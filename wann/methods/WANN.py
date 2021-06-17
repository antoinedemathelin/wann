"""
Weighting Adversarial Neural Network (WANN)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.layers import multiply
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras import losses
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer


class _SavePrediction(Callback):  
    """
    Callbacks which stores predicted weights
    and labels in history at each epoch.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        super().__init__()
            
    def on_epoch_end(self, batch, logs={}):
        """Applied at the end of each epoch"""
        if "y_pred" not in self.model.history.history:
            self.model.history.history["y_pred"] = []
        if "weights" not in self.model.history.history:
            self.model.history.history["weights"] = []
        predictions = self.model.predict([self.X, self.X,
                                          self.y, self.y, self.y])
        self.model.history.history["y_pred"].append(
        predictions[0].ravel()
        )
        self.model.history.history["weights"].append(
        predictions[-1].ravel()
        )


@tf.custom_gradient
def _grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class _GradReverse(tf.keras.layers.Layer):
    """
    Gradient Reversal Layer: inverse sign of gradient during
    backpropagation.
    """
    def __init__(self):
        super().__init__()

    def call(self, x):
        return _grad_reverse(x)
    
    
def _get_default_model(shape, activation=None, C=1, name="DefaultModel"):
    """
    Default Network.
    """
    inputs = Input(shape=(shape,))
    modeled = Dense(100, activation='relu',
                         kernel_constraint=MinMaxNorm(0, C),
                         bias_constraint=MinMaxNorm(0, C))(inputs)
    modeled = Dense(1, activation=activation,
                    kernel_constraint=MinMaxNorm(0, C),
                    bias_constraint=MinMaxNorm(0, C))(modeled)
    model = Model(inputs, modeled, name=name)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


class WANN(object):
    """
    WANN: Weighting Adversarial Neural Network is an instance-based domain adaptation
    method suited for regression tasks. It supposes the supervised setting where some
    labeled target data are available.
    
    The goal of WANN is to compute a source instances reweighting which correct
    "shifts" between source and target domain. This is done by minimizing the
    Y-discrepancy distance between source and target distributions
    
    WANN involves three networks:
        - the weighting network which learns the source weights.
        - the task network which learns the task.
        - the discrepancy network which is used to estimate a distance 
          between the reweighted source and target distributions: the Y-discrepancy
    
    Parameters
    ----------
    get_base_model: callable, optional
        Constructor for the two networks: task and discrepancer.
        The constructor should take the four following
        arguments:
        - shape: the input shape
        - C: the projecting constant
        - activation: the last layer activation function
        - name: the model name
        If None, get_default_model is used.
        
    get_weighting_model: callable, optional
        Constructor for the weightig network.
        The constructor should take the same arguments 
        as get_base_model.
        If None, get_base_model is used.
        
    C: float, optional (default=1.)
        Projecting constant: networks should be
        regularized by projecting the weights of each layer
        on the ball of radius C.
        
    C_w: float, optional (default=None)
        Projecting constant of the weighting network.
        If None C_w = C.
        
    optimizer: tf.keras Optimizer, optional (default="adam")
        Optimizer of WANN
        
    save_hist: boolean, optional (default=False)
        Wether to save the predicted weights and labels
        at each epochs or not
    """
    
    def __init__(self, get_base_model=None, get_weighting_model=None,
                 C=1., C_w=None, optimizer='adam', save_hist=False):
        
        self.get_base_model = get_base_model
        if self.get_base_model is None:
            self.get_base_model = _get_default_model
        
        self.get_weighting_model = get_weighting_model
        if self.get_weighting_model is None:
            self.get_weighting_model = get_base_model
        
        self.C = C
        self.C_w = C_w
        if self.C_w is None:
            self.C_w = C
        
        self.save_hist = save_hist
        self.optimizer = optimizer
        

    def fit(self, X, y, index=None, weights_target=None, **fit_params):
        """
        Fit WANN
        
        Parameters
        ----------
        X, y: numpy arrays
            Input data
            
        index: iterable
            Index should contains 2 lists or 1D-arrays
            corresponding to:
            index[0]: indexes of source labeled data in X, y
            index[1]: indexes of target labeled data in X, y
            
        weights_target: numpy array, optional (default=None)
            Weights for target sample.
            If None, all weights are set to 1.
            
        fit_params: key, value arguments
            Arguments to pass to the fit method (epochs, batch_size...)
            
        Returns
        -------
        self 
        """
        self.fit_params = fit_params
        assert hasattr(index, "__iter__"), "index should be iterable"
        assert len(index) == 2, "index length should be 2"
        src_index = index[0]
        tgt_index = index[1]
        self._fit(X, y, src_index, tgt_index, weights_target)        
        return self


    def _fit(self, X, y, src_index, tgt_index, weights_target):
        # Resize source and target index to the same length
        max_size = max((len(src_index), len(tgt_index)))
        resize_src_ind = np.array([src_index[i%len(src_index)]
                                   for i in range(max_size)])
        resize_tgt_ind = np.array([tgt_index[i%len(tgt_index)]
                                   for i in range(max_size)])
        
        # If no target weights, all are set to one 
        if weights_target is None:
             resize_weights_target = np.ones(max_size)
        else:
            resize_weights_target = np.array([weights_target[i%len(weights_target)]
                                              for i in range(max_size)])
                     
        # Create WANN model
        if not hasattr(self, "model"):
            self._create_wann(shape=X.shape[1])

        # Callback to save predicted weights and labels
        callbacks = []
        if self.save_hist:
            callbacks.append(_SavePrediction(X, y))
            
        # Fit
        self.model.fit([X[resize_src_ind], X[resize_tgt_ind],
                        y[resize_src_ind], y[resize_tgt_ind],
                        resize_weights_target],
                       callbacks = callbacks,
                       **self.fit_params)
        return self
            
            
    def _create_wann(self, shape):
        # Build task, weights_predictor and discrepancer network
        # Weights_predictor should end with a relu activation
        self.weights_predictor = self.get_weighting_model(
                shape, activation='relu', C=self.C_w, name="weights")
        self.task = self.get_base_model(
                shape, activation=None, C=self.C, name="task")
        self.discrepancer = self.get_base_model(
                shape, activation=None, C=self.C, name="discrepancer")
        
        # Create input layers for Xs, Xt, ys, yt and target weights
        input_source = Input(shape=(shape,))
        input_target = Input(shape=(shape,))
        output_source = Input(shape=(1,))
        output_target = Input(shape=(1,))
        weights_target = Input(shape=(1,))
        Flip = _GradReverse()
        
        # Get networks output for both source and target
        weights_source = self.weights_predictor(input_source)      
        output_task_s = self.task(input_source)
        output_task_t = self.task(input_target)
        output_disc_s = self.discrepancer(input_source)
        output_disc_t = self.discrepancer(input_target)
        
        # Reversal layer at the end of discrepancer
        output_disc_s = Flip(output_disc_s)
        output_disc_t = Flip(output_disc_t)

        # Create model and define loss
        self.model = Model([input_source, input_target, output_source, output_target, weights_target],
                           [output_task_s, output_task_t, output_disc_s, output_disc_t, weights_source],
                           name='WANN')
            
        loss_task_s = K.mean(multiply([weights_source, K.square(output_source - output_task_s)]))
        loss_task_t = K.mean(multiply([weights_target, K.square(output_target - output_task_t)]))
            
        loss_disc_s = K.mean(multiply([weights_source, K.square(output_source - output_disc_s)]))
        loss_disc_t = K.mean(multiply([weights_target, K.square(output_target - output_disc_t)]))
            
        loss_task = loss_task_s + loss_task_t
        loss_disc = loss_disc_t - loss_disc_s
                         
        loss = loss_task + loss_disc
   
        self.model.add_loss(loss)
        self.model.add_metric(tf.reduce_sum(K.mean(weights_source)), name="weights", aggregation="mean")
        self.model.add_metric(tf.reduce_sum(loss_task_s), name="task_s", aggregation="mean")
        self.model.add_metric(tf.reduce_sum(loss_task_t), name="task_t", aggregation="mean")
        self.model.add_metric(tf.reduce_sum(loss_disc), name="disc", aggregation="mean")
        self.model.add_metric(tf.reduce_sum(loss_disc_s), name="disc_s", aggregation="mean")
        self.model.add_metric(tf.reduce_sum(loss_disc_t), name="disc_t", aggregation="mean")
        self.model.compile(optimizer=self.optimizer)
        return self
    
    
    def predict(self, X):
        """
        Predict method: return the prediction of task network
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        y_pred: array
            prediction of task network
        """
        return self.task.predict(X)
    
    
    def get_weight(self, X):
        """
        Return the predictions of weighting network
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        array:
            weights
        """
        return self.weights_predictor.predict(X)
    
    
    def save(self, path):
        """
        Save task network
        
        Parameters
        ----------
        path: str
            path where to save the model
        """
        self.task.save(path)
        self.weights_predictor.save(path + "_weights")
        return self