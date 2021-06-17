"""
Weighting Adversarial Neural Network (WANN)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.layers import multiply, subtract, Reshape
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras import losses
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer


EPS = 1e-20


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
                                          self.y, self.y])
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
    
    
class NormalizationLayer(tf.keras.layers.Layer):
    
    def __init__(self, norm):
        self.norm = K.variable(norm)
        super().__init__(name = "normalization_layer")
        
    def call(self, x):
        return self.norm * x


# class Normalize(Callback):
    
#     def __init__(self, X):
#         self.X = X
#         super().__init__()
            
#     def on_epoch_begin(self, batch, logs={}):
#         sum_of_weights = np.mean(self.model.get_layer("weights").predict(self.X))
#         self.model.get_layer("normalization_layer").norm.assign(1. / (sum_of_weights + EPS))
    
    
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


class BalancedWeighting(object):
    """
    WANN: Weighting Adversarial Neural Network is an instance-based domain adaptation
    method suited for regression tasks. It supposes the supervised setting where some
    labeled target data are available.
    
    The goal of WANN is to compute a source instances reweighting which correct
    "shifts" between source and target domain. This is done by minimizing the
    Y-discrepancy distance between source and target distributions
    
    WANN involves three networks:
        - W: the weighting network which learns the source weights.
        - h_t: the task network which learns the task.
        - h_d: the discrepancy network which is used to estimate a distance 
               between the reweighted source and target distributions: the Y-discrepancy
    
    WANN objective function is composed of the average loss on the source instances and
    the estimated Y-discrepancy distance between the reweighted source and target instances.
    
    .. math:: G(W, h_t, h_d) = \sum_{(x_i, y_i) \in S} W(x_i) (h_t(x_i)-y_i)^2 
    + \sum_{(x_i, y_i) \in T} (h_t(x_i)-y_i)^2 
    + \left|\sum_{(x_i, y_i) \in S} W(x_i) (h_d(x_i)-y_i)^2
    - \sum_{(x_j, y_j) \in T}(h_d(x_j)-y_j)^2 \right|^2
    
    A reversal layer is placed at the output of h_d, thus W and h_t are trained in order to
    minimize G, whereas h_d is trained to optimize it.
    
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
        
    C: float, optional (default=1.)
        Projecting constant: networks should be
        regularized by projecting the weights of each layer
        on the ball of radius C.
        
    optimizer: tf.keras Optimizer, optional (default="adam")
        Optimizer of WANN
        
    save_hist: boolean, optional (default=False)
        Wether to save the predicted weights and labels
        at each epochs or not
    """
    
    def __init__(self, get_base_model=None, C=1., optimizer='adam'):
        
        self.get_base_model = get_base_model
        if self.get_base_model is None:
            self.get_base_model = _get_default_model
        self.C = C
        self.optimizer = optimizer
        

    def fit(self, X, y, index=None, **fit_params):
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
        self._fit(X, y, src_index, tgt_index)
        return self


    def _fit(self, X, y, src_index, tgt_index):
        # Resize source and target index to the same length
        max_size = max((len(src_index), len(tgt_index)))
        resize_src_ind = np.array([src_index[i%len(src_index)]
                                   for i in range(max_size)])
        resize_tgt_ind = np.array([tgt_index[i%len(tgt_index)]
                                   for i in range(max_size)])
                     
        # Create model
        if not hasattr(self, "model"):
            self._create_model(shape=X.shape[1], norm=1.)

        # Fit
        self.model.fit([X[resize_src_ind], X[resize_tgt_ind],
                        y[resize_src_ind], y[resize_tgt_ind]],
                       **self.fit_params)
        return self
            
            
    def _create_model(self, shape, norm):

        self.task = self.get_base_model(
                shape, activation=None, C=self.C, name="task")
        
        # Create input layers for Xs, Xt, ys, yt and target weights
        input_source = Input(shape=(shape,))
        input_target = Input(shape=(shape,))
        output_source = Input(shape=(1,))
        output_target = Input(shape=(1,))
        
        output_task_s = self.task(input_source)
        output_task_t = self.task(input_target)
        
        # Create model and define loss
        self.model = Model([input_source, input_target, output_source, output_target],
                           [output_task_t, output_task_s],
                           name='Naive')

        loss_task_s = K.mean(K.square(subtract([output_source, output_task_s])))
        loss_task_t = K.mean(K.square(subtract([output_target, output_task_t])))

        loss = loss_task_s + loss_task_t
        
        self.model.add_loss(loss)
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
    
    
    def save(self, path):
        """
        Save task network
        
        Parameters
        ----------
        path: str
            path where to save the model
        """
        self.task.save(path)
        return self