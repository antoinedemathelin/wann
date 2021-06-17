"""
Adversarial Discriminative Domain Adaptation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import multiply
from tensorflow.keras import losses
import tensorflow.keras.backend as K


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


class ADDA:
    """
    ADDA: Adversarial Discriminative Domain Adaptation
    
    Reference: Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell.
    "Adversarial discriminative domain adaptation".
    In Computer Vision and Pattern Recognition (CVPR), 2017
    
    Parameters
    ----------
    get_encoder: callable, optional
        Constructor for encoder network.
        The constructor should take at least
        the "shape" argument.
        
    get_task: callable, optional
        Constructor for the task network.
        The constructor should take at least
        the "shape" argument.
        
    get_discriminer: callable, optional
        Constructor for the discriminer network.
        The constructor should take at least
        the "shape" argument.
    
    optimizer: tf Optimizer, optional (default="adam")
        Networks optimizer
            
    kwargs: key, value arguments, optional
        Additional arguments for constructors.
    """
    def __init__(self, get_encoder, get_task, get_discriminer,
                 optimizer="adam", optimizer_task="adam", **kwargs):
        self.get_encoder = get_encoder
        self.get_task = get_task
        self.get_discriminer = get_discriminer
        self.optimizer = optimizer
        self.optimizer_task = optimizer_task
        self.kwargs = kwargs

    
    def fit(self, X, y, index, fit_task_params={}, **fit_params):
        """
        Fit ADDA
        
        Parameters
        ----------
        X, y: numpy arrays
            Input data
            
        index: iterable
            Index should contains 2 lists or 1D-arrays
            corresponding to:
            index[0]: indexes of source labeled data in X, y
            index[1]: indexes of target unlabeled data in X, y
            
        fit_task_params: key, value arguments
            Arguments to pass to the fit method of task network
            (epochs, batch_size...)
        
        fit_params: key, value arguments
            Arguments to pass to the fit method of discriminer
            (epochs, batch_size...)
            
        Returns
        -------
        self 
        """
        assert hasattr(index, "__iter__"), "index should be iterable"
        assert len(index) in (2, 3), "index length should be 2 or 3"
        
        if len(index) == 2:
            src_index = index[0]
            tgt_index = index[1]
            tgt_train_index = np.array([])
            
        if len(index) == 3:
            src_index = index[0]
            tgt_index = index[1]
            tgt_train_index = index[2]
            task_index = np.concatenate((src_index, tgt_train_index))

        self._create_model(X.shape[1:], y.shape[1:])
        
        max_size = max(len(src_index), len(tgt_index))
        resize_tgt_ind = np.resize(tgt_index, max_size)
        resize_src_ind = np.resize(src_index, max_size)
        
        self.src_model_.fit(X[task_index], y[task_index],
                            **fit_params)
        self.tgt_model_.layers[1].set_weights(self.src_encoder_.get_weights())
        self.tgt_model_.fit([self.src_encoder_.predict(X[resize_src_ind]),
                            X[resize_tgt_ind]],
                            **fit_params)
        return self
    
    
    def _create_model(self, shape_X, shape_y):
        
        
        self.src_encoder_ = self.get_encoder(shape=shape_X[0], **self.kwargs)
        self.tgt_encoder_ = self.get_encoder(shape=shape_X[0], **self.kwargs)
        
        self.task_ = self.get_task(shape=self.src_encoder_.output_shape[1], **self.kwargs)
        self.discriminator_ = self.get_discriminer(shape=self.src_encoder_.output_shape[1],
                                                     activation="sigmoid", **self.kwargs)
        
        input_task = Input(shape_X)
        encoded_source = self.src_encoder_(input_task)
        tasked = self.task_(encoded_source)
        self.src_model_ = Model(input_task, tasked, name="ModelSource")
        self.src_model_.compile(optimizer=self.optimizer_task, loss="mse")
               
        input_source = Input(self.src_encoder_.output_shape[1:])
        input_target = Input(shape_X)
        encoded_target = self.tgt_encoder_(input_target)
        discrimined_target = _GradReverse()(encoded_target)
        discrimined_target = self.discriminator_(discrimined_target)
        discrimined_source = self.discriminator_(input_source)
        
        loss = -(K.mean(K.log(1 - discrimined_target + 1e-5)) +
                K.mean(K.log(discrimined_source + 1e-5)))
        
        self.tgt_model_ = Model([input_source, input_target],
                                [discrimined_source, discrimined_target],
                                name="ModelTarget")
        self.tgt_model_.add_loss(loss)
        
        self.tgt_model_.compile(optimizer=self.optimizer)
        
        tasked_tgt = self.task_(encoded_target)
        self.task_to_save = Model(input_target, tasked_tgt)
        self.task_to_save.compile(optimizer="adam", loss="mean_squared_error")
        
        return self


    def predict(self, X, domain="target"):
        """
        Return the predictions of task network on the encoded feature space.

        ``domain`` arguments specify how features from ``X``
        will be considered: as ``"source"`` or ``"target"`` features.
        If ``"source"``, source encoder will be used. 
        If ``"target"``, target encoder will be used.

        Parameters
        ----------
        X : array
            Input data.

        domain : str, optional (default="target")
            Choose between ``"source"`` and ``"target"`` encoder.

        Returns
        -------
        y_pred : array
            Prediction of task network.

        Notes
        -----
        As ADDA is an anti-symetric feature-based method, one should
        indicates the domain of ``X`` in order to apply the appropriate
        feature transformation.
        """
        if domain == "target":
            X = self.tgt_encoder_.predict(X)
        elif domain == "source":
            X = self.src_encoder_.predict(X)
        else:
            raise ValueError("Choose between source or target for domain name")
        return self.task_.predict(X)
    
    
    def save(self, path):
        """
        Save task network
        
        Parameters
        ----------
        path: str
            path where to save the model
        
        Returns
        -------
        self
        """
        self.task_to_save.save(path)
        self.tgt_encoder_.save(path + "_encoder")
        return self
