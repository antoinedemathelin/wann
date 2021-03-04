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


class DANN(object):
    """
    DANN: Discriminative Adversarial Neural Network is a feature-based domain adaptation
    method. Originally introduced for unsupervised classification DA it could be
    widen to other task in supervised DA straightforwardly.
    
    The goal of DANN is to find a new representation of the input features in which
    source and target data could not be distinguish by any "discriminer" network.
    This new representation is learned by an "encoder" network in an adversarial fashion.
    A "task" network is learned on the encoded space in parallel to the encoder and 
    discriminer networks.
        
    A reversal layer is placed at the input of the discriminer network which then optimizes
    an opposite objective than the two other networks.
    
    Reference: Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette,
    M. Marchand, and V. Lempitsky. "Domain-adversarial training of neural networks". 
    J. Mach. Learn. Res., 17(1):2096–2030, January 2016.
    
    Parameters
    ----------
    get_encoder: callable, optional
        Constructor for encoder network.
        The constructor should take at least
        the "shape" argument.
        
    get_task: callable, optional
        Constructor for two networks: task and discriminer.
        The constructor should take at least
        the "shape" and "activation" arguments.
        
    optimizer: tf Optimizer, optional (default="adam")
        DANN optimizer
        
    lambda_: float, optional (default=1.)
        trade-off parameter
        
    kwargs: key, value arguments, optional
        Additional arguments for constructors
    """
    def __init__(self, get_encoder, get_task, optimizer="adam", lambda_=1., **kwargs):
        self.optimizer = optimizer
        self.get_encoder = get_encoder
        self.get_task = get_task
        self.lambda_ = lambda_
        self.kwargs = kwargs

    
    def fit(self, X, y, index, **fit_params):
        """
        Fit DANN
        
        Parameters
        ----------
        X, y: numpy arrays
            Input data
            
        index: iterable
            Index should contains 2 or 3 lists or 1D-arrays
            corresponding to:
            index[0]: indexes of source labeled data in X, y
            index[1]: indexes of target unlabeled data in X, y
            index[2]: indexes of target labeled data in X, y (optional)
            
        fit_params: key, value arguments
            Arguments to pass to the fit method (epochs, batch_size...)
            
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
            
        self._create_model(shape=X.shape[1])
        
        task_index = np.concatenate((src_index, tgt_train_index))
        disc_index = np.concatenate((src_index, tgt_index))
        labels = np.array([0] * len(src_index) + [1] * len(tgt_index))
        max_size = len(disc_index)
        resize_task_ind = np.array([task_index[i%len(task_index)]
                                   for i in range(max_size)])
        self.model.fit([X[resize_task_ind], X[disc_index]], [y[resize_task_ind], labels],
                      **fit_params)
        return self
    
    
    def predict(self, X):
        """
        Predict method: return the prediction of task network
        on the encoded features
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        y_pred: array
            prediction of task network
        """
        return self.task.predict(self.encoder.predict(X))
        
        
    def _create_model(self, shape):
        
        self.encoder = self.get_encoder(shape=shape, **self.kwargs)
        self.task = self.get_task(shape=self.encoder.output_shape[1],
                                  activation=None, **self.kwargs)
        self.discriminer = self.get_task(shape=self.encoder.output_shape[1],
                                         activation="sigmoid", **self.kwargs)
        
        input_task = Input((shape,))
        input_disc = Input((shape,))
        
        encoded_task = self.encoder(input_task)
        encoded_disc = self.encoder(input_disc)
        
        tasked = self.task(encoded_task)
        discrimined = _GradReverse()(encoded_disc)
        discrimined = self.discriminer(discrimined)
        
        
        self.model = Model([input_task, input_disc],
                           [tasked, discrimined], name="DANN")
        self.model.compile(optimizer=self.optimizer,
                           loss=["mean_squared_error", "binary_crossentropy"],
                           loss_weights=[1., self.lambda_])
        
        self.task_to_save = Model(input_task, tasked)
        self.task_to_save.compile(optimizer="adam", loss="mean_squared_error")
        
        return self
    
    
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
        self.encoder.save(path + "_encoder")
        return self
            
