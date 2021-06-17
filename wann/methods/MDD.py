import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import multiply, subtract
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


class MDD(object):
    """
    MCD: Maximum Disparity Discrepancy is a feature-based domain adaptation
    method originally introduced for unsupervised classification DA.
    
    The goal of MDD is to find a new representation of the input features which
    minimizes the disparity discrepancy between the source and target domains 
    
    The disparity discrepancy is defined as follow:
        
        disp_h(Q, P) = sup_h' | R_S(h, h') - R_T(h, h') |  
    
    The disparity discrepancy is estimated through adversarial training of three networks:
    An encoder and two classifiers. the first classifier learns the task on the source domain.
    The second is used to compute the disparity discrepancy between source and target domains. 
    A reversal layer is placed between the encoder and this last classifier 
    to perform adversarial training.
    
    Reference: Yuchen Zhang, Tianle Liu, Mingsheng Long, and Michael Jordan.
    "Bridging theory and algorithm for domain adaptation".
    In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors,
    Proceedings of the 36th International Conference onMachine Learning,
    pages 7404–7413, Long Beach, California, USA, 2019. PMLR
    
    Parameters
    ----------
    get_encoder: callable, optional
        Constructor for encoder network.
        The constructor should take at least
        the "shape" argument.
        
    get_task: callable, optional
        Constructor for two networks: task and discriminer.
        The constructor should take at least
        the "shape" argument.
        
    optimizer: tf Optimizer, optional (default="adam")
        MDD Optimizer
        
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
        Fit MDD
        
        Parameters
        ----------
        X, y: numpy arrays
            Input data
            
        index: iterable
            Index should contains 2 lists or 1D-arrays
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
            src_index = np.concatenate((src_index, tgt_train_index))
            
        self._create_model(X.shape[1:], y.shape[1:])
        
        max_size = max((len(src_index), len(tgt_index)))
        resize_src_ind = np.array([src_index[i%len(src_index)]
                                   for i in range(max_size)])
        resize_tgt_ind = np.array([tgt_index[i%len(tgt_index)]
                                   for i in range(max_size)])
        
        self.model.fit([X[resize_src_ind], X[resize_tgt_ind], y[resize_src_ind]],
                      **fit_params)
        return self
    
    
    def predict(self, X):
        """
        Predict method: return the prediction of classifier 1
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
        
        
    def _create_model(self, shape_X, shape_y):
        
        self.encoder = self.get_encoder(shape=shape_X[0], **self.kwargs)
        self.task = self.get_task(shape=self.encoder.output_shape[1], **self.kwargs)
        self.discrepancer = self.get_task(shape=self.encoder.output_shape[1], **self.kwargs)
        
        input_src = Input(shape_X)
        output_src = Input(shape_y)
        input_tgt = Input(shape_X)
        
        encoded_src = self.encoder(input_src)
        encoded_tgt = self.encoder(input_tgt)
        
        task_src = self.task(encoded_src)
        task_tgt = self.task(encoded_tgt)
#         task_src_disc = _GradReverse()(task_src)
#         task_tgt_disc = _GradReverse()(task_tgt)
        
        discrepanced_src = _GradReverse()(encoded_src)
        discrepanced_tgt = _GradReverse()(encoded_tgt)
        discrepanced_src = self.discrepancer(discrepanced_src)
        discrepanced_tgt = self.discrepancer(discrepanced_tgt)
        discrepanced_src = _GradReverse()(discrepanced_src)
        discrepanced_tgt = _GradReverse()(discrepanced_tgt)
        
        task_loss = K.mean(K.square(subtract([task_src, output_src])))
        
        disc_loss_src = K.mean(K.square(subtract([discrepanced_src, task_src])))
        disc_loss_tgt = K.mean(K.square(subtract([discrepanced_tgt, task_tgt])))
        
        disc_loss = K.abs(disc_loss_src - disc_loss_tgt)
        
        loss = task_loss + self.lambda_ * disc_loss
        
        self.model = Model([input_src, input_tgt, output_src],
                           [task_src, task_tgt,
                            discrepanced_src, discrepanced_tgt],
                           name="MDD")
        self.model.add_loss(loss)
        self.model.compile(optimizer=self.optimizer)
        
        self.task_to_save = Model(input_src, task_src)
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
            
