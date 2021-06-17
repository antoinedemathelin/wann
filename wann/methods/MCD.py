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


class MCD(object):
    """
    MCD: Maximum Classifier Discrepancy is a feature-based domain adaptation
    method originally introduced for unsupervised classification DA.
    
    The goal of MCD is to find a new representation of the input features which
    minimizes the discrepancy between the source and target domains 
    
    The discrepancy is estimated through adversarial training of three networks:
    An encoder and two classifiers. These two learn the task on the source domains
    and are used to compute the discrepancy. A reversal layer is placed between
    the encoder and the two classifiers to perform adversarial training.
    
    Reference: Saito, K., Watanabe, K., Ushiku, Y., and Harada, T.
    "Maximum  classifier  discrepancy  for  unsupervised  domain adaptation".
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3723–3732, 2018.
    
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
        MCD Optimizer
        
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
        Fit MCD
        
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
        return self.classifier_1.predict(self.encoder.predict(X))
        
        
    def _create_model(self, shape_X, shape_y):
                
        self.encoder = self.get_encoder(shape=shape_X[0], **self.kwargs)
        self.classifier_1 = self.get_task(shape=self.encoder.output_shape[1], **self.kwargs)
        self.classifier_2 = self.get_task(shape=self.encoder.output_shape[1], **self.kwargs)
        
        input_src = Input(shape_X)
        output_src = Input(shape_y)
        input_tgt = Input(shape_X)
        
        encoded_src = self.encoder(input_src)
        encoded_tgt = self.encoder(input_tgt)
        
        classified_1_src = self.classifier_1(encoded_src)
        classified_2_src = self.classifier_2(encoded_src)
        
        classified_1_tgt = _GradReverse()(encoded_tgt)
        classified_2_tgt = _GradReverse()(encoded_tgt)
        classified_1_tgt = self.classifier_1(classified_1_tgt)
        classified_2_tgt = self.classifier_2(classified_2_tgt)

        src_loss = (K.mean(K.square(subtract([classified_1_src, output_src]))) +
                    K.mean(K.square(subtract([classified_2_src, output_src]))))
        disc_loss = K.mean(K.abs(subtract([classified_1_tgt, classified_2_tgt])))
                
        loss = src_loss - self.lambda_ * disc_loss
        
        self.model = Model([input_src, input_tgt, output_src],
                           [classified_1_src, classified_2_src,
                            classified_1_tgt, classified_2_tgt],
                           name="MCD")
        self.model.add_loss(loss)
        self.model.compile(optimizer=self.optimizer)
        
        self.task_to_save = Model(input_src, classified_1_src)
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
            
            
