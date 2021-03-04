'''
Kernel Mean Matching (KMM)
'''

import numpy as np
import copy
from sklearn.metrics import pairwise
from cvxopt import matrix, solvers


class KMM(object):
    """
    KMM: Kernel Mean Matching
    
    Sample bias correction method for domain adaptation based on the
    minimization of the MMD distance between source and target domains
    
    The MMD is computed in the features space induced by a Gaussian kernel
    
    Reference: Jiayuan Huang, Arthur Gretton, Karsten Borgwardt, Bernhard Schölkopf,
    and Alex J. Smola. "Correcting sample selection bias by unlabeled data."
    In B. Schölkopf, J. C. Platt, and T. Hoffman, editors,
    Advances in Neural Information Processing Systems 19, pages 601–608. MIT Press,2007
    (https://papers.nips.cc/paper/3075-correcting-sample-selection-bias-by-unlabeled-data.pdf)
    
    Parameters
    ----------
    estimator: object, optional (default=None)
        Estimator should implement fit and predict methods
        
    sigma: float, optional (default=0.1)
        kernel bandwidth
        
    B: float, optional (default=1000)
        Constraint parameter
        
    epsilon: float, optional
        Constraint parameter, if None epsilon is set to
        (np.sqrt(len(X)) - 1)/np.sqrt(len(X))
    """
    
    def __init__(self, estimator=None,
                 sigma=0.1, B=1000, epsilon=None):
        self.estimator = estimator
        self.sigma = sigma
        self.B = B
        self.epsilon = epsilon
    
    
    def fit(self, X, y=None, index=None, **fit_params):
        """
        Two-stage fitting:
        - First fit the optimal weighting scheme for sample
          bias correction
        - If y is not None, fit estimator on the 
          reweighted training instances.
        
        Parameters
        ----------
        X, y: arrays
            input and output data
            
        index: iterable
            Index should contains 2 or 3 lists or 1D-arrays
            corresponding to:
            index[0]: indexes of source labeled data in X, y
            index[1]: indexes of target unlabeled data in X, y
            index[2]: indexes of target labeled data in X, y
            
        fit_params: key, value arguments
            Arguments to pass to the fit method (epochs, batch_size...)
            
        Returns
        -------
        self 
        """
        
        if index is None:
            index = [range(len(X))] * 2
        
        if len(index) == 2:
            src_index = index[0]
            tgt_index = index[1]
        if len(index) == 3:
            src_index = np.concatenate((index[0], index[2]))
            tgt_index = index[1]
        
        # Fit training weights
        self.training_weights_ = self._fit_weights(X[src_index], X[tgt_index])
        
        # Fit estimator
        if y is not None:
            self.estimator_ = copy.deepcopy(self.estimator)
            self.estimator_.fit(X[src_index], y[src_index], 
                                sample_weight = self.training_weights_,
                                **fit_params)
        return self


    def _fit_weights(self, X_src, X_tgt):
        """
        Compute training weights by solving QP
        """       
        m = len(X_src)
        n = len(X_tgt)
        
        # Get epsilon
        if self.epsilon is None:
            self.epsilon = (np.sqrt(m) - 1)/np.sqrt(m)

        # Compute Kernel Matrix
        K_src = pairwise.rbf_kernel(X_src, X_src, self.sigma)
        K = (1/2) * (K_src + K_src.transpose())

        # Compute q
        K_tgt = pairwise.rbf_kernel(X_src, X_tgt, self.sigma)
        q = -(m/n) * np.dot(K_tgt, np.ones((n, 1)))

        # Set up QP
        G = np.concatenate((np.ones((1,m)),
                            -np.ones((1,m)),
                            -np.eye(m),
                            np.eye(m)))
        h = np.concatenate((np.array([[m*(self.epsilon+1)]]),
                            np.array([[m*(self.epsilon-1)]]),
                            -np.zeros((m,1)),
                            np.ones((m,1))*self.B))
        P = matrix(K, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        
        # Solve QP
        solvers.options['show_progress'] = False
        weights = solvers.qp(P,q,G,h)['x']
        return np.array(weights).ravel()
    
    
    def get_weight(self):
        """
        Return fitted training weights
        """
        return self.training_weights_
    
    
    def predict(self, X):
        """
        Return estimator predictions
        
        Parameter
        ---------
        X: array
            input array
        
        Returns
        -------
        array: predictions
        """
        return self.estimator_.predict(X)