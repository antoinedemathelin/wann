'''
KLIEP 

Implementation inspired from Ashwin: 
https://github.com/ashwinkumarm/biasingComparison/blob/master/biasingTechniqueComparison/kliep/kliep.py
'''
import copy
import numpy as np
from sklearn.base import clone

class KLIEP(object):
    """
    Kullback–Leibler importance estimation procedure (KLIEP)
    
    KLIEP is an mportance estimation method for domain adaptation. The algorithm finds an importance
    estimate w(x) = \sum{k} \alpha_k \phi_k(x) such that the Kullback–Leibler divergence from the target
    input density p_t(x) to the reweighted source one w(x)p_s(x) is minimized.
    \phi_k(x) are kernels models.
    
    This is the LCV version of KLIEP, where the kernels bandwith are selected through cross-validation
    On different values of sigmas
    
    Reference: M. Sugiyama, S. Nakajima, H. Kashima, P. von Bünau and  M. Kawanabe.
    "Direct importance estimation with model selection and its application to covariateshift adaptation".
    InProceedings of the 20th International Conference on Neural InformationProcessing Systems,
    NIPS’07, page 1433–1440, Red Hook, NY, USA, 2007. Curran AssociatesInc
    
    Parameters
    ----------
    estimator: object, optional (default=None)
        Estimator should implement fit and predict methods
        
    cv: int, optional (default=3)
        cross-validation split parameter
        
    sigmas: list, optional (default=[2^-5, .. , 2^5])
        list of kernel bandwidth to select through
        cross-validation
        
    random_state: int, optional (default=0)
        seed number
    """
    
    def __init__(self, estimator=None, cv=3,
                 sigmas=[2**(i-5) for i in range(10)],
                 random_state=0):
        self.estimator = estimator
        self.cv = cv
        self.sigmas = sigmas
        self.random_state = random_state
        
    
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
            train_index = index[0]
            test_index = index[1]
        if len(index) == 3:
            train_index = np.concatenate((index[0], index[2]))
            test_index = index[1]
        
        self._fit_weights(X[train_index], X[test_index])
                
        if y is not None:
            self.training_weights_ = self.get_weight(X[train_index])
            self.estimator_ = clone(self.estimator, safe=False)
            self.estimator_.fit(X[train_index], y[train_index], 
                                sample_weight = self.training_weights_,
                                **fit_params)
        return self
    
    
    def _fit_weights(self, X_src, X_tgt):
        np.random.seed(self.random_state)
        X_tgt_shuffled = np.copy(X_tgt)
        np.random.shuffle(X_tgt_shuffled)
        split = int(len(X_tgt)/self.cv)
        
        j_scores = {}
        
        for n_sample in [0.1, 0.2]:
            for sigma in self.sigmas:
                j_scores[(n_sample, sigma)] = np.zeros(self.cv)
                for k in range(self.cv):
                    X_tgt_fold = X_tgt_shuffled[k * split: (k + 1) * split, :] 
                    j_scores[(n_sample, sigma)][k] = self._fit(X_src=X_src,
                                                               X_tgt=X_tgt_fold,
                                                               n_sample=n_sample,
                                                               sigma=sigma)
                j_scores[(n_sample, sigma)] = np.mean(j_scores[(n_sample, sigma)])

        sorted_scores = sorted([x for x in j_scores.items() if np.isfinite(x[1])],
                               key=lambda x :x[1], reverse=True)
        self.sigma_ = sorted_scores[0][0][1]
        self.n_sample_ = sorted_scores[0][0][0]
        self.j_scores_ = sorted_scores
        self.j_ = self._fit(X_src, X_tgt_shuffled, self.n_sample_, self.sigma_)
        
        return self
        
    
    def _fit(self, X_src, X_tgt, n_sample, sigma):
        n_sample = int(n_sample * len(X_tgt))
        indices = np.random.choice(len(X_tgt), size=n_sample, replace=False)
        self.test_vectors_ = np.copy(X_tgt[indices,:])
        
        X_src = X_src.reshape((X_src.shape[0], 1, X_src.shape[1]))
        X_tgt = X_tgt.reshape((X_tgt.shape[0], 1, X_tgt.shape[1]))
        
        self._find_alpha(X_src=X_src,
                         X_tgt=X_tgt,
                         n_sample=n_sample,
                         sigma=sigma)
        
        return np.log(self.get_weight(X_tgt, sigma=sigma)).sum()/len(X_tgt)
        
    
    def _phi(self, X, sigma):
        return np.exp(-np.sum((X-self.test_vectors_)**2, axis=-1)/
                      (2 * sigma**2))

    
    def _find_alpha(self, X_src, X_tgt, n_sample, sigma):
        A = self._phi(X_tgt, sigma)
        b = self._phi(X_src, sigma).sum(axis=0) / len(X_src)
        b = b.reshape((n_sample, 1))
        
        alpha = np.ones((n_sample, 1))/float(n_sample)
        for k in range(5000):
            alpha += 1e-4 * np.dot(np.transpose(A), 1./np.dot(A, alpha))
            alpha += b * (((1-np.dot(np.transpose(b), alpha))/np.dot(np.transpose(b), b)))
            alpha = np.maximum(0, alpha)
            alpha /= (np.dot(np.transpose(b), alpha))
        self.alpha_ = alpha
        
    
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
    
    
    def get_weight(self, X, sigma=None):
        """
        Return fitted training weights
        """
        if sigma is None:
            sigma = self.sigma_
        if len(X.shape) != 3:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        return np.dot(self._phi(X, sigma), self.alpha_).reshape((X.shape[0],))