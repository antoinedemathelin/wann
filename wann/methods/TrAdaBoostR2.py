"""
TwoStageTrAdaBoostR2
"""

import numpy as np
import copy
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples


def _binary_search(func, a=0, b=1, c_iter=0, tol=1.e-3, max_iter=1000, best=1, best_score=1):
    if c_iter > max_iter:
        print("Binary search's goal not meeted! Value is set to be the available best!")
        return best
    elif np.abs(func(a)) < tol:
        return a
    elif np.abs(func(b)) < tol:
        return b
    else:
        c = (a + b) / 2
        if func(c) < best_score:
            best = c
            best_score = func(c)
        if func(c) * func(a) <= 0:
            c_iter += 1
            return _binary_search(func, a, c, c_iter, tol, max_iter, best, best_score)
        else:
            c_iter += 1
            return _binary_search(func, c, b, c_iter, tol, max_iter, best, best_score)



class AdaBoostR2prime(AdaBoostRegressor):
    
    def fit(self, X, y, sample_weight, source_size):
        
        # Check parameters
        self._validate_estimator()
              
        # Init sample_weights_
        self.sample_weights_ = []

        # Make sure sample_weight sum to 1
        assert (np.sum(sample_weight) - 1. < 0.0001)
        
        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        
        random_state = check_random_state(self.random_state)
        
        for iboost in range(self.n_estimators):
            self.sample_weights_.append(np.copy(sample_weight))
            
            copy_sample_weight = np.copy(sample_weight)

            sample_weight, estimator_weight, estimator_error = self._boost(iboost, X, y,
                                                         sample_weight,
                                                         random_state)
            
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight
            

            sample_weight[:source_size] = copy_sample_weight[:source_size]
            
            sample_weight[source_size:] *= ((1 - np.sum(sample_weight[:source_size]))
                                             / np.sum(sample_weight[source_size:]))
        
        return self
    


class TwoStageTrAdaBoostR2(object):
    """
    TwoStage-TrAdaBoostR2
    
    TrAdaBoostR2 algorithm is an instances-based domain adaptation method suited for regression task.
    The method is based on a "reverse boosting" principle where the weights of source instances
    poorly predicted decrease at each boosting iteration.
    
    The implemented algorithm is the two-stage version of the method, where the source and target
    instances weights are alternatively fixed from one stage to another. The total sum of target
    weights increase at each boosting iteration and a cross-validation score is computed to select
    the best estimator over all iterations.
    
    Reference: David Pardoe and Peter Stone.
    "Boosting for regression transfer".
    InProceedings of the 27thInternational Conference on Machine Learning (ICML), June 2010.
    
    Parameters
    ----------
    base_estimator: object
        base_estimator should implement a fit and
        predict method.
        
    n_estimators: int, optional (default=10)
        Number of first stage boosting iteration
        
    fold: int, optional (default=5)
        Number of fold for cross-validation
        
    verbose: int, optional (default=0)
        If verbose, print cross-validation score
        at each first stage iteration
        
    stage: int, optional (default=2)
        stage=1 or 2, if stage=1, no boosting
        are done to fine-tune target weights
        
    random_state: int, optional (default=None)
        Seed Number
    """
    def __init__(self,
                 base_estimator = None,
                 n_estimators = 10,
                 fold = 5,
                 learning_rate = 1.,
                 verbose=0,
                 stage=1,
                 random_state = None,
                 save_hist = False):
        
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.fold = fold
        self.verbose = verbose
        self.stage = stage
        self.random_state = random_state
        

    
    def fit(self, X, y, index):
        """
        Fit TwoStageTrAdaBoostR2
        
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
        X = np.concatenate((X[index[0]], X[index[1]]))
        y = np.concatenate((y[index[0]], y[index[1]]))
        
        # Init sample_weights_
        self.sample_weights_ = []

        # Initialize weights to 1 / n_samples
        sample_weight = np.ones(len(X)) / len(X)
        
        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        
        for iboost in range(self.n_estimators):
            self.sample_weights_.append(np.copy(sample_weight))

            cv_score = self._cross_val_score(X, y, sample_weight, len(index[0]))
            
            self.estimator_errors_[iboost] = cv_score.mean()
            
            if self.verbose:
                print("cv error of estimator %i: %.3f (%.10f)"%
                     (iboost, cv_score.mean(), cv_score.std()))

            sample_weight = self._boost(iboost, X, y, sample_weight, len(index[0]), len(index[1]))

            sample_weight_sum = np.sum(sample_weight)
                   
            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        
        return self


    def _boost(self, iboost, X, y, sample_weight, source_size, target_size):
        estimator = clone(self.base_estimator)
        self.estimators_.append(estimator)
        bootstrap_idx = np.random.choice(
            len(X), size=len(X), replace=True,
            p=sample_weight
        )
        if self.stage == 1:
            estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        else:
            estimator.fit(X, y, sample_weight, source_size)
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()
        if error_max != 0:
            error_vect /= error_max
            
        beta = self._get_beta(iboost, sample_weight, error_vect,
                              source_size, target_size)
        
        if not iboost == self.n_estimators - 1:
            # Source updating weights
            sample_weight[:source_size] *= np.power(
                beta, error_vect[:source_size]
            )
        return sample_weight


    def _get_beta(self, iboost, sample_weight, error_vect, source_size, target_size):
        
        K_t = (target_size/len(sample_weight) + (iboost/(self.n_estimators - 1)) *
               (1 - (target_size/len(sample_weight))))
        
        C_t = np.sum(sample_weight[-target_size:]) * ((1 - K_t) / K_t)
        
        def func(x):
            return np.dot(sample_weight[:source_size],
                          np.power(x, error_vect[:source_size])) - C_t       
        return _binary_search(func, a=0, b=1, c_iter=0,
                              tol=1.e-3, max_iter=1000,
                              best=1, best_score=1)

                                        
    def _cross_val_score(self, X, y, sample_weight, source_size):
        kf = KFold(n_splits = self.fold)
        error = []
        X_target = X[source_size:]
        X_source = X[:source_size]
        y_target = y[source_size:]
        y_source = y[:source_size]
        target_weight = sample_weight[source_size:]
        source_weight = sample_weight[:source_size]
        for train, test in kf.split(X_target):
            est = clone(self.base_estimator)
            X_train = np.concatenate((X_source, X_target[train]))
            y_train = np.concatenate((y_source, y_target[train]))
            X_test = X_target[test]
            y_test = y_target[test]
            # make sure the sum weight of the target data do not change with CV's split sampling
            target_weight_train = target_weight[train]*np.sum(target_weight)/np.sum(target_weight[train])           
                        
            bootstrap_idx = np.random.choice(
            len(X_train), size=len(X_train), replace=True,
            p=np.concatenate((source_weight, target_weight_train))
            )

            if self.stage == 1:
                est.fit(X_train[bootstrap_idx], y_train[bootstrap_idx])
            else:
                est.fit(X_train, y_train, np.concatenate((source_weight, target_weight_train)), source_size)
            
            y_predict = est.predict(X_test)
            error.append(mean_squared_error(y_predict, y_test))
        return np.array(error)

    
    def predict(self, X):
        """
        Return the best estimator predictions
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        y_pred: array
            prediction of the estimator of
            minimal cross-validation score
        """
        # select the model with the least CV error
        fmodel = self.estimators_[np.array(self.estimator_errors_).argmin()]
        predictions = fmodel.predict(X)
        return predictions