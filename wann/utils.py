import os
import tempfile
import urllib
import zipfile
import gzip
import tarfile
import shutil
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from methods.WANN import WANN
from methods.DANN import DANN
from methods.MCD import MCD
from methods.MDD import MDD
from methods.KMM import KMM


class BaggingModels(object):
    """
    BaggingModels compute several copy of the same tf.keras model 
    and fit them using parallel computing with the joblib library.
    
    Parameters
    ----------
    func: callable
        Constructor for the base_estimator
        (should return a tf.keras Model or 
        implement a save method)
        
    n_models: int, optional (default=8)
        Number of bagged models
        
    n_jobs: int, optional (default=None)
        Number of jobs for parallel computing
        
    random_state: int, optional (default=None)
        Seed Number
        
    kwargs: key, value args
        Arguments passed to func constructor
    """
    def __init__(self, func, n_models=8, n_jobs=None, random_state=None, **kwargs):
        self.func = func
        self.n_models = n_models
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.random_state = random_state
        
    
    def fit(self, X, y, **fit_params):
        """
        Fit n_models constructed with func and save them
        in temporary directory
        
        Parameters
        ----------
        X, y: numpy arrays
            Input data
            
        fit_params: key, value arguments
            Arguments to pass to the fit method (epochs, batch_size...)
            
        Returns
        -------
        self
        """
        
        self.models = []
        self.weights_predictors = []
        self.encoders = []
        self.fit_params = fit_params
        self.model_paths = [os.path.join(tempfile.gettempdir(),
                                         "tmp_model_" + str(num_model) + ".h5")
                      for num_model in range(self.n_models)]
        
        
        np.random.seed(self.random_state)
        random_states = np.random.choice(2**30, self.n_models)
        
        if self.n_jobs is None:
            for path, state in zip(self.model_paths, random_states):
                self._fit(X, y, path, state)
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            parallel(delayed(self._fit)(X, y, path, state)
                     for path, state in zip(self.model_paths,
                                            random_states))
            
    
    def _fit(self, X, y, path, random_state):
        K.clear_session()
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        model = self.func(**self.kwargs)
        model.fit(X, y, **self.fit_params)
        model.save(path)
        
        
    def predict(self, X):
        """
        Return the predictions of the models
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        y_pred: array
            models predictions
        """
        if self.models == []:
            for path in self.model_paths:
                self.models.append(load_model(path))
        predictions = np.stack([model.predict(X).ravel()
                                for model in self.models], 1)
        return predictions
    
    
    def get_weight(self, X):
        """
        Return the weights (for WANN)
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        array:
            weights
        """
        if self.weights_predictors == []:
            for path in self.model_paths:
                self.weights_predictors.append(load_model(path + "_weights"))
        weights = np.stack([model.predict(X).ravel()
                            for model in self.weights_predictors], 1)
        return weights
    
    
    def get_feature(self, X):
        """
        Return the encoded features (for DANN)
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        array:
            features
        """
        if self.encoders == []:
            for path in self.model_paths:
                self.encoders.append(load_model(path + "_encoder"))
        features = np.stack([model.predict(X)
                             for model in self.encoders], -1)
        return features
    
    
def cv_split(X, y, i, split, src_index, tgt_index, tgt_train_index, param, method, fit_params, kwargs):
    test = tgt_train_index[i * split: (i + 1) * split]
    train = np.array(list(set(tgt_train_index) - set(test)))
    
    np.random.seed(0)
    tf.random.set_seed(0)

    if method == "KMM":
        model = KMM(sigma=param, **kwargs)
        model.fit(X, y, index=[src_index, tgt_index, train], **fit_params)

    if method == "WANN":            
        model = WANN(C_w=param, **kwargs)
        model.fit(X, y, index=[src_index, train], **fit_params)

    if method == "DANN":               
        model = DANN(lambda_=param, **kwargs)
        model.fit(X, y, index=[src_index, tgt_index, train], **fit_params)

    if method == "MCD":               
        model = MCD(lambda_=param, **kwargs)
        model.fit(X, y, index=[src_index, tgt_index, train], **fit_params)

    if method == "MDD":               
        model = MDD(lambda_=param, **kwargs)
        model.fit(X, y, index=[src_index, tgt_index, train], **fit_params)
        
    y_pred = model.predict(X)
    score = mean_squared_error(y[test], y_pred[test])
    return score


def cross_val(method, X, y, src_index, tgt_index, tgt_train_index,
              params, cv=5, fit_params={}, parallel=False, **kwargs):
    """
    Cross Validation function for WANN, DANN and KMM methods
    """
    best_param = params[0]
    best_score = np.inf
    for param in params:
        split = int(len(tgt_train_index) / cv)
        scores = []
        if parallel:
            parallel = Parallel(n_jobs=5)
            scores = parallel(delayed(cv_split)(X, y, i, split, src_index, tgt_index, tgt_train_index, param, method, fit_params, kwargs)
                     for i in range(cv))
            
        else:
            for i in range(cv):
                score = cv_split(X, y, i, split, src_index, tgt_index, tgt_train_index, param, method, fit_params, kwargs)
                scores.append(score)
        
        print("Cross Validation: param = %.3f | score = %.4f"%(param, np.mean(scores)))
        if np.mean(scores) <= best_score:
            best_score = np.mean(scores)
            best_param = param 
    print("Best: param = %.3f | score = %.4f"%(best_param, best_score))
    return best_param


def kin(name='kin-8fh'):
    """
    Load kin 8xy family dataset by name
    """
    folder = os.path.dirname(__file__)
    path = folder + "/../dataset/kin/"
    try:
        data = open(path + name + ".txt")
    except:
        print("Downloading kin data files...")
        download_kin(path)
        print("Kin data files successfully downloaded and saved in 'dataset/kin' folder")
        data = open(path + name + ".txt")
    A = []
    for line in data: # files are iterable
        x = str(line).replace("b", '').replace("\\n", '').replace("'", '').split(" ")
        y = []
        for u in x:
            try:
                float(u)
                y.append(float(u))
            except:
                pass
        A.append(y)
    A = np.array(A)
    data = pd.DataFrame(A[np.isfinite(A).any(1)])
    X = data.drop([data.columns[-1]], 1).__array__()
    y = data[data.columns[-1]].__array__()
    return X, y
   
    
def superconduct():
    """
    Load and preprocess superconductivity dataset
    """
    folder = os.path.dirname(__file__)
    path = folder + "/../dataset/uci/"
    try:
        data = pd.read_csv(path + "superconductivity.csv")
    except:
        print("Downloading superconductivity data file...")
        download_uci(path)
        print("Superconductivity data file successfully downloaded and saved in 'dataset/uci' folder")
        data = pd.read_csv(path + "superconductivity.csv")
    
    split_col = (data.corr().iloc[:, -1].abs() - 0.3).abs().sort_values().head(1).index[0]
    X = data.drop([data.columns[-1], split_col], 1).__array__()
    
    y = data[data.columns[-1]].__array__()

    cuts = np.percentile(data[split_col].values, [25, 50, 75])
    file = "superconduct"
    
    return data.values, X, y, cuts, list(data.columns).index(split_col)


def domain(data, cuts, split_col, i):
    """
    Get indexes of ith split of data
    """
    if i == 0:
        return np.argwhere(data[:, split_col] <= cuts[0]).ravel()
    elif i == len(cuts) or i == -1:
        return np.argwhere(data[:, split_col] > cuts[-1]).ravel()
    else:
        return np.argwhere((data[:, split_col] <= cuts[i]) & (data[:, split_col] > cuts[i-1])).ravel()
    
    
def download_uci(path):
    try:
        os.mkdir(os.path.dirname(path))
    except:
        os.mkdir(os.path.dirname(os.path.dirname(path)))
        os.mkdir(os.path.dirname(path))
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip",
                               path + "superconduct.zip")
    
    with zipfile.ZipFile(path + "superconduct.zip", 'r') as zip_ref:
        zip_ref.extractall(path)
    os.remove(path + "superconduct.zip")
    os.remove(path + "unique_m.csv")
    os.rename(path + "train.csv", path + "superconductivity.csv")
    
    
def download_kin(path):
    try:
        os.mkdir(os.path.dirname(path))
    except:
        os.mkdir(os.path.dirname(os.path.dirname(path)))
        os.mkdir(os.path.dirname(path))
    
    for kin in ['kin-8fh', 'kin-8fm', 'kin-8nh', 'kin-8nm']:
        urllib.request.urlretrieve("ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/kin-family/" + kin + ".tar.gz",
                                   path + kin + ".tar.gz")
    
        tar = tarfile.open(path + kin + ".tar.gz", "r:gz")
        tar.extractall(path + kin)
        tar.close()
    
        with gzip.open(path + kin + "/" + kin + "/Dataset.data.gz", 'rb') as f_in:
            with open(path + kin + '.txt', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(path + kin + ".tar.gz")
        
