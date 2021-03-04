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
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from methods.WANN import WANN
from methods.DANN import DANN
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
    
    
def cross_val(method, X, y, src_index, tgt_index, tgt_train_index,
              params, cv=5, fit_params={}, **kwargs):
    """
    Cross Validation function for WANN, DANN and KMM methods
    """
    best_param = params[0]
    best_score = np.inf
    for param in params:
        split = int(len(tgt_train_index) / cv)
        scores = []
        for i in range(cv):
            test = tgt_train_index[i * split: (i + 1) * split]
            train = np.array(list(set(tgt_train_index) - set(test)))
            
            if method == "KMM":
                model = KMM(sigma=param, **kwargs)
                model.fit(X, y, index=[src_index, tgt_index, tgt_train_index], **fit_params)
            
            if method == "WANN":            
                model = BaggingModels(WANN, n_models=1, n_jobs=None, random_state=0,
                                      C_w=param, **kwargs)
                model.fit(X, y, index=[src_index, train], **fit_params)
                
            if method == "DANN":
                if tgt_index is None:
                    resize_tgt_ind = np.array([train[i%len(train)]
                                           for i in range(len(src_index))])
                else:
                    resize_tgt_ind = tgt_index
                
                model = BaggingModels(DANN, n_models=1, n_jobs=None, random_state=0,
                                      lambda_=param, **kwargs)
                model.fit(X, y, index=[src_index, resize_tgt_ind, train], **fit_params)

            y_pred = model.predict(X)
            score = mean_squared_error(y[test], y_pred[test])
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


def sa(source, target):
    """
    Load sentiment analysis dataset giving source and target domain names
    """
    folder = os.path.dirname(__file__)
    path = folder + "/../dataset/sa/"
    try:
        file = open(path + "kitchen.txt")
    except:
        print("Downloading sentiment analysis data files...")
        download_sa(path)
        print("Sentiment analysis data files successfully downloaded and saved in 'dataset/sa' folder")
    return _get_Xy(source, target)



def _get_reviews_and_labels_from_txt(file, max_=3000):
    """
    Open text file of amazon reviews and add all reviews and
    ratings in two list.
    max_ gives the maximum number of reviews to extract.
    """
    file = open(file)
    
    reviews = []
    labels = []
    capture_label = False
    capture_review = False
    n_class = int(max_/4)
    count_class = {1.:0, 2.:0, 4.:0, 5.:0}
    stop = False
    for line in file:
        if capture_label and count_class[float(line)] >= n_class:
            stop = True
        if capture_label and count_class[float(line)] < n_class:
            labels.append(float(line))
            count_class[float(line)] += 1
        if capture_review:
            reviews.append(str(line))
        
        capture_label = False
        capture_review = False

        if "<rating>" in line:
            capture_label = True
        if "<review_text>" in line and stop == False:
            capture_review = True
        if "<review_text>" in line and stop == True:
            stop = False
        if len(reviews) >= max_ and len(reviews) == len(labels):
            break
    return reviews, labels


def _decontracted(phrase):
    """
    Decontract english common contraction
    """
    phrase=re.sub(r"won't","will not",phrase)
    phrase=re.sub(r"can't","can not",phrase)
    phrase=re.sub(r"n\'t","not",phrase)
    phrase=re.sub(r"\'re","are",phrase)
    phrase=re.sub(r"\'s","is",phrase)
    phrase=re.sub(r"\'d","would",phrase)
    phrase=re.sub(r"\'ll","will",phrase)    
    phrase=re.sub(r"\'t","not",phrase)
    phrase=re.sub(r"\'ve","have",phrase)
    phrase=re.sub(r"\'m","am",phrase)
    return phrase


def _preprocess_review(reviews):
    """
    Preprocess text in reviews list
    """
    stop=set(stopwords.words('english'))
    snow = SnowballStemmer('english')
    
    preprocessed_reviews=[]

    for sentence in reviews:
        sentence=re.sub(r"http\S+"," ",sentence)
        cleanr=re.compile('<.*?>')
        sentence=re.sub(cleanr,' ',sentence)
        sentence=_decontracted(sentence)
        sentence=re.sub("\S\*\d\S*"," ",sentence)
        sentence=re.sub("[^A-Za-z]+"," ",sentence)
        sentence=re.sub(r'[?|!|\'|"|#]',r' ',sentence)
        sentence=re.sub(r'[.|,|)|(|\|/]',r' ',sentence)
        sentence='  '.join(snow.stem(e.lower()) for e in sentence.split() if e.lower() not in stop)
        preprocessed_reviews.append(sentence.strip())
        
    return preprocessed_reviews


def _get_uni_and_bi_gram(reviews, max_features=1000):
    """
    Return uni and bi-gram of an ensemble of sentences
    """
    count=CountVectorizer(ngram_range=(1,2), max_features=max_features)
    return count.fit_transform(reviews)


def _get_reviews(domain):
    """
    Return preprocessed reviews and labels
    """
    folder = os.path.dirname(__file__)
    reviews, labels = _get_reviews_and_labels_from_txt(folder + "/../dataset/sa/" + domain + ".txt")    
    reviews = _preprocess_review(reviews)
    return reviews, labels


def _get_Xy(source, target):
    """
    Concatenate preprocessed source and target reviews,
    get uni and bigrams and return X, y and src and tgt indexes.
    """
    reviews_s, labels_s = _get_reviews(source)
    reviews_t, labels_t = _get_reviews(target)
    
    src_index = range(len(reviews_s))
    tgt_index = range(len(reviews_s), len(reviews_s) + len(reviews_t))
    
    reviews = reviews_s + reviews_t
    labels = labels_s + labels_t
    
    X = _get_uni_and_bi_gram(reviews).toarray()
    y = np.array(labels)
    
    return X, y, src_index, tgt_index
   
    
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
        
        
def download_sa(path):
    try:
        os.mkdir(os.path.dirname(path))
    except:
        os.mkdir(os.path.dirname(os.path.dirname(path)))
        os.mkdir(os.path.dirname(path))
    
    urllib.request.urlretrieve("https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz",
                               path + "domain_sentiment_data.tar.gz")
    urllib.request.urlretrieve("https://www.cs.jhu.edu/~mdredze/datasets/sentiment/book.unlabeled.gz",
                               path + "book.unlabeled.gz")
    tar = tarfile.open(path + "domain_sentiment_data.tar.gz", "r:gz")
    tar.extractall(path + "domain_sentiment_data")
    tar.close()
    with gzip.open(path + "book.unlabeled.gz", 'rb') as f_in:
        with open(path + 'books.txt', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    shutil.move(path + "domain_sentiment_data/sorted_data_acl/dvd/unlabeled.review",
                path + 'dvd.txt')
    shutil.move(path + "domain_sentiment_data/sorted_data_acl/electronics/unlabeled.review",
                path + 'electronics.txt')
    shutil.move(path + "domain_sentiment_data/sorted_data_acl/kitchen_&_housewares/unlabeled.review",
                path + 'kitchen.txt')
    
    os.remove(path + "book.unlabeled.gz")
    os.remove(path + "domain_sentiment_data.tar.gz")
    shutil.rmtree(path + "domain_sentiment_data")