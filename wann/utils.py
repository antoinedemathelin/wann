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
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer



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
