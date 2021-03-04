import os
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras import backend as K

from utils import sa, BaggingModels, cross_val
from methods.KLIEP import KLIEP
from methods.KMM import KMM
from methods.TrAdaBoostR2_keras import TwoStageTrAdaBoostR2
from methods.WANN import WANN
from methods.DANN import DANN

def run_sa_experiments(method, get_base_model, get_encoder, get_task,
                       C, C_w, lambda_, sigma, epochs, batch_size, n_models, n_jobs,
                       n_source, n_target_unlabeled, n_target_labeled, n_target_test,
                       random_state, save):
    """
    Run experiments on amazon review sentiment analysis dataset
    
    Parameters
    ----------
    method: str
        name of the method used: should one of the following:
            - NoReweight
            - KMM
            - KLIEP
            - GDM
            - TrAdaBoost
            - WANN
            
    get_base_model: callable
        constructor for the base learner, should takes
        C, shape, activation and name as arguments
        
    get_encoder: callable
        constructor for the DANN encoder network
        
    get_task: callable
        constructor for the DANN task network
        
    C: float
        projecting constant for networks (args of get_base_model)
        
    C_w: float
        projecting constant for WANN
        
    lambda_: float
        DANN trade-off parameter
        
    sigma: float
        kernel bandwith for KMM
        
    epochs: int
        number of epochs
        
    batch_size: int
        size of the batches
        
    n_models: int
        number of bagged models
        
    n_jobs: int
        number of jobs to run in parallel, if n_jobs=None
        no paralllel computing is done.
        
    n_source: int
        number of training source labeled data
        
    n_target_unlabeled: int
        number of training target unlabeled data
        
    n_target_labeled: int
        number of training target labeled data
        
    n_target_test: int
        number of testing target data to compute mse
        
    random_state: int
        seed number of the experiment
        
    save: boolean
        whether to save results in csv or not
        
    Returns
    -------
    df: DataFrame
        dataframe containing mse scores
    """
    print("Experiment for method: %s"%method)
    print("\n")
    
    folder = os.path.dirname(__file__)
    save_path = folder + "/../dataset/results/" + "sa_" + method + "_" + str(random_state)
    df = pd.DataFrame(columns=['state', 'method', 'source', 'target', 'score'])
    if save:
        try:
            df.to_csv(save_path + ".csv")
        except:
            try:
                os.mkdir(folder + "/../dataset/results")
            except:
                os.mkdir(folder + "/../dataset")
                os.mkdir(folder + "/../dataset/results")
            df.to_csv(save_path + ".csv")
    
    for source in ['dvd', 'books', 'electronics', 'kitchen']:
        
        print("############# " + source + " #############")
        
        target_list = ['dvd', 'books', 'electronics', 'kitchen']
        target_list.remove(source)
        for target in target_list:

            print("--------- %s ----------"%target)

            X, y, src_index, tgt_index = sa(source, target)
            y = (y-y[src_index].mean())/y[src_index].std()
            shape = X.shape[1]

            np.random.seed(0)
            src_index = np.random.choice(src_index, n_source, replace=False)
            tgt_index, tgt_test_index = train_test_split(tgt_index,
                                                         train_size=n_target_unlabeled,
                                                         test_size=n_target_test)
            tgt_train_index = np.random.choice(tgt_index, n_target_labeled, replace=False)
            train_index = np.concatenate((src_index, tgt_train_index))

            base_estimator = BaggingModels(func=get_base_model,
                                           n_models=n_models,
                                           n_jobs=n_jobs,
                                           shape=shape,
                                           C=C,
                                           random_state=random_state)
            fit_params = dict(epochs=epochs, batch_size=batch_size, verbose=0)

            if method == "NoReweight":
                model = copy.deepcopy(base_estimator)
                model.fit(X[train_index], y[train_index], **fit_params)
            
            if method == "KMM":
                if sigma is None:
                    try:
                        sigma = DICT_KMM[source[-2:] + "_" + target[-2:]]
                    except:
                        sigma = cross_val("KMM", X, y, src_index, tgt_index, tgt_train_index,
                                          params=[2**(i-5) for i in range(10)],
                                          fit_params=fit_params, cv=5,
                                          estimator=base_estimator)
                        try:
                            DICT[source[-2:] + "_" + target[-2:]] = sigma
                        except:
                            pass
                print("sigma: %.3f"%sigma)
                model = KMM(base_estimator, sigma=sigma)
                model.fit(X, y, index=[src_index, tgt_index, tgt_train_index], **fit_params)
                
            if method == "KLIEP":
                model = KLIEP(base_estimator)
                model.fit(X, y, index=[src_index, tgt_index, tgt_train_index], **fit_params)
                
            if method == "TrAdaBoost":
                model = TwoStageTrAdaBoostR2(func=get_base_model,
                                             random_state=random_state,
                                             n_jobs=n_jobs,
                                             C=C,
                                             shape=X.shape[1])
                model.fit(X, y, [src_index, tgt_train_index], **fit_params)
                
            if method == "WANN":
                model = BaggingModels(WANN,
                                      get_base_model=get_base_model,
                                      C=C,
                                      C_w=C_w,
                                      n_models=n_models,
                                      n_jobs=n_jobs,
                                      random_state=random_state)
                model.fit(X, y, index=[src_index, tgt_train_index], **fit_params)
                
            if method == "DANN":
                if lambda_ is None:
                    try:
                        lambda_ = DICT_DANN[source[-2:] + "_" + target[-2:]]
                    except:
                        lambda_ = cross_val("DANN", X, y, src_index, tgt_index, tgt_train_index,
                                            params=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                                            fit_params=fit_params, cv=5,
                                            get_encoder=get_encoder, get_task=get_task)
                        try:
                            DICT_DANN[source[-2:] + "_" + target[-2:]] = lambda_
                        except:
                            pass
                print("lambda: %.3f"%lambda_)
                model = BaggingModels(DANN,
                                      get_encoder=get_encoder,
                                      get_task=get_task,
                                      C=C,
                                      lambda_=lambda_,
                                      n_models=n_models,
                                      n_jobs=n_jobs,
                                      random_state=random_state)
                model.fit(X, y, index=[src_index, tgt_index, tgt_train_index], **fit_params)
                
            y_pred = model.predict(X)
            score = mean_squared_error(y[tgt_test_index], y_pred[tgt_test_index])
            _line = pd.DataFrame([[random_state, method, source, target, score]],
                                 columns=['state', 'method', 'source', 'target', 'score'])
            df = df.append(_line, ignore_index=True)
            if save:
                df.to_csv(save_path + ".csv")
            print('Target_score: %.3f'%score)
            K.clear_session()
    return df
            
            
            
def get_base_model(shape, activation=None, C=1, name="BaseModel"):
    inputs = Input(shape=(shape,))
    modeled = Dense(100, activation='relu',
                         kernel_constraint=MinMaxNorm(0, C),
                         bias_constraint=MinMaxNorm(0, C))(inputs)
    modeled = Dropout(0.5)(modeled)
    modeled = Dense(10, activation='relu',
                         kernel_constraint=MinMaxNorm(0, C),
                         bias_constraint=MinMaxNorm(0, C))(modeled)
    modeled = Dropout(0.2)(modeled)
    modeled = Dense(1, activation=activation,
                    kernel_constraint=MinMaxNorm(0, C),
                    bias_constraint=MinMaxNorm(0, C))(modeled)
    model = Model(inputs, modeled, name=name)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def get_encoder(shape, C=1, name="encoder"):
    inputs = Input(shape=(shape,))
    modeled = Dense(100, activation='relu',
                         kernel_constraint=MinMaxNorm(0, C),
                         bias_constraint=MinMaxNorm(0, C))(inputs)
    modeled = Dropout(0.5)(modeled)
    modeled = Dense(10, activation='relu',
                         kernel_constraint=MinMaxNorm(0, C),
                         bias_constraint=MinMaxNorm(0, C))(modeled)
    modeled = Dropout(0.2)(modeled)
    model = Model(inputs, modeled)
    model.compile(optimizer="adam", loss='mean_squared_error')
    return model


def get_task(shape, C=1, activation=None, name="task"):
    inputs = Input(shape=(shape,))
    modeled = Dense(1, activation=activation,
                         kernel_constraint=MinMaxNorm(0, C),
                         bias_constraint=MinMaxNorm(0, C))(inputs)
    model = Model(inputs, modeled)
    model.compile(optimizer="adam", loss='mean_squared_error')
    return model



if __name__ == "__main__":
    
    DICT_KMM = {}
    DICT_DANN = {}
    
    get_base_model=get_base_model
    get_encoder=get_encoder
    get_task=get_task
    C=1
    C_w=0.2
    lambda_=None
    sigma=None
    epochs=200
    batch_size=64
    n_models=1
    n_jobs=None
    n_source=700
    n_target_unlabeled=700
    n_target_labeled=50
    n_target_test=1000
    save=True
    
    for method in ["WANN", "NoReweight", "TrAdaBoost", "KLIEP", "KMM", "DANN"]:
        for state in range(10):
            run_sa_experiments(method=method,
                               get_base_model=get_base_model,
                               get_encoder=get_encoder,
                               get_task=get_task,
                               C=C,
                               C_w=C_w,
                               lambda_=lambda_,
                               sigma=sigma,
                               epochs=epochs,
                               batch_size=batch_size,
                               n_models=n_models,
                               n_jobs=n_jobs,
                               n_source=n_source,
                               n_target_unlabeled=n_target_unlabeled,
                               n_target_labeled=n_target_labeled,
                               n_target_test=n_target_test,
                               random_state=state,
                               save=save);