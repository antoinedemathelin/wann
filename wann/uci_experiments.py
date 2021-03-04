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

from utils import superconduct, domain, BaggingModels, cross_val
from methods.KLIEP import KLIEP
from methods.KMM import KMM
from methods.TrAdaBoostR2_keras import TwoStageTrAdaBoostR2
from methods.WANN import WANN
from methods.DANN import DANN


def run_uci_experiments(method, get_base_model, get_encoder, get_task,
                        C, C_w, epochs, batch_size, n_models, n_jobs,
                        n_target_labeled, random_state, save, **kwargs):
    """
    Run experiments on superconductivity dataset
    
    Parameters
    ----------
    method: str
        name of the method used: should one of the following:
            - NoReweight
            - TrAdaBoost
            - WANN
            - SrcOnly
            - TgtOnly
            
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
        projecting constant for WANN weighting network
        
    lambda_: float
        DANN trade-off parameter
        
    epochs: int
        number of epochs
        
    batch_size: int
        size of the batches
        
    n_models: int
        number of bagged models
        
    n_jobs: int
        number of jobs to run in parallel, if n_jobs=None
        no paralllel computing is done.
        
    n_target_labeled: int
        number of training target labeled data
        
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
    save_path = folder + "/../dataset/results/" + "uci_" + method + "_" + str(random_state)
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
    
    for source in [0, 1, 2, 3]:
        
        print("############# " + str(source) + " #############")
        
        target_list = [0, 1, 2, 3]
        target_list.remove(source)
        for target in target_list:

            print("--------- %s ----------"%str(target))

            data, X, y, cuts, split_col = superconduct()
            shape = X.shape[1]

            src_index = domain(data, cuts, split_col, source)
            tgt_index = domain(data, cuts, split_col, target)
            
            np.random.seed(0)
            tgt_train_index, tgt_test_index = train_test_split(tgt_index, train_size=n_target_labeled)
            train_index = np.concatenate((src_index, tgt_train_index))
            
            std_sc = StandardScaler()
            std_sc.fit(X[train_index])
            X = std_sc.transform(X)
            y = (y - y[train_index].mean()) / y[train_index].std()

            base_estimator = BaggingModels(func=get_base_model,
                                           n_models=n_models,
                                           n_jobs=n_jobs,
                                           shape=shape,
                                           C=C,
                                           random_state=random_state)
            fit_params = dict(epochs=epochs, batch_size=batch_size, verbose=0)

            if method == "SrcOnly":
                model = copy.deepcopy(base_estimator)
                model.fit(X[src_index], y[src_index], **fit_params)
            
            if method == "TgtOnly":
                model = copy.deepcopy(base_estimator)
                model.fit(X[tgt_train_index], y[tgt_train_index], **fit_params)
            
            if method == "NoReweight":
                model = copy.deepcopy(base_estimator)
                model.fit(X[train_index], y[train_index], **fit_params)
                
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
                        lambda_ = DICT_DANN[str(source) + "_" + str(target)]
                    except:
                        lambda_ = cross_val("DANN", X, y, src_index, None, tgt_train_index,
                                            params=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                                            fit_params=fit_params, cv=5,
                                            get_encoder=get_encoder, get_task=get_task)
                        try:
                            DICT_DANN[str(source) + "_" + str(target)] = lambda_
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
                resize_tgt_ind = np.array([tgt_train_index[i%len(tgt_train_index)]
                                           for i in range(len(src_index))])
                model.fit(X, y, index=[src_index, resize_tgt_ind, tgt_train_index], **fit_params)
                
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
    
    DICT_DANN = {}
    
    get_base_model=get_base_model
    get_encoder=get_encoder
    get_task=get_task
    C=1
    C_w=0.1
    lambda_=None
    epochs=200
    batch_size=1000
    n_models=1
    n_jobs=None
    n_target_labeled=10
    save=True
    
    for method in ["WANN", "NoReweight", "TgtOnly", "SrcOnly", "TrAdaBoost", "DANN"]:
        for state in range(10):
            run_uci_experiments(method=method,
                                get_base_model=get_base_model,
                                get_encoder=get_encoder,
                                get_task=get_task,
                                C=C,
                                C_w=C_w,
                                lambda_=lambda_,
                                epochs=epochs,
                                batch_size=batch_size,
                                n_models=n_models,
                                n_jobs=n_jobs,
                                n_target_labeled=n_target_labeled,
                                random_state=state,
                                save=save);