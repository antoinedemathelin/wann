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
from tensorflow.keras.optimizers import Adam

from utils import kin, BaggingModels, cross_val
from methods.KLIEP import KLIEP
from methods.KMM import KMM
from methods.TrAdaBoostR2_keras import TwoStageTrAdaBoostR2
from methods.WANN import WANN
from methods.BalancedWeighting import BalancedWeighting


def run_kin_experiments(method, get_base_model,
                        C, C_w, sigma, epochs, batch_size, n_models, n_jobs,
                        n_source, n_target_unlabeled, n_target_labeled,
                        n_target_test, random_state, save):
    """
    Run experiments on kin 8xy datasets
    
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
            - BalancedWeighting
            
    get_base_model: callable
        constructor for the base learner, should takes
        C, shape, activation and name as arguments
        
    C: float
        projecting constant for networks (args of get_base_model)
        
    C_w: float
        projecting constant for WANN
        
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
    print(" ")
    print("Experiment for method: %s"%method)
    print(" ")   
    bug = False
    folder = os.path.dirname(__file__)
    save_path = folder + "/../dataset/results/" + "kin_" + method + "_" + str(random_state)
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
    
    for source in ['kin-8fh', 'kin-8fm', 'kin-8nh', 'kin-8nm']:
        
        print("############# " + source + " #############")
        
        target_list = ['kin-8fh', 'kin-8fm', 'kin-8nh', 'kin-8nm']
        target_list.remove(source)
        for target in target_list:

            print("--------- %s ----------"%target)

            Xs, ys = kin(source)
            Xt, yt = kin(target)
            X = np.concatenate((Xs, Xt))
            y = np.concatenate((ys, yt))
            shape = X.shape[1]

            np.random.seed(0)
            src_index = np.random.choice(len(Xs), n_source, replace=False)
            tgt_index, tgt_test_index = train_test_split(range(len(Xs), len(Xs)+len(Xt)),
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
                model = KMM(base_estimator, sigma=sigma)
                model.fit(X, y, index=[src_index, tgt_index, tgt_train_index], **fit_params)
                
            if method == "KLIEP":
                model = KLIEP(base_estimator)
                model.fit(X, y, index=[src_index, tgt_index, tgt_train_index], **fit_params)
                
            if method == "TrAdaBoost":
                model = TwoStageTrAdaBoostR2(func=get_base_model,
                                             random_state=random_state,
                                             n_jobs=5,
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
                
                
            if method == "BalancedWeighting":                
                model = BaggingModels(BalancedWeighting,
                                      get_base_model=get_base_model,
                                      C=C,
                                      n_models=n_models,
                                      n_jobs=n_jobs,
                                      random_state=random_state)
                model.fit(X, y, index=[src_index, tgt_train_index], **fit_params)

                
            y_pred = model.predict(X)
            score = mean_squared_error(y[tgt_test_index], y_pred[tgt_test_index])
            _line = pd.DataFrame([[random_state, method, source, target, score]],
                                columns=['state', 'method', 'source', 'target', 'score'])
            df = df.append(_line, ignore_index=True)
            if save:
                df.to_csv(save_path + ".csv")
            print('Target_score: %.4f'%score)
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


if __name__ == "__main__":
    
    get_base_model=get_base_model
    C=1
    C_w=1
    sigma=0.1
    epochs=300
    batch_size=32
    n_models=1
    n_jobs=None
    n_source=200
    n_target_unlabeled=200
    n_target_labeled=10
    n_target_test=400
    save=True
              
    for state in range(10):
        for method in ["KLIEP", "KMM", "WANN", "NoReweight", "BalancedWeighting", "TrAdaBoost"]:
            run_kin_experiments(method=method,
                                    get_base_model=get_base_model,
                                    C=C,
                                    C_w=C_w,
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