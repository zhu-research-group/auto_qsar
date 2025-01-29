import os
import pandas as pd
import numpy as np

# project imports
# basic sklearn stuff
from sklearn import pipeline
from sklearn import model_selection

# preprocessing/data selection
from sklearn.preprocessing import StandardScaler

# unsupervised ml
from sklearn.decomposition import PCA

#supervised ml
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import xgboost as xgb
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR, SVC
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


seed = 42
def split_train_test(X, y, n_split, test_set_size, seed):
    """ Splits data into training and test sets"""

    assert X.shape[0] == y.shape[0], 'The lengths of X and y do not match X == {}, y == {}.'.format(X.shape[0],
                                                                                                    y.shape[0])

    # split X into training sets and test set the size of test_set_size
    if test_set_size != 0:
        batch_size = int(y.shape[0] * (1 - test_set_size) // n_split)  # calculating batch size
        train_size = int(batch_size * n_split)

        X_train_tmp, X_test, y_train_tmp, y_test = model_selection.train_test_split(X,y,train_size=train_size,random_state=seed)
    else:
        X_train_tmp = X
        y_train_tmp = y
        X_test = None
        y_test = None

    cv = model_selection.KFold(shuffle=True, n_splits=n_split, random_state=seed)
    valid_idx = []  # indexes for new train dataset
    for (_, valid) in cv.split(X_train_tmp, y_train_tmp):
        valid_idx += valid.tolist()

    X_train = X_train_tmp.iloc[valid_idx]
    y_train = y_train_tmp.iloc[valid_idx]

    return X_train, y_train, X_test, y_test

# hyperparameters for classification and regression models
""" Algs is a list of tuples:
item 1 is the alg name
item 2 is a scikit-learn machine learning classifiers
item 3 is the paramaters to grid search through
"""

CLASSIFIER_ALGS = [
    ('rf', RandomForestClassifier(max_depth=10, # max depth 10 to prevent overfitting
                                  class_weight='balanced',
                                  random_state=seed), {'rf__n_estimators':[5, 10, 25]}),
    ('nb', GaussianNB(), {}),
    ('knn', KNeighborsClassifier(metric='euclidean'), {'knn__n_neighbors':[1, 3, 5],
                                                        'knn__weights': ['uniform', 'distance']}),
    ('svc', SVC(probability=True,
                class_weight='balanced',
                random_state=seed), {'svc__kernel': ['rbf'],
                                     'svc__gamma': ['auto', 1e-2, 1e-3],
                                     'svc__C': [1, 10, 100]}),
    ('bnb', BernoulliNB(alpha=1.0), {}),
    ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.9, random_state=seed), {})
]

REGRESSOR_ALGS = [
    ('rfr', RandomForestRegressor(random_state=seed,
                                  n_jobs=-1), {'rfr__n_estimators':[200], # [200,300,400]
                                               'rfr__min_samples_leaf': [1, 2, 4], 
                                               'rfr__max_depth': [10, 15]}) # max depth prevent overfitting
    ,('svr', SVR(), {'svr__kernel': ['poly','rbf'],
                     'svr__gamma': ['auto', 1e-2, 1e-3],
                     'svr__C': [1, 10, 100]})
    ,
    ('xgbr', GradientBoostingRegressor(random_state=seed), {'xgbr__subsample': [0.7],
                                                            'xgbr__learning_rate': [0.05], # [0.1, 0.05]
                                                             'xgbr__max_depth' : [10], # [10, 15]
                                                             'xgbr__n_estimators': [100]}) #[100, 400]
    # ,
    # ('plsr', PLSRegression(),  {'plsr__n_components':range(2,10)})
]


def get_regress_stats(y, predictions):
    """
    :param predictions: predicted endpoints
    :param y: correct endpoint
    :return:
    """
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    return {'r2': r2, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}

def cal_fold_error(y, predictions):
    """
    :param predictions: predicted endpoints
    :param y: correct endpoint
    :return:
    """
    fold = abs(predictions/y)
    # calculaye geometrical mean fold error
    gmfe = np.exp(np.mean(abs(np.log(fold))))
    # calculate fold error for each compounds and count how many are below 2-fold, 3-fold, 5-fold error
    fold_es = []
    for i in range(len(y)):
        fold_e = max(fold.iloc[i], 1/fold.iloc[i])
        fold_es.append(fold_e)
    fold_ess = pd.Series(fold_es)
    lf2 = fold_ess[fold_ess<2].count()/len(y)
    lf3 = fold_ess[fold_ess<3].count()/len(y)
    lf5 = fold_ess[fold_ess<5].count()/len(y)

    return {'gmfe': gmfe, 'lt2fold': lf2, 'lt3fold': lf3, 'lt5fold': lf5}