from config import directory_check
from classic_ml_regressor import split_train_test, REGRESSOR_ALGS, get_regress_stats, cal_fold_error

from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import make_scorer
import joblib

import pandas as pd
import os
import argparse
import sys
import numpy as np
import math

print('running')
parser = argparse.ArgumentParser(description='Build QSAR Models')

parser.add_argument('-ds', '--dataset', metavar='ds', type=str, help='training set name')
parser.add_argument('-f', '--features', metavar='f', type=str, help='features to build model with')
parser.add_argument('-ns', '--n_splits', metavar='ns', type=int, help='number of splits for cross validation')
parser.add_argument('-dd', '--data_dir', metavar='dd', type=str, help='project directory')
parser.add_argument('-nc', '--name_col', metavar='nc', type=str, help='name of name column in sdf file')
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str, help='end point to model')
parser.add_argument('-ts', '--test_set_size', metavar='ts', type=float, help='size of the test set')

args = parser.parse_args()
dataset = args.dataset
features = args.features
n_splits = args.n_splits
seed = 42
data_dir = args.data_dir
name_col = args.name_col
endpoint = args.endpoint
test_set_size = args.test_set_size

# Check to see if necessary directories are present and if not, create them
directory_check(data_dir)

regress_scoring = {'MAPE': make_scorer(mean_absolute_percentage_error, greater_is_better=False), 'r2_score': make_scorer(r2_score)}
# read the file containing descriptor + endpoint
df = pd.read_csv(os.path.join(data_dir, 'caches', f'{dataset}_{features}_{endpoint}_regression.csv'),index_col=0)
X, y_regress = df.iloc[:,:-1], df.iloc[:,-1]

X_train, y_train_regress, X_test, y_test_regress = split_train_test(X, y_regress, n_splits, test_set_size, seed)

cv = model_selection.KFold(shuffle=True, n_splits=n_splits, random_state=seed)

for name, clf, params in REGRESSOR_ALGS:
    pipe = pipeline.Pipeline([
        ('scaler', StandardScaler()),
        (name, clf)])

    grid_search = model_selection.GridSearchCV(pipe,
                                               param_grid=params,
                                               cv=cv,
                                               scoring=regress_scoring,
                                               refit='r2_score')

    grid_search.fit(X_train, y_train_regress)

    best_estimator = grid_search.best_estimator_
    best_estimator.fit(X_train, y_train_regress)
    print("=======Results for {}=======".format(name))

    cv_predictions = cross_val_predict(best_estimator, X_train, y_train_regress, cv=cv)
    cv_stats = get_regress_stats(y_train_regress, cv_predictions)
    cv_fold = cal_fold_error(y_train_regress, cv_predictions)
    cv_stats.update(cv_fold)

    # record the predictions and the results
    cv_df = pd.DataFrame(cv_predictions, index=y_train_regress.index, columns=['predicted'])
    cv_df['actual'] = y_train_regress
    cv_df.to_csv(os.path.join(data_dir, 'predictions',
                      f'{name}_{dataset}_{features}_{endpoint}_{n_splits}fcv_predictions.csv'))

    # print the 5-fold cv accuracy
    print("Best {}-fcv r2_score: {}".format(n_splits, grid_search.best_score_))

    print("All k-fold results:")
    for score, val in cv_stats.items():
        print(score, val)

    # write 5-fold cv results to csv
    pd.Series(cv_stats).to_csv(
        os.path.join(data_dir, 'results', f'{name}_{dataset}_{features}_{endpoint}_{n_splits}fcv_results.csv'))

    # if use the lg transformed endpoint, convert to the original scale and do the evaluation
    if 'lg' in endpoint:
        cv_predictions_orig = 10**(cv_predictions)
        cv_stats_orig = get_regress_stats(10**y_train_regress, cv_predictions_orig)
        cv_fold_orig = cal_fold_error(10**y_train_regress, cv_predictions_orig)
        cv_stats_orig.update(cv_fold_orig)

        print("All k-fold results in original scale:")
        for score, val in cv_stats_orig.items():
            print(score, val)

    # make predictions on training data, then test data
    train_preds = best_estimator.predict(X_train)

    train_df = pd.DataFrame(train_preds, index=y_train_regress.index, columns=['predicted'])
    train_df['actual'] = y_train_regress
    train_df.to_csv(os.path.join(data_dir, 'predictions',
                                 f'{name}_{dataset}_{features}_{endpoint}_train_predictions.csv'))

    #print("Training data prediction results: ")
    train_stats = get_regress_stats(y_train_regress, train_preds)
    train_fold = cal_fold_error(y_train_regress, train_preds)
    train_stats.update(train_fold)
    #for score, val in train_stats.items():
    #    print(score, val)    # write training predictions and stats also

    if test_set_size != 0:
        test_preds = best_estimator.predict(X_test)
    else:
        test_preds = None

    if test_preds is not None:
        test_df = pd.DataFrame(test_preds, index=y_test_regress.index, columns = ['predicted'])
        test_df['actual'] = y_test_regress
        test_df.to_csv(
            os.path.join(data_dir, 'predictions', f'{name}_{dataset}_{features}_{endpoint}_test_predictions.csv'))

        print("Test data results")
        test_stats = get_regress_stats(y_test_regress, test_preds)
        test_fold = cal_fold_error(y_test_regress, test_preds)
        test_stats.update(test_fold)
        for score, val in test_stats.items():
            print(score, val)
        if 'lg' in endpoint:
            test_preds_orig = 10**(test_preds)
            test_stats_orig = get_regress_stats(10**y_test_regress, test_preds_orig)
            test_fold_orig = cal_fold_error(10**y_test_regress, test_preds_orig)
            test_stats_orig.update(test_fold_orig)
            for score, val in test_stats_orig.items():
                print(score, val)
    else:
        test_stats ={}
        for score, val in train_stats.items():
            test_stats[score] = np.nan

    pd.DataFrame([train_stats, test_stats], index=['Training', 'Test']).to_csv(
        os.path.join(data_dir, 'results', f'{name}_{dataset}_{features}_{endpoint}_train_test_results.csv'))

    # save model
    save_dir = os.path.join(data_dir, 'ML_models', f'{name}_{dataset}_{features}_{endpoint}_pipeline.pkl')
    joblib.dump(best_estimator, save_dir)


