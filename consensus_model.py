import argparse
import os

import pandas as pd

from stats import get_class_stats
from molecules_and_features import make_dataset

parser = argparse.ArgumentParser(description='Generate consensus predictions using different algorithms')
parser.add_argument('-ds', '--dataset', metavar='ds', type=str, help='training set name')
parser.add_argument('-f', '--features', metavar='f', type=str, help='features to build model with')
parser.add_argument('-dd', '--data_dir', metavar='dd', type=str, help='environmental variable of project directory')
parser.add_argument('-nc', '--name_col', metavar='nc', type=str, help='name of name column in sdf file')
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str, help='endpoint to model')
parser.add_argument('-t', '--threshold', metavar='t', type=int, help='threshold of endpoint')
parser.add_argument('-es', '--eval_set', metavar='es', type=str, help='set to make consensus predictions on')
parser.add_argument('-di', '--dont_include', metavar='dir', type=str, help='models not to include...should be a csv '
                                                                           'string')

args = parser.parse_args()
features = args.features
endpoint = args.endpoint
name = args.dataset
threshold = args.threshold
directory = os.getenv(args.data_dir)
eval_set = args.eval_set
name_col = args.name_col
dont_include = args.dont_include.split(',')

X, y = make_dataset(f'{name}.sdf', data_dir=directory, features=features, name_col=name_col, endpoint=endpoint,
                    threshold=threshold)

algorithms = [alg for alg in ['knn', 'nb', 'rf', 'svc'] if alg not in dont_include]
files = []

for alg in algorithms:
    file = os.path.join(directory, 'predictions',
                        f'{alg}_{name}_{features}_{endpoint}_{threshold}_{eval_set}_predictions.csv')
    files.append(file)

all_predictions = pd.concat([pd.read_csv(file, index_col=0, header=None) for file in files], axis=1)
all_predictions.columns = algorithms

# get the mode across the predictions as the consensus model
# in the case of evens, just predict as inactive

y_predictions = all_predictions.mode(1).iloc[:, 0]
y_true = y[y_predictions.index]
stats = get_class_stats(None, y_true, y_predictions)
y_predictions.to_csv(os.path.join(directory, 'predictions',
                                  f'consensus_{name}_{features}_{endpoint}_{threshold}_{eval_set}_predictions.csv'))
pd.Series(stats).to_csv(
    os.path.join(directory, 'results', f'consensus_{name}_{features}_{endpoint}_{threshold}_{eval_set}_results.csv'))
