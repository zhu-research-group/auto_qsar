import os
from argparse import ArgumentParser

import pandas as pd

from stats import get_class_stats
from molecules_and_features import make_dataset

parser = ArgumentParser(description='Generate consensus predictions using different descriptor sets')
parser.add_argument('-ds', '--dataset', metavar='ds', type=str, help='training set name')
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str, help='endpoint to model')
parser.add_argument('-es', '--eval_set', metavar='es', type=str, help='set to make consensus predictions on')
parser.add_argument('-ev', '--env_var', metavar='dd', type=str, help='environmental variable of project directory')
parser.add_argument('-f', '--features', metavar='f', type=str, help='features to include in consensus as csv string')
parser.add_argument('-nc', '--name_col', metavar='nc', type=str, help='name of name column in sdf file')
parser.add_argument('-t', '--threshold', metavar='t', type=int, help='threshold of endpoint')

args = parser.parse_args()
name = args.dataset
endpoint = args.endpoint
eval_set = args.eval_set
directory = os.getenv(args.env_var)
features = args.features.split(',')
name_col = args.name_col
threshold = args.threshold

X, y = make_dataset(f'{name}.sdf', data_dir=directory, features='ECFP6', name_col=name_col, endpoint=endpoint,
                    threshold=threshold)
dfs = []

for feature in features:
    file = os.path.join(directory + 'predictions',
                        f'consensus_{name}_{feature}_{endpoint}_{threshold}_{eval_set}_predictions.csv')
    dfs.append(pd.read_csv(file, index_col=0, header=None))

consensus = pd.concat(dfs, axis=1)
consensus.columns = features
y_predictions = consensus.mode(1).iloc[:, 0]
y_true = y[y_predictions.index]
stats = get_class_stats(None, y_true, y_predictions)

print(stats)
