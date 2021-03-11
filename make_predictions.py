import os

from argparse import ArgumentParser

import pandas as pd

from rdkit.Chem import SDWriter
from sklearn.externals.joblib import load

from molecules_and_features import generate_molecules, make_dataset

parser = ArgumentParser(description='Make predictions using trained QSAR models')
parser.add_argument('-ds', '--train_name', metavar='ds', type=str, help='training set name')
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str, help='endpoint to model')
parser.add_argument('-ev', '--env_var', metavar='dd', type=str, help='environmental variable of project directory')
parser.add_argument('-f', '--features', metavar='f', type=str, help='features to build model with')
parser.add_argument('-nc', '--name_col', metavar='nc', type=str, help='name of name column in sdf file')
parser.add_argument('-ps', '--prediction_set', metavar='ps', type=str, help='prediction set name')
parser.add_argument('-t', '--threshold', metavar='t', type=int, help='toxic threshold')
parser.add_argument('-di', '--dont_include', metavar='dir', type=str, help='models not to include...should be a csv '
                                                                           'string')

args = parser.parse_args()
data_dir = os.getenv(args.env_var)
env_var = args.env_var
features = args.features.split(',')
name_col = args.name_col
prediction_set = args.prediction_set
endpoint = args.endpoint
threshold = args.threshold
evaluate = args.evaluate
train_name = args.train_name
dont_include = args.dont_include.split(',')

X_pred = make_dataset(f'{train_name}.sdf', data_dir=data_dir, features=features, name_col=name_col, endpoint=endpoint,
                      threshold=threshold, pred_set=True)
algorithms = [alg for alg in ['ada', 'bnb', 'knn', 'nb', 'rf', 'svc'] if alg not in dont_include]
preds = []

for alg in algorithms:
    model_name = f'{alg}_{train_name}_{features}_{endpoint}_{threshold}_pipeline'
    model_file_path = os.path.join(data_dir, 'ML_models', f'{model_name}.pkl')

    if os.path.exists(model_file_path):
        loaded_model = load(model_file_path)
        probabilities = loaded_model.predict_proba(X_pred)
        preds.append(probabilities)

concatenated = pd.concat(preds, axis=0)
consensus_preds = concatenated.groupby(concatenated.index).mean()
preds.append(consensus_preds)
algorithms.append('consensus')

final_preds = pd.concat(preds, axis=1)
final_preds.columns = algorithms
final_preds[final_preds >= 0.5] = 1
final_preds[final_preds < 0.5] = 0

X, y = make_dataset(f'{train_name}.sdf', data_dir=data_dir, features=features, name_col=name_col, endpoint=endpoint,
                    threshold=threshold)
y = y.loc[X_pred.index]
y[y.isnull()] = consensus_preds
y[y in ['nan', 'NaN']] = consensus_preds

molecules = generate_molecules(os.path.join(data_dir, f'{train_name}.sdf'))

for molecule in molecules:
    if not molecule.HasProp(endpoint):
        molecule.SetProp(endpoint, str(y.loc[molecule.GetProp(name_col)]))

w = SDWriter(os.path.join(data_dir, f'{train_name}_with_predictions.sdf'))

for molecule in molecules:
    w.write(molecule)

w.close()
