# Wasserstein Barycenter Transport for Multi-source Domain Adaptation

import os
import json
import argparse
import numpy as np
from scipy.io import loadmat

# For reproducibility
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help="""Path to folder containing the data files""")
parser.add_argument('--n_folds', default=5, type=int)
args = parser.parse_args()

data_path = args.data_path
n_folds = args.n_folds
fnames = os.listdir(data_path)
domains = [fname.split('.mat')[0] for fname in fnames]
print(domains)

d = []
X = []
y = []
for i, fname in enumerate(fnames):
    mat = loadmat(os.path.join(data_path, fname))
    X.append(mat['fea'])
    y.append(mat['gnd'])
    d.append([i] * len(X[-1]))

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
d = np.concatenate(d, axis=0).reshape(-1, 1)

dataset = np.concatenate([X, y, d], axis=1)
np.save('./data/FR.npy', dataset)

# Create cross validation indices
fold_dict = {}

for dom in np.unique(d):
    ind_domain = np.where(d == dom)[0]
    y_d = y[ind_domain]
    fold_dict['Domain {}'.format(dom + 1)] = {}
    for fold in range(n_folds):
        fold_dict['Domain {}'.format(dom + 1)]['Fold {}'.format(fold + 1)] = []
        for cl in np.unique(y_d):
            ind_cl = np.where(y_d == cl)[0]
            samples_per_fold = len(ind_cl) // n_folds
            if fold < n_folds - 1:
                fold_dict['Domain {}'.format(dom + 1)]['Fold {}'.format(fold + 1)].append(
                    ind_domain[ind_cl[fold * samples_per_fold: (fold + 1) * samples_per_fold]]
                )
            else:
                fold_dict['Domain {}'.format(dom + 1)]['Fold {}'.format(fold + 1)].append(
                    ind_domain[ind_cl[fold * samples_per_fold:]]
                )
        fold_dict['Domain {}'.format(dom + 1)]['Fold {}'.format(fold + 1)] = np.concatenate(
            fold_dict['Domain {}'.format(dom + 1)]['Fold {}'.format(fold + 1)], axis=0
        ).tolist()

with open('./data/Faces_crossval_index.json', 'w') as f:
    f.write(json.dumps(fold_dict))
