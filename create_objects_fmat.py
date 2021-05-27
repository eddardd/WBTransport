# Wasserstein Barycenter Transport for Multi-source Domain Adaptation
#
# References
# ----------
# [1] Saenko, Kate, et al. "Adapting visual category models to new domains." European conference on computer vision.
#     Springer, Berlin, Heidelberg, 2010.

import os
import json
import argparse
import numpy as np

from scipy.io import loadmat

# For reproducibility
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help="""Path to folder containing the data files""")
parser.add_argument('--feat_type', default='Decaf')
parser.add_argument('--n_folds', default=5, type=int)
args = parser.parse_args()

data_path = args.data_path
feat_type = args.feat_type
n_folds = args.n_folds
fnames = [fname for fname in os.listdir(os.path.join(data_path, feat_type)) if 'readme' not in fname]
print({fname: i for i, fname in enumerate(fnames)})

all_X = []
all_Y = []
all_D = []
for i, fname in enumerate(fnames):
    fpath = os.path.join(data_path, feat_type, fname)
    arr = loadmat(fpath)
    all_X.append(arr['feas'])
    all_Y.append(arr['labels'])
    all_D.append(np.array([i] * len(all_X[-1])))

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_Y, axis=0).reshape(-1, 1)
d = np.concatenate(all_D, axis=0).reshape(-1, 1)
dataset = np.concatenate([X, y, d], axis=1)
np.save('./data/Objects_{}.npy'.format(feat_type), dataset)

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

with open('./data/Objects_crossval_index.json', 'w') as f:
    f.write(json.dumps(fold_dict))
