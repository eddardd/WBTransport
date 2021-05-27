import os
import ot
import msda
import json
import warnings
import argparse
import numpy as np

from models import KerasMLP
from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from msda import JCPOTTransport
from msda import MultiSourceOTDAClassifier
from msda import DAPrincipalComponentAnalysis
from msda import TransferComponentsClassifier
from msda import ImportanceWeightedClassifier
from msda import WassersteinBarycenterTransport
from msda.barycenters import sinkhorn_barycenter

from ot.da import SinkhornTransport
from ot.da import SinkhornL1l2Transport

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--task', default="Objects", type=str,
                    help="""available tasks: MGR, MSD, Objects, Faces""")
parser.add_argument('--algorithm', default="WBT", type=str,
                    help="""Select between WBT, SinT, JCPOT, KMM or TCA""")
parser.add_argument('--data_path', default="./data/", type=str,
                    help="""Path to folder containing the data files""")
parser.add_argument('--numItermax',
                    help='Parameter for WBT algorithm. Determines the maximum number of WBT iterations',
                    default=1, type=int)
parser.add_argument('--reg_e_bar',
                    help='Parameter for WBT algorithm. Entropy penalty for the barycenter',
                    default=1e-2, type=float)
parser.add_argument('--reg_e',
                    help="Parameter for OT Algorithms. Entropy penalty for the transport to the target domain",
                    default=1e-2, type=float)
parser.add_argument('--reg_cl',
                    help="Parameter for OT Algorithms. Class Penalty",
                    default=1e-2, type=float)
parser.add_argument('--n_components',
                    help="Parameter for PCA and TCA. Determines the number of dimensions in the latent space.",
                    default=1000, type=int)
parser.add_argument('--mu',
                    help="Parameter for TCA. Regularization term.",
                    default=0.5, type=float)
args = parser.parse_args()


task = args.task
algorithm = args.algorithm
data_path = args.data_path
numItermax = args.numItermax
reg_e_bar = args.reg_e_bar
reg_e = args.reg_e
reg_cl = args.reg_cl
n_components = args.n_components
mu = args.mu

if 'WBT' in algorithm:
    algorithm_params = {'numItermax': numItermax,
                        'reg_e_bar': reg_e_bar,
                        'reg_e': reg_e,
                        'reg_cl': reg_cl}
elif algorithm == 'SinT':
    algorithm_params = {'reg_e': reg_e, 'reg_cl': reg_cl}
elif "JCPOT" in algorithm:
    algorithm_params = {'reg_e': reg_e}
elif algorithm == 'PCA':
    algorithm_params = {'n_components': n_components}
elif algorithm == 'TCA':
    algorithm_params = {'n_components': n_components, 'mu': mu}
elif algorithm == 'KMM':
    algorithm_params = {}

print('Chosen algorithm: {}\n Parameters: {}'.format(algorithm, algorithm_params))


if task == "MGR":
    dataset = np.load(os.path.join(data_path, 'MGR.npy'))
    with open(os.path.join(data_path, 'MGR_crossval_index.json'), 'r') as f:
        fold_dict = json.loads(f.read())
    clf = RandomForestClassifier(n_estimators=1000, max_depth=13, n_jobs=-1, random_state=0)
    domain_names = ['Noiseless', "buccaneer2", "destroyerengine", "f16", "factory2"]
elif task == "MSD":
    dataset = np.load(os.path.join(data_path, 'MSD.npy'))
    with open(os.path.join(data_path, 'MSD_crossval_index.json'), 'r') as f:
        fold_dict = json.loads(f.read())
    clf = RandomForestClassifier(n_estimators=1000, max_depth=13, n_jobs=-1, random_state=0)
    domain_names = ['Noiseless', "buccaneer2", "destroyerengine", "f16", "factory2"]
elif task == 'Faces':
    dataset = np.load(os.path.join(data_path, 'FR.npy'))
    with open(os.path.join(data_path, 'Faces_crossval_index.json'), 'r') as f:
        fold_dict = json.loads(f.read())
    clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    domain_names = ['PIE07', 'PIE29', 'PIE05', 'PIE09']
elif task == 'Objects':
    dataset = np.load(os.path.join(data_path, 'Objects_Decaf.npy'))
    with open('./data/Objects_crossval_index.json', 'r') as f:
        fold_dict = json.loads(f.read())
    clf = KerasMLP(lr=1e-2, l2_penalty=1e-3, n_epochs=500, verbose=False)
    domain_names = ['Webcam', 'Amazon', 'dslr', 'Caltech']
else:
    raise ValueError("Expected '--task' to be either MGR, MSD, Objects or Faces, but got {}".format(task))
# Features
X = dataset[:, :-2]

if task == "MGR" or task == "MSD":
    # For MGR and MSD only.
    # We drop the 17th column because it is constant and equal to zero.
    X = np.delete(X, 17, axis=1)
# Feature Scaling
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
# Using standardization improves overall results, but the reported accuracies were acquired using
# feature scaling.
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# Class and domain labels
y = dataset[:, -2]
m = dataset[:, -1]

domains = np.unique(m).astype(int)
if task == "MGR" or task == "MSD":
    targets = [i for i in range(1, 5)]
else:
    targets = domains


print('\n\n')
print('-' * 79)
print('|{:^77}|'.format(algorithm))
print('-' * 79)
print('|{:^25}|{:^25}|{:^25}|'.format('Domain', 'Mean Accuracy', 'Std Deviation'))
for target in targets:
    sources = [d for d in domains if d != target]
    accs = []
    for i in [1, 2, 3, 4, 5]:
        selected_folds = [j for j in range(1, 6) if j != i]
        unselected_fold = i
        inds = [
            np.concatenate([
                fold_dict['Domain {}'.format(s + 1)]['Fold {}'.format(f)] for f in selected_folds
            ]) for s in sources
        ]

        indt = np.concatenate([
            fold_dict['Domain {}'.format(target + 1)]['Fold {}'.format(f)] for f in selected_folds
        ])
        indt_ts = fold_dict['Domain {}'.format(target + 1)]['Fold {}'.format(i)]

        Xs = [X[ind] for ind in inds]
        ys = [y[ind] for ind in inds]

        cXs = np.concatenate(Xs, axis=0)
        cys = np.concatenate(ys, axis=0)

        Xt = X[indt]
        yt = y[indt]

        # Selects the algorithm for performing MSDA
        if algorithm == 'WBT':
            barycenter_solver = partial(sinkhorn_barycenter, numItermax=numItermax, reg=reg_e_bar, limit_max=1e+3, stopThr=1)
            if reg_cl > 0.0:
                transport_solver = partial(SinkhornL1l2Transport, reg_e=reg_e, reg_cl=reg_cl, norm='max')
            else:
                transport_solver = partial(SinkhornTransport, reg_e=reg_e, norm='max')
            
            baryT = msda.WassersteinBarycenterTransport(barycenter_solver=barycenter_solver,
                                                        transport_solver=transport_solver,
                                                        barycenter_initialization="random_cls")
            model = MultiSourceOTDAClassifier(clf=clf, ot_method=baryT, semi_supervised=False)
            model.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
            yp = model.predict(Xs=Xs, Xt=Xt)
            yp = yp[:len(yt)] # get only predictions on test samples
        elif algorithm == 'WBT_reg':
            yb = np.concatenate(ys, axis=0)
            barycenter_solver = partial(sinkhorn_barycenter, numItermax=numItermax, reg=reg_e_bar, limit_max=1e+3, stopThr=1,
                                        ys=ys, ybar=yb, verbose=False)
            if reg_cl > 0.0:
                transport_solver = partial(SinkhornL1l2Transport, reg_e=reg_e, reg_cl=reg_cl, norm='max')
            else:
                transport_solver = partial(SinkhornTransport, reg_e=reg_e, norm='max')
            
            baryT = msda.WassersteinBarycenterTransport(barycenter_solver=barycenter_solver,
                                                        transport_solver=transport_solver,
                                                        barycenter_initialization="random_cls")
            model = MultiSourceOTDAClassifier(clf=clf, ot_method=baryT, semi_supervised=False)
            model.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
            yp = model.predict(Xs=Xs, Xt=Xt)
            yp = yp[:len(yt)] # get only predictions on test samples
        elif algorithm == 'SinT':
            if reg_cl > 0.0:
                T = msda.SinkhornLaplaceTransport(reg_e=reg_e, reg_lap=reg_cl, similarity_param=80, norm='max')
            else:
                T = ot.da.SinkhornTransport(reg_e=reg_e, norm='max')
            T.fit(Xs=cXs, ys=cys, Xt=Xt, yt=None)
            TXs = T.transform(Xs=cXs)
            clf.fit(TXs, cys)
            yp = clf.predict(Xt)
        elif algorithm == "JCPOT":
            jcpot = JCPOTTransport(reg_e=reg_e, max_iter=numItermax, metric='sqeuclidean', tol=1e-9, norm='max')
            jcpot.fit(Xs, ys, Xt)
            TXs = np.concatenate(jcpot.transform(Xs), axis=0)

            clf.fit(TXs, cys)
            yp = clf.predict(Xt)
        elif algorithm == "JCPOT-LP":
            jcpot = JCPOTTransport(reg_e=reg_e, max_iter=numItermax, metric='sqeuclidean', tol=1e-9, norm='max')
            jcpot.fit(Xs, ys, Xt)
            TXs = jcpot.transform(Xs)
            yp = jcpot.transform_labels(ys).argmax(axis=1) + 1
        elif algorithm == "KMM":
            bandwidth = (.5 * (cXs.var() + Xt.var())) ** -1
            clf_iw = ImportanceWeightedClassifier(clf=clf, kernel_type='rbf', kernel_param=bandwidth)
            clf_iw.fit(cXs, cys, Xt)
            yp = clf_iw.predict(Xt)
        elif algorithm == "TCA":
            if task == 'Objects':
                _clf = KerasMLP(lr=1e-2, l2_penalty=1e-3, input_shape=(n_components,))
                clf_tca = TransferComponentsClassifier(clf=_clf, mu=mu, num_components=n_components,
                                                       kernel_type='linear')
            else:
                clf_tca = TransferComponentsClassifier(clf=clf, mu=mu, num_components=n_components,
                                                       kernel_type='linear')
            clf_tca.fit(cXs, cys, Xt)
            yp = clf_tca.predict(Xt)
        elif algorithm == "PCA":
            if task == 'Objects':
                _clf = KerasMLP(lr=1e-2, l2_penalty=1e-3, input_shape=(n_components,))
                clf_pca = DAPrincipalComponentAnalysis(clf=_clf, num_components=n_components,)
            else:
                clf_pca = DAPrincipalComponentAnalysis(clf=clf, num_components=n_components,)
            clf_pca.fit(cXs, cys, Xt)
            yp = clf_pca.predict(Xt)
        acc = 100 * accuracy_score(yt, yp)
        accs.append(acc)
        print('|{:^25}|{:^25}|{:^25}|'.format(domain_names[target], acc, i))    
    print('|{:^25}|{:^25}|{:^25}|'.format(domain_names[target], np.mean(accs), np.std(accs)))