import os
import ot
import msda
import json
import argparse
import numpy as np

from models import KerasMLP
from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--task', default="Objects", type=str,
                    help="""available tasks: MGR or MSD""")
parser.add_argument('--algorithm', default="WBT", type=str,
                    help="""Select between WBT, SinT, JCPOT, KMM or TCA""")
parser.add_argument('--data_path', default="./data/", type=str,
                    help="""Path to folder containing the data files""")
parser.add_argument('--numItermax', default=1, type=int)
parser.add_argument('--reg_e_bar', default=1e-2, type=float)
parser.add_argument('--reg_e', default=1e-2, type=float)
parser.add_argument('--reg_cl', default=1e-2, type=float)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--mu', default=0.5, type=float)
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

print('numItermax: {}, reg_e_bar: {}, reg_e: {}, reg_cl: {}'.format(numItermax, reg_e_bar, reg_e, reg_cl))

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
else:
    raise ValueError("Expected '--task' to be either MGR or MSD, but got {}".format(task))
# Features
X = dataset[:, :-2]

if task == "MGR" or task == "MSD":
    # For MGR and MSD only.
    # We drop the 17th column because it is constant and equal to zero.
    X = np.delete(X, 17, axis=1)
# Normalize data
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
# Class and domain labels
y = dataset[:, -2]
m = dataset[:, -1]

domains = np.unique(m).astype(int)
if task == "MGR" or task == "MSD":
    targets = [i for i in range(1, 5)]
else:
    targets = domains

for target in targets:
    sources = [d for d in domains if d != target]
    accs_baseline = []
    accs_transfer = []
    accs_reg = []
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
        cXs = np.concatenate(Xs, axis=0)
        ys = [y[ind] for ind in inds]
        cys = np.concatenate(ys, axis=0)
        Xt = X[indt]
        yt = y[indt]

        # Baseline evaluation
        clf.fit(cXs, cys)
        yp = clf.predict(Xt)
        accs_baseline.append(100 * accuracy_score(yp, yt))
        print("[Target: {:^7}] Baseline Accuracy on fold {}: {}".format(domain_names[target], i, accs_baseline[-1]))

        if algorithm == 'WBT':
            # Wasserstein Barycenter Transport
            yb = np.concatenate(ys, axis=0)
            barycenter_solver = partial(msda.barycenters.sinkhorn_barycenter,
                                        numItermax=numItermax,
                                        numInnerItermax=1000,
                                        stopThr=1,
                                        verbose=False,
                                        log=False,
                                        metric="sqeuclidean",
                                        norm="max",
                                        reg=reg_e_bar,
                                        line_search=False,
                                        limit_max=1e+3)

            if reg_cl > 0:
                transport_solver = partial(ot.da.SinkhornL1l2Transport,
                                           reg_e=reg_e,
                                           reg_cl=reg_cl,
                                           metric="sqeuclidean",
                                           norm="max",
                                           max_iter=1000,
                                           verbose=False)
            else:
                transport_solver = partial(ot.da.SinkhornTransport,
                                           reg_e=reg_e,
                                           metric="sqeuclidean",
                                           norm="max",
                                           max_iter=1000,
                                           verbose=False)

            baryT = msda.WassersteinBarycenterTransport(barycenter_solver=barycenter_solver,
                                                        transport_solver=transport_solver,
                                                        barycenter_initialization="random_cls")

            model = msda.MultiSourceOTDAClassifier(clf=clf, ot_method=baryT, semi_supervised=False)
            model.fit(Xs=Xs, Xt=Xt, ys=ys, yt=yt)

            Ypred = model.predict(Xs=Xs, Xt=Xt)
            Ytest = np.concatenate([yt, np.concatenate(ys, axis=0)], axis=0)

            accs_transfer.append(100 * accuracy_score(Ytest[:len(yt)], Ypred[:len(yt)]))
            print("[Target: {:^7}] WBT Accuracy on fold {}:      {}".format(domain_names[target], i, accs_transfer[-1]))

            barycenter_solver = partial(msda.barycenters.sinkhorn_barycenter,
                                        ys=ys,
                                        ybar=yb,
                                        numItermax=numItermax,
                                        numInnerItermax=1000,
                                        stopThr=1,
                                        verbose=False,
                                        log=False,
                                        metric="sqeuclidean",
                                        norm="max",
                                        reg=args.reg_e_bar,
                                        line_search=False,
                                        limit_max=1e+3)

            if args.reg_cl > 0:
                transport_solver = partial(ot.da.SinkhornL1l2Transport,
                                           reg_e=reg_e,
                                           reg_cl=reg_cl,
                                           metric="sqeuclidean",
                                           norm="max",
                                           max_iter=1000,
                                           verbose=False)
            else:
                transport_solver = partial(ot.da.SinkhornTransport,
                                           reg_e=reg_e,
                                           metric="sqeuclidean",
                                           norm="max",
                                           max_iter=1000,
                                           verbose=False)

            baryT = msda.WassersteinBarycenterTransport(barycenter_solver=barycenter_solver,
                                                        transport_solver=transport_solver,
                                                        barycenter_initialization="random_cls")

            model = msda.MultiSourceOTDAClassifier(clf=clf, ot_method=baryT, semi_supervised=False)
            model.fit(Xs=Xs, Xt=Xt, ys=ys, yt=yt)

            Ypred = model.predict(Xs=Xs, Xt=Xt)
            Ytest = np.concatenate([yt, np.concatenate(ys, axis=0)], axis=0)

            accs_reg.append(100 * accuracy_score(Ytest[:len(yt)], Ypred[:len(yt)]))
            print("[Target: {:^7}] WBTreg Accuracy on fold {}:   {}".format(domain_names[target], i, accs_reg[-1]))
        elif algorithm == "SinT":
            # Single Source approximation
            T = ot.da.SinkhornTransport(reg_e=reg_e, norm='max')
            T.fit(Xs=cXs, ys=cys, Xt=Xt, yt=None)
            TXs = T.transform(Xs=cXs)
            clf.fit(TXs, cys)
            Ypred = clf.predict(Xt)
            accs_transfer.append(100 * accuracy_score(yt, Ypred))
            print("[Target: {:^7}] SinT Accuracy on fold {}:     {}".format(domain_names[target], i, accs_transfer[-1]))

            # T = ot.da.SinkhornL1l2Transport(reg_e=reg_e, reg_cl=reg_cl, norm='max')
            T = msda.SinkhornLaplaceTransport(reg_e=reg_e, reg_lap=reg_cl, similarity_param=80, norm='max')
            T.fit(Xs=cXs, ys=cys, Xt=Xt, yt=None)
            TXs = T.transform(Xs=cXs)
            clf.fit(TXs, cys)
            Ypred = clf.predict(Xt)
            accs_reg.append(100 * accuracy_score(yt, Ypred))
            print("[Target: {:^7}] SinTreg Accuracy on fold {}:  {}".format(domain_names[target], i, accs_reg[-1]))
        elif algorithm == "JCPOT":
            # JCPOT-based transport
            jcpot = msda.JCPOTTransport(reg_e=reg_e, max_iter=numItermax, metric='sqeuclidean', tol=1e-9, norm='max')
            jcpot.fit(Xs, ys, Xt)
            TXs = jcpot.transform(Xs)
            Tys = jcpot.transform_labels(ys).argmax(axis=1) + 1
            TXs = np.concatenate(TXs, axis=0)

            clf.fit(TXs, cys)
            yp = clf.predict(Xt)
            accs_transfer.append(100 * accuracy_score(yt, yp))
            accs_reg.append(100 * accuracy_score(yt, Tys))

            print("[Target: {:^7}] JCPOT Accuracy on fold {}:    {}".format(domain_names[target], i, accs_transfer[-1]))
            print("[Target: {:^7}] JCPOT-LP Accuracy on fold {}: {}".format(domain_names[target], i, accs_reg[-1]))
        elif algorithm == "KMM":
            # Importance Weighting Classifier
            bandwidth = (.5 * (cXs.var() + Xt.var())) ** -1
            clf_iw = msda.ImportanceWeightedClassifier(clf=clf, kernel_type='rbf', kernel_param=bandwidth)
            clf_iw.fit(cXs, cys, Xt)
            yp = clf_iw.predict(Xt)
            accs_transfer.append(100 * accuracy_score(yt, yp))
            accs_reg.append(-1)
            print("[Target: {:^7}] KMM Accuracy on fold {}:      {}".format(domain_names[target], i, accs_transfer[-1]))
        elif algorithm == "TCA":
            # Transfer Component Analysis Classifier
            clf_tca = msda.TransferComponentsClassifier(clf=clf, mu=mu, num_components=n_components,
                                                        kernel_type='linear')
            clf_tca.fit(cXs, cys, Xt)
            yp = clf_tca.predict(Xt)
            accs_transfer.append(100 * accuracy_score(yt, yp))
            accs_reg.append(-1)
            print("[Target: {:^7}] TCA Accuracy on fold {}:      {}".format(domain_names[target], i, accs_transfer[-1]))
        elif algorithm == "TargetOnly":
            Xt_ts = X[indt_ts]
            yt_ts = y[indt_ts]

            # Baseline evaluation
            clf.fit(Xt, yt)
            yp = clf.predict(Xt_ts)
            accs_transfer.append(100 * accuracy_score(yp, yt_ts))
            print("[Target: {:^7}] Target-Only Accuracy on fold  {}: {}".format(domain_names[target], i, accs_transfer[-1]))
        else:
            raise ValueError("""expected '--algorithm' to be either WBT, SinT, JCPOT, IW, TCA, or TargetOnly,
                             but got {}""".format(algorithm))

    print("[Target: {:^7}] Accuracy Avg [baseline]:    {:<10.5f}".format(domain_names[target], np.mean(accs_baseline)))
    print("[Target: {:^7}] Accuracy Var [baseline]:    {:<10.5f}".format(domain_names[target], np.std(accs_baseline)))
    print("[Target: {:^7}] Accuracy Avg [transfer]:    {:<10.5f}".format(domain_names[target], np.mean(accs_transfer)))
    print("[Target: {:^7}] Accuracy Var [transfer]:    {:<10.5f}".format(domain_names[target], np.std(accs_transfer)))
    print("[Target: {:^7}] Accuracy Avg [regularized]: {:<10.5f}".format(domain_names[target], np.mean(accs_reg)))
    print("[Target: {:^7}] Accuracy Var [regularized]: {:<10.5f}".format(domain_names[target], np.std(accs_reg)))
