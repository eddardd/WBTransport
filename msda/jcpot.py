import ot
import numpy as np


def jcpot_barycenter(Xs, Ys, Xt, reg, metric='sqeuclidean', norm=None,
                     numItermax=100, stopThr=1e-6, verbose=False, log=False,
                     **kwargs):
    nbclasses = len(np.unique(Ys[0]))
    nbdomains = len(Xs)

    # log dictionary
    if log:
        log = {'niter': 0, 'err': [], 'M': [], 'D1': [], 'D2': [], 'gamma': []}

    K = []
    M = []
    D1 = []
    D2 = []

    # For each source domain, build cost matrices M, Gibbs kernels K and corresponding matrices D_1 and D_2
    for d in range(nbdomains):
        dom = {}
        nsk = Xs[d].shape[0]  # get number of elements for this domain
        dom['nbelem'] = nsk
        classes = np.unique(Ys[d])  # get number of classes for this domain

        # format classes to start from 0 for convenience
        if np.min(classes) != 0:
            Ys[d] = Ys[d] - np.min(classes)
            classes = np.unique(Ys[d])

        # build the corresponding D_1 and D_2 matrices
        Dtmp1 = np.zeros((nbclasses, nsk))
        Dtmp2 = np.zeros((nbclasses, nsk))

        for c in classes:
            nbelemperclass = np.sum(Ys[d] == c)
            if nbelemperclass != 0:
                Dtmp1[int(c), Ys[d] == c] = 1.
                Dtmp2[int(c), Ys[d] == c] = 1. / (nbelemperclass)
        D1.append(Dtmp1)
        D2.append(Dtmp2)

        # build the cost matrix and the Gibbs kernel
        Mtmp = ot.dist(Xs[d], Xt, metric=metric)
        if norm is not None:
            Mtmp = ot.utils.cost_normalization(Mtmp, norm=norm)
        M.append(Mtmp)

        Ktmp = np.empty(Mtmp.shape, dtype=Mtmp.dtype)
        np.divide(Mtmp, -reg, out=Ktmp)
        np.exp(Ktmp, out=Ktmp)
        K.append(Ktmp)

    # uniform target distribution
    a = ot.unif(np.shape(Xt)[0])

    cpt = 0  # iterations count
    err = 1
    old_bary = np.ones((nbclasses))

    while (err > stopThr and cpt < numItermax):

        bary = np.zeros((nbclasses))

        # update coupling matrices for marginal constraints w.r.t. uniform target distribution
        for d in range(nbdomains):
            K[d] = ot.bregman.projC(K[d], a)
            other = np.sum(K[d], axis=1)
            bary = bary + np.log(np.dot(D1[d], other)) / nbdomains

        bary = np.exp(bary)

        # update coupling matrices for marginal constraints w.r.t. unknown proportions based on [Prop 4., 27]
        for d in range(nbdomains):
            new = np.dot(D2[d].T, bary)
            K[d] = ot.bregman.projR(K[d], new)

        err = np.linalg.norm(bary - old_bary)
        cpt = cpt + 1
        old_bary = bary

        if log:
            log['err'].append(err)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

    bary = bary / np.sum(bary)

    if log:
        log['niter'] = cpt
        log['M'] = M
        log['D1'] = D1
        log['D2'] = D2
        log['gamma'] = K
        return bary, log
    else:
        return bary


class JCPOTTransport(ot.da.BaseTransport):
    def __init__(self, reg_e=.1, max_iter=10,
                 tol=10e-9, verbose=False, log=False,
                 metric="sqeuclidean", norm='max',
                 out_of_sample_map='ferradans'):
        self.reg_e = reg_e
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.out_of_sample_map = out_of_sample_map


    def fit(self, Xs, ys=None, Xt=None, yt=None):
        # check the necessary inputs parameters are here
        if ot.utils.check_params(Xs=Xs, Xt=Xt, ys=ys):

            self.xs_ = Xs
            self.xt_ = Xt

            returned_ = jcpot_barycenter(Xs=Xs, Ys=ys, Xt=Xt, reg=self.reg_e, metric=self.metric,
                                         norm=self.norm, distrinumItermax=self.max_iter, stopThr=self.tol,
                                         verbose=self.verbose, log=True)

            self.coupling_ = returned_[1]['gamma']

            # deal with the value of log
            if self.log:
                self.proportions_, self.log_ = returned_
            else:
                self.proportions_ = returned_
                self.log_ = dict()

        return self

    def transform(self, Xs=None, ys=None, Xt=None, yt=None, batch_size=128):
        transp_Xs = []

        # check the necessary inputs parameters are here
        if ot.utils.check_params(Xs=Xs):

            if all([np.allclose(x, y) for x, y in zip(self.xs_, Xs)]):

                # perform standard barycentric mapping for each source domain

                for coupling in self.coupling_:
                    transp = coupling / np.sum(coupling, 1)[:, None]

                    # set nans to 0
                    transp[~ np.isfinite(transp)] = 0

                    # compute transported samples
                    transp_Xs.append(np.dot(transp, self.xt_))
            else:
                # perform out of sample mapping
                indices = np.arange(Xs.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xs = []

                for bi in batch_ind:
                    transp_Xs_ = []

                    # get the nearest neighbor in the sources domains
                    xs = np.concatenate(self.xs_, axis=0)
                    idx = np.argmin(dist(Xs[bi], xs), axis=1)

                    # transport the source samples
                    for coupling in self.coupling_:
                        transp = coupling / np.sum(
                            coupling, 1)[:, None]
                        transp[~ np.isfinite(transp)] = 0
                        transp_Xs_.append(np.dot(transp, self.xt_))

                    transp_Xs_ = np.concatenate(transp_Xs_, axis=0)

                    # define the transported points
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - xs[idx, :]
                    transp_Xs.append(transp_Xs_)

                transp_Xs = np.concatenate(transp_Xs, axis=0)

            return transp_Xs

    def transform_labels(self, ys=None):
        # check the necessary inputs parameters are here
        if ot.utils.check_params(ys=ys):
            yt = np.zeros((len(np.unique(np.concatenate(ys))), self.xt_.shape[0]))
            for i in range(len(ys)):
                ysTemp = ot.utils.label_normalization(np.copy(ys[i]))
                classes = np.unique(ysTemp)
                n = len(classes)
                ns = len(ysTemp)

                # perform label propagation
                transp = self.coupling_[i] / np.sum(self.coupling_[i], 1)[:, None]

                # set nans to 0
                transp[~ np.isfinite(transp)] = 0

                if self.log:
                    D1 = self.log_['D1'][i]
                else:
                    D1 = np.zeros((n, ns))

                    for c in classes:
                        D1[int(c), ysTemp == c] = 1

                # compute propagated labels
                yt = yt + np.dot(D1, transp) / len(ys)

            return yt.T

    def inverse_transform_labels(self, yt=None):
        # check the necessary inputs parameters are here
        if ot.utils.check_params(yt=yt):
            transp_ys = []
            ytTemp = ot.utils.label_normalization(np.copy(yt))
            classes = np.unique(ytTemp)
            n = len(classes)
            D1 = np.zeros((n, len(ytTemp)))

            for c in classes:
                D1[int(c), ytTemp == c] = 1

            for i in range(len(self.xs_)):

                # perform label propagation
                transp = self.coupling_[i] / np.sum(self.coupling_[i], 1)[:, None]

                # set nans to 0
                transp[~ np.isfinite(transp)] = 0

                # compute propagated labels
                transp_ys.append(np.dot(D1, transp.T).T)

            return transp_ys
