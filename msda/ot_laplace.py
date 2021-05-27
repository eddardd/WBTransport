import numpy as np
from ot.bregman import sinkhorn
from ot.da import BaseTransport
from ot.da import distribution_estimation_uniform

from ot.optim import line_search_armijo, cg
from ot.utils import dist, kernel, laplacian, dots, unif


def gcg(a, b, M, reg1, reg2, f, df, G0=None, numItermax=10, numInnerItermax=200,
        stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False, method="sinkhorn"):
    r"""
    Solve the general regularized OT problem with the generalized conditional gradient

        The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg1\cdot\Omega(\gamma) + reg2\cdot f(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`f` is the regularization term ( and df is its gradient)
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the generalized conditional gradient as discussed in  [5,7]_


    Parameters
    ----------
    a : ndarray, shape (ns,)
        samples weights in the source domain
    b : ndarrayv (nt,)
        samples in the target domain
    M : ndarray, shape (ns, nt)
        loss matrix
    reg1 : float
        Entropic Regularization term >0
    reg2 : float
        Second Regularization term >0
    G0 : ndarray, shape (ns, nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations of Sinkhorn
    stopThr : float, optional
        Stop threshol on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshol on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    method : string, optional (default="sinkhorn")
        String specifying the Sinkhorn algorithm implementation. Should be either
        'sinkhorn', 'greenkhorn', 'sinkhorn_stabilized' or 'sinkhorn_epsilon_scaling'.

    Returns
    -------
    gamma : ndarray, shape (ns, nt)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE
           Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1
    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient: analysis of
           convergence and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.optim.cg : conditional gradient

    """

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = np.outer(a, b)
    else:
        G = G0

    def cost(G):
        return np.sum(M * G) + reg1 * np.sum(G * np.log(G)) + reg2 * f(G)

    f_val = cost(G)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, 0, 0))

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + reg2 * df(G)

        # solve linear program with Sinkhorn
        Gc = sinkhorn(a, b, Mi, reg1, numItermax=numInnerItermax)

        deltaG = Gc - G

        # line search
        dcost = Mi + reg1 * (1 + np.log(G))  # ??
        alpha, _, f_val = line_search_armijo(cost, G, deltaG, dcost, f_val)

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)

        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, relative_delta_fval, abs_delta_fval))

    if log:
        return G, log
    else:
        return G


def sinkhorn_laplace(a, b, xs, xt, M, ys=None, reg_e=.1, sim="knn", sim_param=None, reg="pos", reg_lap=1,
                     alpha=.5, numItermax=100, numInnerItermax=100000, stopThr=1e-9, stopInnerThr=1e-9,
                     log=False, verbose=False):
    r"""Solve the Optimal Transport problem (OT) with Laplacian regularization.

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + \lambda_{e}H(\gamma) + \lambda_{lap}\Omega_{lap}(\gamma; \alpha)
        s.t.\ \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0

    where:

    - a and b are source and target weights (sum to 1)
    - xs and xt are source and target samples
    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega_{reg}` is the laplacian regularization term. It takes the form
      :math:`\Omega_{reg} = \sum_{i=1}^{n_{s}}\sum_{j=1}^{n_{s}}S_{\alpha}(i,j)||\hat{{x}}_{i} - \hat{{x}}_{j}||`,
      where :math:`S_{\alpha} = (1-\alpha)S_{s} + \alpha S_{t}`, for the source and target similarity matrices
      :math:`S_{s}` and :math:`S_{t}`.
    - :math:`H(\gamma)` is the entropic regularization term, with penalty :math:`\lambda_{e}`.

    Parameters
    ----------
    a : :class:`numpy.ndarray`
        Numpy array of shape (n_s,) containing the sample weights for the source domain.
    b : :class:`numpy.ndarray`
        Numpy array of shape (n_t,) containing the sample weights for the target domain.
    xs : :class:`numpy.ndarray`
        Numpy array of shape (n_s, n_features) containing the source domain samples.
    xt : :class:`numpy.ndarray`
        Numpy array of shape (n_t, n_features) containing the target domain samples.
    M : :class:`numpy.ndarray`
        Numpy array of shape (n_s, n_t) containing the loss between source and target samples.
    ys : :class:`numpy.ndarray`
        Numpy array of shape (n_s,) containing the labels of source samples. Pass it if you want
        to encourage group-sparsity on the transport plan.
    sim : string
        Type of similarity (‘knn’ or ‘gauss’) used to construct the Laplacian.
    """

    if not isinstance(sim_param, (int, float, type(None))):
        raise ValueError(
            "Similarity parameter should be an int or a float. Got {type}"
            "instead.".format(type=type(sim_param).__name__))

    if sim == 'gauss':
        if sim_param is None:
            sim_param = 1 / (2 * (np.mean(dist(xs, xs, 'sqeuclidean')) ** 2))
        sS = kernel(xs, xs, method=sim, sigma=sim_param)
        sT = kernel(xt, xt, method=sim, sigma=sim_param)

    elif sim == 'knn':
        if sim_param is None:
            sim_param = 3

        from sklearn.neighbors import kneighbors_graph

        sS = kneighbors_graph(X=xs, n_neighbors=int(sim_param)).toarray()
        sS = (sS + sS.T) / 2
        sT = kneighbors_graph(xt, n_neighbors=int(sim_param)).toarray()
        sT = (sT + sT.T) / 2
    else:
        raise ValueError("Unknown similarity type {sim}. Currently supported similarity"
                         "types are 'knn' and 'gauss'.".format(sim=sim))

    if ys is not None:
        # Create group sparsity matrix
        group_sparsity_matrix = np.ones(sS.shape)
        for i in range(sS.shape[0]):
            ind_diff = np.where(ys != ys[i])[0]
            group_sparsity_matrix[ind_diff] = 0
        # Sparsify similarity matrix
        sS *= group_sparsity_matrix

    lS = laplacian(sS)
    lT = laplacian(sT)

    def f(G):
        return alpha * np.trace(np.dot(xt.T, np.dot(G.T, np.dot(lS, np.dot(G, xt))))) \
            + (1 - alpha) * np.trace(np.dot(xs.T, np.dot(G, np.dot(lT, np.dot(G.T, xs)))))

    ls2 = lS + lS.T
    lt2 = lT + lT.T
    xt2 = np.dot(xt, xt.T)

    if reg == 'disp':
        Cs = -reg_lap * alpha / xs.shape[0] * dots(ls2, xs, xt.T)
        Ct = -reg_lap * (1 - alpha) / xt.shape[0] * dots(xs, xt.T, lt2)
        M = M + Cs + Ct

    def df(G):
        return alpha * np.dot(ls2, np.dot(G, xt2))\
            + (1 - alpha) * np.dot(xs, np.dot(xs.T, np.dot(G, lt2)))

    return gcg(a, b, M, reg_e, reg_lap, f, df, G0=None, numItermax=numItermax,
               numInnerItermax=numInnerItermax, stopThr=stopInnerThr, stopThr2=stopThr,
               verbose=verbose, log=log)


class SinkhornLaplaceTransport(BaseTransport):
    def __init__(self, reg_type='pos', reg_e=1., reg_lap=1., metric="sqeuclidean",
                 norm=None, similarity="knn", similarity_param=None, alpha=0.5,
                 max_iter=100, tol=1e-9, max_inner_iter=1000, inner_tol=1e-9,
                 log=False, verbose=False, distribution_estimation=distribution_estimation_uniform,
                 limit_max=np.infty, out_of_sample_map='ferradans'):
        self.reg = reg_type
        self.reg_e = reg_e
        self.reg_lap = reg_lap
        self.metric = metric
        self.norm = norm
        self.similarity = similarity
        self.sim_param = similarity_param
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.log = log
        self.verbose = verbose
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.limit_max = limit_max

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        super(SinkhornLaplaceTransport, self).fit(Xs, ys, Xt, yt)

        returned_ = sinkhorn_laplace(a=self.mu_s,
                                     b=self.mu_t,
                                     xs=self.xs_,
                                     xt=self.xt_,
                                     M=self.cost_,
                                     ys=ys,
                                     reg_e=self.reg_e,
                                     sim=self.similarity,
                                     sim_param=self.sim_param,
                                     reg=self.reg,
                                     reg_lap=self.reg_lap,
                                     alpha=self.alpha,
                                     numItermax=self.max_iter,
                                     numInnerItermax=self.max_inner_iter,
                                     stopThr=self.tol,
                                     stopInnerThr=self.inner_tol,
                                     log=self.log,
                                     verbose=self.verbose)
        # coupling estimation
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()
        return self
