import ot
import time
import numpy as np

from ot.bregman import sinkhorn
from msda.utils import barycentric_mapping
from msda.utils import semisupervised_penalty


def sinkhorn_barycenter(mu_s, Xs, Xbar, ys=None, ybar=None, reg=1e-3, b=None, weights=None,
                        method="sinkhorn", norm="max", metric="sqeuclidean", numItermax=100,
                        numInnerItermax=1000, stopThr=1e-4, verbose=False, innerVerbose=False,
                        log=False, line_search=False, limit_max=np.infty, callbacks=None, **kwargs):
    r"""Compute the entropic regularized Wasserstein barycenter of distributions
    in :math:`\mu_{s}`. This function solves the follwing optimization problem:

    .. math::

        \mu^{*} = \underset{\mu}{argmin}\sum_{j=1}^{N}W_{reg}(\mu,\mu_{j})

    where:

        - :math:`W_{reg}` is the regularized Wasserstein distance.
        - :math:`\mu_{j}` are the distributions in the list mu_s

    The algorithm used for solving the problem is the Sinkhorn-Knopp algorithm
    (default) as proposed in [1]_. This is an implementation of algorithm 2
    from [2]_, by assuming :math:`a` (the barycenter distribution) constant and
    uniform. Indeed, one may consider only the optimization over the barycenter
    support.

    Parameters
    ----------
    mu_s : list of array-like objects of shape (n_source_k,)
        List containing the weights for each sample in the source domain.
    Xs : list of array-like objects of shape (n_source_k, n_features)
        List containing matrices. Each matrix in Xs corresponds to the support
        of the respective weight in mu_s.
    Xt : array-like object of shape (n_support_bar, n_features) (default=None)
        Initialization for the barycenter support. If None, draws N samples from
        a standard normal distribution, where N is the total number of samples in
        the source domain.
    reg : float (default=1e-3)
        Regularization parameter.
    b : array-like object of shape (n_support_bar,) (default=None)
        Weights for each sample on the barycenter's support.
    method : str (default='sinkhorn')
        Method for the Sinkhorn algorithm. By default uses Sinkhorn-Knopp, but
        others, such as log-domain computations are available ('sinkhorn-stabilized').
    norm : str (default='max')
        Method for normalizing the cost matrix. If no normalization is to be
        performed, use None.
    metric : str (default="sqeuclidean")
        Ground metric for OT problem.
    numItermax : int (default=100)
        Number of iterations to find the Barycenter.
    numInnerItermax : int (default=1000)
        Number of iterations for the Sinkhorn algorithm at each iteration.
    stopThr : float (default=1e-4)
        Threshold for the displacement in the Barycenter support.
    verbose : bool (default=False)
        Display convergence information at each iteration.
    innerVerbose: bool (default=False)
        Display convergencer information about the Sinkhorn algorithm.
    log : bool (default=False)
        If True, returns informations from each iteration.

    Returns
    -------
    Xt : array-like object with shape (n_support_bar, n_features)
        Barycenter's support.
    transport_plans : list of array-like objects with shape (n_support_bar, n_source_k)
        The kth list element is a transport plan between the kth source to the Barycenter.

    References
    ----------
    [1] Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport.
        In Advances in neural information processing systems (pp. 2292-2300).
    [2] Cuturi, M., & Doucet, A. (2014). Fast computation of Wasserstein barycenters.
    """
    N = len(mu_s)
    k = Xbar.shape[0]
    d = Xbar.shape[1]
    if b is None:
        b = np.ones([k, ]) / k
    if weights is None:
        weights = [1 / N] * N

    displacement = stopThr + 1
    count = 0
    comp_start = time.time()
    log_dict = {'displacement_square_norms': [],
                'barycenter_coordinates': [Xbar]}
    old_Xbar = np.zeros([k, d])
    while (displacement > stopThr and count < numItermax):
        tstart = time.time()
        T_sum = np.zeros([k, d])
        transport_plans = []

        for i in range(N):
            Mi = ot.dist(Xs[i], Xbar, metric=metric)
            Mi = ot.utils.cost_normalization(Mi, norm=norm)
            if ys is not None and ybar is not None:
                Mi = semisupervised_penalty(ys=ys[i], yt=ybar, M=Mi, limit_max=limit_max)
            T_i = sinkhorn(mu_s[i], b, Mi, reg, numItermax=numInnerItermax, verbose=innerVerbose, **kwargs)
            transport_plans.append(T_i.T)
        T_sum = sum([
            wi * barycentric_mapping(Xt=Xsi, coupling=Ti) for wi, Ti, Xsi in zip(weights, transport_plans, Xs)
        ])

        if line_search:
            alpha = naive_line_search(_barycenter_cost, Xbar, T_sum, args=(Xs, b, transport_plans), max_iter=21,
                                      verbose=verbose)

            if verbose:
                print("Alpha determined through line search: {}".format(alpha))
        else:
            alpha = 1.0

        Xbar = (1 - alpha) * Xbar + alpha * T_sum
        displacement = np.sum(np.square(Xbar - old_Xbar))
        old_Xbar = Xbar.copy()
        tfinish = time.time()

        if callbacks is not None:
            for callback in callbacks:
                callback(Xbar, ybar, transport_plans)

        if verbose:
            ndigits_it = int(np.log10(numItermax)) + 1
            tdelta = tfinish - tstart
            print("It: {:<{width}} || displacement: {:<25}"
                  " || It took: {:<10}".format(count, displacement, tdelta, width=ndigits_it))
        if log:
            log_dict["displacement_square_norms"].append(displacement)
            log_dict["barycenter_coordinates"].append(Xbar)

        count += 1
    if verbose:
        print("Barycenter calculation took {} seconds".format(time.time() - comp_start))
    if log:
        return Xbar, transport_plans, log_dict
    else:
        return Xbar, transport_plans
