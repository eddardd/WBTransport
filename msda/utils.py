import numpy as np


def bar_zeros_initializer(Xs, ys):
    r"""Initializes the barycenter support as zeros.

    Xs : array-like object
        Concatenaed source samples. Expected shape: (sum n_source_k, n_features)
    ys : array-like
        Concatenated source labels. Expected shape: (sum n_source_k)
    """
    return np.zeros(Xs.shape), ys


def bar_random_initializer(Xs, ys):
    r"""Initializes the barycenter at random positions drawn from a gaussian distribution .

    Xs : array-like object
        Concatenaed source samples. Expected shape: (sum n_source_k, n_features)
    ys : array-like
        Concatenated source labels. Expected shape: (sum n_source_k)
    """
    return np.random.randn(*Xs.shape), ys


def bar_random_cls_initializer(Xs, ys):
    r"""Initializes the barycenter support as zeros.

    Xs : array-like object
        Concatenaed source samples. Expected shape: (sum n_source_k, n_features)
    ys : array-like
        Concatenated source labels. Expected shape: (sum n_source_k)
    """
    Xbar = np.zeros(Xs.shape)
    for cl in np.unique(ys):
        ind = np.where(ys == cl)[0]
        class_mean = np.mean(Xs[ind], axis=0)
        Xbar[ind, :] = class_mean + np.random.randn(len(ind), Xs.shape[1])

    return Xbar, ys


def semisupervised_penalty(ys, yt, M, limit_max=np.infty):
    r"""Adds the semisupervised penalty defined in [1]_, and given by the following equation:

    .. math::

        \begin{eqnarray}
            \Omega_{semi}(\gamma; y_{s}, y_{t}) &=  \text{limit\_max} & \textbf{if $y_{i}^{s} \neq y_{j}^{t}$}
                                                &=  0 & \text{otherwise}
        \end{eqnarray}

    Parameters
    ----------
    ys : array-like object
        Array containing the source samples. Expected shape: (n_source,)
    yt : array-like object
        Array containing the target samples. Expected shape: (n_target,)
    M : array-like object
        Pairwise distance matrix. Expected shape: (n_source, n_target)
    limit_max : float
        Value to use as infinity. Default: np.infty.
    """
    assert M.shape == (ys.shape[0], yt.shape[0]), "Expected cost matrix to have shape ({}, {})," \
                                                  " but got {}".format(ys.shape[0], yt.shape[0], M.shape)

    _M = M.copy()
    classes = [c for c in np.unique(ys) if c != -1]
    for c in classes:
        idx_s = np.where((ys != c) & (ys != -1))[0]
        idx_t = np.where(yt == c)[0]

        for j in idx_t:
            _M[idx_s, j] = limit_max

    return _M


def barycentric_mapping(Xt, coupling):
    r"""Given an optimal transport plan :math:`\gamma`, estimate
    the transported source samples :math:`\hat{X}_{s}` using
    the Barycentric Mapping [1]_ formula:

    .. math::

        \hat{X}_{s} = diag(\gamma^{T}\mathbf{1}_{n_{s}})^{-1}\gamma\mathbf{X}_{t}

    Parameters
    ----------
    Xt : array-like object
        Numpy array of shape (n_t, n_features). Holds the target domain samples.
    coupling : array-like object
        Estimated optimal transport plan. Numpy array of shape (n_s, n_t).

    Returns
    -------
    transp_Xs : array-like object
        Estimated transported source samples. Numpy array of shape (n_s, n_features).

    References
    ----------
    [1] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy, "Optimal Transport for Domain Adaptation,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 9, pp. 1853-1865,
        1 Sept. 2017, doi: 10.1109/TPAMI.2016.2615921.
    """
    transp_Xs = coupling / np.sum(coupling, 1)[:, None]
    transp_Xs[~np.isfinite(transp_Xs)] = 0
    transp_Xs = np.dot(transp_Xs, Xt)

    return transp_Xs


def ferradans_mapping(Xs_tr, Xs_ts, Xt, coupling, batch_size=32, interpolation="diff"):
    r"""Out of sampling mapping following the method described in [1]_

    Parameters
    ----------
    Xs_tr : array-like object
        Numpy array used to fit the transport plan. Expected shape: (n_source_tr, n_features)
    Xs_ts : array-like object
        Numpy array for which we want to estimate the transport mapping.
    Xt : array-like object
        Numpy array used to fit the transport plan. Expected shape: (n_target, n_features)
    coupling : array-like object
        Numpy array holding the transport plan. Expected shape: (n_source_tt, n_target).
    batch_size : int
        Number of points on each batch for the estimation.

    References
    [1] Ferradans, Sira, et al. "Regularized discrete optimal transport." SIAM Journal on Imaging Sciences 7.3 (2014):
        1853-1882.
    """
    assert interpolation in ["diff", "avg"]

    indices = np.arange(Xs_ts.shape[0])
    batch_ind = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    transported_points = []

    # Transport Xs onto Xt
    transp_Xs = barycentric_mapping(Xt=Xt, coupling=coupling)
    for bi in batch_ind:
        D0 = dist(Xs_ts[bi], Xs_tr)
        idx = np.argmin(D0, axis=1)

        if interpolation == "diff":
            points = transp_Xs[idx, :] + Xs_ts[bi] - Xs_tr[idx, :]
        else:
            points = (transp_Xs[idx, :] + Xs_ts[bi] + Xs_tr[idx, :]) / 3
        transported_points.append(points)
    transported_points = np.concatenate(transported_points, axis=0)

    return transported_points
