import numpy as np

from ot.utils import unif
from ot.da import SinkhornTransport

from msda.barycenters import sinkhorn_barycenter

from msda.utils import ferradans_mapping
from msda.utils import barycentric_mapping
from msda.utils import bar_zeros_initializer
from msda.utils import bar_random_initializer
from msda.utils import bar_random_cls_initializer


class WassersteinBarycenterTransport:
    r"""Multi-source domain adaptation using Wasserstein barycenters. This class is intended
    to solve the domain adaptation problem when one has multiple sources s1, ..., sM. First,
    the Wasserstein barycenter of source measures :math:`\{\mu_{k}\}_{k=1}^{N}` is estimated,
    using the Fréchet mean,

    .. math::

        \mu^{*} = \underset{\mu}{min}\sum_{k=1}^{N}\omega_{k}W_{p}(\mu, \mu_{k}).

    Being :math:`\mu^{*}` the Barycenter (discrete) measure, it is expressed as,

    .. math::

        \mu^{*}(\mathbf{x}) = \sum_{i=1}^{N_{bar}}p_{i}\delta(\mathbf{x}-\mathbf{x}^{supp}_{i}),

    where,

        - :math:`p_{i}` are the weights for points in the support of :math:`\mu^{*}`.
        - :math:`\mathbf{x}^{supp}_{i}` are the points in the support of :math:`\mu^{*}`.

    We consider :math:`p_{i}` constant and uniform (hence, no optimization is done over this variable). The only
    variable being optimized in the Barycenter estimation, are the support points :math:`X_{bar} = \{x_{i}^{supp}\}`.
    The algorithm used for that end is Algorithm 2 of [1]_. After estimating :math:`X_{bar}`, the points of each
    domain are transported onto the Barycenter, using the Barycentric Mapping defined as,

    .. math::
        \hat{\mathbf{X}}_{s_{j}} = diag(a)^{-1}\gamma_{s_{j}}X_{bar}

    This yields a single source, :math:`\tilde{X}_{s}` for which we may apply standard single-source OT as in [2]_.

    References
    ----------
    [1] Cuturi, M., & Doucet, A. (2014). Fast computation of Wasserstein barycenters.
    [2] Flamary, R. (2016). Optimal transport for domain adaptation.
    """

    def __init__(self,
                 barycenter_initialization="zeros",
                 weights=None,
                 verbose=False,
                 barycenter_solver=sinkhorn_barycenter,
                 transport_solver=SinkhornTransport):
        self.barycenter_initialization = barycenter_initialization
        self.weights = None
        self.verbose = verbose
        self.barycenter_solver = barycenter_solver
        self.transport_solver = transport_solver

    def fit(self, Xs=None, Xt=None, ys=None, yt=None):
        r"""Estimates the coupling matrices between each source domain and the barycenter.
        After calculating the barycenter of sources, transport each source domain to the
        barycenter. Then, transport the sources onto the target.

        Parameters
        ----------
        Xs : list of K array-like objects, shape K x (n_samples_domain_k, n_features)
            A list containing the source samples of each domain.
        ys : list of K array-like objects, shape K x (n_samples_domain_k,)
            A list containing the labels of each sample on each source domain.
        Xt : array-like object, shape (n_samples_target, n_features)
            An array containing the target domain samples.
        yt : array-like object shape (n_samples_target,)
            An array containing the target domain labels. If given, semi-supervised
            loss is added to the cost matrix.
        """
        self.xs_ = Xs
        self.ys_ = ys
        self.xt_ = Xt

        # Barycenter variables
        mu_s = [unif(X.shape[0]) for X in Xs]

        if self.barycenter_initialization == "zeros":
            if self.verbose:
                print("[INFO] initializing barycenter position as zeros")
            self.Xbar, self.ybar = bar_zeros_initializer(np.concatenate(self.xs_, axis=0),
                                                         np.concatenate(self.ys_, axis=0))
        elif self.barycenter_initialization == "random":
            if self.verbose:
                print("[INFO] initializing barycenter at random positions")
            self.Xbar, self.ybar = bar_random_initializer(np.concatenate(self.xs_, axis=0),
                                                          np.concatenate(self.ys_, axis=0))
        elif self.barycenter_initialization == "random_cls":
            if self.verbose:
                print("[INFO] initializing barycenter at random positions using classes")
            self.Xbar, self.ybar = bar_random_cls_initializer(np.concatenate(self.xs_, axis=0),
                                                              np.concatenate(self.ys_, axis=0))
        else:
            raise ValueError("Expected 'barycenter_initialization' to be either"
                             "'zeros', 'random' or 'random_cls', but got {}".format(self.barycenter_initialization))

        if self.weights is None:
            self.weights = unif(len(self.xs_))

        if self.verbose:
            print("Estimating Barycenter")
            print("---------------------")

        # Barycenter estimation
        bary, couplings = self.barycenter_solver(mu_s=mu_s, Xs=self.xs_, Xbar=self.Xbar)

        couplings = [coupling.T for coupling in couplings]

        self.coupling_ = {
            'Barycenter Coupling {}'.format(i): coupling for i, coupling in enumerate(couplings)
        }
        self.Xbar = bary
        self.Xs = bary.copy()

        # Transport estimation
        self.Tbt = self.transport_solver()
        if self.verbose:
            print("\n")
            print("Estimating transport Barycenter => Target")
            print("-----------------------------------------")
        self.Tbt.fit(Xs=self.Xs, ys=self.ybar, Xt=Xt, yt=yt)

        self.coupling_["Bar->Target Coupling"] = self.Tbt.coupling_

    def transform(self, Xs=None, ys=None, Xt=None, yt=None):
        # Verify if samples given were the ones used to fit the optimal transport plan
        if all([np.array_equal(xs_k, Xs_k) for xs_k, Xs_k in zip(self.xs_, Xs)]):
            # If they were the same, applies barycentric mapping
            self.txs_ = self.Tbt.transform(Xs=self.Xs)
        else:
            # Otherwise, uses Ferradans mapping
            _Xs = []
            for Xs_ts_k, Xs_tr_k, gamma_k in zip(Xs, self.xs_, self.coupling_):
                # 1. Transport each medium to the barycenter using Ferradans mapping
                # For each xj in Xs_ts Finds nearest neighbor in Xs_ts and applies
                # the barycentric mapping
                _Xs.append(
                    ferradans_mapping(Xs_tr=Xs_tr_k,
                                      Xs_ts=Xs_ts_k,
                                      Xt=self.Xbar,
                                      coupling=self.coupling_[gamma_k])
                )
            # 2. Using the transported points onto the barycenter, transport them onto the target.
            self.txs_ = np.concatenate(_Xs, axis=0)
        return self.txs_
