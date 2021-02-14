import numpy as np

from libtlda import tca
from sklearn.decomposition import PCA

from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist


def da_pca(Xs, Xt, num_components=2):
    """Principal Component Analysis for Domain Adaptation

    Accorindg to [1], performs a projection of the entire dataset
    (source + target) onto a low-dimensional sub-space using PCA.
    The dimensionality of such sub-space is determined through
    num_components.

    Parameters
    ----------
    Xs : :class:`numpy.ndarray`
        Numpy array of shape (n_source, n_features) containing the source domain samples
    Xt : :class:`numpy.ndarray`
        Numpy array of shape (n_target, n_features) containing the target domain samples
    num_components : int
        Integer determining the number of principal components for the data projection.

    Returns
    -------
    proj_Xs : :class:`numpy.ndarray`
        Numpy array of shape (n_source, num_components) with the projected source domain data.
    pca : :class:`sklearn.decomposition.PCA`
        Principal Component Analysis object for transforming new data.
    """
    all_X = np.concatenate([Xs, Xt], axis=0)
    pca = PCA(n_components=num_components)
    pca.fit(all_X)
    
    proj_X = pca.transform(all_X)
    proj_Xs = proj_X[:len(Xs)]
    
    return proj_Xs, pca



class DAPrincipalComponentAnalysis:
    def __init__(self, clf, num_components=2):
        self.clf = clf
        self.num_components = num_components

    def fit(self, Xs, ys, Xt):
        proj_Xs, self.pca = da_pca(Xs, Xt, num_components=self.num_components)
        self.clf.fit(proj_Xs, ys)

    def predict(self, X):
        proj_X = self.pca.transform(X)

        return self.clf.predict(proj_X)


class TransferComponentsClassifier(tca.TransferComponentClassifier):
    def __init__(self, clf, kernel_type='rbf', kernel_param='auto',
                 mu=1.0, num_components=1):
        self.clf = clf
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.mu = mu
        self.num_components = num_components
        self.XsXt = None
        self.C = None
        self.is_trained = False
        self.train_data_dim = None

    def __check_pos_def(self, K):
        return np.all(np.linalg.eigvals(K) > 0)

    def __check_dims(self, Xs, Xt):
        _, ds = Xs.shape
        _, dt = Xt.shape

        if not ds == dt:
            raise ValueError('Dimensionalities of X and Z should be equal.')

    def __kernel(self, Xs, Xt):
        self.__check_dims(Xs, Xt)

        if self.kernel_type == 'linear':
            return np.dot(Xs, Xt.T)
        elif self.kernel_type == 'rbf':
            bandwidth = 1 / Xs.shape[0] if self.kernel_param == 'auto' else self.kernel_param
            return np.exp(-cdist(Xs, Xt) / (2. * bandwidth ** 2))
        elif self.kernel_type == 'polynomial':
            p = 2.0 if self.kernel_param == 'auto' else self.kernel_param
            return (np.dot(Xs, Xt.T) + 1) ** p
        elif self.kernel_type == 'sigmoid':
            return 1 / (1 + np.exp(np.dot(Xs, Xt.T)))
        else:
            raise NotImplementedError('Kernel Type {} not implemented.'.format(self.kernel_type))

    def __tca(self, Xs, Xt):
        ns, _ = Xs.shape
        nt, _ = Xt.shape
        self.__check_dims(Xs, Xt)

        XsXt = np.concatenate([Xs, Xt], axis=0)
        K = self.__kernel(Xs=XsXt, Xt=XsXt)

        if not self.__check_pos_def(K):
            print('Warning: covariate matrices not PSD.')
            regct = -6

            while not self.__check_pos_def(K):
                print('Adding regularization: ' + str(10**regct))
                K += np.eye(ns + nt) * 10. ** regct
                regct += 1

        L = np.vstack([
            np.hstack([np.ones([ns, ns]) / (ns ** 2), - np.ones([ns, nt]) / (ns * nt)]),
            np.hstack([- np.ones([nt, ns]) / (ns * nt), np.ones([nt, nt]) / (nt * nt)])
        ])

        H = np.eye(ns + nt) - np.ones([ns + nt, ns + nt]) / (ns + nt)

        J = np.dot(np.linalg.pinv(np.eye(ns + nt) + self.mu * np.dot(np.dot(K, L), K)), np.dot(np.dot(K, H), K))
        _, C = eigs(J, k=self.num_components)

        return np.real(C), K

    def fit(self, Xs, ys, Xt):
        self.__check_dims(Xs, Xt)

        self.XsXt = np.concatenate([Xs, Xt], axis=0)
        self.C, K = self.__tca(Xs, Xt)
        TX = np.dot(K[:Xs.shape[0], :], self.C)

        self.clf.fit(TX, ys)
        self.is_trained = True
        self.train_data_dim = Xs.shape[1]

    def predict(self, X):
        K = self.__kernel(X, self.XsXt)
        Z = np.dot(K, self.C)

        return self.clf.predict(Z)

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()

    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained
