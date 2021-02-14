import warnings
import numpy as np

from libtlda import iw


class ImportanceWeightedClassifier(iw.ImportanceWeightedClassifier):
    def __init__(self, clf, kernel_type='rbf', kernel_param=1.0):
        self.clf = clf
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.is_trained = False
        self.weights = []
        self.ie_fn = iw.ImportanceWeightedClassifier(kernel_type=kernel_type,
                                                     bandwidth=kernel_param).iwe_kernel_mean_matching

    def fit(self, Xs, ys, Xt):
        self.iw = self.ie_fn(Xs, Xt)
        self.iw /= np.sum(self.iw)
        self.clf.fit(Xs, ys, sample_weight=self.iw)
        self.is_trained = True

    def predict(self, X):
        return self.clf.predict(X)

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()

    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained
