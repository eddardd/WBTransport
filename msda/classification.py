import numpy as np


class MultiSourceOTDAClassifier:
    def __init__(self, clf, ot_method, semi_supervised=False):
        self.clf = clf
        self.ot_method = ot_method
        self.semi_supervised = semi_supervised

    def fit(self, Xs, ys, Xt=None, yt=None):
        """Fits two objects on data (Xs, ys, Xt, yt):

        1. The classifier provided on the class constructor,
        2. The OT method provided on the class constructor.

        After fitting the OT method, this class estimates the
        transported source samples. The classifier is then fitted
        on the transported source samples.

        Parameters
        ----------
        Xs : array-like object
            Numpy array of shape (n_s, n_features). Contains the training
            source samples available during training.
        ys : array-like object
            Numpy array of shape (n_s, n_features). Contains the labels of
            training source samples available during training.
        Xt : array-like object (optional, default=None)
            Numpy array of shape (n_t, n_features). Contains the target samples
            available during training.
        yt : array-like object (optional, default=None)
            Numpy array of shape (n_t, n_features). Contains the labels of
            target samples available during training. If semi-supervised
            strategies are to be applied, this parameter should be specified.
        """
        if self.semi_supervised:
            self.ot_method.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)
        else:
            self.ot_method.fit(Xs=Xs, ys=ys, Xt=Xt)

        self.transp_Xs = self.ot_method.transform(Xs=Xs)
        X, y = self.transp_Xs, np.concatenate(ys, axis=0)
        self.clf.fit(X, y)

        return self

    def predict(self, Xs, Xt):
        _Xs_ts = self.ot_method.transform(Xs=Xs)

        yp = self.clf.predict(Xt)

        return yp
