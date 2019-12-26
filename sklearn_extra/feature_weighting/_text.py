import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_X_y
from sklearn.preprocessing import LabelEncoder


class TfigmTransformer(BaseEstimator, TransformerMixin):
    """TF-IGM feature weighting.

    TF-IGM (Inverse Gravity Momentum) is a supervised
    feature weighting scheme for classification tasks that measures
    class distinguishing power.
    
    See User Guide for mode details.

    Parameters
    ----------
    alpha : float, default=0.15
      regularization parameter. Known good default values are 0.14 - 0.20.
    tf_scale : {"sqrt", "log1p"}, default=None
      if not None, scaling applied to term frequency. Possible scaling values are,
       - "sqrt":  square root scaling
       - "log1p": ``log(1 + tf)`` scaling. This option corresponds to
       ``sublinear_tf=True`` parameter in
       :class:`~sklearn.feature_extraction.text.TfidfTransformer`.

    Attributes
    ----------
    igm_ : array of shape (n_features,)
        The Inverse Gravity Moment (IGM) weight.
    coef_ : array of shape (n_features,)
        Regularized IGM weight corresponding to ``alpha + igm_``

    Examples
    --------
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn_extra.feature_weighting import TfigmTransformer
    >>> corpus = ['this is the first document',
    ...           'this document is the second document',
    ...           'and this is the third one',
    ...           'is this the first document']
    >>> y = ['1', '2', '1', '2']
    >>> pipe = Pipeline([('count', CountVectorizer()),
    ...                  ('tfigm', TfigmTransformer())]).fit(corpus, y)
    >>> pipe['count'].transform(corpus).toarray()
    array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
           [0, 2, 0, 1, 0, 1, 1, 0, 1],
           [1, 0, 0, 1, 1, 0, 1, 1, 1],
           [0, 1, 1, 1, 0, 0, 1, 0, 1]])
    >>> pipe['tfigm'].igm_
    array([1.  , 0.25, 0.  , 0.  , 1.  , 1.  , 0.  , 1.  , 0.  ])
    >>> pipe['tfigm'].coef_
    array([1.15, 0.4 , 0.15, 0.15, 1.15, 1.15, 0.15, 1.15, 0.15])
    >>> pipe.transform(corpus).shape
    (4, 9)

    References
    ----------
    Chen, Kewen, et al. "Turning from TF-IDF to TF-IGM for term weighting
    in text classification." Expert Systems with Applications 66 (2016):
    245-260.
    """
    def __init__(self, alpha=0.15, tf_scale=None):
        self.alpha = alpha
        self.tf_scale = tf_scale

    def _fit(self, X, y):
        """Learn the igm vector (global term weights)

        Parameters
        ----------
        X : {array-like, sparse matrix} of (n_samples, n_features)
            a matrix of term/token counts
        y : array-like of shape (n_samples,)
            target classes
        """
        self._le = LabelEncoder().fit(y)
        n_class = len(self._le.classes_)
        class_freq = np.zeros((n_class, X.shape[1]))

        X_nz = X != 0
        if sp.issparse(X_nz):
            X_nz = X_nz.asformat("csr", copy=False)

        for idx, class_label in enumerate(self._le.classes_):
            y_mask = y == class_label
            n_samples = y_mask.sum()
            class_freq[idx, :] = X_nz[y_mask].sum(axis=0) / n_samples

        self._class_freq = class_freq
        class_freq_sort = np.sort(self._class_freq, axis=0)
        f1 = class_freq_sort[-1, :]

        fk = (class_freq_sort * np.arange(n_class, 0, -1)[:, None]).sum(axis=0)
        # avoid division by zero
        igm = np.divide(f1, fk, out=np.zeros_like(f1), where=(fk != 0))
        # scale weights to [0, 1]
        self.igm_ = ((1 + n_class) * n_class * igm - 2) / (
            (1 + n_class) * n_class - 2
        )
        self.coef_ = self.alpha + self.igm_
        return self

    def fit(self, X, y):
        """Learn the igm vector (global term weights)

        Parameters
        ----------
        X : {array-like, sparse matrix} of (n_samples, n_features)
            a matrix of term/token counts
        y : array-like of shape (n_samples,)
            target classes
        """
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"])
        self._fit(X, y)
        return self

    def _transform(self, X):
        """Transform a count matrix to a TF-IGM representation

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            a matrix of term/token counts

        Returns
        -------
        vectors : {ndarray, sparse matrix} of shape (n_samples, n_features)
            transformed matrix
        """
        if self.tf_scale is None:
            pass
        elif self.tf_scale == "sqrt":
            X = np.sqrt(X)
        elif self.tf_scale == "log1p":
            X = np.log1p(X)
        else:
            raise ValueError

        if sp.issparse(X):
            X_tr = X @ sp.diags(self.coef_)
        else:
            X_tr = X * self.coef_[None, :]
        return X_tr

    def transform(self, X):
        """Transform a count matrix to a TF-IGM representation

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            a matrix of term/token counts

        Returns
        -------
        vectors : {ndarray, sparse matrix} of shape (n_samples, n_features)
            transformed matrix
        """
        X = check_array(X, accept_sparse=["csr", "csc"])
        X_tr = self._transform(X)
        return X_tr

    def fit_transform(self, X, y):
        """Transform a count matrix to a TF-IGM representation

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            a matrix of term/token counts

        Returns
        -------
        vectors : {ndarray, sparse matrix} of shape (n_samples, n_features)
            transformed matrix
        """
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"])
        self._fit(X, y)
        X_tr = self._transform(X)
        return X_tr
