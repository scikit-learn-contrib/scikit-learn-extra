# NSWG-based ANN classification
# Authors: Lev Svalov <leos3112@gmail.com>
#          Stanislav Protasov <stanislav.protasov@gmail.com>
# License: BSD 3 clause

from sklearn.base import BaseEstimator, ClassifierMixin
from ._navigable_small_world_graph import BaseNSWGraph
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.utils.multiclass import type_of_target
import numpy as np


def _check_positive_int(value, desc):
    """Validates if value is a valid integer > 0"""
    if value is None or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(
            "%s should be a positive integer. " "%s was given" % (desc, value)
        )


def _check_label_type(y):
    """Validates if labels type is correct for the estimator"""
    if type_of_target(y) not in ["binary", "multiclass"]:
        raise ValueError("Unknown label type: ")


class NSWGraph(BaseEstimator, ClassifierMixin, BaseNSWGraph):
    """Nearest Neighbors search using Navigable small world graphs.

    Read more in the :ref:`User Guide <nswgraph>`.

    Parameters
    ----------
    regularity : int, default: 16
        The size of the friends list of every vertex in the graph.
        Higher regularity leads to more accurate but slower search.

    guard_hops : int, default: 100
         The number of bi-directional links created for every new element in the graph.

    quantize : bool, default: False
         If True, use a product quantization for the preliminary dimensionality reduction of the data.

    quantization_levels : int, default: 20
         (Used if quantize=True)
         The number of the values used in quantization approximation of the dataset.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    y_: ndarray of shape (n_samples, )
        A list of labels for the corresponding targets

    n_features_in_: ndarray of shape (n_features, )
        A number of features in the provided data

    is_fitted_: boolean
        The boolean flag, indicates that the estimator was constructed with provided targets,
        so that the predict() can be used.

    is_constructed_: boolean
        The boolean flag, indicates that the estimator was constructed without provided targets,
        so that the eestimator can query for neighbours regardless its labels

    Examples
    --------
    >>> from sklearn_extra.neighbors import NSWGraph
    >>> import numpy as np
    >>> rng = np.random.RandomState(10)
    >>> X = rng.random_sample((50, 128))
    >>> nswgraph = NSWGraph()
    >>> nswgraph.build(X)
    NSWGraph(regularity=16, guard_hops=100, attempts=2, quantize=False, quantization_levels=20)
    >>> X_val = rng.random_sample((5, 128))
    >>> dist, ind = nswgraph.query(X_val, k=3)

    References
    ----------
    * Malkov, Y., Ponomarenko, A., Logvinov, A., & Krylov, V. (2014). Approximate nearest neighbor algorithm based on navigable small world graphs. Information Systems, 45, 61-68.


    """

    def __init__(
        self,
        regularity=16,
        guard_hops=100,
        attempts=2,
        quantize=False,
        quantization_levels=20,
    ):
        super().__init__()
        self.regularity = regularity
        self.guard_hops = guard_hops
        self.attempts = attempts
        self.quantize = quantize
        self.quantization_levels = quantization_levels
        self._check_init_args()

    def _check_init_args(self):
        """Validation of the initialization arguments"""
        _check_positive_int(self.regularity, "regularity")
        _check_positive_int(self.guard_hops, "guard_hops")
        _check_positive_int(self.attempts, "attempts")
        _check_positive_int(self.quantization_levels, "quantization_levels")
        if not isinstance(self.quantize, bool):
            raise ValueError(
                "%s should be a boolean. "
                "%s was given" % ("Quantization switch", self.quantize)
            )

    def _check_dimension_correspondence(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Wrong dimensionality of the data."
                "Estimator is built with %s, but %s was given"
                % (self.n_features_in_, X.shape[1])
            )

    def __repr__(self, **kwargs):
        return f"NSWGraph(regularity={self.regularity}, guard_hops={self.guard_hops}, attempts={self.attempts}, quantize={self.quantize}, quantization_levels={self.quantization_levels})"

    def build(self, X, y=None):
        """Build NSWGraph on the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features),
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self: NSWGraph
        The constructed NSWGraph

        """
        self._check_init_args()

        if y is not None:
            self.fit(X, y)
        else:
            X = check_array(X, dtype=[np.float64, np.float32])
            super().build(X)
            self.is_constructed_ = True
            self.n_features_in_ = X.shape[1]

        return self

    def fit(self, X, y):
        """Build NSWGraph on the provided data and link it with the labels.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features),
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self: NSWGraph
        The constructed NSWGraph and fitted nearest neighbors classifier.
        """
        self._check_init_args()
        X = check_array(X)
        X, y = check_X_y(X, y)
        _check_label_type(y)
        super().build(X)
        self.classes_ = np.unique(y)
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        self.is_constructed_ = True

        return self

    def query(self, X, k=1, return_distance=True):
        """Query the NSWGraph for the k nearest neighbors

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features),
            An array of points to query

        k : int, default=1
            The number of nearest neighbors to return

        return_distance: bool, default=True
        if True, return a tuple (dist, ind) of dists and indices if False, return array i

        Returns
        -------
        dist: ndarray of shape X.shape[:-1] + (k,), dtype=double
        Each entry gives the list of distances to the neighbors of the corresponding point.

        ind: ndarray of shape X.shape[:-1] + (k,), dtype=int
        Each entry gives the list of indices of neighbors of the corresponding point.
        """

        check_is_fitted(
            self,
            "is_constructed_",
            msg="This %(name)s instance is not constructed yet. Call 'build' with "
            "appropriate arguments before using this estimator.",
        )

        X = check_array(X, dtype=[np.float64, np.float32])
        self._check_dimension_correspondence(X)
        _check_positive_int(k, "k-closests")
        dist, ind = super().query(X, k)

        if return_distance:
            return dist, ind
        else:
            return ind

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class labels for the provided query data.
        The label of the closest neighbor is supposed to be predicted label

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features),
            An array of data vectors to query

        Returns
        -------
        y : ndarray of shape (n_queries,)
            Label for each data sample.
        """

        check_is_fitted(self, "is_fitted_")
        X = check_array(X, dtype=[np.float64, np.float32])
        self._check_dimension_correspondence(X)

        _, ind = super().query(X, k=1)
        result = np.array([self.y_[res[0]] for res in ind])
        return result
