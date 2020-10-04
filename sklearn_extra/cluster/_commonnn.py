# -*- coding: utf-8 -*-
"""Density-Based Common-Nearest-Neighbors Clustering
"""

# Author: Jan-Oliver Joswig <jan.joswig@fu-berlin.de>
#
# License: BSD 3 clause

from distutils.version import LooseVersion
import warnings

import numpy as np
from scipy import sparse

import sklearn
from sklearn.base import BaseEstimator, ClusterMixin

if LooseVersion(sklearn.__version__) < LooseVersion("0.23.0"):
    from sklearn.utils import check_array, check_consistent_length

    # In scikit-learn version 0.23.x use
    # sklearn.base.BaseEstimator._validate_data
else:
    from sklearn.utils.validation import _check_sample_weight

    # TODO
    # from sklearn.utils.validation import _deprecate_positional_args

from sklearn.neighbors import NearestNeighbors

from ._commonnn_inner import commonnn_inner


def commonnn(
    X,
    eps=0.5,
    min_samples=5,
    metric="minkowski",
    metric_params=None,
    algorithm="auto",
    leaf_size=30,
    p=2,
    sample_weight=None,
    n_jobs=None,
):
    """Common-nearest-neighbor clustering

    Cluster from vector array or distance matrix.

    Read more in the :ref:`User Guide <commonnn>`.

    Parameters
    ----------
    X : {array-like, sparse (CSR) matrix} of shape
        (n_samples, n_features) or (n_samples, n_samples)
        A feature array, or array of distances between samples if
        `metric='precomputed'`.

    eps : float, default=0.5
        The maximum distance between two samples for one to be
        considered as in the neighborhood of the other. This is not
        a maximum bound on the distances of points within a cluster.
        The clustering will use `min_samples` within `eps` as
        the density criterion.  The lower `eps`,
        the higher the required sample density.

    min_samples : int, default=5
        The number of samples that need to be shared as neighbors for
        two points being part of the same cluster.  The clustering will
        use `min_samples` within `eps` as the density
        criterion.  The larger `min_samples`, the higher the required
        sample density.

    metric : string, or callable
        The metric to use when calculating distance between instances in
        a feature array. If metric is a string or callable, it must be
        one of the options allowed by
        :func:`sklearn.metrics.pairwise_distances` for its metric
        parameter.
        If metric is "precomputed", X is assumed to be a distance
        matrix and must be square during fit.  X may be a
        :term:`sparse graph <sparse graph>`,
        in which case only "nonzero" elements may be considered
        neighbors.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'},
        default='auto'
        The algorithm to be used by :class:`NearestNeighbors`
        to compute pointwise distances and find nearest neighbors.

    leaf_size : int, default=30
        Leaf size passed to tree :class:`NearestNeighbors` depending on
        `algorithm`.  This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, default=2
        The power of the Minkowski metric to be used to calculate
        distance between points.

    sample_weight : array-like of shape (n_samples,), default=None
        Weight of each sample.  Note, that this option does not effect
        the clustering at the moment.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        `None` means 1 unless in a :obj:`joblib.parallel_backend`
        context. `-1` means using all processors. See
        :term:`Glossary <n_jobs>` for more details.
        If precomputed distance are used, parallel execution is not
        available and thus `n_jobs` will have no effect.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.
        Noisy samples are given the label -1.

    See also
    --------
    CommonNNClustering
        An estimator interface for this clustering algorithm.
    """

    est = CommonNNClustering(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        n_jobs=n_jobs,
    )
    est.fit(X, sample_weight=sample_weight)
    return est.labels_


class CommonNNClustering(ClusterMixin, BaseEstimator):
    """Density-Based common-nearest-neighbors clustering.

    Read more in the :ref:`User Guide <commonnn>`.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be
        considered as in the neighborhood of the other. This is not
        a maximum bound on the distances of points within a cluster.
        The clustering will use `min_samples` within `eps` as
        the density criterion.  The lower `eps`,
        the higher the required sample density.

    min_samples : int, default=5
        The number of samples that need to be shared as neighbors for
        two points being part of the same cluster.  The clustering will
        use `min_samples` within `eps` as the density
        criterion.  The larger `min_samples`, the higher the required
        sample density.

    metric : string, or callable, default='euclidean'
        The metric to use
        when calculating distance between instances in a feature array.
        If metric is a string or callable, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise_distances` for its
        metric parameter. If metric is "precomputed", X is assumed to be
        a distance matrix and must be square. X may be a :term:`Glossary
        <sparse graph>`, in which case only "nonzero" elements may be
        considered neighbors.

    metric_params : dict, default=None
        Additional keyword arguments for
        the metric function.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'},
        default='auto'
        The algorithm to be used by :class:`NearestNeighbors`
        to compute pointwise distances and find nearest
        neighbors.

    leaf_size : int, default=30
        Leaf size passed to tree :class:`NearestNeighbors` depending on
        `algorithm`.
        This can affect the speed of the construction and query, as well
        as the memory required to store the tree. The optimal value
        depends on the nature of the problem.

    p : float, default=None
        The power of the Minkowski metric to be used
        to calculate distance between points.

    n_jobs : int, default=None
        The number of parallel jobs to run.
        `None` means 1 unless in a :obj:`joblib.parallel_backend`
        context. `-1` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    Attributes
    ----------

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point
        in the dataset given to fit().
        Noisy samples are given the label -1.

    Examples
    --------
    >>> from sklearn_extra.cluster import CommonNNClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    >>> clustering = CommonNNClustering(eps=3, min_samples=0).fit(X)
    >>> clustering.labels_
    array([ 0,  0,  0,  1,  1, -1])

    See also
    --------
    commonnn
        A function interface for this cluster algorithm.

    sklearn.cluster.DBSCAN
        A similar clustering providing a different notion of the
        point density.  The implementation is (like this present
        :class:`CommonNNClustering` implementation) optimized for speed.

    sklearn.cluster.OPTICS
        A similar clustering
        at multiple values of `eps`.
        The implementation is optimized for
        memory usage.

    Notes
    -----

    This implementation bulk-computes all neighborhood queries, which
    increases the memory complexity to :math:`O(n ⋅ n_n)` where
    :math:`n_n` is the average
    number of neighbors, similar to the present implementation of
    :class:`sklearn.cluster.DBSCAN`.  It may attract a higher memory
    complexity
    when querying these nearest neighborhoods, depending on the
    `algorithm`.

    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    `mode='distance'`, then using `metric='precomputed'` here.

    :class:`sklearn.cluster.OPTICS` provides a similar clustering with
    lower memory usage.

    References
    ----------
    B. Keller, X. Daura, W. F. van Gunsteren "Comparing Geometric and
    Kinetic Cluster Algorithms for Molecular Simulation Data" J. Chem.
    Phys., 2010, 132, 074110.

    O. Lemke, B.G. Keller "Density-based Cluster Algorithms for the
    Identification of Core Sets" J. Chem. Phys., 2016, 145, 164104.

    O. Lemke, B.G. Keller "Common nearest neighbor clustering - a
    benchmark" Algorithms, 2018, 11, 19.
    """

    # TODO Use
    # @_deprecate_positional_args
    #     not in scikit-learn version 0.21.3
    def __init__(
        self,
        eps=0.5,
        *,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None):
        """Perform common-nearest-neighbor clustering

        Cluster from features, or distance matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape
            (n_samples, n_features), or (n_samples, n_samples)
            Training instances to cluster, or distances between
            instances if `metric='precomputed'`.
            If a sparse matrix is provided, it will
            be converted into a sparse `csr_matrix`.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample.  Note, that this option is not
            fully supported at the moment.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """

        if LooseVersion(sklearn.__version__) < LooseVersion("0.23.0"):
            X = check_array(X, accept_sparse="csr")
        else:
            X = self._validate_data(X, accept_sparse="csr")

        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")

        if sample_weight is not None:
            warnings.warn(
                "Sample weights are not fully supported, yet.", UserWarning
            )
            if LooseVersion(sklearn.__version__) < LooseVersion("0.23.0"):
                sample_weight = np.asarray(sample_weight)
                check_consistent_length(X, sample_weight)
            else:
                sample_weight = _check_sample_weight(sample_weight, X)

        # Calculate neighborhood for all samples. This leaves the
        # original point in, which needs to be considered later
        # (i.e. point i is in the
        # neighborhood of point i). While True, its useless information
        if self.metric == "precomputed" and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())

        neighbors_model = NearestNeighbors(
            radius=self.eps,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            metric_params=self.metric_params,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(
            X, return_distance=False
        )

        if sample_weight is None:
            n_neighbors = np.array(
                [len(neighbors) for neighbors in neighborhoods]
            )
        else:
            n_neighbors = np.array(
                [
                    np.sum(sample_weight[neighbors])
                    for neighbors in neighborhoods
                ]
            )

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # Account for self neighbour membership (self.min_samples + 2)
        corrected_min_samples = self.min_samples + 2

        # Array tracking points qualified for similarity check
        core_candidates = np.asarray(n_neighbors >= corrected_min_samples)

        commonnn_inner(
            neighborhoods, labels, core_candidates, corrected_min_samples
        )

        self.labels_ = labels

        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Perform common-nearest-neighbor clustering

        Cluster from features or distance matrix,
        and return cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            `metric='precomputed'`. If a sparse matrix is provided, it will
            be converted into a sparse `csr_matrix`.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample.  Note, that this option is not
            fully supported at the moment.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        self.fit(X, sample_weight=sample_weight)
        return self.labels_
