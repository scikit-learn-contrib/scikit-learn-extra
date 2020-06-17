# -*- coding: utf-8 -*-
"""Density-Based Common-Nearest-Neighbours (CNN) Clustering
"""

# Author: Jan-Oliver Joswig <jan.joswig@fu-berlin.de>
#
# License: BSD 3 clause

import warnings

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array
# TODO Use sklearn.base.BaseEstimator._validate_data instead
#     not in scikit-learn version 0.21.3

# TODO Use
# from sklearn.utils.validation import _check_sample_weight
#     not in scikit-learn version 0.21.3

# TODO Use
# from sklearn.utils.validation import _deprecate_positional_args
#     not in scikit-learn version 0.21.3

from sklearn.neighbors import NearestNeighbors

from ._cnn_inner import cnn_inner


def cnn(
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
    """Perform CNN clustering from vector array or distance matrix.

    Read more in the :ref:`User Guide <cnn>`.

    Parameters
    ----------
    X : {array-like, sparse (CSR) matrix} of shape
        (n_samples, n_features) or (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    eps : float, default=0.5
        The maximum distance between two samples for one to be
        considered as in the neighborhood of the other. This is not
        a maximum bound on the distances of points within a cluster.
        The clustering will use `min_samples`/`eps` as the density
        criterion. The lower `eps`, the higher the required sample
        density.

    min_samples : int, default=5
        The number of samples that need to be shared as neighbors for
        two points being part of the same cluster.  The clustering will
        use `min_samples`/`eps` as the density
        criterion. The larger `min_samples`, the higher the required
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
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the
        speed
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
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See
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
    CNN
        An estimator interface for this clustering algorithm.

    DBSCAN
        An estimator interface for a similar clustering algorithm
        providing a different notion of the point density.

    OPTICS
        A similar estimator interface clustering at multiple values of eps. Our
        implementation is optimized for memory usage.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_cnn.py`.

    This implementation bulk-computes all neighborhood queries,
    which increases
    the memory complexity to O(n.d) where d is the average number of
    neighbors. It may attract a higher memory complexity when querying
    these nearest neighborhoods, depending
    on the ``algorithm``.

    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.

    :func:`cluster.optics <sklearn.cluster.optics>` provides a similar
    clustering with lower memory usage.

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

    est = CNN(
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


class CNN(ClusterMixin, BaseEstimator):
    """Perform CNN clustering from vector array or distance matrix.

    CNN - Density-Based Common-Nearest-Neighbours Clustering.

    Read more in the :ref:`User Guide <cnn>`.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be
        considered as in the neighborhood of the other. This is not
        a maximum bound on the distances of points within a cluster.
        The clustering will use `min_samples`/`eps` as the density
        criterion. The lower `eps`, the higher the required sample
        density.

    min_samples : int, default=5
        The number of samples that need to be shared as neighbors for
        two points being part of the same cluster.  The clustering will
        use `min_samples`/`eps` as the density
        criterion. The larger `min_samples`, the higher the required
        sample density.

    metric : string, or callable, default='euclidean' The metric to use
        when calculating distance between instances in a feature array.
        If metric is a string or callable, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise_distances` for its
        metric parameter. If metric is "precomputed", X is assumed to be
        a distance matrix and must be square. X may be a :term:`Glossary
        <sparse graph>`, in which case only "nonzero" elements may be
        considered neighbors.

    metric_params : dict, default=None Additional keyword arguments for
        the metric function.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'},
        default='auto' The algorithm to be used by the NearestNeighbors
        module to compute pointwise distances and find nearest
        neighbors. See NearestNeighbors module documentation for
        details.

    leaf_size : int, default=30 Leaf size passed to BallTree or cKDTree.
        This can affect the speed of the construction and query, as well
        as the memory required to store the tree. The optimal value
        depends on the nature of the problem.

    p : float, default=None The power of the Minkowski metric to be used
        to calculate distance between points.

    n_jobs : int, default=None The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    Attributes
    ----------

    labels_ : ndarray of shape (n_samples) Cluster labels for each point
        in the dataset given to fit(). Noisy samples are given the label
        -1.

    Examples
    --------
    >>> from sklearn.cluster import CNN
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    >>> clustering = CNN(eps=3, min_samples=0).fit(X)
    >>> clustering.labels_
    array([ 0,  0,  0,  1,  1, -1])
    >>> clustering
    CNN(eps=3, min_samples=0)

    See also
    --------
    DBSCAN
        A similar clustering providing a different notion of the
        point density. The implementation is (like this present CNN
        implementation) optimized for speed.

    OPTICS
        A similar clustering
        at multiple values of eps. The implementation is optimized for
        memory usage.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_cnn.py
    <sphx_glr_auto_examples_cluster_plot_cnn.py>`.

    This implementation bulk-computes all neighborhood queries, which
    increases the memory complexity to O(n.d) where d is the average
    number of neighbors, similar to the present implementation of
    :class:`cluster.DBSCAN`. It may attract a higher memory complexity
    when querying these nearest neighborhoods, depending on the
    ``algorithm``.

    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.

    Another way to reduce memory and computation time is to remove
    (near-)duplicate points and use ``sample_weight`` instead.

    :class:`cluster.OPTICS` provides a similar clustering with lower
    memory usage.

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
        """Perform CNN clustering from features, or distance matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape
            (n_samples, n_features), or (n_samples, n_samples)
            Training instances to cluster, or distances between
            instances if ``metric='precomputed'``.
            If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample.  Note, that this option does not
            effect the clustering at the moment.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """

        X = check_array(X, accept_sparse="csr")
        # TODO Use
        # X = self._validate_data(X, accept_sparse="csr")
        #    not in scikit-learn version 0.21.3

        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")

        if sample_weight is not None:
            raise NotImplementedError(
                "Sample weights are not yet supported."
                )
        #     sample_weight = _check_sample_weight(sample_weight, X)

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

        cnn_inner(
            neighborhoods, labels, core_candidates, corrected_min_samples
        )

        self.labels_ = labels

        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Perform CNN clustering from features or distance matrix,
        and return cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample.  Note, that this option does not
            effect the clustering at the moment.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        self.fit(X, sample_weight=sample_weight)
        return self.labels_
