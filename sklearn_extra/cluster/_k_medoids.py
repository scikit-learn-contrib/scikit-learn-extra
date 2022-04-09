"""K-medoids clustering"""

# Authors: Timo Erkkilä <timo.erkkila@gmail.com>
#          Antti Lehmussola <antti.lehmussola@gmail.com>
#          Kornel Kiełczewski <kornel.mail@gmail.com>
#          Zane Dufour <zane.dufour@gmail.com>
# License: BSD 3 clause

import abc
import numbers
import warnings
from enum import Enum

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted

# cython implementation of steps in PAM algorithm.
from ._k_medoids_helper import _build, _compute_optimal_swap


class _InitMethod(str, Enum):
    RANDOM = "random"
    HEURISTIC = "heuristic"
    KMEDOIDSPP = "k-medoids++"
    BUILD = "build"


def _is_array(v):
    return hasattr(v, "__array__")


def _compute_inertia(distances):
    """Compute inertia of new samples. Inertia is defined as the sum of the
    sample distances to closest cluster centers.

    Parameters
    ----------
    distances : {array-like, sparse matrix}, shape=(n_samples, n_clusters)
        Distances to cluster centers.

    Returns
    -------
    Sum of sample distances to closest cluster centers.
    """

    # Define inertia as the sum of the sample-distances
    # to closest cluster centers
    inertia = np.sum(np.min(distances, axis=1))

    return inertia


class _BaseMethod(abc.ABC):
    def converged(self, current_iter, max_iter):
        if np.all(self.old_medoid_idxs == self.medoid_idxs):
            return True
        elif current_iter == max_iter - 1:
            warnings.warn(
                "Maximum number of iteration reached before "
                "convergence. Consider increasing max_iter to "
                "improve the fit.",
                ConvergenceWarning,
            )
            return True
        return False

    def next(self):
        self.old_medoid_idxs = np.copy(self.medoid_idxs)
        self._next()

    @abc.abstractmethod
    def _next(self):
        ...


class _PAM(_BaseMethod):
    def __init__(self, D, medoid_idxs, n_clusters, max_iter):
        self.D = D
        self.medoid_idxs = medoid_idxs
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        # Compute the distance to the first and second closest points
        # among medoids.

        if n_clusters == 1 and max_iter > 0:
            # PAM SWAP step can only be used for n_clusters > 1
            warnings.warn(
                "n_clusters should be larger than 2 if max_iter != 0 "
                "setting max_iter to 0."
            )
            max_iter = 0
        elif max_iter > 0:
            self.Djs, self.Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]
        else:
            self.Djs = None
            self.Ejs = None

    def _next(self):
        not_medoid_idxs = np.delete(np.arange(len(self.D)), self.medoid_idxs)
        optimal_swap = _compute_optimal_swap(
            self.D,
            self.medoid_idxs.astype(np.intc),
            not_medoid_idxs.astype(np.intc),
            self.Djs,
            self.Ejs,
            self.n_clusters,
        )
        if optimal_swap is not None:
            i, j, _ = optimal_swap
            self.medoid_idxs[self.medoid_idxs == i] = j

            # update Djs and Ejs with new medoids
            self.Djs, self.Ejs = np.sort(self.D[self.medoid_idxs], axis=0)[
                [0, 1]
            ]


class _Alternate(_BaseMethod):
    def __init__(self, D, medoid_idxs, n_clusters, *args):
        self.D = D
        self.medoid_idxs = medoid_idxs
        self.n_clusters = n_clusters

    def _next(self):
        labels = np.argmin(self.D[self.medoid_idxs, :], axis=0)
        # Update the medoids for each cluster
        for k in range(self.n_clusters):
            # Extract the distance matrix between the data points
            # inside the cluster k
            cluster_k_idxs = np.where(labels == k)[0]

            if len(cluster_k_idxs) == 0:
                warnings.warn(
                    "Cluster {k} is empty! "
                    "self.labels_[self.medoid_indices_[{k}]] "
                    "may not be labeled with "
                    "its corresponding cluster ({k}).".format(k=k)
                )
                continue

            in_cluster_distances = self.D[
                cluster_k_idxs, cluster_k_idxs[:, np.newaxis]
            ]

            # Calculate all costs from each point to all others in the cluster
            in_cluster_all_costs = np.sum(in_cluster_distances, axis=1)

            min_cost_idx = np.argmin(in_cluster_all_costs)
            min_cost = in_cluster_all_costs[min_cost_idx]
            curr_cost = in_cluster_all_costs[
                np.argmax(cluster_k_idxs == self.medoid_idxs[k])
            ]

            # Adopt a new medoid if its distance is smaller then the current
            if min_cost < curr_cost:
                self.medoid_idxs[k] = cluster_k_idxs[min_cost_idx]


class KMedoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """k-medoids clustering.

    Read more in the :ref:`User Guide <k_medoids>`.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of medoids to
        generate.

    metric : string, or callable, optional, default: 'euclidean'
        What distance metric to use. See :func:metrics.pairwise_distances
        metric can be 'precomputed', the user must then feed the fit method
        with a precomputed kernel matrix and not the design matrix X.

    method : {'alternate', 'pam'}, default: 'alternate'
        Which algorithm to use. 'alternate' is faster while 'pam' is more accurate.

    init : {'random', 'heuristic', 'k-medoids++', 'build'}, or array-like of shape
        (n_clusters, n_features), optional, default: 'heuristic'
        Specify medoid initialization method. 'random' selects n_clusters
        elements from the dataset. 'heuristic' picks the n_clusters points
        with the smallest sum distance to every other point. 'k-medoids++'
        follows an approach based on k-means++_, and in general, gives initial
        medoids which are more separated than those generated by the other methods.
        'build' is a greedy initialization of the medoids used in the original PAM
        algorithm. Often 'build' is more efficient but slower than other
        initializations on big datasets and it is also very non-robust,
        if there are outliers in the dataset, use another initialization.
        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        .. _k-means++: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting. It can be zero in
        which case only the initialization is computed which may be suitable for
        large datasets when the initialization is sufficiently efficient
        (i.e. for 'build' init).

    random_state : int, RandomState instance or None, optional
        Specify random state for the random number generator. Used to
        initialise medoids when init='random'.

    Attributes
    ----------
    cluster_centers_ : array, shape = (n_clusters, n_features)
            or None if metric == 'precomputed'
        Cluster centers, i.e. medoids (elements from the original dataset)

    medoid_indices_ : array, shape = (n_clusters,)
        The indices of the medoid rows in X

    labels_ : array, shape = (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Examples
    --------
    >>> from sklearn_extra.cluster import KMedoids
    >>> import numpy as np

    >>> X = np.asarray([[1, 2], [1, 4], [1, 0],
    ...                 [4, 2], [4, 4], [4, 0]])
    >>> kmedoids = KMedoids(n_clusters=2, random_state=0).fit(X)
    >>> kmedoids.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> kmedoids.predict([[0,0], [4,4]])
    array([0, 1])
    >>> kmedoids.cluster_centers_
    array([[1., 2.],
           [4., 2.]])
    >>> kmedoids.inertia_
    8.0

    See scikit-learn-extra/examples/plot_kmedoids_digits.py for examples
    of KMedoids with various distance metrics.

    References
    ----------
    Maranzana, F.E., 1963. On the location of supply points to minimize
      transportation costs. IBM Systems Journal, 2(2), pp.129-135.
    Park, H.S.and Jun, C.H., 2009. A simple and fast algorithm for K-medoids
      clustering.  Expert systems with applications, 36(2), pp.3336-3341.

    See also
    --------

    KMeans
        The KMeans algorithm minimizes the within-cluster sum-of-squares
        criterion. It scales well to large number of samples.

    Notes
    -----
    Since all pairwise distances are calculated and stored in memory for
    the duration of fit, the space complexity is O(n_samples ** 2).

    """

    def __init__(
        self,
        n_clusters=8,
        metric="euclidean",
        method="alternate",
        init="heuristic",
        max_iter=300,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state

    def _check_non_negative(self, v, param, zero_included):
        error = ValueError(
            f"{param} should be a nonnegative integer, got {v}."
        )
        if v is None:
            raise error
        greater_than_zero = v >= 0 if zero_included else v > 0
        if not (isinstance(v, numbers.Integral) and greater_than_zero):
            raise error

    def _check_n_clusters(self, X, init):

        if _is_array(init):
            warnings.warn(
                "n_clusters should be equal to size of array-like if init "
                "is array-like setting n_clusters to {}.".format(init.shape[0])
            )
            return init.shape[0]

        self._check_non_negative(
            self.n_clusters, "n_clusters", zero_included=False
        )

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )
        return self.n_clusters

    def _check_max_iter(self):
        self._check_non_negative(self.max_iter, "max_iter", zero_included=True)
        return self.max_iter

    def _check_init(self):
        methods = [x.value for x in _InitMethod]
        is_not_valid_method = not (
            isinstance(self.init, str) and self.init in methods
        )
        is_not_array_like = not _is_array(self.init)
        if is_not_array_like and is_not_valid_method:
            msg = f"init should be one of: {methods + ['array-like']}, got {self.init}."
            raise ValueError(msg)

        return self.init

    def _check_method(self):
        if self.method not in ["pam", "alternate"]:
            raise ValueError(
                f"method={self.method} is not supported. Supported methods "
                f"are 'pam' and 'alternate'."
            )

    def _check_X(self, X):
        return check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

    def fit(self, X, y=None):
        """Fit K-Medoids to the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features), \
                or (n_samples, n_samples) if metric == 'precomputed'
            Dataset to cluster.

        y : Ignored

        Returns
        -------
        self
        """
        random_state = check_random_state(self.random_state)
        self._check_method()
        self._init = self._check_init()
        X = self._check_X(X)
        n_clusters = self._check_n_clusters(X, self._init)
        max_iter = self._check_max_iter()

        D = pairwise_distances(X, metric=self.metric)

        initial_medoid_idxs = self._initialize_medoids(
            D, n_clusters, random_state, X
        )

        algorithm_class = _PAM if self.method == "pam" else _Alternate
        algorithm = algorithm_class(
            D, initial_medoid_idxs, n_clusters, max_iter
        )

        for self.n_iter_ in range(0, max_iter):
            algorithm.next()
            if algorithm.converged(self.n_iter_, max_iter):
                break

        self.cluster_centers_ = (
            X[algorithm.medoid_idxs] if self.metric != "precomputed" else None
        )

        self.labels_ = np.argmin(D[algorithm.medoid_idxs, :], axis=0)
        self.medoid_indices_ = algorithm.medoid_idxs
        self.inertia_ = _compute_inertia(self.transform(X))

        return self

    def transform(self, X):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Data to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return X[:, self.medoid_indices_]
        else:
            check_is_fitted(self, "cluster_centers_")

            Y = self.cluster_centers_
            kwargs = {}
            if self.metric == "seuclidean":
                kwargs["V"] = np.var(np.vstack([X, Y]), axis=0, ddof=1)
            DXY = pairwise_distances(X, Y=Y, metric=self.metric, **kwargs)

            return DXY

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            New data to predict.

        Returns
        -------
        labels : array, shape = (n_query,)
            Index of the cluster each sample belongs to.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return np.argmin(X[:, self.medoid_indices_], axis=1)
        else:
            check_is_fitted(self, "cluster_centers_")

            # Return data points to clusters based on which cluster assignment
            # yields the smallest distance
            kwargs = {}
            if self.metric == "seuclidean":
                kwargs["V"] = np.var(
                    np.vstack([X, self.cluster_centers_]), axis=0, ddof=1
                )
            pd_argmin = pairwise_distances_argmin(
                X,
                Y=self.cluster_centers_,
                metric=self.metric,
                metric_kwargs=kwargs,
            )

            return pd_argmin

    def _initialize_medoids(
        self,
        D,
        n_clusters,
        random_state,
        X=None,
    ):
        """Select initial mediods when beginning clustering."""
        if _is_array(self._init):  # Pre assign cluster
            return np.hstack(
                [np.where((X == c).all(axis=1)) for c in self._init]
            ).ravel()

        if self._init == _InitMethod.RANDOM:
            return random_state.choice(len(D), n_clusters, replace=False)

        if self._init == _InitMethod.KMEDOIDSPP:
            return self._kpp_init(D, n_clusters, random_state)

        if self._init == _InitMethod.HEURISTIC:  # Initialization by heuristic
            # Pick K first data points that have the smallest sum distance
            # to every other point. These are the initial medoids.
            return np.argpartition(np.sum(D, axis=1), n_clusters - 1)[
                :n_clusters
            ]

        if self._init == _InitMethod.BUILD:  # Build initialization
            return _build(D, n_clusters).astype(np.int64)

    # Copied from sklearn.cluster.k_means_._k_init
    def _kpp_init(self, D, n_clusters, random_state_, n_local_trials=None):
        """Init n_clusters seeds with a method similar to k-means++

        Parameters
        -----------
        D : array, shape (n_samples, n_samples)
            The distance matrix we will use to select medoid indices.

        n_clusters : integer
            The number of seeds to choose

        random_state : RandomState
            The generator used to initialize the centers.

        n_local_trials : integer, optional
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.

        Notes
        -----
        Selects initial cluster centers for k-medoid clustering in a smart way
        to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
        "k-means++: the advantages of careful seeding". ACM-SIAM symposium
        on Discrete algorithms. 2007

        Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
        which is the implementation used in the aforementioned paper.
        """
        n_samples, _ = D.shape

        centers = np.empty(n_clusters, dtype=int)

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(n_clusters))

        center_id = random_state_.randint(n_samples)
        centers[0] = center_id

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = D[centers[0], :] ** 2
        current_pot = closest_dist_sq.sum()

        # pick the remaining n_clusters-1 points
        for cluster_index in range(1, n_clusters):
            rand_vals = (
                random_state_.random_sample(n_local_trials) * current_pot
            )
            candidate_ids = np.searchsorted(
                stable_cumsum(closest_dist_sq), rand_vals
            )

            # Compute distances to center candidates
            distance_to_candidates = D[candidate_ids, :] ** 2

            # Decide which candidate is the best
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # Compute potential when including center candidate
                new_dist_sq = np.minimum(
                    closest_dist_sq, distance_to_candidates[trial]
                )
                new_pot = new_dist_sq.sum()

                # Store result if it is the best local trial so far
                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[cluster_index] = best_candidate
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        return centers


class CLARA(BaseEstimator, ClusterMixin, TransformerMixin):
    """CLARA clustering.

    Read more in the :ref:`User Guide <CLARA>`.
    CLARA (Clustering for Large Applications) extends k-medoids approach for a
    large number of objects. This algorithm use a sampling approach.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of medoids to
        generate.

    metric : string, or callable, optional, default: 'euclidean'
        What distance metric to use. See :func:metrics.pairwise_distances

    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting PAM. It can be zero
        in which case only the initialization is computed.

    n_sampling : int or None, optional, default : None
        Size of the sampled dataset at each iteration. sampling-size a trade-off
        between complexity and efficiency. If None, then sampling-size is set
        to min(sample_size, 40 + 2 * self.n_clusters) as suggested by the authors of the
        algorithm. must be smaller than sample_size.

    n_sampling_iter : int, optional, default : 5
        Number of different samples that have to be done, or number of iterations.

    random_state : int, RandomState instance or None, optional
        Specify random state for the random number generator. Used to
        initialise medoids when init='random'.

    Attributes
    ----------
    cluster_centers_ : array, shape = (n_clusters, n_features)
            or None if metric == 'precomputed'
        Cluster centers, i.e. medoids (elements from the original dataset)

    medoid_indices_ : array, shape = (n_clusters,)
        The indices of the medoid rows in X

    labels_ : array, shape = (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Examples
    --------
    >>> from sklearn_extra.cluster import CLARA
    >>> import numpy as np
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(centers=[[0,0],[1,1]], n_features=2,random_state=0)
    >>> clara = CLARA(n_clusters=2, random_state=0).fit(X)
    >>> clara.predict([[0,0], [4,4]])
    array([0, 1])
    >>> clara.inertia_
    122.44919397611667

    References
    ----------
        Kaufman, L. and Rousseeuw, P.J. (2008). Partitioning Around Medoids (Program PAM).
        In Finding Groups in Data (eds L. Kaufman and P.J. Rousseeuw).
        doi:10.1002/9780470316801.ch2

    See also
    --------

    KMedoids
        CLARA is a variant of KMedoids that use sub-sampling scheme as such if the
        dataset is sufficiently small, KMedoids is preferable.

    Notes
    -----
    Contrary to KMedoids, CLARA is linear in N the sample size for both the spacial
    and time complexity. On the other hand, it scales quadratically with n_sampling.

    """

    def __init__(
        self,
        n_clusters=8,
        metric="euclidean",
        init="build",
        max_iter=300,
        n_sampling=None,
        n_sampling_iter=5,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.init = init
        self.max_iter = max_iter
        self.n_sampling = n_sampling
        self.n_sampling_iter = n_sampling_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit CLARA to the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features), \
                or (n_n_sampling_iter, n_n_sampling_iter) if metric == 'precomputed'
            Dataset to cluster.

        y : Ignored

        Returns
        -------
        self
        """
        X = check_array(X, dtype=[np.float64, np.float32])
        n = len(X)

        random_state_ = check_random_state(self.random_state)

        if self.n_sampling is None:
            n_sampling = max(
                min(n, 40 + 2 * self.n_clusters), self.n_clusters + 1
            )
        else:
            n_sampling = self.n_sampling

        # Check n_sampling.

        if n < self.n_clusters:
            raise ValueError(
                "sample_size should be greater than self.n_clusters"
            )

        if self.n_clusters >= n_sampling:
            raise ValueError(
                "sampling size must be strictly greater than self.n_clusters"
            )

        medoids_idxs = random_state_.choice(
            np.arange(n), size=self.n_clusters, replace=False
        )
        best_score = np.inf
        for _ in range(self.n_sampling_iter):
            if n_sampling >= n:
                sample_idxs = np.arange(n)
            else:
                sample_idxs = np.hstack(
                    [
                        medoids_idxs,
                        random_state_.choice(
                            np.delete(np.arange(n), medoids_idxs),
                            size=n_sampling - self.n_clusters,
                            replace=False,
                        ),
                    ]
                )
            pam = KMedoids(
                n_clusters=self.n_clusters,
                metric=self.metric,
                method="pam",
                init=self.init,
                max_iter=self.max_iter,
                random_state=random_state_,
            )
            pam.fit(X[sample_idxs])
            self.cluster_centers_ = pam.cluster_centers_
            self.inertia_ = _compute_inertia(self.transform(X))

            if pam.inertia_ < best_score:
                best_score = self.inertia_
                medoids_idxs = pam.medoid_indices_

        self.medoid_indices_ = sample_idxs[medoids_idxs]
        self.labels_ = np.argmin(self.transform(X), axis=1)
        self.n_iter_ = self.n_sampling_iter

        return self

    def transform(self, X):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Data to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return X[:, self.medoid_indices_]
        else:
            check_is_fitted(self, "cluster_centers_")

            Y = self.cluster_centers_
            return pairwise_distances(X, Y=Y, metric=self.metric)

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            New data to predict.

        Returns
        -------
        labels : array, shape = (n_query,)
            Index of the cluster each sample belongs to.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return np.argmin(X[:, self.medoid_indices_], axis=1)
        else:
            check_is_fitted(self, "cluster_centers_")

            # Return data points to clusters based on which cluster assignment
            # yields the smallest distance
            return pairwise_distances_argmin(
                X, Y=self.cluster_centers_, metric=self.metric
            )
