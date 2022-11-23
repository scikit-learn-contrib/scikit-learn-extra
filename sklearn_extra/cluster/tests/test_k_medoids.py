"""Testing for K-Medoids"""
import warnings
import numpy as np
from unittest import mock
from scipy.sparse import csc_matrix
import pytest

from sklearn.datasets import load_iris, fetch_20newsgroups_vectorized
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from numpy.testing import assert_allclose, assert_array_equal

from sklearn_extra.cluster import KMedoids, CLARA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


seed = 0
X = np.random.RandomState(seed).rand(100, 5)

# test kmedoid's results
rng = np.random.RandomState(seed)
X_cc, y_cc = make_blobs(
    n_samples=100,
    centers=np.array([[-1, -1], [1, 1]]),
    random_state=rng,
    shuffle=False,
    cluster_std=0.2,
)


@pytest.mark.parametrize("method", ["alternate", "pam"])
@pytest.mark.parametrize(
    "init", ["random", "heuristic", "build", "k-medoids++"]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmedoid_results(method, init, dtype):
    expected = np.hstack([np.zeros(50), np.ones(50)])
    km = KMedoids(n_clusters=2, init=init, method=method, random_state=rng)
    km.fit(X_cc.astype(dtype))
    # This test use data that are not perfectly separable so the
    # accuracy is not 1. Accuracy around 0.85
    assert (np.mean(km.labels_ == expected) > 0.8) or (
        1 - np.mean(km.labels_ == expected) > 0.8
    )
    assert dtype is np.dtype(km.cluster_centers_.dtype).type
    assert dtype is np.dtype(km.transform(X_cc.astype(dtype)).dtype).type


@pytest.mark.parametrize("method", ["alternate", "pam"])
@pytest.mark.parametrize(
    "init", ["random", "heuristic", "build", "k-medoids++"]
)
def test_kmedoid_nclusters(method, init):
    n_clusters = 50

    km = KMedoids(
        n_clusters=n_clusters,
        init=init,
        method=method,
        max_iter=1,
        random_state=rng,
    )
    km.fit(X_cc)
    assert len(np.unique(km.medoid_indices_)) == n_clusters


def test_clara_results():
    expected = np.hstack([np.zeros(50), np.ones(50)])
    km = CLARA(n_clusters=2)
    km.fit(X_cc)
    # This test use data that are not perfectly separable so the
    # accuracy is not 1. Accuracy around 0.85
    assert (np.mean(km.labels_ == expected) > 0.8) or (
        1 - np.mean(km.labels_ == expected) > 0.8
    )


def test_medoids_invalid_method():
    with pytest.raises(ValueError, match="invalid is not supported"):
        KMedoids(n_clusters=1, method="invalid").fit([[0, 1], [1, 1]])


def test_medoids_invalid_init():
    with pytest.raises(ValueError, match="init needs to be one of"):
        KMedoids(n_clusters=1, init="invalid").fit([[0, 1], [1, 1]])


def test_kmedoids_input_validation_and_fit_check():
    rng = np.random.RandomState(seed)
    # Invalid parameters
    msg = "n_clusters should be a nonnegative integer. 0 was given"
    with pytest.raises(ValueError, match=msg):
        KMedoids(n_clusters=0).fit(X)

    msg = "n_clusters should be a nonnegative integer. None was given"
    with pytest.raises(ValueError, match=msg):
        KMedoids(n_clusters=None).fit(X)

    msg = "max_iter should be a nonnegative integer. -1 was given"
    with pytest.raises(ValueError, match=msg):
        KMedoids(n_clusters=1, max_iter=-1).fit(X)

    msg = "max_iter should be a nonnegative integer. None was given"
    with pytest.raises(ValueError, match=msg):
        KMedoids(n_clusters=1, max_iter=None).fit(X)

    msg = (
        r"init needs to be one of the following: "
        r".*random.*heuristic.*k-medoids\+\+"
    )
    with pytest.raises(ValueError, match=msg):
        KMedoids(init=None).fit(X)

    # Trying to fit 3 samples to 8 clusters
    msg = (
        "The number of medoids \(8\) must be less "
        "than the number of samples 5."
    )
    Xsmall = rng.rand(5, 2)
    with pytest.raises(ValueError, match=msg):
        KMedoids(n_clusters=8).fit(Xsmall)


def test_random_deterministic():
    """Random_state should determine 'random' init output."""
    rng = np.random.RandomState(seed)

    X = load_iris()["data"]
    D = euclidean_distances(X)

    medoids = KMedoids(init="random")._initialize_medoids(D, 4, rng)
    assert_array_equal(medoids, [114, 62, 33, 107])


def test_heuristic_deterministic():
    """Result of heuristic init method should not depend on rnadom state."""
    rng1 = np.random.RandomState(1)
    rng2 = np.random.RandomState(2)
    X = load_iris()["data"]
    D = euclidean_distances(X)

    medoids_1 = KMedoids(init="heuristic")._initialize_medoids(D, 10, rng1)

    medoids_2 = KMedoids(init="heuristic")._initialize_medoids(D, 10, rng2)

    assert_array_equal(medoids_1, medoids_2)


def test_update_medoid_idxs_empty_cluster():
    """Label is unchanged for an empty cluster."""
    D = np.zeros((3, 3))
    labels = np.array([0, 0, 0])
    medoid_idxs = np.array([0, 1])
    kmedoids = KMedoids(n_clusters=2)

    # Swallow empty cluster warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmedoids._update_medoid_idxs_in_place(D, labels, medoid_idxs)

    assert_array_equal(medoid_idxs, [0, 1])


def test_kmedoids_empty_clusters():
    """When a cluster is empty, it should throw a warning."""
    rng = np.random.RandomState(seed)
    X = [[1], [1], [1]]
    kmedoids = KMedoids(n_clusters=2, random_state=rng)
    with pytest.warns(UserWarning, match="Cluster 1 is empty!"):
        kmedoids.fit(X)


@mock.patch.object(KMedoids, "_kpp_init", return_value=object())
def test_kpp_called(_kpp_init_mocked):
    """KMedoids._kpp_init method should be called by _initialize_medoids"""
    D = np.array([[0, 1], [1, 0]])
    n_clusters = 2
    rng = np.random.RandomState(seed)
    kmedoids = KMedoids()
    kmedoids.init = "k-medoids++"
    # set _kpp_init_mocked.return_value to a singleton
    initial_medoids = kmedoids._initialize_medoids(D, n_clusters, rng)

    # assert that _kpp_init was called and its result was returned.
    _kpp_init_mocked.assert_called_once_with(D, n_clusters, rng)
    assert initial_medoids == _kpp_init_mocked.return_value


def test_kmedoids_pp():
    """Initial clusters should be well-separated for k-medoids++"""
    rng = np.random.RandomState(seed)
    kmedoids = KMedoids()
    X = [
        [10, 0],
        [11, 0],
        [0, 10],
        [0, 11],
        [10, 10],
        [11, 10],
        [12, 10],
        [10, 11],
    ]
    D = euclidean_distances(X)

    centers = kmedoids._kpp_init(D, n_clusters=3, random_state_=rng)

    assert len(centers) == 3

    inter_medoid_distances = D[centers][:, centers]
    assert np.all((inter_medoid_distances > 5) | (inter_medoid_distances == 0))


def test_precomputed():
    """Test the 'precomputed' distance metric."""
    rng = np.random.RandomState(seed)
    X_1 = [[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]]
    D_1 = euclidean_distances(X_1)
    X_2 = [[1.1, 0.0], [0.0, 0.9]]
    D_2 = euclidean_distances(X_2, X_1)

    kmedoids = KMedoids(metric="precomputed", n_clusters=2, random_state=rng)
    kmedoids.fit(D_1)

    assert_allclose(kmedoids.inertia_, 0.2)
    assert_array_equal(kmedoids.medoid_indices_, [2, 0])
    assert_array_equal(kmedoids.labels_, [1, 1, 0, 0])
    assert kmedoids.cluster_centers_ is None

    med_1, med_2 = tuple(kmedoids.medoid_indices_)
    predictions = kmedoids.predict(D_2)
    assert_array_equal(predictions, [med_1 // 2, med_2 // 2])

    transformed = kmedoids.transform(D_2)
    assert_array_equal(transformed, D_2[:, kmedoids.medoid_indices_])


def test_kmedoids_fit_naive():
    n_clusters = 3
    metric = "euclidean"

    model = KMedoids(n_clusters=n_clusters, metric=metric)
    Xnaive = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    model.fit(Xnaive)

    assert_array_equal(
        model.cluster_centers_, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    assert_array_equal(model.labels_, [0, 1, 2])
    assert model.inertia_ == 0.0

    # diagonal must be zero, off-diagonals must be positive
    X_new = model.transform(Xnaive)
    for c in range(n_clusters):
        assert X_new[c, c] == 0
        for c2 in range(n_clusters):
            if c != c2:
                assert X_new[c, c2] > 0


def test_max_iter():
    """Test that warning message is thrown when max_iter is reached."""
    rng = np.random.RandomState(seed)
    X_iris = load_iris()["data"]

    model = KMedoids(
        n_clusters=10, init="random", random_state=rng, max_iter=1
    )
    msg = "Maximum number of iteration reached before"

    with pytest.warns(UserWarning, match=msg):
        model.fit(X_iris)


def test_kmedoids_iris():
    """Test kmedoids on the Iris dataset"""
    rng = np.random.RandomState(seed)
    X_iris = load_iris()["data"]

    ref_model = KMeans(n_clusters=3).fit(X_iris)

    avg_dist_to_closest_centroid = (
        ref_model.transform(X_iris).min(axis=1).mean()
    )

    for init in ["random", "heuristic", "k-medoids++"]:
        distance_metric = "euclidean"
        model = KMedoids(
            n_clusters=3, metric=distance_metric, init=init, random_state=rng
        )
        model.fit(X_iris)

        # test convergence in reasonable number of steps
        assert model.n_iter_ < (len(X_iris) // 10)

        distances = PAIRWISE_DISTANCE_FUNCTIONS[distance_metric](X_iris)
        avg_dist_to_random_medoid = np.mean(distances.ravel())
        avg_dist_to_closest_medoid = model.inertia_ / X_iris.shape[0]
        # We want distance-to-closest-medoid to be reduced from average
        # distance by more than 50%
        assert avg_dist_to_random_medoid > 2 * avg_dist_to_closest_medoid
        # When K-Medoids is using Euclidean distance,
        # we can compare its performance to
        # K-Means. We want the average distance to cluster centers
        # to be similar between K-Means and K-Medoids
        assert_allclose(
            avg_dist_to_closest_medoid, avg_dist_to_closest_centroid, rtol=0.1
        )


def test_kmedoids_fit_predict_transform():
    rng = np.random.RandomState(seed)
    model = KMedoids(random_state=rng)

    labels1 = model.fit_predict(X)
    assert len(labels1) == 100
    assert_array_equal(labels1, model.labels_)

    labels2 = model.predict(X)
    assert_array_equal(labels1, labels2)

    Xt1 = model.fit_transform(X)
    assert_array_equal(Xt1.shape, (100, model.n_clusters))

    Xt2 = model.transform(X)
    assert_array_equal(Xt1, Xt2)


def test_callable_distance_metric():
    rng = np.random.RandomState(seed)

    def my_metric(a, b):
        return np.sqrt(np.sum(np.power(a - b, 2)))

    model = KMedoids(random_state=rng, metric=my_metric)
    labels1 = model.fit_predict(X)
    assert len(labels1) == 100
    assert_array_equal(labels1, model.labels_)


def test_outlier_robustness():
    rng = np.random.RandomState(seed)
    kmeans = KMeans(n_clusters=2, random_state=rng)
    kmedoids = KMedoids(n_clusters=2, random_state=rng)

    X = [[-11, 0], [-10, 0], [-9, 0], [0, 0], [1, 0], [2, 0], [1000, 0]]

    kmeans.fit(X)
    kmedoids.fit(X)

    assert_array_equal(kmeans.labels_, [0, 0, 0, 0, 0, 0, 1])
    assert_array_equal(kmedoids.labels_, [0, 0, 0, 1, 1, 1, 1])


def test_kmedoids_on_sparse_input():
    rng = np.random.RandomState(seed)
    model = KMedoids(n_clusters=2, random_state=rng)
    row = np.array([1, 0])
    col = np.array([0, 4])
    data = np.array([1, 1])
    X = csc_matrix((data, (row, col)), shape=(2, 5))
    labels = model.fit_predict(X)
    assert len(labels) == 2
    assert_array_equal(labels, model.labels_)


# Test the build initialization.
def test_build():
    X, y = fetch_20newsgroups_vectorized(return_X_y=True)
    # Select only the first 500 samples
    X = X[:500]
    y = y[:500]
    # Precompute cosine distance matrix
    diss = cosine_distances(X)
    # run build
    ske = KMedoids(20, "precomputed", init="build", max_iter=0)
    ske.fit(diss)
    assert ske.inertia_ <= 230
    assert len(np.unique(ske.labels_)) == 20


def test_clara_consistency_iris():
    # test that CLARA is PAM when full sample is used

    rng = np.random.RandomState(seed)
    X_iris = load_iris()["data"]

    clara = CLARA(
        n_clusters=3,
        n_sampling_iter=1,
        n_sampling=len(X_iris),
        random_state=rng,
    )

    model = KMedoids(n_clusters=3, init="build", random_state=rng)

    model.fit(X_iris)
    clara.fit(X_iris)
    assert np.sum(model.labels_ == clara.labels_) == len(X_iris)


def test_seuclidean():
    with pytest.warns(None) as record:
        km = KMedoids(2, metric="seuclidean", method="pam")
        km.fit(np.array([0, 0, 0, 1]).reshape((4, 1)))
        km.predict(np.array([0, 0, 0, 1]).reshape((4, 1)))
        km.transform(np.array([0, 0, 0, 1]).reshape((4, 1)))
    assert len(record) == 0


def test_medoids_indices():
    rng = np.random.RandomState(seed)
    X_iris = load_iris()["data"]

    clara = CLARA(
        n_clusters=3,
        n_sampling_iter=1,
        n_sampling=len(X_iris),
        random_state=rng,
    )

    model = KMedoids(n_clusters=3, init="build", random_state=rng)

    centroids = np.array([X_iris[0], X_iris[50]])
    array_like_model = KMedoids(
        n_clusters=len(centroids), init=centroids, max_iter=0
    )

    model.fit(X_iris)
    clara.fit(X_iris)
    array_like_model.fit(X_iris)
    assert_array_equal(X_iris[model.medoid_indices_], model.cluster_centers_)
    assert_array_equal(X_iris[clara.medoid_indices_], clara.cluster_centers_)
    assert_array_equal(centroids, array_like_model.cluster_centers_)


def test_array_like_init():
    centroids = np.array([X_cc[0], X_cc[50]])

    expected = np.hstack([np.zeros(50), np.ones(50)])
    km = KMedoids(n_clusters=len(centroids), init=centroids)
    km.fit(X_cc)
    # # This test use data that are not perfectly separable so the
    # # accuracy is not 1. Accuracy around 0.85
    assert (np.mean(km.labels_ == expected) > 0.8) or (
        1 - np.mean(km.labels_ == expected) > 0.8
    )

    # Override n_clusters if array-like init method is used
    km = KMedoids(n_clusters=len(centroids) + 2, init=centroids)
    km.fit(X_cc)

    assert len(km.cluster_centers_) == len(centroids)
