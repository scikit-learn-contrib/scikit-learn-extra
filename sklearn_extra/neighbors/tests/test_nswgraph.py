import pytest
from sklearn_extra.neighbors import NSWGraph
from sklearn.utils.validation import check_array, check_random_state
from numpy.testing import assert_array_almost_equal
import numpy as np
from sklearn.metrics import DistanceMetric


def brute_force_neighbors(X, Y, k, metric, **kwargs):
    """True neighbours for assertion check. Taken from BallTree tests in Scikit-Learn"""
    X, Y = check_array(X), check_array(Y)
    D = DistanceMetric.get_metric(metric, **kwargs).pairwise(Y, X)
    ind = np.argsort(D, axis=1)[:, :k]
    return ind


def test_array_object_type():
    """Check that we do not accept object dtype array. Taken from BallTree tests in Scikit-Learn"""
    X = np.array([(1, 2, 3), (2, 5), (5, 5, 1, 2)], dtype=object)
    nswgraph = NSWGraph()
    with pytest.raises(
        ValueError, match="setting an array element with a sequence"
    ):
        nswgraph.build(X)


def test_init_types():
    """Make sure that the init args validation check works properly"""
    regularity = -1
    with pytest.raises(ValueError):
        nswgraph = NSWGraph(regularity=regularity)

    guard_hops = "something"
    with pytest.raises(ValueError):
        nswgraph = NSWGraph(guard_hops=guard_hops)

    quantize = "True"
    with pytest.raises(ValueError):
        nswgraph = NSWGraph(quantize=quantize)

    quantization_levels = 1.5
    with pytest.raises(ValueError):
        nswgraph = NSWGraph(
            quantize=True, quantization_levels=quantization_levels
        )


def test_query():
    """Make sure that neighbours query works satisfactory using NSWGraph"""

    rng = check_random_state(0)
    X = rng.random_sample((30, 16))
    nswgraph = NSWGraph()
    nswgraph.build(X)
    k = 3
    X_val = X[:1]
    ind1 = nswgraph.query(X_val, k=k, return_distance=False)
    ind2 = brute_force_neighbors(X, X_val, k=k, metric="euclidean")
    assert_array_almost_equal(ind1, ind2)
