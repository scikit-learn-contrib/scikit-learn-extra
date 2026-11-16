"""Tests for EMeans."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score

from sklearn_extra.cluster import EMeans
from sklearn_extra.cluster._e_means import (
    _assignment_pool,
    _chunk_size,
    _emeans_core,
    _epoch_rows,
    _resolve_recipe,
    _label_sums,
    _nearest_centers,
    _round_power_of_two,
    _scheduled_alpha,
    _take_rows,
)
from sklearn.utils.extmath import row_norms


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=3000, centers=6, n_features=10, random_state=0)
    return X, y


def test_recovers_blobs(blobs):
    X, y = blobs
    em = EMeans(n_clusters=6, init="k-means++", random_state=0).fit(X)
    assert adjusted_rand_score(y, em.labels_) > 0.99
    assert em.cluster_centers_.shape == (6, 10)
    assert_array_equal(em.labels_, em.predict(X))
    assert em.n_iter_ == 5
    assert em.n_steps_ > em.n_iter_


def test_inertia_matches_labels(blobs):
    X, _ = blobs
    em = EMeans(n_clusters=6, random_state=0).fit(X)
    expected = ((X - em.cluster_centers_[em.labels_]) ** 2).sum()
    assert_allclose(em.inertia_, expected, rtol=1e-6)
    assert_allclose(em.score(X), -expected, rtol=1e-6)


def test_competitive_with_lloyd():
    rng = np.random.RandomState(0)
    X = rng.randn(20000, 32)
    km = KMeans(
        n_clusters=100, n_init=1, max_iter=25, tol=0, random_state=0
    ).fit(X)
    em = EMeans(n_clusters=100, n_epochs=5, random_state=0).fit(X)
    assert em.inertia_ < 1.01 * km.inertia_


def test_reproducible(blobs):
    X, _ = blobs
    a = EMeans(n_clusters=6, random_state=42).fit(X)
    b = EMeans(n_clusters=6, random_state=42).fit(X)
    assert_array_equal(a.cluster_centers_, b.cluster_centers_)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_preserved(blobs, dtype):
    X = blobs[0].astype(dtype)
    em = EMeans(n_clusters=6, random_state=0).fit(X)
    assert em.cluster_centers_.dtype == dtype


def test_init_array_and_kmeanspp(blobs):
    X, _ = blobs
    init = X[:6].copy()
    em = EMeans(n_clusters=6, init=init, random_state=0).fit(X)
    assert em.cluster_centers_.shape == (6, 10)
    em = EMeans(n_clusters=6, init="k-means++", random_state=0).fit(X)
    assert em.cluster_centers_.shape == (6, 10)


def test_fractional_epochs(blobs):
    X, _ = blobs
    em = EMeans(n_clusters=6, n_epochs=1.5, random_state=0).fit(X)
    assert em.n_iter_ == 2
    assert _epoch_rows(1000, 1.5) == [500, 1000]
    assert _epoch_rows(1000, 0.5) == [500]
    assert _epoch_rows(1000, 3) == [1000, 1000, 1000]


def test_manual_batch_and_step(blobs):
    X, _ = blobs
    em = EMeans(n_clusters=6, batch_size=100, alpha0=0.3, random_state=0).fit(
        X
    )
    assert em.batch_size_ == 100
    assert em.alpha0_ == 0.3
    assert em.n_steps_ == 5 * 30


def test_revival_splits_dead_clusters():
    # Every point starts closest to one center, so the others die and
    # must be revived by splits.
    rng = np.random.RandomState(0)
    X = rng.randn(4000, 8)
    init = np.vstack([np.zeros((1, 8)), 1e3 + rng.randn(19, 8)])
    em = EMeans(n_clusters=20, init=init, random_state=0).fit(X)
    assert em.n_splits_ > 0
    assert len(np.unique(em.labels_)) == 20
    em = EMeans(n_clusters=20, init=init, unused_threshold=0, random_state=0)
    with pytest.warns(ConvergenceWarning):
        em.fit(X)
    assert em.n_splits_ == 0


def test_alpha_one_is_lloyd_on_full_batch(blobs):
    # With a full batch and a constant step of 1, each update replaces the
    # centers by the means of their points: one Lloyd iteration per epoch.
    X, _ = blobs
    init = X[:6].copy()
    em = EMeans(
        n_clusters=6,
        n_epochs=1,
        batch_size=len(X),
        alpha0=1.0,
        init=init,
        unused_threshold=0,
    ).fit(X)
    km = KMeans(
        n_clusters=6, init=init, n_init=1, max_iter=1, algorithm="lloyd"
    ).fit(X)
    assert_allclose(em.cluster_centers_, km.cluster_centers_, rtol=1e-10)


def test_recipe_helpers():
    assert _round_power_of_two(0.3) == 1
    assert _round_power_of_two(5) == 4
    assert _round_power_of_two(6) == 8
    batch, alpha0 = _resolve_recipe(10, 4, 5, "throughput", None, None)
    assert 1 <= batch <= 10 and 0 < alpha0 <= 1
    assert _scheduled_alpha(0.5, 0, 10) == pytest.approx(0.5)
    assert _scheduled_alpha(0.5, 10, 10) == pytest.approx(0.05)


def test_core_equivalent_to_reference_step():
    # One batch from a known state: the EMA of counts and sums.
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    init = np.array([[1.0], [11.0]])
    centers, labels, steps, _, _ = _emeans_core(
        X,
        init,
        batch_size=4,
        alpha0=0.5,
        epoch_rows=[4],
        epoch_orders=lambda e: np.arange(4),
        unused_threshold=0,
        split_rng=np.random.RandomState(0),
        chunk_size=4,
        x_squared_norms=row_norms(X, squared=True),
    )
    assert steps == 1
    assert_array_equal(labels, [0, 0, 1, 1])
    # Closing means of the last epoch.
    assert_allclose(centers, [[1.0], [11.0]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_threaded_assignment_matches_serial(dtype, monkeypatch):
    # Force a pool of 3 threads over blocks of 64 rows.
    monkeypatch.setattr(
        "sklearn_extra.cluster._e_means._blas_num_threads", lambda: 3
    )
    rng = np.random.RandomState(0)
    X = rng.randn(1000, 7).astype(dtype)
    centers = rng.randn(40, 7).astype(dtype)
    norms = row_norms(X, squared=True)
    exact = ((X[:, None, :] - centers[None]) ** 2).sum(axis=2)
    serial = _nearest_centers(X, centers, norms, 64)
    # Enough work for a pool: its 3 threads get blocks of at most 64 rows.
    monkeypatch.setattr("sklearn_extra.cluster._e_means._MIN_TASK_WORK", 1)
    with _assignment_pool(len(X), 40, 7, 64, X.dtype.itemsize) as pool:
        assert pool.n_threads == 3
        threaded = _nearest_centers(X, centers, norms, 64, pool)
        indices = rng.permutation(1000)
        rows = np.empty((3000, 7), dtype=dtype)
        _take_rows(X, np.tile(indices, 3), rows, pool)
    assert_array_equal(rows, np.tile(X[indices], (3, 1)))
    for distances, labels in (serial, threaded):
        assert_array_equal(labels, exact.argmin(axis=1))
        assert_allclose(distances, exact.min(axis=1), rtol=1e-4, atol=1e-4)
    # Too few rows to share: no pool.
    with _assignment_pool(63, 40, 7, 64, X.dtype.itemsize) as pool:
        assert pool is None


def test_label_sums_match_add_at():
    rng = np.random.RandomState(0)
    rows = rng.randn(500, 5).astype(np.float32)
    labels = rng.randint(30, size=500)
    expected = np.zeros((40, 5), dtype=np.float32)
    np.add.at(expected, labels, rows)
    sums = _label_sums(labels, rows, 40)
    assert sums.dtype == np.float32
    # Same order of additions, so the sums are bitwise equal.
    assert_array_equal(sums, expected)


def test_chunk_size_bounds():
    assert _chunk_size(2048, 4) == 128
    assert _chunk_size(1, 8) == 4096
    assert _chunk_size(10**6, 8) == 64
    assert _chunk_size(10**9, 8) == 1


@pytest.mark.parametrize(
    "params, match",
    [
        ({"n_clusters": 0}, "n_clusters"),
        ({"n_clusters": 5000}, "n_samples"),
        ({"n_epochs": 0}, "n_epochs"),
        ({"n_epochs": float("inf")}, "n_epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"alpha0": 1.5}, "alpha0"),
        ({"batch_law": "fast"}, "batch_law"),
        ({"unused_threshold": -1}, "unused_threshold"),
        ({"init": "kmeans"}, "init"),
        ({"init": np.zeros((2, 2))}, "init"),
    ],
)
def test_invalid_params(blobs, params, match):
    X, _ = blobs
    with pytest.raises(ValueError, match=match):
        EMeans(**params).fit(X)
