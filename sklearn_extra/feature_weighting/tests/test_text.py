# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import scipy.sparse as sp

import pytest

from sklearn_extra.feature_weighting import TfigmTransformer
from sklearn.datasets import make_classification


@pytest.mark.parametrize("array_format", ["dense", "csr", "csc", "coo"])
def test_tfigm_transform(array_format):
    X = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1]])
    if array_format != "dense":
        X = sp.csr_matrix(X).asformat(array_format)
    y = np.array(["a", "b", "a", "c"])

    alpha = 0.2
    est = TfigmTransformer(alpha=alpha)
    X_tr = est.fit_transform(X, y)

    assert_allclose(est.igm_, [0.20, 0.40, 0.0])
    assert_allclose(est.igm_ + alpha, est.coef_)

    assert X_tr.shape == X.shape
    assert sp.issparse(X_tr) is (array_format != "dense")

    if array_format == "dense":
        assert_allclose(X * est.coef_[None, :], X_tr)
    else:
        assert_allclose(X.A * est.coef_[None, :], X_tr.A)


def test_tfigm_synthetic():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        random_state=0,
        n_classes=5,
        shuffle=False,
    )
    X = (X > 0).astype(np.float)

    est = TfigmTransformer()
    est.fit(X, y)
    # informative features have higher IGM weights than noisy ones.
    # (athough here we lose a lot of information due to thresholding of X).
    assert est.igm_[:5].mean() / est.igm_[5:].mean() > 3


@pytest.mark.parametrize("n_class", [2, 5])
def test_tfigm_random_distribution(n_class):
    rng = np.random.RandomState(0)
    n_samples, n_features = 500, 4
    X = rng.randint(2, size=(n_samples, n_features))
    y = rng.randint(n_class, size=(n_samples,))

    est = TfigmTransformer()
    X_tr = est.fit_transform(X, y)

    # all weighs are strictly positive
    assert_array_less(0, est.igm_)
    # and close to zero, since none of the features are discriminant
    assert_array_less(est.igm_, 0.05)


def test_tfigm_valid_target():
    X = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1]])
    y = None

    est = TfigmTransformer()
    with pytest.raises(ValueError, match="y cannot be None"):
        est.fit(X, y)

    # check asymptotic behaviour for 1 class
    y = [1, 1, 1, 1]
    est = TfigmTransformer()
    est.fit(X, y)
    assert_allclose(est.igm_, np.ones(3))


def test_tfigm_valid_target():
    X = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1]])
    y = [1, 1, 2, 2]

    est = TfigmTransformer(alpha=-1)
    with pytest.raises(ValueError, match="alpha=-1 must be a positive number"):
        est.fit(X, y)

    est = TfigmTransformer(tf_scale="unknown")
    msg = r"tf_scale=unknown should be one of \[None, 'sqrt'"
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)
