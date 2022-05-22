import numpy as np
import pytest

from sklearn_extra.robust.mean_estimators import (
    median_of_means,
    huber,
    make_huber_metric,
)
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import HuberRegressor

rng = np.random.RandomState(42)

sample = rng.normal(size=100)

# Check good in normal case
def test_normal():
    assert np.abs(median_of_means(sample, 3, rng)) < 1e-1
    assert np.abs(huber(sample, 1)) < 1e-1


# Check breakdown point for median of means_blocks


def test_mom():
    for num_out in range(1, 49):
        sample_cor = sample
        sample_cor[:num_out] = np.inf
        assert np.abs(median_of_means(sample_cor, num_out, rng)) < 2


def test_huber():
    X = np.hstack([np.zeros(90), np.ones(10)])
    with pytest.warns(None) as record:
        mu = huber(X, c=0.5)
    assert len(record) == 0
    assert np.abs(mu) < 0.1


def test_robust_metric():
    robust_mse = make_huber_metric(mean_squared_error, c=5)
    y_true = np.hstack([np.zeros(95), 20 * np.ones(5)])
    np.random.shuffle(y_true)
    y_pred = np.zeros(100)

    assert robust_mse(y_true, y_pred) < 1


def test_check_robust_cv():

    robust_mse = make_huber_metric(mean_squared_error, c=9)
    rng = np.random.RandomState(42)

    X = rng.uniform(size=100)[:, np.newaxis]
    y = 3 * X.ravel()

    y[[42 // 2, 42, 42 * 2]] = 200  # outliers

    huber_reg = HuberRegressor()
    error_Hub_reg = error_ols = np.mean(
        cross_val_score(huber_reg, X, y, scoring=make_scorer(robust_mse))
    )
    assert error_Hub_reg < 1
