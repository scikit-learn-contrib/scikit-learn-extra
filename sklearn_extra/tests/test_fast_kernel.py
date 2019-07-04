import numpy as np

from sklearn.datasets import make_regression, make_classification
from sklearn.utils.testing import assert_array_almost_equal
from sklearn_extra.fast_kernel import FKR_EigenPro, FKC_EigenPro

import pytest

np.random.seed(1)
# Tests for Fast Kernel Regression and Classification.


def gen_regression(params):
    """Generate a regression problem with make_regression
    where random_state=1"""
    return make_regression(**params, random_state=1)


def gen_classification(params):
    """Generate a classification problem with make_classification
    where random_state=1"""
    return make_classification(**params, random_state=1)


@pytest.mark.parametrize(
    "estimator, data",
    [
        (FKR_EigenPro, gen_regression({})),
        (FKC_EigenPro, gen_classification({})),
    ],
)
@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"kernel": "not_a_kernel"}, "Unknown kernel 'not_a_kernel'"),
        ({"n_epoch": 0}, "n_epoch should be positive, was 0"),
        ({"n_epoch": -1}, "n_epoch should be positive, was -1"),
        ({"n_components": -1}, "n_components should be non-negative, was -1"),
        (
            {"subsample_size": -1},
            "subsample_size should be non-negative, was -1",
        ),
        ({"batch_size": 0}, "batch_size should be positive, was 0"),
        ({"batch_size": -1}, "batch_size should be positive, was -1"),
        ({"bandwidth": 0}, "bandwidth should be positive, was 0"),
        ({"bandwidth": -1}, "bandwidth should be positive, was -1"),
    ],
)
def test_parameter_validation(estimator, data, params, err_msg):
    X, y = data
    with pytest.raises(ValueError, match=err_msg):
        estimator(**params).fit(X, y)


@pytest.mark.parametrize(
    "data, estimator",
    [
        # Test gaussian kernel
        (
            gen_regression({}),
            FKR_EigenPro(
                kernel="gaussian", n_epoch=100, bandwidth=10, random_state=1
            ),
        ),
        # Test laplacian kernel
        (
            gen_regression({}),
            FKR_EigenPro(
                kernel="laplace", n_epoch=100, bandwidth=8, random_state=1
            ),
        ),
        # Test cauchy kernel
        (
            gen_regression({}),
            FKR_EigenPro(
                kernel="cauchy",
                n_epoch=100,
                bandwidth=10,
                subsample_size=1000,
                random_state=1,
            ),
        ),
        # Test with multiple outputs
        (
            gen_regression({"n_features": 200, "n_targets": 30}),
            FKR_EigenPro(
                kernel="gaussian", n_epoch=100, bandwidth=14, random_state=1
            ),
        ),
        # Test with a very large number of input features
        (
            gen_regression({"n_features": 10000}),
            FKR_EigenPro(
                kernel="gaussian", n_epoch=100, bandwidth=1, random_state=1
            ),
        ),
        # Test a very simple underlying distribution
        (
            gen_regression({"n_informative": 1}),
            FKR_EigenPro(
                batch_size=500,
                kernel="gaussian",
                n_epoch=100,
                bandwidth=10,
                random_state=1,
            ),
        ),
        # Test a very complex underlying distribution
        (
            gen_regression({"n_samples": 500, "n_informative": 100}),
            FKR_EigenPro(
                kernel="gaussian", n_epoch=60, bandwidth=10, random_state=1
            ),
        ),
    ],
)
def test_regressor_accuracy(data, estimator):
    """
    Test the accuracy of the Fast Kernel Regressor on multiple
    data sets with different parameter inputs. We expect that the
    regressor should achieve near-zero training error after sufficient
    training time.
    :param data: A tuple containing the input and output training data
    :param Estimator: The regressor to do predictions with.
    """
    X, y = data
    prediction = estimator.fit(X, y).predict(X)
    assert_array_almost_equal(abs(prediction / y) / 2.0, 0.5, decimal=2)


def test_fast_kernel_regression_duplicate_data():
    """Test the performance when some data is repeated"""
    X, y = make_regression(random_state=1)
    X, y = np.concatenate([X, X]), np.concatenate([y, y])
    fkr_prediction = (
        FKR_EigenPro(
            kernel="gaussian", n_epoch=100, bandwidth=5, random_state=1
        )
        .fit(X, y)
        .predict(X)
    )
    assert_array_almost_equal(abs(fkr_prediction / y) / 2.0, 0.5, decimal=2)


def test_fast_kernel_regression_conflict_data():
    """Make sure the regressor doesn't crash when conflicting
    data is given"""
    X, y = make_regression(random_state=1)
    y = np.reshape(y, (-1, 1))
    X, y = X, np.hstack([y, y + 2])
    # Make sure we don't throw an error when fitting or predicting
    FKR_EigenPro(kernel="linear", n_epoch=5, bandwidth=1, random_state=1).fit(
        X, y
    ).predict(X)


# Tests for FastKernelClassification


@pytest.mark.parametrize(
    "data, estimator",
    [
        # Test gaussian kernel
        (
            gen_classification({"n_samples": 10, "hypercube": False}),
            FKC_EigenPro(
                batch_size=9,
                kernel="gaussian",
                bandwidth=2.5,
                n_epoch=100,
                random_state=1,
            ),
        ),
        # Test laplacian kernel
        (
            gen_classification({}),
            FKC_EigenPro(
                kernel="laplace", n_epoch=100, bandwidth=13, random_state=1
            ),
        ),
        # Test cauchy kernel
        (
            gen_classification({}),
            FKC_EigenPro(
                kernel="cauchy", n_epoch=100, bandwidth=10, random_state=1
            ),
        ),
        # Test with a very large number of input features
        # and samples, shifted around and scaled
        (
            gen_classification(
                {
                    "n_samples": 500,
                    "n_features": 500,
                    "n_informative": 160,
                    "scale": 30,
                    "shift": 6,
                }
            ),
            FKC_EigenPro(
                kernel="gaussian", n_epoch=100, bandwidth=20, random_state=1
            ),
        ),
        # Test a distribution that has been shifted
        (
            gen_classification({"shift": 1, "hypercube": False}),
            FKC_EigenPro(
                kernel="gaussian", n_epoch=200, bandwidth=8, random_state=1
            ),
        ),
        # Test with many redundant features.
        (
            gen_classification({"n_redundant": 18}),
            FKC_EigenPro(
                kernel="laplace", n_epoch=100, bandwidth=20, random_state=1
            ),
        ),
    ],
)
def test_classifier_accuracy(data, estimator):
    """
    Test the accuracy of the Fast Kernel Classification on multiple
    data sets with different parameter inputs. We expect that the
    classification should achieve zero training error after sufficient
    training time.
    :param data: A tuple containing the input and output training data
    :param Estimator: The classifier to do predictions with.
    """
    X, y = data
    prediction = estimator.fit(X, y).predict(X)
    assert_array_almost_equal(prediction, y)


def test_fast_kernel_classification_duplicate_data():
    """
    Make sure that the classifier correctly handles cases
    where some data is repeated.
    """
    X, y = make_classification(n_features=200, n_repeated=50, random_state=1)
    fkc_prediction = (
        FKC_EigenPro(
            kernel="gaussian", n_epoch=60, bandwidth=1, random_state=1
        )
        .fit(X, y)
        .predict(X)
    )
    assert_array_almost_equal(fkc_prediction, y)


def test_fast_kernel_classification_conflict_data():
    """Make sure that the classifier doesn't crash
    when given conflicting input data"""
    X, y = make_classification(random_state=1)
    X, y = np.concatenate([X, X]), np.concatenate([y, 1 - y])
    # Make sure we don't throw an error when fitting or predicting
    FKC_EigenPro(kernel="linear", n_epoch=5, bandwidth=5, random_state=1).fit(
        X, y
    ).predict(X)
