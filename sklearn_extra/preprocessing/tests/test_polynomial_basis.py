import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklearn_extra.preprocessing import BernsteinFeatures

from sklearn.utils._testing import (
    assert_array_almost_equal,
)

feature_1d = np.array([0, 0.5, 1, np.nan]).reshape(-1, 1)
feature_2d = np.array([[0, 0.25], [0.5, 0.5], [np.nan, 0.75]])


@parametrize_with_checks([BernsteinFeatures()])
def test_sklearn_compatibility(estimator, check):
    check(estimator)


def test_correct_param_types():
    with pytest.raises(TypeError):
        BernsteinFeatures(na_value="a").fit(feature_1d)

    with pytest.raises(ValueError):
        BernsteinFeatures(degree=-1).fit(feature_1d)

    with pytest.raises(TypeError):
        BernsteinFeatures(degree="a").fit(feature_1d)

    with pytest.raises(TypeError):
        BernsteinFeatures(bias=-1).fit(feature_1d)

    with pytest.raises(TypeError):
        BernsteinFeatures(interactions=-1).fit(feature_1d)


def test_correct_output_one_feature():
    bbt = BernsteinFeatures(degree=2).fit(feature_1d)
    output = bbt.transform(feature_1d)
    expected_output = np.array(
        [[0.0, 0.0], [0.5, 0.25], [0.0, 1.0], [0.0, 0.0]]
    )
    assert_array_almost_equal(output, expected_output)


def test_correct_output_two_features():
    bbt = BernsteinFeatures(degree=2).fit(feature_2d)
    output = bbt.transform(feature_2d)
    expected_output = np.array(
        [
            [0.0, 0.0, 0.375, 0.0625],
            [0.5, 0.25, 0.5, 0.25],
            [0.0, 0.0, 0.375, 0.5625],
        ]
    )
    assert_array_almost_equal(output, expected_output)


def test_correct_output_interactions():
    bbt = BernsteinFeatures(degree=2, interactions=True).fit(feature_2d)
    output = bbt.transform(feature_2d)
    expected_output = np.array(
        [
            [0.0, 0.0, 0.375, 0.0, 0.0, 0.0625, 0.0, 0.0],
            [0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert_array_almost_equal(output, expected_output)
