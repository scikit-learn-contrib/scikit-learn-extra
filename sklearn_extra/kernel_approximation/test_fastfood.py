import pytest
import numpy as np

from numpy.testing import assert_array_almost_equal
from sklearn.metrics.pairwise import rbf_kernel

from sklearn_extra.kernel_approximation import Fastfood


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= X.sum(axis=1)[:, np.newaxis]
Y /= Y.sum(axis=1)[:, np.newaxis]


@pytest.mark.parametrize(
    "message, input_, expected",
    [
        ("test n is scaled to be a multiple of d", (16, 20), (16, 32, 2)),
        ("test n equals d", (16, 16), (16, 16, 1)),
        ("test n becomes power of two", (3, 16), (4, 16, 4)),
        ("test all", (7, 12), (8, 16, 2)),
    ],
)
def test_fastfood_enforce_dimensionality_constraint(message, input_, expected):
    d, n = input_
    output = Fastfood._enforce_dimensionality_constraints(d, n)
    assert expected == output, message


def test_fastfood():
    """test that Fastfood fast approximates kernel on random data"""
    # compute exact kernel
    gamma = 10.0
    kernel = rbf_kernel(X, Y, gamma=gamma)

    sigma = np.sqrt(1 / (2 * gamma))

    # approximate kernel mapping
    ff_transform = Fastfood(sigma, n_components=1000, random_state=42)

    pars = ff_transform.fit(X)
    X_trans = pars.transform(X)
    Y_trans = ff_transform.transform(Y)

    kernel_approx = np.dot(X_trans, Y_trans.T)

    print("approximation:", kernel_approx[:5, :5])
    print("true kernel:", kernel[:5, :5])
    assert_array_almost_equal(kernel, kernel_approx, decimal=1)
