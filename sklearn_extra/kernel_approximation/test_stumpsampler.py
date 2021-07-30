import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn_extra.kernel_approximation import StumpsSampler

rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))


def test_abss_expected_output_shape():
    N_COMPONENTS = 100
    abss = StumpsSampler(n_components=N_COMPONENTS, random_state=rng)
    Xt = abss.fit_transform(X)
    assert X.shape[0] == Xt.shape[0]
    assert N_COMPONENTS == Xt.shape[1]


def test_abss_output_values():
    N_COMPONENTS = 100
    Xt_manual = np.zeros((X.shape[0], N_COMPONENTS))
    abss = StumpsSampler(n_components=N_COMPONENTS, random_state=rng)
    Xt = abss.fit_transform(X)
    for col in range(N_COMPONENTS):
        Xt_manual[:, col] = np.sign(
            X[:, abss.random_columns_[col]] - abss.random_offsets_[col]
        )
    assert_array_almost_equal(Xt_manual, Xt)
