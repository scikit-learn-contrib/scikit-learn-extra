import numpy as np

from sklearn_extra.robust.mean_estimators import median_of_means, huber
from numpy.testing import assert_almost_equal

rng = np.random.RandomState(42)

sample = rng.normal(size=100)

# Check good in normal case
def test_normal():
    assert np.abs(median_of_means(sample,3, rng)) < 1e-1
    assert np.abs(huber(sample,1)) < 1e-1

# Check breakdown point for median of means_blocks

def test_mom():
    for num_out in range(1,49):
        sample_cor = sample
        sample_cor[:num_out] = np.inf
        assert np.abs(median_of_means(sample_cor,num_out, rng)) < 2
