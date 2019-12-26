import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse as sp
from sklearn_extra.feature_weighting import TfigmTransformer


def test_tfigm_transform():
    X = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1]])
    X = sp.csr_matrix(X)
    y = np.array(["a", "b", "a", "c"])

    est = TfigmTransformer()
    X_tr = est.fit_transform(X, y)
    assert X_tr.shape == X.shape
    assert_allclose(est.coef_, [3.333333, 4.5, 2.166667], rtol=1e-5)
