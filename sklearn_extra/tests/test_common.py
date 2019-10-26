from sklearn.utils import estimator_checks

from sklearn_extra.kernel_approximation import Fastfood
from sklearn_extra.kernel_methods import _eigenpro
from sklearn_extra.cluster import KMedoids

ALL_ESTIMATORS = [
    Fastfood,
    KMedoids,
    _eigenpro.EigenProClassifier,
    _eigenpro.EigenProRegressor,
]

if hasattr(estimator_checks, "parametrize_with_checks"):
    # Common tests are only run on scikit-learn 0.22+

    @estimator_checks.parametrize_with_checks(ALL_ESTIMATORS)
    def test_all_estimators(estimator, check):
        return check(estimator)
