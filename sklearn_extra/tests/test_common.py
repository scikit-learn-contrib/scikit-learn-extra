import pytest
from sklearn.utils import estimator_checks

from sklearn_extra.kernel_approximation import Fastfood
from sklearn_extra.kernel_methods import EigenProClassifier, EigenProRegressor
from sklearn_extra.cluster import KMedoids, CommonNNClustering, CLARA
from sklearn_extra.robust import (
    RobustWeightedClassifier,
    RobustWeightedRegressor,
    RobustWeightedKMeans,
)


ALL_ESTIMATORS = [
    Fastfood,
    KMedoids,
    CLARA,
    EigenProClassifier,
    EigenProRegressor,
    CommonNNClustering,
    RobustWeightedKMeans,
    RobustWeightedRegressor,
    RobustWeightedClassifier,
]


@estimator_checks.parametrize_with_checks([cls() for cls in ALL_ESTIMATORS])
def test_all_estimators(estimator, check, request):
    # TODO: fix this common test failure cf #41
    if isinstance(
        estimator, EigenProClassifier
    ) and "function check_classifier_multioutput" in str(check):
        request.applymarker(
            pytest.mark.xfail(run=False, reason="See issue #41")
        )

    return check(estimator)
