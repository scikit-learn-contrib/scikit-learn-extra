import pytest
from sklearn.utils import estimator_checks

from sklearn_extra.kernel_approximation import Fastfood
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
    CommonNNClustering,
    RobustWeightedKMeans,
    RobustWeightedRegressor,
    RobustWeightedClassifier,
]


@estimator_checks.parametrize_with_checks([cls() for cls in ALL_ESTIMATORS])
def test_all_estimators(estimator, check, request):
    # TODO: fix this common test failure cf #41

    # TODO: fix this later, ask people at sklearn to advise on it.
    if isinstance(estimator, RobustWeightedRegressor) and (
        ("function check_regressors_train" in str(check))
        or ("function check_estimators_dtypes" in str(check))
    ):
        request.applymarker(pytest.mark.xfail(run=False))
    if isinstance(estimator, RobustWeightedClassifier) and (
        ("function check_classifiers_train" in str(check))
        or ("function check_estimators_dtypes" in str(check))
    ):
        request.applymarker(pytest.mark.xfail(run=False))

    return check(estimator)
