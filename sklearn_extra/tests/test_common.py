import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklearn_extra.kernel_approximation import Fastfood
from sklearn_extra import fast_kernel


@pytest.mark.parametrize(
    "Estimator", [Fastfood, fast_kernel.FKCEigenPro, fast_kernel.FKREigenPro]
)
def test_all_estimators(Estimator, request):
    return check_estimator(Estimator)
