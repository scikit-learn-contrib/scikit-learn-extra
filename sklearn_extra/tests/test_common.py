import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklearn_extra import TemplateEstimator
from sklearn_extra import TemplateClassifier
from sklearn_extra import TemplateTransformer
from sklearn_extra.kernel_approximation import Fastfood


@pytest.mark.parametrize(
    "Estimator",
    [TemplateEstimator, TemplateTransformer, TemplateClassifier,
     Fastfood]
)
def test_all_estimators(Estimator, request):
    return check_estimator(Estimator)
