import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklearn_extra import TemplateEstimator
from sklearn_extra import TemplateClassifier
from sklearn_extra import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
