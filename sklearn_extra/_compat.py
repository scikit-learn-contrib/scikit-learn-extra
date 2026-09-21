try:
    from sklearn.utils.validation import validate_data
except ImportError:

    def validate_data(estimator, *args, **kwargs):
        return estimator._validate_data(*args, **kwargs)


try:
    from sklearn.utils import get_tags
except ImportError:

    def estimator_type(estimator):
        return estimator._estimator_type

else:

    def estimator_type(estimator):
        return get_tags(estimator).estimator_type
