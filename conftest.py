import sys
from distutils.version import LooseVersion
import sklearn

import pytest
from _pytest.doctest import DoctestItem


def pytest_collection_modifyitems(config, items):

    # numpy changed the str/repr formatting of numpy arrays in 1.14. We want to
    # run doctests only for numpy >= 1.14.
    skip_doctests = False
    try:
        import numpy as np

        if LooseVersion(np.__version__) < LooseVersion("1.14") or LooseVersion(
            sklearn.__version__
        ) < LooseVersion("0.23.0"):
            reason = (
                "doctests are only run for numpy >= 1.14 "
                "and scikit-learn >=0.23.0"
            )
            skip_doctests = True
        elif sys.platform.startswith("win32"):
            reason = (
                "doctests are not run for Windows because numpy arrays "
                "repr is inconsistent across platforms."
            )
            skip_doctests = True
    except ImportError:
        pass

    if skip_doctests:
        skip_marker = pytest.mark.skip(reason=reason)

        for item in items:
            if isinstance(item, DoctestItem):
                item.add_marker(skip_marker)
