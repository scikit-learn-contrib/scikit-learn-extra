#! /usr/bin/env python
import os

from setuptools import find_packages, setup, Extension

import numpy as np

from Cython.Build import cythonize
from Cython.Distutils import build_ext

# get __version__ from _version.py
ver_file = os.path.join("sklearn_extra", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "scikit-learn-extra"
DESCRIPTION = "A set of tools for scikit-learn."
with open("README.rst", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()
URL = "https://github.com/scikit-learn-contrib/scikit-learn-extra"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://github.com/scikit-learn-contrib/scikit-learn-extra"
VERSION = __version__  # noqa
INSTALL_REQUIRES = [
    # 1.22.4 is the oldest numpy compatible with numpy 2 builds and scipy>=1.13.1.
    "numpy>=1.22.4",
    # First scipy and scikit-learn releases with numpy 2 support.
    "scipy>=1.13.1",
    "scikit-learn>=1.4.2,<2",
    "packaging",
]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: Implementation :: CPython",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov"],
    "docs": [
        "pillow",
        "sphinx",
        "sphinx-gallery",
        "sphinx_rtd_theme",
        "numpydoc",
        "matplotlib",
    ],
}
libraries = []
if os.name == "posix":
    libraries.append("m")

# Target the numpy C-API level matching the numpy floor in INSTALL_REQUIRES,
# so it does not drift with the numpy version used to build.
define_macros = [("NPY_TARGET_VERSION", "NPY_1_22_API_VERSION")]

args = {
    "ext_modules": cythonize(
        [
            Extension(
                "sklearn_extra.utils._cyfht",
                ["sklearn_extra/utils/_cyfht.pyx"],
                include_dirs=[np.get_include()],
                define_macros=define_macros,
            ),
            Extension(
                "sklearn_extra.cluster._k_medoids_helper",
                ["sklearn_extra/cluster/_k_medoids_helper.pyx"],
                include_dirs=[np.get_include()],
                define_macros=define_macros,
            ),
            Extension(
                "sklearn_extra.robust._robust_weighted_estimator_helper",
                ["sklearn_extra/robust/_robust_weighted_estimator_helper.pyx"],
                include_dirs=[np.get_include()],
                define_macros=define_macros,
                libraries=libraries,
            ),
            Extension(
                "sklearn_extra.cluster._commonnn_inner",
                ["sklearn_extra/cluster/_commonnn_inner.pyx"],
                include_dirs=[np.get_include()],
                define_macros=define_macros,
                language="c++",
            ),
        ]
    ),
    "cmdclass": dict(build_ext=build_ext),
}


setup(
    name=DISTNAME,
    description=DESCRIPTION,
    long_description_content_type="text/x-rst",
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.9",
    **args,
)
