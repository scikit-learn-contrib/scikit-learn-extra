"""BernsteinTransformer for generating polynomial features using the Bernstein basis."""

# Author: Alex Shtoff
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_is_fitted,
    check_scalar,
)
from itertools import combinations
from scipy.stats import binom


class PolynomialBasisTransformer(TransformerMixin, BaseEstimator):
    """
    Polynomial basis transformer for generating polynomial features.

    This transformer generates polynomial features of a given degree
    for each data column separately, or tensor-product features for every
    pair of columns if interactions=True.

    Parameters
    ----------

    degree : int, default=5
        The degree of the polynomial basis.

    bias : bool, default=False
        If True, avoids generating the first basis function, assuming it
        represents the constant term.

    na_value : float, default=0.
        The value to replace NaNs in the input data with.

    interactions : bool, default=False
        If True, generates tensor-product features for every pair of columns. If
        bias=True, the product of the first two basis functions is excluded.


    Notes
    -----
    Inheriting classes should override the `vandermonde_matrix` method to
    generate the Vandermonde matrix of the concrete polynomial basis.


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Vandermonde_matrix
    """

    def __init__(self, degree=5, bias=False, na_value=0.0, interactions=False):
        self.degree = degree
        self.bias = bias
        self.na_value = na_value
        self.interactions = interactions

    def fit(self, X, y=None):
        self.degree = check_scalar(self.degree, "degree", int, min_val=0)
        self.bias = check_scalar(self.bias, "bias", bool)
        self.na_value = check_scalar(self.na_value, "na_value", float)
        self.interactions = check_scalar(
            self.interactions, "interactions", bool
        )
        self._validate_data(X, force_all_finite="allow-nan")
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)

        X = self._validate_data(
            X,
            dtype=FLOAT_DTYPES,
            reset=False,
            force_all_finite="allow-nan",
        )

        # Get the number of columns in the input array
        n_rows, n_features = X.shape

        # Compute the specific polynomial basis for each column
        basis_features = [
            self.feature_matrix(X[:, i]) for i in range(n_features)
        ]

        # create interaction features - basis tensor products
        if self.interactions:
            interaction_features = [
                (u[:, None, :] * v[:, :, None]).reshape(n_rows, -1)
                for u, v in combinations(basis_features, 2)
            ]
            result_basis = interaction_features
        else:
            result_basis = basis_features

        # remove the first basis function, if no bias is required
        if not self.bias:
            result_basis = [basis[:, 1:] for basis in result_basis]

        return np.hstack(result_basis)

    def feature_matrix(self, column):
        vander = self.vandermonde_matrix(column)
        return np.nan_to_num(vander, self.na_value)

    def vandermonde_matrix(self, column):
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_tags(self):
        base_tags = super()._get_tags()
        return base_tags | {
            "allow_nan": True,
            "requires_y": False,
        }


class BernsteinFeatures(PolynomialBasisTransformer):
    """
    Polynomial basis transformer for generating polynomial features using the Bernstein basis.

    See Also
    --------
    PolynomialBasisTransformer

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Bernstein_polynomial
    [2]: https://alexshtf.github.io/2024/02/11/Bernstein-Sklearn.html
    """

    def vandermonde_matrix(self, column):
        basis_idx = np.arange(1 + self.degree)
        basis = binom.pmf(basis_idx, self.degree, column[:, None])
        return basis
