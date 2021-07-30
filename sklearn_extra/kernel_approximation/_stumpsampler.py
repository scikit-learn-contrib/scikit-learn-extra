import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin, BaseEstimator


class StumpSampler(TransformerMixin, BaseEstimator):
    """Approximates feature map of Stump Kernel
    by Monte Carlo approximation::

        K(x, x') = 1 - 1/a * ||x - x'||_1

    where a - width of the kernel.
    This method requires scaling of the input X, since one width
    is used for all of the features.

    Parameters
    ----------
    a : float, default=1.0
        The width of the kernel.
    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Attributes
    ----------
    random_offsets_ : ndarray of shape (n_components,), dtype=float
        Random offsets used to split features.
    random_columns_ : ndarray of shape (n_components,), dtype=int
        Column indices used to create random stumps.

    [1] "Uniform  Approximation  of  Functions  with  Random  Bases"
    by A. Rahimi and Benjamin Recht.
    (https://authors.library.caltech.edu/75528/1/04797607.pdf)
    [2] "Infinite Ensemble Learning with Support Vector Machines"
    by Hsuan-Tien Lin and Ling Li
    (https://jmlr.org/papers/volume9/lin08a/lin08a.pdf)
    """

    def __init__(self, *, a=1.0, n_components=100, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.a = a

    def fit(self, X, y=None):
        """Fit the model with X.
        Samples random projection according to n_features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        X = self._validate_data(X, accept_sparse=False)
        random_state = check_random_state(self.random_state)
        self.random_columns_ = random_state.randint(
            0, X.shape[1], size=self.n_components
        )
        # widths proportional to max abs of columns
        a = 1.0 / self.a
        self.random_offsets_ = np.asarray(
            [random_state.uniform(-a[i], a[i]) for i in self.random_columns_]
        )
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
        Xt = np.sign(X[:, self.random_columns_] - self.random_offsets_)
        return Xt
