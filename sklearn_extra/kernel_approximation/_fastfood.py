# License: BSD 3 clause

import numpy as np
from scipy.stats import chi

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array, check_random_state

from ..utils._cyfht import fht2 as cyfht


class Fastfood(BaseEstimator, TransformerMixin):
    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.

    Fastfood replaces the random matrix of Random Kitchen Sinks (RBFSampler)
    with an approximation that uses the Walsh-Hadamard transformation to gain
    significant speed and storage advantages.  The computational complexity for
    mapping a single example is O(n_components log d).  The space complexity is
    O(n_components).  Hint: n_components should be a power of two. If this is
    not the case, the next higher number that fulfills this constraint is
    chosen automatically.

    Parameters
    ----------
    sigma : float
        Parameter of RBF kernel: exp(-(1/(2*sigma^2)) * x^2)

    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.

    tradeoff_mem_accuracy : "accuracy" or "mem", default: 'accuracy'
        mem:        This version is not as accurate as the option "accuracy",
                    but is consuming less memory.
        accuracy:   The final feature space is of dimension 2*n_components,
                    while being more accurate and consuming more memory.

    random_state : {int, RandomState}, optional
        If int, random_state is the seed used by the random number generator;
        if RandomState instance, random_state is the random number generator.

    Notes
    -----
    See "Fastfood | Approximating Kernel Expansions in Loglinear Time" by
    Quoc Le, Tamas Sarl and Alex Smola.

    Examples
    --------
    See scikit-learn-fastfood/examples/plot_digits_classification_fastfood.py
    for an example how to use fastfood with a primal classifier in comparison
    to an usual rbf-kernel with a dual classifier.

    """

    def __init__(
        self,
        sigma=np.sqrt(1 / 2),
        n_components=100,
        tradeoff_mem_accuracy="accuracy",
        random_state=None,
    ):
        self.sigma = sigma
        self.n_components = n_components
        self.random_state = random_state
        # map to 2*n_components features or to n_components features with less
        # accuracy
        self.tradeoff_mem_accuracy = tradeoff_mem_accuracy

    @staticmethod
    def _is_number_power_of_two(n):
        return n != 0 and ((n & (n - 1)) == 0)

    @staticmethod
    def _enforce_dimensionality_constraints(d, n):
        if not (Fastfood._is_number_power_of_two(d)):
            # find d that fulfills 2^l
            d = np.power(2, np.floor(np.log2(d)) + 1)
        divisor, remainder = divmod(n, d)
        times_to_stack_v = int(divisor)
        if remainder != 0:
            # output info, that we increase n so that d is a divider of n
            n = (divisor + 1) * d
            times_to_stack_v = int(divisor + 1)
        return int(d), int(n), times_to_stack_v

    def _pad_with_zeros(self, X):
        try:
            X_padded = np.pad(
                X,
                ((0, 0), (0, self._number_of_features_to_pad_with_zeros)),
                "constant",
            )
        except AttributeError:
            zeros = np.zeros(
                (X.shape[0], self._number_of_features_to_pad_with_zeros)
            )
            X_padded = np.concatenate((X, zeros), axis=1)

        return X_padded

    @staticmethod
    def _approx_fourier_transformation_multi_dim(result):
        cyfht(result)

    @staticmethod
    def _l2norm_along_axis1(X):
        return np.sqrt(np.einsum("ij,ij->i", X, X))

    def _uniform_vector(self, rng):
        if self.tradeoff_mem_accuracy != "accuracy":
            return rng.uniform(0, 2 * np.pi, size=self._n)
        else:
            return None

    def _apply_approximate_gaussian_matrix(self, B, G, P, X):
        """ Create mapping of all x_i by applying B, G and P step-wise """
        num_examples = X.shape[0]

        result = np.multiply(B, X.reshape((1, num_examples, 1, self._d)))
        result = result.reshape(
            (num_examples * self._times_to_stack_v, self._d)
        )
        Fastfood._approx_fourier_transformation_multi_dim(result)
        result = result.reshape((num_examples, -1))
        np.take(result, P, axis=1, mode="wrap", out=result)
        np.multiply(
            np.ravel(G), result.reshape(num_examples, self._n), out=result
        )
        result = result.reshape(num_examples * self._times_to_stack_v, self._d)
        Fastfood._approx_fourier_transformation_multi_dim(result)
        return result

    def _scale_transformed_data(self, S, VX):
        """ Scale mapped data VX to match kernel(e.g. RBF-Kernel) """
        VX = VX.reshape(-1, self._times_to_stack_v * self._d)

        return (
            1 / (self.sigma * np.sqrt(self._d)) * np.multiply(np.ravel(S), VX)
        )

    def _phi(self, X):
        if self.tradeoff_mem_accuracy == "accuracy":
            return (1 / np.sqrt(X.shape[1])) * np.hstack(
                [np.cos(X), np.sin(X)]
            )
        else:
            np.cos(X + self._U, X)
            return X * np.sqrt(2.0 / X.shape[1])

    def fit(self, X, y=None):
        """Fit the model with X.

        Samples a couple of random based vectors to approximate a Gaussian
        random projection matrix to generate n_components features.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        X = check_array(X, dtype=np.float64)

        d_orig = X.shape[1]
        rng = check_random_state(self.random_state)

        (
            self._d,
            self._n,
            self._times_to_stack_v,
        ) = Fastfood._enforce_dimensionality_constraints(
            d_orig, self.n_components
        )
        self._number_of_features_to_pad_with_zeros = self._d - d_orig

        self._G = rng.normal(size=(self._times_to_stack_v, self._d))
        self._B = rng.choice(
            [-1, 1], size=(self._times_to_stack_v, self._d), replace=True
        )
        self._P = np.hstack(
            [
                (i * self._d) + rng.permutation(self._d)
                for i in range(self._times_to_stack_v)
            ]
        )
        self._S = np.multiply(
            1 / self._l2norm_along_axis1(self._G).reshape((-1, 1)),
            chi.rvs(
                self._d,
                size=(self._times_to_stack_v, self._d),
                random_state=rng,
            ),
        )

        self._U = self._uniform_vector(rng)

        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X = check_array(X, dtype=np.float64)
        X_padded = self._pad_with_zeros(X)
        HGPHBX = self._apply_approximate_gaussian_matrix(
            self._B, self._G, self._P, X_padded
        )
        VX = self._scale_transformed_data(self._S, HGPHBX)
        return self._phi(VX)
