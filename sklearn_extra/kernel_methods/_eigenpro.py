# Authors: Alex Li <7Alex7Li@gmail.com>
#          Siyuan Ma <Siyuan.ma9@gmail.com>

import numpy as np
from scipy.linalg import eigh, LinAlgError
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_X_y


class BaseEigenPro(BaseEstimator):
    """
    Base class for EigenPro iteration.
    """

    def __init__(
        self,
        batch_size="auto",
        n_epoch=2,
        n_components=1000,
        subsample_size="auto",
        kernel="rbf",
        gamma="scale",
        degree=3,
        coef0=1,
        kernel_params=None,
        random_state=None,
    ):
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.n_components = n_components
        self.subsample_size = subsample_size
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.random_state = random_state

    def _kernel(self, X, Y):
        """Calculate the kernel matrix

        Parameters
        ---------
        X : {float, array}, shape = [n_samples, n_features]
            Input data.

        Y : {float, array}, shape = [n_centers, n_targets]
            Kernel centers.

        Returns
        -------
        K : {float, array}, shape = [n_samples, n_centers]
            Kernel matrix.
        """
        if (
            self.kernel != "rbf"
            and self.kernel != "laplace"
            and self.kernel != "cauchy"
        ):
            if callable(self.kernel):
                params = self.kernel_params or {}
            else:
                params = {
                    "gamma": self.gamma_,
                    "degree": self.degree,
                    "coef0": self.coef0,
                }
            return pairwise_kernels(
                X, Y, metric=self.kernel, filter_params=True, **params
            )
        distance = euclidean_distances(X, Y, squared=True)
        bandwidth = np.float32(1.0 / np.sqrt(2.0 * self.gamma_))
        if self.kernel == "rbf":
            distance = -self.gamma_ * distance
            K = np.exp(distance)
        elif self.kernel == "laplace":
            d = np.maximum(distance, 0)
            K = np.exp(-np.sqrt(d) / bandwidth)
        else:  # self.kernel == "cauchy":
            K = 1 / (1 + 2.0 * self.gamma_ * distance)
        return K

    def _nystrom_svd(self, X, n_components):
        """Compute the top eigensystem of a kernel
        operator using Nystrom method

        Parameters
        ----------
        X : {float, array}, shape = [n_subsamples, n_features]
            Subsample feature matrix.

        n_components : int
            Number of top eigencomponents to be restored.

        Returns
        -------
        E : {float, array}, shape = [k]
            Top eigenvalues.

        Lambda : {float, array}, shape = [n_subsamples, k]
            Top eigenvectors of a subsample kernel matrix (which can be
            directly used to approximate the eigenfunctions of the kernel
            operator).
        """
        m, _ = X.shape
        K = self._kernel(X, X)

        W = K / m
        try:
            E, Lambda = eigh(W, eigvals=(m - n_components, m - 1))
        except LinAlgError:
            # Use float64 when eigh fails due to precision
            W = np.float64(W)
            E, Lambda = eigh(W, eigvals=(m - n_components, m - 1))
            E, Lambda = np.float32(E), np.float32(Lambda)
        # Flip so eigenvalues are in descending order.
        E = np.maximum(np.float32(1e-7), np.flipud(E))
        Lambda = np.fliplr(Lambda)[:, :n_components] / np.sqrt(
            m, dtype="float32"
        )

        return E, Lambda

    def _setup(self, feat, max_components, mG, alpha):
        """Compute preconditioner and scale factors for EigenPro iteration

        Parameters
        ----------
        feat : {float, array}, shape = [n_samples, n_features]
            Feature matrix (normally from training data).

        max_components : int
            Maximum number of components to be used in EigenPro iteration.

        mG : int
            Maximum batch size to fit in memory.

        alpha : float
            Exponential factor (< 1) for eigenvalue ratio.

        Returns
        -------
        max_S : float
            Normalized largest eigenvalue.

        max_kxx : float
            Maximum of k(x,x) where k is the EigenPro kernel.

        E : {float, array}, shape = [k]
            Preconditioner for EigenPro

        Lambda : {float, array}, shape = [n_subsamples, k]
            Top eigenvectors of a subsample kernel matrix
        """
        alpha = np.float32(alpha)

        # Estimate eigenvalues (S) and eigenvectors (V) of the kernel matrix
        # corresponding to the feature matrix.
        E, Lambda = self._nystrom_svd(feat, max_components)
        n_subsamples = feat.shape[0]

        # Calculate the number of components to be used such that the
        # corresponding batch size is bounded by the subsample size and the
        # memory size.
        max_bs = min(max(n_subsamples / 5, mG), n_subsamples)
        n_components = np.sum(np.power(1 / E, alpha) < max_bs) - 1
        if n_components < 2:
            n_components = min(E.shape[0] - 1, 2)

        Lambda = Lambda[:, :n_components]
        scale = np.power(E[0] / E[n_components], alpha)

        # Compute part of the preconditioner for step 2 of gradient descent in
        # the eigenpro model
        D = (1 - np.power(E[n_components] / E[:n_components], alpha)) / E[
            :n_components
        ]

        max_S = E[0].astype(np.float32)
        kxx = 1 - np.sum(Lambda ** 2, axis=1) * n_subsamples
        return max_S / scale, np.max(kxx), D, Lambda

    def _initialize_params(self, X, Y, random_state):
        """
        Validate parameters passed to the model, choose parameters
        that have not been passed in, and run setup for EigenPro iteration.
        Parameters
        ----------
        X : {float, array}, shape = [n_samples, n_features]
            Training data.

        Y : {float, array}, shape = [n_samples, n_targets]
            Training targets.

        random_state : RandomState instance
            The random state to use for random number generation

        Returns
        -------
        Y : {float, array}, shape = [n_samples, n_targets]
            Training targets. If Y was originally of shape
            [n_samples], it is now [n_samples, 1].

        E : {float, array}, shape = [k]
            Preconditioner for EigenPro

        Lambda : {float, array}, shape = [n_subsamples, k]
            Top eigenvectors of a subsample kernel matrix

        eta : float
            The learning rate

        pinx : {int, array}, shape = [sample_size]
            The rows of X used to calculate E and Lambda
        """
        n, d = X.shape
        n_label = 1 if len(Y.shape) == 1 else Y.shape[1]
        self.centers_ = X

        # Calculate the subsample size to be used.
        if self.subsample_size == "auto":
            if n < 100000:
                sample_size = 4000
            else:
                sample_size = 12000
        else:
            sample_size = self.subsample_size
        sample_size = min(n, sample_size)

        n_components = min(sample_size - 1, self.n_components)
        n_components = max(1, n_components)

        # Approximate amount of memory that we want to use
        mem_bytes = 0.1 * 1024 ** 3
        # Memory used with a certain sample size
        mem_usages = (d + n_label + 2 * np.arange(sample_size)) * n * 4
        mG = np.int32(np.sum(mem_usages < mem_bytes))

        # Calculate largest eigenvalue and max{k(x,x)} using subsamples.
        pinx = random_state.choice(n, sample_size, replace=False).astype(
            "int32"
        )
        if self.gamma == "scale":
            self.gamma_ = np.float32(1.0 / (X.var() * d))
        else:
            self.gamma_ = self.gamma
        max_S, beta, E, Lambda = self._setup(
            X[pinx], n_components, mG, alpha=0.95
        )
        # Calculate best batch size.
        if self.batch_size == "auto":
            bs = min(np.int32(beta / max_S), mG) + 1
        else:
            bs = self.batch_size
        self.bs_ = min(bs, n)

        # Calculate best step size.
        if self.bs_ < beta / max_S + 1:
            eta = self.bs_ / beta
        elif self.bs_ < n:
            eta = 2.0 * self.bs_ / (beta + (self.bs_ - 1) * max_S)
        else:
            eta = 0.95 * 2 / max_S
        # Remember the shape of Y for predict() and ensure it's shape is 2-D.
        self.was_1D_ = False
        if len(Y.shape) == 1:
            Y = np.reshape(Y, (Y.shape[0], 1))
            self.was_1D_ = True
        return Y, E, Lambda, np.float32(eta), pinx

    def validate_parameters(self):
        """
        Validate the parameters of the model to ensure that no unreasonable
        values were passed in.
        """
        if self.n_epoch <= 0:
            raise ValueError(
                "n_epoch should be positive, was " + str(self.n_epoch)
            )
        if self.n_components < 0:
            raise ValueError(
                "n_components should be non-negative, was "
                + str(self.n_components)
            )
        if self.subsample_size != "auto" and self.subsample_size < 0:
            raise ValueError(
                "subsample_size should be non-negative, was "
                + str(self.subsample_size)
            )
        if self.batch_size != "auto" and self.batch_size <= 0:
            raise ValueError(
                "batch_size should be positive, was " + str(self.batch_size)
            )
        if self.gamma != "scale" and self.gamma <= 0:
            raise ValueError(
                "gamma should be positive, was " + str(self.gamma)
            )

    def _raw_fit(self, X, Y):
        """Train eigenpro regression model

        Parameters
        ----------
        X : {float, array}, shape = [n_samples, n_features]
            Training data.

        Y : {float, array}, shape = [n_samples, n_targets]
            Training targets.

        Returns
        -------
        self : returns an instance of self.
        """
        X, Y = check_X_y(
            X,
            Y,
            dtype=np.float32,
            multi_output=True,
            ensure_min_samples=3,
            y_numeric=True,
        )
        Y = Y.astype(np.float32)
        random_state = check_random_state(self.random_state)

        self.validate_parameters()
        """Parameter Initialization"""
        Y, D, V, eta, pinx = self._initialize_params(X, Y, random_state)

        """Training loop"""
        n = self.centers_.shape[0]

        self.coef_ = np.zeros((n, Y.shape[1]), dtype=np.float32)
        step = np.float32(eta / self.bs_)
        for epoch in range(0, self.n_epoch):
            epoch_inds = random_state.choice(
                n, n // self.bs_ * self.bs_, replace=False
            ).astype("int32")

            for batch_inds in np.array_split(epoch_inds, n // self.bs_):
                batch_x = self.centers_[batch_inds]
                kfeat = self._kernel(batch_x, self.centers_)
                batch_y = Y[batch_inds]

                # Update 1: Sampled Coordinate Block.
                gradient = np.dot(kfeat, self.coef_) - batch_y

                self.coef_[batch_inds] -= step * gradient

                # Update 2: Fixed Coordinate Block
                delta = np.dot(
                    V * D, np.dot(V.T, np.dot(kfeat[:, pinx].T, gradient))
                )
                self.coef_[pinx] += step * delta
        return self

    def _raw_predict(self, X):
        """Predict using the kernel regression model

        Parameters
        ----------
        X : {float, array}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        Y : {float, array}, shape = [n_samples, n_targets]
            Predicted targets.
        """
        check_is_fitted(
            self, ["bs_", "centers_", "coef_", "was_1D_", "gamma_"]
        )
        X = np.asarray(X, dtype=np.float64)

        if len(X.shape) == 1:
            raise ValueError(
                "Reshape your data. X should be a matrix of shape"
                " (n_samples, n_features)."
            )
        n = X.shape[0]

        Ys = []
        for batch_inds in np.array_split(range(n), max(1, n // self.bs_)):
            batch_x = X[batch_inds]
            kfeat = self._kernel(batch_x, self.centers_)

            pred = np.dot(kfeat, self.coef_)
            Ys.append(pred)
        Y = np.vstack(Ys)
        if self.was_1D_:
            Y = np.reshape(Y, Y.shape[0])
        return Y

    def _get_tags(self):
        return {"multioutput": True}


class EigenProRegressor(BaseEigenPro, RegressorMixin):
    """Regression using EigenPro iteration.

    Train least squared kernel regression model with mini-batch EigenPro
    iteration.

    Parameters
    ----------
    batch_size : int, default = 'auto'
        Mini-batch size for gradient descent.

    n_epoch : int, default = 2
        The number of passes over the training data.

    n_components : int, default = 1000
        the maximum number of eigendirections used in modifying the kernel
        operator. Convergence rate speedup over normal gradient descent is
        approximately the largest eigenvalue over the n_componentth
        eigenvalue, however, it may take time to compute eigenvalues for
        large n_components

    subsample_size : int, default = 'auto'
        The number of subsamples used for estimating the largest
        n_component eigenvalues and eigenvectors. When it is set to 'auto',
        it will be 4000 if there are less than 100,000 samples
        (for training), and otherwise 12000.

    kernel : string or callable, default = "rbf"
        Kernel mapping used internally. Strings can be anything supported
        by scikit-learn, however, there is special support for the
        rbf, laplace, and cauchy kernels. If a callable is given, it should
        accept two arguments and return a floating point number.

    gamma : float, default='scale'
        Kernel coefficient. If 'scale', gamma = 1/(n_features*X.var()).
        Interpretation of the default value is left to the kernel;
        see the documentation for sklearn.metrics.pairwise.
        For kernels that use bandwidth, bandwidth = 1/sqrt(2*gamma).

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any
        Additional parameters (keyword arguments) for kernel function
        passed as callable object.

    random_state : int, RandomState instance or None, (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.  If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator
        is the RandomState instance used by `np.random`.

    References
    ----------
    * Siyuan Ma, Mikhail Belkin
      "Diving into the shallows: a computational perspective on
      large-scale machine learning", NIPS 2017.

    Examples
    --------
    >>> from sklearn_extra.kernel_methods import EigenProRegressor
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 4000, 20, 3
    >>> rng = np.random.RandomState(1)
    >>> x_train = rng.randn(n_samples, n_features)
    >>> y_train = rng.randn(n_samples, n_targets)
    >>> rgs = EigenProRegressor(n_epoch=3, gamma=.5, subsample_size=50)
    >>> rgs.fit(x_train, y_train)
    EigenProRegressor(batch_size='auto', coef0=1, degree=3, gamma=0.5, kernel='rbf',
                      kernel_params=None, n_components=1000, n_epoch=3,
                      random_state=None, subsample_size=50)
    >>> y_pred = rgs.predict(x_train)
    >>> loss = np.mean(np.square(y_train - y_pred))
    """

    def __init__(
        self,
        batch_size="auto",
        n_epoch=2,
        n_components=1000,
        subsample_size="auto",
        kernel="rbf",
        gamma="scale",
        degree=3,
        coef0=1,
        kernel_params=None,
        random_state=None,
    ):
        super().__init__(
            batch_size=batch_size,
            n_epoch=n_epoch,
            n_components=n_components,
            subsample_size=subsample_size,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            random_state=random_state,
        )

    def fit(self, X, Y):
        return self._raw_fit(X, Y)

    def predict(self, X):
        return self._raw_predict(X)


class EigenProClassifier(BaseEigenPro, ClassifierMixin):
    """Classification using EigenPro iteration.

    Train least squared kernel classification model with mini-batch EigenPro
    iteration.

    Parameters
    ----------
    batch_size : int, default = 'auto'
        Mini-batch size for gradient descent.

    n_epoch : int, default = 2
        The number of passes over the training data.

    n_components : int, default = 1000
        the maximum number of eigendirections used in modifying the
        kernel operator. Convergence rate speedup over normal gradient
        descent is approximately the largest eigenvalue over the
        n_componenth eigenvalue, however, it may take time to compute
        eigenvalues for large n_components

    subsample_size : int, default = 'auto'
        The size of subsamples used for estimating the largest
        n_component eigenvalues and eigenvectors. When it is set to
        'auto', it will be 4000 if there are less than 100,000 samples
        (for training), and otherwise 12000.

    kernel : string or callable, default = "rbf"
        Kernel mapping used internally. Strings can be anything supported
        by scikit-learn, however, there is special support for the
        rbf, laplace, and cauchy kernels. If a callable is given, it should
        accept two arguments and return a floating point number.

    gamma : float, default='scale'
        Kernel coefficient. If 'scale', gamma = 1/(n_features*X.var()).
        Interpretation of the default value is left to the kernel;
        see the documentation for sklearn.metrics.pairwise.
        For kernels that use bandwidth, bandwidth = 1/sqrt(2*gamma).

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels. Ignored by
        other kernels.

    kernel_params : mapping of string to any
        Additional parameters (keyword arguments) for kernel function
        passed as callable object.

    random_state : int, RandomState instance or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.  If int, random_state is the seed used by
        the random number generator; If RandomState instance,
        random_state is the random number generator;
        If None, the random number generator is the RandomState
        instance used by `np.random`.

    References
    ----------
    * Siyuan Ma, Mikhail Belkin
      "Diving into the shallows: a computational perspective on
      large-scale machine learning", NIPS 2017.

    Examples
    --------
    >>> from sklearn_extra.kernel_methods import EigenProClassifier
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 4000, 20, 3
    >>> rng = np.random.RandomState(1)
    >>> x_train = rng.randn(n_samples, n_features)
    >>> y_train = rng.randint(n_targets, size=n_samples)
    >>> rgs = EigenProClassifier(n_epoch=3, gamma=.01, subsample_size=50)
    >>> rgs.fit(x_train, y_train)
    EigenProClassifier(batch_size='auto', coef0=1, degree=3, gamma=0.01,
                       kernel='rbf', kernel_params=None, n_components=1000,
                       n_epoch=3, random_state=None, subsample_size=50)
    >>> y_pred = rgs.predict(x_train)
    >>> loss = np.mean(y_train != y_pred)
    """

    def __init__(
        self,
        batch_size="auto",
        n_epoch=2,
        n_components=1000,
        subsample_size="auto",
        kernel="rbf",
        gamma=0.02,
        degree=3,
        coef0=1,
        kernel_params=None,
        random_state=None,
    ):
        super().__init__(
            batch_size=batch_size,
            n_epoch=n_epoch,
            n_components=n_components,
            subsample_size=subsample_size,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            random_state=random_state,
        )

    def fit(self, X, Y):
        """ Train eigenpro classification model

        Parameters
        ----------
        X : {float, array}, shape = [n_samples, n_raw_feature]
            The raw input feature matrix.

        Y : {float, array}, shape =[n_samples]
            The labels corresponding to the features of X.

        Returns
        -------
        self : returns an instance of self.
       """
        X, Y = check_X_y(
            X,
            Y,
            dtype=np.float32,
            force_all_finite=True,
            multi_output=False,
            ensure_min_samples=3,
        )
        check_classification_targets(Y)
        self.classes_ = np.unique(Y)

        loc = {}
        for ind, label in enumerate(self.classes_):
            loc[label] = ind

        class_matrix = np.zeros((Y.shape[0], self.classes_.shape[0]))

        for ind, label in enumerate(Y):
            class_matrix[ind, loc[label]] = 1
        self._raw_fit(X, class_matrix)
        return self

    def predict(self, X):
        """Predict using the kernel classification model

        Parameters
        ----------
        X : {float, array}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y : {float, array}, shape = [n_samples]
            Predicted labels.
        """
        Y = self._raw_predict(X)
        return self.classes_[np.argmax(Y, axis=1)]
