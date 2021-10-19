"""RobustWeightedEstimator meta estimator."""

# Author: Timothee Mathieu
# License: BSD 3 clause

import numpy as np
import warnings
from scipy.stats import iqr


from sklearn.base import (
    BaseEstimator,
    clone,
    ClassifierMixin,
    RegressorMixin,
    ClusterMixin,
)
from sklearn.utils import (
    check_random_state,
    check_array,
    check_consistent_length,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.metaestimators import if_delegate_has_method

# Tool library in which we get robust mean estimators.
from .mean_estimators import median_of_means_blocked, block_mom, huber
from ._robust_weighted_estimator_helper import _kmeans_loss


# cython implementation of loss functions, copied from scikit-learn with light
# modifications.
from ._robust_weighted_estimator_helper import (
    _kmeans_loss,
    Log,
    SquaredLoss,
    Hinge,
    Huber,
    ModifiedHuber,
    SquaredHinge,
)


LOSS_FUNCTIONS = {
    "hinge": (Hinge,),
    "log": (Log,),
    "squared_error": (SquaredLoss,),
    "squared_loss": (SquaredLoss,),
    "squared_hinge": (SquaredHinge,),
    "modified_huber": (ModifiedHuber,),
    "huber": (Huber, 1.35),  # 1.35 is default value. TODO : set as parameter
}

# Test version of sklearn, in version older than v1.0 squared_loss must be used
import sklearn

if sklearn.__version__[0] == "0":
    SQ_LOSS = "squared_loss"
else:
    SQ_LOSS = "squared_error"


def _huber_psisx(x, c):
    """Huber-loss weight for RobustWeightedEstimator algorithm"""
    res = np.zeros(len(x))
    res[np.abs(x) <= c] = 1
    res[np.abs(x) > c] = c / np.abs(x)[np.abs(x) > c]
    res[~np.isfinite(x)] = 0
    return res


def _mom_psisx(med_block, n):
    """MOM weight for RobustWeightedEstimator algorithm"""
    res = np.zeros(n)
    res[med_block] = 1
    return res


class _RobustWeightedEstimator(BaseEstimator):
    """Meta algorithm for robust regression and binary classification.

    This model uses iterative reweighting of samples to make a regression or
    classification estimator robust.

    The principle of the algorithm is to use an empirical risk minimization
    principle where the risk is estimated using a robust estimator (for example
    Huber estimator or median-of-means estimator)[1], [3]. The idea behind this
    algorithm was mentioned before in [2].
    This idea translates in an iterative algorithm where the sample_weight
    are changed at each iterations and are dependent of the sample. Informally
    the outliers should have small weight while the inliers should have big
    weight, where outliers are sample with a big loss function.

    This algorithm enjoy a non-zero breakdown-point (it can handle arbitrarily
    bad outliers). When the "mom" weighting scheme is used, k outliers can be
    tolerated. When the "Huber" weighting scheme is used, asymptotically the
    number of outliers has to be less than half the sample size.

    Read more in the :ref:`User Guide <robust>`.

    Parameters
    ----------

    base_estimator : object, mandatory
        The base estimator to fit. For now only SGDRegressor and SGDClassifier
        are supported.
        If None, then the base estimator is SGDRegressor with squared loss.

    loss : string or callable, mandatory
        Name of the loss used, must be the same loss as the one optimized in
        base_estimator.
        Classification losses supported : 'log', 'hinge', 'squared_hinge',
        'modified_huber'. If 'log', then the base_estimator must support
        predict_proba. Regression losses supported : 'squared_error', 'huber'.
        If callable, the function is used as loss function ro construct
        the weights.

    weighting : string, default="huber"
        Weighting scheme used to make the estimator robust.
        Can be 'huber' for huber-type weights or  'mom' for median-of-means
        type weights.

    max_iter : int, default=100
        Maximum number of iterations.
        For more information, see the optimization scheme of base_estimator
        and the eta0 and burn_in parameter.

    burn_in : int, default=10
        Number of steps used without changing the learning rate.
        Can be useful to make the weight estimation better at the beginning.

    eta0 : float, default=0.01
        Constant step-size used during the burn_in period. Used only if
        burn_in>0. Can have a big effect on efficiency.

    c : float>0 or None, default=None
        Parameter used for Huber weighting procedure, used only if weightings
        is 'huber'. Measure the robustness of the weighting procedure. A small
        value of c means a more robust estimator.
        Can have a big effect on efficiency.
        If None, c is estimated at each step using half the Inter-quartile
        range, this tends to be conservative (robust).

    k : int < sample_size/2, default=1
        Parameter used for mom weighting procedure, used only if weightings
        is 'mom'. 2k+1 is the number of blocks used for median-of-means
        estimation, higher value of k means a more robust estimator.
        Can have a big effect on efficiency.
        If None, k is estimated using the number of points distant from the
        median of means of more than 2 times a robust estimate of the scale
        (using the inter-quartile range), this tends to be conservative
        (robust).

    tol : float or None, (default = 1e-3)
        The stopping criterion. If it is not None, training will stop when
        (loss > best_loss - tol) for n_iter_no_change consecutive epochs.

    n_iter_no_change : int, default=10
        Number of iterations with no improvement to wait before early stopping.

    verbose: int, default=0
        If >0 will display the (robust) estimated loss every 10 epochs.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.



    Attributes
    ----------
    base_estimator_ : object,
        The fitted base_estimator.

    weights_ : array like, length = n_sample.
        Weight of each sample at the end of the algorithm. Can be used as a
        measure of how much of an outlier a sample is.

    Notes
    -----
    For now only scikit-learn SGDRegressor and SGDClassifier are officially
    supported but one can use any estimator compatible with scikit-learn,
    as long as this estimator support partial_fit, warm_start and sample_weight
    . It must have the parameters max_iter and support "constant" learning rate
    with learning rate called "eta0".

    For now, only binary classification is implemented. See sklearn.multiclass
    if you want to use this algorithm in multiclass classification.


    References
    ----------

    [1] Guillaume Lecué, Matthieu Lerasle and Timothée Mathieu.
        "Robust classification via MOM minimization", arXiv preprint (2018).
        arXiv:1808.03106

    [2] Christian Brownlees, Emilien Joly and Gábor Lugosi.
        "Empirical risk minimization for heavy-tailed losses", Ann. Statist.
        Volume 43, Number 6 (2015), 2507-2536.

    [3] Stanislav Minsker and Timothée Mathieu.
        "Excess risk bounds in robust empirical risk minimization"
        arXiv preprint (2019). arXiv:1910.07485.



    """

    def __init__(
        self,
        base_estimator,
        loss,
        weighting="huber",
        max_iter=100,
        burn_in=10,
        eta0=0.1,
        c=None,
        k=0,
        tol=1e-5,
        n_iter_no_change=10,
        verbose=0,
        random_state=None,
    ):
        self.base_estimator = base_estimator
        self.weighting = weighting
        self.eta0 = eta0
        self.burn_in = burn_in
        self.c = c
        self.k = k
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns an estimator trained with RobustWeightedEstimator.
        """
        X = check_array(X)
        if y is not None:
            y = check_array(y, ensure_2d=False)
            check_consistent_length(X, y)

        random_state = check_random_state(self.random_state)

        self._validate_hyperparameters(len(X))

        # Initialization of all parameters in the base_estimator.
        base_estimator = clone(self.base_estimator)
        loss_param = self.loss

        # Get actual loss function from its name.
        loss = self._get_loss_function(loss_param)

        parameters = list(base_estimator.get_params().keys())
        if "warm_start" in parameters:
            base_estimator.set_params(warm_start=True)

        if ("loss" in parameters) and (loss_param != "squared_error"):
            base_estimator.set_params(loss=loss_param)

        if "eta0" in parameters:
            base_estimator.set_params(eta0=self.eta0)

        if "n_iter_no_change" in parameters:
            base_estimator.set_params(n_iter_no_change=self.n_iter_no_change)

        base_estimator.set_params(random_state=random_state)
        if self.burn_in > 0:
            learning_rate = base_estimator.learning_rate
            base_estimator.set_params(learning_rate="constant", eta0=self.eta0)

        # Initialization
        if self._estimator_type == "classifier":
            classes = np.unique(y)
            if len(classes) > 2:
                raise ValueError("y must be binary.")
            # Initialization of the estimator.
            # Partial fit for the estimator to be set to "fitted" to be able
            # to predict.
            base_estimator.partial_fit(X, y, classes=classes)
            # As the partial fit is here non-robust, override the
            # learned coefs.
            base_estimator.coef_ = np.zeros([1, len(X[0])])
            base_estimator.intercept_ = np.array([0])
            self.classes_ = base_estimator.classes_
        elif self._estimator_type == "regressor":
            # Initialization of the estimator
            # Partial fit for the estimator to be set to "fitted" to be able
            # to predict.
            base_estimator.partial_fit(X, y)
            # As the partial fit is here non-robust, override the
            # learned coefs.
            base_estimator.coef_ = np.zeros([len(X[0])])
            base_estimator.intercept_ = np.array([0])
        elif self._estimator_type == "clusterer":
            # Partial fit for the estimator to be set to "fitted" to be able
            # to predict.
            base_estimator.partial_fit(X, y)

        # Initialization of final weights
        final_weights = np.zeros(len(X))
        best_loss = np.inf
        n_iter_no_change_ = 0

        # Optimization algorithm
        for epoch in range(self.max_iter):

            if epoch > self.burn_in and self.burn_in > 0:
                # If not in the burn_in phase anymore, change the learning_rate
                # calibration to the one edicted by self.base_estimator.
                base_estimator.set_params(learning_rate=learning_rate)

            if self._estimator_type == "classifier":
                # If in classification, use decision_function
                pred = base_estimator.decision_function(X)
            else:
                pred = base_estimator.predict(X)

            # Compute the loss of each sample
            if self._estimator_type == "clusterer":
                loss_values = loss(X, pred)
            elif self._estimator_type == "classifier":
                # For classifiers, set the labels to {-1,1} for compatibility.
                loss_values = loss(2 * y.flatten() - 1, pred)
            else:
                loss_values = loss(y.flatten(), pred)
            # Compute the weight associated with each losses.
            # Samples whose loss is far from the mean loss (robust estimation)
            # will have a small weight.
            weights, current_loss = self._get_weights(
                loss_values, random_state
            )

            if (self.verbose > 0) and (epoch % 10 == 0):
                print("Epoch ", epoch, " loss: %.2F" % (current_loss))
            # Use the optimization algorithm of self.base_estimator for one
            # epoch using the previously computed weights. Also shuffle the data.
            perm = random_state.permutation(len(X))

            base_estimator.partial_fit(X, y, sample_weight=weights)

            if (self.tol is not None) and (
                current_loss > best_loss - self.tol
            ):
                n_iter_no_change_ += 1
            else:
                n_iter_no_change_ = 0

            if current_loss < best_loss:
                best_loss = current_loss

            if n_iter_no_change_ == self.n_iter_no_change:
                break

            # Shuffle the data at each step.
            if self._estimator_type == "clusterer":
                # Here y is None
                base_estimator.partial_fit(
                    X[perm], y, sample_weight=weights[perm]
                )
            else:
                base_estimator.partial_fit(
                    X[perm], y[perm], sample_weight=weights[perm]
                )
            if (self.tol is not None) and (
                current_loss > best_loss - self.tol
            ):
                n_iter_no_change_ += 1
            else:
                n_iter_no_change_ = 0

            if current_loss < best_loss:
                best_loss = current_loss

            if n_iter_no_change_ == self.n_iter_no_change:
                break
            elif epoch == self.max_iter - 1:
                warnings.warn(
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit.",
                    ConvergenceWarning,
                )

            if self.weighting == "mom":
                final_weights += weights

        if self.weighting == "mom":
            self.weights_ = final_weights / self.max_iter
        else:
            self.weights_ = weights
        self.base_estimator_ = base_estimator
        self.n_iter_ = self.max_iter * len(X)

        if hasattr(base_estimator, "coef_"):
            self.coef_ = base_estimator.coef_
            self.intercept_ = base_estimator.intercept_
        if hasattr(base_estimator, "labels_"):
            self.labels_ = self.base_estimator_.labels_
        if hasattr(base_estimator, "cluster_centers_"):
            self.cluster_centers_ = self.base_estimator_.cluster_centers_
            self.inertia_ = self.base_estimator_.inertia_
        return self

    def _get_loss_function(self, loss):
        """Get concrete ''LossFunction'' object for str ''loss''."""
        if type(loss) == str:
            eff_loss = LOSS_FUNCTIONS.get(loss)
            if eff_loss is None:
                raise ValueError("The loss %s is not supported. " % self.loss)

            loss_class, args = eff_loss[0], eff_loss[1:]

            return np.vectorize(getattr(loss_class(*args), "py_loss"))
        else:
            return loss

    def _validate_hyperparameters(self, n):
        # Check the hyperparameters.

        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0, got %s." % self.max_iter)

        if not (self.c is None) and (self.c <= 0):
            raise ValueError("c must be > 0, got %s." % self.c)

        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0, got %s." % self.burn_in)

        if (self.burn_in > 0) and (self.eta0 <= 0):
            raise ValueError("eta0 must be > 0, got %s." % self.eta0)

        if not (self.k is None) and (
            not isinstance(self.k, int)
            or self.k < 0
            or self.k > np.floor(n / 2)
        ):
            raise ValueError(
                "k must be integer >= 0, and smaller than floor(sample_size/2)"
                " got %s." % self.k
            )

    def _get_weights(self, loss_values, random_state):
        # Compute the robust weight of the samples.
        if self.weighting == "huber":
            if self.c is None:
                # If no c parameter given, estimate using inter quartile range.
                c = iqr(loss_values) / 2
                if c == 0:
                    warnings.warn(
                        "Too many samples are parfectly predicted "
                        "according to the loss function. "
                        "Switching to constant c = 1.35. "
                        "Consider using another weighting scheme, "
                        "or using a constant c value to remove "
                        "this warning."
                    )
                    c = 1.35
            else:
                c = self.c

            def psisx(x):
                return _huber_psisx(x, c)

            # Robust estimation of the risk is in mu.
            mu = huber(loss_values, c)

        elif self.weighting == "mom":
            if self.k is None:
                med = np.median(loss_values)
                # scale estimator using iqr, rescaled by what would be if the
                # loss was Gaussian.
                scale = iqr(np.abs(loss_values - med)) / 1.37
                k = np.sum(np.abs(loss_values - med) > 2 * scale)
            else:
                k = self.k
            # Choose (randomly) 2k+1 (almost-)equal blocks of data.
            blocks = block_mom(loss_values, k, random_state)
            # Compute the median-of-means of the losses using these blocks.
            # Return also the index at which this median-of-means is attained.
            mu, idmom = median_of_means_blocked(loss_values, blocks)

            def psisx(x):
                return _mom_psisx(blocks[idmom], len(loss_values))

        else:
            raise ValueError("No such weighting scheme")
        # Compute the unnormalized weights.
        w = psisx(loss_values - mu)
        if self._estimator_type == "regressor":
            return w / np.sum(w) * len(w), mu
        else:
            return w / np.sum(w), mu

    def predict(self, X):
        """Predict using the estimator trained with RobustWeightedEstimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self, attributes=["base_estimator_"])
        return self.base_estimator_.predict(X)

    def _check_proba(self):
        if self.loss != "log":
            raise AttributeError(
                "Probability estimates are not available for"
                " loss=%r" % self.loss
            )

    @property
    def predict_proba(self):
        check_is_fitted(self, attributes=["base_estimator_"])
        self._check_proba()
        return self._predict_proba

    def _predict_proba(self, X):
        return self.base_estimator_.predict_proba(X)

    @property
    def _estimator_type(self):
        return self.base_estimator._estimator_type

    def score(self, X, y=None):
        """Returns the score on the given data, using
        ``base_estimator_.score``.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) or (n_samples,), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        Returns
        -------
        score : float
        """
        check_is_fitted(self, attributes=["base_estimator_"])
        return self.base_estimator_.score(X, y)

    @if_delegate_has_method(delegate="base_estimator")
    def decision_function(self, X):
        """Predict using the linear model. For classifiers only.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples,)
           Predicted target values per element in X.
        """
        check_is_fitted(self, attributes=["base_estimator_"])
        return self.base_estimator_.decision_function(X)


class RobustWeightedClassifier(BaseEstimator, ClassifierMixin):
    """Algorithm for robust classification using reweighting algorithm.

    This model uses iterative reweighting of samples to make a regression or
    classification estimator robust.

    The principle of the algorithm is to use an empirical risk minimization
    principle where the risk is estimated using a robust estimator (for example
    Huber estimator or median-of-means estimator)[1], [3]. The idea behind this
    algorithm was mentioned before in [2].
    This idea translates in an iterative algorithm where the sample_weight
    are changed at each iterations and are dependent of the sample. Informally
    the outliers should have small weight while the inliers should have big
    weight, where outliers are sample with a big loss function.

    This algorithm enjoy a non-zero breakdown-point (it can handle arbitrarily
    bad outliers). When the "mom" weighting scheme is used, k outliers can be
    tolerated. When the "Huber" weighting scheme is used, asymptotically the
    number of outliers has to be less than half the sample size.

    Read more in the :ref:`User Guide <robust>`.

    Parameters
    ----------

    weighting : string, default="huber"
        Weighting scheme used to make the estimator robust.
        Can be 'huber' for huber-type weights or  'mom' for median-of-means
        type weights.

    max_iter : int, default=100
        Maximum number of iterations.
        For more information, see the optimization scheme of base_estimator
        and the eta0 and burn_in parameter.

    burn_in : int, default=10
        Number of steps used without changing the learning rate.
        Can be useful to make the weight estimation better at the beginning.

    eta0 : float, default=0.01
        Constant step-size used during the burn_in period. Used only if
        burn_in>0. Can have a big effect on efficiency.

    c : float>0 or None, default=None
        Parameter used for Huber weighting procedure, used only if weightings
        is 'huber'. Measure the robustness of the weighting procedure. A small
        value of c means a more robust estimator.
        Can have a big effect on efficiency.
        If None, c is estimated at each step using half the Inter-quartile
        range, this tends to be conservative (robust).

    k : int < sample_size/2, default=1
        Parameter used for mom weighting procedure, used only if weightings
        is 'mom'. 2k+1 is the number of blocks used for median-of-means
        estimation, higher value of k means a more robust estimator.
        Can have a big effect on efficiency.
        If None, k is estimated using the number of points distant from the
        median of means of more than 2 times a robust estimate of the scale
        (using the inter-quartile range), this tends to be conservative
        (robust).

    loss : string, None or callable, default="log"
        Classification losses supported : 'log', 'hinge', 'modified_huber'.
        If 'log', then the base_estimator must support predict_proba.

    sgd_args : dict, default={}
        arguments of the SGDClassifier base estimator.

    multi_class : string, default="ovr"
        multi-class scheme. Can be either "ovo" for OneVsOneClassifier or "ovr"
        for OneVsRestClassifier or "binary" for binary classification.

    n_jobs : int, default=1
        number of jobs used in the multi-class meta-algorithm computation.

    tol : float or None, (default = 1e-3)
        The stopping criterion. If it is not None, training will stop when
        (loss > best_loss - tol) for n_iter_no_change consecutive epochs.

    n_iter_no_change : int, default=10

        Number of iterations with no improvement to wait before early stopping.

    verbose: int, default=0
        If >0 will display the (robust) estimated loss every 10 epochs.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.

    Attributes
    ----------

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function. Only available if
        multi_class = "binary"

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        Only available if multi_class = "binary"

    n_iter_ : ndarray of shape (n_classes,) or (1, )
        Actual number of iterations for all classes. If binary or multinomial,
        it returns only 1 element. For liblinear solver, only the maximum
        number of iteration across all classes is given.

    base_estimator_ : object,
        The fitted base estimator SGDCLassifier.

    weights_ : array like, length = n_sample.
        Weight of each sample at the end of the algorithm. Can be used as a
        measure of how much of an outlier a sample is. Only available if
        multi_class = "binary"


    Notes
    -----

    Often, there is a need to use RobustScaler as preprocessing.

    Examples
    --------

    >>> from sklearn_extra.robust import RobustWeightedClassifier
    >>> from sklearn.datasets import make_blobs
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> X,y = make_blobs(n_samples=100, centers=np.array([[-1, -1], [1, 1]]),
    ...                  random_state=rng)
    >>> clf=RobustWeightedClassifier()
    >>> _ = clf.fit(X, y)
    >>> score = np.mean(clf.predict(X)==y)

    References
    ----------

    [1] Guillaume Lecué, Matthieu Lerasle and Timothée Mathieu.
        "Robust classification via MOM minimization", Mach Learn 109, (2020).
        https://doi.org/10.1007/s10994-019-05863-6 (2018).
        arXiv:1808.03106

    [2] Christian Brownlees, Emilien Joly and Gábor Lugosi.
        "Empirical risk minimization for heavy-tailed losses", Ann. Statist.
        Volume 43, Number 6 (2015), 2507-2536.

    [3] Stanislav Minsker and Timothée Mathieu.
        "Excess risk bounds in robust empirical risk minimization"
        arXiv preprint (2019). arXiv:1910.07485.

    """

    def __init__(
        self,
        weighting="huber",
        max_iter=100,
        burn_in=10,
        eta0=0.01,
        c=None,
        k=0,
        loss="log",
        sgd_args=None,
        multi_class="ovr",
        n_jobs=1,
        tol=1e-3,
        n_iter_no_change=10,
        verbose=0,
        random_state=None,
    ):
        self.weighting = weighting
        self.max_iter = max_iter
        self.burn_in = burn_in
        self.eta0 = eta0
        self.c = c
        self.k = k
        self.loss = loss
        self.sgd_args = sgd_args
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns an estimator trained with RobustWeightedClassifier.
        """

        if self.sgd_args is None:
            sgd_args = {}
        else:
            sgd_args = self.sgd_args

        # Define the base estimator
        X, y = self._validate_data(X, y, y_numeric=False)

        base_robust_estimator_ = _RobustWeightedEstimator(
            SGDClassifier(**sgd_args, eta0=self.eta0),
            weighting=self.weighting,
            loss=self.loss,
            burn_in=self.burn_in,
            c=self.c,
            k=self.k,
            eta0=self.eta0,
            max_iter=self.max_iter,
            tol=self.tol,
            n_iter_no_change=self.n_iter_no_change,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        if self.multi_class == "ovr":
            self.base_estimator_ = OneVsRestClassifier(
                base_robust_estimator_, n_jobs=self.n_jobs
            )
        elif self.multi_class == "binary":
            self.base_estimator_ = base_robust_estimator_
        elif self.multi_class == "ovo":
            self.base_estimator_ = OneVsOneClassifier(
                base_robust_estimator_, n_jobs=self.n_jobs
            )
        else:
            raise ValueError("No such multiclass method implemented.")

        self.base_estimator_.fit(X, y)
        if self.multi_class == "binary":
            self.weights_ = self.base_estimator_.weights_
            self.coef_ = self.base_estimator_.coef_
            self.intercept_ = self.base_estimator_.intercept_
        self.n_iter_ = self.max_iter * len(X)
        self.classes_ = self.base_estimator_.classes_
        return self

    def predict(self, X):
        """Predict using the estimator trained with RobustWeightedClassifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples, n_outputs)
            The predicted values.
        """

        check_is_fitted(self, attributes=["base_estimator_"])
        return self.base_estimator_.predict(X)

    def _check_proba(self):
        if self.loss != "log":
            raise AttributeError(
                "Probability estimates are not available for"
                " loss=%r" % self.loss
            )

    @property
    def predict_proba(self):
        """
        Probability estimates when binary classification.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
        """
        check_is_fitted(self, attributes=["base_estimator_"])
        self._check_proba()
        return self._predict_proba

    def _predict_proba(self, X):
        return self.base_estimator_.predict_proba(X)

    def score(self, X, y=None):
        """Returns the score on the given data, using
        ``base_estimator_.score``.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) or (n_samples,), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        Returns
        -------
        score : float
        """
        check_is_fitted(self, attributes=["base_estimator_"])
        return self.base_estimator_.score(X, y)

    def decision_function(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples,)
           Predicted target values per element in X.
        """
        check_is_fitted(self, attributes=["base_estimator_"])
        return self.base_estimator_.decision_function(X)


class RobustWeightedRegressor(BaseEstimator, RegressorMixin):
    """Algorithm for robust regression using reweighting algorithm.

    This model uses iterative reweighting of samples to make a regression or
    classification estimator robust.

    The principle of the algorithm is to use an empirical risk minimization
    principle where the risk is estimated using a robust estimator (for example
    Huber estimator or median-of-means estimator)[1], [3]. The idea behind this
    algorithm was mentioned before in [2].
    This idea translates in an iterative algorithm where the sample_weight
    are changed at each iterations and are dependent of the sample. Informally
    the outliers should have small weight while the inliers should have big
    weight, where outliers are sample with a big loss function.

    This algorithm enjoy a non-zero breakdown-point (it can handle arbitrarily
    bad outliers). When the "mom" weighting scheme is used, k outliers can be
    tolerated. When the "Huber" weighting scheme is used, asymptotically the
    number of outliers has to be less than half the sample size.

    Read more in the :ref:`User Guide <robust>`.

    Parameters
    ----------

    weighting : string, default="huber"
        Weighting scheme used to make the estimator robust.
        Can be 'huber' for huber-type weights or  'mom' for median-of-means
        type weights.

    max_iter : int, default=100
        Maximum number of iterations.
        For more information, see the optimization scheme of base_estimator
        and the eta0 and burn_in parameter.

    burn_in : int, default=10
        Number of steps used without changing the learning rate.
        Can be useful to make the weight estimation better at the beginning.

    eta0 : float, default=0.01
        Constant step-size used during the burn_in period. Used only if
        burn_in>0. Can have a big effect on efficiency.

    c : float>0 or None, default=None
        Parameter used for Huber weighting procedure, used only if weightings
        is 'huber'. Measure the robustness of the weighting procedure. A small
        value of c means a more robust estimator.
        Can have a big effect on efficiency.
        If None, c is estimated at each step using half the Inter-quartile
        range, this tends to be conservative (robust).

    k : int < sample_size/2, default=1
        Parameter used for mom weighting procedure, used only if weightings
        is 'mom'. 2k+1 is the number of blocks used for median-of-means
        estimation, higher value of k means a more robust estimator.
        Can have a big effect on efficiency.
        If None, k is estimated using the number of points distant from the
        median of means of more than 2 times a robust estimate of the scale
        (using the inter-quartile range), this tends to be conservative
        (robust).

    loss : string, None or callable, default="squared_error"
        For now, only "squared_error" and "huber" are implemented.

    sgd_args : dict, default={}
        arguments of the SGDClassifier base estimator.
    tol : float or None, (default = 1e-3)
        The stopping criterion. If it is not None, training will stop when
        (loss > best_loss - tol) for n_iter_no_change consecutive epochs.

    n_iter_no_change : int, default=10
        Number of iterations with no improvement to wait before early stopping.

    verbose: int, default=0
        If >0 will display the (robust) estimated loss every 10 epochs.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.


    Attributes
    ----------


    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features.

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias).

    n_iter_ : ndarray of shape (n_classes,) or (1, )
        Actual number of iterations.

    base_estimator_ : object,
        The fitted base_estimator.

    weights_ : array like, length = n_sample.
        Weight of each sample at the end of the algorithm. Can be used as a
        measure of how much of an outlier a sample is.

    Notes
    -----

    Often, there is a need to use RobustScaler as preprocessing.

    Examples
    --------

    >>> from sklearn_extra.robust import RobustWeightedRegressor
    >>> from sklearn.datasets import make_regression
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> X, y = make_regression()
    >>> reg = RobustWeightedRegressor()
    >>> _ = reg.fit(X, y)
    >>> score = np.mean(reg.predict(X)==y)

    References
    ----------

    [1] Guillaume Lecué, Matthieu Lerasle and Timothée Mathieu.
        "Robust classification via MOM minimization", Mach Learn 109, (2020).
        https://doi.org/10.1007/s10994-019-05863-6 (2018).
        arXiv:1808.03106

    [2] Christian Brownlees, Emilien Joly and Gábor Lugosi.
        "Empirical risk minimization for heavy-tailed losses", Ann. Statist.
        Volume 43, Number 6 (2015), 2507-2536.

    [3] Stanislav Minsker and Timothée Mathieu.
        "Excess risk bounds in robust empirical risk minimization"
        arXiv preprint (2019). arXiv:1910.07485.

    """

    def __init__(
        self,
        weighting="huber",
        max_iter=100,
        burn_in=10,
        eta0=0.01,
        c=None,
        k=0,
        loss=SQ_LOSS,
        sgd_args=None,
        tol=1e-3,
        n_iter_no_change=10,
        verbose=0,
        random_state=None,
    ):

        self.weighting = weighting
        self.max_iter = max_iter
        self.burn_in = burn_in
        self.eta0 = eta0
        self.c = c
        self.k = k
        self.loss = loss
        self.sgd_args = sgd_args
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns an estimator trained with RobustWeightedClassifier.
        """
        if self.sgd_args is None:
            sgd_args = {}
        else:
            sgd_args = self.sgd_args

        # Define the base estimator

        X, y = self._validate_data(X, y, y_numeric=True)

        self.base_estimator_ = _RobustWeightedEstimator(
            SGDRegressor(**sgd_args, eta0=self.eta0),
            weighting=self.weighting,
            loss=self.loss,
            burn_in=self.burn_in,
            c=self.c,
            k=self.k,
            eta0=self.eta0,
            max_iter=self.max_iter,
            tol=self.tol,
            n_iter_no_change=self.n_iter_no_change,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        self.base_estimator_.fit(X, y)

        self.weights_ = self.base_estimator_.weights_
        self.n_iter_ = self.max_iter * len(X)
        self.coef_ = self.base_estimator_.coef_
        self.intercept_ = self.base_estimator_.intercept_
        return self

    def predict(self, X):
        """Predict using the estimator trained with RobustWeightedRegressor.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples, n_outputs)
            The predicted values.
        """

        check_is_fitted(self, attributes=["base_estimator_"])
        return self.base_estimator_.predict(X)

    def score(self, X, y=None):
        """Returns the score on the given data, using
        ``base_estimator_.score``.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) or (n_samples,), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        Returns
        -------
        score : float
        """
        check_is_fitted(self, attributes=["base_estimator_"])
        return self.base_estimator_.score(X, y)


class RobustWeightedKMeans(BaseEstimator, ClusterMixin):
    """Algorithm for robust kmeans clustering using reweighting algorithm.

    This model uses iterative reweighting of samples to make a regression or
    classification estimator robust.

    The principle of the algorithm is to use an empirical risk minimization
    principle where the risk is estimated using a robust estimator (for example
    Huber estimator or median-of-means estimator)[1], [3]. The idea behind this
    algorithm was mentioned before in [2].
    This idea translates in an iterative algorithm where the sample_weight
    are changed at each iterations and are dependent of the sample. Informally
    the outliers should have small weight while the inliers should have big
    weight, where outliers are sample with a big loss function.

    This algorithm enjoy a non-zero breakdown-point (it can handle arbitrarily
    bad outliers). When the "mom" weighting scheme is used, k outliers can be
    tolerated. When the "Huber" weighting scheme is used, asymptotically the
    number of outliers has to be less than half the sample size.

    Read more in the :ref:`User Guide <robust>`.

    Parameters
    ----------

    weighting : string, default="huber"
        Weighting scheme used to make the estimator robust.
        Can be 'huber' for huber-type weights or  'mom' for median-of-means
        type weights.

    max_iter : int, default=100
        Maximum number of iterations.
        For more information, see the optimization scheme of base_estimator
        and the eta0 and burn_in parameter.

    eta0 : float, default=0.01
        Constant step-size used during the burn_in period. Used only if
        burn_in>0. Can have a big effect on efficiency.

    c : float>0 or None, default=None
        Parameter used for Huber weighting procedure, used only if weightings
        is 'huber'. Measure the robustness of the weighting procedure. A small
        value of c means a more robust estimator.
        Can have a big effect on efficiency.
        If None, c is estimated at each step using half the Inter-quartile
        range, this tends to be conservative (robust).

    k : int < sample_size/2, default=1
        Parameter used for mom weighting procedure, used only if weightings
        is 'mom'. 2k+1 is the number of blocks used for median-of-means
        estimation, higher value of k means a more robust estimator.
        Can have a big effect on efficiency.
        If None, k is estimated using the number of points distant from the
        median of means of more than 2 times a robust estimate of the scale
        (using the inter-quartile range), this tends to be conservative
        (robust).

    kmeans_args : dict, default={}
        arguments of the MiniBatchKMeans base estimator. Must not contain
        batch_size.

    tol : float or None, (default = 1e-3)
        The stopping criterion. If it is not None, training will stop when
        (loss > best_loss - tol) for n_iter_no_change consecutive epochs.

    n_iter_no_change : int, default=10
        Number of iterations with no improvement to wait before early stopping.

    verbose: int, default=0
        If >0 will display the (robust) estimated loss every 10 epochs.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.


    Attributes
    ----------

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.

    base_estimator_ : object,
        The fitted base_estimator.

    weights_ : array like, length = n_sample.
        Weight of each sample at the end of the algorithm. Can be used as a
        measure of how much of an outlier a sample is.

    Notes
    -----

    One may need to use RobustScaler as a preprocessing.

    Examples
    --------

    >>> from sklearn_extra.robust import RobustWeightedKMeans
    >>> from sklearn.datasets import make_blobs
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> X,y = make_blobs(n_samples=100, centers=np.array([[-1, -1], [1, 1]]),
    ...                  random_state=rng)
    >>> km = RobustWeightedKMeans()
    >>> _ = km.fit(X)
    >>> score = np.mean((km.predict(X)-y)**2)

    References
    ----------

    [1] Guillaume Lecué, Matthieu Lerasle and Timothée Mathieu.
        "Robust classification via MOM minimization", Mach Learn 109, (2020).
        https://doi.org/10.1007/s10994-019-05863-6 (2018).
        arXiv:1808.03106

    [2] Christian Brownlees, Emilien Joly and Gábor Lugosi.
        "Empirical risk minimization for heavy-tailed losses", Ann. Statist.
        Volume 43, Number 6 (2015), 2507-2536.

    [3] Stanislav Minsker and Timothée Mathieu.
        "Excess risk bounds in robust empirical risk minimization"
        arXiv preprint (2019). arXiv:1910.07485.

    """

    def __init__(
        self,
        n_clusters=8,
        weighting="huber",
        max_iter=100,
        eta0=0.01,
        c=None,
        k=0,
        kmeans_args=None,
        tol=1e-3,
        n_iter_no_change=10,
        verbose=0,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.weighting = weighting
        self.max_iter = max_iter
        self.eta0 = eta0
        self.c = c
        self.k = k
        self.kmeans_args = kmeans_args
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        if self.kmeans_args is None:
            kmeans_args = {}
        else:
            kmeans_args = self.kmeans_args
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        self.base_estimator_ = _RobustWeightedEstimator(
            MiniBatchKMeans(
                self.n_clusters,
                batch_size=X.shape[0],
                random_state=self.random_state,
                **kmeans_args
            ),
            burn_in=0,  # Important because it does not mean anything to
            # have burn-in
            # steps for kmeans. It must be 0.
            weighting=self.weighting,
            loss=_kmeans_loss,
            max_iter=self.max_iter,
            eta0=self.eta0,
            c=self.c,
            k=self.k,
            tol=self.tol,
            n_iter_no_change=self.n_iter_no_change,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        self.base_estimator_.fit(X)
        self.cluster_centers_ = self.base_estimator_.cluster_centers_
        self.n_iter_ = self.max_iter * len(X)
        self.labels_ = self.predict(X)
        self.inertia_ = self.base_estimator_.inertia_
        self.weights_ = self.base_estimator_.weights_
        return self

    def predict(self, X):
        """Predict using the estimator trained with RobustWeightedClassifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples, n_outputs)
            The predicted values.
        """

        check_is_fitted(self, attributes=["base_estimator_"])
        return self.base_estimator_.predict(X)

    def score(self, X, y=None):
        """Returns the score on the given data, using
        ``base_estimator_.score``.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) or (n_samples,), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        Returns
        -------
        score : float
        """
        check_is_fitted(self, attributes=["base_estimator_"])
        return self.base_estimator_.score(X, y)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return euclidean_distances(X, self.cluster_centers_)

    def fit_transform(self, X, y=None):
        return self.fit(X)._transform(X)
