"""RobustWeightedEstimator meta estimator."""

# Author: Timothee Mathieu
# License: BSD 3 clause

import numpy as np
import warnings
from scipy.stats import iqr

from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils import (
    check_random_state,
    check_array,
    check_consistent_length,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import SGDRegressor

# Loss functions import. Taken from scikit-learn linear SGD estimators.

try:
    from sklearn.linear_model._sgd_fast import (
        Log,
        SquaredLoss,
        Hinge,
        SquaredHinge,
    )
except ImportError:
    from sklearn.linear_model.sgd_fast import (
        Log,
        SquaredLoss,
        Hinge,
        SquaredHinge,
    )

# Tool library in which we get robust mean estimators.
from .mean_estimators import median_of_means_blocked, blockMOM, huber

LOSS_FUNCTIONS = {
    "hinge": (Hinge, 1.0),
    "squared_hinge": (SquaredHinge, 1.0),
    "log": (Log,),
    "squared_loss": (SquaredLoss,),
}


def _huber_psisx(x, c):
    """Huber-loss weight for RobustWeightedEstimator algorithm"""
    res = np.ones(len(x))
    res[x != 0] = (2 * (x[x != 0] > 0) - 1) * c / x[x != 0]
    res[np.abs(x) < c] = 1
    res[~np.isfinite(x)] = 0
    return res


def _mom_psisx(med_block, n):
    """MOM weight for RobustWeightedEstimator algorithm"""
    res = np.zeros(n)
    res[med_block] = 1
    return lambda x: res


class RobustWeightedEstimator(MetaEstimatorMixin, BaseEstimator):
    """Meta algorithm for robust regression and (Binary) classification.

    This model use iterative reweighting of samples to make a regression or
    classification estimator robust.

    The principle of the algorithm is to use an empirical risk minimization
    principle where the risk is estimated using a robust estimator (for example
    Huber estimator or median-of-means estimator)[1], [3]. The idea behind this
    algorithm was mentionned before in [2].
    This idea translates in an iterative algorithm where the sample_weight
    are changed at each iterations and are dependent of the sample. Informally
    the outliers should have small weight while the inliers should have big
    weight, where outliers are sample with a big loss function.

    This algorithm enjoy a non-zero breakdown-point (it can handle arbitrarily
    bad outliers). When the "mom" weighting scheme is used, k outliers can be
    tolerated. When the "Huber" weighting scheme is used, asymptotically the
    number of outliers has to be less than half the sample size.

    Read more in the :ref:`User Guide <RobustWeightedEstimator>`.

    Parameters
    ----------

    base_estimator : object or None, default=None
        The base estimator to fit. For now only SGDRegressor and SGDClassifier
        are supported.
        If None, then the base estimator is SGDRegressor with squared loss.

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
        is 'huber'. Measure the robustness of the weightint procedure. A small
        value of c means a more robust estimator.
        Can have a big effect on efficiency.
        If None, c is estimated at each step using half the Inter-quartile
        range.

    k : int < sample_size/2, default=1
        Parameter used for mom weighting procedure, used only if weightings
        is 'mom'. 2k+1 is the number of blocks used for median-of-means
        estimation, higher value of k means a more robust estimator.
        Can have a big effect on efficiency.

    loss : string or None, default=None
        Name of the loss used, must be the same loss as the one optimized in
        base_estimator.
        Classification losses supported : 'log', 'hinge'.
        If 'log', then the base_estimator must support predict_proba.
        Regression losses supported : 'squared_loss'.
        If None, loss='squared_loss'

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.



    Attributes
    ----------
    estimator : object,
        estimator trained using the RobustWeightedEstimator scheme, can be used
        as any sklearn estimator.

    weights : array like, length = n_sample.
        Weight of each sample at the end of the algorithm. Can be used as a
        measure of how much of an outlier a sample is.

    Notes
    -----
    For now only scikit-learn SGDRegressor and SGDClassifier are officially
    supported but one can use any estimator compatible with scikit-learn,
    as long as this estimator support partial_fit, warm_start and sample_weight
    . It must have the parameters max_iter and batch_size if the computation is
    done on batches. It must also support "constant" learning rate with
    learning rate called "eta0".

    For now, only binary classification is implemented. See sklearn.multiclass
    if you want to use this algorithm in multiclass classification.

    Examples
    --------

    >>> from sklearn_extra.robust import RobustWeightedEstimator
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.datasets import make_blobs
    >>> import numpy as np
    >>> X,y = make_blobs(n_samples=100, centers=np.array([[-1, -1], [1, 1]]))
    >>> clf = clf=RobustWeightedEstimator(base_estimator=SGDClassifier(),
    ...                                   weighting='mom', k=3, loss='hinge')
    >>> clf.fit(X, y)
    >>> print(mean(clf.predict(X)==y))
    0.9

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
        base_estimator=None,
        weighting="huber",
        max_iter=100,
        burn_in=10,
        eta0=0.01,
        c=None,
        k=1,
        loss=None,
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
        self.random_state = check_random_state(random_state)

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
        self : returns an estimator trained with RobustWeightedEstimator.
        """
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        check_consistent_length(X, y)

        classes = np.unique(y)

        self._validate_hyperparameters(len(X))

        # Initialization of all parameters in the base_estimator.
        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = SGDRegressor()

        learning_rate = base_estimator.learning_rate

        if self.loss is None:
            self.loss = "squared_loss"

        base_estimator.set_params(
            warm_start=True,
            learning_rate="constant",
            loss=self.loss,
            eta0=self.eta0,
        )

        # Get actual loss function from its name.
        loss = self._get_loss_function()

        # Weight initialization : do one non-robust epoch.
        if self.loss in ["log", "hinge", "squared_hinge"]:
            # If in a classification task, precise the classes.
            base_estimator.partial_fit(X, y, classes=classes)
        else:
            base_estimator.partial_fit(X, y)

        # Outlying Measure
        final_weight = np.zeros(len(X))

        # Optimization algorithm
        for epoch in range(self.max_iter):
            if base_estimator.loss == "log":
                # If log-loss use probabilties. Select only the probability
                # that it is 1.
                pred = base_estimator.predict_proba(X)[:, 1]
            elif base_estimator.loss == "hinge":
                # If in classification, not using log-loss use
                # decision_function
                pred = base_estimator.decision_function(X)
            else:
                pred = base_estimator.predict(X)

            # Compute the loss of each sample
            loss_values = loss(y, pred)

            # Compute the weight associated with each losses.
            # Samples whose loss is far from the mean loss (robust estimation)
            # will have a small weight.
            weights = self._get_weights(loss_values)

            # Use the optimization algorithm of self.base_estimator for one
            # epoch using the previously computed weights.
            base_estimator.partial_fit(X, y, sample_weight=weights)

            if epoch > self.burn_in:
                # If not in the burn_in phase anymore, change the learning_rate
                # calibration to the one edicted by self.base_estimator.
                base_estimator.set_params(learning_rate=learning_rate)

            # Shuffle the data at each step.

            perm = self.random_state.permutation(len(X))
            X = X[perm]
            y = y[perm]
            weights = weights[perm]

            if self.weighting == "mom":
                final_weight += weights

        if self.weighting == "mom":
            self.weights = final_weight / self.max_iter
        else:
            self.weights = weights
        self.estimator = base_estimator
        return self

    def _get_loss_function(self):
        """Get concrete ''LossFunction'' object for str ''loss''. """

        loss_ = LOSS_FUNCTIONS.get(self.loss)
        if loss_ is None:
            raise ValueError("The loss %s is not supported. " % self.loss)

        loss_class, args = loss_[0], loss_[1:]
        return np.vectorize(loss_class(*args).dloss)

    def _validate_hyperparameters(self, n):
        # Check the hyperparameters.

        if self.max_iter <= 0:
            raise ValueError(
                "RobustWeightedEstimator: "
                "max_iter must be > 0, got %s." % self.max_iter
            )

        if not (self.c is None) and (self.c <= 0):
            raise ValueError(
                "RobustWeightedEstimator: " "c must be > 0, got %s." % self.c
            )

        if self.burn_in < 0:
            raise ValueError(
                "RobustWeightedEstimator: "
                "burn_in must be >= 0, got %s." % self.c
            )

        if (self.burn_in > 0) and (self.eta0 <= 0):
            raise ValueError(
                "RobustWeightedEstimator: "
                "eta0 must be > 0, got %s." % self.eta0
            )

        if (
            not isinstance(self.k, int)
            or self.k < 0
            or self.k > np.floor(n / 2)
        ):
            raise ValueError(
                "RobustWeightedEstimator: "
                "k must be integer >= 0, and smaller than floor(sample_size/2)"
                " got %s." % self.k
            )

    def _get_weights(self, loss_values):
        # Compute the robust weight of the samples.
        if self.weighting == "huber":
            if self.c is None:
                # If no c parameter given, estimate using inter quartile range.
                self.c = iqr(np.abs(loss_values - np.median(loss_values))) / 2
                if self.c == 0:
                    warnings.warn(
                        "RobustWeightedEstimator: "
                        "too many sampled are parfectly predicted "
                        "according to the loss function. "
                        "Switching to constant c = 1.35. "
                        "Consider using another weighting scheme, "
                        "or using a constant c value to remove "
                        "this warning."
                    )
                    self.c = 1.35

            def psisx(x):
                return _huber_psisx(x, self.c)

            # Robust estimation of the risk is in mu.
            mu = huber(loss_values, self.c)

        elif self.weighting == "mom":
            # Choose (randomly) 2k+1 (almost-)equal blocks of data.
            blocks = blockMOM(loss_values, self.k, self.random_state)
            # Compute the median-of-means of the losses using these blocks.
            # Return also the index at which this median-of-means is attained.
            mu, idmom = median_of_means_blocked(loss_values, blocks)
            psisx = _mom_psisx(blocks[idmom], len(loss_values))
        else:
            raise ValueError(
                "RobustWeightedEstimator: " "no such weighting scheme"
            )
        # Compute the unnormalized weights.
        w = psisx(loss_values - mu)
        return w / np.sum(w) * len(loss_values)

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
        check_is_fitted(self.estimator)
        return self.estimator.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self.estimator)
        if self.loss != "log":
            raise ValueError(
                "RobustWeightedEstimator: "
                "predict_proba only supported if loss is log."
            )
        else:
            return self.estimator.predict_proba(X)
