"""RobustWeightedEstimator meta estimator."""

# Author: Timothee Mathieu
# License: BSD 3 clause

import numpy as np
import warnings
from scipy.stats import iqr

from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.linear_model import SGDRegressor

# Loss functions import. Taken from scikit-learn linear SGD estimators.
from sklearn.linear_model.sgd_fast import Hinge
from sklearn.linear_model.sgd_fast import SquaredHinge
from sklearn.linear_model.sgd_fast import Log
from sklearn.linear_model.sgd_fast import SquaredLoss


# Tool library in which we get robust mean estimators.
from .mean_estimators import MOM, blockMOM, huber


def _huber_psisx(x, c):
    """Huber-loss weight for RobustWeightedEstimator algorithm"""
    def psisx(x):
        if not(np.isfinite(x)):
            return 0
        elif np.abs(x) < c:
            return 1
        else:
            return (2*(x > 0)-1)*c/x
    return np.vectorize(psisx)(x)


def _mom_psisx(med_block, n):
    """MOM weight for RobustWeightedEstimator algorithm"""
    res = np.zeros(n)
    res[med_block] = 1
    return lambda x: res


class RobustWeightedEstimator(MetaEstimatorMixin, BaseEstimator):
    """Meta algorithm for robust regression and (Binary) classification.

    This model use iterative reweighting of samples to make a regression or
    classification estimator robust.

    This algorithm is stille very new and still in development.
    The principle of the algorithm is to use an empirical risk minimization
    principle where the risk is estimated using a robust estimator (for example
    Huber estimator or median-of-means estimator)[1], [3]. The idea behind this
    algorithm was mentionned before in [2].
    This idea translates in an iterative algorithm where the sample_weight
    are changed at each iterations and are dependent of the sample. Informally
    the outliers should have small weight while the inliers should have big
    weight.

    This algorithm enjoy a non-zero breakdown-point (it can handle arbitrarily
    bad outliers). When the "mom" weighting scheme is used, K/2 outliers can be
    tolerated. When the "Huber" weighting scheme is used, asymptotically the
    number of outliers has to be less than half the sample size.

    Read more in the :ref:`User Guide <RobustWeightedEstimator>`.

    Parameters
    ----------

    base_estimator : object or None, optional (default=None)
        The base estimator to fit. For now only SGDRegressor and SGDClassifier
        are supported.
        If None, then the base estimator is SGDRegressor with squared loss.

    weighting : string, optional (default="huber")
        Weighting scheme used to make the estimator robust.
        Can be 'huber' for huber-type weights or  'mom' for median-of-means
        type weights.

    max_iter : int, optional (default=100)
        Maximum number of iterations.
        For more information, see the optimization scheme of base_estimator
        and the eta0 and burn_in parameter.

    burn_in : int, optional (delaut=10)
        Number of steps used without changing the learning rate.
        Can be useful to make the weight estimation better at the beginning.

    eta0 : float, optional (default=0.01)
        Constant step-size used during the burn_in period. Used only if
        burn_in>0. Can have a big effect on efficiency.

    c : float>0 or None, optional (default None)
        Parameter used for Huber weighting procedure, used only if weightings
        is 'huber'. Measure the robustness of the weightint procedure. A small
        value of c means a more robust estimator.
        Can have a big effect on efficiency.
        If None, c is estimated at each step using half the Inter-quartile
        range.

    K : int, optional (default=3)
        Parameter used for mom weighting procedure, used only if weightings
        is 'huber'. It is the number of blocks used for median-of-means
        estimation, higher value of K means a more robust estimator.
        Can have a big effect on efficiency.

    loss : string or None, optional (default=None)
        Name of the loss used, must be the same loss as the one optimized in
        base_estimator.
        Classification losses supported : 'log', 'hinge'.
        If 'log', then the base_estimator must support predict_proba.
        Regression losses supported : 'squared_loss'.
        If None, loss='squared_loss'


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

    References
    ----------

    [1] Guillaume Lecué, Matthieu Lerasle and Timothée Mathieu.
        "Robust classification via MOM minimization", arXiv preprint (2019).
        arXiv:1808.03106

    [2] Christian Brownlees, Emilien Joly, and Gábor Lugosi.
        "Empirical risk minimization for heavy-tailed losses", Ann. Statist.
        Volume 43, Number 6 (2015), 2507-2536.

    [3] Stanislav Minsker, Timothée Mathieu.
        "Excess risk bounds in robust empirical risk minimization"
        arXiv preprint (2019). arXiv:1910.07485.

    """
    def __init__(self, base_estimator=None, weighting="huber", max_iter=100,
                 burn_in=10, eta0=0.01, c=None, K=3, loss=None):
        self.base_estimator = base_estimator
        self.weighting = weighting
        self.eta0 = eta0
        self.burn_in = burn_in
        self.c = c
        self.K = K
        self.loss = loss
        self.max_iter = max_iter
        if self.loss is None:
            warnings.warn("RobustWeightedEstimator: No loss"
                          " function given. Using squared loss"
                          " function for regression.")
            self.loss = "squared_loss"

        self.loss_functions = {
                "hinge": (Hinge, 1.0),
                "squared_hinge": (SquaredHinge, 1.0),
                "log": (Log, ),
                "squared_loss": (SquaredLoss, )

            }

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
        self._validate_hyperparameters()

        # Initialization of all parameters in the base_estimator.
        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = SGDRegressor()
        base_estimator.warm_start = True
        base_estimator.shuffle = False

        # Set one batch as the entirety of the data.
        base_estimator.batch_size = len(X)

        learning_rate = base_estimator.learning_rate
        base_estimator.learning_rate = "constant"
        base_estimator.loss = self.loss
        base_estimator.eta0 = self.eta0

        # Get actual loss function from its name.
        loss = self._get_loss_function()

        # Weight initialization : do one non-robust epoch.

        if self.loss in ['log', 'hinge', 'squared_hinge']:
            # If in a classification task, precise the classes.
            base_estimator.partial_fit(X, y, classes=list(set(y)))
        else:
            base_estimator.partial_fit(X, y)

        # Optimization algorithm
        for epoch in range(self.max_iter):
            if base_estimator.loss == 'log':
                # If log-loss use probabilties. Select only the probability
                # that it is 1.
                pred = base_estimator.predict_proba(X)[:, 1]
            elif base_estimator.loss == 'hinge':
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
            weights = self._weighting(loss_values)

            # Use the optimization algorithm of self.base_estimator for one
            # epoch using the previously computed weights.
            base_estimator.partial_fit(X, y, sample_weight=weights)

            if epoch > self.burn_in:
                # If not in the burn_in phase anymore, change the learning_rate
                # calibration to the one edicted by self.base_estimator.
                base_estimator.learning_rate = learning_rate

        self.weights = weights
        self.estimator = base_estimator
        return self

    def _get_loss_function(self):
        """Get concrete ``LossFunction`` object for str ``loss``. """
        try:
            loss_ = self.loss_functions[self.loss]
            loss_class, args = loss_[0], loss_[1:]
            return np.vectorize(loss_class(*args).dloss)
        except KeyError:
            raise ValueError("The loss %s is not supported. " % self.loss)

    def _validate_hyperparameters(self):
        # Check the hyperparameters.

        if self.max_iter <= 0:
            raise ValueError("RobustWeightedEstimator: "
                             "max_iter must be > 0, got %s." % self.max_iter)

        if not (self.c is None) and (self.c <= 0):
            raise ValueError("RobustWeightedEstimator: "
                             "c must be > 0, got %s." % self.c)

        if self.burn_in < 0:
            raise ValueError("RobustWeightedEstimator: "
                             "burn_in must be >= 0, got %s." % self.c)

        if (self.burn_in > 0) and (self.eta0 <= 0):
            raise ValueError("RobustWeightedEstimator: "
                             "eta0 must be > 0, got %s." % self.eta0)

        if not isinstance(self.K, int):
            raise ValueError("RobustWeightedEstimator: "
                             "K must be integer, got %s." % self.K)

    def _weighting(self, loss_values):
        # Compute the robust weight of the samples.
        if self.weighting == 'huber':
            if self.c is None:
                # If no c parameter given, estimate using inter quartile range.
                self.c = iqr(np.abs(loss_values-np.median(loss_values)))/2
                if self.c == 0:
                    warnings.warn("RobustWeightedEstimator: "
                                  "too many sampled are parfectly predicted "
                                  "according to the loss function. "
                                  "Switching to constant c = 1.35. "
                                  "Consider using another weighting scheme, "
                                  "or using a constant c value to remove "
                                  "this warning.")
                    self.c = 1.35

            def psisx(x):
                return _huber_psisx(x, self.c)

            # Robust estimation of the risk is in mu.
            mu = huber(loss_values, self.c)

        elif self.weighting == 'mom':
            # Choose (randomly) K (almost-)equal blocks of data.
            blocks = blockMOM(loss_values, self.K)
            # Compute the median-of-means of the losses using these blocks.
            # Return also the index at which this median-of-means is attained.
            mu, idmom = MOM(loss_values, blocks)
            psisx = _mom_psisx(blocks[idmom], len(loss_values))
        else:
            raise ValueError("RobustWeightedEstimator: "
                             "no such weighting scheme")
        # Compute the unnormalized weights.
        w = psisx(loss_values-mu)
        return w/np.sum(w)*len(loss_values)

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
        return self.estimator.predict(X)
