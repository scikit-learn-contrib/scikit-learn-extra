
===================================================
Robust algorithms for Regression and Classification
===================================================

From Wikipedia: https://en.wikipedia.org/wiki/Robust_statistics, "Robust
statistics seek to provide methods that emulate popular statistical methods,
but which are not unduly affected by outliers or other small departures from
model assumptions." In particular, in machine learning, we want to bound the
influence that any minority of the dataset can have on the prediction.

.. |robust_regression| image:: ../robust_plot_regression.png
       :target: ../examples/plot_robust_regression_toy.py
       :scale: 70

.. centered:: |robust_regression|

What is an outlier ?
====================

We use the term "outlier" to mean a discordant minority of the dataset and it
can come in a lot of different forms. Often, outliers are points that are
situated outside the bulk of the data but it is a simplistic view as shown in the figure below.
There can be outliers in the features X or in the labels Y or in both.

One way one could define an outlier is with respect to the task we have
to do, and in this sense an outlier is a point with a large value of the loss
function (with respect to an optimal "oracle" estimator defined on the inliers).
Visually, in the following scatter plots, in the case of classification, we can
see that the points in the up-right corner are outliers while the points
in the bottom-left corner are not, this is a supervised learning definition of
outliers as opposed to unsupervised learning where both cluster of points would be
considered outliers as they are outside of the bulk of the data.

.. |outlier| image:: ../robust_def_outliers.png
      :scale: 70

.. centered:: |outlier|

Outliers can be caused by a lot of things, among them are human errors, captor errors or inherent causes.
For example one can think in an economical setting to income which is highly
corrupted by outliers (there are people with very high income).

Here, we limit ourselves to linear estimators, but non-linear estimators are
also plagued with the same non-robustness properties. See scikit-learn RANSAC
documentation (`scikit-learn <https://scikit-learn.org/stable/modules/linear_model.html#ransac-random-sample-consensus>`__)
for an example of outliers for non-linear estimators.

Robust estimation with robust weighting
=======================================

Glimpse of the theory
---------------------

A lot of learning algorithms are based on a paradigm known as empirical risk
minimization (ERM) which consists in finding the estimator :math:`\widehat{f}` that minimizes an
estimation of the risk.

.. math::

  \frac{1}{n} \sum_{i=1}^n \ell(\widehat{f}(X_i),y_i)= \min_{f}\, \frac{1}{n} \sum_{i=1}^n \ell(f(X_i),y_i)

where the :math:`ell` is a loss function (one can think of the squared distance in
regression). Said in another way, we are trying to minimize an estimation of
the expected risk and this estimation is done by means of the empirical mean.
However, it is well known that the empirical mean is not robust to extreme data
and these extreme values will have a big influence on the estimation of :math:`\widehat{f}`. The principle behind the robust weighting algorithm is to
use a robust estimator of the mean, instead of the empirical mean, one use
either the median-of-means (MOM) or the Huber estimator to estimate the mean.
And then we find an estimator :math:`\widehat{f}` that minimizes that robust estimation of the risk.
We call this Robust Empirical Risk Minimization (RERM) [1]_.

In practice, for a large range of robust estimators of the mean, one can
define weights :math:`w_i` that depends on the :math:`i^{th}` sample, with the weight :math:`w_i` being
very small when the :math:`i^{th}` data is an outlier and large :math:`w_i` when the point is not
an outlier. Then, we are reduced to the following optimization.

.. math::

  \frac{1}{n} \sum_{i=1}^n w_i \ell(\widehat{f}(X_i),y_i)= = \min_{f}\, \frac{1}{n} \sum_{i=1}^n w_i\ell(f(X_i),y_i)

Remark that the weights :math:`w_i` depends on :math:`\widehat{f}`, and the resulting algorithm is then an alternate
optimization scheme, iteratively doing one step to optimize with respect to :math:`f`
while the weights stay fixed and then one step to estimate the weights while :math:`f` stays fixed.
These two steps are then repeated until convergence.

Robust estimation in practice
=============================

The algorithm
-------------

We use a meta algorithm that take into entry a base estimator (for example a
stochastic gradient descent algorithm like SGDClassifier or SGDRegressor), the
conditions for an estimator to be used are mainly that it must support
partial_fit and sample_weight but for now only SGDClassifier and SGDRegressor
are officially supported. One can see the example using KMeans as an example if one wants to
adapt the method to other estimators.

At each step we estimates some sample weights that are meant to be small for
outliers and large for inliers and then we do one optimization step using the
base_estimator optimization algorithm.

There are two weighting scheme supported in this algorithm: Huber-like weights
and median-of-means weights, these two types of weights both comes with a
parameter that will determine the robustness/efficiency trade-off of the
estimation.

* Huber weights : in the case of "huber" weighting parameter, the parameter "c" used
  to express the trade-off between robustness and efficiency.
  It is a positive real number. When c goes to 0, the estimator
  gets more robust and less efficient while when c goes to infinity, the Huber
  estimator is not robust but more efficient if there were only inliers (points with a small loss).
  A good heuristic would be to choose "c" as an estimate of the standard deviation of the losses of the inliers.
  In practice, if c=None, it is estimated with the inter-quartile range
  but it can also be fixed to a constant and then tuned via `cross-validation <https://scikit-learn.org/stable/modules/cross_validation.html>`__.


* Median-of-means weights : in the case of "mom" weighting parameter, the parameter "k" is
  used to express the trade-off between robustness and efficiency. "k" is a non-negative integer,
  when k=0 then the estimator exactly the same as base_estimator and
  when k=sample_size/2 the estimator is very robust but less efficient on inliers.
  A good heuristic would be to choose k as an estimate of the number of outliers.
  In practice, if k=None, it is estimated using the number of points distant from
  the median of more than a 1.45 times the inter-quartile range but it can also
  be fixed to a constant and then tuned via `cross-validation <https://scikit-learn.org/stable/modules/cross_validation.html>`__.

  .. table:: Robustness/Efficiency tradeoff and choice of parameters
     :widths: auto
     :align: center

     +-----------+----------------------+-----------------+-------------------+
     | weighting | Robustness parameter | Small parameter | Large parameter   |
     +===========+======================+=================+===================+
     |    mom    |          k           |   Non robust    |      Robust       |
     +-----------+----------------------+-----------------+-------------------+
     |   huber   |          c           |   Robust        |      Non robust   |
     +-----------+----------------------+-----------------+-------------------+


The choice of the optimization parameters max_iter and eta0 are also very
important for the efficiency of this estimator and one may want to use
cross-validation to fix these hyper-parameters, choosing eta0 too large can have the effect of
making the estimator non-robust. One should also take care that it can be
important to rescale the data (the same way as it is important to do it for SGD)
. In the context of a corrupted dataset, please use 'RobustScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html>'.

This algorithm has been studied in the context of "mom" weights in the article [1]_,
the context of "huber" weights has been mentioned in [2]_. Both weighting scheme can be seen as
a special cases of the algorithm in [3]_.

Comparison with other robust estimators
---------------------------------------

There are already some robust algorithms in scikit-learn but one major difference is that robust
algorithms in scikit-learn are primarily meant for Regression, see `robustness in regression <https://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outliers-and-modeling-errors>`__.
Hence, we will not talk about classification algorithms in this comparison.

As such we only compare ourselves to TheilSenRegressor and RANSACRegressor as they
both deal with outliers in X and in Y and are closer to RobustWeightedEstimator.

**Warning:** the Huber weights we use here is very different from HuberRegressor
or other regression with "robust losses". Indeed, this kind of regression is robust
only to outliers in the label Y but not in X. This can be seen in the examples where
the chosen estimator is SGDRegressor which use the Hinge loss, a robust loss (in Y),
in a corrupted setting (in X) SGDRegressor will fail while RobustWeightedEstimator
will estimate correctly if tuned appropriately.

In regression, we have the following pros for RobustWeightedEstimator.

* RANSACRegressor and TheilSenRegressor both use a hard rejection of outlier.
  This can be interpreted as though there was an outlier detection step and then a
  regression step whereas RobustWeightedEstimator is directly robust to outliers.
  Empirically, robust estimators has been found to be more efficient than the
  two steps procedure outlier detection and then regression. Another way to say that is that the outliers are treated as though they have no influence, while RobustWeightedEstimator
  acknowledge the presence of outliers but it bounds their influence on the prediction.
* RobustWeightedEstimator provides a weight_ output that can be considered as an "outlying score".
* RobustWeightedEstimator can use any regularization that is part of SGD algorithms.


And the cons.

* There are cases where we want outliers to have no influence (captor error for example).
* In general, in small dimension, RobustWeightedEstimator with "mom" weights is
  less efficient than both TheilSenRegressor and RANSACRegressor when the sample_size is small.
* In general, this algorithm is slower than both  TheilSenRegressor and RANSACRegressor.
  The specifics depend on the sample_size, dimension and base_estimator.

One other advantage of RobustWeightedEstimator is that it can be used for a broad range of algorithms.
For example, one can do robust unsupervised learning with RobustWeightedEstimator, an example of which
is given using KMeans algorithm.

Speed and limits of the algorithm
---------------------------------

Most of the time, it is interesting to do robust statistics only when there
are outliers. Generally, one can compute both a robust and a non-robust
estimator and if there is no big discrepancies between the two, a robust
estimator may not be needed. On the other hand, there can be a great gain in
using robust algorithms for dataset that are highly corrupted. See the examples on real datasets.
A lot of dataset have previously been "cleaned" of any outlier, for small dataset this
can be done by an expert for example, on these dataset this algorithm is often not useful.

With respect to the dimensionality, the algorithm is expected to be as good (or as bad) as
the base_estimator do in high dimension.

Complexity:

* If weighting="huber": the computation is in general slower but the complexity order of magnitude is not changed compared
  to base_estimator complexity.

* If weighting="mom": the parameter k represent a trade-off efficiency vs computational time.
  Indeed, as said previously it is advised for efficiency to choose k equal to about
  the number of outliers. On the other hand the larger k is, the faster the algorithm will perform.


Limitations and comparison of the two weighting scheme:
-------------------------------------------------------

The parameter weighting="mom" is advised only with sufficiently large dataset
(thumb rule sample_size > 500 the specifics depend on the dataset), this weighting
scheme uses a smart sub-sampling of the dataset and as such, small dataset are not
a good fit with median-of-means, weighting="huber" does not present this drawback.
On the other hand, median-of-means estimation can be beneficial when the sample size
is large, in particular because of the complexity.

Warning about cross-validation
------------------------------

On a real dataset, one should be aware that there can be outliers in the training
set but also in the test set. To deal with outliers in the test set when evaluating
the model, one way of doing things is to choose a robust loss function: `accuracy_score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score>`__
or `roc_auc_score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`__
are examples of robust losses in Classification and
`median_absolute_error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html>`__
is an example in Regression. Another possibility is to use a robust estimator of the mean. For example in the California housing
real data example, we used the median instead of the mean to estimate the test loss, but a more efficient estimator (huber estimator for example) could also be used.

.. topic:: References:

    .. [1] Guillaume Lecué, Matthieu Lerasle and Timothée Mathieu.
           `"Robust classification via MOM minimization" <https://arxiv.org/abs/1808.03106>`_, arXiv preprint (2018).
           arXiv:1808.03106

    .. [2] Christian Brownlees, Emilien Joly and Gábor Lugosi.
           `"Empirical risk minimization for heavy-tailed losses" <https://projecteuclid.org/euclid.aos/1444222083>`_, Ann. Statist.
           Volume 43, Number 6 (2015), 2507-2536.

    .. [3] Stanislav Minsker and Timothée Mathieu.
           `"Excess risk bounds in robust empirical risk minimization" <https://arxiv.org/abs/1910.07485>`_
           arXiv preprint (2019). arXiv:1910.07485.
