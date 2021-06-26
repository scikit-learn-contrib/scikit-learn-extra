"""Robust Mean estimation."""

# Author: Timothee Mathieu
# License: BSD 3 clause

import numpy as np
from scipy.stats import iqr
from sklearn.metrics import mean_squared_error


def block_mom(X, k, random_state):
    """Sample the indices of 2k+1 blocks for data x using a random permutation

    Parameters
    ----------

    X : array like, length = n_sample
        sample whose size correspong to the size of the sample we want to do
        blocks for.

    k : int
        we use 2k+1 blocks

    random_state : RandomState instance
        The seed of the pseudo random number generator to use when shuffling
        the data.

    Returns
    -------

    list of size K containing the lists of the indices of the blocks,
    the size of the lists are contained in [n_sample/K,2n_sample/K]
    """
    x = X.flatten()
    K = 2 * k + 1
    # Sample a permutation to shuffle the data.
    perm = random_state.permutation(len(x))
    return np.array_split(perm, K)


def median_of_means_blocked(X, blocks):
    """Compute the median of means of X using the blocks blocks

    Parameters
    ----------

    X : array like, length = n_sample
        sample from which we want an estimator of the mean

    blocks : list of list, provided by the function blockMOM.

    Return
    ------

    The median of means of x using the block blocks, a float.
    """
    x = X.flatten()

    # Compute the mean of each block
    means_blocks = [np.mean([x[f] for f in ind]) for ind in blocks]

    # Find the indice for which the mean of block is the median-of-means.
    indice = np.argsort(means_blocks)[int(np.floor(len(means_blocks) / 2))]
    return means_blocks[indice], indice


def median_of_means(X, k, random_state=np.random.RandomState(42)):
    """Compute the median of means of X using 2k+1 blocks

    Parameters
    ----------

    X : array like, length = n_sample
        sample from which we want an estimator of the mean

    k : int.

    random_state : RandomState instance
        The seed of the pseudo random number generator to use when shuffling
        the data.

    Return
    ------

    The median of means of x using 2k+1 random blocks, a float.
    """
    x = X.flatten()

    blocks = block_mom(x, k, random_state)
    return median_of_means_blocked(x, blocks)[0]


def huber(X, c=1.35, n_iter=20):
    """Compute the Huber estimator of location of X with parameter c

    Parameters
    ----------

    X : array like, length = n_sample
        sample from which we want an estimator of the mean

    c : float >0, default = 1.35
        parameter that control the robustness of the estimator.
        c going to zero gives a  behavior close to the median.
        c going to infinity gives a behavior close to sample mean.

    n_iter : int, default = 20
        Number of iterations of the algorithm.

    Return
    ------

    The Huber estimator of location on x with parameter c, a float.

    """
    x = X.flatten()

    # Initialize the algorithm with a robust first-guess : the median.
    mu = np.median(x)

    def psisx(x, c):
        # Huber weight function.
        res = np.zeros(len(x))
        mask = np.abs(x) <= c
        res[mask] = 1
        res[~mask] = c / np.abs(x[~mask])
        return res

    # Run the iterative reweighting algorithm to compute M-estimator.
    for t in range(n_iter):
        # Compute the weights
        w = psisx(x - mu, c)

        # Infinite coordinates in x gives zero weight, we take them out.
        ind_pos = w > 0

        # Update the value of the estimate with the new estimate using the
        # new weights.
        mu = np.sum(np.array(w[ind_pos]) * x[ind_pos]) / np.sum(w[ind_pos])
    return mu


def make_huber_metric(
    score_func=mean_squared_error, sample_weight=None, c=None, n_iter=20
):
    """
    Make a robust metric using Huber estimator.

    Read more in the :ref:`User Guide <make_huber_metric>`.

    Parameters
    ----------

    score_func :  callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    sample_weight: array-like of shape (n_samples,), default=None
        Sample weights.


    c : float >0, default = None
        parameter that control the robustness of the estimator.
        c going to zero gives a  behavior close to the median.
        c going to infinity gives a behavior close to sample mean.
        if c is None, the iqr (inter quartile range) is used as heuristic.

    n_iter : int, default = 20
        Number of iterations of the algorithm.

    Return
    ------

    Robust metric function, a callable  with signature
    ``score_func(y, y_pred, **kwargs).

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.metrics import mean_squared_error
    >>> from sklearn_extra.robust import make_huber_metric
    >>> robust_mse = make_huber_metric(mean_squared_error, c=5)
    >>> y_true = np.hstack([np.zeros(98), 20*np.ones(2)]) # corrupted test values
    >>> np.random.shuffle(y_true) # shuffle them
    >>> y_pred = np.zeros(100) # predicted values
    >>> robust_mse(y_true, y_pred)
    0.1020408163265306
    """

    def metric(y_true, y_pred):
        # change size in order to use the raw multisample
        # to have individual values
        y1 = [y_true]
        y2 = [y_pred]
        values = score_func(
            y1, y2, sample_weight=sample_weight, multioutput="raw_values"
        )
        if c is None:
            c_ = iqr(values)
        else:
            c_ = c
        if c_ == 0:
            return np.median(values)
        else:
            return huber(values, c_, n_iter)

    return metric
