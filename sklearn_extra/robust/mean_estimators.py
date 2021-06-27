"""Robust Mean estimation."""

# Author: Timothee Mathieu
# License: BSD 3 clause

import numpy as np


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


def huber(X, scale_param=None, n_iter=20, tol=1e-3):
    """Compute the Huber estimator of location of X with parameter c

    Parameters
    ----------

    X : array like, length = n_sample
        sample from which we want an estimator of the mean

    scale_param : float >0, default = None
        parameter that control the robustness of the estimator.
        scale_param going to zero gives a  behavior close to the median.
        scale_param going to infinity gives a behavior close to sample mean.
        if scale_param is None, the interquartile range (IQR) is used
        as heuristic.

    n_iter : int, default = 20
        Number of iterations of the algorithm.

    tol : float, default=1e-3
        Tolerance on stopping criterion.

    Return
    ------

    The Huber estimator of location on x with parameter c, a float.

    """
    x = X.flatten()

    # Initialize the algorithm with a robust first-guess : the median.
    mu = np.median(x)

    if scale_param is None:
        scale_param_ = iqr(x)
    else:
        scale_param_ = scale_param

    def psisx(x, c):
        # Huber weight function.
        res = np.zeros(len(x))
        mask = np.abs(x) <= scale_param_
        res[mask] = 1
        res[~mask] = scale_param_ / np.abs(x[~mask])
        return res

    # Create a list to keep the ten last values of mu
    last_mu = mu

    # Run the iterative reweighting algorithm to compute M-estimator.
    for t in range(n_iter):
        # Compute the weights
        w = psisx(x - mu, scale_param_)

        # Infinite coordinates in x gives zero weight, we take them out.
        ind_pos = w > 0

        # Update the value of the estimate with the new estimate using the
        # new weights.
        mu = np.sum(np.array(w[ind_pos]) * x[ind_pos]) / np.sum(w[ind_pos])

        # Stopping criterion. The error is decreasing at each iteration
        if np.abs(mu - last_mu) < tol:
            break
        else:
            last_mu = mu

    return mu
