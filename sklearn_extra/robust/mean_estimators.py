"""Robust Mean estimation."""

# Author: Timothee Mathieu
# License: BSD 3 clause

import numpy as np


def blockMOM(x, k, random_state):
    """Sample the indices of 2k+1 blocks for data x using a random permutation

    Parameters
    ----------

    k : int
        we use 2k+1 blocks

    x : array like, length = n_sample
        sample whose size correspong to the size of the sample we want to do
        blocks for.

    random_state : RandomState instance
        The seed of the pseudo random number generator to use when shuffling
        the data.

    Returns
    -------

    list of size K containing the lists of the indices of the blocks,
    the size of the lists are contained in [n_sample/K,2n_sample/K]
    """
    K = 2 * k + 1
    b = int(np.floor(len(x) / K))
    nb = K - (len(x) - b * K)
    nbpu = len(x) - b * K
    # Sample a permutation to shuffle the data.
    perm = random_state.permutation(len(x))
    # Construct K blocks of approximately equal size
    blocks = [[(b + 1) * g + f for f in range(b + 1)] for g in range(nbpu)]
    blocks += [
        [nbpu * (b + 1) + b * g + f for f in range(b)] for g in range(nb)
    ]
    return [perm[b] for b in blocks]


def median_of_means_blocked(x, blocks):
    """Compute the median of means of x using the blocks blocks

    Parameters
    ----------

    x : array like, length = n_sample
        sample from which we want an estimator of the mean

    blocks : list of list, provided by the function blockMOM.

    Return
    ------

    The median of means of x using the block blocks, a float.
    """
    # Compute the mean of each block
    means_blocks = [np.mean([x[f] for f in ind]) for ind in blocks]

    # Find the indice for which the mean of block is the median-of-means.
    indice = np.argsort(means_blocks)[int(np.floor(len(means_blocks) / 2))]
    return means_blocks[indice], indice


def median_of_means(x, k, random_state=np.random.RandomState(42)):
    """Compute the median of means of x using 2k+1 blocks

    Parameters
    ----------

    x : array like, length = n_sample
        sample from which we want an estimator of the mean

    k : int.

    random_state : RandomState instance
        The seed of the pseudo random number generator to use when shuffling
        the data.

    Return
    ------

    The median of means of x using 2k+1 random blocks, a float.
    """
    blocks = blockMOM(x, k, random_state)
    return median_of_means_blocked(x, blocks)[0]


def huber(x, c=1.35, T=20):
    """Compute the Huber estimator of location of x with parameter c

    Parameters
    ----------

    x : array like, length = n_sample
        sample from which we want an estimator of the mean

    c : float >0, default = 1.35
        parameter that control the robustness of the estimator.
        c going to zero gives a  behavior close to the median.
        c going to infinity gives a behavior close to sample mean.

    T : int, default = 20
        Number of iterations of the algorithm.

    Return
    ------

    The Huber estimator of location on x with parameter c, a float.

    """
    # Initialize the algorithm with a robust first-guess : the median.
    mu = np.median(x)

    def psisx(x, c):
        # Huber weight function.
        res = np.ones(len(x))
        res[x != 0] = (2 * (x[x != 0] > 0) - 1) * c / x[x != 0]
        res[np.abs(x) < c] = 1
        res[~np.isfinite(x)] = 0
        return res

    # Run the iterative reweighting algorithm to compute M-estimator.
    for t in range(T):
        # Compute the weights
        w = psisx(x - mu, c)

        # Infinite coordinates in x gives zero weight, we take them out.
        ind_pos = w > 0

        # Update the value of the estimate with the new estimate using the
        # new weights.
        mu = np.sum(np.array(w[ind_pos]) * x[ind_pos]) / np.sum(w[ind_pos])
    return mu
