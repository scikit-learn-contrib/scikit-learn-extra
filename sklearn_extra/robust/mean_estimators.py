"""Robust Mean estimation."""

# Author: Timothee Mathieu
# License: BSD 3 clause

import numpy as np
from sklearn.utils import check_random_state



def blockMOM(x, K, random_state):
    """Sample the indices of K blocks for data x using a random permutation

    Parameters
    ----------

    K : int
        number of blocks

    x : array like, length = n_sample
        sample whose size correspong to the size of the sample we want to do
        blocks for.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.

    Returns
    -------

    list of size K containing the lists of the indices of the blocks,
    the size of the lists are contained in [n_sample/K,2n_sample/K]
    """
    b = int(np.floor(len(x) / K))
    nb = K - (len(x) - b * K)
    nbpu = len(x) - b * K
    # Sample a permutation to shuffle the data.
    random_state = check_random_state(random_state)
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


def median_of_means(x, K):
    """Compute the median of means of x using K blocks

    Parameters
    ----------

    x : array like, length = n_sample
        sample from which we want an estimator of the mean

    K : int.

    Return
    ------

    The median of means of x using K random blocks, a float.
    """
    blocks = blockMOM(K, x)
    return MOM(x, blocks)[0]


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
        if not (np.isfinite(x)):
            return 0
        else:
            return 1 if np.abs(x) < c else (2 * (x > 0) - 1) * c / x

    def get_weight(x, mu, c):
        # Compute weight.
        if x - mu == 0:
            return 1
        else:
            return psisx(x - mu, c)

    # Run the iterative reweighting algorithm to compute M-estimator.
    for t in range(T):
        # Compute the weights
        w = np.array([get_weight(xx, mu, c) for xx in x])

        # Infinite coordinates in x gives zero weight, we take them out.
        ind_pos = w > 0

        # Update the value of the estimate with the new estimate using the
        # new weights.
        mu = np.sum(np.array(w[ind_pos]) * x[ind_pos]) / np.sum(w[ind_pos])
    return mu
