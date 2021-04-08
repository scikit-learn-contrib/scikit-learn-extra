# cython: infer_types=True
# Fast swap step in PAM algorithm for k_medoid.
# Author: Timothée Mathieu
# License: 3-clause BSD

cimport cython
import numpy as np
cimport numpy as np

from sklearn.utils.extmath import row_norms
from cython cimport floating

import sys
from time import time

from libc.math cimport exp, log, sqrt, pow, fabs
cimport numpy as np
from numpy.math cimport INFINITY


# Modified from sklearn.cluster._k_means_fast.pyx
np.import_array()

cdef floating _euclidean_dense_dense(
        floating* a,  # IN
        floating* b,  # IN
        int n_features) nogil:
    """Euclidean distance between a dense and b dense"""
    cdef:
        int i
        int n = n_features // 4
        int rem = n_features % 4
        floating result = 0

    # We manually unroll the loop for better cache optimization.
    for i in range(n):
        result += ((a[0] - b[0]) * (a[0] - b[0])
                  +(a[1] - b[1]) * (a[1] - b[1])
                  +(a[2] - b[2]) * (a[2] - b[2])
                  +(a[3] - b[3]) * (a[3] - b[3]))
        a += 4; b += 4

    for i in range(rem):
        result += (a[i] - b[i]) * (a[i] - b[i])

    return result



cpdef np.ndarray[floating] _kmeans_loss(np.ndarray[floating, ndim=2, mode='c'] X,
                                        int[:] labels):
    """Compute inertia

    squared distancez between each sample and its assigned center.
    """
    if floating is float:
        dtype = np.float32
    elif floating is double:
        dtype = np.double

    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int i, j
        int n_classes = len(np.unique(labels))
        np.ndarray[floating, ndim=2] centers = np.zeros([n_classes,
                                                         n_features],
                                                         dtype = dtype)
        np.ndarray[long] num_in_cluster = np.zeros(n_classes, dtype = int)
        np.ndarray[floating] inertias = np.zeros(n_samples, dtype = dtype)
    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j]
        num_in_cluster[labels[i]] += 1

    for i in range(n_classes):
        for j in range(n_features):
            centers[i, j] /= num_in_cluster[i]

    for i in range(n_samples):
        j = labels[i]
        inertias[i] = _euclidean_dense_dense(&X[i, 0], &centers[j, 0], n_features)
    return inertias





# Regression and Classification losses, from scikit-learn.




# ----------------------------------------
# Extension Types for Loss Functions
# ----------------------------------------

cdef class LossFunction:
    """Base class for convex loss functions"""

    cdef double loss(self, double p, double y) nogil:
        """Evaluate the loss function.

        Parameters
        ----------
        p : double
            The prediction, p = w^T x
        y : double
            The true value (aka target)

        Returns
        -------
        double
            The loss evaluated at `p` and `y`.
        """
        return 0.

    def py_dloss(self, double p, double y):
        """Python version of `dloss` for testing.

        Pytest needs a python function and can't use cdef functions.
        """
        return self.dloss(p, y)

    def py_loss(self, double p, double y):
        """Python version of `dloss` for testing.

        Pytest needs a python function and can't use cdef functions.
        """
        return self.loss(p, y)


    cdef double dloss(self, double p, double y) nogil:
        """Evaluate the derivative of the loss function with respect to
        the prediction `p`.

        Parameters
        ----------
        p : double
            The prediction, p = w^T x
        y : double
            The true value (aka target)
        Returns
        -------
        double
            The derivative of the loss function with regards to `p`.
        """
        return 0.


cdef class Regression(LossFunction):
    """Base class for loss functions for regression"""

    cdef double loss(self, double p, double y) nogil:
        return 0.

    cdef double dloss(self, double p, double y) nogil:
        return 0.


cdef class Classification(LossFunction):
    """Base class for loss functions for classification"""

    cdef double loss(self, double p, double y) nogil:
        return 0.

    cdef double dloss(self, double p, double y) nogil:
        return 0.


cdef class ModifiedHuber(Classification):
    """Modified Huber loss for binary classification with y in {-1, 1}

    This is equivalent to quadratically smoothed SVM with gamma = 2.

    See T. Zhang 'Solving Large Scale Linear Prediction Problems Using
    Stochastic Gradient Descent', ICML'04.
    """
    cdef double loss(self, double p, double y) nogil:
        cdef double z = p * y
        if z >= 1.0:
            return 0.0
        elif z >= -1.0:
            return (1.0 - z) * (1.0 - z)
        else:
            return -4.0 * z

    cdef double dloss(self, double p, double y) nogil:
        cdef double z = p * y
        if z >= 1.0:
            return 0.0
        elif z >= -1.0:
            return 2.0 * (1.0 - z) * -y
        else:
            return -4.0 * y

    def __reduce__(self):
        return ModifiedHuber, ()


cdef class Hinge(Classification):
    """Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by SVM.
        When threshold=0.0, one gets the loss used by the Perceptron.
    """

    cdef double threshold

    def __init__(self, double threshold=1.0):
        self.threshold = threshold

    cdef double loss(self, double p, double y) nogil:
        cdef double z = p * y
        if z <= self.threshold:
            return self.threshold - z
        return 0.0

    cdef double dloss(self, double p, double y) nogil:
        cdef double z = p * y
        if z <= self.threshold:
            return -y
        return 0.0

    def __reduce__(self):
        return Hinge, (self.threshold,)


cdef class SquaredHinge(Classification):
    """Squared Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by
        (quadratically penalized) SVM.
    """

    cdef double threshold

    def __init__(self, double threshold=1.0):
        self.threshold = threshold

    cdef double loss(self, double p, double y) nogil:
        cdef double z = self.threshold - p * y
        if z > 0:
            return z * z
        return 0.0

    cdef double dloss(self, double p, double y) nogil:
        cdef double z = self.threshold - p * y
        if z > 0:
            return -2 * y * z
        return 0.0

    def __reduce__(self):
        return SquaredHinge, (self.threshold,)


cdef class Log(Classification):
    """Logistic regression loss for binary classification with y in {-1, 1}"""

    cdef double loss(self, double p, double y) nogil:
        cdef double z = p * y
        # approximately equal and saves the computation of the log
        if z > 18:
            return exp(-z)
        if z < -18:
            return -z
        return log(1.0 + exp(-z))

    cdef double dloss(self, double p, double y) nogil:
        cdef double z = p * y
        # approximately equal and saves the computation of the log
        if z > 18.0:
            return exp(-z) * -y
        if z < -18.0:
            return -y
        return -y / (exp(z) + 1.0)

    def __reduce__(self):
        return Log, ()


cdef class SquaredLoss(Regression):
    """Squared loss traditional used in linear regression."""
    cdef double loss(self, double p, double y) nogil:
        return 0.5 * (p - y) * (p - y)

    cdef double dloss(self, double p, double y) nogil:
        return p - y

    def __reduce__(self):
        return SquaredLoss, ()


cdef class Huber(Regression):
    """Huber regression loss

    Variant of the SquaredLoss that is robust to outliers (quadratic near zero,
    linear in for large errors).

    https://en.wikipedia.org/wiki/Huber_Loss_Function
    """

    cdef double c

    def __init__(self, double c):
        self.c = c

    cdef double loss(self, double p, double y) nogil:
        cdef double r = p - y
        cdef double abs_r = fabs(r)
        if abs_r <= self.c:
            return 0.5 * r * r
        else:
            return self.c * abs_r - (0.5 * self.c * self.c)

    cdef double dloss(self, double p, double y) nogil:
        cdef double r = p - y
        cdef double abs_r = fabs(r)
        if abs_r <= self.c:
            return r
        elif r > 0.0:
            return self.c
        else:
            return -self.c

    def __reduce__(self):
        return Huber, (self.c,)
