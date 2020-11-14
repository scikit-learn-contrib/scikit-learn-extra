# cython: infer_types=True
# Fast swap step in PAM algorithm for k_medoid.
# Author: Timothée Mathieu
# License: 3-clause BSD

cimport cython
import numpy as np
cimport numpy as np

from sklearn.utils.extmath import row_norms
from cython cimport floating

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
