# cython: infer_types=True
# Fast swap step in PAM algorithm for k_medoid.
# Author: Timothée Mathieu
# License: 3-clause BSD

cimport cython
from cython.parallel import prange

def _compute_optimal_swap( double[:,:] D,
                           long[:] medoid_idxs,
                           long[:] not_medoid_idxs,
                           double[:] Djs,
                           double[:] Ejs,
                           int n_clusters,
                           int n_threads):
    """Compute best cost change for all the possible swaps"""

    # Initialize best cost change and the associated swap couple.
    cdef (int, int, double) best_cost_change = (1, 1, 0.0)
    cdef int sample_size = len(D)
    cdef int i, j, h, id_i, id_h, id_j
    cdef double T
    cdef int not_medoid_shape = sample_size - n_clusters
    cdef bint cluster_i_bool, not_cluster_i_bool, second_best_medoid, not_second_best_medoid
    cdef double to_add

    # Compute the change in cost for each swap.
    for h in prange(not_medoid_shape,  nogil=True, num_threads = n_threads):
        # id of the potential new medoid.
        id_h = not_medoid_idxs[h]
        for i in range(n_clusters):
            # id of the medoid we want to replace.
            id_i = medoid_idxs[i]
            T = 0.0
            # compute for all not-selected points the change in cost
            for j in range(not_medoid_shape):
                id_j = not_medoid_idxs[j]
                cluster_i_bool = D[id_i, id_j] == Djs[id_j]
                not_cluster_i_bool =D[id_i, id_j] != Djs[id_j]
                second_best_medoid = D[id_h, id_j] < Ejs[id_j]
                not_second_best_medoid = D[id_h, id_j] >= Ejs[id_j]
                if cluster_i_bool & second_best_medoid:
                    T = T + D[id_j, id_h] -Djs[id_j]
                elif cluster_i_bool & not_second_best_medoid:
                    T = T + Ejs[id_j] - Djs[id_j]
                elif not_cluster_i_bool & (D[id_j, id_h] < Djs[id_j]):
                    T = T + D[id_j, id_h] - Djs[id_j]

            # same for i
            second_best_medoid = D[id_h, id_i] < Ejs[id_i]
            if  second_best_medoid:
                T = T + D[id_i, id_h]
            else:
                T = T + Ejs[id_i]

            if T < best_cost_change[2]:
                best_cost_change = (id_i, id_h, T)

    # If one of the swap decrease the objective, return that swap.
    if best_cost_change[2] < 0:
        return best_cost_change
    else:
        return None
