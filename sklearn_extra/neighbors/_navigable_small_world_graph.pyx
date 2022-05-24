# distutils: language = c++
#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

# NSWG-based ANN classification
# Authors: Lev Svalov <leos3112@gmail.com>
#          Stanislav Protasov <stanislav.protasov@gmail.com>
# License: BSD 3 clause

cimport numpy as np
np.import_array()
from libcpp.vector cimport vector
from libcpp.set cimport set as set_c
from libcpp.pair cimport pair as pair
from libc.math cimport pow
from libcpp.queue cimport priority_queue
from libc.stdlib cimport rand
import itertools
import numpy as np

cdef class BaseNSWGraph:
    """
    Cython-Optimized implementation of the Navigable small world graph structure

    Parameters
    ----------
    regularity : int, default: 16
        The size of the friends list of every vertex in the graph.
        Higher regularity leads to more accurate but slower search.

    guard_hops : int, default: 100
         The number of bi-directional links created for every new element in the graph.

    quantize : bool, default: False
         If True, use a product quantization for the preliminary dimensionality reduction of the data.

    quantization_levels : int, default: 20
         (Used if quantize=True)
         The number of the values used in quantization approximation of the dataset.

    """
    def __init__(self, ITYPE_t regularity=16,
                       ITYPE_t guard_hops=100,
                       ITYPE_t attempts=2,
                       BTYPE_t quantize=False,
                       ITYPE_t quantization_levels=20):
        self.regularity = regularity
        self.guard_hops = guard_hops
        self.attempts = attempts
        self.quantize = quantize,
        self.quantization_levels = quantization_levels

    cdef priority_queue[pair[DTYPE_t, ITYPE_t]] delete_duplicate(self, priority_queue[pair[DTYPE_t, ITYPE_t]] queue) nogil:
        """
        Auxiliary method for removing the duplicated nodes from the neighbor candidates sequence
        
        Parameters
        ----------
        queue: priority_queue of pairs consisting double distance value and the index of the particular node
        """
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] new_que
        cdef set_c[ITYPE_t] tmp_set
        new_que.push(queue.top())
        tmp_set.insert(queue.top().second)
        queue.pop()
        while queue.size() != 0:
            if tmp_set.find(queue.top().second) == tmp_set.end():
                tmp_set.insert(queue.top().second)
                new_que.push(queue.top())
            queue.pop()
        return new_que

    cdef DTYPE_t eucl_dist(self, vector[DTYPE_t] v1, vector[DTYPE_t] v2) nogil:
        """
        Calculation of the reduced Euclidean distance between two data vectors
        
        Parameters
        ----------
        v1, v2: vector of double features values
        
        Returns
        -------
        d: double, reduced Euclidean distance value
        """
        cdef ITYPE_t i = 0
        cdef DTYPE_t res = 0
        if self.quantize:
            for i in range(v1.size()):
                res += self.lookup_table[int(v2[i])][int(v1[i])]
        else:
            for i in range(v1.size()):
                res += pow(v1[i] - v2[i], 2)
        return res


    cdef void search_nsw_basic(self, vector[DTYPE_t] query,
                               set_c[ITYPE_t]* visitedSet,
                               priority_queue[pair[DTYPE_t, ITYPE_t]]* candidates,
                               priority_queue[pair[DTYPE_t, ITYPE_t]]* result,
                               ITYPE_t* res_hops,
                               ITYPE_t k) nogil:
        """
        Single search for neighbors candidates for the provided query vector
        
        Parameters
        ----------
        query: query data vector consisting double features values
        visitedSet: pointer set of nodes indices that was already visited by attempted searches
        candidates: pointer to sequence of possible neighbors for the query 
        result: pointer to final sequence of neighbors of the query
        res_hops: pointer to the result number of hops obtained after the search
        k: number of neighbors
        """
        cdef ITYPE_t entry = rand() % self.nodes.size()
        cdef ITYPE_t hops = 0
        cdef DTYPE_t closest_dist = 0
        cdef ITYPE_t closest_id = 0
        cdef ITYPE_t e = 0
        cdef DTYPE_t d = 0
        cdef pair[DTYPE_t, ITYPE_t] tmp_pair

        d = self.eucl_dist(query, self.nodes[entry])
        tmp_pair.first = d * (-1)
        tmp_pair.second = entry

        if visitedSet[0].find(entry) == visitedSet[0].end():
            candidates[0].push(tmp_pair)
        tmp_pair.first = tmp_pair.first * (-1)
        result[0].push(tmp_pair)
        hops = 0

        while hops < self.guard_hops:
            hops += 1
            if candidates[0].size() == 0:
                break
            tmp_pair = candidates[0].top()
            candidates.pop()
            closest_dist = tmp_pair.first * (-1)
            closest_id = tmp_pair.second
            if result[0].size() >= k:
                while result[0].size() > k:
                    result[0].pop()

                if result[0].top().first < closest_dist:
                    break

            for e in self.neighbors[closest_id]:
                if visitedSet[0].find(e) == visitedSet[0].end():
                    d = self.eucl_dist(query, self.nodes[e])
                    visitedSet[0].insert(e)
                    tmp_pair.first = d
                    tmp_pair.second = e
                    result.push(tmp_pair)
                    tmp_pair.first = tmp_pair.first * (-1)
                    candidates.push(tmp_pair)
        res_hops[0] = hops


    cdef np.ndarray _get_quantized(self, np.ndarray vector):
        """ Auxiliary method for transformation the initial data vector to quantized version """
        result = []
        for i, data_value in enumerate(vector):
            result.append((np.abs(self.quantization_values - data_value)).argmin())
        return np.array(result)


    cdef pair[vector[ITYPE_t], vector[DTYPE_t]] _multi_search(self, vector[DTYPE_t] query, ITYPE_t k) nogil:
        """
        Main neighbors search function that combines results from multiple attempted single search and deletes duplicated results
        
        Parameters
        ----------
        query: query data vector consisting double features values
        k: number of neighbors
        
        Returns
        -------
        ind, dist: pair of sequences: indices of neighbor vector and corresponding distance to the query vector 
        """
        cdef set_c[ITYPE_t] visitedSet
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] candidates
        cdef priority_queue[pair[DTYPE_t, ITYPE_t]] result
        cdef vector[ITYPE_t] res
        cdef vector[DTYPE_t] dist
        cdef ITYPE_t i
        cdef ITYPE_t hops
        cdef pair[DTYPE_t, ITYPE_t] j
        cdef ITYPE_t id
        cdef DTYPE_t d

        for i in range(self.attempts):
            self.search_nsw_basic(query, &visitedSet, &candidates, &result, &hops, k)
            result = self.delete_duplicate(result)
        while result.size() > k:
            result.pop()
        while res.size() < k:
            if result.empty():
                break
            el = result.top().second
            d = result.top().first
            dist.insert(dist.begin(), d)
            res.insert(res.begin(), el)
            result.pop()

        return pair[vector[ITYPE_t], vector[DTYPE_t]](res, dist)


    cdef void _build_navigable_graph(self, vector[vector[DTYPE_t]] X) nogil:
        """
        Build the Navigable small world graph
        
        Parameters
        ----------
        X: query data vectors that are constructing the data structure
        """
        cdef vector[DTYPE_t] val
        cdef vector[ITYPE_t] closest
        cdef ITYPE_t c
        cdef ITYPE_t i
        cdef vector[ITYPE_t] res
        cdef set_c[ITYPE_t] tmp_set
        if X.size() != self.number_nodes:
            raise Exception("Number of nodes don't match")
        if X[0].size() != self.dimension:
            raise Exception("Dimension doesn't match")

        self.nodes.clear()
        self.neighbors.clear()

        self.nodes.push_back(X[0])
        for i in range(self.number_nodes):
            self.neighbors.push_back(tmp_set)

        for i in range(1, self.number_nodes):
            val = X[i]
            closest.clear()
            closest = self._multi_search(val, k=self.regularity).first
            self.nodes.push_back(val)
            for c in closest:
                self.neighbors[i].insert(c)
                self.neighbors[c].insert(i)

    cdef vector[vector[DTYPE_t]] ndarray_to_vector_2(self, np.ndarray X):
        """ 
        Auxiliary method for conversion the numpy array of data vectors to libcpp 2d vector
        
        Parameters
        ---------- 
        X: numpy array to convert to the 2d vector
        
        Returns
        -------
        X_vector: libcpp 2d vector
        """
        cdef vector[vector[DTYPE_t]] X_vector
        cdef ITYPE_t i
        for i in range(len(X)):
            X_vector.push_back((X[i]))
        return X_vector

    cdef np.ndarray _quantization(self, np.ndarray X):
        """ 
        Auxiliary method for quantization of the given data.
        It quantizes the data vectors and constructs the lookup table of reduced distances
        
        Parameters
        ---------- 
        X: the given data to build NSWG
        
        Returns
        -------
        X_quantized: the quantizers data with reduced dimensionality
        """
        self.quantization_values = np.linspace(0.0, 1.0, num=self.quantization_levels)
        self.lookup_table = np.zeros(shape=(self.quantization_levels,self.quantization_levels))
        for v in itertools.combinations(enumerate(self.quantization_values), 2):
            i = v[0][0]
            j = v[1][0]
            self.lookup_table[i][j] = pow(np.abs(v[0][1]-v[1][1]),2)
            self.lookup_table[j][i] = pow(np.abs(v[1][1]-v[0][1]),2)
        X_quantized = []
        for i, vector in enumerate(X):
            X_quantized.append(self._get_quantized(vector))
        return np.array(X_quantized)

    def build(self, X):
        """
        Build BaseNSWGraph on the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features),
            Training data.
        """
        self.number_nodes = len(X)
        self.dimension = len(X[0])
        if self.quantize:
            quantized_data = self._quantization(X)
            X = quantized_data
        cdef vector[vector[DTYPE_t]] X_vector = self.ndarray_to_vector_2(X)
        self._build_navigable_graph(X_vector)

    def query(self, np.ndarray queries, ITYPE_t k=1):
        """Query the BaseNSWGraph for the k nearest neighbors

        Parameters
        ----------
        queries : array-like, shape = (n_samples, n_features),
            An array of points to query

        k : int, default=1
            The number of nearest neighbors to return

        Returns
        -------
        dist: ndarray of shape X.shape[:-1] + (k,), dtype=double
        Each entry gives the list of distances to the neighbors of the corresponding point.

        ind: ndarray of shape X.shape[:-1] + (k,), dtype=int
        Each entry gives the list of indices of neighbors of the corresponding point.
        """
        ind = []
        dist = []
        cdef pair[vector[ITYPE_t], vector[DTYPE_t]] res
        cdef vector[vector[DTYPE_t]] query_vector
        for query in queries:
            if self.quantize:
                normalized_query = query
                query = self._get_quantized(normalized_query)
            query = np.array([query])
            query_vector = self.ndarray_to_vector_2(query)
            res = self._multi_search(query_vector[0], k)
            ind.append(res.first)
            dist.append(res.second)
        return np.array(dist, dtype=object), np.array(ind, dtype=object)
