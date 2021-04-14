.. _cluster:

=====================================================
Clustering with KMedoids and Common-nearest-neighbors
=====================================================
.. _k_medoids:
.. currentmodule:: sklearn_extra.cluster

K-Medoids
=========


:class:`KMedoids` is related to the :class:`KMeans <sklearn.cluster.KMeans>` algorithm. While
:class:`KMeans <sklearn.cluster.KMeans>` tries to minimize the within cluster sum-of-squares,
:class:`KMedoids` tries to minimize the sum of distances between each point and
the medoid of its cluster. The medoid is a data point (unlike the centroid)
which has the least total distance to the other members of its cluster. The use of
a data point to represent each cluster's center allows the use of any distance
metric for clustering. It may also be a practical advantage, for instance K-Medoids
algorithms have been used for facial recognition for which the medoid is a
typical photo of the person to recognize while K-Means would have obtained a blurry
image that mixed several pictures of the person to recognize.

:class:`KMedoids` can be more robust to noise and outliers than :class:`KMeans <sklearn.cluster.KMeans>`
as it will choose one of the cluster members as the medoid while
:class:`KMeans <sklearn.cluster.KMeans>` will move the center of the cluster towards the outlier which
might in turn move other points away from the cluster centre.

:class:`KMedoids` is also different from K-Medians, which is analogous to :class:`KMeans <sklearn.cluster.KMeans>`
except that the Manhattan Median is used for each cluster center instead of
the centroid. K-Medians is robust to outliers, but it is limited to the
Manhattan Distance metric and, similar to :class:`KMeans <sklearn.cluster.KMeans>`, it does not guarantee
that the center of each cluster will be a member of the original dataset.

The complexity of K-Medoids is :math:`O(N^2 K T)` where :math:`N` is the number
of samples, :math:`T` is the number of iterations and :math:`K` is the number of
clusters. This makes it more suitable for smaller datasets in comparison to
:class:`KMeans <sklearn.cluster.KMeans>` which is :math:`O(N K T)`.

.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_plot_kmedoids_digits.py`: Applying K-Medoids on digits
    with various distance metrics.


**Algorithm description:**
There are several algorithms to compute K-Medoids, though :class:`KMedoids`
currently only supports K-Medoids solver analogous to K-Means called alternate
and the algorithm PAM (partitioning around medoids). Alternate algorithm is used
when speed is an issue.


* Alternate method works as follows:

    * Initialize: Select ``n_clusters`` from the dataset as the medoids using
      a heuristic, random, or k-medoids++ approach (configurable using the ``init`` parameter).
    * Assignment step: assign each element from the dataset to the closest medoid.
    * Update step: Identify the new medoid of each cluster.
    * Repeat the assignment and update step while the medoids keep changing or
      maximum number of iterations ``max_iter`` is reached.

* PAM method works as follows:

    * Initialize: Greedy initialization of ``n_clusters``. First select the point
      in the dataset that minimizes the sum of distances to a point. Then, add one
      point that minimizes the cost and loop until ``n_clusters`` points are selected.
      This is the ``init`` parameter called ``build``.
    * Swap Step: for all medoids already selected, compute the cost of swapping this
      medoid with any non-medoid point. Then, make the swap that decreases the cost
      the most. Loop and stop when there is no change anymore.

.. topic:: References:

  * Maranzana, F.E., 1963. On the location of supply points to minimize
    transportation costs. IBM Systems Journal, 2(2), pp.129-135.
  * Park, H.S. and Jun, C.H., 2009. A simple and fast algorithm for K-medoids
    clustering. Expert systems with applications, 36(2), pp.3336-3341.
  * Kaufman, L. and Rousseeuw, P.J. (2008). Partitioning Around Medoids (Program PAM).
    In Finding Groups in Data (eds L. Kaufman and P.J. Rousseeuw).
    doi:10.1002/9780470316801.ch2
  * Bhat, Aruna (2014).K-medoids clustering using partitioning around medoids
    for performing face recognition. International Journal of Soft Computing,
    Mathematics and Control, 3(3), pp 1-12.

.. _commonnn:

Common-nearest-neighbors clustering
===================================

:class:`CommonNNClustering <sklearn_extra.cluster.CommonNNClustering>`
provides an interface to density-based
common-nearest-neighbors clustering. Density-based clustering identifies
clusters as dense regions of high point density, separated by sparse
regions of lower density. Common-nearest-neighbors clustering
approximates local density as the number of shared (common) neighbors
between two points with respect to a neighbor search radius. A density
threshold (density criterion) is used – defined by the cluster
parameters ``min_samples`` (number of common neighbors) and ``eps`` (search
radius) – to distinguish high from low density. A high value of
``min_samples`` and a low value of ``eps`` corresponds to high density.

As such the method is related to other density-based cluster algorithms
like :class:`DBSCAN <sklearn.cluster.DBSCAN>` or Jarvis-Patrick. DBSCAN
approximates local density as the number of points in the neighborhood
of a single point. The Jarvis-Patrick algorithm uses the number of
common neighbors shared by two points among the :math:`k` nearest neighbors.
As these approaches each provide a different notion of how density is
estimated from point samples, they can be used complementarily. Their
relative suitability for a classification problem depends on the nature
of the clustered data. Common-nearest-neighbors clustering (as
density-based clustering in general) has the following advantages over
other clustering techniques:

  * The cluster result is deterministic. The same set of cluster
    parameters always leads to the same classification for a data set.
    A different ordering of the data set leads to a different ordering
    of the cluster assignment, but does not change the assignment
    qualitatively.
  * Little prior knowledge about the data is required, e.g. the number
    of resulting clusters does not need to be known beforehand (although
    cluster parameters need to be tuned to obtain a desired result).
  * Identified clusters are not restricted in their shape or size.
  * Points can be considered noise (outliers) if they do not fullfil
    the density criterion.

The common-nearest-neighbors algorithm tests the density criterion for
pairs of neighbors (do they have at least ``min_samples`` points in the
intersection of their neighborhoods at a radius ``eps``). Two points that
fullfil this criterion are directly part of the same dense data region,
i.e. they are *density reachable*. A *density connected* network of
density reachable points (a connected component if density reachability
is viewed as a graph structure) constitutes a separated dense region and
therefore a cluster. Note, that for example in contrast to
:class:`DBSCAN <sklearn.cluster.DBSCAN>` there is no differentiation in
*core* (dense points) and *edge* points (points that are not dense
themselves but neighbors of dense points). The assignment of points on
the cluster rims to a cluster is possible, but can be ambiguous. The
cluster result is returned as a 1D container of labels, i.e. a sequence
of integers (zero-based) of length :math:`n` for a data set of :math:`n`
points,
denoting the assignment of points to a specific cluster. Noise is
labeled with ``-1``. Valid clusters have at least two members. The
clusters are not sorted by cluster member count. In same cases the
algorithm tends to identify small clusters that can be filtered out
manually.

.. topic:: Examples:

  * :ref:`examples/cluster/plot_commonnn.py <sphx_glr_auto_examples_plot_commonnn.py>`
    Basic usage of the
    :class:`CommonNNClustering <sklearn_extra.cluster.CommonNNClustering>`
  * :ref:`examples/cluster/plot_commonnn_data_sets.py <sphx_glr_auto_examples_plot_commonnn_data_sets.py>`
    Common-nearest-neighbors clustering of toy data sets

.. topic:: Implementation:

  The present implementation of the common-nearest-neighbors algorithm in
  :class:`CommonNNClustering <sklearn_extra.cluster.CommonNNClustering>`
  shares some
  commonalities with the current
  scikit-learn implementation of :class:`DBSCAN <sklearn.cluster.DBSCAN>`.
  It computes neighborhoods from points in bulk with
  :class:`NearestNeighbors <sklearn.neighbors.NearestNeighbors>` before
  the actual clustering. Consequently, to store the neighborhoods
  it requires memory on the order of
  :math:`O(n ⋅ n_n)` for :math:`n` points in the data set where :math:`n_n`
  is the
  average number of neighbors (which is proportional to ``eps``), that is at
  worst :math:`O(n^2)`. Depending on the input structure (dense or sparse
  points or similarity matrix) the additional memory demand varies.
  The clustering itself follows a
  breadth-first-search scheme, checking the density criterion at every
  node expansion. The linear time complexity is roughly proportional to
  the number of data points :math:`n`, the total number of neighbors :math:`N`
  and the value of ``min_samples``. For density-based clustering
  schemes with lower memory demand, also consider:

    * :class:`OPTICS <sklearn.cluster.OPTICS>` – Density-based clustering
      related to DBSCAN using a ``eps`` value range.
    * `cnnclustering <https://pypi.org/project/cnnclustering/>`_ – A
      different implementation of common-nearest-neighbors clustering.

.. topic:: Notes:

  * :class:`DBSCAN <sklearn.cluster.DBSCAN>` provides an option to
    specify data point weights with ``sample_weights``. This feature is
    experimentally at the moment for :class:`CommonNNClustering` as
    weights are not well defined for checking the common-nearest-neighbor
    density criterion. It should not be used in production, yet.

.. topic:: References:

  * B. Keller, X. Daura, W. F. van Gunsteren "Comparing Geometric and
    Kinetic Cluster Algorithms for Molecular Simulation Data" J. Chem.
    Phys., 2010, 132, 074110.

  * O. Lemke, B.G. Keller "Density-based Cluster Algorithms for the
    Identification of Core Sets" J. Chem. Phys., 2016, 145, 164104.

  * O. Lemke, B.G. Keller "Common nearest neighbor clustering - a
    benchmark" Algorithms, 2018, 11, 19.
