.. title:: User guide : contents

.. _user_guide:


==========
User guide
==========

.. toctree::
     :numbered:

     modules/eigenpro.rst

.. _k_medoids:

K-Medoids
=========

:class:`KMedoids` is related to the :class:`KMeans` algorithm. While
:class:`KMeans` tries to minimize the within cluster sum-of-squares,
:class:`KMedoids` tries to minimize the sum of distances between each point and
the medoid of its cluster. The medoid is a data point (unlike the centroid)
which has least total distance to the other members of its cluster. The use of
a data point to represent each cluster's center allows the use of any distance
metric for clustering.

:class:`KMedoids` can be more robust to noise and outliers than :class:`KMeans`
as it will choose one of the cluster members as the medoid while
:class:`KMeans` will move the center of the cluster towards the outlier which
might in turn move other points away from the cluster centre.

:class:`KMedoids` is also different from K-Medians, which is analogous to :class:`KMeans`
except that the Manhattan Median is used for each cluster center instead of
the centroid. K-Medians is robust to outliers, but it is limited to the
Manhattan Distance metric and, similar to :class:`KMeans`, it does not guarantee
that the center of each cluster will be a member of the original dataset.

The complexity of K-Medoids is :math:`O(N^2 K T)` where :math:`N` is the number
of samples, :math:`T` is the number of iterations and :math:`K` is the number of
clusters. This makes it more suitable for smaller datasets in comparison to
:class:`KMeans` which is :math:`O(N K T)`.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_plot_kmedoids_digits.py`: Applying K-Medoids on digits
   with various distance metrics.


**Algorithm description:**
There are several algorithms to compute K-Medoids, though :class:`KMedoids`
currently only supports K-Medoids solver analogous to K-Means. Other frequently
used approach is partitioning around medoids (PAM) which is currently not
implemented.

This version works as follows:

* Initialize: Select ``n_clusters`` from the dataset as the medoids using
  a heuristic, random, or k-medoids++ approach (configurable using the ``init`` parameter).
* Assignment step: assign each element from the dataset to the closest medoid.
* Update step: Identify the new medoid of each cluster.
* Repeat the assignment and update step while the medoids keep changing or
  maximum number of iterations ``max_iter`` is reached.

.. topic:: References:

* Maranzana, F.E., 1963. On the location of supply points to minimize
  transportation costs. IBM Systems Journal, 2(2), pp.129-135.
* Park, H.S. and Jun, C.H., 2009. A simple and fast algorithm for K-medoids
  clustering. Expert systems with applications, 36(2), pp.3336-3341.

.. _commonnn:

Common-nearest-neighbors clustering
===================================

:class:`CommonNNClassifier` provides an interface to density-based
common-nearest-neighbors clustering. Density-based clustering identifies
clusters as dense regions of high point density, separated by sparse
regions of lower density. Common-nearest-neighbors clustering
approximates local density as the number of shared (common) neighbors
between two points with respect to a neighbor search radius. A density
threshold (density criterion) is used as cluster parameter to
distinguish high from low density. As such
the method is related to other density-based cluster algorithms like
DBSCAN or Jarvis-Patrick. DBSCAN approximates local density as the
number of points in the neighborhood of a single point. The
Jarvis-Patrick algorithm uses the number of common neighbors shared by
two points among the $k$ nearest neighbors. As these three approaches
each provide a different notion of how density is estimated from
point samples, they can be used complementary.

.. topic:: Examples:

* :ref:`examples/cluster/plot_commonnn.py
    <sphx_glr_auto_examples_cluster_plot_commonnn.py>`

.. topic:: References:

* B. Keller, X. Daura, W. F. van Gunsteren "Comparing Geometric and
  Kinetic Cluster Algorithms for Molecular Simulation Data" J. Chem.
  Phys., 2010, 132, 074110.

* O. Lemke, B.G. Keller "Density-based Cluster Algorithms for the
  Identification of Core Sets" J. Chem. Phys., 2016, 145, 164104.

* O. Lemke, B.G. Keller "Common nearest neighbor clustering - a
  benchmark" Algorithms, 2018, 11, 19.
