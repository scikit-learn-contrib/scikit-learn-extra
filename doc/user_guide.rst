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
currently only supports a non-standard version of K-Medoids substantially
different from the well-known PAM algorithm.

This version works as follows:

* Initialize: Select ``n_clusters`` from the dataset as the medoids using
  a heuristic, random, or k-medoids++ approach (configurable using the ``init`` parameter).
* Assignment step: assign each element from the dataset to the closest medoid.
* Update step: Identify the new medoid of each cluster.
* Repeat the assignment and update step while the medoids keep changing or
  maximum number of iterations ``max_iter`` is reached.

.. topic:: References:
