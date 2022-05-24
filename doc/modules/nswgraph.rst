.. _neighbors:

============================================================
Neighbors search with NSW graphs
============================================================
.. _nswgraph:
.. currentmodule:: sklearn_extra.neighbors


A navigable small-world graph is a type of mathematical graph in which most nodes are not neighbors of one another,
but the neighbors of any given node are likely to be neighbors of each other and most nodes can be reached
from every other node by some small number of hops or steps [1]_.
The number of steps regulates by the property which must be satisfied by the navigable small-world graph:

* The minimum number of edges that must be traversed to travel between two randomly chosen nodes grows proportionally to the logarithm of the number of nodes in the network [2]_.

:class:`NSWGraph` is the approximate nearest neighbor algorithm based on navigable small world graphs.
The algorithm tends to be more optimal in case of high-dimensional data [3]_ in comparison with
existing Scikit-Learn approximate nearest neighbor algorithms based on :class:`KDTree <sklearn.neighbors.KDTree>`
and :class:`BallTree <sklearn.neighbors.BallTree>`.

See `Scikit-Learn User-guide <https://scikit-learn.org/stable/modules/neighbors.html>`_
for more general information on Nearest Neighbors search.


.. topic:: References:

    .. [1] Porter, Mason A. “Small-World Network.” Scholarpedia.
           Available at: http://www.scholarpedia.org/article/Small-world_network.

    .. [2] Kleinberg, Jon. "The small-world phenomenon and decentralized search." SiAM News 37.3 (2004): 1-2.

    .. [3] Malkov, Y., Ponomarenko, A., Logvinov, A., & Krylov, V. (2014).
           Approximate nearest neighbor algorithm based on navigable small world graphs.
           Information Systems, 45, 61-68.
