.. _kernel_approximation:

==================================================
Kernel map approximation for faster kernel methods
==================================================

.. currentmodule:: sklearn_extra.kernel_approximation

Kernel methods, which are among the most flexible and influential tools in
machine learning with applications in virtually all areas of the field, rely
on high-dimensional feature spaces in order to construct powerfull classifiers or
regressors or clustering algorithms. The main drawback of kernel methods
is their prohibitive computational complexity. Both spatial and temporal complexity
 is at least quadratic because we have to compute the whole kernel matrix.

One of the popular way to improve the computational scalability of kernel methods is
to approximate the feature map impicit behind the kernel method. In practice,
this means that we will compute a low dimensional approximation of the
the otherwise high-dimensional embedding used to define the kernel method.

:class:`Fastfood` approximates feature map of an RBF kernel by Monte Carlo approximation
of its Fourier transform.

Fastfood replaces the random matrix of Random Kitchen Sinks
(`RBFSampler <https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html#sklearn.kernel_approximation.RBFSampler>`_)
with an approximation that uses the Walsh-Hadamard transformation to gain
significant speed and storage advantages.  The computational complexity for
mapping a single example is O(n_components log d).  The space complexity is
O(n_components).

See `scikit-learn User-guide <https://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation>`_ for more general informations on kernel approximations.
