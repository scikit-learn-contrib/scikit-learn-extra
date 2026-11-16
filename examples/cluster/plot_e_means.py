"""
===================================
e-means versus Lloyd's k-means
===================================

Compare the training error reached by :class:`~sklearn_extra.cluster.EMeans`
and by Lloyd's algorithm (:class:`sklearn.cluster.KMeans`) as a function of
the number of passes over the data, on random high-dimensional data with
many clusters.

One Lloyd iteration and one e-means epoch both assign every sample to its
nearest center once, so they cost about the same.
"""

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn_extra.cluster import EMeans

X, _ = make_blobs(
    n_samples=20000,
    centers=200,
    n_features=32,
    cluster_std=4.0,
    random_state=0,
)
n_clusters = 300

passes = [1, 2, 3, 5, 8, 12, 25]
lloyd, emeans = [], []
for p in passes:
    km = KMeans(
        n_clusters=n_clusters,
        init="random",
        n_init=1,
        max_iter=p,
        tol=0,
        algorithm="lloyd",
        random_state=0,
    ).fit(X)
    lloyd.append(km.inertia_)
    em = EMeans(n_clusters=n_clusters, n_epochs=p, random_state=0).fit(X)
    emeans.append(em.inertia_)

plt.plot(passes, lloyd, "o-", label="Lloyd (KMeans)")
plt.plot(passes, emeans, "s-", label="e-means (EMeans)")
plt.axhline(
    lloyd[-1], color="gray", linestyle=":", label="Lloyd, 25 iterations"
)
plt.xscale("log")
plt.xlabel("passes over the data")
plt.ylabel("training inertia")
plt.legend()
plt.title("e-means versus Lloyd's k-means")
plt.show()
