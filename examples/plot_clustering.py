# -*- coding: utf-8 -*-
"""
===================================================================
A demo of several clustering algorithms on a corrupted dataset
===================================================================
In this example we exhibit the results of various 
scikit-learn and scikit-learn-extra clustering algorithms on
a dataset with outliers.
KMedoids is the most stable and efficient 
algorithm for this application (change the seed to
see different behavior for SpectralClustering and 
the robust kmeans.
The mean-shift algorithm, once correctly 
parameterized, detects the outliers as a class of 
its own.
"""
print(__doc__)

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, mixture
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from itertools import cycle, islice

from sklearn_extra.robust import RobustWeightedEstimator
from sklearn_extra.cluster import KMedoids

rng = np.random.RandomState(42)

centers = [[1, 1], [-1, -1], [1, -1]]

kmeans = KMeans(n_clusters=3, random_state=rng)
kmedoid = KMedoids(n_clusters=3, random_state=rng)


def kmeans_loss(X, pred):
    return np.array(
        [
            np.linalg.norm(X[pred[i]] - np.mean(X[pred == pred[i]])) ** 2
            for i in range(len(X))
        ]
    )


two_means = cluster.MiniBatchKMeans(n_clusters=3)
spectral = cluster.SpectralClustering(
    n_clusters=3, eigen_solver="arpack", affinity="nearest_neighbors"
)
dbscan = cluster.DBSCAN()
optics = cluster.OPTICS(min_samples=20, xi=0.1, min_cluster_size=0.2)
affinity_propagation = cluster.AffinityPropagation(
    damping=0.75, preference=-220
)
birch = cluster.Birch(n_clusters=3)
gmm = mixture.GaussianMixture(n_components=3, covariance_type="full")


for N in [300, 3000]:
    # Construct the dataset
    n_clusters = len(centers)
    X, labels_true = make_blobs(
        n_samples=N, centers=centers, cluster_std=0.4, random_state=rng
    )

    # Change the first 1% entries to outliers
    for f in range(int(N / 100)):
        X[f] = [20, 3] + rng.normal(size=2) * 0.1
    # Shuffle the data so that we don't know where the outlier is.
    X = shuffle(X, random_state=rng)

    # Define two other clustering algorithms
    kmeans_rob = RobustWeightedEstimator(
        MiniBatchKMeans(3, batch_size=len(X), init="random", random_state=rng),
        # in theory, init=kmeans++ is very non-robust
        burn_in=0,
        eta0=0.1,
        weighting="mom",
        loss=kmeans_loss,
        max_iter=100,
        k=int(N / 100),
        random_state=rng,
    )
    bandwidth = cluster.estimate_bandwidth(X, 0.2)

    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

    clustering_algorithms = (
        ("MiniBatchKMeans", two_means),
        ("AffinityPropagation", affinity_propagation),
        ("MeanShift", ms),
        ("SpectralClustering", spectral),
        ("DBSCAN", dbscan),
        ("OPTICS", optics),
        ("Birch", birch),
        ("GaussianMixture", gmm),
        ("K-Medoid", kmedoid),
        ("Robust K-Means", kmeans_rob),
    )

    plot_num = 1
    fig = plt.figure(figsize=(9 * 2 + 3, 5))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.85, wspace=0.05, hspace=0.18
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(2, int(len(clustering_algorithms) / 2), plot_num)
        plt.title(name, size=18)

        plt.scatter(X[:, 0], X[:, 1], s=10, c=y_pred)

        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        plt.suptitle(
            f"Dataset with {N} samples, {N // 100} outliers.", size=20,
        )
        plot_num += 1

    plt.show()
