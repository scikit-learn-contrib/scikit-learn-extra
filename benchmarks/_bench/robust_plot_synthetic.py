"""
==================================================================
Plot of accuracy and time as sample_size and num_features increase
==================================================================
We show that the increase in computation time is linear when
increasing the number of features or the sample size increases.
"""

import matplotlib.pyplot as plt
import numpy as np
from time import time

from sklearn_extra.robust import RobustWeightedEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(42)
x_label = "Number of features"
dimensions = np.linspace(50, 5000, num=8).astype(int)
sample_sizes = np.linspace(50, 5000, num=8).astype(int)
accuracies = []
times = []

# Get the accuracy and time of computations for a dataset with varying number
# of features

for d in dimensions:
    # Make an example in dimension d. Use a scale factor for the problem to be
    # easy even in high dimension.
    X, y = make_classification(
        n_samples=200, n_features=d, scale=1 / np.sqrt(2 * d), random_state=rng
    )
    stime = time()
    clf = RobustWeightedEstimator(
        SGDClassifier(loss="hinge", penalty="l1"),
        loss="hinge",
        random_state=rng,
    )
    accuracies.append(np.mean(cross_val_score(clf, X, y, cv=10)))
    times.append(time() - stime)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(dimensions, accuracies)
axs[0, 0].set_xlabel(x_label)
axs[0, 0].set_ylabel("accuracy")
axs[0, 1].plot(dimensions, times)
axs[0, 1].set_xlabel(x_label)
axs[0, 1].set_ylabel("Time to fit and predict (s)")

accuracies = []
times = []

# Get the accuracy and time of computations for a dataset with varying number
# of samples

for n in sample_sizes:
    X, y = make_classification(n_samples=n, n_features=5, random_state=rng)
    stime = time()
    clf = RobustWeightedEstimator(
        SGDClassifier(loss="hinge", penalty="l1"),
        loss="hinge",
        random_state=rng,
    )
    accuracies.append(np.mean(cross_val_score(clf, X, y, cv=10)))
    times.append(time() - stime)

axs[1, 0].plot(dimensions, accuracies)
axs[1, 0].set_xlabel(x_label)
axs[1, 0].set_ylabel("accuracy")
axs[1, 1].plot(dimensions, times)
axs[1, 1].set_xlabel(x_label)
axs[1, 1].set_ylabel("Time to fit and predict (s)")


plt.show()
