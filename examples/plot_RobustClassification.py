# -*- coding: utf-8 -*-
"""
=============================================================
A demo of Robust Classification on Simulated corrupted dataset
=============================================================
In this example we compare the RobustWeightedEstimator using SGDClassifier
for classification with the vanilla SGDClassifier with various losses.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn_extra.robust import RobustWeightedEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs

np.random.seed(42)

# Sample two Gaussian blobs
X, y = make_blobs(n_samples=100, centers=np.array([[-1, -1], [1, 1]]))

# Change the first 5 entries to outliers
for f in range(3):
    X[f] = [20, 3] + np.random.normal(size=2) * 0.1
    y[f] = 0

# Shuffle the data so that we don't know where the outlier is.
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

estimators = [
    ("SGDClassifier, Hinge loss", SGDClassifier(loss="hinge")),
    ("SGDClassifier, log loss", SGDClassifier(loss="log")),
    (
        "SGDClassifier, modified_huber loss",
        SGDClassifier(loss="modified_huber"),
    ),
    (
        "RobustWeightedEstimator",
        RobustWeightedEstimator(
            base_estimator=SGDClassifier(), loss="log", weighting="huber"
        ),
    ),
]


# Helping function to represent estimators
def plot_classif(clf, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y)


for i, (name, estimator) in enumerate(estimators):
    plt.subplot(2, 2, i + 1)
    estimator.fit(X, y)
    plot_classif(estimator, X, y)
    plt.title(name)


plt.suptitle(
    "Scatter plot of training set and representation of"
    " estimation functions"
)
plt.show()
