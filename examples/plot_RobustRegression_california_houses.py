# -*- coding: utf-8 -*-
"""
================================================================
A demo of Robust Regression on Real dataset "california housing"
================================================================
In this example we compare the RobustWeightedEstimator using SGDRegressor
for regression on the real dataset california housing.
WARNING: running this example can take some time (<1hour).

One of the main point of this example is the importance of taking into account
outliers in the test dataset when dealing with real datasets.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn_extra.robust import RobustWeightedEstimator
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def eval(est, X, y, X_test, y_test):
    est.fit(X, y)
    return (est.predict(X_test) - y_test) ** 2


X, y = fetch_california_housing(return_X_y=True)

# Scale the dataset with sklearn RobustScaler (important for this algorithm)
X = RobustScaler().fit_transform(X)

# Using GridSearchCV, we tune the parameters for SGDRegressor and
# RobustWeightedEstimator.
reg = SGDRegressor(
    learning_rate="adaptive", eta0=1e-6, max_iter=2000, n_iter_no_change=100
)
reg_rob = RobustWeightedEstimator(
    SGDRegressor(
        learning_rate="adaptive",
        eta0=1e-6,
        max_iter=1000,
        n_iter_no_change=100,
    ),
    weighting="huber",
    c=0.5,
    eta0=1e-6,
    max_iter=1000,
)

M = 30
res = np.zeros(shape=[4, M])

for f in range(M):
    print("\r Progress: epoch %s / %s" % (f + 1, M), end="")

    # Split in a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    cv_not_rob = eval(reg, X_train, y_train, X_test, y_test)
    cv_rob = eval(reg_rob, X_train, y_train, X_test, y_test)

    # It is preferable to use the median of the validation losses
    # because it is possible that some outliers are present in the test set.
    # We compute both for comparison.
    res[0, f] = np.mean(cv_not_rob)
    res[1, f] = np.median(cv_not_rob)
    res[2, f] = np.mean(cv_rob)
    res[3, f] = np.median(cv_rob)

fig, (axe1, axe2) = plt.subplots(1, 2)

axe1.boxplot(
    np.array([res[0, :], res[2, :]]).T,
    labels=["SGDRegressor", "RobustWeightedEstimator"],
)

axe2.boxplot(
    np.array([res[1, :], res[3, :]]).T,
    labels=["SGDRegressor", "RobustWeightedEstimator"],
)

axe1.set_title("Boxplot of the mean test squared error")
axe2.set_title("Boxplot of the median test squared error")

plt.show()
