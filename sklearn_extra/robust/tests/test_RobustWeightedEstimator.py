import numpy as np

from sklearn_extra.robust import RobustWeightedEstimator
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, median_absolute_error
from sklearn.linear_model import SGDClassifier, SGDRegressor

# Classification tests
np.random.seed(42)
X_c, y_c = make_blobs(n_samples=100, centers=np.array([[-1, -1], [1, 1]]))
for f in range(3):
    X_c[f] = [20, 5]+np.random.normal(size=2)*0.1
    y_c[f] = 0
perm = np.random.permutation(len(X_c))
X_c = X_c[perm]
y_c = y_c[perm]

classif_losses = ['log', 'hinge', 'squared_hinge']


def test_classif():
    for loss in classif_losses:
        clf = RobustWeightedEstimator(SGDClassifier(), loss=loss,
                                      max_iter=50)
        clf.fit(X_c, y_c)
        score = accuracy_score(clf.predict(X_c), y_c)
        assert score > 0.7


# Regression tests
X_r = np.random.uniform(-1, 1, size=[200])
y_r = X_r+0.1*np.random.normal(size=200)
X_r[-1] = 10
X_r = X_r.reshape(-1, 1)
y_r[-1] = -1
perm = np.random.permutation(len(X_r))
X_r = X_r[perm]
y_r = y_r[perm]
regression_losses = ['squared_loss']


def test_regression():
    for loss in regression_losses:
        reg = RobustWeightedEstimator(SGDRegressor(), eta0=1,
                                      loss=loss, max_iter=50)
        reg.fit(X_r, y_r)
        score = median_absolute_error(reg.predict(X_r), y_r)
        assert score < 0.1
