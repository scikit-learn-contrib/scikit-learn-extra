import numpy as np
import pytest

from sklearn_extra.robust import RobustWeightedEstimator
from sklearn.datasets import make_blobs
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.utils import shuffle


# Check if estimator adheres to scikit-learn conventions.

# Classification test with outliers
rng = np.random.RandomState(42)
X_cc, y_cc = make_blobs(
    n_samples=100, centers=np.array([[-1, -1], [1, 1]]), random_state=rng
)
for f in range(3):
    X_cc[f] = [20, 5] + rng.normal(size=2) * 0.1
    y_cc[f] = 0
X_cc, y_cc = shuffle(X_cc, y_cc, random_state=rng)

classif_losses = ["log", "hinge"]
weightings = ["huber", "mom"]


@pytest.mark.parametrize("loss", classif_losses)
@pytest.mark.parametrize("weighting", weightings)
def test_corrupted_classif(loss, weighting):
    clf = RobustWeightedEstimator(
        SGDClassifier(),
        loss=loss,
        max_iter=50,
        weighting=weighting,
        k=5,
        c=None,
        random_state=rng,
    )
    clf.fit(X_cc, y_cc)
    score = clf.score(X_cc, y_cc)
    assert score > 0.75


# Classification test without outliers
rng = np.random.RandomState(42)
X_c, y_c = make_blobs(
    n_samples=100, centers=np.array([[-1, -1], [1, 1]]), random_state=rng
)

# Check that the fit is close to SGD when in extremal parameter cases
@pytest.mark.parametrize("loss", classif_losses)
@pytest.mark.parametrize("weighting", weightings)
def test_not_robust_classif(loss, weighting):
    clf = RobustWeightedEstimator(
        SGDClassifier(),
        loss=loss,
        max_iter=100,
        weighting=weighting,
        k=0,
        c=1e7,
        burn_in=0,
        random_state=rng,
    )
    clf_not_rob = SGDClassifier(loss=loss, random_state=rng)
    clf.fit(X_c, y_c)
    clf_not_rob.fit(X_c, y_c)
    pred1 = clf.base_estimator_.decision_function(X_c)
    pred2 = clf_not_rob.decision_function(X_c)

    assert (
        np.linalg.norm(pred1 - pred2) / np.linalg.norm(pred2)
        - np.linalg.norm(pred1 - y_c) / np.linalg.norm(y_c)
        < 0.1
    )


# Case "log" loss, test predict_proba
@pytest.mark.parametrize("weighting", weightings)
def test_predict_proba(weighting):
    clf = RobustWeightedEstimator(
        SGDClassifier(loss="log"),
        loss="log",
        max_iter=100,
        weighting=weighting,
        k=0,
        c=1e7,
        burn_in=0,
        random_state=rng,
    )
    clf_not_rob = SGDClassifier(loss="log", random_state=rng)
    clf.fit(X_c, y_c)
    clf_not_rob.fit(X_c, y_c)
    pred1 = clf.base_estimator_.predict_proba(X_c)[:, 1]
    pred2 = clf_not_rob.predict_proba(X_c)[:, 1]

    assert (
        np.linalg.norm(pred1 - pred2) / np.linalg.norm(pred2)
        - np.linalg.norm(pred1 - y_c) / np.linalg.norm(y_c)
        < 0.1
    )


# Regression test with outliers
X_rc = rng.uniform(-1, 1, size=[200])
y_rc = X_rc + 0.1 * rng.normal(size=200)
X_rc[-1] = 10
X_rc = X_rc.reshape(-1, 1)
y_rc[-1] = -1
X_rc, y_rc = shuffle(X_rc, y_rc, random_state=rng)
regression_losses = ["squared_loss"]


@pytest.mark.parametrize("loss", regression_losses)
@pytest.mark.parametrize("weighting", weightings)
def test_corrupted_regression(loss, weighting):
    reg = RobustWeightedEstimator(
        SGDRegressor(),
        loss=loss,
        max_iter=50,
        weighting=weighting,
        k=4,
        c=None,
        random_state=rng,
    )
    reg.fit(X_rc, y_rc)
    score = median_absolute_error(reg.predict(X_rc), y_rc)
    assert score < 0.2


X_r = rng.uniform(-1, 1, size=[200])
y_r = X_r + 0.1 * rng.normal(size=200)
X_r = X_r.reshape(-1, 1)

# Check that the fit is close to SGD when in extremal parameter cases
@pytest.mark.parametrize("loss", regression_losses)
@pytest.mark.parametrize("weighting", weightings)
def test_not_robust_regression(loss, weighting):
    clf = RobustWeightedEstimator(
        SGDRegressor(),
        loss=loss,
        max_iter=100,
        weighting=weighting,
        k=0,
        c=1e7,
        burn_in=0,
        random_state=rng,
    )
    clf_not_rob = SGDRegressor(loss=loss, random_state=rng)
    clf.fit(X_r, y_r)
    clf_not_rob.fit(X_r, y_r)
    pred1 = clf.predict(X_r)
    pred2 = clf_not_rob.predict(X_r)

    assert np.linalg.norm(pred1 - pred2) / np.linalg.norm(
        pred2
    ) < np.linalg.norm(pred1 - y_r) / np.linalg.norm(y_r)
