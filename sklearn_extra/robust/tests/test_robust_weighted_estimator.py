import numpy as np
import pytest

from sklearn_extra.robust import (
    RobustWeightedClassifier,
    RobustWeightedRegressor,
    RobustWeightedKMeans,
)
from sklearn.datasets import make_blobs
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

k_values = [None, 5]  # values of k for test robust
c_values = [None, 1e-4]  # values of c for test robust

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
multi_class = ["ovr", "ovo"]


@pytest.mark.parametrize("loss", classif_losses)
@pytest.mark.parametrize("weighting", weightings)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("c", c_values)
@pytest.mark.parametrize("multi_class", multi_class)
def test_corrupted_classif(loss, weighting, k, c, multi_class):
    clf = RobustWeightedClassifier(
        loss=loss,
        max_iter=50,
        weighting=weighting,
        k=5,
        c=None,
        multi_class=multi_class,
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
@pytest.mark.parametrize("multi_class", multi_class)
def test_not_robust_classif(loss, weighting, multi_class):
    clf = RobustWeightedClassifier(
        loss=loss,
        max_iter=100,
        weighting=weighting,
        k=0,
        c=1e7,
        burn_in=0,
        multi_class=multi_class,
        random_state=rng,
    )
    clf_not_rob = SGDClassifier(loss=loss, random_state=rng)
    clf.fit(X_c, y_c)
    clf_not_rob.fit(X_c, y_c)
    pred1 = clf.base_estimator_.decision_function(X_c)
    pred2 = clf_not_rob.decision_function(X_c)

    assert np.mean((pred1 > 0) == (pred2 > 0)) > 0.8


@pytest.mark.parametrize("weighting", weightings)
def test_classif_binary(weighting):
    clf = RobustWeightedClassifier(
        max_iter=100,
        weighting=weighting,
        k=0,
        c=1e7,
        burn_in=0,
        multi_class="binary",
        random_state=rng,
    )
    clf_not_rob = SGDClassifier(loss="log", random_state=rng)
    clf.fit(X_c, y_c)
    clf_not_rob.fit(X_c, y_c)
    norm_coef1 = np.linalg.norm(np.hstack([clf.coef_.ravel(), clf.intercept_]))
    norm_coef2 = np.linalg.norm(
        np.hstack([clf_not_rob.coef_.ravel(), clf_not_rob.intercept_])
    )
    coef1 = clf.coef_ / norm_coef1
    coef2 = clf_not_rob.coef_ / norm_coef2

    intercept1 = clf.intercept_ / norm_coef1
    intercept2 = clf_not_rob.intercept_ / norm_coef2

    assert np.linalg.norm(coef1 - coef2) < 0.5
    assert np.linalg.norm(intercept1 - intercept2) < 0.5

    assert len(clf.weights_) == len(X_c)


# Case "log" loss, test predict_proba
@pytest.mark.parametrize("weighting", weightings)
def test_predict_proba(weighting):
    clf = RobustWeightedClassifier(
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

    assert np.mean((pred1 > 1 / 2) == (pred2 > 1 / 2)) > 0.8


# check that classifier with another loss than log raises an error
def test_robust_no_proba():
    est = RobustWeightedClassifier(loss="hinge").fit(X_c, y_c)
    msg = "Probability estimates are not available for loss='hinge'"
    with pytest.raises(AttributeError, match=msg):
        est.predict_proba(X_c)


# Regression test with outliers
X_rc = rng.uniform(-1, 1, size=[200])
y_rc = X_rc + 0.1 * rng.normal(size=200)
X_rc[-1] = 10
X_rc = X_rc.reshape(-1, 1)
y_rc[-1] = -1
X_rc, y_rc = shuffle(X_rc, y_rc, random_state=rng)
regression_losses = ["squared_loss", "huber"]


@pytest.mark.parametrize("loss", regression_losses)
@pytest.mark.parametrize("weighting", weightings)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("c", c_values)
def test_corrupted_regression(loss, weighting, k, c):
    reg = RobustWeightedRegressor(
        loss=loss,
        max_iter=50,
        weighting=weighting,
        k=k,
        c=None,
        random_state=rng,
    )
    reg.fit(X_rc, y_rc)
    score = median_absolute_error(reg.predict(X_rc), y_rc)
    assert score < 0.2


X_r = rng.uniform(-1, 1, size=[1000])
y_r = X_r + 0.1 * rng.normal(size=1000)
X_r = X_r.reshape(-1, 1)

# Check that the fit is close to SGD when in extremal parameter cases
@pytest.mark.parametrize("loss", regression_losses)
@pytest.mark.parametrize("weighting", weightings)
def test_not_robust_regression(loss, weighting):
    clf = RobustWeightedRegressor(
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
    difference = [
        np.linalg.norm(pred1[i] - pred2[i]) for i in range(len(pred1))
    ]
    assert np.mean(difference) < 1e-1


# Clustering test with outliers


rng = np.random.RandomState(42)
X_cc, y_cc = make_blobs(
    n_samples=100, centers=np.array([[-1, -1], [1, 1]]), random_state=rng
)
for f in range(3):
    X_cc[f] = [20, 5] + rng.normal(size=2) * 0.1
    y_cc[f] = 0
X_cc, y_cc = shuffle(X_cc, y_cc, random_state=rng)

weightings = ["huber", "mom"]


@pytest.mark.parametrize("weighting", weightings)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("c", c_values)
def test_corrupted_cluster(weighting, k, c):
    km = RobustWeightedKMeans(
        n_clusters=2,
        max_iter=50,
        weighting=weighting,
        k=5,
        c=None,
        random_state=rng,
    )
    km.fit(X_cc)
    error = np.mean((km.predict(X_cc) - y_cc) ** 2)
    assert error < 100


# Clustering test without outliers
rng = np.random.RandomState(42)
X_c, y_c = make_blobs(
    n_samples=100, centers=np.array([[-1, -1], [1, 1]]), random_state=rng
)

# Check that the fit is close to KMeans when in extremal parameter cases
@pytest.mark.parametrize("weighting", weightings)
def test_not_robust_cluster(weighting):
    clf = RobustWeightedKMeans(
        n_clusters=2,
        max_iter=100,
        weighting=weighting,
        k=0,
        c=1e7,
        random_state=rng,
    )
    clf_not_rob = KMeans(2, random_state=rng)
    clf.fit(X_c)
    clf_not_rob.fit(X_c)
    pred1 = [clf.cluster_centers_[i] for i in clf.predict(X_c)]
    pred2 = [clf_not_rob.cluster_centers_[i] for i in clf_not_rob.predict(X_c)]
    difference = [
        np.linalg.norm(pred1[i] - pred2[i]) for i in range(len(pred1))
    ]
    assert np.mean(difference) < 1
