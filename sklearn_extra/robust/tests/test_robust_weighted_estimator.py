import numpy as np
import pytest

from sklearn_extra.robust import (
    RobustWeightedClassifier,
    RobustWeightedRegressor,
    RobustWeightedKMeans,
)
from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier, SGDRegressor, HuberRegressor
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_almost_equal,
)

# Test version of sklearn, in version older than v1.0 squared_loss must be used
import sklearn

if sklearn.__version__[0] == "0":
    SQ_LOSS = "squared_loss"
else:
    SQ_LOSS = "squared_error"

k_values = [None, 10]  # values of k for test robust
c_values = [None, 1e-3]  # values of c for test robust

# Classification test with outliers
rng = np.random.RandomState(42)
X_cc, y_cc = make_blobs(
    n_samples=100,
    centers=np.array([[-1, -1], [1, 1]]),
    random_state=rng,
)
for f in range(3):
    X_cc[f] = [10, 5] + rng.normal(size=2) * 0.1
    y_cc[f] = 0

classif_losses = ["log", "hinge"]
weightings = ["huber", "mom"]
multi_class = ["ovr", "ovo"]


def test_robust_estimator_max_iter():
    """Test that warning message is thrown when max_iter is reached."""
    model = RobustWeightedClassifier(max_iter=1)
    msg = "Maximum number of iteration reached before"
    with pytest.warns(UserWarning, match=msg):
        model.fit(X_cc, y_cc)


def test_robust_estimator_unsupported_loss():
    """Test that warning message is thrown when unsupported loss."""
    model = RobustWeightedClassifier(loss="invalid")
    msg = "The loss invalid is not supported. "
    with pytest.raises(ValueError, match=msg):
        model.fit(X_cc, y_cc)


def test_robust_estimator_unsupported_weighting():
    """Test that warning message is thrown when unsupported weighting."""
    model = RobustWeightedClassifier(weighting="invalid")
    msg = "No such weighting scheme"
    with pytest.raises(ValueError, match=msg):
        model.fit(X_cc, y_cc)


def test_robust_estimator_unsupported_multiclass():
    """Test that warning message is thrown when unsupported weighting."""
    model = RobustWeightedClassifier(multi_class="invalid")
    msg = "No such multiclass method implemented."
    with pytest.raises(ValueError, match=msg):
        model.fit(X_cc, y_cc)


def test_robust_estimator_input_validation_and_fit_check():
    # Invalid parameters
    msg = "max_iter must be > 0, got 0."
    with pytest.raises(ValueError, match=msg):
        RobustWeightedKMeans(max_iter=0).fit(X_cc)

    msg = "c must be > 0, got 0."
    with pytest.raises(ValueError, match=msg):
        RobustWeightedKMeans(c=0).fit(X_cc)

    msg = "burn_in must be >= 0, got -1."
    with pytest.raises(ValueError, match=msg):
        RobustWeightedClassifier(burn_in=-1).fit(X_cc, y_cc)

    msg = "eta0 must be > 0, got 0."
    with pytest.raises(ValueError, match=msg):
        RobustWeightedClassifier(burn_in=1, eta0=0).fit(X_cc, y_cc)

    msg = "k must be integer >= 0, and smaller than floor"
    with pytest.raises(ValueError, match=msg):
        RobustWeightedKMeans(k=-1).fit(X_cc)


@pytest.mark.parametrize("loss", classif_losses)
@pytest.mark.parametrize("weighting", weightings)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("c", c_values)
@pytest.mark.parametrize("multi_class", multi_class)
def test_corrupted_classif(loss, weighting, k, c, multi_class):
    clf = RobustWeightedClassifier(
        loss=loss,
        max_iter=100,
        weighting=weighting,
        k=k,
        c=c,
        multi_class=multi_class,
        random_state=rng,
    )
    clf.fit(X_cc, y_cc)
    score = clf.score(X_cc, y_cc)
    assert score > 0.8


# Classification test without outliers
rng = np.random.RandomState(42)
X_c, y_c = make_blobs(
    n_samples=100,
    centers=np.array([[-1, -1], [1, 1], [3, -1]]),
    random_state=rng,
)

# check binary throw an error
def test_robust_estimator_unsupported_loss():
    model = RobustWeightedClassifier(multi_class="binary")
    msg = "y must be binary."
    with pytest.raises(ValueError, match=msg):
        model.fit(X_c, y_c)


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
    pred1 = clf.predict(X_c)
    pred2 = clf_not_rob.predict(X_c)

    assert np.mean((pred1 > 0) == (pred2 > 0)) > 0.8
    assert clf.score(X_c, y_c) == np.mean(pred1 == y_c)


# Make binary uncorrupted dataset
X_cb, y_cb = make_blobs(
    n_samples=100, centers=np.array([[-1, -1], [1, 1]]), random_state=rng
)


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
    clf.fit(X_cb, y_cb)
    clf_not_rob.fit(X_cb, y_cb)
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

    assert len(clf.weights_) == len(X_cb)


# Check that weights_ parameter can be used as outlier score.
@pytest.mark.parametrize("weighting", weightings)
def test_classif_corrupted_weights(weighting):
    clf = RobustWeightedClassifier(
        max_iter=100,
        weighting=weighting,
        k=5,
        c=1,
        burn_in=0,
        multi_class="binary",
        random_state=rng,
    )
    clf.fit(X_cc, y_cc)
    assert np.mean(clf.weights_[:3]) < np.mean(clf.weights_[3:])


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
X_rc[0] = 10
X_rc = X_rc.reshape(-1, 1)
y_rc[0] = -1

regression_losses = [SQ_LOSS, "huber"]


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
        c=c,
        random_state=rng,
        n_iter_no_change=20,
    )
    reg.fit(X_rc, y_rc)
    assert np.abs(reg.coef_[0] - 1) < 0.1
    assert np.abs(reg.intercept_[0]) < 0.1


# Check that weights_ parameter can be used as outlier score.
@pytest.mark.parametrize("weighting", weightings)
def test_regression_corrupted_weights(weighting):
    reg = RobustWeightedRegressor(
        max_iter=100,
        weighting=weighting,
        k=5,
        c=1,
        burn_in=0,
        random_state=rng,
    )
    reg.fit(X_rc, y_rc)
    assert reg.weights_[0] < np.mean(reg.weights_[1:])


X_r = rng.uniform(-1, 1, size=[1000])
y_r = X_r + 0.1 * rng.normal(size=1000)
X_r = X_r.reshape(-1, 1)

# Check that the fit is close to SGD when in extremal parameter cases
@pytest.mark.parametrize("loss", regression_losses)
@pytest.mark.parametrize("weighting", weightings)
def test_not_robust_regression(loss, weighting):
    reg = RobustWeightedRegressor(
        loss=loss,
        max_iter=100,
        weighting=weighting,
        k=0,
        c=1e7,
        burn_in=0,
        random_state=rng,
    )
    reg_not_rob = SGDRegressor(loss=loss, random_state=rng)
    reg.fit(X_r, y_r)
    reg_not_rob.fit(X_r, y_r)
    pred1 = reg.predict(X_r)
    pred2 = reg_not_rob.predict(X_r)
    difference = [
        np.linalg.norm(pred1[i] - pred2[i]) for i in range(len(pred1))
    ]
    assert np.mean(difference) < 1
    assert_almost_equal(reg.score(X_r, y_r), r2_score(y_r, reg.predict(X_r)))


# Compare with HuberRegressor on dataset corrupted in y
X_rcy = rng.uniform(-1, 1, size=[200])
y_rcy = X_rcy + 0.1 * rng.normal(size=200)
X_rcy = X_rcy.reshape(-1, 1)
y_rcy[0] = -1


def test_vs_huber():
    reg1 = RobustWeightedRegressor(
        max_iter=100,
        weighting="huber",
        k=5,
        c=1,
        burn_in=0,
        sgd_args={"learning_rate": "adaptive"},  # test sgd_args
        random_state=rng,
    )
    reg2 = HuberRegressor()
    reg1.fit(X_rcy, y_rcy)
    reg2.fit(X_rcy, y_rcy)
    assert np.abs(reg1.coef_[0] - reg2.coef_[0]) < 1e-2


# Clustering test with outliers


rng = np.random.RandomState(42)
X_clusterc, y_clusterc = make_blobs(
    n_samples=100, centers=np.array([[-1, -1], [1, 1]]), random_state=rng
)
for f in range(3):
    X_clusterc[f] = [20, 5] + rng.normal(size=2) * 0.1
    y_clusterc[f] = 0
X_cluster, y_cluster = shuffle(X_clusterc, y_clusterc, random_state=rng)

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
    km.fit(X_clusterc)
    error = np.mean((km.predict(X_clusterc) - y_clusterc) ** 2)
    assert error < 100


# Clustering test without outliers
rng = np.random.RandomState(42)
X_cluster, y_cluster = make_blobs(
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
    clf.fit(X_cluster)
    clf_not_rob.fit(X_cluster)
    pred1 = [clf.cluster_centers_[i] for i in clf.predict(X_cluster)]
    pred2 = [
        clf_not_rob.cluster_centers_[i] for i in clf_not_rob.predict(X_cluster)
    ]
    difference = [
        np.linalg.norm(pred1[i] - pred2[i]) for i in range(len(pred1))
    ]
    assert np.mean(difference) < 1


def test_transform():
    n_clusters = 2
    km = RobustWeightedKMeans(n_clusters=n_clusters, random_state=rng)
    km.fit(X_cluster)
    X_new = km.transform(km.cluster_centers_)

    for c in range(n_clusters):
        assert X_new[c, c] == 0
        for c2 in range(n_clusters):
            if c != c2:
                assert X_new[c, c2] > 0


def test_fit_transform():
    X1 = (
        RobustWeightedKMeans(n_clusters=2, random_state=42)
        .fit(X_cluster)
        .transform(X_cluster)
    )
    X2 = RobustWeightedKMeans(n_clusters=2, random_state=42).fit_transform(
        X_cluster
    )
    assert_array_almost_equal(X1, X2)
