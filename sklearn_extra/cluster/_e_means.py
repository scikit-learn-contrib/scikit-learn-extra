"""e-means: mini-batch k-means with exponential moving averages.

Port of the reference implementation in Faiss
(``faiss/contrib/e_means``, MIT license, Copyright (c) Meta Platforms, Inc.
and affiliates) of

    S. Bessa, P.-E. Mazaré, G. Szilvasy, H. Touvron, D. Picard, M. Douze and
    H. Jégou, "e-means: outperforming k-means (in one-tenth the epochs)",
    arXiv:2611.09501, 2026.

Algorithm 1, equations and constants below refer to that paper. The fitted
batch-size and step laws are the coefficients shipped with Faiss.
"""

# Parts of this file are ported from Faiss (faiss/contrib/e_means), which is
# distributed under the following license:
#
#   MIT License
#
#   Copyright (c) Meta Platforms, Inc. and affiliates.
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
# License: BSD 3 clause

import math
import numbers
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_info, threadpool_limits

from .._compat import validate_data

# Epsilon in Equation 5: the cosine schedule's minimum step / initial step.
_ALPHA_MIN_FACTOR = 0.1
# Number of batch updates between revival checks (Algorithm 1, line 9).
_CHECK_UNUSED_EVERY = 5
# Initial EMA mass N_j (Algorithm 1, line 1). Faiss uses 2**-16.
_INIT_USAGE = 2.0**-16
# Relative coordinate perturbation for donor splits (Appendix J, Eq. 8).
_SPLIT_EPSILON = 1e-4
# Revival threshold tau: revive N_j < tau * mean(N).
_UNUSED_THRESHOLD = 0.1
# Revival is off by default at or below this dimension.
_NO_REVIVAL_MAX_D = 2
# batch_law="auto" uses the quality law at or below this dimension.
_AUTO_QUALITY_MAX_D = 16
# Batch law B = C_B * k**b_k * n**b_n * T**b_T (Equation 7).
_BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}
# Step law alpha0 = C * k**a_k * B**a_B * n**a_n * T**a_T (Equation 6).
_ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)
# Assignment works on (chunk, k) blocks of about this many bytes, so that a
# block stays in a core's cache between the matrix product and the argmin.
_CHUNK_BYTES = 1 << 20
# Rows per block: enough to amortize a pass over the centers, at most this.
_CHUNK_MIN_ROWS = 64
_CHUNK_MAX_ROWS = 4096
# Cap on the total size of the blocks of all assignment threads.
_MAX_SCRATCH_BYTES = 512 << 20
# A thread is worth waking for at least this many multiply-adds (about a
# millisecond) and this many rows, and a pool for at least this many threads.
_MIN_TASK_WORK = 1 << 23
_MIN_TASK_ROWS = 32
_MIN_POOL_THREADS = 4
# Per-thread (chunk, k) block, reused between calls by the pool's threads.
_scratch = threading.local()


def _round_power_of_two(value):
    """Round a positive value to the nearest power of two."""
    if value <= 1:
        return 1
    low = 2 ** int(math.floor(math.log2(value)))
    high = 2 * low
    return low if value / low < high / value else high


def _step_law(n, k, n_epochs, batch_size):
    """Initial EMA step for a batch size, from the fitted step law."""
    c, a_k, a_b, a_n, a_t = _ALPHA_LAW
    alpha0 = c * k**a_k * batch_size**a_b * n**a_n * n_epochs**a_t
    return min(max(alpha0, 1e-4), 1.0)


def _resolve_batch_law(batch_law, d):
    if batch_law != "auto":
        return batch_law
    return "quality" if d <= _AUTO_QUALITY_MAX_D else "throughput"


def _resolve_recipe(n, k, n_epochs, batch_law, batch_size, alpha0):
    """Return the batch size B and the initial EMA step alpha0."""
    if batch_size is None:
        c, b_k, b_n, b_t = _BATCH_LAWS[batch_law]
        batch_size = _round_power_of_two(
            c * k**b_k * n**b_n * n_epochs**b_t
        )
        if batch_law == "throughput":
            batch_size = min(batch_size, _round_power_of_two(n / 2))
        else:
            batch_size = min(batch_size, n // 2)
        batch_size = max(1, min(batch_size, n))
    else:
        batch_size = min(int(batch_size), n)
    if alpha0 is None:
        alpha0 = _step_law(n, k, n_epochs, batch_size)
    return batch_size, float(alpha0)


def _epoch_rows(n, n_epochs):
    """Rows per epoch: whole passes, a fractional prefix, then a full pass.

    The last epoch is always a full pass, since it provides the closing
    means (Algorithm 1, line 12).
    """
    whole = math.floor(n_epochs)
    remainder = int(round((n_epochs - whole) * n))
    if whole == 0:
        return [remainder]
    return [n] * (whole - 1) + ([remainder] if remainder else []) + [n]


def _scheduled_alpha(alpha0, step, total_steps):
    """Cosine-anneal the EMA step to one tenth of its initial value (Eq. 5)."""
    fraction = min(step, total_steps) / max(1, total_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return alpha0 * (_ALPHA_MIN_FACTOR + (1.0 - _ALPHA_MIN_FACTOR) * cosine)


def _label_sums(labels, rows, k):
    """Sum of the rows of each label, as ``np.add.at`` on zeros.

    Uses the sparse product ``one-hot @ rows`` in CSC form, which adds the
    rows of a label in row order and is much faster than ``np.add.at``.
    """
    n = len(labels)
    onehot = sp.csc_matrix(
        (np.ones(n, dtype=rows.dtype), labels, np.arange(n + 1)),
        shape=(k, n),
    )
    return onehot @ rows


def _chunk_size(n_clusters, itemsize):
    """Rows per (chunk, n_clusters) assignment block."""
    row_bytes = n_clusters * itemsize
    rows = min(
        _CHUNK_MAX_ROWS, max(_CHUNK_MIN_ROWS, _CHUNK_BYTES // row_bytes)
    )
    return max(1, min(rows, _MAX_SCRATCH_BYTES // row_bytes))


def _blas_num_threads():
    """Threads BLAS uses now; follows ``threadpool_limits`` and the env."""
    return max(
        (
            module["num_threads"]
            for module in threadpool_info()
            if module["user_api"] == "blas"
        ),
        default=1,
    )


def _min_task_rows(n_clusters, n_features):
    """Fewest rows worth handing to a thread of the assignment pool."""
    return max(_MIN_TASK_ROWS, -(-_MIN_TASK_WORK // (n_clusters * n_features)))


class _AssignmentPool(ThreadPoolExecutor):
    """Thread pool that records its number of threads."""

    def __init__(self, n_threads):
        super().__init__(n_threads)
        self.n_threads = n_threads


@contextmanager
def _assignment_pool(n_rows, n_clusters, n_features, chunk_size, itemsize):
    """Yield a thread pool for ``_nearest_centers``, or None.

    A multi-threaded matrix product over a large block leaves the argmin
    single-threaded and the block out of cache. Instead, the pool's threads
    each run single-threaded BLAS and the argmin on their own small blocks,
    so BLAS is limited to one thread while the pool is open. No pool is made
    when ``n_rows`` rows are too little work to share between a few threads
    or BLAS is limited to one thread.
    """
    block_bytes = chunk_size * n_clusters * itemsize
    n_threads = min(
        n_rows // _min_task_rows(n_clusters, n_features),
        max(1, _MAX_SCRATCH_BYTES // block_bytes),
    )
    # Ask for the BLAS threads (about a millisecond) only if it matters.
    if n_threads > 1:
        blas_threads = _blas_num_threads()
        n_threads = min(n_threads, blas_threads)
    if n_threads <= 1 or n_threads < min(blas_threads, _MIN_POOL_THREADS):
        yield None
        return
    with _AssignmentPool(n_threads) as pool:
        with threadpool_limits(limits=1, user_api="blas"):
            yield pool


def _take_rows(X, indices, out, pool):
    """``out[i] = X[indices[i]]``, in the pool's threads if there is one."""
    n = len(indices)
    if pool is None or n < 2 * 1024:
        np.take(X, indices, axis=0, out=out, mode="clip")
        return

    def take(sl):
        np.take(X, indices[sl], axis=0, out=out[sl], mode="clip")

    list(pool.map(take, gen_batches(n, max(1024, -(-n // pool.n_threads)))))


def _separate_splits(donor, child):
    """Make identical donor and child centroids differ in one coordinate."""
    collapsed = (donor == child).all(axis=1)
    value = child[:, 0]
    with np.errstate(over="ignore"):
        neighbor = np.nextafter(value, value.dtype.type(np.inf))
    neighbor = np.where(
        np.isfinite(neighbor),
        neighbor,
        np.nextafter(value, value.dtype.type(-np.inf)),
    )
    child[:, 0] = np.where(collapsed, neighbor, value)


def _nearest_centers(X, centers, x_squared_norms, chunk_size, pool=None):
    """Squared distance to, and index of, the nearest center of each row.

    ``pool`` is a thread pool from ``_assignment_pool``, or None to run in
    the calling thread. With a pool, the blocks have at most ``chunk_size``
    rows and are small enough to give every thread one.
    """
    n = X.shape[0]
    k = centers.shape[0]
    if pool is not None:
        chunk_size = min(
            chunk_size,
            max(_min_task_rows(k, X.shape[1]), -(-n // pool.n_threads)),
        )
    labels = np.empty(n, dtype=np.int64)
    distances = np.empty(n, dtype=X.dtype)
    c_squared_norms = row_norms(centers, squared=True)
    # Scaling by -2 is exact, so this equals -2 * (X @ centers.T) bitwise.
    minus_two_centers_t = (-2 * centers).T
    shape = (min(chunk_size, n), k)

    def assign(sl, block):
        # ||x||^2 is constant per row, so the argmin needs only the rest.
        # (chunk, k) layout keeps the argmin over contiguous rows, and
        # writing into a reused block avoids faulting in fresh pages.
        partial = block[: sl.stop - sl.start]
        np.matmul(X[sl], minus_two_centers_t, out=partial)
        partial += c_squared_norms
        best = partial.argmin(axis=1)
        labels[sl] = best
        np.add(
            partial[np.arange(len(best)), best],
            x_squared_norms[sl],
            out=distances[sl],
        )

    def assign_in_pool(sl):
        block = getattr(_scratch, "block", None)
        if block is None or block.shape != shape or block.dtype != X.dtype:
            block = _scratch.block = np.empty(shape, dtype=X.dtype)
        assign(sl, block)

    if pool is None or n <= chunk_size:
        block = np.empty(shape, dtype=X.dtype)
        for sl in gen_batches(n, chunk_size):
            assign(sl, block)
    else:
        list(pool.map(assign_in_pool, gen_batches(n, chunk_size)))
    np.maximum(distances, 0, out=distances)
    return distances, labels


def _choose_splits(usage, threshold_factor, rng):
    """Pick dead clusters to revive and donors proportional to their mass."""
    k = len(usage)
    total = usage.sum(dtype=usage.dtype)
    threshold = usage.dtype.type(total * threshold_factor) / usage.dtype.type(
        k
    )
    unused = usage < threshold
    donors = (~unused) & (usage > 0)
    nsplit = min(int(unused.sum()), int(donors.sum()))
    if nsplit == 0:
        return None, None
    # Gumbel-top-k: donors sampled without replacement by mass.
    uniform = np.maximum(rng.random_sample(k), np.finfo("float64").tiny)
    keys = np.full(k, -np.inf)
    keys[donors] = np.log(usage[donors]) - np.log(-np.log(uniform[donors]))
    source = np.argsort(keys)[-nsplit:][::-1]
    target = np.flatnonzero(unused)[:nsplit]
    return source, target


def _apply_splits(centers, usage, source, target):
    d = centers.shape[1]
    split = centers[source]
    sign = np.ones(d, dtype=centers.dtype)
    sign[1::2] = -1.0
    offset = (_SPLIT_EPSILON * sign).astype(centers.dtype)
    donor = split * (1.0 - offset)
    child = split * (1.0 + offset)
    _separate_splits(donor, child)
    half_usage = usage[source] / usage.dtype.type(2)
    usage[source] = half_usage
    centers[source] = donor
    usage[target] = half_usage
    centers[target] = child
    return len(source)


def _emeans_core(
    X,
    init_centers,
    *,
    batch_size,
    alpha0,
    epoch_rows,
    epoch_orders,
    unused_threshold,
    split_rng,
    chunk_size,
    x_squared_norms,
    pool=None,
):
    """Run Algorithm 1 from given initial centers.

    ``epoch_orders(epoch)`` returns the row permutation used for that epoch;
    the first ``epoch_rows[epoch]`` rows of it are visited. Returns the
    closing centers, the last-epoch labels, the number of batch updates,
    the number of revived clusters and the moving objective of each epoch.
    ``pool`` is a thread pool from ``_assignment_pool``, or None.
    """
    dtype = X.dtype
    k, d = init_centers.shape
    centers = np.array(init_centers, dtype=dtype, order="C", copy=True)
    usage = np.full(k, _INIT_USAGE, dtype=dtype)
    epoch_steps = [(r + batch_size - 1) // batch_size for r in epoch_rows]
    total_steps = sum(epoch_steps)
    last_epoch = len(epoch_rows) - 1
    labels_last = np.full(X.shape[0], -1, dtype=np.int64)
    batch_rows = np.empty((min(batch_size, X.shape[0]), d), dtype=dtype)
    step = 0
    n_splits = 0
    objectives = []

    for epoch, rows in enumerate(epoch_rows):
        order = epoch_orders(epoch)
        pass_counts = np.zeros(k, dtype=np.float64)
        pass_sums = (
            np.zeros((k, d), dtype=dtype) if epoch == last_epoch else None
        )
        objective = 0.0
        for begin in range(0, rows, batch_size):
            batch = order[begin : min(begin + batch_size, rows)]
            # Lines 6-7: assign the batch against one fixed snapshot.
            rows_x = batch_rows[: len(batch)]
            _take_rows(X, batch, rows_x, pool)
            dist, lab = _nearest_centers(
                rows_x, centers, x_squared_norms[batch], chunk_size, pool
            )
            if epoch == last_epoch:
                labels_last[batch] = lab
            objective += dist.sum(dtype=np.float64)
            batch_counts = np.bincount(lab, minlength=k).astype(dtype)
            batch_sums = _label_sums(lab, rows_x, k)

            # Line 8: EMA of counts and sums, stored as mass and position.
            alpha = _scheduled_alpha(alpha0, step, total_steps)
            decay = 1.0 - alpha
            pass_counts += batch_counts
            if pass_sums is not None:
                pass_sums += batch_sums
            touched = batch_counts > 0
            if alpha == 1:
                retained = np.zeros(int(touched.sum()), dtype=dtype)
            else:
                usage *= dtype.type(decay)
                retained = usage[touched]
            counts_t = batch_counts[touched]
            mass = retained + dtype.type(alpha) * counts_t
            mean = batch_sums[touched] / counts_t[:, None]
            weight = dtype.type(alpha) * counts_t / mass
            position = centers[touched] + weight[:, None] * (
                mean - centers[touched]
            )
            position[weight == 1] = mean[weight == 1]
            centers[touched] = position
            usage[touched] = mass

            # Line 9: revive dead clusters by splitting heavy ones. The RNG
            # is drawn only when a split happens, as in the reference.
            if unused_threshold > 0 and (step + 1) % _CHECK_UNUSED_EVERY == 0:
                source, target = _choose_splits(
                    usage, unused_threshold, split_rng
                )
                if source is not None:
                    n_splits += _apply_splits(centers, usage, source, target)
            step += 1
        objectives.append(objective)

    # Line 12: closing means of the last epoch's (moving) assignments.
    nonempty = pass_counts > 0
    centers[nonempty] = pass_sums[nonempty] / pass_counts[
        nonempty, None
    ].astype(dtype)
    return centers, labels_last, step, n_splits, objectives


class EMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    """e-means clustering: mini-batch k-means with exponential moving averages.

    e-means minimizes the k-means objective with mini-batch updates. Each
    cluster keeps an exponential moving average (EMA) of the counts and sums
    of the points assigned to it, and its center is their ratio. The EMA step
    follows a cosine schedule, and both the batch size and the initial step
    are set by scaling laws fitted on large embedding datasets, so there are
    no hyperparameters to tune. Clusters that stop receiving points are
    revived by splitting heavily used ones. The final centers are the means
    of the points assigned to each cluster during the last epoch.

    On high-dimensional data, e-means typically reaches the training error
    of 25 Lloyd iterations in 2 to 3 epochs.

    Read more in the :ref:`User Guide <e_means>`.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.

    n_epochs : float, default=5
        Number of passes over the data. Fractional values run a partial
        pass before the last one, which is always a full pass.

    batch_size : int or None, default=None
        Mini-batch size. If None, it is set by the batch law selected by
        ``batch_law``.

    alpha0 : float or None, default=None
        Initial EMA step (one minus the decay), in (0, 1]. If None, it is
        set from the batch size by the step law.

    batch_law : {"auto", "throughput", "quality"}, default="auto"
        Scaling law used to set ``batch_size`` when it is None.
        ``"quality"`` gives batches about 8 times smaller than
        ``"throughput"``. ``"auto"`` uses ``"quality"`` when the number of
        features is at most 16 and ``"throughput"`` otherwise.

    unused_threshold : float or None, default=None
        Clusters whose EMA mass falls below this fraction of the mean mass
        are revived every 5 batches by splitting a cluster sampled in
        proportion to its mass. 0 disables revival. If None, 0.1 is used,
        or 0 when the number of features is at most 2.

    init : {"random", "k-means++"} or array of shape \
            (n_clusters, n_features), default="random"
        Initialization method. ``"random"`` samples ``n_clusters`` distinct
        rows of ``X``, as in the paper; it is cheap for many clusters.
        ``"k-means++"`` usually gives better results for a few clusters
        but its cost can exceed that of training when ``n_clusters`` is
        large.

    compute_labels : bool, default=True
        Compute ``labels_`` and ``inertia_`` with a final assignment pass
        over the training data using the final centers.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for the initialization, the
        order of the rows in each epoch and the revival of clusters.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each training point, from the final assignment pass if
        ``compute_labels=True``, otherwise from the last epoch.

    inertia_ : float
        Sum of squared distances of training samples to their closest
        cluster center. Only set if ``compute_labels=True``.

    n_iter_ : int
        Number of epochs run. Equal to ``ceil(n_epochs)``.

    n_steps_ : int
        Number of mini-batch updates.

    batch_size_ : int
        The mini-batch size that was used.

    alpha0_ : float
        The initial EMA step that was used.

    n_splits_ : int
        Number of dead clusters that were revived.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    See Also
    --------
    sklearn.cluster.KMeans : Lloyd's (or Elkan's) full-batch k-means.
    sklearn.cluster.MiniBatchKMeans : Mini-batch k-means with per-center
        learning rates.

    Notes
    -----
    Each epoch costs one assignment of every training point to its nearest
    center, O(n_samples * n_clusters * n_features), done in chunks with
    matrix products. Large batches are assigned by a pool of threads, as
    many as BLAS is allowed to use (see ``threadpoolctl.threadpool_limits``),
    and BLAS is limited to one thread while they run.

    This is a port of the reference implementation shipped in Faiss
    (``faiss/contrib/e_means``).

    References
    ----------
    S. Bessa, P.-E. Mazaré, G. Szilvasy, H. Touvron, D. Picard, M. Douze and
    H. Jégou. "e-means: outperforming k-means (in one-tenth the epochs)".
    arXiv:2611.09501, 2026.

    Examples
    --------
    >>> from sklearn_extra.cluster import EMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> em = EMeans(n_clusters=2, random_state=0).fit(X)
    >>> em.cluster_centers_.shape
    (2, 2)
    >>> em.predict([[0, 0], [4, 4]]).shape
    (2,)
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        n_epochs=5,
        batch_size=None,
        alpha0=None,
        batch_law="auto",
        unused_threshold=None,
        init="random",
        compute_labels=True,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.batch_law = batch_law
        self.unused_threshold = unused_threshold
        self.init = init
        self.compute_labels = compute_labels
        self.random_state = random_state

    def _check_params(self, X):
        n_samples, n_features = X.shape
        if (
            not isinstance(self.n_clusters, numbers.Integral)
            or self.n_clusters < 1
        ):
            raise ValueError(
                "n_clusters should be a positive integer, got %r."
                % (self.n_clusters,)
            )
        if n_samples < self.n_clusters:
            raise ValueError(
                "n_samples=%d should be >= n_clusters=%d."
                % (n_samples, self.n_clusters)
            )
        if (
            not isinstance(self.n_epochs, numbers.Real)
            or not 0 < self.n_epochs < math.inf
        ):
            raise ValueError(
                "n_epochs should be a positive finite number, got %r."
                % (self.n_epochs,)
            )
        if self.batch_size is not None and (
            not isinstance(self.batch_size, numbers.Integral)
            or self.batch_size < 1
        ):
            raise ValueError(
                "batch_size should be a positive integer or None, got %r."
                % (self.batch_size,)
            )
        if self.alpha0 is not None and (
            not isinstance(self.alpha0, numbers.Real)
            or not 0 < self.alpha0 <= 1
        ):
            raise ValueError(
                "alpha0 should be in (0, 1] or None, got %r." % (self.alpha0,)
            )
        if self.batch_law not in ("auto", "throughput", "quality"):
            raise ValueError(
                "batch_law should be 'auto', 'throughput' or 'quality', "
                "got %r." % (self.batch_law,)
            )
        if self.unused_threshold is not None and (
            not isinstance(self.unused_threshold, numbers.Real)
            or not self.unused_threshold >= 0
        ):
            raise ValueError(
                "unused_threshold should be >= 0 or None, got %r."
                % (self.unused_threshold,)
            )
        if isinstance(self.init, str):
            if self.init not in ("random", "k-means++"):
                raise ValueError(
                    "init should be 'random', 'k-means++' or an array, "
                    "got %r." % (self.init,)
                )
        else:
            init = np.asarray(self.init)
            if init.shape != (self.n_clusters, n_features):
                raise ValueError(
                    "init array should have shape (%d, %d), got %r."
                    % (self.n_clusters, n_features, init.shape)
                )

    def _init_centers(self, X, x_squared_norms, rng):
        if isinstance(self.init, str) and self.init == "random":
            return X[rng.permutation(X.shape[0])[: self.n_clusters]]
        if isinstance(self.init, str):
            centers, _ = kmeans_plusplus(
                X,
                self.n_clusters,
                x_squared_norms=x_squared_norms,
                random_state=rng,
            )
            return centers
        return np.asarray(self.init, dtype=X.dtype)

    def _assign(self, X, centers, x_squared_norms):
        """Squared distance to, and index of, the nearest center."""
        n, d = X.shape
        k, itemsize = centers.shape[0], X.dtype.itemsize
        chunk_size = _chunk_size(k, itemsize)
        with _assignment_pool(n, k, d, chunk_size, itemsize) as pool:
            return _nearest_centers(
                X, centers, x_squared_norms, chunk_size, pool
            )

    def fit(self, X, y=None):
        """Compute e-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            order="C",
            copy=False,
        )
        self._check_params(X)
        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        k = self.n_clusters

        x_squared_norms = row_norms(X, squared=True)
        init_centers = self._init_centers(X, x_squared_norms, rng)
        epoch_seed = int(rng.randint(np.iinfo(np.int32).max))
        split_rng = np.random.RandomState(rng.randint(np.iinfo(np.int32).max))

        batch_law = _resolve_batch_law(self.batch_law, n_features)
        batch_size, alpha0 = _resolve_recipe(
            n_samples,
            k,
            self.n_epochs,
            batch_law,
            self.batch_size,
            self.alpha0,
        )
        if self.unused_threshold is None:
            unused_threshold = (
                0.0 if n_features <= _NO_REVIVAL_MAX_D else _UNUSED_THRESHOLD
            )
        else:
            unused_threshold = float(self.unused_threshold)
        epoch_rows = _epoch_rows(n_samples, self.n_epochs)

        def epoch_orders(epoch):
            return np.random.RandomState(
                (epoch_seed * 1000003 + epoch) % 2**32
            ).permutation(n_samples)

        itemsize = X.dtype.itemsize
        chunk_size = _chunk_size(k, itemsize)
        with _assignment_pool(
            batch_size, k, n_features, chunk_size, itemsize
        ) as pool:
            centers, labels, n_steps, n_splits, _ = _emeans_core(
                X,
                init_centers,
                batch_size=batch_size,
                alpha0=alpha0,
                epoch_rows=epoch_rows,
                epoch_orders=epoch_orders,
                unused_threshold=unused_threshold,
                split_rng=split_rng,
                chunk_size=chunk_size,
                x_squared_norms=x_squared_norms,
                pool=pool,
            )

        if self.compute_labels:
            distances, labels = self._assign(X, centers, x_squared_norms)
            self.inertia_ = float(distances.sum(dtype=np.float64))
            n_distinct = len(np.unique(labels))
            if n_distinct < k:
                warnings.warn(
                    "Number of distinct clusters (%d) found smaller than "
                    "n_clusters (%d). Possibly due to duplicate points "
                    "in X." % (n_distinct, k),
                    ConvergenceWarning,
                    stacklevel=2,
                )

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_iter_ = len(epoch_rows)
        self.n_steps_ = n_steps
        self.batch_size_ = batch_size
        self.alpha0_ = alpha0
        self.n_splits_ = n_splits
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            order="C",
            reset=False,
        )
        centers = self.cluster_centers_.astype(X.dtype, copy=False)
        _, labels = self._assign(X, centers, row_norms(X, squared=True))
        return labels

    def transform(self, X):
        """Transform X to a cluster-distance space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            Euclidean distances of X to the cluster centers.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            reset=False,
        )
        return euclidean_distances(X, self.cluster_centers_)

    def score(self, X, y=None):
        """Opposite of the sum of squared distances of X to their closest center.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            Opposite of the k-means objective on X.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            order="C",
            reset=False,
        )
        centers = self.cluster_centers_.astype(X.dtype, copy=False)
        distances, _ = self._assign(X, centers, row_norms(X, squared=True))
        return -float(distances.sum(dtype=np.float64))
