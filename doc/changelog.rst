Changelog
=========

Unreleased
----------

- Add `CLARA` (Clustering for Large Applications) which extends k-medoids to
    be more scalable using a sampling approach.
    [`#83 <https://github.com/scikit-learn-contrib/scikit-learn-extra/pull/83>`_].
- Fix `_estimator_type` for :class:`~sklearn_extra.robust` estimators. Fix
  misbehavior of scikit-learn's :class:`~sklearn.model_selection.cross_val_score` and
  :class:`~sklearn.grid_search.GridSearchCV` for :class:`~sklearn_extra.robust.RobustWeightedClassifier`

Version 0.2.0
-------------
*April 14, 2021*

- Add :class:`~sklearn_extra.robust.RobustWeightedClassifier`,
  :class:`~sklearn_extra.robust.RobustWeightedRegressor` and
  :class:`~sklearn_extra.robust.RobustWeightedKMeans` estimators that rely on
  iterative reweighting of samples to be robust to
  outliers. [`#42 <https://github.com/scikit-learn-contrib/scikit-learn-extra/pull/42>`_].
- Added Common Nearest-Neighbors clustering estimator
  :class:`~sklearn_extra.cluster.CommonNNClustering`
  [`#64 <https://github.com/scikit-learn-contrib/scikit-learn-extra/pull/64>`_]
- Added PAM algorithm to :class:`~sklearn_extra.cluster.KMedoids` with ``method="pam"`` parameter
  which produces better solutions but at higher computational cost
  [`#73 <https://github.com/scikit-learn-contrib/scikit-learn-extra/pull/73>`_]
- Binary wheels were uploaded to PyPi, making the installation possible without a C compiler
  [`#66 <https://github.com/scikit-learn-contrib/scikit-learn-extra/pull/66>`_]

List of contributors (in alphabetical order)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Christos Aridas, Jan-Oliver Joswig, Timothée Mathieu, Roman Yurchak
