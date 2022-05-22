# -*- coding: utf-8 -*-
"""
================================================================
An example of a robust cross-validation evaluation in regression
================================================================
In this example we compare `LinearRegression` (OLS) with `HuberRegressor`  from
scikit-learn using cross-validation.

We show that a robust cross-validation scheme gives a better
evaluation of the generalisation error in a corrupted dataset.

In this example, we do robust cross-validation by using an alternative to the
empirical mean to aggregate the errors. This alternative is a robust estimator
of the mean (the trimmed mean is an example of such a robust estimator, but here
we use Huber's estimator). This robust estimator of the mean is used on each
fold of the cross-validation and then, we return the empirical mean of the
obtained robust scores to get the final score.
"""
print(__doc__)

import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn_extra.robust import make_huber_metric
from sklearn.linear_model import LinearRegression, HuberRegressor

robust_mse = make_huber_metric(mean_squared_error, c=9)
rng = np.random.RandomState(42)

X = rng.uniform(size=100)[:, np.newaxis]
y = 3 * X.ravel()
# Remark y <= 3

y[[42 // 2, 42, 42 * 2]] = 200  # outliers

print("Non robust error:")
for reg in [LinearRegression(), HuberRegressor()]:
    print(
        reg,
        " mse : %.2F"
        % (
            np.mean(
                cross_val_score(
                    reg, X, y, scoring=make_scorer(mean_squared_error)
                )
            )
        ),
    )


print("\n")
print("Robust error:")
for reg in [LinearRegression(), HuberRegressor()]:
    print(
        reg,
        " mse : %.2F"
        % (
            np.mean(
                cross_val_score(reg, X, y, scoring=make_scorer(robust_mse))
            )
        ),
    )
