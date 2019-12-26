# License: BSD 3 clause
#
# Authors: Roman Yurchak <rth.yurchak@gmail.com>

import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score

from sklearn_extra.feature_weighting import TfigmTransformer

if "CI" in os.environ:
    # make this example run faster in CI
    categories = ["sci.crypt", "comp.graphics", "comp.sys.mac.hardware"]
else:
    categories = None

docs, y = fetch_20newsgroups(return_X_y=True, categories=categories)


vect = CountVectorizer(min_df=5, stop_words="english", ngram_range=(1, 1))
X = vect.fit_transform(docs)

res = []

for scaler_label, scaler in [
    ("TF", FunctionTransformer(lambda x: x)),
    ("TF-IDF(sublinear_tf=False)", TfidfTransformer()),
    ("TF-IDF(sublinear_tf=True)", TfidfTransformer(sublinear_tf=True)),
    ("TF-IGM(tf_scale=None)", TfigmTransformer()),
    ("TF-IGM(tf_scale='sqrt')", TfigmTransformer(tf_scale="sqrt"),),
    ("TF-IGM(tf_scale='log1p')", TfigmTransformer(tf_scale="log1p"),),
]:
    pipe = make_pipeline(scaler, Normalizer())
    X_tr = pipe.fit_transform(X, y)
    est = LinearSVC()
    scoring = {
        "F1-macro": lambda est, X, y: f1_score(
            y, est.predict(X), average="macro"
        ),
        "balanced_accuracy": "balanced_accuracy",
    }
    scores = cross_validate(est, X_tr, y, scoring=scoring,)
    for key, val in scores.items():
        if not key.endswith("_time"):
            res.append(
                {
                    "metric": "_".join(key.split("_")[1:]),
                    "subset": key.split("_")[0],
                    "preprocessing": scaler_label,
                    "score": f"{val.mean():.3f}±{val.std():.3f}",
                }
            )
scores = (
    pd.DataFrame(res)
    .set_index(["preprocessing", "metric", "subset"])["score"]
    .unstack(-1)
)
scores = scores["test"].unstack(-1).sort_values("F1-macro", ascending=False)
print(scores)
