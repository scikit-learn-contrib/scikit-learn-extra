import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score

from sklearn_extra.feature_weighting import TfigmTransformer


X, y = fetch_20newsgroups(return_X_y=True)

#print('classes:', pd.Series(y).value_counts())
res = []

for scaler_label, scaler in tqdm([
        ("identity", FunctionTransformer(lambda x: x)),
        ("TF-IDF(sublinar_tf=False)", TfidfTransformer()),
        ("TF-IDF(sublinear_tf=True)", TfidfTransformer(sublinear_tf=True)),
        ("TF-IGM(tf_scale=None)", TfigmTransformer(alpha=7)),
        ("TF-IGM(tf_scale='sqrt')", TfigmTransformer(alpha=7, tf_scale="sqrt")),
        ("TF-IGM(tf_scale='log1p')", TfigmTransformer(alpha=7, tf_scale="log1p")),
    ]):
    pipe = make_pipeline(
        CountVectorizer(min_df=5, stop_words="english"),
        scaler,
        Normalizer()
    )
    X_tr = pipe.fit_transform(X, y)
    est = LogisticRegression(random_state=2, solver="liblinear")
    est = LinearSVC()
    scoring={
        'F1-macro': lambda est, X, y: f1_score(y, est.predict(X), average="macro"),
        'balanced_accuracy': "balanced_accuracy"
    }
    scores = cross_validate(
        est, X_tr, y, verbose=0,
        n_jobs=6,
        scoring=scoring,
        return_train_score=True
    )
    res.extend([{'metric': "_".join(key.split('_')[1:]),
               'subset': key.split('_')[0],
               "preprocessing": scaler_label,
               "score": f"{val.mean():.3f}+-{val.std():.3f}"}
              for key, val in scores.items() if not key.endswith('_time')])
scores = pd.DataFrame(res).set_index(["preprocessing", "metric", 'subset'])['score'].unstack(-1)
scores = scores['test'].unstack(-1).sort_values("F1-macro", ascending=False)
print(scores)
