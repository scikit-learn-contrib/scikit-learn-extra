from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from sklearn_extra.feature_weighting import TfigmTransformer


X, y = fetch_20newsgroups(return_X_y=True)

for scaler in [TfidfTransformer(), TfigmTransformer(alpha=9)]:
    pipe = make_pipeline(
        CountVectorizer(min_df=5, stop_words="english"),
        scaler,
        Normalizer()
    )
    X_tr = pipe.fit_transform(X, y)
    est = LogisticRegression(random_state=2, solver="liblinear")
    scores = cross_val_score(
        est, X_tr, y, verbose=1,
        scoring=lambda est, X, y: f1_score(y, est.predict(X), average="macro"),
    )
    print(f"{scaler.__class__.__name__} F1-macro score: "
          f"{scores.mean():.3f}+-{scores.std():.3f}")
