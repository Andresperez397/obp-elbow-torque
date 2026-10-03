"""Model pipelines and grouped cross-validation (ANALYSIS_PLAN.md sections 3-4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column at its training 1st/99th percentiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class InnerGroupElasticNet(BaseEstimator):
    """ElasticNetCV whose inner folds are grouped by pitcher."""

    def __init__(self, n_inner: int = 5):
        self.n_inner = n_inner

    def fit(self, X, y, groups):
        inner = list(GroupKFold(n_splits=self.n_inner).split(X, y, groups))
        self.model_ = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=50, cv=inner,
                                   max_iter=20000).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def preprocessing():
    """Per-fold preprocessing for the linear models: impute, winsorize, standardize."""
    return make_pipeline(SimpleImputer(strategy="median"), Winsorizer(), StandardScaler())


def make_model(kind: str):
    """Learners that need only (X, y). The grouped elastic net is built in fit_predict."""
    if kind == "mean":
        return DummyRegressor(strategy="mean")
    if kind == "ols":
        return make_pipeline(preprocessing(), LinearRegression())
    if kind == "gbm":
        return make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                           max_iter=300, min_samples_leaf=20,
                                                           random_state=0))
    raise ValueError(f"unknown model kind: {kind}")


def fit_predict(kind: str, X_tr, y_tr, g_tr, X_te) -> np.ndarray:
    """Fit on the training fold only and predict the held-out fold."""
    if kind == "enet":
        prep = preprocessing().fit(X_tr)
        net = InnerGroupElasticNet().fit(prep.transform(X_tr), y_tr, g_tr)
        return net.predict(prep.transform(X_te))
    return make_model(kind).fit(X_tr, y_tr).predict(X_te)


def shuffled_group_folds(groups: np.ndarray, n_splits: int, seed: int):
    """GroupKFold with a random pitcher-to-fold assignment (GroupKFold itself is deterministic)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = rng.permutation(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in groups])
    for k in range(n_splits):
        yield np.where(f != k)[0], np.where(f == k)[0]


def oof_predictions(df: pd.DataFrame, features: list[str], kind: str, outcome: str, group: str,
                    seed: int, scheme: str = "group", n_splits: int = 10) -> np.ndarray:
    X = df[features].to_numpy(dtype=float) if features else np.zeros((len(df), 1))
    y = df[outcome].to_numpy(dtype=float)
    g = df[group].to_numpy()
    pred = np.full(len(df), np.nan)
    if scheme == "group":
        splits = shuffled_group_folds(g, n_splits, seed)
    elif scheme == "random":
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
    else:
        raise ValueError(scheme)
    for tr, te in splits:
        pred[te] = fit_predict(kind, X[tr], y[tr], g[tr], X[te])
    return pred


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2_vs(y, p, p0) -> float:
    """R2 relative to a reference prediction (the out-of-fold training mean, M0)."""
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - p0) ** 2))
