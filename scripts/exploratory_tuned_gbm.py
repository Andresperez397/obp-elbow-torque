"""EXPLORATORY (not in the frozen plan; see DEVIATIONS.md D4).

Are the boosted trees behind the elastic net only because the elastic net is tuned and the trees
are not? Same outer folds as run_analysis.py (grouped by pitcher, repeats 0-4), M4 features. The
trees' depth, learning rate, rounds and leaf size are tuned by an inner 5-fold GroupKFold on the
training pitchers only; the elastic net is refit on the same folds for a paired comparison.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from obp_evt import data, models  # noqa: E402

N_REPEATS = 5
GRID = [dict(max_depth=d, learning_rate=lr, max_iter=it, min_samples_leaf=leaf)
        for d, lr, it, leaf in itertools.product([2, 3], [0.03, 0.1], [100, 300], [10, 30])]


def gbm(params):
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(early_stopping=False, random_state=0, **params))


def tuned_gbm_predict(X_tr, y_tr, g_tr, X_te):
    inner = list(GroupKFold(n_splits=5).split(X_tr, y_tr, g_tr))
    best, best_mse = None, np.inf
    for params in GRID:
        mse = np.mean([np.mean((gbm(params).fit(X_tr[a], y_tr[a]).predict(X_tr[b]) - y_tr[b]) ** 2)
                       for a, b in inner])
        if mse < best_mse:
            best, best_mse = params, mse
    return gbm(best).fit(X_tr, y_tr).predict(X_te), best


def main() -> None:
    df = data.load()
    X = df[data.blocks(df)["M4"]].to_numpy(dtype=float)
    y = df[data.OUTCOME].to_numpy(dtype=float)
    g = df[data.GROUP].to_numpy()
    rows, chosen = [], []
    for seed in range(N_REPEATS):
        p_gbm, p_net, p_mean = (np.full(len(y), np.nan) for _ in range(3))
        for tr, te in models.shuffled_group_folds(g, 10, seed):
            p_gbm[te], best = tuned_gbm_predict(X[tr], y[tr], g[tr], X[te])
            chosen.append(best)
            p_net[te] = models.fit_predict("enet", X[tr], y[tr], g[tr], X[te])
            p_mean[te] = y[tr].mean()
        rows.append({"seed": seed, "r2_tuned_gbm": models.r2_vs(y, p_gbm, p_mean),
                     "r2_enet": models.r2_vs(y, p_net, p_mean),
                     "rmse_tuned_gbm": models.rmse(y, p_gbm), "rmse_enet": models.rmse(y, p_net)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    tables = data.ROOT / "reports" / "tables"
    out.to_csv(tables / "exploratory_tuned_gbm.csv", index=False)
    summary = {"mean": out.mean(numeric_only=True).round(4).to_dict(),
               "enet_better_in": int((out.r2_enet > out.r2_tuned_gbm).sum()),
               "n_repeats": N_REPEATS,
               "most_chosen_params": pd.Series([json.dumps(c, sort_keys=True) for c in chosen])
               .value_counts().head(3).to_dict()}
    with open(tables / "exploratory_tuned_gbm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
