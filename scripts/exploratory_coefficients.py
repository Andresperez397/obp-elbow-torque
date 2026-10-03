"""EXPLORATORY (not in the frozen plan; see DEVIATIONS.md D1).

Which kinematic variables does the M4 elastic net lean on, and are they stable across pitcher
resamples? Fits the M4 elastic net on 500 pitcher-bootstrap samples and records each
standardized coefficient. Describes the fitted model; it is not a causal claim.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from obp_evt import data, models  # noqa: E402

N_BOOT = 500


def fit_coefs(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].to_numpy(dtype=float)
    prep = make_pipeline(*models._prep()).fit(X)
    net = models.InnerGroupElasticNet().fit(prep.transform(X), df[data.OUTCOME].to_numpy(),
                                            df[data.GROUP].to_numpy())
    return net.model_.coef_


def main() -> None:
    df = data.load()
    cols = data.blocks(df)["M4"]
    full = fit_coefs(df, cols)
    rng = np.random.default_rng(0)
    pitchers = df[data.GROUP].unique()
    boots = []
    for b in range(N_BOOT):
        pick = rng.choice(pitchers, size=len(pitchers), replace=True)
        # Relabel duplicated pitchers so inner grouped CV treats each copy as its own pitcher.
        parts = [df[df[data.GROUP] == p].assign(**{data.GROUP: i}) for i, p in enumerate(pick)]
        boots.append(fit_coefs(pd.concat(parts, ignore_index=True), cols))
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{N_BOOT}", flush=True)
    B = np.array(boots)
    out = pd.DataFrame({
        "feature": cols, "coef_full_nm_per_sd": full,
        "ci_low": np.percentile(B, 2.5, axis=0), "ci_high": np.percentile(B, 97.5, axis=0),
        "selected_share": (B != 0).mean(0),
        "same_sign_share": (np.sign(B) == np.sign(full)).mean(0),
    }).sort_values("coef_full_nm_per_sd", key=np.abs, ascending=False)
    path = data.ROOT / "reports" / "tables" / "exploratory_enet_coefficients.csv"
    out.to_csv(path, index=False)
    print(out.round(2).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
