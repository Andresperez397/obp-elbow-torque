"""Mixed models, bootstrap intervals and the normalization test (ANALYSIS_PLAN.md sections 4-6)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

G = 9.80665


def _fit_mixed(formula: str, df: pd.DataFrame, group: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, df, groups=df[group])
        for method in ("lbfgs", "powell", "nm"):
            try:
                res = model.fit(reml=True, method=method)
                if res.converged:
                    return res
            except Exception:  # noqa: BLE001 - try the next optimizer
                continue
    raise RuntimeError(f"mixed model did not converge: {formula}")


def icc(df: pd.DataFrame, outcome: str, group: str, n_boot: int = 1000, seed: int = 0) -> dict:
    res = _fit_mixed(f"{outcome} ~ 1", df, group)
    tau2 = float(res.cov_re.iloc[0, 0])
    sigma2 = float(res.scale)
    est = tau2 / (tau2 + sigma2)

    # Parametric bootstrap: simulate from the fitted model, refit, recompute ICC.
    rng = np.random.default_rng(seed)
    groups = df[group].to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    mu = float(res.fe_params.iloc[0])
    boots = []
    sim = df[[group]].copy()
    for _ in range(n_boot):
        u = rng.normal(0, np.sqrt(tau2), len(uniq))[inv]
        sim["y"] = mu + u + rng.normal(0, np.sqrt(sigma2), len(df))
        try:
            r = _fit_mixed("y ~ 1", sim, group)
        except RuntimeError:
            continue
        t, s = float(r.cov_re.iloc[0, 0]), float(r.scale)
        boots.append(t / (t + s))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"icc": est, "ci_low": float(lo), "ci_high": float(hi), "tau_nm": np.sqrt(tau2),
            "sigma_nm": np.sqrt(sigma2), "n_boot_ok": len(boots)}


def within_between(df: pd.DataFrame, outcome: str, group: str) -> pd.DataFrame:
    d = df.copy()
    d["speed_between"] = d.groupby(group)["pitch_speed_mph"].transform("mean")
    d["speed_within"] = d["pitch_speed_mph"] - d["speed_between"]
    res = _fit_mixed(f"{outcome} ~ speed_between + speed_within + mass_kg + height_m + lefty",
                     d, group)
    ci = res.conf_int()
    rows = []
    for term in ["speed_between", "speed_within", "mass_kg", "height_m", "lefty"]:
        rows.append({"term": term, "estimate": res.params[term], "ci_low": ci.loc[term, 0],
                     "ci_high": ci.loc[term, 1], "p": res.pvalues[term]})
    out = pd.DataFrame(rows)
    out.attrs["within_speed_sd_mph"] = float(d["speed_within"].std())
    return out


def normalization_test(df: pd.DataFrame, outcome: str, group: str) -> dict:
    p = df.groupby(group).agg(evt=(outcome, "mean"), mass=("mass_kg", "first"),
                              height=("height_m", "first")).reset_index()
    p["bwh"] = p["mass"] * G * p["height"]  # N*m
    res = smf.ols("evt ~ bwh", p).fit()
    ci = res.conf_int()
    p["norm"] = p["evt"] / p["bwh"]
    r = float(np.corrcoef(p["norm"], p["bwh"])[0, 1])
    z, se = np.arctanh(r), 1 / np.sqrt(len(p) - 3)  # Fisher z interval
    r_ci = (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))
    return {"intercept_nm": res.params["Intercept"], "intercept_ci": (ci.loc["Intercept", 0],
            ci.loc["Intercept", 1]), "slope": res.params["bwh"], "r2": res.rsquared,
            "corr_normalized_vs_bwh": r, "corr_ci": r_ci, "n_pitchers": len(p),
            "pitcher_table": p}


def cluster_bootstrap(df: pd.DataFrame, group: str, stat_fn, n_boot: int = 2000, seed: int = 0):
    """Resample pitchers with replacement; stat_fn(sub_df) -> float or dict of floats."""
    rng = np.random.default_rng(seed)
    idx_by_g = {g: np.flatnonzero(df[group].to_numpy() == g) for g in df[group].unique()}
    keys = np.array(list(idx_by_g))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([idx_by_g[k] for k in pick])
        out.append(stat_fn(df.iloc[rows]))
    return pd.DataFrame(out)
