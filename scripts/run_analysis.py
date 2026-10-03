"""Run every pre-specified analysis in ANALYSIS_PLAN.md and write tables to reports/tables/."""
from __future__ import annotations

import os

# Nested BLAS/OpenMP threads oversubscribe the CPU during repeated small fits; pin to one.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from obp_evt import data, models, stats  # noqa: E402

OUT = data.ROOT / "reports" / "tables"
N_REPEATS = 20
SPECS = [  # (label, block, learner)
    ("M0 mean", None, "mean"),
    ("M1 body size", "M1", "ols"),
    ("M2 + velocity", "M2", "ols"),
    ("M3 + literature mechanics", "M3", "ols"),
    ("M4 all kinematics (enet)", "M4", "enet"),
    ("M4 all kinematics (gbm)", "M4", "gbm"),
    ("M5 + lower-body kinetics (enet)", "M5", "enet"),
    ("M5 + lower-body kinetics (gbm)", "M5", "gbm"),
]
GAINS = [  # (label, model, reference)
    ("body size over mean", "M1 body size", "M0 mean"),
    ("velocity over body size", "M2 + velocity", "M1 body size"),
    ("literature mechanics over velocity", "M3 + literature mechanics", "M2 + velocity"),
    ("all kinematics (enet) over velocity", "M4 all kinematics (enet)", "M2 + velocity"),
    ("all kinematics (enet) over literature", "M4 all kinematics (enet)", "M3 + literature mechanics"),
    ("all kinematics (gbm) over velocity", "M4 all kinematics (gbm)", "M2 + velocity"),
    ("lower-body kinetics (enet) over M4 enet", "M5 + lower-body kinetics (enet)", "M4 all kinematics (enet)"),
    ("lower-body kinetics (gbm) over M4 gbm", "M5 + lower-body kinetics (gbm)", "M4 all kinematics (gbm)"),
]


def run_scheme(df, blocks, scheme):
    y = df[data.OUTCOME].to_numpy()
    rows, oof_seed0 = [], {}
    for seed in range(N_REPEATS):
        preds = {}
        for label, block, kind in SPECS:
            feats = blocks[block] if block else []
            preds[label] = models.oof_predictions(df, feats, kind, data.OUTCOME, data.GROUP,
                                                  seed=seed, scheme=scheme)
        for label in preds:
            rows.append({"scheme": scheme, "seed": seed, "model": label,
                         "rmse": models.rmse(y, preds[label]),
                         "r2": models.r2_vs(y, preds[label], preds["M0 mean"])})
        if seed == 0:
            oof_seed0 = preds
        print(f"  {scheme} repeat {seed + 1}/{N_REPEATS}", flush=True)
    return pd.DataFrame(rows), oof_seed0


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = data.load()
    blocks = data.blocks(df)
    results: dict = {"n_pitches": len(df), "n_pitchers": int(df[data.GROUP].nunique()),
                     "block_sizes": {k: len(v) for k, v in blocks.items()}}
    json.dump(blocks, open(OUT / "feature_blocks.json", "w"), indent=2)

    cohort = df.groupby(data.GROUP).agg(level=("playing_level", "first"), lefty=("lefty", "first"),
                                        mass=("mass_kg", "first"), height=("height_m", "first"),
                                        age=("age_yrs", "first"),
                                        speed=("pitch_speed_mph", "mean"),
                                        evt=(data.OUTCOME, "mean"), n=(data.OUTCOME, "size"))
    cohort.describe().T.to_csv(OUT / "cohort_summary.csv")
    results["levels"] = cohort["level"].value_counts().to_dict()
    results["lefty_pitchers"] = int(cohort["lefty"].sum())

    print("Q1 ICC")
    results["q1_icc"] = stats.icc(df, data.OUTCOME, data.GROUP)

    print("Q2/Q3 cross-validation")
    grouped, oof = run_scheme(df, blocks, "group")
    random, _ = run_scheme(df, blocks, "random")
    cv = pd.concat([grouped, random])
    cv.to_csv(OUT / "cv_repeats.csv", index=False)
    summ = cv.groupby(["scheme", "model"], sort=False)[["rmse", "r2"]].agg(["mean", "std"])
    summ.to_csv(OUT / "cv_summary.csv")

    # Pitcher-cluster bootstrap on the seed-0 grouped out-of-fold predictions.
    y = df[data.OUTCOME].to_numpy()
    boot_df = pd.DataFrame({"g": df[data.GROUP].to_numpy(), "y": y,
                            **{k: v for k, v in oof.items()}})

    def stat_fn(sub):
        out = {}
        for label, _, _ in SPECS:
            out[f"rmse|{label}"] = models.rmse(sub["y"], sub[label])
            out[f"r2|{label}"] = models.r2_vs(sub["y"], sub[label], sub["M0 mean"])
        return out

    boot = stats.cluster_bootstrap(boot_df, "g", stat_fn, n_boot=2000, seed=0)
    point = stat_fn(boot_df)
    model_rows = []
    for label, _, _ in SPECS:
        g_mean = grouped[grouped.model == label][["rmse", "r2"]].mean()
        r_mean = random[random.model == label][["rmse", "r2"]].mean()
        pm = boot_df.groupby("g")[list(dict.fromkeys(["y", label, "M0 mean"]))].mean()
        model_rows.append({
            "model": label,
            "rmse_grouped": g_mean["rmse"], "r2_grouped": g_mean["r2"],
            "r2_seed0": point[f"r2|{label}"],
            "r2_ci_low": np.percentile(boot[f"r2|{label}"], 2.5),
            "r2_ci_high": np.percentile(boot[f"r2|{label}"], 97.5),
            "rmse_ci_low": np.percentile(boot[f"rmse|{label}"], 2.5),
            "rmse_ci_high": np.percentile(boot[f"rmse|{label}"], 97.5),
            "r2_pitcher_level": models.r2_vs(pm["y"].to_numpy(), pm[label].to_numpy(),
                                             pm["M0 mean"].to_numpy()),
            "rmse_random": r_mean["rmse"], "r2_random": r_mean["r2"],
            "leakage_inflation_r2": r_mean["r2"] - g_mean["r2"],
        })
    model_table = pd.DataFrame(model_rows)
    model_table.to_csv(OUT / "model_table.csv", index=False)

    gain_rows = []
    for label, m, ref in GAINS:
        d = boot[f"r2|{m}"] - boot[f"r2|{ref}"]
        gain_rows.append({"comparison": label, "gain_r2": point[f"r2|{m}"] - point[f"r2|{ref}"],
                          "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                          "adds_information": bool(np.percentile(d, 2.5) > 0)})
    pd.DataFrame(gain_rows).to_csv(OUT / "block_gains.csv", index=False)
    boot_df.to_csv(OUT / "oof_predictions_seed0.csv", index=False)

    print("Q4 within/between")
    wb = stats.within_between(df, data.OUTCOME, data.GROUP)
    wb.to_csv(OUT / "within_between.csv", index=False)
    results["q4_within_speed_sd_mph"] = wb.attrs["within_speed_sd_mph"]

    print("Q5 normalization")
    nt = stats.normalization_test(df, data.OUTCOME, data.GROUP)
    nt.pop("pitcher_table").to_csv(OUT / "pitcher_means.csv", index=False)
    results["q5_normalization"] = nt

    results["runtime_s"] = round(time.time() - t0, 1)
    json.dump(results, open(OUT / "results.json", "w"), indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))
    print(model_table.round(3).to_string())
    print(pd.DataFrame(gain_rows).round(3).to_string())
    print(wb.round(3).to_string())


if __name__ == "__main__":
    main()
