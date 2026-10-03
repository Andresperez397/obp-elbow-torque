"""Build the report figures from reports/tables/ (run after run_analysis.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TAB, FIG = ROOT / "reports" / "tables", ROOT / "reports" / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def model_ladder():
    t = pd.read_csv(TAB / "model_table.csv")
    t = t[t["model"] != "M0 mean"].reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hlines(y, t["r2_ci_low"], t["r2_ci_high"], color=BLUE, lw=2)
    ax.plot(t["r2_grouped"], y, "o", ms=8, color=BLUE, label="New pitchers (grouped CV)")
    ax.plot(t["r2_random"], y, "o", ms=8, mfc=SURFACE, mec=ORANGE, mew=2,
            label="Random pitch split (leaky)")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, t["model"])
    ax.set_xlabel("Out-of-sample R²  (share of torque variance explained)")
    ax.set_title("Torque explained by each block of information")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_model_ladder.png", dpi=200)
    plt.close(fig)


def variance_split(res):
    q = res["q1_icc"]
    between, within = q["tau_nm"] ** 2, q["sigma_nm"] ** 2
    total = between + within
    fig, ax = plt.subplots(figsize=(7.6, 1.6))
    ax.barh([0], [between / total], color=BLUE, height=0.5)
    ax.barh([0], [within / total], left=[between / total], color=ORANGE, height=0.5,
            edgecolor=SURFACE, linewidth=2)
    ax.text(between / total / 2, 0, f"Between pitchers  {between / total:.0%}", ha="center",
            va="center", color="white", fontweight="bold")
    ax.text(1.01, 0, f"Within a pitcher  {within / total:.0%}", ha="left", va="center",
            color=INK, fontweight="bold")
    ax.set_xlim(0, 1.32)
    ax.axis("off")
    ax.set_title(f"Where torque varies (ICC {q['icc']:.2f}, 95% CI {q['ci_low']:.2f}–{q['ci_high']:.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_variance_split.png", dpi=200)
    plt.close(fig)


def observed_vs_predicted():
    t = pd.read_csv(TAB / "model_table.csv").set_index("model")
    oof = pd.read_csv(TAB / "oof_predictions_seed0.csv")
    best = t.drop(index="M0 mean")["r2_grouped"].idxmax()
    pm = oof.groupby("g")[["y", best]].mean()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lo, hi = pm.min().min() - 5, pm.max().max() + 5
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    ax.plot(pm[best], pm["y"], "o", ms=6, color=BLUE, alpha=0.8, mec=SURFACE, mew=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted torque, pitcher mean (Nm)")
    ax.set_ylabel("Measured torque, pitcher mean (Nm)")
    ax.set_title(f"Unseen pitchers: {best}")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_observed_vs_predicted.png", dpi=200)
    plt.close(fig)


def normalization(res):
    p = pd.read_csv(TAB / "pitcher_means.csv")
    q = res["q5_normalization"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(p["bwh"], p["evt"], "o", ms=6, color=BLUE, alpha=0.75, mec=SURFACE, mew=1)
    x = np.linspace(0, p["bwh"].max() * 1.05, 50)
    ax.plot(x, q["intercept_nm"] + q["slope"] * x, color=BLUE, lw=2, label="Fitted line")
    ratio = (p["evt"] / p["bwh"]).mean()
    ax.plot(x, ratio * x, color=ORANGE, lw=2, ls="--",
            label="What ratio normalization assumes (through 0)")
    ax.set_xlim(0, x.max())
    ax.set_ylim(0, p["evt"].max() * 1.1)
    ax.set_xlabel("Body weight × height (N·m)")
    ax.set_ylabel("Elbow varus torque, pitcher mean (Nm)")
    lo, hi = q["intercept_ci"]
    ax.set_title(f"Intercept {q['intercept_nm']:.0f} Nm (95% CI {lo:.0f} to {hi:.0f})")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_normalization.png", dpi=200)
    plt.close(fig)


def velocity_slopes():
    wb = pd.read_csv(TAB / "within_between.csv").set_index("term")
    rows = [("Between pitchers", "speed_between", BLUE), ("Within a pitcher", "speed_within", ORANGE)]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for i, (_, term, c) in enumerate(rows):
        r = wb.loc[term]
        ax.hlines(i, r["ci_low"], r["ci_high"], color=c, lw=2)
        ax.plot(r["estimate"], i, "o", ms=8, color=c)
        ax.text(r["estimate"], i + 0.22, f"{r['estimate']:.2f} Nm/mph", color=INK2, ha="center")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([0, 1], [r[0] for r in rows])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Nm of torque per +1 mph (adjusted for body size, handedness)")
    ax.set_title("Torque cost of velocity")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_velocity_slopes.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    with open(TAB / "results.json") as f:
        res = json.load(f)
    model_ladder()
    variance_split(res)
    observed_vs_predicted()
    normalization(res)
    velocity_slopes()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
