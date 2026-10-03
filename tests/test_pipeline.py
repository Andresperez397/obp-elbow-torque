import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from obp_evt import data, models, stats  # noqa: E402

HAVE_DATA = (data.RAW / "poi_metrics.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run scripts/fetch_data.py first")


@pytest.fixture(scope="module")
def df():
    return data.load()


@needs_data
def test_join_keeps_every_pitch(df):
    assert len(df) == len(pd.read_csv(data.RAW / "poi_metrics.csv"))
    assert df["session_pitch"].is_unique
    assert df[["mass_kg", "height_m", data.GROUP]].notna().all().all()


@needs_data
def test_one_session_per_pitcher(df):
    assert (df.groupby(data.GROUP)["session"].nunique() == 1).all()


@needs_data
def test_no_leakage_or_outcome_in_any_block(df):
    for name, cols in data.blocks(df).items():
        assert data.OUTCOME not in cols, name
        assert not set(cols) & set(data.LEAKAGE), name
        assert len(cols) == len(set(cols)), name


@needs_data
def test_blocks_are_nested(df):
    b = data.blocks(df)
    assert set(b["M1"]) < set(b["M2"]) < set(b["M3"]) <= set(b["M4"]) < set(b["M5"])


@needs_data
def test_lower_kinetics_only_in_m5(df):
    lk = set(data.lower_kinetic_columns(df))
    assert lk and not lk & set(data.blocks(df)["M4"])
    assert "lead_knee_extension_angular_velo_max" in data.blocks(df)["M4"]


def test_grouped_folds_never_share_a_pitcher():
    g = np.repeat(np.arange(30), 4)
    seen = []
    for tr, te in models.shuffled_group_folds(g, 10, seed=3):
        assert not set(g[tr]) & set(g[te])
        seen.extend(te)
    assert sorted(seen) == list(range(len(g)))


def test_winsorizer_uses_training_quantiles_only():
    w = models.Winsorizer().fit(np.arange(100, dtype=float).reshape(-1, 1))
    assert w.transform(np.array([[1e6]]))[0, 0] == pytest.approx(np.quantile(np.arange(100), 0.99))


def test_r2_reference():
    y = np.array([1.0, 2.0, 3.0])
    assert models.r2_vs(y, y, np.full(3, 2.0)) == 1.0
    assert models.r2_vs(y, np.full(3, 2.0), np.full(3, 2.0)) == 0.0


def test_icc_recovers_known_value():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(200), 5)
    u = rng.normal(0, 2, 200)
    y = u[g] + rng.normal(0, 1, len(g))  # population ICC = 4/5
    realized = np.var(u, ddof=1) / (np.var(u, ddof=1) + 1)  # ICC this sample actually holds
    est = stats.icc(pd.DataFrame({"y": y, "g": g}), "y", "g", n_boot=100)
    assert est["ci_low"] < realized < est["ci_high"]
    assert est["icc"] == pytest.approx(realized, abs=0.03)


def test_oof_mean_model_uses_only_training_folds():
    """The M0 prediction for each held-out pitcher must equal the mean of the OTHER folds."""
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 3)
    df = pd.DataFrame({"y": rng.normal(100, 10, len(g)), "g": g})
    pred = models.oof_predictions(df, [], "mean", "y", "g", seed=1, n_splits=5)
    for tr, te in models.shuffled_group_folds(g, 5, seed=1):
        assert np.allclose(pred[te], df["y"].iloc[tr].mean())


@pytest.mark.parametrize("kind", ["ols", "enet", "gbm"])
def test_held_out_pitcher_outcome_cannot_affect_its_own_prediction(kind):
    """Corrupt one pitcher's torque; that pitcher's out-of-fold predictions must not move."""
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(40), 3)
    X = rng.normal(size=(len(g), 4))
    y = X @ np.array([3.0, -2.0, 0.0, 1.0]) + rng.normal(0, 1, len(g))
    df = pd.DataFrame(X, columns=list("abcd")).assign(y=y, g=g)
    before = models.oof_predictions(df, list("abcd"), kind, "y", "g", seed=0, n_splits=5)
    bad = df.copy()
    bad.loc[bad["g"] == 7, "y"] = 1e6
    after = models.oof_predictions(bad, list("abcd"), kind, "y", "g", seed=0, n_splits=5)
    rows = (df["g"] == 7).to_numpy()
    assert np.allclose(before[rows], after[rows])
    assert not np.allclose(before[~rows], after[~rows])  # sanity: the corruption is seen elsewhere


def test_cluster_bootstrap_resamples_whole_pitchers():
    """Within every resample, each pitcher contributes all of its rows or none, possibly repeated."""
    sizes = {1: 1, 2: 2, 3: 3, 4: 5}
    df = pd.DataFrame({"g": np.repeat(list(sizes), list(sizes.values()))})

    def whole_blocks(sub):
        counts = sub["g"].value_counts()
        return {"ok": all(counts[k] % sizes[k] == 0 for k in counts.index), "n_g": len(sub)}

    out = stats.cluster_bootstrap(df, "g", whole_blocks, n_boot=300, seed=0)
    assert out["ok"].all()


def test_gbm_settings_are_the_planned_fixed_ones():
    hgb = models.make_model("gbm").steps[-1][1]
    params = hgb.get_params()
    assert params["early_stopping"] is False  # no silent internal (non-grouped) validation split
    assert (params["max_depth"], params["learning_rate"], params["max_iter"],
            params["min_samples_leaf"]) == (3, 0.05, 300, 20)
