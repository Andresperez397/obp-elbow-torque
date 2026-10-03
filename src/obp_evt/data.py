"""Load and join the OBP pitching POI data and define the pre-specified predictor blocks."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

OUTCOME = "elbow_varus_moment"
GROUP = "pitcher_id"

# Same throwing-arm inverse-dynamics solution as the outcome -> never used as predictors.
LEAKAGE = [
    "shoulder_internal_rotation_moment",
    "shoulder_transfer_fp_br", "shoulder_generation_fp_br", "shoulder_absorption_fp_br",
    "elbow_transfer_fp_br", "elbow_generation_fp_br", "elbow_absorption_fp_br",
    "thorax_distal_transfer_fp_br",
]

BODY = ["mass_kg", "height_m", "lefty"]
VELO = ["pitch_speed_mph"]
LITERATURE = [
    "max_shoulder_external_rotation", "elbow_flexion_mer", "shoulder_abduction_fp", "arm_slot",
    "torso_lateral_tilt_br", "torso_rotation_fp", "timing_peak_torso_to_peak_pelvis_rot_velo",
    "max_torso_rotational_velo", "max_pelvis_rotational_velo", "stride_length",
]
LOWER_KINETICS_PREFIXES = ("lead_hip_", "lead_knee_transfer", "lead_knee_generation",
                           "lead_knee_absorption", "rear_hip_", "rear_knee_", "pelvis_lumbar_",
                           "rear_grf_", "lead_grf_", "peak_rfd_")

ID_COLS = ["session_pitch", "session", "p_throws", "pitch_type"]


def load(raw_dir: Path = RAW) -> pd.DataFrame:
    poi = pd.read_csv(raw_dir / "poi_metrics.csv")
    meta = pd.read_csv(raw_dir / "metadata.csv")
    meta = meta.rename(columns={"user": GROUP, "session_mass_kg": "mass_kg",
                                "session_height_m": "height_m"})
    df = poi.merge(meta[["session_pitch", GROUP, "mass_kg", "height_m", "age_yrs", "playing_level"]],
                   on="session_pitch", how="inner", validate="one_to_one")
    if len(df) != len(poi):
        raise ValueError(f"join dropped rows: {len(poi)} -> {len(df)}")
    df["lefty"] = (df["p_throws"] == "L").astype(int)
    return df


def kinematic_columns(df: pd.DataFrame) -> list[str]:
    """Every POI that is not the outcome, an identifier, leakage, or a lower-body kinetic."""
    skip = set(ID_COLS) | set(LEAKAGE) | {OUTCOME, GROUP, "mass_kg", "height_m", "lefty",
                                          "age_yrs", "playing_level", "pitch_speed_mph"}
    return [c for c in df.columns
            if c not in skip and not c.startswith(LOWER_KINETICS_PREFIXES)
            and pd.api.types.is_numeric_dtype(df[c])]


def lower_kinetic_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(LOWER_KINETICS_PREFIXES)]


def blocks(df: pd.DataFrame) -> dict[str, list[str]]:
    m1 = BODY
    m2 = m1 + VELO
    m3 = m2 + LITERATURE
    m4 = m2 + kinematic_columns(df)
    m5 = m4 + lower_kinetic_columns(df)
    return {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5}
