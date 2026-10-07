"""Elbow torque explorer: why validation holds out whole pitchers, and what body size says.

    streamlit run app/streamlit_app.py

Reads only the committed tables in reports/tables (derived from the OpenBiomechanics Project).
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

T = Path(__file__).resolve().parents[1] / "reports" / "tables"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9890"

st.set_page_config(page_title="Elbow Torque Explorer", layout="wide")


@st.cache_data
def load():
    models = pd.read_csv(T / "model_table.csv")
    means = pd.read_csv(T / "pitcher_means.csv")
    res = json.loads((T / "results.json").read_text())
    return models, means, res


models, means, res = load()
icc = res["q1_icc"]

st.title("Predicting elbow torque for pitchers the model has never seen")
st.caption(
    "411 fastballs from 100 pitchers in the OpenBiomechanics Project (Driveline Baseball). Peak elbow "
    "varus torque from "
    "marker-based motion capture. Every accuracy figure is for held-out pitchers unless it says otherwise."
)
tab1, tab2, tab3 = st.tabs(["The validation trap", "Pitcher or pitch?", "Body-size estimate"])

with tab1:
    st.subheader("The same model looks far better if you split by pitch instead of by pitcher")
    st.write(
        "A pitcher's other fastballs are nearly copies of the one being predicted. If they sit in the "
        "training data, "
        "a flexible model can match the pitch to the pitcher. Pick a model to compare the honest number "
        "(whole "
        "pitchers held out) with the shortcut (random pitches held out)."
    )
    name = st.selectbox(
        "Model", list(models["model"]), index=int(models.index[models["model"].str.contains("gbm")][0])
    )
    m = models[models["model"] == name].iloc[0]
    long = pd.DataFrame(
        {
            "split": ["Hold out whole pitchers (honest)", "Hold out random pitches (leaky)"],
            "R²": [m["r2_grouped"], m["r2_random"]],
        }
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Honest R² (new pitchers)", f"{m['r2_grouped']:.2f}")
    c2.metric("Leaky R² (random pitches)", f"{m['r2_random']:.2f}")
    c3.metric("Overstatement", f"+{m['leakage_inflation_r2']:.2f}")
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("R²:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("split:N", title=None, sort=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(range=[BLUE, ORANGE]),
                legend=None,
            ),
            tooltip=["split", alt.Tooltip("R²:Q", format=".2f")],
        )
        .properties(height=130)
    )
    st.altair_chart(chart, width="stretch")
    full = models.assign(Overstatement=models["leakage_inflation_r2"])
    st.dataframe(
        full[["model", "r2_grouped", "r2_random", "Overstatement"]].rename(
            columns={"model": "Model", "r2_grouped": "Honest R²", "r2_random": "Leaky R²"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Boosted trees look like the best model under the leaky split (0.87) and fall to 0.51 on new "
        "pitchers, below a "
        "plain elastic net (0.63). A random split would have picked the wrong model."
    )

with tab2:
    st.subheader("Torque is a pitcher trait")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Share of variance between pitchers (ICC)",
        f"{icc['icc']:.2f}",
        help="95% CI {:.2f} to {:.2f}".format(icc["ci_low"], icc["ci_high"]),
    )
    c2.metric("Spread between pitchers (SD)", f"{icc['tau_nm']:.0f} Nm")
    c3.metric("Spread of one pitcher's fastballs (SD)", f"{icc['sigma_nm']:.0f} Nm")
    st.write(
        "One pitcher's fastballs differ by about 4 Nm; different pitchers differ by about 19 Nm. So the "
        "useful target "
        "is the pitcher, and an honest test has to hold out whole pitchers, which is what tab 1 shows."
    )
    hist = (
        alt.Chart(means)
        .mark_bar(color=GREY)
        .encode(
            x=alt.X("evt:Q", bin=alt.Bin(maxbins=25), title="Pitcher's mean peak elbow varus torque (Nm)"),
            y=alt.Y("count()", title="Pitchers"),
        )
    )
    st.altair_chart(hist.properties(height=260), width="stretch")

with tab3:
    st.subheader("What body size alone says about a pitcher's torque")
    st.write(
        "A straight-line fit of mean torque on body mass and height across the 100 pitchers. It is the "
        "simplest model in "
        "the project (honest R² 0.35) and a baseline, not a prediction tool."
    )
    c1, c2 = st.columns(2)
    mass = c1.slider("Body mass (kg)", 60.0, 130.0, 90.0, 1.0)
    height = c2.slider("Height (m)", 1.60, 2.10, 1.88, 0.01)
    X = np.column_stack([np.ones(len(means)), means["mass"], means["height"]])
    beta, *_ = np.linalg.lstsq(X, means["evt"].to_numpy(), rcond=None)
    pred = float(beta @ [1.0, mass, height])
    rmse = float(models.loc[models["model"] == "M1 body size", "rmse_grouped"].iloc[0])
    lo, hi = pred - 1.645 * rmse, pred + 1.645 * rmse
    a, b, c = st.columns(3)
    a.metric("Estimated peak torque", f"{pred:.0f} Nm")
    b.metric("Typical range (90%)", f"{lo:.0f} to {hi:.0f} Nm")
    c.metric("Honest error (RMSE)", f"{rmse:.0f} Nm")
    pts = (
        alt.Chart(means)
        .mark_circle(color=GREY, opacity=0.7, size=50)
        .encode(
            x=alt.X("mass:Q", scale=alt.Scale(zero=False), title="Body mass (kg)"),
            y=alt.Y("evt:Q", scale=alt.Scale(zero=False), title="Mean peak torque (Nm)"),
            tooltip=["mass", "height", "evt"],
        )
    )
    you = (
        alt.Chart(pd.DataFrame({"mass": [mass], "evt": [pred]}))
        .mark_point(color=ORANGE, size=200, filled=True)
        .encode(x="mass:Q", y="evt:Q")
    )
    st.altair_chart((pts + you).properties(height=300), width="stretch")
    st.caption(
        "Limits: 100 mostly college pitchers tested in one lab, one session each; torque is a modeled "
        "inverse-dynamics "
        "estimate whose level scales with body mass. Do not use this for individual medical or workload "
        "decisions."
    )
