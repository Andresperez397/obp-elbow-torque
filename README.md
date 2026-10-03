# Predicting elbow varus torque in pitchers the model has never seen

**What it is:** a pre-registered analysis of how much of a pitcher's peak elbow varus torque can be predicted from body size, velocity and mechanics. It also measures how badly the usual validation shortcut overstates the answer.

**Data:** 411 fastballs from 100 pitchers in the [OpenBiomechanics Project](https://www.openbiomechanics.org) (Driveline Baseball). Torque comes from marker-based lab motion capture and inverse dynamics.

**Author:** Andres Perez, M.S. Kinesiology (Biomechanics)

![Model ladder](reports/figures/fig1_model_ladder.png)

## Key findings

**1. Torque is a pitcher trait.**
- 95% of the variance in elbow varus torque lies between pitchers (ICC 0.95, 95% CI 0.93–0.96).
- A pitcher's fastballs differ from one another by about 4 Nm. Pitchers differ from each other by about 19 Nm.
- So the useful prediction target is the *pitcher*, and validation has to hold out whole pitchers.

**2. On pitchers the model has never seen, mechanics roughly double what body size explains.**

| Model (validated on held-out pitchers) | R² | 95% CI | RMSE (Nm) |
|---|---|---|---|
| Body size + handedness | 0.35 | 0.12–0.50 | 16.0 |
| + velocity | 0.45 | 0.27–0.57 | 14.8 |
| + 10 literature-chosen mechanics | 0.53 | 0.36–0.63 | 13.6 |
| **All 42 kinematic metrics (elastic net)** | **0.63** | **0.52–0.72** | **12.0** |
| All kinematics (gradient-boosted trees) | 0.51 | 0.36–0.61 | 14.0 |
| + lower-body kinetics and ground reaction forces (elastic net) | 0.63 | 0.50–0.72 | 12.1 |

Under the pre-declared rule, a block of information counts as adding something only if the 95% interval of its R² gain excludes zero:
- Velocity adds information over body size (+0.10).
- The full kinematic set adds information (+0.19 over velocity, +0.12 over the literature subset).
- The 10 literature mechanics on their own do not clear the bar (+0.08, CI −0.05 to 0.19).
- Lower-body kinetics and ground reaction forces add nothing on top of kinematics.

**3. Leaky validation inflates the result and reverses the model ranking.**
- If pitches are split at random, so a pitcher's other fastballs sit in the training set, gradient-boosted trees look like the best model (R² 0.87).
- On new pitchers the same model reaches only R² 0.51, worse than a linear elastic net.
- The trees were memorizing pitchers, not learning mechanics.
- Leakage inflation is 0.04 to 0.16 for the linear models and 0.37 to 0.39 for the trees.

**4. Velocity costs about 1.3 Nm per mph, within and between pitchers.**
- Adjusted for body size and handedness, a pitcher who throws 1 mph harder than another carries 1.29 Nm more torque (CI 0.68–1.91).
- When the *same* pitcher throws 1 mph harder, torque rises 1.25 Nm (CI 0.46–2.03).
- Caveat: within-pitcher speed varied little (SD 0.5 mph in one session), so the within estimate is imprecise.

![Velocity slopes](reports/figures/fig5_velocity_slopes.png)

**5. Ratio normalization (torque ÷ body weight × height) is not rejected in this sample.**
- Pitcher-mean torque against BW×H has an intercept of 21 Nm (95% CI −2 to 43).
- The interval includes zero, which proportional scaling requires, but only barely.
- Normalized torque is still weakly negatively correlated with BW×H (r = −0.15).
- So normalization over-corrects slightly for bigger pitchers. That is worth checking in any dataset before normalizing.

![Normalization](reports/figures/fig4_normalization.png)

## How the analysis was done

- **The plan came first.** [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) fixed the questions, predictor blocks, models, metrics and decision rules. It was committed to git before any outcome model was fit, and the commit history shows the order. Post-hoc changes are logged in [`DEVIATIONS.md`](DEVIATIONS.md).
- **No leakage predictors.** Shoulder internal-rotation moment and the throwing-arm energy-flow terms come from the same inverse-dynamics solution as the outcome, so they are excluded.
- **Grouped validation.** 10-fold cross-validation by pitcher, repeated 20 times. All preprocessing (winsorizing, imputation, scaling) and elastic-net tuning (inner grouped CV) is fit inside the training folds.
- **Honest uncertainty.** 95% intervals come from a 2,000-resample bootstrap over pitchers, not pitches.
- **Mixed models.** A random-intercept model gives the ICC, with a parametric-bootstrap CI. A within/between decomposition separates the two velocity effects.
- **Tests.** `tests/` checks:
  - the join keeps every pitch
  - no outcome or leakage column enters any block
  - grouped folds never share a pitcher
  - preprocessing uses only training data
  - the ICC estimator recovers a known value in simulation

### Exploratory (not pre-registered): what the elastic net relies on

The table below comes from 500 pitcher-bootstrap refits of the all-kinematics model, using standardized predictors. It describes what the model leans on, not causal effects.
- The most stable predictors are body mass, velocity, and *lower* peak shoulder external rotation (layback) at a given velocity.
- Next come glove-arm and throwing-arm shoulder abduction at foot plant, and trunk lateral tilt.
- See [`reports/tables/exploratory_enet_coefficients.csv`](reports/tables/exploratory_enet_coefficients.csv).

## Limitations

- **Sample:** 100 mostly college pitchers (75 college, 12 independent, 7 high school, 6 MiLB) from one private training facility, one lab session each, fastballs only. Do not assume the results transfer to MLB or in-game data.
- **The outcome is modeled:** torque is an inverse-dynamics estimate. Its level depends on a segment-inertia model that scales with body mass, so part of the body-size effect is built in by construction.
- **Small sample:** with 100 pitchers, R² gains below about 0.05 can't be reliably separated from zero. Treat the confidence intervals as the result.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py          # downloads the OBP files (not redistributed here)
python scripts/run_analysis.py        # about 5 minutes; writes reports/tables/
python scripts/make_figures.py
python scripts/exploratory_coefficients.py   # optional
pytest -q
```

## Licensing and attribution

- **Data:** The OpenBiomechanics Project, Driveline Baseball, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) with an additional professional-organization exclusion (see the OBP license). The raw data are not redistributed here.
- **Derived results:** the tables and figures in `reports/` are derived works, shared under CC BY-NC-SA 4.0.
- **Code:** MIT (see `LICENSE`).
- This project is not affiliated with or endorsed by Driveline Baseball.
