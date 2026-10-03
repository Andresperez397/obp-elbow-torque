# Analysis Plan (frozen before any outcome modeling)

**Project:** How much of a pitcher's elbow varus torque can we predict, and from what?
**Data:** The OpenBiomechanics Project (OBP), baseball pitching point-of-interest (POI) metrics, Driveline Baseball. Licensed CC BY-NC-SA 4.0.
**Author:** Andres Perez
**Plan written:** 2026-10-02. It was committed to git before the analysis code was run, so the commit history is the audit trail.

The only data inspection done before writing this plan was a check of the shape, column names, units, missingness and summary statistics. No outcome model was fit, and no correlation with the outcome was examined.

## 1. Questions

- **Q1. Where does the variation live?** What share of the variance in peak elbow varus torque (EVT) lies between pitchers, and what share is pitch-to-pitch variation within a pitcher?
- **Q2. How well can EVT be predicted for a pitcher the model has never seen?** How much does each block of information add: body size, then velocity, then a small set of mechanics chosen from the literature, then all kinematics, then lower-body kinetics?
- **Q3. How much does the validation scheme matter?** How much does random pitch-level cross-validation (which leaks a pitcher's own pitches into training) inflate apparent accuracy, compared with grouped, leave-pitchers-out validation?
- **Q4. Velocity and torque within and between pitchers.** Is the torque cost of 1 mph the same when one pitcher throws harder (within) as when comparing harder- and softer-throwing pitchers (between)?
- **Q5. Is ratio normalization justified?** The common practice of normalizing torque by body weight × height assumes torque is proportional to BW×H, meaning a regression line through the origin. We test that assumption.

## 2. Data and units

- **Unit of analysis:** the pitch, nested within a pitcher. The dataset has 411 fastballs from 100 pitchers, 2 to 5 pitches each, one session per pitcher.
- **Outcome:** `elbow_varus_moment`, the peak elbow varus moment in Nm. This is an inverse-dynamics estimate from marker-based lab motion capture, not a direct measurement.
- **Body size:** `session_mass_kg` and `session_height_m` come from `metadata.csv`, joined on `session_pitch`.

## 3. Predictor blocks (pre-specified)

**Excluded as leakage**, because they come from the same throwing-arm inverse-dynamics solution as the outcome:
- `shoulder_internal_rotation_moment`
- all `shoulder_*` and `elbow_*` energy-flow terms
- `thorax_distal_transfer_fp_br`

Using them would "predict" torque from torque.

| Model | Adds | Variables |
|---|---|---|
| M0 | none | training-fold mean |
| M1 | body size + handedness | mass, height, `p_throws` |
| M2 | velocity | M1 + `pitch_speed_mph` |
| M3 | literature mechanics (10) | M2 + `max_shoulder_external_rotation`, `elbow_flexion_mer`, `shoulder_abduction_fp`, `arm_slot`, `torso_lateral_tilt_br`, `torso_rotation_fp`, `timing_peak_torso_to_peak_pelvis_rot_velo`, `max_torso_rotational_velo`, `max_pelvis_rotational_velo`, `stride_length` |
| M4 | all kinematics | M2 + every kinematic POI (angles, angular velocities, timing, center-of-gravity velocity, stride) |
| M5 | lower-body kinetics | M4 + lower-body energy-flow terms (`lead_*`, `rear_*`, `pelvis_lumbar_transfer_fp_br`) + ground reaction forces + rate of force development |

**Why the M3 variables?** Each has a published mechanical link to elbow loading:
- elbow flexion and shoulder abduction (Aguinaldo & Chambers 2009)
- contralateral trunk tilt (Solomito et al. 2015)
- shoulder external rotation and trunk/pelvis rotation and timing (Fleisig et al. 1995; Aguinaldo & Chambers 2009)
- arm slot and stride length, as common coaching variables

**Learners:**
- M1 to M3: ordinary least squares.
- M4 and M5: elastic net, plus a gradient-boosted trees model (scikit-learn `HistGradientBoostingRegressor`) as a nonlinear check.
- Elastic-net penalties are tuned by inner grouped cross-validation. GBM hyperparameters are fixed in advance: `max_depth=3`, `learning_rate=0.05`, `max_iter=300`, `min_samples_leaf=20`.

**Preprocessing**, fit inside each training fold only:
- winsorize every continuous predictor at its training-fold 1st and 99th percentiles
- median-impute the 8 pitches that lack force-plate data (M5 only)
- standardize

## 4. Validation

- **Primary scheme:** GroupKFold by pitcher, 10 folds, repeated 20 times with different random pitcher-to-fold assignments. No pitcher's pitches ever appear in both training and test.
- **Primary metrics:** out-of-fold RMSE (Nm) and R² (1 − SSE/SST, using the training-fold mean as the M0 reference), averaged over repeats.
- **Uncertainty:** a pitcher-level cluster bootstrap (2,000 resamples of pitchers) on the out-of-fold predictions of one fixed repeat (seed 0), giving 95% intervals for RMSE, R², and the gain over the previous block.
- **Pitcher-level view:** the same metrics computed on each pitcher's mean torque against the mean prediction, reported as secondary.
- **Leakage comparison (Q3):** the same models under ordinary 10-fold KFold over pitches, 20 repeats. The difference in R² is the leakage inflation.

## 5. Variance and velocity models (Q1, Q4)

- **Q1:** a random-intercept mixed model, EVT ~ 1 + (1 | pitcher), fit by REML. ICC = between variance / total variance, with a 95% CI from a parametric bootstrap (1,000 resamples).
- **Q4:** EVT ~ speed_between + speed_within + mass + height + handedness + (1 | pitcher), where speed_between is the pitcher-mean speed and speed_within = speed − pitcher mean. We report both slopes in Nm per mph with 95% Wald CIs. Within-pitcher speed ranges are small (fastballs from one session), so the within slope is expected to be imprecise, and we report it as such.

## 6. Normalization (Q5)

Using pitcher-mean data (one row per pitcher), we fit EVT = a + b·(BW×H), with BW in newtons and H in metres:
- If a is clearly non-zero (its 95% CI excludes 0), ratio normalization does not remove body size and is not appropriate.
- We also report the correlation between normalized torque (EVT / BW×H) and BW×H. Proportional scaling predicts zero.

## 7. Pre-declared interpretation rules

- A block "adds information" only if the 95% bootstrap interval of its R² gain excludes zero.
- We do not choose models by test performance. All six models are reported regardless of their results.
- Any change to this plan after results are seen is logged as a dated deviation in `DEVIATIONS.md`, with its reason, and both versions are reported.

## 8. Known limitations (stated in advance)

- The pitchers are mostly college athletes (314 of 411 pitches) training at one private facility, and the data come from a single lab session per pitcher. Results may not transfer to professional or in-game settings.
- EVT is a model output (inverse dynamics) whose absolute level depends on the segment-inertia model, which scales with body mass. Some association with body size is built in by construction.
- With 100 pitchers, gains smaller than a few points of R² cannot be reliably distinguished from zero.
