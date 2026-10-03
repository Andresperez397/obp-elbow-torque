# Deviations from ANALYSIS_PLAN.md

Every change made after the plan was frozen is logged here, dated, with its reason.

## D1 (2026-10-02): added an exploratory coefficient analysis

- **What:** `scripts/exploratory_coefficients.py` refits the M4 elastic net on 500 pitcher-bootstrap samples. It reports each standardized coefficient, with a percentile interval, the share of resamples that selected it, and the share that kept its sign.
- **Why:** after seeing that M4 (elastic net) was the best model on unseen pitchers, the natural next question is which variables it relies on. The plan did not include this.
- **Status:** exploratory and descriptive only. It does not change any pre-specified result, and it is labelled as exploratory wherever it appears. The coefficients describe a regularized predictive model, not causal effects.

## Implementation fixes (no change to the analysis)

- **2026-10-02:** scikit-learn 1.9 renamed `ElasticNetCV(n_alphas=...)` to `alphas=<int>`. Same grid of 50 penalties.
- **2026-10-02:** pinned BLAS/OpenMP to 1 thread because nested threading made repeated fits very slow. Results are unaffected.
- **2026-10-02:** fixed a duplicate-column bug in the pitcher-level metric for the M0 row. The run crashed before writing any results, so no result changed.

## D2 (2026-10-02): citation rationale corrected (variables unchanged)

Checking `ANALYSIS_PLAN.md` §3 against the source abstracts on PubMed showed two imprecise attributions:
- **Aguinaldo & Chambers 2009** (doi:10.1177/0363546509336721) reports associations of elbow valgus torque with maximum shoulder external rotation, elbow flexion, trunk-rotation onset and arm slot (sidearm higher). It does **not** report shoulder abduction, which the plan attributed to it.
- **Fleisig et al. 1995** (doi:10.1177/036354659502300218) is a kinetics description (peak varus torque shortly before maximum external rotation). It is weak support for trunk and pelvis rotation timing.
- **Solomito et al. 2015** (doi:10.1177/0363546515574060) is cited correctly: contralateral trunk lean raises the elbow varus moment.

The M3 variable list is unchanged. Shoulder abduction at foot plant stays in as a common coaching variable without a specific citation, and the frozen plan text is left as written. The corrected references are listed in the README.

## D3 (2026-10-02): added a confidence interval for one reported correlation

`normalization_test` now also returns a Fisher-z 95% CI for the correlation between normalized torque and BW×H (r = −0.15, CI −0.34 to 0.04). The README no longer describes that correlation as an over-correction. It is not distinguishable from zero. No other result changed.

## Code-quality changes (outputs verified byte-identical)

- **2026-10-02:** refactored the model factory (no string sentinel) and closed files properly.
- **2026-10-02:** added tests: held-out pitcher corruption for every learner, out-of-fold mean, whole-pitcher bootstrap.
- **2026-10-02:** added the ruff config.
- Re-running the analysis reproduced every table byte for byte.
