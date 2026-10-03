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
