# Diagnostics And Quality Plots Guide

This note explains the diagnostic and quality plots used in our current
`run_simulations.ipynb` workflow for:

- `ABC kernel`
- `GMM`
- `SBI / NPE`

It focuses on three practical questions:

1. Is the posterior shape sensible?
2. Is the method calibrated (where applicable)?
3. How well does it predict held-out analogue rows?


## 1) Posterior Comparison Plots

### What is plotted
- Baseline histogram: all retained analogue pairs after selection and color filtering.
- Method posteriors: `ABC kernel`, `GMM`, and `SBI`.

### What it tells us
- How much each method updates away from the baseline.
- Whether methods broadly agree on central tendency and spread.
- Whether one method is much sharper/wider than the others.

### How to interpret
- Good sign: posteriors are different from baseline but still overlap in plausible regions.
- Caution sign: one method is extremely narrow while others are broad.
- Caution sign: modes are far apart with no overlap; inspect features/support next.


## 2) Feature-Space Support And Predictive Checks

### Support check (SBI)
The support table reports for each feature:
- observed value `x_obs`
- training min/max range
- whether `x_obs` is outside range
- nearest in-sample feature vector

### What it tells us
- If `x_obs` is outside training support, inference is extrapolative.

### How to interpret
- Good sign: `outside_range=False` for most/all features and nearest distance is moderate.
- Caution sign: multiple `outside_range=True` entries or very large nearest distance.

### Feature-space comparison (ABC/SBI)
- ABC row: histogram reweighted by kernel weights.
- SBI row: approximate posterior predictive feature samples.

These are sanity checks for whether conditioning behaves as expected in feature space.


## 3) SBI Calibration Plots

SBI has explicit held-out calibration diagnostics from `fit_catalog_npe(...)`.

### Mean calibration log-probability
- Scalar summary of held-out fit quality.
- Higher (less negative) generally indicates better held-out density fit.
- Best used for relative comparison across SBI runs with the same data split.

### SBC rank histogram (marginal)
- Tests whether posterior ranks on held-out rows are approximately uniform.
- Uniform-ish histogram means calibration is roughly reasonable.

Interpretation:
- U-shape: posterior too narrow (overconfident).
- Hump in center: posterior too wide (underconfident).
- Strong skew: systematic bias.

### Expected coverage plot
- Compares nominal interval coverage (x-axis) vs empirical coverage (y-axis).
- Ideal is near diagonal `y=x`.

Interpretation:
- Curve below diagonal: intervals too narrow (undercover).
- Curve above diagonal: intervals too wide (overcover).


## 4) GMM Fit-Quality Diagnostics

When `n_components` is selected by `bic` or `aic`, we log:
- score vs number of components
- selected component count

### What it tells us
- Whether model complexity selection is stable.
- Whether the chosen component count is near a clear optimum.

### How to interpret
- Clear minimum with margin: stable complexity choice.
- Flat/noisy curve: multiple complexities are similarly plausible.
- Sharp swings: potential sensitivity to random seed or preprocessing.


## 5) Residual Diagnostics On Held-Out Analogues

These are pseudo-observation diagnostics:
- split retained analogue rows into train/test
- fit methods on train
- predict posterior mean for each test `x_i`
- residual: `theta_true - theta_pred_mean`

### Metrics
- `MAE`: mean absolute error (lower is better).
- `RMSE`: root mean squared error (lower is better, penalizes large misses more).
- `R2`: variance explained relative to mean baseline (higher is better).

`R2` interpretation:
- `R2 = 1`: perfect.
- `R2 = 0`: no better than predicting test-set mean.
- `R2 < 0`: worse than mean baseline.

### Residual histogram
- Center near zero is good.
- Narrower spread is better.
- Heavy tails suggest occasional large misses.

### Residual vs predicted-mean scatter
- Should look like noise around zero, not a strong trend.
- Funnel shape can indicate heteroscedasticity.
- Curvature can indicate model misspecification.

### R2 bar chart
- Quick cross-method summary on the same test split.
- Useful for relative ranking, not as an absolute truth metric.


## 6) Why Loss Curves Are Uneven Across Methods

- `ABC kernel`: no optimizer, so no training loss exists.
- `GMM`: optimization exists, but in this notebook we expose complexity diagnostics (`bic/aic`) rather than iterative loss traces.
- `SBI`: training is optimization-based, but current helper output does not expose full epoch-by-epoch losses; we rely on held-out calibration diagnostics instead.

So “loss diagnostics” are method-specific:
- explicit N/A where no loss is meaningful
- proxy quality diagnostics where direct loss traces are unavailable


## 7) Practical Reading Order

Use this order when reviewing a run:

1. Posterior comparison plot (overall behavior).
2. Support check and feature-space sanity (is inference in-domain?).
3. SBI calibration plots (is neural posterior calibrated?).
4. Residual metrics and residual plots (held-out predictive quality).
5. GMM criterion trajectory and selected components (fit stability).

If methods disagree strongly:
- inspect support first
- then check calibration/coverage
- then compare residual metrics and tails


## 8) Important Caveat

Residual and `R2` diagnostics here are computed on held-out simulation analogues,
not on the real Local Group target (which is unknown).

So these diagnostics answer:
- “How well does each method generalize within this simulation-derived dataset?”

They do **not** by themselves prove correctness for the real universe.
