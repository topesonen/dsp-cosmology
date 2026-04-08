# SBI Explained For This Project

This note explains what we are doing with simulation-based inference (SBI) in this project, how the workflow is implemented, and why we structured it the way we did.

The short version is:

We are not running a new physical simulator inside `sbi`.

Instead, we first build a catalogue of Local Group analogue pairs from the cosmological simulation. Each row in that catalogue already contains:

- observed-like quantities `x`, such as pair separation and velocities
- hidden quantities `theta`, such as total dark matter mass or halo `M200c`

That means we can treat the catalogue as an empirical sample from a joint distribution

`p(theta, x | S)`

where `S` is the whole analogue-selection pipeline.

Then SBI learns a conditional density for

`p(theta | x_obs, S)`.

In words:

Given a Local Group-like observed system `x_obs`, and given the way we selected analogue pairs, what does the simulation imply about the hidden quantity `theta`?

## Common Idea Behind Both Versions

Both the condensed workflow and the larger workflow use the same scientific idea:

1. Load the TNG subhalo catalogue.
2. Select relevant objects with the analogue pipeline.
3. Find candidate pairs and apply pair-level cuts.
4. Build a per-pair catalogue.
5. Choose:
   `x` = the observable features we condition on.
6. Choose:
   `theta` = the hidden quantity we want to infer.
7. Train Neural Posterior Estimation (NPE) with `sbi`.
8. Evaluate the posterior at a chosen observation `x_obs`.

So the difference between the two versions is not the scientific target.
The difference is how much machinery we wrap around the core NPE workflow.

## 1. Condensed Practical-Guide Iteration

Files:

- [`scripts/lg_sbi_practical_guide.py`](/home/tp/Documents/uni/dsp/scripts/lg_sbi_practical_guide.py)
- [`notebooks/run_sbi_practical_guide_style.ipynb`](/home/tp/Documents/uni/dsp/notebooks/run_sbi_practical_guide_style.ipynb)

### What This Version Does

This version is intentionally simple and stays close to the style used in the `sbi-practical-guide` examples.

The notebook does the important `sbi` steps directly:

1. build the Local Group pair catalogue
2. choose `ACTIVE_FEATURES`, `TARGET`, and `x_obs`
3. prepare `X` and `theta`
4. convert them to torch tensors
5. create `posterior_nn(...)`
6. create `NPE(...)`
7. call `append_simulations(...)`
8. call `train(...)`
9. call `build_posterior(...)`
10. sample from the posterior at `x_obs`

That directness is the main point of this version.

### How It Works

The thin helper file only does a few lightweight tasks:

- build the observation vector from feature metadata
- add `M200c`-based columns to the pair catalogue
- extract a clean `(X, theta)` problem from the catalogue
- summarize 1D posterior samples
- make simple sanity plots

The actual SBI training logic is left in the notebook on purpose.

This is closer to how the `sbi` guide teaches the workflow:

- keep the tensors visible
- keep the training call visible
- keep the posterior construction visible

### Why We Made This Version

This version is useful when the main goal is understanding.

It makes the flow easier to read:

- where the data come from
- what exactly counts as `x`
- what exactly counts as `theta`
- where `sbi` starts
- what the model is actually trained on

It is also easier to compare mentally with standard SBI tutorials, because the notebook itself contains the core NPE calls instead of hiding them behind a bigger wrapper.

### Why This Version Is Good For Us

For this project, the biggest conceptual hurdle is often not the neural density estimator itself.
It is understanding that our “simulator output” is really a selected catalogue of analogue pairs.

The condensed version makes that clearer:

- the simulation is TNG
- the analogue-selection pipeline creates the empirical dataset
- `sbi` learns from that dataset

That makes it a better teaching and debugging starting point.

### Limitations Of The Condensed Version

This version is intentionally minimal.
So it does less for us automatically.

It does not wrap:

- train/calibration splitting
- support checks
- SBC-style diagnostics
- expected coverage diagnostics
- approximate posterior predictive summaries

It is best when we want a clean baseline implementation that is easy to inspect and reason about.

## 2. Larger Wrapped Workflow

Files:

- [`scripts/lg_sbi.py`](/home/tp/Documents/uni/dsp/scripts/lg_sbi.py)
- [`notebooks/run_simulations.ipynb`](/home/tp/Documents/uni/dsp/notebooks/run_simulations.ipynb)
- [`notebooks/run_sbi_tuning.ipynb`](/home/tp/Documents/uni/dsp/notebooks/run_sbi_tuning.ipynb)

### What This Version Does

This version still uses NPE, but it wraps much more of the workflow into reusable helper functions.

Its main functions are:

- `prepare_sbi_problem(...)`
- `fit_catalog_npe(...)`
- `infer_with_npe(...)`
- plotting helpers for posterior and diagnostics

So instead of exposing the raw `sbi` flow in the notebook, this version moves more logic into a project helper module.

### How It Works

`prepare_sbi_problem(...)`:

- extracts the requested feature columns from the pair catalogue
- extracts the target `theta`
- optionally log-transforms features and/or target
- filters out non-finite rows

`fit_catalog_npe(...)`:

- splits the data into training and calibration subsets
- builds the neural density estimator
- trains the NPE model
- builds the posterior
- computes some diagnostics on held-out data

`infer_with_npe(...)`:

- evaluates the trained posterior at `x_obs`
- draws posterior samples
- summarizes the posterior
- performs a support check
- builds an approximate posterior predictive in feature space

### Why This Version Exists

The larger version is more useful once we move beyond “does NPE run?” and into “can we trust this result?” or “can we compare it cleanly with other methods?”

That matters because our project is not only about running SBI.
It is about comparing several inference approaches on the same analogue catalogue:

- ABC kernel
- GMM
- SBI / NPE

For that setting, convenience and repeated diagnostics matter a lot.

### What The Extra Machinery Buys Us

The larger workflow adds several things that are useful in practice:

Train/calibration split:

- lets us inspect held-out behaviour instead of only trusting the training fit

Support check:

- tells us whether `x_obs` lies inside the empirical support of the catalogue
- helps flag extrapolation problems

Marginal SBC-style rank check:

- gives a rough calibration diagnostic on held-out rows

Expected coverage for 1D theta:

- checks whether nominal credible intervals are approximately calibrated

Approximate posterior predictive:

- gives a rough feature-space sanity check, even though NPE directly models `p(theta | x)` rather than `p(x | theta)`

### Why This Bigger Version Is Still Reasonable

Even though it is more complex, the bigger version is still scientifically consistent with the condensed one.

It is still catalogue-based SBI.

It still learns from rows of `(theta, x)` that come from the selected pair catalogue.

The difference is that it adds more project-specific structure around that core learning problem.

### Trade-Offs

The larger version is better for:

- repeated experiments
- method comparison notebooks
- diagnostics
- reusable project helpers

The larger version is worse for:

- quickly seeing the essential `sbi` calls
- mapping the workflow onto tutorial examples
- teaching the basic idea to someone new to SBI

So it is more convenient, but less transparent.

## Why We Use NPE At All

NPE is a good fit here because our target is a posterior over a hidden quantity given observed features.

That is exactly what NPE is designed for:

- input: many examples of `(theta, x)`
- output: a learned approximation to `p(theta | x)`

In our case, the catalogue already gives us those examples after selection.

This is especially attractive because:

- we do not need to hand-design a kernel over `theta`
- we get a full posterior, not only point estimates
- nonlinear relationships between `x` and `theta` can be learned flexibly

## What This Is Not

It is important not to oversell what we are doing.

This is not a full simulator-in-the-loop SBI setup where we:

- propose a parameter
- run a forward physical simulator
- compare synthetic observations to real data
- adaptively simulate again

Instead, we are doing empirical, catalogue-based SBI on a bank of already-simulated examples.

That is still a legitimate SBI-style workflow, but it is conceptually simpler:

- TNG provides the simulated examples
- our analogue-selection pipeline defines the conditioning context `S`
- `sbi` learns the conditional mapping from the retained rows

## Which Version To Use

Use the condensed practical-guide-style version when:

- you want the clearest conceptual picture
- you want something close to tutorial code
- you want to debug the core NPE flow
- you want a simple baseline

Use the larger wrapped workflow when:

- you want diagnostics built in
- you want to compare SBI with ABC/KDE/GMM in one notebook
- you want reusable helper functions
- you want a more feature-complete project workflow

## Bottom Line

The condensed version is the best explanation of what we are doing.

The larger version is the best working project tool.

They solve the same inference problem, but at different levels of abstraction:

- the condensed version is closest to the underlying SBI idea
- the larger version is closest to the needs of this project
