# dsp-cosmology

Priors for unknown properties of the Local Group from large cosmological simulations.

This project uses the TNG300 group catalogue to build simulated Local Group analogue pairs and infer hidden quantities, such as total virial mass, from observed-like constraints such as pair separation, relative velocity, and stellar masses. The main inference workflow compares Approximate Bayesian Computation (ABC), Gaussian Mixture Models (GMM), and Simulation-Based Inference (SBI / NPE) on the same selected analogue catalogue.

## Repository Layout

- `analogues/`: reusable analogue-selection pipeline, pair finding, filters, and data containers.
- `scripts/`: inference helpers for ABC, GMM/KDE, SBI, Optuna tuning, and multi-configuration posterior sweeps.
- `config/config.json`: shared analysis settings for selection cuts, feature metadata, and target metadata.
- `notebooks/generate_analogues.ipynb`: quick notebook for building and inspecting the analogue sample.
- `notebooks/run_simulations.ipynb`: main ABC/GMM/SBI comparison notebook.
- `notebooks/run_multi_config_posteriors.ipynb`: compact notebook for comparing posterior curves across different feature configurations.
- `sbi_explained.md` and `diagnostics_explained.md`: short notes explaining the SBI workflow and diagnostics.

## Setup

The notebooks expect the TNG300 snapshot 99 group catalogue to be available locally. By default, the code looks for:

```text
tng300/outputs/groups_099/
```

relative to the repository root. You can also set either `TNG_BASE_PATH` or `ILLUSTRIS_BASE_PATH` to point to the TNG outputs directory.

Typical environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install sbi
```

The project uses `illustris_python` for TNG catalogue access, plus common scientific Python packages such as `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `sbi`, and `optuna`.

## How `config/config.json` Works

The shared config file controls most of the scientific choices used by the notebooks.

`selection_config` defines the first-stage galaxy/sample cuts, for example stellar-mass limits, pair-separation limits, and the blue/red colour threshold.

`filter_config` defines pair-level cuts after candidate pairs are generated, for example allowed radial/tangential/total velocity ranges and isolation settings.

`feature_info` defines the observable-like conditioning variables. Each feature entry contains:

- `obs`: observed Local Group value used as `x_obs`
- `sigma`: observational uncertainty used by ABC-style distance weighting
- `active`: whether the feature is included by config-driven notebooks
- `label` and `description`: display metadata for tables and plots

`target_presets` defines possible inferred quantities. Exactly one target must have `"active": true`; `config.loader.load_config()` raises an error otherwise. The current default target is `total_virial_mass`, which points to the pair-level `log_tot_virial_mass` column.

Important detail: `run_simulations.ipynb` uses the active features from `feature_info`, while `run_multi_config_posteriors.ipynb` uses explicit feature sets inside the notebook so it can compare several configurations without toggling `active` flags.

## `run_simulations.ipynb`

This is the main end-to-end comparison notebook. It:

1. loads `config/config.json`;
2. resolves the local TNG outputs path;
3. rebuilds the analogue sample using `analogues.AnalogueSample`;
4. keeps blue-blue and red-red pairs and removes mixed blue-red pairs;
5. builds the final pair catalogue and feature matrix;
6. runs ABC kernel weighting, GMM conditioning, and SBI/NPE on the same `(X, theta, x_obs)` problem;
7. produces posterior summaries, posterior plots, feature-space checks, support checks, SBI calibration diagnostics, GMM model-selection diagnostics, and held-out residual diagnostics.

Use this notebook when you want the full method comparison and diagnostics for one active feature set from the config.

## `run_multi_config_posteriors.ipynb`

This notebook is a lighter plotting workflow for comparing posterior curves across several feature sets. It builds the analogue catalogue once, then reruns a selected inference method for configurations such as:

- separation only
- separation + radial velocity
- separation + radial velocity + tangential velocity
- separation + velocities + stellar masses

The main settings are in the notebook's settings cell:

- `PAIR_SUBSET`: `"blue_blue"`, `"red_red"`, `"blue_or_red"`, or `"all"`
- `INFERENCE_METHOD`: `"abc"`, `"gmm"`, or `"sbi"`
- `FEATURE_CONFIGS`: list of feature groups to compare

Use this notebook when you want a compact figure showing how the inferred halo-mass posterior changes as more observational constraints are added.

## Inference Methods

All three inference methods use the same retained analogue-pair catalogue and aim to estimate:

```text
p(theta | x_obs, S)
```

where `S` is the analogue-selection pipeline, `x_obs` is the observed Local Group feature vector, and `theta` is the hidden quantity such as total virial mass.

- ABC reweights existing analogue rows by distance to `x_obs`, scaled by observational uncertainties.
- GMM fits a Gaussian mixture to the joint distribution of `(theta, X)` and conditions analytically on `x_obs`.
- SBI/NPE trains a neural conditional density estimator to sample directly from the learned posterior.

The methods are intentionally compared together: ABC is transparent, GMM is a smooth parametric middle ground, and SBI is the most flexible but needs stronger diagnostics.

## Useful References

- TNG300 data: https://www.tng-project.org/data/downloads/TNG300-1/
- TNG data specifications: https://www.tng-project.org/data/docs/specifications/#sec2a
- TNG example scripts: https://www.tng-project.org/data/docs/scripts/
