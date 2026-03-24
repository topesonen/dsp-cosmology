
"""Minimal example for running the new LG helpers after pair finding.

Assumed objects already exist in the notebook:
- sample
- pipeline.pairs   or pair_set after the final pair-level cuts
- sub              optional, only needed if the catalogue also includes SFR fields

This example mirrors the variable names used in run_abc_explained.
"""

import numpy as np

from lg_abc import build_pair_catalog
from lg_gmm_kde import (
    prepare_density_problem,
    run_kde_conditioning,
    fit_joint_gmm,
    condition_gmm,
    posterior_predictive_from_weights,
    plot_theta_posteriors,
    plot_feature_posteriors,
)
from lg_sbi import (
    prepare_sbi_problem,
    fit_catalog_npe,
    infer_with_npe,
    plot_sbi_theta_posterior,
    plot_sbc_rank_hist,
    plot_expected_coverage,
)

# 1) Build a per-pair catalogue.
cat = build_pair_catalog(
    sample=sample,
    pairs=pipeline.pairs,
    sub=sub,
    sfr_field="SubhaloSFRinRad",
)

# 2) Choose features X and target theta.
features = ["r_kpc", "v_r", "v_t"]
target = "mdm_sum"

x_obs = np.array([770.0, -109.0, 17.0], dtype=float)
sigma = np.array([50.0, 20.0, 30.0], dtype=float)

# 3) KDE + GMM.
problem = prepare_density_problem(
    catalog=cat,
    features=features,
    target=target,
    target_log10=True,
)

kde_result = run_kde_conditioning(
    problem["X"],
    problem["theta"],
    x_obs=x_obs,
    bandwidth=sigma,
)
print("KDE posterior:", kde_result["summary"])
print("KDE predictive X:", posterior_predictive_from_weights(problem["X"], kde_result["weights"], feature_names=features))

gmm_fit = fit_joint_gmm(
    problem["X"],
    problem["theta"],
    n_components="bic",
    component_range=range(1, 9),
    random_state=0,
)
gmm_result = condition_gmm(
    gmm_fit,
    x_obs=x_obs,
    n_samples=20_000,
    random_state=0,
)
print("GMM posterior:", gmm_result["summary"])

plot_theta_posteriors(problem["theta"], kde_result=kde_result, gmm_result=gmm_result)
plot_feature_posteriors(problem["X"], features, x_obs=x_obs, kde_result=kde_result)

# 4) SBI / NPE with normalizing flows.
#    This needs: pip install sbi
sbi_problem = prepare_sbi_problem(
    catalog=cat,
    features=features,
    target=target,
    target_log10=True,
)

sbi_fit = fit_catalog_npe(
    X=sbi_problem["X"],
    theta=sbi_problem["theta"],
    x_obs=x_obs,
    feature_names=sbi_problem["feature_names"],
    target_name=sbi_problem["target_name"],
    density_model="nsf",
    hidden_features=64,
    num_transforms=5,
    training_batch_size=128,
    learning_rate=5e-4,
    validation_fraction=0.1,
    calibration_fraction=0.15,
    stop_after_epochs=30,
    max_num_epochs=300,
    random_state=0,
)
print("SBI diagnostics:", sbi_fit["diagnostics"])

sbi_result = infer_with_npe(
    sbi_fit,
    x_obs=x_obs,
    num_samples=20_000,
    ppc_draws=500,
    ppc_neighbors=25,
    random_state=0,
)
print("SBI posterior:", sbi_result["summary"])
print("SBI approximate predictive X:", sbi_result["posterior_predictive_summary"])

plot_sbi_theta_posterior(sbi_problem["theta"], sbi_result["samples"])

ranks = sbi_fit["diagnostics"]["sbc_ranks"]
if ranks.size > 0:
    plot_sbc_rank_hist(
        ranks,
        num_posterior_samples=sbi_fit["diagnostics"]["sbc_summary"]["num_posterior_samples"],
        parameter_label="log10(mdm_sum)",
    )

if "expected_coverage_1d" in sbi_fit["diagnostics"]:
    plot_expected_coverage(sbi_fit["diagnostics"]["expected_coverage_1d"])
