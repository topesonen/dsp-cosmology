"""Helpers for comparing halo-mass posteriors across feature configurations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analogues import AnalogueSample, FilterConfig, SelectionConfig
from lg_abc import run_abc_catalog
from lg_gmm_kde import (
    condition_gmm,
    fit_joint_gmm,
    prepare_density_problem,
    support_check,
    weighted_gaussian_kde_1d,
)
from lg_sbi import fit_catalog_npe, infer_with_npe


DEFAULT_SBI_CONFIG: dict[str, Any] = {
    "density_model": "maf",
    "hidden_features": 128,
    "num_transforms": 4,
    "training_batch_size": 256,
    "learning_rate": 3e-4,
    "validation_fraction": 0.1,
    "calibration_fraction": 0.2,
    "stop_after_epochs": 30,
    "max_num_epochs": 300,
    "random_state": 42,
    "show_train_summary": False,
}

PAIR_SUBSET_LABELS = {
    "all": "All pairs",
    "blue_blue": "Blue-Blue",
    "red_red": "Red-Red",
    "blue_or_red": "Blue-Blue + Red-Red",
}


def resolve_base_path(repo_root: Path) -> str:
    raw_candidates = [
        os.environ.get("TNG_BASE_PATH"),
        os.environ.get("ILLUSTRIS_BASE_PATH"),
        str(repo_root / "tng300" / "outputs"),
    ]

    for raw in raw_candidates:
        if not raw:
            continue
        base = Path(raw).expanduser()
        for candidate in (base, base / "tng300" / "outputs"):
            if (candidate / "groups_099").exists():
                return str(candidate.resolve())

    raise FileNotFoundError(
        "Could not find a valid TNG outputs directory. Set TNG_BASE_PATH or "
        "ILLUSTRIS_BASE_PATH to the outputs directory, or to a root that contains "
        "tng300/outputs."
    )


def build_analogue_sample(
    repo_root: Path,
    *,
    selection_config: dict[str, Any],
    filter_config: dict[str, Any],
    snap: int = 99,
    verbose: bool = True,
) -> AnalogueSample:
    return AnalogueSample(
        base_path=resolve_base_path(repo_root),
        selection_config=SelectionConfig(**selection_config),
        filter_config=FilterConfig(**filter_config),
        snap=snap,
        verbose=verbose,
    )


def pair_subset_display_name(pair_subset: str) -> str:
    if pair_subset not in PAIR_SUBSET_LABELS:
        raise ValueError(
            f"Unsupported pair_subset '{pair_subset}'. "
            f"Choose from {sorted(PAIR_SUBSET_LABELS)}."
        )
    return PAIR_SUBSET_LABELS[pair_subset]


def build_pair_catalog(
    analogue_sample: AnalogueSample,
    *,
    pair_subset: str = "blue_or_red",
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    sample = analogue_sample._sample
    pairs = analogue_sample.pairs

    is_blue_blue = np.asarray(pairs.is_blue_blue, dtype=bool)
    is_red_red = np.asarray(pairs.is_red_red, dtype=bool)
    is_blue_red = np.asarray(pairs.is_blue_red, dtype=bool)

    if pair_subset == "all":
        keep_mask = np.ones(pairs.i.size, dtype=bool)
    elif pair_subset == "blue_blue":
        keep_mask = is_blue_blue
    elif pair_subset == "red_red":
        keep_mask = is_red_red
    elif pair_subset == "blue_or_red":
        keep_mask = is_blue_blue | is_red_red
    else:
        raise ValueError(
            f"Unsupported pair_subset '{pair_subset}'. "
            f"Choose from {sorted(PAIR_SUBSET_LABELS)}."
        )

    pair_i = pairs.i
    pair_j = pairs.j

    mstar_i = sample.m_stellar[pair_i].astype(np.float64, copy=False)
    mstar_j = sample.m_stellar[pair_j].astype(np.float64, copy=False)
    mdm_i = sample.m_dark_matter[pair_i].astype(np.float64, copy=False)
    mdm_j = sample.m_dark_matter[pair_j].astype(np.float64, copy=False)
    k_band_i = sample.k_band[pair_i].astype(np.float64, copy=False)
    k_band_j = sample.k_band[pair_j].astype(np.float64, copy=False)

    catalog = {
        "r_kpc": pairs.separation.astype(np.float64, copy=False),
        "v_r": pairs.vr.astype(np.float64, copy=False),
        "v_t": pairs.vt.astype(np.float64, copy=False),
        "same_host": pairs.have_same_host.astype(np.float64, copy=False),
        "mstar_i": mstar_i,
        "mstar_j": mstar_j,
        "k_band_small": k_band_i,
        "k_band_large": k_band_j,
        "mstar_big": np.maximum(mstar_i, mstar_j),
        "mstar_small": np.minimum(mstar_i, mstar_j),
        "mdm_i": mdm_i,
        "mdm_j": mdm_j,
        "mdm_sum": mdm_i + mdm_j,
        "mdm_big": np.maximum(mdm_i, mdm_j),
        "mdm_small": np.minimum(mdm_i, mdm_j),
        "dm_mass_ratio": np.maximum(mdm_i, mdm_j)
        / np.clip(np.minimum(mdm_i, mdm_j), 1e-30, None),
        "mstar_sum": mstar_i + mstar_j,
        "log_tot_virial_mass": pairs.log_tot_virial_mass.astype(np.float64, copy=False),
        "pair_is_red_red": is_red_red.astype(np.float64),
        "pair_is_blue_blue": is_blue_blue.astype(np.float64),
    }

    for key in list(catalog.keys()):
        catalog[key] = np.asarray(catalog[key])[keep_mask]

    color_summary = pd.DataFrame(
        [
            {
                "pair_subset": pair_subset_display_name(pair_subset),
                "total_pairs_before_subset": int(pairs.i.size),
                "blue_blue": int(is_blue_blue.sum()),
                "red_red": int(is_red_red.sum()),
                "mixed_blue_red": int(is_blue_red.sum()),
                "pairs_after_subset": int(keep_mask.sum()),
            }
        ]
    )

    return catalog, color_summary


def build_x_obs_and_sigma(
    feature_info: dict[str, dict[str, Any]],
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_obs = np.array([feature_info[name]["obs"] for name in features], dtype=float)
    sigma = np.array([feature_info[name]["sigma"] for name in features], dtype=float)
    return x_obs, sigma


def run_single_configuration(
    *,
    catalog: dict[str, np.ndarray],
    feature_info: dict[str, dict[str, Any]],
    features: list[str],
    target_column: str,
    target_label: str,
    method: str = "abc",
    label: str | None = None,
    log10_features: set[str] | None = None,
    target_log10: bool = False,
    abc_accept_frac: float = 0.05,
    abc_kernel_bandwidth: float = 1.0,
    gmm_components: int | str = "bic",
    gmm_random_state: int = 42,
    sbi_config: dict[str, Any] | None = None,
    num_samples: int = 20_000,
    random_state: int = 42,
) -> dict[str, Any]:
    if not features:
        raise ValueError("features must contain at least one feature name")

    x_obs, sigma = build_x_obs_and_sigma(feature_info, features)
    problem = prepare_density_problem(
        catalog=catalog,
        features=features,
        target=target_column,
        log10_features=log10_features,
        target_log10=target_log10,
    )

    X = problem["X"]
    theta = problem["theta"]
    method_key = method.lower()

    if method_key == "abc":
        raw_result = run_abc_catalog(
            X=X,
            theta=theta,
            x_obs=x_obs,
            sigma=sigma,
            accept_frac=abc_accept_frac,
            kernel_bandwidth=abc_kernel_bandwidth,
        )
        theta_grid, theta_density = weighted_gaussian_kde_1d(
            theta,
            weights=raw_result["weights"],
        )
        summary = dict(raw_result["summary_kernel"])
        support = support_check(X, x_obs, scale=sigma)
    elif method_key == "gmm":
        gmm_fit = fit_joint_gmm(
            X=X,
            theta=theta,
            n_components=gmm_components,
            random_state=gmm_random_state,
        )
        raw_result = condition_gmm(
            gmm_fit,
            x_obs=x_obs,
            n_samples=num_samples,
            random_state=random_state,
        )
        theta_grid = raw_result["theta_grid"]
        theta_density = raw_result["theta_density"]
        summary = dict(raw_result["summary"])
        support = raw_result["support_check"]
    elif method_key == "sbi":
        fit_kwargs = dict(DEFAULT_SBI_CONFIG)
        if sbi_config:
            fit_kwargs.update(sbi_config)
        fit = fit_catalog_npe(
            X=X,
            theta=theta,
            x_obs=x_obs,
            feature_names=problem["feature_names"],
            target_name=target_label,
            **fit_kwargs,
        )
        raw_result = infer_with_npe(
            fit,
            x_obs=x_obs,
            num_samples=num_samples,
            random_state=random_state,
            ppc_draws=500,
            ppc_neighbors=25,
        )
        theta_grid, theta_density = weighted_gaussian_kde_1d(raw_result["samples"])
        summary = dict(raw_result["summary"])
        support = raw_result["support_check"]
    else:
        raise ValueError("method must be one of {'abc', 'gmm', 'sbi'}")

    return {
        "label": label if label is not None else " + ".join(features),
        "features": list(features),
        "method": method_key,
        "x_obs": x_obs,
        "sigma": sigma,
        "problem": problem,
        "summary": summary,
        "support_check": support,
        "theta_grid": theta_grid,
        "theta_density": theta_density,
        "raw_result": raw_result,
    }


def run_configuration_grid(
    *,
    catalog: dict[str, np.ndarray],
    feature_info: dict[str, dict[str, Any]],
    configurations: list[dict[str, Any]],
    target_column: str,
    target_label: str,
    method: str = "abc",
    log10_features: set[str] | None = None,
    target_log10: bool = False,
    abc_accept_frac: float = 0.05,
    abc_kernel_bandwidth: float = 1.0,
    gmm_components: int | str = "bic",
    gmm_random_state: int = 42,
    sbi_config: dict[str, Any] | None = None,
    num_samples: int = 20_000,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for config in configurations:
        label = config.get("label")
        features = list(config["features"])
        result = run_single_configuration(
            catalog=catalog,
            feature_info=feature_info,
            features=features,
            target_column=target_column,
            target_label=target_label,
            method=method,
            label=label,
            log10_features=log10_features,
            target_log10=target_log10,
            abc_accept_frac=abc_accept_frac,
            abc_kernel_bandwidth=abc_kernel_bandwidth,
            gmm_components=gmm_components,
            gmm_random_state=gmm_random_state,
            sbi_config=sbi_config,
            num_samples=num_samples,
            random_state=random_state,
        )
        results.append(result)

        summary = result["summary"]
        support = result["support_check"]
        rows.append(
            {
                "label": result["label"],
                "method": result["method"],
                "features": ", ".join(result["features"]),
                "n_features": len(result["features"]),
                "n_pairs_used": int(result["problem"]["X"].shape[0]),
                "dropped_nonfinite": int(result["problem"]["dropped_rows"]),
                "support_n_outside": int(support["n_outside"]),
                "support_nearest_distance": float(support["nearest_distance"]),
                "mean": float(summary["mean"]),
                "median": float(summary["median"]),
                "q16": float(summary["q16"]),
                "q84": float(summary["q84"]),
                "ess": float(summary.get("ess", np.nan)),
            }
        )

    return results, pd.DataFrame(rows)


def plot_configuration_posteriors(
    results: list[dict[str, Any]],
    *,
    title: str,
    x_label: str,
    reference_value: float | None = None,
    reference_label: str | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)

    for result in results:
        ax.plot(
            result["theta_grid"],
            result["theta_density"],
            linewidth=2.2,
            label=result["label"],
        )

    if reference_value is not None:
        ax.axvline(
            float(reference_value),
            color="black",
            linestyle="--",
            linewidth=1.6,
            label=reference_label if reference_label else f"Reference = {reference_value:.2f}",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig, ax
