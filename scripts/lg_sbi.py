
"""Notebook-friendly SBI helpers for Local Group analogue studies.

This module uses Neural Posterior Estimation (NPE) with normalizing flows
through the `sbi` package.

Why this is a reasonable fit here
---------------------------------
After the analogue-selection pipeline S, the pair catalogue already contains
joint samples of:

- theta: the hidden property to infer, such as log10(total mass)
- x: observed-like features, such as r_kpc, v_r, v_t

That means the catalogue can be treated as an empirical sample from the
joint distribution p(theta, x | S). The NPE model then learns a conditional
density q_phi(theta | x) directly from those rows.

This is not a full forward-simulator workflow with new simulations generated
on demand. It is a catalogue-based version of the same idea:
learn the conditional distribution from simulated examples, then evaluate it
at x_obs.

Practical choices in this helper
--------------------------------
The structure follows the practical advice in the SBI guide:

- start with NPE using normalizing flows
- keep a held-out calibration split
- use validation during training
- run simple diagnostics before trusting the posterior
- check whether x_obs is inside the empirical support of the simulation bank

Important limitation
--------------------
NPE returns a posterior over theta directly.
Unlike KDE/ABC, it does not naturally return exact weights over pair IDs.
If the notebook needs explicit pair weights, use the KDE helper from
`lg_gmm_kde.py` alongside this module.

Installation
------------
This helper expects:
    pip install sbi

Minimal notebook usage
----------------------
from lg_abc import build_pair_catalog
from lg_sbi import (
    prepare_sbi_problem,
    fit_catalog_npe,
    infer_with_npe,
    plot_sbi_theta_posterior,
    plot_sbc_rank_hist,
    plot_expected_coverage,
)

cat = build_pair_catalog(sample=sample, pairs=pipeline.pairs, sub=sub, sfr_field="SubhaloSFRinRad")

problem = prepare_sbi_problem(
    catalog=cat,
    features=["r_kpc", "v_r", "v_t"],
    target="mdm_sum",
    target_log10=True,
)

x_obs = np.array([770.0, -109.0, 17.0], dtype=float)

fit = fit_catalog_npe(
    X=problem["X"],
    theta=problem["theta"],
    x_obs=x_obs,
    feature_names=problem["feature_names"],
    target_name=problem["target_name"],
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

result = infer_with_npe(fit, x_obs=x_obs, num_samples=20_000, ppc_draws=500)
print(result["summary"])
print(fit["diagnostics"])
plot_sbi_theta_posterior(problem["theta"], result["samples"])
"""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


def _as_1d(values: np.ndarray | Iterable[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return arr


def _as_2d(values: np.ndarray | Iterable[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (or 1D, which is promoted to 2D)")
    return arr


def _safe_log10(arr: np.ndarray, name: str) -> np.ndarray:
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} contains non-positive values; cannot apply log10")
    return np.log10(arr)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = _as_1d(values, name="values")
    weights = _as_1d(weights, name="weights")
    if values.size != weights.size:
        raise ValueError("values and weights must have the same length")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")

    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(mask):
        return float("nan")

    v = values[mask]
    w = weights[mask]
    total = w.sum()
    if total <= 0.0:
        return float("nan")

    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / total
    cdf = np.r_[0.0, cdf]
    vpad = np.r_[v[0], v]
    return float(np.interp(q, cdf, vpad))


def summarize_1d(values: np.ndarray) -> dict[str, float]:
    values = _as_1d(values, name="values")
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "q16": float("nan"), "q84": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.quantile(values, 0.5)),
        "q16": float(np.quantile(values, 0.16)),
        "q84": float(np.quantile(values, 0.84)),
    }


def prepare_sbi_problem(
    catalog: dict[str, np.ndarray],
    features: list[str],
    target: str,
    *,
    log10_features: set[str] | None = None,
    target_log10: bool = False,
    drop_nonfinite: bool = True,
) -> dict[str, Any]:
    """Extract X and theta arrays from a pair catalogue."""
    if not features:
        raise ValueError("features must contain at least one name")
    if target not in catalog:
        raise KeyError(f"target '{target}' not found in catalog")

    log10_features = set() if log10_features is None else set(log10_features)

    cols: list[np.ndarray] = []
    n_rows: int | None = None
    for name in features:
        if name not in catalog:
            raise KeyError(f"feature '{name}' not found in catalog")
        col = _as_1d(catalog[name], name=name).astype(np.float64, copy=False)
        if name in log10_features:
            col = _safe_log10(col, name)
        if n_rows is None:
            n_rows = col.size
        elif col.size != n_rows:
            raise ValueError("all feature columns must have the same length")
        cols.append(col)

    theta = _as_1d(catalog[target], name=target).astype(np.float64, copy=False)
    if target_log10:
        theta = _safe_log10(theta, target)
    if n_rows is None:
        raise ValueError("could not infer row count")
    if theta.size != n_rows:
        raise ValueError("target length does not match feature lengths")

    X = np.column_stack(cols)
    keep_mask = np.ones(n_rows, dtype=bool)
    if drop_nonfinite:
        keep_mask &= np.all(np.isfinite(X), axis=1)
        keep_mask &= np.isfinite(theta)

    X = X[keep_mask]
    theta = theta[keep_mask]
    dropped_rows = int(n_rows - np.sum(keep_mask))

    if X.shape[0] == 0:
        raise ValueError("no rows remain after filtering non-finite values")

    return {
        "X": X,
        "theta": theta,
        "feature_names": list(features),
        "target_name": target,
        "feature_log10": sorted(log10_features),
        "target_log10": bool(target_log10),
        "keep_mask": keep_mask,
        "dropped_rows": dropped_rows,
    }


def support_check(X: np.ndarray, x_obs: np.ndarray, scale: np.ndarray | None = None) -> dict[str, Any]:
    """Basic misspecification check in feature space."""
    X = _as_2d(X, name="X")
    x_obs = _as_1d(x_obs, name="x_obs")
    if X.shape[1] != x_obs.size:
        raise ValueError("dimension mismatch between X and x_obs")

    if scale is None:
        scale = np.std(X, axis=0)
    scale = _as_1d(scale, name="scale")
    if scale.size != X.shape[1]:
        raise ValueError("scale must have one entry per feature")
    scale = np.where(scale <= 0.0, 1.0, scale)

    feature_min = X.min(axis=0)
    feature_max = X.max(axis=0)
    outside = (x_obs < feature_min) | (x_obs > feature_max)

    z = (X - x_obs[None, :]) / scale[None, :]
    d = np.sqrt(np.sum(z * z, axis=1))
    nn_index = int(np.argmin(d))
    return {
        "outside_range": outside,
        "n_outside": int(np.sum(outside)),
        "feature_min": feature_min,
        "feature_max": feature_max,
        "nearest_distance": float(d[nn_index]),
        "nearest_index": nn_index,
        "nearest_x": X[nn_index].copy(),
    }


def _train_calibration_split(
    n: int,
    *,
    calibration_fraction: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in (0, 1)")
    if n < 20:
        raise ValueError("at least 20 rows are recommended for the SBI helper")

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    n_cal = int(np.round(calibration_fraction * n))
    n_cal = min(max(n_cal, 5), n - 5)
    cal_idx = np.sort(perm[:n_cal])
    train_idx = np.sort(perm[n_cal:])
    return train_idx, cal_idx


def _import_sbi():
    try:
        import torch
        from sbi.inference import NPE
        from sbi.neural_nets import posterior_nn
    except ImportError as exc:
        raise ImportError(
            "lg_sbi.py requires the `sbi` package. Install it in the notebook environment with `pip install sbi`."
        ) from exc
    return torch, NPE, posterior_nn


def _to_tensor_2d(torch_mod, arr: np.ndarray, device: str):
    arr = _as_2d(arr, name="arr")
    return torch_mod.as_tensor(arr, dtype=torch_mod.float32, device=device)


def _posterior_log_prob_mean(posterior, theta_cal_t, x_cal_t) -> float:
    values: list[float] = []
    for i in range(theta_cal_t.shape[0]):
        lp = posterior.log_prob(theta_cal_t[i : i + 1], x=x_cal_t[i])
        values.append(float(lp.detach().cpu().numpy().reshape(-1)[0]))
    return float(np.mean(values)) if values else float("nan")


def _run_marginal_sbc(
    posterior,
    torch_mod,
    theta_cal_t,
    x_cal_t,
    *,
    num_eval: int,
    num_posterior_samples: int,
    random_state: int | None,
) -> np.ndarray:
    """Approximate marginal SBC ranks on a held-out calibration split."""
    rng = np.random.default_rng(random_state)
    n_cal = theta_cal_t.shape[0]
    num_eval = min(int(num_eval), n_cal)
    if num_eval <= 0:
        return np.empty((0, theta_cal_t.shape[1]), dtype=np.int64)

    chosen = rng.choice(n_cal, size=num_eval, replace=False)
    ranks = np.zeros((num_eval, theta_cal_t.shape[1]), dtype=np.int64)

    for out_i, cal_i in enumerate(chosen):
        x_i = x_cal_t[cal_i]
        theta_true = theta_cal_t[cal_i].detach().cpu().numpy()
        samples = posterior.sample((int(num_posterior_samples),), x=x_i)
        samples_np = samples.detach().cpu().numpy()
        ranks[out_i] = np.sum(samples_np < theta_true[None, :], axis=0)

    return ranks


def _sbc_summary(ranks: np.ndarray, num_posterior_samples: int) -> dict[str, Any]:
    if ranks.size == 0:
        return {"num_eval": 0}
    fractions = ranks / float(num_posterior_samples)
    return {
        "num_eval": int(ranks.shape[0]),
        "num_posterior_samples": int(num_posterior_samples),
        "mean_rank_fraction": np.mean(fractions, axis=0),
        "std_rank_fraction": np.std(fractions, axis=0),
    }


def _run_expected_coverage_1d(
    posterior,
    theta_cal_t,
    x_cal_t,
    *,
    num_eval: int,
    num_posterior_samples: int,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9),
    random_state: int | None,
) -> dict[str, Any]:
    """Simple expected-coverage estimate for a 1D theta."""
    rng = np.random.default_rng(random_state)
    n_cal = theta_cal_t.shape[0]
    num_eval = min(int(num_eval), n_cal)
    if num_eval <= 0:
        return {"num_eval": 0, "levels": np.asarray(levels, dtype=np.float64), "empirical": np.asarray([])}

    chosen = rng.choice(n_cal, size=num_eval, replace=False)
    hits = np.zeros(len(levels), dtype=np.float64)

    for cal_i in chosen:
        x_i = x_cal_t[cal_i]
        theta_true = float(theta_cal_t[cal_i].detach().cpu().numpy().reshape(-1)[0])
        samples = posterior.sample((int(num_posterior_samples),), x=x_i).detach().cpu().numpy().reshape(-1)

        for j, level in enumerate(levels):
            alpha = 0.5 * (1.0 - level)
            lo = float(np.quantile(samples, alpha))
            hi = float(np.quantile(samples, 1.0 - alpha))
            hits[j] += float(lo <= theta_true <= hi)

    empirical = hits / float(num_eval)
    return {
        "num_eval": int(num_eval),
        "num_posterior_samples": int(num_posterior_samples),
        "levels": np.asarray(levels, dtype=np.float64),
        "empirical": empirical,
    }


def fit_catalog_npe(
    X: np.ndarray,
    theta: np.ndarray,
    *,
    x_obs: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    target_name: str = "theta",
    density_model: str = "nsf",
    hidden_features: int = 64,
    num_transforms: int = 5,
    training_batch_size: int = 128,
    learning_rate: float = 5e-4,
    validation_fraction: float = 0.1,
    calibration_fraction: float = 0.15,
    stop_after_epochs: int = 30,
    max_num_epochs: int = 300,
    clip_max_norm: float | None = 5.0,
    random_state: int | None = 0,
    device: str | None = None,
    sbc_num_eval: int = 50,
    sbc_num_posterior_samples: int = 200,
    coverage_num_eval: int = 50,
    coverage_num_posterior_samples: int = 500,
    show_train_summary: bool = False,
) -> dict[str, Any]:
    """Fit NPE on a catalogue of joint samples (theta, X).

    Returns
    -------
    fit:
        Dict containing the trained posterior object, the held-out diagnostics,
        and the training/calibration split.

    Notes
    -----
    - The catalogue itself defines the empirical prior under the selection S.
    - `prior=None` is used in the NPE trainer because the goal here is direct
      conditional density estimation on the empirical joint sample.
    """
    X = _as_2d(X, name="X")
    theta = _as_1d(theta, name="theta")
    if X.shape[0] != theta.size:
        raise ValueError("X and theta must have the same number of rows")

    torch_mod, NPE, posterior_nn = _import_sbi()
    if device is None:
        device = "cuda" if torch_mod.cuda.is_available() else "cpu"

    train_idx, cal_idx = _train_calibration_split(
        X.shape[0],
        calibration_fraction=calibration_fraction,
        random_state=random_state,
    )

    X_train = X[train_idx]
    theta_train = theta[train_idx][:, None]
    X_cal = X[cal_idx]
    theta_cal = theta[cal_idx][:, None]

    density_estimator = posterior_nn(
        model=density_model,
        z_score_theta="independent",
        z_score_x="independent",
        hidden_features=int(hidden_features),
        num_transforms=int(num_transforms),
    )

    inference = NPE(prior=None, density_estimator=density_estimator, device=device)

    theta_train_t = _to_tensor_2d(torch_mod, theta_train, device)
    X_train_t = _to_tensor_2d(torch_mod, X_train, device)
    theta_cal_t = _to_tensor_2d(torch_mod, theta_cal, device)
    X_cal_t = _to_tensor_2d(torch_mod, X_cal, device)

    inference.append_simulations(theta_train_t, X_train_t)
    neural_posterior = inference.train(
        training_batch_size=int(training_batch_size),
        learning_rate=float(learning_rate),
        validation_fraction=float(validation_fraction),
        stop_after_epochs=int(stop_after_epochs),
        max_num_epochs=int(max_num_epochs),
        clip_max_norm=clip_max_norm,
        show_train_summary=bool(show_train_summary),
    )
    posterior = inference.build_posterior(neural_posterior, sample_with="direct")

    diagnostics: dict[str, Any] = {
        "calibration_size": int(theta_cal.shape[0]),
        "train_size": int(theta_train.shape[0]),
        "mean_calibration_log_prob": _posterior_log_prob_mean(posterior, theta_cal_t, X_cal_t),
    }

    ranks = _run_marginal_sbc(
        posterior,
        torch_mod,
        theta_cal_t,
        X_cal_t,
        num_eval=sbc_num_eval,
        num_posterior_samples=sbc_num_posterior_samples,
        random_state=random_state,
    )
    diagnostics["sbc_ranks"] = ranks
    diagnostics["sbc_summary"] = _sbc_summary(ranks, num_posterior_samples=sbc_num_posterior_samples)

    if theta_train.shape[1] == 1:
        diagnostics["expected_coverage_1d"] = _run_expected_coverage_1d(
            posterior,
            theta_cal_t,
            X_cal_t,
            num_eval=coverage_num_eval,
            num_posterior_samples=coverage_num_posterior_samples,
            random_state=random_state,
        )

    if x_obs is not None:
        diagnostics["x_obs_support_check"] = support_check(X_train, x_obs)

    return {
        "method": "sbi_npe",
        "posterior": posterior,
        "inference": inference,
        "device": device,
        "X_train": X_train,
        "theta_train": theta_train.reshape(-1),
        "X_calibration": X_cal,
        "theta_calibration": theta_cal.reshape(-1),
        "feature_names": feature_names if feature_names is not None else [f"x{j}" for j in range(X.shape[1])],
        "target_name": target_name,
        "diagnostics": diagnostics,
        "training_config": {
            "density_model": density_model,
            "hidden_features": int(hidden_features),
            "num_transforms": int(num_transforms),
            "training_batch_size": int(training_batch_size),
            "learning_rate": float(learning_rate),
            "validation_fraction": float(validation_fraction),
            "calibration_fraction": float(calibration_fraction),
            "stop_after_epochs": int(stop_after_epochs),
            "max_num_epochs": int(max_num_epochs),
            "device": device,
        },
    }


def approximate_catalog_posterior_predictive(
    theta_train: np.ndarray,
    X_train: np.ndarray,
    theta_posterior_samples: np.ndarray,
    *,
    num_draws: int = 500,
    num_neighbors: int = 25,
    random_state: int | None = 0,
) -> np.ndarray:
    """Approximate posterior predictive X samples using the catalogue itself.

    Because NPE learns q(theta | x) directly, it does not provide p(x | theta).
    This helper uses a simple empirical approximation:
    for each posterior theta sample, find nearby theta values in the training
    catalogue and resample one of their X rows.

    This is a heuristic.
    It is useful as a quick sanity check, not as a formal posterior predictive.
    """
    theta_train = _as_2d(theta_train, name="theta_train")
    X_train = _as_2d(X_train, name="X_train")
    theta_posterior_samples = _as_2d(theta_posterior_samples, name="theta_posterior_samples")

    if theta_train.shape[0] != X_train.shape[0]:
        raise ValueError("theta_train and X_train must have the same number of rows")
    if theta_train.shape[1] != theta_posterior_samples.shape[1]:
        raise ValueError("theta dimension mismatch between training data and posterior samples")

    num_draws = min(int(num_draws), theta_posterior_samples.shape[0])
    num_neighbors = max(1, min(int(num_neighbors), theta_train.shape[0]))

    rng = np.random.default_rng(random_state)
    chosen = rng.choice(theta_posterior_samples.shape[0], size=num_draws, replace=False)

    pred = np.zeros((num_draws, X_train.shape[1]), dtype=np.float64)
    for out_i, sample_i in enumerate(chosen):
        theta_i = theta_posterior_samples[sample_i : sample_i + 1]
        d = np.sqrt(np.sum((theta_train - theta_i) ** 2, axis=1))
        nn = np.argsort(d, kind="mergesort")[:num_neighbors]
        picked = int(rng.choice(nn))
        pred[out_i] = X_train[picked]
    return pred


def infer_with_npe(
    fit: dict[str, Any],
    *,
    x_obs: np.ndarray,
    num_samples: int = 20_000,
    random_state: int | None = 0,
    ppc_draws: int = 500,
    ppc_neighbors: int = 25,
) -> dict[str, Any]:
    """Evaluate the trained NPE posterior at x_obs."""
    torch_mod, _, _ = _import_sbi()

    x_obs = _as_1d(x_obs, name="x_obs")
    feature_names = fit["feature_names"]
    if len(feature_names) != x_obs.size:
        raise ValueError("x_obs length must match the number of feature names")

    posterior = fit["posterior"]
    device = fit["device"]

    if random_state is not None:
        np.random.seed(random_state)
        torch_mod.manual_seed(int(random_state))

    x_obs_t = torch_mod.as_tensor(x_obs, dtype=torch_mod.float32, device=device)
    samples_t = posterior.sample((int(num_samples),), x=x_obs_t)
    samples = samples_t.detach().cpu().numpy()
    if samples.ndim == 2 and samples.shape[1] == 1:
        samples_1d = samples[:, 0]
    else:
        samples_1d = samples.reshape(-1)

    summary = summarize_1d(samples_1d)
    support = support_check(fit["X_train"], x_obs)

    x_pred = approximate_catalog_posterior_predictive(
        fit["theta_train"][:, None],
        fit["X_train"],
        samples if samples.ndim == 2 else samples[:, None],
        num_draws=ppc_draws,
        num_neighbors=ppc_neighbors,
        random_state=random_state,
    )

    ppc_summary: dict[str, dict[str, float]] = {}
    x_obs_percentile: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        ppc_summary[name] = summarize_1d(x_pred[:, j])
        x_obs_percentile[name] = float(np.mean(x_pred[:, j] <= x_obs[j]))

    return {
        "method": "sbi_npe",
        "samples": samples_1d,
        "summary": summary,
        "x_obs": x_obs.copy(),
        "support_check": support,
        "x_predictive_samples": x_pred,
        "posterior_predictive_summary": ppc_summary,
        "x_obs_predictive_percentile": x_obs_percentile,
    }


def plot_sbi_theta_posterior(
    theta_prior: np.ndarray,
    posterior_samples: np.ndarray,
    *,
    bins: int = 40,
    title: str = "SBI posterior over theta",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the baseline theta distribution and the SBI posterior samples."""
    theta_prior = _as_1d(theta_prior, name="theta_prior")
    posterior_samples = _as_1d(posterior_samples, name="posterior_samples")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(theta_prior, bins=bins, density=True, alpha=0.35, label="all pairs after S")
    ax.hist(posterior_samples, bins=bins, density=True, alpha=0.55, label="SBI posterior samples")
    ax.set_xlabel("theta")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig, ax


def plot_sbc_rank_hist(
    ranks: np.ndarray,
    *,
    num_posterior_samples: int,
    parameter_label: str = "theta",
    bins: int = 20,
    title: str = "Held-out marginal SBC ranks",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a simple SBC rank histogram for the first marginal."""
    ranks = np.asarray(ranks)
    if ranks.ndim != 2 or ranks.shape[1] < 1:
        raise ValueError("ranks must have shape (n_eval, n_params)")
    first = ranks[:, 0] / float(num_posterior_samples)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(first, bins=bins, range=(0.0, 1.0), alpha=0.75)
    ax.axhline(len(first) / bins, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"rank fraction for {parameter_label}")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return fig, ax


def plot_expected_coverage(
    coverage_result: dict[str, Any],
    *,
    title: str = "Held-out expected coverage",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot nominal vs empirical coverage for the 1D expected-coverage check."""
    levels = np.asarray(coverage_result.get("levels", []), dtype=np.float64)
    empirical = np.asarray(coverage_result.get("empirical", []), dtype=np.float64)
    if levels.size == 0 or empirical.size == 0:
        raise ValueError("coverage_result does not contain usable coverage data")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="k", linewidth=1.0)
    ax.plot(levels, empirical, marker="o")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return fig, ax
