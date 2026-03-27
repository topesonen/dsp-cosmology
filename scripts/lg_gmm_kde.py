
"""Notebook-friendly GMM + KDE helpers for Local Group analogue studies.

This module is meant to run after pair finding and pair-level filtering.
The basic idea is:

1. Start from a pair catalogue where each row is one simulated Local Group analogue.
2. Choose observed-like features X, for example:
   - r_kpc
   - v_r
   - v_t
3. Choose a target theta, for example:
   - log10(mdm_sum)
4. Condition on an observed feature vector x_obs.
5. Get an estimate of p(theta | x_obs, S), where S is the analogue-selection pipeline.

Two estimators are included here:

- KDE-style conditioning:
  Uses a Gaussian kernel in feature space.
  This is close in spirit to kernel ABC, but phrased as conditional density estimation.
  It gives pair weights directly, so it is useful when the notebook wants a posterior over pairs.

- Joint GMM:
  Fits a Gaussian mixture model to the empirical joint sample [theta, X].
  Conditioning on x_obs then gives an analytic Gaussian-mixture approximation
  to p(theta | x_obs, S).

Design choices:
- The code is intentionally explicit and notebook-friendly.
- The target theta is assumed to be 1D for the summary and plotting helpers.
  This matches the first scientific targets such as total mass, mass ratio, or energy proxy.
- The feature matrix X can have any number of columns.

Minimal notebook usage
----------------------
from lg_abc import build_pair_catalog
from lg_gmm_kde import (
    prepare_density_problem,
    run_kde_conditioning,
    fit_joint_gmm,
    condition_gmm,
    plot_theta_posteriors,
    plot_feature_posteriors,
)

cat = build_pair_catalog(sample=sample, pairs=pipeline.pairs, sub=sub, sfr_field="SubhaloSFRinRad")

problem = prepare_density_problem(
    catalog=cat,
    features=["r_kpc", "v_r", "v_t"],
    target="mdm_sum",
    target_log10=True,
)

x_obs = np.array([770.0, -109.0, 17.0], dtype=float)
bandwidth = np.array([50.0, 20.0, 30.0], dtype=float)

kde_result = run_kde_conditioning(problem["X"], problem["theta"], x_obs=x_obs, bandwidth=bandwidth)

gmm_fit = fit_joint_gmm(problem["X"], problem["theta"], n_components="bic", random_state=0)
gmm_result = condition_gmm(gmm_fit, x_obs=x_obs, n_samples=20_000, random_state=0)

print(kde_result["summary"])
print(gmm_result["summary"])
plot_theta_posteriors(problem["theta"], kde_result=kde_result, gmm_result=gmm_result)
plot_feature_posteriors(problem["X"], problem["feature_names"], x_obs=x_obs, kde_result=kde_result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


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


def summarize_1d(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    values = _as_1d(values, name="values")
    if values.size == 0:
        out = {"mean": float("nan"), "median": float("nan"), "q16": float("nan"), "q84": float("nan")}
        if weights is not None:
            out["ess"] = 0.0
        return out

    if weights is None:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.quantile(values, 0.5)),
            "q16": float(np.quantile(values, 0.16)),
            "q84": float(np.quantile(values, 0.84)),
        }

    weights = _as_1d(weights, name="weights")
    if weights.size != values.size:
        raise ValueError("values and weights must have the same length")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite and non-negative")

    total = weights.sum()
    if total <= 0.0:
        return {"mean": float("nan"), "median": float("nan"), "q16": float("nan"), "q84": float("nan"), "ess": 0.0}

    wn = weights / total
    ess = 1.0 / float(np.sum(wn * wn))
    return {
        "mean": float(np.sum(wn * values)),
        "median": weighted_quantile(values, wn, 0.5),
        "q16": weighted_quantile(values, wn, 0.16),
        "q84": weighted_quantile(values, wn, 0.84),
        "ess": float(ess),
    }


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        x = _as_2d(x, name="x")
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std = np.where(std <= 0.0, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = _as_2d(x, name="x")
        return (x - self.mean[None, :]) / self.std[None, :]

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = _as_2d(x, name="x")
        return x * self.std[None, :] + self.mean[None, :]


def prepare_density_problem(
    catalog: dict[str, np.ndarray],
    features: list[str],
    target: str,
    *,
    log10_features: set[str] | None = None,
    target_log10: bool = False,
    drop_nonfinite: bool = True,
) -> dict[str, Any]:
    """Extract X and theta arrays from a pair catalogue.

    Parameters
    ----------
    catalog:
        Dict of per-pair arrays. The usual input is the output of
        `lg_abc.build_pair_catalog(...)`.
    features:
        Ordered feature names for X.
    target:
        One target column name for theta.
    log10_features:
        Optional subset of feature names to transform with base-10 log.
    target_log10:
        If True, apply base-10 log to theta.
    drop_nonfinite:
        If True, rows with NaN or inf in X or theta are removed.

    Returns
    -------
    problem:
        Dict with keys:
        - X
        - theta
        - feature_names
        - target_name
        - keep_mask
        - dropped_rows
    """
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
    """Basic misspecification check in feature space.

    This is intentionally simple:
    - checks whether x_obs is outside the min/max range in any feature
    - computes the nearest standardized distance to the empirical sample

    This is not a proof that the model is well specified.
    It is only a quick warning light.
    """
    X = _as_2d(X, name="X")
    x_obs = _as_1d(x_obs, name="x_obs")
    if X.shape[1] != x_obs.size:
        raise ValueError("dimension mismatch between X and x_obs")

    if scale is None:
        scale = np.std(X, axis=0)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.ndim != 1 or scale.size != X.shape[1]:
        raise ValueError("scale must be a 1D array with one entry per feature")
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


def _gaussian_kernel_weights(X: np.ndarray, x_obs: np.ndarray, bandwidth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = _as_2d(X, name="X")
    x_obs = _as_1d(x_obs, name="x_obs")
    bandwidth = _as_1d(bandwidth, name="bandwidth")
    if X.shape[1] != x_obs.size or X.shape[1] != bandwidth.size:
        raise ValueError("dimension mismatch among X, x_obs, and bandwidth")
    if np.any(bandwidth <= 0.0) or not np.all(np.isfinite(bandwidth)):
        raise ValueError("bandwidth must be finite and strictly positive")

    z = (X - x_obs[None, :]) / bandwidth[None, :]
    logw = -0.5 * np.sum(z * z, axis=1)

    finite = np.isfinite(logw)
    if not np.any(finite):
        return np.zeros(X.shape[0], dtype=np.float64), np.full(X.shape[0], np.inf, dtype=np.float64)

    max_logw = np.max(logw[finite])
    w = np.zeros(X.shape[0], dtype=np.float64)
    w[finite] = np.exp(logw[finite] - max_logw)
    total = np.sum(w)
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros(X.shape[0], dtype=np.float64), np.sqrt(np.sum(z * z, axis=1))

    w /= total
    d = np.sqrt(np.sum(z * z, axis=1))
    return w, d


def _silverman_bandwidth_1d(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = _as_1d(values, name="values")
    if values.size < 2:
        return 1.0

    if weights is None:
        std = float(np.std(values, ddof=1))
        n_eff = float(values.size)
    else:
        weights = _as_1d(weights, name="weights")
        total = float(np.sum(weights))
        if total <= 0.0:
            return 1.0
        wn = weights / total
        mean = float(np.sum(wn * values))
        std = float(np.sqrt(np.sum(wn * (values - mean) ** 2)))
        n_eff = float(1.0 / np.sum(wn * wn))

    if std <= 0.0 or not np.isfinite(std):
        std = max(float(np.max(values) - np.min(values)), 1.0) / 6.0
    if n_eff <= 1.0 or not np.isfinite(n_eff):
        n_eff = float(max(values.size, 2))
    return max(1e-6, 1.06 * std * n_eff ** (-1.0 / 5.0))


def weighted_gaussian_kde_1d(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    grid: np.ndarray | None = None,
    grid_size: int = 400,
    bandwidth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple weighted 1D Gaussian KDE for plotting."""
    values = _as_1d(values, name="values")
    if weights is None:
        weights = np.ones_like(values, dtype=np.float64)
    else:
        weights = _as_1d(weights, name="weights").astype(np.float64, copy=False)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        if grid is None:
            grid = np.linspace(0.0, 1.0, grid_size)
        return np.asarray(grid, dtype=np.float64), np.full_like(np.asarray(grid, dtype=np.float64), np.nan)

    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    weights = weights / total

    if bandwidth is None:
        bandwidth = _silverman_bandwidth_1d(values, weights)
    if bandwidth <= 0.0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be finite and strictly positive")

    if grid is None:
        low = float(np.min(values))
        high = float(np.max(values))
        pad = max(0.1 * (high - low), 3.0 * bandwidth)
        grid = np.linspace(low - pad, high + pad, grid_size, dtype=np.float64)
    else:
        grid = np.asarray(grid, dtype=np.float64)

    z = (grid[:, None] - values[None, :]) / bandwidth
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * bandwidth)
    density = norm * np.sum(weights[None, :] * np.exp(-0.5 * z * z), axis=1)
    return grid, density


def run_kde_conditioning(
    X: np.ndarray,
    theta: np.ndarray,
    *,
    x_obs: np.ndarray,
    bandwidth: np.ndarray,
    density_grid_size: int = 400,
    theta_density_bandwidth: float | None = None,
) -> dict[str, Any]:
    """Condition on x_obs with a Gaussian kernel in feature space.

    Notes
    -----
    - `bandwidth` plays the same role as the sigma vector in a kernel-ABC style run.
    - The output weights define an empirical posterior over catalogue rows.
    - The resulting theta posterior is the weighted empirical distribution over theta.
    """
    X = _as_2d(X, name="X")
    theta = _as_1d(theta, name="theta")
    if X.shape[0] != theta.size:
        raise ValueError("X and theta must have the same number of rows")

    weights, distance = _gaussian_kernel_weights(X, x_obs, bandwidth)
    summary = summarize_1d(theta, weights=weights)
    grid, density = weighted_gaussian_kde_1d(
        theta,
        weights=weights,
        grid_size=density_grid_size,
        bandwidth=theta_density_bandwidth,
    )
    rank = np.argsort(distance, kind="mergesort")
    return {
        "method": "kde",
        "weights": weights,
        "distance": distance,
        "summary": summary,
        "theta_grid": grid,
        "theta_density": density,
        "top_indices": rank[:10].copy(),
        "support_check": support_check(X, x_obs, scale=bandwidth),
        "x_obs": np.asarray(x_obs, dtype=np.float64).copy(),
        "bandwidth": np.asarray(bandwidth, dtype=np.float64).copy(),
    }


def posterior_predictive_from_weights(
    X: np.ndarray,
    weights: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Summarize the weighted empirical posterior predictive for X."""
    X = _as_2d(X, name="X")
    weights = _as_1d(weights, name="weights")
    if X.shape[0] != weights.size:
        raise ValueError("X and weights must have the same number of rows")

    if feature_names is None:
        feature_names = [f"x{j}" for j in range(X.shape[1])]
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match X.shape[1]")

    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(feature_names):
        out[name] = summarize_1d(X[:, j], weights=weights)
    return out


def _multivariate_normal_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    x = _as_1d(x, name="x")
    mean = _as_1d(mean, name="mean")
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")
    if x.size != mean.size or mean.size != cov.shape[0]:
        raise ValueError("dimension mismatch in multivariate normal logpdf")

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + np.eye(cov.shape[0], dtype=np.float64) * 1e-8
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise np.linalg.LinAlgError("covariance is not positive definite")

    delta = x - mean
    quad = float(delta @ np.linalg.solve(cov, delta))
    d = float(x.size)
    return float(-0.5 * (d * np.log(2.0 * np.pi) + logdet + quad))


def _choose_component_count(
    Z: np.ndarray,
    component_range: Iterable[int],
    *,
    criterion: str,
    covariance_type: str,
    reg_covar: float,
    random_state: int | None,
) -> tuple[GaussianMixture, list[dict[str, float]]]:
    criterion = criterion.lower()
    if criterion not in {"bic", "aic"}:
        raise ValueError("criterion must be 'bic' or 'aic'")

    candidates: list[dict[str, float]] = []
    best_score = np.inf
    best_model: GaussianMixture | None = None

    for n_components in component_range:
        model = GaussianMixture(
            n_components=int(n_components),
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        model.fit(Z)
        score = float(model.bic(Z) if criterion == "bic" else model.aic(Z))
        candidates.append({"n_components": int(n_components), criterion: score})
        if score < best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("failed to fit any GMM candidate")
    return best_model, candidates


def fit_joint_gmm(
    X: np.ndarray,
    theta: np.ndarray,
    *,
    n_components: int | str = "bic",
    component_range: Iterable[int] = range(1, 9),
    covariance_type: str = "full",
    standardize: bool = True,
    reg_covar: float = 1e-6,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Fit a joint GMM to [theta, X].

    Parameters
    ----------
    n_components:
        Either an integer or one of {"bic", "aic"}.
        If a string is passed, the best component count is selected from component_range.
    """
    X = _as_2d(X, name="X")
    theta = _as_1d(theta, name="theta")
    if X.shape[0] != theta.size:
        raise ValueError("X and theta must have the same number of rows")

    theta2 = theta[:, None]
    Z = np.column_stack([theta2, X])

    scaler = Standardizer.fit(Z) if standardize else Standardizer(
        mean=np.zeros(Z.shape[1], dtype=np.float64),
        std=np.ones(Z.shape[1], dtype=np.float64),
    )
    Zs = scaler.transform(Z)

    if isinstance(n_components, str):
        model, scores = _choose_component_count(
            Zs,
            component_range=component_range,
            criterion=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        selected_by = n_components.lower()
    else:
        model = GaussianMixture(
            n_components=int(n_components),
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        model.fit(Zs)
        scores = []
        selected_by = "manual"

    return {
        "model": model,
        "scaler": scaler,
        "standardize": bool(standardize),
        "n_components": int(model.n_components),
        "selection_scores": scores,
        "selected_by": selected_by,
        "X": X.copy(),
        "theta": theta.copy(),
    }


def _conditional_mixture_from_joint_gmm(fit: dict[str, Any], x_obs: np.ndarray) -> dict[str, Any]:
    model: GaussianMixture = fit["model"]
    scaler: Standardizer = fit["scaler"]

    x_obs = _as_1d(x_obs, name="x_obs")
    d_theta = 1
    d_x = x_obs.size

    x_obs_joint = np.concatenate([np.zeros(d_theta, dtype=np.float64), x_obs])
    x_obs_scaled = scaler.transform(x_obs_joint[None, :])[0, d_theta:]

    cond_weights: list[float] = []
    cond_means: list[float] = []
    cond_vars: list[float] = []

    for k in range(model.n_components):
        mu = model.means_[k]
        cov = model.covariances_[k]

        mu_t = mu[:d_theta]
        mu_x = mu[d_theta:]
        s_tt = cov[:d_theta, :d_theta]
        s_tx = cov[:d_theta, d_theta:]
        s_xt = cov[d_theta:, :d_theta]
        s_xx = cov[d_theta:, d_theta:]

        jitter = 1e-8 * np.eye(d_x, dtype=np.float64)
        s_xx_reg = s_xx + jitter
        solve_term = np.linalg.solve(s_xx_reg, (x_obs_scaled - mu_x))
        gain = s_tx @ np.linalg.inv(s_xx_reg)

        cond_mean = mu_t + s_tx @ solve_term
        cond_cov = s_tt - gain @ s_xt

        cond_cov = 0.5 * (cond_cov + cond_cov.T)
        cond_cov[0, 0] = max(float(cond_cov[0, 0]), 1e-8)

        log_weight = np.log(model.weights_[k]) + _multivariate_normal_logpdf(x_obs_scaled, mu_x, s_xx_reg)
        cond_weights.append(float(log_weight))
        cond_means.append(float(cond_mean[0]))
        cond_vars.append(float(cond_cov[0, 0]))

    logw = np.asarray(cond_weights, dtype=np.float64)
    logw = logw - np.max(logw)
    weights = np.exp(logw)
    weights = weights / np.sum(weights)

    theta_mean = scaler.mean[0]
    theta_std = scaler.std[0]
    means_original = theta_mean + theta_std * np.asarray(cond_means, dtype=np.float64)
    stds_original = theta_std * np.sqrt(np.asarray(cond_vars, dtype=np.float64))

    return {
        "weights": weights,
        "means": means_original,
        "stds": stds_original,
        "means_scaled": np.asarray(cond_means, dtype=np.float64),
        "vars_scaled": np.asarray(cond_vars, dtype=np.float64),
    }


def _sample_mixture_1d(
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    *,
    n_samples: int,
    random_state: int | None = None,
) -> np.ndarray:
    weights = _as_1d(weights, name="weights")
    means = _as_1d(means, name="means")
    stds = _as_1d(stds, name="stds")
    if not (weights.size == means.size == stds.size):
        raise ValueError("weights, means, and stds must have the same length")
    if np.any(stds <= 0.0):
        raise ValueError("stds must be strictly positive")

    rng = np.random.default_rng(random_state)
    comps = rng.choice(weights.size, size=int(n_samples), p=weights)
    samples = rng.normal(loc=means[comps], scale=stds[comps])
    return samples.astype(np.float64, copy=False)


def _mixture_density_1d(grid: np.ndarray, weights: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    grid = _as_1d(grid, name="grid")
    weights = _as_1d(weights, name="weights")
    means = _as_1d(means, name="means")
    stds = _as_1d(stds, name="stds")
    z = (grid[:, None] - means[None, :]) / stds[None, :]
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * stds[None, :])
    return np.sum(weights[None, :] * norm * np.exp(-0.5 * z * z), axis=1)


def condition_gmm(
    fit: dict[str, Any],
    *,
    x_obs: np.ndarray,
    n_samples: int = 20_000,
    density_grid_size: int = 400,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Condition the joint GMM on x_obs and return a 1D theta posterior."""
    theta = _as_1d(fit["theta"], name="theta")
    cond = _conditional_mixture_from_joint_gmm(fit, x_obs=x_obs)

    samples = _sample_mixture_1d(
        cond["weights"],
        cond["means"],
        cond["stds"],
        n_samples=n_samples,
        random_state=random_state,
    )
    summary = summarize_1d(samples)

    low = min(float(np.min(theta)), float(np.min(samples)))
    high = max(float(np.max(theta)), float(np.max(samples)))
    pad = 0.1 * max(high - low, 1e-6)
    grid = np.linspace(low - pad, high + pad, density_grid_size, dtype=np.float64)
    density = _mixture_density_1d(grid, cond["weights"], cond["means"], cond["stds"])

    return {
        "method": "gmm",
        "samples": samples,
        "summary": summary,
        "theta_grid": grid,
        "theta_density": density,
        "component_weights": cond["weights"],
        "component_means": cond["means"],
        "component_stds": cond["stds"],
        "x_obs": np.asarray(x_obs, dtype=np.float64).copy(),
        "support_check": support_check(fit["X"], x_obs),
    }


def plot_theta_posteriors(
    theta_prior: np.ndarray,
    *,
    kde_result: dict[str, Any] | None = None,
    gmm_result: dict[str, Any] | None = None,
    bins: int = 40,
    title: str = "Posterior over theta after conditioning on x_obs",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the prior-like baseline and the KDE/GMM posteriors."""
    theta_prior = _as_1d(theta_prior, name="theta_prior")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(theta_prior, bins=bins, density=True, alpha=0.35, label="all pairs after S")

    if kde_result is not None:
        ax.plot(kde_result["theta_grid"], kde_result["theta_density"], lw=2.0, label="KDE posterior")
    if gmm_result is not None:
        ax.plot(gmm_result["theta_grid"], gmm_result["theta_density"], lw=2.0, label="GMM posterior")

    ax.set_xlabel("theta")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig, ax


def plot_feature_posteriors(
    X: np.ndarray,
    feature_names: list[str],
    *,
    x_obs: np.ndarray,
    kde_result: dict[str, Any] | None = None,
    bins: int = 40,
    title: str = "Feature distributions after conditioning",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot feature marginals before and after KDE weighting.

    This plot is most meaningful for the KDE result because KDE returns direct
    weights over catalogue rows.
    """
    X = _as_2d(X, name="X")
    x_obs = _as_1d(x_obs, name="x_obs")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match number of feature columns")
    if X.shape[1] != x_obs.size:
        raise ValueError("x_obs length must match number of feature columns")

    n_cols = X.shape[1]
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.0))
    axes = np.atleast_1d(axes)

    weights = None if kde_result is None else kde_result["weights"]
    for j, ax in enumerate(axes):
        ax.hist(X[:, j], bins=bins, density=True, alpha=0.35, label="all pairs after S")
        if weights is not None:
            ax.hist(X[:, j], bins=bins, weights=weights, density=True, alpha=0.55, label="KDE weighted")
        ax.axvline(x_obs[j], color="k", linestyle="--", linewidth=1.0)
        ax.set_xlabel(feature_names[j])
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)

    axes[0].legend()
    fig.suptitle(title)
    plt.tight_layout()
    return fig, axes
