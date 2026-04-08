"""Thin helpers for a practical-guide-style SBI notebook.

This file intentionally stays close to the direct `sbi` example flow used in
the `sbi-practical-guide` repository and the official `sbi` tutorials:

1. build a catalogue of `(theta, x)` rows,
2. choose features and a target,
3. convert arrays to torch tensors in the notebook,
4. call `NPE().append_simulations(...).train()`,
5. build the posterior and sample from it.

Compared with `lg_sbi.py`, this module is deliberately smaller and avoids
extra wrappers around training, diagnostics, and approximate predictive checks.
It exists to support a clearer, more notebook-centric workflow.
"""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from lg_analogues import PairSet, Sample


def _as_1d(values: np.ndarray | Iterable[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return arr


def _safe_log10(values: np.ndarray, *, name: str) -> np.ndarray:
    if np.any(values <= 0.0):
        raise ValueError(f"{name} contains non-positive values; cannot apply log10")
    return np.log10(values)


def build_observation_vector(
    feature_info: dict[str, dict[str, Any]],
    active_features: list[str],
    *,
    overrides: dict[str, float] | None = None,
) -> np.ndarray:
    """Construct the observation vector in the chosen feature order."""
    overrides = {} if overrides is None else dict(overrides)
    values: list[float] = []
    for name in active_features:
        if name not in feature_info:
            raise KeyError(f"Feature '{name}' is missing from feature_info")
        values.append(float(overrides.get(name, feature_info[name]["obs"])))
    return np.asarray(values, dtype=np.float64)


def add_group_m200c_columns(
    catalog: dict[str, np.ndarray],
    *,
    sample: Sample,
    pairs: PairSet,
    sub: dict[str, Any],
    halos: dict[str, Any] | np.ndarray,
    h: float,
) -> dict[str, np.ndarray]:
    """Return a copy of `catalog` with FoF-halo M200c-derived columns added."""
    out = {name: np.asarray(values).copy() for name, values in catalog.items()}

    selected_group_numbers = np.asarray(sub["SubhaloGrNr"], dtype=np.int64)[sample.keep_idx]
    pair_i = np.asarray(pairs.i, dtype=np.int64)
    pair_j = np.asarray(pairs.j, dtype=np.int64)

    group_i = selected_group_numbers[pair_i]
    group_j = selected_group_numbers[pair_j]

    if isinstance(halos, dict):
        group_m200c_native = np.asarray(halos["Group_M_Crit200"], dtype=np.float64)
    else:
        group_m200c_native = np.asarray(halos, dtype=np.float64)

    group_m200c_msun = group_m200c_native * 1.0e10 / float(h)

    m200c_i = group_m200c_msun[group_i]
    m200c_j = group_m200c_msun[group_j]
    m200c_big = np.maximum(m200c_i, m200c_j)
    m200c_small = np.minimum(m200c_i, m200c_j)

    out["m200c_i"] = m200c_i
    out["m200c_j"] = m200c_j
    out["m200c_big"] = m200c_big
    out["m200c_small"] = m200c_small
    out["m200c_sum"] = m200c_i + m200c_j
    out["m200c_ratio"] = m200c_big / np.clip(m200c_small, 1e-30, None)
    return out


def prepare_1d_posterior_problem(
    catalog: dict[str, np.ndarray],
    *,
    features: list[str],
    target: str,
    log10_features: set[str] | None = None,
    target_log10: bool = False,
) -> dict[str, Any]:
    """Extract a clean `(X, theta)` problem from a pair catalogue."""
    if not features:
        raise ValueError("features must contain at least one name")
    if target not in catalog:
        raise KeyError(f"Target '{target}' not found in catalog")

    log10_features = set() if log10_features is None else set(log10_features)

    columns: list[np.ndarray] = []
    n_rows: int | None = None
    for name in features:
        if name not in catalog:
            raise KeyError(f"Feature '{name}' not found in catalog")
        col = _as_1d(catalog[name], name=name).astype(np.float64, copy=False)
        if name in log10_features:
            col = _safe_log10(col, name=name)
        if n_rows is None:
            n_rows = col.size
        elif col.size != n_rows:
            raise ValueError("All feature columns must have the same length")
        columns.append(col)

    theta = _as_1d(catalog[target], name=target).astype(np.float64, copy=False)
    if n_rows is None:
        raise ValueError("Could not infer the row count from features")
    if theta.size != n_rows:
        raise ValueError("Target length does not match feature lengths")
    if target_log10:
        theta = _safe_log10(theta, name=target)

    X = np.column_stack(columns)
    keep = np.all(np.isfinite(X), axis=1) & np.isfinite(theta)
    X = X[keep]
    theta = theta[keep]

    if X.shape[0] == 0:
        raise ValueError("No finite rows remain after filtering")

    return {
        "X": X,
        "theta": theta,
        "feature_names": list(features),
        "target_name": target,
        "dropped_rows": int(n_rows - X.shape[0]),
    }


def summarize_1d(values: np.ndarray | Iterable[float]) -> dict[str, float]:
    arr = _as_1d(values, name="values")
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "q16": float("nan"), "q84": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.quantile(arr, 0.5)),
        "q16": float(np.quantile(arr, 0.16)),
        "q84": float(np.quantile(arr, 0.84)),
    }


def plot_1d_posterior_comparison(
    theta_catalog: np.ndarray,
    posterior_samples: np.ndarray,
    *,
    target_name: str,
    bins: int = 40,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the catalogue-wide theta distribution against posterior samples."""
    theta_catalog = _as_1d(theta_catalog, name="theta_catalog")
    posterior_samples = _as_1d(posterior_samples, name="posterior_samples")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(theta_catalog, bins=bins, density=True, alpha=0.3, label="all pairs after S")
    ax.hist(posterior_samples, bins=bins, density=True, alpha=0.55, label="NPE posterior samples")
    ax.set_xlabel(target_name)
    ax.set_ylabel("density")
    ax.set_title(title or f"NPE posterior for {target_name}")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig, ax


def plot_feature_histograms(
    X: np.ndarray,
    x_obs: np.ndarray,
    *,
    feature_names: list[str],
) -> tuple[plt.Figure, np.ndarray]:
    """Plot one histogram per feature with the observation overlaid."""
    X = np.asarray(X, dtype=np.float64)
    x_obs = _as_1d(x_obs, name="x_obs")
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != x_obs.size or X.shape[1] != len(feature_names):
        raise ValueError("Dimension mismatch between X, x_obs, and feature_names")

    fig, axes = plt.subplots(1, X.shape[1], figsize=(5 * X.shape[1], 3.8), constrained_layout=True)
    if X.shape[1] == 1:
        axes = np.asarray([axes], dtype=object)

    for j, name in enumerate(feature_names):
        ax = axes[j]
        ax.hist(X[:, j], bins=30, alpha=0.75)
        ax.axvline(x_obs[j], linestyle="--", linewidth=2)
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)

    return fig, axes
