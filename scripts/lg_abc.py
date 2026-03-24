"""Approximate Bayesian Computation (ABC) helpers for Local Group analogue studies.

These utilities are intentionally simple and notebook-friendly.
They operate on outputs from `lg_analogues.Sample` and `lg_analogues.PairSet`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lg_analogues import PairSet, Sample


def _resolve_keep_mask(n: int, keep_pairs: np.ndarray | None) -> np.ndarray:
    """Return a boolean mask of length `n` from bool mask or integer indices."""
    if keep_pairs is None:
        return np.ones(n, dtype=bool)

    keep_pairs = np.asarray(keep_pairs)
    if keep_pairs.ndim != 1:
        raise ValueError("keep_pairs must be a 1D array")

    if keep_pairs.dtype == bool:
        if keep_pairs.size != n:
            raise ValueError(f"Boolean keep_pairs length {keep_pairs.size} != number of pairs {n}")
        return keep_pairs.copy()

    if not np.issubdtype(keep_pairs.dtype, np.integer):
        raise ValueError("keep_pairs must be either a boolean mask or integer indices")

    mask = np.zeros(n, dtype=bool)
    if keep_pairs.size == 0:
        return mask

    if np.any((keep_pairs < 0) | (keep_pairs >= n)):
        raise ValueError("keep_pairs integer indices are out of bounds")
    mask[keep_pairs] = True
    return mask


def build_pair_catalog(
    sample: Sample,
    pairs: PairSet,
    *,
    keep_pairs: np.ndarray | None = None,
    sub: dict[str, Any] | None = None,
    sfr_field: str | None = None,
) -> dict[str, np.ndarray]:
    """Build per-pair catalog columns from a selected sample and pair set.

    Args:
        sample: Output sample from LG analogue selection.
        pairs: Pair set referencing `sample` rows.
        keep_pairs: Optional boolean mask or integer indices selecting pair rows.
        sub: Optional full subhalo catalog dict (needed for SFR columns).
        sfr_field: Name of SFR field in `sub` (e.g. ``SubhaloSFRinRad``).

    Returns:
        Dict of per-pair NumPy arrays (all same length).
    """
    n_pairs = pairs.i.size
    mask = _resolve_keep_mask(n_pairs, keep_pairs)

    i = pairs.i[mask]
    j = pairs.j[mask]

    mstar_i = sample.mstar[i].astype(np.float64, copy=False)
    mstar_j = sample.mstar[j].astype(np.float64, copy=False)
    mdm_i = sample.mdm[i].astype(np.float64, copy=False)
    mdm_j = sample.mdm[j].astype(np.float64, copy=False)

    catalog: dict[str, np.ndarray] = {
        "r_kpc": pairs.dist_kpc[mask].astype(np.float64, copy=False),
        "v_r": pairs.v_r[mask].astype(np.float64, copy=False),
        "v_t": pairs.v_t[mask].astype(np.float64, copy=False),
        "same_host": pairs.same_host[mask].astype(np.float64, copy=False),
        "mstar_i": mstar_i,
        "mstar_j": mstar_j,
        "mstar_big": np.maximum(mstar_i, mstar_j),
        "mstar_small": np.minimum(mstar_i, mstar_j),
        "mdm_i": mdm_i,
        "mdm_j": mdm_j,
        "mdm_sum": mdm_i + mdm_j,
    }

    if sub is not None and sfr_field is not None:
        if sfr_field not in sub:
            raise KeyError(f"sfr_field '{sfr_field}' not found in sub")

        keep_idx = sample.keep_idx
        sfr_all = np.asarray(sub[sfr_field], dtype=np.float64)
        sfr_sample = sfr_all[keep_idx]
        sfr_i = sfr_sample[i]
        sfr_j = sfr_sample[j]

        ssfr_i = np.full(i.size, np.nan, dtype=np.float64)
        ssfr_j = np.full(i.size, np.nan, dtype=np.float64)

        pos_i = mstar_i > 0.0
        pos_j = mstar_j > 0.0
        ssfr_i[pos_i] = sfr_i[pos_i] / mstar_i[pos_i]
        ssfr_j[pos_j] = sfr_j[pos_j] / mstar_j[pos_j]

        catalog.update(
            {
                "sfr_i": sfr_i,
                "sfr_j": sfr_j,
                "sfr_sum": sfr_i + sfr_j,
                "ssfr_i": ssfr_i,
                "ssfr_j": ssfr_j,
            }
        )

    return catalog


def make_feature_matrix(
    catalog: dict[str, np.ndarray],
    features: list[str],
    *,
    log10: set[str] | None = None,
    standardize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a feature matrix in a requested feature order.

    Args:
        catalog: Dict of per-pair arrays.
        features: Ordered feature names.
        log10: Subset of `features` to transform with log10.
        standardize: If True, return z-scored columns.

    Returns:
        ``(X, meta)`` where ``X`` is shape (N, D) and ``meta`` stores
        feature names, transform info, means and stds used.
    """
    if len(features) == 0:
        raise ValueError("features must contain at least one name")

    log10_set = set() if log10 is None else set(log10)

    cols: list[np.ndarray] = []
    n_rows: int | None = None
    for name in features:
        if name not in catalog:
            raise KeyError(f"Feature '{name}' not found in catalog")

        col = np.asarray(catalog[name], dtype=np.float64)
        if col.ndim != 1:
            raise ValueError(f"Feature '{name}' must be a 1D array")

        if n_rows is None:
            n_rows = col.size
        elif col.size != n_rows:
            raise ValueError("All catalog columns must have the same length")

        if name in log10_set:
            if np.any(col <= 0):
                raise ValueError(f"Feature '{name}' has non-positive values; cannot apply log10")
            col = np.log10(col)

        cols.append(col)

    assert n_rows is not None
    X = np.column_stack(cols) if n_rows > 0 else np.empty((0, len(features)), dtype=np.float64)

    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix contains non-finite values")

    means = np.zeros(len(features), dtype=np.float64)
    stds = np.ones(len(features), dtype=np.float64)

    if standardize and X.shape[0] > 0:
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds == 0.0] = 1.0
        X = (X - means) / stds

    meta: dict[str, Any] = {
        "features": list(features),
        "log10": sorted(log10_set),
        "standardized": bool(standardize),
        "means": means,
        "stds": stds,
    }
    return X, meta


def make_theta_vector(catalog: dict[str, np.ndarray], target: str, *, log10: bool = False) -> np.ndarray:
    """Extract 1D theta values for posterior summaries.

    Args:
        catalog: Dict of per-pair arrays.
        target: Target column name in `catalog`.
        log10: Apply base-10 logarithm to target values.

    Returns:
        Theta vector as float64 array.
    """
    if target not in catalog:
        raise KeyError(f"Target '{target}' not found in catalog")

    theta = np.asarray(catalog[target], dtype=np.float64)
    if theta.ndim != 1:
        raise ValueError("Target array must be 1D")

    if log10:
        if np.any(theta <= 0):
            raise ValueError(f"Target '{target}' has non-positive values; cannot apply log10")
        theta = np.log10(theta)

    return theta


def abc_distance_diagonal(X: np.ndarray, x_obs: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Compute diagonal-covariance ABC distances.

    Distance for row i:
        ``d_i = sqrt(sum(((X_i - x_obs) / sigma)^2))``
    """
    X = np.asarray(X, dtype=np.float64)
    x_obs = np.asarray(x_obs, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if x_obs.ndim != 1 or sigma.ndim != 1:
        raise ValueError("x_obs and sigma must be 1D arrays")
    if X.shape[1] != x_obs.size or X.shape[1] != sigma.size:
        raise ValueError("Dimension mismatch between X, x_obs, and sigma")
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must be finite and strictly positive")

    z = (X - x_obs[None, :]) / sigma[None, :]
    return np.sqrt(np.einsum("ij,ij->i", z, z))


def abc_rejection_mask(
    d: np.ndarray,
    *,
    eps: float | None = None,
    accept_frac: float | None = None,
    accept_n: int | None = None,
    kernel_bandwidth: float = None
) -> np.ndarray:
    """Create rejection-ABC acceptance mask.

    Exactly one of these should be provided:

    - eps: accept distances ``d <= eps``
    - accept_frac: accept the closest fraction of rows
    - accept_n: accept the closest fixed number of rows
    """
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError("d must be a 1D array")

    bw = float(kernel_bandwidth)
    if (not np.isfinite(bw)) or (bw <= 0.0):
        raise ValueError("bandwidth must be finite and > 0")

    n = d.size
    if n == 0:
        return np.zeros(0, dtype=bool)

    provided = int(eps is not None) + int(accept_frac is not None) + int(accept_n is not None)
    if provided != 1:
        raise ValueError("Provide exactly one of eps, accept_frac, or accept_n")

    if eps is not None:
        return d <= float(eps)

    if accept_n is not None:
        k = int(accept_n)
        if k <= 0:
            raise ValueError("accept_n must be >= 1")
        k = min(k, n)
        order = np.argsort(d, kind="mergesort")
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
        return mask

    # accept_frac case
    if not (0.0 < float(accept_frac) <= 1.0):
        raise ValueError("accept_frac must be in (0, 1]")

    k = int(np.ceil(float(accept_frac) * n))
    k = max(1, min(k, n))

    order = np.argsort(d, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def abc_kernel_weights(d: np.ndarray, *, bandwidth: float = 1.0) -> np.ndarray:
    """Compute normalized Gaussian-kernel ABC weights from distances.

    Weights are computed as:
        ``w_i ∝ exp(-0.5 * (d_i / bandwidth)^2)``

    With the default ``bandwidth=1``, this matches the original implementation.
    """
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError("d must be a 1D array")

    bw = float(bandwidth)
    if (not np.isfinite(bw)) or (bw <= 0.0):
        raise ValueError("bandwidth must be finite and > 0")

    w = np.zeros(d.size, dtype=np.float64)
    finite = np.isfinite(d)
    if not np.any(finite):
        return w

    z = d[finite] / bw
    w[finite] = np.exp(-0.5 * z * z)
    total = w.sum()
    if (not np.isfinite(total)) or (total <= 0.0):
        return np.zeros_like(w)

    w /= total
    return w




def run_abc_catalog(
    X: np.ndarray,
    theta: np.ndarray,
    *,
    x_obs: np.ndarray,
    sigma: np.ndarray,
    eps: float | None = None,
    accept_frac: float | None = None,
    accept_n: int | None = None,
    kernel_bandwidth: float = 1.0,
) -> dict[str, Any]:
    """Run simple rejection + kernel ABC on an existing catalogue.

    This mirrors the structure of a classic ABC tutorial (like a coin-flip example),
    except that the "simulator" is replaced by an existing catalogue:

    - each row i is one simulated system (one LG analogue pair)
    - X[i] are the "observables-like" summary features for row i
    - theta[i] is the target quantity to infer (posterior samples)

    Returns a dict with distances, acceptance mask, kernel weights, and posterior summaries.
    """
    X = np.asarray(X, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if theta.ndim != 1:
        raise ValueError("theta must be a 1D array")
    if X.shape[0] != theta.size:
        raise ValueError("X and theta must have the same number of rows")

    d = abc_distance_diagonal(X, x_obs, sigma)
    acc = abc_rejection_mask(d, eps=eps, accept_frac=accept_frac, accept_n=accept_n, kernel_bandwidth=kernel_bandwidth)
    w = abc_kernel_weights(d, bandwidth=kernel_bandwidth)

    theta_acc = theta[acc]
    summary_rejection = summarize_posterior(theta_acc) if theta_acc.size > 0 else None
    summary_kernel = summarize_posterior(theta, w)

    out: dict[str, Any] = {
        "d": d,
        "accept_mask": acc,
        "accept_rate": float(acc.sum() / acc.size) if acc.size > 0 else 0.0,
        "weights": w,
        "ess": float(effective_sample_size(w)),
        "theta_accepted": theta_acc,
        "summary_rejection": summary_rejection,
        "summary_kernel": summary_kernel,
    }
    return out
def effective_sample_size(w: np.ndarray) -> float:
    """Compute effective sample size (ESS) from normalized-like weights."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("w must be a 1D array")
    if w.size == 0:
        return 0.0

    if not np.all(np.isfinite(w)):
        return 0.0

    s = w.sum()
    if s <= 0.0:
        return 0.0

    wn = w / s
    denom = np.sum(wn * wn)
    if denom <= 0.0:
        return 0.0
    return float(1.0 / denom)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute a 1D weighted quantile in [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError("values and weights must be 1D arrays")
    if values.size != weights.size:
        raise ValueError("values and weights must have the same length")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    if values.size == 0:
        return float("nan")

    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(mask):
        return float("nan")

    v = values[mask]
    w = weights[mask]
    wsum = w.sum()
    if wsum <= 0.0:
        return float("nan")

    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]

    cdf = np.cumsum(w) / wsum
    cdf = np.r_[0.0, cdf]
    vpad = np.r_[v[0], v]
    return float(np.interp(q, cdf, vpad))


def summarize_posterior(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    """Summarize posterior samples with plain or weighted statistics."""
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("values must be 1D")

    if values.size == 0:
        if weights is None:
            return {"mean": float("nan"), "median": float("nan"), "q16": float("nan"), "q84": float("nan")}
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q16": float("nan"),
            "q84": float("nan"),
            "ess": 0.0,
        }

    if weights is None:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.quantile(values, 0.5)),
            "q16": float(np.quantile(values, 0.16)),
            "q84": float(np.quantile(values, 0.84)),
        }

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size != values.size:
        raise ValueError("weights must be a 1D array with same length as values")

    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")

    wsum = weights.sum()
    if wsum <= 0.0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q16": float("nan"),
            "q84": float("nan"),
            "ess": 0.0,
        }

    wn = weights / wsum
    return {
        "mean": float(np.sum(wn * values)),
        "median": weighted_quantile(values, wn, 0.5),
        "q16": weighted_quantile(values, wn, 0.16),
        "q84": weighted_quantile(values, wn, 0.84),
        "ess": float(effective_sample_size(wn)),
    }


def resample_weighted(
    values: np.ndarray,
    weights: np.ndarray,
    n: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Draw weighted bootstrap samples from `values`.

    If all weights are zero, falls back to uniform resampling.
    """
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=np.float64)

    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError("values and weights must be 1D")
    if values.size != weights.size:
        raise ValueError("values and weights must have same length")
    if n < 0:
        raise ValueError("n must be >= 0")

    if n == 0:
        return values[:0].copy()

    if values.size == 0:
        raise ValueError("Cannot resample from empty values")

    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite and non-negative")

    wsum = weights.sum()
    if wsum > 0.0:
        p = weights / wsum
    else:
        p = np.full(values.size, 1.0 / values.size, dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.choice(values.size, size=n, replace=True, p=p)
    return values[idx]
