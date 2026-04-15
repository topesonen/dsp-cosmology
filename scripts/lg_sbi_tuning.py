"""Optuna-based hyperparameter tuning helpers for catalogue SBI.

This module keeps the existing `lg_sbi.fit_catalog_npe(...)` workflow intact and
adds a light tuning layer around it. The first tuning objective is the held-out
mean calibration log probability already returned by the current helper.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from lg_sbi import fit_catalog_npe


DEFAULT_FIXED_CONFIG: dict[str, Any] = {
    "max_num_epochs": 300,
    "clip_max_norm": 5.0,
    "calibration_fraction": 0.15,
    "random_state": 0,
    "show_train_summary": False,
}


def _import_optuna():
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as e:
        raise ImportError(
            "lg_sbi_tuning.py requires the `optuna` package."
        ) from e
    return optuna, TPESampler


def _trial_search_params(trial) -> dict[str, Any]:
    return {
        "density_model": trial.suggest_categorical("density_model", ["maf", "nsf"]),
        "hidden_features": trial.suggest_categorical("hidden_features", [32, 64, 128, 256]),
        "num_transforms": trial.suggest_int("num_transforms", 3, 10),
        "training_batch_size": trial.suggest_categorical("training_batch_size", [64, 128, 256, 512]),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True),
        "validation_fraction": trial.suggest_float("validation_fraction", 0.05, 0.3),
        "stop_after_epochs": trial.suggest_categorical("stop_after_epochs", [20, 30, 50, 80]),
    }


def make_sbi_optuna_objective(
    X: np.ndarray,
    theta: np.ndarray,
    *,
    x_obs: np.ndarray,
    feature_names: list[str] | None = None,
    target_name: str = "theta",
    fixed_config: dict[str, Any] | None = None,
    device: str | None = None,
) -> Callable[[Any], float]:
    """Create an Optuna objective around `fit_catalog_npe(...)`.

    Per trial, the expensive SBC and expected-coverage diagnostics are disabled.
    The objective value is the held-out mean calibration log probability.
    """
    X = np.asarray(X, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    config = dict(DEFAULT_FIXED_CONFIG)
    if fixed_config is not None:
        config.update(dict(fixed_config))

    def objective(trial) -> float:
        params = _trial_search_params(trial)
        trial_fit_config = dict(config)
        trial_fit_config.update(params)
        fit = fit_catalog_npe(
            X=X,
            theta=theta,
            x_obs=x_obs,
            feature_names=feature_names,
            target_name=target_name,
            device=device,
            sbc_num_eval=0,
            coverage_num_eval=0,
            **trial_fit_config,
        )

        diagnostics = fit["diagnostics"]
        score = float(diagnostics["mean_calibration_log_prob"])
        if not np.isfinite(score):
            raise ValueError("mean_calibration_log_prob is not finite for this trial")

        trial.set_user_attr("train_size", int(diagnostics["train_size"]))
        trial.set_user_attr("calibration_size", int(diagnostics["calibration_size"]))
        return score

    return objective


def _study_trials_dataframe(study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": int(trial.number),
            "state": str(trial.state.name),
            "value": float(trial.value) if trial.value is not None else np.nan,
        }
        for key, value in trial.params.items():
            row[f"param_{key}"] = value
        for key, value in trial.user_attrs.items():
            row[f"user_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def run_sbi_optuna_study(
    X: np.ndarray,
    theta: np.ndarray,
    *,
    x_obs: np.ndarray,
    feature_names: list[str] | None = None,
    target_name: str = "theta",
    fixed_config: dict[str, Any] | None = None,
    study_name: str | None = None,
    n_trials: int = 60,
    sampler_seed: int = 0,
    device: str | None = None,
    refit_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an Optuna study and refit the best SBI model with full diagnostics."""
    optuna, TPESampler = _import_optuna()

    config = dict(DEFAULT_FIXED_CONFIG)
    if fixed_config is not None:
        config.update(dict(fixed_config))

    objective = make_sbi_optuna_objective(
        X,
        theta,
        x_obs=x_obs,
        feature_names=feature_names,
        target_name=target_name,
        fixed_config=config,
        device=device,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=TPESampler(seed=int(sampler_seed)),
    )
    study.optimize(objective, n_trials=int(n_trials), catch=(Exception,))

    complete = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(float(trial.value))
    ]
    if not complete:
        raise RuntimeError("Optuna study completed with no successful trials")

    best_trial = study.best_trial
    best_params = dict(best_trial.params)

    refit_config = dict(config)
    if refit_overrides is not None:
        refit_config.update(dict(refit_overrides))
    best_fit_config = dict(refit_config)
    best_fit_config.update(best_params)

    best_fit = fit_catalog_npe(
        X=np.asarray(X, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        x_obs=x_obs,
        feature_names=feature_names,
        target_name=target_name,
        device=device,
        **best_fit_config,
    )

    return {
        "study": study,
        "best_params": best_params,
        "best_value": float(best_trial.value),
        "best_trial_number": int(best_trial.number),
        "best_fit": best_fit,
        "trials_df": _study_trials_dataframe(study),
        "fixed_config": config,
    }
