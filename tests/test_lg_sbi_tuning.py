import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import lg_sbi_tuning


def _have_optuna() -> bool:
    try:
        import optuna  # noqa: F401
    except ImportError:
        return False
    return True


class SbiTuningTests(unittest.TestCase):
    @unittest.skipUnless(_have_optuna(), "optuna is not installed in this environment")
    def test_optuna_objective_and_study_smoke(self):
        X = np.array(
            [
                [0.1, -0.2],
                [0.0, 0.3],
                [0.2, 0.1],
                [-0.1, -0.1],
            ],
            dtype=np.float64,
        )
        theta = np.array([1.0, 1.3, 0.8, 1.1], dtype=np.float64)
        x_obs = np.array([0.05, 0.0], dtype=np.float64)

        def fake_fit_catalog_npe(*, X, theta, x_obs, feature_names, target_name, **kwargs):
            learning_rate = float(kwargs["learning_rate"])
            hidden_features = int(kwargs["hidden_features"])
            num_transforms = int(kwargs["num_transforms"])
            score = (
                -abs(np.log10(learning_rate) + 3.0)
                + 0.01 * hidden_features
                - 0.02 * abs(num_transforms - 5)
            )
            return {
                "posterior": None,
                "inference": None,
                "device": kwargs.get("device", "cpu"),
                "X_train": np.asarray(X, dtype=np.float64),
                "theta_train": np.asarray(theta, dtype=np.float64),
                "X_calibration": np.asarray(X, dtype=np.float64),
                "theta_calibration": np.asarray(theta, dtype=np.float64),
                "feature_names": feature_names,
                "target_name": target_name,
                "diagnostics": {
                    "calibration_size": int(len(theta)),
                    "train_size": int(len(theta)),
                    "mean_calibration_log_prob": float(score),
                    "sbc_ranks": np.empty((0, 1), dtype=np.int64),
                    "sbc_summary": {"num_eval": 0},
                    "expected_coverage_1d": {"num_eval": 0, "levels": np.asarray([]), "empirical": np.asarray([])},
                },
                "training_config": kwargs,
            }

        with patch.object(lg_sbi_tuning, "fit_catalog_npe", side_effect=fake_fit_catalog_npe):
            objective = lg_sbi_tuning.make_sbi_optuna_objective(
                X,
                theta,
                x_obs=x_obs,
                feature_names=["x0", "x1"],
                target_name="theta",
            )

            import optuna

            trial = optuna.trial.FixedTrial(
                {
                    "density_model": "maf",
                    "hidden_features": 64,
                    "num_transforms": 5,
                    "training_batch_size": 128,
                    "learning_rate": 1e-3,
                    "validation_fraction": 0.1,
                    "stop_after_epochs": 30,
                }
            )
            score = objective(trial)
            self.assertTrue(np.isfinite(score))

            study_result = lg_sbi_tuning.run_sbi_optuna_study(
                X,
                theta,
                x_obs=x_obs,
                feature_names=["x0", "x1"],
                target_name="theta",
                n_trials=3,
            )

            self.assertIn("best_params", study_result)
            self.assertTrue(np.isfinite(study_result["best_value"]))
            self.assertFalse(study_result["trials_df"].empty)


if __name__ == "__main__":
    unittest.main()
