from typing import Callable

import numpy as np


class MCIntegrator:
    """
    A  Monte Carlo integrator that marginalizes over observational
    uncertainty by propagating samples through a predictive model.

    >>> import numpy as np
    >>>
    >>> # 1. Define a simple model: theta = 2 * x
    >>> def simple_model(x): return 2 * x[0]
    >>>
    >>> # 2. Define observation: x = 10 +/- 0.1
    >>> x_obs = np.array([10.0])
    >>> x_err = np.array([0.1])
    >>>
    >>> # 3. Initialize and seed for reproducibility
    >>> np.random.seed(42)
    >>> integrator = MCIntegrator(simple_model, x_obs, x_err)
    >>>
    >>> # 4. Run integration
    >>> samples = integrator.run(n_realizations=5)
    >>> np.round(samples, 2)
    array([20.1 , 19.97, 20.13, 20.3 , 19.95])
    >>>
    >>> # 5. Check credible intervals
    >>> ci = integrator.get_ci(samples)
    >>> {k: round(v, 2) for k, v in ci.items()}
    {'median': 20.1, 'plus': 0.09, 'minus': 0.13}
    """

    def __init__(self, model: Callable, x_obs: np.ndarray, x_err: np.ndarray):
        """
        Args:
            model (Callable): A function that accepts an observation array 'x'
                              and returns the predicted property 'theta'.
                              Expected signature: model(x_array) -> theta_samples
            x_obs: 1D array of the observed mean values.
            x_err: 1D array (standard deviations) or 2D array (covariance matrix)
                   representing the observational uncertainty.
        """
        self.model = model
        self.x_obs = x_obs
        self.x_err = x_err

        if self.x_err.ndim == 1:
            self.cov = np.diag(self.x_err**2)
        else:
            self.cov = self.x_err

        self._is_vectorized = self._check_vectorization(model, len(x_obs))

    def run(self, n_realizations: int = 100) -> np.ndarray:
        """
        Run the Monte Carlo integration.

        Args:
            n_realizations: Number of Monte Carlo realizations to run.
                            Defaults to 100.

        Returns:
            1D array of the integrated values.
        """
        x_samples = np.random.multivariate_normal(
            self.x_obs, self.cov, size=n_realizations
        )
        if self._is_vectorized:
            theta_samples = self.model(x_samples)
        else:
            theta_samples = [self.model(x_i) for x_i in x_samples]

        return np.concatenate(np.atleast_2d(theta_samples)).flatten()

    @staticmethod
    def get_ci(samples):
        """Return standard Bayesian credible intervals."""
        q = np.percentile(samples, [16, 50, 84])
        return {
            "median": float(q[1]),
            "plus": float(q[2] - q[1]),
            "minus": float(q[1] - q[0]),
        }

    def _check_vectorization(self, model, n_features: int) -> bool:
        """
        Determines if a model function is vectorized.

        A vectorized model should accept an input of shape (N, n_features)
        and return an output of shape (N, ...) or (N*S, ...).
        """
        batch_size = 3
        dummy_batch = np.zeros((batch_size, n_features))

        try:
            output = model(dummy_batch)
            if hasattr(output, "shape") and output.shape[0] == batch_size:
                return True
            return False
        except Exception:
            return False


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    print("Doctests completed.")
