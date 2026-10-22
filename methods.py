"""Estimators sharing a fit(data, rng) interface."""

from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression, QuantileRegressor


class Method(ABC):
    """Return a coefficient vector of shape (p,).

    Each fit receives its own Generator. OLS and LAD are deterministic;
    accepting rng keeps the interface usable for randomized estimators.
    settings records every estimator option changed from its default.
    """

    @abstractmethod
    def fit(self, data, rng):
        """Fit a fresh estimator to data['X'] and data['y']."""


class OLS(Method):
    name = "OLS"
    settings = {"fit_intercept": False}

    def fit(self, data, rng):
        return LinearRegression(**self.settings).fit(data["X"], data["y"]).coef_


class LAD(Method):
    name = "LAD"
    settings = {"quantile": 0.5, "alpha": 0.0,
                "fit_intercept": False, "solver": "highs"}

    def fit(self, data, rng):
        return QuantileRegressor(**self.settings).fit(data["X"], data["y"]).coef_
