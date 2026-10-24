"""Estimators sharing a fit(data, rng) interface."""

from abc import ABC, abstractmethod

import numpy as np
import statsmodels.api as sm
from scipy import stats
from knockpy.knockoff_filter import KnockoffFilter
import knockpy.knockoff_stats as kstats

class Method(ABC):
    """Return a list of selected variables after controlling for false discovery
    at the nominal rate alpha.

    Each fit receives its own Generator. models are deterministic;
    accepting rng keeps the interface usable for randomized estimators.
    settings records every estimator option changed from its default.
    """

    @abstractmethod
    def fit(self, data, alpha, rng):
        """Fit a fresh estimator to data['X'] and data['y']."""


class BHq(Method):
    name = "BHq"
    settings = {}

    def fit(self, data, alpha, rng):
        model = sm.OLS(data['y'], data['X']).fit()
        pvals = model.pvalues
        fdr = stats.false_discovery_control(pvals, method = 'BH')
        return (fdr <= alpha).astype(int)


class knockoff(Method):
    name = "knockoff"
    settings = {"ksampler": 'gaussian', 'fstat' : 'lasso',
                "knockoff_kwargs" : {'method':'sdp',
                                     'dsdp_warning':False}, 

                }

    def fit(self, data, alpha, rng):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            filter = KnockoffFilter(**self.settings) 
            _ = filter.forward(data["X"],data["y"],fdr=alpha)
            w = filter.W
            thresh = kstats.data_dependent_threshhold(W=w, fdr=alpha, offset=0)
            return (w >= thresh).astype(int)


class knockoff_plus(knockoff):
    # offset is 1 by default; can just return output of the forward wrapper
    name = 'knockoff+'

    def fit(self, data, alpha, rng):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            filter = KnockoffFilter(**self.settings)
            rejections = (filter.forward(data["X"], data["y"],fdr = alpha)).astype(int) 
            return rejections
