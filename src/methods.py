"""Implement two interchangeable tests of a population mean.

MeanTest defines the common test(data, null_mean, rng) interface.
TTest uses Student's t reference distribution; NormalApproxMeanTest uses
normal tails with the same estimated standard error. METHODS maps the
configuration names to these two methods.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm, ttest_1samp


class MeanTest(ABC):
    """Define the interface and descriptors for a two-sided mean test.

    Subclasses supply method_id, name, and test(). Stable integer method
    IDs identify their random streams; names describe them in figures.
    """

    @abstractmethod
    def test(self, data, null_mean, rng):
        """Return the estimated mean, SE, statistic, reference df, and p-value.

        Parameters
        ----------
        data : numpy.ndarray
            One-dimensional sample with at least two observations.
        null_mean : float
            Population mean under the null hypothesis.
        rng : numpy.random.Generator
            Stream for internal randomness; deterministic methods ignore it.

        Returns
        -------
        dict
            Numeric fields estimate, se, statistic, df, and p_value.
            Infinite df denotes a normal reference distribution.
        """
        pass


class TTest(MeanTest):
    """Override MeanTest.test with the one-sample Student's t-test.

    method_id is 1 and name identifies the method in reports.
    """

    method_id = 1
    name = "One-sample t-test"

    def test(self, data, null_mean, rng):
        """Return a two-sided t-test result using the MeanTest interface.

        data, null_mean, and rng have the meanings documented in MeanTest.
        Return its five numeric output fields; rng is accepted but unused.
        """
        fit = ttest_1samp(data, popmean=null_mean, alternative="two-sided")
        return {
            "estimate": float(np.mean(data)),
            "se": float(np.std(data, ddof=1) / np.sqrt(len(data))),
            "statistic": float(fit.statistic), "df": float(fit.df),
            "p_value": float(fit.pvalue),
        }


class NormalApproxMeanTest(MeanTest):
    """Override MeanTest.test with an estimated-SE normal approximation.

    method_id is 2 and name identifies the method in reports. This method
    estimates the variance; it is not a known-variance z-test.
    """

    method_id = 2
    name = "Normal approximation"

    def test(self, data, null_mean, rng):
        """Return a normal-tail result using the MeanTest interface.

        data, null_mean, and rng have the meanings documented in MeanTest.
        Return its five numeric output fields with df=inf. Use the sample
        standard deviation (ddof=1); rng is accepted but unused.
        """
        estimate = float(np.mean(data))
        se = float(np.std(data, ddof=1) / np.sqrt(len(data)))
        statistic = (estimate - null_mean) / se
        return {
            "estimate": estimate, "se": se, "statistic": statistic,
            "df": np.inf, "p_value": float(2 * norm.sf(abs(statistic))),
        }


METHODS = {"t_test": TTest(), "normal_approximation": NormalApproxMeanTest()}
