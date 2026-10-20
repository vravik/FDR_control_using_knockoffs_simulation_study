"""Define the simulation scenarios and their random streams.

Scenario stores a frozen SciPy distribution and its stable configuration ID.
make_scenarios constructs the design from configuration dictionaries.
make_rng constructs a NumPy Generator for one random task.
RNG_PLAN, DATA, and METHOD describe the recorded stream assignment.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import expon, lognorm, norm

DATA, METHOD = 0, 1
RNG_PLAN = {
    "bit_generator": "PCG64",
    "seed_sequence_entropy": [
        "scenario_id", "replicate_id", "purpose_id", "method_id", "root_seed"
    ],
    "purposes": {"data": DATA, "method": METHOD},
    "data_method_id": 0,
    "replicate_ids": "zero-based",
}


@dataclass(frozen=True)
class Scenario:
    """Store a data-generating condition without wrapping SciPy's sampler.

    Attributes
    ----------
    scenario_id : int
        Stable ID for this configuration, independent of execution order.
    label : str
        Distribution label for tables and figures.
    n : int
        Observations per repetition.
    p : int
        Number of covariates
    A : float
        Amplitude of signal covariates
    k : int
        Sparsity of model
    rho : float
        correlation between features
    dgp : object
        Frozen SciPy distribution providing rvs() and mean().
    """

    scenario_id: int
    label: str
    n: int
    p : int
    A : float
    k : int
    rho : float
    dgp: object


def make_scenarios(settings):
    """Return the scenarios specified by a simulation settings dictionary.

    Parameters
    ----------
    settings : dict
        The simulation section of config.json, including its scenario list.

    Returns
    -------
    list of Scenario
        Conditions with frozen distributions and their true means available.
    """
    distributions = {"normal": norm, "exponential": expon, "lognormal": lognorm}
    return [
        Scenario(item["scenario_id"], item["label"], item["n"],
                 distributions[item["distribution"]](**item["parameters"]))
        for item in settings["scenarios"]
    ]


def make_rng(root_seed, scenario_id, replicate_id, *, purpose_id, method_id=0):
    """Return a generator identified by its random task, not execution order.

    Parameters
    ----------
    root_seed : int
        Recorded master seed for the study.
    scenario_id, replicate_id : int
        Stable scenario ID and zero-based repetition ID.
    purpose_id : int
        Keyword-only stream purpose: DATA (0) or METHOD (1).
    method_id : int, optional
        Keyword-only method ID; leave at zero for data streams.

    Returns
    -------
    numpy.random.Generator
        PCG64 stream initialized through SeedSequence. Use distinct,
        nonnegative task IDs; do not assign IDs from list positions.
    """
    key = [scenario_id, replicate_id, purpose_id, method_id, root_seed]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))