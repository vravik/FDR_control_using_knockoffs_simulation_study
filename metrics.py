"""Losses and one helper for Monte Carlo uncertainty in a paired ratio."""

import numpy as np


def evaluate(beta_hat, data):
    """Realized squared losses, without averaging or dividing by p.

    Prediction loss concerns the conditional mean at a new predictor with
    the same covariance; it does not include new response noise.
    """
    difference = beta_hat - data["beta_true"]
    return {"coefficient_error": float(difference @ difference),
            "prediction_error": float(difference @ data["covariance"] @ difference)}


def paired_ratio(lad, ols):
    """Return mean(LAD)/mean(OLS) and its paired, delta-method MCSE.

    Inputs are one-dimensional loss arrays for the SAME successful
    replicates, in the same order, with at least two pairs. Pair by replicate
    ID before calling. This helper supplies the uncertainty calculation,
    not the data reshaping or plotting.

    For R = mean(LAD)/mean(OLS), the replicate influence values are
    (LAD_b - R * OLS_b) / mean(OLS). Their SD/sqrt(B) accounts for pairing.
    """
    lad, ols = np.asarray(lad), np.asarray(ols)
    ratio = lad.mean() / ols.mean()
    influence = (lad - ratio * ols) / ols.mean()
    return ratio, influence.std(ddof=1) / np.sqrt(len(influence))
