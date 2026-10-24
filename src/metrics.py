"""Losses and one helper for Monte Carlo uncertainty in a paired ratio.

Since evaluation does not consider magnitude of recovered coefficient, we convert
true beta to a binary vector 
"""

import numpy as np


def evaluate(beta_hat, data):
    """Realized power and false discovery rates, without averaging"""
    rejections = beta_hat.sum()
    if(rejections == 0):
        return({'fdp' : 0, 
                'power' : 0})
    else:
        return {"fdp": np.dot(beta_hat, data['beta_true'] == 0) / rejections,             
            "power": np.dot(beta_hat, data['beta_true'] != 0) /
            (data['beta_true']!=0).sum()}


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
