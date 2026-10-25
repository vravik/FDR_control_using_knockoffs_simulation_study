"""Regression DGPs sharing the generate(rng) interface from Lecture 6.

We only have one error distribution. 

Design matrix depends on the rho parameter -- we sample this once for each 
scenario-id. 
Instances of the class vary in how beta is defined -- we pass these parameters 
as inputs for the generate function

We can use the methods in the knockpy package to generate the design matrix and
coefficients
"""


from abc import ABC, abstractmethod

import numpy as np
import knockpy


class RegressionDGP(ABC):
    """Gaussian predictors with variable feature correlations, randomly sampled 
    beta, fixed error distribution.

    n , p fixed. Rho defines correlation structure of features.
    Parameters for generating regresison output include sparsity k, 
    Amplitude A that are directly passed to generate method

    Parameters satisfy 0 < p < n and 0 <= rho < 1. generate() takes a
    NumPy Generator, arguments k, A and returns X, y, beta_true in a dictionary.
    Array shapes are (n, p), (n,), (p,), (p, p), respectively.
    """

    def __init__(self, n, p):
        self.n, self.p= n, p
        self.beta = np.zeros(p)
                
    @abstractmethod
    def generate(self, rng, rho, k, A):
        """Return Design matrix and variables from the regression model."""
        

class NormalDGP(RegressionDGP):
    def generate(self, rng, rho=0, k=10, A=3.5):
        index = np.arange(self.p)
        if(rho == 0):
            covariance = np.identity(self.p)
        else:
            covariance = rho ** np.abs(index[:, None] - index[None, :])
        factor = np.linalg.cholesky(covariance)
        X = rng.normal(size=(self.n, self.p)) @ factor.T
        self.X = X / np.linalg.norm(X, axis=0)
        indices = rng.choice(self.p, size = k, replace=False)
        coef_sign = rng.choice([-1, 1], size = k)
        self.beta = np.zeros(self.p)
        self.beta[indices] = coef_sign * A
        return {"X": self.X, "y": self.X @ self.beta + rng.normal(size=self.n),
                "beta_true": self.beta, "covariance":covariance}

DGP_CLASSES = {"normal": NormalDGP}
