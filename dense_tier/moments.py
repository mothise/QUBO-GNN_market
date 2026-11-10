from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass
class FactoredMoments:
    mu: np.ndarray
    B: np.ndarray
    D: np.ndarray
    shrinkage: float

    @property
    def n(self):
        return self.mu.shape[0]

    def dense_sigma(self):
        return self.B @ self.B.T + np.diag(self.D)

    def corr(self):
        S = self.dense_sigma()
        d = np.sqrt(np.clip(np.diag(S), 1e-18, None))
        return S / np.outer(d, d)


def estimate_moments(R_window):
    T, n = R_window.shape
    mu = R_window.mean(axis=0)
    lw = LedoitWolf(assume_centered=False).fit(R_window)
    d = float(lw.shrinkage_)
    Xc = R_window - mu
    S_trace_over_n = float(np.einsum("ij,ij->", Xc, Xc)) / (T * n)
    B = np.sqrt((1.0 - d) / T) * Xc.T
    D = np.full(n, d * S_trace_over_n)
    return FactoredMoments(mu=mu, B=B, D=D, shrinkage=d)
