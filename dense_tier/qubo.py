from dataclasses import dataclass

import numpy as np
import torch

from dense_tier.moments import FactoredMoments


def default_lambda_schedule(lam_final, lam_init_frac=0.01):
    lam0 = lam_init_frac * lam_final

    def lam(frac):
        return lam0 + (lam_final - lam0) * float(frac)

    return lam

@dataclass
class PortfolioQubo:
    fm: FactoredMoments
    K: int
    q: float = 0.5
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.n = self.fm.n
        self._Bt = torch.tensor(self.fm.B.T, dtype=self.dtype, device=self.device)
        self._D = torch.tensor(self.fm.D, dtype=self.dtype, device=self.device)
        self._mu = torch.tensor(self.fm.mu, dtype=self.dtype, device=self.device)
        sigma_ii = np.einsum("ij,ij->i", self.fm.B, self.fm.B) + self.fm.D
        scale = (self.q * np.median(sigma_ii) / self.K**2
                 + (1 - self.q) * np.median(np.abs(self.fm.mu)) / self.K)
        self.lam_final = 100.0 * float(scale)
        self.lam = default_lambda_schedule(self.lam_final)

    def quad(self, p):
        v = self._Bt @ p
        return (v * v).sum() + (self._D * p * p).sum()

    def objective_relaxed(self, p):
        return (self.q / self.K**2) * self.quad(p) - ((1 - self.q) / self.K) * (self._mu @ p)

    def train_loss(self, p, frac):
        pen = (p.sum() - self.K) ** 2
        return self.objective_relaxed(p) + self.lam(frac) * pen

    def feasible_objective(self, x):
        x = np.asarray(x, dtype=np.float64)
        v = self.fm.B.T @ x
        quad = float(v @ v + self.fm.D @ x)
        return (self.q / self.K**2) * quad - ((1 - self.q) / self.K) * float(self.fm.mu @ x)

    def dense_Q(self, lam):
        n = self.n
        Q = (self.q / self.K**2) * self.fm.dense_sigma() + lam * np.ones((n, n))
        Q[np.diag_indices(n)] += -((1 - self.q) / self.K) * self.fm.mu - 2 * lam * self.K
        return Q

    def check_exactness(self, n_trials=64, seed=0):
        rng = np.random.default_rng(seed)
        lam = self.lam(1.0)
        Q = self.dense_Q(lam)
        worst = 0.0
        for _ in range(n_trials):
            x = (rng.random(self.n) < rng.random()).astype(np.float64)
            dense = float(x @ Q @ x) + lam * self.K**2
            fact = self.feasible_objective(x) + lam * (x.sum() - self.K) ** 2
            worst = max(worst, abs(dense - fact) / max(1e-30, abs(dense)))
        return worst
