"""PQQA (Ichikawa & Arai, 2025) at the settings of its paper: parallel annealing of relaxed variables, no network."""
import time

import numpy as np
import torch

SETTINGS = dict(lr=0.1, weight_decay=0.01, temperature=1e-3, entropy_exponent=4, gamma_min=-2.0, gamma_max=0.1,
                communication=0.2, penalty=2.0, betas=(0.9, 0.999), adam_eps=1e-8)
STEPS = 3000
STD_FLOOR = 1e-12


class IndependentSet:
    def __init__(self, edges, n):
        self.n = int(n)
        self.edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self._adjacency = None

    def adjacency(self, P):
        if self._adjacency is None:
            e = torch.as_tensor(self.edges, device=P.device)
            A = torch.zeros((self.n, self.n), dtype=P.dtype, device=P.device)
            A[e[:, 0], e[:, 1]] = 1.0
            A[e[:, 1], e[:, 0]] = 1.0
            self._adjacency = A
        return self._adjacency

    def loss(self, P):
        return -P.sum(dim=1) + 0.5 * SETTINGS["penalty"] * ((P @ self.adjacency(P)) * P).sum(dim=1)

    def feasible_sizes(self, P):
        sets = P > 0.5
        chosen = sets.to(P.dtype)
        violated = ((chosen @ self.adjacency(P)) * chosen).sum(dim=1)
        return torch.where(violated == 0, sets.sum(dim=1), torch.full_like(sets.sum(dim=1), -1))


class Portfolio:
    def __init__(self, fm, K, q, form):
        self.n = int(fm.n)
        self.K = int(K)
        self.a, self.b = q / self.K**2, (1.0 - q) / self.K
        self.B, self.D, self.mu = (np.asarray(v, dtype=float) for v in (fm.B, fm.D, fm.mu))
        sigma_ii = np.einsum("ij,ij->i", self.B, self.B) + self.D
        self.lam = 100.0 * (q * float(np.median(np.abs(sigma_ii))) / self.K**2
                            + (1.0 - q) * float(np.median(np.abs(self.mu))) / self.K)
        self.form = form
        self.unit = self.lam / 100.0
        self._tensors = None

    def loss(self, P):
        if self._tensors is None:
            self._tensors = tuple(torch.as_tensor(v, dtype=P.dtype, device=P.device) for v in (self.B, self.D, self.mu))
        B, D, mu = self._tensors
        V = P @ B
        quad = (V * V).sum(dim=1) + (P * P) @ D
        total = P.sum(dim=1)
        if self.form == "convex":
            loss = self.a * quad - self.b * (P @ mu) + self.lam * (total - self.K) ** 2
        else:
            squares = (P * P).sum(dim=1)
            loss = (self.a * quad - self.b * ((P * P) @ mu)
                    + self.lam * (total**2 - 2 * self.K * squares + self.K**2))
        return loss / self.unit


def anneal(problem, runs, steps, seed, device, lr=SETTINGS["lr"], track_every=0):
    s = dict(SETTINGS, lr=lr)
    t0 = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    P0 = np.random.default_rng(seed).random((runs, problem.n))
    P = torch.nn.Parameter(torch.as_tensor(P0, dtype=torch.float32, device=device).clone())
    optimizer = torch.optim.AdamW([P], lr=s["lr"], betas=s["betas"], eps=s["adam_eps"], weight_decay=s["weight_decay"])
    a = s["entropy_exponent"]
    noise = float(np.sqrt(2.0 * s["lr"] * s["temperature"]))
    best_value, best_step = -1, None
    for t in range(steps):
        gamma = s["gamma_min"] + (s["gamma_max"] - s["gamma_min"]) * t / (steps - 1)
        loss = (problem.loss(P) + gamma * (1.0 - (2.0 * P - 1.0) ** a).sum(dim=1)).sum()
        if runs > 1:
            variance = ((P - P.mean(dim=0, keepdim=True)) ** 2).mean(dim=0)
            loss = loss - runs * s["communication"] * torch.sqrt(variance + STD_FLOOR).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            P.add_(noise * torch.randn(P.shape, generator=generator, dtype=torch.float32, device=device))
            P.clamp_(0.0, 1.0)
            if track_every and (t % track_every == 0 or t == steps - 1):
                top = int(problem.feasible_sizes(P).max())
                if top > best_value:
                    best_value, best_step = top, t
    P_end = P.detach().cpu().numpy().astype(np.float64)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    s.update(runs=int(runs), steps=int(steps), betas=list(s["betas"]))
    return P_end, s, seconds, (best_value, best_step) if track_every else None
