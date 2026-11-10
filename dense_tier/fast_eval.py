from itertools import combinations, islice

import numpy as np

REL_TOL = 1e-9
ABS_TOL = 1e-15


def tol_of(f_star):
    f_star = np.asarray(f_star, dtype=float)
    return REL_TOL * np.abs(f_star) + ABS_TOL


class FastEval:
    def __init__(self, fm, K, q):
        self.B = np.ascontiguousarray(fm.B)
        self.D = fm.D
        self.mu = fm.mu
        self.n, self.k = self.B.shape
        self.K = K
        self.a = q / K**2
        self.b = (1.0 - q) / K
        self.Bn2 = np.einsum("ij,ij->i", self.B, self.B)

    def f(self, x):
        v = self.B.T @ x
        return self.a * (v @ v + self.D @ x) - self.b * (self.mu @ x)

    def f_batch(self, X):
        V = X @ self.B
        return self.a * (np.einsum("ij,ij->i", V, V) + X @ self.D) \
            - self.b * (X @ self.mu)

    def greedy(self):
        x = np.zeros(self.n)
        v = np.zeros(self.k)
        chosen = np.zeros(self.n, dtype=bool)
        for _ in range(self.K):
            c = self.B @ v
            delta = self.a * (2 * c + self.Bn2 + self.D) - self.b * self.mu
            delta[chosen] = np.inf
            j = int(np.argmin(delta))
            x[j] = 1.0
            chosen[j] = True
            v += self.B[j]
        return x

    def one_swap(self, x0, max_swaps=2000):
        x = x0.copy()
        v = self.B.T @ x
        f_cur = self.a * (v @ v + self.D @ x) - self.b * (self.mu @ x)
        for it in range(max_swaps):
            ins = np.flatnonzero(x > 0.5)
            outs = np.flatnonzero(x < 0.5)
            c = self.B @ v
            cross = self.B[ins] @ self.B[outs].T
            dq = (-2.0 * c[ins][:, None] + 2.0 * c[outs][None, :]
                  + self.Bn2[ins][:, None] + self.Bn2[outs][None, :]
                  - 2.0 * cross
                  + self.D[outs][None, :] - self.D[ins][:, None])
            df = self.a * dq - self.b * (self.mu[outs][None, :] - self.mu[ins][:, None])
            bi, bj = np.unravel_index(int(np.argmin(df)), df.shape)
            if df[bi, bj] < -1e-15:
                i, j = int(ins[bi]), int(outs[bj])
                x[i], x[j] = 0.0, 1.0
                v += self.B[j] - self.B[i]
                f_cur += float(df[bi, bj])
            else:
                return x, float(f_cur), it
        return x, float(f_cur), max_swaps

    def random_start(self, rng):
        x = np.zeros(self.n)
        x[rng.choice(self.n, self.K, replace=False)] = 1.0
        return x

    def exact_enum(self, batch=200_000):
        best_val, best_x = np.inf, None
        it = combinations(range(self.n), self.K)
        while True:
            buf = list(islice(it, batch))
            if not buf:
                break
            X = np.zeros((len(buf), self.n))
            rows = np.repeat(np.arange(len(buf)), self.K)
            X[rows, np.asarray(buf).ravel()] = 1.0
            vals = self.f_batch(X)
            bidx = int(np.argmin(vals))
            if vals[bidx] < best_val:
                best_val, best_x = float(vals[bidx]), X[bidx]
        return best_x, best_val
