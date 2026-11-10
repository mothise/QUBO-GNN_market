"""Certify dense instances with HiGHS"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, identity

from dense_tier.experiments.certify_gurobi import warm_start
from dense_tier.fast_eval import FastEval
from dense_tier.moments import estimate_moments
from shared.bruni import load_market, rolling_windows
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"
Y_SCALE = 100.0
GAP_TOL = 2e-6
GRID = 24


def certify(fm, K, q, time_limit, x0, f0, milp_time=300.0):
    ev = FastEval(fm, K, q)
    n, k = ev.B.shape
    scale = Y_SCALE ** 2
    Bs = Y_SCALE * ev.B
    linear = scale * (ev.a * ev.D - ev.b * ev.mu)
    t_start = time.perf_counter()
    x_best, f_best = x0.copy(), float(f0)

    ordered = np.sort(Bs, axis=0)
    y_low, y_high = ordered[:K].sum(axis=0), ordered[-K:].sum(axis=0)

    n_vars = n + 2 * k
    objective = np.concatenate([linear, np.zeros(k), ev.a * np.ones(k)])
    integrality = np.concatenate([np.ones(n), np.zeros(2 * k)])
    bounds = Bounds(np.concatenate([np.zeros(n), y_low, np.zeros(k)]),
                    np.concatenate([np.ones(n), y_high, np.full(k, np.inf)]))
    link = hstack([csr_matrix(Bs.T), -identity(k, format="csr"), csr_matrix((k, k))])
    cardinality = csr_matrix(np.concatenate([np.ones(n), np.zeros(2 * k)])[None, :])

    rows, cols, vals, rhs = [], [], [], []
    placed = [set() for _ in range(k)]

    def add_tangents(y):
        for j in range(k):
            u = float(y[j])
            key = round(u, 9)
            if key in placed[j]:
                continue
            placed[j].add(key)
            r = len(rhs)
            rows.extend([r, r])
            cols.extend([n + k + j, n + j])
            vals.extend([1.0, -2.0 * u])
            rhs.append(-u * u)

    for y in np.linspace(y_low, y_high, GRID):
        add_tangents(y)
    add_tangents(Bs.T @ x_best)

    lower, status, rounds = -np.inf, "time_limit", 0
    while True:
        remaining = time_limit - (time.perf_counter() - t_start)
        if remaining <= 1.0:
            break
        tangents = coo_matrix((vals, (rows, cols)), shape=(len(rhs), n_vars)).tocsr()
        constraints = [LinearConstraint(link, 0.0, 0.0), LinearConstraint(cardinality, K, K),
                       LinearConstraint(tangents, np.array(rhs), np.inf)]
        res = milp(objective, constraints=constraints, integrality=integrality, bounds=bounds,
                   options=dict(time_limit=min(milp_time, remaining), mip_rel_gap=0.0))
        rounds += 1
        dual = getattr(res, "mip_dual_bound", None)
        if dual is not None and np.isfinite(dual):
            lower = max(lower, float(dual))
        if res.x is None:
            status = "failed"
            break
        x = np.round(res.x[:n])
        f_x = ev.f(x)
        x_swapped, f_swapped, _ = ev.one_swap(x)
        for cand, f_cand in ((x, f_x), (x_swapped, f_swapped)):
            if f_cand < f_best:
                x_best, f_best = cand.copy(), float(f_cand)
        if lower >= scale * f_best - GAP_TOL * max(1.0, abs(scale * f_best)):
            status = "optimal"
            break
        before = len(rhs)
        add_tangents(Bs.T @ x)
        add_tangents(Bs.T @ x_swapped)
        if len(rhs) == before and res.status == 0:
            status = "failed"
            break
    bound = lower / scale
    return {"status": status, "obj": f_best, "bound": bound,
            "mip_gap": (f_best - bound) / abs(f_best) if f_best != 0 else None,
            "seconds": time.perf_counter() - t_start, "rounds": rounds, "tangents": len(rhs),
            "support": np.flatnonzero(x_best > 0.5).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", default=str(RESULTS / "hard_subset.csv"))
    ap.add_argument("--out", default=str(RESULTS / "certify_highs.jsonl"))
    ap.add_argument("--time_limit", type=float, default=1200.0)
    args = ap.parse_args()

    hard = pd.read_csv(args.hard, float_precision="round_trip")
    rows = hard.sort_values(["n", "market", "window", "K"]).to_dict("records")
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["market"], int(r["window"]), int(r["K"]), float(r["q"])))

    cache = {}
    with open(args.out, "a") as fh:
        for i, r in enumerate(rows):
            key = (r["market"], int(r["window"]), int(r["K"]), float(r["q"]))
            if key in done:
                continue
            if key[:2] not in cache:
                R = load_market(r["market"])
                tr, _ = rolling_windows(R.shape[0])[int(r["window"])]
                cache[key[:2]] = estimate_moments(R[tr])
            fm = cache[key[:2]]
            K, q = int(r["K"]), float(r["q"])

            ev = FastEval(fm, K, q)
            x_ws, f_ws = warm_start(ev, seed=int(r["window"]))
            res = certify(fm, K, q, args.time_limit, x_ws, f_ws)
            rec = {"market": r["market"], "window": int(r["window"]), "K": K, "q": q,
                   "n": int(fm.n), "time_limit": float(args.time_limit), "f_warmstart": f_ws,
                   "f_best_screen": float(r["f_best"]), "engine": "highs, separable tangents", **res}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{i + 1}/{len(rows)} {r['market']}_w{r['window']}_K{K} n={fm.n}: "
                  f"{res['status']} obj={res['obj']} obj-bound={res['obj'] - res['bound']:.1e} "
                  f"rounds={res['rounds']} ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
