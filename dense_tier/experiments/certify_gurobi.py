"""Certify the discriminative instances with Gurobi."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from dense_tier.fast_eval import FastEval, tol_of
from dense_tier.moments import estimate_moments
from shared.bruni import load_market, rolling_windows
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"
PIP_QUADRATIC_VAR_CAP = 200


def check_license():
    try:
        import gurobipy as gp
    except Exception as e:
        return f"gurobipy not importable: {e}"
    try:
        m = gp.Model()
        m.Params.OutputFlag = 0
        x = m.addMVar(PIP_QUADRATIC_VAR_CAP + 1, vtype=gp.GRB.BINARY)
        m.setObjective(x @ x)
        m.optimize()
        return f"gurobi {gp.gurobi.version()}: full licence"
    except Exception as e:
        return (f"gurobi {gp.gurobi.version()}: size-limited licence [{type(e).__name__}], "
                f"n <= 148 only; n >= 442 will be rejected")


def warm_start(ev, n_starts=50, seed=0):
    rng = np.random.default_rng(seed)
    x_best, f_best = None, np.inf
    xg = ev.greedy()
    for x0 in [xg] + [ev.random_start(rng) for _ in range(n_starts)]:
        x, f, _ = ev.one_swap(x0)
        if f < f_best:
            x_best, f_best = x, f
    return x_best, float(f_best)


def certify(fm, K, q, time_limit, x0=None, threads=2):
    import gurobipy as gp
    from gurobipy import GRB

    n, k = fm.B.shape
    a = q / K**2
    b = (1.0 - q) / K

    m = gp.Model("ccmv")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.0
    m.Params.Threads = threads

    x = m.addMVar(n, vtype=GRB.BINARY, name="x")
    y = m.addMVar(k, lb=-GRB.INFINITY, name="y")
    m.addConstr(y == fm.B.T @ x)
    m.addConstr(x.sum() == K)
    m.setObjective(a * (y @ y) + (a * fm.D - b * fm.mu) @ x, GRB.MINIMIZE)
    if x0 is not None:
        x.Start = x0
    m.optimize()

    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
              GRB.INTERRUPTED: "interrupted"}.get(m.Status, str(m.Status))
    sol = x.X if m.SolCount > 0 else None
    return {"status": status,
            "obj": float(m.ObjVal) if m.SolCount > 0 else None,
            "bound": float(m.ObjBound),
            "mip_gap": float(m.MIPGap) if m.SolCount > 0 else None,
            "seconds": float(m.Runtime),
            "support": np.flatnonzero(sol > 0.5).tolist() if sol is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", default=str(RESULTS / "hard_subset.csv"))
    ap.add_argument("--out", default=str(RESULTS / "certify.jsonl"))
    ap.add_argument("--time_limit_small", type=float, default=60.0)
    ap.add_argument("--time_limit", type=float, default=3600.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--check_license", action="store_true")
    args = ap.parse_args()

    print(check_license(), flush=True)
    if args.check_license:
        return

    hard = pd.read_csv(args.hard, float_precision="round_trip")
    rows = hard.sort_values(["n", "market", "window", "K"]).to_dict("records")
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            if not str(r["status"]).startswith("error"):
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
            tl = args.time_limit_small if fm.n < 100 else args.time_limit

            ev = FastEval(fm, K, q)
            x_ws, f_ws = warm_start(ev, seed=int(r["window"]))
            t0 = time.time()
            try:
                res = certify(fm, K, q, tl, x0=x_ws, threads=args.threads)
            except Exception as e:
                res = {"status": f"error: {e}", "obj": None, "bound": None,
                       "mip_gap": None, "seconds": time.time() - t0, "support": None}
            if res["status"] == "optimal" and f_ws < res["obj"] - tol_of(res["obj"]):
                raise SystemExit(f"warm start {f_ws!r} below certified optimum {res['obj']!r} on {key}")
            rec = {"market": r["market"], "window": int(r["window"]), "K": K, "q": q,
                   "n": int(fm.n), "time_limit": float(tl), "f_warmstart": f_ws,
                   "f_best_screen": float(r["f_best"]), **res}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{i + 1}/{len(rows)} {r['market']}_w{r['window']}_K{K} n={fm.n}: "
                  f"{res['status']} obj={res['obj']} gap={res['mip_gap']} ({res['seconds']:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
