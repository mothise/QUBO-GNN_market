"""Hardness screen over every (market, window, K, q) cell.

Per instance: greedy, greedy + 1-swap, S random-start 1-swaps, hit_rate (fraction of starts reaching the best),
n_locopt, and three structure metrics of Sigma.
"""
import argparse
import json
import math
import os
import time
import zlib

import numpy as np
import pandas as pd

from dense_tier.fast_eval import FastEval, tol_of
from dense_tier.moments import estimate_moments
from shared.bruni import MARKETS, load_market, rolling_windows
from shared.paths import REPO_ROOT

EXACT_LIMIT = 150_000
RESULTS = REPO_ROOT / "dense_tier" / "results"
OUT_DEFAULT = RESULTS / "hardness_screen.jsonl"
HARD_DEFAULT = RESULTS / "hard_subset.csv"


def instance_seed(market, widx, K):
    return zlib.crc32(f"{market}|{widx}|{K}".encode()) % 2**31


def structure_metrics(fm):
    G = fm.B.T @ fm.B
    eigs = np.linalg.eigvalsh(G) + fm.D[0]
    lam1 = float(eigs[-1])
    tr = float(np.sum(eigs) + (fm.n - len(eigs)) * fm.D[0])
    return {"shrinkage": float(fm.shrinkage), "lam1_share": lam1 / tr, "stable_rank": tr / lam1}


def run_instance(fm, K, q, n_starts, seed):
    ev = FastEval(fm, K, q)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    x_g = ev.greedy()
    f_g = ev.f(x_g)
    _, f_gs, _ = ev.one_swap(x_g)

    finals = np.array([ev.one_swap(ev.random_start(rng))[1] for _ in range(n_starts)])
    f_best = min(f_gs, float(finals.min()))
    f_exact = None
    if math.comb(ev.n, K) <= EXACT_LIMIT:
        _, f_exact = ev.exact_enum()
        f_best = min(f_best, f_exact)

    tol = tol_of(f_best)
    uniq = np.unique(np.round(finals / max(abs(f_best), 1e-30), 9))
    return {
        "f_greedy": f_g, "f_greedy_swap": f_gs, "f_best": f_best, "f_exact": f_exact,
        "gap_greedy": (f_g - f_best) / abs(f_best),
        "gap_greedy_swap": (f_gs - f_best) / abs(f_best),
        "greedy_optimal": bool(f_g <= f_best + tol),
        "gswap_optimal": bool(f_gs <= f_best + tol),
        "hit_rate": float(np.mean(finals <= f_best + tol)),
        "n_locopt": int(len(uniq)),
        "seconds": time.time() - t0,
    }


def window_task(args):
    market, widx, tr_idx, R, Ks, qs, n_starts = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    fm = estimate_moments(R[tr_idx])
    struct = structure_metrics(fm)
    rows = []
    for K in Ks:
        if K >= fm.n:
            continue
        for q in qs:
            rec = run_instance(fm, K, q, n_starts, seed=instance_seed(market, widx, K))
            rec.update({"market": market, "window": widx, "K": K, "q": q, "n": fm.n, **struct})
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=MARKETS)
    ap.add_argument("--Ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--qs", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--n_starts", type=int, default=10)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--hard_out", default=str(HARD_DEFAULT))
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    tasks = []
    for market in args.markets:
        R = load_market(market)
        for widx, (tr_idx, _) in enumerate(rolling_windows(R.shape[0])):
            tasks.append((market, widx, tr_idx, R, args.Ks, args.qs, args.n_starts))
    print(f"{len(tasks)} windows -> up to {len(tasks) * len(args.Ks) * len(args.qs)} "
          f"instances, {args.procs} workers", flush=True)

    t0 = time.time()
    n_rows = 0
    with open(args.out, "w") as fh:
        if args.procs > 1:
            import multiprocessing as mp
            with mp.Pool(args.procs) as pool:
                for i, rows in enumerate(pool.imap_unordered(window_task, tasks, chunksize=2)):
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                    n_rows += len(rows)
                    if (i + 1) % 25 == 0:
                        print(f"  {i + 1}/{len(tasks)} windows, {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
        else:
            for t in tasks:
                for r in window_task(t):
                    fh.write(json.dumps(r) + "\n")
                    n_rows += 1
    print(f"done: {n_rows} rows in {time.time() - t0:.0f}s -> {args.out}", flush=True)

    df = pd.DataFrame([json.loads(l) for l in open(args.out)])
    hard = df[(df.hit_rate < 0.5) | (df.gap_greedy_swap > 1e-4)]
    hard.sort_values(["market", "window", "K", "q"]).to_csv(args.hard_out, index=False)
    print(f"discriminative instances: {len(hard)}/{len(df)} -> {args.hard_out}", flush=True)


if __name__ == "__main__":
    main()
