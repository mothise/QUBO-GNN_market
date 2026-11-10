"""
Best portfolio known on the instances the exact solver left open
"""
import argparse
import json
import os
import time

import numpy as np

from dense_tier.fast_eval import FastEval, tol_of
from dense_tier.moments import estimate_moments
from shared.bruni import load_market, rolling_windows
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"
CERTIFY = RESULTS / "hard_end_certify_large.jsonl"
SCREEN = RESULTS / "hard_end_screen.jsonl"
STARTS = 30000
BLOCKS = (1, 30, 300, 3000, 10000, 30000)


def key_of(r):
    return (r["market"], int(r["window"]), int(r["K"]), float(r["q"]))


def indicator(n, support):
    x = np.zeros(n)
    x[np.asarray(support, dtype=int)] = 1.0
    return x


def block_minima(ends, k):
    return ends[: (len(ends) // k) * k].reshape(-1, k).min(axis=1)


def reference_row(task):
    c, s = task
    R = load_market(c["market"])
    fm = estimate_moments(R[rolling_windows(R.shape[0])[c["window"]][0]])
    K, q = int(c["K"]), float(c["q"])
    ev = FastEval(fm, K, q)

    x_solver = indicator(fm.n, c["support"])
    if abs(ev.f(x_solver) - c["obj"]) > 1e-6 * abs(c["obj"]):
        raise ValueError("certificate mismatch")
    x_screen = indicator(fm.n, s["support"])
    if abs(ev.f(x_screen) - s["f_best"]) > 1e-6 * abs(s["f_best"]):
        raise ValueError("screen mismatch")

    rng = np.random.default_rng([1711, int(c["window"]), K, int(round(1000 * q))])
    ends = np.empty(STARTS)
    f_restarts, x_restarts = np.inf, None
    t0 = time.perf_counter()
    for i in range(STARTS):
        x, f, _ = ev.one_swap(ev.random_start(rng))
        ends[i] = f
        if f < f_restarts:
            f_restarts, x_restarts = f, x
    seconds = time.perf_counter() - t0

    cands = [("solver incumbent", float(ev.f(x_solver)), x_solver),
             ("30 starts of the screen", float(ev.f(x_screen)), x_screen),
             (f"{STARTS} starts", float(ev.f(x_restarts)), x_restarts)]
    f_best = min(f for _, f, _ in cands)
    tol = float(tol_of(f_best))
    x_best = next(x for _, f, x in cands if f <= f_best + tol)
    return {
        "market": c["market"], "window": int(c["window"]), "K": K, "q": q, "n": int(fm.n),
        "status": "best_known", "obj": float(ev.f(x_best)),
        "support": [int(i) for i in np.flatnonzero(x_best > 0.5)],
        "held_by": [name for name, f, _ in cands if f <= f_best + tol],
        "f_solver": cands[0][1], "f_screen30": cands[1][1], "f_restarts": cands[2][1],
        "solver_bound": float(c["bound"]), "open_gap": (f_best - float(c["bound"])) / abs(f_best),
        "starts": STARTS,
        "block_best": {str(k): float(block_minima(ends, k).mean()) for k in BLOCKS},
        "block_at_best": {str(k): float((block_minima(ends, k) <= f_best + tol).mean()) for k in BLOCKS},
        "share_of_starts_at_best": float(np.mean(ends <= f_best + tol)),
        "distinct_ends": int(len(np.unique(np.round(ends / abs(f_best), 9)))),
        "seconds_per_start": seconds / STARTS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "hard_end_best_known.jsonl"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    certs = [json.loads(line) for line in open(CERTIFY)]
    certs = [c for c in certs if c["status"] != "optimal" and c.get("support")]
    certs.sort(key=key_of)
    done = set()
    if os.path.exists(args.out):
        done = {key_of(json.loads(line)) for line in open(args.out)}
    wanted = {key_of(c) for c in certs} - done
    screen = {}
    for line in open(SCREEN):
        r = json.loads(line)
        if r["part"] == "protocol" and r["est"] == "lw" and key_of(r) in wanted:
            screen[key_of(r)] = r
    tasks = [(c, screen[key_of(c)]) for c in certs if key_of(c) in wanted]
    print(f"{len(certs)} open instances, {len(tasks)} to do, {STARTS} starts each, {args.workers} workers", flush=True)

    with open(args.out, "a") as fh:
        if args.workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(args.workers) as pool:
                _write(pool.imap_unordered(reference_row, tasks, chunksize=1), fh, len(tasks))
        else:
            _write(map(reference_row, tasks), fh, len(tasks))


def _write(results, fh, total):
    for j, row in enumerate(results):
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(f"  [{j + 1}/{total}] {row['market']}_w{row['window']}_K{row['K']} q={row['q']:g}: obj={row['obj']:.6e} "
              f"held by {', '.join(row['held_by'])}; at best {row['share_of_starts_at_best']:.3%} of starts, "
              f"{row['distinct_ends']} distinct ends, gap to bound {row['open_gap']:.1%} "
              f"({1000 * row['seconds_per_start']:.1f} ms per start)", flush=True)


if __name__ == "__main__":
    main()
