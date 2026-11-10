"""
Exhaustive enumeration against the HiGHS certificates
"""
import argparse
import json
import math
import os
import time

import numpy as np

from dense_tier.fast_eval import FastEval, tol_of
from dense_tier.moments import estimate_moments
from shared.bruni import load_market, rolling_windows
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"
LISTS = ("hard_end_certify_flagged.jsonl", "hard_end_certify_independent.jsonl")
LIMIT = 3e7


def key_of(r):
    return (r["market"], int(r["window"]), int(r["K"]), float(r["q"]))


def enumerate_one(c):
    market, window, K, q = key_of(c)
    R = load_market(market)
    fm = estimate_moments(R[rolling_windows(R.shape[0])[window][0]])
    ev = FastEval(fm, K, q)
    t0 = time.perf_counter()
    x, f = ev.exact_enum()
    seconds = time.perf_counter() - t0
    x_cert = np.zeros(fm.n)
    x_cert[np.asarray(c["support"], dtype=int)] = 1.0
    f_cert = float(ev.f(x_cert))
    return {"market": market, "window": window, "K": K, "q": q, "n": int(fm.n), "portfolios": math.comb(fm.n, K),
            "f_enumeration": float(f), "f_certificate": f_cert,
            "same_objective": bool(abs(f - f_cert) <= tol_of(f_cert)),
            "same_assets": bool(np.array_equal(x > 0.5, x_cert > 0.5)), "seconds": seconds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "hard_end_enum.jsonl"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    certs = {}
    for name in LISTS:
        for line in open(RESULTS / name):
            c = json.loads(line)
            if c["status"] == "optimal" and math.comb(int(c["n"]), int(c["K"])) <= LIMIT:
                certs[key_of(c)] = c
    done = set()
    if os.path.exists(args.out):
        done = {key_of(json.loads(line)) for line in open(args.out)}
    tasks = [c for k, c in sorted(certs.items(), key=lambda kv: math.comb(kv[1]["n"], kv[1]["K"])) if k not in done]
    print(f"{len(certs)} instances with at most {LIMIT:.0e} portfolios, {len(tasks)} to do", flush=True)

    with open(args.out, "a") as fh:
        if args.workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(args.workers) as pool:
                _write(pool.imap_unordered(enumerate_one, tasks, chunksize=1), fh)
        else:
            _write(map(enumerate_one, tasks), fh)


def _write(results, fh):
    for row in results:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(f"  {row['market']}_w{row['window']}_K{row['K']} q={row['q']:g}: {row['portfolios']:,} portfolios, "
              f"same_objective={int(row['same_objective'])} same_assets={int(row['same_assets'])} "
              f"({row['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
