"""Hardness screen of the portfolio instances at a fixed covariance shrinkage, from 0 to 0.9."""
import argparse
import json
import math
import os
import time

import numpy as np

import pandas as pd
from sklearn.covariance import LedoitWolf

from dense_tier.fast_eval import FastEval, tol_of
from dense_tier.moments import FactoredMoments
from shared.bruni import load_market, rolling_windows
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"

EXACT_LIMIT = 2_000_000


def imposed_moments(R_window, delta):
    T, n = R_window.shape
    mu = R_window.mean(axis=0)
    Xc = R_window - mu
    tr_over_n = float(np.einsum("ij,ij->", Xc, Xc)) / (T * n)
    B = np.sqrt(max(1.0 - delta, 0.0) / T) * Xc.T
    D = np.full(n, delta * tr_over_n)
    return FactoredMoments(mu=mu, B=B, D=D, shrinkage=delta)


def lw_delta(R_window):
    return float(LedoitWolf(assume_centered=False).fit(R_window).shrinkage_)


def hardness_metrics(fm, K, q, n_starts, seed):
    ev = FastEval(fm, K, q)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    x_g = ev.greedy()
    f_g = ev.f(x_g)
    _, f_gs, _ = ev.one_swap(x_g)

    finals = np.array([ev.one_swap(ev.random_start(rng))[1]
                       for _ in range(n_starts)])

    f_best = min(f_gs, float(finals.min()))
    f_exact = None
    if math.comb(ev.n, K) <= EXACT_LIMIT:
        _, f_exact = ev.exact_enum()
        f_best = min(f_best, f_exact)

    tol = tol_of(f_best)
    uniq = np.unique(np.round(finals / max(abs(f_best), 1e-30), 9))
    return {
        "f_greedy": f_g, "f_greedy_swap": f_gs, "f_best": f_best,
        "f_exact": f_exact,
        "gap_greedy_swap": (f_gs - f_best) / abs(f_best),
        "gswap_optimal": bool(f_gs <= f_best + tol),
        "hit_rate": float(np.mean(finals <= f_best + tol)),
        "n_locopt": int(len(uniq)),
        "seconds": time.time() - t0,
    }


def pick_instances(hard_csv):
    hard = pd.read_csv(hard_csv, float_precision="round_trip")
    picks = []
    for market, g in hard.groupby("market"):
        g = g.sort_values("gap_greedy_swap", ascending=False)
        take = g if g.n.iloc[0] <= 100 else g.head(3)
        for _, r in take.iterrows():
            picks.append({"market": str(market), "window": int(r.window),
                          "K": int(r.K), "q": float(r.q),
                          "n": int(r.n),
                          "screen_hit": float(r.hit_rate),
                          "screen_gap": float(r.gap_greedy_swap)})
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", default=str(RESULTS / "hard_subset.csv"))
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--n_starts", type=int, default=30)
    ap.add_argument("--max_n", type=int, default=10_000)
    ap.add_argument("--out", default=str(RESULTS / "delta_sweep.jsonl"))
    args = ap.parse_args()

    picks = [p for p in pick_instances(args.hard) if p["n"] <= args.max_n]

    cache = {}
    mode = "a" if os.path.exists(args.out) else "w"
    done = set()
    if mode == "a":
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["market"], r["window"], r["K"], r["q"], round(r["delta"], 3)))

    with open(args.out, mode) as fh:
        for p in picks:
            key = (p["market"], p["window"])
            if key not in cache:
                R = load_market(p["market"])
                tr, _ = rolling_windows(R.shape[0])[p["window"]]
                cache[key] = (R[tr], lw_delta(R[tr]))
            Rw, d_hat = cache[key]
            for delta in args.deltas:
                if (p["market"], p["window"], p["K"], p["q"], round(delta, 3)) in done:
                    continue
                fm = imposed_moments(Rw, delta)
                rec = hardness_metrics(fm, p["K"], p["q"], args.n_starts,
                                       seed=(p["window"] * 1000 + p["K"]))
                rec.update({"market": p["market"], "window": p["window"],
                            "K": p["K"], "q": p["q"], "n": p["n"],
                            "delta": delta, "lw_delta_hat": d_hat,
                            "screen_hit": p["screen_hit"],
                            "screen_gap": p["screen_gap"]})
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            print(f"  done {p['market']}_w{p['window']}_K{p['K']} "
                  f"(n={p['n']}, LW delta_hat={d_hat:.3f})", flush=True)


if __name__ == "__main__":
    main()
