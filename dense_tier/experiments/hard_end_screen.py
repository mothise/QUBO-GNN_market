"""Hardness screen at risk weights from 0.9 to 1.

Sample and Ledoit-Wolf covariance, four window lengths, asset subsamples, and samples of a known population.
"""
import argparse
import json
import os
import time
import zlib

import numpy as np

from dense_tier.fast_eval import FastEval, tol_of
from dense_tier.moments import FactoredMoments, estimate_moments
from shared.bruni import MARKETS, load_market, rolling_windows
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"
PROTOCOL_WEEKS = 52
STEP = 12
LARGE_MARKETS = ("SP500", "NASDAQComp")
QS = (0.9, 0.95, 0.99, 1.0)
KS = {"protocol": (5, 10, 20), "length": (10,), "subsample": (10,), "synthetic": (10,), "synthetic_factor": (10,)}
LENGTHS = (26, 104, 208)
SUBSAMPLES = (50, 100, 200, 400, 800)
SIM_WEEKS = (26, 52, 104, 208, 416)
POP_WEEKS = 208
POP_FACTORS = 10
EVERY = 5
N_STARTS = 30


def seed_of(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % 2**31


def sample_moments(R_window):
    T, n = R_window.shape
    mu = R_window.mean(axis=0)
    Xc = R_window - mu
    return FactoredMoments(mu=mu, B=Xc.T / np.sqrt(T), D=np.zeros(n), shrinkage=0.0)


def factor_moments(R_window, n_factors):
    T = R_window.shape[0]
    mu = R_window.mean(axis=0)
    Xc = R_window - mu
    _, sing, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_factors, len(sing))
    B = Vt[:k].T * (sing[:k] / np.sqrt(T))
    total = np.einsum("ij,ij->j", Xc, Xc) / T
    D = np.maximum(total - np.einsum("ij,ij->i", B, B), 1e-3 * total)
    return FactoredMoments(mu=mu, B=B, D=D, shrinkage=0.0)


def estimators(R_window):
    lw = estimate_moments(R_window)
    return {"lw": lw, "sample": sample_moments(R_window)}, float(lw.shrinkage)


def effective_q(q, delta):
    if q >= 1.0:
        return 1.0
    r = (1.0 - delta) * q / (1.0 - q)
    return r / (1.0 + r)


def variance(fm, x, K):
    v = fm.B.T @ x
    return float(v @ v + fm.D @ x) / K**2


def return_variance(R_rows, support):
    if R_rows is None:
        return None
    return float(R_rows[:, support].mean(axis=1).var())


def screen(fm, K, q, seed):
    ev = FastEval(fm, K, q)
    rng = np.random.default_rng(seed)
    x_g = ev.greedy()
    f_g = ev.f(x_g)
    x_gs, f_gs, _ = ev.one_swap(x_g)
    ends, finals = [], []
    for _ in range(N_STARTS):
        x_r, f_r, _ = ev.one_swap(ev.random_start(rng))
        ends.append(x_r)
        finals.append(f_r)
    finals = np.array(finals)
    best = int(np.argmin(finals))
    x_best, f_best = (ends[best], float(finals[best])) if finals[best] < f_gs else (x_gs, float(f_gs))
    tol = tol_of(f_best)
    scale = max(abs(f_best), 1e-30)
    rec = {
        "f_best": f_best, "f_greedy": float(f_g),
        "gap_greedy_swap": float((f_gs - f_best) / scale),
        "hit_rate": float(np.mean(finals <= f_best + tol)),
        "n_locopt": int(len(np.unique(np.round(finals / scale, 9)))),
        "overlap_mean": float(np.mean([x @ x_best for x in ends]) / K),
        "support": np.flatnonzero(x_best > 0.5).tolist(),
    }
    return rec, x_best


def holdout_fields(R, cols, weeks, support):
    end = int(weeks[-1]) + 1
    blocks = {"in": R[weeks][:, cols],
              "out12": R[end:end + 12][:, cols] if end + 12 <= R.shape[0] else None,
              "out52": R[end:end + 52][:, cols] if end + 52 <= R.shape[0] else None}
    return {f"var_{name}": return_variance(rows, support) for name, rows in blocks.items()}


_RETURNS = {}


def market_returns(market):
    if market not in _RETURNS:
        _RETURNS[market] = load_market(market)
    return _RETURNS[market]


def real_block(task):
    part, market, w, T, n_sub = task
    R = market_returns(market)
    end = STEP * w + PROTOCOL_WEEKS
    weeks = np.arange(end - T, end)
    if n_sub is None:
        cols = np.arange(R.shape[1])
    else:
        rng = np.random.default_rng(seed_of("subset", market, w, n_sub))
        cols = np.sort(rng.choice(R.shape[1], n_sub, replace=False))
    fms, delta = estimators(R[weeks][:, cols])
    n = len(cols)
    rows = []
    for K in KS[part]:
        for q in QS:
            for est, fm in fms.items():
                rec, _ = screen(fm, K, q, seed_of(part, market, w, T, n_sub, K))
                rec.update(holdout_fields(R, cols, weeks, np.asarray(rec["support"])))
                rec.update({"part": part, "market": market, "window": w, "T": T, "n": n, "n_over_T": n / T,
                            "n_sub": n_sub, "K": K, "q": q, "est": est, "delta": delta,
                            "q_eff": effective_q(q, delta) if est == "lw" else q, "sim_T": None})
                rows.append(rec)
    return rows


def synthetic_block(task):
    part, market, w, pop_weeks, _ = task
    R = market_returns(market)
    end = STEP * w + PROTOCOL_WEEKS
    long_window = R[end - pop_weeks:end]
    pop = factor_moments(long_window, POP_FACTORS) if part == "synthetic_factor" else estimate_moments(long_window)
    n, k = pop.B.shape
    rows = []
    for K in KS[part]:
        for q in QS:
            seed = seed_of(part, market, w, pop_weeks, K)
            rec, x_pop = screen(pop, K, q, seed)
            base = {"part": part, "market": market, "window": w, "T": pop_weeks, "n": n, "n_sub": None,
                    "K": K, "q": q}
            rec.update(base, est="population", sim_T=None, n_over_T=None, delta=float(pop.shrinkage), q_eff=q,
                       var_in=variance(pop, x_pop, K), var_true=variance(pop, x_pop, K))
            rows.append(rec)
            for sim_T in SIM_WEEKS:
                rng = np.random.default_rng(seed_of("draw", market, w, pop_weeks, sim_T))
                X = (pop.mu + rng.standard_normal((sim_T, k)) @ pop.B.T
                     + np.sqrt(pop.D) * rng.standard_normal((sim_T, n)))
                fms, delta = estimators(X)
                for est, fm in fms.items():
                    rec, x_best = screen(fm, K, q, seed)
                    rec.update(base, est=est, sim_T=sim_T, n_over_T=n / sim_T, delta=delta,
                               q_eff=effective_q(q, delta) if est == "lw" else q,
                               var_in=variance(fm, x_best, K), var_true=variance(pop, x_best, K),
                               pop_overlap=float(x_best @ x_pop) / K)
                    rows.append(rec)
    return rows


def block_task(task):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    block = synthetic_block if task[0].startswith("synthetic") else real_block
    return task, block(task)


def block_id(task):
    return "|".join(str(v) for v in task)


def build_tasks():
    tasks = []
    for market in MARKETS:
        R = market_returns(market)
        for w in range(len(rolling_windows(R.shape[0]))):
            end = STEP * w + PROTOCOL_WEEKS
            tasks.append(("protocol", market, w, PROTOCOL_WEEKS, None))
            tasks += [("length", market, w, T, None) for T in LENGTHS if end - T >= 0]
            if market in LARGE_MARKETS:
                tasks += [("subsample", market, w, PROTOCOL_WEEKS, m) for m in SUBSAMPLES if m < R.shape[1]]
            if w % EVERY == 0 and end - POP_WEEKS >= 0:
                tasks += [(part, market, w, POP_WEEKS, None) for part in ("synthetic", "synthetic_factor")]
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "hard_end_screen.jsonl"))
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    tasks = build_tasks()
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["block"] for line in open(args.out)}
    todo = [t for t in tasks if block_id(t) not in done]
    print(f"{len(tasks)} blocks, {len(todo)} to do, {args.procs} workers", flush=True)

    t0 = time.time()
    with open(args.out, "a") as fh:
        if args.procs > 1:
            import multiprocessing as mp
            with mp.Pool(args.procs) as pool:
                n_rows = _write(pool.imap_unordered(block_task, todo, chunksize=1), fh, len(todo), t0)
        else:
            n_rows = _write(map(block_task, todo), fh, len(todo), t0)
    print(f"done: {n_rows} rows in {time.time() - t0:.0f}s -> {args.out}", flush=True)


def _write(results, fh, total, t0):
    n_rows = 0
    for i, (task, rows) in enumerate(results):
        for r in rows:
            r["block"] = block_id(task)
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        n_rows += len(rows)
        if (i + 1) % 25 == 0 or i + 1 == total:
            print(f"  {i + 1}/{total} blocks, {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
    return n_rows


if __name__ == "__main__":
    main()
