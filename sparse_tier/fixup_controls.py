"""The greedy fix-up of the QRF-GNN decoder applied to sets that no network produced, and as a restart heuristic."""
import argparse
import json
import os
import time
import zlib

import numpy as np

from shared.paths import REPO_ROOT
from sparse_tier.gnn_mis import fix_independent_set_fast
from sparse_tier.market_graph import correlation, load_returns_bruni, threshold_graph

RESULTS = REPO_ROOT / "sparse_tier" / "results"
SCREENS = ("screen_SP500.jsonl", "screen_NASDAQComp.jsonl")
RUNS = RESULTS / "gnn_qrf_6000_readout.jsonl"
NAMES = RESULTS / "instances_full42.txt"
JITTERS = (0.1, 0.3)
CLASS_COUNTS = (1, 4, 10, 20)
N_SHUFFLES = 300
PEEL_SHARES = (0.1, 0.3)
PEEL_PREFIXES = (50, 300)
RESTARTS = 1200

_RETURNS, _GRAPHS = {}, {}


def instance_graph(row):
    if row["name"] not in _GRAPHS:
        if row["source"] not in _RETURNS:
            _RETURNS[row["source"]] = load_returns_bruni(row["source"])
        R, ws = _RETURNS[row["source"]]
        G = threshold_graph(correlation(R[ws[row["window"]][0]]), row["tau"])
        if G.number_of_edges() != row["m"]:
            raise ValueError("screen/instance mismatch")
        edges = np.array(G.edges(), dtype=np.int64).reshape(-1, 2)
        deg = np.array([d for _, d in sorted(G.degree())], dtype=np.int64)
        _GRAPHS[row["name"]] = (G, edges, deg)
    return _GRAPHS[row["name"]]


def fixed_size(bits, edges, deg):
    return int(fix_independent_set_fast(bits, edges, deg).sum())


def set_of(indices, n):
    bits = np.zeros(n, dtype=bool)
    bits[indices] = True
    return bits


def neighbour_lists(edges, n):
    both = np.concatenate([edges, edges[:, ::-1]])
    both = both[np.argsort(both[:, 0], kind="stable")]
    return np.split(both[:, 1], np.cumsum(np.bincount(both[:, 0], minlength=n))[:-1])


def peel(neighbours, deg, share, rng):
    n = len(deg)
    sel = np.ones(n, dtype=bool)
    cnt = deg.astype(float)
    scale = float(deg.max() + 1)
    while True:
        top = cnt.max()
        if top <= 0:
            return sel
        if share == 0:
            key = np.where(cnt > 0, cnt * scale + deg, -1.0)
            v = int(np.flatnonzero(key == key.max())[0])
        else:
            v = int(rng.choice(np.flatnonzero(cnt >= max((1.0 - share) * top, 1.0))))
        sel[v] = False
        cnt[neighbours[v]] -= 1
        cnt[v] = -n


def one_run(run, screen_row):
    t0 = time.perf_counter()
    _, edges, deg = instance_graph(screen_row)
    n = len(deg)
    if list(deg) != list(run["degree"]):
        raise ValueError("run/instance mismatch")
    rng = np.random.default_rng(zlib.crc32(f"{run['name']}|{run['seed']}".encode()))

    sizes = np.asarray(run["traj_raw"], dtype=int)
    stored = np.asarray(run["traj_feasible"], dtype=float)
    partial = (sizes > 0) & (sizes < n)
    sizes, stored = sizes[partial], stored[partial]

    streams = {"uniform": [], "degree": []}
    streams.update({f"jitter_{j}": [] for j in JITTERS})
    for s in sizes:
        s = int(s)
        streams["uniform"].append(fixed_size(set_of(rng.choice(n, s, replace=False), n), edges, deg))
        order = np.lexsort((rng.random(n), deg))
        streams["degree"].append(fixed_size(set_of(order[:s], n), edges, deg))
        place = np.empty(n)
        place[order] = np.arange(n)
        for j in JITTERS:
            moved = np.argsort(place + rng.uniform(-j * n, j * n, n))
            streams[f"jitter_{j}"].append(fixed_size(set_of(moved[:s], n), edges, deg))

    selected = np.asarray(run["best_p"]) >= 0.5
    own = fixed_size(selected, edges, deg)
    by_degree = np.lexsort((np.arange(n), deg))
    shuffles = {}
    for b in CLASS_COUNTS:
        values = []
        for _ in range(N_SHUFFLES):
            bits = selected.copy()
            for cls in np.array_split(by_degree, b):
                bits[cls] = selected[rng.permutation(cls)]
            values.append(fixed_size(bits, edges, deg))
        values = np.array(values)
        shuffles[str(b)] = {"mean": float(values.mean()), "best": int(values.max()),
                            "share_at_least_own": float(np.mean(values >= own))}

    return {"type": "run", "name": run["name"], "seed": run["seed"], "n": n, "mean_degree": run["mean_degree"],
            "reference": float(run["reference"]), "stream_length": int(len(sizes)),
            "stored_epochs_all_or_none": int((~partial).sum()),
            "net_best_value": float(run["best_value"]), "net_best_stored_epochs": float(stored.max()),
            "net_final_value": float(run["final_value_nodrop"]),
            "own_fix_up": own, "own_matches_stored": bool(own == int(round(run["best_value"]))),
            "streams": {k: {"best": int(max(v)), "mean": float(np.mean(v))} for k, v in streams.items()},
            "shuffles": shuffles, "n_shuffles": N_SHUFFLES, "seconds": time.perf_counter() - t0}


def one_graph(screen_row):
    t0 = time.perf_counter()
    _, edges, deg = instance_graph(screen_row)
    n = len(deg)
    neighbours = neighbour_lists(edges, n)
    rng = np.random.default_rng(zlib.crc32(f"peel|{screen_row['name']}".encode()))
    all_ones = fixed_size(np.ones(n, dtype=bool), edges, deg)
    streams = {}
    for share in PEEL_SHARES:
        values = np.array([int(peel(neighbours, deg, share, rng).sum()) for _ in range(RESTARTS)])
        streams[str(share)] = {"best": int(values.max()),
                               **{f"best_{k}": float(np.mean([values[i:i + k].max()
                                                              for i in range(0, len(values) - k + 1, k)]))
                                  for k in PEEL_PREFIXES}}
    return {"type": "peel", "name": screen_row["name"], "seed": None, "n": n,
            "mean_degree": screen_row["mean_degree"], "reference": float(screen_row["reference"]),
            "all_ones": all_ones,
            "peel_0_matches_fix_up": bool(int(peel(neighbours, deg, 0, rng).sum()) == all_ones),
            "peel": streams, "restarts": RESTARTS, "seconds": time.perf_counter() - t0}


def work(job):
    kind, *args = job
    return one_graph(*args) if kind == "peel" else one_run(*args)


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "fixup_controls.jsonl"))
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    screen = {r["name"]: r for name in SCREENS for r in read_jsonl(RESULTS / name)}
    names = {l.strip() for l in open(NAMES) if l.strip() and not l.startswith("#")}
    done = {(r["type"], r["name"], r["seed"]) for r in read_jsonl(args.out)} if os.path.exists(args.out) else set()
    runs = [r for r in read_jsonl(RUNS)
            if r["rung"] == "problem" and r["name"] in names and ("run", r["name"], r["seed"]) not in done]
    runs.sort(key=lambda r: (-r["m_problem"], r["seed"]))
    jobs = [("run", r, screen[r["name"]]) for r in runs]
    jobs += [("peel", screen[name]) for name in sorted(names) if ("peel", name, None) not in done]
    print(f"{len(runs)} stored runs and {len(jobs) - len(runs)} graphs to do, {args.procs} workers", flush=True)

    t0 = time.time()
    with open(args.out, "a") as fh:
        if args.procs > 1:
            import multiprocessing as mp
            with mp.Pool(args.procs) as pool:
                _write(pool.imap_unordered(work, jobs, chunksize=1), fh, len(jobs), t0)
        else:
            _write(map(work, jobs), fh, len(jobs), t0)


def _write(results, fh, total, t0):
    for i, rec in enumerate(results):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if rec["type"] == "peel":
            print(f"  [{i + 1}/{total}] {rec['name']} restarts: fix-up once {rec['all_ones']}, "
                  + ", ".join(f"share {k} best {v['best']}" for k, v in rec["peel"].items())
                  + f" ({rec['seconds']:.0f}s, {time.time() - t0:.0f}s in all)", flush=True)
        else:
            print(f"  [{i + 1}/{total}] {rec['name']} s{rec['seed']}: network {rec['net_best_stored_epochs']:.0f}, "
                  f"uniform {rec['streams']['uniform']['best']}, degree {rec['streams']['degree']['best']} "
                  f"({rec['seconds']:.0f}s, {time.time() - t0:.0f}s in all)", flush=True)


if __name__ == "__main__":
    main()
