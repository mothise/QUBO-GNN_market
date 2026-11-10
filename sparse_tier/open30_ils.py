"""Iterated local search on the 30 open market graphs, given each network's median training time."""
import json
import os

import numpy as np

from sparse_tier.market_graph import correlation, load_returns_bruni, threshold_graph
from sparse_tier.mis_baselines import ils_arw
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "sparse_tier" / "results"
SCREENS = ["screen_SP500.jsonl", "screen_NASDAQComp.jsonl", "screen_cpsat.jsonl"]
SOLVERS = {"QRF-GNN 6,000": "gnn_open30_qrf6000.jsonl", "CRA with GraphSAGE": "gnn_open30_cra_sage.jsonl"}
OUT = RESULTS / "open30_ils_matched.jsonl"
SEEDS = (0, 1)


def main():
    screen = {}
    for name in SCREENS:
        for line in open(RESULTS / name):
            r = json.loads(line)
            screen[r["name"]] = r
    names = [l.strip() for l in open(RESULTS / "instances_open30.txt") if l.strip() and not l.startswith("#")]
    seconds = {s: {} for s in SOLVERS}
    for s, name in SOLVERS.items():
        for line in open(RESULTS / name):
            r = json.loads(line)
            seconds[s].setdefault(r["name"], []).append(r["train_seconds"])

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["name"], r["seed"]))

    cache = {}
    with open(OUT, "a") as fh:
        for name in names:
            budgets = sorted((float(np.median(seconds[s][name])), s) for s in SOLVERS if name in seconds[s])
            if not budgets:
                continue
            r = screen[name]
            if r["source"] not in cache:
                cache[r["source"]] = load_returns_bruni(r["source"])
            R, ws = cache[r["source"]]
            G = threshold_graph(correlation(R[ws[r["window"]][0]]), r["tau"])
            if G.number_of_edges() != r["m"]:
                raise SystemExit("screen/instance mismatch")
            for seed in SEEDS:
                if (name, seed) in done:
                    continue
                rng = np.random.default_rng([20261113, seed])
                rec = {"name": name, "seed": seed, "at": {}}
                start, spent = None, 0.0
                for t, s in budgets:
                    res = ils_arw(G, time_budget=max(0.0, t - spent), rng=rng, start=start, stop_at_target=False)
                    spent = t
                    start = set(res["best_set"])
                    rec["at"][s] = {"budget": t, "value": float(res["best_value"])}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(name, seed, {s: v["value"] for s, v in rec["at"].items()}, flush=True)


if __name__ == "__main__":
    main()
