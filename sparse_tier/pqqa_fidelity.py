"""PQQA on the kinds of graphs of its paper: Erdos-Renyi, 3-SAT and random regular graphs."""
import argparse
import json
import os

import networkx as nx
import numpy as np
import torch

from shared import pqqa
from shared.paths import REPO_ROOT
from sparse_tier.mis_baselines import greedy_min_degree, ils_arw

RESULTS = REPO_ROOT / "sparse_tier" / "results"
LRS = (1.0, 0.1, 0.01)
ILS_SECONDS = 20.0


def er_graph(index):
    rng = np.random.default_rng(100 + index)
    n = int(rng.integers(700, 801))
    return nx.fast_gnp_random_graph(n, 0.15, seed=1000 + index), None


def sat_graph(index):
    rng = np.random.default_rng(200 + index)
    variables, clauses = 100, int(rng.integers(403, 450))
    hidden = rng.integers(0, 2, variables).astype(bool)
    literals = []
    while len(literals) < clauses:
        chosen = rng.choice(variables, 3, replace=False)
        signs = rng.integers(0, 2, 3).astype(bool)
        if (hidden[chosen] == signs).any():
            literals.append(list(zip(chosen.tolist(), signs.tolist())))
    G = nx.Graph()
    G.add_nodes_from(range(3 * clauses))
    by_literal = {}
    for c, clause in enumerate(literals):
        G.add_edges_from([(3 * c, 3 * c + 1), (3 * c, 3 * c + 2), (3 * c + 1, 3 * c + 2)])
        for k, literal in enumerate(clause):
            by_literal.setdefault(literal, []).append(3 * c + k)
    for (var, sign), nodes in by_literal.items():
        if sign:
            G.add_edges_from((u, v) for u in nodes for v in by_literal.get((var, False), []))
    return G, clauses


def regular(degree):
    def make(index):
        return nx.random_regular_graph(degree, 1000, seed=index), None
    return make


FAMILIES = {"er": er_graph, "sat": sat_graph, "regular10": regular(10), "regular20": regular(20),
            "regular30": regular(30)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(RESULTS / "pqqa_fidelity.jsonl"))
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        done = {(r["family"], r["graph"], r["lr"], r["runs"]) for r in map(json.loads, open(args.out))}
    plan = []
    for family in FAMILIES:
        is_regular = family.startswith("regular")
        for lr in ((pqqa.SETTINGS["lr"],) if is_regular else LRS):
            for runs in ((1, 100) if is_regular else (100,)):
                plan += [(family, g, lr, runs) for g in range(5 if is_regular else 16)]
    plan = [p for p in plan if p not in done]
    print(f"{len(plan)} runs to do, device {args.device}", flush=True)

    yardstick = {}
    with open(args.out, "a") as fout:
        for family, g, lr, runs in plan:
            G, optimum = FAMILIES[family](g)
            n = G.number_of_nodes()
            edges = np.array(G.edges)
            if (family, g) not in yardstick:
                nx.set_node_attributes(G, 1.0, "w")
                ils = ils_arw(G, time_budget=ILS_SECONDS, rng=np.random.default_rng(g), stop_at_target=False)
                yardstick[(family, g)] = (int(ils["best_value"]), len(greedy_min_degree(G)))
            ils_best, greedy = yardstick[(family, g)]
            P, settings, seconds, _ = pqqa.anneal(pqqa.IndependentSet(edges, n), runs, pqqa.STEPS, g,
                                                  args.device, lr=lr)
            sets = P > 0.5
            sizes = sets.sum(axis=1)
            feasible = (sets[:, edges[:, 0]] & sets[:, edges[:, 1]]).sum(axis=1) == 0
            rec = {"family": family, "graph": g, "n": n, "m": int(len(edges)), "lr": lr, "runs": runs,
                   "best_feasible": int(sizes[feasible].max()) if feasible.any() else 0,
                   "runs_feasible": float(feasible.mean()), "optimum": optimum, "ils": ils_best, "greedy": greedy,
                   "seconds": seconds, "model_settings": settings}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"  {family} graph {g} lr {lr:g} runs {runs}: best feasible {rec['best_feasible']}, ILS {ils_best}"
                  + (f", optimum {optimum}" if optimum else "") + f" ({seconds:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
