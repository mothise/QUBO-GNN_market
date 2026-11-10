"""PI-GNN and CRA-PI-GNN on the benchmark of their papers: independent sets on random regular graphs."""
import argparse
import json
import os

import networkx as nx
import numpy as np
import torch

from sparse_tier.gnn_mis import model_defaults, train_one
from sparse_tier.mis_baselines import greedy_min_degree, ils_arw
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "sparse_tier" / "results"
N = 1000
GRAPH_SEEDS = (0, 1, 2, 3, 4)
ILS_SECONDS = 20.0
CELLS = ([(d, m, c) for d in (10, 20, 30) for m in ("pignn", "cra") for c in ("sage", "gcn")]
         + [(d, "pignn", "gcn") for d in (3, 5)])


def run(cells, out, device):
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["model"], r["conv"], r["degree"], r["graph_seed"]))

    yardstick = {}
    with open(out, "a") as fout:
        for gs in GRAPH_SEEDS:
            for degree, model, conv in cells:
                if (model, conv, degree, gs) in done:
                    continue
                G = nx.random_regular_graph(degree, N, seed=gs)
                nx.set_node_attributes(G, 1.0, "w")
                if (degree, gs) not in yardstick:
                    ils = ils_arw(G, time_budget=ILS_SECONDS, rng=np.random.default_rng(gs), stop_at_target=False)
                    yardstick[(degree, gs)] = (len(greedy_min_degree(G)), ils["best_value"])
                greedy, ils_best = yardstick[(degree, gs)]
                d = model_defaults(model)
                res = train_one(G, G, np.ones(N), seed=gs, epochs=d["epochs"], lr=d["lr"], p_min=d["p_min"],
                                p_max=d["p_max"], patience=0, dim_embedding=10, hidden_dim=50, dropout=d["dropout"],
                                device=device, log_every=0, model=model, conv=conv, traj_stride=1000)
                row = dict(model=model, conv=conv, degree=degree, graph_seed=gs, n=N,
                           final_value=res["final_value"], final_raw=res["final_raw"], final_viol=res["final_viol"],
                           best_value=res["best_value"], best_epoch=res["best_epoch"], epochs_run=res["epochs_run"],
                           stop_reason=res["stop_reason"], train_seconds=res["train_seconds"],
                           phi_final=res.get("phi_final"), greedy=greedy, ils_best=ils_best,
                           model_settings=res["model_settings"])
                fout.write(json.dumps(row) + "\n"); fout.flush()
                print(f"{model:5s} {conv:4s} d={degree:2d} graph {gs}: final {res['final_value']:.0f} "
                      f"(raw {res['final_raw']}, viol {res['final_viol']}), ILS {ils_best:.0f}; "
                      f"epochs {res['epochs_run']}, {res['train_seconds']:.0f}s, stop {res['stop_reason']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(RESULTS / "family_fidelity.jsonl"))
    args = ap.parse_args()
    run(CELLS, args.out, args.device)


if __name__ == "__main__":
    main()
