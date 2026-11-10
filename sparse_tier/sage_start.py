"""PI-GNN and CRA-PI-GNN with GraphSAGE layers on random regular graphs,
 with the layer weights drawn as in DGL."""
import argparse
import json
import os

import networkx as nx
import numpy as np
import torch

from sparse_tier import gnn_mis
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "sparse_tier" / "results"
N = 1000
GRAPH_SEEDS = (0, 1, 2)
CELLS = [(d, m) for m in ("cra", "pignn") for d in (10, 20, 30)]
library_build = gnn_mis.build_family


def dgl_start(net):
    gain = torch.nn.init.calculate_gain("relu")
    for layer in (net.conv1, net.conv2):
        for p in layer.parameters():
            if p.dim() == 2:
                torch.nn.init.xavier_uniform_(p, gain=gain)


def build_family(model, n_nodes, device, dtype, conv="gcn", lr=None, dropout=None):
    net, opt, s = library_build(model, n_nodes, device, dtype, conv=conv, lr=lr, dropout=dropout)
    dgl_start(net)
    s["start"] = "dgl"
    return net, opt, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(RESULTS / "family_fidelity_dgl_start.jsonl"))
    args = ap.parse_args()
    gnn_mis.build_family = build_family

    yardstick = {}
    for line in open(RESULTS / "family_fidelity.jsonl"):
        r = json.loads(line)
        yardstick[(r["degree"], r["graph_seed"])] = (r["greedy"], r["ils_best"])
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["model"], r["degree"], r["graph_seed"]))

    with open(args.out, "a") as fout:
        for degree, model in CELLS:
            for gs in GRAPH_SEEDS:
                if (model, degree, gs) in done:
                    continue
                G = nx.random_regular_graph(degree, N, seed=gs)
                nx.set_node_attributes(G, 1.0, "w")
                greedy, ils_best = yardstick[(degree, gs)]
                d = gnn_mis.model_defaults(model)
                res = gnn_mis.train_one(G, G, np.ones(N), seed=gs, epochs=d["epochs"], lr=d["lr"], p_min=d["p_min"],
                                        p_max=d["p_max"], patience=0, dim_embedding=10, hidden_dim=50,
                                        dropout=d["dropout"], device=args.device, log_every=0, model=model,
                                        conv="sage", traj_stride=1000)
                row = dict(start="dgl", model=model, conv="sage", degree=degree, graph_seed=gs, n=N,
                           final_value=res["final_value"], final_raw=res["final_raw"], final_viol=res["final_viol"],
                           best_value=res["best_value"], best_epoch=res["best_epoch"], epochs_run=res["epochs_run"],
                           stop_reason=res["stop_reason"], train_seconds=res["train_seconds"],
                           phi_final=res.get("phi_final"), greedy=greedy, ils_best=ils_best,
                           model_settings=res["model_settings"])
                fout.write(json.dumps(row) + "\n"); fout.flush()
                print(f"{model:5s} sage d={degree:2d} graph {gs}: final {res['final_value']:.0f} "
                      f"(raw {res['final_raw']}, viol {res['final_viol']}), ILS {ils_best:.0f}; "
                      f"epochs {res['epochs_run']}, {res['train_seconds']:.0f}s, stop {res['stop_reason']}", flush=True)


if __name__ == "__main__":
    main()
