"""PQQA on the weighted market graphs: the weights in the linear term, a single penalty or a penalty per edge."""
import argparse
import json
import os
import time

import numpy as np
import torch

from shared import pqqa
from shared.paths import REPO_ROOT
from sparse_tier.gnn_mis import fix_independent_set_fast, load_screen
from sparse_tier.market_graph import correlation, load_returns_bruni, threshold_graph, weighted_window
from sparse_tier.mis_baselines import local_search_arw, value

RESULTS = REPO_ROOT / "sparse_tier" / "results"
SCREEN = RESULTS / "screen_return_abs.jsonl"
NAMES = RESULTS / "instances_full42_return_abs.txt"
SEEDS = (0, 1, 2, 3, 4)
STEPS = pqqa.STEPS


class WeightedIndependentSet:
    def __init__(self, edges, weights, penalty, edge_penalty, scale):
        self.n = len(weights)
        i, j = edges[:, 0], edges[:, 1]
        if edge_penalty == "uniform":
            cost = np.full(len(edges), float(penalty))
        else:
            cost = float(penalty) * np.minimum(weights[i], weights[j])
        self.linear = scale * weights
        self.C = np.zeros((self.n, self.n))
        self.C[i, j] = scale * cost
        self.C[j, i] = scale * cost
        self._tensors = None

    def loss(self, P):
        if self._tensors is None:
            self._tensors = tuple(torch.as_tensor(v, dtype=P.dtype, device=P.device) for v in (self.linear, self.C))
        linear, C = self._tensors
        return -(P @ linear) + 0.5 * ((P @ C) * P).sum(dim=1)


def weighted_graph(row, returns):
    R, ws = returns
    Rw, w = weighted_window(R[ws[row["window"]][0]], row["weights"])
    G = threshold_graph(correlation(Rw), row["tau"], weights=w, edges=row["edges"])
    if (G.number_of_nodes(), G.number_of_edges()) != (row["n"], row["m"]):
        raise SystemExit("screen/instance mismatch")
    edges = np.array(G.edges(), dtype=np.int64).reshape(-1, 2)
    return G, edges, np.bincount(edges.ravel(), minlength=len(w)), np.asarray(w, dtype=np.float64)


def solve(row, G, edges, deg, weights, seed, runs, lr, penalty, edge_penalty, loss_scale, device, steps=STEPS):
    scale = 1.0 / float(np.median(weights)) if loss_scale == "median" else 1.0
    problem = WeightedIndependentSet(edges, weights, penalty, edge_penalty, scale)
    P, settings, seconds, _ = pqqa.anneal(problem, runs, steps, seed, device, lr=lr)
    settings.update(penalty=float(penalty), edge_penalty=edge_penalty, loss_scale=loss_scale)
    t0 = time.perf_counter()
    sets = P > 0.5
    raw = sets.sum(axis=1)
    viol = (sets[:, edges[:, 0]] & sets[:, edges[:, 1]]).sum(axis=1)
    fixed = [fix_independent_set_fast(s, edges, deg) if v else s for s, v in zip(sets, viol)]
    values = np.array([float(weights[f].sum()) for f in fixed])
    best = int(np.argmax(values))
    members = np.flatnonzero(fixed[best])
    plus_ls = value(G, local_search_arw(G, set(int(v) for v in members)))
    decode_seconds = time.perf_counter() - t0
    return {
        "name": row["name"], "source": row["source"], "window": row["window"], "tau": row["tau"], "rung": "problem",
        "seed": seed, "n": int(len(weights)), "m_problem": int(len(edges)), "mean_degree": row["mean_degree"],
        "reference": float(row["reference"]), "reference_is_certified": bool(row["reference_is_certified"]),
        "model": "pqqa", "runs": int(runs), "steps": int(steps), "model_settings": settings,
        "p_min": float(penalty), "p_max": float(penalty), "edge_penalty": edge_penalty,
        "loss_scale": loss_scale, "loss_scale_value": scale, "device": device,
        "final_value": float(values[best]), "final_raw": int(raw[best]), "final_viol": int(viol[best]),
        "final_size": int(len(members)), "final_plus_ls": float(plus_ls), "final_set": members.tolist(),
        "final_value_mean": float(values.mean()), "collapsed": bool(raw[best] == 0),
        "run_raw": raw.tolist(), "run_viol": viol.tolist(), "run_value": values.tolist(),
        "runs_feasible": float(np.mean(viol == 0)), "runs_distinct": int(len({s.tobytes() for s in sets})),
        "binary_share": float(np.mean((P < 0.01) | (P > 0.99))), "train_seconds": seconds,
        "decode_seconds": decode_seconds, "wall_seconds": seconds + decode_seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalty", type=float, default=pqqa.SETTINGS["penalty"])
    ap.add_argument("--edge_penalty", default="uniform", choices=["uniform", "min"])
    ap.add_argument("--loss_scale", default="none", choices=["none", "median"])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=pqqa.SETTINGS["lr"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(RESULTS / "pqqa_return_abs.jsonl"))
    args = ap.parse_args()

    names = [l.strip() for l in open(NAMES) if l.strip() and not l.startswith("#")]
    rows = load_screen([SCREEN], names=names)
    wanted = dict(lr=args.lr, penalty=args.penalty, edge_penalty=args.edge_penalty, loss_scale=args.loss_scale)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["name"], r["seed"], r["runs"]))
            if any(r["model_settings"][k] != v for k, v in wanted.items()):
                raise SystemExit(f"{args.out} holds runs of other settings")
    print(f"{len(rows)} graphs x {len(SEEDS)} seeds, {STEPS} steps, {args.runs} runs, penalty {args.penalty:g} "
          f"({args.edge_penalty}), loss scale {args.loss_scale}, lr {args.lr:g}, device {args.device}", flush=True)

    returns, graphs = {}, {}
    with open(args.out, "a") as fout:
        for seed in SEEDS:
            for row in rows:
                if (row["name"], seed, args.runs) in done:
                    continue
                if row["name"] not in graphs:
                    if row["source"] not in returns:
                        returns[row["source"]] = load_returns_bruni(row["source"])
                    graphs[row["name"]] = weighted_graph(row, returns[row["source"]])
                rec = solve(row, *graphs[row["name"]], seed, args.runs, args.lr, args.penalty, args.edge_penalty,
                            args.loss_scale, args.device)
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                print(f"  {rec['name']} s{seed}: final={rec['final_value']:.3f} (raw {rec['final_raw']}, "
                      f"viol {rec['final_viol']}) mean={rec['final_value_mean']:.3f} ref={rec['reference']:.3f} "
                      f"train={rec['train_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
