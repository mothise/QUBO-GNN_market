"""PQQA on market-graph independent sets."""
import argparse
import json
import os
import time

import numpy as np
import torch

from shared import pqqa
from shared.paths import REPO_ROOT
from sparse_tier.gnn_mis import fix_independent_set_fast, load_screen
from sparse_tier.market_graph import correlation, load_returns_bruni, threshold_graph
from sparse_tier.mis_baselines import local_search_arw

RESULTS = REPO_ROOT / "sparse_tier" / "results"
SCREENS = [str(RESULTS / "screen_SP500.jsonl"), str(RESULTS / "screen_NASDAQComp.jsonl")]
SEEDS = (0, 1, 2, 3, 4)
TRACK_EVERY = 10


def solve(row, G, seed, runs, lr, device):
    edges = np.array(G.edges(), dtype=np.int64).reshape(-1, 2)
    deg = np.array([d for _, d in sorted(G.degree())], dtype=np.int64)
    n = len(deg)
    problem = pqqa.IndependentSet(edges, n)
    P, settings, seconds, (best_value, best_step) = pqqa.anneal(problem, runs, pqqa.STEPS, seed, device,
                                                               lr=lr, track_every=TRACK_EVERY)
    t0 = time.perf_counter()
    sets = P > 0.5
    raw = sets.sum(axis=1)
    viol = (sets[:, edges[:, 0]] & sets[:, edges[:, 1]]).sum(axis=1)
    fixed = [fix_independent_set_fast(s, edges, deg) if v else s for s, v in zip(sets, viol)]
    value = np.array([int(f.sum()) for f in fixed])
    best = int(np.argmax(value))
    members = [int(i) for i in np.flatnonzero(fixed[best])]
    return {
        "name": row["name"], "source": row["source"], "window": row["window"], "tau": row["tau"], "seed": seed,
        "n": n, "m_problem": int(len(edges)), "mean_degree": row["mean_degree"],
        "reference": float(row["reference"]), "reference_is_certified": bool(row["reference_is_certified"]),
        "model": "pqqa", "runs": int(runs), "steps": settings["steps"],
        "model_settings": settings, "device": device,
        "final_value": float(value[best]), "final_raw": int(raw[best]), "final_viol": int(viol[best]),
        "final_plus_ls": float(len(local_search_arw(G, set(members)))), "final_set": members,
        "final_value_mean": float(value.mean()), "run_value": value.tolist(),
        "best_value": float(best_value), "best_step": best_step,
        "runs_feasible": float(np.mean(viol == 0)),
        "train_seconds": seconds, "decode_seconds": time.perf_counter() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", nargs="+", default=SCREENS)
    ap.add_argument("--names_file", default=str(RESULTS / "instances_full42.txt"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=pqqa.SETTINGS["lr"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(RESULTS / "pqqa_runs.jsonl"))
    args = ap.parse_args()

    names = [l.strip() for l in open(args.names_file) if l.strip() and not l.startswith("#")]
    rows = load_screen(args.screen, only_certified=not args.all, names=names)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["name"], r["seed"], r["steps"], r["runs"]))
            if r["model_settings"]["lr"] != args.lr:
                raise SystemExit(f"{args.out} holds runs of another learning rate")
    print(f"{len(rows)} graphs x {len(SEEDS)} seeds, {pqqa.STEPS} steps, {args.runs} runs, "
          f"lr {args.lr:g}, device {args.device}", flush=True)

    returns, graphs = {}, {}
    with open(args.out, "a") as fout:
        for seed in SEEDS:
            for row in rows:
                if (row["name"], seed, pqqa.STEPS, args.runs) in done:
                    continue
                if row["name"] not in graphs:
                    if row["source"] not in returns:
                        returns[row["source"]] = load_returns_bruni(row["source"])
                    R, ws = returns[row["source"]]
                    graphs[row["name"]] = threshold_graph(correlation(R[ws[row["window"]][0]]), row["tau"])
                    if graphs[row["name"]].number_of_edges() != row["m"]:
                        raise SystemExit("screen/instance mismatch")
                rec = solve(row, graphs[row["name"]], seed, args.runs, args.lr, args.device)
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                print(f"  {rec['name']} s{seed}: final={rec['final_value']:.0f} (raw {rec['final_raw']}, "
                      f"viol {rec['final_viol']}) mean={rec['final_value_mean']:.1f} +ls={rec['final_plus_ls']:.0f} "
                      f"best={rec['best_value']:.0f}@{rec['best_step']} ref={rec['reference']:.0f} "
                      f"train={rec['train_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
