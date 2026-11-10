"""PQQA on the certified dense instances."""
import argparse
import json
import os
import time

import numpy as np

from dense_tier.experiments.signal_anatomy import load_instances
from dense_tier.fast_eval import FastEval, tol_of
from shared import pqqa
from shared.metrics import auc_score
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"
SEEDS = (0, 1, 2, 3)
RUNS = 100


def solve(c, fm, seed, form, lr, device):
    K, q = int(c["K"]), float(c["q"])
    ev = FastEval(fm, K, q)
    x_star = np.zeros(fm.n)
    x_star[np.asarray(c["support"], dtype=int)] = 1.0
    f_star = float(ev.f(x_star))
    tol = tol_of(f_star)

    problem = pqqa.Portfolio(fm, K, q, form)
    P, settings, seconds, _ = pqqa.anneal(problem, RUNS, pqqa.STEPS, seed, device, lr=lr)
    X = np.zeros(P.shape)
    np.put_along_axis(X, np.argsort(-P, axis=1, kind="stable")[:, :K], 1.0, axis=1)
    f_runs = ev.f_batch(X)
    best = int(np.argmin(f_runs))
    _, f_swap, _ = ev.one_swap(X[best])
    return {
        "type": "run", "market": c["market"], "window": int(c["window"]), "K": K, "q": q, "n": int(fm.n),
        "seed": seed, "model": "pqqa", "loss_form": form, "runs": RUNS,
        "steps": settings["steps"], "model_settings": settings, "device": device,
        "loss_unit": float(problem.unit), "lam": float(problem.lam),
        "f_star": f_star, "f_final": float(f_runs[best]), "hit_raw": bool(f_runs[best] <= f_star + tol),
        "gap_raw": float((f_runs[best] - f_star) / abs(f_star)),
        "f_swap": float(f_swap), "hit_swap": bool(f_swap <= f_star + tol),
        "runs_hit": float(np.mean(f_runs <= f_star + tol)), "f_runs_mean": float(f_runs.mean()),
        "final_p": P[best].tolist(), "auc_final": auc_score(P[best], x_star > 0.5),
        "mean_p_auc": auc_score(P.mean(axis=0), x_star > 0.5),
        "run_count": (P > 0.5).sum(axis=1).tolist(),
        "seconds": seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certify_jsonl", default=RESULTS / "certify.jsonl")
    ap.add_argument("--runs_out", default=RESULTS / "signal_runs_pqqa.jsonl")
    ap.add_argument("--loss_form", default="convex", choices=["convex", "qubo"])
    ap.add_argument("--lr", type=float, default=pqqa.SETTINGS["lr"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    instances = load_instances(args.certify_jsonl)

    def key_of(r):
        return (r["market"], int(r["window"]), int(r["K"]), float(r["q"]), r["seed"], r["loss_form"])

    done = set()
    if os.path.exists(args.runs_out):
        for line in open(args.runs_out):
            r = json.loads(line)
            done.add(key_of(r))
            if r["model_settings"]["lr"] != args.lr:
                raise SystemExit(f"{args.runs_out} holds runs of another learning rate")
    print(f"{len(instances)} instances x {len(SEEDS)} seeds, loss form {args.loss_form}, lr {args.lr:g}, "
          f"device {args.device}", flush=True)

    t_start = time.time()
    with open(args.runs_out, "a") as fh:
        for seed in SEEDS:
            for c, fm in instances:
                if key_of(dict(c, seed=seed, loss_form=args.loss_form)) in done:
                    continue
                rec = solve(c, fm, seed, args.loss_form, args.lr, args.device)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"  {rec['market']}_w{rec['window']}_K{rec['K']} q={rec['q']:g} s{seed}: "
                      f"auc={rec['auc_final']:.3f} hit_raw={int(rec['hit_raw'])} hit_swap={int(rec['hit_swap'])} "
                      f"runs_hit={rec['runs_hit']:.2f} ({rec['seconds']:.1f}s, {time.time() - t_start:.0f}s in all)",
                      flush=True)


if __name__ == "__main__":
    main()
