"""The training run of the dense loss without a network"""
import argparse
import json

import numpy as np

from dense_tier.experiments.signal_anatomy import load_instances
from dense_tier.fast_eval import FastEval, tol_of
from shared.metrics import auc_score
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"
INITS = ("half", "kn")


def free_logits(ev, q, init, seed=0, epochs=3000, lr=0.014):
    rng = np.random.default_rng(seed)
    n, K = ev.n, ev.K
    sigma_ii = ev.Bn2 + ev.D
    lam_final = 100.0 * (q * np.median(sigma_ii) / K**2 + (1 - q) * np.median(np.abs(ev.mu)) / K)
    lam0 = 0.01 * lam_final
    z = rng.normal(0, 0.1, n) + (0.0 if init == "half" else np.log((K / n) / (1 - K / n)))
    m, v = np.zeros(n), np.zeros(n)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for ep in range(epochs):
        lam = lam0 + (lam_final - lam0) * ep / (epochs - 1)
        p = 1 / (1 + np.exp(-z))
        grad_p = 2 * ev.a * (ev.B @ (ev.B.T @ p) + ev.D * p) - ev.b * ev.mu + 2 * lam * (p.sum() - K)
        g = grad_p * p * (1 - p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        z -= lr * (m / (1 - b1 ** (ep + 1))) / (np.sqrt(v / (1 - b2 ** (ep + 1))) + eps)
    return 1 / (1 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certify_jsonl", default=RESULTS / "certify_highs.jsonl")
    ap.add_argument("--out", default=RESULTS / "free_logits.jsonl")
    args = ap.parse_args()

    with open(args.out, "w") as fh:
        for c, fm in load_instances(args.certify_jsonl):
            K, q = int(c["K"]), float(c["q"])
            ev = FastEval(fm, K, q)
            x_star = np.zeros(fm.n)
            x_star[np.asarray(c["support"], dtype=int)] = 1.0
            f_star = float(ev.f(x_star))
            if abs(f_star - c["obj"]) > tol_of(f_star):
                raise SystemExit(f"certificate mismatch: {f_star} vs {c['obj']}")
            tol = tol_of(f_star)
            for init in INITS:
                p = free_logits(ev, q, init)
                x = np.zeros(fm.n)
                x[np.argsort(-p, kind="stable")[:K]] = 1.0
                _, f_sw, _ = ev.one_swap(x)
                rec = {"market": c["market"], "window": int(c["window"]), "K": K, "q": q, "n": int(fm.n),
                       "init": init, "auc": auc_score(p, x_star > 0.5), "sum_p": float(p.sum()),
                       "hit": bool(ev.f(x) <= f_star + tol), "hit_swap": bool(f_sw <= f_star + tol)}
                fh.write(json.dumps(rec) + "\n")
                print(f"{c['market']}_w{c['window']}_K{K} n={fm.n} init={init}: auc={rec['auc']:.3f} "
                      f"hit={int(rec['hit'])} hit_swap={int(rec['hit_swap'])}", flush=True)


if __name__ == "__main__":
    main()
