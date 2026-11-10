import argparse
import json

import numpy as np
from scipy.optimize import lsq_linear

from dense_tier.experiments.signal_anatomy import load_instances
from dense_tier.fast_eval import FastEval, tol_of
from shared.metrics import auc_score
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"


def relaxed_minimiser(ev, lam, f_star):
    n, d = ev.n, ev.D[0]
    A = np.vstack([np.sqrt(ev.a) * ev.B.T, np.sqrt(ev.a * d) * np.eye(n), np.sqrt(lam) * np.ones((1, n))])
    y = np.concatenate([np.zeros(ev.k), np.sqrt(ev.a * d) * ev.b * ev.mu / (2 * ev.a * d), [np.sqrt(lam) * ev.K]])
    unit = np.sqrt(abs(f_star))
    p = lsq_linear(A / unit, y / unit, bounds=(0, 1), method="bvls", tol=1e-13, max_iter=20000).x
    p[p < 1e-9] = 0.0
    p[p > 1 - 1e-9] = 1.0
    return p


def analyse(c, fm):
    K, q = int(c["K"]), float(c["q"])
    ev = FastEval(fm, K, q)
    x_star = np.zeros(fm.n)
    x_star[np.asarray(c["support"], dtype=int)] = 1.0
    f_star = float(ev.f(x_star))
    if abs(f_star - c["obj"]) > tol_of(f_star):
        raise SystemExit(f"certificate mismatch: {f_star} vs {c['obj']}")
    s_star = x_star > 0.5
    tol = tol_of(f_star)

    sigma_ii = ev.Bn2 + ev.D
    lam = 100 * (q * np.median(sigma_ii) / K**2 + (1 - q) * np.median(np.abs(fm.mu)) / K)
    p = relaxed_minimiser(ev, lam, f_star)
    v = ev.B.T @ p
    loss = ev.a * (v @ v + (ev.D * p * p).sum()) - ev.b * (ev.mu @ p) + lam * (p.sum() - K) ** 2

    def decode(score):
        x = np.zeros(fm.n)
        x[np.argsort(-score)[:K]] = 1.0
        f_x = ev.f(x)
        _, f_sw, _ = ev.one_swap(x)
        return {"overlap": float(x[s_star].mean()), "hit": bool(f_x <= f_star + tol),
                "gap": (f_x - f_star) / abs(f_star), "hit_swap": bool(f_sw <= f_star + tol)}

    sharpe = fm.mu / np.sqrt(sigma_ii)
    relaxed, by_sharpe, by_mu = decode(p), decode(sharpe), decode(fm.mu)
    return {
        "market": c["market"], "window": int(c["window"]), "K": K, "q": q, "n": int(fm.n),
        "f_star": f_star, "lam": float(lam), "shrink": float(fm.shrinkage),
        "loss_relax": float(loss), "relax_minus_fstar_rel": float((loss - f_star) / abs(f_star)),
        "sum_p": float(p.sum()), "n_pos": int((p > 0.01).sum()),
        "n_frac": int(((p > 0.01) & (p < 0.99)).sum()),
        "auc_relax": auc_score(p, s_star), "overlap_relax": relaxed["overlap"],
        "hit_relax": relaxed["hit"], "hit_relax_swap": relaxed["hit_swap"], "gap_relax": relaxed["gap"],
        "auc_sharpe": auc_score(sharpe, s_star), "overlap_sharpe": by_sharpe["overlap"],
        "hit_sharpe": by_sharpe["hit"], "hit_sharpe_swap": by_sharpe["hit_swap"],
        "auc_mu": auc_score(fm.mu, s_star), "overlap_mu": by_mu["overlap"], "hit_mu_swap": by_mu["hit_swap"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certify_jsonl", default=RESULTS / "certify_highs.jsonl")
    ap.add_argument("--out", default=RESULTS / "relaxed_optimum.jsonl")
    args = ap.parse_args()

    with open(args.out, "w") as fh:
        for c, fm in load_instances(args.certify_jsonl):
            rec = analyse(c, fm)
            fh.write(json.dumps(rec) + "\n")
            print(f"{rec['market']}_w{rec['window']}_K{rec['K']} n={rec['n']}: auc={rec['auc_relax']:.3f} "
                  f"hit={int(rec['hit_relax'])} hit_swap={int(rec['hit_relax_swap'])} "
                  f"sharpe auc={rec['auc_sharpe']:.3f} hit_swap={int(rec['hit_sharpe_swap'])}", flush=True)


if __name__ == "__main__":
    main()