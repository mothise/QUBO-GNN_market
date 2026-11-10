import argparse
import math

import numpy as np

from dense_tier.fast_eval import FastEval
from dense_tier.gnn_factored import run_gnn_factored
from dense_tier.graphs import GRAPH_BUILDERS
from dense_tier.moments import estimate_moments
from dense_tier.qubo import PortfolioQubo
from shared.bruni import load_market, rolling_windows
from shared.gnn import nx_to_pyg_edge_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="DowJones")
    ap.add_argument("--window_idx", type=int, default=0)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--graphs", nargs="+", default=["complete", "knn"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    R = load_market(args.market)
    tr, _ = rolling_windows(R.shape[0])[args.window_idx]
    fm = estimate_moments(R[tr])
    print(f"{args.market} window {args.window_idx}: T={len(tr)} n={fm.n} "
          f"LW shrinkage={fm.shrinkage:.4f}")

    qubo = PortfolioQubo(fm, K=args.K, q=args.q, device=args.device)
    print(f"lam_final={qubo.lam_final:.3e}  factored-vs-dense rel err={qubo.check_exactness():.3e}")

    ev = FastEval(fm, args.K, args.q)
    rng = np.random.default_rng(0)
    x_g = ev.greedy()
    _, f_gs, _ = ev.one_swap(x_g)
    rep_vals = [ev.one_swap(ev.random_start(rng))[1] for _ in range(20)]
    rand_vals = [ev.f(ev.random_start(rng)) for _ in range(200)]
    n_comb = math.comb(fm.n, args.K)
    f_star = ev.exact_enum()[1] if n_comb <= 5_000_000 else np.nan

    print(f"\nbaselines (lower is better; exact over C({fm.n},{args.K})={n_comb:,})")
    print(f"  exact optimum      : {f_star:+.6e}")
    print(f"  greedy             : {ev.f(x_g):+.6e}")
    print(f"  greedy + 1-swap    : {f_gs:+.6e}")
    print(f"  repair-only (20x)  : best {min(rep_vals):+.6e}  mean {np.mean(rep_vals):+.6e}")
    print(f"  random-K (200x)    : best {min(rand_vals):+.6e}  mean {np.mean(rand_vals):+.6e}")

    results = {}
    for gname in args.graphs:
        G = GRAPH_BUILDERS[gname](fm)
        ei = nx_to_pyg_edge_index(G)
        print(f"\n{gname}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        best_raw, best_pol, cards = [], [], []
        for seed in args.seeds:
            x, obj, ep, diag = run_gnn_factored(
                qubo.train_loss, qubo.feasible_objective, args.K, ei, G,
                number_epochs=args.epochs, seed=seed, log_every=0, device=args.device)
            _, f_pol, _ = ev.one_swap(x)
            best_raw.append(obj)
            best_pol.append(f_pol)
            cards.append(diag["thr_card"][-1])
            print(f"  seed {seed}: raw topK={obj:+.6e} (+swap {f_pol:+.6e}) "
                  f"@ep {ep}/{diag['epochs_run']}  sum(p)_end={diag['sum_p'][-1]:.2f}  "
                  f"|thr@0.5|={diag['thr_card'][-1]}  t={diag['seconds']:.1f}s")
        results[gname] = (min(best_raw), min(best_pol), cards)

    print("\ngap to exact optimum (best of seeds)")
    for gname, (raw, pol, cards) in results.items():
        gr = (raw - f_star) / abs(f_star) if np.isfinite(f_star) else np.nan
        gp = (pol - f_star) / abs(f_star) if np.isfinite(f_star) else np.nan
        print(f"  {gname:10s} raw {raw:+.6e} (gap {gr:+.2%}) | +swap {pol:+.6e} (gap {gp:+.2%}) | thr-card {cards}")
    print(f"  {'greedy':10s} {ev.f(x_g):+.6e} (gap {(ev.f(x_g) - f_star) / abs(f_star):+.2%})")


if __name__ == "__main__":
    main()
