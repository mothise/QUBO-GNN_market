"""
Hardness screen for market-graph independent sets.
"""
import argparse
import json
import os
import time

import numpy as np

from sparse_tier.market_graph import TAUS_DEFAULT, bruni_instances, er_control
from sparse_tier.mis_baselines import greedy_plus_ls, ils_arw, restart_hit_rate, value
from sparse_tier.mis_exact import certify
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "sparse_tier" / "results"

_STAT_KEYS = ("n", "m", "density", "mean_degree", "max_degree", "isolated_frac",
              "n_components", "largest_component", "kernel_size", "forced_in")


def screen_graph(G, *, restarts, time_limit, seed, engine,
                 ils_seconds=5.0, exact=None):
    cert = exact if exact is not None else certify(G, time_limit=time_limit, engine=engine)
    g, gl = greedy_plus_ls(G)
    vg, vgl = value(G, g), value(G, gl)
    certified = cert["status"] == "optimal"
    ref = cert["value"] if certified else max(cert["value"], vgl)

    rr = restart_hit_rate(G, ref, restarts, seed=seed)
    if rr["best"] > ref + 1e-9:
        ref = rr["best"]
        rr = restart_hit_rate(G, ref, restarts, seed=seed)

    ils = None
    if ils_seconds > 0:
        ils = ils_arw(G, time_budget=ils_seconds, rng=np.random.default_rng(seed + 7919),
                      reference=ref, stop_at_target=certified)
        if ils["best_value"] > ref + 1e-9:
            ref = ils["best_value"]
            rr = restart_hit_rate(G, ref, restarts, seed=seed)

    gap_gl = (ref - vgl) / max(abs(ref), 1e-12)
    out = dict(
        certified=cert["status"], exact_value=cert["value"], exact_bound=cert["bound"],
        exact_gap=cert["gap"], exact_seconds=cert["seconds"], exact_rows=cert.get("n_clique_rows", 0),
        exact_set=cert["set"], reference=ref, reference_is_certified=certified,
        greedy=vg, greedy_ls=vgl, greedy_ls_gap=gap_gl,
        rr_hit_rate=rr["hit_rate"], rr_best=rr["best"], rr_mean=rr["mean"],
        rr_ms=1000 * rr["seconds_per_restart"], rr_R=restarts,
    )
    if ils is not None:
        out.update(
            ils_best=ils["best_value"], ils_gap=(ref - ils["best_value"]) / max(abs(ref), 1e-12),
            ils_hit=bool(ils["best_value"] >= ref - 1e-9), ils_time_to_target=ils["time_to_target"],
            ils_iterations=ils["iterations"], ils_seconds=ils["seconds"], ils_budget=ils_seconds,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="*", default=["SP500", "NASDAQComp"])
    ap.add_argument("--taus", nargs="+", type=float, default=list(TAUS_DEFAULT))
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--weights", default="unit", choices=["unit", "return"])
    ap.add_argument("--edges", default="signed", choices=["signed", "abs"])
    ap.add_argument("--names_file", default=None)
    ap.add_argument("--restarts", type=int, default=50)
    ap.add_argument("--time_limit", type=float, default=300.0)
    ap.add_argument("--engine", default="highs", choices=["highs", "cpsat"])
    ap.add_argument("--no_control", action="store_true")
    ap.add_argument("--max_mean_degree", type=float, default=None)
    ap.add_argument("--ils_seconds", type=float, default=5.0)
    ap.add_argument("--reuse_exact_from", nargs="*", default=[])
    ap.add_argument("--only_certified", action="store_true")
    ap.add_argument("--only_uncertified", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "screen.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = None
    if args.names_file:
        with open(args.names_file) as f:
            names = {l.strip() for l in f if l.strip() and not l.startswith("#")}

    reuse = {}
    for p in args.reuse_exact_from:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                reuse[r["name"]] = dict(status=r["certified"], value=r["exact_value"], bound=r["exact_bound"],
                                        gap=r["exact_gap"], seconds=r["exact_seconds"],
                                        n_clique_rows=r.get("exact_rows", 0), set=r["exact_set"])

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                done.add(json.loads(line)["name"])

    def instances():
        for m in args.markets:
            yield from bruni_instances(m, args.taus, every=args.every, weights=args.weights, edges=args.edges)

    with open(args.out, "a") as fout:
        for inst in instances():
            if inst.name in done or (names is not None and inst.name not in names):
                continue
            if args.only_certified and reuse.get(inst.name, {}).get("status") != "optimal":
                continue
            if args.only_uncertified and reuse.get(inst.name, {}).get("status") == "optimal":
                continue
            st = inst.G.graph
            if args.max_mean_degree is not None and st["mean_degree"] > args.max_mean_degree:
                continue
            t0 = time.time()
            row = dict(name=inst.name, source=inst.source, window=inst.window, tau=inst.tau,
                       weights=args.weights, edges=args.edges, m_negative=st["m_negative"],
                       n_assets_window=inst.meta.get("n_assets_window"),
                       seed=args.seed, restarts=args.restarts,
                       time_limit=args.time_limit, engine=args.engine, **{k: st[k] for k in _STAT_KEYS})
            row.update(screen_graph(inst.G, restarts=args.restarts, time_limit=args.time_limit, seed=args.seed,
                                    engine=args.engine, ils_seconds=args.ils_seconds,
                                    exact=None if args.only_uncertified else reuse.get(inst.name)))
            if not args.no_control and inst.G.number_of_edges() > 0:
                H = er_control(inst.G, seed=args.seed + inst.window)
                c = screen_graph(H, restarts=args.restarts, time_limit=args.time_limit, seed=args.seed,
                                 engine=args.engine, ils_seconds=args.ils_seconds)
                c.pop("exact_set", None)
                row["control"] = dict(kernel_size=H.graph["kernel_size"], isolated_frac=H.graph["isolated_frac"],
                                      n_components=H.graph["n_components"], **c)
            row["seconds"] = time.time() - t0
            fout.write(json.dumps(row) + "\n"); fout.flush()
            ctrl = row.get("control", {})
            ttt = row.get("ils_time_to_target")
            print(f"{inst.name:26s} n={row['n']:4d} deg={row['mean_degree']:6.1f} iso={row['isolated_frac']:.2f} "
                  f"kern={row['kernel_size']:4d} | opt={row['exact_value']:.0f} {row['certified'][:5]} "
                  f"{row['exact_seconds']:6.1f}s | greedy={row['greedy']:.0f} +ls={row['greedy_ls']:.0f} "
                  f"| rr hit={row['rr_hit_rate']:.2f} best={row['rr_best']:.0f} {row['rr_ms']:.0f}ms"
                  + (f" | ILS best={row['ils_best']:.0f} hit={int(row['ils_hit'])} "
                     f"ttt={'-' if ttt is None else f'{ttt:.2f}s'}" if "ils_best" in row else "")
                  + (f" | ER: opt={ctrl.get('exact_value', float('nan')):.0f} +ls={ctrl.get('greedy_ls', float('nan')):.0f} "
                     f"hit={ctrl.get('rr_hit_rate', float('nan')):.2f}" if ctrl else "")
                  + f" | {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
