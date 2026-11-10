"""Anatomy of the GNN's ranking signal on the certified instances."""
import argparse
import json
import os
import time

import numpy as np

from dense_tier.fast_eval import FastEval, tol_of
from dense_tier.moments import estimate_moments
from shared.bruni import load_market, rolling_windows
from shared.metrics import auc_score
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"

N_GRID = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
EPOCH_GRID = [250, 500, 1000, 3000]
SEEDS = list(range(8))
GRAPHS = ["knn", "complete"]
AUC_EVERY = 50
RAND_TRIALS = 40
RESTART_SEARCHES = 300


def random_subsets(rng, n, K, m):
    U = rng.random((m, n))
    idx = np.argpartition(U, K - 1, axis=1)[:, :K]
    X = np.zeros((m, n))
    np.put_along_axis(X, idx, 1.0, axis=1)
    return X


def prefix_curve(ev, objs, supports, f_star,
                 n_grid=N_GRID):
    tol = tol_of(f_star)
    m = len(objs)
    run_best = np.empty(m, dtype=int)
    bi = 0
    for t in range(m):
        if objs[t] < objs[bi]:
            bi = t
        run_best[t] = bi
    curve = []
    for N in n_grid:
        if N > m:
            break
        b = run_best[N - 1]
        f_raw = float(objs[b])
        x = np.zeros(ev.n)
        x[supports[b]] = 1.0
        _, f_sw, _ = ev.one_swap(x)
        curve.append({
            "N": N,
            "raw_hit": bool(f_raw <= f_star + tol),
            "raw_gap": (f_raw - f_star) / abs(f_star),
            "swap_hit": bool(f_sw <= f_star + tol),
            "swap_gap": (f_sw - f_star) / abs(f_star),
        })
    return curve


def compress_traj(objs):
    out = []
    best = np.inf
    for t, v in enumerate(objs):
        if v < best:
            best = float(v)
            out.append((t, best))
    return out


def load_instances(certify_jsonl):
    certs = [json.loads(l) for l in open(certify_jsonl)]
    certs = [c for c in certs if c.get("status") in ("optimal", "best_known") and c.get("support")]
    certs.sort(key=lambda c: (c["market"], c["window"], c["K"]))
    cache = {}
    out = []
    for c in certs:
        mkey = (c["market"], c["window"])
        if mkey not in cache:
            R = load_market(c["market"])
            tr, _ = rolling_windows(R.shape[0])[c["window"]]
            cache[mkey] = estimate_moments(R[tr])
        out.append((c, cache[mkey]))
    return out


def instance_record(c, fm, seed_tag, n_grid=N_GRID):
    K, q = int(c["K"]), float(c["q"])
    ev = FastEval(fm, K, q)

    x_star = np.zeros(fm.n)
    x_star[np.asarray(c["support"], dtype=int)] = 1.0
    f_star = float(ev.f(x_star))
    if abs(f_star - c["obj"]) > tol_of(f_star):
        raise SystemExit(f"certificate mismatch: {f_star} vs {c['obj']}")
    s_star = x_star > 0.5
    tol = tol_of(f_star)

    x_g = ev.greedy()
    x_gs, f_gs, _ = ev.one_swap(x_g)
    greedy_overlap = float((x_g[s_star] > 0.5).mean())

    curves = []
    for t in range(RAND_TRIALS):
        rng = np.random.default_rng([1310, seed_tag, t])
        X = random_subsets(rng, ev.n, ev.K, max(n_grid))
        objs = ev.f_batch(X)
        sup = np.argsort(-X, axis=1)[:, :K]
        curves.append(prefix_curve(ev, objs, sup, f_star, n_grid=n_grid))
    rand_curve = []
    for i, N in enumerate(n_grid):
        rand_curve.append({
            "N": N,
            "raw_hit_rate": float(np.mean([cu[i]["raw_hit"] for cu in curves])),
            "swap_hit_rate": float(np.mean([cu[i]["swap_hit"] for cu in curves])),
            "raw_gap_mean": float(np.mean([cu[i]["raw_gap"] for cu in curves])),
            "swap_gap_mean": float(np.mean([cu[i]["swap_gap"] for cu in curves])),
        })

    rng = np.random.default_rng([1311, seed_tag])
    hits = 0
    t0 = time.perf_counter()
    for _ in range(RESTART_SEARCHES):
        _, f_r, _ = ev.one_swap(ev.random_start(rng))
        if f_r <= f_star + tol:
            hits += 1
    restart_dt = time.perf_counter() - t0

    Sig_ii = np.einsum("ij,ij->i", fm.B, fm.B) + fm.D
    corr_deg = np.abs(fm.corr()).sum(axis=1) - 1.0
    feats = {
        "mu": fm.mu,
        "neg_var": -Sig_ii,
        "mu_over_var": fm.mu / Sig_ii,
        "sharpe": fm.mu / np.sqrt(Sig_ii),
        "neg_corr_degree": -corr_deg,
        "greedy_member": (x_g > 0.5).astype(float),
        "greedy_swap_member": (x_gs > 0.5).astype(float),
    }
    trivial = {k: auc_score(v, s_star) for k, v in feats.items()}

    return {
        "type": "instance",
        "market": c["market"], "window": int(c["window"]), "K": K, "q": q,
        "n": int(fm.n), "f_star": f_star,
        "shrinkage": float(fm.shrinkage),
        "greedy_overlap": greedy_overlap,
        "gswap_hit": bool(f_gs <= f_star + tol),
        "gap_gswap": (f_gs - f_star) / abs(f_star),
        "random_curve": rand_curve,
        "restart_hit_rate": hits / RESTART_SEARCHES,
        "restart_searches": RESTART_SEARCHES,
        "restart_search_seconds": restart_dt / RESTART_SEARCHES,
        "trivial_auc": trivial,
    }


def family_loss(qb, scale="objective"):
    unit = qb.lam_final / (100.0 if scale == "objective" else 2.0)

    def loss(p, frac):
        return qb.train_loss(p, 1.0) / unit

    return loss, float(unit)


def qubo_form_loss(qb):
    a = (1 - qb.q) / qb.K

    def loss(p, frac):
        lam = qb.lam(frac)
        p2 = p * p
        return ((qb.q / qb.K**2) * qb.quad(p) - a * (qb._mu @ p2)
                + lam * (p.sum() ** 2 - 2 * qb.K * p2.sum() + qb.K**2))

    return loss


def gnn_run(c, fm, graph, seed, epochs, decode_mode="dropout",
            model="qrf", conv="gcn", n_grid=N_GRID, family_scale="objective", gamma0=None, device="cpu",
            init_kn=False, embed_input=False, dropout=0.5, loss_form="convex"):
    import torch
    from dense_tier.gnn_factored import decode_topk, run_gnn_factored
    from dense_tier.graphs import GRAPH_BUILDERS
    from dense_tier.qubo import PortfolioQubo
    from shared.gnn import nx_to_pyg_edge_index

    torch.set_num_threads(int(os.environ.get("GNN_TORCH_THREADS", "1")))
    K, q = int(c["K"]), float(c["q"])
    ev = FastEval(fm, K, q)
    qb = PortfolioQubo(fm, K=K, q=q, device=device)

    x_star = np.zeros(fm.n)
    x_star[np.asarray(c["support"], dtype=int)] = 1.0
    f_star = float(ev.f(x_star))
    s_star = x_star > 0.5
    tol = tol_of(f_star)

    G = GRAPH_BUILDERS[graph](fm)
    ei = nx_to_pyg_edge_index(G, device=device)

    probs_log = []
    objs = []
    sups = []

    def capture_decode(p_t):
        p = p_t.detach().cpu().numpy().copy()
        probs_log.append(p)
        x = decode_topk(p_t, K)
        sups.append(np.flatnonzero(x > 0.5))
        return x

    if model == "qrf":
        train_loss, loss_unit = (qb.train_loss if loss_form == "convex" else qubo_form_loss(qb)), None
    else:
        train_loss, loss_unit = family_loss(qb, family_scale)
    gamma0 = float(gamma0) if (gamma0 is not None and model == "cra") else None

    t0 = time.time()
    x_raw, f_raw, best_ep, diag = run_gnn_factored(
        train_loss, qb.feasible_objective, K, ei, G,
        decode_fn=capture_decode,
        number_epochs=epochs, seed=seed, log_every=0,
        decode_mode=decode_mode, model=model, conv=conv,
        family_overrides={"gamma0": gamma0} if gamma0 is not None else None, device=device,
        init_p=(K / fm.n) if init_kn else None, embed_input=embed_input, dropout=dropout)
    train_time = time.time() - t0
    X = np.zeros((len(sups), fm.n))
    for i, s in enumerate(sups):
        X[i, s] = 1.0
    objs = ev.f_batch(X)

    _, f_swap, _ = ev.one_swap(np.asarray(x_raw))
    p_best = probs_log[best_ep]
    p_final = probs_log[-1]
    topk = np.zeros(fm.n)
    topk[np.argsort(-p_best)[:K]] = 1.0

    auc_traj = [[int(e), auc_score(probs_log[e], s_star)]
                for e in range(0, len(probs_log), AUC_EVERY)]

    rec = {
        "type": "run",
        "market": c["market"], "window": int(c["window"]), "K": K, "q": q,
        "n": int(fm.n), "graph": graph, "seed": seed, "epochs": epochs,
        "decode_mode": decode_mode,
        "model": model, "conv": conv if model != "qrf" else None,
        "model_settings": diag["model_settings"], "loss_unit": loss_unit,
        "family_scale": family_scale if model != "qrf" else None,
        "gamma0_override": gamma0,
        "stop_reason": diag["stop_reason"],
        "gamma_at_stop": diag.get("gamma_at_stop"), "phi_final": diag.get("phi_final"),
        "epochs_run": int(diag["epochs_run"]),
        "final_train_loss": float(diag["final_train_loss"]),
        "final_loss_readout": float(diag["final_loss_readout"]),
        "final_p": diag["final_p"],
        "init_sum_p": float(diag["sum_p"][0]),
        "arm": {"init_kn": bool(init_kn), "embed_input": bool(embed_input), "dropout": float(dropout),
                "loss_form": loss_form},
        "best_epoch": int(best_ep),
        "f_star": f_star,
        "f_gnn_raw": float(f_raw),
        "f_gnn_swap": float(f_swap),
        "gap_raw": (f_raw - f_star) / abs(f_star),
        "gap_swap": (f_swap - f_star) / abs(f_star),
        "hit_raw": bool(f_raw <= f_star + tol),
        "hit_swap": bool(f_swap <= f_star + tol),
        "auc_best": auc_score(p_best, s_star),
        "auc_final": auc_score(p_final, s_star),
        "prec_at_k": float((topk[s_star] > 0.5).mean()),
        "prec_random": K / fm.n,
        "auc_traj": auc_traj,
        "traj_best": compress_traj(objs),
        "seconds": train_time,
        "edges": G.number_of_edges(),
        "device": str(device),
    }
    rec["curve"] = prefix_curve(ev, np.asarray(objs), np.stack(sups), f_star, n_grid=n_grid)
    return rec


def worker(task):
    (_, c, graph, seed, epochs), opts = task
    key = (c["market"], c["window"])
    fm = _FM_CACHE.get(key)
    if fm is None:
        R = load_market(c["market"])
        tr, _ = rolling_windows(R.shape[0])[c["window"]]
        fm = estimate_moments(R[tr])
        _FM_CACHE[key] = fm
    return gnn_run(c, fm, graph, seed, epochs, **opts)


_FM_CACHE = {}
DEFAULT_ARM = {"init_kn": False, "embed_input": False, "dropout": 0.5, "loss_form": "convex"}


def run_config(r):
    model = r.get("model", "qrf")
    return (model, r.get("conv"), (r.get("family_scale") or "objective") if model != "qrf" else None,
            r.get("gamma0_override"), sorted((r.get("arm") or DEFAULT_ARM).items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certify_jsonl", default=RESULTS / "certify.jsonl")
    ap.add_argument("--runs_out", default=RESULTS / "signal_runs.jsonl")
    ap.add_argument("--inst_out", default=RESULTS / "signal_instances.jsonl")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--graphs", nargs="+", default=GRAPHS)
    ap.add_argument("--epoch_grid", type=int, nargs="+", default=EPOCH_GRID)
    ap.add_argument("--limit_instances", type=int, default=0)
    ap.add_argument("--decode_mode", default="dropout", choices=("dropout", "nodrop"))
    ap.add_argument("--model", default="qrf", choices=["qrf", "pignn", "cra"])
    ap.add_argument("--conv", default="gcn", choices=["gcn", "sage"])
    ap.add_argument("--family_scale", default="objective", choices=["objective", "penalty"])
    ap.add_argument("--gamma0", type=float, default=None)
    ap.add_argument("--n_grid_extra", type=int, nargs="*", default=[])
    ap.add_argument("--init_kn", action="store_true")
    ap.add_argument("--embed_input", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--loss_form", default="convex", choices=["convex", "qubo"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    grid = sorted(set(N_GRID) | set(args.n_grid_extra))
    opts = {"decode_mode": args.decode_mode,
            "model": args.model, "conv": args.conv, "n_grid": grid,
            "family_scale": args.family_scale, "gamma0": args.gamma0, "device": args.device,
            "init_kn": args.init_kn, "embed_input": args.embed_input, "dropout": args.dropout,
            "loss_form": args.loss_form}
    config = run_config({
        "model": args.model, "conv": args.conv if args.model != "qrf" else None,
        "family_scale": args.family_scale, "gamma0_override": args.gamma0 if args.model == "cra" else None,
        "arm": {"init_kn": args.init_kn, "embed_input": args.embed_input, "dropout": float(args.dropout),
                "loss_form": args.loss_form}})
    print(f"model={args.model} device={args.device} decode_mode={args.decode_mode}", flush=True)

    instances = load_instances(args.certify_jsonl)
    if args.limit_instances:
        instances = instances[: args.limit_instances]

    done_inst = set()
    if os.path.exists(args.inst_out):
        for line in open(args.inst_out):
            r = json.loads(line)
            done_inst.add((r["market"], r["window"], r["K"], r["q"]))
            if [pt["N"] for pt in r["random_curve"]] != grid:
                raise SystemExit(f"{args.inst_out} was written with another candidate grid")
    with open(args.inst_out, "a") as fh:
        for i, (c, fm) in enumerate(instances):
            key = (c["market"], c["window"], c["K"], c["q"])
            if key in done_inst:
                continue
            t0 = time.time()
            rec = instance_record(c, fm, i, n_grid=grid)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  [{i+1}/{len(instances)}] {key} h={rec['restart_hit_rate']:.2f} "
                  f"rand_swap@{grid[-1]}={rec['random_curve'][-1]['swap_hit_rate']:.2f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    done = set()
    if os.path.exists(args.runs_out):
        for line in open(args.runs_out):
            r = json.loads(line)
            done.add((r["market"], r["window"], r["K"], r["q"], r["graph"],
                      r["seed"], r["epochs"],
                      r.get("decode_mode", "dropout")))
            if run_config(r) != config:
                raise SystemExit(f"{args.runs_out} holds runs of another configuration")

    tasks = []
    for ci, (c, fm) in enumerate(instances):
        for graph in args.graphs:
            for seed in args.seeds:
                for E in args.epoch_grid:
                    k = (c["market"], c["window"], c["K"], c["q"], graph, seed, E,
                         args.decode_mode)
                    if k not in done:
                        tasks.append(((ci, c, graph, seed, E), opts))
    tasks.sort(key=lambda t: (-t[0][4], t[0][2] != "complete"))
    print(f"{len(tasks)} GNN runs to do "
          f"({len(done)} already present)", flush=True)

    t_start = time.time()
    if args.workers <= 1:
        it = map(worker, tasks)
        with open(args.runs_out, "a") as fh:
            for j, rec in enumerate(it):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                _progress(j, len(tasks), rec, t_start)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool, open(args.runs_out, "a") as fh:
            for j, rec in enumerate(pool.imap_unordered(worker, tasks,
                                                        chunksize=1)):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                _progress(j, len(tasks), rec, t_start)

    print(f"done in {(time.time()-t_start)/60:.1f} min -> "
          f"{args.runs_out}", flush=True)


def _progress(j, total, rec, t_start):
    el = time.time() - t_start
    eta = el / (j + 1) * (total - j - 1)
    print(f"  [{j+1}/{total}] {rec['market']}_w{rec['window']}_K{rec['K']} "
          f"{rec['graph']} s{rec['seed']} E{rec['epochs']}: "
          f"auc={rec['auc_best']:.3f} hit_swap={int(rec['hit_swap'])} "
          f"({rec['seconds']:.1f}s) eta {eta/60:.0f}m", flush=True)


if __name__ == "__main__":
    main()
