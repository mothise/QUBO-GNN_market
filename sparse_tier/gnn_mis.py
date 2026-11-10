"""
QRF-GNN, PI-GNN and CRA-PI-GNN on market-graph independent sets; QRF-GNN also over a ladder of message graphs.
"""
import argparse
import json
import os
import time
from collections import deque

import networkx as nx
import numpy as np

import torch
import torch.nn as nn

from shared.gnn import build_gnn, build_static_inputs, nx_to_pyg_edge_index, qubo_loss
from shared.mis import build_mis_qubo_components, linear_penalty, make_mis_q_train
from sparse_tier.market_graph import (correlation, er_control, kernel_strong, knn_graph, load_returns_bruni,
                          rewired_control, threshold_graph, weighted_window)
from sparse_tier.mis_baselines import ils_arw, local_search_arw, restart_hit_rate, value
from sparse_tier.mis_exact import certify
from shared.paths import REPO_ROOT
from shared.solver_family import CONVS, MODELS, SETTINGS, PlateauStop, build_family, cra_gamma, cra_phi

RESULTS = REPO_ROOT / "sparse_tier" / "results"
QRF_DEFAULTS = dict(epochs=3000, patience=1000, lr=1.4e-2, p_min=0.01, p_max=2.0, dropout=0.5)


def model_defaults(model):
    if model == "qrf":
        return dict(QRF_DEFAULTS)
    s = SETTINGS[model]
    return dict(epochs=s["max_epochs"], patience=0, lr=s["lr"], p_min=s["penalty"], p_max=s["penalty"],
                dropout=s["dropout"])


def threshold_for_edge_count(C, target_m):
    iu = np.triu_indices(C.shape[0], 1)
    c = np.sort(C[iu])[::-1]
    target_m = int(min(max(target_m, 1), len(c) - 1))
    return float((c[target_m - 1] + c[target_m]) / 2)


def build_rung(rung, G, C, seed):
    n, m = G.number_of_nodes(), G.number_of_edges()
    if rung == "problem":
        H = nx.Graph(G); H.graph["builder"] = "problem"
    elif rung == "empty":
        H = nx.empty_graph(n); H.graph["builder"] = "empty"
    elif rung.startswith("dense") or rung.startswith("er_dense"):
        M = int(rung.split("dense")[1])
        tau2 = threshold_for_edge_count(C, M * m)
        H = threshold_graph(C, tau2)
        if rung.startswith("er_"):
            H = er_control(H, seed=seed)
        H.graph["effective_tau"] = tau2
    elif rung == "knn":
        k = min(max(1, int(round(2 * 2 * m / n))), n - 1)
        H = knn_graph(C, k)
        H.graph["k"] = k
    else:
        raise ValueError(rung)
    H.graph["rung"] = rung
    H.graph["m_message"] = H.number_of_edges()
    H.graph["m_problem"] = m
    return H


def fix_independent_set_fast(bits, edges, deg):
    sel = bits.copy()
    if len(edges) == 0:
        return sel
    e = edges[sel[edges[:, 0]] & sel[edges[:, 1]]]
    n = len(sel)
    scale = float(deg.max() + 1)
    while len(e):
        cnt = np.bincount(e.ravel(), minlength=n)
        key = cnt * scale + deg
        key[cnt == 0] = -1
        worst = int(np.flatnonzero(key == key.max())[0])
        sel[worst] = False
        e = e[(e[:, 0] != worst) & (e[:, 1] != worst)]
    return sel


def readout_no_dropout(net, edge_index, static, h0):
    saved = net.dropout_frac
    net.dropout_frac = 0.0
    with torch.no_grad():
        probs, _ = net(edge_index, static, h0)
    net.dropout_frac = saved
    return probs


def train_one(G_problem, G_message, w, *, seed, epochs,
              lr, p_min, p_max, patience, dim_embedding,
              hidden_dim, dropout, device, log_every,
              decode_every=1,
              model="qrf", conv="gcn", traj_stride=1, ramp_epochs=0,
              edge_penalty="uniform"):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device); dtype = torch.float32
    n = G_problem.number_of_nodes()

    _, Q_off = build_mis_qubo_components(G_problem, torch_dtype=dtype, torch_device=dev)
    w_t = torch.tensor(w, dtype=dtype, device=dev)
    if edge_penalty == "min":
        Q_off = Q_off * torch.minimum(w_t[:, None], w_t[None, :])
    Q_train = make_mis_q_train(torch.diag(-w_t), Q_off, p_min=p_min, p_max=p_max)

    if G_message.number_of_edges() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=dev)
    else:
        edge_index = nx_to_pyg_edge_index(G_message, device=dev)

    if model == "qrf":
        hypers = dict(dim_embedding=dim_embedding, hidden_dim=hidden_dim, dropout=dropout, number_classes=1)
        net, _, opt = build_gnn(n, hypers, {"lr": lr}, dev, dtype)
        static = build_static_inputs(n, dim_embedding, G_problem, dev, dtype)
        fam, stopper = None, None
    else:
        net, opt, fam = build_family(model, n, dev, dtype, conv=conv, lr=lr, dropout=dropout)
        static = None
        stopper = PlateauStop(fam["tol"], fam["patience"])
    h0 = torch.zeros(n, 1, dtype=dtype, device=dev)
    stop_reason = "epochs"

    edges_arr = np.array(G_problem.edges(), dtype=np.int64).reshape(-1, 2)
    deg_arr = np.array([d for _, d in sorted(G_problem.degree())], dtype=np.int64)
    traj_feasible, traj_raw, traj_viol, traj_loss = [], [], [], []
    best_val, best_epoch, best_p, best_set = -np.inf, 0, None, None
    since_improve = 0
    denom = max(1, (ramp_epochs or epochs) - 1)
    loss_window, paper_stop_epoch = deque(maxlen=500), None
    t0 = time.perf_counter()
    decode_time = 0.0
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    for ep in range(epochs):
        Qt = Q_train(ep / denom)
        probs, h0_new = net(edge_index, static, h0.detach())
        probs = probs.squeeze()
        loss = qubo_loss(probs, Qt)
        if model == "cra":
            loss = loss + cra_gamma(fam, ep) * cra_phi(probs, fam["alpha"])
        plateau = stopper is not None and stopper.step(float(loss.detach()))
        if model == "qrf" and paper_stop_epoch is None:
            loss_now = float(loss.detach())
            quiet = bool(loss_window) and abs(loss_now - loss_window[-1]) < 1e-5
            loss_window.append(loss_now)
            if quiet and len(loss_window) == loss_window.maxlen and max(loss_window) - min(loss_window) < 1e-5:
                paper_stop_epoch = ep
        do_decode = (ep % decode_every == 0) or (ep == epochs - 1) or plateau
        if not do_decode:
            h0 = h0_new
            traj_feasible.append(traj_feasible[-1] if traj_feasible else 0.0)
            traj_raw.append(traj_raw[-1] if traj_raw else 0); traj_viol.append(traj_viol[-1] if traj_viol else 0)
            traj_loss.append(float(loss.detach())); since_improve += 1
            if patience and since_improve >= patience:
                stop_reason = "objective_patience"
                break
            opt.zero_grad(); loss.backward()
            if model == "qrf":
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
            opt.step()
            continue
        sync(); td = time.perf_counter()
        with torch.no_grad():
            pd_np = probs.detach().cpu().numpy()
            bits = pd_np >= 0.5
            viol = int((bits[edges_arr[:, 0]] & bits[edges_arr[:, 1]]).sum()) if len(edges_arr) else 0
            fixed_bits = fix_independent_set_fast(bits, edges_arr, deg_arr) if viol else bits
            val = float(w[fixed_bits].sum())
        decode_time += time.perf_counter() - td
        h0 = h0_new
        traj_feasible.append(val); traj_raw.append(int(bits.sum())); traj_viol.append(viol); traj_loss.append(float(loss.detach()))
        if val > best_val + 1e-12:
            best_val, best_epoch, since_improve = val, ep, 0
            best_p = pd_np.copy(); best_set = [int(i) for i in np.flatnonzero(fixed_bits)]
        else:
            since_improve += 1
        if log_every and ep % log_every == 0:
            print(f"    ep={ep:5d} loss={float(loss):+9.3f} raw={int(bits.sum()):4d} viol={viol:4d} "
                  f"feasible={val:7.2f} best={best_val:7.2f}@{best_epoch}", flush=True)
        if patience and since_improve >= patience:
            stop_reason = "objective_patience"
            break
        if plateau:
            stop_reason = "loss_plateau"
            break
        opt.zero_grad(); loss.backward()
        if model == "qrf":
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        opt.step()
    sync()
    wall = time.perf_counter() - t0
    ls = local_search_arw(G_problem, set(best_set))
    n_ep = max(1, len(traj_feasible))
    with torch.no_grad():
        pf, _ = net(edge_index, static, h0.detach())
        final_p = pf.squeeze().detach().cpu().numpy()
    fbits = final_p >= 0.5
    final_fixed = fix_independent_set_fast(fbits, edges_arr, deg_arr)
    final_value = float(w[final_fixed].sum())
    nodrop_p = readout_no_dropout(net, edge_index, static, h0.detach()).squeeze().detach().cpu().numpy()
    nodrop_bits = nodrop_p >= 0.5
    nodrop_viol = int((nodrop_bits[edges_arr[:, 0]] & nodrop_bits[edges_arr[:, 1]]).sum()) if len(edges_arr) else 0
    nodrop_fixed = fix_independent_set_fast(nodrop_bits, edges_arr, deg_arr)
    rng_nc = np.random.default_rng(seed + 991)
    k_best = int(max(1, (best_p >= 0.5).sum())) if best_p is not None else n // 2
    noise_vals = []
    for _ in range(20):
        rb = np.zeros(n, dtype=bool); rb[rng_nc.choice(n, size=min(k_best, n), replace=False)] = True
        noise_vals.append(float(w[fix_independent_set_fast(rb, edges_arr, deg_arr)].sum()))
    n_run = len(traj_feasible)
    stride = max(1, int(traj_stride))
    final_raw, final_viol = traj_raw[-1], traj_viol[-1]
    family_fields = dict(model=model, traj_stride=stride, stop_reason=stop_reason)
    if fam is not None:
        family_fields["model_settings"] = fam
    if model == "cra":
        family_fields["gamma_at_stop"] = float(cra_gamma(fam, n_run - 1))
        family_fields["phi_final"] = float(np.sum(1.0 - (2.0 * final_p - 1.0) ** fam["alpha"]))
    value_at_paper_stop = None if paper_stop_epoch is None else float(traj_feasible[paper_stop_epoch])
    traj_feasible, traj_raw, traj_viol = traj_feasible[::stride], traj_raw[::stride], traj_viol[::stride]
    return dict(final_p=[float(x) for x in final_p], final_value=final_value, **family_fields,
                final_p_nodrop=[float(x) for x in nodrop_p], final_value_nodrop=float(w[nodrop_fixed].sum()),
                final_plus_ls_nodrop=value(G_problem, local_search_arw(G_problem, set(np.flatnonzero(nodrop_fixed).tolist()))),
                final_raw_nodrop=int(nodrop_bits.sum()), final_viol_nodrop=nodrop_viol,
                ramp_epochs=int(ramp_epochs or epochs), edge_penalty=edge_penalty,
                paper_stop_epoch=paper_stop_epoch, value_at_paper_stop=value_at_paper_stop,
                final_plus_ls=value(G_problem, local_search_arw(G_problem, set(np.flatnonzero(final_fixed).tolist()))),
                k_raw_best=k_best, noise_fix_mean=float(np.mean(noise_vals)), noise_fix_max=float(np.max(noise_vals)),
                best_value=best_val, best_epoch=best_epoch, epochs_run=n_run,
                p_min=float(p_min), p_max=float(p_max), patience=int(patience), epochs_nominal=int(epochs),
                penalty_at_stop=float(linear_penalty((n_run - 1) / denom, p_min=p_min, p_max=p_max)),
                penalty_at_best=float(linear_penalty(best_epoch / denom, p_min=p_min, p_max=p_max)),
                final_sat1=float(np.mean(final_p > 0.99)), final_sat0=float(np.mean(final_p < 0.01)),
                wall_seconds=wall, train_seconds=wall - decode_time, decode_seconds=decode_time,
                train_seconds_per_epoch=(wall - decode_time) / n_ep,
                best_plus_ls=value(G_problem, ls), best_set=best_set, best_p=[float(x) for x in best_p],
                traj_feasible=traj_feasible, traj_raw=traj_raw, traj_viol=traj_viol,
                final_raw=final_raw, final_viol=final_viol, collapsed=bool(final_raw == 0))


def load_screen(paths, *, only_certified=True, min_tau=None, max_mean_degree=None,
                names=None):
    by = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                by[r["name"]] = r
    rows = []
    names = set(names) if names else None
    for r in by.values():
        if names is not None and r["name"] not in names:
            continue
        if only_certified and not r.get("reference_is_certified"):
            continue
        if min_tau is not None and r["tau"] < min_tau:
            continue
        if max_mean_degree is not None and r["mean_degree"] > max_mean_degree:
            continue
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", nargs="+", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min_strong_kernel", type=int, default=0)
    ap.add_argument("--decode_every", type=int, default=1)
    ap.add_argument("--min_tau", type=float, default=None)
    ap.add_argument("--max_mean_degree", type=float, default=None)
    ap.add_argument("--max_instances", type=int, default=None)
    ap.add_argument("--rungs", nargs="+", default=["problem", "dense2", "dense4", "knn", "er_dense2", "empty"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--p_min", type=float, default=None)
    ap.add_argument("--p_max", type=float, default=None)
    ap.add_argument("--dim_embedding", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_every", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "gnn_runs.jsonl"))
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--names_file", default=None)
    ap.add_argument("--problem_control", default="none", choices=["none", "er", "rewired"])
    ap.add_argument("--control_seed", type=int, default=20261031)
    ap.add_argument("--control_ref_time", type=float, default=20.0)
    ap.add_argument("--control_ils_time", type=float, default=5.0)
    ap.add_argument("--model", default="qrf", choices=list(MODELS))
    ap.add_argument("--conv", default="gcn", choices=list(CONVS))
    ap.add_argument("--traj_stride", type=int, default=1)
    ap.add_argument("--ramp_epochs", type=int, default=0)
    ap.add_argument("--edge_penalty", default="uniform", choices=["uniform", "min"])
    args = ap.parse_args()
    for key, val in model_defaults(args.model).items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    names = list(args.names) if args.names else []
    if args.names_file:
        with open(args.names_file) as f:
            names += [l.strip() for l in f if l.strip() and not l.startswith("#")]
    rows = load_screen(args.screen, only_certified=not args.all, min_tau=args.min_tau,
                       max_mean_degree=args.max_mean_degree, names=names or None)
    if names:
        missing = sorted(set(names) - {r["name"] for r in rows})
        if missing:
            print(f"{len(missing)} requested names not found / filtered out: {missing[:5]}...")
    if args.max_instances:
        rows = rows[:args.max_instances]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line); done.add((r["name"], r["rung"], r["seed"]))
                if r.get("model", "qrf") != args.model:
                    raise SystemExit(f"{args.out} holds runs of model {r.get('model', 'qrf')}")
                if r.get("edge_penalty", "uniform") != args.edge_penalty:
                    raise SystemExit(f"{args.out} holds runs of edge penalty {r.get('edge_penalty', 'uniform')}")

    cache = {}
    with open(args.out, "a") as fout:
        for r in rows:
            if r["source"] not in cache:
                cache[r["source"]] = load_returns_bruni(r["source"])
            R, ws = cache[r["source"]]
            Rw, w = weighted_window(R[ws[r["window"]][0]], r.get("weights", "unit"))
            C = correlation(Rw)
            G = threshold_graph(C, r["tau"], weights=w, edges=r.get("edges", "signed"))
            if (G.number_of_nodes(), G.number_of_edges()) != (r["n"], r["m"]):
                raise SystemExit("screen/instance mismatch")
            name = r["name"]
            ref, ref_cert, exact_set, ils_ttt, control_info = (r["reference"], r["reference_is_certified"],
                                                                r.get("exact_set"), r.get("ils_time_to_target"), None)
            if args.problem_control != "none":
                name = f"{r['name']}__{args.problem_control}"
                if all((name, rg, s) in done for rg in args.rungs for s in args.seeds):
                    continue
                cseed = args.control_seed + 7919 * int(r["window"]) + int(round(100 * r["tau"]))
                t_c = time.perf_counter()
                Gc = er_control(G, seed=cseed) if args.problem_control == "er" else rewired_control(G, seed=cseed)
                t_build = time.perf_counter() - t_c
                cert = certify(Gc, time_limit=args.control_ref_time)
                ils = ils_arw(Gc, time_budget=args.control_ils_time, rng=np.random.default_rng(cseed),
                              reference=cert["value"] if cert["status"] == "optimal" else None, stop_at_target=True)
                if cert["status"] == "optimal":
                    ref, ref_cert, exact_set = cert["value"], True, cert["set"]
                elif ils["best_value"] >= cert["value"]:
                    ref, ref_cert, exact_set = ils["best_value"], False, sorted(int(v) for v in ils["best_set"])
                else:
                    ref, ref_cert, exact_set = cert["value"], False, cert["set"]
                ils_ttt = ils["time_to_target"]
                rr = restart_hit_rate(Gc, ref, 20, seed=cseed)
                r = dict(r, rr_ms=1000.0 * rr["seconds_per_restart"], rr_hit_rate=rr["hit_rate"], rr_best=rr["best"])
                control_info = dict(control=args.problem_control, control_seed=cseed, control_build_seconds=t_build,
                                    control_cert_status=cert["status"], control_cert_bound=cert["bound"],
                                    control_cert_seconds=cert["seconds"], control_ils_best=ils["best_value"],
                                    control_rr_hit_rate=rr["hit_rate"], control_rr_ms=1000.0 * rr["seconds_per_restart"],
                                    control_market_reference=r["reference"], control_market_mean_degree=r["mean_degree"])
                print(f"{name}: control built in {t_build:.1f}s; HiGHS {cert['status']} value {cert['value']:.0f} "
                      f"bound {cert['bound']:.0f} ({cert['seconds']:.1f}s); ILS best {ils['best_value']:.0f}; reference "
                      f"{ref:.0f} (certified={ref_cert})", flush=True)
                G = Gc
            ks = kernel_strong(G)
            if ks["kernel_size"] < args.min_strong_kernel:
                print(f"{name}: strong kernel {ks['kernel_size']} < {args.min_strong_kernel}, skipped")
                continue
            unfixed = sorted(int(v) for v in ks["unfixed"])
            mean_degree = float(np.mean([d for _, d in G.degree()]))
            for rung in args.rungs:
                for seed in args.seeds:
                    if (name, rung, seed) in done:
                        continue
                    H = build_rung(rung, G, C, seed)
                    res = train_one(G, H, w, seed=seed, epochs=args.epochs, lr=args.lr, p_min=args.p_min,
                                    p_max=args.p_max, patience=args.patience, dim_embedding=args.dim_embedding,
                                    hidden_dim=args.hidden_dim, dropout=args.dropout, device=args.device,
                                    log_every=args.log_every, decode_every=args.decode_every,
                                    model=args.model, conv=args.conv, traj_stride=args.traj_stride,
                                    ramp_epochs=args.ramp_epochs, edge_penalty=args.edge_penalty)
                    out = dict(name=name, source=r["source"], window=r["window"], tau=r["tau"], rung=rung,
                               seed=seed, n=G.number_of_nodes(), m_problem=G.number_of_edges(),
                               m_message=H.number_of_edges(), effective_tau=H.graph.get("effective_tau"),
                               mean_degree=mean_degree, reference=ref,
                               reference_is_certified=ref_cert, exact_set=exact_set,
                               kernel_strong=int(ks["kernel_size"]), unfixed_strong=unfixed,
                               degree=[int(d) for _, d in sorted(G.degree())],
                               ils_time_to_target=ils_ttt, rr_ms=r.get("rr_ms"),
                               device=args.device, **(control_info or {}), **res)
                    fout.write(json.dumps(out) + "\n"); fout.flush()
                    print(f"    best={res['best_value']:.0f}@{res['best_epoch']} +ls={res['best_plus_ls']:.0f} "
                          f"final={res['final_value']:.0f} (no dropout {res['final_value_nodrop']:.0f}) "
                          f"noise-fix={res['noise_fix_mean']:.0f} "
                          f"ref={ref:.0f} train={res['train_seconds']:.1f}s (+decode {res['decode_seconds']:.1f}s) "
                          f"raw_final={res['final_raw']} viol_final={res['final_viol']} collapsed={res['collapsed']} "
                          f"epochs={res['epochs_run']} stop={res['stop_reason']}")


if __name__ == "__main__":
    main()
