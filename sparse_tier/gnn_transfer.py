"""
Transfer across windows.
"""
import argparse
import copy
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from shared.metrics import auc_score
from sparse_tier.metrics import auc_stratified, residualise
from shared.gnn import build_gnn, nx_to_pyg_edge_index, pagerank_features, qubo_loss
from shared.mis import build_mis_qubo_components, make_mis_q_train
from sparse_tier.gnn_mis import fix_independent_set_fast
from sparse_tier.market_graph import correlation, kernel_strong, load_returns_bruni, threshold_graph
from sparse_tier.mis_baselines import greedy_min_degree, ils_arw, local_search_arw, value
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "sparse_tier" / "results"


def load_screen(paths, *, names=None):
    by = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line); by[r["name"]] = r
    names = set(names) if names else None
    return [r for r in by.values() if r.get("reference_is_certified") and (names is None or r["name"] in names)]


def build_instance(r, R, ws):
    Rw = R[ws[r["window"]][0]]
    C = correlation(Rw)
    G = threshold_graph(C, r["tau"])
    w = np.ones(G.number_of_nodes())
    if G.number_of_edges() != r["m"]:
        raise SystemExit(f"{r['name']}: screen/instance mismatch")
    ks = kernel_strong(G)
    n = G.number_of_nodes()
    kern = np.zeros(n, bool); kern[sorted(int(v) for v in ks["unfixed"])] = True
    y = np.zeros(n); y[r["exact_set"]] = 1
    deg = np.array([d for _, d in sorted(G.degree())], dtype=float)
    return dict(name=r["name"], window=r["window"], tau=r["tau"], G=G, w=w, n=n, alpha=float(r["reference"]),
                y=y, kern=kern, deg=deg, kernel_strong=int(ks["kernel_size"]),
                edges=np.array(G.edges(), dtype=np.int64).reshape(-1, 2), degi=deg.astype(np.int64),
                rr_ms=r.get("rr_ms"), ils_ttt=r.get("ils_time_to_target"), greedy_ls=r.get("greedy_ls"))


@contextmanager
def dropout_off(net):
    p = net.dropout_frac
    net.dropout_frac = 0.0
    try:
        yield
    finally:
        net.dropout_frac = p


def static_from_block(rand_block, G, dtype):
    dev = rand_block.device
    walk = pagerank_features(G, 2 * rand_block.shape[1]).to(dev).type(dtype)
    return torch.cat([rand_block, torch.ones_like(rand_block), torch.ones_like(rand_block), walk], 1)


def decode(p_np, inst):
    bits = p_np >= 0.5
    e = inst["edges"]
    viol = int((bits[e[:, 0]] & bits[e[:, 1]]).sum()) if len(e) else 0
    fixed = fix_independent_set_fast(bits, e, inst["degi"]) if viol else bits
    S = set(np.flatnonzero(fixed).tolist())
    val = float(inst["w"][fixed].sum())
    ls = local_search_arw(inst["G"], S)
    return dict(raw=int(bits.sum()), viol=viol, value=val, value_ls=value(inst["G"], ls), set_ls=sorted(int(v) for v in ls))


def rank_metrics(p_np, inst):
    y, kern, deg = inst["y"], inst["kern"], inst["deg"]
    out = dict(auc_all=auc_score(p_np, y))
    if kern.sum() >= 10 and 0 < y[kern].sum() < kern.sum():
        out["auc_kernel"] = auc_score(p_np[kern], y[kern])
        out["auc_resid_kernel"] = auc_score(residualise(p_np[kern], -deg[kern]), y[kern])
        out["auc_strat_kernel"] = auc_stratified(p_np[kern], y[kern], deg[kern])
        out["auc_degree_kernel"] = auc_score(-deg[kern], y[kern])
    return out


def train_source(inst, *, seed, epochs, lr, p_min, p_max, patience,
                 dim_embedding, hidden_dim, dropout, device):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device); dtype = torch.float32
    G, w, n = inst["G"], inst["w"], inst["n"]
    Q_diag, Q_off = build_mis_qubo_components(G, torch_dtype=dtype, torch_device=dev)
    Q_train = make_mis_q_train(Q_diag, Q_off, p_min=p_min, p_max=p_max)
    edge_index = nx_to_pyg_edge_index(G, device=dev)
    hypers = dict(dim_embedding=dim_embedding, hidden_dim=hidden_dim, dropout=dropout, number_classes=1)
    net, _, opt = build_gnn(n, hypers, {"lr": lr}, dev, dtype)
    rand_block = torch.rand((n, dim_embedding), dtype=dtype, device=dev)
    static = static_from_block(rand_block, G, dtype)
    h0 = torch.zeros(n, 1, dtype=dtype, device=dev)
    best_val, best_epoch, since = -np.inf, 0, 0
    denom = max(1, epochs - 1)
    t0 = time.perf_counter()
    ep_run = 0
    for ep in range(epochs):
        ep_run = ep + 1
        probs, h0_new = net(edge_index, static, h0.detach())
        probs = probs.squeeze()
        loss = qubo_loss(probs, Q_train(ep / denom))
        with torch.no_grad():
            pd_np = probs.detach().cpu().numpy()
            bits = pd_np >= 0.5
            e = inst["edges"]
            viol = int((bits[e[:, 0]] & bits[e[:, 1]]).sum()) if len(e) else 0
            fixed = fix_independent_set_fast(bits, e, inst["degi"]) if viol else bits
            val = float(w[fixed].sum())
        h0 = h0_new
        if val > best_val + 1e-12:
            best_val, best_epoch, since = val, ep, 0
        else:
            since += 1
        if patience and since >= patience:
            break
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    with torch.no_grad():
        pf, _ = net(edge_index, static, h0.detach())
        final_p = pf.squeeze().detach().cpu().numpy()
    dec = decode(final_p, inst)
    return dict(net=net, rand_block=rand_block, h0_final=h0.detach(), final_p=final_p, final_decode=dec,
                best_value=float(best_val), best_epoch=int(best_epoch), epochs_run=int(ep_run), train_seconds=train_s,
                static_src=static)


def zero_shot(net, rand_block, inst_t, *, iters, h0_init, bn_train,
              snapshots=(1, 5, 10, 50, 200)):
    dev = rand_block.device; dtype = rand_block.dtype
    G = inst_t["G"]
    static = static_from_block(rand_block, G, dtype)
    edge_index = nx_to_pyg_edge_index(G, device=dev)
    h0 = torch.zeros(inst_t["n"], 1, dtype=dtype, device=dev) if h0_init is None else h0_init.clone()
    was_training = net.training
    net.train(bn_train)
    snaps = {}
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), dropout_off(net):
        p = None
        for k in range(1, iters + 1):
            p, h0 = net(edge_index, static, h0)
            if k in snapshots:
                snaps[str(k)] = p.squeeze().detach().cpu().numpy()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    net.train(was_training)
    return dict(p=p.squeeze().detach().cpu().numpy(), snaps=snaps, wall_seconds=wall)


def finetune(net0, rand_block, inst_t, *, epochs, lr, penalty, from_scratch,
             hypers, seed):
    dev = rand_block.device; dtype = rand_block.dtype
    G, n = inst_t["G"], inst_t["n"]
    torch.manual_seed(seed)
    if from_scratch:
        net, _, opt = build_gnn(n, hypers, {"lr": lr}, dev, dtype)
        rb = torch.rand((n, hypers["dim_embedding"]), dtype=dtype, device=dev)
    else:
        net = copy.deepcopy(net0)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        rb = rand_block
    net.train(True)
    static = static_from_block(rb, G, dtype)
    edge_index = nx_to_pyg_edge_index(G, device=dev)
    Q_diag, Q_off = build_mis_qubo_components(G, torch_dtype=dtype, torch_device=dev)
    Q_train = make_mis_q_train(Q_diag, Q_off, p_min=penalty, p_max=penalty)
    denom = max(1, epochs - 1)
    h0 = torch.zeros(n, 1, dtype=dtype, device=dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(epochs):
        probs, h0_new = net(edge_index, static, h0.detach())
        loss = qubo_loss(probs.squeeze(), Q_train(ep / denom))
        h0 = h0_new
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        opt.step()
    with torch.no_grad(), dropout_off(net):
        net.train(False)
        p, _ = net(edge_index, static, h0.detach())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return dict(p=p.squeeze().detach().cpu().numpy(), wall_seconds=time.perf_counter() - t0)


def make_pairs(insts, all_pairs):
    by_tau = defaultdict(list)
    for nm, inst in insts.items():
        by_tau[inst["tau"]].append(nm)
    consecutive, extra = [], []
    for names in by_tau.values():
        names = sorted(names, key=lambda nm: insts[nm]["window"])
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                if j == i + 1:
                    consecutive.append((a, b))
                elif all_pairs:
                    extra.append((a, b))
    return consecutive, extra


def main_run(args):
    rows = load_screen(args.screen, names=_read_names(args.names_file))
    rows = [r for r in rows if r["source"] == "SP500"]
    R, ws = load_returns_bruni("SP500")
    insts = {}
    for r in rows:
        inst = build_instance(r, R, ws)
        if inst["kernel_strong"] >= args.min_strong_kernel:
            insts[inst["name"]] = inst
    consecutive, extra = make_pairs(insts, args.all_pairs)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line); done.add((r["source_name"], r["target_name"], r["seed"], r["mode"], r.get("identity", "keep")))
    hypers = dict(dim_embedding=args.dim_embedding, hidden_dim=args.hidden_dim, dropout=args.dropout, number_classes=1)
    identities = ["keep", "shuffle"]

    targets_of = defaultdict(list)
    for a, b in consecutive:
        targets_of[a].append((b, True))
    for a, b in extra:
        targets_of[a].append((b, False))

    with open(args.out, "a") as fout:
        def emit(src, tgt, seed, mode, identity, p_np, extra_fields, p_decode=None, do_ils=False):
            inst_t = insts[tgt]
            dec = decode(p_np if p_decode is None else p_decode, inst_t)
            rec = dict(source_name=src, target_name=tgt, source_window=insts[src]["window"], target_window=inst_t["window"],
                       window_gap=inst_t["window"] - insts[src]["window"], tau=inst_t["tau"], seed=seed, mode=mode,
                       identity=identity, n=inst_t["n"], alpha=inst_t["alpha"], kernel_strong=inst_t["kernel_strong"],
                       mean_degree=float(inst_t["deg"].mean()), raw=dec["raw"], viol=dec["viol"], value=dec["value"],
                       value_ls=dec["value_ls"], gap_ls=(inst_t["alpha"] - dec["value_ls"]) / inst_t["alpha"],
                       hit_ls=bool(dec["value_ls"] >= inst_t["alpha"] - 1e-9), greedy_ls=inst_t["greedy_ls"],
                       ils_ttt_screen=inst_t["ils_ttt"], **rank_metrics(p_np, inst_t), **extra_fields)
            if do_ils and args.ils_seconds > 0 and mode in ("zero%d" % max(args.zero_iters), "persist_p", "degree", "ft%d" % max(args.ft_epochs)):
                res = ils_arw(inst_t["G"], time_budget=args.ils_seconds, rng=np.random.default_rng(seed),
                              reference=inst_t["alpha"], start=set(dec["set_ls"]), stop_at_target=True)
                rec["ils_ttt_from_decode"] = res["time_to_target"]; rec["ils_hit_from_decode"] = bool(res["hit"])
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            return rec

        for src, tlist in targets_of.items():
            if not tlist:
                continue
            inst_s = insts[src]
            for seed in args.seeds:
                need = [(t, c) for t, c in tlist if any((src, t, seed, f"zero{min(args.zero_iters)}", idn) not in done
                                                        for idn in identities)]
                if not need:
                    continue
                print(f"{src} seed={seed} ...", end=" ", flush=True)
                S = train_source(inst_s, seed=seed, epochs=args.epochs, lr=args.lr, p_min=args.p_min, p_max=args.p_max,
                                 patience=args.patience, dim_embedding=args.dim_embedding, hidden_dim=args.hidden_dim,
                                 dropout=args.dropout, device=args.device)
                print(f"train {S['train_seconds']:.1f}s, epochs {S['epochs_run']}, final decode {S['final_decode']['value']:.0f} "
                      f"(+LS {S['final_decode']['value_ls']:.0f}) / alpha {inst_s['alpha']:.0f}", flush=True)
                src_fields = dict(source_train_seconds=S["train_seconds"], source_epochs_run=S["epochs_run"],
                                  source_final_value=S["final_decode"]["value"], source_alpha=inst_s["alpha"])
                if (src, src, seed, "self_final", "keep") not in done:
                    emit(src, src, seed, "self_final", "keep", S["final_p"], dict(src_fields, wall_seconds=0.0))
                gen = torch.Generator(device="cpu").manual_seed(10_000 + seed)
                for tgt, is_consecutive in need:
                    inst_t = insts[tgt]
                    if (src, tgt, seed, "degree", "keep") not in done:
                        gset = np.zeros(inst_t["n"]); gset[list(greedy_min_degree(inst_t["G"]))] = 1.0
                        emit(src, tgt, seed, "degree", "keep", -inst_t["deg"], dict(src_fields, wall_seconds=0.0), p_decode=gset, do_ils=is_consecutive)
                    if (src, tgt, seed, "persist_p", "keep") not in done:
                        emit(src, tgt, seed, "persist_p", "keep", S["final_p"], dict(src_fields, wall_seconds=0.0), do_ils=is_consecutive)
                    for idn in identities:
                        rb = S["rand_block"] if idn == "keep" else S["rand_block"][torch.randperm(inst_t["n"], generator=gen).to(S["rand_block"].device)]
                        if (src, tgt, seed, "zero1", idn) not in done:
                            K = max(args.zero_iters)
                            Z = zero_shot(S["net"], rb, inst_t, iters=K, h0_init=None, bn_train=False,
                                          snapshots=tuple(args.zero_iters))
                            for k in args.zero_iters:
                                pk = Z["snaps"].get(str(k))
                                if pk is None:
                                    continue
                                emit(src, tgt, seed, f"zero{k}", idn, pk, dict(src_fields, wall_seconds=Z["wall_seconds"] * k / K, iters=k),
                                     do_ils=is_consecutive and idn == "keep")
                        if idn == "keep" and (src, tgt, seed, f"zero{max(args.zero_iters)}_h0src", idn) not in done:
                            K = max(args.zero_iters)
                            Z = zero_shot(S["net"], rb, inst_t, iters=K, h0_init=S["h0_final"], bn_train=False,
                                          snapshots=(K,))
                            emit(src, tgt, seed, f"zero{K}_h0src", idn, Z["p"], dict(src_fields, wall_seconds=Z["wall_seconds"], iters=K))
                        if is_consecutive:
                            for E in args.ft_epochs:
                                if (src, tgt, seed, f"ft{E}", idn) not in done:
                                    F = finetune(S["net"], rb, inst_t, epochs=E, lr=args.lr, penalty=args.p_ft,
                                                 from_scratch=False, hypers=hypers, seed=seed)
                                    emit(src, tgt, seed, f"ft{E}", idn, F["p"], dict(src_fields, wall_seconds=F["wall_seconds"], epochs=E),
                                         do_ils=(idn == "keep"))
                                if idn == "keep" and (src, tgt, seed, f"scratch{E}", idn) not in done:
                                    F = finetune(S["net"], rb, inst_t, epochs=E, lr=args.lr, penalty=args.p_ft,
                                                 from_scratch=True, hypers=hypers, seed=seed)
                                    emit(src, tgt, seed, f"scratch{E}", idn, F["p"], dict(src_fields, wall_seconds=F["wall_seconds"], epochs=E))
                    print(f"    -> {tgt} (gap {inst_t['window'] - inst_s['window']}) done", flush=True)
                del S
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()


def _read_names(path):
    if not path:
        return None
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", nargs="+", default=[str(RESULTS / "screen_SP500.jsonl")])
    ap.add_argument("--names_file", default=str(RESULTS / "instances_full42.txt"))
    ap.add_argument("--min_strong_kernel", type=int, default=30)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--all_pairs", action="store_true", help="zero-shot on ALL ordered same-tau pairs (transfer-vs-gap curve)")
    ap.add_argument("--zero_iters", nargs="+", type=int, default=[1, 5, 10, 50, 200])
    ap.add_argument("--ft_epochs", nargs="+", type=int, default=[10, 50, 200])
    ap.add_argument("--p_ft", type=float, default=2.0, help="constant penalty for fine-tuning epochs and the scratch arm")
    ap.add_argument("--ils_seconds", type=float, default=2.0, help="ILS budget for time-to-target from the decoded set (0 = skip)")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1.4e-2)
    ap.add_argument("--p_min", type=float, default=0.01)
    ap.add_argument("--p_max", type=float, default=2.0)
    ap.add_argument("--dim_embedding", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(RESULTS / "gnn_transfer.jsonl"))
    args = ap.parse_args()
    main_run(args)


if __name__ == "__main__":
    main()
