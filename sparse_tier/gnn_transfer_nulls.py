"""
Controls for the transfer result.
"""
import argparse
import copy
import json
import os
import time
from collections import defaultdict

import numpy as np

from shared.metrics import auc_score
from sparse_tier.market_graph import er_control, load_returns_bruni, rewired_control
from sparse_tier import gnn_transfer as T
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "sparse_tier" / "results"

KINDS = ("market", "rewired", "er", "random")


def control_instance(inst, kind, seed):
    G = er_control(inst["G"], seed=seed) if kind == "er" else rewired_control(inst["G"], seed=seed)
    n = inst["n"]
    deg = np.array([d for _, d in sorted(G.degree())], dtype=float)
    return dict(name=f"{inst['name']}__{kind}", window=inst["window"], tau=inst["tau"], G=G, w=np.ones(n), n=n,
                deg=deg, degi=deg.astype(np.int64), edges=np.array(G.edges(), dtype=np.int64).reshape(-1, 2))


def untrained_network(n, seed, args):
    import torch
    from shared.gnn import build_gnn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(args.device)
    hypers = dict(dim_embedding=args.dim_embedding, hidden_dim=args.hidden_dim, dropout=args.dropout, number_classes=1)
    net, _, _ = build_gnn(n, hypers, {"lr": args.lr}, dev, torch.float32)
    rand_block = torch.rand((n, args.dim_embedding), dtype=torch.float32, device=dev)
    return dict(net=net, rand_block=rand_block, train_seconds=0.0, epochs_run=0, final_p=None, final_decode=None)


def main_run(args):
    rows = T.load_screen(args.screen, names=T._read_names(args.names_file))
    rows = [r for r in rows if r["source"] == "SP500"]
    R, ws = load_returns_bruni("SP500")
    insts = {}
    for r in rows:
        inst = T.build_instance(r, R, ws)
        if inst["kernel_strong"] >= args.min_strong_kernel:
            insts[inst["name"]] = inst
    pairs, _ = T.make_pairs(insts, False)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    targets_of = defaultdict(list)
    for a, b in pairs:
        targets_of[a].append(b)
    K = max(args.zero_iters)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["source_name"], r["target_name"], r["seed"], r["mode"]))
    from scipy.stats import rankdata
    controls = {}
    t_start = time.time()
    with open(args.out, "a") as fout:
        def emit(src, tgt, seed, mode, kind, p_np, extra):
            inst_t = insts[tgt]
            dec = T.decode(p_np, inst_t)
            k = inst_t["kern"]
            rec = dict(source_name=src, target_name=tgt, source_window=insts[src]["window"], target_window=inst_t["window"],
                       tau=inst_t["tau"], seed=seed, mode=mode, kind=kind, n=inst_t["n"], alpha=inst_t["alpha"],
                       kernel_strong=inst_t["kernel_strong"], mean_degree=float(inst_t["deg"].mean()),
                       source_mean_degree=float(insts[src]["deg"].mean()), raw=dec["raw"], viol=dec["viol"], value=dec["value"],
                       value_ls=dec["value_ls"], gap_ls=(inst_t["alpha"] - dec["value_ls"]) / inst_t["alpha"],
                       **T.rank_metrics(p_np, inst_t), **extra)
            if k.sum() >= 10 and 0 < inst_t["y"][k].sum() < k.sum():
                rec["auc_comb_kernel"] = auc_score(rankdata(p_np[k]) + rankdata(-inst_t["deg"][k]), inst_t["y"][k])
            fout.write(json.dumps(rec) + "\n"); fout.flush()

        for src, tlist in targets_of.items():
            inst_s = insts[src]
            for seed in args.seeds:
                for kind in args.kinds:
                    need = [t for t in tlist if any((src, t, seed, f"{kind}_zero{k}{sfx}") not in done
                                                    for k in args.zero_iters for sfx in ("", "_bn"))]
                    if not need:
                        continue
                    if kind == "random":
                        S = untrained_network(inst_s["n"], seed, args)
                    else:
                        if kind == "market":
                            train_inst = inst_s
                        else:
                            if (src, kind) not in controls:
                                cseed = 20261109 + 7919 * int(inst_s["window"]) + int(round(100 * inst_s["tau"]))
                                controls[(src, kind)] = control_instance(inst_s, kind, cseed)
                            train_inst = controls[(src, kind)]
                        S = T.train_source(train_inst, seed=seed, epochs=args.epochs, lr=args.lr, p_min=args.p_min, p_max=args.p_max,
                                           patience=args.patience, dim_embedding=args.dim_embedding, hidden_dim=args.hidden_dim,
                                           dropout=args.dropout, device=args.device)
                    src_fields = dict(source_train_seconds=S["train_seconds"], source_epochs_run=S["epochs_run"])
                    if S.get("final_decode") is not None:
                        src_fields.update(source_final_raw=S["final_decode"]["raw"], source_final_viol=S["final_decode"]["viol"],
                                          source_final_value=S["final_decode"]["value"])
                    if kind == "market" and (src, src, seed, "market_home") not in done:
                        emit(src, src, seed, "market_home", kind, S["final_p"], dict(src_fields, wall_seconds=0.0, iters=0, bn_train=False))
                    for tgt in need:
                        for bn_train, sfx in ((False, ""), (True, "_bn")):
                            net = copy.deepcopy(S["net"]) if bn_train else S["net"]
                            Z = T.zero_shot(net, S["rand_block"], insts[tgt], iters=K, h0_init=None, bn_train=bn_train,
                                            snapshots=tuple(args.zero_iters))
                            for k in args.zero_iters:
                                mode = f"{kind}_zero{k}{sfx}"
                                pk = Z["snaps"].get(str(k))
                                if pk is None or (src, tgt, seed, mode) in done:
                                    continue
                                emit(src, tgt, seed, mode, kind, pk, dict(src_fields, wall_seconds=Z["wall_seconds"] * k / K, iters=k, bn_train=bn_train))
                    del S
                print(f"{src} seed {seed}: {len(tlist)} target(s) done, {time.time() - t_start:.0f}s elapsed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", nargs="+", default=[str(RESULTS / "screen_SP500.jsonl")])
    ap.add_argument("--names_file", default=str(RESULTS / "instances_full42.txt"))
    ap.add_argument("--min_strong_kernel", type=int, default=30)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--zero_iters", nargs="+", type=int, default=[1, 200])
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1.4e-2)
    ap.add_argument("--p_min", type=float, default=0.01)
    ap.add_argument("--p_max", type=float, default=2.0)
    ap.add_argument("--dim_embedding", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(RESULTS / "gnn_transfer_nulls.jsonl"))
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main_run(args)


if __name__ == "__main__":
    main()
