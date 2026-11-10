"""QRF-GNN at a constant penalty on market graphs, logged through training, against a lower learning rate, no dropout and the penalty warm-up."""
import argparse
import json
import os
import time

import numpy as np

from shared.metrics import auc_score
from sparse_tier.market_graph import correlation, load_returns_bruni, threshold_graph
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "sparse_tier" / "results"

DEFAULT_NAMES = ["SP500_w32_tau0.4", "SP500_w44_tau0.5", "SP500_w36_tau0.4", "SP500_w12_tau0.4", "SP500_w20_tau0.4",
                 "SP500_w44_tau0.4", "SP500_w16_tau0.5", "SP500_w16_tau0.4", "SP500_w28_tau0.5", "SP500_w28_tau0.4"]
CHECKPOINTS = (5, 10, 30, 100, 300, 1000, 2000)


def configs(args):
    return {
        "base": dict(lr=args.lr, dropout=args.dropout, p_min=2.0, p_max=2.0, epochs=args.epochs, nominal=args.epochs),
        "lr_low": dict(lr=args.lr / 10, dropout=args.dropout, p_min=2.0, p_max=2.0, epochs=args.variant_epochs, nominal=args.variant_epochs),
        "nodrop": dict(lr=args.lr, dropout=0.0, p_min=2.0, p_max=2.0, epochs=args.variant_epochs, nominal=args.variant_epochs),
        "warmup": dict(lr=args.lr, dropout=args.dropout, p_min=0.01, p_max=2.0, epochs=args.warmup_epochs, nominal=3000),
    }


def probe_run(G, *, seed, epochs, nominal, lr, p_min, p_max, dropout, device, dim_embedding, hidden_dim, log_first, log_every):
    import torch
    import torch.nn as nn
    from shared.gnn import build_gnn, build_static_inputs, nx_to_pyg_edge_index, qubo_loss
    from shared.mis import build_mis_qubo_components, linear_penalty, make_mis_q_train

    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device); dtype = torch.float32
    n = G.number_of_nodes()
    Q_diag, Q_off = build_mis_qubo_components(G, torch_dtype=dtype, torch_device=dev)
    Q_train = make_mis_q_train(Q_diag, Q_off, p_min=p_min, p_max=p_max)
    edge_index = nx_to_pyg_edge_index(G, device=dev)
    hypers = dict(dim_embedding=dim_embedding, hidden_dim=hidden_dim, dropout=dropout, number_classes=1)
    net, _, opt = build_gnn(n, hypers, {"lr": lr}, dev, dtype)
    static = build_static_inputs(n, dim_embedding, G, dev, dtype)
    h0 = torch.zeros(n, 1, dtype=dtype, device=dev)

    edges = np.array(G.edges(), dtype=np.int64).reshape(-1, 2)
    deg = np.array([d for _, d in sorted(G.degree())], dtype=float)
    q1, q3 = np.quantile(deg, [0.25, 0.75])
    low, high = deg <= q1, deg >= q3
    log = dict(epoch=[], penalty=[], loss=[], mean_p=[], max_p=[], mean_logit=[], raw=[], share_up=[], gnorm_p=[], gnorm_w=[],
               mean_p_lowdeg=[], mean_p_highdeg=[])
    traj_raw = []
    denom = max(1, nominal - 1)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(epochs):
        Qt = Q_train(ep / denom)
        probs, h0_new = net(edge_index, static, h0.detach())
        probs = probs.squeeze()
        probs.retain_grad()
        loss = qubo_loss(probs, Qt)
        h0 = h0_new
        opt.zero_grad(); loss.backward()
        gw = float(nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2))
        p_np = probs.detach().cpu().numpy()
        raw = int((p_np >= 0.5).sum())
        traj_raw.append(raw)
        if ep < log_first or ep % log_every == 0 or ep == epochs - 1:
            g = probs.grad.detach().cpu().numpy()
            log["epoch"].append(ep); log["penalty"].append(float(linear_penalty(ep / denom, p_min=p_min, p_max=p_max)))
            log["loss"].append(float(loss.detach())); log["mean_p"].append(float(p_np.mean())); log["max_p"].append(float(p_np.max()))
            log["mean_logit"].append(float(h0_new.detach().mean())); log["raw"].append(raw)
            log["share_up"].append(float((g < 0).mean())); log["gnorm_p"].append(float(np.linalg.norm(g))); log["gnorm_w"].append(gw)
            log["mean_p_lowdeg"].append(float(p_np[low].mean())); log["mean_p_highdeg"].append(float(p_np[high].mean()))
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    with torch.no_grad():
        pf, _ = net(edge_index, static, h0.detach())
        final_p = pf.squeeze().detach().cpu().numpy()
    bits = final_p >= 0.5
    viol = int((bits[edges[:, 0]] & bits[edges[:, 1]]).sum()) if len(edges) else 0
    return dict(final_p=final_p, final_raw=int(bits.sum()), final_viol=viol, traj_raw=traj_raw, log=log, train_seconds=train_s)


def summarise(res):
    traj = res["traj_raw"]
    empties = [i for i, x in enumerate(traj) if x == 0]
    out = dict(final_raw=res["final_raw"], final_viol=res["final_viol"], final_max_p=float(np.max(res["final_p"])),
               first_empty_epoch=(empties[0] if empties else None), last_empty_epoch=(empties[-1] if empties else None),
               n_empty_epochs=len(empties), epochs_run=len(traj), train_seconds=res["train_seconds"], log=res["log"])
    for c in CHECKPOINTS:
        out[f"raw_at_{c}"] = int(traj[c]) if c < len(traj) else None
    return out


def main_run(args):
    rows = {}
    for p in args.screen:
        for line in open(p):
            r = json.loads(line)
            rows[r["name"]] = r
    names = [nm for nm in args.names if nm in rows]
    missing = [nm for nm in args.names if nm not in rows]
    if missing:
        print("not in the screen files:", missing)
    if args.max_instances:
        names = names[:args.max_instances]
    cfgs = {k: v for k, v in configs(args).items() if k in args.configs}
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["name"], r["config"], r["seed"]))
    R, ws = load_returns_bruni("SP500")
    t_start = time.time()
    with open(args.out, "a") as fout:
        for name in names:
            r = rows[name]
            G = threshold_graph(correlation(R[ws[r["window"]][0]]), r["tau"])
            if G.number_of_edges() != r["m"]:
                raise SystemExit(f"{name}: screen/instance mismatch")
            n = G.number_of_nodes()
            deg = np.array([d for _, d in sorted(G.degree())], dtype=float)
            y = np.zeros(n); y[r["exact_set"]] = 1
            for cname, c in cfgs.items():
                for seed in args.seeds:
                    if (name, cname, seed) in done:
                        continue
                    res = probe_run(G, seed=seed, epochs=c["epochs"], nominal=c["nominal"], lr=c["lr"], p_min=c["p_min"], p_max=c["p_max"],
                                    dropout=c["dropout"], device=args.device, dim_embedding=args.dim_embedding, hidden_dim=args.hidden_dim,
                                    log_first=args.log_first, log_every=args.log_every)
                    row = dict(name=name, config=cname, seed=seed, n=n, mean_degree=float(deg.mean()), alpha=float(r["reference"]),
                               lr=c["lr"], dropout=c["dropout"], p_min=c["p_min"], p_max=c["p_max"], epochs=c["epochs"],
                               auc_final_p=auc_score(res["final_p"], y), auc_degree=auc_score(-deg, y), **summarise(res))
                    fout.write(json.dumps(row) + "\n"); fout.flush()
                print(f"{name:18s} {cname:7s} done, {time.time() - t_start:.0f}s elapsed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", nargs="+", default=[str(RESULTS / "screen_SP500.jsonl")])
    ap.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    ap.add_argument("--configs", nargs="+", default=["base", "lr_low", "nodrop", "warmup"], choices=["base", "lr_low", "nodrop", "warmup"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--epochs", type=int, default=3000, help="epochs of the base configuration (no early stop)")
    ap.add_argument("--variant_epochs", type=int, default=1500, help="epochs of lr_low and nodrop")
    ap.add_argument("--warmup_epochs", type=int, default=300, help="epochs of the warm-up contrast (schedule nominal 3000)")
    ap.add_argument("--lr", type=float, default=1.4e-2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--dim_embedding", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=50)
    ap.add_argument("--log_first", type=int, default=60, help="log every epoch up to this one")
    ap.add_argument("--log_every", type=int, default=50, help="then every this many epochs")
    ap.add_argument("--max_instances", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(RESULTS / "gnn_collapse_probe.jsonl"))
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main_run(args)


if __name__ == "__main__":
    main()
