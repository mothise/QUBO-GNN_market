import time

import numpy as np
import torch
import torch.nn as nn

from shared.gnn import build_gnn, build_static_inputs
from shared.solver_family import PlateauStop, build_family, cra_gamma, cra_phi


def decode_topk(probs, K):
    idx = torch.topk(probs, K).indices.cpu().numpy()
    x = np.zeros(probs.shape[0], dtype=np.float64)
    x[idx] = 1.0
    return x


def _readout(net, edge_index, static_inputs, h0_in):
    saved = net.dropout_frac
    net.dropout_frac = 0.0
    try:
        p, _ = net(edge_index, static_inputs, h0_in)
    finally:
        net.train()
        net.dropout_frac = saved
    return p


def run_gnn_factored(train_loss_fn, feasible_obj_fn, K, edge_index, nx_graph, *,
                     decode_fn=None, dim_embedding=10, hidden_dim=50, number_classes=1,
                     dropout=0.5, lr=1.4e-2, number_epochs=10_000, window=500, window_tol=1e-7,
                     log_every=1000, seed=0, device="cpu", dtype=torch.float32,
                     decode_mode="dropout",
                     model="qrf", conv="gcn", family_overrides=None, init_p=None, embed_input=False):
    if model != "qrf" and (init_p is not None or embed_input):
        raise ValueError("init_p and embed_input are for the qrf network only")
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_nodes = nx_graph.number_of_nodes()
    dev = torch.device(device)
    if model == "qrf":
        gnn_hypers = {"dim_embedding": dim_embedding, "hidden_dim": hidden_dim,
                      "dropout": dropout, "number_classes": number_classes}
        net, embed, optimizer = build_gnn(n_nodes, gnn_hypers, {"lr": lr}, dev, dtype)
        static_inputs = build_static_inputs(n_nodes, dim_embedding, nx_graph, dev, dtype)
        if init_p is not None:
            bias = next(q for name, q in net.layers[-1].named_parameters() if name.endswith("bias"))
            with torch.no_grad():
                bias.fill_(float(np.log(init_p / (1.0 - init_p))))
        fam, stopper = None, None
    else:
        net, optimizer, fam = build_family(model, n_nodes, dev, dtype, conv=conv)
        if family_overrides:
            fam.update(family_overrides)
        static_inputs = None
        stopper = PlateauStop(fam["tol"], fam["patience"])
    h0 = torch.zeros(n_nodes, number_classes, dtype=dtype, device=dev)

    best_x = None
    best_obj = float("inf")
    best_epoch = -1
    train_window = []
    diag = {"sum_p": [], "thr_card": [], "epochs_run": 0,
            "decode_mode": decode_mode,
            "model_settings": fam, "stop_reason": "epochs"}
    denom = max(1, number_epochs - 1)

    t0 = time.time()
    for epoch in range(number_epochs):
        frac = epoch / denom
        h0_in = h0.detach()
        x_in = torch.cat([embed.weight, static_inputs[:, dim_embedding:]], 1) if embed_input else static_inputs
        probs, h0 = net(edge_index, x_in, h0_in)
        probs = probs.squeeze(-1)

        loss = train_loss_fn(probs, frac)
        if model == "cra":
            loss = loss + cra_gamma(fam, epoch) * cra_phi(probs, fam["alpha"])
        loss_val = float(loss.detach().item())

        with torch.no_grad():
            if decode_mode == "dropout":
                p = probs.detach()
            else:
                p = _readout(net, edge_index, x_in, h0_in).squeeze(-1)
            x_topk = decode_fn(p) if decode_fn is not None else decode_topk(p, K)
            obj = feasible_obj_fn(x_topk)
            if obj < best_obj:
                best_obj, best_x, best_epoch = obj, x_topk, epoch
            if epoch % 50 == 0:
                diag["sum_p"].append(float(p.sum().item()))
                diag["thr_card"].append(int((p >= 0.5).sum().item()))

        if log_every and epoch % log_every == 0:
            print(f"  ep={epoch:6d} loss={loss_val:+.6e} sum(p)={float(p.sum()):7.2f} "
                  f"|thr|={int((p >= 0.5).sum())} topK_obj={obj:+.6e} "
                  f"(best {best_obj:+.6e} @ {best_epoch})")

        if stopper is None:
            train_window.append(loss_val)
            if len(train_window) > window:
                train_window.pop(0)
            if window > 0 and len(train_window) == window \
                    and max(train_window) - min(train_window) < window_tol:
                diag["stop_reason"] = "loss_window"
                break
        elif stopper.step(loss_val):
            diag["stop_reason"] = "loss_plateau"
            break

        optimizer.zero_grad()
        loss.backward()
        if model == "qrf":
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()

    diag["epochs_run"] = epoch + 1
    diag["final_train_loss"] = loss_val
    with torch.no_grad():
        diag["final_loss_readout"] = float(train_loss_fn(p, frac).item())
    diag["final_p"] = [float(v) for v in p.detach().cpu().numpy()]
    if model == "cra":
        diag["gamma_at_stop"] = float(cra_gamma(fam, epoch))
        diag["phi_final"] = float(cra_phi(p, fam["alpha"]).item())
    diag["final_probs_sorted"] = np.sort(p.detach().cpu().numpy())[::-1]
    diag["seconds"] = time.time() - t0
    return best_x, best_obj, best_epoch, diag
