"""PI-GNN (Schuetz et al., 2022) and CRA-PI-GNN (Ichikawa, 2024) at the settings of their papers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

MODELS = ("qrf", "pignn", "cra")
CONVS = ("gcn", "sage", "gcn_self", "sage_noroot")

SETTINGS = {
    "pignn": dict(optimizer="adam", lr=1e-4, weight_decay=0.0, dropout=0.0, max_epochs=100_000,
                  tol=1e-4, patience=1000, penalty=2.0),
    "cra": dict(optimizer="adamw", lr=1e-4, weight_decay=1e-2, dropout=0.0, max_epochs=50_000,
                tol=1e-5, patience=1000, penalty=2.0, gamma0=-20.0, eps=1e-3, alpha=2),
}


def embed_dims(model, n_nodes):
    if model == "pignn":
        root = n_nodes ** 0.5 if n_nodes >= 100_000 else n_nodes ** (1.0 / 3.0)
        d0 = int(root + 1e-9)
        d1 = int(d0 / 2)
    else:
        d0 = int(n_nodes ** 0.8)
        d1 = int(n_nodes ** 0.8 / 2)
    return max(d0, 1), max(d1, 1)


class TwoLayerGNN(nn.Module):
    def __init__(self, n_nodes, d0, d1, conv="gcn", dropout=0.0):
        super().__init__()
        self.dropout_frac = dropout
        self.embed = nn.Embedding(n_nodes, d0)
        if conv in ("gcn", "gcn_self"):
            loops = conv == "gcn_self"
            self.conv1 = GCNConv(d0, d1, add_self_loops=loops, cached=True)
            self.conv2 = GCNConv(d1, 1, add_self_loops=loops, cached=True)
        else:
            root = conv == "sage"
            self.conv1 = SAGEConv(d0, d1, aggr="mean", root_weight=root)
            self.conv2 = SAGEConv(d1, 1, aggr="mean", root_weight=root)

    def forward(self, edge_index, static=None, h0=None):
        h = self.conv1(self.embed.weight, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout_frac, training=self.training)
        logits = self.conv2(h, edge_index)
        return torch.sigmoid(logits), logits


def build_family(model, n_nodes, device, dtype, conv="gcn", lr=None, dropout=None):
    s = dict(SETTINGS[model])
    if lr is not None:
        s["lr"] = float(lr)
    if dropout is not None:
        s["dropout"] = float(dropout)
    d0, d1 = embed_dims(model, n_nodes)
    net = TwoLayerGNN(n_nodes, d0, d1, conv=conv, dropout=s["dropout"]).type(dtype).to(device)
    if s["optimizer"] == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    else:
        opt = torch.optim.Adam(net.parameters(), lr=s["lr"])
    s.update(model=model, conv=conv, d0=d0, d1=d1)
    return net, opt, s


def cra_phi(probs, alpha=2):
    return (1.0 - (2.0 * probs - 1.0) ** alpha).sum()


def cra_gamma(settings, epoch):
    return settings["gamma0"] + settings["eps"] * epoch


class PlateauStop:
    def __init__(self, tol, patience):
        self.tol = tol
        self.patience = patience
        self.prev = None
        self.count = 0

    def step(self, loss_value):
        if self.prev is not None and abs(loss_value - self.prev) <= self.tol:
            self.count += 1
        else:
            self.count = 0
        self.prev = loss_value
        return self.count >= self.patience
