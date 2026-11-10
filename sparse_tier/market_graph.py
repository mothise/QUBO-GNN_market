"""Market graphs for the sparse problem.

Vertices are assets; an edge joins two assets whose sample return correlation exceeds
tau.
"""
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from shared.bruni import load_market, rolling_windows

TAUS_DEFAULT = (0.3, 0.4, 0.5, 0.6, 0.7)


def load_returns_bruni(market):
    R = load_market(market)
    return R, rolling_windows(R.shape[0])


def correlation(R_window):
    X = R_window - R_window.mean(axis=0)
    s = X.std(axis=0)
    s[s == 0] = np.inf
    X = X / s
    C = (X.T @ X) / X.shape[0]
    np.fill_diagonal(C, 1.0)
    return C


def weighted_window(R_window, scheme):
    if scheme == "unit":
        return R_window, np.ones(R_window.shape[1])
    mu = R_window.mean(axis=0)
    kept = mu > 0
    return R_window[:, kept], mu[kept] / mu[kept].max()


def threshold_graph(C, tau, *, weights=None, edges="signed"):
    n = C.shape[0]
    iu = np.triu_indices(n, 1)
    mask = (np.abs(C[iu]) if edges == "abs" else C[iu]) > tau
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(iu[0][mask].tolist(), iu[1][mask].tolist()))
    nx.set_node_attributes(G, 1.0 if weights is None else dict(enumerate(weights.tolist())), "w")
    G.graph.update(builder="threshold", tau=float(tau), m_negative=int((C[iu] < -tau).sum()), **graph_stats(G))
    return G


def knn_graph(C, k):
    n = C.shape[0]
    k = int(min(max(k, 1), n - 1))
    A = np.abs(C).copy()
    np.fill_diagonal(A, -np.inf)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        idx = np.argpartition(-A[i], k - 1)[:k] if k < n - 1 else np.flatnonzero(np.arange(n) != i)
        for j in idx:
            G.add_edge(i, int(j))
    G.graph.update(builder="knn", k=int(k), **graph_stats(G))
    return G


def er_control(G, seed):
    n, m = G.number_of_nodes(), G.number_of_edges()
    H = nx.gnm_random_graph(n, m, seed=seed)
    nx.set_node_attributes(H, dict(G.nodes(data="w", default=1.0)), "w")
    H.graph.update(builder="er_control", control_of=G.graph.get("builder"), seed=int(seed), **graph_stats(H))
    return H


def rewired_control(G, seed, *, swaps_per_edge=3.0):
    n, m = G.number_of_nodes(), G.number_of_edges()
    H = nx.Graph(); H.add_nodes_from(range(n)); H.add_edges_from(G.edges())
    dens = m / (n * (n - 1) / 2) if n > 1 else 0.0
    use_complement = dens > 0.5
    W = nx.complement(H) if use_complement else H
    m_work = W.number_of_edges()
    complete = True
    nswap = int(max(1, swaps_per_edge * m_work)) if m_work >= 2 else 0
    if nswap:
        try:
            nx.double_edge_swap(W, nswap=nswap, max_tries=20 * nswap + 100, seed=seed)
        except nx.NetworkXAlgorithmError:
            complete = False
    if use_complement:
        H = nx.complement(W)
        H.add_nodes_from(range(n))
    nx.set_node_attributes(H, dict(G.nodes(data="w", default=1.0)), "w")
    H.graph.update(builder="rewired_control", control_of=G.graph.get("builder"), seed=int(seed),
                   swaps_requested=int(nswap), swaps_complete=bool(complete), swapped_complement=bool(use_complement),
                   **graph_stats(H))
    return H


def kernel_after_reductions(G):
    H = G.copy()
    forced, removed = set(), set()
    unit = all(abs(w - 1.0) < 1e-12 for _, w in G.nodes(data="w", default=1.0))
    changed = True
    while changed:
        changed = False
        for v in list(H.nodes):
            if v not in H:
                continue
            d = H.degree(v)
            if d == 0:
                forced.add(v); H.remove_node(v); changed = True
            elif d == 1 and unit:
                (u,) = list(H.neighbors(v))
                forced.add(v); removed.add(u); H.remove_nodes_from([v, u]); changed = True
    return forced, removed, set(H.nodes)


def kernel_strong(G, *, max_passes=10_000):
    unit = all(abs(w - 1.0) < 1e-12 for _, w in G.nodes(data="w", default=1.0))
    adj = {v: set(G.neighbors(v)) for v in G.nodes}
    members = {v: {v} for v in G.nodes}
    n_forced = n_folds = n_dom = 0
    next_id = max(G.nodes, default=-1) + 1

    def remove(v):
        for u in adj[v]:
            adj[u].discard(v)
        del adj[v]

    changed = True
    passes = 0
    while changed and passes < max_passes:
        changed = False; passes += 1
        for v in list(adj):
            if v not in adj:
                continue
            N = adj[v]
            if not N:
                n_forced += 1; remove(v); changed = True; continue
            if not unit:
                continue
            Nl = list(N)
            if all(Nl[j] in adj[Nl[i]] for i in range(len(Nl)) for j in range(i + 1, len(Nl))):
                n_forced += 1
                for u in Nl:
                    remove(u)
                remove(v); changed = True; continue
            if len(N) == 2:
                u, w = Nl
                if w not in adj[u]:
                    newN = (adj[u] | adj[w]) - {v, u, w}
                    vp = next_id; next_id += 1
                    members[vp] = members[u] | members[v] | members[w]
                    remove(u); remove(w); remove(v)
                    adj[vp] = set(newN)
                    for x in newN:
                        adj[x].add(vp)
                    n_folds += 1; changed = True; continue
            for u in Nl:
                if len(adj[u]) <= len(N) and (adj[u] - {v}) <= N:
                    n_dom += 1; remove(v); changed = True; break
    unfixed = set()
    for v in adj:
        unfixed |= members[v]
    return dict(kernel_size=len(adj), n_forced=n_forced, n_folds=n_folds, n_dominated=n_dom,
                unfixed=unfixed, reduced_adj=adj)


def graph_stats(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    degs = np.array([d for _, d in G.degree()]) if n else np.zeros(0)
    comps = list(nx.connected_components(G)) if n else []
    forced, _, kernel = kernel_after_reductions(G) if n else (set(), set(), set())
    return dict(
        n=int(n), m=int(m),
        density=float(m / (n * (n - 1) / 2)) if n > 1 else 0.0,
        mean_degree=float(degs.mean()) if n else 0.0,
        max_degree=int(degs.max()) if n else 0,
        isolated_frac=float((degs == 0).mean()) if n else 0.0,
        n_components=len(comps),
        largest_component=int(max((len(c) for c in comps), default=0)),
        kernel_size=int(len(kernel)),
        forced_in=int(len(forced)),
    )


@dataclass
class Instance:
    name: str
    source: str
    window: int
    tau: float
    G: nx.Graph
    w: np.ndarray
    C: np.ndarray | None = None
    R: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.G.number_of_nodes()


def bruni_instances(market, taus=TAUS_DEFAULT, *, every=1, weights="unit", edges="signed"):
    R, ws = load_returns_bruni(market)
    suffix = ("_return" if weights == "return" else "") + ("_abs" if edges == "abs" else "")
    for wi, (tr, _) in enumerate(ws):
        if wi % every:
            continue
        Rw, w = weighted_window(R[tr], weights)
        C = correlation(Rw)
        for tau in taus:
            G = threshold_graph(C, tau, weights=w, edges=edges)
            yield Instance(name=f"{market}_w{wi}_tau{tau}{suffix}", source=market, window=wi, tau=float(tau),
                           G=G, w=w, C=C, R=Rw, meta=dict(G.graph, n_assets_window=int(R.shape[1])))
