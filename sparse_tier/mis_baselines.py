"""
Classical baselines for maximum independent set.
"""
import time

import numpy as np


def _adj(G):
    return {v: set(G.neighbors(v)) for v in G.nodes}


def _w(G):
    return {v: float(d.get("w", 1.0)) for v, d in G.nodes(data=True)}


def is_independent(G, S):
    S = set(S)
    return not any(u in S and v in S for u, v in G.edges())


def value(G, S):
    w = _w(G)
    return float(sum(w[v] for v in S))


def greedy_min_degree(G):
    adj = _adj(G)
    w = _w(G)
    alive = set(G.nodes)
    deg = {v: len(adj[v]) for v in alive}
    S = set()
    while alive:
        v = max(alive, key=lambda x: (w[x] / (deg[x] + 1.0), -x))
        S.add(v)
        rem = (adj[v] & alive) | {v}
        for u in rem:
            alive.discard(u)
        for u in rem:
            for x in adj[u]:
                if x in alive:
                    deg[x] -= 1
    return S


def _ls_core(adj, w, nodes, S, *, max_passes=10_000):
    tight = {v: 0 for v in nodes}
    for v in S:
        for u in adj[v]:
            tight[u] += 1

    def add(v):
        S.add(v)
        for u in adj[v]:
            tight[u] += 1

    def remove(v):
        S.discard(v)
        for u in adj[v]:
            tight[u] -= 1

    for _ in range(max_passes):
        improved = False
        for v in nodes:
            if v not in S and tight[v] == 0 and w[v] > 1e-12:
                add(v); improved = True
        for x in list(S):
            if x not in S:
                continue
            one_tight = [u for u in adj[x] if tight[u] == 1 and u not in S]
            if not one_tight:
                continue
            done = False
            one_tight.sort(key=lambda u: -w[u])
            for i, u in enumerate(one_tight):
                for v2 in one_tight[i + 1:]:
                    if v2 not in adj[u] and w[u] + w[v2] > w[x] + 1e-12:
                        remove(x); add(u); add(v2)
                        improved = done = True
                        break
                if done:
                    break
            if done:
                continue
            u = one_tight[0]
            if w[u] > w[x] + 1e-12:
                remove(x); add(u); improved = True
        if not improved:
            break
    return S


def local_search_arw(G, S, *, max_passes=10_000):
    return _ls_core(_adj(G), _w(G), list(G.nodes), set(S), max_passes=max_passes)


def ils_arw(G, *, time_budget, rng, reference=None,
            start=None, tol=1e-9, stop_at_target=True):
    adj, w, nodes = _adj(G), _w(G), list(G.nodes)
    t0 = time.perf_counter()
    S = set(start) if start is not None else greedy_min_degree(G)
    S = _ls_core(adj, w, nodes, S)
    vS = sum(w[v] for v in S)
    best, best_val = set(S), vS
    hit_time = None
    if reference is not None and best_val >= reference - tol:
        hit_time = time.perf_counter() - t0
    iters = 0
    pos = {v: i for i, v in enumerate(nodes)}
    while (time.perf_counter() - t0) < time_budget and not (stop_at_target and hit_time is not None):
        iters += 1
        k = 1 if rng.random() < 0.5 else 1 + int(rng.geometric(0.5))
        T = set(S)
        out = [v for v in nodes if v not in T]
        if not out:
            break
        for v in rng.choice(out, size=min(k, len(out)), replace=False):
            v = nodes[pos[v]] if v in pos else int(v)
            for u in adj[v]:
                T.discard(u)
            T.add(v)
        T = _ls_core(adj, w, nodes, T)
        vT = sum(w[v] for v in T)
        if vT >= vS - tol or rng.random() < np.exp(-(vS - vT)):
            S, vS = T, vT
        if vT > best_val + tol:
            best, best_val = set(T), vT
            if reference is not None and best_val >= reference - tol and hit_time is None:
                hit_time = time.perf_counter() - t0
    return dict(best_set=best, best_value=float(best_val), seconds=float(time.perf_counter() - t0),
                iterations=int(iters), time_to_target=hit_time, hit=hit_time is not None)


def repair_only(G, rng, *, p_init=0.5,
                local_search=True):
    adj = _adj(G)
    nodes = list(G.nodes)
    n = len(nodes)
    x = rng.random(n) < p_init
    S = {nodes[i] for i in range(n) if x[i]}
    order = rng.permutation(n)
    for i in order:
        v = nodes[i]
        if v in S and any(u in S for u in adj[v]):
            S.discard(v)
    tight = {v: sum(1 for u in adj[v] if u in S) for v in nodes}
    for i in rng.permutation(n):
        v = nodes[i]
        if v not in S and tight[v] == 0:
            S.add(v)
            for u in adj[v]:
                tight[u] += 1
    if local_search:
        S = _ls_core(adj, _w(G), nodes, S)
    return S


def restart_hit_rate(G, reference, R, *, seed=0,
                     tol=1e-9, local_search=True):
    rng = np.random.default_rng(seed)
    vals = []
    t0 = time.perf_counter()
    for _ in range(R):
        S = repair_only(G, rng, local_search=local_search)
        vals.append(value(G, S))
    dt = (time.perf_counter() - t0) / max(1, R)
    vals = np.asarray(vals)
    return dict(hit_rate=float((vals >= reference - tol).mean()), best=float(vals.max()),
                mean=float(vals.mean()), min=float(vals.min()), seconds_per_restart=float(dt), R=int(R))


def greedy_plus_ls(G):
    S0 = greedy_min_degree(G)
    return S0, local_search_arw(G, S0)
