"""
Certified maximum independent set.
"""
import time

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def greedy_edge_clique_cover(G):
    nodes = list(G.nodes)
    n = len(nodes)
    pos = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((n, n), dtype=bool)
    for u, v in G.edges():
        A[pos[u], pos[v]] = A[pos[v], pos[u]] = True
    deg = A.sum(1)
    order = np.argsort(-deg, kind="stable")
    covered = np.zeros((n, n), dtype=bool)
    cliques = []
    for u in order:
        nb = np.flatnonzero(A[u] & ~covered[u])
        for v in nb[np.argsort(-deg[nb], kind="stable")]:
            if covered[u, v]:
                continue
            Q = [u, v]
            cand = A[u] & A[v]
            while cand.any():
                idx = np.flatnonzero(cand)
                x = idx[np.argmax(deg[idx])]
                Q.append(x)
                cand &= A[x]
            Qa = np.asarray(Q)
            covered[np.ix_(Qa, Qa)] = True
            cliques.append([nodes[i] for i in Q])
    return cliques


def _constraint_matrix(H, idx):
    n = len(idx)
    E = np.array([(idx[u], idx[v]) for u, v in H.edges()])
    rows = np.repeat(np.arange(len(E)), 2)
    return coo_matrix((np.ones(2 * len(E)), (rows, E.ravel())), shape=(len(E), n)).tocsr(), len(E)


def _solve_component_highs(H, w, time_limit):
    nodes = list(H.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    A, nrows = _constraint_matrix(H, idx)
    c = -np.array([w[v] for v in nodes])
    res = milp(c, constraints=LinearConstraint(A, -np.inf, 1.0),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options=dict(time_limit=float(time_limit), disp=False, mip_rel_gap=0.0))
    trivial_bound = float(sum(max(w[v], 0.0) for v in nodes))
    if res.x is None:
        db = getattr(res, "mip_dual_bound", None)
        return None, -np.inf, (-float(db) if db is not None and np.isfinite(db) else trivial_bound), "failed", nrows
    x = np.rint(res.x).astype(int)
    S = {nodes[i] for i in range(n) if x[i] == 1}
    val = float(sum(w[v] for v in S))
    db = getattr(res, "mip_dual_bound", None)
    if res.status == 0:
        bound = val
    elif db is not None and np.isfinite(db):
        bound = max(val, -float(db))
    else:
        bound = trivial_bound
    status = "optimal" if res.status == 0 else ("time_limit" if res.status == 1 else f"status{res.status}")
    return S, val, bound, status, nrows


def _solve_component_cpsat(H, w, time_limit):
    from ortools.sat.python import cp_model

    nodes = list(H.nodes)
    cliques = greedy_edge_clique_cover(H)
    m = cp_model.CpModel()
    x = {v: m.NewBoolVar(f"x{v}") for v in nodes}
    for Q in cliques:
        m.AddAtMostOne(x[v] for v in Q)
    scale = 10 ** 6
    m.Maximize(sum(int(round(w[v] * scale)) * x[v] for v in nodes))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit)
    s.parameters.num_workers = 8
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, -np.inf, float(sum(max(w[v], 0.0) for v in nodes)), "failed", len(cliques)
    S = {v for v in nodes if s.Value(x[v])}
    val = float(sum(w[v] for v in S))
    bound = s.BestObjectiveBound() / scale
    status = "optimal" if st == cp_model.OPTIMAL else "time_limit"
    return S, val, bound, status, len(cliques)


def certify(G, *, time_limit=120.0, engine="highs"):
    solve = _solve_component_cpsat if engine == "cpsat" else _solve_component_highs
    w = {v: float(d.get("w", 1.0)) for v, d in G.nodes(data=True)}
    t0 = time.perf_counter()
    S, val, bound, rows, solved = set(), 0.0, 0.0, 0, 0
    status = "optimal"
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    deadline = t0 + time_limit
    for comp in comps:
        if len(comp) == 1:
            (v,) = comp
            if w[v] > 0:
                S.add(v); val += w[v]; bound += w[v]
            continue
        remaining = max(1.0, deadline - time.perf_counter())
        Sc, vc, bc, stc, nrows = solve(G.subgraph(comp).copy(), w, remaining)
        solved += 1; rows += nrows
        if Sc is None:
            status = "failed"; bound += bc; continue
        S |= Sc; val += vc; bound += bc
        if stc != "optimal":
            status = "time_limit" if status != "failed" else status
    return dict(status=status, value=float(val), bound=float(bound),
                gap=float((bound - val) / max(abs(bound), 1e-12)), set=sorted(int(v) for v in S),
                seconds=float(time.perf_counter() - t0), n_components_solved=int(solved),
                n_clique_rows=int(rows), engine=engine)


def brute_force(G):
    Gc = nx.complement(G)
    for v, d in G.nodes(data=True):
        Gc.nodes[v]["weight"] = int(round(1000 * float(d.get("w", 1.0))))
    cl, wt = nx.max_weight_clique(Gc, weight="weight")
    return set(cl), wt / 1000.0
