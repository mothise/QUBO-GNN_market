import networkx as nx
import numpy as np


def mst_edges(C):
    n = C.shape[0]
    D = np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    iu = np.triu_indices(n, 1)
    for i, j, w in zip(*iu, D[iu]):
        G.add_edge(int(i), int(j), weight=float(w))
    return list(nx.minimum_spanning_tree(G, weight="weight").edges())


def _build(n, edges, C, connect_mst):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in edges:
        i, j = int(i), int(j)
        if i != j:
            G.add_edge(i, j, weight=float(abs(C[i, j])))
    if connect_mst and n > 1 and not nx.is_connected(G):
        for i, j in mst_edges(C):
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(abs(C[i, j])))
    return G


def complete_graph(C):
    n = C.shape[0]
    iu = np.triu_indices(n, 1)
    return _build(n, zip(*iu), C, connect_mst=False)


def knn_graph(C, k=5):
    n = C.shape[0]
    A = np.abs(C).copy()
    np.fill_diagonal(A, -np.inf)
    edges = []
    for i in range(n):
        for j in np.argpartition(A[i], -k)[-k:]:
            edges.append((i, int(j)))
    return _build(n, edges, C, connect_mst=True)


GRAPH_BUILDERS = {
    "complete": lambda fm: complete_graph(fm.corr()),
    "knn": lambda fm, k=5: knn_graph(fm.corr(), k),
}
