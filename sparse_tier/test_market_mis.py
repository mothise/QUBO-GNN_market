import networkx as nx
import numpy as np

from sparse_tier.market_graph import (correlation, er_control, kernel_after_reductions,
                          threshold_graph)
from sparse_tier.mis_baselines import (greedy_min_degree, is_independent, local_search_arw,
                           repair_only, value)
from sparse_tier.mis_exact import brute_force, certify


def rand_graph(rng, n, p, weighted):
    G = nx.gnp_random_graph(n, p, seed=int(rng.integers(1 << 30)))
    w = rng.uniform(0.2, 2.0, n) if weighted else np.ones(n)
    nx.set_node_attributes(G, {i: float(w[i]) for i in range(n)}, "w")
    return G


def main():
    rng = np.random.default_rng(0)
    n_ok = 0
    for t in range(60):
        n = int(rng.integers(8, 26)); p = float(rng.uniform(0.05, 0.5)); weighted = bool(t % 2)
        G = rand_graph(rng, n, p, weighted)
        _, vb = brute_force(G)
        c = certify(G, time_limit=30)
        assert c["status"] == "optimal", c["status"]
        assert abs(c["value"] - vb) < 1e-3 * n, (c["value"], vb)
        assert is_independent(G, c["set"])
        g = greedy_min_degree(G); gl = local_search_arw(G, g)
        assert is_independent(G, g) and is_independent(G, gl)
        assert value(G, g) - 1e-12 <= value(G, gl) <= c["value"] + 1e-9
        r = repair_only(G, rng)
        assert is_independent(G, r) and value(G, r) <= c["value"] + 1e-9
        forced, _, kernel = kernel_after_reductions(G)
        if not weighted:
            ck = certify(G.subgraph(kernel).copy(), time_limit=30)
            assert abs(len(forced) + ck["value"] - c["value"]) < 1e-9, "kernel reduction lost optimality"
        n_ok += 1
    print(f"certify == brute force, baselines feasible/bounded, kernel exact: {n_ok}/60 graphs OK")

    P = nx.path_graph(3); nx.set_node_attributes(P, 1.0, "w")
    assert local_search_arw(P, {1}) == {0, 2}
    print("(1,2)-swap on P3 OK")

    R = rng.normal(size=(52, 40)); C = correlation(R)
    assert np.allclose(C, np.corrcoef(R.T), atol=1e-12)
    G = threshold_graph(C, 0.2)
    iu = np.triu_indices(40, 1)
    assert G.number_of_edges() == int((C[iu] > 0.2).sum())
    H = er_control(G, seed=3)
    assert (H.number_of_nodes(), H.number_of_edges()) == (G.number_of_nodes(), G.number_of_edges())
    print("threshold_graph / er_control OK")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
