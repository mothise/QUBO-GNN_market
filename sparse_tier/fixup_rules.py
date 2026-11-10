"""The repair step of gnn_mis.py and the heaviest independent subset, applied to the outputs on the weighted graphs."""
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from shared.paths import REPO_ROOT
from sparse_tier.gnn_mis import fix_independent_set_fast
from sparse_tier.market_graph import correlation, load_returns_bruni, threshold_graph, weighted_window

RESULTS = REPO_ROOT / "sparse_tier" / "results"
SCREEN = RESULTS / "screen_return_abs.jsonl"
NAMES = RESULTS / "instances_full42_return_abs.txt"
OUT = RESULTS / "return_abs_fixup_rules.jsonl"
RUNS = [("cra", "single 2", "gnn_return_abs_cra_sage.jsonl"), ("cra", "single 1", "gnn_return_abs_p1_cra_sage.jsonl"),
        ("cra", "single 0.25", "gnn_return_abs_p025_cra_sage.jsonl"),
        ("cra", "per-edge", "gnn_return_abs_edgemin_cra_sage.jsonl"),
        ("qrf", "single 2", "gnn_return_abs_qrf6000.jsonl"), ("qrf", "single 1", "gnn_return_abs_p1_qrf6000.jsonl"),
        ("qrf", "single 0.25", "gnn_return_abs_p025_qrf6000.jsonl"),
        ("qrf", "per-edge", "gnn_return_abs_edgemin_qrf6000.jsonl")]


def violated(sel, edges):
    return edges[sel[edges[:, 0]] & sel[edges[:, 1]]]


def heaviest_subset(bits, edges, w):
    sel = bits.copy()
    e = violated(sel, edges)
    if not len(e):
        return sel
    verts = np.unique(e.ravel())
    col = np.searchsorted(verts, e.ravel())
    A = csr_matrix((np.ones(2 * len(e)), (np.repeat(np.arange(len(e)), 2), col)), shape=(len(e), len(verts)))
    res = milp(c=w[verts], constraints=LinearConstraint(A, lb=1, ub=np.inf), integrality=np.ones(len(verts)),
               bounds=Bounds(0, 1))
    sel[verts[np.round(res.x) > 0.5]] = False
    return sel


def main():
    screen = {}
    for line in open(SCREEN):
        r = json.loads(line)
        screen[r["name"]] = r
    names = [l.strip() for l in open(NAMES) if l.strip() and not l.startswith("#")]
    returns, graphs = {}, {}
    for name in names:
        s = screen[name]
        if s["source"] not in returns:
            returns[s["source"]] = load_returns_bruni(s["source"])
        R, ws = returns[s["source"]]
        Rw, w = weighted_window(R[ws[s["window"]][0]], s["weights"])
        G = threshold_graph(correlation(Rw), s["tau"], weights=w, edges=s["edges"])
        if (G.number_of_nodes(), G.number_of_edges()) != (s["n"], s["m"]):
            raise SystemExit("screen/instance mismatch")
        edges = np.array(G.edges(), dtype=np.int64).reshape(-1, 2)
        graphs[name] = (np.asarray(w, dtype=np.float64), edges, np.bincount(edges.ravel(), minlength=len(w)))

    with open(OUT, "w") as fout:
        for model, penalty, file_name in RUNS:
            acted = 0
            for line in open(RESULTS / file_name):
                r = json.loads(line)
                w, edges, deg = graphs[r["name"]]
                s = screen[r["name"]]
                bits = np.asarray(r["final_p_nodrop"], dtype=float) >= 0.5
                as_run = fix_independent_set_fast(bits, edges, deg)
                weight_aware = heaviest_subset(bits, edges, w)
                removed = bits & ~as_run
                row = dict(model=model, penalty=penalty, name=r["name"], seed=r["seed"], source=s["source"],
                           window=s["window"], mean_degree=s["mean_degree"], reference=s["reference"],
                           selected=int(bits.sum()), selected_weight=float(w[bits].sum()),
                           violated_edges=int(len(violated(bits, edges))), removed_as_run=int(removed.sum()),
                           removed_weight_aware=int((bits & ~weight_aware).sum()),
                           removed_to_selected_weight=float(w[removed].mean() / w[bits].mean()) if removed.any() else None,
                           value_as_run=float(w[as_run].sum()), value_weight_aware=float(w[weight_aware].sum()))
                fout.write(json.dumps(row) + "\n")
                acted += row["violated_edges"] > 0
            print(f"{model} {penalty}: the repair step acts in {acted} runs", flush=True)


if __name__ == "__main__":
    main()
