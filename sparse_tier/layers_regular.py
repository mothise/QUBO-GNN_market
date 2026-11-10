"""CRA-PI-GNN with four layer types on random regular graphs of degree 30, 100 and 200."""
import argparse

import torch

from sparse_tier.family_fidelity import run
from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "sparse_tier" / "results"
LAYERS = ("gcn", "gcn_self", "sage_noroot", "sage")
CELLS = ([(d, "cra", c) for d in (100, 200) for c in LAYERS]
         + [(30, "cra", c) for c in ("gcn_self", "sage_noroot")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(RESULTS / "layers_regular.jsonl"))
    args = ap.parse_args()
    run(CELLS, args.out, args.device)


if __name__ == "__main__":
    main()
