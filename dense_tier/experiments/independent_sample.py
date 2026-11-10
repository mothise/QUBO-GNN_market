"""
Instance list for the independent-window sample.
"""
import argparse
import json

import pandas as pd

from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default=str(RESULTS / "hardness_screen.jsonl"))
    ap.add_argument("--out", default=str(RESULTS / "independent_subset.csv"))
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--q", type=float, default=0.9)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max_n", type=int, default=148)
    args = ap.parse_args()

    df = pd.DataFrame([json.loads(l) for l in open(args.screen)])
    keep = ((df["n"] <= args.max_n) & (df["K"] == args.K) & (df["q"] == args.q)
            & (df["window"] % args.every == 0))
    sub = df[keep].sort_values(["n", "market", "window"])

    sub.to_csv(args.out, index=False)
    print(f"{len(sub)} instances (K={args.K}, q={args.q}, every {args.every}th window) -> {args.out}")


if __name__ == "__main__":
    main()
