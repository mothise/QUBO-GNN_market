"""Instance lists from the screen at risk weights from 0.9 to 1."""
import json

import pandas as pd

from shared.paths import REPO_ROOT

RESULTS = REPO_ROOT / "dense_tier" / "results"
QS_FLAGGED = (0.99, 1.0)
QS_INDEPENDENT = (1.0,)
QS_LARGE = (0.99, 1.0)
MAX_SMALL_N = 83
PER_MARKET = 6
K = 10
EVERY = 5
COLUMNS = ["market", "window", "K", "q", "n", "hit_rate", "gap_greedy_swap", "f_best", "n_locopt", "delta",
           "q_eff", "flagged"]


def main():
    df = pd.DataFrame([json.loads(line) for line in open(RESULTS / "hard_end_screen.jsonl")])
    df = df[(df["part"] == "protocol") & (df["est"] == "lw")].copy()
    df["flagged"] = (df["hit_rate"] < 0.5) | (df["gap_greedy_swap"] > 1e-4)

    flagged = df[df["flagged"] & df["q"].isin(QS_FLAGGED) & (df["n"] <= MAX_SMALL_N)]
    flagged = flagged.sort_values(["hit_rate", "n_locopt", "window", "K"], ascending=[True, False, True, True])
    flagged = flagged.drop_duplicates(["market", "window", "q"]).groupby(["market", "q"], sort=True).head(PER_MARKET)
    flagged = flagged.sort_values(["n", "market", "q", "window", "K"])[COLUMNS]

    one_k = df[(df["K"] == K) & (df["window"] % EVERY == 0)]
    independent = one_k[one_k["q"].isin(QS_INDEPENDENT) & (one_k["n"] <= MAX_SMALL_N)]
    independent = independent.sort_values(["n", "market", "q", "window"])[COLUMNS]
    large = one_k[one_k["q"].isin(QS_LARGE) & (one_k["n"] > MAX_SMALL_N)]
    large = large.sort_values(["n", "market", "q", "window"])[COLUMNS]

    for name, d in (("flagged", flagged), ("independent", independent), ("large", large)):
        d.to_csv(RESULTS / f"hard_end_subset_{name}.csv", index=False)
        print(f"{name}: {len(d)} instances, {int(d['flagged'].sum())} flagged", flush=True)


if __name__ == "__main__":
    main()
