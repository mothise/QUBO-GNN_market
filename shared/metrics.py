"""AUC of a score vector as a ranker for a 0/1 label vector (ties count 0.5)."""
import numpy as np
from scipy.stats import rankdata


def auc_score(score, positives):
    score = np.asarray(score, dtype=float)
    pos = np.asarray(positives, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
