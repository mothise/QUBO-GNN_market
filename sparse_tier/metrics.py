"""
Ranking metrics for the sparse tier.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from shared.metrics import auc_score


def residualise(score, covariate):
    rs, rc = rankdata(score), rankdata(covariate)
    if np.std(rc) < 1e-12:
        return rs
    beta = np.cov(rs, rc, bias=True)[0, 1] / np.var(rc)
    return rs - beta * rc


def auc_stratified(score, positives, strata_of, n_strata=10):
    pos = positives.astype(bool)
    if pos.sum() == 0 or (~pos).sum() == 0 or len(score) < 20:
        return float("nan")
    r = rankdata(strata_of, method="average")
    try:
        bins = pd.qcut(r, q=min(n_strata, max(2, len(np.unique(r)))), labels=False, duplicates="drop")
    except ValueError:
        return float("nan")
    num = den = 0.0
    for b in np.unique(bins):
        m = bins == b
        n1, n0 = pos[m].sum(), (~pos[m]).sum()
        if n1 and n0:
            num += auc_score(score[m], positives[m]) * n1 * n0
            den += n1 * n0
    return float(num / den) if den else float("nan")
