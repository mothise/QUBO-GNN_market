"""Bruni et al. (2016) weekly return data and the 52/12/12 rolling protocol."""
import numpy as np
import scipy.io as sio

from shared.paths import DATA_DIR

BRUNI_DIR = DATA_DIR / "bruni"
MARKETS = ["DowJones", "FF49Industries", "FTSE100", "NASDAQ100", "SP500", "NASDAQComp"]


def load_market(market):
    path = BRUNI_DIR / market / f"{market}.mat"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - place the Bruni datasets under {BRUNI_DIR}")
    return np.asarray(sio.loadmat(path)["Assets_Returns"], dtype=np.float64)


def rolling_windows(T, in_sample=52, holding=12, step=12):
    out = []
    start = 0
    while start + in_sample + holding <= T:
        tr = np.arange(start, start + in_sample)
        te = np.arange(start + in_sample, start + in_sample + holding)
        out.append((tr, te))
        start += step
    return out
