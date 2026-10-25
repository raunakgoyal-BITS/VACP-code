"""Data pipeline: log-increment targets, sliding windows, purged chronological splits.

Target for origin t and horizon h is the cumulative log-increment
    R_{t,h} = log(P_{t+h} / P_t),
and level-space bounds are recovered as P_t * exp(bound).  The input window for origin t
covers steps t-L+1 .. t (everything observable when the forecast is issued).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class WindowedData:
    X: np.ndarray          # (N, L, F)
    R: np.ndarray          # (N, H) cumulative log-increments
    P0: np.ndarray         # (N,) level at origin
    t0: np.ndarray         # (N,) integer index of the origin in the raw series
    horizons: list = field(default_factory=list)

    def subset(self, a, b):
        return WindowedData(self.X[a:b], self.R[a:b], self.P0[a:b], self.t0[a:b], self.horizons)

    def level_target(self):
        """(N, H) realised levels P_{t+h}."""
        return self.P0[:, None] * np.exp(self.R)


def make_returns(series):
    s = np.asarray(series, float)
    r = np.zeros_like(s)
    r[1:] = np.log(s[1:] / s[:-1])
    return r


def build_windows(series, features, seq_len, horizons, start=None):
    """features: (T, F) already transformed to stationary form (see experiments/features.py)."""
    s = np.asarray(series, float)
    T = len(s)
    H = list(horizons); hmax = max(H)
    X, R, P0, t0 = [], [], [], []
    first = seq_len - 1 if start is None else max(start, seq_len - 1)
    for t in range(first, T - hmax):
        X.append(features[t - seq_len + 1:t + 1])
        P0.append(s[t]); t0.append(t)
        R.append([np.log(s[t + h] / s[t]) for h in H])
    return WindowedData(np.asarray(X, np.float32), np.asarray(R, np.float32),
                        np.asarray(P0, float), np.asarray(t0, int), H)


def purged_split(data: WindowedData, train=0.70, val=0.15):
    """Chronological 70/15/15 split with an embargo of H_max windows after train and val,
    so that no training target overlaps the validation period and no validation target
    overlaps the test period (Lopez de Prado, 2018)."""
    N = len(data.X); hmax = max(data.horizons)
    n_tr, n_va = int(N * train), int(N * val)
    tr = data.subset(0, n_tr - hmax)
    va = data.subset(n_tr, n_tr + n_va - hmax)
    te = data.subset(n_tr + n_va, N)
    return tr, va, te


def standardize(train: WindowedData, *others: WindowedData):
    F = train.X.shape[-1]
    mu = train.X.reshape(-1, F).mean(0)
    sd = train.X.reshape(-1, F).std(0) + 1e-8

    def apply(d):
        return WindowedData(((d.X - mu) / sd).astype(np.float32), d.R, d.P0, d.t0, d.horizons)
    return (mu, sd), apply(train), *[apply(o) for o in others]
