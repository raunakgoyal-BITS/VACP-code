"""Interval-forecasting baselines / ablations sharing the VACP-MH backbone where possible.

Included here (runnable now):
  * NaiveVolInterval     -- random-walk return + rolling-vol Gaussian interval (strong, cheap)
  * BackboneSplitConformal-- backbone median/sigma + STATIC split conformal (exchangeability;
                             demonstrates the rolling-window collapse under shift)
  * BackbonePlainACI      -- backbone median + UN-normalised online conformal (ablation: shows
                             width stops tracking volatility when sigma-normalisation is removed)

Extend with external competitors (ACI/EnbPI/Conformal-PID/CopulaCPTS/CQR/DeepAR/TFT/GARCH)
by wrapping their outputs into the same (lower, upper) interface -- see README for the map.
The VACP-MH class in vacp.py IS the full proposed method (normalised + online).
"""
from __future__ import annotations
import numpy as np
from scipy import stats

from .conformal import SplitConformal, OnlineConformal


def _z(level):
    return stats.norm.ppf(0.5 + level / 2.0)


class NaiveVolInterval:
    """Random walk in return space: median return 0, width from trailing realised vol.

    For horizon h, sigma_h = sqrt(h) * rolling_std(returns). Gaussian interval on the
    cumulative return, reconstructed to price space. A deliberately strong, assumption-light
    benchmark (daily indices are near-random-walk).
    """

    def __init__(self, horizons, vol_window=20):
        self.horizons = list(horizons)
        self.vol_window = vol_window

    def predict_intervals(self, prices_test, returns_test, P0, level):
        z = _z(level)
        out = {"return": {}, "price": {}}
        rstd = np.array([returns_test[max(0, t - self.vol_window):t].std() if t > 1 else 0.0
                         for t in range(len(returns_test))])
        for h in self.horizons:
            sig_h = np.sqrt(h) * rstd
            lo_r, hi_r = -z * sig_h, z * sig_h
            out["return"][h] = (lo_r, hi_r)
            out["price"][h] = (P0 * np.exp(lo_r), P0 * np.exp(hi_r))
        return out


class BackboneStaticSplit:
    """Reuse a fitted VACP_MH backbone but calibrate with STATIC split conformal (no online)."""

    def __init__(self, vacp, normalized=True):
        self.vacp = vacp
        self.normalized = normalized
        self.split = {}  # (hi, level) -> SplitConformal

    def calibrate(self, val):
        med, sig = self.vacp._predict_raw(val)
        resid = np.abs(val.R - med)
        for hi in range(len(self.vacp.horizons)):
            for lvl in self.vacp.coverage_levels:
                sc = SplitConformal(alpha_target=1 - lvl, normalized=self.normalized)
                sc.fit(resid[:, hi], sig[:, hi])
                self.split[(hi, lvl)] = sc
        return self

    def predict_intervals(self, test, level):
        med, sig = self.vacp._predict_raw(test)
        N = med.shape[0]
        out = {"return": {}, "price": {}}
        for hi, h in enumerate(self.vacp.horizons):
            sc = self.split[(hi, level)]
            lo_r = np.empty(N); hi_r = np.empty(N)
            for t in range(N):
                l, u = sc.interval(med[t, hi], sig[t, hi])
                lo_r[t], hi_r[t] = l, u
            out["return"][h] = (lo_r, hi_r)
            out["price"][h] = (test.P0 * np.exp(lo_r), test.P0 * np.exp(hi_r))
        return out
