"""Build every interval method from saved backbone outputs (light, no training).

All functions return {(h, level): (lo, hi)} in LEVEL space on the requested stream.
Online calibrators are warm-started on a calibration stream and rolled forward with
delay-h feedback (see vacp_mh.conformal.rollout).
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy import stats

from vacp_mh.conformal import OnlineConformal, SplitConformal, rollout, _conformal_quantile
from vacp_mh.backbone import QR_LEVELS
from vacp_mh.garch import GarchT

# ----------------------------------------------------------------- tuning grids
GRID_PI = [dict(eta=e, ki=k, kappa=c) for e, k, c in
           itertools.product([0.005, 0.01, 0.02, 0.05, 0.1], [0.0, 0.005, 0.01, 0.02], [0.0, 0.5, 1.0])]
GRID_ACI = [dict(gamma=g, kappa=c) for g, c in
            itertools.product([0.001, 0.002, 0.005, 0.01, 0.02, 0.05], [0.0, 0.5, 1.0])]
VACP_DEFAULT = dict(method="pi", normalized=True, sigma_floor_q=0.10)


class Stream:
    """Model-space arrays for one stream (val or test) of one backbone."""

    def __init__(self, R, P0, m, sigma=None, q=None, target="logret", lvl_stats=None):
        self.R, self.P0, self.m, self.sigma, self.q = R, P0, m, sigma, q
        self.target, self.lvl_stats = target, lvl_stats
        self.y_level = P0[:, None] * np.exp(R)
        if target == "logret":
            self.y = R
        else:
            pm, ps = lvl_stats
            self.y = (self.y_level - pm) / ps

    def to_level(self, lo, hi, hi_idx):
        if self.target == "logret":
            return self.P0 * np.exp(lo), self.P0 * np.exp(hi)
        pm, ps = self.lvl_stats
        return lo * ps + pm, hi * ps + pm

    def sub(self, a, b):
        g = lambda x: None if x is None else x[a:b]
        s = Stream.__new__(Stream)
        s.R, s.P0, s.m, s.sigma, s.q = g(self.R), g(self.P0), g(self.m), g(self.sigma), g(self.q)
        s.target, s.lvl_stats, s.y_level, s.y = self.target, self.lvl_stats, g(self.y_level), g(self.y)
        return s


def conformal_method(cal_stream, ev_stream, horizons, levels, track=False, **kw):
    """Generic (online or split) conformal wrapper around the backbone median."""
    out, traces = {}, {}
    static = kw.pop("static", False)
    for hi, h in enumerate(horizons):
        for lvl in levels:
            if static:
                c = SplitConformal(alpha=1 - lvl, normalized=kw.get("normalized", True),
                                   sigma_floor_q=kw.get("sigma_floor_q", 0.10), h=h)
            else:
                c = OnlineConformal(alpha=1 - lvl, h=h, **kw)
            sc = cal_stream.sigma[:, hi] if cal_stream.sigma is not None else np.ones(len(cal_stream.m))
            se = ev_stream.sigma[:, hi] if ev_stream.sigma is not None else np.ones(len(ev_stream.m))
            c.warm_start(cal_stream.y[:, hi], cal_stream.m[:, hi], sc)
            lo, up, R = rollout(c, ev_stream.y[:, hi], ev_stream.m[:, hi], se, track=True)
            out[(h, lvl)] = ev_stream.to_level(lo, up, hi)
            traces[(h, lvl)] = R
    return (out, traces) if track else out


def prob_lstm(ev, horizons, levels, df=4.0):
    out = {}
    for hi, h in enumerate(horizons):
        for lvl in levels:
            z = stats.t.ppf(0.5 + lvl / 2, df) if np.isfinite(df) else stats.norm.ppf(0.5 + lvl / 2)
            out[(h, lvl)] = ev.to_level(ev.m[:, hi] - z * ev.sigma[:, hi],
                                        ev.m[:, hi] + z * ev.sigma[:, hi], hi)
    return out


def _qpair(lvl):
    a = round((1 - lvl) / 2, 4)
    return QR_LEVELS.index(a), QR_LEVELS.index(round(1 - a, 4))


def qr_lstm(ev, horizons, levels):
    out = {}
    for hi, h in enumerate(horizons):
        for lvl in levels:
            i, j = _qpair(lvl)
            out[(h, lvl)] = ev.to_level(ev.q[:, hi, i], ev.q[:, hi, j], hi)
    return out


def cqr(cal, ev, horizons, levels):
    """Conformalised quantile regression (Romano et al., 2019), split version."""
    out = {}
    for hi, h in enumerate(horizons):
        for lvl in levels:
            i, j = _qpair(lvl)
            E = np.maximum(cal.q[:, hi, i] - cal.y[:, hi], cal.y[:, hi] - cal.q[:, hi, j])
            Q = _conformal_quantile(E, lvl)
            out[(h, lvl)] = ev.to_level(ev.q[:, hi, i] - Q, ev.q[:, hi, j] + Q, hi)
    return out


def naive_vol(r_full, t0, P0, horizons, levels, window):
    """Random walk + trailing realised volatility of increments up to the origin."""
    import pandas as pd
    rv = pd.Series(r_full).rolling(window, min_periods=2).std().to_numpy()[t0]
    out = {}
    for h in horizons:
        for lvl in levels:
            half = stats.norm.ppf(0.5 + lvl / 2) * np.sqrt(h) * rv
            out[(h, lvl)] = (P0 * np.exp(-half), P0 * np.exp(half))
    return out


def garch(r_full, t_train_end, t0, P0, horizons, levels):
    g = GarchT().fit(r_full[1:t_train_end + 1])
    out = {}
    for lvl in levels:
        iv = g.intervals(r_full, t0, horizons, lvl)
        for h in horizons:
            out[(h, lvl)] = (P0 * np.exp(iv[h][0]), P0 * np.exp(iv[h][1]))
    return out, g.params()


# ----------------------------------------------------------------- tuning on validation
def interval_score(y, lo, hi, lvl):
    a = 1 - lvl
    return (hi - lo) + 2 / a * ((lo - y) * (y < lo) + (y - hi) * (y > hi))


def tune(val_stream, horizons, levels, grid, ref, base_kw, max_covdev=0.025):
    """Select controller gains on the validation stream only.

    The validation stream is split in half: the first half warm-starts the calibrator,
    the second half is rolled forward.  Criterion: mean (over horizons and levels) of the
    interval score relative to the reference method `ref` (Naive-Vol on the same half),
    among configurations whose mean coverage deviation is <= max_covdev.
    """
    n = len(val_stream.m); half = n // 2
    cal, ev = val_stream.sub(0, half), val_stream.sub(half, n)
    rows = []
    for g in grid:
        out = conformal_method(cal, ev, horizons, levels, **{**base_kw, **g})
        rel, cd = [], []
        for hi, h in enumerate(horizons):
            for lvl in levels:
                y = ev.y_level[:, hi]; lo, up = out[(h, lvl)]
                rel.append(interval_score(y, lo, up, lvl).mean() / ref[(h, lvl)])
                cd.append(abs(np.mean((y >= lo) & (y <= up)) - lvl))
        rows.append(dict(**g, relIS=float(np.mean(rel)), covdev=float(np.mean(cd))))
    ok = [r for r in rows if r["covdev"] <= max_covdev]
    best = min(ok, key=lambda r: r["relIS"]) if ok else min(rows, key=lambda r: r["covdev"])
    return {k: best[k] for k in grid[0]}, rows
