"""Online and split conformal calibration layers (numpy).

All calibrators work on ONE horizon h and ONE nominal level 1-alpha, in the model's
target space (log-increment by default).  Scores are

    s_t = |y_t - m_t|                       (plain)
    s_t = |y_t - m_t| / max(sigma_t, c)     (dispersion-normalised, Lei et al. 2018)

where c is a data-driven floor (a low quantile of the calibration-set sigmas).

Delayed feedback
----------------
For horizon h the outcome of the interval issued at origin t is only known at t+h.
`rollout` therefore applies the feedback of origin t-h just before issuing the interval
for origin t.  With h = 1 this is the usual online protocol; for h > 1 it removes the
look-ahead that a naive "update immediately" loop would introduce.

Controllers
-----------
* "aci"  : adaptive conformal inference (Gibbs & Candes, 2021).  The working miscoverage
           alpha_t is updated, and the radius is the (1-alpha_t) empirical quantile of a
           sliding window of scores.
* "qt"   : quantile tracking (P-control on the radius, Angelopoulos et al., 2023).
* "pi"   : quantile tracking + a bounded (anti-windup) integral term.  This is the
           controller used by the proposed method.

eta and ki are relative to the warm-start radius q0 (step = eta*q0 per unit error), so the
same values mean the same thing for plain and normalised scores and for every dataset.
Gains are scaled with the horizon, eta_h = eta / h**kappa, because the h-step errors of
consecutive origins overlap and are strongly autocorrelated: a single shock produces up
to h consecutive misses.
"""
from __future__ import annotations
from collections import deque
import numpy as np


def _conformal_quantile(scores, level):
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, q, method="higher"))


class OnlineConformal:
    def __init__(self, alpha=0.10, method="pi", normalized=True, eta=0.05, ki=0.01,
                 i_bound=20.0, gamma=0.005, window=500, sigma_floor_q=0.10,
                 h=1, kappa=0.5):
        assert 0 < alpha < 1 and method in ("aci", "qt", "pi")
        self.alpha, self.method, self.normalized = alpha, method, normalized
        scale = float(h) ** kappa
        self.eta = eta / scale            # radius step (score units)
        self.ki = ki / scale              # integral gain
        self.i_bound = i_bound            # anti-windup clamp on the error integral
        self.gamma = gamma / scale        # ACI step on alpha_t
        self.window = window
        self.sigma_floor_q = sigma_floor_q
        self.sigma_floor = 0.0
        self.h = h
        # state
        self.q = 0.0
        self.q0 = 1.0
        self.E = 0.0
        self.alpha_t = alpha
        self.scores = deque(maxlen=window)

    # -- scores ------------------------------------------------------------
    def _sig(self, sigma):
        return np.maximum(np.asarray(sigma, float), self.sigma_floor)

    def score(self, y, m, sigma):
        r = np.abs(np.asarray(y, float) - np.asarray(m, float))
        return r / self._sig(sigma) if self.normalized else r

    def warm_start(self, y, m, sigma):
        """Initialise from a calibration stream (arrays, chronological)."""
        if self.normalized and self.sigma_floor_q > 0:
            s = np.asarray(sigma, float)
            self.sigma_floor = float(np.quantile(s[s > 0], self.sigma_floor_q))
        sc = self.score(y, m, sigma)
        self.q = _conformal_quantile(sc, 1 - self.alpha)
        self.q0 = max(self.q, 1e-12)      # gains are expressed relative to this radius
        for v in sc[-self.window:]:
            self.scores.append(float(v))
        return self

    # -- radius --------------------------------------------------------------
    def radius(self):
        if self.method == "aci":
            lvl = 1.0 - self.alpha_t
            if lvl >= 1.0:
                return float(max(self.scores))           # finite stand-in for +inf
            if lvl <= 0.0:
                return 0.0
            return _conformal_quantile(np.asarray(self.scores), lvl)
        r = self.q + (self.ki * self.q0 * self.E if self.method == "pi" else 0.0)
        return max(r, 0.0)

    def feedback(self, s, r_used):
        """Consume the realised score s of an origin whose outcome is now known.

        r_used is the radius that was actually issued for that origin, so err is the
        true miss indicator of the published interval."""
        err = 1.0 if s > r_used else 0.0
        if self.method == "aci":
            self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
            self.scores.append(float(s))
        else:
            self.q = max(self.q + self.eta * self.q0 * (err - self.alpha), 0.0)
            if self.method == "pi":
                self.E = float(np.clip(self.E + (err - self.alpha), -self.i_bound, self.i_bound))
        return err


def rollout(cal, y, m, sigma, track=False):
    """Walk-forward with delay-h feedback. Returns (lo, hi[, radius trace]) in model space.

    cal is an OnlineConformal (or SplitConformal) with cal.h set.  y, m, sigma: (N,).
    """
    N = len(m)
    lo = np.empty(N); hi = np.empty(N); R = np.empty(N)
    sig = cal._sig(sigma) if cal.normalized else np.ones(N)
    s_real = cal.score(y, m, sigma)
    online = isinstance(cal, OnlineConformal)
    for t in range(N):
        if online and t - cal.h >= 0:
            cal.feedback(s_real[t - cal.h], R[t - cal.h])
        r = cal.radius()
        R[t] = r
        lo[t] = m[t] - r * sig[t]
        hi[t] = m[t] + r * sig[t]
    return (lo, hi, R) if track else (lo, hi)


class SplitConformal:
    """Static split conformal; normalized=True gives locally weighted CP (Lei et al. 2018)."""

    def __init__(self, alpha=0.10, normalized=True, sigma_floor_q=0.10, h=1):
        self.alpha, self.normalized, self.sigma_floor_q, self.h = alpha, normalized, sigma_floor_q, h
        self.sigma_floor = 0.0
        self.Q = None

    _sig = OnlineConformal._sig
    score = OnlineConformal.score

    def warm_start(self, y, m, sigma):
        if self.normalized and self.sigma_floor_q > 0:
            s = np.asarray(sigma, float)
            self.sigma_floor = float(np.quantile(s[s > 0], self.sigma_floor_q))
        self.Q = _conformal_quantile(self.score(y, m, sigma), 1 - self.alpha)
        return self

    def radius(self):
        return self.Q
