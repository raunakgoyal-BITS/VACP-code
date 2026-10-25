"""GARCH(1,1) with Student-t innovations (Bollerslev, 1986/1987), fitted by MLE (scipy).

Fitted on training-period increments only; the variance recursion is then filtered
through the whole series with frozen parameters, so the forecast for origin t uses data
up to t only.  h-step intervals use the closed-form cumulative variance forecast.
"""
from __future__ import annotations
import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln


def _filter(r, mu, omega, a, b):
    e = r - mu
    s2 = np.empty(len(r) + 1)
    s2[0] = np.var(e)
    for t in range(len(r)):
        s2[t + 1] = omega + a * e[t] ** 2 + b * s2[t]
    return s2          # s2[t] = conditional variance of r[t]; s2[T] = one-step-ahead


def _nll(theta, r):
    mu, lw, la, lb, lnu = theta
    omega, a, b = np.exp(lw), 1 / (1 + np.exp(-la)), 1 / (1 + np.exp(-lb))
    nu = 2.05 + np.exp(lnu)
    if a + b >= 0.999:
        return 1e10
    s2 = _filter(r, mu, omega, a, b)[:-1]
    z2 = (r - mu) ** 2 / (s2 * (nu - 2) / nu)
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
          - 0.5 * np.log(s2 * (nu - 2) / nu) - (nu + 1) / 2 * np.log1p(z2 / nu))
    return -ll.sum()


class GarchT:
    def fit(self, r_train):
        r = np.asarray(r_train, float)
        v = r.var()
        x0 = np.array([r.mean(), np.log(0.05 * v), -2.5, 2.0, np.log(4.0)])
        res = optimize.minimize(_nll, x0, args=(r,), method="Nelder-Mead",
                                options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
        mu, lw, la, lb, lnu = res.x
        self.mu, self.omega = mu, np.exp(lw)
        self.a, self.b = 1 / (1 + np.exp(-la)), 1 / (1 + np.exp(-lb))
        self.nu = 2.05 + np.exp(lnu)
        return self

    def params(self):
        return dict(mu=self.mu, omega=self.omega, alpha=self.a, beta=self.b, nu=self.nu)

    def intervals(self, r_full, t0, horizons, level):
        """r_full: increments of the full series (r[t] = log s_t/s_{t-1}).
        Returns {h: (lo, hi)} in cumulative log-increment space for origins t0."""
        s2 = _filter(r_full, self.mu, self.omega, self.a, self.b)
        nxt = s2[t0 + 1]                       # variance of r[t0+1], known at t0
        p = self.a + self.b
        lr = self.omega / (1 - p)
        out = {}
        for h in horizons:
            V = lr * h + (nxt - lr) * (1 - p ** h) / (1 - p)   # sum_k E[s2_{t+k}]
            if h == 1:
                zq = stats.t.ppf(0.5 + level / 2, self.nu) * np.sqrt((self.nu - 2) / self.nu)
            else:
                zq = stats.norm.ppf(0.5 + level / 2)
            half = zq * np.sqrt(np.maximum(V, 1e-12))
            out[h] = (h * self.mu - half, h * self.mu + half)
        return out
