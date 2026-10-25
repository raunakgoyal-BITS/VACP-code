"""Statistical validation utilities (numpy/scipy; statsmodels for data diagnostics)."""
from __future__ import annotations
import numpy as np
from scipy import stats


# ---------------------------------------------------------------- forecast comparison
def diebold_mariano(loss_a, loss_b, h=1):
    """DM test with HAC variance (Bartlett, lag h-1) and the Harvey-Leybourne-Newbold
    small-sample correction.  Negative statistic => A has the lower loss."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d); dbar = d.mean(); dc = d - dbar
    var = np.dot(dc, dc) / n
    for lag in range(1, h):
        var += 2 * (1 - lag / h) * np.dot(dc[lag:], dc[:-lag]) / n
    dm = dbar / np.sqrt(max(var, 1e-300) / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


# ---------------------------------------------------------------- bootstrap
def block_indices(n, block, n_boot, seed=0):
    """Moving-block bootstrap index matrix (n_boot, n) (Kunsch, 1989)."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return idx


def block_bootstrap_ci(x, block=20, n_boot=2000, alpha=0.05, seed=0, idx=None):
    x = np.asarray(x, float)
    if idx is None:
        idx = block_indices(len(x), block, n_boot, seed)
    means = x[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def t_ci(values, alpha=0.05):
    """Mean and t-based CI half-width across independent runs (seeds)."""
    v = np.asarray(values, float); n = len(v)
    if n < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(stats.t.ppf(1 - alpha / 2, n - 1) * v.std(ddof=1) / np.sqrt(n))


# ---------------------------------------------------------------- coverage backtests
def _ll_bern(k, n, p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return k * np.log(p) + (n - k) * np.log(1 - p)


def kupiec_pof(hits, alpha):
    """Kupiec (1995) proportion-of-failures LR test.  hits = 1 if y outside the interval."""
    hits = np.asarray(hits, int); n = len(hits); k = hits.sum()
    lr = -2 * (_ll_bern(k, n, alpha) - _ll_bern(k, n, k / n if n else 0))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen(hits, alpha):
    """Christoffersen (1998) independence and conditional-coverage LR tests.
    Returns dict(LR_ind, p_ind, LR_cc, p_cc)."""
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_ind = _ll_bern(n01, n00 + n01, p01) + _ll_bern(n11, n10 + n11, p11)
    ll_0 = _ll_bern(n01 + n11, n00 + n01 + n10 + n11, p)
    lr_ind = float(max(-2 * (ll_0 - ll_ind), 0.0))
    lr_pof, _ = kupiec_pof(h[1:], alpha)
    lr_cc = lr_pof + lr_ind
    return dict(LR_ind=lr_ind, p_ind=float(1 - stats.chi2.cdf(lr_ind, 1)),
                LR_cc=float(lr_cc), p_cc=float(1 - stats.chi2.cdf(lr_cc, 2)))


# ---------------------------------------------------------------- multiple comparison
def friedman_nemenyi(score_matrix, names, alpha=0.05):
    """Friedman test over blocks (rows) and Nemenyi critical difference (Demsar, 2006)."""
    S = np.asarray(score_matrix, float); D, M = S.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, S)
    stat, p = stats.friedmanchisquare(*[S[:, m] for m in range(M)])
    # q_alpha = studentized range quantile / sqrt(2)
    q = stats.studentized_range.ppf(1 - alpha, M, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(M * (M + 1) / (6.0 * D))
    return dict(friedman_stat=float(stat), friedman_p=float(p),
                avg_ranks=dict(zip(names, ranks.mean(0).tolist())), cd=float(cd), q=float(q))


# ---------------------------------------------------------------- data diagnostics
def series_diagnostics(r):
    """ADF (stationarity), Engle's ARCH-LM (heteroscedasticity), Ljung-Box on r^2."""
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
    r = np.asarray(r, float)
    adf = adfuller(r, autolag="AIC")
    arch = het_arch(r, nlags=10)
    lb = acorr_ljungbox(r ** 2, lags=[10], return_df=True)
    return dict(n=len(r), mean=float(r.mean()), std=float(r.std()),
                skew=float(stats.skew(r)), ex_kurt=float(stats.kurtosis(r)),
                adf_stat=float(adf[0]), adf_p=float(adf[1]),
                archlm_stat=float(arch[0]), archlm_p=float(arch[1]),
                lb_sq_stat=float(lb["lb_stat"].iloc[0]), lb_sq_p=float(lb["lb_pvalue"].iloc[0]))
