"""Interval / probabilistic forecasting metrics (numpy-only, no torch needed).

All functions operate on numpy arrays. Shapes:
  y      : (N,)              true values
  lower  : (N,)              lower interval bound
  upper  : (N,)              upper interval bound
  nominal: float in (0,1)    target coverage, e.g. 0.90  (miscoverage alpha = 1 - nominal)

The headline metric is the interval (Winkler) score: it penalises BOTH width and
non-coverage, so a method cannot win simply by producing very wide intervals -- which is
exactly the failure mode documented for DeepPIPE (MPIW ~3841 at PICP ~0.995).
"""
from __future__ import annotations
import numpy as np


def picp(y, lower, upper):
    """Prediction Interval Coverage Probability = fraction of y inside [lower, upper]."""
    y, lower, upper = map(np.asarray, (y, lower, upper))
    return float(np.mean((y >= lower) & (y <= upper)))


def mpiw(lower, upper):
    """Mean Prediction Interval Width."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    return float(np.mean(upper - lower))


def normalized_mpiw(lower, upper, y):
    """MPIW normalised by the range of y (comparable across datasets/scales)."""
    y = np.asarray(y)
    rng = np.ptp(y)
    return mpiw(lower, upper) / rng if rng > 0 else np.nan


def coverage_deviation(y, lower, upper, nominal):
    """|empirical coverage - nominal|. Lower is better."""
    return abs(picp(y, lower, upper) - nominal)


def interval_score(y, lower, upper, nominal):
    """Winkler / interval score for a central (nominal)-level interval. Lower is better.

    IS_alpha = (u - l) + (2/alpha)(l - y) 1{y<l} + (2/alpha)(y - u) 1{y>u},  alpha = 1 - nominal
    """
    y, lower, upper = map(np.asarray, (y, lower, upper))
    alpha = 1.0 - nominal
    width = upper - lower
    below = (lower - y) * (y < lower)
    above = (y - upper) * (y > upper)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def weighted_interval_score(y, quantile_preds, quantile_levels, median=None):
    """Weighted Interval Score (WIS) aggregating several central intervals + the median.

    quantile_preds : dict {q_level: array(N,)} for symmetric pairs (e.g. 0.05,0.95,0.1,0.9,...)
    quantile_levels: list of the LOWER quantile levels used (e.g. [0.05, 0.10, 0.25]); each
                     pairs with its upper (1 - q). nominal = 1 - 2*q.
    median         : array(N,) median forecast; if None uses quantile_preds[0.5] if present.
    Reference: Bracher et al. (2021), "Evaluating epidemic forecasts in an interval format".
    """
    y = np.asarray(y)
    if median is None:
        median = quantile_preds.get(0.5)
    if median is None:
        raise ValueError("Provide `median` or include 0.5 in quantile_preds.")
    median = np.asarray(median)
    K = len(quantile_levels)
    total = 0.5 * np.abs(y - median)
    for q in quantile_levels:
        lo = np.asarray(quantile_preds[q])
        hi = np.asarray(quantile_preds[round(1 - q, 10)])
        nominal = 1 - 2 * q
        alpha = 1 - nominal
        is_k = (hi - lo) + (2.0 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
        total = total + (alpha / 2.0) * is_k
    return float(np.mean(total / (K + 0.5)))


def pinball_loss(y, q_pred, q_level):
    """Pinball (quantile) loss at level q_level."""
    y, q_pred = np.asarray(y), np.asarray(q_pred)
    diff = y - q_pred
    return float(np.mean(np.maximum(q_level * diff, (q_level - 1) * diff)))


def crps_from_samples(y, samples):
    """CRPS estimated from Monte-Carlo samples of the predictive distribution.

    y       : (N,)
    samples : (N, M)  M samples per observation.
    CRPS = E|X - y| - 0.5 E|X - X'|  (empirical / energy form).
    """
    y = np.asarray(y)[:, None]
    samples = np.asarray(samples)
    term1 = np.mean(np.abs(samples - y), axis=1)
    # 0.5 E|X - X'| via mean pairwise abs diff (O(M^2); fine for modest M)
    M = samples.shape[1]
    diffs = np.abs(samples[:, :, None] - samples[:, None, :])
    term2 = 0.5 * diffs.reshape(samples.shape[0], -1).mean(axis=1) * (M * M) / (M * M)
    return float(np.mean(term1 - term2))


def summary(y, lower, upper, nominal):
    """Convenience: return the standard interval-metric dict for one (level, horizon)."""
    return {
        "PICP": picp(y, lower, upper),
        "MPIW": mpiw(lower, upper),
        "nMPIW": normalized_mpiw(lower, upper, y),
        "CovDev": coverage_deviation(y, lower, upper, nominal),
        "IntervalScore": interval_score(y, lower, upper, nominal),
    }
