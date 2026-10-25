"""Training objectives for the VACP-MH backbone (torch)."""
from __future__ import annotations
import math
import torch


def multi_horizon_pinball(q_pred, target, quantile_levels):
    """Pinball loss summed over quantile levels and horizons.

    q_pred : (B, H, Q)  predicted quantiles
    target : (B, H)     true cumulative returns per horizon
    quantile_levels : 1D tensor (Q,)
    """
    target = target.unsqueeze(-1)                       # (B, H, 1)
    ql = quantile_levels.view(1, 1, -1)                 # (1, 1, Q)
    diff = target - q_pred                              # (B, H, Q)
    loss = torch.maximum(ql * diff, (ql - 1.0) * diff)
    return loss.mean()


def student_t_nll(mu, log_scale, target, df=4.0):
    """Negative log-likelihood of a Student-t (heavy tails) per horizon.

    mu, log_scale, target : (B, H). df = degrees of freedom (smaller = heavier tails).
    This trains the conditional dispersion head sigma_hat = exp(log_scale) that VACP-MH
    uses to normalise conformal scores (volatility-aware width).
    """
    scale = torch.exp(log_scale).clamp_min(1e-6)
    z = (target - mu) / scale
    nll = (
        -math.lgamma((df + 1) / 2) + math.lgamma(df / 2)
        + 0.5 * math.log(df * math.pi)
        + log_scale
        + 0.5 * (df + 1) * torch.log1p(z * z / df)
    )
    return nll.mean()
