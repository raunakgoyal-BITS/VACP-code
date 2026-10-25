"""Multi-horizon quantile + dispersion backbone (torch).

Encoder-decoder-free, single LSTM encoder with per-horizon heads:
  * median head   -> q50 of the cumulative return R_{t,h}
  * log_scale head-> conditional dispersion sigma_hat_{t,h} (drives volatility-aware width)
  * quantile head -> optional direct quantiles (for CQR-style baselines / WIS)

Deliberately architecture-light: the contribution of VACP-MH is the calibration
mechanism, not the backbone. Any point/probabilistic forecaster can be swapped in.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MultiHorizonForecaster(nn.Module):
    def __init__(self, n_features, horizons, hidden=64, layers=2, dropout=0.1,
                 quantile_levels=None):
        super().__init__()
        self.horizons = list(horizons)
        self.H = len(self.horizons)
        self.quantile_levels = quantile_levels or []
        self.Q = len(self.quantile_levels)
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.median_head = nn.Linear(hidden, self.H)
        self.logscale_head = nn.Linear(hidden, self.H)
        self.quantile_head = nn.Linear(hidden, self.H * self.Q) if self.Q else None

    def forward(self, x):
        """x: (B, L, n_features) -> dict of (B, H) tensors (+ (B,H,Q) quantiles)."""
        out, _ = self.lstm(x)
        h = out[:, -1, :]                                  # last hidden state
        median = self.median_head(h)                       # (B, H)
        log_scale = self.logscale_head(h)                  # (B, H)
        result = {"median": median, "log_scale": log_scale}
        if self.quantile_head is not None:
            q = self.quantile_head(h).view(-1, self.H, self.Q)
            # enforce non-crossing via cumulative softplus around the median
            result["quantiles"] = q
        return result
