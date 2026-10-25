"""VACP-MH: Volatility-Aware Conformalised Prediction for Multi-Horizon intervals.

Pipeline
--------
1. Train the multi-horizon backbone (median q50 + dispersion sigma_hat) in RETURN space,
   with a Student-t NLL (heavy tails) [+ optional pinball on quantile heads].
2. Warm-start one OnlineConformal calibrator PER (horizon, coverage level) from validation
   residuals, normalised by sigma_hat  ->  volatility-aware radius.
3. At test time, roll forward: emit interval q50 +/- Q*sigma_hat, observe truth, update the
   online calibrator (ACI/PID). Reconstruct price-level intervals by exponentiation.

The conformal layer has (essentially) no learned parameters and runs at inference, so it
adds negligible cost -- deployment cost matches a plain point/dispersion forecaster.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import MultiHorizonForecaster
from .losses import student_t_nll, multi_horizon_pinball
from .conformal import OnlineConformal


class VACP_MH:
    def __init__(self, horizons=(1, 3, 5, 10, 20), coverage_levels=(0.80, 0.90, 0.95),
                 seq_len=60, hidden=64, layers=2, df=4.0,
                 conformal_method="pid", gamma=0.03, window=250, normalized=True,
                 ki=0.01, kd=0.0, device=None):
        self.horizons = list(horizons)
        self.coverage_levels = list(coverage_levels)
        self.seq_len = seq_len
        self.hidden, self.layers, self.df = hidden, layers, df
        self.cfg_conf = dict(method=conformal_method, gamma=gamma, window=window,
                             normalized=normalized, ki=ki, kd=kd)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.calibrators = {}  # {(h_index, level): OnlineConformal}

    # ---- training -------------------------------------------------------
    def fit(self, train, val, epochs=40, lr=1e-3, batch=32, patience=8, verbose=True):
        F = train.X.shape[-1]
        self.model = MultiHorizonForecaster(F, self.horizons, self.hidden, self.layers).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

        def loader(d, shuffle):
            ds = TensorDataset(torch.tensor(d.X), torch.tensor(d.R))
            return DataLoader(ds, batch_size=batch, shuffle=shuffle)

        tr, va = loader(train, True), loader(val, False)
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            self.model.train()
            for xb, rb in tr:
                xb, rb = xb.to(self.device), rb.to(self.device)
                out = self.model(xb)
                loss = student_t_nll(out["median"], out["log_scale"], rb, df=self.df)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
            vloss = self._val_loss(va)
            sched.step(vloss)
            if vloss < best - 1e-6:
                best, best_state, bad = vloss, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                bad += 1
            if verbose:
                print(f"epoch {ep+1:03d}  val_nll={vloss:.5f}  best={best:.5f}")
            if bad >= patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._calibrate(val)
        return self

    def _val_loss(self, va):
        self.model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for xb, rb in va:
                xb, rb = xb.to(self.device), rb.to(self.device)
                out = self.model(xb)
                l = student_t_nll(out["median"], out["log_scale"], rb, df=self.df)
                tot += l.item() * len(xb); n += len(xb)
        return tot / max(n, 1)

    def _predict_raw(self, d):
        """Return (median (N,H), sigma (N,H)) in return space."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(d.X).to(self.device))
        med = out["median"].cpu().numpy()
        sig = np.exp(out["log_scale"].cpu().numpy())
        return med, sig

    # ---- calibration ----------------------------------------------------
    def _calibrate(self, val):
        med, sig = self._predict_raw(val)
        resid = np.abs(val.R - med)                      # (N, H)
        for hi in range(len(self.horizons)):
            for lvl in self.coverage_levels:
                c = OnlineConformal(alpha_target=1 - lvl, **self.cfg_conf)
                c.warm_start(resid[:, hi], sig[:, hi])
                self.calibrators[(hi, lvl)] = c

    # ---- prediction (online rollout) -----------------------------------
    def predict_intervals(self, test, level, update_online=True):
        """Return dict with per-horizon return-space and price-space intervals.

        Rolls through the test set in order, updating the online calibrator after each
        observation (walk-forward). Set update_online=False for a frozen (static) run.
        """
        med, sig = self._predict_raw(test)
        N, H = med.shape
        out = {"return": {}, "price": {}, "picp_online": {}}
        for hi, h in enumerate(self.horizons):
            c = self.calibrators[(hi, level)]
            lo_r = np.empty(N); hi_r = np.empty(N)
            for t in range(N):
                l, u = c.interval(med[t, hi], sig[t, hi])
                lo_r[t], hi_r[t] = l, u
                if update_online:
                    c.update(test.R[t, hi], med[t, hi], sig[t, hi])
            out["return"][h] = (lo_r, hi_r)
            out["price"][h] = (test.P0 * np.exp(lo_r), test.P0 * np.exp(hi_r))
        return out, med, sig
