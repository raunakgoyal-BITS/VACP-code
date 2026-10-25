"""Backbone training for VACP-MH and its learned baselines/ablations (torch).

One LSTM encoder, per-horizon heads.  Supported objectives:
  * "student_t" : location m and log-scale for a Student-t likelihood (VACP-MH default)
  * "gaussian"  : the same with a Gaussian likelihood (ablation)
  * "pinball"   : direct quantile regression (QR-LSTM baseline and CQR)

Targets live in "model space": cumulative log-increments by default, or standardised
levels for the level-target ablation.
"""
from __future__ import annotations
import math, time, copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import MultiHorizonForecaster
from .losses import student_t_nll, multi_horizon_pinball

QR_LEVELS = [0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975]


def gaussian_nll(mu, log_scale, target):
    z = (target - mu) / torch.exp(log_scale).clamp_min(1e-6)
    return (log_scale + 0.5 * z * z + 0.5 * math.log(2 * math.pi)).mean()


class Backbone:
    def __init__(self, n_features, horizons, loss="student_t", df=4.0, hidden=64, layers=2,
                 dropout=0.1, lr=1e-3, batch=32, epochs=60, patience=8, weight_decay=1e-2,
                 seed=0, device="cpu"):
        self.loss_name, self.df = loss, df
        self.horizons = list(horizons)
        self.cfg = dict(lr=lr, batch=batch, epochs=epochs, patience=patience, wd=weight_decay)
        self.seed, self.device = seed, device
        torch.manual_seed(seed); np.random.seed(seed)
        ql = QR_LEVELS if loss == "pinball" else None
        self.model = MultiHorizonForecaster(n_features, horizons, hidden, layers, dropout,
                                            quantile_levels=ql).to(device)
        self.qlev = torch.tensor(QR_LEVELS) if ql else None
        self.history = []          # (epoch, train_loss, val_loss, seconds)
        self.train_seconds = 0.0

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def _loss(self, out, y):
        if self.loss_name == "student_t":
            return student_t_nll(out["median"], out["log_scale"], y, df=self.df)
        if self.loss_name == "gaussian":
            return gaussian_nll(out["median"], out["log_scale"], y)
        return multi_horizon_pinball(out["quantiles"], y, self.qlev.to(y.device))

    def fit(self, Xtr, Ytr, Xva, Yva, verbose=False):
        g = torch.Generator().manual_seed(self.seed)
        tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                        batch_size=self.cfg["batch"], shuffle=True, generator=g)
        Xva_t, Yva_t = torch.tensor(Xva).to(self.device), torch.tensor(Yva).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"],
                                weight_decay=self.cfg["wd"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        best, best_state, bad = np.inf, None, 0
        t_start = time.perf_counter()
        for ep in range(self.cfg["epochs"]):
            t_ep = time.perf_counter()
            self.model.train(); tot, n = 0.0, 0
            for xb, yb in tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = self._loss(self.model(xb), yb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                tot += loss.item() * len(xb); n += len(xb)
            self.model.eval()
            with torch.no_grad():
                vl = self._loss(self.model(Xva_t), Yva_t).item()
            sched.step(vl)
            self.history.append((ep + 1, tot / n, vl, time.perf_counter() - t_ep))
            if verbose:
                print(f"  ep{ep+1:03d} train={tot/n:.4f} val={vl:.4f}")
            if vl < best - 1e-6:
                best, best_state, bad = vl, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        self.train_seconds = time.perf_counter() - t_start
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, X):
        """Return dict: m (N,H), sigma (N,H) [likelihood losses] or q (N,H,Q) [pinball]."""
        self.model.eval()
        out = self.model(torch.tensor(X).to(self.device))
        res = {"m": out["median"].cpu().numpy()}
        if self.loss_name == "pinball":
            q = np.sort(out["quantiles"].cpu().numpy(), axis=-1)   # remove crossings
            res["q"] = q
            res["m"] = q[..., QR_LEVELS.index(0.50)]
        else:
            res["sigma"] = np.exp(out["log_scale"].cpu().numpy())
        return res
