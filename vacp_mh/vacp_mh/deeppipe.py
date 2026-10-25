"""DeepPIPE reference baseline (Wang et al., 2020) -- used ONLY as a competitor/reference.

Reproduces the encoder-decoder LSTM with three heads (point, lower, upper via softplus
offsets) and the hybrid loss L = beta*MAE + MPIW + lambda*(max(0, alpha - PICP))^2.
Supports single-phase (original) and two-phase (paper's fix) training so we can benchmark
VACP-MH against BOTH the collapse and the paper's proposed remedy.

Operates in PRICE space (as in the reference paper), which is exactly the setting where the
level-OOD problem appears -- this is intentional, so the comparison is faithful.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepPIPE(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.point_head = nn.Linear(hidden, 1)
        self.lower_head = nn.Linear(hidden, 1)
        self.upper_head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.encoder(x)
        h = out[:, -1, :]
        yhat = self.point_head(h).squeeze(-1)
        lo = yhat - F.softplus(self.lower_head(h).squeeze(-1))
        up = yhat + F.softplus(self.upper_head(h).squeeze(-1))
        return yhat, lo, up


def deeppipe_loss(yhat, lo, up, y, beta=0.5, lam=15.0, alpha=0.90):
    """Hybrid loss. Soft PICP via sigmoid for differentiability."""
    mae = torch.mean(torch.abs(y - yhat))
    mpiw = torch.mean(up - lo)
    # soft coverage
    soft_in = torch.sigmoid(50.0 * (y - lo)) * torch.sigmoid(50.0 * (up - y))
    picp = torch.mean(soft_in)
    penalty = torch.clamp(alpha - picp, min=0.0) ** 2
    return beta * mae + mpiw + lam * penalty, mae, mpiw, picp


def train_deeppipe(Xtr, ytr, Xva, yva, input_size, mode="two_phase",
                   hidden=64, layers=2, epochs=120, warmup=40, lr=1e-3,
                   beta=0.5, lam=15.0, alpha=0.90, batch=32, patience=25,
                   device="cpu", seed=42, verbose=False):
    """Train DeepPIPE in price space. mode in {"single_phase","two_phase"}.

    Returns the trained model. X arrays: (N,L,F); y arrays: (N,) price target at t+1.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = DeepPIPE(input_size, hidden, layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(device)
    yva_t = torch.tensor(yva, dtype=torch.float32).to(device)
    n = len(Xtr_t)

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr_t[idx].to(device); yb = ytr_t[idx].to(device)
            yhat, lo, up = model(xb)
            if mode == "two_phase" and ep < warmup:
                loss = torch.mean(torch.abs(yb - yhat))          # point-only warm-up
            else:
                loss, *_ = deeppipe_loss(yhat, lo, up, yb, beta, lam, alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # validation (full hybrid loss for model selection once intervals active)
        model.eval()
        with torch.no_grad():
            yhat, lo, up = model(Xva_t)
            if mode == "two_phase" and ep < warmup:
                vloss = torch.mean(torch.abs(yva_t - yhat)).item()
            else:
                vl, *_ = deeppipe_loss(yhat, lo, up, yva_t, beta, lam, alpha)
                vloss = vl.item()
        sched.step(vloss)
        # early stopping only during the interval phase (matches paper's guidance)
        active = not (mode == "two_phase" and ep < warmup)
        if active:
            if vloss < best - 1e-6:
                best, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if bad >= patience:
                break
        if verbose and ep % 20 == 0:
            print(f"  [deeppipe {mode}] ep{ep} vloss={vloss:.2f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_deeppipe(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        yhat, lo, up = model(torch.tensor(X, dtype=torch.float32).to(device))
    return yhat.cpu().numpy(), lo.cpu().numpy(), up.cpu().numpy()
