"""Shared loading, feature engineering and windowing for both case studies."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vacp_mh import data as D  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
GLOBAL = ["SP500", "NASDAQ", "DOW", "NIKKEI", "HANGSENG", "BRENT", "GOLD", "USDINR"]
POLL = ["PM10", "SO2", "NO2", "CO", "O3"]
MET = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
CAL = ["HOUR_SIN", "HOUR_COS", "MONTH_SIN", "MONTH_COS"]


def load_cfg(name):
    with open(os.path.join(ROOT, "experiments", "configs", f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def _lr(x):
    return D.make_returns(np.asarray(x, float))


def load_dataset(cfg):
    """Return (df, series, feature_groups: {name: (T, k) array, column names}, scale signal)."""
    df = pd.read_csv(os.path.join(ROOT, cfg["csv"]))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    s = df[cfg["target"]].to_numpy(float)
    g = {}
    if cfg["name"] == "nifty":
        g["own"] = (np.column_stack([_lr(s),
                                     np.log(df.NIFTY_High / df.NIFTY_Low),
                                     np.log(df.NIFTY_Close / df.NIFTY_Open)]),
                    ["r_NIFTY", "log(H/L)", "log(C/O)"])
        lv = np.log(df.INDIA_VIX_Close.to_numpy(float))
        g["vix"] = (np.column_stack([lv, np.r_[0.0, np.diff(lv)]]), ["log VIX", "dlog VIX"])
        g["global"] = (np.column_stack([_lr(df[f"{k}_Close"]) for k in GLOBAL]),
                       [f"r_{k}" for k in GLOBAL])
        scale = df.INDIA_VIX_Close.to_numpy(float)
    else:
        r = _lr(s)
        g["own"] = (np.column_stack([r, np.log(s)]), ["r_PM2.5", "log PM2.5"])
        g["poll"] = (np.column_stack([np.log1p(df[c].to_numpy(float)) for c in POLL]),
                     [f"log1p {c}" for c in POLL])
        met = [df[c].to_numpy(float) for c in MET]
        met[MET.index("RAIN")] = np.log1p(met[MET.index("RAIN")])
        g["met"] = (np.column_stack(met + [df[c].to_numpy(float) for c in CAL]), MET + CAL)
        # trailing 48-h realised volatility of log-increments, observable at the origin
        scale = pd.Series(r).rolling(48, min_periods=2).std().fillna(0.0).to_numpy()
    return df, s, g, scale


def make_windows(cfg, groups, series, seq_len=None):
    F = np.column_stack([groups[k][0] for k in groups_order(cfg, groups)])
    return D.build_windows(series, F, seq_len or cfg["seq_len"], cfg["horizons"],
                           start=cfg["start_index"])


def groups_order(cfg, groups):
    return [k for k in cfg["feature_groups"] if k in groups]


def split_std(cfg, wd):
    tr, va, te = D.purged_split(wd, **cfg["split"])
    _, trs, vas, tes = D.standardize(tr, va, te)
    return trs, vas, tes
