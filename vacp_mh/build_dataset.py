"""Build market_data.csv for 2008-03 to 2025-12 (VIX-aligned start).

Downloads NIFTY 50, India VIX, global indices, commodities, and FX from Yahoo
Finance, aligns them on NIFTY trading days, forward/back-fills gaps, and writes
the same schema the experiment expects:
  Date, <SYM>_Close, <SYM>_High, <SYM>_Low, <SYM>_Open, <SYM>_Volume  (x each SYM)

All data is REAL (Yahoo Finance). India VIX (^INDIAVIX) begins 2008-03-03, which
sets the honest start of the dataset; NIFTY itself is available from 2007-09 but
we align the whole panel to the VIX start so every feature is real from day one.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
START, END = "2008-03-01", "2025-12-31"

# order matters: NIFTY first (defines the trading calendar)
SYMBOLS = [
    ("^NSEI", "NIFTY"),
    ("^INDIAVIX", "INDIA_VIX"),
    ("^GSPC", "SP500"),
    ("^IXIC", "NASDAQ"),
    ("^DJI", "DOW"),
    ("^N225", "NIKKEI"),
    ("^HSI", "HANGSENG"),
    ("BZ=F", "BRENT"),
    ("GC=F", "GOLD"),
    ("USDINR=X", "USDINR"),
]
FIELDS = ["Close", "High", "Low", "Open", "Volume"]


def fetch(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=START, end=END, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"empty download for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):      # yfinance sometimes returns MultiIndex
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in FIELDS if c in df.columns]].copy()
    for f in FIELDS:
        if f not in df.columns:
            df[f] = 0.0
    return df[FIELDS]


def main():
    frames = {}
    for sym, label in SYMBOLS:
        d = fetch(sym)
        print(f"{label:<10} {len(d):>5} rows  {d.index.min().date()} -> {d.index.max().date()}")
        frames[label] = d

    # NIFTY trading calendar is the master index
    master_idx = frames["NIFTY"].index
    cols = {}
    for label, d in frames.items():
        d = d.reindex(master_idx)                  # align to NIFTY days
        for f in FIELDS:
            cols[f"{label}_{f}"] = d[f].to_numpy(float)

    out = pd.DataFrame(cols, index=master_idx)
    out.index.name = "Date"
    # fill non-trading-day gaps in foreign markets, then any residual leading NaNs
    out = out.ffill().bfill()
    # volume NaNs -> 0
    for c in out.columns:
        if c.endswith("_Volume"):
            out[c] = out[c].fillna(0.0)
    out = out.dropna(subset=["NIFTY_Close"]).reset_index()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")

    # reorder columns to match original schema exactly
    ordered = ["Date"]
    for _, label in SYMBOLS:
        for f in FIELDS:
            ordered.append(f"{label}_{f}")
    out = out[ordered]

    path = os.path.join(HERE, "market_data.csv")
    out.to_csv(path, index=False)
    print(f"\nWROTE {path}")
    print(f"rows={len(out)}  {out.Date.iloc[0]} -> {out.Date.iloc[-1]}")
    print(f"NIFTY close range: {out.NIFTY_Close.min():.0f} - {out.NIFTY_Close.max():.0f}")


if __name__ == "__main__":
    main()
