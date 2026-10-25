"""Build air_quality.csv from the UCI Beijing Multi-Site Air-Quality dataset.

Path-A evaluation: forecast PM2.5, a strongly heteroscedastic pollutant series
(pronounced volatility clustering: calm clean days vs. sudden pollution spikes),
which is the regime where the framework's dispersion normaliser should help --
unlike a near-random-walk daily equity index.

We use one station (Aotizhongxin) at hourly resolution, 2013-03-01..2017-02-28
(35,064 hours). Heterogeneous features: the other pollutants (PM10, SO2, NO2, CO,
O3) + meteorology (TEMP, PRES, DEWP, RAIN, WSPM) + cyclical hour/month encodings.

The experiment forecasts in log-increment space; because PM2.5 can be 0, we model
TARGET = PM2.5 + 1 (a strictly positive surrogate). PICP / MPIW / interval score
are all invariant to this +1 shift, so results read directly as PM2.5.

Output schema mirrors the price CSV so the harness's loader works unchanged:
  Date, <TARGET as 'Close'>, plus feature columns.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

SRC = "/tmp/beijing/inner/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "air_quality.csv")

FEATURES = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def main():
    df = pd.read_csv(SRC)
    df["Date"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("Date").reset_index(drop=True)

    # target: PM2.5 (+1 positive surrogate); interpolate short gaps then ffill/bfill
    df["PM2.5"] = df["PM2.5"].interpolate(limit=6).ffill().bfill()
    target = df["PM2.5"].to_numpy(float) + 1.0

    for c in FEATURES:
        df[c] = df[c].interpolate(limit=6).ffill().bfill()

    # cyclical time encodings (heterogeneous exogenous variables)
    hr = df["hour"].to_numpy(float)
    mo = df["month"].to_numpy(float)
    hour_sin = np.sin(2 * np.pi * hr / 24.0)
    hour_cos = np.cos(2 * np.pi * hr / 24.0)
    month_sin = np.sin(2 * np.pi * mo / 12.0)
    month_cos = np.cos(2 * np.pi * mo / 12.0)

    out = pd.DataFrame({"Date": df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "Close": target})
    for c in FEATURES:
        out[c] = df[c].to_numpy(float)
    out["HOUR_SIN"] = hour_sin
    out["HOUR_COS"] = hour_cos
    out["MONTH_SIN"] = month_sin
    out["MONTH_COS"] = month_cos

    out = out.dropna(subset=["Close"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"WROTE {OUT}")
    print(f"rows={len(out)}  {out.Date.iloc[0]} -> {out.Date.iloc[-1]}")
    pm = out['Close'].to_numpy() - 1.0
    print(f"PM2.5 range: {pm.min():.0f} - {pm.max():.0f}  mean={pm.mean():.1f}  std={pm.std():.1f}")
    # heteroscedasticity check: std of hourly log-increments in rolling windows
    r = np.diff(np.log(out['Close'].to_numpy()))
    rv = pd.Series(r).rolling(168).std()  # weekly rolling vol of increments
    print(f"rolling weekly increment-vol: min={rv.min():.3f} max={rv.max():.3f} "
          f"ratio(max/min)={rv.max()/max(rv.min(),1e-9):.1f}x  (higher => more heteroscedastic)")


if __name__ == "__main__":
    main()
