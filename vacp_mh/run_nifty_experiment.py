# LEGACY (first draft). Not used for the revised paper; see experiments/.
"""FULL NIFTY 50 experiment: VACP-MH vs DeepPIPE (single/two-phase) vs baselines.

Multi-horizon {1,3,5,10,20}, multi-level {0.80,0.90,0.95}. Produces:
  results/metrics_all.csv           -- every (method,horizon,level) metric row
  results/summary_h1.csv            -- headline next-day table (paper Table 2 style)
  results/reliability.csv           -- nominal vs empirical coverage per method
  results/stats.txt                 -- DM tests, bootstrap CIs, VIX-width correlations
  figures/*.png                     -- all visualizations

Run:  python run_nifty_experiment.py            (full)
      python run_nifty_experiment.py --fast     (fewer epochs, quick smoke test)
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from vacp_mh import data as D
from vacp_mh import metrics as Mx
from vacp_mh import plots as P
from vacp_mh import stats as St
from vacp_mh.vacp import VACP_MH
from vacp_mh.baselines import NaiveVolInterval, BackboneStaticSplit
from vacp_mh.deeppipe import train_deeppipe, predict_deeppipe

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

HORIZONS = [1, 3, 5, 10, 20]
LEVELS = [0.80, 0.90, 0.95]
SEQ = 60
FEATURES = ["NIFTY_Open", "NIFTY_High", "NIFTY_Low", "NIFTY_Close", "NIFTY_Volume",
            "INDIA_VIX_Close", "SP500_Close", "NASDAQ_Close", "DOW_Close",
            "NIKKEI_Close", "HANGSENG_Close", "BRENT_Close", "GOLD_Close", "USDINR_Close"]

FAST = "--fast" in sys.argv
EPOCHS = 8 if FAST else 60
DP_EPOCHS = 12 if FAST else 120
DP_WARMUP = 4 if FAST else 40


def load_nifty():
    df = pd.read_csv(os.path.join(HERE, "market_data.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").ffill().bfill().reset_index(drop=True)
    prices = df["NIFTY_Close"].to_numpy(float)
    vix = df["INDIA_VIX_Close"].to_numpy(float)
    feats = df[FEATURES].to_numpy(float)
    return df, prices, vix, feats


def price_target_at_h(P0, R_h):
    return P0 * np.exp(R_h)


def main():
    torch.manual_seed(42); np.random.seed(42)
    df, prices, vix, feats = load_nifty()
    print(f"Loaded NIFTY: {len(prices)} days, close {prices.min():.0f}-{prices.max():.0f}")

    # ---- windows in return space for VACP-MH & baselines ----
    wd = D.build_windows(prices, features=feats, seq_len=SEQ, horizons=HORIZONS)
    tr, va, te = D.chronological_split(wd, train=0.70, val=0.15)
    _, trs, vas, tes = D.standardize(tr, va, te)
    print(f"Windows: train={len(trs.X)} val={len(vas.X)} test={len(tes.X)}  "
          f"train_max_price={tr.P0.max():.0f} test_max_price={te.P0.max():.0f}")

    rows = []                       # metric rows
    reliability = {}                # {method: {level: empirical_picp at h=1}}
    width_traces = {}               # {method: (width_h1_at_0.90, vix_aligned)}
    forecast_cache = {}             # for the band figure

    # ================= VACP-MH (proposed) =================
    print("\n=== Training VACP-MH backbone (return space) ===")
    vacp = VACP_MH(horizons=HORIZONS, coverage_levels=LEVELS, seq_len=SEQ,
                   conformal_method="pid", gamma=0.03, ki=0.01, window=200, normalized=True)
    vacp.fit(trs, vas, epochs=EPOCHS, verbose=True)

    # static-split (exchangeability ablation) & plain-ACI (no vol-norm ablation) share backbone
    static = BackboneStaticSplit(vacp, normalized=True).calibrate(vas)
    plain = VACP_MH(horizons=HORIZONS, coverage_levels=LEVELS, normalized=False,
                    conformal_method="aci", gamma=0.03)
    plain.model = vacp.model; plain._calibrate(vas)

    def eval_return_method(name, out_price):
        for lvl in LEVELS:
            for hi, h in enumerate(HORIZONS):
                y = price_target_at_h(tes.P0, tes.R[:, hi])
                lo, up = out_price[lvl]["price"][h]
                s = Mx.summary(y, lo, up, lvl)
                rows.append(dict(method=name, horizon=h, level=lvl, **s))
                if h == 1:
                    reliability.setdefault(name, {})[lvl] = s["PICP"]
                    if abs(lvl - 0.90) < 1e-9:
                        width_traces[name] = (up - lo)

    print("=== Evaluating VACP-MH / ablations / naive (walk-forward) ===")
    vacp_out = {lvl: vacp.predict_intervals(tes, lvl, update_online=True)[0] for lvl in LEVELS}
    plain_out = {lvl: plain.predict_intervals(tes, lvl, update_online=True)[0] for lvl in LEVELS}
    static_out = {lvl: static.predict_intervals(tes, lvl) for lvl in LEVELS}
    eval_return_method("VACP-MH", vacp_out)
    eval_return_method("Ablate:StaticSplit", static_out)
    eval_return_method("Ablate:PlainACI", plain_out)

    # naive vol interval
    ret_series = D.make_returns(prices)
    naive = NaiveVolInterval(HORIZONS)
    naive_out = {}
    n_test = len(tes.P0)
    ret_test = ret_series[-n_test:]
    for lvl in LEVELS:
        naive_out[lvl] = naive.predict_intervals(prices[-n_test:], ret_test, tes.P0, lvl)
    eval_return_method("Naive-Vol", naive_out)

    # cache VACP band for figure (h=1, level 0.90)
    lo90, up90 = vacp_out[0.90]["price"][1]
    forecast_cache["VACP-MH"] = (price_target_at_h(tes.P0, tes.R[:, 0]),
                                 vacp._predict_raw(tes)[0][:, 0], lo90, up90)

    # ================= DeepPIPE reference (price space, next-day) =================
    # Build price-space next-day arrays aligned to same windows.
    print("\n=== Training DeepPIPE reference (single & two-phase) ===")
    Xtr, Xva, Xte = trs.X, vas.X, tes.X
    ytr = price_target_at_h(tr.P0, tr.R[:, 0]); ytr = (ytr - prices.mean()) / prices.std()
    # DeepPIPE trained on standardized price target for stability; invert for metrics
    pm, ps = prices.mean(), prices.std()
    def std_price(P0, Rh): return (price_target_at_h(P0, Rh) - pm) / ps
    ytr = std_price(tr.P0, tr.R[:, 0]); yva = std_price(va.P0, va.R[:, 0])

    for mode, label in [("single_phase", "DeepPIPE(single)"), ("two_phase", "DeepPIPE(two-phase)")]:
        dp = train_deeppipe(Xtr, ytr, Xva, yva, input_size=Xtr.shape[-1], mode=mode,
                            epochs=DP_EPOCHS, warmup=DP_WARMUP, alpha=0.90, seed=42)
        yhat, lo, up = predict_deeppipe(dp, Xte)
        yhat = yhat * ps + pm; lo = lo * ps + pm; up = up * ps + pm
        y = price_target_at_h(tes.P0, tes.R[:, 0])
        # DeepPIPE has a fixed interval (single 90% design); report at 0.90 only, but
        # record the same width at all levels to expose its non-responsiveness.
        for lvl in LEVELS:
            s = Mx.summary(y, lo, up, lvl)
            rows.append(dict(method=label, horizon=1, level=lvl, **s))
            reliability.setdefault(label, {})[lvl] = s["PICP"]
            if abs(lvl - 0.90) < 1e-9:
                width_traces[label] = (up - lo)
        if mode == "two_phase":
            forecast_cache["DeepPIPE(two-phase)"] = (y, yhat, lo, up)

    # ================= save tables =================
    dfm = pd.DataFrame(rows)
    dfm.to_csv(os.path.join(RES, "metrics_all.csv"), index=False)
    h1 = dfm[(dfm.horizon == 1) & (np.isclose(dfm.level, 0.90))].copy()
    h1 = h1.sort_values("IntervalScore")
    h1.to_csv(os.path.join(RES, "summary_h1.csv"), index=False)
    print("\n=== Headline (h=1, level=0.90), sorted by interval score ===")
    print(h1[["method", "PICP", "MPIW", "CovDev", "IntervalScore"]].to_string(index=False))

    rel_rows = [dict(method=m, level=l, empirical=reliability[m][l])
                for m in reliability for l in reliability[m]]
    pd.DataFrame(rel_rows).to_csv(os.path.join(RES, "reliability.csv"), index=False)

    # ================= statistics =================
    with open(os.path.join(RES, "stats.txt"), "w") as f:
        f.write("== VIX-width correlation (h=1, level 0.90): adaptivity ==\n")
        vix_test = vix[-len(tes.P0):]
        for m, w in width_traces.items():
            if len(w) == len(vix_test):
                c = np.corrcoef(vix_test, w)[0, 1]
                f.write(f"  corr(width, VIX)  {m:<22} = {c:+.3f}\n")
        f.write("\n== Interval-score DM test vs VACP-MH (h=1, level 0.90) ==\n")
        # build per-point interval-score losses
        def is_loss(y, lo, up, lvl):
            a = 1 - lvl; w = up - lo
            return w + (2/a)*((lo - y)*(y < lo) + (y - up)*(y > up))
        y = price_target_at_h(tes.P0, tes.R[:, 0])
        lo_v, up_v = vacp_out[0.90]["price"][1]
        L_v = is_loss(y, lo_v, up_v, 0.90)
        for name, out in [("StaticSplit", static_out[0.90]["price"][1]),
                          ("PlainACI", plain_out[0.90]["price"][1]),
                          ("Naive-Vol", naive_out[0.90]["price"][1])]:
            L_b = is_loss(y, out[0], out[1], 0.90)
            dm, p = St.diebold_mariano(L_v, L_b, h=1)
            f.write(f"  VACP-MH vs {name:<12} DM={dm:+.2f} p={p:.4f} "
                    f"({'VACP better' if dm < 0 else 'baseline better'})\n")
        mean, lo_ci, hi_ci = St.block_bootstrap_ci(L_v, block=20)
        f.write(f"\n  VACP-MH interval-score mean={mean:.1f} 95%CI=[{lo_ci:.1f},{hi_ci:.1f}]\n")
    print("\nstats written to results/stats.txt")

    # ================= figures =================
    print("Generating figures...")
    idx = np.arange(len(tes.P0))
    y_true, y_pred, lo90, up90 = forecast_cache["VACP-MH"]
    P.plot_forecast_band(idx, y_true, y_pred, lo90, up90, 0.90,
                         "VACP-MH: point forecast and 90% interval (NIFTY 50 test)",
                         os.path.join(FIG, "fig1_vacp_band.png"))
    if "DeepPIPE(two-phase)" in forecast_cache:
        yt, yp, lo, up = forecast_cache["DeepPIPE(two-phase)"]
        P.plot_forecast_band(idx, yt, yp, lo, up, 0.90,
                             "DeepPIPE (two-phase): 90% interval (NIFTY 50 test)",
                             os.path.join(FIG, "fig2_deeppipe_band.png"))
    P.plot_reliability(reliability, list(reliability.keys()),
                       os.path.join(FIG, "fig3_reliability.png"))
    mpiw_by = {m: [dfm[(dfm.method == m) & (dfm.horizon == h) & np.isclose(dfm.level, 0.90)]
                   ["MPIW"].mean() for h in HORIZONS]
               for m in ["VACP-MH", "Ablate:StaticSplit", "Ablate:PlainACI", "Naive-Vol"]}
    P.plot_width_vs_horizon(mpiw_by, HORIZONS, os.path.join(FIG, "fig4_width_vs_horizon.png"))
    vix_test = vix[-len(tes.P0):]
    if len(width_traces.get("VACP-MH", [])) == len(vix_test):
        P.plot_width_vs_volatility(width_traces["VACP-MH"], vix_test,
                                   os.path.join(FIG, "fig5_vacp_width_vs_vix.png"),
                                   title="VACP-MH width vs VIX (should be POSITIVE)")
    if "DeepPIPE(two-phase)" in width_traces and len(width_traces["DeepPIPE(two-phase)"]) == len(vix_test):
        P.plot_width_vs_volatility(width_traces["DeepPIPE(two-phase)"], vix_test,
                                   os.path.join(FIG, "fig6_deeppipe_width_vs_vix.png"),
                                   title="DeepPIPE width vs VIX (paper reports -0.50)")
    names = h1["method"].tolist(); vals = h1["IntervalScore"].tolist()
    P.plot_metric_bars(names, vals, "Interval score (lower=better)",
                       "Interval score by method (h=1, 90%)",
                       os.path.join(FIG, "fig7_intervalscore_bars.png"))
    P.plot_coverage_over_time(y_true, lo90, up90,
                              os.path.join(FIG, "fig8_vacp_rolling_coverage.png"))
    print(f"Figures saved to {FIG}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
