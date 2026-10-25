# LEGACY (first draft). Not used for the revised paper; see experiments/.
"""PATH-A experiment: VACP-MH on Beijing PM2.5 (strongly heteroscedastic target).

Hypothesis under test: on a target with strong, persistent conditional
heteroscedasticity (volatility clustering), the dispersion-normalised online
conformal (VACP-MH) should improve over the dispersion-FREE online ablation
(PlainACI) and the static-split ablation -- unlike the near-random-walk daily
equity index, where it did not.

Target: PM2.5+1 (positive surrogate; metrics invariant to the +1 shift).
Scale signal (VIX analog): trailing realised volatility of log-increments.
Horizons (hourly): {1,3,6,12,24}. Levels: {0.80,0.90,0.95}.

Outputs mirror the NIFTY run under results_aq/ and figures_aq/.
"""
from __future__ import annotations
import os, sys
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
FIG = os.path.join(HERE, "figures_aq"); RES = os.path.join(HERE, "results_aq")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24]
LEVELS = [0.80, 0.90, 0.95]
SEQ = 48                       # 48-hour lookback
FEATURES = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
            "HOUR_SIN", "HOUR_COS", "MONTH_SIN", "MONTH_COS"]

FAST = "--fast" in sys.argv
EPOCHS = 6 if FAST else 30
DP_EPOCHS = 10 if FAST else 40
DP_WARMUP = 3 if FAST else 12


def load_aq():
    df = pd.read_csv(os.path.join(HERE, "air_quality.csv"))
    df = df.reset_index(drop=True)
    target = df["Close"].to_numpy(float)            # PM2.5 + 1 (positive)
    feats = df[FEATURES].to_numpy(float)
    return df, target, feats


def rolling_vol(series, window=48):
    r = D.make_returns(series)
    return np.array([r[max(0, t - window):t].std() if t > 1 else 0.0
                     for t in range(len(series))])


def target_at_h(P0, R_h):
    return P0 * np.exp(R_h)


def main():
    torch.manual_seed(42); np.random.seed(42)
    df, target, feats = load_aq()
    scale_signal = rolling_vol(target, window=48)          # VIX analog
    print(f"Loaded PM2.5: {len(target)} hours, PM2.5 {target.min()-1:.0f}-{target.max()-1:.0f}")

    wd = D.build_windows(target, features=feats, seq_len=SEQ, horizons=HORIZONS)
    tr, va, te = D.chronological_split(wd, train=0.70, val=0.15)
    _, trs, vas, tes = D.standardize(tr, va, te)
    print(f"Windows: train={len(trs.X)} val={len(vas.X)} test={len(tes.X)}  "
          f"train_max={tr.P0.max()-1:.0f} test_max={te.P0.max()-1:.0f}")

    rows = []
    reliability = {}
    width_traces = {}
    forecast_cache = {}

    print("\n=== Training VACP-MH backbone (log-increment space) ===")
    vacp = VACP_MH(horizons=HORIZONS, coverage_levels=LEVELS, seq_len=SEQ,
                   conformal_method="pid", gamma=0.03, ki=0.01, window=200, normalized=True)
    vacp.fit(trs, vas, epochs=EPOCHS, verbose=True)

    static = BackboneStaticSplit(vacp, normalized=True).calibrate(vas)
    plain = VACP_MH(horizons=HORIZONS, coverage_levels=LEVELS, normalized=False,
                    conformal_method="aci", gamma=0.03)
    plain.model = vacp.model; plain._calibrate(vas)

    def eval_method(name, out_price):
        for lvl in LEVELS:
            for hi, h in enumerate(HORIZONS):
                y = target_at_h(tes.P0, tes.R[:, hi])
                lo, up = out_price[lvl]["price"][h]
                s = Mx.summary(y, lo, up, lvl)
                rows.append(dict(method=name, horizon=h, level=lvl, **s))
                if h == 1:
                    reliability.setdefault(name, {})[lvl] = s["PICP"]
                    if abs(lvl - 0.90) < 1e-9:
                        width_traces[name] = (up - lo)

    print("=== Evaluating VACP-MH / ablations / naive ===")
    vacp_out = {lvl: vacp.predict_intervals(tes, lvl, update_online=True)[0] for lvl in LEVELS}
    plain_out = {lvl: plain.predict_intervals(tes, lvl, update_online=True)[0] for lvl in LEVELS}
    static_out = {lvl: static.predict_intervals(tes, lvl) for lvl in LEVELS}
    eval_method("VACP-MH", vacp_out)
    eval_method("Ablate:StaticSplit", static_out)
    eval_method("Ablate:PlainACI", plain_out)

    ret_series = D.make_returns(target)
    naive = NaiveVolInterval(HORIZONS, vol_window=48)
    naive_out = {}
    n_test = len(tes.P0)
    ret_test = ret_series[-n_test:]
    for lvl in LEVELS:
        naive_out[lvl] = naive.predict_intervals(target[-n_test:], ret_test, tes.P0, lvl)
    eval_method("Naive-Vol", naive_out)

    lo90, up90 = vacp_out[0.90]["price"][1]
    forecast_cache["VACP-MH"] = (target_at_h(tes.P0, tes.R[:, 0]),
                                 vacp._predict_raw(tes)[0][:, 0], lo90, up90)

    # DeepPIPE reference (target space, next-step)
    print("\n=== Training DeepPIPE reference (single & two-phase) ===")
    Xtr, Xva, Xte = trs.X, vas.X, tes.X
    pm, ps = target.mean(), target.std()
    def std_t(P0, Rh): return (target_at_h(P0, Rh) - pm) / ps
    ytr = std_t(tr.P0, tr.R[:, 0]); yva = std_t(va.P0, va.R[:, 0])
    for mode, label in [("single_phase", "DeepPIPE(single)"), ("two_phase", "DeepPIPE(two-phase)")]:
        dp = train_deeppipe(Xtr, ytr, Xva, yva, input_size=Xtr.shape[-1], mode=mode,
                            epochs=DP_EPOCHS, warmup=DP_WARMUP, alpha=0.90, seed=42)
        yhat, lo, up = predict_deeppipe(dp, Xte)
        yhat = yhat * ps + pm; lo = lo * ps + pm; up = up * ps + pm
        y = target_at_h(tes.P0, tes.R[:, 0])
        for lvl in LEVELS:
            s = Mx.summary(y, lo, up, lvl)
            rows.append(dict(method=label, horizon=1, level=lvl, **s))
            reliability.setdefault(label, {})[lvl] = s["PICP"]
            if abs(lvl - 0.90) < 1e-9:
                width_traces[label] = (up - lo)
        if mode == "two_phase":
            forecast_cache["DeepPIPE(two-phase)"] = (y, yhat, lo, up)

    # save tables
    dfm = pd.DataFrame(rows)
    dfm.to_csv(os.path.join(RES, "metrics_all.csv"), index=False)
    h1 = dfm[(dfm.horizon == 1) & (np.isclose(dfm.level, 0.90))].copy().sort_values("IntervalScore")
    h1.to_csv(os.path.join(RES, "summary_h1.csv"), index=False)
    print("\n=== Headline (h=1, level=0.90), sorted by interval score ===")
    print(h1[["method", "PICP", "MPIW", "CovDev", "IntervalScore"]].to_string(index=False))

    rel_rows = [dict(method=m, level=l, empirical=reliability[m][l])
                for m in reliability for l in reliability[m]]
    pd.DataFrame(rel_rows).to_csv(os.path.join(RES, "reliability.csv"), index=False)

    # statistics
    scale_test = scale_signal[-len(tes.P0):]
    with open(os.path.join(RES, "stats.txt"), "w") as f:
        f.write("== width-vs-scale correlation (h=1, level 0.90): adaptivity ==\n")
        for m, w in width_traces.items():
            if len(w) == len(scale_test):
                c = np.corrcoef(scale_test, w)[0, 1]
                f.write(f"  corr(width, scale)  {m:<22} = {c:+.3f}\n")
        f.write("\n== Interval-score DM test vs VACP-MH (h=1, level 0.90) ==\n")
        def is_loss(y, lo, up, lvl):
            a = 1 - lvl; w = up - lo
            return w + (2/a)*((lo - y)*(y < lo) + (y - up)*(y > up))
        y = target_at_h(tes.P0, tes.R[:, 0])
        lo_v, up_v = vacp_out[0.90]["price"][1]
        L_v = is_loss(y, lo_v, up_v, 0.90)
        for name, out in [("StaticSplit", static_out[0.90]["price"][1]),
                          ("PlainACI", plain_out[0.90]["price"][1]),
                          ("Naive-Vol", naive_out[0.90]["price"][1])]:
            L_b = is_loss(y, out[0], out[1], 0.90)
            dm, p = St.diebold_mariano(L_v, L_b, h=1)
            f.write(f"  VACP-MH vs {name:<12} DM={dm:+.2f} p={p:.4f} "
                    f"({'VACP better' if dm < 0 else 'baseline better'})\n")
        mean, lo_ci, hi_ci = St.block_bootstrap_ci(L_v, block=48)
        f.write(f"\n  VACP-MH interval-score mean={mean:.1f} 95%CI=[{lo_ci:.1f},{hi_ci:.1f}]\n")

        # ---- REGIME-CONDITIONAL COVERAGE (the discriminating Path-A test) ----
        # Split the test set into calm / normal / turbulent thirds by the
        # conditional-scale signal, and report per-regime PICP at 90%. A method
        # earns the dispersion normaliser if it HOLDS coverage in the turbulent
        # regime where fixed-width methods under-cover.
        f.write("\n== Regime-conditional coverage @90% (calm / mid / turbulent thirds by scale) ==\n")
        f.write("   target = 0.900; deviation from 0.900 in the turbulent third is the key number\n")
        qs = np.quantile(scale_test, [1/3, 2/3])
        regime = np.where(scale_test <= qs[0], 0, np.where(scale_test <= qs[1], 1, 2))
        rlabels = ["calm", "mid", "turbulent"]
        methods_io = [("VACP-MH", vacp_out[0.90]["price"][1]),
                      ("StaticSplit", static_out[0.90]["price"][1]),
                      ("PlainACI", plain_out[0.90]["price"][1]),
                      ("Naive-Vol", naive_out[0.90]["price"][1])]
        f.write(f"   {'method':<12}" + "".join(f"{r:>12}" for r in rlabels) + f"{'turb|dev|':>12}\n")
        regime_cov = {}
        for name, (lo_m, up_m) in methods_io:
            cov = (y >= lo_m) & (y <= up_m)
            row = []
            for g in range(3):
                m = regime == g
                row.append(cov[m].mean() if m.any() else float("nan"))
            regime_cov[name] = row
            f.write(f"   {name:<12}" + "".join(f"{c:>12.3f}" for c in row)
                    + f"{abs(row[2]-0.90):>12.3f}\n")
        # widths per regime too
        f.write("\n== Regime-conditional mean width @90% ==\n")
        f.write(f"   {'method':<12}" + "".join(f"{r:>12}" for r in rlabels) + "\n")
        for name, (lo_m, up_m) in methods_io:
            w = up_m - lo_m
            row = [w[regime == g].mean() if (regime == g).any() else float("nan") for g in range(3)]
            f.write(f"   {name:<12}" + "".join(f"{v:>12.1f}" for v in row) + "\n")
    print("stats written to results_aq/stats.txt")

    # figures
    print("Generating figures...")
    idx = np.arange(len(tes.P0))
    y_true, y_pred, lo90, up90 = forecast_cache["VACP-MH"]
    P.plot_forecast_band(idx, y_true, y_pred, lo90, up90, 0.90,
                         "VACP-MH: PM2.5 point forecast and 90% interval (Beijing test)",
                         os.path.join(FIG, "fig1_vacp_band.png"))
    P.plot_reliability(reliability, list(reliability.keys()),
                       os.path.join(FIG, "fig3_reliability.png"))
    mpiw_by = {m: [dfm[(dfm.method == m) & (dfm.horizon == h) & np.isclose(dfm.level, 0.90)]
                   ["MPIW"].mean() for h in HORIZONS]
               for m in ["VACP-MH", "Ablate:StaticSplit", "Ablate:PlainACI", "Naive-Vol"]}
    P.plot_width_vs_horizon(mpiw_by, HORIZONS, os.path.join(FIG, "fig4_width_vs_horizon.png"))
    if len(width_traces.get("VACP-MH", [])) == len(scale_test):
        P.plot_width_vs_volatility(width_traces["VACP-MH"], scale_test,
                                   os.path.join(FIG, "fig5_vacp_width_vs_scale.png"),
                                   title="VACP-MH width vs realised-vol scale (should be POSITIVE)")
    if "Ablate:PlainACI" in width_traces and len(width_traces["Ablate:PlainACI"]) == len(scale_test):
        P.plot_width_vs_volatility(width_traces["Ablate:PlainACI"], scale_test,
                                   os.path.join(FIG, "fig6_plainaci_width_vs_scale.png"),
                                   title="PlainACI (no vol-norm) width vs scale")
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
