"""Make every paper table (LaTeX) and figure from outputs/<data>/analysis.pkl.

Usage: python experiments/report.py
Tables -> ../paper/tables/*.tex     Figures -> ../paper/figs/*.pdf (+ .png)
"""
from __future__ import annotations
import os, pickle, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, ROOT, load_cfg, load_dataset
from vacp_mh import stats as St
from analyze import MAIN, GROUP, ABL, PROPOSED, LEGACY_NAMES

PAPER = os.path.join(os.path.dirname(ROOT), "paper")
TAB = os.path.join(PAPER, "tables"); FIG = os.path.join(PAPER, "figs")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": .3,
                     "font.family": "serif"})
DS = {"nifty": "NIFTY 50", "pm25": "PM2.5"}
COL = {PROPOSED: "#c0392b", "CP-PID": "#2471a3", "ACI": "#5dade2", "LW-CP": "#7d3c98",
       "Split-CP": "#bb8fce", "GARCH-t": "#1e8449", "Naive-Vol": "#82e0aa",
       "QR-LSTM": "#d68910", "CQR": "#f5b041", "Prob-LSTM": "#7f8c8d",
       "DeepPIPE-R": "#34495e", "DeepPIPE-P": "#aab7b8"}
ONLINE = ["ACI", "CP-PID", PROPOSED]
TEX = {"w/o sigma floor": r"w/o $\sigma$ floor", "PI -> ACI": r"PI $\rightarrow$ ACI"}


def load(d):
    with open(os.path.join(OUT, d, "analysis.pkl"), "rb") as f:
        a = pickle.load(f)
    # Older analysis.pkl files store the proposed method under its former name.
    for v in a.values():
        if isinstance(v, pd.DataFrame):
            for c in ("method", "baseline"):
                if c in v.columns:
                    v[c] = v[c].replace(LEGACY_NAMES)
    return a


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)


def write(name, s):
    with open(os.path.join(TAB, name + ".tex"), "w") as f:
        f.write(s)


def note(text, after_gap=False):
    """Explanatory note printed under a table (captions are kept short)."""
    box = r"\parbox{\linewidth}{\scriptsize " + text + "}"
    return box if after_gap else r"\par\smallskip" + box


RANK = "ranked on unrounded values"  # how ties in bold/underline marks are broken


def mark(vals, lower=True, target=None):
    """Return list of 'b'/'u'/'' for best and second best."""
    v = np.array([np.nan if x is None else x for x in vals], float)
    key = np.abs(v - target) if target is not None else (v if lower else -v)
    order = [i for i in np.argsort(np.where(np.isnan(key), np.inf, key)) if not np.isnan(key[i])]
    out = [""] * len(v)
    if order:
        out[order[0]] = "b"
    if len(order) > 1:
        out[order[1]] = "u"
    return out


def fmt(x, m, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if abs(x) >= 1e4:
        e = int(np.floor(np.log10(abs(x))))
        s = f"${x / 10**e:.1f}{{\\times}}10^{{{e}}}$"
    else:
        s = f"{x:.{nd}f}"
    return r"\textbf{" + s + "}" if m == "b" else (r"\underline{" + s + "}" if m == "u" else s)


def seed_ci(df, col):
    """mean and t half-width over seeds of a per-seed quantity."""
    v = df.groupby("seed")[col].mean().to_numpy()
    return St.t_ci(v)


# ======================================================================= tables
def table_datasets(A):
    rows = []
    for d, a in A.items():
        cfg = a["cfg"]; s = a["series"]
        r = np.diff(np.log(s))
        dg = St.series_diagnostics(r)
        n_te = len(a["t0"]); dates = pd.to_datetime(a["dates"])
        rows.append((d, cfg, dg, n_te, dates, a))
    L = [r"\begin{table}[!t]\centering\caption{Dataset characteristics and diagnostics.}\label{tab:datasets}",
         r"\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{lcc}\toprule",
         r" & NIFTY~50 & Beijing PM2.5 \\\midrule"]
    def row(lbl, f):
        L.append(lbl + " & " + " & ".join(f(x) for x in rows) + r" \\")
    row("Frequency", lambda x: "daily (trading days)" if x[0] == "nifty" else "hourly")
    row("Period", lambda x: f"{x[4].min():%b %Y}--{x[4].max():%b %Y}")
    row("Observations", lambda x: f"{len(x[5]['series']):,}")
    row("Test origins", lambda x: f"{x[3]:,}")
    row("Horizons $h$", lambda x: ", ".join(str(h) for h in x[1]["horizons"]))
    row("Input features", lambda x: str(sum(1 for _ in [0]) and {"nifty": 13, "pm25": 16}[x[0]]))
    row("Scale signal $v_t$", lambda x: "India VIX" if x[0] == "nifty" else "48-h realised vol.")
    row("Std. of increments", lambda x: f"{x[2]['std']:.4f}")
    row("Skewness", lambda x: f"{x[2]['skew']:.2f}")
    row("Excess kurtosis", lambda x: f"{x[2]['ex_kurt']:.1f}")
    row("ADF stat. ($p$)", lambda x: f"{x[2]['adf_stat']:.1f} ({x[2]['adf_p']:.3f})")
    row("ARCH-LM stat. ($p$)", lambda x: f"{x[2]['archlm_stat']:.0f} ({x[2]['archlm_p']:.3f})")
    row("LB$^2$ stat. ($p$)", lambda x: f"{x[2]['lb_sq_stat']:.0f} ({x[2]['lb_sq_p']:.3f})")
    L += [r"\bottomrule\end{tabular}",
          note(r"Diagnostics are computed on the one-step log-increments. ADF: augmented Dickey--Fuller; ARCH-LM: Engle's test with 10 lags; LB$^2$: Ljung--Box on squared increments (10 lags)."),
          r"\end{table}"]
    write("datasets", "\n".join(L))
    return {x[0]: x[2] for x in rows}


def table_main(A):
    """Headline comparison at the 90% level."""
    L = [r"\begin{table*}[!t]\centering",
         r"\caption{Main comparison at the 90\% level.}\label{tab:main}",
         r"\scriptsize\setlength{\tabcolsep}{2.6pt}\resizebox{\textwidth}{!}{\begin{tabular}{ll" + "ccccc" * 2 + "}\\toprule",
         r" & & \multicolumn{5}{c}{NIFTY~50} & \multicolumn{5}{c}{Beijing PM2.5}\\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
         r"Group & Method & PICP$_{h=1}$ & IS$_{h=1}$ & $\overline{\mathrm{CovDev}}$ & $\overline{\mathrm{relIS}}$ & WRD & PICP$_{h=1}$ & IS$_{h=1}$ & $\overline{\mathrm{CovDev}}$ & $\overline{\mathrm{relIS}}$ & WRD\\\midrule"]
    cols = {}
    for d, a in A.items():
        ci = a["ci"]; met = a["metrics"]
        c1 = ci[(ci.h == 1) & np.isclose(ci.level, 0.9)].set_index("method")
        m9 = met[np.isclose(met.level, 0.9)]
        vals = {}
        for k in MAIN:
            r1 = c1.loc[k] if k in c1.index else None
            mk = m9[m9.method == k]
            multi = mk.h.nunique() > 1
            cd = seed_ci(mk, "CovDev") if multi else (None, None)
            ri = seed_ci(mk, "relIS") if multi else (None, None)
            wr = mk.worst_regime_dev.mean() if multi else None
            vals[k] = (r1, cd, ri, wr)
        cols[d] = vals
    marks = {}
    for d, vals in cols.items():
        marks[d] = dict(
            picp=mark([v[0]["PICP"] if v[0] is not None else None for v in vals.values()], target=0.9),
            is1=mark([v[0]["IS"] if v[0] is not None else None for v in vals.values()]),
            cd=mark([v[1][0] for v in vals.values()]), ri=mark([v[2][0] for v in vals.values()]),
            wr=mark([v[3] for v in vals.values()]))
    prev = None
    for i, k in enumerate(MAIN):
        g = GROUP[k] if GROUP[k] != prev else ""
        if GROUP[k] != prev and prev is not None:
            L.append(r"\addlinespace[1pt]")
        prev = GROUP[k]
        cells = [g, k]
        for d in A:
            r1, cd, ri, wr = cols[d][k]; mk = marks[d]
            if r1 is None:
                cells += ["--"] * 5; continue
            cells.append(fmt(r1["PICP"], mk["picp"][i]) + f" {{\\tiny[{r1['PICP_lo']:.2f},{r1['PICP_hi']:.2f}]}}")
            nd = 0 if r1["IS"] > 100 else 1
            cells.append(fmt(r1["IS"], mk["is1"][i], nd) + ("" if r1["IS"] >= 1e4 else f" {{\\tiny[{r1['IS_lo']:.{nd}f},{r1['IS_hi']:.{nd}f}]}}"))
            cells.append("--" if cd[0] is None else fmt(cd[0], mk["cd"][i]) + (f"{{\\tiny$\\pm${cd[1]:.3f}}}" if cd[1] > 0 else ""))
            cells.append("--" if ri[0] is None else fmt(ri[0], mk["ri"][i]) + (f"{{\\tiny$\\pm${ri[1]:.3f}}}" if ri[1] > 0 else ""))
            cells.append("--" if wr is None else fmt(wr, mk["wr"][i]))
        L.append(" & ".join(cells) + r"\\")
    L += [r"\bottomrule\end{tabular}}",
          note(r"PICP and IS at $h=1$ are seed-averaged, with 95\% moving-block bootstrap CIs in brackets. "
               r"$\overline{\mathrm{CovDev}}$ and $\overline{\mathrm{relIS}}$ are averaged over all horizons (relIS: interval score relative to Naive-Vol, lower is better) and shown as mean $\pm$ 95\% CI half-width over five seeds. "
               r"WRD: worst-regime coverage deviation averaged over horizons. Best in bold, second best underlined, " + RANK + ". DeepPIPE is a single-horizon, single-level method."),
          r"\end{table*}"]
    write("main", "\n".join(L))


def table_horizon(A):
    """PICP and relIS per horizon at 90% for the multi-horizon methods (seed means)."""
    L = [r"\begin{table*}[!t]\centering\caption{Coverage and relIS by horizon at 90\%.}\label{tab:horizon}",
         r"\scriptsize\setlength{\tabcolsep}{2.4pt}"]
    for d, a in A.items():
        H = a["H"]; met = a["metrics"]; m9 = met[np.isclose(met.level, .9)]
        meths = [k for k in MAIN if m9[m9.method == k].h.nunique() > 1]
        L.append(r"\begin{tabular}{l" + "cc" * len(H) + "}\\toprule")
        L.append(r"\multicolumn{" + str(1 + 2 * len(H)) + r"}{c}{" + DS[d] + r"}\\")
        L.append("Method & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$h={h}$}}" for h in H) + r"\\")
        L.append(" & " + " & ".join(["PICP & relIS"] * len(H)) + r"\\\midrule")
        piv_p = m9.groupby(["method", "h"]).PICP.mean().unstack()
        piv_r = m9.groupby(["method", "h"]).relIS.mean().unstack()
        mk = {h: mark([piv_r.loc[k, h] for k in meths]) for h in H}
        for i, k in enumerate(meths):
            cells = [k]
            for h in H:
                p = piv_p.loc[k, h]
                ps = f"{p:.3f}" if 0.88 <= p <= 0.92 else r"\textcolor{gray}{" + f"{p:.3f}" + "}"
                cells += [ps, fmt(piv_r.loc[k, h], mk[h][i])]
            L.append(" & ".join(cells) + r"\\")
        L.append(r"\bottomrule\end{tabular}\\[4pt]")
    L.append(note(r"Mean over five seeds. Best relIS per column in bold, second best underlined, " + RANK + r"; PICP outside $[0.88,0.92]$ in grey.", after_gap=True))
    L.append(r"\end{table*}")
    write("horizon", "\n".join(L))


def table_tests(A):
    """Diebold-Mariano (Holm) at 90% for every horizon, and coverage backtests at h=1."""
    L = [r"\begin{table*}[!t]\centering\caption{Statistical tests at the 90\% level.}\label{tab:tests}",
         r"\scriptsize\setlength{\tabcolsep}{2.6pt}"]
    for d, a in A.items():
        H = a["H"]; dm = a["dm"]; dm9 = dm[np.isclose(dm.level, .9)]
        cv = a["covtests"]; cv1 = cv[(cv.h == 1) & np.isclose(cv.level, .9)].set_index("method")
        L.append(r"\begin{tabular}{l" + "c" * len(H) + "ccc}\\toprule")
        L.append(r" & \multicolumn{" + str(len(H)) + r"}{c}{DM vs proposed, " + DS[d] + r"} & \multicolumn{3}{c}{Backtests, $h=1$}\\")
        L.append(r"\cmidrule(lr){2-" + str(1 + len(H)) + r"}\cmidrule(lr){" + str(2 + len(H)) + "-" + str(4 + len(H)) + "}")
        L.append("Method & " + " & ".join(f"$h={h}$" for h in H) + r" & UC & IND & CC\\\midrule")
        for k in MAIN:
            cells = [k]
            for h in H:
                r = dm9[(dm9.baseline == k) & (dm9.h == h)]
                if k == PROPOSED or r.empty:
                    cells.append("--"); continue
                r = r.iloc[0]; p = r.p_holm
                st = "^{***}" if p < .01 else "^{**}" if p < .05 else "^{*}" if p < .1 else ""
                cells.append(f"${r.DM:+.2f}{st}$")
            if k in cv1.index:
                c = cv1.loc[k]
                cells += [f"{c.kupiec_p:.3f}", f"{c.christ_ind_p:.3f}", f"{c.christ_cc_p:.3f}"]
            else:
                cells += ["--"] * 3
            L.append(" & ".join(cells) + r"\\")
        L.append(r"\bottomrule\end{tabular}\\[4pt]")
    L.append(note(r"Left: Diebold--Mariano statistics on the interval-score loss of the proposed method minus the baseline (negative values favour the proposed method), HAC lag $h-1$ with the Harvey correction; $^{*}$, $^{**}$, $^{***}$: Holm-adjusted $p<0.10, 0.05, 0.01$ across the baselines of each column. "
                  r"Right: $p$-values at $h=1$ of Kupiec's unconditional coverage (UC), Christoffersen's independence (IND) and conditional coverage (CC) tests (median over seeds; $p<0.05$ rejects calibration).", after_gap=True))
    L.append(r"\end{table*}")
    write("tests", "\n".join(L))


def table_regime(A):
    L = [r"\begin{table}[!t]\centering\caption{Regime-conditional coverage at $h=1$ (90\%).}\label{tab:regime}",
         r"\scriptsize\setlength{\tabcolsep}{2pt}\begin{tabular}{lccc}\toprule",
         r"Method & Calm & Moderate & Turbulent\\\midrule"]
    keep = ["Naive-Vol", "GARCH-t", "QR-LSTM", "CQR", "LW-CP", "ACI", "CP-PID", PROPOSED]
    for d, a in A.items():
        ci = a["ci"]; c1 = ci[(ci.h == 1) & np.isclose(ci.level, .9)].set_index("method")
        L.append(r"\multicolumn{4}{l}{\textit{" + DS[d] + r"}}\\")
        mk = {g: mark([c1.loc[k, g] for k in keep], target=0.9) for g in ["calm", "moderate", "turbulent"]}
        for i, k in enumerate(keep):
            cells = [k] + [fmt(c1.loc[k, g], mk[g][i]) + f" {{\\tiny[{c1.loc[k, g+'_lo']:.2f},{c1.loc[k, g+'_hi']:.2f}]}}"
                           for g in ["calm", "moderate", "turbulent"]]
            L.append(" & ".join(cells) + r"\\")
    L += [r"\bottomrule\end{tabular}",
          note(r"Regimes are terciles of the scale signal at the forecast origin. Brackets: 95\% block-bootstrap CI. Bold: closest to 0.90; underlined: second closest, " + RANK + "."),
          r"\end{table}"]
    write("regime", "\n".join(L))


def table_ablation(A):
    comp = {  # norm, online, integral, hscale, floor, student-t, log-incr
        PROPOSED: "1111111", "w/o normalisation": "0111011", "w/o online update": "1000111",
        "w/o integral term": "1101111", "PI -> ACI": "11--111", "w/o horizon gain scaling": "1110111",
        "w/o sigma floor": "1111011", "Gaussian likelihood": "1111101", "Level target": "1111110"}
    sym = {"1": r"\cmark", "0": r"\xmark", "-": "--"}
    L = [r"\begin{table*}[!t]\centering\caption{Component ablation.}\label{tab:ablation}",
         r"\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lccccccc" + "ccc" * 2 + "}\\toprule",
         r" & \multicolumn{7}{c}{Components} & \multicolumn{3}{c}{NIFTY~50} & \multicolumn{3}{c}{Beijing PM2.5}\\",
         r"\cmidrule(lr){2-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
         r"Variant & Norm. & Online & Int. & $h$-scale & Floor & t-NLL & Log & CovDev & relIS & WRD & CovDev & relIS & WRD\\\midrule"]
    vals = {}
    for d, a in A.items():
        m = a["metrics_abl"]
        vals[d] = {k: (m[m.method == k].CovDev.mean(), m[m.method == k].relIS.mean(),
                       m[m.method == k].worst_regime_dev.mean()) for k in ABL}
    mk = {d: [mark([vals[d][k][j] for k in ABL]) for j in range(3)] for d in A}
    for i, k in enumerate(ABL):
        cells = [{**TEX, PROPOSED: PROPOSED + " (full)"}.get(k, k)] + [sym[c] for c in comp[k]]
        for d in A:
            cells += [fmt(vals[d][k][j], mk[d][j][i]) for j in range(3)]
        L.append(" & ".join(cells) + r"\\")
    L += [r"\bottomrule\end{tabular}",
          note(r"Each variant makes one change to the proposed method; every other setting is fixed to the validation-selected configuration. Columns are averaged over all horizons, levels and five seeds. "
               r"Norm.: dispersion-normalised score; Online: online update; Int.: integral term; $h$-scale: horizon gain scaling; Floor: $\sigma$ floor; t-NLL: Student-$t$ likelihood; Log: log-increment target. "
               r"Best in bold, second underlined, " + RANK + r". Because these averages include the 80\% and 95\% levels, the full-model values differ slightly from the 90\% values in Table~\ref{tab:main}."),
          r"\end{table*}"]
    write("ablation", "\n".join(L))


def table_featlook(A):
    L = [r"\begin{table}[!t]\centering\caption{Input-feature and look-back ablations.}\label{tab:featlook}",
         r"\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{llccc}\toprule",
         r"Data & Variant & CovDev & relIS & WRD\\\midrule"]
    names = {"F_own": "own history", "F_own+vix": "+ India VIX", "F_own+poll": "+ co-pollutants"}
    for d, a in A.items():
        f = a["featlook"]; cfg = a["cfg"]
        full_lbl = "+ global markets (full)" if d == "nifty" else "+ meteorology, calendar (full)"
        order = [k for k in ["F_own", "F_own+vix", "F_own+poll"] if k in set(f.method)] + ["FULL"]
        rows = [(names.get(k, full_lbl if k == "FULL" else k), f[f.method == k]) for k in order]
        Ls = [f"L{x}" for x in cfg["lookback_grid"] if x != cfg["seq_len"]]
        rows += [(f"look-back $L={k[1:]}$", f[f.method == k]) for k in Ls if k in set(f.method)]
        rows.insert(len(order) + sum(1 for k in Ls if k in set(f.method) and int(k[1:]) < cfg["seq_len"]),
                    (f"look-back $L={cfg['seq_len']}$ (default)", f[f.method == "FULL"]))
        v = [(r.CovDev.mean(), r.relIS.mean(), r.worst_regime_dev.mean()) for _, r in rows]
        mk = [mark([x[j] for x in v]) for j in range(3)]
        for i, (lbl, _) in enumerate(rows):
            L.append((DS[d] if i == 0 else "") + " & " + lbl + " & " + " & ".join(fmt(v[i][j], mk[j][i]) for j in range(3)) + r"\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule\end{tabular}"
    L += [note(r"The calibrator of the proposed method is re-tuned on validation for each backbone. Averages over all horizons, levels and seeds. Best in bold, second underlined, " + RANK + "."),
          r"\end{table}"]
    write("featlook", "\n".join(L))


def table_runtime(A):
    L = [r"\begin{table}[!t]\centering\caption{Computational cost.}\label{tab:runtime}",
         r"\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lcccc}\toprule",
         r"Component & \multicolumn{2}{c}{NIFTY~50} & \multicolumn{2}{c}{PM2.5}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r" & Params & Time & Params & Time\\\midrule"]
    def mm(a, k, f):
        return np.mean([f(m[k]) for m in a["meta"] if k in m])
    items = [("LSTM backbone, training (s)", "BT", lambda m: m["train_seconds"]),
             ("QR-LSTM, training (s)", "QR", lambda m: m["train_seconds"]),
             ("DeepPIPE, training (s)", "DPr", lambda m: m["train_seconds"])]
    for lbl, k, f in items:
        cells = [lbl]
        for d, a in A.items():
            cells += [f"{int(mm(a, k, lambda m: m['n_params'])):,}", f"{mm(a, k, f):.0f}"]
        L.append(" & ".join(cells) + r"\\")
    cells = ["Backbone inference (ms / origin)"]
    for d, a in A.items():
        cells += ["", f"{1e3 * mm(a, 'BT', lambda m: m['infer_seconds_test'] / m['n_test']):.3f}"]
    L.append(" & ".join(cells) + r"\\")
    cells = [PROPOSED + r" calibrator ($\mu$s / step)"]
    for d, a in A.items():
        cells += ["0", f"{1e6 * np.mean(a['calib_sec']):.1f}"]
    L.append(" & ".join(cells) + r"\\")
    cells = ["Validation tuning of calibrators (s)"]
    for d, a in A.items():
        cells += ["", f"{np.mean([t['tune_seconds'] for t in a['tuned']]):.0f}"]
    L.append(" & ".join(cells) + r"\\")
    cells = ["GARCH(1,1)-t fit (s)"]
    for d, a in A.items():
        cells += ["5", f"{a['garch']['seconds']:.1f}"]
    L.append(" & ".join(cells) + r"\\")
    L += [r"\bottomrule\end{tabular}",
          note("Apple M3 Pro CPU, two threads per run; mean over seeds. Calibration cost is per forecast origin, horizon and level."),
          r"\end{table}"]
    write("runtime", "\n".join(L))


def table_hyper(A):
    def mode(lst):
        s = pd.Series([json.dumps(x, sort_keys=True) for x in lst]).value_counts()
        return json.loads(s.index[0]), int(s.iloc[0])
    L = [r"\begin{table}[!t]\centering\caption{Hyperparameters.}\label{tab:hyper}",
         r"\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}p{2.5cm}p{2.9cm}p{2.9cm}@{}}\toprule", r"Setting & NIFTY~50 & PM2.5\\\midrule"]
    a1, a2 = A.get("nifty"), A.get("pm25")
    def r(lbl, f):
        L.append(lbl + " & " + " & ".join(f(a) if a is not None else "--" for a in (a1, a2)) + r"\\")
    r("Look-back $L$", lambda a: str(a["cfg"]["seq_len"]))
    r("LSTM layers / hidden", lambda a: f"{a['cfg']['backbone']['layers']} / {a['cfg']['backbone']['hidden']}")
    r("Dropout / weight decay", lambda a: f"{a['cfg']['backbone']['dropout']} / {a['cfg']['backbone']['weight_decay']}")
    r("Batch / max epochs / patience", lambda a: f"{a['cfg']['backbone']['batch']} / {a['cfg']['backbone']['epochs']} / {a['cfg']['backbone']['patience']}")
    r("Learning rate (AdamW)", lambda a: f"{a['cfg']['backbone']['lr']}")
    r(r"Student-$t$ d.o.f. $\nu$", lambda a: f"{a['cfg']['backbone']['df']}")
    r(r"$\sigma$ floor quantile", lambda a: "0.10")
    for key, lbl in (("vacp", PROPOSED), ("cp_pid", "CP-PID"), ("aci", "ACI")):
        def f(a, key=key):
            cfg, n = mode([t[key] for t in a["tuned"]])
            return ", ".join(f"{k}={v}" for k, v in cfg.items()).replace("eta", r"$\eta$").replace("ki", r"$k_I$").replace("kappa", r"$\kappa$").replace("gamma", r"$\gamma$") + f" ({n}/{len(a['tuned'])})"
        r(lbl + " gains", f)
    r("Bootstrap block length", lambda a: str(a["cfg"]["bootstrap_block"]))
    r("Seeds", lambda a: ", ".join(map(str, a["cfg"]["seeds"])))
    L += [r"\bottomrule\end{tabular}",
          note("Calibrator gains were selected on the second half of the validation stream (grid in the text). The most frequent choice over seeds is shown with its count."),
          r"\end{table}"]
    write("hyper", "\n".join(L))


def table_utility(A):
    L = [r"\begin{table}[!t]\centering\caption{Practical utility of the intervals.}\label{tab:utility}",
         r"\scriptsize\setlength{\tabcolsep}{2.2pt}"]
    if "nifty" in A:
        u = A["nifty"]["util"]; g = u.groupby("method").mean(numeric_only=True)
        keep = ["Buy and hold", "Naive-Vol", "GARCH-t", "QR-LSTM", "CQR", "LW-CP", "ACI", "CP-PID", PROPOSED]
        L.append(r"\resizebox{\columnwidth}{!}{\begin{tabular}{lcccccc}\toprule Method & VaR$_{5\%}$ rate ($p$) & VaR$_{2.5\%}$ rate ($p$) & Sharpe & Ann. ret. & MDD & Expo.\\\midrule")
        for k in keep:
            if k not in g.index:
                continue
            r = g.loc[k]
            v1 = "--" if np.isnan(r.get("VaR0.050_rate", np.nan)) else f"{r['VaR0.050_rate']:.3f} ({r['VaR0.050_kupiec_p']:.2f})"
            v2 = "--" if np.isnan(r.get("VaR0.025_rate", np.nan)) else f"{r['VaR0.025_rate']:.3f} ({r['VaR0.025_kupiec_p']:.2f})"
            L.append(f"{k} & {v1} & {v2} & {r.sharpe:.2f} & {100*r.ann_ret:.1f}\\% & {100*r.mdd:.1f}\\% & {r.avg_expo:.2f}\\\\")
        L.append(r"\bottomrule\end{tabular}}\\[4pt]")
    if "pm25" in A:
        u = A["pm25"]["util"]; keep = ["Point forecast", "Naive-Vol", "GARCH-t", "QR-LSTM", "CQR", "LW-CP", "ACI", "CP-PID", PROPOSED]
        hs = sorted(u.h.unique())
        L.append(r"\begin{tabular}{l" + "ccc" * len(hs) + "}\\toprule")
        L.append(" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$h={h}$}}" for h in hs) + r"\\")
        L.append("Method & " + " & ".join(["POD & FAR & PSS"] * len(hs)) + r"\\\midrule")
        g = u.groupby(["method", "h"]).mean(numeric_only=True)
        pss = {h: mark([g.loc[(k, h), "PSS"] if (k, h) in g.index else None for k in keep], lower=False) for h in hs}
        for i, k in enumerate(keep):
            cells = [k]
            for h in hs:
                if (k, h) not in g.index:
                    cells += ["--"] * 3; continue
                r = g.loc[(k, h)]
                cells += [f"{r.POD:.2f}", f"{r.FAR:.2f}", fmt(r.PSS, pss[h][i], 2)]
            L.append(" & ".join(cells) + r"\\")
        L.append(r"\bottomrule\end{tabular}")
    L.append(note(r"Top (NIFTY~50): lower bound of the $h=1$ interval used as Value-at-Risk at 5\% (90\% interval) and 2.5\% (95\% interval), with Kupiec $p$-values, and a causal de-risking rule driven by the 90\% width (5~bp cost per unit turnover). "
                  r"Bottom (PM2.5): alerts when the upper 90\% bound exceeds 75~$\mu$g/m$^3$; POD: probability of detection; FAR: false-alarm ratio; PSS: Peirce skill score (best in bold, second underlined, " + RANK + "). Seed means."))
    L.append(r"\end{table}")
    write("utility", "\n".join(L))


def table_mechanism(A):
    """Score homogeneity: ratio of the 90% quantile of scores in turbulent vs calm origins."""
    L = [r"\begin{table}[!t]\centering\caption{Turbulent-to-calm ratio of the nonconformity score.}\label{tab:mechanism}",
         r"\scriptsize\begin{tabular}{lcc}\toprule Score & NIFTY~50 & PM2.5\\\midrule"]
    vals = {}
    for d, a in A.items():
        reg = a["reg"]; raw, nor = [], []
        for mch in a["mech"]:
            s_raw = np.abs(mch["y"] - mch["m"]); s_n = s_raw / np.maximum(mch["sigma"], np.quantile(mch["sigma"], .1))
            q = lambda s, g: np.quantile(s[reg == g], .9)
            raw.append(q(s_raw, 2) / q(s_raw, 0)); nor.append(q(s_n, 2) / q(s_n, 0))
        vals[d] = (raw, nor)
    for j, lbl in enumerate([r"Plain $|y-\hat m|$", r"Normalised $|y-\hat m|/\hat\sigma$"]):
        L.append(lbl + " & " + " & ".join(f"{np.mean(vals[d][j]):.2f} [{np.min(vals[d][j]):.2f}, {np.max(vals[d][j]):.2f}]" for d in A) + r"\\")
    L += [r"\bottomrule\end{tabular}",
          note(r"Ratio of the 0.9-quantile of the $h=1$ score in the turbulent tercile to that in the calm tercile (1 = exchangeable across regimes). Mean over seeds, with the range in brackets."),
          r"\end{table}"]
    write("mechanism", "\n".join(L))
    return vals


def friedman(A):
    blocks, names = [], [k for k in MAIN if not k.startswith("DeepPIPE")]
    for d, a in A.items():
        m = a["metrics"]
        piv = m.groupby(["method", "h", "level"]).relIS.mean().unstack("method")[names]
        blocks.append(piv.to_numpy())
    S = np.vstack(blocks)
    res = St.friedman_nemenyi(S, names)
    S2 = []
    for d, a in A.items():
        m = a["metrics"]
        S2.append(m.groupby(["method", "h", "level"]).CovDev.mean().unstack("method")[names].to_numpy())
    res_cd = St.friedman_nemenyi(np.vstack(S2), names)
    return res, res_cd, S.shape[0]


# ======================================================================= figures
def fig_data(A):
    fig, axs = plt.subplots(2, 1, figsize=(7.1, 3.6))
    for ax, (d, a) in zip(axs, A.items()):
        dt = pd.to_datetime(a["dates"]); s = a["series"]
        if d == "pm25":
            s = s - 1
        ax.plot(dt, s, lw=.4, color="#2c3e50")
        t_te = dt[a["t0"][0]]; t_va = dt[a["t_train_end"] + 1]
        ax.axvspan(t_va, t_te, color="#f9e79f", alpha=.5, label="validation")
        ax.axvspan(t_te, dt[-1], color="#f5b7b1", alpha=.5, label="test")
        ax.set_ylabel("NIFTY 50 close" if d == "nifty" else r"PM2.5 ($\mu$g/m$^3$)")
        ax2 = ax.twinx(); ax2.plot(dt, a["scale_full"], lw=.3, color="#c0392b", alpha=.6)
        ax2.set_ylabel("India VIX" if d == "nifty" else "48-h realised vol.", color="#c0392b"); ax2.grid(False)
        ax.legend(loc="upper left")
    fig.tight_layout(); save(fig, "fig_data")


def fig_calgap(A):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(wspace=0.05))
    for ai, (ax, (d, a)) in enumerate(zip(axs, A.items())):
        m = a["metrics"]; meths = [k for k in MAIN if m[m.method == k].h.nunique() > 1]
        piv = m.groupby(["method", "h", "level"]).PICP.mean().reset_index()
        piv["gap"] = piv.PICP - piv.level
        M = np.array([[piv[(piv.method == k) & (piv.h == h) & np.isclose(piv.level, l)].gap.iloc[0]
                       for l in a["L"] for h in a["H"]] for k in meths])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-.08, vmax=.08, aspect="auto")
        ax.set_yticks(range(len(meths))); ax.set_yticklabels(meths if ai == 0 else [])
        ax.set_xticks(range(len(a["L"]) * len(a["H"])))
        ax.set_xticklabels([f"{h}" for l in a["L"] for h in a["H"]])
        for j in range(1, len(a["L"])):
            ax.axvline(j * len(a["H"]) - .5, color="k", lw=.8)
        ax.set_xlabel("horizon $h$  (blocks: 80% | 90% | 95%)"); ax.set_title(DS[d]); ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{100*M[i,j]:+.0f}", ha="center", va="center", fontsize=4.2,
                        color="w" if abs(M[i, j]) > .05 else "k")
    fig.colorbar(im, ax=axs, shrink=.8, label="PICP $-$ nominal (cell text: percentage points)")
    save(fig, "fig_calgap")


def fig_reliability(A):
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for ax, (d, a) in zip(axs, A.items()):
        ci = a["ci"]; c1 = ci[ci.h == 1]
        ax.plot([.75, 1], [.75, 1], "k--", lw=.8)
        for k in ["Naive-Vol", "GARCH-t", "QR-LSTM", "LW-CP", "ACI", "CP-PID", PROPOSED]:
            r = c1[c1.method == k].sort_values("level")
            ax.errorbar(r.level + (0.003 if k == PROPOSED else 0), r.PICP,
                        yerr=[r.PICP - r.PICP_lo, r.PICP_hi - r.PICP], marker="o", ms=3,
                        lw=1.6 if k == PROPOSED else .8, capsize=2, color=COL[k], label=k)
        ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage ($h=1$)")
        ax.set_title(DS[d]); ax.set_xlim(.77, .97); ax.set_ylim(.72, 1.0)
    axs[0].legend(ncol=2)
    fig.tight_layout(); save(fig, "fig_reliability")


def fig_pareto(A):
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for ax, (d, a) in zip(axs, A.items()):
        m = a["metrics"]; m9 = m[np.isclose(m.level, .9)]
        for k in [k for k in MAIN if m9[m9.method == k].h.nunique() > 1]:
            g = m9[m9.method == k].groupby("seed")[["CovDev", "relIS"]].mean()
            ax.scatter(g.CovDev, g.relIS, s=8, color=COL[k], alpha=.5)
            ax.scatter(g.CovDev.mean(), g.relIS.mean(), s=40, color=COL[k], edgecolor="k",
                       marker="*" if k == PROPOSED else "o", label=k, zorder=3)
        ax.set_xlabel("mean coverage deviation (90%)"); ax.set_ylabel("mean relIS (lower is better)")
        ax.set_title(DS[d])
        if d == "pm25":
            ax.set_yscale("log"); ax.set_ylabel("mean relIS (log scale)")
    axs[1].legend(ncol=2, loc="upper right")
    fig.tight_layout(); save(fig, "fig_pareto")


def fig_regime(A):
    keep = ["Naive-Vol", "GARCH-t", "QR-LSTM", "LW-CP", "ACI", "CP-PID", PROPOSED]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.7))
    for ax, (d, a) in zip(axs, A.items()):
        ci = a["ci"]; c1 = ci[(ci.h == 1) & np.isclose(ci.level, .9)].set_index("method")
        w = .11
        for j, k in enumerate(keep):
            x = np.arange(3) + (j - len(keep) / 2) * w
            v = [c1.loc[k, g] for g in ["calm", "moderate", "turbulent"]]
            e = [[c1.loc[k, g] - c1.loc[k, g + "_lo"] for g in ["calm", "moderate", "turbulent"]],
                 [c1.loc[k, g + "_hi"] - c1.loc[k, g] for g in ["calm", "moderate", "turbulent"]]]
            ax.bar(x, v, w, yerr=e, color=COL[k], label=k, capsize=1.2, error_kw=dict(lw=.6))
        ax.axhline(.9, color="k", ls="--", lw=.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["calm", "moderate", "turbulent"])
        ax.set_ylim(.7, 1.0); ax.set_ylabel("PICP ($h=1$, 90%)"); ax.set_title(DS[d])
    axs[0].legend(ncol=2, fontsize=6)
    fig.tight_layout(); save(fig, "fig_regime")


def fig_width_scale(A):
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.6))
    for ax, (d, a) in zip(axs, A.items()):
        sc = a["scale"]; bins = np.quantile(sc, np.linspace(0, 1, 11)); b = np.clip(np.digitize(sc, bins[1:-1]), 0, 9)
        ex = {PROPOSED: a["vacp_example"], "ACI": a["aci_example"], "CP-PID": a["pid_example"],
              "Naive-Vol": a["nv"]}
        for k, dct in ex.items():
            lo, up = dct[(1, .9)]; rw = (up - lo) / a["P0"]
            ax.plot(range(10), [rw[b == i].mean() for i in range(10)], marker="o", ms=3, color=COL[k], label=k)
        ax.set_xlabel("scale-signal decile at origin"); ax.set_ylabel("relative width ($h=1$, 90%)")
        ax.set_title(DS[d])
    axs[0].legend()
    fig.tight_layout(); save(fig, "fig_width_scale")


def fig_mechanism(A):
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 2.2))
    for j, (d, a) in enumerate(A.items()):
        mch = a["mech"][0]; reg = a["reg"]
        s_raw = np.abs(mch["y"] - mch["m"]); s_n = s_raw / np.maximum(mch["sigma"], np.quantile(mch["sigma"], .1))
        for i, (s, lbl) in enumerate(((s_raw, "plain score"), (s_n, "normalised score"))):
            ax = axs[2 * j + i]
            ax.boxplot([s[reg == g] / np.median(s) for g in range(3)], showfliers=False,
                       tick_labels=["calm", "mod.", "turb."], widths=.5)
            ax.set_title(f"{DS[d]}: {lbl}", fontsize=7); ax.set_ylabel("score / median" if i == 0 else "")
    fig.tight_layout(); save(fig, "fig_mechanism")


def fig_ablation(A):
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    for ax, (d, a) in zip(axs, A.items()):
        m = a["metrics_abl"]
        base = m[m.method == PROPOSED]
        dv = [(m[m.method == k].relIS.mean() - base.relIS.mean()) * 100 for k in ABL[1:]]
        dc = [(m[m.method == k].worst_regime_dev.mean() - base.worst_regime_dev.mean()) * 100 for k in ABL[1:]]
        y = np.arange(len(ABL) - 1)
        ax.barh(y - .18, dv, .36, color="#2471a3", label=r"$\Delta$ relIS ($\times 100$)")
        ax.barh(y + .18, dc, .36, color="#c0392b", label=r"$\Delta$ WRD (pp)")
        ax.axvline(0, color="k", lw=.8); ax.set_yticks(y); ax.set_yticklabels(ABL[1:])
        ax.set_title(f"{DS[d]} (positive = worse than full model)")
        lim = max(5, np.nanpercentile(np.abs(dv + dc), 90) * 1.3); ax.set_xlim(-lim, lim)
        for yi, v in zip(y, dv):
            if abs(v) > lim:
                ax.text(np.sign(v) * lim * .95, yi - .18, f"{v:.0f}", va="center", ha="right" if v > 0 else "left", fontsize=6)
    axs[0].legend(loc="lower left")
    fig.tight_layout(); save(fig, "fig_ablation")


def fig_sensitivity(A):
    params = [("eta", r"$\eta$"), ("ki", r"$k_I$"), ("kappa", r"$\kappa$"), ("sigma_floor_q", r"$\sigma$-floor quantile")]
    fig, axs = plt.subplots(2, 4, figsize=(7.1, 3.4))
    for c, (p, lbl) in enumerate(params):
        for d, a in A.items():
            s = a["sens"]; s = s[s.param == p]
            per = s.groupby(["value", "seed"])[["relIS", "CovDev"]].mean().reset_index()
            for r, col in enumerate(["relIS", "CovDev"]):
                g = per.groupby("value")[col]
                mu = g.mean(); hw = g.apply(lambda v: St.t_ci(v)[1])
                ax = axs[r, c]
                ax.plot(mu.index, mu.values, marker="o", ms=3, label=DS[d])
                ax.fill_between(mu.index, mu - hw, mu + hw, alpha=.2)
                tuned = pd.Series([t["vacp"].get(p, .1 if p == "sigma_floor_q" else np.nan) for t in a["tuned"]]).mode()
                if len(tuned):
                    ax.axvline(tuned.iloc[0], ls=":", lw=.7, color=ax.lines[-1].get_color())
                ax.set_xlabel(lbl); ax.set_ylabel(col if c == 0 else "")
                if p in ("eta",):
                    ax.set_xscale("log")
    axs[0, 0].legend()
    fig.tight_layout(); save(fig, "fig_sensitivity")


def fig_dynamics(A):
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.0))
    for j, (d, a) in enumerate(A.items()):
        ax = axs[0, j]
        for i, m in enumerate(a["meta"]):
            h = np.array(m["BT"]["history"])
            ax.plot(h[:, 0], h[:, 1], color="#2471a3", lw=.7, alpha=.7, label="train" if i == 0 else None)
            ax.plot(h[:, 0], h[:, 2], color="#c0392b", lw=.7, alpha=.7, label="validation" if i == 0 else None)
        ax.set_xlabel("epoch"); ax.set_ylabel("Student-$t$ NLL"); ax.set_title(f"{DS[d]}: backbone training")
        ax.legend()
        ax = axs[1, j]
        tr = a["traces"][0]; H = a["H"]
        for h, c in ((H[0], "#c0392b"), (H[-1], "#2471a3")):
            R = tr[(h, .9)]
            ax.plot(R / R[0], lw=.7, color=c, label=f"radius $q_t/q_0$, $h={h}$")
        lo, up = a["vacp_example"][(1, .9)]; y = a["y"][:, 0]
        cov = pd.Series(((y >= lo) & (y <= up)).astype(float)).rolling(100 if d == "nifty" else 500).mean()
        ax2 = ax.twinx(); ax2.plot(cov.values, color="k", lw=.6, label="rolling PICP ($h=1$)")
        ax2.axhline(.9, ls="--", lw=.6, color="k"); ax2.set_ylim(.7, 1.0); ax2.grid(False)
        ax.set_xlabel("test origin"); ax.set_title(f"{DS[d]}: online controller (90%)")
        ax.legend(loc="lower left", fontsize=6); ax2.legend(loc="lower right", fontsize=6)
    fig.tight_layout(); save(fig, "fig_dynamics")


def fig_cd(res, title, name):
    ranks = res["avg_ranks"]; cd = res["cd"]
    items = sorted(ranks.items(), key=lambda x: x[1]); k = len(items)
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    lo, hi = 1, k
    ax.set_xlim(lo - .3, hi + .3); ax.set_ylim(0, 1); ax.axis("off")
    ax.hlines(.85, lo, hi, color="k", lw=.8)
    for t in range(lo, hi + 1):
        ax.vlines(t, .83, .87, color="k", lw=.8); ax.text(t, .9, str(t), ha="center", fontsize=6)
    half = (k + 1) // 2
    for i, (nm, r) in enumerate(items):
        left = i < half
        yy = .75 - (i if left else k - 1 - i) * .085
        xx = lo - .2 if left else hi + .2
        ax.plot([r, r, xx], [.85, yy, yy], color=COL.get(nm, "k"), lw=.7)
        ax.text(xx, yy, f"{nm} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=6,
                fontweight="bold" if nm == PROPOSED else None)
    ax.hlines(.97, lo, lo + cd, color="k", lw=1.5); ax.text(lo + cd / 2, .99, f"CD = {cd:.2f}", ha="center", fontsize=6)
    ax.set_title(title, fontsize=7, y=1.05)
    save(fig, name)


def fig_band(A):
    fig, axs = plt.subplots(2, 1, figsize=(7.1, 3.6))
    for ax, (d, a) in zip(axs, A.items()):
        y = a["y"][:, 0]; n = len(y); reg = a["reg"]
        # window of 120 origins with the largest share of turbulent origins
        w = 120 if d == "nifty" else 240
        sc = pd.Series((reg == 2).astype(float)).rolling(w).mean().to_numpy()
        e = int(np.nanargmax(sc)); s0 = e - w + 1; sl = slice(s0, e + 1); x = np.arange(s0, e + 1)
        sub = (lambda v: v - 1) if d == "pm25" else (lambda v: v)
        for k, dct, c in (("ACI", a["aci_example"], COL["ACI"]), (PROPOSED, a["vacp_example"], COL[PROPOSED])):
            lo, up = dct[(1, .9)]
            ax.fill_between(x, sub(lo[sl]), sub(up[sl]), color=c, alpha=.25 if k == PROPOSED else .18, label=f"{k} 90% interval")
        ax.plot(x, sub(y[sl]), color="k", lw=.8, label="observed")
        ax.set_title(f"{DS[d]}: most turbulent {w}-step test window, $h=1$"); ax.legend(ncol=3, fontsize=6)
    fig.tight_layout(); save(fig, "fig_band")


def main():
    A = {d: load(d) for d in ("nifty", "pm25") if os.path.exists(os.path.join(OUT, d, "analysis.pkl"))}
    diag = table_datasets(A)
    table_main(A); table_horizon(A); table_tests(A); table_regime(A); table_ablation(A)
    table_featlook(A); table_runtime(A); table_hyper(A); table_utility(A)
    mech = table_mechanism(A)
    summary = {"diagnostics": diag, "mechanism": {d: dict(raw=v[0], norm=v[1]) for d, v in mech.items()}}
    if len(A) == 2:
        res, res_cd, nb = friedman(A)
        fig_cd(res, f"Mean rank of relIS over {nb} (dataset, $h$, level) cells", "fig_cd")
        fig_cd(res_cd, f"Mean rank of coverage deviation over {nb} cells", "fig_cd_covdev")
        summary["friedman_relIS"] = res; summary["friedman_covdev"] = res_cd
    for f in (fig_data, fig_calgap, fig_reliability, fig_pareto, fig_regime, fig_width_scale,
              fig_mechanism, fig_ablation, fig_sensitivity, fig_dynamics, fig_band):
        f(A)
    with open(os.path.join(TAB, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print("tables ->", TAB, "\nfigures ->", FIG)


if __name__ == "__main__":
    main()
