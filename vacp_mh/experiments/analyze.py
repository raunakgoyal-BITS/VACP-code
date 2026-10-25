"""Calibrate, evaluate and test every method for one dataset across all seeds.

Usage: python experiments/analyze.py --data nifty
Writes outputs/<data>/analysis.pkl and CSV tables under outputs/<data>/tables/.
"""
from __future__ import annotations
import argparse, json, os, pickle, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy import stats

from common import load_cfg, load_dataset, OUT
from vacp_mh import stats as St
from vacp_mh.data import make_returns
import methods as M

# Display name of the proposed method in every table, figure and saved result.
PROPOSED = "Proposed"
LEGACY_NAMES = {"VACP-MH": PROPOSED}  # name used in analysis.pkl files written before Oct 2026

MAIN = ["Naive-Vol", "GARCH-t", "Prob-LSTM", "QR-LSTM", "CQR", "Split-CP", "LW-CP",
        "ACI", "CP-PID", "DeepPIPE-P", "DeepPIPE-R", PROPOSED]
GROUP = {"Naive-Vol": "Heuristic", "GARCH-t": "Heuristic", "Prob-LSTM": "Probabilistic",
         "QR-LSTM": "Quantile regression", "CQR": "Quantile regression",
         "Split-CP": "Static conformal", "LW-CP": "Static conformal",
         "ACI": "Online conformal", "CP-PID": "Online conformal",
         "DeepPIPE-P": "Interval learning", "DeepPIPE-R": "Interval learning",
         PROPOSED: "This work"}
ABL = [PROPOSED, "w/o normalisation", "w/o online update", "w/o integral term",
       "PI -> ACI", "w/o horizon gain scaling", "w/o sigma floor",
       "Gaussian likelihood", "Level target"]
SENS = {"eta": [0.005, 0.01, 0.02, 0.05, 0.1], "ki": [0.0, 0.005, 0.01, 0.02],
        "kappa": [0.0, 0.5, 1.0], "sigma_floor_q": [0.0, 0.05, 0.10, 0.20]}


# ============================================================== per-seed construction
def load_seed(data, seed, part="main"):
    p = os.path.join(OUT, data, f"seed{seed}", f"{part}.npz")
    if not os.path.exists(p):
        return None, None
    z = dict(np.load(p))
    with open(p.replace(".npz", "_meta.json")) as f:
        meta = json.load(f)
    return z, meta


def streams(z, key, lvl_stats=None, target="logret"):
    mk = lambda sp: M.Stream(z[f"{sp}/R"], z[f"{sp}/P0"], z[f"{key}/{sp}/m"],
                             z.get(f"{key}/{sp}/sigma"), z.get(f"{key}/{sp}/q"),
                             target, lvl_stats)
    return mk("val"), mk("test")


def streams_from(zmain, zpart, key):
    mk = lambda sp: M.Stream(zmain[f"{sp}/R"], zmain[f"{sp}/P0"], zpart[f"{key}/{sp}/m"],
                             zpart.get(f"{key}/{sp}/sigma"), None)
    return mk("val"), mk("test")


def ref_naive(ctx, cfg, val):
    """Naive-Vol interval score on the second half of the validation stream (tuning ref)."""
    n = len(val.m); half = n // 2
    t0 = ctx["t0_val"][half:]; P0 = val.P0[half:]
    nv = M.naive_vol(ctx["r"], t0, P0, cfg["horizons"], cfg["levels"], ctx["nv_window"])
    ref = {}
    for hi, h in enumerate(cfg["horizons"]):
        for lvl in cfg["levels"]:
            lo, up = nv[(h, lvl)]
            ref[(h, lvl)] = M.interval_score(val.y_level[half:, hi], lo, up, lvl).mean()
    return ref


def run_seed(args):
    data, seed, cfg, ctx = args
    H, L = cfg["horizons"], cfg["levels"]
    z, meta = load_seed(data, seed)
    if z is None:
        return None
    lvl_stats = tuple(z["lvl_stats"])
    vT, eT = streams(z, "BT")
    ref = ref_naive(ctx, cfg, vT)
    res = {"methods": {}, "abl": {}, "sens": {}, "tuned": {}, "traces": {}, "meta": meta}
    t = time.perf_counter()
    vacp_cfg, rows_v = M.tune(vT, H, L, M.GRID_PI, ref, dict(M.VACP_DEFAULT))
    pid_cfg, rows_p = M.tune(vT, H, L, M.GRID_PI, ref, dict(method="pi", normalized=False))
    aci_cfg, rows_a = M.tune(vT, H, L, M.GRID_ACI, ref, dict(method="aci", normalized=False))
    res["tuned"] = dict(vacp=vacp_cfg, cp_pid=pid_cfg, aci=aci_cfg, tune_seconds=time.perf_counter() - t,
                        grid_vacp=rows_v)
    VK = {**M.VACP_DEFAULT, **vacp_cfg}
    m = res["methods"]
    t = time.perf_counter()
    m[PROPOSED], tr = M.conformal_method(vT, eT, H, L, track=True, **VK)
    res["calib_seconds_per_step"] = (time.perf_counter() - t) / (len(eT.m) * len(H) * len(L))
    res["traces"] = {k: v for k, v in tr.items() if k[1] == 0.90}
    m["CP-PID"] = M.conformal_method(vT, eT, H, L, method="pi", normalized=False, **pid_cfg)
    m["ACI"] = M.conformal_method(vT, eT, H, L, method="aci", normalized=False, **aci_cfg)
    m["Split-CP"] = M.conformal_method(vT, eT, H, L, static=True, normalized=False)
    m["LW-CP"] = M.conformal_method(vT, eT, H, L, static=True, normalized=True, sigma_floor_q=0.10)
    m["Prob-LSTM"] = M.prob_lstm(eT, H, L, df=cfg["backbone"]["df"])
    vQ, eQ = streams(z, "QR")
    m["QR-LSTM"] = M.qr_lstm(eQ, H, L)
    m["CQR"] = M.cqr(vQ, eQ, H, L)
    for k, name in (("DPp", "DeepPIPE-P"), ("DPr", "DeepPIPE-R")):
        m[name] = {(1, 0.90): (z[f"{k}/test/lo"].astype(float), z[f"{k}/test/hi"].astype(float))}
    # ---------------- component ablation (one change at a time from the proposed method)
    a = res["abl"]
    a[PROPOSED] = m[PROPOSED]
    a["w/o normalisation"] = M.conformal_method(vT, eT, H, L, **{**VK, "normalized": False})
    a["w/o online update"] = m["LW-CP"]
    a["w/o integral term"] = M.conformal_method(vT, eT, H, L, **{**VK, "ki": 0.0})
    a["PI -> ACI"] = M.conformal_method(vT, eT, H, L, method="aci", normalized=True,
                                        sigma_floor_q=0.10, **aci_cfg)
    a["w/o horizon gain scaling"] = M.conformal_method(vT, eT, H, L, **{**VK, "kappa": 0.0})
    a["w/o sigma floor"] = M.conformal_method(vT, eT, H, L, **{**VK, "sigma_floor_q": 0.0})
    vG, eG = streams(z, "BG")
    a["Gaussian likelihood"] = M.conformal_method(vG, eG, H, L, **VK)
    vL, eL = streams(z, "BL", lvl_stats, target="level")
    a["Level target"] = M.conformal_method(vL, eL, H, L, **VK)
    # ---------------- one-at-a-time sensitivity on the test stream (reported as such)
    for p, grid in SENS.items():
        for v in grid:
            res["sens"][(p, v)] = M.conformal_method(vT, eT, H, L, **{**VK, p: v})
    # ---------------- feature and look-back ablations (re-tuned on their own validation)
    res["feat"], res["look"] = {}, {}
    for part, dst in (("features", res["feat"]), ("lookback", res["look"])):
        zp, mp = load_seed(data, seed, part)
        if zp is None:
            continue
        keys = sorted({k.split("/")[0] for k in zp})
        for k in keys:
            v, e = streams_from(z, zp, k)
            c, _ = M.tune(v, H, L, M.GRID_PI, ref, dict(M.VACP_DEFAULT))
            dst[k] = M.conformal_method(v, e, H, L, **{**M.VACP_DEFAULT, **c})
        res["meta"].update({f"{part}:{k}": vv for k, vv in mp.items()})
    res["mech"] = dict(m=eT.m[:, 0], sigma=eT.sigma[:, 0], y=eT.y[:, 0])
    return res


# ============================================================== metrics
def cell_arrays(y, lo, up, lvl):
    cov = ((y >= lo) & (y <= up)).astype(float)
    return cov, M.interval_score(y, lo, up, lvl), up - lo


def regime_labels(scale):
    q = np.quantile(scale, [1 / 3, 2 / 3])
    return np.where(scale <= q[0], 0, np.where(scale <= q[1], 1, 2))


def metrics_table(store, y_lv, P0, scale, reg, H, L, nv_is):
    """store: {method: [per-seed {(h,l): (lo,hi)}]} -> long DataFrame (one row per seed)."""
    rows = []
    for name, seeds in store.items():
        for si, d in enumerate(seeds):
            for (h, lvl), (lo, up) in d.items():
                hi = H.index(h); y = y_lv[:, hi]
                cov, isc, w = cell_arrays(y, lo, up, lvl)
                picp = cov.mean(); pinaw = w.mean() / np.ptp(y)
                cwc = pinaw * (1 + (picp < lvl) * np.exp(-50 * (picp - lvl)))
                rw = w / P0
                row = dict(method=name, seed=si, h=h, level=lvl, PICP=picp, MPIW=w.mean(),
                           PINAW=pinaw, CovDev=abs(picp - lvl), IS=isc.mean(),
                           relIS=isc.mean() / nv_is[(h, lvl)], CWC=cwc,
                           rho_width_scale=stats.spearmanr(rw, scale).correlation)
                for g, nm in enumerate(["calm", "moderate", "turbulent"]):
                    row[f"PICP_{nm}"] = cov[reg == g].mean()
                    row[f"MPIW_{nm}"] = w[reg == g].mean()
                row["worst_regime_dev"] = max(abs(row[f"PICP_{k}"] - lvl)
                                              for k in ["calm", "moderate", "turbulent"])
                rows.append(row)
    return pd.DataFrame(rows)


def seed_avg_series(seeds, key, y, lvl):
    covs, iss, ws = [], [], []
    for d in seeds:
        if key not in d:
            return None
        c, s, w = cell_arrays(y, *d[key], lvl)
        covs.append(c); iss.append(s); ws.append(w)
    return np.mean(covs, 0), np.mean(iss, 0), np.mean(ws, 0), np.array(covs)


def boot_and_tests(store, y_lv, reg, H, L, block, ref=PROPOSED):
    """Bootstrap CIs (seed-averaged series), DM vs ref with Holm, Kupiec/Christoffersen."""
    n = y_lv.shape[0]
    idx = St.block_indices(n, block, 2000, seed=123)
    ci_rows, dm_rows, cov_rows = [], [], []
    for (h, lvl) in [(h, l) for h in H for l in L]:
        hi = H.index(h); y = y_lv[:, hi]
        series = {k: seed_avg_series(v, (h, lvl), y, lvl) for k, v in store.items()}
        series = {k: v for k, v in series.items() if v is not None}
        for k, (c, s, w, call) in series.items():
            pc, plo, phi = St.block_bootstrap_ci(c, idx=idx)
            ic, ilo, ihi = St.block_bootstrap_ci(s, idx=idx)
            row = dict(method=k, h=h, level=lvl, PICP=pc, PICP_lo=plo, PICP_hi=phi,
                       IS=ic, IS_lo=ilo, IS_hi=ihi)
            # regime coverage CIs (bootstrap over time, restricted to the regime)
            for g, nm in enumerate(["calm", "moderate", "turbulent"]):
                msk = (reg == g).astype(float)
                bm = (c[idx] * msk[idx]).sum(1) / np.maximum(msk[idx].sum(1), 1)
                row[f"{nm}"] = float(c[reg == g].mean())
                row[f"{nm}_lo"], row[f"{nm}_hi"] = np.quantile(bm, [0.025, 0.975])
            ci_rows.append(row)
            # coverage backtests: h=1 on the full stream; h>1 on every h-th origin
            hits_all = 1 - call                             # (seeds, n)
            ps_k, ps_ci, ps_cc = [], [], []
            for hits in hits_all:
                hh = hits[::h].astype(int)
                ps_k.append(St.kupiec_pof(hh, 1 - lvl)[1])
                cr = St.christoffersen(hh, 1 - lvl)
                ps_ci.append(cr["p_ind"]); ps_cc.append(cr["p_cc"])
            cov_rows.append(dict(method=k, h=h, level=lvl, kupiec_p=np.median(ps_k),
                                 christ_ind_p=np.median(ps_ci), christ_cc_p=np.median(ps_cc),
                                 kupiec_rej=int(np.sum(np.array(ps_k) < 0.05)),
                                 ind_rej=int(np.sum(np.array(ps_ci) < 0.05)),
                                 cc_rej=int(np.sum(np.array(ps_cc) < 0.05)),
                                 n_seeds=len(ps_k)))
        if ref in series:
            names, dms, ps = [], [], []
            for k, v in series.items():
                if k == ref:
                    continue
                dm, p = St.diebold_mariano(series[ref][1], v[1], h=h)
                names.append(k); dms.append(dm); ps.append(p)
            for k, dm, p, pa in zip(names, dms, ps, St.holm(ps)):
                dm_rows.append(dict(baseline=k, h=h, level=lvl, DM=dm, p=p, p_holm=pa))
    return pd.DataFrame(ci_rows), pd.DataFrame(dm_rows), pd.DataFrame(cov_rows)


# ============================================================== utility analyses
def finance_utility(store, y_lv, P0, r_next, sigma_target, cost=5e-4):
    """Volatility targeting with the 90% h=1 interval as the risk forecast, plus VaR tests."""
    z = stats.norm.ppf(0.95)
    rows = []

    def perf(ret, expo):
        turn = np.abs(np.diff(np.r_[expo[0], expo]))
        net = ret * expo - cost * turn
        eq = np.cumsum(net)
        mdd = float(np.max(np.maximum.accumulate(eq) - eq))
        sr = net.mean() / net.std() * np.sqrt(252)
        return dict(ann_ret=net.mean() * 252, ann_vol=net.std() * np.sqrt(252), sharpe=sr,
                    mdd=mdd, avg_expo=expo.mean(), turnover=turn.mean() * 252)
    rows.append(dict(method="Buy and hold", seed=0, **perf(r_next, np.ones_like(r_next))))
    for name, seeds in store.items():
        for si, d in enumerate(seeds):
            if (1, 0.90) not in d:
                continue
            lo, up = d[(1, 0.90)]
            s_imp = np.log(up / lo) / (2 * z)
            # causal de-risking: compare today's risk forecast with its own trailing
            # 250-day median (only past forecasts are used)
            ref = pd.Series(s_imp).shift(1).rolling(250, min_periods=20).median()
            ref = ref.fillna(pd.Series(s_imp).expanding().median().shift(1)).fillna(s_imp[0]).to_numpy()
            expo = np.minimum(1.0, ref / np.maximum(s_imp, 1e-12))
            row = dict(method=name, seed=si, **perf(r_next, expo))
            for lvl in (0.90, 0.95):
                if (1, lvl) in d:
                    hits = (y_lv[:, 0] < d[(1, lvl)][0]).astype(int)   # lower-tail VaR breach
                    row[f"VaR{(1-lvl)/2:.3f}_rate"] = hits.mean()
                    row[f"VaR{(1-lvl)/2:.3f}_kupiec_p"] = St.kupiec_pof(hits, (1 - lvl) / 2)[1]
                    row[f"VaR{(1-lvl)/2:.3f}_ind_p"] = St.christoffersen(hits, (1 - lvl) / 2)["p_ind"]
            rows.append(row)
    return pd.DataFrame(rows)


def alert_utility(store, y_lv, point_lv, H, thr=75.0, hs=(1, 6, 24), lvl=0.90):
    """Exceedance alerts: warn when the upper 90% bound exceeds the threshold."""
    rows = []

    def score(ev, al):
        tp = np.sum(ev & al); fp = np.sum(~ev & al); fn = np.sum(ev & ~al); tn = np.sum(~ev & ~al)
        pod = tp / max(tp + fn, 1); pofd = fp / max(fp + tn, 1)
        return dict(POD=pod, FAR=fp / max(tp + fp, 1), POFD=pofd, PSS=pod - pofd,
                    alert_rate=al.mean(), event_rate=ev.mean())
    for h in hs:
        if h not in H:
            continue
        hi = H.index(h); ev = (y_lv[:, hi] - 1) > thr
        for si, p in enumerate(point_lv):
            rows.append(dict(method="Point forecast", seed=si, h=h, **score(ev, (p[:, hi] - 1) > thr)))
        for name, seeds in store.items():
            for si, d in enumerate(seeds):
                if (h, lvl) in d:
                    rows.append(dict(method=name, seed=si, h=h,
                                     **score(ev, (d[(h, lvl)][1] - 1) > thr)))
    return pd.DataFrame(rows)


# ============================================================== main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    cfg = load_cfg(a.data); H, L = cfg["horizons"], cfg["levels"]
    df, s, groups, scale_full = load_dataset(cfg)
    r = make_returns(s)
    z0, _ = load_seed(a.data, cfg["seeds"][0])
    ctx = dict(r=r, t0_val=z0["val/t0"], nv_window=20 if a.data == "nifty" else 48)
    t0 = z0["test/t0"]; P0 = z0["test/P0"]; y_lv = P0[:, None] * np.exp(z0["test/R"])
    scale = scale_full[t0]; reg = regime_labels(scale)
    with Pool(a.procs) as pool:
        per_seed = [x for x in pool.map(run_seed, [(a.data, sd, cfg, ctx) for sd in cfg["seeds"]])
                    if x is not None]
    print(f"{a.data}: {len(per_seed)} seeds analysed")
    # deterministic baselines
    nv = M.naive_vol(r, t0, P0, H, L, ctx["nv_window"])
    t_train_end = int(z0["val/t0"][0]) - 1
    tg = time.perf_counter()
    ga, gparams = M.garch(r, t_train_end, t0, P0, H, L)
    garch_seconds = time.perf_counter() - tg
    nseed = len(per_seed)
    main_store = {k: [ps["methods"][k] for ps in per_seed] for k in per_seed[0]["methods"]}
    main_store["Naive-Vol"] = [nv] * nseed
    main_store["GARCH-t"] = [ga] * nseed
    main_store = {k: main_store[k] for k in MAIN}
    nv_is = {(h, l): M.interval_score(y_lv[:, H.index(h)], *nv[(h, l)], l).mean() for h in H for l in L}
    abl_store = {k: [ps["abl"][k] for ps in per_seed] for k in ABL}
    met = metrics_table(main_store, y_lv, P0, scale, reg, H, L, nv_is)
    met_abl = metrics_table(abl_store, y_lv, P0, scale, reg, H, L, nv_is)
    ci, dm, covt = boot_and_tests(main_store, y_lv, reg, H, L, cfg["bootstrap_block"])
    ci_abl, dm_abl, _ = boot_and_tests(abl_store, y_lv, reg, H, L, cfg["bootstrap_block"])
    # sensitivity
    sens_rows = []
    for key in per_seed[0]["sens"]:
        st = {f"{key[0]}={key[1]}": [ps["sens"][key] for ps in per_seed]}
        mt = metrics_table(st, y_lv, P0, scale, reg, H, L, nv_is)
        mt["param"], mt["value"] = key
        sens_rows.append(mt)
    sens = pd.concat(sens_rows)
    # feature / look-back
    fl = []
    for part in ("feat", "look"):
        keys = sorted({k for ps in per_seed for k in ps[part]})
        for k in keys:
            st = {k: [ps[part][k] for ps in per_seed if k in ps[part]]}
            mt = metrics_table(st, y_lv, P0, scale, reg, H, L, nv_is); mt["part"] = part
            fl.append(mt)
    full = met[met.method == PROPOSED].copy(); full["method"] = "FULL"; full["part"] = "both"
    featlook = pd.concat(fl + [full]) if fl else full
    # utilities
    if a.data == "nifty":
        r_next = y_lv[:, 0] / P0
        r_next = np.log(r_next)
        tr_r = r[1:t_train_end + 1]
        util = finance_utility({k: v for k, v in main_store.items()}, y_lv, P0, r_next,
                               sigma_target=float(tr_r.std()))
    else:
        point = []
        for sd in cfg["seeds"]:
            zz, _ = load_seed(a.data, sd)
            if zz is not None:
                point.append(P0[:, None] * np.exp(zz["BT/test/m"]))
        util = alert_utility(main_store, y_lv, point, H)
    out = dict(cfg=cfg, H=H, L=L, t0=t0, P0=P0, y=y_lv, scale=scale, reg=reg,
               metrics=met, metrics_abl=met_abl, ci=ci, dm=dm, covtests=covt,
               ci_abl=ci_abl, dm_abl=dm_abl, sens=sens, featlook=featlook, util=util,
               tuned=[ps["tuned"] for ps in per_seed], meta=[ps["meta"] for ps in per_seed],
               calib_sec=[ps["calib_seconds_per_step"] for ps in per_seed],
               traces=[ps["traces"] for ps in per_seed], mech=[ps["mech"] for ps in per_seed],
               garch=dict(params=gparams, seconds=garch_seconds),
               vacp_example=per_seed[0]["methods"][PROPOSED],
               aci_example=per_seed[0]["methods"]["ACI"],
               pid_example=per_seed[0]["methods"]["CP-PID"],
               nv=nv, nv_is=nv_is, series=s, dates=df["Date"].to_numpy(),
               scale_full=scale_full, t_train_end=t_train_end)
    odir = os.path.join(OUT, a.data); tdir = os.path.join(odir, "tables"); os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(odir, "analysis.pkl"), "wb") as f:
        pickle.dump(out, f)
    for k in ("metrics", "metrics_abl", "ci", "dm", "covtests", "ci_abl", "dm_abl", "sens",
              "featlook", "util"):
        out[k].to_csv(os.path.join(tdir, f"{k}.csv"), index=False)
    print("saved", odir)


if __name__ == "__main__":
    main()
