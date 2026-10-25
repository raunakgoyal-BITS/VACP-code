"""Train every learned model for one (dataset, seed) and save raw outputs.

Usage:  python experiments/run_seed.py --data nifty --seed 0 --part main
        parts: main | features | lookback

Nothing here touches conformal calibration; analyze.py builds every interval method
from these saved backbone outputs, so calibration choices can be studied without
retraining.  The test split is only used for prediction, never for fitting or selection.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

from common import load_cfg, load_dataset, make_windows, split_std, OUT
from vacp_mh.backbone import Backbone
from vacp_mh.deeppipe import train_deeppipe, predict_deeppipe


def fit_backbone(cfg, seed, trs, vas, tes, loss="student_t", target="logret", lvl_stats=None):
    b = cfg["backbone"]
    if target == "logret":
        Ytr, Yva = trs.R, vas.R
    else:  # standardised level target
        pm, ps = lvl_stats
        Ytr = ((trs.level_target() - pm) / ps).astype(np.float32)
        Yva = ((vas.level_target() - pm) / ps).astype(np.float32)
    bb = Backbone(trs.X.shape[-1], cfg["horizons"], loss=loss, df=b["df"], hidden=b["hidden"],
                  layers=b["layers"], dropout=b["dropout"], lr=b["lr"], batch=b["batch"],
                  epochs=b["epochs"], patience=b["patience"], weight_decay=b["weight_decay"],
                  seed=seed)
    bb.fit(trs.X, Ytr, vas.X, Yva)
    pv = bb.predict(vas.X)
    t = time.perf_counter(); pt = bb.predict(tes.X); inf = time.perf_counter() - t
    meta = dict(n_params=bb.n_params(), train_seconds=bb.train_seconds,
                epochs_run=len(bb.history), history=bb.history,
                infer_seconds_test=inf, n_test=len(tes.X))
    return pv, pt, meta


def pack(store, key, pv, pt):
    for split, p in (("val", pv), ("test", pt)):
        for k, v in p.items():
            store[f"{key}/{split}/{k}"] = v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="main"); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_cfg(a.data)
    df, s, groups, scale = load_dataset(cfg)
    odir = os.path.join(OUT, a.data, f"seed{a.seed}"); os.makedirs(odir, exist_ok=True)
    store, meta = {}, {}

    wd = make_windows(cfg, groups, s)
    trs, vas, tes = split_std(cfg, wd)
    lvl = trs.level_target(); lvl_stats = (float(lvl.mean()), float(lvl.std()))

    if a.part == "main":
        for split, d in (("val", vas), ("test", tes)):
            store[f"{split}/R"] = d.R; store[f"{split}/P0"] = d.P0; store[f"{split}/t0"] = d.t0
        store["lvl_stats"] = np.array(lvl_stats)
        for key, loss, target in [("BT", "student_t", "logret"), ("BG", "gaussian", "logret"),
                                  ("BL", "student_t", "level"), ("QR", "pinball", "logret")]:
            print(f"[{a.data} s{a.seed}] backbone {key}", flush=True)
            pv, pt, m = fit_backbone(cfg, a.seed, trs, vas, tes, loss, target, lvl_stats)
            pack(store, key, pv, pt); meta[key] = m
        # DeepPIPE two-phase, h = 1, designed for 90%: price space (original) and return space
        dp = cfg["deeppipe"]
        pm, ps = lvl_stats
        rs = float(trs.R[:, 0].std())
        spaces = {"DPp": (lambda d: (d.level_target()[:, 0] - pm) / ps,
                          lambda z, d: z * ps + pm),
                  "DPr": (lambda d: d.R[:, 0] / rs,
                          lambda z, d: d.P0 * np.exp(z * rs))}
        for key, (fwd, inv) in spaces.items():
            print(f"[{a.data} s{a.seed}] {key}", flush=True)
            t = time.perf_counter()
            mdl = train_deeppipe(trs.X, fwd(trs).astype(np.float32), vas.X,
                                 fwd(vas).astype(np.float32), input_size=trs.X.shape[-1],
                                 mode="two_phase", epochs=dp["epochs"], warmup=dp["warmup"],
                                 batch=dp["batch"], beta=dp["beta"], lam=dp["lam"],
                                 alpha=0.90, seed=a.seed)
            tt = time.perf_counter() - t
            yhat, lo, up = predict_deeppipe(mdl, tes.X)
            store[f"{key}/test/yhat"] = inv(yhat, tes); store[f"{key}/test/lo"] = inv(lo, tes)
            store[f"{key}/test/hi"] = inv(up, tes)
            meta[key] = dict(train_seconds=tt,
                             n_params=sum(p.numel() for p in mdl.parameters()))
    elif a.part == "features":
        for gs in cfg["feature_ablation"][:-1]:          # the full set is BT from 'main'
            sub = {k: groups[k] for k in gs}
            w = make_windows(cfg, sub, s); t1, v1, e1 = split_std(cfg, w)
            key = "F_" + "+".join(gs)
            print(f"[{a.data} s{a.seed}] {key}", flush=True)
            pv, pt, m = fit_backbone(cfg, a.seed, t1, v1, e1)
            pack(store, key, pv, pt); meta[key] = m
    elif a.part == "lookback":
        for L in cfg["lookback_grid"]:
            if L == cfg["seq_len"]:
                continue
            w = make_windows(cfg, groups, s, seq_len=L); t1, v1, e1 = split_std(cfg, w)
            assert np.array_equal(e1.t0, tes.t0)
            key = f"L{L}"
            print(f"[{a.data} s{a.seed}] {key}", flush=True)
            pv, pt, m = fit_backbone(cfg, a.seed, t1, v1, e1)
            pack(store, key, pv, pt); meta[key] = m

    np.savez_compressed(os.path.join(odir, f"{a.part}.npz"), **store)
    with open(os.path.join(odir, f"{a.part}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{a.data} s{a.seed}] {a.part} done", flush=True)


if __name__ == "__main__":
    main()
