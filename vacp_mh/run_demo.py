# LEGACY (first draft). Not used for the revised paper; see experiments/.
"""End-to-end demo: VACP-MH vs static-split vs plain-ACI vs naive, on synthetic GARCH data.

Runs WITHOUT any external data file. Swap `synthetic_garch_series(...)` for your NIFTY
loader (see data.load_csv_prices) to reproduce on real data.

    python run_demo.py
"""
from __future__ import annotations
import numpy as np
import torch

from vacp_mh import data as D
from vacp_mh import metrics as Mx
from vacp_mh.vacp import VACP_MH
from vacp_mh.baselines import NaiveVolInterval, BackboneStaticSplit


def row(name, y, lo, up, level):
    s = Mx.summary(y, lo, up, level)
    return (f"{name:<26} PICP={s['PICP']:.3f}  MPIW={s['MPIW']:.4f}  "
            f"CovDev={s['CovDev']:.3f}  IntScore={s['IntervalScore']:.4f}")


def main(seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    horizons = [1, 5, 10]
    levels = [0.80, 0.90, 0.95]

    prices, exog = D.synthetic_garch_series(T=3000, seed=seed)     # includes a vol regime shift
    wd = D.build_windows(prices, features=exog, seq_len=60, horizons=horizons)
    tr, va, te = D.chronological_split(wd, train=0.7, val=0.15)
    _, tr, va, te = D.standardize(tr, va, te)

    print("== training VACP-MH backbone ==")
    model = VACP_MH(horizons=horizons, coverage_levels=levels, seq_len=60,
                    conformal_method="pid", gamma=0.03, ki=0.01, window=200, normalized=True)
    model.fit(tr, va, epochs=8, verbose=True)     # bump epochs for real runs

    # static-split (exchangeability) and plain-ACI (no vol-normalisation) comparators
    static = BackboneStaticSplit(model, normalized=True).calibrate(va)

    plain = VACP_MH(horizons=horizons, coverage_levels=levels, normalized=False,
                    conformal_method="aci", gamma=0.03)
    plain.model = model.model                      # reuse the trained backbone
    plain._calibrate(va)

    ret_test = D.make_returns(prices)[-len(te.P0):]  # approx aligned return series for naive
    naive = NaiveVolInterval(horizons)

    print("\n== TEST metrics (price space) ==")
    for level in levels:
        print(f"\n-- target coverage {int(level*100)}% --")
        v_out, _, _ = model.predict_intervals(te, level, update_online=True)
        p_out, _, _ = plain.predict_intervals(te, level, update_online=True)
        s_out = static.predict_intervals(te, level)
        n_out = naive.predict_intervals(prices[-len(te.P0):], ret_test, te.P0, level)
        for h in horizons:
            y = te.P0 * np.exp(te.R[:, horizons.index(h)])
            print(f"  horizon h={h}")
            print("   ", row("VACP-MH (norm+PID)", y, *v_out["price"][h], level))
            print("   ", row("ablate: plain-ACI", y, *p_out["price"][h], level))
            print("   ", row("baseline: static-split", y, *s_out["price"][h], level))
            print("   ", row("baseline: naive-vol", y, *n_out["price"][h], level))


if __name__ == "__main__":
    main()
