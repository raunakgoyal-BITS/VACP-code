# Code

Volatility- and delay-aware online conformal calibration of multi-horizon prediction
intervals. This folder holds the code behind the Neurocomputing manuscript. The tables and figures it
writes are in `../paper/tables` and `../paper/figs`.
Every number in the paper comes from the pipeline below.

In the manuscript and in every generated table and figure, the method is
called "Proposed" (set by `PROPOSED` in `experiments/analyze.py`); the Python package is
`vacp_mh`. The saved `analysis.pkl` files use the earlier label "VACP-MH"; `report.py`
renames it when it loads them.

## Layout
```
vacp_mh/                 library
  backbone.py            two-layer LSTM; Student-t / Gaussian NLL or pinball (QR) heads
  models.py, losses.py   network module and loss functions used by backbone.py
  conformal.py           OnlineConformal (PI quantile tracking, ACI), SplitConformal, rollout
                         with h-step delayed feedback
  garch.py               GARCH(1,1)-t baseline
  deeppipe.py            DeepPIPE interval-learning baseline
  stats.py               block bootstrap, DM-HLN, Holm, Kupiec, Christoffersen,
                         Friedman-Nemenyi, data diagnostics
  data.py                returns, windows, purged split, train-only standardisation
experiments/
  configs/{nifty,pm25}.yaml   all settings, horizons, levels and seeds
  run_seed.py            trains every learned model for one dataset and seed (no calibration)
  methods.py             builds every interval method from the saved backbone outputs
  analyze.py             validation-only tuning, test evaluation, tests, utility studies
  report.py              writes paper/tables/*.tex and paper/figs/*.pdf
build_dataset.py         rebuilds market_data.csv (NIFTY 50, India VIX, global markets)
build_airquality.py      rebuilds air_quality.csv (Beijing Aotizhongxin station, UCI)
outputs/                 saved model outputs, analysis.pkl and CSV tables per dataset
```

## Reproduce
```bash
pip install -r requirements.txt
python build_dataset.py   # NIFTY 50 data from Yahoo Finance (not redistributed here)
cd experiments
for d in nifty pm25; do
  for s in 0 1 2 3 4; do
    for p in main features lookback; do python run_seed.py --data $d --seed $s --part $p; done
  done
  python analyze.py --data $d
done
python report.py          # tables and figures only; about 10 s from the saved analysis.pkl files
```
Training uses the train split only. Calibrator gains are chosen on the second half of the
validation split. The test split is used only for the final evaluation.

## Data
- NIFTY 50, India VIX and global series: Yahoo Finance, via `build_dataset.py`. The CSV is
  not included in this repository. Because Yahoo Finance may revise past data, a rebuilt file
  can differ slightly from the one used in the paper. The saved outputs in `outputs/` are
  the ones behind every reported number.
- Beijing Multi-Site Air-Quality data (Zhang et al., 2017), UCI Machine Learning Repository,
  station Aotizhongxin, via `build_airquality.py`. Set `SRC` in that script to the
  downloaded CSV. The resulting `air_quality.csv` is included (CC BY 4.0).

## Licence
The code is released under the MIT licence; see `../LICENSE`.
