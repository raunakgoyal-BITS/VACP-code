# Volatility- and Delay-Aware Online Conformal Calibration of Multi-Horizon Prediction Intervals

This repository holds the code, configuration files and saved outputs behind the paper
*Volatility- and Delay-Aware Online Conformal Calibration of Multi-Horizon Prediction
Intervals*, by Raunak Goyal, Srikrishna Agarwal, Jabez Christopher and Vasan Arunachalam
(manuscript prepared for *Neurocomputing*).

The method is a two-layer LSTM with location and scale heads. Its intervals are calibrated
online by a per-horizon, per-level PI quantile-tracking controller. The controller uses a
normalised score with a floor on the predicted scale, a horizon-scaled gain, and $h$-step
delayed feedback. It is evaluated on the NIFTY 50 index (daily) and on Beijing PM2.5 (hourly).
In the paper and in every generated table and figure, the method is called "Proposed". In the
code it is the `vacp_mh` package.

## Layout
```
vacp_mh/            code, data scripts, configs and saved outputs (see vacp_mh/README.md)
paper/tables/       LaTeX tables written by vacp_mh/experiments/report.py
paper/figs/         figures written by report.py (PDF and PNG), and fig_arch.tex (Fig. 1)
LICENSE             MIT licence
```

## Quick start: rebuild every table and figure from the saved outputs
```bash
cd vacp_mh
pip install -r requirements.txt
cd experiments
python report.py        # about 10 s; writes ../../paper/tables and ../../paper/figs
```
`report.py` reads the saved results in `vacp_mh/outputs/{nifty,pm25}/analysis.pkl`. It does
not need the raw data or a GPU.

## Full reproduction
See `vacp_mh/README.md` for training every model on five seeds, the validation-only tuning
and the test evaluation. The NIFTY 50 data are not redistributed here. Rebuild them from
Yahoo Finance with `python vacp_mh/build_dataset.py`. The PM2.5 data (`vacp_mh/air_quality.csv`)
come from the UCI Beijing Multi-Site Air-Quality dataset (Zhang et al., 2017; CC BY 4.0).

## Licence
MIT; see `LICENSE`.
