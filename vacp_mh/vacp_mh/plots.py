"""Publication-quality visualizations for the interval-forecasting study."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})


def plot_forecast_band(dates_idx, y_true, y_pred, lower, upper, level, title, path):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(dates_idx, lower, upper, color="#a1d99b",
                    alpha=0.5, label=f"{int(level*100)}% prediction interval")
    ax.plot(dates_idx, y_true, color="#08306b", lw=1.6, label="Actual")
    if y_pred is not None:
        ax.plot(dates_idx, y_pred, color="#d94801", lw=1.2, ls="--", label="Point forecast")
    ax.set_xlabel("Test trading day index"); ax.set_ylabel("NIFTY 50 close")
    ax.set_title(title); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_reliability(results_by_level, methods, path, title="Calibration: nominal vs empirical"):
    """results_by_level: {method: {nominal_level: empirical_picp}}."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
    for m in methods:
        levels = sorted(results_by_level[m].keys())
        emp = [results_by_level[m][l] for l in levels]
        ax.plot(levels, emp, marker="o", label=m)
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage (PICP)")
    ax.set_title(title); ax.legend(); ax.set_xlim(0.5, 1); ax.set_ylim(0.4, 1.02)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_width_vs_horizon(mpiw_by_method, horizons, path,
                          title="Interval width grows with horizon"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, widths in mpiw_by_method.items():
        ax.plot(horizons, widths, marker="s", label=m)
    ax.set_xlabel("Forecast horizon h (days)"); ax.set_ylabel("MPIW (price units)")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_width_vs_volatility(width, vix, path,
                             title="Interval width vs. volatility (adaptivity)"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(vix, width, s=12, alpha=0.5, color="#2171b5")
    if len(vix) > 2:
        c = np.corrcoef(vix, width)[0, 1]
        z = np.polyfit(vix, width, 1); xs = np.linspace(min(vix), max(vix), 50)
        ax.plot(xs, np.polyval(z, xs), "r-", lw=1.5, label=f"corr = {c:+.2f}")
        ax.legend()
    ax.set_xlabel("India VIX (volatility)"); ax.set_ylabel("Interval width")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_metric_bars(names, values, ylabel, title, path, highlight="VACP-MH"):
    order = np.argsort(values)
    names = [names[i] for i in order]; values = [values[i] for i in order]
    colors = ["#2ca25f" if highlight in n else "#3182bd" for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, values, color=colors, edgecolor="black", alpha=0.9)
    off = max(values) * 0.02
    for b, v in zip(bars, values):
        ax.text(v + off, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_coverage_over_time(y_true, lower, upper, path,
                            title="Rolling coverage over test period", window=30):
    covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
    roll = np.convolve(covered, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(roll, color="#238b45", lw=1.5)
    ax.axhline(0.9, color="k", ls="--", lw=1, label="target 0.90")
    ax.set_xlabel("Test day"); ax.set_ylabel(f"rolling PICP (w={window})")
    ax.set_ylim(0, 1.02); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
