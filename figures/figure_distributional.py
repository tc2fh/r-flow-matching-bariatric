"""figure_distributional.py - render the distributional (``dist_*``) CSVs as ONE figure.

run_tte._run_distributional writes ``dist_proper_scores_test``, ``dist_coverage_curve_test``,
``dist_threshold_calibration_{bmi35,hba1c57}_test``, ``dist_calibration_slope_citl``,
``dist_attrition_sensitivity`` and ``dist_modeC_marginal_distance`` as CSV. This module turns
the four that carry the argument into one main figure so they travel back as a PNG/PDF/SVG.

Figure C2 (2 rows x 2 cols):
  A  Calibration slope & CITL - the SOPHIA drift readout (1.0 / 0.0 = ideal).
  B  Coverage calibration - nominal vs empirical predictive-interval coverage.
  C  Threshold reliability - P(BMI<35) and P(HbA1c<5.7) predicted vs observed, ECE/Brier.
  D  Attrition sensitivity - naive (observed-only) vs IPCW-weighted headline metrics.

Real run:   OMP_NUM_THREADS=1 python -m figures.figure_distributional --eval-dir runs/frozen/<ts>/evaluation
Demo:       python -m figures.figure_distributional --demo --out /tmp/figc

Calibration DRIFT (train-early / test-late collapse) is a CROSS-RUN delta: pass a second
run's dist_calibration_slope_citl.csv via --temporal-eval-dir to overlay the two folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe on a display-less VM (incl. Windows); before pyplot import

from . import style as S


_DIST_FILES = {
    "slope": "dist_calibration_slope_citl.csv",
    "coverage": "dist_coverage_curve_test.csv",
    "thr_bmi": "dist_threshold_calibration_bmi35_test.csv",
    "thr_hba1c": "dist_threshold_calibration_hba1c57_test.csv",
    "attrition": "dist_attrition_sensitivity.csv",
    "modec": "dist_modeC_marginal_distance.csv",
}


def load_distributional(eval_dir: str | Path) -> dict:
    eval_dir = Path(eval_dir)
    out = {}
    for key, fname in _DIST_FILES.items():
        p = eval_dir / fname
        out[key] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return out


def demo_distributional(temporal: bool = False) -> dict:
    """Illustrative numbers. ``temporal=True`` degrades slope/coverage to mimic the
    out-of-time fold, so the two overlaid demonstrate the drift readout."""
    horizons = ["bmi_12m", "bmi_2y", "hba1c_12m", "hba1c_2y"]
    if temporal:
        slope_vals = [0.79, 0.71, 0.83, 0.74]; citl_vals = [0.14, 0.22, 0.10, 0.18]
    else:
        slope_vals = [0.96, 0.92, 0.98, 0.94]; citl_vals = [0.03, 0.06, 0.02, 0.05]
    slope = pd.DataFrame(dict(family=["bmi35", "bmi35", "hba1c57", "hba1c57"],
                             horizon=horizons, weighting="naive",
                             slope=slope_vals, citl=citl_vals, n_obs=[3128, 2443, 1357, 1015],
                             split_strategy=("temporal" if temporal else "surgery")))
    cov_rows = []
    for nm, drop in (("bmi_12m", 0.02), ("bmi_2y", 0.05), ("hba1c_12m", 0.03)):
        for nominal in (0.5, 0.8, 0.9, 0.95):
            emp = max(0.0, nominal - drop - (0.03 if nominal >= 0.9 else 0.0))
            cov_rows.append(dict(horizon=nm, nominal=nominal, empirical_naive=emp,
                                 mean_width_naive=np.nan))
    coverage = pd.DataFrame(cov_rows)

    def _rel(seed, ece, brier):
        rng = np.random.default_rng(seed)
        preds = np.linspace(0.05, 0.9, 8)
        obs = np.clip(preds + rng.normal(0, ece, preds.size), 0, 1)
        return pd.DataFrame(dict(weighting="naive", ece=ece, brier=brier,
                                 bin_pred=preds, bin_obs=obs, bin_n=(rng.integers(60, 400, preds.size))))
    thr_bmi = _rel(1, 0.04, 0.16)
    thr_hba1c = _rel(2, 0.07, 0.11)
    at_rows = []
    for nm, cn, cw in (("bmi_12m", 1.05, 1.28), ("bmi_2y", 1.9, 2.35),
                       ("hba1c_12m", 0.30, 0.36), ("hba1c_2y", 0.33, 0.41)):
        at_rows.append(dict(horizon=nm, metric="crps", value_naive=cn, value_ipcw=cw,
                            naive_minus_ipcw_gap=cn - cw))
    attrition = pd.DataFrame(at_rows)
    modec = pd.DataFrame([
        dict(horizon="bmi_12m", n_obs=3128, wasserstein1=0.74, ks_stat=0.068, median_shift=-0.74),
        dict(horizon="bmi_2y", n_obs=2443, wasserstein1=1.66, ks_stat=0.128, median_shift=1.66),
        dict(horizon="hba1c_12m", n_obs=1357, wasserstein1=0.10, ks_stat=0.131, median_shift=-0.08),
    ])
    return dict(slope=slope, coverage=coverage, thr_bmi=thr_bmi, thr_hba1c=thr_hba1c,
                attrition=attrition, modec=modec, demo=True)


def _empty(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color=S.MUTED, style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)


def _plot_slope(ax, slope: pd.DataFrame, slope_t: pd.DataFrame | None) -> None:
    if not len(slope):
        _empty(ax, "no dist_calibration_slope_citl.csv"); ax.set_title("A  Calibration slope / CITL", loc="left"); return
    s = slope[slope.get("weighting", "naive") == "naive"] if "weighting" in slope else slope
    labels = list(s["horizon"])
    y = np.arange(len(s))[::-1]
    ax.axvline(1.0, color=S.BASELINE, lw=1.1, zorder=1)
    ax.plot(s["slope"], y, marker="o", ms=6.5, ls="none", color=S.ACCENT,
            markeredgecolor=S.SURFACE, markeredgewidth=1.2, zorder=4,
            label=f"this fold ({s['split_strategy'].iloc[0] if 'split_strategy' in s else 'run'})")
    if slope_t is not None and len(slope_t):
        st = slope_t[slope_t.get("weighting", "naive") == "naive"] if "weighting" in slope_t else slope_t
        st = st.set_index("horizon").reindex(labels).reset_index()
        ax.plot(st["slope"], y, marker="D", ms=6, ls="none", color=S.CRITICAL,
                markeredgecolor=S.SURFACE, markeredgewidth=1.0, zorder=4, label="temporal fold")
        for yi, a, bb in zip(y, s["slope"].values, st["slope"].values):
            if np.isfinite(a) and np.isfinite(bb):
                ax.plot([a, bb], [yi, yi], color=S.MUTED, lw=1.0, alpha=0.6, zorder=2)
    for yi, sl, ci in zip(y, s["slope"], s["citl"]):
        ax.annotate(f"CITL {ci:+.2f}", (sl, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.3, color=S.MUTED)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.8, len(s) - 0.2)
    ax.set_xlim(0.68, 1.03)
    ax.set_xlabel("calibration slope (1.0 = ideal; <1 = overfit/optimistic)", fontsize=7.4)
    ax.set_title("A  Calibration slope & CITL - the SOPHIA drift readout", loc="left")
    ax.legend(fontsize=6.8, loc="lower right")
    S.style_axis(ax, xgrid=True, ygrid=False)


def _plot_coverage(ax, coverage: pd.DataFrame) -> None:
    if not len(coverage):
        _empty(ax, "no dist_coverage_curve_test.csv"); ax.set_title("B  Coverage calibration", loc="left"); return
    ax.plot([0, 1], [0, 1], ls="--", color=S.BASELINE, lw=1.2, label="ideal", zorder=1)
    palette = [S.ACCENT, S.CONTRAST, S.AQUA, S.INK_SECONDARY]
    for i, (nm, g) in enumerate(coverage.groupby("horizon")):
        g = g.sort_values("nominal")
        ax.plot(g["nominal"], g["empirical_naive"], marker="o", ms=5, color=palette[i % len(palette)],
                label=nm, zorder=3)
    ax.set(xlim=(0.4, 1.0), ylim=(0.4, 1.0))
    ax.set_xlabel("nominal coverage", fontsize=7.6); ax.set_ylabel("empirical coverage", fontsize=7.6)
    ax.set_title("B  Predictive-interval coverage (on the diagonal = calibrated)", loc="left")
    ax.legend(fontsize=6.8, loc="upper left")
    S.style_axis(ax, xgrid=True, ygrid=True)


def _plot_reliability(ax, tb: pd.DataFrame, th: pd.DataFrame) -> None:
    have = False
    ax.plot([0, 1], [0, 1], ls="--", color=S.BASELINE, lw=1.2, zorder=1)
    for df, color, name in ((tb, S.ACCENT, "P(BMI<35)"), (th, S.CONTRAST, "P(HbA1c<5.7)")):
        if not len(df) or "bin_pred" not in df:
            continue
        d = df.dropna(subset=["bin_pred", "bin_obs"])
        if not len(d):
            continue
        have = True
        ece = float(d["ece"].iloc[0]) if "ece" in d else np.nan
        brier = float(d["brier"].iloc[0]) if "brier" in d else np.nan
        ax.plot(d["bin_pred"], d["bin_obs"], marker="o", ms=5, color=color,
                label=f"{name}  ECE {ece:.02f} / Brier {brier:.02f}", zorder=3)
    if not have:
        _empty(ax, "no dist_threshold_calibration_*.csv"); ax.set_title("C  Threshold reliability", loc="left"); return
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel("predicted probability", fontsize=7.6); ax.set_ylabel("observed frequency", fontsize=7.6)
    ax.set_title("C  Threshold-probability reliability (the delivered numbers)", loc="left")
    ax.legend(fontsize=6.8, loc="upper left")
    S.style_axis(ax, xgrid=True, ygrid=True)


def _plot_attrition(ax, at: pd.DataFrame) -> None:
    d = at[at.get("metric") == "crps"] if len(at) and "metric" in at else at
    d = d.dropna(subset=["value_naive", "value_ipcw"]) if len(d) else d
    if not len(d):
        _empty(ax, "no dist_attrition_sensitivity.csv"); ax.set_title("D  Attrition sensitivity", loc="left"); return
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r["value_naive"], r["value_ipcw"]], [yi, yi], color=S.MUTED, lw=1.4, zorder=2)
        ax.plot([r["value_naive"]], [yi], marker="o", ms=6, color=S.ACCENT,
                markeredgecolor=S.SURFACE, markeredgewidth=1.1, zorder=4)
        ax.plot([r["value_ipcw"]], [yi], marker="o", ms=6, color=S.CONTRAST,
                markeredgecolor=S.SURFACE, markeredgewidth=1.1, zorder=4)
    ax.plot([], [], marker="o", ls="none", color=S.ACCENT, label="naive (observed-only)")
    ax.plot([], [], marker="o", ls="none", color=S.CONTRAST, label="IPCW-weighted")
    ax.set_yticks(y); ax.set_yticklabels(list(d["horizon"]), fontsize=7.5)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("CRPS (lower = better) - gap = informative-attrition bias", fontsize=7.2)
    ax.set_title("D  Attrition sensitivity (naive vs censoring-weighted)", loc="left")
    ax.legend(fontsize=6.8, loc="lower right")
    S.style_axis(ax, xgrid=True, ygrid=False)


def _build_fig(data: dict, stem: Path, slope_t: pd.DataFrame | None = None) -> list[Path]:
    S.apply_rcparams()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    demo = data.get("demo", False)
    fig = plt.figure(figsize=(11.0, 8.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.26,
                  left=0.09, right=0.97, top=0.90, bottom=0.10)
    _plot_slope(fig.add_subplot(gs[0, 0]), data["slope"], slope_t)
    _plot_coverage(fig.add_subplot(gs[0, 1]), data["coverage"])
    _plot_reliability(fig.add_subplot(gs[1, 0]), data["thr_bmi"], data["thr_hba1c"])
    _plot_attrition(fig.add_subplot(gs[1, 1]), data["attrition"])

    title = "Distributional validation of the twin (held-out test set)"
    if demo:
        title += "   -   ILLUSTRATIVE DEMO (synthetic numbers)"
    fig.suptitle(title, x=0.02, ha="left", fontsize=12.5, weight="semibold")
    cap = ("Grades the predictive distribution, not just point error. Slope<1 / |CITL|>0 and coverage below the "
           "diagonal = optimistic (the failure mode that broke prior calculators out-of-sample). The drift is the "
           "delta between the internal (surgery) and temporal folds - overlay both with --temporal-eval-dir.")
    if demo:
        cap = "DEMO - numbers are illustrative placeholders, NOT a real run.  " + cap
    S.caption(fig, cap, y=0.055, color=S.CRITICAL if demo else S.INK_SECONDARY, size=7.0)
    return S.save_figure(fig, stem)


def build(art, stem: Path, *, demo: bool = False) -> list[Path]:
    data = demo_distributional() if demo else load_distributional(art.eval_dir)
    if not demo and not len(data.get("slope", [])) and not len(data.get("coverage", [])):
        return []
    return _build_fig(data, Path(stem))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=None)
    ap.add_argument("--temporal-eval-dir", default=None, help="second fold to overlay for the drift readout")
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if not args.demo and not args.eval_dir:
        raise SystemExit("Pass --eval-dir <run>/evaluation, or --demo.")
    if args.demo:
        data = demo_distributional()
        slope_t = demo_distributional(temporal=True)["slope"]
    else:
        data = load_distributional(args.eval_dir)
        slope_t = (load_distributional(args.temporal_eval_dir)["slope"]
                   if args.temporal_eval_dir else None)
    out = Path(args.out) if args.out else Path(args.eval_dir).parent / "figures" / "main"
    out.mkdir(parents=True, exist_ok=True)
    written = _build_fig(data, out / "figC2_distributional", slope_t=slope_t)
    print("wrote:", ", ".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
