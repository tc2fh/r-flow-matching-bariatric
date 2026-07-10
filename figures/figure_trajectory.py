"""Figure 4 - Trajectory fit with CONFORMAL-CALIBRATED predictive intervals.

Two panels (BMI kg/m^2, HbA1c %-points), absolute native units. Per horizon the raw
flow predictive interval [p10, p90] is widened by that horizon's split-conformal
half-width q (from W5 coverage_post machinery) into a calibrated 90% interval; the
figure shows the median-patient calibrated band, the predicted-median trajectory, and
the observed cohort median. Non-calibrated horizons are de-emphasised (open markers +
hatched band) so an un-earned interval is never shown as trustworthy.

Reads: twin_run_dir/test_predictions.csv (pred_mean/pred_p10/pred_p90/observed/
observed_mask per horizon), eval_flow_calibration_coverage.csv (conformal_q per
horizon), eval_flow_calibration_pit.csv (regime -> trustworthy).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from . import artifacts as A
from .artifacts import RunArtifacts


def _panel(ax, group: str, pred, q_by_h: dict, trust_rows: list[dict]) -> None:
    horizons = A.GROUP_HORIZONS[group]
    months = np.array([A.HORIZON_MONTHS[h] for h in horizons], float)
    trust = np.array([r["trustworthy"] for r in trust_rows], bool)

    center = np.full(len(horizons), np.nan)
    lo = np.full(len(horizons), np.nan)
    hi = np.full(len(horizons), np.nan)
    obs = np.full(len(horizons), np.nan)
    for i, h in enumerate(horizons):
        mean_c, p10_c, p90_c = f"pred_mean_{h}", f"pred_p10_{h}", f"pred_p90_{h}"
        obs_c, mask_c = f"observed_{h}", f"observed_mask_{h}"
        if mean_c not in pred:
            continue
        q = float(q_by_h.get(h, 0.0) or 0.0)
        center[i] = np.nanmedian(pred[mean_c].to_numpy(float))
        lo[i] = np.nanmedian(pred[p10_c].to_numpy(float) - q)
        hi[i] = np.nanmedian(pred[p90_c].to_numpy(float) + q)
        mask = pred[mask_c].to_numpy(float) == 1
        if mask.any():
            obs[i] = np.nanmedian(pred[obs_c].to_numpy(float)[mask])

    valid = ~np.isnan(center)
    style.draw_calibrated_band(ax, months[valid], lo[valid], hi[valid], trust[valid],
                               style.ACCENT, label="calibrated 90% predictive interval")
    ax.plot(months[valid], center[valid], color=style.ACCENT, linewidth=2.0, zorder=3,
            label="predicted median")
    style.draw_calibrated_markers(ax, months[valid], center[valid], trust[valid], style.ACCENT)
    obs_present = ~np.isnan(obs)
    if obs_present.any():
        ax.plot(months[obs_present], obs[obs_present], linestyle="none", marker="D",
                markersize=5, color=style.INK, markeredgecolor=style.SURFACE,
                markeredgewidth=1.0, zorder=6, label="observed (cohort median)")

    thr = A.THRESHOLD[group]
    ax.axhline(thr, color=style.MUTED, linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
    ax.text(months[0], thr, f"{A.THRESHOLD_LABEL[group]}", ha="left", va="bottom",
            fontsize=6.8, color=style.MUTED)

    ax.set_xticks(months)
    ax.set_xticklabels(style.month_ticklabels(months))
    ax.set(xlabel="Time since surgery", ylabel=f"{A.GROUP_LABEL[group]} ({A.GROUP_UNIT[group]})",
           title=f"{A.GROUP_LABEL[group]} trajectory")
    style.style_axis(ax)


def build(art: RunArtifacts, out_stem: Path) -> list[Path]:
    style.apply_rcparams()
    pred = art.flow_predictions()
    cov = art.calibration_coverage()
    q_by_h = dict(zip(cov["horizon"].astype(str), cov["conformal_q"].astype(float)))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
    for ax, group in zip(axes, ("bmi", "hba1c")):
        _panel(ax, group, pred, q_by_h, art.trust_table(group))

    handles, labels = axes[0].get_legend_handles_labels()
    handles = handles + style.calibration_legend_handles()
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Digital-twin trajectory fit with conformal-calibrated intervals", y=1.05)
    style.caption(fig, style.CALIBRATION_CAUTION, y=-0.16, size=6.8)
    return style.save_figure(fig, out_stem)
