"""Supplement figure - four-arm trajectory ablation (does event conditioning help?).

Grouped bars of pooled CRPS (lower = better) for the four arms across overall / BMI /
HbA1c, with the decisive event-conditioned-vs-unconditioned flow contrast as the two
accent hues and the XGB / Ridge point regressors as neutral references. Annotates the
paired-test evidence for the coupling claim.

Reads (optional): trajectory_comparison_metrics.csv (pooled __overall__/__bmi__/
__hba1c__ rows), trajectory_comparison_paired_tests.csv (event_flow_vs_no_event_flow).
Returns [] (skips) if the ablation CSV is absent - it is emitted by a separate
`evaluate_twin.py --trajectory-comparison` run, not by freeze_run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .artifacts import RunArtifacts

ARM_ORDER = ["event_flow", "no_event_flow", "xgb", "ridge"]
ARM_LABEL = {"event_flow": "event-conditioned flow", "no_event_flow": "unconditioned flow",
             "xgb": "XGBoost (point)", "ridge": "Ridge (point)"}
ARM_COLOR = {"event_flow": style.ACCENT, "no_event_flow": style.CONTRAST,
             "xgb": style.MUTED, "ridge": style.BASELINE}
GROUPS = [("overall", "Overall"), ("bmi", "BMI"), ("hba1c", "HbA1c")]


def build(art: RunArtifacts, out_stem: Path) -> list[Path]:
    metrics = art.trajectory_metrics()
    if metrics is None:
        print("[figure_ablation] trajectory_comparison_metrics.csv absent - skipping "
              "(run evaluate_twin.py --trajectory-comparison to emit it).")
        return []
    style.apply_rcparams()
    paired = art.trajectory_paired()

    pooled = metrics[metrics["horizon"].astype(str).str.startswith("__")].copy()
    pooled["gkey"] = pooled["group"].astype(str)

    arms_present = [a for a in ARM_ORDER if a in set(metrics["arm"])]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    n_arms = len(arms_present)
    gwidth = 0.8
    bw = gwidth / n_arms
    xbase = np.arange(len(GROUPS))
    for j, arm in enumerate(arms_present):
        vals = []
        for gkey, _ in GROUPS:
            sub = pooled[(pooled["gkey"] == gkey) & (pooled["arm"] == arm)]
            vals.append(float(sub["crps"].iloc[0]) if len(sub) else np.nan)
        offset = (j - (n_arms - 1) / 2) * bw
        bars = ax.bar(xbase + offset, vals, width=bw * 0.86, color=ARM_COLOR[arm],
                      edgecolor=style.SURFACE, linewidth=1.2, label=ARM_LABEL[arm], zorder=3)
        for rect, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=6.4, color=style.INK_SECONDARY)

    ax.set_xticks(xbase)
    ax.set_xticklabels([lbl for _, lbl in GROUPS])
    ax.set(ylabel="CRPS (lower is better)", title="Trajectory ablation: event conditioning vs baselines")
    ax.legend(loc="upper left", ncol=2, fontsize=7.0)
    style.style_axis(ax)

    # coupling evidence annotation
    note = ""
    ev = pooled[(pooled["gkey"] == "overall") & (pooled["arm"] == "event_flow")]
    nv = pooled[(pooled["gkey"] == "overall") & (pooled["arm"] == "no_event_flow")]
    if len(ev) and len(nv):
        d = float(ev["crps"].iloc[0]) - float(nv["crps"].iloc[0])
        note = f"event - no-event overall CRPS delta = {d:+.3f} ({'favors event' if d < 0 else 'favors no-event'})"
    if paired is not None:
        key = paired[(paired["comparison"] == "event_flow_vs_no_event_flow") & (paired["metric"] == "crps")]
        if len(key):
            k = int((key["wilcoxon_p"] < 0.05).sum())
            note += f"; Wilcoxon p<0.05 favoring event at {k}/{len(key)} horizons"
    if note:
        style.caption(fig, note, y=-0.02, size=6.8)
    return style.save_figure(fig, out_stem)
