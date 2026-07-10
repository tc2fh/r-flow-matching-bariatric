"""Figure 2 - GBM composite-event risk model: discrimination + calibration + net benefit.

Three panels, one accent colour + neutral references:
  (A) ROC with AUROC (+ bootstrap 95% CI read from eval_gbm_discrimination_test.csv).
  (B) Reliability curve (observed vs predicted risk, marker area ~ bin n) + Brier.
  (C) Decision curve (net benefit vs threshold probability). No DCA artifact exists,
      so net benefit is COMPUTED here from the per-patient risks + labels:
        NB(p_t) = TP/n - FP/n * (p_t / (1 - p_t)),   positive call iff risk >= p_t
      against the two references treat-all and treat-none.

Reads: gbm_run_dir/test_predictions.csv (subject_id, y_true, prob_unweighted) and
eval_gbm_discrimination_test.csv (auroc/auroc_lo/auroc_hi/brier, 'raw' row).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .artifacts import RunArtifacts


def _roc(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-s, kind="mergesort")
    y = y[order].astype(float)
    s_sorted = s[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    P, N = max(y.sum(), 1.0), max((1.0 - y).sum(), 1.0)
    keep = np.r_[np.where(np.diff(s_sorted) != 0)[0], len(s_sorted) - 1]
    tpr = np.r_[0.0, tps[keep] / P]
    fpr = np.r_[0.0, fps[keep] / N]
    auroc = float(np.trapezoid(tpr, fpr)) if tpr.size > 1 else float("nan")
    return fpr, tpr, auroc


def _reliability(y: np.ndarray, s: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    pred, obs, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            pred.append(float(s[m].mean()))
            obs.append(float(y[m].mean()))
            cnt.append(int(m.sum()))
    return np.asarray(pred), np.asarray(obs), np.asarray(cnt)


def _net_benefit(y: np.ndarray, s: np.ndarray, thresholds: np.ndarray):
    n = y.size
    prev = float(y.mean()) if n else float("nan")
    nb_model, nb_all = [], []
    for pt in thresholds:
        w = pt / (1.0 - pt)
        pos = s >= pt
        tp = float(np.sum(pos & (y == 1)))
        fp = float(np.sum(pos & (y == 0)))
        nb_model.append(tp / n - fp / n * w)
        nb_all.append(prev - (1.0 - prev) * w)
    return np.asarray(nb_model), np.asarray(nb_all)


def build(art: RunArtifacts, out_stem: Path, *, prob_col: str = "prob_unweighted") -> list[Path]:
    style.apply_rcparams()
    pred = art.gbm_predictions()
    y = pred["y_true"].to_numpy().astype(int)
    s = pred[prob_col].to_numpy().astype(float)

    disc = art.gbm_discrimination()
    raw = disc[disc["score"] == "raw"]
    row = raw.iloc[0] if len(raw) else disc.iloc[0]
    auroc_ci = (float(row["auroc"]), float(row["auroc_lo"]), float(row["auroc_hi"]))
    brier = float(row["brier"])

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), constrained_layout=True)

    # -- (A) ROC ------------------------------------------------------------- #
    ax = axes[0]
    fpr, tpr, auroc_num = _roc(y, s)
    ax.plot([0, 1], [0, 1], color=style.MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.step(fpr, tpr, where="post", color=style.ACCENT, linewidth=2.2, zorder=3)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="1 - specificity (false-positive rate)",
           ylabel="Sensitivity (true-positive rate)", title="A  Discrimination (ROC)")
    ax.set_aspect("equal")
    ax.text(0.97, 0.06, f"AUROC {auroc_ci[0]:.2f}\n(95% CI {auroc_ci[1]:.2f}-{auroc_ci[2]:.2f})",
            ha="right", va="bottom", fontsize=8.0, color=style.INK)
    style.style_axis(ax)

    # -- (B) Reliability ----------------------------------------------------- #
    ax = axes[1]
    pred_m, obs_m, cnt = _reliability(y, s, n_bins=10)
    ax.plot([0, 1], [0, 1], color=style.MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    if pred_m.size:
        sizes = 20 + 90 * (cnt / cnt.max())
        ax.plot(pred_m, obs_m, color=style.ACCENT, linewidth=1.6, zorder=2)
        ax.scatter(pred_m, obs_m, s=sizes, color=style.ACCENT, edgecolor=style.SURFACE,
                   linewidth=1.0, zorder=4)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Predicted risk",
           ylabel="Observed event frequency", title="B  Calibration (reliability)")
    ax.set_aspect("equal")
    ax.text(0.97, 0.06, f"Brier {brier:.3f}", ha="right", va="bottom", fontsize=8.0, color=style.INK)
    style.style_axis(ax)

    # -- (C) Decision curve (computed) --------------------------------------- #
    ax = axes[2]
    prev = float(y.mean())
    thr = np.linspace(0.01, min(0.6, max(0.2, prev * 3 + 0.1)), 120)
    nb_model, nb_all = _net_benefit(y, s, thr)
    ax.axhline(0.0, color=style.BASELINE, linewidth=1.0, zorder=1)                 # treat none
    ax.plot(thr, nb_all, color=style.MUTED, linewidth=1.3, linestyle=(0, (4, 3)),
            zorder=2, label="treat all")
    ax.plot(thr, nb_model, color=style.ACCENT, linewidth=2.2, zorder=3, label="GBM model")
    ax.set_ylim(min(-0.02, float(np.nanmin(nb_model)) - 0.01), max(0.02, prev + 0.02))
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="C  Decision curve (net benefit)")
    style.direct_label(ax, thr[-1], nb_model[-1], "GBM", style.ACCENT, dx=4, va="center")
    ax.legend(loc="upper right", handlelength=1.8)
    style.style_axis(ax)

    fig.suptitle("Composite-event (MACE / nephropathy / retinopathy) risk model", y=1.06)
    style.caption(fig,
                  f"n = {y.size} test patients, event prevalence {prev:.1%}. "
                  "Decision-curve net benefit computed from per-patient predicted risk and labels; "
                  "treat-none = 0, treat-all = prevalence-weighted reference.",
                  y=-0.02, size=6.8)
    return style.save_figure(fig, out_stem)
