"""Figure 3 / Figure spec B - cohort RYGB-vs-sleeve comparative effectiveness.

The headline counterfactual. The whole held-out test cohort is clamped to RYGB and to
sleeve; per patient we take the sample median at each timepoint; the signed difference
(RYGB - sleeve) is formed in ABSOLUTE native units (BMI kg/m^2, HbA1c %-points); the
cohort MEDIAN of that difference is plotted per timepoint with a patient-bootstrap 95%
band. Negative => RYGB lower (the expected direction). Non-calibrated horizons are
de-emphasised (the counterfactual inherits the flow's tail calibration).

Uses figures.sampling.cohort_surgery_medians (re-samples the frozen twin) + trust_table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from . import artifacts as A
from .artifacts import RunArtifacts
from . import sampling as S


def _bootstrap_median(col: np.ndarray, n_boot: int, rng: np.random.Generator):
    n = col.size
    if n == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(col))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.median(col[idx], axis=1)
    return med, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _panel(ax, group: str, diff: np.ndarray, name_to_col: dict, trust_rows: list[dict],
           n_boot: int, rng: np.random.Generator) -> None:
    horizons = A.GROUP_HORIZONS[group]
    months = np.array([A.HORIZON_MONTHS[h] for h in horizons], float)
    trust = np.array([r["trustworthy"] for r in trust_rows], bool)
    med = np.full(len(horizons), np.nan); lo = np.full(len(horizons), np.nan); hi = np.full(len(horizons), np.nan)
    for i, h in enumerate(horizons):
        if h in name_to_col:
            med[i], lo[i], hi[i] = _bootstrap_median(diff[:, name_to_col[h]], n_boot, rng)

    valid = ~np.isnan(med)
    ax.axhline(0.0, color=style.BASELINE, linewidth=1.1, zorder=1)
    style.draw_calibrated_band(ax, months[valid], lo[valid], hi[valid], trust[valid],
                               style.ACCENT, label="bootstrap 95% CI")
    ax.plot(months[valid], med[valid], color=style.ACCENT, linewidth=2.0, zorder=3,
            label="median RYGB - sleeve")
    style.draw_calibrated_markers(ax, months[valid], med[valid], trust[valid], style.ACCENT)

    ax.set_xticks(months); ax.set_xticklabels(style.month_ticklabels(months))
    ax.set(xlabel="Time since surgery",
           ylabel=f"{A.GROUP_LABEL[group]} difference ({A.GROUP_UNIT[group]})\nRYGB - sleeve",
           title=f"{A.GROUP_LABEL[group]}")
    style.style_axis(ax)
    # direction cue
    ymin, ymax = ax.get_ylim()
    if ymin < 0:
        ax.annotate("RYGB lower", xy=(months[0], ymin * 0.82), fontsize=6.6,
                    color=style.GOOD, ha="left", va="center")


def build(art: RunArtifacts, out_stem: Path, *, bundle: S.TwinBundle | None = None,
          n_samples: int = 200, n_steps: int = 50, seed: int = 0, n_boot: int = 1000,
          device: str = "cpu", csv_path: str | None = None) -> list[Path]:
    style.apply_rcparams()
    if bundle is None:
        bundle = S.load_frozen(art, device=device, csv_path=csv_path)
    cont_names, medians = S.cohort_surgery_medians(bundle, n_samples=n_samples, n_steps=n_steps, seed=seed)
    diff = medians["rnygb"] - medians["sleeve"]           # (n_test, 15) native units
    name_to_col = {n: j for j, n in enumerate(cont_names)}
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
    for ax, group in zip(axes, ("bmi", "hba1c")):
        _panel(ax, group, diff, name_to_col, art.trust_table(group), n_boot, rng)

    handles, labels = axes[0].get_legend_handles_labels()
    handles = handles + style.calibration_legend_handles()
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("RYGB vs sleeve: cohort comparative effectiveness (held-out test set)", y=1.05)
    style.caption(fig,
                  f"n = {diff.shape[0]} test patients, {n_samples} samples/patient, {n_boot} bootstrap resamples. "
                  "Negative = RYGB lower. Display samples are truncated to predeclared physiological bounds; "
                  "the counterfactual safety supplement reports every raw violation. " + style.CALIBRATION_CAUTION,
                  y=-0.16, size=6.8)
    return style.save_figure(fig, out_stem)
