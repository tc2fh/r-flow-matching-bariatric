"""Figure 1 - Cohort / CONSORT attrition funnel.

Built from ``debug_attrition`` output (reuses ``python_attrition`` so the filter logic
is never duplicated). The GLP-1 filter (SQL ``PriorGLP1 = 0``) is drawn as its OWN
explicit, labelled exclusion node at the top of the funnel - never a silent drop -
even on the local CSV path where its per-clause count is VM-only.

Reads: debug_attrition.python_attrition(raw_df) stages (raw loaded -> analytic cohort),
optional db_prefilter_counts() (VM) for the SQL per-clause n, and (optional) the
eval_twin_summary split sizes for the terminal train/val/test split.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from . import style
from .artifacts import RunArtifacts, _read_json


def _box(ax, x, y, w, h, text, *, facecolor, edgecolor, textcolor=style.INK,
         dashed=False, weight="normal", fontsize=8.0, ha="center"):
    boxstyle = "round,pad=0.02,rounding_size=0.02"
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=boxstyle,
                           linewidth=1.0, facecolor=facecolor, edgecolor=edgecolor,
                           linestyle="--" if dashed else "-", zorder=3)
    ax.add_patch(patch)
    tx = x if ha == "center" else x - w / 2 + 0.015
    ax.text(tx, y, text, ha=ha, va="center", fontsize=fontsize, color=textcolor,
            zorder=4, linespacing=1.25)


def _down_arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.0, color=style.MUTED, zorder=2))


def _exclusion(ax, x_main, x_excl, y, w, h, text):
    # elbow from the main spine out to a right-hand exclusion box
    ax.add_patch(FancyArrowPatch((x_main, y), (x_excl - w / 2, y), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=1.0, color=style.MUTED, zorder=2))
    _box(ax, x_excl, y, w, h, text, facecolor="#f6f5f2", edgecolor=style.BASELINE,
         textcolor=style.INK_SECONDARY, fontsize=6.6, ha="left")


def build(art: RunArtifacts, out_stem: Path, *, use_db: bool = False,
          csv_path: str | None = None) -> list[Path]:
    import debug_attrition as da

    style.apply_rcparams()
    src = csv_path or art.source_csv
    if src is None:
        raise FileNotFoundError("No source CSV recorded for this run; pass csv_path=... to build the CONSORT figure.")
    src_path = Path(src)
    if not src_path.exists() and not src_path.is_absolute():
        src_path = Path.cwd() / src
    raw_df = da.load_raw_csv(src_path)
    info = da.python_attrition(raw_df)
    db_counts = da.db_prefilter_counts() if use_db else None

    stages = info["stages"]                 # [{stage, dropped, remaining}], first is 'raw loaded'
    n_raw = info["n_raw"]
    n_final = stages[-1]["remaining"]

    # PriorGLP1 clause count if we are on the VM (else surfaced as an explicit but
    # count-deferred step so it is never a silent drop).
    glp1_n = None
    sql_removed = None
    true_total = None
    if db_counts and "true_total" in db_counts:
        true_total = db_counts["true_total"]
        sql_removed = true_total - n_raw
        for entry in db_counts.get("clauses", []):
            if "PriorGLP1" in str(entry.get("clause", "")):
                glp1_n = entry.get("rows_excluded_alone")

    # ---- layout: a central spine of retained-cohort nodes, right-hand exclusions ---- #
    fig, ax = plt.subplots(figsize=(8.0, 8.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    x_main, x_excl = 0.29, 0.74
    node_w, node_h = 0.40, 0.075
    excl_w, excl_h = 0.44, 0.078

    # Build the ordered list of retained nodes and the exclusions leading into each.
    retained = [("MBSCohort - SQL cohort definition\n(inclusion WHERE incl. PriorGLP1 = 0)",
                 true_total, True)]
    exclusions = []
    # SQL WHERE exclusion (incl. the GLP-1 filter) -> post-SQL export
    if sql_removed is not None:
        sql_txt = (f"Excluded in SQL cohort def.: n = {sql_removed}\n"
                   f"  - PriorGLP1 = 0 (prior GLP-1): n = "
                   f"{glp1_n if glp1_n is not None else 'n/a'}\n"
                   f"  - PMH_PriorMBS=0, BMI 35-75, eGFR>=20, ...")
    else:
        sql_txt = ("Excluded in SQL cohort def. (pre-export):\n"
                   "  - PriorGLP1 = 0 (prior GLP-1 users)\n"
                   "  - PMH_PriorMBS=0, BMI 35-75, eGFR>=20, ...\n"
                   "  [per-clause n: VM-only]")
    exclusions.append(sql_txt)
    retained.append((f"Records exported (post-SQL WHERE)", n_raw, False))

    # python_attrition drop stages -> subsequent retained nodes
    stage_labels = {
        "CPT unrecognized (not sleeve/rnygb)": "CPT code not sleeve / RYGB",
        "composite event w/o valid interval": "Composite event without a valid interval",
        "PostOpGLP1 w/o valid start interval": "Post-op GLP-1 without valid start interval",
        "missing required conditioning": "Missing required baseline conditioning",
    }
    for stage in stages[1:]:
        pretty = stage_labels.get(stage["stage"], stage["stage"])
        exclusions.append(f"Excluded: {pretty}\n  n = {stage['dropped']}")
        retained.append(("Analytic modeling cohort" if stage is stages[-1]
                         else f"After: {pretty}", stage["remaining"], stage is stages[-1]))

    # y positions: evenly spaced top->bottom
    n_nodes = len(retained)
    ys = np.linspace(0.965, 0.085, n_nodes)
    for i, (label, n, is_terminal) in enumerate(retained):
        y = ys[i]
        if i == 0:  # SQL definition node (dashed, muted)
            n_txt = f"n = {n}" if n is not None else "n = (VM-only)"
            _box(ax, x_main, y, node_w, node_h + 0.02, f"{label}\n{n_txt}",
                 facecolor="#f3f2ef", edgecolor=style.MUTED, dashed=True,
                 textcolor=style.INK_SECONDARY, fontsize=7.6)
        else:
            is_final = i == n_nodes - 1
            fc = "#e9f1fb" if is_final else style.SURFACE
            ec = style.ACCENT if is_final else style.BASELINE
            pct = 100.0 * n / n_raw if n_raw else float("nan")
            _box(ax, x_main, y, node_w, node_h, f"{label}\nn = {n}  ({pct:.1f}% of exported)",
                 facecolor=fc, edgecolor=ec, weight="semibold" if is_final else "normal",
                 textcolor=style.INK, fontsize=7.8)
        if i < n_nodes - 1:
            _down_arrow(ax, x_main, y - (node_h / 2 + 0.006), ys[i + 1] + node_h / 2 + 0.006)
        # exclusion box sits between node i and node i+1
        if i < len(exclusions):
            y_mid = (ys[i] + ys[i + 1]) / 2
            _exclusion(ax, x_main + node_w / 2 - 0.01, x_excl, y_mid, excl_w, excl_h, exclusions[i])

    # terminal split annotation (train/val/test) if the eval summary is present
    summary = _read_json(art.eval_dir / "eval_twin_summary.json")
    sizes = summary.get("split_sizes") if summary else None
    subtitle = f"Analytic cohort N = {n_final} of {n_raw} exported records ({100*n_final/n_raw:.1f}% retained)"
    if sizes:
        subtitle += (f"   |   split: train {sizes.get('train','?')} / "
                     f"val {sizes.get('val','?')} / test {sizes.get('test','?')}")
    fig.suptitle("Cohort selection (CONSORT)", y=0.985)
    style.caption(fig, subtitle, y=0.055, size=7.2)
    style.caption(fig,
                  "Prior GLP-1 users (PriorGLP1 = 0) are excluded at the SQL cohort-definition step and reported "
                  "explicitly above; on the local CSV export that step is pre-applied and its per-clause count is VM-only.",
                  y=0.035, size=6.6)

    return style.save_figure(fig, out_stem)
