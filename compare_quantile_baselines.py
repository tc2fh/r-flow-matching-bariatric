"""Flow matching vs. quantile baselines for the BMI / HbA1c trajectory (the head-to-head).

The digital-twin flow claims that modelling each patient's BMI/HbA1c trajectory as a
*conditional generative* process beats strong probabilistic regressors. The point baselines
in ``baselines_trajectory`` (XGBoost, Ridge) can only answer "does the flow beat a point
predictor" - their CRPS degenerates to MAE and they have no calibrated interval. This script
runs the harder, fairer test the collaborator asked for: does the flow beat genuinely
*distribution-valued* baselines -

  * ``qgbm`` - QUANTILE GRADIENT BOOSTING: one pinball-loss XGBoost per horizon predicting a
               whole grid of conditional quantiles at once (``reg:quantileerror``), and
  * ``qreg`` - CONDITIONAL QUANTILE REGRESSION: linear Koenker-Bassett quantile regression,
               one linear program per quantile level.

Every arm is scored by IDENTICAL code on the SAME shared split, the SAME conditioning
information, and the SAME observed test rows. Because the quantile arms' rearranged quantile
grid is a valid predictive ensemble ``[n_test, n_quantiles, n_horizon]`` - the exact shape of
the flow's per-patient sample block - they flow through the repo's proper-scoring machinery
(``baselines_trajectory.horizon_score`` for MAD/RMSE/CRPS/NLL and ``distributional_metrics``
for coverage / interval score / pinball / sharpness) with no special-casing. The comparison is
reported per horizon and pooled by group (BMI, HbA1c, overall), with paired Wilcoxon + t tests
on per-patient CRPS (flow as the reference arm) and a summary figure of CRPS-over-time and
coverage calibration.

Typical use (after a twin run / pipeline exists)::

    OMP_NUM_THREADS=1 python compare_quantile_baselines.py --pipeline runs/twin_pipeline/<ts>
    OMP_NUM_THREADS=1 python compare_quantile_baselines.py --twin-run runs/.../twin_final \
        --csv fake_data/fake_mbs_cohort.csv

Without a trained twin (baselines only, e.g. a pre-training sanity check)::

    OMP_NUM_THREADS=1 python compare_quantile_baselines.py --csv fake_data/fake_mbs_cohort.csv --no-flow
"""

from __future__ import annotations

import os
import sys

# macOS dual-OpenMP guard (xgboost + torch): pin OMP threads before torch loads. See
# baselines_trajectory for the rationale; harmless on Linux.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: this is a batch script, never an interactive session
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train_flow_matching as fm
import train_flow_matching_multitask as mt
import train_flow_matching_twin as tw
import baselines_trajectory as bt
import distributional_metrics as dm
import evaluate_twin as et


# Flow arm(s) carry a predictive spread; so do the quantile arms. The point arms (if included)
# do not, so their NLL/coverage stay NaN - decided per arm from the sample count, not a list.
ARM_LABELS = {
    "event_flow": "flow (event-conditioned)",
    "no_event_flow": "flow (no event)",
    bt.ARM_QGBM: "quantile GBM",
    bt.ARM_QREG: "quantile regression",
    bt.ARM_XGB: "xgboost (point)",
    bt.ARM_RIDGE: "ridge (point)",
}
ARM_ORDER = ["event_flow", "no_event_flow", bt.ARM_QGBM, bt.ARM_QREG, bt.ARM_XGB, bt.ARM_RIDGE]
ARM_COLORS = {
    "event_flow": "#1b6ca8", "no_event_flow": "#7fb2d6",
    bt.ARM_QGBM: "#d1495b", bt.ARM_QREG: "#edae49",
    bt.ARM_XGB: "#66a182", bt.ARM_RIDGE: "#8d99ae",
}
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)


# --------------------------------------------------------------------------- #
# Horizon-time parsing (for the "over time" x-axis)
# --------------------------------------------------------------------------- #
def horizon_months(name: str) -> float:
    """Months-since-surgery for a horizon name like ``bmi_3m`` / ``hba1c_2y`` (for the x-axis)."""
    token = name.split("_")[-1]
    if token.endswith("m"):
        return float(token[:-1])
    if token.endswith("y"):
        return float(token[:-1]) * 12.0
    return float("nan")


# --------------------------------------------------------------------------- #
# Arm assembly
# --------------------------------------------------------------------------- #
def assemble_arms(args) -> dict:
    """Fit / sample every requested arm on ONE shared split and return aligned test arrays.

    Returns a dict with ``obs``/``mask`` ([n_test, 15] original units, test order),
    ``cont_names``/``cont_groups``, ``arm_samples`` ({arm: [n_test, m, 15]}), ``quantiles``,
    and provenance fields for the summary. The flow arm(s) reuse ``evaluate_twin``'s sampler so
    the flow is scored exactly as it is elsewhere in the repo; the baselines reuse
    ``baselines_trajectory`` so they see the flow's conditioning + split.
    """
    device = et.choose_device(args.device)
    dataset = et.load_dataset(Path(args.csv) if args.csv else None)

    # The split/preprocessing come from the flow's own config when a twin run is given, so every
    # baseline is trained and tested on the flow's exact partition. Without a flow, build the
    # split from the CLI split args (baselines-only sanity mode).
    event_twin_run = None
    if not args.no_flow:
        event_twin_run = _resolve_event_run(args)
        cfg = et.load_twin_config(event_twin_run)
    else:
        cfg = tw.TwinConfig(split_strategy=args.split_strategy, split_seed=args.split_seed,
                            train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac)

    splits = tw.make_splits(dataset, cfg)
    test_idx = splits["test"]
    pre = mt.fit_preprocessing(dataset, splits["train"])
    test_arrays = mt.split_arrays(dataset, splits, pre)["test"]
    obs, mask = test_arrays["original_x"], test_arrays["original_mask"]  # [n_test, 15] original units

    arm_samples: dict[str, np.ndarray] = {}

    if not args.no_flow:
        event_model = et.restore_twin(event_twin_run, cfg, device)
        arm_samples["event_flow"] = et._sample_twin_arm(
            event_model, dataset, test_idx, pre, cfg, args.n_samples, args.n_steps, device)
        if args.with_noevent_flow:
            noevent_model, noevent_cfg, _ = et._resolve_noevent_arm(
                dataset, cfg, None, args.noevent_num_steps, Path(args.output_dir or "."), device)
            arm_samples["no_event_flow"] = et._sample_twin_arm(
                noevent_model, dataset, test_idx, pre, noevent_cfg, args.n_samples, args.n_steps, device)

    # Quantile arms (the point of this script): real predictive ensembles [n_test, n_q, 15].
    q_baselines = bt.fit_quantile_baselines(
        dataset, splits, quantiles=bt.DEFAULT_QUANTILES, use_event=not args.baseline_no_event, seed=args.seed)
    arm_samples[bt.ARM_QGBM] = q_baselines["qgbm_pred"]
    arm_samples[bt.ARM_QREG] = q_baselines["qreg_pred"]

    # Optional point arms, so the table can also show the "does the flow beat a point predictor"
    # question next to the harder distributional one. Degenerate one-sample "ensembles".
    point_baselines = None
    if args.with_point:
        point_baselines = bt.fit_trajectory_baselines(
            dataset, splits, use_event=not args.baseline_no_event, seed=args.seed)
        arm_samples[bt.ARM_XGB] = point_baselines["xgb_pred"][:, None, :]
        arm_samples[bt.ARM_RIDGE] = point_baselines["ridge_pred"][:, None, :]

    present = [a for a in ARM_ORDER if a in arm_samples]
    return {
        "dataset_source": dataset.source_label,
        "event_twin_run": str(event_twin_run) if event_twin_run else None,
        "obs": obs, "mask": mask,
        "cont_names": list(tw.CONT_NAMES), "cont_groups": list(tw.CONT_GROUPS),
        "arm_samples": arm_samples, "present": present,
        "quantiles": q_baselines["quantiles"],
        "baseline_feature_names": q_baselines["feature_names"],
        "n_test": int(test_idx.size),
    }


def _resolve_event_run(args) -> Path:
    """Resolve the event-conditioned twin run dir from --twin-run or --pipeline (or newest)."""
    if args.twin_run:
        return Path(args.twin_run)
    pipeline = Path(args.pipeline) if args.pipeline else et.find_latest_pipeline()
    if pipeline is None:
        raise SystemExit("Provide --twin-run or --pipeline (or run with --no-flow for baselines only).")
    manifest = et.resolve_from_pipeline(pipeline)
    return Path(manifest["twin_final_run_dir"])


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _observed_target(obs_h: np.ndarray, mask_h: np.ndarray, group: str) -> np.ndarray:
    """Observed target with unobserved / physiologically-implausible rows set to NaN.

    This is the SAME observed set ``horizon_score`` scores on (mask==1 AND in-range), so the
    calibration read-outs from ``distributional_metrics`` (which mask on finite obs) grade
    exactly the patients the proper scores do."""
    obs_h = np.asarray(obs_h, dtype=np.float64)
    keep = (np.asarray(mask_h) == 1) & bt.plausible_mask(obs_h, bt.PHYSIOLOGIC_BOUNDS.get(group))
    return np.where(keep, obs_h, np.nan)


def score_all(bundle: dict) -> dict:
    """Score every present arm per horizon and pooled by group.

    Returns ``metrics`` (MAD/RMSE/CRPS/NLL rows), ``calibration`` (coverage/interval/pinball/
    sharpness rows), and ``per_patient`` (per-arm per-horizon score dicts, for paired tests).
    """
    obs, mask = bundle["obs"], bundle["mask"]
    cont_names, cont_groups = bundle["cont_names"], bundle["cont_groups"]
    present, arm_samples = bundle["present"], bundle["arm_samples"]

    per_patient = {a: {} for a in present}
    metric_rows, calib_rows = [], []

    for h, name in enumerate(cont_names):
        group = cont_groups[h]
        obs_eff = _observed_target(obs[:, h], mask[:, h], group)
        for arm in present:
            s_h = arm_samples[arm][:, :, h]                      # [n_test, m]
            has_density = s_h.shape[1] > 1
            score = bt.horizon_score(s_h, obs[:, h], mask[:, h], has_density=has_density,
                                     obs_bounds=bt.PHYSIOLOGIC_BOUNDS.get(group))
            per_patient[arm][h] = score
            metric_rows.append({"horizon": name, "group": group, "arm": arm, "n_obs": score["n_obs"],
                                 "mad": score["mad"], "rmse": score["rmse"],
                                 "crps": score["crps"], "nll": score["nll"]})
            calib_rows.append({"horizon": name, "group": group, "arm": arm,
                               **_calibration_row(s_h, obs_eff, has_density)})

    group_dims = {"overall": list(range(len(cont_names))),
                  "bmi": [i for i, g in enumerate(cont_groups) if g == "bmi"],
                  "hba1c": [i for i, g in enumerate(cont_groups) if g == "hba1c"]}
    for gname, dims in group_dims.items():
        for arm in present:
            abs_err = _pool(per_patient, arm, dims, "abs_err")
            sq_err = _pool(per_patient, arm, dims, "sq_err")
            crps_pp = _pool(per_patient, arm, dims, "crps_pp")
            nll_pp = _pool(per_patient, arm, dims, "nll_pp")
            metric_rows.append({"horizon": f"__{gname}__", "group": gname, "arm": arm, "n_obs": int(abs_err.size),
                                "mad": float(np.median(abs_err)) if abs_err.size else float("nan"),
                                "rmse": float(np.sqrt(np.mean(sq_err))) if sq_err.size else float("nan"),
                                "crps": float(np.mean(crps_pp)) if crps_pp.size else float("nan"),
                                "nll": float(np.nanmean(nll_pp)) if np.isfinite(nll_pp).any() else float("nan")})
            # Pooled calibration: stack the [n, m] blocks and obs across the group's horizons.
            s_pool, obs_pool, has_density = _pool_samples(arm_samples, arm, dims, obs, mask, cont_groups)
            calib_rows.append({"horizon": f"__{gname}__", "group": gname, "arm": arm,
                               **_calibration_row(s_pool, obs_pool, has_density)})

    return {"metrics": pd.DataFrame(metric_rows), "calibration": pd.DataFrame(calib_rows),
            "per_patient": per_patient}


def _calibration_row(s_h: np.ndarray, obs_eff: np.ndarray, has_density: bool) -> dict:
    """Coverage@levels + widths + 90% interval score + median pinball + sharpness for one block.

    A point arm (m==1) has no interval, so every calibration cell is NaN. Uses the repo's
    ``distributional_metrics`` so the coverage / interval / pinball conventions match the rest of
    the evaluation suite (guards fewer than ``dm.MIN_N`` observed rows by returning NaN)."""
    row = {f"cov_{int(c * 100)}": float("nan") for c in COVERAGE_LEVELS}
    row.update({f"width_{int(c * 100)}": float("nan") for c in COVERAGE_LEVELS})
    row.update({"interval_score_90": float("nan"), "pinball_50": float("nan"), "mean_sd": float("nan")})
    if not has_density:
        return row
    for r in dm.coverage_curve(s_h, obs_eff, levels=COVERAGE_LEVELS):
        c = int(r["nominal"] * 100)
        row[f"cov_{c}"] = r["empirical"]
        row[f"width_{c}"] = r["mean_width"]
    row["interval_score_90"], _ = dm.interval_score(s_h, obs_eff, alpha=0.10)
    row["pinball_50"] = dm.pinball_loss(s_h, obs_eff, quantiles=(0.5,)).get(0.5, float("nan"))
    row["mean_sd"] = dm.sharpness(s_h)["mean_sd"]
    return row


def _pool(per_patient: dict, arm: str, dims: list[int], key: str) -> np.ndarray:
    if not dims:
        return np.array([], dtype=np.float64)
    return np.concatenate([per_patient[arm][h][key] for h in dims])


def _pool_samples(arm_samples, arm, dims, obs, mask, cont_groups):
    """Stack an arm's [n, m] blocks and matching observed targets across a group's horizons."""
    if not dims:
        return np.empty((0, 1)), np.array([]), False
    blocks = [arm_samples[arm][:, :, h] for h in dims]
    obs_blocks = [_observed_target(obs[:, h], mask[:, h], cont_groups[h]) for h in dims]
    s_pool = np.concatenate(blocks, axis=0)
    obs_pool = np.concatenate(obs_blocks, axis=0)
    return s_pool, obs_pool, s_pool.shape[1] > 1


# --------------------------------------------------------------------------- #
# Paired tests (flow as reference)
# --------------------------------------------------------------------------- #
def paired_tests(bundle: dict, per_patient: dict) -> pd.DataFrame:
    """Paired Wilcoxon + t on per-patient CRPS: reference arm vs every other present arm.

    Reference is the event-conditioned flow when present, else the first present arm. A negative
    ``mean_diff`` means the reference (flow) scores LOWER CRPS - i.e. the flow wins."""
    present = bundle["present"]
    cont_names, cont_groups = bundle["cont_names"], bundle["cont_groups"]
    ref = "event_flow" if "event_flow" in present else present[0]
    others = [a for a in present if a != ref]
    group_dims = {"overall": list(range(len(cont_names))),
                  "bmi": [i for i, g in enumerate(cont_groups) if g == "bmi"],
                  "hba1c": [i for i, g in enumerate(cont_groups) if g == "hba1c"]}
    rows = []
    for h, name in enumerate(cont_names):
        for other in others:
            pt = bt.paired_test(per_patient[ref][h]["crps_pp"], per_patient[other][h]["crps_pp"])
            rows.append({"horizon": name, "group": cont_groups[h], "comparison": f"{ref}_vs_{other}", **pt})
    for gname, dims in group_dims.items():
        for other in others:
            a = np.concatenate([per_patient[ref][h]["crps_pp"] for h in dims]) if dims else np.array([])
            b = np.concatenate([per_patient[other][h]["crps_pp"] for h in dims]) if dims else np.array([])
            pt = bt.paired_test(a, b)
            rows.append({"horizon": f"__{gname}__", "group": gname, "comparison": f"{ref}_vs_{other}", **pt})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Scorecard (pooled CRPS + NLL per arm) - the compact "figure or table" of scores
# --------------------------------------------------------------------------- #
SCORECARD_KEYS = ["crps_overall", "crps_bmi", "crps_hba1c", "nll_overall", "nll_bmi", "nll_hba1c"]
SCORECARD_COLS = ["CRPS\noverall", "CRPS\nBMI", "CRPS\nHbA1c", "NLL\noverall", "NLL\nBMI", "NLL\nHbA1c"]


def build_scorecard(metrics: pd.DataFrame, present: list[str]) -> pd.DataFrame:
    """Pooled CRPS + Gaussian predictive NLL per arm (overall / BMI / HbA1c), one row per arm.

    CRPS is defined for every arm (point arms' CRPS == MAE); NLL only for the spread-carrying
    arms (flow, qgbm, qreg) - it stays NaN (rendered ``n/a``) for the xgb/ridge point arms."""
    pooled = metrics[metrics["horizon"].str.startswith("__")]
    rows = []
    for arm in present:
        row = {"arm": arm, "label": ARM_LABELS.get(arm, arm)}
        for g in ("overall", "bmi", "hba1c"):
            sub = pooled[(pooled["arm"] == arm) & (pooled["group"] == g)]
            row[f"crps_{g}"] = float(sub["crps"].iloc[0]) if len(sub) else float("nan")
            row[f"nll_{g}"] = float(sub["nll"].iloc[0]) if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_score(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.1e}" if abs(v) >= 1000 else f"{v:.3f}"


def _render_scorecard(ax, scorecard: pd.DataFrame) -> None:
    """Draw the pooled CRPS/NLL scorecard as a table in ``ax``; bold the best (min) per column."""
    ax.axis("off")
    best = {}
    for k in SCORECARD_KEYS:
        vals = scorecard[k].to_numpy(dtype=float)
        best[k] = int(np.nanargmin(vals)) if np.isfinite(vals).any() else -1
    cell_text = [[_fmt_score(scorecard.iloc[i][k]) for k in SCORECARD_KEYS] for i in range(len(scorecard))]
    tbl = ax.table(cellText=cell_text, rowLabels=list(scorecard["label"]),
                   colLabels=SCORECARD_COLS, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.5)
    for j, k in enumerate(SCORECARD_KEYS):
        if best[k] >= 0:
            tbl[best[k] + 1, j].set_text_props(weight="bold", color="#0a7d0a")  # +1 for header row
    ax.set_title("Pooled scorecard: CRPS & NLL (bold = best; lower is better)", fontsize=10)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(bundle: dict, metrics: pd.DataFrame, calibration: pd.DataFrame,
                scorecard: pd.DataFrame, out_path: Path) -> None:
    """Six-panel summary: CRPS-over-time (BMI/HbA1c), coverage, pooled CRPS + NLL bars, scorecard."""
    present = bundle["present"]
    per_h = metrics[~metrics["horizon"].str.startswith("__")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    # Row 1, cols 0-1: CRPS over time for BMI and HbA1c.
    for ax, group, title in ((axes[0, 0], "bmi", "BMI trajectory"),
                             (axes[0, 1], "hba1c", "HbA1c trajectory")):
        sub = per_h[per_h["group"] == group]
        names = [n for n in bundle["cont_names"] if n in set(sub["horizon"])]
        months = [horizon_months(n) for n in names]
        for arm in present:
            arm_sub = sub[sub["arm"] == arm].set_index("horizon").reindex(names)
            ax.plot(months, arm_sub["crps"].to_numpy(), marker="o", ms=4,
                    color=ARM_COLORS.get(arm), label=ARM_LABELS.get(arm, arm))
        ax.set_title(f"{title}: CRPS over time (lower is better)")
        ax.set_xlabel("months since surgery")
        ax.set_ylabel("CRPS (original units)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")

    # Row 1, col 2: coverage calibration (pooled overall) - nominal vs empirical, diagonal = ideal.
    ax = axes[0, 2]
    overall_cal = calibration[calibration["horizon"] == "__overall__"].set_index("arm")
    noms = np.array(COVERAGE_LEVELS)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="ideal")
    any_cov = False
    for arm in present:
        if arm not in overall_cal.index:
            continue
        emp = np.array([overall_cal.loc[arm, f"cov_{int(c * 100)}"] for c in COVERAGE_LEVELS], dtype=float)
        if np.isfinite(emp).any():
            any_cov = True
            ax.plot(noms, emp, marker="s", ms=5, color=ARM_COLORS.get(arm), label=ARM_LABELS.get(arm, arm))
    ax.set_title("Coverage calibration (pooled, overall)")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="best")
    if not any_cov:
        ax.text(0.5, 0.5, f"coverage NaN\n(<{dm.MIN_N} obs/group)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")

    # Row 2, col 0: pooled CRPS by arm (BMI vs HbA1c) - the "who wins on sharpness" bar.
    _pooled_bars(axes[1, 0], metrics, present, "crps", "Pooled CRPS by arm (lower is better)",
                 "pooled CRPS", symlog=False)
    # Row 2, col 1: pooled NLL by arm (density arms only; symlog absorbs fake-data extremes).
    _pooled_bars(axes[1, 1], metrics, [a for a in present if a in _density_arms(bundle)], "nll",
                 "Pooled predictive NLL by arm (density arms; lower is better)", "pooled NLL", symlog=True)
    # Row 2, col 2: the exact pooled CRPS/NLL numbers as a table.
    _render_scorecard(axes[1, 2], scorecard)

    fig.suptitle("Flow matching vs. quantile baselines - BMI / HbA1c trajectory (CRPS & NLL)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _density_arms(bundle: dict) -> list[str]:
    """Arms carrying a predictive spread (>1 sample) -> NLL is defined."""
    return [a for a in bundle["present"] if bundle["arm_samples"][a].shape[1] > 1]


def _pooled_bars(ax, metrics: pd.DataFrame, arms: list[str], metric: str, title: str,
                 ylabel: str, symlog: bool) -> None:
    """Grouped BMI/HbA1c bars of a pooled metric per arm."""
    pooled = metrics[metrics["horizon"].isin(["__bmi__", "__hba1c__"])]
    if not arms:
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "no density arms", ha="center", va="center", transform=ax.transAxes, color="gray")
        return
    x = np.arange(len(arms))
    w = 0.38
    bmi_vals = [pooled[(pooled["arm"] == a) & (pooled["group"] == "bmi")][metric].mean() for a in arms]
    hb_vals = [pooled[(pooled["arm"] == a) & (pooled["group"] == "hba1c")][metric].mean() for a in arms]
    ax.bar(x - w / 2, bmi_vals, w, label="BMI", color="#1b6ca8")
    ax.bar(x + w / 2, hb_vals, w, label="HbA1c", color="#d1495b")
    if symlog:
        ax.set_yscale("symlog")  # NLL can span orders of magnitude on tiny data; symlog keeps all bars legible
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS.get(a, a) for a in arms], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    bundle = assemble_arms(args)
    scored = score_all(bundle)
    metrics, calibration = scored["metrics"], scored["calibration"]
    paired = paired_tests(bundle, scored["per_patient"])
    scorecard = build_scorecard(metrics, bundle["present"])

    out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(bundle)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "quantile_comparison_metrics.csv"
    calib_path = out_dir / "quantile_comparison_calibration.csv"
    paired_path = out_dir / "quantile_comparison_paired_tests.csv"
    scorecard_path = out_dir / "quantile_comparison_scorecard.csv"
    fig_path = out_dir / "quantile_comparison.png"
    summary_path = out_dir / "quantile_comparison_summary.json"

    metrics.to_csv(metrics_path, index=False)
    calibration.to_csv(calib_path, index=False)
    paired.to_csv(paired_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    make_figure(bundle, metrics, calibration, scorecard, fig_path)

    summary = {
        "output_dir": str(out_dir),
        "dataset_source": bundle["dataset_source"],
        "event_twin_run": bundle["event_twin_run"],
        "arms": bundle["present"],
        "arm_labels": {a: ARM_LABELS.get(a, a) for a in bundle["present"]},
        "n_test": bundle["n_test"],
        "n_quantiles": int(np.asarray(bundle["quantiles"]).size),
        "quantiles": [float(q) for q in np.asarray(bundle["quantiles"])],
        "n_samples_flow": args.n_samples,
        "n_steps_flow": args.n_steps,
        "baseline_feature_names": bundle["baseline_feature_names"],
        "reference_arm": "event_flow" if "event_flow" in bundle["present"] else bundle["present"][0],
        "metrics_csv": str(metrics_path),
        "calibration_csv": str(calib_path),
        "paired_tests_csv": str(paired_path),
        "scorecard_csv": str(scorecard_path),
        "figure": str(fig_path),
        "notes": {
            "crps": "proper score defined for every arm; point arms' CRPS == MAE (degenerate ensemble)",
            "nll": "moment-matched Gaussian predictive NLL; NaN for point arms (no spread)",
            "coverage": "empirical coverage of the central band at each nominal level; NaN if <"
                        f"{dm.MIN_N} observed rows in the (pooled) block",
            "paired_test": "Wilcoxon signed-rank + paired t on per-patient CRPS; mean_diff<0 => reference (flow) better",
            "qgbm": "quantile gradient boosting (xgboost reg:quantileerror, all quantiles jointly, rearranged)",
            "qreg": "linear conditional quantile regression (Koenker-Bassett; sklearn QuantileRegressor, per level)",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    _print_summary(bundle, metrics, calibration, paired, out_dir)
    return summary


def _default_output_dir(bundle: dict) -> Path:
    if bundle["event_twin_run"]:
        return Path(bundle["event_twin_run"]) / "quantile_comparison"
    return fm.REPO_ROOT / "runs" / "quantile_comparison"


def _print_summary(bundle, metrics, calibration, paired, out_dir) -> None:
    present = bundle["present"]
    ref = "event_flow" if "event_flow" in present else present[0]
    print("\n" + "=" * 78)
    print(f"Flow vs. quantile baselines  |  arms: {', '.join(present)}  |  n_test={bundle['n_test']}")
    print("=" * 78)
    pooled = metrics[metrics["horizon"].str.startswith("__")].copy()
    pooled["arm"] = pd.Categorical(pooled["arm"], categories=[a for a in ARM_ORDER if a in present], ordered=True)
    print("\nPooled MAD / RMSE / CRPS / NLL by group (lower is better):")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        show = pooled.sort_values(["group", "arm"])[["group", "arm", "n_obs", "mad", "rmse", "crps", "nll"]]
        print(show.round(4).to_string(index=False))
    ov_cal = calibration[calibration["horizon"] == "__overall__"]
    cov_cols = [f"cov_{int(c * 100)}" for c in COVERAGE_LEVELS]
    if ov_cal[cov_cols].notna().any().any():
        print("\nPooled coverage (overall) - empirical vs nominal 50/80/90/95:")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(ov_cal[["arm"] + cov_cols + ["interval_score_90"]].round(3).to_string(index=False))
    else:
        print(f"\nCoverage NaN on this split (<{dm.MIN_N} observed rows/group); populated on the real cohort.")
    print(f"\nPaired CRPS tests vs reference '{ref}' (overall; mean_diff<0 => flow better):")
    ov = paired[paired["group"] == "overall"][["comparison", "n_pairs", "mean_diff", "wilcoxon_p", "ttest_p"]]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(ov.round(4).to_string(index=False))
    print(f"\nSaved comparison + figure to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", "--csv-path", dest="csv", type=str, default=None,
                   help="Cohort CSV (defaults to the twin run's / repo fake cohort).")
    p.add_argument("--twin-run", dest="twin_run", type=str, default=None,
                   help="Event-conditioned twin run dir (the flow arm).")
    p.add_argument("--pipeline", type=str, default=None,
                   help="Twin pipeline dir; the event arm is resolved from its manifest.")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-flow", dest="no_flow", action="store_true",
                   help="Score the quantile baselines only (no trained twin needed).")
    p.add_argument("--with-noevent-flow", dest="with_noevent_flow", action="store_true",
                   help="Also include the no-event flow arm.")
    p.add_argument("--with-point", dest="with_point", action="store_true",
                   help="Also include the xgb/ridge point baselines.")
    p.add_argument("--baseline-no-event", dest="baseline_no_event", action="store_true",
                   help="Drop the event feature from the baselines (mirror the no-event flow).")
    p.add_argument("--n-samples", type=int, default=200, help="Flow predictive draws per patient.")
    p.add_argument("--n-steps", type=int, default=50, help="Flow ODE sampling steps.")
    p.add_argument("--noevent-num-steps", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    # Split args (used only in --no-flow mode; otherwise inherited from the twin config).
    p.add_argument("--split-strategy", type=str, default="surgery", choices=["surgery", "outcome"])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
