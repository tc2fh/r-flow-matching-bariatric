"""Evaluate a saved GBM composite-MACE baseline run.

The baseline trains fast and deterministically and does not serialize the tree
model, so this evaluator reloads the run's ``config.json`` and *retrains* the
estimator (identical seed/config -> identical model) before producing a richer,
Cosmos-comparable MACE evaluation:

    eval_mace_metrics_test.csv          AUROC / AUPRC / Brier (raw + calibrated)
    eval_mace_operating_points_test.csv default / Youden / fixed-specificity
    eval_mace_predictions_test.csv      per-patient factual + counterfactual risk
    eval_mace_curves_test.png           ROC / PR / reliability
    eval_mace_counterfactual_test.png   surgery counterfactual (flip CPT)
    eval_summary.json

Run::

    python evaluate_gbm_mace_baseline.py --run runs/gbm_mace_baseline/<run> \
        --csv fake_data/fake_mbs_cohort.csv
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, roc_curve

import gbm_mace_baseline as gb
import train_flow_matching as fm
import evaluate_flow_matching as ev  # for the shared render_table helper

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


DEFAULT_LOG_DIR = Path("runs/gbm_mace_baseline")
PRIMARY_VARIANT_BALANCED = True  # the variant used for curves/counterfactual plots


# --------------------------------------------------------------------------- #
# Run resolution / loading
# --------------------------------------------------------------------------- #
def is_gbm_run_dir(path: Path) -> bool:
    config = path / "config.json"
    if not config.exists():
        return False
    try:
        raw = json.loads(config.read_text(encoding="utf-8"))
    except Exception:
        return False
    return "feature_names" in raw or raw.get("label") == gb.MACE_LABEL_NAME


def resolve_run_dir(path: Path | None, log_dir: Path) -> Path:
    search_root = log_dir if path is None else Path(path)
    if is_gbm_run_dir(search_root):
        return search_root
    if search_root.exists():
        candidates = [p for p in search_root.rglob("run_*") if p.is_dir() and is_gbm_run_dir(p)]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    raise SystemExit(f"No GBM baseline run with config.json found under {search_root}")


def load_config(run_dir: Path) -> gb.GBMConfig:
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    valid = {field.name for field in fields(gb.GBMConfig)}
    return gb.GBMConfig(**{key: value for key, value in raw.items() if key in valid})


def load_dataset(csv_path: Path | None) -> fm.FlowDataset:
    if csv_path is not None:
        return fm.load_dataset_from_csv(csv_path)
    try:
        return fm.load_dataset_from_database()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\n\nPass --csv <path> to evaluate from a saved CSV export.") from exc


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_curves(y_true: np.ndarray, prob: np.ndarray, output_path: Path) -> None:
    prevalence = float(y_true.mean()) if y_true.size else float("nan")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    try:
        fpr, tpr, _ = roc_curve(y_true, prob)
        axes[0].plot(fpr, tpr)
    except Exception:
        pass
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    try:
        prec, rec, _ = precision_recall_curve(y_true, prob)
        axes[1].plot(rec, prec)
    except Exception:
        pass
    axes[1].axhline(prevalence, color="k", ls="--", alpha=0.4)
    axes[1].set(title=f"Precision-Recall (test, prevalence={prevalence:.3f})", xlabel="Recall", ylabel="Precision")
    table = gb.reliability_table(y_true, prob)
    if not table.empty:
        axes[2].plot(table["mean_pred"], table["frac_pos"], marker="o")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[2].set(title="Reliability (test)", xlabel="Mean predicted", ylabel="Observed frequency")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    gb.report_saved(output_path)


def plot_counterfactual(
    surgery_type: np.ndarray, factual: np.ndarray, counterfactual: np.ndarray, output_path: Path
) -> dict[str, Any]:
    delta = counterfactual - factual
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"sleeve": "tab:blue", "rnygb": "tab:orange"}
    for proc in ("sleeve", "rnygb"):
        sel = surgery_type == proc
        if sel.any():
            axes[0].scatter(factual[sel], counterfactual[sel], s=14, alpha=0.5, color=colors.get(proc, "tab:gray"), label=f"factual {proc}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set(title="Per-patient MACE risk: factual vs counterfactual surgery", xlabel="Factual risk", ylabel="Counterfactual risk", xlim=[0, 1], ylim=[0, 1])
    axes[0].legend(fontsize=8)
    axes[1].hist(delta, bins=40, color="tab:green", alpha=0.8)
    axes[1].axvline(0.0, color="k", lw=1.0)
    axes[1].axvline(float(np.median(delta)), color="tab:red", lw=2.0, label=f"median delta={np.median(delta):.4f}")
    axes[1].set(title="Counterfactual - factual MACE risk", xlabel="Delta risk", ylabel="Patients")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    gb.report_saved(output_path)

    summary: dict[str, Any] = {
        "median_counterfactual_minus_factual": float(np.median(delta)),
        "mean_counterfactual_minus_factual": float(np.mean(delta)),
        "frac_increased_risk": float(np.mean(delta > 0)),
    }
    if scipy_stats is not None and np.any(delta != 0):
        try:
            wilcoxon = scipy_stats.wilcoxon(counterfactual, factual)
            summary["wilcoxon_statistic"] = float(wilcoxon.statistic)
            summary["wilcoxon_p_value"] = float(wilcoxon.pvalue)
        except Exception:
            summary["wilcoxon_p_value"] = None
    return summary


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_run(run_dir: Path, output_dir: Path | None, csv_path: Path | None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = run_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(run_dir)
    dataset = load_dataset(csv_path)
    x, feature_names, y = gb.assemble_features(dataset)
    splits = gb.make_splits(dataset, cfg)
    test_idx = splits["test"]
    surgery_col = feature_names.index("surgery_idx")

    metric_rows: list[dict] = []
    operating_rows: list[dict] = []
    predictions = pd.DataFrame(
        {
            "subject_id": dataset.subject_ids[test_idx],
            "surgery": dataset.surgery_type[test_idx],
            "y_true": y[test_idx],
        }
    )
    counterfactual_summary: dict[str, Any] = {}
    primary_prob = None

    for balanced in (False, True):
        variant = "balanced" if balanced else "unweighted"
        result = gb.fit_predict_variant(cfg, balanced, x, y, splits)
        for row in result["metric_rows"]:
            if row["split"] in ("test", "test_calibrated"):
                metric_rows.append(row)
        for op in result["operating_points"]:
            operating_rows.append(op)
        predictions[f"prob_{variant}"] = result["test_prob"]
        if result["calibrated"]:
            predictions[f"prob_{variant}_calibrated"] = result["test_prob_calibrated"]

        # Surgery counterfactual: flip the CPT-derived surgery feature, re-predict.
        x_cf = x.copy()
        x_cf[:, surgery_col] = 1.0 - x_cf[:, surgery_col]
        prob_cf = result["estimator"].predict_proba(x_cf[test_idx])[:, 1]
        predictions[f"prob_{variant}_counterfactual"] = prob_cf

        if balanced == PRIMARY_VARIANT_BALANCED:
            primary_prob = result["test_prob"]
            counterfactual_summary = plot_counterfactual(
                dataset.surgery_type[test_idx], result["test_prob"], prob_cf,
                output_dir / "eval_mace_counterfactual_test.png",
            )

    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "eval_mace_metrics_test.csv"
    metrics.to_csv(metrics_path, index=False)
    gb.report_saved(metrics_path, "MACE discrimination metrics")
    ev.render_table(metrics, output_dir / "eval_mace_metrics_test.png", "Composite MACE discrimination (test)")
    operating_df = pd.DataFrame(operating_rows)
    op_path = output_dir / "eval_mace_operating_points_test.csv"
    operating_df.to_csv(op_path, index=False)
    gb.report_saved(op_path, "MACE operating points")
    ev.render_table(operating_df, output_dir / "eval_mace_operating_points_test.png", "MACE operating points (test)")
    pred_path = output_dir / "eval_mace_predictions_test.csv"
    predictions.to_csv(pred_path, index=False)
    gb.report_saved(pred_path, "MACE per-patient predictions (factual + counterfactual)")
    if primary_prob is not None:
        plot_curves(y[test_idx], primary_prob, output_dir / "eval_mace_curves_test.png")

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "csv_path": None if csv_path is None else str(csv_path),
        "model": "gbm_mace_baseline",
        "primary_variant": "balanced" if PRIMARY_VARIANT_BALANCED else "unweighted",
        "split_strategy": cfg.split_strategy,
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "feature_names": feature_names,
        "mace_metrics_test": metrics.to_dict(orient="records"),
        "operating_points_test": operating_rows,
        "counterfactual_surgery": counterfactual_summary,
    }
    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    gb.report_saved(summary_path, "evaluation summary")

    print(f"Run: {run_dir}")
    print(f"Split sizes: {summary['split_sizes']}")
    print("MACE classification (test):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(metrics.to_string(index=False))
    print(f"Saved outputs to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="GBM baseline run dir, or a parent to search.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run, args.log_dir)
    evaluate_run(run_dir=run_dir, output_dir=args.output_dir, csv_path=args.csv_path)


if __name__ == "__main__":
    main()
