"""Head-to-head composite-MACE comparison: GBM baseline vs multi-task model.

Runs both models on the SAME held-out test split (it asserts the two runs share
split seed/fractions, then reproduces that split once) and tabulates
AUROC / AUPRC / Brier so the models are directly comparable. Optionally adds a
Cosmos flow-model reference row from that run's saved test predictions -- clearly
flagged, because the Cosmos model uses a different (surgery-only) split.

    python compare_mace_models.py \
        --gbm-run runs/gbm_mace_baseline/<run> \
        --mt-run  runs/python_flow_matching_multitask/<run> \
        --csv fake_data/fake_mbs_cohort.csv \
        [--cosmos-run runs/python_flow_matching_optuna/<study>/best_model/<run>]

Outputs (default runs/mace_model_comparison/run_<ts>/):
    mace_model_comparison.csv
    mace_model_comparison.png
    comparison_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression

import train_flow_matching_multitask as mt
import gbm_mace_baseline as gb
import evaluate_flow_matching as ev
import evaluate_gbm_mace_baseline as egb
import evaluate_multitask as emt


DEFAULT_OUTPUT_DIR = Path("runs/mace_model_comparison")
SPLIT_ATTRS = ("split_strategy", "split_seed", "train_frac", "val_frac", "test_frac")


def assert_split_compatible(gbm_cfg: gb.GBMConfig, mt_cfg: mt.MultiTaskConfig) -> None:
    for attr in SPLIT_ATTRS:
        gbm_value, mt_value = getattr(gbm_cfg, attr), getattr(mt_cfg, attr)
        if gbm_value != mt_value:
            raise SystemExit(
                f"Cannot compare on the same split: {attr} differs (gbm={gbm_value}, multitask={mt_value}).\n"
                "Retrain both models with matching split_seed and train/val/test fractions."
            )


def metrics_row(model: str, variant: str, y_true: np.ndarray, prob: np.ndarray, note: str = "") -> dict[str, Any]:
    m = mt.discrimination_metrics(y_true.astype(np.int64), prob)
    return {
        "model": model,
        "variant": variant,
        "n": m["n"],
        "n_pos": m["n_pos"],
        "prevalence": m["prevalence"],
        "auroc": m["auroc"],
        "auprc": m["auprc"],
        "auprc_baseline": m["auprc_baseline"],
        "brier": m["brier"],
        "note": note,
    }


def isotonic_calibrate(val_prob: np.ndarray, y_val: np.ndarray, test_prob: np.ndarray) -> tuple[np.ndarray, bool]:
    if val_prob.size >= 10 and len(np.unique(y_val)) == 2:
        try:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(val_prob, y_val)
            return iso.transform(test_prob), True
        except Exception:
            return test_prob, False
    return test_prob, False


def cosmos_reference_row(cosmos_run: Path, note: str) -> dict[str, Any] | None:
    pred_path = cosmos_run / "test_predictions.csv"
    if not pred_path.exists():
        print(f"WARNING: cosmos run has no test_predictions.csv at {pred_path}; skipping reference row.")
        return None
    pred = pd.read_csv(pred_path)
    if "pred_mean_mace_ever" not in pred.columns or "observed_mace_ever" not in pred.columns:
        print("WARNING: cosmos test_predictions.csv lacks mace_ever columns; skipping reference row.")
        return None
    mask_col = "observed_mask_mace_ever"
    keep = pred[mask_col] == 1 if mask_col in pred.columns else np.ones(len(pred), dtype=bool)
    y = pred.loc[keep, "observed_mace_ever"].to_numpy(dtype=np.float64)
    prob = pred.loc[keep, "pred_mean_mace_ever"].to_numpy(dtype=np.float64).clip(0.0, 1.0)
    return metrics_row("cosmos_flow", "pred_mean", y, prob, note=note)


def make_output_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def plot_comparison(table: pd.DataFrame, output_path: Path) -> None:
    labels = [f"{row.model}\n{row.variant}" for row in table.itertuples()]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.5 * len(labels)), 5))
    width = 0.4
    axes[0].bar(x - width / 2, table["auroc"].to_numpy(), width, label="AUROC", color="tab:blue")
    axes[0].bar(x + width / 2, table["auprc"].to_numpy(), width, label="AUPRC", color="tab:orange")
    for j, base in enumerate(table["auprc_baseline"].to_numpy()):
        axes[0].plot([x[j] - width, x[j] + width], [base, base], color="k", ls="--", lw=1.0, alpha=0.5)
    axes[0].set(title="Discrimination (higher better; dashed = AUPRC prevalence baseline)", ylabel="score", ylim=(0, 1))
    axes[0].set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    axes[0].legend()
    axes[1].bar(x, table["brier"].to_numpy(), color="tab:green")
    axes[1].set(title="Brier score (lower better)", ylabel="Brier")
    axes[1].set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    ev.report_saved(output_path, "comparison bar chart")


def compare(
    gbm_run: Path,
    mt_run: Path,
    csv_path: Path | None,
    output_dir: Path | None,
    cosmos_run: Path | None,
    device_name: str,
) -> dict[str, Any]:
    gbm_run = egb.resolve_run_dir(gbm_run, egb.DEFAULT_LOG_DIR)
    mt_run = ev.resolve_run_dir(mt_run, emt.DEFAULT_LOG_DIR)
    gbm_cfg = egb.load_config(gbm_run)
    mt_cfg = emt.load_config(mt_run)
    assert_split_compatible(gbm_cfg, mt_cfg)

    device = ev.choose_device(device_name)
    dataset = egb.load_dataset(csv_path)

    # One shared split (gb/mt dispatch identically; same strategy + seed/fracs asserted above).
    y_all = dataset.x[:, mt.MACE_DIM].astype(np.int64)
    splits = mt.make_splits(dataset, mt_cfg)
    val_idx, test_idx = splits["val"], splits["test"]
    y_val, y_test = y_all[val_idx], y_all[test_idx]

    rows: list[dict[str, Any]] = [
        metrics_row("baseline", "predict_prevalence", y_test, np.full(test_idx.size, float(y_test.mean()))),
    ]

    # GBM baseline (both weighting variants, raw + isotonic-calibrated).
    x, _, y = gb.assemble_features(dataset)
    for balanced in (False, True):
        variant = "balanced" if balanced else "unweighted"
        result = gb.fit_predict_variant(gbm_cfg, balanced, x, y, splits)
        rows.append(metrics_row("gbm", variant, y_test, result["test_prob"]))
        if result["calibrated"]:
            rows.append(metrics_row("gbm", f"{variant}_calibrated", y_test, result["test_prob_calibrated"]))

    # Multi-task classification head (raw + isotonic-calibrated).
    _, pre, model = emt.load_run(mt_run, device)
    mt_val_prob = emt.predict_mace(model, dataset, val_idx, pre, device)
    mt_test_prob = emt.predict_mace(model, dataset, test_idx, pre, device)
    rows.append(metrics_row("multitask", "raw", y_test, mt_test_prob))
    mt_test_cal, mt_calibrated = isotonic_calibrate(mt_val_prob, y_val, mt_test_prob)
    if mt_calibrated:
        rows.append(metrics_row("multitask", "calibrated", y_test, mt_test_cal))

    cosmos_note = None
    if cosmos_run is not None:
        cosmos_run = ev.resolve_run_dir(cosmos_run, Path("runs/python_flow_matching_optuna"))
        cosmos_cfg = ev.load_config(cosmos_run)[0]
        same_split = (
            mt_cfg.split_strategy == "surgery"
            and cosmos_cfg.split_seed == mt_cfg.split_seed
            and cosmos_cfg.train_frac == mt_cfg.train_frac
            and cosmos_cfg.val_frac == mt_cfg.val_frac
            and cosmos_cfg.test_frac == mt_cfg.test_frac
        )
        note = "same surgery-stratified split (patient-for-patient)" if same_split else "DIFFERENT test split"
        cosmos = cosmos_reference_row(cosmos_run, note)
        if cosmos is not None:
            rows.append(cosmos)
            cosmos_note = str(cosmos_run)

    table = pd.DataFrame(rows)
    run_dir = make_output_dir(output_dir or DEFAULT_OUTPUT_DIR)
    comparison_csv = run_dir / "mace_model_comparison.csv"
    table.to_csv(comparison_csv, index=False)
    ev.report_saved(comparison_csv, "head-to-head comparison table")
    ev.render_table(table, run_dir / "mace_model_comparison_table.png", "Composite MACE: head-to-head (shared test split)")
    plot_comparison(table, run_dir / "mace_model_comparison.png")

    summary = {
        "gbm_run": str(gbm_run),
        "mt_run": str(mt_run),
        "cosmos_run": cosmos_note,
        "csv_path": None if csv_path is None else str(csv_path),
        "shared_split": {attr: getattr(mt_cfg, attr) for attr in SPLIT_ATTRS},
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "test_prevalence": float(y_test.mean()),
        "comparison": table.to_dict(orient="records"),
    }
    summary_path = run_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ev.report_saved(summary_path, "comparison summary")

    print(f"Shared split (seed={mt_cfg.split_seed}): {summary['split_sizes']}  test_prevalence={y_test.mean():.4f}")
    print("Head-to-head composite-MACE comparison (test split):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.to_string(index=False))
    print(f"Saved comparison to {run_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gbm-run", type=Path, default=None, help="GBM baseline run dir (or parent to search).")
    parser.add_argument("--mt-run", type=Path, default=None, help="Multi-task run dir, study dir, or best_model dir.")
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cosmos-run", type=Path, default=None, help="Optional Cosmos flow run for a reference row.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    compare(
        gbm_run=args.gbm_run,
        mt_run=args.mt_run,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        cosmos_run=args.cosmos_run,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
