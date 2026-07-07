"""Standalone gradient-boosted baseline for composite MACE risk prediction.

This is a *baseline* for the per-patient composite-event risk task
(MACE OR Nephropathy OR Retinopathy, the same `mace_ever` label the flow model
generates). It deliberately conditions on the exact same patient features the
flow model uses, so its discrimination/calibration are an apples-to-apples
reference point for the joint multi-task network in
``train_flow_matching_multitask.py``.

Design notes
------------
* Data loading/preprocessing is reused from ``train_flow_matching`` (imported,
  never modified) so the cohort, filters, and feature engineering match.
* The estimator prefers XGBoost (``scale_pos_weight`` for imbalance) and falls
  back to scikit-learn's ``HistGradientBoostingClassifier`` (``class_weight``)
  when XGBoost is unavailable. Both natively handle NaNs, so continuous features
  are passed through *un-imputed* (better than mean-filling for trees).
* Imbalance is handled at the loss level (class weighting), never by resampling.
  We train an *unweighted honest baseline* and a *balanced* variant and report
  both -- weighting usually moves the operating point, not AUROC/AUPRC, while
  costing calibration, so seeing both side by side is the point.
* Evaluation is imbalance-aware: AUROC, AUPRC (with prevalence baseline), Brier,
  isotonic recalibration, threshold tuning (Youden + fixed-specificity), and a
  reliability curve. Accuracy and 0.5-threshold metrics are intentionally
  de-emphasized.

Run (local smoke test from the fake CSV)::

    python gbm_mace_baseline.py --csv fake_data/fake_mbs_cohort.csv

Run (standalone against Cosmos via the imported pyodbc path)::

    python gbm_mace_baseline.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import train_flow_matching as fm


DEFAULT_OUTPUT_DIR = fm.REPO_ROOT / "runs" / "gbm_mace_baseline"
MACE_LABEL_NAME = "mace_ever"

# Extra comorbidity features pulled straight from ``dataset.frame`` and handed to
# the GBM ONLY. They are deliberately NOT in ``fm.PATIENT_FEATURES`` -- that shared
# list feeds the flow + multi-task models too, and these belong to the risk model.
# Trees route missing values natively, so they are passed through un-imputed (NaNs
# preserved). See MACE_MODELING_DECISIONS.md.
#
# NOTE: ``PMH_dyslipidemia`` is a requested risk feature but is intentionally absent
# here -- it lives in ``fm.PATIENT_FEATURES`` (shared with the flow), so it already
# reaches the GBM via ``patient_features_raw``; listing it here too would duplicate
# the column. ``osa`` likewise arrives through the shared vector.
GBM_EXTRA_FRAME_FEATURES = [
    "PMH_DM2",
    "PMH_hypertension",
    "PMH_MI",
    "PMH_stroke",
    "PMH_AFib",
    "PMH_VTE",
]


def report_saved(path: Path, description: str = "") -> Path:
    """Announce each saved artifact (with full path) so it is easy to locate in
    the terminal when running on the Cosmos VM."""
    tag = f" {description}" if description else ""
    print(f"  [saved]{tag} -> {path}", flush=True)
    return path


@dataclass
class GBMConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    seed: int = 0
    split_seed: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    # "surgery" reproduces the Cosmos flow split (fm.make_stratified_splits) so test
    # patients line up one-for-one; "outcome" stratifies jointly by surgery and MACE.
    split_strategy: str = "surgery"
    # HistGradientBoosting hyperparameters (used when XGBoost is unavailable).
    learning_rate: float = 0.05
    max_iter: int = 400
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    early_stopping: bool = True
    n_iter_no_change: int = 25
    validation_fraction: float = 0.1
    # XGBoost hyperparameters (used when XGBoost imports successfully).
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 4
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    # Evaluation.
    target_specificity: float = 0.90
    recalibrate: bool = True
    permutation_importance_repeats: int = 10


# --------------------------------------------------------------------------- #
# Feature assembly + splitting
# --------------------------------------------------------------------------- #
def frame_feature(dataset: fm.FlowDataset, canonical: str) -> np.ndarray | None:
    """Pull a numeric column from ``dataset.frame`` by its canonical name.

    Tolerant of Cosmos casing / join suffixes (``.y``, ``_mbs`` ...): reuses the
    same normalized-name matching ``fm.canonicalize_columns`` uses, so a column
    that survives the SQL export under a slightly different spelling is still
    found. ``dataset.frame`` is row-aligned with ``dataset.x`` /
    ``patient_features_raw`` (all built from the same post-filter DataFrame), so
    the returned vector lines up with the label and the rest of the matrix.
    Returns a float64 array with NaNs preserved (trees handle them), or ``None``
    when the column is absent.
    """
    matched = fm.find_compatible_column(list(dataset.frame.columns), canonical)
    if matched is None:
        return None
    return fm.numeric(dataset.frame[matched]).to_numpy(dtype=np.float64)


def assemble_features(dataset: fm.FlowDataset) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build the model matrix for the composite-MACE risk GBM.

    Features: the shared ``fm.PATIENT_FEATURES`` (which now include ``osa`` and
    ``dyslipidemia`` alongside the original demographics/labs) + surgery type
    (sleeve/rnygb) + the extra comorbidity flags in ``GBM_EXTRA_FRAME_FEATURES``
    (``PMH_DM2``, ``PMH_hypertension``, ``PMH_MI``, ``PMH_stroke``, ``PMH_AFib``,
    ``PMH_VTE``) pulled from ``dataset.frame``. The extras are GBM-only -- they are
    pointedly NOT in ``fm.PATIENT_FEATURES`` (which also feeds the flow/multi-task
    models); ``dyslipidemia`` reaches the GBM through the shared vector instead.
    Continuous/binary columns keep their NaNs; the tree learners route missing
    values natively. Label: the composite ``mace_ever`` indicator (MACE OR
    Nephropathy OR Retinopathy).
    """
    patient = dataset.patient_features_raw.astype(np.float64)
    surgery = dataset.surgery_idx.astype(np.float64).reshape(-1, 1)
    columns = [patient, surgery]
    feature_names = list(dataset.patient_feature_names) + ["surgery_idx"]
    for canonical in GBM_EXTRA_FRAME_FEATURES:
        values = frame_feature(dataset, canonical)
        if values is None:
            warnings.warn(
                f"GBM extra feature {canonical!r} not found in dataset.frame; skipping it.",
                stacklevel=2,
            )
            continue
        columns.append(values.reshape(-1, 1))
        feature_names.append(canonical)
    x = np.hstack(columns)
    mace_dim = fm.TARGET_NAMES.index(MACE_LABEL_NAME)
    y = dataset.x[:, mace_dim].astype(np.int64)
    return x, feature_names, y


def make_splits(dataset: fm.FlowDataset, cfg: GBMConfig) -> dict[str, np.ndarray]:
    """Dispatch on cfg.split_strategy.

    "surgery" delegates to fm.make_stratified_splits -- the exact split used by
    train_flow_matching.py / tune_flow_matching_optuna.py -- so with the same
    split_seed and fractions the baseline shares its test patients with the
    Cosmos flow model. "outcome" stratifies jointly by surgery and the MACE label.
    """
    if cfg.split_strategy == "surgery":
        return fm.make_stratified_splits(
            dataset,
            fm.TrainConfig(
                split_seed=cfg.split_seed,
                train_frac=cfg.train_frac,
                val_frac=cfg.val_frac,
                test_frac=cfg.test_frac,
            ),
        )
    if cfg.split_strategy != "outcome":
        raise ValueError(f"Unknown split_strategy: {cfg.split_strategy!r} (expected 'surgery' or 'outcome')")
    y = dataset.x[:, fm.TARGET_NAMES.index(MACE_LABEL_NAME)].astype(np.int64)
    return stratified_splits_by_outcome(dataset.surgery_type, y, cfg)


def stratified_splits_by_outcome(
    surgery_type: np.ndarray, y: np.ndarray, cfg: GBMConfig
) -> dict[str, np.ndarray]:
    """Split stratified jointly by surgery type and the (rare) MACE outcome.

    Stratifying on the outcome -- which ``train_flow_matching`` does not do --
    keeps the positive rate stable across train/val/test, which matters for a
    low-prevalence target.
    """
    if not np.isclose(cfg.train_frac + cfg.val_frac + cfg.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    rng = np.random.default_rng(cfg.split_seed)
    train_parts, val_parts, test_parts = [], [], []
    surgeries = sorted(set(surgery_type.tolist()))
    for surgery in surgeries:
        for label in (0, 1):
            idx = np.where((surgery_type == surgery) & (y == label))[0]
            if idx.size == 0:
                continue
            rng.shuffle(idx)
            n_train = int(np.floor(idx.size * cfg.train_frac))
            n_val = int(np.floor(idx.size * cfg.val_frac))
            train_parts.append(idx[:n_train])
            val_parts.append(idx[n_train : n_train + n_val])
            test_parts.append(idx[n_train + n_val :])
    splits = {
        "train": np.concatenate(train_parts).astype(np.int64),
        "val": np.concatenate(val_parts).astype(np.int64),
        "test": np.concatenate(test_parts).astype(np.int64),
    }
    for key in splits:
        rng.shuffle(splits[key])
    return splits


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


def make_estimator(cfg: GBMConfig, balanced: bool, n_pos: int, n_neg: int):
    """Return an unfitted estimator. Prefer XGBoost, else HistGradientBoosting."""
    if xgboost_available():
        import xgboost as xgb

        scale_pos_weight = (n_neg / max(n_pos, 1)) if balanced else 1.0
        return (
            "xgboost",
            xgb.XGBClassifier(
                n_estimators=cfg.xgb_n_estimators,
                max_depth=cfg.xgb_max_depth,
                learning_rate=cfg.learning_rate,
                subsample=cfg.xgb_subsample,
                colsample_bytree=cfg.xgb_colsample_bytree,
                reg_lambda=cfg.l2_regularization,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=cfg.seed,
                n_jobs=-1,
            ),
        )
    return (
        "hist_gradient_boosting",
        HistGradientBoostingClassifier(
            learning_rate=cfg.learning_rate,
            max_iter=cfg.max_iter,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization,
            early_stopping=cfg.early_stopping,
            n_iter_no_change=cfg.n_iter_no_change,
            validation_fraction=cfg.validation_fraction,
            class_weight="balanced" if balanced else None,
            random_state=cfg.seed,
        ),
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def safe(metric_fn, *args) -> float:
    try:
        value = float(metric_fn(*args))
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def discrimination_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    prevalence = float(y_true.mean()) if y_true.size else float("nan")
    return {
        "n": int(y_true.size),
        "n_pos": int(y_true.sum()),
        "prevalence": prevalence,
        "auroc": safe(roc_auc_score, y_true, y_prob),
        "auprc": safe(average_precision_score, y_true, y_prob),
        "auprc_baseline": prevalence,
        "brier": safe(brier_score_loss, y_true, y_prob),
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        return float(thr[int(np.argmax(j))])
    except Exception:
        return 0.5


def threshold_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, target_spec: float) -> float:
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        specificity = 1.0 - fpr
        feasible = specificity >= target_spec
        if not feasible.any():
            return 1.0
        # Among thresholds meeting the specificity floor, take the most sensitive.
        candidate_tpr = np.where(feasible, tpr, -np.inf)
        return float(thr[int(np.argmax(candidate_tpr))])
    except Exception:
        return 0.5


def operating_point(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, label: str) -> dict:
    pred = (y_prob >= threshold).astype(np.int64)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    except Exception:
        tn = fp = fn = tp = 0
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "operating_point": label,
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = bin_idx == b
        if not sel.any():
            continue
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "count": int(sel.sum()),
                "mean_pred": float(y_prob[sel].mean()),
                "frac_pos": float(y_true[sel].mean()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting (best-effort; never fatal)
# --------------------------------------------------------------------------- #
def save_plots(run_dir: Path, variant_probs: dict[str, np.ndarray], y_true: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve as _roc

        prevalence = float(y_true.mean()) if y_true.size else float("nan")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for name, prob in variant_probs.items():
            try:
                fpr, tpr, _ = _roc(y_true, prob)
                axes[0].plot(fpr, tpr, label=name)
            except Exception:
                pass
            try:
                prec, rec, _ = precision_recall_curve(y_true, prob)
                axes[1].plot(rec, prec, label=name)
            except Exception:
                pass
            table = reliability_table(y_true, prob)
            if not table.empty:
                axes[2].plot(table["mean_pred"], table["frac_pos"], marker="o", label=name)
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
        axes[0].legend()
        axes[1].axhline(prevalence, color="k", ls="--", alpha=0.4, label="prevalence")
        axes[1].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
        axes[1].legend()
        axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[2].set(title="Reliability (test)", xlabel="Mean predicted", ylabel="Observed frequency")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(run_dir / "evaluation_curves.png", dpi=120)
        plt.close(fig)
        report_saved(run_dir / "evaluation_curves.png", "ROC/PR/reliability curves")
    except Exception as exc:  # pragma: no cover - plotting is optional
        warnings.warn(f"Skipped plotting: {exc}", stacklevel=2)


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def fit_predict_variant(
    cfg: GBMConfig,
    balanced: bool,
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
) -> dict:
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    n_pos = int(y[train_idx].sum())
    n_neg = int((y[train_idx] == 0).sum())
    backend, estimator = make_estimator(cfg, balanced, n_pos=n_pos, n_neg=n_neg)
    estimator.fit(x[train_idx], y[train_idx])

    def proba(idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.zeros(0, dtype=np.float64)
        return estimator.predict_proba(x[idx])[:, 1]

    prob = {"train": proba(train_idx), "val": proba(val_idx), "test": proba(test_idx)}

    # Isotonic recalibration fit on validation, applied to the test split.
    test_prob_cal = prob["test"]
    calibrated = False
    if cfg.recalibrate and val_idx.size >= 10 and len(np.unique(y[val_idx])) == 2:
        try:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(prob["val"], y[val_idx])
            test_prob_cal = iso.transform(prob["test"])
            calibrated = True
        except Exception as exc:
            warnings.warn(f"Isotonic recalibration skipped: {exc}", stacklevel=2)

    metric_rows = []
    for split_name in ("train", "val", "test"):
        row = {"variant": "balanced" if balanced else "unweighted", "backend": backend, "split": split_name}
        row.update(discrimination_metrics(y[splits[split_name]], prob[split_name]))
        metric_rows.append(row)
    if calibrated:
        cal_row = {"variant": "balanced" if balanced else "unweighted", "backend": backend, "split": "test_calibrated"}
        cal_row.update(discrimination_metrics(y[test_idx], test_prob_cal))
        metric_rows.append(cal_row)

    # Thresholds chosen on validation, reported on the test split.
    thr_youden = youden_threshold(y[val_idx], prob["val"]) if val_idx.size else 0.5
    thr_spec = (
        threshold_at_specificity(y[val_idx], prob["val"], cfg.target_specificity)
        if val_idx.size
        else 0.5
    )
    operating_points = [
        operating_point(y[test_idx], prob["test"], 0.5, "default_0.5"),
        operating_point(y[test_idx], prob["test"], thr_youden, "youden_val"),
        operating_point(y[test_idx], prob["test"], thr_spec, f"spec>={cfg.target_specificity:g}_val"),
    ]
    for op in operating_points:
        op["variant"] = "balanced" if balanced else "unweighted"

    return {
        "backend": backend,
        "estimator": estimator,
        "metric_rows": metric_rows,
        "operating_points": operating_points,
        "test_prob": prob["test"],
        "test_prob_calibrated": test_prob_cal,
        "calibrated": calibrated,
    }


def run(dataset: fm.FlowDataset, cfg: GBMConfig) -> dict:
    x, feature_names, y = assemble_features(dataset)
    splits = make_splits(dataset, cfg)
    run_dir = make_run_dir(cfg.output_dir)

    overall_prevalence = float(y.mean())
    print(
        f"Patients: {len(y)} "
        f"(train={splits['train'].size}, val={splits['val'].size}, test={splits['test'].size})"
    )
    print(
        f"Composite MACE prevalence: overall={overall_prevalence:.4f}  "
        f"train={float(y[splits['train']].mean()):.4f}  "
        f"val={float(y[splits['val']].mean()):.4f}  "
        f"test={float(y[splits['test']].mean()):.4f}"
    )

    all_metric_rows: list[dict] = []
    all_operating_points: list[dict] = []
    variant_test_probs: dict[str, np.ndarray] = {}
    importances_frames: list[pd.DataFrame] = []
    backend_used = None

    for balanced in (False, True):
        result = fit_predict_variant(cfg, balanced, x, y, splits)
        backend_used = result["backend"]
        all_metric_rows.extend(result["metric_rows"])
        all_operating_points.extend(result["operating_points"])
        variant_name = "balanced" if balanced else "unweighted"
        variant_test_probs[variant_name] = result["test_prob"]

        # Permutation importance on the test split (model-agnostic).
        if splits["test"].size >= 10 and cfg.permutation_importance_repeats > 0:
            try:
                pi = permutation_importance(
                    result["estimator"],
                    x[splits["test"]],
                    y[splits["test"]],
                    scoring="average_precision",
                    n_repeats=cfg.permutation_importance_repeats,
                    random_state=cfg.seed,
                )
                importances_frames.append(
                    pd.DataFrame(
                        {
                            "variant": variant_name,
                            "feature": feature_names,
                            "importance_mean": pi.importances_mean,
                            "importance_std": pi.importances_std,
                        }
                    )
                )
            except Exception as exc:
                warnings.warn(f"Permutation importance skipped ({variant_name}): {exc}", stacklevel=2)

    metrics_df = pd.DataFrame(all_metric_rows)
    operating_df = pd.DataFrame(all_operating_points)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    report_saved(run_dir / "metrics.csv", "discrimination/calibration metrics")
    operating_df.to_csv(run_dir / "operating_points.csv", index=False)
    report_saved(run_dir / "operating_points.csv", "operating points")
    if importances_frames:
        pd.concat(importances_frames, ignore_index=True).to_csv(run_dir / "feature_importances.csv", index=False)
        report_saved(run_dir / "feature_importances.csv", "permutation importances")

    # Test-set predictions for both variants.
    test_idx = splits["test"]
    predictions = pd.DataFrame({"subject_id": dataset.subject_ids[test_idx], "y_true": y[test_idx]})
    for variant_name, prob in variant_test_probs.items():
        predictions[f"prob_{variant_name}"] = prob
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    report_saved(run_dir / "test_predictions.csv", "test predictions")

    save_plots(run_dir, variant_test_probs, y[test_idx])

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(cfg),
                "backend": backend_used,
                "feature_names": feature_names,
                "label": MACE_LABEL_NAME,
                "overall_prevalence": overall_prevalence,
            },
            f,
            indent=2,
        )
    report_saved(run_dir / "config.json", "run config")

    print(f"\nDiscrimination/calibration (lower Brier better; AUPRC vs baseline={overall_prevalence:.3f}):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(metrics_df.to_string(index=False))
    print(f"\nSaved baseline artifacts to {run_dir}")
    return {"run_dir": run_dir, "metrics": metrics_df, "operating_points": operating_df}


def make_run_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def run_from_csv(csv_path: str | Path, cfg: GBMConfig | None = None) -> dict:
    cfg = cfg or GBMConfig()
    return run(fm.load_dataset_from_csv(csv_path), cfg)


def run_from_database(cfg: GBMConfig | None = None) -> dict:
    cfg = cfg or GBMConfig()
    return run(fm.load_dataset_from_database(), cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--target-specificity", type=float, default=0.90)
    parser.add_argument("--split-strategy", type=str, default="surgery", choices=["surgery", "outcome"])
    parser.add_argument("--no-recalibrate", action="store_true")
    args = parser.parse_args()

    cfg = GBMConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        split_seed=args.split_seed,
        target_specificity=args.target_specificity,
        split_strategy=args.split_strategy,
        recalibrate=not args.no_recalibrate,
    )
    try:
        if args.csv_path:
            run_from_csv(args.csv_path, cfg)
        else:
            run_from_database(cfg)
    except RuntimeError as exc:
        print(f"ERROR: {exc}\n\nPass --csv <path> to run from a saved CSV export.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
