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

# Informative-missingness rule: a feature column missing in MORE than this fraction
# of patients gets a companion 0/1 ``<name>_ismissing`` column (the missing fraction
# itself is computed from the data at hand in ``append_missingness_indicators``).
# eGFR is ~10.3% missing on the real cohort, so ``eGFRatEvent_ismissing`` fires there;
# the fully-populated 52-row fake CSV triggers none.
MISSINGNESS_INDICATOR_THRESHOLD = 0.05

# Extra comorbidity + baseline-risk features pulled straight from ``dataset.frame`` and
# handed to the GBM ONLY. They are deliberately NOT in ``fm.PATIENT_FEATURES`` -- that
# shared list feeds the flow + multi-task models too, and these belong to the risk model.
# Trees route missing values natively, so they are passed through un-imputed (NaNs
# preserved). See MACE_MODELING_DECISIONS.md.
#
# NOTE: ``dyslipidemia`` (PMH_dyslipidemia), ``osa`` (PMH_OSA) and ``insulin_status``
# (InsulinStatus) are requested risk features but are intentionally ABSENT here -- they
# already live in ``fm.PATIENT_FEATURES`` (shared with the flow), so they reach the GBM
# via ``patient_features_raw``; listing them here too would duplicate the column.
#
# LEAKAGE VERDICT (W2 numeric tier -- every column below is a pre-operative / baseline
# value, verified against ``table structure.txt`` -> dbo.MBSCohort and MACE_MODELING_DECISIONS.md):
#   * ``BiguanideStatus`` / ``SGLT2Status`` are BASELINE diabetes-medication flags. In the
#     MBSCohort schema they sit interleaved with the PMH_* baseline comorbidity flags
#     (InsulinStatus, BiguanideStatus, SGLT2Status -- between PMH_stroke and
#     PMH_dialysis_transplant) and carry NO ``PostOp``/``PostEvent``/``Interval`` marking,
#     in explicit contrast to the one post-op drug column ``PostOpGLP1``. The decisions log
#     recommends exactly these two as safe next features and quarantines only ``PostOp*`` /
#     GLP1 fields as leakage. (``InsulinStatus`` is the same baseline family but is excluded
#     above for DUPLICATION, not leakage.)
#   * ``eGFRatEvent`` is baseline renal function derived from CreatinineAtEvent + age + sex,
#     hence collinear with ``creatinine_at_surgery`` already in the model -- fine for trees,
#     but do NOT over-read its split importance.
#   * ``Svi*`` are CDC Social-Vulnerability-Index percentiles for the patient's baseline
#     residence (social determinants). Race stays OUT of the model here (fairness audit only).
GBM_EXTRA_FRAME_FEATURES = [
    "PMH_DM2",
    "PMH_hypertension",
    "PMH_MI",
    "PMH_stroke",
    "PMH_AFib",
    "PMH_VTE",
    # --- W2 numeric tier (all baseline / pre-op, NaN-native for trees) ---
    "eGFRatEvent",
    "BiguanideStatus",
    "SGLT2Status",
    "SviOverall",
    "SviHousehold",
    "SviTransportation",
    "SviMinority",
    "SviSES",
]

# Curated roster of leakage-safe *native-categorical* features for the GBM (named and
# ready to enable). Unlike GBM_EXTRA_FRAME_FEATURES these are NOT numeric-coerced: they
# travel through a separate pandas ``category``-dtype path so the tree backends split on
# the raw string levels and route missing cells natively (no one-hot, no imputation).
# LEAKAGE VERDICT (both are baseline / pre-operative administrative fields, verified
# against ``table structure.txt`` -> dbo.MBSCohort; neither carries any
# ``PostOp``/``PostEvent``/``Interval`` marking):
#   * ``RUCA``          -- Rural-Urban Commuting Area code for the patient's baseline
#                          residence (access-to-care / social-determinant proxy; low
#                          cardinality).
#   * ``CoverageClass`` -- insurance / payer class at baseline (Medicare/Commercial/...).
# Race/ethnicity is deliberately EXCLUDED here (fairness audit only -- never a model
# input). The following are left OFF (uncomment to trial; each has a caveat):
#   # "PreferredLanguage",  # baseline, but a coarse culture/access proxy -- fairness-validate first
#   # "StateOrProvince",    # HIGH-CARDINALITY site/geography proxy -- risks leaking site identity
#   # "GLP1Name",           # near-empty under the PriorGLP1==0 filter (mostly NaN) -- low signal
GBM_CANDIDATE_CATEGORICAL_FEATURES = [
    "RUCA",
    "CoverageClass",
]

# ACTIVE categorical set handed to the GBM. EMPTY by default -> ``assemble_features``
# returns the exact float64 ndarray it always has and ``make_estimator`` builds the
# estimator exactly as before, so the five external importers (evaluate_twin.py,
# bmi_threshold_probability.py, compare_mace_models.py, train_twin_pipeline.py,
# evaluate_gbm_mace_baseline.py) are byte-for-byte unaffected. Set this to
# ``GBM_CANDIDATE_CATEGORICAL_FEATURES`` (or any subset) to switch on the DataFrame /
# native-categorical path. Kept as a separate toggle from the curated roster above so the
# shipped default stays OFF while the vetted names stay documented in one place.
GBM_CATEGORICAL_FRAME_FEATURES: list[str] = []


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
    # patients line up one-for-one; "temporal" is the out-of-time fold
    # (fm.make_temporal_splits: earliest surgeries -> train, latest -> test);
    # "outcome" stratifies jointly by surgery and MACE.
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


def frame_categorical(dataset: fm.FlowDataset, canonical: str) -> np.ndarray | None:
    """Pull a native-*categorical* column from ``dataset.frame`` by its canonical name.

    Companion to ``frame_feature`` for the ``GBM_CATEGORICAL_FRAME_FEATURES`` roster:
    the same tolerant normalized-name matching, but the raw string levels are PRESERVED
    (no ``fm.numeric`` coercion). Returns a positional object ndarray of the levels with
    ``np.nan`` for missing cells -- the caller wraps it in a ``category``-dtype column, so
    missing stays a non-level that the tree backends route natively (no imputation).
    ``dataset.frame`` is loaded with ``dtype=str`` and is row-aligned with
    ``patient_features_raw`` (both come from the same post-filter frame in the same order),
    so the returned vector lines up positionally with the numeric block assembled in
    ``assemble_features``. Blank / whitespace-only cells collapse to NaN rather than
    becoming a spurious ``""`` level. Returns ``None`` when the column is absent (the
    caller logs the skip loudly, exactly like ``frame_feature``).
    """
    matched = fm.find_compatible_column(list(dataset.frame.columns), canonical)
    if matched is None:
        return None
    text = dataset.frame[matched].astype("string").str.strip()
    text = text.mask(text.eq(""))
    return text.to_numpy(dtype=object, na_value=np.nan)


def append_missingness_indicators(
    x: np.ndarray,
    feature_names: list[str],
    threshold: float = MISSINGNESS_INDICATOR_THRESHOLD,
) -> tuple[np.ndarray, list[str]]:
    """Append an informative-missingness companion column for every feature that is
    missing in MORE than ``threshold`` of patients.

    The per-column missing fraction is computed from the data at hand (the assembled
    matrix ``x`` itself, over all rows) -- not a hard-coded rate -- and compared to
    ``threshold``. For each flagged column a 0/1 ``<name>_ismissing`` column is appended
    (1 where the original value is NaN). The ORIGINAL columns are left untouched with
    their NaNs intact (trees route them natively); only the appended indicators are
    dense. The feature set has to be identical across train/val/test for the matrix to
    line up, so the include/exclude decision is made ONCE here on the full matrix; it
    uses only feature-marginal missingness and never the label, so it is leakage-free.

    Returns the widened matrix and the extended name list (both grow together, so
    ``x.shape[1] == len(feature_names)`` is preserved for every downstream consumer).
    """
    if x.shape[0] == 0 or x.shape[1] == 0:
        return x, list(feature_names)
    base_names = list(feature_names)
    missing_frac = np.isnan(x).mean(axis=0)
    flagged = np.where(missing_frac > threshold)[0]
    if flagged.size == 0:
        max_missing = float(missing_frac.max()) if missing_frac.size else 0.0
        print(
            f"  [missingness] no feature exceeds the >{threshold:.0%} rule "
            f"(max column missingness = {max_missing:.3f}); no _ismissing columns added.",
            flush=True,
        )
        return x, base_names
    for j in flagged:
        print(
            f"  [missingness] {base_names[j]}: {missing_frac[j]:.1%} missing "
            f"-> added {base_names[j]}_ismissing",
            flush=True,
        )
    indicators = np.isnan(x[:, flagged]).astype(np.float64)
    new_names = [f"{base_names[j]}_ismissing" for j in flagged]
    x = np.hstack([x, indicators])
    return x, base_names + new_names


def assemble_features(
    dataset: fm.FlowDataset,
) -> tuple[np.ndarray | pd.DataFrame, list[str], np.ndarray]:
    """Build the model matrix for the composite-MACE risk GBM.

    Features: the shared ``fm.PATIENT_FEATURES`` (demographics/labs plus ``osa``,
    ``dyslipidemia`` and ``insulin_status``) + surgery type (sleeve/rnygb) + the
    GBM-only extras in ``GBM_EXTRA_FRAME_FEATURES`` -- the comorbidity flags
    (``PMH_DM2``/``PMH_hypertension``/``PMH_MI``/``PMH_stroke``/``PMH_AFib``/``PMH_VTE``)
    and the W2 numeric tier (``eGFRatEvent``; the baseline diabetes-drug flags
    ``BiguanideStatus``/``SGLT2Status``; the SVI social-determinant percentiles
    ``SviOverall``/``SviHousehold``/``SviTransportation``/``SviMinority``/``SviSES``)
    pulled from ``dataset.frame``. The extras are pointedly NOT in
    ``fm.PATIENT_FEATURES`` (which also feeds the flow/multi-task models);
    ``dyslipidemia``/``osa``/``insulin_status`` reach the GBM through the shared
    vector instead, so they are not re-listed as extras.

    ``append_missingness_indicators`` then appends a 0/1 ``<name>_ismissing`` companion
    for every NUMERIC column missing in >5% of patients (informative missingness).
    Continuous/binary columns keep their NaNs; the tree learners route missing values
    natively. Label: the composite ``mace_ever`` indicator (MACE OR Nephropathy OR
    Retinopathy).

    Return contract (two modes, decided ONLY by whether ``GBM_CATEGORICAL_FRAME_FEATURES``
    is non-empty):
      * EMPTY (default): returns ``(x, feature_names, y)`` with ``x`` the float64
        **ndarray** exactly as before -- same columns, order, dtype and
        ``x.shape[1] == len(feature_names)``. The five external importers depend on this
        (positional 2D indexing / in-place numeric edits), so the default is untouched.
      * NON-EMPTY: returns the same triple but with ``x`` a pandas **DataFrame** -- the
        numeric block stays float64 (NaNs preserved) and each configured categorical is
        appended as a ``category``-dtype column (raw string levels; NaN left native, so no
        ``_ismissing`` companion is added for categoricals). ``df.shape[1] ==
        len(feature_names)`` still holds (one appended name per appended column). ``y`` is
        the int64 label ndarray in BOTH modes.
    """
    patient = dataset.patient_features_raw.astype(np.float64)
    surgery = dataset.surgery_idx.astype(np.float64).reshape(-1, 1)
    columns = [patient, surgery]
    feature_names = list(dataset.patient_feature_names) + ["surgery_idx"]
    for canonical in GBM_EXTRA_FRAME_FEATURES:
        values = frame_feature(dataset, canonical)
        if values is None:
            # Logged loudly (warn + print) so a column absent from a given export is
            # never silently dropped: the tiny fake CSV may lack some, the real cohort
            # has all. A dropped column shrinks the matrix, so make it visible.
            message = f"GBM extra feature {canonical!r} not found in dataset.frame; skipping it."
            warnings.warn(message, stacklevel=2)
            print(f"  [feature-skip] {message}", flush=True)
            continue
        columns.append(values.reshape(-1, 1))
        feature_names.append(canonical)
    x = np.hstack(columns)
    # Informative-missingness companions for any feature (shared or extra) missing in
    # more than MISSINGNESS_INDICATOR_THRESHOLD of patients. Grows x and feature_names
    # together; the NaNs in the original columns are left in place for the trees.
    x, feature_names = append_missingness_indicators(x, feature_names)
    mace_dim = fm.TARGET_NAMES.index(MACE_LABEL_NAME)
    y = dataset.x[:, mace_dim].astype(np.int64)

    if not GBM_CATEGORICAL_FRAME_FEATURES:
        # DEFAULT PATH -- byte-for-byte unchanged: hand back the float64 ndarray exactly as
        # before (same columns/order/dtype). The five external importers rely on this.
        return x, feature_names, y

    # CATEGORICAL PATH -- wrap the numeric block (float64, NaNs preserved) in a DataFrame,
    # then append one native ``category``-dtype column per configured categorical. Numpy
    # arrays carry no index, so ``frame[name] = <object ndarray>`` aligns POSITIONALLY with
    # the numeric rows (both are in dataset.frame order). ``feature_names`` grows in lock-
    # step with the columns, preserving ``df.shape[1] == len(feature_names)``.
    frame = pd.DataFrame(x, columns=feature_names)
    for canonical in GBM_CATEGORICAL_FRAME_FEATURES:
        values = frame_categorical(dataset, canonical)
        if values is None:
            # Same loud absent-column handling as the numeric extras: a dropped column
            # would silently shrink the matrix, so make it visible (the tiny fake CSV may
            # lack some; the real cohort has them).
            message = f"GBM categorical feature {canonical!r} not found in dataset.frame; skipping it."
            warnings.warn(message, stacklevel=2)
            print(f"  [feature-skip] {message}", flush=True)
            continue
        frame[canonical] = pd.Series(values, index=frame.index).astype("category")
        feature_names.append(canonical)
    if frame.shape[1] != len(feature_names):  # invariant the 5 importers rely on
        raise AssertionError(
            f"assemble_features column/name mismatch: df has {frame.shape[1]} columns but "
            f"{len(feature_names)} feature_names"
        )
    return frame, feature_names, y


def make_splits(dataset: fm.FlowDataset, cfg: GBMConfig) -> dict[str, np.ndarray]:
    """Dispatch on cfg.split_strategy.

    "surgery" delegates to fm.make_stratified_splits -- the exact split used by
    train_flow_matching.py / tune_flow_matching_optuna.py -- so with the same
    split_seed and fractions the baseline shares its test patients with the
    Cosmos flow model. "temporal" delegates to fm.make_temporal_splits (earliest
    surgeries -> train, latest -> test); with the same split_seed/fractions it stays
    patient-for-patient aligned with the twin flow and Table 1 (all three call the
    same fm.make_temporal_splits). "outcome" stratifies jointly by surgery and the
    MACE label.
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
    if cfg.split_strategy == "temporal":
        return fm.make_temporal_splits(
            dataset,
            fm.TrainConfig(
                split_seed=cfg.split_seed,
                train_frac=cfg.train_frac,
                val_frac=cfg.val_frac,
                test_frac=cfg.test_frac,
            ),
        )
    if cfg.split_strategy != "outcome":
        raise ValueError(
            f"Unknown split_strategy: {cfg.split_strategy!r} (expected 'surgery', 'temporal', or 'outcome')"
        )
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
    """Return an unfitted estimator. Prefer XGBoost, else HistGradientBoosting.

    When ``GBM_CATEGORICAL_FRAME_FEATURES`` is active the backend is told to split on the
    ``category``-dtype columns ``assemble_features`` emits (xgboost ``enable_categorical``;
    HistGradientBoosting ``categorical_features="from_dtype"``). When it is EMPTY (the
    default) NO such kwarg is passed, so the estimator is constructed with exactly the same
    parameters as before -- the default path is byte-for-byte unchanged. Signature is left
    intact so the external ``make_estimator`` call sites are unaffected.
    """
    categorical_active = bool(GBM_CATEGORICAL_FRAME_FEATURES)
    if xgboost_available():
        import xgboost as xgb

        scale_pos_weight = (n_neg / max(n_pos, 1)) if balanced else 1.0
        xgb_kwargs = dict(
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
        )
        if categorical_active:
            # xgboost >= 2.0 splits natively on ``category`` dtype (the active backend).
            xgb_kwargs["enable_categorical"] = True
        return ("xgboost", xgb.XGBClassifier(**xgb_kwargs))
    hist_kwargs = dict(
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
    )
    if categorical_active:
        # sklearn >= 1.4 auto-detects the ``category``-dtype columns from the DataFrame.
        hist_kwargs["categorical_features"] = "from_dtype"
    return ("hist_gradient_boosting", HistGradientBoostingClassifier(**hist_kwargs))


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
def take_rows(x: np.ndarray | pd.DataFrame, idx: np.ndarray) -> np.ndarray | pd.DataFrame:
    """Positional row selection that works for BOTH feature-matrix modes.

    ndarray (default / categoricals-off): ``x[idx]`` -- byte-for-byte the prior behavior.
    DataFrame (categoricals-on): ``x.iloc[idx]`` -- positional, so the ``category`` dtype,
    the float columns and the column order all survive for xgboost's ``enable_categorical``
    (and sklearn's permutation_importance). Only the DataFrame branch is new; the ndarray
    branch is exactly what every call site did before.
    """
    if isinstance(x, pd.DataFrame):
        return x.iloc[idx]
    return x[idx]


def fit_predict_variant(
    cfg: GBMConfig,
    balanced: bool,
    x: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
) -> dict:
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    n_pos = int(y[train_idx].sum())
    n_neg = int((y[train_idx] == 0).sum())
    backend, estimator = make_estimator(cfg, balanced, n_pos=n_pos, n_neg=n_neg)
    estimator.fit(take_rows(x, train_idx), y[train_idx])

    def proba(idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.zeros(0, dtype=np.float64)
        return estimator.predict_proba(take_rows(x, idx))[:, 1]

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
                    take_rows(x, splits["test"]),
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
    parser.add_argument("--split-strategy", type=str, default="surgery", choices=["surgery", "temporal", "outcome"])
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
