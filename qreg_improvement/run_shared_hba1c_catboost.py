#!/usr/bin/env python3
"""Train, validate, calibrate, freeze, and serve a shared three-arm HbA1c CatBoost model.

The script consumes a completed ``run_metabolic_trajectory_study.py`` run.  It reuses the
verified patient splits, baseline-origin HbA1c rows, follow-up/observation weights, and the
calibrated predictions from the older task-specific roster.  It then:

* fits one treatment- and horizon-conditioned CatBoost multi-quantile model;
* benchmarks it against the roster on the same validation patients;
* calibrates 50/80/90% intervals and a configurable HbA1c threshold probability;
* evaluates factual held-out predictions;
* creates INTERNAL treatment-label scenario projections matched on baseline HbA1c;
* writes a portable research model, CSV scorer, model card, public figure book, and bundles.

The treatment-label scenarios are prognostic model projections, not treatment effects or
individual treatment recommendations.  Patient-level scenarios remain under INTERNAL and are
excluded from returnable bundles.

Typical use on the VM::

    python run_shared_hba1c_catboost.py --from-run RESULTS/metabolic_trajectory_YYYYMMDD_HHMMSS

Use ``--horizons 12`` if the 24-month surgical cells do not pass support/calibration gates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Keep CatBoost/sklearn deterministic and friendly to the production VM.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ARMS = ("sleeve", "rygb", "incretin")
HELD_OUT = ("temporal_test", "geographic_test")
DEVELOPMENT = ("train", "validation", "calibration")
OUTCOME = "hba1c"
DEFAULT_HORIZONS = (3, 6, 12)
DEFAULT_THRESHOLD = 7.0
MODEL_VERSION = "shared-hba1c-catboost-1.0.0"
MODEL_CANDIDATE = "shared_hba1c_catboost"
MODEL_ARCHITECTURE = "shared_horizon_catboost_multiquantile"

PAGE_FILES = (
    "00_validation_vs_roster.png",
    "01_heldout_factual_performance.png",
    "02_interval_calibration.png",
    "03_threshold_discrimination.png",
    "04_matching_and_scope.png",
    "05_serving_contract.png",
)
FIGURE_BOOK = "shared_hba1c_figure_book.pdf"
MODEL_BUNDLE = "experimental_shared_hba1c_model_bundle.zip"
RESULTS_BUNDLE = "shared_hba1c_results_bundle.zip"
FIXED_ZIP_TIME = (2001, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class Config:
    source_run: Path
    output_dir: Path
    study_script: Path
    horizons: tuple[int, ...]
    threshold: float = DEFAULT_THRESHOLD
    seed: int = 20260721
    max_training_rows: int = 120_000
    domain_training_rows: int = 200_000
    iterations: int = 500
    patients_per_arm: int = 5
    match_caliper: float = 1.0
    noninferiority_margin: float = 0.05
    coverage_tolerance: float = 0.05
    domain_floor: float = 0.05
    high_missingness_threshold: float = 0.50
    bootstrap_replicates: int = 400
    overwrite: bool = False
    self_test: bool = False

    @property
    def internal(self) -> Path:
        return self.output_dir / "INTERNAL"

    @property
    def export(self) -> Path:
        return self.output_dir / "FIGURES_TO_EXPORT"

    @property
    def model_dir(self) -> Path:
        return self.internal / "shared_hba1c_model"

    @property
    def scenario_dir(self) -> Path:
        return self.internal / "scenario_projections"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_token(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def find_study_script(explicit: str | None, scripts_dir: str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    roots = [Path(scripts_dir).expanduser().resolve()] if scripts_dir else []
    roots += [Path(__file__).resolve().parent, Path.cwd().resolve()]
    names = (
        "run_metabolic_trajectory_study.py",
        "run_metabolic_trajectory_study(1).py",
        "run_metabolic_trajectory_study(2).py",
    )
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError("Could not locate run_metabolic_trajectory_study.py; use --study-script")


def load_study(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("run_metabolic_trajectory_study", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load_runtime_packages()

    # Production worker checkpoints were pickled while their classes lived in __main__.
    main = sys.modules["__main__"]
    for name in (
        "DataBundle", "TaskPartitionedStore", "RowStore", "PredictionStore", "TabularEncoder",
        "RunConfig", "RunContext", "TargetWindow", "CoverageRecord", "CoverageEpisode",
        "PreflightError",
    ):
        value = getattr(module, name, None)
        if value is not None:
            setattr(main, name, value)
            if isinstance(value, type):
                value.__module__ = "__main__"
    return module


def arm_series(frame: Any) -> Any:
    pd = __import__("pandas")
    cohort = frame["cohort"].astype("string").str.lower()
    treatment = frame.get("treatment", pd.Series(pd.NA, index=frame.index, dtype="string"))
    treatment = treatment.astype("string").str.lower()
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    result.loc[cohort.eq("incretin")] = "incretin"
    surgery = cohort.eq("surgery")
    result.loc[surgery & treatment.str.contains(r"rygb|roux|gastric\s*bypass", na=False)] = "rygb"
    result.loc[surgery & treatment.str.contains("sleeve", na=False)] = "sleeve"
    return result


def feature_lists(study: Any, *, include_horizon: bool, include_arm: bool) -> tuple[list[str], list[str]]:
    numeric = [
        "baseline_value", "baseline_cross_outcome", "age_at_index", "diabetes_flag",
        "hypertension", "dyslipidemia", "osa", "insulin", "biguanide", "sglt2",
        "svi", "index_year", *study.OPTIONAL_WIDE_NUMERIC_COVARIATES,
    ]
    if include_horizon:
        numeric.append("target_month")
    categorical = ["sex", "race", "ethnicity", "coverage", "smoking"]
    if include_arm:
        categorical.insert(0, "scenario_arm")
    return numeric, categorical


def fit_encoder(study: Any, frame: Any, numeric: Sequence[str], categorical: Sequence[str]) -> dict[str, Any]:
    numeric = [name for name in numeric if name in frame]
    categorical = [name for name in categorical if name in frame]
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    levels: dict[str, list[str]] = {}
    for name in numeric:
        values = study.pd.to_numeric(frame[name], errors="coerce")
        medians[name] = float(values.median()) if values.notna().any() else 0.0
        scale = float(values.std(ddof=0)) if values.notna().sum() > 1 else 1.0
        scales[name] = scale if math.isfinite(scale) and scale > 1e-8 else 1.0
    for name in categorical:
        values = frame[name].astype("string").fillna("<MISSING>")
        levels[name] = sorted(map(str, values.unique()))
    return {
        "numeric": list(numeric),
        "categorical": list(categorical),
        "medians": medians,
        "scales": scales,
        "levels": levels,
        "informative_missing": sorted(set(numeric).intersection(study.INFORMATIVE_MISSING_FEATURES)),
    }


def transform_encoder(study: Any, encoder: Mapping[str, Any], frame: Any) -> Any:
    columns = []
    informative = set(encoder.get("informative_missing", []))
    for name in encoder["numeric"]:
        values = (study.pd.to_numeric(frame[name], errors="coerce") if name in frame
                  else study.pd.Series(study.np.nan, index=frame.index))
        missing = values.isna().to_numpy(float)
        normalized = (values.to_numpy(float) - encoder["medians"][name]) / encoder["scales"][name]
        if name not in informative:
            normalized = study.np.where(missing.astype(bool), 0.0, normalized)
        columns.extend([normalized, missing])
    for name in encoder["categorical"]:
        values = (frame[name].astype("string").fillna("<MISSING>") if name in frame
                  else study.pd.Series("<MISSING>", index=frame.index))
        known = set(encoder["levels"][name])
        columns.extend(values.eq(level).to_numpy(float) for level in encoder["levels"][name])
        columns.append((~values.isin(known)).to_numpy(float))
    return (study.np.column_stack(columns).astype(study.np.float32) if columns
            else study.np.empty((len(frame), 0), dtype=study.np.float32))


def input_missingness(study: Any, encoder: Mapping[str, Any], frame: Any, threshold: float) -> tuple[Any, Any]:
    features = [name for name in (*encoder["numeric"], *encoder["categorical"])
                if name not in {"scenario_arm", "target_month"}]
    if not features:
        fraction = study.np.zeros(len(frame), dtype=float)
    else:
        columns = []
        for name in features:
            if name not in frame:
                columns.append(study.np.ones(len(frame), dtype=float))
                continue
            values = frame[name]
            missing = values.isna()
            if name in encoder["categorical"]:
                missing = missing | values.astype("string").str.strip().eq("").fillna(True)
            columns.append(missing.to_numpy(float))
        fraction = study.np.column_stack(columns).mean(axis=1)
    return fraction, fraction >= float(threshold)


def stratified_sample(study: Any, frame: Any, maximum: int, seed: int) -> Any:
    if len(frame) <= maximum:
        return frame.reset_index(drop=True)
    groups = list(frame.groupby(["scenario_arm", "target_month"], sort=True, observed=True))
    quota = max(20, maximum // max(len(groups), 1))
    pieces = [group.sample(min(len(group), quota), random_state=seed + index)
              for index, (_key, group) in enumerate(groups)]
    sampled = study.pd.concat(pieces, ignore_index=True)
    if len(sampled) > maximum:
        sampled = sampled.sample(maximum, random_state=seed)
    return sampled.reset_index(drop=True)


def repair_store_root(store: Any, expected: Path) -> Any:
    if expected.exists():
        store.root = expected
    return store


def source_context(study: Any, source_run: Path) -> tuple[Any, Any, dict[str, Any]]:
    manifest = study.read_json(source_run / "run_manifest.json", {}) or {}
    configuration = manifest.get("configuration", {}) or {}
    mode = str(configuration.get("mode", "production"))
    if mode not in {"production", "smoke"}:
        mode = "production"
    cfg = study.RunConfig.create(
        mode,
        str(source_run),
        False,
        incretin_qualifying_months=int(configuration.get(
            "incretin_qualifying_months", study.INCRETIN_QUALIFYING_MONTHS)),
    )
    cfg = replace(cfg, seed=int(configuration.get("seed", study.SEED)))
    context = study.load_run_context(cfg)
    weights = study.require_checkpoint(context, "weights")
    derived = study.require_checkpoint(context, "weights_derived")
    calibration = study.require_checkpoint(context, "calibration")
    repair_store_root(weights["rows"], source_run / "INTERNAL" / "rows" / "weighted")
    repair_store_root(calibration["calibrated"], source_run / "INTERNAL" / "predictions" / "calibrated")
    metadata = {
        "manifest": manifest,
        "source_manifest_sha256": sha256_file(source_run / "run_manifest.json"),
        "source_seed": int(configuration.get("seed", study.SEED)),
    }
    metadata["scale_map"] = derived["scale_map"]
    return weights["rows"], calibration, metadata


def rebuild_source_rows(study: Any, source_run: Path, cfg: Config) -> tuple[Any, Any, dict[str, Any]]:
    """Reconstruct weighted HbA1c prediction rows at cfg.horizons from a completed run's retained data.

    The trajectory study only pre-builds HbA1c targets at 12/24/36/48/60 months, so training on the
    BMI model's 3/6/12-month horizons requires building those rows here. The run's cohorts checkpoint
    retains the raw per-patient BMI/HbA1c measurements and global_splits retains the patient splits and
    baselines, so the study's own audited row builder and cross-fitted weighting are reused verbatim -
    only the target-month set is temporarily narrowed to the requested horizons at origin 0. No source
    re-query is performed, and the source run is not modified (rebuilt stores live under the output dir).
    """
    manifest = study.read_json(source_run / "run_manifest.json", {}) or {}
    configuration = manifest.get("configuration", {}) or {}
    mode = str(configuration.get("mode", "production"))
    if mode not in {"production", "smoke"}:
        mode = "production"
    source_seed = int(configuration.get("seed", study.SEED))
    context_cfg = study.RunConfig.create(
        mode, str(source_run), False,
        incretin_qualifying_months=int(configuration.get(
            "incretin_qualifying_months", study.INCRETIN_QUALIFYING_MONTHS)))
    context_cfg = replace(context_cfg, seed=source_seed)
    context = study.load_run_context(context_cfg)
    cohorts = study.require_checkpoint(context, "global_splits")["cohorts"]
    measurements = study.require_checkpoint(context, "cohorts")["measurements"]
    calibration = study.require_checkpoint(context, "calibration")
    repair_store_root(calibration["calibrated"], source_run / "INTERNAL" / "predictions" / "calibrated")

    # Rebuilt stores are intermediate and are read fully into memory by collect_source_rows, so keep
    # them in a temp dir: writing under the output dir would trip run_pipeline's "not empty" guard, and
    # the source run stays read-only.
    rebuild_root = Path(tempfile.mkdtemp(prefix="hba1c_rebuild_"))
    saved_targets, saved_landmarks = dict(study.TARGET_MONTHS), study.LANDMARK_MONTHS
    try:
        # Build only the requested HbA1c horizons at baseline origin; skip BMI and other origins.
        study.TARGET_MONTHS = {"bmi": (), OUTCOME: tuple(cfg.horizons)}
        study.LANDMARK_MONTHS = (0,)
        unweighted = study.build_prediction_rows(cohorts, measurements, study.RowStore(rebuild_root / "unweighted"))
        weighted, _ = study.estimate_weights_over_store(
            unweighted, study.RowStore(rebuild_root / "weighted"), source_seed)
    finally:
        study.TARGET_MONTHS, study.LANDMARK_MONTHS = saved_targets, saved_landmarks

    all_rows = study.concat_frames([weighted.read(key) for key in weighted.keys()])
    print(f"[hba1c] rebuilt {len(all_rows):,} weighted HbA1c rows at horizons "
          f"{','.join(map(str, cfg.horizons))} from retained measurements (no source re-query)", flush=True)
    metadata = {
        "manifest": manifest,
        "source_manifest_sha256": sha256_file(source_run / "run_manifest.json"),
        "source_seed": source_seed,
        "scale_map": study.development_iqr_scale_map(all_rows),
        "rebuilt_horizons": list(cfg.horizons),
    }
    return weighted, calibration, metadata


def collect_source_rows(study: Any, store: Any, cfg: Config) -> dict[str, Any]:
    buckets: dict[str, list[Any]] = {name: [] for name in (*DEVELOPMENT, *HELD_OUT)}
    profiles: list[Any] = []
    domain: list[Any] = []
    first_horizon = min(cfg.horizons)
    available = set(store.keys())
    for key in (("surgery", OUTCOME, 0), ("incretin", OUTCOME, 0)):
        if key not in available:
            continue
        frame = store.read(key)
        frame = frame.loc[
            study.pd.to_numeric(frame["target_month"], errors="coerce").isin(list(cfg.horizons))
        ].copy()
        frame["scenario_arm"] = arm_series(frame)
        frame = frame.loc[
            frame["scenario_arm"].isin(ARMS)
            & study.pd.to_numeric(frame["baseline_value"], errors="coerce").notna()
        ]
        first = frame.loc[
            study.pd.to_numeric(frame["target_month"], errors="coerce").eq(first_horizon)
        ].drop_duplicates(["patient_id", "cohort"])
        profiles.append(first.loc[first["split"].astype(str).isin(HELD_OUT)].copy())
        domain.append(first.loc[first["split"].astype(str).isin(DEVELOPMENT)].copy())
        observed = frame.loc[
            frame["target_observed"].fillna(False).astype(bool)
            & study.pd.to_numeric(frame["target_value"], errors="coerce").notna()
        ]
        for split in buckets:
            cell = observed.loc[observed["split"].astype(str).eq(split)].copy()
            if not cell.empty:
                buckets[split].append(cell)
    combined = {name: study.concat_frames(parts) for name, parts in buckets.items()}
    combined["train"] = stratified_sample(study, combined["train"], cfg.max_training_rows, cfg.seed)
    combined["heldout"] = study.concat_frames([combined[name] for name in HELD_OUT])
    combined["profiles"] = study.concat_frames(profiles)
    combined["domain"] = study.concat_frames(domain)
    if combined["train"].empty:
        raise RuntimeError("No observed baseline-origin HbA1c training rows were available")
    if combined["validation"].empty or combined["calibration"].empty:
        raise RuntimeError("Validation or calibration HbA1c rows were unavailable")
    return combined


def collect_reference_predictions(study: Any, calibration: Mapping[str, Any],
                                  horizons: Sequence[int], scale_map: Mapping[Any, float]) -> tuple[Any, Any]:
    """Re-select the old roster on the same HbA1c horizons, then return its predictions."""
    store = calibration["calibrated"]
    leaderboards = []
    partitions: dict[tuple[str, str, int], Any] = {}
    for key in (("surgery", OUTCOME, 0), ("incretin", OUTCOME, 0)):
        if key not in set(store.keys()):
            continue
        frame = store.read(key)
        frame = frame.loc[
            study.pd.to_numeric(frame["target_month"], errors="coerce").isin(list(horizons))
        ].copy()
        # A future source run may already contain another experimental shared model; the benchmark
        # remains the original roster.
        frame = frame.loc[~frame["candidate"].astype(str).str.startswith("shared_")]
        partitions[key] = frame
        leaderboards.append(study.candidate_validation_scores(frame, scale_map))
    leaderboard = study.concat_frames(leaderboards)
    selected = study.select_models(leaderboard)
    mapping = {
        (str(row.cohort), str(row.outcome), int(row.origin_month)): str(row.selected_candidate)
        for row in selected.itertuples(index=False)
        if str(row.selected_candidate) not in {"", "not_estimable"}
    }
    pieces = []
    for key, frame in partitions.items():
        candidate = mapping.get(key)
        if not candidate:
            continue
        selected_frame = frame.loc[
            frame["candidate"].astype(str).eq(candidate)
            & frame["split"].astype(str).isin(("validation", *HELD_OUT))
            & frame["target_observed"].fillna(False).astype(bool)
        ].copy()
        selected_frame["scenario_arm"] = arm_series(selected_frame)
        selected_frame["reference_candidate"] = candidate
        pieces.append(selected_frame.loc[selected_frame["scenario_arm"].isin(ARMS)])
    reference = study.concat_frames(pieces)
    if reference.empty:
        raise RuntimeError("Could not reconstruct a horizon-matched HbA1c roster benchmark")
    return reference, selected

def fit_catboost(study: Any, train: Any, cfg: Config) -> dict[str, Any]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise RuntimeError("CatBoost is required for the shared HbA1c model") from error
    numeric, categorical = feature_lists(study, include_horizon=True, include_arm=True)
    encoder = fit_encoder(study, train, numeric, categorical)
    x = study.np.nan_to_num(transform_encoder(study, encoder, train))
    y = study.pd.to_numeric(train["target_value"], errors="coerce").to_numpy(float)
    weights = study.pd.to_numeric(train["analysis_weight"], errors="coerce").fillna(1.0).to_numpy(float)
    model = CatBoostRegressor(
        loss_function="MultiQuantile:alpha=" + ",".join(map(str, study.QUANTILES)),
        iterations=cfg.iterations,
        depth=7,
        learning_rate=0.04,
        l2_leaf_reg=8.0,
        random_seed=cfg.seed,
        random_strength=0.0,
        bootstrap_type="No",
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    model.fit(x, y, sample_weight=weights)
    return {"kind": "catboost", "encoder": encoder, "model": model}


def predict_quantiles(study: Any, model: Mapping[str, Any], frame: Any) -> Any:
    if frame.empty:
        return study.np.empty((0, len(study.QUANTILES)), dtype=float)
    x = study.np.nan_to_num(transform_encoder(study, model["encoder"], frame))
    matrix = study.np.asarray(model["model"].predict(x), dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, len(study.QUANTILES))
    return study.np.clip(study.rearrange_quantiles(matrix), *study.PLAUSIBLE_RANGES[OUTCOME])


def attach_predictions(study: Any, model: Mapping[str, Any], frame: Any) -> Any:
    result = frame.copy().reset_index(drop=True)
    matrix = predict_quantiles(study, model, result)
    result[list(study.QUANTILE_COLUMNS)] = study.stored_quantiles(matrix)
    return result


def fit_conformal(study: Any, calibration: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    intervals = ((0, 6, 0.90), (1, 5, 0.80), (2, 4, 0.50))
    for (arm, month), cell in calibration.groupby(["scenario_arm", "target_month"], sort=True):
        y = cell["target_value"].to_numpy(float)
        matrix = cell[list(study.QUANTILE_COLUMNS)].to_numpy(float)
        for lower, upper, coverage in intervals:
            if len(cell) < study.MIN_CELL_SIZE:
                correction, status = 0.0, "insufficient_calibration_support"
            else:
                scores = study.np.maximum(matrix[:, lower] - y, y - matrix[:, upper])
                correction = max(0.0, study.finite_sample_quantile(scores, coverage))
                status = "calibrated"
            records.append({
                "arm": str(arm), "target_month": int(month), "coverage": coverage,
                "lower_index": lower, "upper_index": upper, "correction": correction,
                "n_calibration": int(len(cell)), "status": status,
            })
    return records


def apply_conformal(study: Any, frame: Any, corrections: Sequence[Mapping[str, Any]]) -> Any:
    result = frame.copy().reset_index(drop=True)
    matrix = result[list(study.QUANTILE_COLUMNS)].to_numpy(float)
    lookup = {(str(item["arm"]), int(item["target_month"]), int(item["lower_index"]), int(item["upper_index"])):
              float(item["correction"]) for item in corrections}
    for position, row in enumerate(result.itertuples(index=False)):
        arm, month = str(row.scenario_arm), int(row.target_month)
        for lower, upper in ((0, 6), (1, 5), (2, 4)):
            value = lookup.get((arm, month, lower, upper), 0.0)
            matrix[position, lower] -= value
            matrix[position, upper] += value
    matrix = study.np.clip(study.rearrange_quantiles(matrix), *study.PLAUSIBLE_RANGES[OUTCOME])
    result[list(study.QUANTILE_COLUMNS)] = study.stored_quantiles(matrix)
    return result


def correction_status(corrections: Sequence[Mapping[str, Any]], arm: str, month: int) -> str:
    statuses = [str(item["status"]) for item in corrections
                if str(item["arm"]) == arm and int(item["target_month"]) == int(month)]
    return "calibrated" if statuses and all(value == "calibrated" for value in statuses) else "insufficient_calibration_support"


def cdf_below(row: Sequence[float], threshold: float, levels: Sequence[float]) -> float:
    import numpy as np
    ladder = np.maximum.accumulate(np.asarray(row, dtype=float))
    probabilities = np.asarray(levels, dtype=float)
    if threshold < ladder[0]:
        width = max(float(ladder[1] - ladder[0]), 1e-6)
        value = probabilities[0] - (ladder[0] - threshold) * probabilities[0] / width
    elif threshold > ladder[-1]:
        width = max(float(ladder[-1] - ladder[-2]), 1e-6)
        value = probabilities[-1] + (threshold - ladder[-1]) * (1.0 - probabilities[-1]) / width
    else:
        value = np.interp(threshold, ladder, probabilities)
    return float(np.clip(value, 0.0, 1.0))


def probabilities_below(study: Any, matrix: Any, threshold: float) -> Any:
    return study.np.asarray([cdf_below(row, threshold, study.QUANTILES) for row in matrix], dtype=float)


def fit_probability_calibrators(study: Any, calibration: Any, cfg: Config) -> tuple[dict[str, Any], Any]:
    from sklearn.isotonic import IsotonicRegression
    calibrators: dict[str, Any] = {}
    records = []
    for (arm, month), cell in calibration.groupby(["scenario_arm", "target_month"], sort=True):
        cell = cell.drop_duplicates("patient_id")
        event = (cell["target_value"].to_numpy(float) < cfg.threshold).astype(int)
        weight = study.pd.to_numeric(cell["analysis_weight"], errors="coerce").fillna(1.0).to_numpy(float)
        raw = probabilities_below(study, cell[list(study.QUANTILE_COLUMNS)].to_numpy(float), cfg.threshold)
        events, nonevents = int(event.sum()), int((1 - event).sum())
        record: dict[str, Any] = {"status": "insufficient_event_or_nonevent_calibration_support"}
        calibrated = raw
        if events >= study.MIN_CELL_SIZE and nonevents >= study.MIN_CELL_SIZE:
            fitted = IsotonicRegression(out_of_bounds="clip").fit(raw, event, sample_weight=weight)
            record = {
                "status": "isotonic_calibrated",
                "x_thresholds": [float(value) for value in fitted.X_thresholds_],
                "y_thresholds": [float(value) for value in fitted.y_thresholds_],
            }
            calibrated = fitted.predict(raw)
        calibrators[f"{arm}|{int(month)}"] = record
        clipped = study.np.clip(calibrated, 1e-6, 1.0 - 1e-6)
        records.append({
            "arm": str(arm), "target_month": int(month), "n": int(len(cell)),
            "events": events, "nonevents": nonevents, "status": record["status"],
            "raw_auroc": study.weighted_auroc(event, raw, weight),
            "calibrated_auroc": study.weighted_auroc(event, calibrated, weight),
            "brier": study.weighted_mean((calibrated - event) ** 2, weight),
            "log_loss": study.weighted_mean(
                -(event * study.np.log(clipped) + (1 - event) * study.np.log1p(-clipped)), weight),
        })
    return calibrators, study.pd.DataFrame(records)


def apply_probability_calibrator(study: Any, calibrators: Mapping[str, Any], raw: Any,
                                 arm: str, month: int) -> tuple[Any, str]:
    values = study.np.asarray(raw, dtype=float)
    item = calibrators.get(f"{arm}|{int(month)}", {})
    if item.get("status") != "isotonic_calibrated":
        return study.np.clip(values, 0.0, 1.0), str(item.get("status", "raw_predictive_cdf_no_calibrator"))
    x = study.np.asarray(item["x_thresholds"], dtype=float)
    y = study.np.asarray(item["y_thresholds"], dtype=float)
    return study.np.clip(study.np.interp(values, x, y, left=y[0], right=y[-1]), 0.0, 1.0), "isotonic_calibrated"


def basic_metrics(study: Any, cell: Any, columns: Sequence[str]) -> dict[str, float]:
    y = cell["target_value"].to_numpy(float)
    matrix = cell[list(columns)].to_numpy(float)
    weight = study.pd.to_numeric(cell["analysis_weight"], errors="coerce").fillna(1.0).to_numpy(float)
    median = matrix[:, 3]
    return {
        "n": int(len(cell)),
        "ess": float(study.effective_sample_size(weight)),
        "crps": float(study.quantile_crps(y, matrix, weight)),
        "rmse": float(study.np.sqrt(study.np.average((y - median) ** 2, weights=weight))),
        "mae": float(study.np.average(study.np.abs(y - median), weights=weight)),
        "bias": float(study.np.average(median - y, weights=weight)),
        "coverage_80": float(study.np.average((y >= matrix[:, 1]) & (y <= matrix[:, 5]), weights=weight)),
        "coverage_90": float(study.np.average((y >= matrix[:, 0]) & (y <= matrix[:, 6]), weights=weight)),
        "width_80": float(study.np.average(matrix[:, 5] - matrix[:, 1], weights=weight)),
    }


def paired_frame(study: Any, shared: Any, reference: Any) -> Any:
    # Pair on the natural per-observation key, not row_id: rebuilt early-horizon rows get fresh row_ids
    # that do not match the roster's, but (patient, cohort, target_month) identifies the same held-out
    # observation in both. Left join keeps every shared held-out row so the model's own factual metrics
    # are reported at every requested horizon, even those (e.g. 3, 6 mo) the roster never forecast.
    keys = ["patient_id", "cohort", "target_month"]

    def normalize(frame: Any) -> Any:
        out = frame.copy()
        out["patient_id"] = out["patient_id"].astype(str)
        out["cohort"] = out["cohort"].astype(str)
        out["target_month"] = study.pd.to_numeric(out["target_month"], errors="coerce").astype("int64")
        return out

    ref_columns = [*keys, "reference_candidate", *study.QUANTILE_COLUMNS]
    renamed = normalize(reference)[ref_columns].rename(
        columns={column: f"reference_{column}" for column in study.QUANTILE_COLUMNS})
    return normalize(shared).merge(renamed, on=keys, how="left", validate="one_to_one")


def bootstrap_crps_difference(study: Any, cell: Any, shared_columns: Sequence[str],
                              reference_columns: Sequence[str], replicates: int, seed: int) -> tuple[float, float]:
    ids = cell["patient_id"].astype(str).to_numpy()
    unique = study.np.asarray(sorted(set(ids)))
    if len(unique) < 2 or replicates < 2:
        return math.nan, math.nan
    positions = {value: study.np.flatnonzero(ids == value) for value in unique}
    y = cell["target_value"].to_numpy(float)
    weight = study.pd.to_numeric(cell["analysis_weight"], errors="coerce").fillna(1.0).to_numpy(float)
    shared = cell[list(shared_columns)].to_numpy(float)
    reference = cell[list(reference_columns)].to_numpy(float)
    rng, values = study.np.random.default_rng(seed), []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = study.np.concatenate([positions[value] for value in sampled])
        values.append(study.quantile_crps(y[index], shared[index], weight[index])
                      - study.quantile_crps(y[index], reference[index], weight[index]))
    return tuple(float(value) for value in study.np.quantile(values, [0.025, 0.975]))


def compare_with_roster(study: Any, shared: Any, reference: Any, cfg: Config,
                        *, gate: bool, scope: str) -> Any:
    shared_columns = list(study.QUANTILE_COLUMNS)
    reference_columns = [f"reference_{column}" for column in study.QUANTILE_COLUMNS]
    paired = paired_frame(study, shared, reference)
    # Horizons the prior roster actually forecast HbA1c for. Rebuilt early horizons (3, 6 mo) have no
    # roster benchmark, so they are scored on their own factual accuracy/calibration but not gated
    # against a roster ratio.
    roster_months = (set(study.pd.to_numeric(reference["target_month"], errors="coerce").dropna().astype(int).tolist())
                     if reference is not None and len(reference) else set())
    rows = []
    for arm in ARMS:
        for month in cfg.horizons:
            cell = paired.loc[
                paired["scenario_arm"].astype(str).eq(arm)
                & study.pd.to_numeric(paired["target_month"], errors="coerce").eq(month)
            ].drop_duplicates("row_id")
            if cell.empty:
                rows.append({"evaluation_scope": scope, "arm": arm, "target_month": month,
                             "n": 0, "roster_available": month in roster_months})
                continue
            shared_metrics = basic_metrics(study, cell, shared_columns)
            priced = cell.loc[study.pd.to_numeric(cell[reference_columns[3]], errors="coerce").notna()]
            roster_available = (month in roster_months) and bool(len(priced))
            if roster_available:
                reference_metrics = basic_metrics(study, priced, reference_columns)
                shared_on_reference = basic_metrics(study, priced, shared_columns)
                low, high = bootstrap_crps_difference(
                    study, priced, shared_columns, reference_columns,
                    min(cfg.bootstrap_replicates, 250 if gate else cfg.bootstrap_replicates),
                    cfg.seed + int(hashlib.sha256(f"{scope}|{arm}|{month}".encode()).hexdigest()[:8], 16),
                )
                ratio = (shared_on_reference["crps"] / reference_metrics["crps"]
                         if reference_metrics["crps"] > 0 else math.nan)
                reference_candidate = str(priced["reference_candidate"].iloc[0])
                reference_crps, reference_rmse = reference_metrics["crps"], reference_metrics["rmse"]
                crps_difference = shared_on_reference["crps"] - reference_metrics["crps"]
                better = bool(shared_on_reference["crps"] < reference_metrics["crps"])
            else:
                low = high = ratio = reference_crps = reference_rmse = crps_difference = math.nan
                reference_candidate, better = "none", False
            row = {
                "evaluation_scope": scope, "arm": arm, "target_month": month,
                **shared_metrics, "roster_available": roster_available,
                "reference_candidate": reference_candidate,
                "reference_crps": reference_crps, "reference_rmse": reference_rmse,
                "crps_ratio_vs_roster": ratio, "crps_difference_vs_roster": crps_difference,
                "difference_ci_low": low, "difference_ci_high": high, "better_than_roster": better,
            }
            if gate:
                # Calibration is a property of the shared model, so it is checked at every horizon;
                # non-inferiority only where a roster exists (vacuously satisfied otherwise).
                row["coverage_ok"] = bool(
                    abs(shared_metrics["coverage_80"] - 0.80) <= cfg.coverage_tolerance
                    and abs(shared_metrics["coverage_90"] - 0.90) <= cfg.coverage_tolerance
                )
                row["noninferior"] = bool(
                    (not roster_available) or (math.isfinite(ratio) and ratio <= 1.0 + cfg.noninferiority_margin))
                row["cell_pass"] = bool(
                    shared_metrics["n"] >= study.MIN_CELL_SIZE
                    and row["coverage_ok"] and row["noninferior"]
                )
            rows.append(row)
    frame = study.pd.DataFrame(rows)
    if gate:
        for column in ("coverage_ok", "noninferior", "cell_pass", "better_than_roster"):
            if column not in frame:
                frame[column] = False
            else:
                frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def heldout_comparisons(study: Any, shared: Any, reference: Any, cfg: Config) -> Any:
    frames = []
    for split in HELD_OUT:
        part = shared.loc[shared["split"].astype(str).eq(split)]
        ref = reference.loc[reference["split"].astype(str).eq(split)]
        frames.append(compare_with_roster(study, part, ref, cfg, gate=False, scope=split))
    frames.append(compare_with_roster(study, shared, reference, cfg, gate=False, scope="pooled_heldout"))
    return study.concat_frames(frames)


def bootstrap_auroc(study: Any, event: Any, score: Any, baseline: Any, weight: Any,
                    patient_ids: Any, replicates: int, seed: int) -> tuple[float, float, float, float]:
    ids = study.np.asarray(patient_ids).astype(str)
    unique = study.np.asarray(sorted(set(ids)))
    positions = {value: study.np.flatnonzero(ids == value) for value in unique}
    rng, estimates, differences = study.np.random.default_rng(seed), [], []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = study.np.concatenate([positions[value] for value in sampled])
        current = study.np.asarray(event)[index]
        if len(study.np.unique(current)) < 2:
            continue
        current_weight = study.np.asarray(weight)[index]
        estimate = study.weighted_auroc(current, study.np.asarray(score)[index], current_weight)
        base = study.weighted_auroc(current, study.np.asarray(baseline)[index], current_weight)
        estimates.append(estimate)
        differences.append(estimate - base)
    if not estimates:
        return math.nan, math.nan, math.nan, math.nan
    return tuple(float(value) for value in (
        *study.np.quantile(estimates, [0.025, 0.975]),
        *study.np.quantile(differences, [0.025, 0.975]),
    ))


def threshold_diagnostics(study: Any, heldout: Any, calibrators: Mapping[str, Any], cfg: Config) -> tuple[Any, Any]:
    from sklearn.metrics import roc_curve
    metric_rows, curve_rows = [], []
    scopes = (*HELD_OUT, "pooled_heldout")
    for scope in scopes:
        scoped = heldout if scope == "pooled_heldout" else heldout.loc[heldout["split"].astype(str).eq(scope)]
        for arm in ARMS:
            for month in cfg.horizons:
                cell = scoped.loc[
                    scoped["scenario_arm"].astype(str).eq(arm)
                    & study.pd.to_numeric(scoped["target_month"], errors="coerce").eq(month)
                ].drop_duplicates("row_id")
                event = (cell["target_value"].to_numpy(float) < cfg.threshold).astype(int) if len(cell) else study.np.array([], int)
                weight = (study.pd.to_numeric(cell["analysis_weight"], errors="coerce").fillna(1.0).to_numpy(float)
                          if len(cell) else study.np.array([], float))
                events, nonevents = int(event.sum()), int(len(event) - event.sum())
                status = "estimable"
                raw = final = baseline = study.np.array([], float)
                raw_auc = auc = baseline_auc = difference = math.nan
                auc_low = auc_high = diff_low = diff_high = brier = log_loss = math.nan
                calibration_status = "unavailable_no_heldout_rows"
                if len(cell):
                    raw = probabilities_below(study, cell[list(study.QUANTILE_COLUMNS)].to_numpy(float), cfg.threshold)
                    final, calibration_status = apply_probability_calibrator(study, calibrators, raw, arm, month)
                    baseline = -study.pd.to_numeric(cell["baseline_value"], errors="coerce").to_numpy(float)
                    brier = study.weighted_mean((final - event) ** 2, weight)
                    clipped = study.np.clip(final, 1e-6, 1.0 - 1e-6)
                    log_loss = study.weighted_mean(
                        -(event * study.np.log(clipped) + (1 - event) * study.np.log1p(-clipped)), weight)
                if events >= study.MIN_CELL_SIZE and nonevents >= study.MIN_CELL_SIZE:
                    raw_auc = study.weighted_auroc(event, raw, weight)
                    auc = study.weighted_auroc(event, final, weight)
                    baseline_auc = study.weighted_auroc(event, baseline, weight)
                    difference = auc - baseline_auc
                    token = int(hashlib.sha256(f"{scope}|{arm}|{month}|threshold".encode()).hexdigest()[:8], 16)
                    auc_low, auc_high, diff_low, diff_high = bootstrap_auroc(
                        study, event, final, baseline, weight, cell["patient_id"],
                        min(cfg.bootstrap_replicates, 400), cfg.seed + token,
                    )
                    if scope == "pooled_heldout":
                        grid = study.np.linspace(0.0, 1.0, 101)
                        for score_type, values in (("shared_model_probability", final),
                                                   ("baseline_hba1c_only", baseline)):
                            fpr, tpr, _ = roc_curve(event, values, sample_weight=weight, drop_intermediate=False)
                            tpr_grid = study.np.maximum.accumulate(study.np.interp(grid, fpr, tpr))
                            curve_rows.extend({
                                "arm": arm, "target_month": month, "score_type": score_type,
                                "fpr": float(x), "tpr": float(y), "evaluation_scope": scope,
                            } for x, y in zip(grid, tpr_grid, strict=True))
                elif not len(cell):
                    status = "unavailable_no_heldout_rows"
                else:
                    status = "unavailable_event_or_nonevent_support"
                metric_rows.append({
                    "evaluation_scope": scope, "arm": arm, "target_month": month,
                    "threshold": cfg.threshold, "n": int(len(cell)), "events": events,
                    "nonevents": nonevents, "effective_sample_size": (
                        study.effective_sample_size(weight) if len(weight) else math.nan),
                    "raw_auroc": raw_auc, "auroc": auc, "auroc_ci_low": auc_low,
                    "auroc_ci_high": auc_high, "baseline_hba1c_auroc": baseline_auc,
                    "auroc_difference_vs_baseline_hba1c": difference,
                    "difference_ci_low": diff_low, "difference_ci_high": diff_high,
                    "brier": brier, "log_loss": log_loss,
                    "probability_calibration_status": calibration_status, "status": status,
                })
    return study.pd.DataFrame(metric_rows), study.pd.DataFrame(curve_rows)


def fit_domain_model(study: Any, rows: Any, cfg: Config) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    frame = rows.drop_duplicates(["patient_id", "cohort"]).copy()
    if len(frame) > cfg.domain_training_rows:
        frame = frame.sample(cfg.domain_training_rows, random_state=cfg.seed).reset_index(drop=True)
    numeric, categorical = feature_lists(study, include_horizon=False, include_arm=False)
    encoder = fit_encoder(study, frame, numeric, categorical)
    x = study.np.nan_to_num(transform_encoder(study, encoder, frame))
    model = LogisticRegression(max_iter=1500, C=0.5, solver="lbfgs", random_state=cfg.seed)
    model.fit(x, frame["scenario_arm"].astype(str))
    return {
        "encoder": encoder, "model": model, "classes": list(map(str, model.classes_)),
        "training_rows": int(len(frame)),
        "interpretation": "descriptive model-domain membership; not a propensity score or causal overlap proof",
    }


def domain_scores(study: Any, domain: Mapping[str, Any], frame: Any) -> Any:
    x = study.np.nan_to_num(transform_encoder(study, domain["encoder"], frame))
    raw = domain["model"].predict_proba(x)
    result = study.np.zeros((len(frame), len(ARMS)), dtype=float)
    for index, name in enumerate(domain["classes"]):
        if name in ARMS:
            result[:, ARMS.index(name)] = raw[:, index]
    return result


def select_profiles(study: Any, profiles: Any, cfg: Config) -> tuple[Any, dict[str, float]]:
    frame = profiles.drop_duplicates(["patient_id", "cohort"]).copy()
    medians = {
        arm: float(study.pd.to_numeric(frame.loc[frame["scenario_arm"].eq(arm), "baseline_value"],
                                       errors="coerce").median())
        for arm in ARMS
    }
    if not all(math.isfinite(value) for value in medians.values()):
        raise RuntimeError(f"Could not compute all held-out baseline HbA1c medians: {medians}")
    selected = []
    for arm in ARMS:
        others = [name for name in ARMS if name != arm]
        target = (medians[others[0]] + medians[others[1]]) / 2.0
        group = frame.loc[frame["scenario_arm"].eq(arm)].copy()
        group["baseline_hba1c"] = study.pd.to_numeric(group["baseline_value"], errors="coerce")
        group = group.loc[group["baseline_hba1c"].notna()]
        group["observed_arm"] = arm
        group["match_target"] = target
        group["match_distance"] = (group["baseline_hba1c"] - target).abs()
        group["within_caliper"] = group["match_distance"].le(cfg.match_caliper)
        group["tie_break"] = [stable_token(str(patient), cfg.seed) for patient in group["patient_id"]]
        group = group.sort_values(["match_distance", "tie_break"], kind="stable").head(cfg.patients_per_arm)
        group["reference_arm_1"], group["reference_arm_2"] = others
        group["profile_label"] = [f"{arm.upper()}-{chr(65 + index)}" for index in range(len(group))]
        selected.append(group)
    return study.concat_frames(selected), medians


def scenario_projections(study: Any, model: Mapping[str, Any], domain: Mapping[str, Any],
                         profiles: Any, corrections: Sequence[Mapping[str, Any]],
                         calibrators: Mapping[str, Any], cell_gates: Mapping[str, bool],
                         cfg: Config) -> tuple[Any, Any, Any]:
    selected, medians = select_profiles(study, profiles, cfg)
    scores = domain_scores(study, domain, selected)
    for index, arm in enumerate(ARMS):
        selected[f"domain_{arm}"] = scores[:, index]
    selected["minimum_domain_score"] = scores.min(axis=1)
    fraction, high = input_missingness(study, model["encoder"], selected, cfg.high_missingness_threshold)
    selected["model_input_missing_fraction"] = fraction
    selected["high_missingness_warning"] = high

    rows = []
    for patient in selected.to_dict("records"):
        for scenario_arm in ARMS:
            template = study.pd.DataFrame([patient] * len(cfg.horizons))
            template["target_month"] = list(cfg.horizons)
            template["scenario_arm"] = scenario_arm
            predicted = attach_predictions(study, model, template)
            predicted = apply_conformal(study, predicted, corrections)
            matrix = predicted[list(study.QUANTILE_COLUMNS)].to_numpy(float)
            raw_probability = probabilities_below(study, matrix, cfg.threshold)
            # Calibrators are horizon-specific, so apply one row at a time.
            final_probability, statuses = [], []
            for raw, month in zip(raw_probability, cfg.horizons, strict=True):
                value, status = apply_probability_calibrator(study, calibrators, [raw], scenario_arm, int(month))
                final_probability.append(float(value[0])); statuses.append(status)
            for index, month in enumerate(cfg.horizons):
                item = {
                    "patient_id": str(patient["patient_id"]),
                    "profile_label": patient["profile_label"],
                    "observed_arm": patient["observed_arm"],
                    "scenario_arm": scenario_arm,
                    "factual_arm": scenario_arm == patient["observed_arm"],
                    "target_month": int(month),
                    "baseline_hba1c": float(patient["baseline_hba1c"]),
                    "baseline_bmi": float(patient.get("baseline_cross_outcome", math.nan)),
                    "match_target": float(patient["match_target"]),
                    "match_distance": float(patient["match_distance"]),
                    "within_caliper": bool(patient["within_caliper"]),
                    "domain_score": float(patient[f"domain_{scenario_arm}"]),
                    "domain_status": ("within_descriptive_domain"
                                      if patient[f"domain_{scenario_arm}"] >= cfg.domain_floor
                                      else "limited_descriptive_domain"),
                    "calibration_status": correction_status(corrections, scenario_arm, int(month)),
                    "raw_probability_below_threshold": float(raw_probability[index]),
                    "probability_below_threshold": float(final_probability[index]),
                    "probability_calibration_status": statuses[index],
                    "threshold": cfg.threshold,
                    "model_input_missing_fraction": float(patient["model_input_missing_fraction"]),
                    "high_missingness_warning": bool(patient["high_missingness_warning"]),
                    "cell_validation_gate_passed": bool(cell_gates.get(f"{scenario_arm}|{int(month)}", False)),
                    "interval_interpretation": (
                        "factual-arm conformal interval" if scenario_arm == patient["observed_arm"]
                        else "receiving-arm-adjusted model interval; transported coverage not established"
                    ),
                }
                item.update({column: float(matrix[index, position])
                             for position, column in enumerate(study.QUANTILE_COLUMNS)})
                rows.append(item)
    scenarios = study.pd.DataFrame(rows)
    matching = study.pd.DataFrame([
        {
            "arm": arm,
            "n": int((profiles["scenario_arm"] == arm).sum()),
            "factual_arm_median_baseline_hba1c": medians[arm],
            "reference_arm_1": [name for name in ARMS if name != arm][0],
            "reference_arm_2": [name for name in ARMS if name != arm][1],
            "matching_target": sum(medians[name] for name in ARMS if name != arm) / 2.0,
        }
        for arm in ARMS
    ])
    return selected, scenarios, matching


def suppress_public(study: Any, frame: Any, count_columns: Sequence[str],
                    structural: Sequence[str]) -> Any:
    if frame.empty:
        return frame.copy()
    saved = frame[[name for name in structural if name in frame]].copy()
    result = study.suppress_small_cells(frame, list(count_columns))
    for name in saved:
        result[name] = saved[name].to_numpy()
    return result


def new_page(study: Any, number: int, title: str, subtitle: str) -> Any:
    figure = study.plt.figure(figsize=(11, 8.5), constrained_layout=False)
    figure.patch.set_facecolor(study.PALETTE["paper"])
    figure.text(0.055, 0.947, f"{number:02d}", fontsize=22, fontweight="bold",
                color=study.PALETTE["blue"], va="top")
    figure.text(0.115, 0.947, title, fontsize=17, fontweight="bold",
                color=study.PALETTE["ink"], va="top")
    figure.text(0.115, 0.915, subtitle, fontsize=9.3, color=study.PALETTE["muted"], va="top")
    figure.lines.append(study.plt.Line2D([0.055, 0.945], [0.893, 0.893],
                                         transform=figure.transFigure, color=study.PALETTE["grid"], lw=1))
    figure.text(0.055, 0.025,
                "Aggregate, disclosure-controlled output | Cells n < 11 suppressed | Prognostic research model only",
                fontsize=7.2, color=study.PALETTE["muted"])
    return figure


def render_public_book(study: Any, cfg: Config, validation: Any, heldout: Any,
                       threshold: Any, curves: Any, matching: Any,
                       gate_summary: Mapping[str, Any], feature_schema: Mapping[str, Any]) -> list[Path]:
    from matplotlib.backends.backend_pdf import PdfPages
    study.configure_figure_style()
    cfg.export.mkdir(parents=True, exist_ok=True)
    pages: list[tuple[str, Any]] = []

    # 00 - validation benchmark
    figure = new_page(study, 0, "Shared HbA1c CatBoost versus the prior roster",
                      "Paired factual validation, baseline-origin forecasts")
    axis = figure.add_axes([0.08, 0.48, 0.84, 0.34])
    display = validation.sort_values(["target_month", "arm"]).copy()
    labels = [f"{row.arm}\n{int(row.target_month)} mo" for row in display.itertuples()]
    ratios = study.pd.to_numeric(display.get("crps_ratio_vs_roster"), errors="coerce").to_numpy(float)
    colors = [study.PALETTE["green"] if bool(value) else study.PALETTE["red"]
              for value in display.get("cell_pass", study.pd.Series(False, index=display.index))]
    axis.bar(range(len(display)), ratios, color=colors)
    axis.axhline(1.0, color=study.PALETTE["ink"], ls="--", lw=1)
    axis.axhline(1.0 + cfg.noninferiority_margin, color=study.PALETTE["red"], ls=":", lw=1)
    axis.set_xticks(range(len(display)), labels, fontsize=7)
    axis.set_ylabel("CRPS ratio: CatBoost / roster")
    axis.set_title("Lower than 1.0 means CatBoost performed better")
    table_axis = figure.add_axes([0.07, 0.09, 0.86, 0.28]); table_axis.axis("off")
    columns = ["arm", "target_month", "n", "crps", "reference_crps", "crps_ratio_vs_roster",
               "coverage_80", "coverage_90", "cell_pass"]
    view = display[[name for name in columns if name in display]].copy()
    labels = ["Arm", "Month", "N", "CRPS", "Roster CRPS", "Ratio", "Cov80", "Cov90", "Pass"]
    artist = table_axis.table(cellText=view.round(3).astype(str).values,
                              colLabels=labels[:len(view.columns)], loc="upper center")
    artist.auto_set_font_size(False); artist.set_fontsize(6.2); artist.scale(1.0, 1.18)
    figure.text(0.07, 0.41,
                f"Global gate: {'PASS' if gate_summary['all_cells_pass'] else 'NOT PASSED'} | "
                f"mean ratio={gate_summary['mean_crps_ratio']:.3f} | "
                f"worst ratio={gate_summary['worst_crps_ratio']:.3f} | "
                f"better cells={gate_summary['better_cells']}/{gate_summary['expected_cells']}",
                fontsize=8.2, fontweight="bold")
    pages.append((PAGE_FILES[0], figure))

    # 01 - held-out table
    figure = new_page(study, 1, "Held-out factual HbA1c performance",
                      "Actual treatment labels only; temporal and geographic tests pooled for display")
    axis = figure.add_axes([0.055, 0.10, 0.89, 0.76]); axis.axis("off")
    pooled = heldout.loc[heldout["evaluation_scope"].eq("pooled_heldout")].copy()
    columns = ["arm", "target_month", "n", "ess", "crps", "reference_crps",
               "crps_ratio_vs_roster", "rmse", "mae", "bias", "coverage_80", "coverage_90"]
    view = pooled[[name for name in columns if name in pooled]].copy()
    labels = ["Arm", "Month", "N", "ESS", "CRPS", "Roster CRPS", "Ratio",
              "RMSE", "MAE", "Bias", "Cov80", "Cov90"]
    artist = axis.table(cellText=view.round(3).astype(str).values,
                        colLabels=labels[:len(view.columns)], loc="upper center")
    artist.auto_set_font_size(False); artist.set_fontsize(6.2); artist.scale(1.0, 1.25)
    figure.text(0.065, 0.075,
                "The roster comparison is paired on row_id, so both models are scored on the same patients. "
                "Switched-arm scenario projections are not evaluated here.", fontsize=7.5)
    pages.append((PAGE_FILES[1], figure))

    # 02 - interval calibration
    figure = new_page(study, 2, "Prediction-interval calibration",
                      "Observed held-out coverage of the conformally adjusted intervals")
    left = figure.add_axes([0.08, 0.18, 0.39, 0.64])
    right = figure.add_axes([0.55, 0.18, 0.39, 0.64])
    colors = {"sleeve": study.PALETTE["blue"], "rygb": study.PALETTE["green"],
              "incretin": study.PALETTE["orange"]}
    for arm in ARMS:
        cell = pooled.loc[pooled["arm"].eq(arm)].sort_values("target_month")
        left.plot(cell["target_month"], cell["coverage_80"], marker="o", color=colors[arm], label=arm)
        right.plot(cell["target_month"], cell["coverage_90"], marker="o", color=colors[arm], label=arm)
    left.axhline(0.80, ls="--", color=study.PALETTE["muted"]); right.axhline(0.90, ls="--", color=study.PALETTE["muted"])
    left.set(title="80% interval", xlabel="Target month", ylabel="Observed coverage", ylim=(0, 1))
    right.set(title="90% interval", xlabel="Target month", ylabel="Observed coverage", ylim=(0, 1))
    left.legend(frameon=False, fontsize=7); right.legend(frameon=False, fontsize=7)
    figure.text(0.07, 0.09,
                "Corrections are fit only on the protected calibration split and separately by observed arm and horizon. "
                "A cell with fewer than 11 calibration outcomes receives no correction and is labelled insufficient.",
                fontsize=7.5)
    pages.append((PAGE_FILES[2], figure))

    # 03 - threshold ROC curves
    figure = new_page(study, 3, f"Held-out discrimination for HbA1c < {cfg.threshold:g}%",
                      "Calibrated predictive-distribution probability versus baseline HbA1c alone")
    months = list(cfg.horizons)
    columns_count = min(3, len(months)); rows_count = int(math.ceil(len(months) / columns_count))
    axes = []
    for index, month in enumerate(months):
        row, column = divmod(index, columns_count)
        width = 0.86 / columns_count; height = 0.60 / rows_count
        axes.append(figure.add_axes([0.07 + column * width, 0.26 + (rows_count - 1 - row) * height,
                                     width - 0.035, height - 0.055]))
        axis = axes[-1]
        plotted = False
        for arm in ARMS:
            metric = threshold.loc[
                threshold["evaluation_scope"].eq("pooled_heldout")
                & threshold["arm"].eq(arm)
                & threshold["target_month"].eq(month)
            ]
            auc = float(metric.iloc[0]["auroc"]) if not metric.empty and study.pd.notna(metric.iloc[0]["auroc"]) else math.nan
            base = float(metric.iloc[0]["baseline_hba1c_auroc"]) if not metric.empty and study.pd.notna(metric.iloc[0]["baseline_hba1c_auroc"]) else math.nan
            for score_type, linestyle, label in (
                ("shared_model_probability", "-", f"{arm} model {auc:.3f}"),
                ("baseline_hba1c_only", "--", f"{arm} baseline {base:.3f}"),
            ):
                curve = curves.loc[
                    curves["arm"].eq(arm) & curves["target_month"].eq(month)
                    & curves["score_type"].eq(score_type)
                ]
                if not curve.empty:
                    axis.plot(curve["fpr"], curve["tpr"], color=colors[arm], ls=linestyle,
                              lw=1.8 if linestyle == "-" else 1.0, label=label)
                    plotted = True
        axis.plot([0, 1], [0, 1], ls=":", color=study.PALETTE["muted"])
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate",
                 ylabel="True-positive rate", title=f"{month} months")
        if plotted:
            axis.legend(frameon=False, fontsize=5.8, loc="lower right")
        else:
            axis.text(0.5, 0.5, "Not estimable", ha="center")
    table_axis = figure.add_axes([0.07, 0.07, 0.86, 0.14]); table_axis.axis("off")
    threshold_view = threshold.loc[threshold["evaluation_scope"].eq("pooled_heldout"),
                                   ["arm", "target_month", "n", "events", "nonevents", "auroc",
                                    "baseline_hba1c_auroc", "auroc_difference_vs_baseline_hba1c", "brier", "status"]]
    labels = ["Arm", "Month", "N", "Events", "Non-events", "Model AUC",
              "Baseline AUC", "Delta AUC", "Brier", "Status"]
    artist = table_axis.table(cellText=threshold_view.round(3).astype(str).values,
                              colLabels=labels, loc="upper center")
    artist.auto_set_font_size(False); artist.set_fontsize(5.4); artist.scale(1.0, 1.12)
    pages.append((PAGE_FILES[3], figure))

    # 04 - matching and scope
    figure = new_page(study, 4, "Scenario-profile matching and interpretation boundary",
                      "Aggregate matching reference; patient-level projections remain INTERNAL")
    axis = figure.add_axes([0.07, 0.49, 0.86, 0.33]); axis.axis("off")
    artist = axis.table(cellText=matching.round(3).astype(str).values,
                        colLabels=list(matching.columns), loc="upper center")
    artist.auto_set_font_size(False); artist.set_fontsize(6.5); artist.scale(1.0, 1.22)
    text = (
        "For each observed arm, profiles are selected solely by proximity of baseline HbA1c to the equal-weight "
        "midpoint of the other two arms' held-out medians. Other baseline characteristics remain as observed.\n\n"
        "The model is then scored under each treatment label while every non-arm input is held fixed. These are "
        "theoretical cross-arm prognostic projections. They do not account for why treatment was chosen, do not "
        "identify a switching effect, and are not an individualized treatment policy.\n\n"
        "The incretin arm remains a future-conditioned sustained-treatment continuer population. Switched-arm "
        "intervals receive the destination arm's calibration correction, but transported coverage is not established."
    )
    figure.text(0.075, 0.41, text, va="top", fontsize=9.2, linespacing=1.45, wrap=True)
    pages.append((PAGE_FILES[4], figure))

    # 05 - serving contract
    figure = new_page(study, 5, "Frozen serving contract",
                      "One portable model, one plain-dict encoder, explicit warnings")
    axis = figure.add_axes([0.07, 0.12, 0.86, 0.70]); axis.axis("off")
    lines = [
        f"Model: CatBoost MultiQuantile, {cfg.iterations} iterations, depth 7, learning rate 0.04, L2=8",
        f"Outcome: absolute HbA1c (%) at months {', '.join(map(str, cfg.horizons))}",
        f"Calibrated endpoint: P(HbA1c < {cfg.threshold:g}%)",
        f"Required serving columns: {', '.join(feature_schema['required'])}",
        f"Global validation gate: {'PASS' if gate_summary['all_cells_pass'] else 'NOT PASSED'}",
        "Per-row outputs: seven quantiles, threshold probability, calibration status, model-domain scores,",
        "input-missingness warning, per-cell validation-gate status, and global gate status.",
        "Portable files: research_model.pkl, catboost_model.cbm, inference.py, feature_schema.json,",
        "environment.json, MODEL_CARD.md, public validation/performance/calibration CSVs.",
        "Intended use: research forecasting and architecture evaluation. Not treatment selection or causal inference.",
    ]
    axis.text(0.02, 0.97, "\n\n".join(lines), va="top", fontsize=10.5, linespacing=1.25)
    pages.append((PAGE_FILES[5], figure))

    pdf_path = cfg.export / FIGURE_BOOK
    with PdfPages(pdf_path, metadata={"Title": "Shared three-arm HbA1c CatBoost study",
                                      "CreationDate": datetime(2001, 1, 1, tzinfo=timezone.utc)}) as writer:
        for name, figure in pages:
            figure.savefig(cfg.export / name, dpi=220, facecolor=figure.get_facecolor())
            writer.savefig(figure, dpi=220, facecolor=figure.get_facecolor())
            study.plt.close(figure)
    return [cfg.export / name for name, _ in pages] + [pdf_path]


def observed_factual_trajectories(study: Any, heldout: Any, selected: Any) -> Any:
    """Observed held-out HbA1c for the selected profiles at each horizon, for the factual-arm overlay.

    Only genuinely observed held-out rows are returned; switched (counterfactual) arms have no
    observed value by construction, so nothing is emitted for them.
    """
    columns = ["profile_label", "patient_id", "target_month", "observed_value"]
    required = {"patient_id", "cohort", "target_month", "target_value"}
    if (selected is None or not len(selected) or heldout is None or not len(heldout)
            or not required.issubset(set(heldout.columns))):
        return study.pd.DataFrame(columns=columns)
    label_by_key = {(str(patient), str(cohort)): str(label) for patient, cohort, label
                    in zip(selected["patient_id"], selected["cohort"], selected["profile_label"])}
    frame = heldout.copy()
    keys = list(zip(frame["patient_id"].astype(str), frame["cohort"].astype(str)))
    frame = frame.loc[[key in label_by_key for key in keys]].copy()
    if frame.empty:
        return study.pd.DataFrame(columns=columns)
    frame["profile_label"] = [label_by_key[key] for key
                              in zip(frame["patient_id"].astype(str), frame["cohort"].astype(str))]
    frame["observed_value"] = study.pd.to_numeric(frame["target_value"], errors="coerce")
    frame["target_month"] = study.pd.to_numeric(frame["target_month"], errors="coerce").astype(int)
    return (frame[columns].dropna(subset=["observed_value"])
            .drop_duplicates(["profile_label", "target_month"])
            .sort_values(["profile_label", "target_month"]).reset_index(drop=True))


def render_internal_scenarios(study: Any, cfg: Config, selected: Any, scenarios: Any,
                              observed: Any = None) -> list[Path]:
    from matplotlib.backends.backend_pdf import PdfPages
    out = cfg.scenario_dir / "FIGURES_TO_EXPORT"
    out.mkdir(parents=True, exist_ok=True)
    study.configure_figure_style()
    pages: list[tuple[str, Any]] = []

    figure = study.plt.figure(figsize=(11, 8.5)); figure.suptitle("Baseline-HbA1c matching audit", fontsize=16, fontweight="bold")
    axis = figure.add_subplot(111); axis.axis("off")
    columns = ["profile_label", "observed_arm", "baseline_hba1c", "match_target",
               "match_distance", "within_caliper", "reference_arm_1", "reference_arm_2"]
    view = selected[columns]
    artist = axis.table(cellText=view.round(3).astype(str).values, colLabels=columns, loc="upper center")
    artist.auto_set_font_size(False); artist.set_fontsize(6.3)
    figure.text(0.05, 0.05, "Profiles were selected before viewing scenario projections; patient IDs are omitted from figures.")
    pages.append(("00_hba1c_matching_audit.png", figure))

    figure = study.plt.figure(figsize=(11, 8.5)); figure.suptitle("Model-domain and missingness audit", fontsize=16, fontweight="bold")
    axis = figure.add_subplot(111); axis.axis("off")
    columns = ["profile_label", "observed_arm", "baseline_hba1c", "domain_sleeve", "domain_rygb",
               "domain_incretin", "minimum_domain_score", "model_input_missing_fraction", "high_missingness_warning"]
    view = selected[columns]
    artist = axis.table(cellText=view.round(3).astype(str).values, colLabels=columns, loc="upper center")
    artist.auto_set_font_size(False); artist.set_fontsize(5.8)
    figure.text(0.05, 0.05, "Domain scores are descriptive membership diagnostics, not propensity scores or causal overlap proof.")
    pages.append(("01_model_domain_and_missingness.png", figure))

    colors = {"sleeve": study.PALETTE["blue"], "rygb": study.PALETTE["green"],
              "incretin": study.PALETTE["orange"]}
    for page_number, observed_arm in enumerate(ARMS, start=2):
        profiles = selected.loc[selected["observed_arm"].eq(observed_arm)].sort_values("profile_label")
        figure, axes = study.plt.subplots(max(len(profiles), 1), 1, figsize=(11, 8.5), squeeze=False)
        figure.suptitle(f"Theoretical HbA1c treatment-label projections: observed {observed_arm} profiles",
                        fontsize=14, fontweight="bold")
        for axis, profile in zip(axes[:, 0], profiles.itertuples()):
            patient = scenarios.loc[scenarios["profile_label"].eq(profile.profile_label)]
            baseline = float(profile.baseline_hba1c)
            for scenario_arm in ARMS:
                rows = patient.loc[patient["scenario_arm"].eq(scenario_arm)].sort_values("target_month")
                x = study.np.r_[0, rows["target_month"].to_numpy(float)]
                y = study.np.r_[baseline, rows["q50"].to_numpy(float)]
                factual = scenario_arm == observed_arm
                axis.plot(x, y, marker="o", ls="-" if factual else "--", lw=2 if factual else 1.2,
                          color=colors[scenario_arm], label=f"{scenario_arm}{' factual' if factual else ' scenario'}")
                low = study.np.r_[baseline, rows["q10"].to_numpy(float)]
                high = study.np.r_[baseline, rows["q90"].to_numpy(float)]
                axis.fill_between(x, low, high, color=colors[scenario_arm], alpha=0.18 if factual else 0.06)
            points = (observed.loc[observed["profile_label"].eq(profile.profile_label)].sort_values("target_month")
                      if observed is not None and len(observed) else None)
            if points is not None and len(points):
                axis.scatter(points["target_month"].to_numpy(float), points["observed_value"].to_numpy(float),
                             marker="X", s=60, color=colors[observed_arm], edgecolor="black", linewidths=0.7,
                             zorder=6, label="observed HbA1c (factual)")
            axis.axhline(cfg.threshold, ls=":", color=study.PALETTE["muted"])
            axis.set_title(f"{profile.profile_label}: baseline {baseline:.2f}%; match target {profile.match_target:.2f}%",
                           loc="left", fontsize=8)
            axis.set_ylabel("HbA1c (%)")
        axes[-1, 0].set_xlabel("Months from observed index")
        axes[0, 0].legend(frameon=False, ncol=4, fontsize=6.6)
        figure.text(0.05, 0.02,
                    "Solid: factual-arm calibrated forecast. Dashed: receiving-arm-adjusted model scenario "
                    "(not an identified treatment effect). Black-edged X: the patient's observed held-out HbA1c "
                    "on the factual arm.", fontsize=7.5)
        pages.append((f"{page_number:02d}_scenario_profiles_{observed_arm}.png", figure))

    pdf = out / "hba1c_scenario_projection_book.pdf"
    with PdfPages(pdf, metadata={"Title": "Internal HbA1c scenario projections"}) as writer:
        for name, figure in pages:
            figure.savefig(out / name, dpi=220, facecolor=figure.get_facecolor())
            writer.savefig(figure, dpi=220, facecolor=figure.get_facecolor())
            study.plt.close(figure)
    return [out / name for name, _ in pages] + [pdf]


def write_inference_script(path: Path) -> None:
    code = r'''#!/usr/bin/env python3
"""Score the frozen shared HbA1c research model from CSV. Not for treatment selection."""
import argparse, pickle, warnings
import numpy as np
import pandas as pd

ARMS=("sleeve","rygb","incretin")
REQUIRED=("baseline_value","scenario_arm","target_month")

def transform(enc,frame):
    cols=[]; informative=set(enc.get("informative_missing",[]))
    for name in enc["numeric"]:
        values=pd.to_numeric(frame[name],errors="coerce") if name in frame else pd.Series(np.nan,index=frame.index)
        missing=values.isna().to_numpy(float)
        normalized=(values.to_numpy(float)-enc["medians"][name])/enc["scales"][name]
        if name not in informative: normalized=np.where(missing.astype(bool),0.0,normalized)
        cols.extend([normalized,missing])
    for name in enc["categorical"]:
        values=frame[name].astype("string").fillna("<MISSING>") if name in frame else pd.Series("<MISSING>",index=frame.index)
        known=set(enc["levels"][name])
        cols.extend(values.eq(level).to_numpy(float) for level in enc["levels"][name])
        cols.append((~values.isin(known)).to_numpy(float))
    return np.column_stack(cols).astype(np.float32) if cols else np.empty((len(frame),0),np.float32)

def missing_fraction(enc,frame):
    features=[name for name in (*enc["numeric"],*enc["categorical"]) if name not in {"scenario_arm","target_month"}]
    if not features:return np.zeros(len(frame))
    cols=[]
    for name in features:
        if name not in frame:cols.append(np.ones(len(frame)));continue
        missing=frame[name].isna()
        if name in enc["categorical"]:missing=missing|frame[name].astype("string").str.strip().eq("").fillna(True)
        cols.append(missing.to_numpy(float))
    return np.column_stack(cols).mean(axis=1)

def predict(bundle,frame):
    x=np.nan_to_num(transform(bundle["model"]["encoder"],frame))
    matrix=np.asarray(bundle["model"]["model"].predict(x),float)
    if matrix.ndim==1:matrix=matrix.reshape(-1,len(bundle["quantiles"]))
    lo,hi=bundle["plausible_range"]
    return np.sort(np.clip(matrix,lo,hi),axis=1)

def correct(bundle,frame,matrix):
    out=matrix.copy(); lookup={(x["arm"],int(x["target_month"]),int(x["lower_index"]),int(x["upper_index"])):float(x["correction"]) for x in bundle.get("corrections",[])}
    for i,row in frame.reset_index(drop=True).iterrows():
        for lower,upper in ((0,6),(1,5),(2,4)):
            value=lookup.get((str(row["scenario_arm"]),int(row["target_month"]),lower,upper),0.0)
            out[i,lower]-=value;out[i,upper]+=value
    lo,hi=bundle["plausible_range"]
    return np.sort(np.clip(out,lo,hi),axis=1)

def cdf(row,threshold,levels):
    row=np.maximum.accumulate(np.asarray(row,float));levels=np.asarray(levels,float)
    if threshold<row[0]:value=levels[0]-(row[0]-threshold)*levels[0]/max(row[1]-row[0],1e-6)
    elif threshold>row[-1]:value=levels[-1]+(threshold-row[-1])*(1-levels[-1])/max(row[-1]-row[-2],1e-6)
    else:value=np.interp(threshold,row,levels)
    return float(np.clip(value,0,1))

def calibrate(bundle,raw,arm,month):
    item=bundle.get("probability_calibrators",{}).get(f"{arm}|{int(month)}",{})
    if item.get("status")!="isotonic_calibrated":return float(raw),item.get("status","raw_predictive_cdf_no_calibrator")
    x=np.asarray(item["x_thresholds"],float);y=np.asarray(item["y_thresholds"],float)
    return float(np.clip(np.interp(raw,x,y,left=y[0],right=y[-1]),0,1)),"isotonic_calibrated"

def domain_scores(bundle,frame):
    domain=bundle["domain_model"]
    raw=domain["model"].predict_proba(np.nan_to_num(transform(domain["encoder"],frame)))
    result=np.zeros((len(frame),len(ARMS)))
    for j,name in enumerate(domain["classes"]):
        if name in ARMS:result[:,ARMS.index(name)]=raw[:,j]
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("model");p.add_argument("input_csv");p.add_argument("output_csv")
    a=p.parse_args();bundle=pickle.load(open(a.model,"rb"));frame=pd.read_csv(a.input_csv)
    missing=[name for name in REQUIRED if name not in frame]
    if missing:raise SystemExit("missing required column(s): "+", ".join(missing))
    if not set(frame["scenario_arm"].astype(str)).issubset(ARMS):raise SystemExit("scenario_arm must be sleeve, rygb, or incretin")
    if not set(map(int,frame["target_month"])).issubset(set(bundle["trained_horizons"])):raise SystemExit("target_month was not trained")
    baseline=pd.to_numeric(frame["baseline_value"],errors="coerce");lo,hi=bundle["plausible_range"]
    if baseline.isna().any() or not baseline.between(lo,hi).all():raise SystemExit(f"baseline_value must be HbA1c (%) within [{lo},{hi}]")
    fraction=missing_fraction(bundle["model"]["encoder"],frame);high=fraction>=float(bundle.get("high_missingness_threshold",.5))
    if high.any():warnings.warn(f"High model-input missingness for {int(high.sum())} row(s).",RuntimeWarning)
    if not bundle.get("global_gate_passed",False):warnings.warn("Global validation gate did not pass; inspect row-level gate flags.",RuntimeWarning)
    matrix=correct(bundle,frame,predict(bundle,frame))
    for j,name in enumerate(bundle["quantile_columns"]):frame[name]=matrix[:,j]
    threshold=float(bundle["threshold"]);raw=np.asarray([cdf(row,threshold,bundle["quantiles"]) for row in matrix])
    calibrated=[calibrate(bundle,value,str(arm),int(month)) for value,arm,month in zip(raw,frame["scenario_arm"],frame["target_month"])]
    frame["hba1c_threshold"]=threshold;frame["raw_probability_below_threshold"]=raw
    frame["probability_below_threshold"]=[value for value,_ in calibrated]
    frame["probability_calibration_status"]=[status for _,status in calibrated]
    scores=domain_scores(bundle,frame)
    for j,arm in enumerate(ARMS):frame[f"model_domain_score_{arm}"]=scores[:,j]
    frame["model_domain_score"]=[scores[i,ARMS.index(str(arm))] for i,arm in enumerate(frame["scenario_arm"])]
    frame["model_domain_status"]=np.where(frame["model_domain_score"]>=float(bundle.get("domain_floor",.05)),"within_descriptive_domain","limited_descriptive_domain")
    frame["input_missing_fraction"]=fraction;frame["high_missingness_warning"]=high
    gates=bundle.get("cell_gate_status",{})
    frame["cell_validation_gate_passed"]=[bool(gates.get(f"{arm}|{int(month)}",False)) for arm,month in zip(frame["scenario_arm"],frame["target_month"])]
    frame["global_validation_gate_passed"]=bool(bundle.get("global_gate_passed",False))
    frame.to_csv(a.output_csv,index=False)
if __name__=="__main__":main()
'''
    path.write_text(code, encoding="utf-8")


def write_model_package(study: Any, cfg: Config, deployment: Mapping[str, Any],
                        validation_public: Any, heldout_public: Any, threshold_public: Any,
                        corrections_public: Any, probability_public: Any) -> tuple[Path, dict[str, Any]]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    encoder = deployment["model"]["encoder"]
    feature_schema = {
        "required": ["baseline_value", "scenario_arm", "target_month"],
        "numeric": encoder["numeric"], "categorical": encoder["categorical"],
        "units": {"baseline_value": "% NGSP HbA1c", "baseline_cross_outcome": "kg/m2",
                  "age_at_index": "years", "target_month": "months"},
        "allowed_arms": list(ARMS), "trained_horizons": list(cfg.horizons),
        "plausible_range": list(study.PLAUSIBLE_RANGES[OUTCOME]),
        "threshold": cfg.threshold,
        "missing_handling": "frozen median/scale plus explicit missing indicators",
        "high_missingness_threshold": cfg.high_missingness_threshold,
    }
    study.atomic_json(cfg.model_dir / "feature_schema.json", feature_schema)
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scikit-learn", "catboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    study.atomic_json(cfg.model_dir / "environment.json", versions)

    portable = {
        key: deployment[key] for key in (
            "version", "outcome", "quantiles", "quantile_columns", "trained_horizons",
            "plausible_range", "threshold", "model", "domain_model", "probability_calibrators",
            "domain_floor", "high_missingness_threshold", "global_gate_passed", "cell_gate_status",
            "claim",
        )
    }
    portable["corrections"] = [
        {name: item[name] for name in ("arm", "target_month", "coverage", "lower_index",
                                      "upper_index", "correction", "status")}
        for item in deployment["corrections"]
    ]
    study.atomic_pickle(cfg.model_dir / "research_model.pkl", portable)
    deployment["model"]["model"].save_model(str(cfg.model_dir / "catboost_model.cbm"))
    write_inference_script(cfg.model_dir / "inference.py")

    gate = "PASS" if deployment["global_gate_passed"] else "NOT PASSED - inspect per-cell flags"
    model_card = f"""# Shared three-arm HbA1c CatBoost research model

Version: `{MODEL_VERSION}`  
Outcome: absolute HbA1c (%)  
Horizons: {', '.join(map(str, cfg.horizons))} months  
Threshold output: P(HbA1c < {cfg.threshold:g}%)  
Global validation gate: **{gate}**

## Intended use

Factual prognostic forecasting and research architecture evaluation. The model returns seven HbA1c
quantiles, calibrated intervals, a calibrated threshold probability when supported, model-domain
scores, input-missingness warnings, and validation-gate flags.

## Not intended for

Individual treatment selection, causal counterfactual claims, treatment recommendations, or digital-
twin use. The incretin cohort is a sustained-treatment continuer population. Changing the arm input
creates a theoretical model scenario and does not identify the effect of switching treatment.

## Serving

Run `python inference.py research_model.pkl input.csv output.csv`. Required columns are
`baseline_value` (baseline HbA1c in %), `scenario_arm`, and `target_month`. Other absent features are
encoded as missing and surfaced through the missingness-warning columns.
"""
    (cfg.model_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    validation_public.to_csv(cfg.model_dir / "validation_vs_roster_public.csv", index=False)
    heldout_public.to_csv(cfg.model_dir / "heldout_performance_public.csv", index=False)
    threshold_public.to_csv(cfg.model_dir / "threshold_performance_public.csv", index=False)
    corrections_public.to_csv(cfg.model_dir / "conformal_corrections_public.csv", index=False)
    probability_public.to_csv(cfg.model_dir / "probability_calibration_public.csv", index=False)

    model_manifest = {
        "version": MODEL_VERSION, "created_utc": utc_now(),
        "outcome": OUTCOME, "trained_horizons": list(cfg.horizons), "threshold": cfg.threshold,
        "global_gate_passed": bool(deployment["global_gate_passed"]),
        "cell_gate_status": dict(deployment["cell_gate_status"]),
        "claim": deployment["claim"], "feature_schema": feature_schema,
    }
    study.atomic_json(cfg.model_dir / "manifest.json", model_manifest)

    names = [
        "research_model.pkl", "catboost_model.cbm", "inference.py", "feature_schema.json",
        "environment.json", "MODEL_CARD.md", "manifest.json", "validation_vs_roster_public.csv",
        "heldout_performance_public.csv", "threshold_performance_public.csv",
        "conformal_corrections_public.csv", "probability_calibration_public.csv",
    ]
    destination = cfg.output_dir / MODEL_BUNDLE
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = cfg.model_dir / name
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return destination, feature_schema


def write_results_bundle(cfg: Config, public_files: Sequence[Path]) -> Path:
    destination = cfg.output_dir / RESULTS_BUNDLE
    members: list[tuple[str, Path]] = []
    for path in public_files:
        if path.is_file() and path.resolve() != destination.resolve():
            members.append((str(path.relative_to(cfg.output_dir)), path))
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(members):
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return destination


def gate_summary(study: Any, validation: Any) -> dict[str, Any]:
    ratios = study.pd.to_numeric(validation["crps_ratio_vs_roster"], errors="coerce")
    finite = ratios[study.np.isfinite(ratios)]
    return {
        "all_cells_pass": bool(len(validation) and validation["cell_pass"].fillna(False).all()),
        "expected_cells": int(len(validation)),
        "passed_cells": int(validation["cell_pass"].fillna(False).sum()),
        "better_cells": int(validation["better_than_roster"].fillna(False).sum()),
        "mean_crps_ratio": float(finite.mean()) if len(finite) else math.inf,
        "worst_crps_ratio": float(finite.max()) if len(finite) else math.inf,
    }


def run_pipeline(study: Any, cfg: Config, data: Mapping[str, Any], reference: Any,
                 source_metadata: Mapping[str, Any]) -> Path:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        if not cfg.overwrite:
            raise RuntimeError(f"Output directory is not empty: {cfg.output_dir}; use --overwrite")
        shutil.rmtree(cfg.output_dir)
    cfg.export.mkdir(parents=True, exist_ok=True)
    cfg.internal.mkdir(parents=True, exist_ok=True)

    print(f"[hba1c] training CatBoost on {len(data['train']):,} rows", flush=True)
    model = fit_catboost(study, data["train"], cfg)
    raw = {name: attach_predictions(study, model, data[name])
           for name in ("validation", "calibration", "heldout")}
    corrections = fit_conformal(study, raw["calibration"])
    calibrated = {name: apply_conformal(study, frame, corrections) for name, frame in raw.items()}

    validation_reference = reference.loc[reference["split"].astype(str).eq("validation")]
    validation = compare_with_roster(
        study, calibrated["validation"], validation_reference, cfg, gate=True, scope="validation")
    summary = gate_summary(study, validation)
    cell_gates = {f"{row.arm}|{int(row.target_month)}": bool(row.cell_pass)
                  for row in validation.itertuples()}
    print(f"[hba1c] validation gate: {'PASS' if summary['all_cells_pass'] else 'NOT PASSED'}; "
          f"mean CRPS ratio={summary['mean_crps_ratio']:.3f}", flush=True)

    calibrators, probability_calibration = fit_probability_calibrators(study, calibrated["calibration"], cfg)
    heldout_reference = reference.loc[reference["split"].astype(str).isin(HELD_OUT)]
    heldout = heldout_comparisons(study, calibrated["heldout"], heldout_reference, cfg)
    threshold, curves = threshold_diagnostics(study, calibrated["heldout"], calibrators, cfg)
    domain = fit_domain_model(study, data["domain"], cfg)
    selected_profiles, scenarios, matching = scenario_projections(
        study, model, domain, data["profiles"], corrections, calibrators, cell_gates, cfg)

    observed_trajectories = observed_factual_trajectories(study, data.get("heldout"), selected_profiles)
    cfg.scenario_dir.mkdir(parents=True, exist_ok=True)
    study.atomic_pickle(cfg.scenario_dir / "patient_scenario_projections.pkl", scenarios)
    scenarios.to_csv(cfg.scenario_dir / "patient_scenario_projections.csv", index=False)
    selected_profiles.to_csv(cfg.scenario_dir / "profile_matching_audit.csv", index=False)
    observed_trajectories.to_csv(cfg.scenario_dir / "observed_factual_trajectories.csv", index=False)
    render_internal_scenarios(study, cfg, selected_profiles, scenarios, observed_trajectories)

    claim = "shared three-arm HbA1c prognostic model; treatment-label scenarios are not identified treatment effects"
    deployment = {
        "version": MODEL_VERSION, "outcome": OUTCOME, "quantiles": list(study.QUANTILES),
        "quantile_columns": list(study.QUANTILE_COLUMNS), "trained_horizons": list(cfg.horizons),
        "plausible_range": list(study.PLAUSIBLE_RANGES[OUTCOME]), "threshold": cfg.threshold,
        "model": model, "domain_model": domain, "corrections": corrections,
        "probability_calibrators": calibrators, "domain_floor": cfg.domain_floor,
        "high_missingness_threshold": cfg.high_missingness_threshold,
        "global_gate_passed": bool(summary["all_cells_pass"]), "cell_gate_status": cell_gates,
        "claim": claim,
    }
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    study.atomic_pickle(cfg.model_dir / "deployment.pkl", {
        **deployment,
        "validation_vs_roster": validation.to_dict("records"),
        "heldout_performance": heldout.to_dict("records"),
        "probability_calibration_diagnostics": probability_calibration.to_dict("records"),
    })

    validation_public = suppress_public(
        study, validation, ["n"], ["evaluation_scope", "arm", "target_month", "reference_candidate",
                                   "better_than_roster", "coverage_ok", "noninferior", "cell_pass"])
    heldout_public = suppress_public(
        study, heldout, ["n"], ["evaluation_scope", "arm", "target_month", "reference_candidate",
                                "better_than_roster"])
    threshold_public = suppress_public(
        study, threshold, ["n", "events", "nonevents"],
        ["evaluation_scope", "arm", "target_month", "threshold", "probability_calibration_status", "status"])
    corrections_frame = study.pd.DataFrame(corrections)
    corrections_public = suppress_public(
        study, corrections_frame, ["n_calibration"],
        ["arm", "target_month", "coverage", "lower_index", "upper_index", "status"])
    probability_public = suppress_public(
        study, probability_calibration, ["n", "events", "nonevents"],
        ["arm", "target_month", "status"])
    matching_public = suppress_public(
        study, matching, ["n"], ["arm", "reference_arm_1", "reference_arm_2"])

    validation_public.to_csv(cfg.output_dir / "validation_vs_roster.csv", index=False)
    heldout_public.to_csv(cfg.output_dir / "heldout_performance.csv", index=False)
    threshold_public.to_csv(cfg.output_dir / "threshold_performance.csv", index=False)
    corrections_public.to_csv(cfg.output_dir / "conformal_corrections.csv", index=False)
    probability_public.to_csv(cfg.output_dir / "probability_calibration.csv", index=False)
    matching_public.to_csv(cfg.output_dir / "matching_reference.csv", index=False)
    curves.to_csv(cfg.output_dir / "threshold_roc_curve_points.csv", index=False)
    roster_selection = source_metadata.get("roster_selection")
    if roster_selection is not None:
        roster_selection.to_csv(cfg.output_dir / "horizon_matched_roster_selection.csv", index=False)

    model_bundle, feature_schema = write_model_package(
        study, cfg, deployment, validation_public, heldout_public, threshold_public,
        corrections_public, probability_public)
    rendered = render_public_book(
        study, cfg, validation_public, heldout_public, threshold_public, curves,
        matching_public, summary, feature_schema)

    manifest = {
        "version": MODEL_VERSION, "completed_utc": utc_now(), "script_sha256": sha256_file(Path(__file__)),
        "source_run": str(cfg.source_run), "source_manifest_sha256": source_metadata.get("source_manifest_sha256"),
        "configuration": {
            "horizons": list(cfg.horizons), "threshold": cfg.threshold, "seed": cfg.seed,
            "max_training_rows": cfg.max_training_rows, "iterations": cfg.iterations,
            "noninferiority_margin": cfg.noninferiority_margin,
            "coverage_tolerance": cfg.coverage_tolerance,
        },
        "training_rows": int(len(data["train"])), "validation_rows": int(len(data["validation"])),
        "calibration_rows": int(len(data["calibration"])), "heldout_rows": int(len(data["heldout"])),
        "gate_summary": summary, "cell_gate_status": cell_gates, "claim": claim,
        "public_figures": [path.name for path in rendered], "model_bundle": model_bundle.name,
        "patient_scenarios_internal_only": True,
    }
    study.atomic_json(cfg.output_dir / "manifest.json", manifest)

    public_files = [
        *rendered,
        cfg.output_dir / "validation_vs_roster.csv",
        cfg.output_dir / "heldout_performance.csv",
        cfg.output_dir / "threshold_performance.csv",
        cfg.output_dir / "conformal_corrections.csv",
        cfg.output_dir / "probability_calibration.csv",
        cfg.output_dir / "matching_reference.csv",
        cfg.output_dir / "threshold_roc_curve_points.csv",
        cfg.output_dir / "horizon_matched_roster_selection.csv",
        cfg.output_dir / "manifest.json",
        model_bundle,
    ]
    results_bundle = write_results_bundle(cfg, public_files)
    print(f"[hba1c] figures: {cfg.export}")
    print(f"[hba1c] portable model: {model_bundle}")
    print(f"[hba1c] public results bundle: {results_bundle}")
    print(f"[hba1c] INTERNAL scenarios: {cfg.scenario_dir}")
    return cfg.output_dir


def synthetic_fixture(study: Any, cfg: Config) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    rng = study.np.random.default_rng(cfg.seed)
    patients = 1200
    arms = study.np.asarray([ARMS[index % 3] for index in range(patients)])
    baseline = study.np.clip(7.8 + 0.8 * rng.normal(size=patients) + 0.25 * (arms == "incretin"), 5.0, 12.0)
    bmi = study.np.clip(43 + 5 * rng.normal(size=patients), 30, 70)
    age = study.np.clip(51 + 11 * rng.normal(size=patients), 18, 85)
    split_cycle = study.np.arange(patients) % 20
    splits = study.np.where(split_cycle < 12, "train",
             study.np.where(split_cycle < 15, "validation",
             study.np.where(split_cycle < 17, "calibration", "temporal_test")))
    rows = []
    effects = {"sleeve": -0.75, "rygb": -1.0, "incretin": -1.15}
    offsets = study.np.asarray([-1.645, -1.282, -0.674, 0.0, 0.674, 1.282, 1.645])
    reference_rows = []
    row_id = 0
    for index in range(patients):
        for month in cfg.horizons:
            horizon_scale = math.sqrt(month / 12.0)
            target = baseline[index] + effects[str(arms[index])] * horizon_scale + 0.12 * (baseline[index] - 7.5)
            target += 0.004 * (age[index] - 50) + rng.normal(scale=0.28 * horizon_scale)
            target = float(study.np.clip(target, 3.5, 15.0))
            cohort = "incretin" if arms[index] == "incretin" else "surgery"
            treatment = arms[index]
            record = {
                "row_id": row_id, "patient_id": f"SYN-{index:05d}", "cohort": cohort,
                "outcome": OUTCOME, "origin_month": 0, "target_month": month,
                "split": str(splits[index]), "target_value": target, "target_observed": True,
                "analysis_weight": float(study.np.clip(rng.lognormal(0, 0.15), 0.5, 2.0)),
                "support_status": "mature_with_target", "treatment": treatment, "center_id": "synthetic",
                "prediction_reference_value": float(baseline[index]), "baseline_value": float(baseline[index]),
                "baseline_cross_outcome": float(bmi[index]), "age_at_index": float(age[index]),
                "diabetes_flag": 1.0, "hypertension": float(age[index] > 52),
                "dyslipidemia": float(index % 4 != 0), "osa": float(bmi[index] > 44),
                "insulin": float(baseline[index] > 8.5), "biguanide": float(index % 5 != 0),
                "sglt2": float(index % 3 == 0), "svi": float(rng.uniform()), "index_year": 2021 + index % 3,
                "sex": "F" if index % 2 else "M", "race": "White" if index % 3 else "Black",
                "ethnicity": "Not Hispanic" if index % 5 else "Hispanic", "coverage": "Commercial" if index % 2 else "Public",
                "smoking": "Never" if index % 4 else "Current", "scenario_arm": str(arms[index]),
            }
            rows.append(record)
            # A deliberately simple old-roster reference: arm/horizon mean plus wider residual error.
            center = baseline[index] + effects[str(arms[index])] * horizon_scale + rng.normal(scale=0.38 * horizon_scale)
            spread = 0.50 * horizon_scale
            reference = {name: record[name] for name in (
                "row_id", "patient_id", "cohort", "outcome", "origin_month", "target_month", "split",
                "target_value", "target_observed", "analysis_weight", "support_status", "treatment",
                "center_id", "prediction_reference_value")}
            reference["scenario_arm"] = str(arms[index]); reference["reference_candidate"] = "synthetic_old_roster"
            for position, column in enumerate(study.QUANTILE_COLUMNS):
                reference[column] = float(study.np.clip(center + offsets[position] * spread, 3.0, 20.0))
            reference_rows.append(reference)
            row_id += 1
    frame = study.pd.DataFrame(rows)
    data = {name: frame.loc[frame["split"].eq(name)].copy() for name in (*DEVELOPMENT, *HELD_OUT)}
    data["heldout"] = study.concat_frames([data[name] for name in HELD_OUT])
    first = frame.loc[frame["target_month"].eq(min(cfg.horizons))].drop_duplicates("patient_id")
    data["profiles"] = first.loc[first["split"].isin(HELD_OUT)].copy()
    data["domain"] = first.loc[first["split"].isin(DEVELOPMENT)].copy()
    data["train"] = stratified_sample(study, data["train"], cfg.max_training_rows, cfg.seed)
    reference = study.pd.DataFrame(reference_rows)
    reference = reference.loc[reference["split"].isin(("validation", *HELD_OUT))].copy()
    return data, reference, {"source_manifest_sha256": "synthetic"}


def run_self_test(study: Any, cfg: Config) -> int:
    with tempfile.TemporaryDirectory(prefix="shared-hba1c-selftest-") as directory:
        temp = Path(directory)
        test_cfg = replace(
            cfg,
            source_run=temp / "source",
            output_dir=temp / "output",
            horizons=(12, 24),
            iterations=80,
            max_training_rows=5000,
            domain_training_rows=5000,
            patients_per_arm=2,
            bootstrap_replicates=40,
            overwrite=True,
            self_test=True,
        )
        data, reference, metadata = synthetic_fixture(study, test_cfg)
        metadata["roster_selection"] = study.pd.DataFrame([
            {"cohort": "surgery", "outcome": OUTCOME, "origin_month": 0,
             "selected_candidate": "synthetic_old_roster"},
            {"cohort": "incretin", "outcome": OUTCOME, "origin_month": 0,
             "selected_candidate": "synthetic_old_roster"},
        ])
        run_pipeline(study, test_cfg, data, reference, metadata)
        required = [
            test_cfg.export / FIGURE_BOOK,
            test_cfg.output_dir / MODEL_BUNDLE,
            test_cfg.output_dir / RESULTS_BUNDLE,
            test_cfg.model_dir / "research_model.pkl",
            test_cfg.model_dir / "inference.py",
            test_cfg.scenario_dir / "patient_scenario_projections.csv",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise AssertionError("missing self-test artifacts: " + ", ".join(missing))
        input_csv = temp / "input.csv"; output_csv = temp / "scored.csv"
        study.pd.DataFrame([{
            "baseline_value": 8.2, "baseline_cross_outcome": 42.0,
            "scenario_arm": "incretin", "target_month": 12,
        }]).to_csv(input_csv, index=False)
        completed = subprocess.run([
            sys.executable, str(test_cfg.model_dir / "inference.py"),
            str(test_cfg.model_dir / "research_model.pkl"), str(input_csv), str(output_csv),
        ], capture_output=True, text=True)
        if completed.returncode != 0:
            raise AssertionError(f"inference self-test failed: {completed.stderr}")
        scored = study.pd.read_csv(output_csv)
        expected = set(study.QUANTILE_COLUMNS) | {"probability_below_threshold", "model_domain_score"}
        if not expected.issubset(scored.columns):
            raise AssertionError(f"inference output missing columns: {sorted(expected - set(scored.columns))}")
        print("SELF-TEST PASSED: training, calibration, figures, portable package, and CSV inference")
    return 0


def parse_horizons(text: str, supported: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))
    if not values:
        raise ValueError("At least one horizon is required")
    invalid = sorted(set(values) - set(map(int, supported)))
    if invalid:
        raise ValueError(f"Unsupported HbA1c horizons: {invalid}; supported={tuple(supported)}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--from-run", metavar="RUN_DIR", help="Completed metabolic trajectory run")
    modes.add_argument("--self-test", action="store_true", help="Deterministic synthetic end-to-end check")
    parser.add_argument("--output-dir", help="Default: <RUN_DIR>/shared_hba1c_catboost")
    parser.add_argument("--study-script", help="Path to run_metabolic_trajectory_study.py")
    parser.add_argument("--scripts-dir", help="Directory containing the study script")
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)),
                        help="Comma-separated HbA1c horizons (default 3,6,12); horizons the source run "
                             "did not pre-build are reconstructed from its retained measurements")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Calibrated P(HbA1c < threshold); default 7.0")
    parser.add_argument("--seed", type=int, default=None, help="Default: source run seed")
    parser.add_argument("--max-training-rows", type=int, default=120_000)
    parser.add_argument("--domain-training-rows", type=int, default=200_000)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--patients-per-arm", type=int, default=5)
    parser.add_argument("--match-caliper", type=float, default=1.0, help="HbA1c percentage points")
    parser.add_argument("--noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--coverage-tolerance", type=float, default=0.05)
    parser.add_argument("--domain-floor", type=float, default=0.05)
    parser.add_argument("--high-missingness-threshold", type=float, default=0.50)
    parser.add_argument("--bootstrap-replicates", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    study_path = find_study_script(args.study_script, args.scripts_dir)
    study = load_study(study_path)
    try:
        horizons = parse_horizons(args.horizons, sorted(study.WINDOW_MONTHS))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not study.PLAUSIBLE_RANGES[OUTCOME][0] < args.threshold < study.PLAUSIBLE_RANGES[OUTCOME][1]:
        raise SystemExit(f"--threshold must lie inside {study.PLAUSIBLE_RANGES[OUTCOME]}")
    positive = {
        "max-training-rows": args.max_training_rows, "domain-training-rows": args.domain_training_rows,
        "iterations": args.iterations, "patients-per-arm": args.patients_per_arm,
        "bootstrap-replicates": args.bootstrap_replicates,
    }
    bad = [name for name, value in positive.items() if int(value) < 1]
    if bad:
        raise SystemExit("These arguments must be positive: " + ", ".join(bad))
    for name, value in (("noninferiority-margin", args.noninferiority_margin),
                        ("coverage-tolerance", args.coverage_tolerance),
                        ("match-caliper", args.match_caliper)):
        if value < 0:
            raise SystemExit(f"--{name} cannot be negative")
    if not 0 < args.domain_floor < 1 or not 0 < args.high_missingness_threshold < 1:
        raise SystemExit("--domain-floor and --high-missingness-threshold must lie inside (0,1)")

    source = Path(args.from_run).expanduser().resolve() if args.from_run else Path(tempfile.gettempdir()) / "shared_hba1c_selftest_source"
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir
              else (source / "shared_hba1c_catboost" if args.from_run
                    else Path(tempfile.gettempdir()) / "shared_hba1c_selftest_output"))
    seed = int(args.seed if args.seed is not None else study.SEED)
    cfg = Config(
        source_run=source, output_dir=output, study_script=study_path, horizons=horizons,
        threshold=float(args.threshold), seed=seed, max_training_rows=int(args.max_training_rows),
        domain_training_rows=int(args.domain_training_rows), iterations=int(args.iterations),
        patients_per_arm=int(args.patients_per_arm), match_caliper=float(args.match_caliper),
        noninferiority_margin=float(args.noninferiority_margin),
        coverage_tolerance=float(args.coverage_tolerance), domain_floor=float(args.domain_floor),
        high_missingness_threshold=float(args.high_missingness_threshold),
        bootstrap_replicates=int(args.bootstrap_replicates), overwrite=bool(args.overwrite),
        self_test=bool(args.self_test),
    )
    if args.self_test:
        return run_self_test(study, cfg)
    if not source.is_dir():
        raise SystemExit(f"Source run does not exist: {source}")
    if set(cfg.horizons) <= set(study.TARGET_MONTHS[OUTCOME]):
        weighted_rows, calibration, metadata = source_context(study, source)
    else:
        weighted_rows, calibration, metadata = rebuild_source_rows(study, source, cfg)
    if args.seed is None:
        cfg = replace(cfg, seed=int(metadata.get("source_seed", study.SEED)))
    data = collect_source_rows(study, weighted_rows, cfg)
    reference, roster_selection = collect_reference_predictions(
        study, calibration, cfg.horizons, metadata["scale_map"])
    metadata["roster_selection"] = roster_selection
    run_pipeline(study, cfg, data, reference, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
