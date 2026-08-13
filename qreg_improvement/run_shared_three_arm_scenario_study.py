#!/usr/bin/env python3
"""Experimental shared three-arm BMI forecasting and scenario-projection study.

This wrapper leaves the existing study scripts unchanged. It:
  * runs their cohort/split/weighting pipeline in a fresh run directory;
  * preserves every original model candidate and baseline;
  * adds shared treatment-conditioned BMI quantile architectures;
  * lets shared candidates compete in canonical origin-0 BMI selection on 3/6/12-month targets;
  * separately selects one shared architecture using factual arm-by-horizon validation;
  * creates BMI-matched theoretical scenario projections for five factual patients
    from EACH observed arm (RYGB, sleeve, sustained incretin treatment); and
  * queues the unchanged final-study and descriptive three-arm scripts.

The switched-arm trajectories are cross-arm prognostic model projections. They are
not identified individual treatment effects and are not treatment recommendations.
Real-patient scenario files and figures remain under INTERNAL/ and are excluded from
returnable bundles by default.
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
import re
import shutil
import subprocess
import sys
import warnings
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ARMS = ("sleeve", "rygb", "incretin")
HELD_OUT = ("temporal_test", "geographic_test")
DEVELOPMENT = ("train", "validation", "calibration")
BMI35_THRESHOLD = 35.0
SHARED_HORIZONS = (3, 6, 12)
CUSTOM = {
    "hgb": ("shared_arm_hgb", "shared_horizon_hgb_quantile"),
    "gbr": ("shared_arm_gbr", "shared_horizon_gbr_quantile"),
    "catboost": ("shared_arm_catboost", "shared_horizon_catboost_multiquantile"),
    "ensemble": ("shared_arm_ensemble", "shared_architecture_ensemble"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", help="Fresh run folder; default: ./results/shared_three_arm_scenario_TIMESTAMP")
    parser.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parent),
                        help="Folder containing the existing study scripts")
    parser.add_argument("--smoke", action="store_true", help="Use the source study's deterministic synthetic fixture")
    parser.add_argument("--smoke-patients", type=int, default=180,
                        help="Synthetic patients in smoke mode; production ignores this option")
    parser.add_argument("--architectures", default="hgb,gbr,catboost,ensemble",
                        help="Comma-separated shared candidates: hgb,gbr,catboost,ensemble")
    parser.add_argument("--max-training-rows", type=int, default=120_000)
    parser.add_argument("--domain-training-rows", type=int, default=200_000,
                        help="Unstratified development-patient cap for the model-domain diagnostic")
    parser.add_argument("--hgb-iterations", type=int, default=220)
    parser.add_argument("--gbr-estimators", type=int, default=180)
    parser.add_argument("--catboost-iterations", type=int, default=500)
    parser.add_argument("--forecast-horizons", default="3,6,12",
                        help="Shared-model horizons; restricted to a subset of 3,6,12")
    parser.add_argument("--patients-per-arm", type=int, default=5)
    parser.add_argument("--bmi-match-caliper", type=float, default=2.0,
                        help="BMI-unit audit threshold; nearest profiles are retained but flagged if outside it")
    parser.add_argument("--domain-floor", type=float, default=0.05,
                        help="Descriptive model-domain flag only; not a propensity or causal positivity threshold")
    parser.add_argument("--high-missingness-threshold", type=float, default=0.50,
                        help="Warn when this fraction of model input features is missing for a scored row")
    parser.add_argument("--noninferiority-margin", type=float, default=0.05,
                        help="Maximum factual CRPS increase versus the original-roster selected model in each cell")
    parser.add_argument("--coverage-tolerance", type=float, default=0.05,
                        help="Allowed absolute error from nominal 80%% and 90%% factual coverage")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--incretin-qualifying-months", type=int, choices=(6, 12), default=12)
    parser.add_argument("--skip-downstream", action="store_true",
                        help="Do not call final-study and descriptive three-arm scripts")
    parser.add_argument("--run-three-arm-full", action="store_true",
                        help="Also queue the independent gated full target trial")

    # Private process-per-stage entry points.
    parser.add_argument("--_original-stage", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-models", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-calibration", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-scenarios", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-render", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_model-script", help=argparse.SUPPRESS)
    return parser


def default_output_dir() -> Path:
    root = Path.cwd().resolve() / "results"
    stem = f"shared_three_arm_scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path, suffix = root / stem, 1
    while path.exists():
        path, suffix = root / f"{stem}_{suffix:02d}", suffix + 1
    return path


def find_script(folder: Path, canonical: str) -> Path:
    for path in (folder / canonical, folder / canonical.replace(".py", "(2).py")):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Could not find {canonical} or its '(2)' upload name under {folder}")


def load_study(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("run_metabolic_trajectory_study", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load_runtime_packages()

    # Source checkpoints are pickled by workers running as __main__. Bind their classes here so
    # this wrapper's workers can exchange those verified checkpoints without changing the source.
    current_main = sys.modules["__main__"]
    for name in ("DataBundle", "RowStore", "PredictionStore", "TabularEncoder", "RunConfig",
                 "RunContext", "TargetWindow", "CoverageRecord", "CoverageEpisode", "PreflightError"):
        value = getattr(module, name, None)
        if value is not None:
            setattr(current_main, name, value)
            if isinstance(value, type):
                value.__module__ = "__main__"
    return module


def run(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    print("[shared]", " ".join(map(str, command)), flush=True)
    completed = subprocess.run(list(command), env=dict(env) if env else None)
    if completed.returncode:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def parse_horizons(text: str, supported: Sequence[int] | None = None) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))
    if not values:
        raise ValueError("At least one forecast horizon is required")
    if supported is not None:
        invalid = sorted(set(values) - set(map(int, supported)))
        if invalid:
            raise ValueError(f"Unsupported BMI horizon(s): {invalid}; supported={tuple(supported)}")
    return values


def requested_architectures(text: str) -> list[str]:
    values = list(dict.fromkeys(item.strip().lower() for item in text.split(",") if item.strip()))
    unknown = sorted(set(values) - set(CUSTOM))
    if unknown:
        raise ValueError(f"Unknown architecture(s): {', '.join(unknown)}")
    if not any(item != "ensemble" for item in values):
        raise ValueError("At least one non-ensemble shared architecture is required")
    return values


def validate_args(args: argparse.Namespace, study: Any) -> tuple[list[str], tuple[int, ...]]:
    architectures = requested_architectures(args.architectures)
    horizons = parse_horizons(args.forecast_horizons, study.TARGET_MONTHS["bmi"])
    if not set(horizons).issubset(SHARED_HORIZONS):
        raise ValueError(f"Shared-model horizons are restricted to {SHARED_HORIZONS}")
    positive = {
        "max-training-rows": args.max_training_rows,
        "domain-training-rows": args.domain_training_rows,
        "hgb-iterations": args.hgb_iterations,
        "gbr-estimators": args.gbr_estimators,
        "catboost-iterations": args.catboost_iterations,
        "patients-per-arm": args.patients_per_arm,
        "smoke-patients": args.smoke_patients,
    }
    bad = [name for name, value in positive.items() if int(value) < 1]
    if bad:
        raise ValueError("These arguments must be positive: " + ", ".join(bad))
    if not 0.0 < args.domain_floor < 1.0:
        raise ValueError("--domain-floor must be strictly between 0 and 1")
    if not 0.0 < args.high_missingness_threshold < 1.0:
        raise ValueError("--high-missingness-threshold must be strictly between 0 and 1")
    if args.bmi_match_caliper < 0 or args.noninferiority_margin < 0 or args.coverage_tolerance < 0:
        raise ValueError("Caliper, noninferiority margin, and coverage tolerance cannot be negative")
    return architectures, horizons


def run_config(study: Any, args: argparse.Namespace) -> Any:
    base = study.RunConfig.create("smoke" if args.smoke else "production", args.output_dir, False,
                                  incretin_qualifying_months=args.incretin_qualifying_months)
    return replace(base, seed=int(args.seed),
                   smoke_patients=int(args.smoke_patients) if args.smoke else base.smoke_patients)


def arm_series(frame: Any) -> Any:
    pd = __import__("pandas")
    cohort = frame["cohort"].astype("string").str.lower()
    treatment = frame.get("treatment", pd.Series(pd.NA, index=frame.index, dtype="string")).astype("string").str.lower()
    if "procedure" in frame:
        procedure = frame["procedure"].astype("string").str.lower()
        treatment = treatment.where(treatment.notna() & treatment.str.strip().ne(""), procedure)
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    result.loc[cohort.eq("incretin")] = "incretin"
    surgery = cohort.eq("surgery")
    result.loc[surgery & treatment.str.contains(r"rygb|roux|gastric\s*bypass", na=False)] = "rygb"
    result.loc[surgery & treatment.str.contains("sleeve", na=False)] = "sleeve"
    return result


def feature_lists(study: Any, include_horizon: bool = True) -> tuple[list[str], list[str]]:
    numeric = [
        "baseline_value", "baseline_cross_outcome", "age_at_index", "diabetes_flag",
        "hypertension", "dyslipidemia", "osa", "insulin", "biguanide", "sglt2",
        "svi", "index_year", *study.OPTIONAL_WIDE_NUMERIC_COVARIATES,
    ]
    if include_horizon:
        numeric.append("target_month")
    categorical = ["scenario_arm", "sex", "race", "ethnicity", "coverage", "smoking"]
    return numeric, categorical


def prepare(frame: Any, scenario_arm: str | None = None) -> Any:
    result = frame.copy()
    result["scenario_arm"] = scenario_arm if scenario_arm is not None else arm_series(result)
    return result


# Plain-dict encoder state keeps deployment.pkl portable; no wrapper or __main__ class is pickled.
def fit_encoder(study: Any, frame: Any, numeric: Sequence[str], categorical: Sequence[str]) -> dict[str, Any]:
    numeric = [name for name in numeric if name in frame]
    categorical = [name for name in categorical if name in frame]
    medians, scales, levels = {}, {}, {}
    for name in numeric:
        values = study.pd.to_numeric(frame[name], errors="coerce")
        medians[name] = float(values.median()) if values.notna().any() else 0.0
        scale = float(values.std(ddof=0)) if values.notna().sum() > 1 else 1.0
        scales[name] = scale if math.isfinite(scale) and scale > 1e-8 else 1.0
    for name in categorical:
        values = frame[name].astype("string").fillna("<MISSING>")
        levels[name] = sorted(map(str, values.unique()))
    return {
        "numeric": list(numeric), "categorical": list(categorical), "medians": medians,
        "scales": scales, "levels": levels,
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


def stratified_sample(study: Any, frame: Any, maximum: int, seed: int,
                      strata: Sequence[str]) -> Any:
    if len(frame) <= maximum:
        return frame.reset_index(drop=True)
    groups = list(frame.groupby(list(strata), sort=True, observed=True))
    quota = max(20, maximum // max(len(groups), 1))
    pieces = [group.sample(min(len(group), quota), random_state=seed + index)
              for index, (_key, group) in enumerate(groups)]
    sampled = study.pd.concat(pieces, ignore_index=True)
    if len(sampled) > maximum:
        sampled = sampled.sample(maximum, random_state=seed)
    return sampled.reset_index(drop=True)


def collect_origin0_bmi(study: Any, rows: Any, split: str, maximum: int, seed: int,
                        horizons: Sequence[int]) -> Any:
    pieces = []
    for key in rows.keys():
        if tuple(key) not in (("surgery", "bmi", 0), ("incretin", "bmi", 0)):
            continue
        frame = rows.read(key)
        keep = (
            frame["split"].astype(str).eq(split)
            & frame["target_observed"].fillna(False).astype(bool)
            & study.pd.to_numeric(frame["target_month"], errors="coerce").isin(list(horizons))
        )
        frame = prepare(frame.loc[keep].copy())
        frame = frame.loc[frame["scenario_arm"].isin(ARMS)]
        if not frame.empty:
            pieces.append(stratified_sample(study, frame, max(500, maximum // 2), seed + len(pieces),
                                             ("scenario_arm", "target_month")))
    if not pieces:
        raise RuntimeError(f"No origin-0 BMI rows in split {split!r}")
    return stratified_sample(study, study.pd.concat(pieces, ignore_index=True), maximum, seed,
                             ("scenario_arm", "target_month"))


def fit_one(study: Any, kind: str, train: Any, args: argparse.Namespace) -> dict[str, Any] | None:
    numeric, categorical = feature_lists(study)
    encoder = fit_encoder(study, train, numeric, categorical)
    x = transform_encoder(study, encoder, train)
    y = study.pd.to_numeric(train["target_value"], errors="coerce").to_numpy(float)
    weights = study.pd.to_numeric(train["analysis_weight"], errors="coerce").fillna(1.0).to_numpy(float)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        models = [HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, max_iter=args.hgb_iterations,
            learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=2.0, early_stopping=False, random_state=args.seed,
        ).fit(x, y, sample_weight=weights) for quantile in study.QUANTILES]
        return {"kind": kind, "encoder": encoder, "models": models}
    if kind == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor
        x = study.np.nan_to_num(x)
        models = [GradientBoostingRegressor(
            loss="quantile", alpha=quantile, n_estimators=args.gbr_estimators,
            learning_rate=0.035, max_depth=3, min_samples_leaf=30,
            random_state=args.seed,
        ).fit(x, y, sample_weight=weights) for quantile in study.QUANTILES]
        return {"kind": kind, "encoder": encoder, "models": models}
    if kind == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            print("[shared] CatBoost unavailable; skipping shared CatBoost", flush=True)
            return None
        model = CatBoostRegressor(
            loss_function="MultiQuantile:alpha=" + ",".join(map(str, study.QUANTILES)),
            iterations=args.catboost_iterations, depth=7, learning_rate=0.04,
            l2_leaf_reg=8.0, random_seed=args.seed, random_strength=0.0,
            bootstrap_type="No", verbose=False, allow_writing_files=False, thread_count=1,
        )
        model.fit(study.np.nan_to_num(x), y, sample_weight=weights)
        return {"kind": kind, "encoder": encoder, "model": model}
    return None


def predict_model(study: Any, bundle: Mapping[str, Any], frame: Any) -> Any:
    if bundle["kind"] == "ensemble":
        matrix = sum(bundle["weights"][name] * predict_model(study, bundle["members"][name], frame)
                     for name in bundle["weights"])
    else:
        x = transform_encoder(study, bundle["encoder"], frame)
        if bundle["kind"] in {"gbr", "catboost"}:
            x = study.np.nan_to_num(x)
        matrix = (study.np.asarray(bundle["model"].predict(x), dtype=float)
                  if bundle["kind"] == "catboost"
                  else study.np.column_stack([model.predict(x) for model in bundle["models"]]))
    return study.np.clip(study.rearrange_quantiles(matrix), *study.PLAUSIBLE_RANGES["bmi"])


def cell_metrics(study: Any, frame: Any, matrix: Any, scales: Mapping[tuple[str, int], float]) -> Any:
    rows = []
    work = frame.reset_index(drop=True)
    for (arm, month), index in work.groupby(["scenario_arm", "target_month"], sort=True).groups.items():
        positions = study.np.asarray(list(index), dtype=int)
        y = work.loc[positions, "target_value"].to_numpy(float)
        weight = work.loc[positions, "analysis_weight"].to_numpy(float)
        q = matrix[positions]
        crps = float(study.quantile_crps(y, q, weight))
        rows.append({
            "arm": str(arm), "target_month": int(month), "n": int(len(positions)),
            "crps": crps, "standardized_crps": crps / max(float(scales[(str(arm), int(month))]), 1e-8),
            "coverage_80": float(study.weighted_mean((y >= q[:, 1]) & (y <= q[:, 5]), weight)),
            "coverage_90": float(study.weighted_mean((y >= q[:, 0]) & (y <= q[:, 6]), weight)),
        })
    return study.pd.DataFrame(rows)


def fit_shared_ensemble(study: Any, fitted: Mapping[str, Any], train: Any, validation: Any,
                        args: argparse.Namespace) -> dict[str, Any] | None:
    if len(fitted) < 2:
        return None
    scales = {}
    for (arm, month), cell in train.groupby(["scenario_arm", "target_month"], sort=True):
        y = cell["target_value"].to_numpy(float)
        q25, q75 = study.np.quantile(y, [0.25, 0.75])
        scales[(str(arm), int(month))] = max(float(q75 - q25), 1e-8)
    matrices = {name: predict_model(study, model, validation) for name, model in fitted.items()}
    base_rows = []
    for name, matrix in matrices.items():
        metrics = cell_metrics(study, validation, matrix, scales)
        base_rows.append({"candidate": name, "score": float(metrics["standardized_crps"].mean()),
                          "metrics": metrics})
    base_rows.sort(key=lambda row: (row["score"], row["candidate"]))
    chosen = [row["candidate"] for row in base_rows[:3]]
    best_base = base_rows[0]
    best_base_metrics = {(row.arm, int(row.target_month)): row for row in best_base["metrics"].itertuples()}
    weights = study.np.full(len(chosen), 1.0 / len(chosen))

    def evaluate(candidate_weights: Any) -> tuple[float, bool]:
        matrix = study.rearrange_quantiles(sum(weight * matrices[name]
                                               for weight, name in zip(candidate_weights, chosen, strict=True)))
        metrics = cell_metrics(study, validation, matrix, scales)
        guard = True
        for row in metrics.itertuples():
            base = best_base_metrics[(row.arm, int(row.target_month))]
            guard &= abs(row.coverage_80 - .80) <= abs(base.coverage_80 - .80) + 1e-8
            guard &= abs(row.coverage_90 - .90) <= abs(base.coverage_90 - .90) + 1e-8
        return float(metrics["standardized_crps"].mean()), bool(guard)

    best_score, _ = evaluate(weights)
    rng = study.np.random.default_rng(args.seed + 917)
    for _ in range(80 if args.smoke else 1000):
        trial = rng.dirichlet(study.np.ones(len(chosen)))
        score, guard = evaluate(trial)
        if guard and score < best_score:
            best_score, weights = score, trial
    return {
        "kind": "ensemble", "members": {name: fitted[name] for name in chosen},
        "weights": {name: float(weight) for name, weight in zip(chosen, weights, strict=True)},
        "validation_standardized_crps": float(best_score),
        "method": "production_style_dirichlet_equal_horizon_crps_with_coverage_guard",
    }


def cohort_domain_frame(study: Any, cohorts: Any) -> Any:
    frame = cohorts.copy()
    frame["scenario_arm"] = arm_series(frame)
    frame = frame.loc[frame["scenario_arm"].isin(ARMS)].copy()
    if "baseline_value" not in frame:
        frame["baseline_value"] = study.pd.to_numeric(frame.get("baseline_bmi"), errors="coerce")
    if "baseline_cross_outcome" not in frame:
        frame["baseline_cross_outcome"] = study.pd.to_numeric(frame.get("baseline_hba1c"), errors="coerce")
    if "index_year" not in frame and "index_date" in frame:
        frame["index_year"] = study.pd.to_datetime(frame["index_date"], errors="coerce").dt.year
    return frame.drop_duplicates(["patient_id", "cohort"], keep="first").reset_index(drop=True)


def fit_domain_model(study: Any, cohorts: Any, maximum: int, seed: int) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    frame = cohort_domain_frame(study, cohorts)
    if "split" in frame:
        frame = frame.loc[frame["split"].astype(str).isin(DEVELOPMENT)].copy()
    prevalence = frame["scenario_arm"].value_counts(normalize=True).sort_index().to_dict()
    if len(frame) > maximum:
        frame = frame.sample(maximum, random_state=seed).reset_index(drop=True)  # unstratified by design
    numeric, categorical = feature_lists(study, include_horizon=False)
    categorical = [name for name in categorical if name != "scenario_arm"]
    encoder = fit_encoder(study, frame, numeric, categorical)
    x = study.np.nan_to_num(transform_encoder(study, encoder, frame))
    model = LogisticRegression(max_iter=1500, C=.5, solver="lbfgs", random_state=seed)
    model.fit(x, frame["scenario_arm"].astype(str))
    return {
        "encoder": encoder, "model": model, "classes": list(map(str, model.classes_)),
        "training_rows": int(len(frame)), "development_prevalence": {str(k): float(v) for k, v in prevalence.items()},
        "interpretation": "model-domain membership diagnostic; not a propensity score or causal overlap proof",
    }


def domain_scores(study: Any, domain: Mapping[str, Any], frame: Any) -> Any:
    x = study.np.nan_to_num(transform_encoder(study, domain["encoder"], frame))
    raw = domain["model"].predict_proba(x)
    result = study.np.zeros((len(frame), len(ARMS)), dtype=float)
    for index, name in enumerate(domain["classes"]):
        if name in ARMS:
            result[:, ARMS.index(name)] = raw[:, index]
    return result


def model_encoder(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the common frozen encoder used by a fitted shared model or ensemble."""
    return (next(iter(model["members"].values()))["encoder"]
            if model["kind"] == "ensemble" else model["encoder"])


def input_missingness(study: Any, encoder: Mapping[str, Any], frame: Any,
                      threshold: float) -> tuple[Any, Any]:
    """Fraction of non-arm/non-horizon model features missing on each scored row."""
    features = [name for name in (*encoder["numeric"], *encoder["categorical"])
                if name not in {"scenario_arm", "target_month"}]
    if not features:
        fraction = study.np.zeros(len(frame), dtype=float)
    else:
        missing = []
        for name in features:
            if name not in frame:
                missing.append(study.np.ones(len(frame), dtype=float))
                continue
            values = frame[name]
            absent = values.isna()
            if name in encoder["categorical"]:
                absent = absent | values.astype("string").str.strip().eq("").fillna(True)
            missing.append(absent.to_numpy(float))
        fraction = study.np.column_stack(missing).mean(axis=1)
    return fraction, fraction >= float(threshold)


def extend_rosters(study: Any) -> None:
    study.CANDIDATE_LEVELS = tuple(sorted(set(study.CANDIDATE_LEVELS).union(value[0] for value in CUSTOM.values())))
    study.ARCHITECTURE_LEVELS = tuple(sorted(set(study.ARCHITECTURE_LEVELS).union(value[1] for value in CUSTOM.values())))


def worker_original_stage(args: argparse.Namespace, study: Any) -> int:
    cfg = run_config(study, args)
    dependencies, issues = study.dependency_manifest(require_database=not args.smoke)
    if issues:
        raise RuntimeError("; ".join(issues))
    study.run_stage_worker(cfg, dependencies, str(args._original_stage))
    return 0


def worker_models(args: argparse.Namespace, study: Any) -> int:
    cfg = run_config(study, args)
    context = study.load_run_context(cfg)
    weight_payload = study.require_checkpoint(context, "weights")
    weighted_rows = weight_payload["rows"]
    derived = study.require_checkpoint(context, "weights_derived")
    ode_gates = study.require_checkpoint(context, "ode_gates")
    dependencies, _ = study.dependency_manifest(require_database=False)
    extend_rosters(study)
    horizons = parse_horizons(args.forecast_horizons, study.TARGET_MONTHS["bmi"])

    # Fit and retain the complete original roster, including its simple baselines and original
    # validation-weighted ensemble. Shared candidates are appended and compete against it.
    original_store = context.new_prediction_store("original_uncalibrated")
    cohorts = measurements = None
    if bool(ode_gates["appropriate"].any()):
        cohorts = study.require_checkpoint(context, "global_splits")["cohorts"]
        measurements = study.require_checkpoint(context, "cohorts")["measurements"]
    original_store, status, details = study.fit_candidate_roster(
        weighted_rows, cfg, dependencies, ode_gates, derived["scale_map"], original_store,
        cohorts=cohorts, measurements=measurements,
    )
    store = context.new_prediction_store("uncalibrated")
    for key in original_store.keys():
        store.add(original_store.read(key), key=key)

    train = collect_origin0_bmi(
        study, weighted_rows, "train", args.max_training_rows, args.seed, horizons)
    validation = collect_origin0_bmi(study, weighted_rows, "validation",
                                     max(20_000, args.max_training_rows // 3), args.seed + 17,
                                     horizons)
    requested = requested_architectures(args.architectures)
    fitted: dict[str, Any] = {}
    for kind in requested:
        if kind == "ensemble":
            continue
        model = fit_one(study, kind, train, args)
        if model is not None:
            fitted[CUSTOM[kind][0]] = model
    if not fitted:
        raise RuntimeError("No shared architecture could be fitted")
    if "ensemble" in requested:
        ensemble = fit_shared_ensemble(study, fitted, train, validation, args)
        if ensemble is not None:
            fitted[CUSTOM["ensemble"][0]] = ensemble

    custom_status = []
    for key in weighted_rows.keys():
        if tuple(key) not in (("surgery", "bmi", 0), ("incretin", "bmi", 0)):
            continue
        task = weighted_rows.read(key)
        task = task.loc[
            study.pd.to_numeric(task["target_month"], errors="coerce").isin(list(horizons))
        ].copy()
        for candidate, model in fitted.items():
            architecture = next(value[1] for value in CUSTOM.values() if value[0] == candidate)
            matrix = predict_model(study, model, prepare(task))
            frame = study.prediction_identity(task, candidate, architecture)
            frame[list(study.QUANTILE_COLUMNS)] = study.stored_quantiles(matrix)
            frame = study.drop_training_predictions(frame)
            if not frame.empty:
                store.add(frame, key=key)
            custom_status.append({
                "cohort": key[0], "outcome": "bmi", "origin_month": 0,
                "candidate": candidate, "architecture": architecture, "status": "fitted",
                "reason": "shared baseline-only arm- and horizon-conditioned BMI model",
            })

    cohort_rows = study.require_checkpoint(context, "global_splits")["cohorts"]
    domain = fit_domain_model(study, cohort_rows, args.domain_training_rows, args.seed)
    deployment = {
        "version": "experimental-shared-three-arm-3.0", "quantiles": list(study.QUANTILES),
        "trained_horizons": list(map(int, horizons)),
        "models": fitted, "domain_model": domain, "training_rows": int(len(train)),
        "domain_floor": float(args.domain_floor),
        "high_missingness_threshold": float(args.high_missingness_threshold),
        "architectures_requested": requested,
        "feature_note": "baseline-only shared features plus scenario_arm and target_month",
        "claim": "cross-arm prognostic scenario model; not an identified individual treatment effect",
    }
    deploy_dir = context.internal / "shared_model"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    study.atomic_pickle(deploy_dir / "deployment.pkl", deployment)
    study.atomic_json(deploy_dir / "manifest.json", {
        "version": deployment["version"], "created_utc": study.utc_now(),
        "wrapper_sha256": study.sha256_file(Path(__file__)), "candidates": list(fitted),
        "training_rows": len(train), "domain_training_rows": domain["training_rows"],
        "trained_horizons": list(horizons),
        "domain_interpretation": domain["interpretation"], "claim": deployment["claim"],
    })

    status = study.pd.concat([status, study.pd.DataFrame(custom_status)], ignore_index=True)
    details["shared_three_arm"] = {
        "candidates": list(fitted), "training_rows": len(train),
        "trained_horizons": list(horizons),
        "domain_training_rows": domain["training_rows"], "manifest": str(deploy_dir / "manifest.json"),
    }
    payload = {"predictions": store, "status": status, "details": details}
    context.save_checkpoint("models_and_predictions", payload,
                            {"weights": study.checkpoint_hash(context, "weights")})
    return 0


def prognostic_metrics(study: Any, cell: Any) -> dict[str, float]:
    y = cell["target_value"].to_numpy(float)
    matrix = cell[list(study.QUANTILE_COLUMNS)].to_numpy(float)
    weight = cell["analysis_weight"].to_numpy(float)
    median = matrix[:, 3]
    return {
        "n": int(len(cell)), "ess": float(study.effective_sample_size(weight)),
        "crps": float(study.quantile_crps(y, matrix, weight)),
        "rmse": float(study.np.sqrt(study.np.average((y - median) ** 2, weights=weight))),
        "mae": float(study.np.average(study.np.abs(y - median), weights=weight)),
        "bias": float(study.np.average(median - y, weights=weight)),
        "coverage_80": float(study.np.average((y >= matrix[:, 1]) & (y <= matrix[:, 5]), weights=weight)),
        "coverage_90": float(study.np.average((y >= matrix[:, 0]) & (y <= matrix[:, 6]), weights=weight)),
        "width_80": float(study.np.average(matrix[:, 5] - matrix[:, 1], weights=weight)),
    }


def factual_metrics(study: Any, store: Any, candidate_map: Mapping[tuple[str, str, int], str] | None,
                    fixed_candidate: str | None, split: str, horizons: Sequence[int]) -> Any:
    """Factual arm-by-horizon metrics.

    With ``fixed_candidate`` set, score that candidate in both source cohorts. With
    ``candidate_map`` set, score the mapped task-specific candidate. With neither set,
    score every candidate present; this is the shared-architecture validation path.
    """
    rows = []
    for key in (("surgery", "bmi", 0), ("incretin", "bmi", 0)):
        if key not in store.keys():
            continue
        frame = store.read(key)
        if fixed_candidate:
            candidates = [str(fixed_candidate)]
        elif candidate_map is not None:
            mapped = candidate_map.get(key)
            candidates = [str(mapped)] if mapped else []
        else:
            candidates = sorted(frame["candidate"].astype(str).dropna().unique())
        for candidate in candidates:
            cell_frame = frame.loc[
                frame["candidate"].astype(str).eq(candidate)
                & frame["split"].astype(str).eq(split)
                & frame["target_observed"].fillna(False).astype(bool)
                & study.pd.to_numeric(frame["target_month"], errors="coerce").isin(list(horizons))
            ].copy()
            cell_frame["arm"] = arm_series(cell_frame)
            cell_frame = cell_frame.loc[cell_frame["arm"].isin(ARMS)]
            for (arm, month), cell in cell_frame.groupby(["arm", "target_month"], sort=True):
                cell = cell.drop_duplicates("patient_id")
                if cell.empty:
                    continue
                rows.append({
                    "candidate": candidate, "cohort": key[0], "arm": str(arm),
                    "target_month": int(month), **prognostic_metrics(study, cell),
                })
    return study.pd.DataFrame(rows)


def select_shared_candidate(study: Any, shared_metrics: Any, reference_metrics: Any,
                            candidates: Sequence[str], horizons: Sequence[int],
                            margin: float, coverage_tolerance: float) -> tuple[str, Any, Any, bool]:
    """Choose one shared candidate only after every factual arm-horizon cell is audited."""
    expected = study.pd.DataFrame(
        [(arm, int(month)) for arm in ARMS for month in horizons],
        columns=["arm", "target_month"],
    )
    if reference_metrics.empty:
        reference_metrics = expected.assign(
            candidate=study.pd.NA, crps=study.np.nan, coverage_80=study.np.nan,
            coverage_90=study.np.nan, n=0, ess=study.np.nan)
    reference = reference_metrics.rename(columns={
        "candidate": "reference_candidate", "crps": "reference_crps",
        "coverage_80": "reference_coverage_80", "coverage_90": "reference_coverage_90",
        "n": "reference_n", "ess": "reference_ess",
    })[["arm", "target_month", "reference_candidate", "reference_crps",
        "reference_coverage_80", "reference_coverage_90", "reference_n", "reference_ess"]]
    detail_rows, summaries = [], []
    for candidate in candidates:
        current = shared_metrics.loc[shared_metrics["candidate"].astype(str).eq(str(candidate))].copy()
        current = expected.merge(current, on=["arm", "target_month"], how="left")
        current = current.merge(reference, on=["arm", "target_month"], how="left")
        current["candidate"] = str(candidate)
        current["crps_ratio_vs_original"] = current["crps"] / current["reference_crps"]
        current["candidate_n_ok"] = current["n"].ge(study.MIN_CELL_SIZE)
        current["reference_n_ok"] = current["reference_n"].ge(study.MIN_CELL_SIZE)
        current["coverage_ok"] = (
            (current["coverage_80"] - .80).abs().le(coverage_tolerance)
            & (current["coverage_90"] - .90).abs().le(coverage_tolerance)
        )
        current["noninferior"] = current["crps_ratio_vs_original"].le(1.0 + margin)
        current["cell_pass"] = (
            current["candidate_n_ok"] & current["reference_n_ok"]
            & current["coverage_ok"] & current["noninferior"]
            & current["crps"].notna() & current["reference_crps"].notna()
        )
        detail_rows.append(current)
        ratios = study.pd.to_numeric(current["crps_ratio_vs_original"], errors="coerce")
        finite = ratios[study.np.isfinite(ratios)]
        summaries.append({
            "candidate": str(candidate), "expected_cells": int(len(expected)),
            "scored_cells": int(current["crps"].notna().sum()),
            "passed_cells": int(current["cell_pass"].fillna(False).sum()),
            "all_cells_pass": bool(current["cell_pass"].fillna(False).all()),
            "worst_crps_ratio": float(finite.max()) if len(finite) else math.inf,
            "mean_crps_ratio": float(finite.mean()) if len(finite) else math.inf,
            "mean_crps": float(study.pd.to_numeric(current["crps"], errors="coerce").mean()),
        })
    details = study.pd.concat(detail_rows, ignore_index=True)
    summary = study.pd.DataFrame(summaries)
    passing = summary.loc[summary["all_cells_pass"].astype(bool)]
    gate_passed = not passing.empty
    pool = passing if gate_passed else summary
    winner = pool.sort_values(
        ["passed_cells", "worst_crps_ratio", "mean_crps_ratio", "mean_crps", "candidate"],
        ascending=[False, True, True, True, True], kind="mergesort",
    ).iloc[0]
    return str(winner["candidate"]), details, summary, bool(gate_passed)

def write_inference_script(path: Path) -> None:
    code = r'''#!/usr/bin/env python3
"""Score the experimental shared BMI model from a CSV. Not for treatment selection."""
import argparse, pickle, warnings
import numpy as np
import pandas as pd

ARMS=("sleeve","rygb","incretin")
REQUIRED=("baseline_value","scenario_arm","target_month")

def transform(enc, frame):
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

def encoder_for(model):
    return next(iter(model["members"].values()))["encoder"] if model["kind"]=="ensemble" else model["encoder"]

def missing_fraction(enc,frame):
    features=[n for n in (*enc["numeric"],*enc["categorical"]) if n not in {"scenario_arm","target_month"}]
    if not features:return np.zeros(len(frame))
    cols=[]
    for name in features:
        if name not in frame: cols.append(np.ones(len(frame)));continue
        missing=frame[name].isna()
        if name in enc["categorical"]:missing=missing|frame[name].astype("string").str.strip().eq("").fillna(True)
        cols.append(missing.to_numpy(float))
    return np.column_stack(cols).mean(axis=1)

def predict(model, frame):
    if model["kind"]=="ensemble":
        matrix=sum(model["weights"][name]*predict(model["members"][name],frame) for name in model["weights"])
    else:
        x=transform(model["encoder"],frame)
        if model["kind"] in {"gbr","catboost"}: x=np.nan_to_num(x)
        matrix=np.asarray(model["model"].predict(x),float) if model["kind"]=="catboost" else np.column_stack([m.predict(x) for m in model["models"]])
    return np.sort(np.clip(matrix,10.0,100.0),axis=1)

def apply_corrections(bundle, frame, matrix):
    out=matrix.copy(); candidate=bundle["selected_candidate"]
    for i,row in frame.reset_index(drop=True).iterrows():
        arm=str(row["scenario_arm"]); month=int(row["target_month"])
        cohort,stratum=("incretin","incretin") if arm=="incretin" else ("surgery",arm)
        for item in bundle.get("corrections",[]):
            if item["candidate"]!=candidate or item["cohort"]!=cohort or item["stratum"]!=stratum or int(item["origin_month"])!=0 or int(item["target_month"])!=month: continue
            pair={.90:(0,6),.80:(1,5),.50:(2,4)}.get(round(float(item["coverage"]),2))
            if pair: out[i,pair[0]]-=float(item["correction"]); out[i,pair[1]]+=float(item["correction"])
    return np.sort(out,axis=1)

def cdf_below(row,threshold,levels):
    ladder=np.maximum.accumulate(np.asarray(row,float));levels=np.asarray(levels,float)
    if threshold<ladder[0]:
        value=levels[0]-(ladder[0]-threshold)*levels[0]/max(ladder[1]-ladder[0],1e-6)
    elif threshold>ladder[-1]:
        value=levels[-1]+(threshold-ladder[-1])*(1-levels[-1])/max(ladder[-1]-ladder[-2],1e-6)
    else:value=np.interp(threshold,ladder,levels)
    return float(np.clip(value,0,1))

def calibrated_probability(bundle,raw,arm,month):
    item=bundle.get("probability_calibrators",{}).get(f"{arm}|{int(month)}",{})
    if item.get("status")!="isotonic_calibrated":return float(raw),item.get("status","raw_predictive_cdf_no_calibrator")
    x=np.asarray(item["x_thresholds"],float);y=np.asarray(item["y_thresholds"],float)
    return float(np.clip(np.interp(raw,x,y,left=y[0],right=y[-1]),0,1)),"isotonic_calibrated"

def model_domain_scores(bundle,frame):
    domain=bundle["domain_model"]
    raw=domain["model"].predict_proba(np.nan_to_num(transform(domain["encoder"],frame)))
    result=np.zeros((len(frame),len(ARMS)))
    for j,name in enumerate(domain["classes"]):
        if name in ARMS:result[:,ARMS.index(name)]=raw[:,j]
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("model"); p.add_argument("input_csv"); p.add_argument("output_csv")
    a=p.parse_args(); bundle=pickle.load(open(a.model,"rb")); frame=pd.read_csv(a.input_csv)
    missing_required=[name for name in REQUIRED if name not in frame]
    if missing_required:raise SystemExit("missing required column(s): "+", ".join(missing_required))
    if not set(frame["scenario_arm"]).issubset(ARMS): raise SystemExit("scenario_arm must be sleeve, rygb, or incretin")
    if not set(map(int,frame["target_month"])).issubset(set(bundle["trained_horizons"])): raise SystemExit("target_month was not trained")
    bmi=pd.to_numeric(frame["baseline_value"],errors="coerce")
    if bmi.isna().any() or not bmi.between(10,100).all(): raise SystemExit("baseline_value must be BMI in kg/m2 within [10,100]")
    model=bundle["models"][bundle["selected_candidate"]]
    fraction=missing_fraction(encoder_for(model),frame);threshold=float(bundle.get("high_missingness_threshold",.5))
    high=fraction>=threshold
    if high.any():warnings.warn(f"High model-input missingness for {int(high.sum())} row(s); inspect output warning columns.",RuntimeWarning)
    if not bundle.get("selection_gate_passed",False):warnings.warn("Shared-model noninferiority gate FAILED; outputs are exploratory.",RuntimeWarning)
    matrix=apply_corrections(bundle,frame,predict(model,frame))
    for j,name in enumerate(bundle["quantile_columns"]): frame[name]=matrix[:,j]
    raw_probability=np.asarray([cdf_below(row,35.0,bundle["quantiles"]) for row in matrix])
    calibrated=[calibrated_probability(bundle,raw,str(arm),int(month)) for raw,arm,month in zip(raw_probability,frame["scenario_arm"],frame["target_month"])]
    frame["raw_probability_bmi_below_35"]=raw_probability
    frame["probability_bmi_below_35"]=[value for value,_ in calibrated]
    frame["probability_calibration_status"]=[status for _,status in calibrated]
    scores=model_domain_scores(bundle,frame)
    for j,arm in enumerate(ARMS):frame[f"model_domain_score_{arm}"]=scores[:,j]
    frame["model_domain_score"]=[scores[i,ARMS.index(str(arm))] for i,arm in enumerate(frame["scenario_arm"])]
    frame["model_domain_status"]=np.where(frame["model_domain_score"]>=float(bundle.get("domain_floor",.05)),"within_descriptive_domain","limited_descriptive_domain")
    frame["input_missing_fraction"]=fraction
    frame["high_missingness_warning"]=high
    frame["shared_model_gate_passed"]=bool(bundle.get("selection_gate_passed",False))
    frame["shared_model_gate_warning"]="" if bundle.get("selection_gate_passed",False) else "FAILED: exploratory output only"
    frame.to_csv(a.output_csv,index=False)
if __name__=="__main__": main()
'''
    path.write_text(code, encoding="utf-8")


def package_model(study: Any, context: Any, deployment: Mapping[str, Any]) -> Path:
    """Create a portable research package without patient-level or sub-11 validation records."""
    deploy_dir = context.internal / "shared_model"
    selected_model = deployment["models"][deployment["selected_candidate"]]
    encoder = model_encoder(selected_model)
    units = {
        "baseline_value": "kg/m2", "baseline_cross_outcome": "% HbA1c",
        "age_at_index": "years", "target_month": "months",
        "sbp_baseline": "mmHg", "dbp_baseline": "mmHg",
    }
    study.atomic_json(deploy_dir / "feature_schema.json", {
        "required": ["baseline_value", "scenario_arm", "target_month"],
        "numeric": encoder["numeric"], "categorical": encoder["categorical"],
        "units": units, "allowed_arms": list(ARMS),
        "trained_horizons": deployment["trained_horizons"],
        "high_missingness_threshold": deployment["high_missingness_threshold"],
        "missing_handling": "paired missing indicators; HGB routes designated informative NaNs natively, while GBR/CatBoost components replace them with zero after frozen scaling",
        "bmi35_probability": "piecewise-linear predictive CDF with explicit tails, then arm/horizon isotonic calibration when calibration support is adequate",
        "out_of_range": {"baseline_value": [10, 100]},
    })
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scikit-learn", "catboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    study.atomic_json(deploy_dir / "environment.json", versions)
    gate = "PASS" if deployment["selection_gate_passed"] else "FAILED - EXPLORATORY ONLY"
    model_card = f"""# Experimental shared three-arm BMI model

Selected shared candidate: `{deployment['selected_candidate']}`.
Shared-model noninferiority gate: **{gate}**.
Trained and gated horizons: {', '.join(map(str, deployment['trained_horizons']))} months.

Intended use: factual architecture evaluation and theoretical cross-arm scenario projection.

The inference output returns model-domain scores, a high-input-missingness warning, and the gate status. Do not suppress those warnings.

Not intended for individual treatment selection, causal counterfactual claims, or digital-twin use. The incretin cohort represents sustained recorded treatment and is not a common-eligibility incident-treatment population. Switched-arm intervals use receiving-arm corrections, but transported coverage is not established. BMI <35 probabilities are quantile-CDF-derived and calibration-adjusted; they are not a separately fitted causal response probability.
"""
    (deploy_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    write_inference_script(deploy_dir / "inference.py")

    # The internal deployment retains full validation diagnostics. The portable pickle omits those
    # records and strips calibration denominators; only information needed for inference remains.
    portable = dict(deployment)
    portable.pop("validation_by_arm_horizon", None)
    portable.pop("selection_summary", None)
    portable.pop("probability_calibration_diagnostics", None)
    portable["corrections"] = [
        {key: item.get(key) for key in (
            "candidate", "cohort", "stratum", "outcome", "origin_month",
            "target_month", "coverage", "correction", "status")}
        for item in deployment.get("corrections", [])
    ]
    study.atomic_pickle(deploy_dir / "research_model.pkl", portable)

    validation = study.pd.DataFrame(deployment.get("validation_by_arm_horizon", []))
    if not validation.empty and "n" in validation:
        validation = suppress_public_metrics(study, validation)
    validation.to_csv(deploy_dir / "validation_public.csv", index=False)
    study.pd.DataFrame(deployment.get("selection_summary", [])).to_csv(
        deploy_dir / "selection_summary_public.csv", index=False)
    probability_calibration = study.pd.DataFrame(
        deployment.get("probability_calibration_diagnostics", []))
    if not probability_calibration.empty:
        structural = probability_calibration[["arm", "target_month", "status"]].copy()
        probability_calibration = study.suppress_small_cells(
            probability_calibration, ["n", "events", "nonevents"])
        probability_calibration[["arm", "target_month", "status"]] = structural
    probability_calibration.to_csv(
        deploy_dir / "bmi35_probability_calibration_public.csv", index=False)

    destination = context.run_dir / "experimental_shared_model_bundle.zip"
    names = [
        "research_model.pkl", "manifest.json", "feature_schema.json", "environment.json",
        "MODEL_CARD.md", "inference.py", "validation_public.csv", "selection_summary_public.csv",
        "bmi35_probability_calibration_public.csv", "augmented_canonical_selection.csv",
        "original_roster_selection.csv",
    ]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = deploy_dir / name
            if path.is_file():
                archive.write(path, name)
    return destination


def horizon_scoped_primary_leaderboard(study: Any, predictions: Any, leaderboard: Any,
                                       scale_map: Mapping[tuple[str, str, int], float],
                                       horizons: Sequence[int]) -> Any:
    """Compare all origin-0 BMI candidates on the same prespecified short horizons."""
    scoped_tasks = {("surgery", "bmi", 0), ("incretin", "bmi", 0)}
    keep = study.pd.Series(True, index=leaderboard.index)
    for cohort, outcome, origin in scoped_tasks:
        keep &= ~(
            leaderboard["cohort"].astype(str).eq(cohort)
            & leaderboard["outcome"].astype(str).eq(outcome)
            & study.pd.to_numeric(leaderboard["origin_month"], errors="coerce").eq(origin)
        )
    pieces = [leaderboard.loc[keep].copy()]
    for key in sorted(scoped_tasks):
        if key not in predictions.keys():
            continue
        frame = predictions.read(key)
        frame = frame.loc[
            study.pd.to_numeric(frame["target_month"], errors="coerce").isin(list(horizons))
        ]
        if not frame.empty:
            pieces.append(study.candidate_validation_scores(frame, scale_map))
    return study.concat_frames(pieces)


def worker_calibration(args: argparse.Namespace, study: Any) -> int:
    cfg = run_config(study, args)
    context = study.load_run_context(cfg)
    model_payload = study.require_checkpoint(context, "models_and_predictions")
    derived = study.require_checkpoint(context, "weights_derived")
    extend_rosters(study)
    calibrated = context.new_prediction_store("calibrated")
    full_leaderboard, corrections, endpoint = study.calibrate_prediction_store(
        model_payload["predictions"], calibrated, derived["scale_map"])

    # Canonical selection remains task-specific, but shared candidates deliberately compete for
    # origin-0 BMI after every candidate is rescored on the same 3/6/12-month horizon set. The
    # separate shared-model gate is benchmarked against an untouched original-roster selection.
    deployment_path = context.internal / "shared_model" / "deployment.pkl"
    deployment = pickle.load(open(deployment_path, "rb"))
    shared_names = set(map(str, deployment["models"]))
    horizons = parse_horizons(args.forecast_horizons, deployment["trained_horizons"])
    leaderboard = horizon_scoped_primary_leaderboard(
        study, model_payload["predictions"], full_leaderboard, derived["scale_map"], horizons)
    canonical_selected = study.select_models(leaderboard)
    original_leaderboard = leaderboard.loc[~leaderboard["candidate"].astype(str).isin(shared_names)].copy()
    original_selected = study.select_models(original_leaderboard)
    original_map = {(str(row.cohort), str(row.outcome), int(row.origin_month)): str(row.selected_candidate)
                    for row in original_selected.itertuples()}
    shared = study.concat_frames([
        factual_metrics(study, calibrated, None, candidate, "validation", horizons)
        for candidate in deployment["models"]
    ])
    reference = factual_metrics(study, calibrated, original_map, None, "validation", horizons)
    winner, validation_detail, selection_summary, gate_passed = select_shared_candidate(
        study, shared, reference, list(deployment["models"]), horizons,
        args.noninferiority_margin, args.coverage_tolerance)
    probability_calibrators, probability_calibration = fit_bmi35_probability_calibrators(
        study, calibrated, winner, horizons)
    if not gate_passed:
        warnings.warn(
            "Shared-model noninferiority gate FAILED; the selected shared candidate and all "
            "scenario projections remain exploratory.", RuntimeWarning, stacklevel=2)

    deploy_dir = context.internal / "shared_model"
    validation_detail.to_csv(deploy_dir / "validation_by_arm_horizon.csv", index=False)
    selection_summary.to_csv(deploy_dir / "selection_summary.csv", index=False)
    original_selected.to_csv(deploy_dir / "original_roster_selection.csv", index=False)
    canonical_selected.to_csv(deploy_dir / "augmented_canonical_selection.csv", index=False)
    full_leaderboard.to_csv(deploy_dir / "source_full_horizon_leaderboard_audit.csv", index=False)
    probability_calibration.to_csv(deploy_dir / "bmi35_probability_calibration.csv", index=False)
    corrections_records = []
    for row in corrections.to_dict("records"):
        corrections_records.append({key: (value.item() if hasattr(value, "item") else value)
                                    for key, value in row.items()})
    deployment.update({
        "selected_candidate": winner, "selection_gate_passed": gate_passed,
        "forecast_horizons": list(horizons), "validation_by_arm_horizon": validation_detail.to_dict("records"),
        "selection_summary": selection_summary.to_dict("records"),
        "original_roster_selection": original_selected.to_dict("records"),
        "probability_calibrators": probability_calibrators,
        "probability_calibration_diagnostics": probability_calibration.to_dict("records"),
        "corrections": corrections_records, "quantile_columns": list(study.QUANTILE_COLUMNS),
    })
    study.atomic_pickle(deployment_path, deployment)
    manifest_path = deploy_dir / "manifest.json"
    manifest = study.read_json(manifest_path, {}) or {}
    manifest.update({
        "selected_candidate": winner, "selection_gate_passed": gate_passed,
        "selection_rule": "factual arm-by-horizon CRPS noninferiority and 80/90 coverage gates",
        "noninferiority_margin": args.noninferiority_margin,
        "coverage_tolerance": args.coverage_tolerance,
        "trained_horizons": list(horizons),
        "high_missingness_threshold": args.high_missingness_threshold,
        "bmi35_probability_method": "predictive CDF with explicit tails and supported arm/horizon isotonic calibration",
        "canonical_selection": "augmented candidate competition; origin-0 BMI ranked on forecast_horizons only",
        "canonical_selection_horizons": list(horizons),
        "later_horizon_rule": "shared candidates emit no rows outside trained_horizons; if selected, later selected-model cells are intentionally unavailable while original-candidate rows remain retained",
        "shared_gate_comparator": "original roster selected per source task on the same forecast horizons",
    })
    study.atomic_json(manifest_path, manifest)
    package_model(study, context, deployment)

    payload = {
        "calibrated": calibrated, "calibration": corrections, "clinical_endpoint": endpoint,
        "leaderboard": leaderboard, "selected": canonical_selected,
        "model_status": model_payload["status"], "model_details": model_payload["details"],
    }
    context.save_checkpoint("calibration", payload,
                            {"models": study.checkpoint_hash(context, "models_and_predictions")})
    return 0


def correction_for(deployment: Mapping[str, Any], arm: str, month: int) -> tuple[dict[tuple[int, int], float], str]:
    cohort, stratum = (("incretin", "incretin") if arm == "incretin" else ("surgery", arm))
    found, statuses = {}, []
    for item in deployment.get("corrections", []):
        if (str(item.get("candidate")) == deployment["selected_candidate"]
                and str(item.get("cohort")) == cohort and str(item.get("stratum")) == stratum
                and str(item.get("outcome")) == "bmi" and int(item.get("origin_month", -1)) == 0
                and int(item.get("target_month", -1)) == int(month)):
            pair = {.90: (0, 6), .80: (1, 5), .50: (2, 4)}.get(round(float(item["coverage"]), 2))
            if pair:
                found[pair] = float(item["correction"])
                statuses.append(str(item.get("status", "")))
    return found, ("calibrated" if statuses and all(status == "calibrated" for status in statuses)
                   else "insufficient_factual_calibration_support")


def apply_receiving_arm_correction(study: Any, deployment: Mapping[str, Any], matrix: Any,
                                   arm: str, horizons: Sequence[int]) -> tuple[Any, list[str]]:
    result = study.np.asarray(matrix, dtype=float).copy()
    statuses = []
    for index, month in enumerate(horizons):
        corrections, status = correction_for(deployment, arm, int(month))
        statuses.append(status)
        for (lower, upper), value in corrections.items():
            result[index, lower] -= value
            result[index, upper] += value
    return study.rearrange_quantiles(result), statuses


def probability_below(quantiles: Sequence[float], threshold: float, levels: Sequence[float]) -> float:
    """Evaluate the piecewise-linear predictive CDF, including explicit 0/1 tails."""
    import numpy as np
    ladder = np.maximum.accumulate(np.asarray(quantiles, dtype=float))
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


def probabilities_below(matrix: Any, threshold: float, levels: Sequence[float]) -> Any:
    return __import__("numpy").asarray(
        [probability_below(row, threshold, levels) for row in matrix], dtype=float)


def probability_calibrator_key(arm: str, month: int) -> str:
    return f"{arm}|{int(month)}"


def apply_probability_calibrator(deployment: Mapping[str, Any], raw: Any,
                                 arm: str, month: int) -> tuple[Any, str]:
    np = __import__("numpy")
    values = np.asarray(raw, dtype=float)
    calibration = deployment.get("probability_calibrators", {}).get(
        probability_calibrator_key(arm, month), {})
    if calibration.get("status") != "isotonic_calibrated":
        return np.clip(values, 0.0, 1.0), str(
            calibration.get("status", "raw_predictive_cdf_no_calibrator"))
    x = np.asarray(calibration["x_thresholds"], dtype=float)
    y = np.asarray(calibration["y_thresholds"], dtype=float)
    return np.clip(np.interp(values, x, y, left=y[0], right=y[-1]), 0.0, 1.0), "isotonic_calibrated"


def fit_bmi35_probability_calibrators(study: Any, store: Any, candidate: str,
                                      horizons: Sequence[int]) -> tuple[dict[str, Any], Any]:
    """Fit arm/horizon isotonic maps on the protected calibration split only."""
    from sklearn.isotonic import IsotonicRegression

    calibrators, rows = {}, []
    for key in (("surgery", "bmi", 0), ("incretin", "bmi", 0)):
        if key not in store.keys():
            continue
        frame = store.read(key)
        frame = frame.loc[
            frame["candidate"].astype(str).eq(str(candidate))
            & frame["split"].astype(str).eq("calibration")
            & frame["target_observed"].fillna(False).astype(bool)
            & study.pd.to_numeric(frame["target_month"], errors="coerce").isin(list(horizons))
        ].copy()
        frame["arm"] = arm_series(frame)
        for (arm, month), cell in frame.groupby(["arm", "target_month"], sort=True):
            cell = cell.drop_duplicates("patient_id")
            event = (study.pd.to_numeric(cell["target_value"], errors="coerce") < BMI35_THRESHOLD).to_numpy(int)
            weight = study.pd.to_numeric(cell["analysis_weight"], errors="coerce").fillna(1.0).to_numpy(float)
            raw = probabilities_below(
                cell[list(study.QUANTILE_COLUMNS)].to_numpy(float), BMI35_THRESHOLD, study.QUANTILES)
            events, nonevents = int(event.sum()), int((1 - event).sum())
            status = "insufficient_event_or_nonevent_calibration_support"
            calibrated = raw
            record: dict[str, Any] = {"status": status}
            if events >= study.MIN_CELL_SIZE and nonevents >= study.MIN_CELL_SIZE:
                fitted = IsotonicRegression(out_of_bounds="clip").fit(raw, event, sample_weight=weight)
                record = {
                    "status": "isotonic_calibrated",
                    "x_thresholds": [float(value) for value in fitted.X_thresholds_],
                    "y_thresholds": [float(value) for value in fitted.y_thresholds_],
                }
                calibrated = fitted.predict(raw)
                status = "isotonic_calibrated"
            calibrators[probability_calibrator_key(str(arm), int(month))] = record
            clipped = study.np.clip(calibrated, 1e-6, 1.0 - 1e-6)
            rows.append({
                "arm": str(arm), "target_month": int(month), "n": int(len(cell)),
                "events": events, "nonevents": nonevents, "status": status,
                "raw_auroc": study.weighted_auroc(event, raw, weight),
                "calibrated_auroc": study.weighted_auroc(event, calibrated, weight),
                "brier": study.weighted_mean((calibrated - event) ** 2, weight),
                "log_loss": study.weighted_mean(
                    -(event * study.np.log(clipped) + (1 - event) * study.np.log1p(-clipped)), weight),
            })
    return calibrators, study.pd.DataFrame(rows)


def heldout_profiles(study: Any, weighted_rows: Any, selection_month: int) -> Any:
    pieces = []
    for key in (("surgery", "bmi", 0), ("incretin", "bmi", 0)):
        if key not in weighted_rows.keys():
            continue
        frame = weighted_rows.read(key)
        frame = frame.loc[
            frame["split"].astype(str).isin(HELD_OUT)
            & study.pd.to_numeric(frame["target_month"], errors="coerce").eq(selection_month)
        ].copy()
        frame["observed_arm"] = arm_series(frame)
        frame = frame.loc[frame["observed_arm"].isin(ARMS)]
        pieces.append(frame)
    if not pieces:
        raise RuntimeError("No held-out origin-0 BMI profiles were available")
    return study.pd.concat(pieces, ignore_index=True).drop_duplicates(["patient_id", "cohort"]).reset_index(drop=True)


def stable_token(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def select_bmi_matched_profiles(study: Any, profiles: Any, count: int, caliper: float,
                                seed: int) -> tuple[Any, dict[str, float]]:
    medians = {arm: float(study.pd.to_numeric(
        profiles.loc[profiles["observed_arm"].eq(arm), "baseline_value"], errors="coerce").median())
        for arm in ARMS}
    if not all(math.isfinite(value) for value in medians.values()):
        raise RuntimeError(f"Could not compute all three held-out factual baseline-BMI medians: {medians}")
    selected = []
    for arm in ARMS:
        other = [name for name in ARMS if name != arm]
        target = float((medians[other[0]] + medians[other[1]]) / 2.0)
        group = profiles.loc[profiles["observed_arm"].eq(arm)].copy()
        group["baseline_bmi"] = study.pd.to_numeric(group["baseline_value"], errors="coerce")
        group = group.loc[group["baseline_bmi"].notna()]
        group["bmi_match_target"] = target
        group["bmi_match_distance"] = (group["baseline_bmi"] - target).abs()
        group["within_bmi_caliper"] = group["bmi_match_distance"].le(caliper) if caliper > 0 else True
        group["tie_break"] = [stable_token(str(patient), seed) for patient in group["patient_id"]]
        group = group.sort_values(["bmi_match_distance", "tie_break"], kind="stable").head(count).copy()
        group["reference_arm_1"], group["reference_arm_2"] = other
        group["reference_median_1"], group["reference_median_2"] = medians[other[0]], medians[other[1]]
        group["profile_label"] = [f"{arm.upper()}-{chr(65 + index)}" for index in range(len(group))]
        selected.append(group)
    return study.pd.concat(selected, ignore_index=True), medians


def suppress_public_metrics(study: Any, frame: Any) -> Any:
    if frame.empty:
        return frame
    counts = [name for name in ("n", "reference_n") if name in frame]
    result = study.suppress_small_cells(frame, counts) if counts else frame.copy()
    for structural in ("target_month", "arm", "candidate", "reference_candidate", "split"):
        if structural in frame:
            result[structural] = frame[structural].to_numpy()
    return result


def bootstrap_auroc(study: Any, event: Any, score: Any, baseline_score: Any,
                    weight: Any, patient_ids: Any, seed: int,
                    replicates: int) -> tuple[float, float, float, float]:
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
        baseline = study.weighted_auroc(
            current, study.np.asarray(baseline_score)[index], current_weight)
        estimates.append(estimate); differences.append(estimate - baseline)
    finite = study.np.asarray(estimates, dtype=float)
    finite = finite[study.np.isfinite(finite)]
    difference = study.np.asarray(differences, dtype=float)
    difference = difference[study.np.isfinite(difference)]
    if not len(finite):
        return math.nan, math.nan, math.nan, math.nan
    return (
        float(study.np.quantile(finite, .025)), float(study.np.quantile(finite, .975)),
        float(study.np.quantile(difference, .025)) if len(difference) else math.nan,
        float(study.np.quantile(difference, .975)) if len(difference) else math.nan,
    )


def bmi35_heldout_diagnostics(study: Any, store: Any, deployment: Mapping[str, Any],
                              horizons: Sequence[int], seed: int,
                              smoke: bool) -> tuple[Any, Any]:
    """Held-out factual BMI<35 probabilities, metrics, and disclosure-smoothed ROC points."""
    from sklearn.metrics import roc_curve

    pieces = []
    candidate = str(deployment["selected_candidate"])
    for key in (("surgery", "bmi", 0), ("incretin", "bmi", 0)):
        if key not in store.keys():
            continue
        frame = store.read(key)
        frame = frame.loc[
            frame["candidate"].astype(str).eq(candidate)
            & frame["split"].astype(str).isin(HELD_OUT)
            & frame["target_observed"].fillna(False).astype(bool)
            & study.pd.to_numeric(frame["target_month"], errors="coerce").isin(list(horizons))
        ].copy()
        frame["arm"] = arm_series(frame)
        frame["patient_key"] = key[0] + "|" + frame["patient_id"].astype(str)
        pieces.append(frame.loc[frame["arm"].isin(ARMS)])
    heldout = study.concat_frames(pieces)
    metrics, curves = [], []
    scopes = [*HELD_OUT, "pooled_heldout"]
    if heldout.empty:
        for scope in scopes:
            for arm in ARMS:
                for month in horizons:
                    metrics.append({
                        "evaluation_scope": scope, "arm": arm, "target_month": int(month),
                        "n": 0, "events": 0, "nonevents": 0, "effective_sample_size": math.nan,
                        "event_rate": math.nan, "raw_auroc": math.nan, "auroc": math.nan,
                        "auroc_ci_low": math.nan, "auroc_ci_high": math.nan, "brier": math.nan,
                        "baseline_bmi_auroc": math.nan, "auroc_difference_vs_baseline_bmi": math.nan,
                        "difference_ci_low": math.nan, "difference_ci_high": math.nan,
                        "log_loss": math.nan,
                        "probability_calibration_status": "unavailable_no_heldout_rows",
                        "status": "unavailable_no_heldout_rows",
                    })
        return study.pd.DataFrame(metrics), study.pd.DataFrame(
            columns=["arm", "target_month", "score_type", "fpr", "tpr", "evaluation_scope"])
    for scope in scopes:
        scoped = heldout if scope == "pooled_heldout" else heldout.loc[heldout["split"].astype(str).eq(scope)]
        for arm in ARMS:
            for month in horizons:
                cell = scoped.loc[
                    scoped["arm"].astype(str).eq(arm)
                    & study.pd.to_numeric(scoped["target_month"], errors="coerce").eq(int(month))
                ].drop_duplicates(["patient_key", "target_month"]).copy()
                event = (study.pd.to_numeric(cell.get("target_value"), errors="coerce") < BMI35_THRESHOLD).to_numpy(int)
                weight = study.pd.to_numeric(cell.get("analysis_weight"), errors="coerce").fillna(1.0).to_numpy(float)
                events, nonevents = int(event.sum()), int((1 - event).sum())
                status = "estimable"
                raw = final = study.np.asarray([], dtype=float)
                probability_status = "unavailable_no_heldout_rows"
                auc = raw_auc = baseline_auc = difference = low = high = difference_low = difference_high = math.nan
                brier = log_loss = event_rate = math.nan
                baseline_score = study.np.asarray([], dtype=float)
                if len(cell):
                    raw = probabilities_below(
                        cell[list(study.QUANTILE_COLUMNS)].to_numpy(float),
                        BMI35_THRESHOLD, study.QUANTILES)
                    final, probability_status = apply_probability_calibrator(
                        deployment, raw, arm, int(month))
                    event_rate = study.weighted_mean(event, weight)
                    brier = study.weighted_mean((final - event) ** 2, weight)
                    clipped = study.np.clip(final, 1e-6, 1.0 - 1e-6)
                    log_loss = study.weighted_mean(
                        -(event * study.np.log(clipped) + (1 - event) * study.np.log1p(-clipped)), weight)
                    baseline_score = -study.pd.to_numeric(
                        cell["prediction_reference_value"], errors="coerce").to_numpy(float)
                if events >= study.MIN_CELL_SIZE and nonevents >= study.MIN_CELL_SIZE:
                    raw_auc = study.weighted_auroc(event, raw, weight)
                    auc = study.weighted_auroc(event, final, weight)
                    baseline_auc = study.weighted_auroc(event, baseline_score, weight)
                    difference = auc - baseline_auc
                    token = int(hashlib.sha256(f"{arm}|{month}|{scope}".encode()).hexdigest()[:8], 16)
                    low, high, difference_low, difference_high = bootstrap_auroc(
                        study, event, final, baseline_score, weight, cell["patient_key"], seed + token,
                        80 if smoke else 400)
                    if scope == "pooled_heldout":
                        grid = study.np.linspace(0.0, 1.0, 101)
                        for score_type, score_values in (
                            ("shared_model_probability", final), ("baseline_bmi_only", baseline_score)):
                            fpr, tpr, _ = roc_curve(
                                event, score_values, sample_weight=weight, drop_intermediate=False)
                            interpolated = study.np.maximum.accumulate(study.np.interp(grid, fpr, tpr))
                            curves.extend({
                                "arm": arm, "target_month": int(month), "score_type": score_type,
                                "fpr": float(x), "tpr": float(y), "evaluation_scope": scope,
                            } for x, y in zip(grid, interpolated, strict=True))
                elif not len(cell):
                    status = "unavailable_no_heldout_rows"
                else:
                    status = "unavailable_event_or_nonevent_support"
                metrics.append({
                    "evaluation_scope": scope, "arm": arm, "target_month": int(month),
                    "n": int(len(cell)), "events": events, "nonevents": nonevents,
                    "effective_sample_size": (study.effective_sample_size(weight) if len(weight) else math.nan),
                    "event_rate": event_rate, "raw_auroc": raw_auc, "auroc": auc,
                    "auroc_ci_low": low, "auroc_ci_high": high, "brier": brier,
                    "baseline_bmi_auroc": baseline_auc,
                    "auroc_difference_vs_baseline_bmi": difference,
                    "difference_ci_low": difference_low, "difference_ci_high": difference_high,
                    "log_loss": log_loss, "probability_calibration_status": probability_status,
                    "status": status,
                })
    return study.pd.DataFrame(metrics), study.pd.DataFrame(curves)


def suppress_bmi35_metrics(study: Any, frame: Any) -> Any:
    if frame.empty:
        return frame
    structural_columns = ["evaluation_scope", "arm", "target_month",
                          "probability_calibration_status", "status"]
    structural = frame[structural_columns].copy()
    result = study.suppress_small_cells(frame, ["n", "events", "nonevents"])
    result[structural_columns] = structural
    return result


def render_public_architecture_book(study: Any, out: Path, validation: Any, summary: Any,
                                    heldout: Any, matching_reference: Any,
                                    bmi35_metrics: Any, bmi35_curves: Any,
                                    winner: str, gate_passed: bool) -> list[Path]:
    from matplotlib.backends.backend_pdf import PdfPages
    out.mkdir(parents=True, exist_ok=True)
    pages = []
    study.configure_figure_style()

    figure = study.plt.figure(figsize=(11, 8.5))
    figure.suptitle("Experimental shared three-arm architecture", fontsize=17, fontweight="bold")
    left, right = figure.add_subplot(121), figure.add_subplot(122)
    ranked = summary.sort_values(["all_cells_pass", "mean_crps_ratio"], ascending=[False, True])
    left.barh(ranked["candidate"], ranked["mean_crps_ratio"])
    left.axvline(1.0, ls="--", lw=1)
    left.set_xlabel("Mean factual CRPS ratio vs original-roster selected model")
    left.set_title(f"Shared winner: {winner}\nAll-cell gate: {'PASS' if gate_passed else 'NOT PASSED'}")
    right.axis("off")
    table = ranked[["candidate", "passed_cells", "expected_cells", "worst_crps_ratio", "all_cells_pass"]].copy()
    right.table(cellText=table.round(3).astype(str).values, colLabels=list(table.columns), loc="center")
    figure.text(.05, .035, "Factual validation only. The global shared-model gate is separate from augmented canonical task-specific selection.")
    pages.append(("00_shared_architecture_selection.png", figure))

    figure = study.plt.figure(figsize=(11, 8.5))
    figure.suptitle("Factual performance by observed arm and horizon", fontsize=17, fontweight="bold")
    axis = figure.add_subplot(111); axis.axis("off")
    display = heldout[["arm", "target_month", "n", "ess", "crps", "rmse", "mae", "bias", "coverage_80", "coverage_90"]]
    axis.table(cellText=display.round(3).astype(str).values, colLabels=list(display.columns), loc="upper center")
    figure.text(.05, .035, "Held-out factual rows at 3, 6, and 12 months; switched projections are not evaluated here.")
    pages.append(("01_factual_performance_3_6_12.png", figure))

    figure = study.plt.figure(figsize=(11, 8.5))
    figure.suptitle("Interpretation boundaries", fontsize=17, fontweight="bold")
    axis = figure.add_subplot(111); axis.axis("off")
    text = (
        "Purpose: test whether one treatment-conditioned quantile architecture preserves factual prediction "
        "quality across observed RYGB, sleeve, and sustained-incretin cohorts.\n\n"
        "Scenario figures hold each selected profile's baseline features fixed and change only the model's arm "
        "input. They are theoretical cross-arm prognostic projections, not identified treatment effects.\n\n"
        "Patient selection is BMI-only. For each factual arm, five profiles are chosen closest to the equal-weight "
        "midpoint of the other two arms' held-out median baseline BMIs.\n\n"
        "The incretin cohort is intentionally restricted to patients with completed recorded treatment. The early "
        "projections therefore describe sustained-treatment continuers and retain the study's future-persistence limitation."
    )
    axis.text(.06, .9, text, va="top", fontsize=11, linespacing=1.5, wrap=True)
    pages.append(("02_scope_and_limitations.png", figure))

    figure = study.plt.figure(figsize=(11, 8.5))
    figure.suptitle("Symmetric BMI-only matching reference", fontsize=17, fontweight="bold")
    axis = figure.add_subplot(111); axis.axis("off")
    axis.table(cellText=matching_reference.round(3).astype(str).values,
               colLabels=list(matching_reference.columns), loc="upper center")
    figure.text(.05, .08,
                "For each factual arm, the matching target is the equal-weight midpoint of the other two arms' held-out median baseline BMIs. Other covariates are deliberately not matched.")
    pages.append(("03_bmi_matching_reference.png", figure))

    figure, axes = study.plt.subplots(1, 3, figsize=(11, 8.5), squeeze=False)
    figure.suptitle("Held-out factual discrimination for attaining BMI <35", fontsize=17, fontweight="bold")
    colors = {"sleeve": study.PALETTE["blue"], "rygb": study.PALETTE["green"],
              "incretin": study.PALETTE["orange"]}
    pooled = bmi35_metrics.loc[bmi35_metrics["evaluation_scope"].eq("pooled_heldout")] if len(bmi35_metrics) else bmi35_metrics
    for axis, month in zip(axes[0], (3, 6, 12), strict=True):
        plotted = False
        for arm in ARMS:
            metric = pooled.loc[
                pooled["arm"].eq(arm)
                & study.pd.to_numeric(pooled["target_month"], errors="coerce").eq(month)
            ] if len(pooled) else pooled
            auc_value = (study.pd.to_numeric(metric.iloc[0]["auroc"], errors="coerce")
                         if not metric.empty else math.nan)
            baseline_auc = (study.pd.to_numeric(metric.iloc[0]["baseline_bmi_auroc"], errors="coerce")
                            if not metric.empty else math.nan)
            if metric.empty or not math.isfinite(float(auc_value)):
                continue
            for score_type, linestyle, width, label in (
                ("shared_model_probability", "-", 2.0, f"{arm} model {float(auc_value):.3f}"),
                ("baseline_bmi_only", "--", 1.1, f"{arm} BMI-only {float(baseline_auc):.3f}"),
            ):
                curve = bmi35_curves.loc[
                    bmi35_curves["arm"].eq(arm)
                    & study.pd.to_numeric(bmi35_curves["target_month"], errors="coerce").eq(month)
                    & bmi35_curves["score_type"].eq(score_type)
                ] if len(bmi35_curves) else bmi35_curves
                if not curve.empty:
                    axis.plot(curve["fpr"], curve["tpr"], color=colors[arm], ls=linestyle,
                              lw=width, label=label)
                    plotted = True
        axis.plot([0, 1], [0, 1], ls="--", lw=.8, color=study.PALETTE["muted"])
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate",
                 ylabel="True-positive rate", title=f"{month} months")
        if plotted:
            axis.legend(frameon=False, fontsize=5.8, loc="lower right")
        else:
            axis.text(.5, .5, "Not estimable", ha="center", color=study.PALETTE["muted"])
    figure.text(.05, .055,
                "Selected shared model; factual temporal and geographic held-out rows pooled for curve display. "
                "Probabilities come from the calibrated predictive CDF and arm/horizon isotonic calibration.\n"
                "Solid curves are the shared model and dashed curves are baseline BMI alone. "
                "Split-specific metrics are retained in the accompanying CSV. "
                f"Shared-model gate={'PASS' if gate_passed else 'FAILED — EXPLORATORY ONLY'}.", fontsize=8)
    figure.tight_layout(rect=[0, .09, 1, .93])
    pages.append(("04_bmi35_heldout_roc_curves.png", figure))

    figure = study.plt.figure(figsize=(11, 8.5))
    figure.suptitle("Held-out BMI <35 probability performance", fontsize=17, fontweight="bold")
    axis = figure.add_subplot(111); axis.axis("off")
    columns = ["arm", "target_month", "n", "events", "nonevents", "auroc",
               "baseline_bmi_auroc", "auroc_difference_vs_baseline_bmi",
               "difference_ci_low", "difference_ci_high", "brier", "status"]
    table = pooled[columns].copy() if len(pooled) else pooled
    if len(table):
        table["status"] = table["status"].replace({
            "estimable": "Estimable",
            "unavailable_event_or_nonevent_support": "Low event support",
            "unavailable_no_heldout_rows": "No held-out rows",
        })
        labels = ["Arm", "Month", "N", "Events", "Non-events", "Model AUC",
                  "BMI-only AUC", "ΔAUC", "Δ CI low", "Δ CI high", "Brier", "Status"]
        artist = axis.table(cellText=table.round(3).astype(str).values,
                            colLabels=labels, loc="upper center")
        artist.auto_set_font_size(False); artist.set_fontsize(5.5); artist.scale(1.0, 1.18)
    else:
        axis.text(.5, .5, "No held-out BMI <35 endpoint rows were available.", ha="center")
    figure.text(.05, .055,
                "AUROC is discrimination, not clinical utility or a treatment effect. Counts and metrics are "
                "suppressed when disclosure or event/non-event support is inadequate.\n"
                f"Shared-model gate={'PASS' if gate_passed else 'FAILED — EXPLORATORY ONLY'}.", fontsize=8)
    pages.append(("05_bmi35_probability_performance.png", figure))

    pdf = out / "shared_architecture_figure_book.pdf"
    with PdfPages(pdf, metadata={"Title": "Experimental shared three-arm architecture"}) as writer:
        for name, figure in pages:
            figure.savefig(out / name, dpi=220, facecolor=figure.get_facecolor())
            writer.savefig(figure, dpi=220, facecolor=figure.get_facecolor())
            study.plt.close(figure)
    return [out / name for name, _ in pages] + [pdf]


def render_internal_scenario_book(study: Any, out: Path, selected: Any, scenarios: Any,
                                  medians: Mapping[str, float], gate_passed: bool) -> list[Path]:
    from matplotlib.backends.backend_pdf import PdfPages
    out.mkdir(parents=True, exist_ok=True)
    pages = []
    study.configure_figure_style()

    figure = study.plt.figure(figsize=(11, 8.5))
    figure.suptitle("BMI-only matching audit for theoretical scenario profiles", fontsize=16, fontweight="bold")
    axis = figure.add_subplot(111); axis.axis("off")
    columns = ["profile_label", "observed_arm", "baseline_bmi", "bmi_match_target",
               "bmi_match_distance", "within_bmi_caliper", "reference_arm_1", "reference_arm_2"]
    view = selected[columns].copy()
    axis.table(cellText=view.round(3).astype(str).values, colLabels=columns, loc="upper center")
    figure.text(.05, .06, f"Held-out medians: {', '.join(f'{arm}={medians[arm]:.2f}' for arm in ARMS)}. "
                         "Profiles were selected before viewing outcomes or projections.")
    pages.append(("00_bmi_matching_audit.png", figure))

    figure = study.plt.figure(figsize=(11, 8.5))
    figure.suptitle("Model-domain scores and input-missingness audit", fontsize=16, fontweight="bold")
    axis = figure.add_subplot(111); axis.axis("off")
    columns = ["profile_label", "observed_arm", "baseline_bmi", "domain_sleeve", "domain_rygb",
               "domain_incretin", "minimum_domain_score", "model_input_missing_fraction",
               "high_missingness_warning"]
    view = selected[columns].copy().rename(columns={
        "profile_label": "Profile", "observed_arm": "Observed arm", "baseline_bmi": "BMI₀",
        "domain_sleeve": "SG domain", "domain_rygb": "RYGB domain",
        "domain_incretin": "IBT domain", "minimum_domain_score": "Min domain",
        "model_input_missing_fraction": "Missing frac.",
        "high_missingness_warning": "High missing",
    })
    artist = axis.table(cellText=view.round(3).astype(str).values,
                        colLabels=list(view.columns), loc="upper center")
    artist.auto_set_font_size(False); artist.set_fontsize(6.3)
    figure.text(.05, .06,
                "Model-domain probabilities are descriptive membership diagnostics, not propensity scores or causal overlap proof. "
                f"Shared-model gate={'PASS' if gate_passed else 'FAILED — EXPLORATORY ONLY'}.", fontsize=8)
    pages.append(("01_model_domain_scores_and_missingness.png", figure))

    colors = {"sleeve": study.PALETTE["blue"], "rygb": study.PALETTE["green"],
              "incretin": study.PALETTE["orange"]}
    for page_number, observed_arm in enumerate(ARMS, start=2):
        profiles = selected.loc[selected["observed_arm"].eq(observed_arm)].sort_values("profile_label")
        figure, axes = study.plt.subplots(max(len(profiles), 1), 1, figsize=(11, 8.5), squeeze=False)
        figure.suptitle(
            f"BMI-matched theoretical treatment-scenario projections: factual {observed_arm} profiles",
            fontsize=14, fontweight="bold")
        for axis, profile in zip(axes[:, 0], profiles.itertuples()):
            patient_rows = scenarios.loc[scenarios["profile_label"].eq(profile.profile_label)]
            baseline = float(profile.baseline_bmi)
            for scenario_arm in ARMS:
                rows = patient_rows.loc[patient_rows["scenario_arm"].eq(scenario_arm)].sort_values("target_month")
                x = study.np.r_[0, rows["target_month"].to_numpy(float)]
                y = study.np.r_[baseline, rows["q50"].to_numpy(float)]
                factual = scenario_arm == observed_arm
                axis.plot(x, y, marker="o", ls="-" if factual else "--", lw=2 if factual else 1.3,
                          color=colors[scenario_arm], label=f"{scenario_arm}{' factual' if factual else ' scenario'}")
                low = study.np.r_[baseline, rows["q10"].to_numpy(float)]
                high = study.np.r_[baseline, rows["q90"].to_numpy(float)]
                axis.fill_between(x, low, high, color=colors[scenario_arm], alpha=.18 if factual else .06)
            domain = patient_rows.groupby("scenario_arm")["domain_score"].first().to_dict()
            axis.set_title(
                f"{profile.profile_label}: baseline {baseline:.2f}; target {profile.bmi_match_target:.2f}; "
                f"distance {profile.bmi_match_distance:.2f} | domain scores "
                + ", ".join(f"{arm} {domain.get(arm, math.nan):.2f}" for arm in ARMS),
                loc="left", fontsize=8)
            axis.set_ylabel("BMI")
        axes[-1, 0].set_xlabel("Months from factual index")
        axes[0, 0].legend(frameon=False, ncol=3, fontsize=7)
        figure.text(.05, .02,
                    "Solid band: factual-arm calibrated forecast. Dashed switched bands: receiving-arm-adjusted "
                    "model-based scenario intervals; transported coverage is not established. "
                    f"Shared validation gate={'PASS' if gate_passed else 'NOT PASSED'}.", fontsize=7)
        figure.tight_layout(rect=[0, .04, 1, .94])
        pages.append((f"{page_number:02d}_scenario_profiles_{observed_arm}.png", figure))

    pdf = out / "bmi_matched_scenario_projection_book.pdf"
    with PdfPages(pdf, metadata={"Title": "Internal BMI-matched theoretical scenario projections"}) as writer:
        for name, figure in pages:
            figure.savefig(out / name, dpi=220, facecolor=figure.get_facecolor())
            writer.savefig(figure, dpi=220, facecolor=figure.get_facecolor())
            study.plt.close(figure)
    return [out / name for name, _ in pages] + [pdf]


def worker_scenarios(args: argparse.Namespace, study: Any) -> int:
    cfg = run_config(study, args)
    context = study.load_run_context(cfg)
    deployment = pickle.load(open(context.internal / "shared_model" / "deployment.pkl", "rb"))
    calibration = study.require_checkpoint(context, "calibration")
    weighted_rows = study.require_checkpoint(context, "weights")["rows"]
    horizons = parse_horizons(args.forecast_horizons, deployment["trained_horizons"])
    profiles = heldout_profiles(study, weighted_rows, min(horizons))
    selected, medians = select_bmi_matched_profiles(
        study, profiles, args.patients_per_arm, args.bmi_match_caliper, args.seed)

    scores = domain_scores(study, deployment["domain_model"], selected)
    for index, arm in enumerate(ARMS):
        selected[f"domain_{arm}"] = scores[:, index]
    selected["minimum_domain_score"] = scores.min(axis=1)
    model = deployment["models"][deployment["selected_candidate"]]
    missing_fraction, high_missingness = input_missingness(
        study, model_encoder(model), selected, args.high_missingness_threshold)
    selected["model_input_missing_fraction"] = missing_fraction
    selected["high_missingness_warning"] = high_missingness
    if high_missingness.any():
        warnings.warn(
            f"High shared-model input missingness for {int(high_missingness.sum())} selected profile(s); "
            "inspect the internal model-domain/missingness table.", RuntimeWarning, stacklevel=2)
    if not deployment["selection_gate_passed"]:
        warnings.warn(
            "Shared-model noninferiority gate FAILED; rendered scenarios are exploratory.",
            RuntimeWarning, stacklevel=2)

    raw_rows = []
    for patient in selected.to_dict("records"):
        template = study.pd.DataFrame([patient] * len(horizons))
        template["target_month"] = list(horizons)
        for scenario_arm in ARMS:
            frame = prepare(template, scenario_arm)
            raw_matrix = predict_model(study, model, frame)
            matrix, calibration_statuses = apply_receiving_arm_correction(
                study, deployment, raw_matrix, scenario_arm, horizons)
            factual = scenario_arm == patient["observed_arm"]
            for index, month in enumerate(horizons):
                row = {
                    "patient_id": str(patient["patient_id"]), "profile_label": patient["profile_label"],
                    "observed_arm": str(patient["observed_arm"]), "scenario_arm": scenario_arm,
                    "factual_arm": factual, "target_month": int(month),
                    "baseline_bmi": float(patient["baseline_bmi"]),
                    "bmi_match_target": float(patient["bmi_match_target"]),
                    "bmi_match_distance": float(patient["bmi_match_distance"]),
                    "within_bmi_caliper": bool(patient["within_bmi_caliper"]),
                    "domain_score": float(patient[f"domain_{scenario_arm}"]),
                    "domain_status": ("within_descriptive_domain" if patient[f"domain_{scenario_arm}"] >= args.domain_floor
                                      else "limited_descriptive_domain"),
                    "interval_interpretation": ("factual-arm conformal interval" if factual
                                                else "receiving-arm-adjusted model-based scenario interval; transported coverage not established"),
                    "calibration_status": calibration_statuses[index],
                    "selected_shared_candidate": deployment["selected_candidate"],
                }
                row.update({f"raw_{column}": float(raw_matrix[index, position])
                            for position, column in enumerate(study.QUANTILE_COLUMNS)})
                row.update({column: float(matrix[index, position])
                            for position, column in enumerate(study.QUANTILE_COLUMNS)})
                row["median_change_from_baseline"] = row["q50"] - row["baseline_bmi"]
                raw_probability = probability_below(
                    matrix[index], BMI35_THRESHOLD, study.QUANTILES)
                calibrated_probability, probability_status = apply_probability_calibrator(
                    deployment, [raw_probability], scenario_arm, int(month))
                row["raw_probability_bmi_below_35"] = raw_probability
                row["probability_bmi_below_35"] = float(calibrated_probability[0])
                row["probability_calibration_status"] = probability_status
                row["model_input_missing_fraction"] = float(patient["model_input_missing_fraction"])
                row["high_missingness_warning"] = bool(patient["high_missingness_warning"])
                row["shared_model_gate_passed"] = bool(deployment["selection_gate_passed"])
                raw_rows.append(row)
    scenarios = study.pd.DataFrame(raw_rows)

    shared_heldout = factual_metrics(study, calibration["calibrated"], None,
                                     deployment["selected_candidate"], "temporal_test", horizons)
    geographic = factual_metrics(study, calibration["calibrated"], None,
                                 deployment["selected_candidate"], "geographic_test", horizons)
    if not geographic.empty:
        geographic["split"] = "geographic_test"
    if not shared_heldout.empty:
        shared_heldout["split"] = "temporal_test"
    heldout = study.concat_frames([shared_heldout, geographic])
    bmi35_metrics, bmi35_curves = bmi35_heldout_diagnostics(
        study, calibration["calibrated"], deployment, horizons, args.seed, args.smoke)

    internal = context.internal / "scenario_projections"
    figures = internal / "FIGURES_TO_EXPORT"
    public = context.run_dir / "shared_architecture_analysis"
    public_figures = public / "FIGURES_TO_EXPORT"
    internal.mkdir(parents=True, exist_ok=True); public.mkdir(parents=True, exist_ok=True)
    study.atomic_pickle(internal / "patient_scenario_projections.pkl", scenarios)
    scenarios.to_csv(internal / "patient_scenario_projections.csv", index=False)
    selected.to_csv(internal / "bmi_matching_audit.csv", index=False)
    bmi35_metrics.to_csv(internal / "bmi35_heldout_probability_metrics_unsuppressed.csv", index=False)
    study.atomic_json(internal / "manifest.json", {
        "created_utc": study.utc_now(), "selection": "BMI-only nearest profiles",
        "symmetric_reference_rule": "for each factual arm, equal-weight midpoint of the other two factual arm medians",
        "heldout_baseline_bmi_medians": medians, "patients_per_arm_requested": args.patients_per_arm,
        "bmi_match_caliper": args.bmi_match_caliper, "forecast_horizons": list(horizons),
        "high_missingness_threshold": args.high_missingness_threshold,
        "bmi35_probability": "predictive CDF with explicit tails plus arm/horizon isotonic calibration when supported",
        "shared_model_gate_passed": bool(deployment["selection_gate_passed"]),
        "claim": deployment["claim"], "patient_level_internal_only": True,
    })

    validation = study.pd.DataFrame(deployment["validation_by_arm_horizon"])
    selection_summary = study.pd.DataFrame(deployment["selection_summary"])
    public_heldout = suppress_public_metrics(study, heldout)
    public_validation = suppress_public_metrics(study, validation)
    public_bmi35_metrics = suppress_bmi35_metrics(study, bmi35_metrics)
    matching_rows = []
    for arm in ARMS:
        other = [name for name in ARMS if name != arm]
        matching_rows.append({
            "arm": arm, "n": int((profiles["observed_arm"] == arm).sum()),
            "factual_arm_median_bmi": medians[arm],
            "reference_arm_1": other[0], "reference_arm_1_median_bmi": medians[other[0]],
            "reference_arm_2": other[1], "reference_arm_2_median_bmi": medians[other[1]],
            "bmi_match_target": (medians[other[0]] + medians[other[1]]) / 2.0,
        })
    matching_reference = study.suppress_small_cells(study.pd.DataFrame(matching_rows), ["n"])
    matching_reference["arm"] = [row["arm"] for row in matching_rows]
    matching_reference["reference_arm_1"] = [row["reference_arm_1"] for row in matching_rows]
    matching_reference["reference_arm_2"] = [row["reference_arm_2"] for row in matching_rows]
    public_heldout.to_csv(public / "factual_performance_by_arm_horizon.csv", index=False)
    public_validation.to_csv(public / "shared_validation_by_arm_horizon.csv", index=False)
    public_bmi35_metrics.to_csv(public / "bmi35_heldout_probability_metrics.csv", index=False)
    bmi35_curves.to_csv(public / "bmi35_heldout_roc_curve_points.csv", index=False)
    selection_summary.to_csv(public / "shared_selection_summary.csv", index=False)
    matching_reference.to_csv(public / "bmi_matching_reference.csv", index=False)
    study.atomic_json(public / "manifest.json", {
        "created_utc": study.utc_now(), "selected_shared_candidate": deployment["selected_candidate"],
        "selection_gate_passed": deployment["selection_gate_passed"],
        "gate_warning": ("" if deployment["selection_gate_passed"]
                         else "FAILED: shared-model outputs are exploratory only"),
        "patient_level_scenarios": "INTERNAL/scenario_projections only",
        "matching_rule": "for each factual arm, equal-weight midpoint of the other two factual-arm held-out median baseline BMIs",
        "claim": deployment["claim"],
        "bmi35_probability_evaluation": "factual held-out temporal/geographic rows at 3, 6, and 12 months",
    })
    render_public_architecture_book(study, public_figures, public_validation, selection_summary,
                                    public_heldout, matching_reference,
                                    public_bmi35_metrics, bmi35_curves,
                                    deployment["selected_candidate"], deployment["selection_gate_passed"])
    render_internal_scenario_book(study, figures, selected, scenarios, medians,
                                  deployment["selection_gate_passed"])
    return 0


def worker_render(args: argparse.Namespace, study: Any) -> int:
    cfg = run_config(study, args)
    context = study.load_run_context(cfg)
    figure_data = study.require_checkpoint(context, "figure_data")
    rendered = study.render_figure_book(figure_data, context.export)
    context.state["status"] = "completed"
    context.state["completed_utc"] = study.utc_now()
    context.state["export_files"] = [path.name for path in rendered]
    study.atomic_json(context.run_dir / "run_state.json", context.state)
    return 0


def stage_runtime_scripts(run_dir: Path, scripts_dir: Path, seed: int,
                          incretin_months: int, horizons: Sequence[int]) -> dict[str, Path]:
    """Stage untouched downstream scripts beside a constant-patched copy of the study module.

    The originals remain unchanged. The runtime copy makes their default seed and incretin-window
    constants agree with this run, because the existing final orchestrator does not forward those
    CLI arguments to the secondary layer. The runtime three-arm copy uses the shared model's
    3/6/12-month scope so a selected shared candidate is never asked for an untrained 24-month row.
    """
    runtime = run_dir / "_runtime_scripts"
    runtime.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for name in ("run_metabolic_trajectory_study.py", "run_secondary_analyses.py",
                 "run_final_study.py", "run_three_arm_target_trial.py"):
        source = find_script(scripts_dir, name)
        destination = runtime / name
        if name == "run_metabolic_trajectory_study.py":
            code = source.read_text(encoding="utf-8")
            code, seed_count = re.subn(r"(?m)^SEED\s*=\s*\d+\s*$", f"SEED = {int(seed)}", code, count=1)
            code, month_count = re.subn(
                r"(?m)^INCRETIN_QUALIFYING_MONTHS\s*=\s*\d+\s*$",
                f"INCRETIN_QUALIFYING_MONTHS = {int(incretin_months)}", code, count=1)
            if seed_count != 1 or month_count != 1:
                raise RuntimeError("Could not patch runtime study constants for downstream reproducibility")
            destination.write_text(code, encoding="utf-8")
        elif name == "run_three_arm_target_trial.py":
            code = source.read_text(encoding="utf-8")
            horizon_literal = "(" + ", ".join(map(str, horizons)) + ("," if len(horizons) == 1 else "") + ")"
            windows = {3: (2.0, 4.5), 6: (4.5, 9.0), 12: (9.0, 18.0)}
            window_literal = "{" + ", ".join(
                f"{month}: {windows[int(month)]!r}" for month in horizons) + "}"
            code, horizon_count = re.subn(
                r"(?m)^HORIZONS\s*=\s*\([^\n]+\)\s*$",
                f"HORIZONS = {horizon_literal}", code, count=1)
            code, window_count = re.subn(
                r"(?m)^OUTCOME_WINDOWS\s*=\s*\{[^\n]+\}\s*$",
                f"OUTCOME_WINDOWS = {window_literal}", code, count=1)
            code, secondary_count = re.subn(
                r'"secondary_horizons"\s*:\s*\[6,\s*24\]',
                f'"secondary_horizons": {[int(month) for month in horizons if int(month) != 12]}',
                code, count=1)
            if horizon_count != 1 or window_count != 1 or secondary_count != 1:
                raise RuntimeError("Could not patch runtime three-arm horizons")
            destination.write_text(code, encoding="utf-8")
        else:
            shutil.copy2(source, destination)
        result[name] = destination
    return result


def merge_png_books(destination: Path, directories: Sequence[Path]) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pages = [path for directory in directories if directory.exists()
             for path in sorted(directory.glob("*.png"))]
    if not pages:
        return 0
    with PdfPages(destination, metadata={"Title": "Extended shared-architecture metabolic study"}) as writer:
        for page in pages:
            image = mpimg.imread(page)
            height, width = image.shape[:2]; dpi = width / 11.0
            figure = plt.figure(figsize=(11, height / dpi), dpi=dpi)
            figure.figimage(image, 0, 0, origin="upper", resize=False)
            writer.savefig(figure, dpi=dpi); plt.close(figure)
    return len(pages)


def bundle_outputs(run_dir: Path) -> Path:
    destination = run_dir / "experimental_shared_three_arm_bundle.zip"
    roots = [
        run_dir / "final_study_figure_book.pdf",
        run_dir / "extended_shared_architecture_figure_book.pdf",
        run_dir / "shared_architecture_analysis",
        run_dir / "three_arm_target_trial" / "three_arm_figure_book.pdf",
        run_dir / "experimental_shared_model_bundle.zip",
        run_dir / "run_manifest.json",
    ]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for root in roots:
            if root.is_file():
                archive.write(root, root.relative_to(run_dir))
            elif root.is_dir():
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(run_dir))
    return destination


def common_worker_args(args: argparse.Namespace, model_path: Path) -> list[str]:
    values = [
        "--output-dir", str(args.output_dir), "--scripts-dir", str(Path(args.scripts_dir).resolve()),
        "--architectures", args.architectures, "--max-training-rows", str(args.max_training_rows),
        "--smoke-patients", str(args.smoke_patients),
        "--domain-training-rows", str(args.domain_training_rows),
        "--hgb-iterations", str(args.hgb_iterations), "--gbr-estimators", str(args.gbr_estimators),
        "--catboost-iterations", str(args.catboost_iterations), "--forecast-horizons", args.forecast_horizons,
        "--patients-per-arm", str(args.patients_per_arm), "--bmi-match-caliper", str(args.bmi_match_caliper),
        "--domain-floor", str(args.domain_floor),
        "--high-missingness-threshold", str(args.high_missingness_threshold),
        "--noninferiority-margin", str(args.noninferiority_margin),
        "--coverage-tolerance", str(args.coverage_tolerance), "--seed", str(args.seed),
        "--incretin-qualifying-months", str(args.incretin_qualifying_months),
        "--_model-script", str(model_path),
    ]
    if args.smoke:
        values.append("--smoke")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser()
    args = cli.parse_args(argv)
    model_path = (Path(args._model_script).resolve() if args._model_script
                  else find_script(Path(args.scripts_dir), "run_metabolic_trajectory_study.py"))
    study = load_study(model_path)

    if args._original_stage:
        return worker_original_stage(args, study)
    if args._worker_models:
        return worker_models(args, study)
    if args._worker_calibration:
        return worker_calibration(args, study)
    if args._worker_scenarios:
        return worker_scenarios(args, study)
    if args._worker_render:
        return worker_render(args, study)

    try:
        architectures, horizons = validate_args(args, study)
    except ValueError as error:
        cli.error(str(error))
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output)
    args.architectures = ",".join(architectures)
    args.forecast_horizons = ",".join(map(str, horizons))
    common = common_worker_args(args, model_path)
    wrapper = str(Path(__file__).resolve())

    for stage in ("acquire_cohorts", "global_splits", "prediction_rows", "weights"):
        run([sys.executable, wrapper, "--_original-stage", stage, *common])
    run([sys.executable, wrapper, "--_worker-models", *common])
    run([sys.executable, wrapper, "--_worker-calibration", *common])
    for stage in ("evaluation", "figure_data"):
        run([sys.executable, wrapper, "--_original-stage", stage, *common])
    run([sys.executable, wrapper, "--_worker-render", *common])
    run([sys.executable, wrapper, "--_worker-scenarios", *common])

    manifest_path = output / "run_manifest.json"
    manifest = study.read_json(manifest_path, {}) or {}
    manifest["experimental_shared_wrapper"] = {
        "path": str(Path(__file__).resolve()), "sha256": study.sha256_file(Path(__file__)),
        "architectures": architectures, "forecast_horizons": list(horizons),
        "shared_training_horizons": list(horizons),
        "high_missingness_threshold": args.high_missingness_threshold,
        "scenario_selection": "BMI-only symmetric other-two-arm midpoint",
        "patient_scenarios_internal_only": True,
    }
    study.atomic_json(manifest_path, manifest)

    downstream = not args.skip_downstream and not args.smoke
    if downstream:
        runtime = stage_runtime_scripts(output, Path(args.scripts_dir).resolve(), args.seed,
                                       args.incretin_qualifying_months, horizons)
        env = os.environ.copy()
        roots = [str(runtime["run_final_study.py"].parent),
                 str(Path(args.scripts_dir).resolve()), str(Path(args.scripts_dir).resolve().parent)]
        env["PYTHONPATH"] = os.pathsep.join(roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        run([sys.executable, str(runtime["run_final_study.py"]), "--from-run", str(output)], env=env)
        run([sys.executable, str(runtime["run_three_arm_target_trial.py"]),
             "--descriptive-from-run", str(output), "--output-dir", str(output / "three_arm_target_trial"),
             "--seed", str(args.seed)], env=env)
        if args.run_three_arm_full:
            run([sys.executable, str(runtime["run_three_arm_target_trial.py"]), "--full",
                 "--output-dir", str(output / "three_arm_target_trial_full"), "--seed", str(args.seed)], env=env)
    elif args.smoke and not args.skip_downstream:
        print("[shared] smoke run: downstream final-study chain skipped because its --from-run mode re-acquires Cosmos")

    page_count = merge_png_books(output / "extended_shared_architecture_figure_book.pdf", [
        output / "FIGURES_TO_EXPORT", output / "secondary" / "FIGURES_TO_EXPORT",
        output / "three_arm_target_trial" / "FIGURES_TO_EXPORT",
        output / "shared_architecture_analysis" / "FIGURES_TO_EXPORT",
    ])
    bundle = bundle_outputs(output)
    study.atomic_json(output / "experimental_shared_run_record.json", {
        "completed_utc": study.utc_now(), "run_dir": str(output), "extended_pages": page_count,
        "bundle": str(bundle), "downstream_queued": downstream,
        "internal_scenario_book": str(output / "INTERNAL" / "scenario_projections" /
                                      "FIGURES_TO_EXPORT" / "bmi_matched_scenario_projection_book.pdf"),
        "internal_model_domain_table": str(output / "INTERNAL" / "scenario_projections" /
                                           "FIGURES_TO_EXPORT" /
                                           "01_model_domain_scores_and_missingness.png"),
        "bmi35_roc_figure": str(output / "shared_architecture_analysis" /
                                "FIGURES_TO_EXPORT" / "04_bmi35_heldout_roc_curves.png"),
    })
    print(f"[shared] completed: {output}")
    print(f"[shared] public architecture analysis: {output / 'shared_architecture_analysis'}")
    print(f"[shared] INTERNAL patient scenarios: {output / 'INTERNAL' / 'scenario_projections'}")
    print(f"[shared] returnable bundle (patient scenarios excluded): {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
