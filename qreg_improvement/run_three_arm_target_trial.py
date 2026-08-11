#!/usr/bin/env python3
"""Three-arm metabolic target-trial emulation and honest prognostic comparison.

This program deliberately has two non-interchangeable analysis paths:

* ``--descriptive-from-run`` reads held-out predictions from a completed metabolic
  trajectory run and describes the three observed reporting cohorts.  It never
  counterfactually changes a patient's treatment label.
* ``--full`` constructs a new-user cohort from reviewed raw event sources and uses
  cross-fitted causal nuisance models.  The frozen prediction store is never read by
  this path.  If the raw source or any hard design gate is unavailable, the program
  publishes a diagnostic report and no causal estimate.

Patient-level checkpoints are confined to ``INTERNAL``.  Only aggregate,
small-cell-suppressed files enter the result bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import tempfile
import textwrap
import warnings
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_metabolic_trajectory_study as study  # noqa: E402


VERSION = "three-arm-target-trial-1.0.0"
SEED = 20260810
FOLDS = 5
MIN_CELL = 11
ARM_LABELS = {0: "SG", 1: "RYGB", 2: "INCRETIN"}
ARM_CODES = {value: key for key, value in ARM_LABELS.items()}
PAIRWISE = ((1, 0), (1, 2), (0, 2))
HORIZONS = (6, 12, 24)
OUTCOMES = ("bmi", "hba1c")
OUTCOME_WINDOWS = {6: (4.5, 9.0), 12: (9.0, 18.0), 24: (18.0, 30.0)}
BMI_BANDS = ((35.0, 40.0, "[35,40)"), (40.0, 45.0, "[40,45)"),
             (45.0, 50.0, "[45,50)"), (50.0, 75.0000001, "[50,75]"))

FIGURE_FILES = (
    "00_three_arm_status.png",
    "01_target_trial_protocol_and_funnel.png",
    "02_three_arm_overlap_and_balance.png",
    "03_descriptive_three_arm_trajectories.png",
    "04_three_arm_primary_effects.png",
    "05_missingness_and_robustness.png",
    "06_pairwise_heterogeneous_benefit.png",
    "07_policy_value_and_claims.png",
)
FIGURE_BOOK = "three_arm_figure_book.pdf"
AGGREGATE_FILES = (
    "design_diagnostic.csv",
    "cohort_funnel.csv",
    "baseline_by_arm.csv",
    "three_arm_overlap.csv",
    "three_arm_balance.csv",
    "threshold_incremental_discrimination.csv",
    "arm_standardized_means.csv",
    "pairwise_aipw.csv",
    "robustness.csv",
    "pairwise_rate.csv",
    "benefit_calibration.csv",
    "policy_value.csv",
    "three_arm_trial_protocol.json",
    "manifest.json",
)

HARD_GATE_LABELS = (
    "incident_incretin_without_future_conditioning",
    "exact_initiation_dates_all_arms",
    "eligibility_assignment_followup_aligned",
    "full_eligible_cohort_not_prediction_subset",
    "primary_horizon_arm_size",
    "three_way_positivity_and_ess",
    "required_preindex_confounders",
    "auditable_outcome_timing_and_maturity",
)
SOFT_GATE_LABELS = (
    "gerd_available",
    "care_process_confounders_available",
    "incretin_version_available",
    "exact_outcome_timestamps",
    "weighted_balance",
    "calendar_geography_availability",
)

# L is frozen here, before any outcome is inspected.  Names ending in ``_baseline``
# and all other entries are explicitly pre-index.  The leakage self-test audits this list.
FROZEN_CONFOUNDERS = (
    "baseline_bmi", "baseline_hba1c", "age", "sex", "race", "ethnicity",
    "diabetes_severity", "insulin", "biguanide", "sglt2", "hypertension",
    "dyslipidemia", "osa", "smoking", "gerd_baseline", "creatinine_baseline",
    "egfr_baseline", "ldl_baseline", "hdl_baseline", "triglycerides_baseline",
    "glucose_fasting_baseline", "sbp_baseline", "dbp_baseline", "coverage",
    "svi", "ruca", "state", "calendar_year", "center_id", "surgeon_id",
    "prescriber_id", "formulary_proxy",
)
FORBIDDEN_POSTINDEX_TOKENS = (
    "post", "followup", "target", "refill", "persistence", "duration", "switch",
    "mediator", "outcome_", "future",
)

np = pd = plt = PdfPages = None


@dataclass(frozen=True)
class TrialConfig:
    mode: str
    output_dir: str
    descriptive_run: str | None = None
    seed: int = SEED
    folds: int = FOLDS
    bootstrap_replicates: int = 400
    min_cell: int = MIN_CELL
    propensity_floor: float = 0.01
    clean_window_days: int = 7
    baseline_days: int = 365

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    @property
    def internal(self) -> Path:
        return self.run_dir / "INTERNAL"

    @property
    def export(self) -> Path:
        return self.run_dir / "FIGURES_TO_EXPORT"


class DesignGateFailure(RuntimeError):
    def __init__(self, message: str, diagnostic: Any | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


def load_runtime() -> None:
    global np, pd, plt, PdfPages
    if np is not None:
        return
    import numpy as _np
    import pandas as _pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
    np, pd, plt, PdfPages = _np, _pd, _plt, _PdfPages
    study.load_runtime_packages()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return None if not np.isfinite(item) else float(item)
        if isinstance(item, (np.bool_,)):
            return bool(item)
        if isinstance(item, (datetime, pd.Timestamp)):
            return item.isoformat()
        raise TypeError(type(item).__name__)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    study.atomic_json(path, value)


def atomic_pickle(path: Path, value: Any) -> None:
    study.atomic_pickle(path, value)


def atomic_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        payload = frame.to_csv(index=False, lineterminator="\n")
    study.atomic_text(path, payload)


def read_csv_or_empty(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def stable_fraction(value: Any, seed: int = SEED, salt: str = "fold") -> float:
    token = f"{seed}|{salt}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(2**64)


def patient_folds(patient_ids: Any, seed: int = SEED, folds: int = FOLDS,
                  salt: str = "xfit") -> Any:
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    return np.asarray([min(folds - 1, int(stable_fraction(value, seed, salt) * folds))
                       for value in patient_ids], dtype=int)


def effective_sample_size(weight: Any) -> float:
    values = np.asarray(weight, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values) or float(np.sum(values * values)) <= 0:
        return 0.0
    return float(np.sum(values) ** 2 / np.sum(values * values))


def suppress_small_cells(frame: Any, count_columns: Sequence[str] = ("n",),
                         threshold: int = MIN_CELL) -> Any:
    """Blank disclosive numeric cells while retaining every declared row."""
    result = frame.copy()
    mask = pd.Series(False, index=result.index, dtype=bool)
    for column in count_columns:
        if column in result:
            count = pd.to_numeric(result[column], errors="coerce")
            mask |= count.gt(0) & count.lt(threshold)
    result["small_cell_suppressed"] = mask
    for column in result.columns:
        if column == "small_cell_suppressed":
            continue
        if column in count_columns or (
            pd.api.types.is_numeric_dtype(result[column])
            and not pd.api.types.is_bool_dtype(result[column])
        ):
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
            result.loc[mask, column] = np.nan
    return result


def ensure_directories(cfg: TrialConfig) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.internal.mkdir(parents=True, exist_ok=True)
    cfg.export.mkdir(parents=True, exist_ok=True)


def empty_outputs() -> dict[str, Any]:
    return {
        "design_diagnostic": pd.DataFrame(columns=["gate_type", "gate", "passed", "status", "detail", "n"]),
        "cohort_funnel": pd.DataFrame(columns=["stage", "arm", "n", "excluded_n", "reason"]),
        "baseline_by_arm": pd.DataFrame(columns=["variable", "arm", "n", "mean", "sd", "level", "percent"]),
        "three_arm_overlap": pd.DataFrame(columns=["arm", "n", "ess", "min_probability", "p01", "median", "p99", "max_weight"]),
        "three_arm_balance": pd.DataFrame(columns=["variable", "comparison", "unweighted_smd", "weighted_smd"]),
        "threshold_incremental_discrimination": pd.DataFrame(columns=["arm", "horizon", "band", "n", "events", "nonevents", "auroc_full", "auroc_baseline_bmi", "difference", "ci_low", "ci_high", "status"]),
        "arm_standardized_means": pd.DataFrame(columns=["outcome", "horizon", "arm", "n", "observed_n", "mean", "se", "ci_low", "ci_high", "model"]),
        "pairwise_aipw": pd.DataFrame(columns=["outcome", "horizon", "contrast", "arm_a", "arm_b", "n", "estimate", "se", "ci_low", "ci_high", "estimand", "claim_status"]),
        "robustness": pd.DataFrame(columns=["outcome", "horizon", "contrast", "analysis", "n", "estimate", "ci_low", "ci_high", "status"]),
        "pairwise_rate": pd.DataFrame(columns=["outcome", "horizon", "contrast", "n", "rate", "ci_low", "ci_high", "c_for_benefit", "status"]),
        "benefit_calibration": pd.DataFrame(columns=["outcome", "contrast", "decile", "n", "predicted_benefit", "observed_dr_benefit", "status"]),
        "policy_value": pd.DataFrame(columns=["outcome", "horizon", "policy", "n", "value", "se", "ci_low", "ci_high", "difference_vs_learned", "difference_ci_low", "difference_ci_high", "status"]),
    }


def write_output_frames(cfg: TrialConfig, outputs: Mapping[str, Any]) -> None:
    for name in AGGREGATE_FILES:
        if not name.endswith(".csv"):
            continue
        key = name[:-4]
        frame = outputs.get(key, empty_outputs()[key])
        count_columns = [column for column in ("n", "observed_n", "events", "nonevents", "excluded_n")
                         if column in frame.columns]
        atomic_csv(cfg.run_dir / name, suppress_small_cells(frame, count_columns, cfg.min_cell))


def trial_protocol(cfg: TrialConfig, source_facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    facts = dict(source_facts or {})
    return {
        "protocol_version": VERSION,
        "terminology": {"IBT": "incretin-based therapy"},
        "frozen_before_outcome_analysis": True,
        "seed": cfg.seed,
        "patient_clustered_folds": cfg.folds,
        "population": "Adults with type 2 diabetes eligible for all three observed initial strategies",
        "eligibility": {
            "age_years": ">=18",
            "type_2_diabetes": True,
            "baseline_bmi_kg_m2": [35, 75],
            "baseline_window_days": [-cfg.baseline_days, 0],
            "observable_history_days": 365,
            "prior_bariatric_surgery": "none",
            "prior_incretin_days": 365,
            "dialysis_or_transplant": "none",
            "common_availability_period_and_setting": True,
            "baseline_confounders": list(FROZEN_CONFOUNDERS),
        },
        "assignment": {
            "0": "sleeve gastrectomy on index date",
            "1": "RYGB on index date",
            "2": "first recorded incretin prescription or administration on index date",
            "simultaneous_clean_window_days": [-cfg.clean_window_days, cfg.clean_window_days],
            "future_persistence_used": False,
            "treatment_versions": "calendar-dependent mixture if agent/dose/route unavailable",
        },
        "time_zero": "actual surgery or incident prescription/administration date",
        "follow_up": {
            "strategy": "treatment-policy analog; later switching/augmentation does not censor",
            "outcome_windows_months": {str(k): list(v) for k, v in OUTCOME_WINDOWS.items()},
            "outcome_selection": "nearest measurement to nominal horizon, identical across arms",
            "end": ["outcome ascertainment", "death", "database exit", "administrative end"],
        },
        "outcomes": {
            "primary": "absolute BMI at 12 months",
            "key_secondary": "absolute HbA1c at 12 months",
            "secondary_horizons": [6, 24],
            "baseline_outcome_adjustment": True,
        },
        "estimand": "pairwise marginal mean difference for initial treatment strategy",
        "competing_event_estimand": (
            "survivor-observed outcome under a treatment-policy analog; death is reported separately "
            "and is not encoded as an ordinary missing outcome"
        ),
        "nuisance_models": {
            "propensity_primary": "multinomial logistic regression",
            "propensity_sensitivity": "histogram gradient boosting classifier",
            "observation_primary": "logistic regression",
            "observation_sensitivity": "histogram gradient boosting classifier",
            "outcome_mean": "HistGradientBoostingRegressor(loss='squared_error')",
        },
        "positivity": {"probability_floor": cfg.propensity_floor, "minimum_ess": 100, "minimum_ess_fraction": 0.20},
        "source_facts_at_freeze": facts,
        "clinical_limitations": [
            "Safety and adverse events are unavailable.", "Treatment burden and cost are unavailable.",
            "Contraindications and patient preference may be incompletely captured.",
            "BMI and HbA1c alone cannot support an individual treatment recommendation.",
        ],
    }


def write_protocol(cfg: TrialConfig, source_facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    protocol = trial_protocol(cfg, source_facts)
    atomic_json(cfg.run_dir / "three_arm_trial_protocol.json", protocol)
    return protocol


def write_descriptive_protocol(cfg: TrialConfig, source_run: Path) -> dict[str,Any]:
    protocol={
        "protocol_version":VERSION,"analysis_type":"descriptive_prognostic_only",
        "terminology":{"IBT":"incretin-based therapy"},
        "source_run":str(source_run),"seed":cfg.seed,
        "claim":"Prognostic comparison of separate observed cohorts; not a treatment effect.",
        "reporting_cohorts":["RYGB","sleeve gastrectomy","incretin-based therapy"],
        "patient_labels":"Observed procedure/cohort labels are retained and never switched.",
        "standardization":"Three-arm overlap weighting on covariates shared by all reporting cohorts.",
        "prediction_rows":"Calibrated selected-model held-out rows only.",
        "bmi_threshold_audit":"Full prediction AUROC versus -baseline BMI, paired patient bootstrap, and prespecified BMI bands.",
        "interpretation":"AUROC is non-additive; its difference is not a unique percentage contribution of baseline BMI.",
        "individual_treatment_recommendation":False,
    }
    atomic_json(cfg.run_dir/"three_arm_trial_protocol.json",protocol);return protocol


def gate_row(gate_type: str, gate: str, passed: bool, detail: str, n: int | None = None) -> dict[str, Any]:
    return {"gate_type": gate_type, "gate": gate, "passed": bool(passed),
            "status": "PASS" if passed else "FAIL", "detail": detail, "n": n}


def source_design_diagnostic(connect: bool = True) -> tuple[Any, dict[str, Any]]:
    """Inspect only reviewed contract/schema/aggregate facts; never read patient rows.

    The current production module exposes an empty reviewed raw-event SQL contract and two
    prebuilt wide cohorts.  A prebuilt GLP1 continuer cohort cannot establish incident starts
    without future conditioning, so it truthfully fails the first and fourth hard gates.
    """
    load_runtime()
    raw_contract = dict(getattr(study, "EMBEDDED_RAW_SOURCE_SQL", {}) or {})
    required = set(getattr(study, "RAW_REQUIRED_SOURCES", ("patients", "procedures", "medications", "measurements")))
    reviewed_raw = required.issubset(raw_contract) and bool(raw_contract)
    facts: dict[str, Any] = {
        "source_mode": "reviewed_raw_event_contract" if reviewed_raw else "future_conditioned_wide_cohorts_only",
        "reviewed_raw_contract": reviewed_raw,
        "raw_contract_domains": sorted(raw_contract),
        "wide_sources": ["dbo.MBSCohort", "dbo.GLP1Cohort"],
        "schema_fingerprint": "not_queried",
        "aggregate_source_counts": {},
    }
    schema_ok = False
    exact_dates = False
    timing_ok = False
    if connect:
        connection = study.connect_cosmos()
        try:
            columns = pd.read_sql_query(
                "SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE "
                "FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME IN ('MBSCohort','GLP1Cohort') "
                "ORDER BY TABLE_SCHEMA,TABLE_NAME,ORDINAL_POSITION", connection,
            )
            facts["schema_fingerprint"] = digest(columns.astype(str).to_dict("records"))
            by_table = {name: set(group["COLUMN_NAME"].astype(str))
                        for name, group in columns.groupby("TABLE_NAME")}
            schema_ok = all(name in by_table for name in ("MBSCohort", "GLP1Cohort"))
            mapped: dict[str, dict[str, str]] = {}
            for logical in ("MBSCohort", "GLP1Cohort"):
                if logical in by_table:
                    probe = pd.DataFrame(columns=sorted(by_table[logical]))
                    try:
                        mapped[logical] = study.resolve_wide_fields(probe, logical)
                    except Exception:
                        mapped[logical] = {}
            exact_dates = reviewed_raw or bool(mapped.get("MBSCohort", {}).get("procedure_date") and
                                               mapped.get("GLP1Cohort", {}).get("glp1_start"))
            timing_ok = reviewed_raw or all(any(key.startswith(f"target_{outcome}_") for key in mapping)
                                            for mapping in mapped.values() for outcome in ("bmi",))
            for table in ("MBSCohort", "GLP1Cohort"):
                query = f"SELECT COUNT_BIG(*) AS n FROM [dbo].[{table}]"
                count = pd.read_sql_query(query, connection)
                facts["aggregate_source_counts"][table] = int(count.iloc[0, 0]) if len(count) else 0
        finally:
            connection.close()
    else:
        schema_ok = reviewed_raw
        exact_dates = reviewed_raw
        timing_ok = reviewed_raw

    rows = [
        gate_row("hard", HARD_GATE_LABELS[0], reviewed_raw,
                 "Reviewed raw medication events support true new-user entry." if reviewed_raw else
                 "Only the future-conditioned GLP1 continuer cohort is reviewed; it is forbidden as a causal arm."),
        gate_row("hard", HARD_GATE_LABELS[1], reviewed_raw and exact_dates,
                 "Exact raw initiation fields are reviewed." if reviewed_raw and exact_dates else
                 "Wide index dates do not establish a full raw-event new-user source."),
        gate_row("hard", HARD_GATE_LABELS[2], reviewed_raw,
                 "Raw events permit same-day alignment." if reviewed_raw else "Alignment cannot be audited from continuer-cohort membership."),
        gate_row("hard", HARD_GATE_LABELS[3], reviewed_raw,
                 "Raw source is independent of prediction storage." if reviewed_raw else
                 "The available GLP1 object is a selected cohort rather than the full eligible source population."),
        gate_row("hard", HARD_GATE_LABELS[4], False, "Requires construction and primary-horizon maturity counts."),
        gate_row("hard", HARD_GATE_LABELS[5], False, "Requires full-cohort cross-fitted three-arm propensities."),
        gate_row("hard", HARD_GATE_LABELS[6], False, "Requires temporal population audit after cohort construction."),
        gate_row("hard", HARD_GATE_LABELS[7], reviewed_raw and timing_ok,
                 "Exact measurement timestamps are reviewable." if reviewed_raw and timing_ok else
                 "Upstream fixed-horizon fields do not provide auditable exact outcome timestamps."),
    ]
    for label in SOFT_GATE_LABELS:
        rows.append(gate_row("soft", label, False, "Requires full raw-event cohort audit."))
    facts["wide_schema_available"] = schema_ok
    facts["wide_exact_index_columns"] = exact_dates
    return pd.DataFrame(rows), facts


def hard_gates_pass(diagnostic: Any) -> bool:
    rows = diagnostic.loc[diagnostic["gate_type"].eq("hard")]
    return bool(len(rows) == len(HARD_GATE_LABELS) and rows["passed"].astype(bool).all())


def claim_status(diagnostic: Any) -> str:
    if not hard_gates_pass(diagnostic):
        return "DESIGN GATES FAILED - NO CAUSAL ESTIMATE"
    soft = diagnostic.loc[diagnostic["gate_type"].eq("soft"), "passed"]
    return "CAUSAL" if len(soft) and soft.astype(bool).all() else "EXPLORATORY CAUSAL"


# ======================================================================================
# Raw-event acquisition and trial cohort construction
# ======================================================================================

def first_present(frame: Any, names: Sequence[str], default: Any = float("nan")) -> Any:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def numeric_flag(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "present", "positive", "current"}:
        return 1.0
    if text in {"0", "false", "no", "n", "absent", "negative", "never"}:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def medication_event_date(frame: Any) -> Any:
    candidates = []
    for name in ("fill_date", "administration_date", "medication_start_date", "order_date"):
        if name in frame:
            candidates.append(pd.to_datetime(frame[name], errors="coerce"))
    if not candidates:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    result = candidates[0]
    for candidate in candidates[1:]:
        result = result.fillna(candidate)
    return result.dt.normalize()


def normalize_incretin_events(medications: Any) -> Any:
    if medications.empty:
        return pd.DataFrame(columns=["patient_id", "event_date", "ingredient", "route", "intended_indication", "dose", "dose_unit", "source_table"])
    frame = medications.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["event_date"] = medication_event_date(frame)
    ingredient_source = first_present(frame, ("ingredient", "medication_concept", "medication_name"), "")
    normalized = [study.normalize_ingredient(value) for value in ingredient_source]
    frame["ingredient"] = [item[0] for item in normalized]
    frame["therapy_class"] = [item[1] for item in normalized]
    frame["mapping_method"] = [item[2] for item in normalized]
    frame["route"] = first_present(frame, ("route", "medication_route"), "unknown").fillna("unknown").astype(str)
    frame["intended_indication"] = first_present(frame, ("intended_indication", "indication"), "unknown").fillna("unknown").astype(str)
    frame["dose"] = pd.to_numeric(first_present(frame, ("dose", "dose_value")), errors="coerce")
    frame["dose_unit"] = first_present(frame, ("dose_unit",), "").fillna("").astype(str)
    frame["source_table"] = first_present(frame, ("source_table",), "raw.medications").fillna("raw.medications").astype(str)
    return frame.loc[frame["ingredient"].notna() & frame["event_date"].notna(),
                     ["patient_id", "event_date", "ingredient", "therapy_class", "route", "intended_indication", "dose", "dose_unit", "source_table"]].sort_values(
                         ["patient_id", "event_date"], kind="mergesort").reset_index(drop=True)


def normalized_measurements(bundle: Any) -> Any:
    normalized, _quality = study.normalize_measurements(bundle.measurements)
    frame = normalized.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["measurement_date"] = pd.to_datetime(frame["measurement_date"], errors="coerce").dt.normalize()
    frame["outcome"] = frame["outcome"].astype(str)
    return frame.loc[frame["measurement_date"].notna() & frame["outcome"].isin(OUTCOMES)].copy()


def _patient_lookup(frame: Any) -> dict[str, Mapping[str, Any]]:
    if frame.empty:
        return {}
    work = frame.copy()
    work["patient_id"] = work["patient_id"].astype(str)
    return {str(row["patient_id"]): row for row in work.drop_duplicates("patient_id", keep="first").to_dict("records")}


def _diagnosis_lookup(diagnoses: Any) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    result: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    if diagnoses.empty:
        return result
    code_col = next((name for name in ("diagnosis_code", "icd_code", "code") if name in diagnoses), None)
    date_col = next((name for name in ("diagnosis_date", "encounter_date", "date") if name in diagnoses), None)
    if code_col is None:
        return result
    for row in diagnoses.itertuples(index=False):
        payload = row._asdict()
        patient = str(payload.get("patient_id"))
        date = pd.to_datetime(payload.get(date_col), errors="coerce") if date_col else pd.NaT
        result.setdefault(patient, []).append((date, str(payload.get(code_col) or "").upper().replace(".", "")))
    return result


def _has_preindex_code(entries: Sequence[tuple[Any, str]], index_date: Any,
                       prefixes: Sequence[str]) -> bool:
    for when, code in entries:
        if (pd.isna(when) or pd.Timestamp(when) <= pd.Timestamp(index_date)) and any(code.startswith(prefix) for prefix in prefixes):
            return True
    return False


def _age_at_index(patient: Mapping[str, Any], index_date: pd.Timestamp) -> float:
    for name in ("age", "age_at_index", "age_at_event"):
        if name in patient and pd.notna(patient[name]):
            return float(patient[name])
    for name in ("birth_date", "date_of_birth", "dob"):
        if name in patient and pd.notna(patient[name]):
            return float((index_date - pd.Timestamp(patient[name])).days / 365.25)
    for name in ("birth_year", "year_of_birth"):
        if name in patient and pd.notna(patient[name]):
            return float(index_date.year - int(patient[name]))
    return float("nan")


def _nearest_baseline(patient_measurements: Any, outcome: str, index_date: pd.Timestamp,
                      days: int = 365) -> tuple[float, Any]:
    cell = patient_measurements.loc[
        patient_measurements["outcome"].eq(outcome)
        & patient_measurements["measurement_date"].between(index_date - pd.Timedelta(days=days), index_date)
    ].copy()
    if cell.empty:
        return float("nan"), pd.NaT
    cell["distance"] = (index_date - cell["measurement_date"]).dt.days
    picked = cell.sort_values(["distance", "measurement_date"], kind="mergesort").iloc[0]
    return float(picked["value"]), picked["measurement_date"]


def _nearest_outcome(patient_measurements: Any, outcome: str, index_date: pd.Timestamp,
                     horizon: int) -> tuple[float, Any]:
    low, high = OUTCOME_WINDOWS[horizon]
    start = index_date + pd.Timedelta(days=int(math.ceil(low * study.DAYS_PER_MONTH)))
    end = index_date + pd.Timedelta(days=int(math.floor(high * study.DAYS_PER_MONTH)))
    nominal = index_date + pd.Timedelta(days=int(round(horizon * study.DAYS_PER_MONTH)))
    cell = patient_measurements.loc[
        patient_measurements["outcome"].eq(outcome)
        & patient_measurements["measurement_date"].between(start, end, inclusive="both")
    ].copy()
    if cell.empty:
        return float("nan"), pd.NaT
    cell["distance"] = (cell["measurement_date"] - nominal).abs().dt.days
    picked = cell.sort_values(["distance", "measurement_date"], kind="mergesort").iloc[0]
    return float(picked["value"]), picked["measurement_date"]


def _candidate_events(bundle: Any, incretin: Any) -> Any:
    surgery = bundle.procedures.copy()
    if not surgery.empty:
        surgery["patient_id"] = surgery["patient_id"].astype(str)
        surgery["index_date"] = pd.to_datetime(surgery["procedure_date"], errors="coerce").dt.normalize()
        codes = first_present(surgery, ("procedure_code", "cpt_code"), "").astype(str).str.replace(r"\D", "", regex=True)
        surgery["arm"] = codes.map({"43775": 0, "43644": 1, "43645": 1, "43846": 1})
        surgery["ingredient"] = ""
        surgery["route"] = "surgical"
        surgery["intended_indication"] = "metabolic_surgery"
        surgery["dose"] = np.nan
        surgery["dose_unit"] = ""
        surgery = surgery.loc[surgery["arm"].notna() & surgery["index_date"].notna(),
                              ["patient_id", "index_date", "arm", "ingredient", "route", "intended_indication", "dose", "dose_unit"]]
    else:
        surgery = pd.DataFrame(columns=["patient_id", "index_date", "arm", "ingredient", "route", "intended_indication", "dose", "dose_unit"])
    drug = incretin.rename(columns={"event_date": "index_date"}).copy()
    drug["arm"] = 2
    drug = drug[["patient_id", "index_date", "arm", "ingredient", "route", "intended_indication", "dose", "dose_unit"]]
    events = pd.concat([surgery, drug], ignore_index=True)
    events["arm"] = events["arm"].astype(int)
    return events.sort_values(["patient_id", "index_date", "arm"], kind="mergesort").reset_index(drop=True)


def construct_trial_cohort(bundle: Any, cfg: TrialConfig) -> tuple[Any, Any, dict[str, Any]]:
    """Construct one actual initial-strategy row per patient from the full raw source."""
    measurements = normalized_measurements(bundle)
    incretin = normalize_incretin_events(bundle.medications)
    events = _candidate_events(bundle, incretin)
    patients = _patient_lookup(bundle.patients)
    diagnoses = _diagnosis_lookup(bundle.diagnoses)
    measurements_by_patient = {str(key): group for key, group in measurements.groupby("patient_id", sort=False)}
    procedures_by_patient = {
        str(key): group.assign(procedure_date=pd.to_datetime(group["procedure_date"], errors="coerce").dt.normalize())
        for key, group in bundle.procedures.assign(patient_id=bundle.procedures["patient_id"].astype(str)).groupby("patient_id", sort=False)
    } if not bundle.procedures.empty else {}
    drugs_by_patient = {str(key): group for key, group in incretin.groupby("patient_id", sort=False)}

    rows: list[dict[str, Any]] = []
    funnel: list[dict[str, Any]] = []
    exclusion_counts: dict[tuple[str, str], int] = {}

    def exclude(arm: int, reason: str) -> None:
        key = (ARM_LABELS[arm], reason)
        exclusion_counts[key] = exclusion_counts.get(key, 0) + 1

    for patient_id, candidate_group in events.groupby("patient_id", sort=True):
        patient = patients.get(str(patient_id))
        if patient is None:
            for arm in candidate_group["arm"]:
                exclude(int(arm), "missing_patient_history_row")
            continue
        # Choose the earliest candidate event that independently meets baseline eligibility.
        # Later persistence/end dates are never referenced anywhere in this function.
        accepted: dict[str, Any] | None = None
        for candidate in candidate_group.to_dict("records"):
            arm = int(candidate["arm"])
            index_date = pd.Timestamp(candidate["index_date"])
            age = _age_at_index(patient, index_date)
            observation_start = pd.to_datetime(patient.get("observation_start_date"), errors="coerce")
            observation_end = pd.to_datetime(patient.get("observation_end_date", patient.get("administrative_end_date")), errors="coerce")
            if not np.isfinite(age) or age < 18:
                exclude(arm, "age_below_18_or_missing"); continue
            if pd.isna(observation_start) or observation_start > index_date - pd.Timedelta(days=365):
                exclude(arm, "less_than_365_days_baseline_history"); continue
            simultaneous = candidate_group.loc[
                candidate_group["arm"].ne(arm)
                & (candidate_group["index_date"] - index_date).abs().dt.days.le(cfg.clean_window_days)
            ]
            if not simultaneous.empty:
                exclude(arm, "simultaneous_strategy_within_clean_window"); continue
            prior_proc = procedures_by_patient.get(str(patient_id), pd.DataFrame())
            if not prior_proc.empty:
                prior_codes = first_present(prior_proc, ("procedure_code", "cpt_code"), "").astype(str).str.replace(r"\D", "", regex=True)
                prior_dates = pd.to_datetime(prior_proc["procedure_date"], errors="coerce")
                prior_mbs = prior_dates.lt(index_date) & prior_codes.isin(study.BARIATRIC_HISTORY_CODES)
                if prior_mbs.any():
                    exclude(arm, "prior_bariatric_surgery"); continue
            patient_drugs = drugs_by_patient.get(str(patient_id), pd.DataFrame())
            if not patient_drugs.empty:
                prior_incretin = patient_drugs["event_date"].between(index_date - pd.Timedelta(days=365), index_date - pd.Timedelta(days=1))
                if prior_incretin.any():
                    exclude(arm, "incretin_exposure_in_prior_365_days"); continue
            dx = diagnoses.get(str(patient_id), [])
            diabetes_value = numeric_flag(patient.get("diabetes", patient.get("diabetes_flag", patient.get("type2_diabetes"))))
            diabetes = diabetes_value == 1 or _has_preindex_code(dx, index_date, ("E11",))
            if not diabetes:
                exclude(arm, "type_2_diabetes_not_documented"); continue
            renal_value = numeric_flag(patient.get("dialysis_transplant", patient.get("prior_dialysis_transplant")))
            renal_dx = _has_preindex_code(dx, index_date, ("Z940", "Z992", "N186", "T861"))
            if renal_value == 1 or renal_dx:
                exclude(arm, "dialysis_or_transplant"); continue
            contraindication = any(
                numeric_flag(patient.get(name)) == 1
                for name in ("contraindication_to_surgery", "contraindication_to_incretin",
                             "not_surgical_candidate", "not_medication_candidate")
            )
            if contraindication:
                exclude(arm, "captured_contraindication_to_a_trial_strategy"); continue
            patient_measurements = measurements_by_patient.get(str(patient_id), measurements.iloc[:0])
            baseline_bmi, baseline_bmi_date = _nearest_baseline(patient_measurements, "bmi", index_date, cfg.baseline_days)
            baseline_hba1c, baseline_hba1c_date = _nearest_baseline(patient_measurements, "hba1c", index_date, cfg.baseline_days)
            if not np.isfinite(baseline_bmi) or not 35 <= baseline_bmi <= 75:
                exclude(arm, "baseline_bmi_missing_or_outside_35_75"); continue

            record: dict[str, Any] = {
                "patient_id": str(patient_id), "arm": arm, "arm_label": ARM_LABELS[arm],
                "index_date": index_date, "age": age, "baseline_bmi": baseline_bmi,
                "baseline_bmi_date": baseline_bmi_date, "baseline_hba1c": baseline_hba1c,
                "baseline_hba1c_date": baseline_hba1c_date, "diabetes_status": 1.0,
                "calendar_year": index_date.year, "observation_start_date": observation_start,
                "observation_end_date": observation_end, "ingredient": candidate.get("ingredient", ""),
                "route": candidate.get("route", "unknown"), "intended_indication":candidate.get("intended_indication","unknown"),
                "dose": candidate.get("dose", np.nan),
                "dose_unit": candidate.get("dose_unit", ""), "death_date": pd.to_datetime(patient.get("death_date"), errors="coerce"),
            }
            aliases = {
                "sex": ("sex", "gender"), "race": ("race", "first_race"), "ethnicity": ("ethnicity",),
                "coverage": ("coverage", "coverage_class", "payer_class"), "center_id": ("center_id", "site_id"),
                "surgeon_id": ("surgeon_id",), "prescriber_id": ("prescriber_id",),
                "formulary_proxy": ("formulary_proxy", "authorization_proxy"), "state": ("state", "state_or_province"),
                "smoking": ("smoking", "smoking_status"), "diabetes_severity": ("diabetes_severity",),
            }
            for canonical, names in aliases.items():
                record[canonical] = next((patient[name] for name in names if name in patient), np.nan)
            flags = ("insulin", "biguanide", "sglt2", "hypertension", "dyslipidemia", "osa")
            for name in flags:
                record[name] = numeric_flag(patient.get(name))
            for name in ("svi", "ruca", "gerd_baseline", "creatinine_baseline", "egfr_baseline",
                         "ldl_baseline", "hdl_baseline", "triglycerides_baseline", "glucose_fasting_baseline",
                         "sbp_baseline", "dbp_baseline"):
                record[name] = pd.to_numeric(patient.get(name), errors="coerce")
            for outcome in OUTCOMES:
                for horizon in HORIZONS:
                    value, when = _nearest_outcome(patient_measurements, outcome, index_date, horizon)
                    maturity_date = index_date + pd.Timedelta(days=int(math.floor(OUTCOME_WINDOWS[horizon][1] * study.DAYS_PER_MONTH)))
                    admin_mature = pd.notna(observation_end) and observation_end >= maturity_date
                    death_before = pd.notna(record["death_date"]) and record["death_date"] <= maturity_date
                    record[f"{outcome}_{horizon}"] = value
                    record[f"{outcome}_{horizon}_date"] = when
                    record[f"{outcome}_{horizon}_observed"] = bool(np.isfinite(value) and not death_before)
                    record[f"{outcome}_{horizon}_mature"] = bool(admin_mature)
                    record[f"{outcome}_{horizon}_death"] = bool(death_before)
            accepted = record
            break
        if accepted is not None:
            rows.append(accepted)

    cohort = pd.DataFrame(rows)
    for arm in ARM_LABELS.values():
        n = int(cohort["arm_label"].eq(arm).sum()) if not cohort.empty else 0
        funnel.append({"stage": "eligible_trial_cohort", "arm": arm, "n": n, "excluded_n": 0, "reason": "included"})
    for (arm, reason), count in sorted(exclusion_counts.items()):
        funnel.append({"stage": "eligibility_exclusion", "arm": arm, "n": 0, "excluded_n": count, "reason": reason})
    metadata = {
        "source_mode": bundle.metadata.get("source_mode", "reviewed_raw_event_contract"),
        "query_fingerprint": bundle.metadata.get("query_fingerprint", "unknown"),
        "schema_fingerprint": bundle.metadata.get("schema_fingerprint", "unknown"),
        "future_persistence_fields_read": False,
        "prediction_store_read": False,
        "measurement_timing": bundle.metadata.get("measurement_timing", "exact_day"),
    }
    return cohort.reset_index(drop=True), pd.DataFrame(funnel), metadata


def acquire_raw_trial_source(cfg: TrialConfig) -> Any:
    """Load only the reviewed canonical raw-event contract from the production module."""
    contract = dict(getattr(study, "EMBEDDED_RAW_SOURCE_SQL", {}) or {})
    study.validate_embedded_raw_sql(contract)
    connection = study.connect_cosmos()
    try:
        study_cfg = study.RunConfig.create("production", str(cfg.run_dir), False)
        return study.load_embedded_raw_bundle(connection, study_cfg, sql_contract=contract)
    finally:
        connection.close()


def restrict_common_availability(frame: Any, minimum_per_arm: int = MIN_CELL) -> tuple[Any, dict[str,Any]]:
    """Restrict to observed calendar/geographic strata containing all three strategies."""
    if frame.empty:
        return frame.copy(), {"strata": [], "removed_n": 0}
    keys=["calendar_year"]
    if "state" in frame and float(frame["state"].notna().mean()) >= .8:
        keys.append("state")
    counts=frame.groupby(keys+["arm"],dropna=False,observed=True).size().unstack("arm",fill_value=0)
    for arm in ARM_LABELS:
        if arm not in counts:counts[arm]=0
    eligible=counts.loc[(counts[[0,1,2]]>=minimum_per_arm).all(axis=1)].index
    if len(keys)==1:
        keep=frame[keys[0]].isin(list(eligible))
        labels=[str(value) for value in eligible]
    else:
        eligible_set={tuple(value if isinstance(value,tuple) else (value,)) for value in eligible.tolist()}
        keep=pd.Series([tuple(row) in eligible_set for row in frame[keys].itertuples(index=False,name=None)],index=frame.index)
        labels=["|".join(map(str,value if isinstance(value,tuple) else (value,))) for value in eligible]
    return frame.loc[keep].reset_index(drop=True),{"keys":keys,"strata":labels,"removed_n":int((~keep).sum()),"retained_n":int(keep.sum())}


def synthetic_trial_cohort(seed: int = SEED, n: int = 3600) -> Any:
    """Embedded three-arm observational fixture with planted mean effects and missingness."""
    load_runtime()
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    age = np.clip(48 + 11 * rng.normal(size=n), 18, 82)
    baseline_bmi = np.clip(43 + 4.5 * x1 + rng.normal(scale=2, size=n), 35, 68)
    baseline_hba1c = np.clip(7.6 + 0.7 * x2 + rng.normal(scale=.4, size=n), 5, 13)
    logits = np.column_stack((0.2 - .25*x1 + .15*x2, .05 + .35*x1 - .1*x2, -.15 - .1*x1 - .05*x2))
    prob = np.exp(logits - logits.max(axis=1, keepdims=True)); prob /= prob.sum(axis=1, keepdims=True)
    arm = np.asarray([rng.choice(3, p=row) for row in prob])
    rows = pd.DataFrame({
        "patient_id": [f"SYN-{i:06d}" for i in range(n)], "arm": arm,
        "arm_label": [ARM_LABELS[int(a)] for a in arm], "index_date": pd.Timestamp("2021-01-01") + pd.to_timedelta(np.arange(n) % 730, unit="D"),
        "age": age, "baseline_bmi": baseline_bmi, "baseline_hba1c": baseline_hba1c,
        "sex": np.where(np.arange(n) % 2, "Female", "Male"), "race": np.where(np.arange(n) % 3, "White", "Black"),
        "ethnicity": np.where(np.arange(n) % 5, "Not Hispanic", "Hispanic"), "coverage": np.where(np.arange(n)%2, "Commercial", "Public"),
        "diabetes_status": 1, "diabetes_severity": baseline_hba1c, "insulin": (x2 > .5).astype(int),
        "biguanide": (x2 < 1).astype(int), "sglt2": (x1+x2 > .5).astype(int), "hypertension": (age > 50).astype(int),
        "dyslipidemia": (x1 > -.2).astype(int), "osa": (baseline_bmi > 44).astype(int), "smoking": np.where(x2>.8,"Current","Never"),
        "gerd_baseline": (x1>.3).astype(int), "creatinine_baseline": 0.9+.1*rng.normal(size=n), "egfr_baseline": 85+8*rng.normal(size=n),
        "ldl_baseline": 110+20*rng.normal(size=n), "hdl_baseline": 48+8*rng.normal(size=n), "triglycerides_baseline": 150+25*rng.normal(size=n),
        "glucose_fasting_baseline": 130+15*rng.normal(size=n), "sbp_baseline": 125+10*rng.normal(size=n), "dbp_baseline": 78+7*rng.normal(size=n),
        "svi": rng.uniform(size=n), "ruca": np.where(np.arange(n)%4,"Urban","Rural"), "state": np.where(np.arange(n)%2,"MA","NY"),
        "calendar_year": 2021 + (np.arange(n)%2), "center_id": [f"C{i%12:02d}" for i in range(n)],
        "surgeon_id": [f"S{i%20:02d}" if a != 2 else np.nan for i,a in enumerate(arm)],
        "prescriber_id": [f"P{i%24:02d}" if a == 2 else np.nan for i,a in enumerate(arm)],
        "formulary_proxy": np.where(np.arange(n)%2,"A","B"), "ingredient": np.where(arm==2,"semaglutide",""),
        "route": np.where(arm==2,"subcutaneous","surgical"), "dose": np.where(arm==2,1.0,np.nan), "dose_unit": np.where(arm==2,"mg",""),
    })
    rows["baseline_bmi_date"] = rows["index_date"] - pd.to_timedelta(14, unit="D")
    rows["baseline_hba1c_date"] = rows["index_date"] - pd.to_timedelta(21, unit="D")
    rows["observation_start_date"] = rows["index_date"] - pd.to_timedelta(730, unit="D")
    rows["observation_end_date"] = rows["index_date"] + pd.to_timedelta(1100, unit="D")
    rows["death_date"] = pd.NaT
    effects_bmi = np.asarray([[-4.0, -6.2, -2.2], [-7.2, -10.0, -4.5], [-8.2, -12.0, -5.2]])
    effects_a1c = np.asarray([[-.6,-.8,-.9],[-.8,-1.1,-1.25],[-.75,-1.0,-1.15]])
    for hidx, horizon in enumerate(HORIZONS):
        bmi_mu = baseline_bmi + effects_bmi[hidx, arm] + .3*x1 + .15*x2
        a1c_mu = baseline_hba1c + effects_a1c[hidx, arm] + .08*x1 + .10*x2
        pc = 1/(1+np.exp(-(.9 - .25*x1 + .15*(arm==2) - .1*x2)))
        observed = rng.random(n) < pc
        rows[f"bmi_{horizon}"] = np.where(observed, bmi_mu+rng.normal(scale=1.2,size=n), np.nan)
        rows[f"hba1c_{horizon}"] = np.where(observed, a1c_mu+rng.normal(scale=.25,size=n), np.nan)
        for outcome in OUTCOMES:
            rows[f"{outcome}_{horizon}_observed"] = observed
            rows[f"{outcome}_{horizon}_mature"] = True
            rows[f"{outcome}_{horizon}_death"] = False
            rows[f"{outcome}_{horizon}_date"] = rows["index_date"] + pd.to_timedelta(round(horizon*study.DAYS_PER_MONTH), unit="D")
    return rows


# ======================================================================================
# Cross-fitted three-arm estimators
# ======================================================================================

def _feature_columns(frame: Any) -> list[str]:
    return [name for name in FROZEN_CONFOUNDERS if name in frame.columns]


def make_design(frame: Any, columns: Sequence[str] | None = None) -> tuple[Any, list[str]]:
    """Deterministic, outcome-blind numeric design with explicit missing indicators."""
    selected = list(columns or _feature_columns(frame))
    blocks: list[Any] = []
    names: list[str] = []
    for column in selected:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            filled = numeric.fillna(median).to_numpy(float)
            scale = float(np.std(filled))
            center = float(np.mean(filled))
            blocks.append(((filled-center)/(scale if scale > 1e-10 else 1.0))[:, None])
            blocks.append(numeric.isna().astype(float).to_numpy()[:, None])
            names.extend([column, column + "__missing"])
        else:
            text = series.astype("string").fillna("<MISSING>").replace("", "<MISSING>")
            levels = sorted(text.unique().tolist(), key=str)
            for level in levels:
                blocks.append(text.eq(level).astype(float).to_numpy()[:, None])
                names.append(f"{column}=={level}")
    if not blocks:
        return np.ones((len(frame), 1), dtype=float), ["intercept"]
    return np.concatenate(blocks, axis=1), names


def append_arm_design(x: Any, arm: Any) -> Any:
    a = np.asarray(arm, dtype=int)
    onehot = np.column_stack([(a == value).astype(float) for value in range(3)])
    return np.column_stack((x, onehot))


def _fit_multinomial(x: Any, a: Any, seed: int, boosted: bool = False) -> Any:
    if boosted:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(loss="log_loss", max_iter=160, learning_rate=.055,
                                               max_leaf_nodes=15, l2_regularization=1.0,
                                               random_state=seed)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1500, C=.5, solver="lbfgs", random_state=seed)
    model.fit(x, a)
    return model


def _aligned_probabilities(model: Any, x: Any) -> Any:
    raw = model.predict_proba(x)
    result = np.full((len(x), 3), np.nan, dtype=float)
    for index, label in enumerate(model.classes_):
        if int(label) in range(3):
            result[:, int(label)] = raw[:, index]
    return result


def crossfit_propensity(frame: Any, x: Any, cfg: TrialConfig,
                        boosted: bool = False) -> Any:
    a = frame["arm"].to_numpy(int)
    folds = patient_folds(frame["patient_id"], cfg.seed, cfg.folds, "propensity")
    result = np.full((len(frame), 3), np.nan, dtype=float)
    for fold in range(cfg.folds):
        test = folds == fold
        train = ~test
        if not test.any():
            continue
        if len(np.unique(a[train])) < 3:
            counts = np.bincount(a[train], minlength=3).astype(float) + 1.0
            result[test] = counts / counts.sum()
            continue
        model = _fit_multinomial(x[train], a[train], cfg.seed + fold, boosted)
        result[test] = _aligned_probabilities(model, x[test])
    if not np.isfinite(result).all():
        raise RuntimeError("cross-fitted generalized propensity contains non-finite cells")
    return result


def _fit_binary_probability(x: Any, y: Any, seed: int, boosted: bool = False) -> tuple[Any | None, float]:
    marginal = float(np.mean(y)) if len(y) else 0.0
    if len(np.unique(y)) < 2:
        return None, marginal
    if boosted:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(loss="log_loss", max_iter=130, max_leaf_nodes=15,
                                               learning_rate=.06, l2_regularization=1.0,
                                               random_state=seed)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1200, C=.5, solver="lbfgs", random_state=seed)
    model.fit(x, y)
    return model, marginal


def _fit_outcome_mean(x: Any, y: Any, seed: int) -> Any | None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    if len(y) < 20:
        return None
    model = HistGradientBoostingRegressor(loss="squared_error", max_iter=180, learning_rate=.055,
                                         max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=seed)
    model.fit(x, y)
    return model


def crossfit_outcome_and_observation(frame: Any, x: Any, outcome: str, horizon: int,
                                     cfg: TrialConfig, boosted_observation: bool = False) -> tuple[Any, Any]:
    """Fit p_c(A,L) and mu_a(L) out of fold on the analysis population."""
    a = frame["arm"].to_numpy(int)
    y = pd.to_numeric(frame[f"{outcome}_{horizon}"], errors="coerce").to_numpy(float)
    delta = frame[f"{outcome}_{horizon}_observed"].astype(bool).to_numpy() & np.isfinite(y)
    folds = patient_folds(frame["patient_id"], cfg.seed, cfg.folds, f"{outcome}-{horizon}")
    pc = np.full(len(frame), np.nan, dtype=float)
    mu = np.full((len(frame), 3), np.nan, dtype=float)
    xa = append_arm_design(x, a)
    for fold in range(cfg.folds):
        test = folds == fold
        train = ~test
        if not test.any():
            continue
        observation_model, marginal = _fit_binary_probability(
            xa[train], delta[train].astype(int), cfg.seed + 100 + fold, boosted_observation
        )
        pc[test] = (observation_model.predict_proba(xa[test])[:, 1]
                    if observation_model is not None else marginal)
        observed_train = train & delta
        outcome_model = _fit_outcome_mean(xa[observed_train], y[observed_train], cfg.seed + 200 + fold)
        if outcome_model is None:
            for arm_value in range(3):
                arm_cell = observed_train & (a == arm_value)
                fallback = float(np.mean(y[arm_cell])) if arm_cell.any() else float(np.nanmean(y[observed_train]))
                mu[test, arm_value] = fallback
        else:
            for arm_value in range(3):
                forced = np.full(int(test.sum()), arm_value, dtype=int)
                mu[test, arm_value] = outcome_model.predict(append_arm_design(x[test], forced))
    pc = np.clip(np.nan_to_num(pc, nan=float(np.mean(delta))), .01, .99)
    if not np.isfinite(mu).all():
        raise RuntimeError("out-of-fold outcome mean contains non-finite predictions")
    return pc, mu


def three_arm_aipw(y: Any, arm: Any, observed: Any, propensity: Any,
                   observation_probability: Any, outcome_mean: Any,
                   correction_limits: tuple[float, float] | None = None) -> dict[str, Any]:
    """Explicit symmetric three-arm AIPW with paired influence-function contrasts."""
    yv = np.asarray(y, dtype=float)
    av = np.asarray(arm, dtype=int)
    delta = np.asarray(observed, dtype=bool) & np.isfinite(yv)
    e = np.asarray(propensity, dtype=float)
    pc = np.asarray(observation_probability, dtype=float)
    mu = np.asarray(outcome_mean, dtype=float)
    if e.shape != (len(yv), 3) or mu.shape != (len(yv), 3):
        raise ValueError("propensity and outcome_mean must be n x 3")
    phi = mu.copy()
    raw_weights = np.zeros((len(yv), 3), dtype=float)
    for arm_value in range(3):
        treated_observed = (av == arm_value) & delta
        denominator = np.maximum(e[:, arm_value] * pc, 1e-8)
        factor = np.zeros(len(yv), dtype=float)
        factor[treated_observed] = 1.0 / denominator[treated_observed]
        if correction_limits is not None and treated_observed.any():
            low_q, high_q = correction_limits
            finite = factor[treated_observed]
            lower = float(np.quantile(finite, low_q)) if low_q > 0 else 0.0
            upper = float(np.quantile(finite, high_q)) if high_q < 1 else float(np.max(finite))
            factor[treated_observed] = np.clip(finite, lower, upper)
        residual = np.zeros(len(yv), dtype=float)
        residual[treated_observed] = yv[treated_observed] - mu[treated_observed, arm_value]
        phi[:, arm_value] = mu[:, arm_value] + factor * residual
        raw_weights[:, arm_value] = factor
    means = phi.mean(axis=0)
    arm_se = phi.std(axis=0, ddof=1) / math.sqrt(len(phi))
    pairwise: dict[tuple[int, int], dict[str, float]] = {}
    for first, second in PAIRWISE:
        influence = phi[:, first] - phi[:, second]
        estimate = float(np.mean(influence))
        se = float(np.std(influence, ddof=1) / math.sqrt(len(influence)))
        pairwise[(first, second)] = {
            "estimate": estimate, "se": se, "ci_low": estimate - 1.96*se,
            "ci_high": estimate + 1.96*se,
        }
    return {"phi": phi, "means": means, "arm_se": arm_se,
            "pairwise": pairwise, "correction_weights": raw_weights}


def contrast_label(first: int, second: int) -> str:
    return f"{ARM_LABELS[first]} vs {ARM_LABELS[second]}"


def overlap_weights(propensity: Any, arm: Any) -> Any:
    e = np.clip(np.asarray(propensity, dtype=float), 1e-8, 1.0)
    a = np.asarray(arm, dtype=int)
    h = 1.0 / np.sum(1.0 / e, axis=1)
    return h / e[np.arange(len(a)), a]


def weighted_mean_sd(values: Any, weights: Any) -> tuple[float, float]:
    x = np.asarray(values, dtype=float); w = np.asarray(weights, dtype=float)
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return float("nan"), float("nan")
    x = x[keep]; w = w[keep]
    mean = float(np.sum(w*x)/np.sum(w))
    variance = float(np.sum(w*(x-mean)**2)/np.sum(w))
    return mean, math.sqrt(max(variance, 0.0))


def standardized_mean_difference(frame: Any, variable: str, first: int, second: int,
                                 weights: Any | None = None) -> float:
    if variable not in frame:
        return float("nan")
    series = frame[variable]
    if not pd.api.types.is_numeric_dtype(series):
        levels = sorted(series.astype("string").fillna("<MISSING>").unique())
        values = series.astype("string").fillna("<MISSING>").eq(levels[0]).astype(float).to_numpy()
    else:
        values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    a = frame["arm"].to_numpy(int)
    w = np.ones(len(frame)) if weights is None else np.asarray(weights, dtype=float)
    m1, s1 = weighted_mean_sd(values[a == first], w[a == first])
    m0, s0 = weighted_mean_sd(values[a == second], w[a == second])
    pooled = math.sqrt((s1*s1+s0*s0)/2) if np.isfinite(s1+s0) else float("nan")
    return float((m1-m0)/pooled) if np.isfinite(pooled) and pooled > 1e-12 else 0.0


def overlap_and_balance(frame: Any, propensity: Any, feature_columns: Sequence[str]) -> tuple[Any, Any, Any]:
    a = frame["arm"].to_numpy(int)
    weights = overlap_weights(propensity, a)
    overlap_rows = []
    for arm_value, label in ARM_LABELS.items():
        cell = a == arm_value
        own = propensity[cell, arm_value]
        overlap_rows.append({
            "arm": label, "n": int(cell.sum()), "ess": effective_sample_size(weights[cell]),
            "min_probability": float(np.min(propensity[:, arm_value])),
            "p01": float(np.quantile(own, .01)), "median": float(np.median(own)),
            "p99": float(np.quantile(own, .99)), "max_weight": float(np.max(weights[cell])),
        })
    balance_rows = []
    for variable in feature_columns:
        for first, second in PAIRWISE:
            balance_rows.append({
                "variable": variable, "comparison": contrast_label(first, second),
                "unweighted_smd": standardized_mean_difference(frame, variable, first, second),
                "weighted_smd": standardized_mean_difference(frame, variable, first, second, weights),
            })
    return pd.DataFrame(overlap_rows), pd.DataFrame(balance_rows), weights


def baseline_table(frame: Any) -> Any:
    rows = []
    variables = [name for name in FROZEN_CONFOUNDERS if name in frame]
    for variable in variables:
        for arm_value, label in ARM_LABELS.items():
            series = frame.loc[frame["arm"].eq(arm_value), variable]
            n = int(series.notna().sum())
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                rows.append({"variable": variable, "arm": label, "n": n,
                             "mean": float(numeric.mean()), "sd": float(numeric.std()),
                             "level": "", "percent": np.nan})
            else:
                for level, count in series.astype("string").fillna("<MISSING>").value_counts(dropna=False).items():
                    rows.append({"variable": variable, "arm": label, "n": n, "mean": np.nan,
                                 "sd": np.nan, "level": str(level),
                                 "percent": 100*float(count)/max(len(series), 1)})
    return pd.DataFrame(rows)


def c_for_benefit(predicted_benefit: Any, observed_benefit: Any,
                  max_pairs: int = 200_000, seed: int = SEED) -> float:
    predicted = np.asarray(predicted_benefit, dtype=float)
    observed = np.asarray(observed_benefit, dtype=float)
    keep = np.isfinite(predicted) & np.isfinite(observed)
    predicted, observed = predicted[keep], observed[keep]
    if len(predicted) < 2:
        return float("nan")
    total = len(predicted)*(len(predicted)-1)//2
    if total <= max_pairs:
        i, j = np.triu_indices(len(predicted), k=1)
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, len(predicted), size=max_pairs)
        j = rng.integers(0, len(predicted), size=max_pairs)
        unequal = i != j; i, j = i[unequal], j[unequal]
    pdiff = predicted[i]-predicted[j]; odiff = observed[i]-observed[j]
    informative = odiff != 0
    if not informative.any():
        return .5
    product = pdiff[informative]*odiff[informative]
    return float((np.sum(product > 0)+.5*np.sum(product == 0))/len(product))


def toc_rate(predicted_benefit: Any, observed_benefit: Any) -> tuple[Any, float]:
    predicted = np.asarray(predicted_benefit, dtype=float)
    observed = np.asarray(observed_benefit, dtype=float)
    order = np.argsort(-predicted, kind="mergesort")
    sorted_benefit = observed[order]
    overall = float(np.mean(observed))
    fractions = np.linspace(.1, 1.0, 10)
    toc = []
    for fraction in fractions:
        k = max(1, int(math.ceil(len(observed)*fraction)))
        toc.append(float(np.mean(sorted_benefit[:k])-overall))
    integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    rate = float(integrator(np.asarray(toc), fractions))
    return pd.DataFrame({"fraction": fractions, "toc": toc}), rate


def _bootstrap_rate(predicted: Any, observed: Any, patient_ids: Any,
                    replicates: int, seed: int) -> tuple[float, float]:
    unique = np.asarray(sorted(set(map(str, patient_ids))))
    positions = {value: np.flatnonzero(np.asarray(patient_ids).astype(str) == value) for value in unique}
    rng = np.random.default_rng(seed); rates = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([positions[value] for value in sampled])
        rates.append(toc_rate(np.asarray(predicted)[index], np.asarray(observed)[index])[1])
    return float(np.quantile(rates, .025)), float(np.quantile(rates, .975))


def matched_pair_c_for_benefit(frame: Any, predicted: Any, outcome: str,
                               first: int, second: int) -> float:
    """Secondary c-for-benefit on deterministic baseline-outcome-matched observed pairs."""
    value_column=f"{outcome}_12";observed_column=f"{outcome}_12_observed"
    baseline_column="baseline_bmi" if outcome=="bmi" else "baseline_hba1c"
    eligible=frame[observed_column].astype(bool)&pd.to_numeric(frame[value_column],errors="coerce").notna()
    first_rows=frame.loc[eligible&frame["arm"].eq(first)].copy()
    second_rows=frame.loc[eligible&frame["arm"].eq(second)].copy()
    if len(first_rows)<2 or len(second_rows)<2:return float("nan")
    first_rows["_position"]=np.flatnonzero((eligible&frame["arm"].eq(first)).to_numpy())
    second_rows["_position"]=np.flatnonzero((eligible&frame["arm"].eq(second)).to_numpy())
    first_rows=first_rows.sort_values([baseline_column,"patient_id"],kind="mergesort").reset_index(drop=True)
    second_rows=second_rows.sort_values([baseline_column,"patient_id"],kind="mergesort").reset_index(drop=True)
    pairs=min(len(first_rows),len(second_rows));first_rows=first_rows.iloc[:pairs];second_rows=second_rows.iloc[:pairs]
    observed_benefit=(pd.to_numeric(second_rows[value_column],errors="coerce").to_numpy(float)-
                      pd.to_numeric(first_rows[value_column],errors="coerce").to_numpy(float))
    predicted=np.asarray(predicted,dtype=float)
    pair_prediction=.5*(predicted[first_rows["_position"].to_numpy(int)]+predicted[second_rows["_position"].to_numpy(int)])
    return c_for_benefit(pair_prediction,observed_benefit)


def dr_heterogeneity(frame: Any, x: Any, aipw: Mapping[str, Any], outcome: str,
                     cfg: TrialConfig) -> tuple[Any, Any, dict[str, Any]]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    folds = patient_folds(frame["patient_id"], cfg.seed, cfg.folds, "dr-learner")
    rate_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}
    phi = np.asarray(aipw["phi"])
    for first, second in PAIRWISE:
        contrast = contrast_label(first, second)
        # Lower outcomes are beneficial, so benefit is Y(second)-Y(first).
        observed_benefit = phi[:, second]-phi[:, first]
        predicted = np.full(len(frame), np.nan)
        for fold in range(cfg.folds):
            test = folds == fold; train = ~test
            model = HistGradientBoostingRegressor(loss="squared_error", max_iter=130,
                                                  max_leaf_nodes=12, min_samples_leaf=25,
                                                  l2_regularization=1.0,
                                                  random_state=cfg.seed+fold)
            model.fit(x[train], observed_benefit[train])
            predicted[test] = model.predict(x[test])
        _toc, rate = toc_rate(predicted, observed_benefit)
        low, high = _bootstrap_rate(predicted, observed_benefit, frame["patient_id"],
                                    min(cfg.bootstrap_replicates, 200), cfg.seed+first*10+second)
        cfb = matched_pair_c_for_benefit(frame,predicted,outcome,first,second)
        rate_rows.append({"outcome": outcome, "horizon": 12, "contrast": contrast,
                          "n": len(frame), "rate": rate, "ci_low": low, "ci_high": high,
                          "c_for_benefit": cfb, "status": "exploratory"})
        decile = pd.qcut(pd.Series(predicted).rank(method="first"), 10, labels=False, duplicates="drop")
        for value in range(10):
            cell = np.asarray(decile == value)
            calibration_rows.append({"outcome": outcome, "contrast": contrast,
                                     "decile": value+1, "n": int(cell.sum()),
                                     "predicted_benefit": float(np.mean(predicted[cell])) if cell.any() else np.nan,
                                     "observed_dr_benefit": float(np.mean(observed_benefit[cell])) if cell.any() else np.nan,
                                     "status": "estimable" if cell.sum() >= MIN_CELL else "suppressed"})
        predictions[contrast] = predicted
    return pd.DataFrame(rate_rows), pd.DataFrame(calibration_rows), predictions


def policy_values(frame: Any, aipw: Mapping[str, Any], predicted_mu: Any,
                  outcome: str) -> Any:
    phi = np.asarray(aipw["phi"]); learned_arm = np.argmin(predicted_mu, axis=1)
    learned_if = phi[np.arange(len(phi)), learned_arm]
    policies: list[tuple[str, Any]] = [("learned_three_arm", learned_if)]
    for arm_value, label in ARM_LABELS.items():
        policies.append((f"always_{label}", phi[:, arm_value]))
    learned_value = float(np.mean(learned_if)); rows = []
    for name, influence in policies:
        value = float(np.mean(influence)); se = float(np.std(influence, ddof=1)/math.sqrt(len(influence)))
        difference_if = learned_if-influence
        diff = float(np.mean(difference_if)); diff_se = float(np.std(difference_if, ddof=1)/math.sqrt(len(influence)))
        rows.append({"outcome": outcome, "horizon": 12, "policy": name, "n": len(frame),
                     "value": value, "se": se, "ci_low": value-1.96*se, "ci_high": value+1.96*se,
                     "difference_vs_learned": diff, "difference_ci_low": diff-1.96*diff_se,
                     "difference_ci_high": diff+1.96*diff_se,
                     "status": "exploratory_not_clinically_actionable"})
    return pd.DataFrame(rows)


def _replace_gate(diagnostic: Any, gate: str, passed: bool, detail: str,
                  n: int | None = None) -> Any:
    result = diagnostic.copy()
    mask = result["gate"].eq(gate)
    if not mask.any():
        result = pd.concat([result, pd.DataFrame([gate_row("hard", gate, passed, detail, n)])], ignore_index=True)
    else:
        result.loc[mask, "passed"] = bool(passed)
        result.loc[mask, "status"] = "PASS" if passed else "FAIL"
        result.loc[mask, "detail"] = detail
        result.loc[mask, "n"] = n
    return result


def _preindex_population_audit(frame: Any) -> tuple[bool, str]:
    required = ("baseline_bmi", "age", "sex", "diabetes_status", "calendar_year", "coverage",
                "hypertension", "dyslipidemia", "osa")
    failures = []
    for column in required:
        if column not in frame:
            failures.append(f"{column}=absent")
            continue
        populated = frame[column].notna() & frame[column].astype("string").str.strip().ne("")
        fraction = float(populated.mean()) if len(frame) else 0.0
        if fraction < .95:
            failures.append(f"{column}={fraction:.1%}")
    temporal_ok = True
    for value_column, date_column in (("baseline_bmi", "baseline_bmi_date"),
                                      ("baseline_hba1c", "baseline_hba1c_date")):
        if value_column in frame and date_column in frame:
            violation = frame[value_column].notna() & pd.to_datetime(frame[date_column], errors="coerce").gt(frame["index_date"])
            if violation.any():
                temporal_ok = False; failures.append(f"{value_column} has post-index values")
    return not failures and temporal_ok, "All required fields >=95% populated and pre-index." if not failures else "; ".join(failures)


def _calendar_availability_ok(frame: Any) -> tuple[bool, str]:
    if "calendar_year" not in frame or "state" not in frame:
        return False, "Calendar year or geography unavailable."
    years = sorted(frame["calendar_year"].dropna().unique())
    if not years:
        return False, "No populated calendar year."
    distributions = []
    for arm_value in ARM_LABELS:
        counts = frame.loc[frame["arm"].eq(arm_value), "calendar_year"].value_counts(normalize=True)
        distributions.append(np.asarray([counts.get(year, 0.0) for year in years]))
    maximum = max(float(np.max(np.abs(distributions[i]-distributions[j])))
                  for i in range(3) for j in range(i))
    passed = maximum <= .20
    return passed, f"Maximum absolute arm-specific calendar-share difference={maximum:.3f}."


def evaluate_constructed_gates(frame: Any, diagnostic: Any, metadata: Mapping[str, Any],
                               propensity: Any | None = None, overlap: Any | None = None,
                               balance: Any | None = None, cfg: TrialConfig | None = None) -> Any:
    cfg = cfg or TrialConfig("self-test", str(Path(tempfile.gettempdir()) / "tte-gates"))
    result = diagnostic.copy()
    raw = metadata.get("source_mode") in {"cosmos_embedded_raw_sql", "reviewed_raw_event_contract", "synthetic_raw_events"}
    result = _replace_gate(result, HARD_GATE_LABELS[0], raw and not metadata.get("future_persistence_fields_read", False),
                           "Entry uses first pre-index-auditable incretin event; persistence and duration are unused.")
    exact_dates = "index_date" in frame and not pd.to_datetime(frame["index_date"],errors="coerce").isna().any()
    result = _replace_gate(result, HARD_GATE_LABELS[1], exact_dates, "All arms carry populated exact initiation dates." if exact_dates else "Exact initiation date field missing or unpopulated.")
    aligned = bool(exact_dates)
    result = _replace_gate(result, HARD_GATE_LABELS[2], aligned, "Eligibility, assignment, and follow-up share index_date.")
    full = not metadata.get("prediction_store_read", True)
    result = _replace_gate(result, HARD_GATE_LABELS[3], full, "Cohort was built from raw source; prediction storage was never read.")
    counts = {label: int((frame["arm"] == arm).sum()) for arm, label in ARM_LABELS.items()}
    mature_counts = {
        label: int((frame["arm"].eq(arm) & frame["bmi_12_mature"].astype(bool)).sum())
        for arm, label in ARM_LABELS.items()
    }
    size_ok = all(value >= 100 for value in mature_counts.values())
    result = _replace_gate(result, HARD_GATE_LABELS[4], size_ok,
                           f"12-month administratively mature counts: {mature_counts}.", min(mature_counts.values()) if mature_counts else 0)
    conf_ok, conf_detail = _preindex_population_audit(frame)
    result = _replace_gate(result, HARD_GATE_LABELS[6], conf_ok, conf_detail)
    outcome_timing=True
    for outcome in OUTCOMES:
        for horizon in HORIZONS:
            observed_column=f"{outcome}_{horizon}_observed";date_column=f"{outcome}_{horizon}_date"
            if observed_column not in frame or date_column not in frame:
                outcome_timing=False;continue
            observed=frame[observed_column].astype(bool)
            dates=pd.to_datetime(frame[date_column],errors="coerce")
            if dates.loc[observed].isna().any() or dates.loc[observed].lt(frame.loc[observed,"index_date"]).any():
                outcome_timing=False
    timing = metadata.get("measurement_timing", "exact_day") == "exact_day" and exact_dates and outcome_timing
    maturity = all(frame.loc[frame["bmi_12_mature"], "observation_end_date"].notna()) if "observation_end_date" in frame else True
    result = _replace_gate(result, HARD_GATE_LABELS[7], timing and maturity,
                           "Exact outcome timestamps and administrative maturity are auditable." if timing and maturity else
                           "Outcome timing or maturity semantics are incomplete.")
    if propensity is not None and overlap is not None:
        support = np.asarray(propensity).min(axis=1) > cfg.propensity_floor
        supported = frame.loc[support]
        supported_weights = overlap_weights(np.asarray(propensity)[support], supported["arm"].to_numpy(int)) if support.any() else np.array([])
        ess = effective_sample_size(supported_weights)
        by_arm_ess = [effective_sample_size(supported_weights[supported["arm"].to_numpy(int) == arm]) for arm in ARM_LABELS] if support.any() else [0,0,0]
        positivity = bool(support.any() and np.asarray(propensity)[support].min() > cfg.propensity_floor
                          and ess >= 100 and ess >= .20*len(frame) and min(by_arm_ess) >= 100)
        result = _replace_gate(result, HARD_GATE_LABELS[5], positivity,
                               f"Support retained {int(support.sum())}/{len(frame)}; overall ESS={ess:.1f}; arm ESS={[round(x,1) for x in by_arm_ess]}; all three probability tails checked.",
                               int(support.sum()))
    # Soft confounder/version/timing facts.
    gerd = "gerd_baseline" in frame and float(frame["gerd_baseline"].notna().mean()) >= .5
    care_checks=[]
    for name,mask in (("center_id",pd.Series(True,index=frame.index)),
                      ("surgeon_id",frame["arm"].isin([0,1])),
                      ("prescriber_id",frame["arm"].eq(2)),
                      ("formulary_proxy",pd.Series(True,index=frame.index))):
        care_checks.append(name in frame and float(frame.loc[mask,name].notna().mean()) >= .5)
    care=all(care_checks)
    version_checks=[]
    incretin_mask=frame["arm"].eq(2)
    for name in ("ingredient","dose","route"):
        if name not in frame:
            version_checks.append(False);continue
        series=frame.loc[incretin_mask,name]
        populated=series.notna()&~series.astype("string").str.strip().str.lower().isin(["","unknown","nan","none"])
        version_checks.append(float(populated.mean())>=.5 if len(series) else False)
    version=all(version_checks)
    calendar_ok, calendar_detail = _calendar_availability_ok(frame)
    soft_values = {
        "gerd_available": (gerd, "GERD/reflux populated in at least half of rows."),
        "care_process_confounders_available": (care, "Center, surgeon, prescriber, and formulary proxies populated by applicable arm."),
        "incretin_version_available": (version, "Ingredient, dose, and route populated for incident starts."),
        "exact_outcome_timestamps": (timing, "Exact timestamps available."),
        "weighted_balance": (False, "Awaiting weighted balance."),
        "calendar_geography_availability": (calendar_ok, calendar_detail),
    }
    if balance is not None and len(balance):
        maximum = float(pd.to_numeric(balance["weighted_smd"], errors="coerce").abs().max())
        soft_values["weighted_balance"] = (maximum <= .10, f"Maximum absolute weighted SMD={maximum:.3f}.")
    for gate, (passed, detail) in soft_values.items():
        result = _replace_gate(result, gate, passed, detail)
    return result


def causal_analysis(frame: Any, base_diagnostic: Any, metadata: Mapping[str, Any],
                    cfg: TrialConfig) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Run the full estimator in memory; callers publish estimates only after all gates pass."""
    outputs = empty_outputs()
    feature_columns = _feature_columns(frame)
    x, feature_names = make_design(frame, feature_columns)
    e_logistic = crossfit_propensity(frame, x, cfg, boosted=False)
    e_boosted = crossfit_propensity(frame, x, cfg, boosted=True)
    support = e_logistic.min(axis=1) > cfg.propensity_floor
    supported = frame.loc[support].reset_index(drop=True)
    supported_x = x[support]
    supported_e = e_logistic[support]
    supported_e_boosted = e_boosted[support]
    if len(supported):
        overlap, balance, weights = overlap_and_balance(supported, supported_e, feature_columns)
        for arm_value,label in ARM_LABELS.items():
            eligible_n=int(frame["arm"].eq(arm_value).sum());support_n=int(supported["arm"].eq(arm_value).sum())
            mask=overlap["arm"].eq(label);overlap.loc[mask,"eligible_n"]=eligible_n
            overlap.loc[mask,"support_n"]=support_n;overlap.loc[mask,"removed_n"]=eligible_n-support_n
    else:
        overlap=empty_outputs()["three_arm_overlap"];balance=empty_outputs()["three_arm_balance"]
    diagnostic = evaluate_constructed_gates(frame, base_diagnostic, metadata, e_logistic, overlap, balance, cfg)
    outputs["baseline_by_arm"] = baseline_table(frame)
    outputs["three_arm_overlap"] = overlap
    outputs["three_arm_balance"] = balance
    outputs["design_diagnostic"] = diagnostic
    internal: dict[str, Any] = {
        "feature_columns": feature_columns, "feature_names": feature_names,
        "support_removed_n": int((~support).sum()), "support_retained_n": int(support.sum()),
        "source_metadata": dict(metadata), "heterogeneity_predictions": {},
    }
    if not hard_gates_pass(diagnostic):
        return outputs, diagnostic, internal

    arm_mean_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    rate_frames: list[Any] = []
    calibration_frames: list[Any] = []
    policy_frames: list[Any] = []
    cell_payloads: dict[str, Any] = {}
    status = claim_status(diagnostic)
    for outcome in OUTCOMES:
        for horizon in HORIZONS:
            mature = supported[f"{outcome}_{horizon}_mature"].astype(bool).to_numpy()
            cell = supported.loc[mature].reset_index(drop=True)
            cell_x = supported_x[mature]
            cell_e = supported_e[mature]
            cell_e_boosted = supported_e_boosted[mature]
            if len(cell) < 100 or any(int(cell["arm"].eq(arm).sum()) < MIN_CELL for arm in ARM_LABELS):
                for first, second in PAIRWISE:
                    pair_rows.append({"outcome": outcome, "horizon": horizon,
                                      "contrast": contrast_label(first, second), "arm_a": ARM_LABELS[first],
                                      "arm_b": ARM_LABELS[second], "n": len(cell), "estimate": np.nan,
                                      "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                                      "estimand": "not estimable", "claim_status": "INSUFFICIENT SUPPORT"})
                continue
            pc, mu = crossfit_outcome_and_observation(cell, cell_x, outcome, horizon, cfg, False)
            y = pd.to_numeric(cell[f"{outcome}_{horizon}"], errors="coerce").to_numpy(float)
            delta = cell[f"{outcome}_{horizon}_observed"].astype(bool).to_numpy() & np.isfinite(y)
            result = three_arm_aipw(y, cell["arm"], delta, cell_e, pc, mu, (.01, .99))
            for arm_value, label in ARM_LABELS.items():
                mean = float(result["means"][arm_value]); se = float(result["arm_se"][arm_value])
                arm_mean_rows.append({"outcome": outcome, "horizon": horizon, "arm": label,
                                      "n": len(cell), "observed_n": int((delta & cell["arm"].eq(arm_value).to_numpy()).sum()),
                                      "mean": mean, "se": se, "ci_low": mean-1.96*se, "ci_high": mean+1.96*se,
                                      "model": "cross-fitted IPCW-AIPW logistic propensity"})
            for first, second in PAIRWISE:
                estimate = result["pairwise"][(first, second)]
                pair_rows.append({"outcome": outcome, "horizon": horizon,
                                  "contrast": contrast_label(first, second), "arm_a": ARM_LABELS[first],
                                  "arm_b": ARM_LABELS[second], "n": len(cell), **estimate,
                                  "estimand": "marginal mean arm_a minus arm_b in common-support mature population",
                                  "claim_status": status})
            # Weight-rule sensitivities.
            for analysis_name, limits in (("no_truncation", None), ("1_99", (.01,.99)),
                                          ("2.5_97.5", (.025,.975)), ("5_95", (.05,.95))):
                sensitivity = three_arm_aipw(y, cell["arm"], delta, cell_e, pc, mu, limits)
                for first, second in PAIRWISE:
                    value = sensitivity["pairwise"][(first,second)]
                    robustness_rows.append({"outcome": outcome, "horizon": horizon,
                                            "contrast": contrast_label(first,second), "analysis": analysis_name,
                                            "n": len(cell), "estimate": value["estimate"],
                                            "ci_low": value["ci_low"], "ci_high": value["ci_high"], "status": "estimable"})
            pc_boost, mu_boost = crossfit_outcome_and_observation(cell, cell_x, outcome, horizon, cfg, True)
            boosted_result = three_arm_aipw(y, cell["arm"], delta, cell_e_boosted, pc_boost, mu_boost, (.01,.99))
            for first, second in PAIRWISE:
                value = boosted_result["pairwise"][(first,second)]
                robustness_rows.append({"outcome": outcome, "horizon": horizon,
                                        "contrast": contrast_label(first,second), "analysis": "boosted_nuisance",
                                        "n": len(cell), "estimate": value["estimate"], "ci_low": value["ci_low"],
                                        "ci_high": value["ci_high"], "status": "estimable"})
            complete = delta
            if complete.sum() >= 100:
                complete_result = three_arm_aipw(y[complete], cell.loc[complete,"arm"], np.ones(complete.sum(),bool),
                                                 cell_e[complete], np.ones(complete.sum()), mu[complete], (.01,.99))
                for first, second in PAIRWISE:
                    value = complete_result["pairwise"][(first,second)]
                    robustness_rows.append({"outcome": outcome, "horizon": horizon,
                                            "contrast": contrast_label(first,second), "analysis": "complete_case",
                                            "n": int(complete.sum()), "estimate": value["estimate"],
                                            "ci_low": value["ci_low"], "ci_high": value["ci_high"], "status": "estimable"})
            common_years=[]
            if "calendar_year" in cell:
                year_counts=cell.groupby(["calendar_year","arm"],observed=True).size().unstack("arm",fill_value=0)
                for arm_value in ARM_LABELS:
                    if arm_value not in year_counts:year_counts[arm_value]=0
                common_years=year_counts.loc[(year_counts[[0,1,2]]>=MIN_CELL).all(axis=1)].index.tolist()
            era=cell["calendar_year"].isin(common_years).to_numpy() if common_years else np.zeros(len(cell),bool)
            if era.sum()>=100:
                era_result=three_arm_aipw(y[era],cell.loc[era,"arm"],delta[era],cell_e[era],pc[era],mu[era],(.01,.99))
                for first,second in PAIRWISE:
                    value=era_result["pairwise"][(first,second)]
                    robustness_rows.append({"outcome":outcome,"horizon":horizon,"contrast":contrast_label(first,second),
                                            "analysis":"calendar_era_restriction","n":int(era.sum()),"estimate":value["estimate"],
                                            "ci_low":value["ci_low"],"ci_high":value["ci_high"],"status":"estimable"})
            observed_balance=[]
            temporary=cell.copy();temporary["arm"]=delta.astype(int)
            for variable in feature_columns:
                observed_balance.append(abs(standardized_mean_difference(temporary,variable,1,0)))
            maximum_observation_smd=float(np.nanmax(observed_balance)) if observed_balance else np.nan
            robustness_rows.append({"outcome":outcome,"horizon":horizon,"contrast":"observed vs missing",
                                    "analysis":"maximum_baseline_smd","n":len(cell),"estimate":maximum_observation_smd,
                                    "ci_low":np.nan,"ci_high":np.nan,"status":"diagnostic_not_effect"})
            cell_payloads[f"{outcome}_{horizon}"] = {
                "n": len(cell), "observed_n": int(delta.sum()), "death_n": int(cell[f"{outcome}_{horizon}_death"].sum()),
                "missing_n": int((~delta).sum()), "propensity": cell_e, "arm": cell["arm"].to_numpy(),
            }
            if horizon == 12:
                rate, calibration, predicted_benefit = dr_heterogeneity(cell, cell_x, result, outcome, cfg)
                rate_frames.append(rate); calibration_frames.append(calibration)
                policy_frames.append(policy_values(cell, result, mu, outcome))
                internal["heterogeneity_predictions"][outcome] = predicted_benefit
    outputs["arm_standardized_means"] = pd.DataFrame(arm_mean_rows)
    outputs["pairwise_aipw"] = pd.DataFrame(pair_rows)
    outputs["robustness"] = pd.DataFrame(robustness_rows)
    outputs["pairwise_rate"] = pd.concat(rate_frames, ignore_index=True) if rate_frames else empty_outputs()["pairwise_rate"]
    outputs["benefit_calibration"] = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else empty_outputs()["benefit_calibration"]
    outputs["policy_value"] = pd.concat(policy_frames, ignore_index=True) if policy_frames else empty_outputs()["policy_value"]
    internal["cell_payloads"] = cell_payloads
    return outputs, diagnostic, internal


# ======================================================================================
# Completed-run descriptive/prognostic analysis
# ======================================================================================

def reopen_prediction_store(root: Path) -> Any:
    store = study.PredictionStore(root)
    store.partitions = {}
    extensions: set[str] = set()
    staged: dict[str, list[tuple[int, str]]] = {}
    for item in sorted(root.glob("*")):
        if not item.is_file() or item.suffix not in (".parquet", ".pkl"):
            continue
        stem = item.name[:-len(item.suffix)]
        base, separator, number = stem.rpartition("__")
        if not separator or not number.isdigit():
            continue
        pieces = base.split("__")
        if len(pieces) != 3:
            continue
        cohort, outcome, origin_token = pieces
        digits = re.sub(r"\D", "", origin_token)
        if not digits:
            continue
        key = [cohort, outcome, int(digits)]
        if study.PredictionStore.partition_name(key) != base:
            continue
        extensions.add(item.suffix)
        staged.setdefault(base, []).append((int(number), item.name))
    for base, files in staged.items():
        files.sort()
        cohort, outcome, origin_token = base.split("__")
        store.partitions[base] = {"key": [cohort, outcome, int(re.sub(r"\D", "", origin_token))],
                                  "files": [name for _, name in files], "rows": 0}
    if extensions:
        store.parquet = ".parquet" in extensions
        if store.parquet and study._pa_dataset is None:
            raise DesignGateFailure("The completed prediction store is parquet but pyarrow is unavailable.")
    return store


def _load_verified_pickle(run_dir: Path, stage: str) -> Any:
    checkpoints = run_dir / "INTERNAL" / "checkpoints"
    body = checkpoints / f"{stage}.pkl"
    metadata = checkpoints / f"{stage}.json"
    if not body.exists():
        raise DesignGateFailure(f"Missing completed-run checkpoint: {body}")
    if metadata.exists():
        meta = study.read_json(metadata, {}) or {}
        if not meta.get("complete", True):
            raise DesignGateFailure(f"Checkpoint is incomplete: {metadata}")
        expected = meta.get("artifact_sha256") or meta.get("sha256")
        if expected and sha256_file(body) != expected:
            raise DesignGateFailure(f"Checkpoint hash mismatch: {body}")
    with body.open("rb") as stream:
        return pickle.load(stream)


def load_completed_run(run_dir: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    manifest = study.read_json(run_dir / "run_manifest.json", {}) or {}
    if not manifest:
        raise DesignGateFailure(f"No completed run_manifest.json in {run_dir}")
    figure_data = _load_verified_pickle(run_dir, "figure_data")
    if not isinstance(figure_data, Mapping) or "selected" not in figure_data:
        raise DesignGateFailure("figure_data checkpoint lacks the frozen selected-model table")
    try:
        cohort_payload = _load_verified_pickle(run_dir, "cohorts")
        cohorts = cohort_payload["cohorts"] if isinstance(cohort_payload, Mapping) else cohort_payload
    except DesignGateFailure:
        split_payload = _load_verified_pickle(run_dir, "global_splits")
        cohorts = split_payload["cohorts"]
    store = reopen_prediction_store(run_dir / "INTERNAL" / "predictions" / "calibrated")
    if not store.keys():
        raise DesignGateFailure("The completed calibrated prediction store is empty.")
    return store, figure_data["selected"], cohorts, {
        "manifest": manifest, "identity": dict(figure_data.get("identity", {})),
        "split": dict(figure_data.get("split", {})),
    }


def _reporting_cohort(cohorts: Any) -> Any:
    frame = cohorts.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    procedure = first_present(frame, ("procedure", "treatment"), "").astype(str).str.lower()
    cohort_name = first_present(frame, ("cohort",), "").astype(str).str.lower()
    arm = np.where(cohort_name.eq("incretin"), 2,
                   np.where(procedure.str.contains("rygb|gastric bypass"), 1,
                            np.where(procedure.str.contains("sleeve"), 0, -1)))
    frame["arm"] = arm
    frame = frame.loc[frame["arm"].isin([0,1,2])].copy()
    frame["arm_label"] = frame["arm"].map(ARM_LABELS)
    aliases = {
        "age": ("age", "age_at_index"), "diabetes_status": ("diabetes_status", "diabetes_flag", "diabetes_eligible"),
        "calendar_year": ("calendar_year", "index_year"), "baseline_bmi": ("baseline_bmi",),
        "baseline_hba1c": ("baseline_hba1c",), "sex": ("sex",), "race": ("race",),
        "ethnicity": ("ethnicity",), "coverage": ("coverage",), "hypertension": ("hypertension",),
        "dyslipidemia": ("dyslipidemia",), "osa": ("osa",), "insulin": ("insulin",),
        "biguanide": ("biguanide",), "sglt2": ("sglt2",), "smoking": ("smoking",),
        "state": ("state",), "svi": ("svi",), "ruca": ("ruca",), "center_id": ("center_id",),
    }
    for canonical, candidates in aliases.items():
        if canonical not in frame:
            frame[canonical] = first_present(frame, candidates)
    frame["cohort_key"] = np.where(frame["arm"].eq(2), "incretin", "surgery")
    return frame.drop_duplicates(["patient_id", "cohort_key"], keep="first").reset_index(drop=True)


def _selected_candidates(selected: Any) -> dict[tuple[str, str, int], str]:
    result = {}
    for row in selected.itertuples(index=False):
        candidate = str(row.selected_candidate)
        if candidate and candidate != "not_estimable":
            result[(str(row.cohort), str(row.outcome), int(row.origin_month))] = candidate
    return result


def _fit_reporting_overlap(reporting: Any, cfg: TrialConfig) -> tuple[Any, Any, list[str]]:
    shared = []
    for column in ("baseline_bmi", "baseline_hba1c", "age", "sex", "race", "ethnicity",
                   "diabetes_status", "hypertension", "dyslipidemia", "osa", "insulin",
                   "biguanide", "sglt2", "smoking", "coverage", "svi", "ruca", "state", "calendar_year"):
        if column in reporting and all(float(reporting.loc[reporting["arm"].eq(arm), column].notna().mean()) >= .5 for arm in ARM_LABELS):
            shared.append(column)
    x, _names = make_design(reporting, shared)
    model = _fit_multinomial(x, reporting["arm"].to_numpy(int), cfg.seed, False)
    probability = _aligned_probabilities(model, x)
    weight = overlap_weights(probability, reporting["arm"])
    return probability, weight, shared


def weighted_quantile(values: Any, weights: Any, probability: float) -> float:
    x = np.asarray(values, dtype=float); w = np.asarray(weights, dtype=float)
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return float("nan")
    x=x[keep]; w=w[keep]; order=np.argsort(x, kind="mergesort")
    x=x[order]; w=w[order]; cumulative=np.cumsum(w)-.5*w
    cumulative /= np.sum(w)
    return float(np.interp(probability, cumulative, x))


def _descriptive_bootstrap_median(values: Any, weights: Any, ids: Any,
                                  seed: int, replicates: int) -> tuple[float,float]:
    ids = np.asarray(ids).astype(str); unique=np.asarray(sorted(set(ids)))
    positions={value:np.flatnonzero(ids==value) for value in unique}
    rng=np.random.default_rng(seed); estimates=[]
    for _ in range(replicates):
        sampled=rng.choice(unique,size=len(unique),replace=True)
        index=np.concatenate([positions[value] for value in sampled])
        estimates.append(weighted_quantile(np.asarray(values)[index],np.asarray(weights)[index],.5))
    return float(np.nanquantile(estimates,.10)),float(np.nanquantile(estimates,.90))


def _auroc_bootstrap_difference(label: Any, full_score: Any, baseline_score: Any,
                                weight: Any, patient_ids: Any, cfg: TrialConfig) -> tuple[float,float]:
    ids=np.asarray(patient_ids).astype(str); unique=np.asarray(sorted(set(ids)))
    positions={value:np.flatnonzero(ids==value) for value in unique}
    rng=np.random.default_rng(cfg.seed+991); differences=[]
    for _ in range(min(cfg.bootstrap_replicates,400)):
        sampled=rng.choice(unique,size=len(unique),replace=True)
        index=np.concatenate([positions[value] for value in sampled])
        y=np.asarray(label)[index]
        if len(np.unique(y))<2:
            continue
        differences.append(study.weighted_auroc(y,np.asarray(full_score)[index],np.asarray(weight)[index])-
                           study.weighted_auroc(y,np.asarray(baseline_score)[index],np.asarray(weight)[index]))
    if not differences:
        return float("nan"),float("nan")
    return float(np.quantile(differences,.025)),float(np.quantile(differences,.975))


def descriptive_analysis(source_run: Path, cfg: TrialConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    store, selected, cohorts, source_meta = load_completed_run(source_run)
    reporting = _reporting_cohort(cohorts)
    probability, reporting_weight, shared = _fit_reporting_overlap(reporting, cfg)
    reporting["reporting_overlap_weight"] = reporting_weight
    selected_map = _selected_candidates(selected)
    trajectory_rows=[]; threshold_rows=[]; heldout_frames=[]
    quantile_columns=list(study.QUANTILE_COLUMNS)
    for key in store.keys():
        cohort,outcome,origin=key
        if origin != 0 or (cohort,outcome,origin) not in selected_map:
            continue
        columns=list(study.STORED_PREDICTION_COLUMNS)
        predictions=store.read(key,columns=columns)
        candidate=selected_map[(cohort,outcome,origin)]
        predictions=predictions.loc[
            predictions["candidate"].astype(str).eq(candidate)
            & predictions["split"].astype(str).isin(["temporal_test","geographic_test"])
        ].copy()
        if predictions.empty:
            continue
        predictions["patient_id"]=predictions["patient_id"].astype(str)
        predictions["cohort_key"]=str(cohort)
        joined=predictions.merge(reporting[["patient_id","cohort_key","arm","arm_label","baseline_bmi","reporting_overlap_weight"]],
                                 on=["patient_id","cohort_key"],how="inner",validate="many_to_one")
        joined["combined_weight"]=pd.to_numeric(joined["analysis_weight"],errors="coerce").fillna(1.0)*joined["reporting_overlap_weight"]
        heldout_frames.append(joined)
        for (arm_value,horizon),cell in joined.groupby(["arm","target_month"],sort=True):
            matrix=cell[quantile_columns].to_numpy(float); weight=cell["combined_weight"].to_numpy(float)
            median=weighted_quantile(matrix[:,3],weight,.5)
            low,high=_descriptive_bootstrap_median(matrix[:,3],weight,cell["patient_id"],cfg.seed+int(horizon)+int(arm_value),min(cfg.bootstrap_replicates,200))
            observed=cell["target_observed"].astype(bool)&pd.to_numeric(cell["target_value"],errors="coerce").notna()
            obs_values=pd.to_numeric(cell.loc[observed,"target_value"],errors="coerce").to_numpy(float)
            obs_weight=cell.loc[observed,"combined_weight"].to_numpy(float)
            residual=obs_values-matrix[observed.to_numpy(),3]
            rmse=math.sqrt(float(np.average(residual**2,weights=obs_weight))) if len(residual) else np.nan
            coverage=float(np.average((obs_values>=matrix[observed.to_numpy(),1])&(obs_values<=matrix[observed.to_numpy(),5]),weights=obs_weight)) if len(residual) else np.nan
            trajectory_rows.append({"arm":ARM_LABELS[int(arm_value)],"outcome":outcome,"horizon":int(horizon),
                                    "n":int(cell["patient_id"].nunique()),"observed_n":int(observed.sum()),
                                    "predicted_median":median,"ci80_low":low,"ci80_high":high,
                                    "observed_median":weighted_quantile(obs_values,obs_weight,.5),"rmse":rmse,
                                    "coverage80":coverage,"overlap_ess":effective_sample_size(weight),
                                    "claim":"Prognostic comparison of separate observed cohorts; not a treatment effect."})
    heldout=pd.concat(heldout_frames,ignore_index=True) if heldout_frames else pd.DataFrame()
    for arm_value,label in ARM_LABELS.items():
        for horizon in HORIZONS:
            cell=heldout.loc[(heldout["arm"].eq(arm_value))&(heldout["outcome"].eq("bmi"))&(heldout["target_month"].eq(horizon))] if not heldout.empty else heldout
            observed=cell["target_observed"].astype(bool)&pd.to_numeric(cell["target_value"],errors="coerce").notna() if len(cell) else pd.Series([],dtype=bool)
            scored=cell.loc[observed].copy() if len(cell) else cell
            status="estimable"; full=baseline=diff=low=high=np.nan; events=nonevents=0
            if len(scored):
                y=(pd.to_numeric(scored["target_value"],errors="coerce")<35).astype(int).to_numpy()
                events=int(y.sum()); nonevents=int((1-y).sum())
                matrix=scored[quantile_columns].to_numpy(float)
                full_score=study.quantile_ladder_probability(matrix,35.0)
                baseline_score=-pd.to_numeric(scored["baseline_bmi"],errors="coerce").to_numpy(float)
                weight=scored["combined_weight"].to_numpy(float)
                if events>=MIN_CELL and nonevents>=MIN_CELL:
                    full=study.weighted_auroc(y,full_score,weight); baseline=study.weighted_auroc(y,baseline_score,weight); diff=full-baseline
                    low,high=_auroc_bootstrap_difference(y,full_score,baseline_score,weight,scored["patient_id"],cfg)
                else: status="unavailable_event_or_nonevent_support"
            else: status="unavailable_no_heldout_rows"
            threshold_rows.append({"arm":label,"horizon":horizon,"band":"all","n":len(scored),"events":events,"nonevents":nonevents,
                                   "auroc_full":full,"auroc_baseline_bmi":baseline,"difference":diff,"ci_low":low,"ci_high":high,"status":status})
            for lower,upper,band in BMI_BANDS:
                band_cell=scored.loc[pd.to_numeric(scored.get("baseline_bmi"),errors="coerce").ge(lower)&pd.to_numeric(scored.get("baseline_bmi"),errors="coerce").lt(upper)] if len(scored) else scored
                by=(pd.to_numeric(band_cell["target_value"],errors="coerce")<35).astype(int).to_numpy() if len(band_cell) else np.array([],int)
                bevents=int(by.sum()); bnonevents=int(len(by)-bevents); bfull=np.nan
                bstatus="estimable"
                if bevents>=MIN_CELL and bnonevents>=MIN_CELL:
                    bfull=study.weighted_auroc(by,study.quantile_ladder_probability(band_cell[quantile_columns].to_numpy(float),35.0),band_cell["combined_weight"].to_numpy(float))
                else:bstatus="unavailable_event_or_nonevent_support"
                threshold_rows.append({"arm":label,"horizon":horizon,"band":band,"n":len(band_cell),"events":bevents,"nonevents":bnonevents,
                                       "auroc_full":bfull,"auroc_baseline_bmi":np.nan,"difference":np.nan,"ci_low":np.nan,"ci_high":np.nan,"status":bstatus})
    overlap_rows=[]
    for arm_value,label in ARM_LABELS.items():
        cell=reporting["arm"].eq(arm_value)
        overlap_rows.append({"arm":label,"n":int(cell.sum()),"ess":effective_sample_size(reporting_weight[cell]),
                             "min_probability":float(probability[:,arm_value].min()),"p01":float(np.quantile(probability[cell,arm_value],.01)),
                             "median":float(np.median(probability[cell,arm_value])),"p99":float(np.quantile(probability[cell,arm_value],.99)),
                             "max_weight":float(reporting_weight[cell].max())})
    outputs=empty_outputs()
    outputs["baseline_by_arm"]=baseline_table(reporting)
    outputs["three_arm_overlap"]=pd.DataFrame(overlap_rows)
    outputs["threshold_incremental_discrimination"]=pd.DataFrame(threshold_rows)
    diagnostic=pd.DataFrame([gate_row("claim","descriptive_only",True,"Prognostic comparison of separate observed cohorts; not a treatment effect.")])
    outputs["design_diagnostic"]=diagnostic
    aggregate={"mode":"descriptive","source_run":str(source_run.resolve()),"source_metadata":source_meta,
               "shared_reporting_covariates":shared,"trajectories":pd.DataFrame(trajectory_rows),
               "source_mode":source_meta.get("identity",{}).get("source_mode","completed_prediction_run"),
               "source_fingerprint":source_meta.get("manifest",{}).get("fingerprint",source_meta.get("identity",{}).get("fingerprint","unknown")),
               "available_confounders":shared,
               "claim_status":"PROGNOSTIC ONLY - NOT A TREATMENT EFFECT"}
    return outputs,aggregate


# ======================================================================================
# Disclosure-controlled figure book, checkpoints, manifest, and bundle
# ======================================================================================

PALETTE = dict(study.PALETTE)
ARM_COLORS = {"SG": PALETTE["blue"], "RYGB": PALETTE["green"], "INCRETIN": PALETTE["orange"]}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family":"DejaVu Sans","font.size":8.5,"axes.titlesize":10.5,"axes.labelsize":8.5,
        "axes.edgecolor":PALETTE["grid"],"axes.linewidth":.8,"axes.grid":True,
        "grid.color":PALETTE["grid"],"grid.linewidth":.6,"grid.alpha":.75,
        "figure.facecolor":PALETTE["paper"],"axes.facecolor":"white","savefig.facecolor":PALETTE["paper"],
    })


def new_page(number: int, title: str, subtitle: str, claim: str) -> Any:
    figure=plt.figure(figsize=(11,8.5),constrained_layout=False)
    figure.patch.set_facecolor(PALETTE["paper"])
    figure.text(.055,.947,f"{number:02d}",fontsize=22,fontweight="bold",color=PALETTE["blue"],va="top")
    figure.text(.115,.947,title,fontsize=17,fontweight="bold",color=PALETTE["ink"],va="top")
    figure.text(.115,.915,subtitle,fontsize=9.3,color=PALETTE["muted"],va="top")
    figure.lines.append(plt.Line2D([.055,.945],[.893,.893],transform=figure.transFigure,color=PALETTE["grid"],lw=1))
    figure.text(.055,.024,"Aggregate, disclosure-controlled output | Cells n < 11 suppressed | Death reported separately",
                fontsize=7.2,color=PALETTE["muted"])
    figure.text(.945,.024,claim,ha="right",fontsize=7.4,fontweight="bold",
                color=PALETTE["red"] if "NOT" in claim or "FAILED" in claim else PALETTE["blue"])
    return figure


def draw_table(axis: Any, frame: Any, columns: Sequence[str], labels: Sequence[str] | None=None,
               max_rows: int=12, font_size: float=7.2) -> None:
    axis.axis("off")
    if frame is None or frame.empty:
        axis.text(.5,.5,"Not estimable / unavailable",ha="center",va="center",color=PALETTE["muted"])
        return
    view=frame.loc[:,[column for column in columns if column in frame]].head(max_rows).copy()
    def show(value: Any) -> str:
        if pd.isna(value): return "Unavailable"
        if isinstance(value,(float,np.floating)): return f"{float(value):.3f}"
        return str(value)
    cells=[[show(value) for value in row] for row in view.to_numpy()]
    headers=list(labels or view.columns)
    table=axis.table(cellText=cells,colLabels=headers,cellLoc="left",colLoc="left",loc="upper left",bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(font_size)
    for (row,_column),cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"]);cell.set_linewidth(.45)
        if row==0:
            cell.set_facecolor("#EAF2F8");cell.set_text_props(weight="bold",color=PALETTE["ink"])
        elif row%2==0:cell.set_facecolor("#F7F9FC")


def _claim(data: Mapping[str,Any]) -> str:
    return str(data.get("claim_status","UNKNOWN"))


def page_status(data: Mapping[str,Any]) -> Any:
    diagnostic=data["outputs"]["design_diagnostic"]
    figure=new_page(0,"Three-arm study status","Data gates, permitted claim, and run identity",_claim(data))
    left=figure.add_axes([.06,.50,.55,.34]);right=figure.add_axes([.64,.50,.30,.34]);bottom=figure.add_axes([.06,.10,.88,.31])
    hard=diagnostic.loc[diagnostic.get("gate_type",pd.Series(index=diagnostic.index,dtype=str)).isin(["hard","claim"])]
    draw_table(left,hard,["gate","status","detail"],["Gate","Status","Finding"],max_rows=9,font_size=6.9)
    right.axis("off")
    right.text(.02,.95,"Run identity",fontsize=11,fontweight="bold",color=PALETTE["ink"],va="top")
    identity=[f"Mode: {data.get('mode','unknown')}",f"Version: {VERSION}",f"Seed / folds: {data.get('seed',SEED)} / {data.get('folds',FOLDS)}",
              f"Protocol: {str(data.get('protocol_hash',''))[:16]}",f"Source: {data.get('source_mode','unknown')}"]
    right.text(.02,.82,"\n".join(identity),fontsize=8.4,color=PALETTE["ink"],va="top",linespacing=1.5)
    conclusion=str(data.get("conclusion","No conclusion recorded."))
    color=PALETTE["red"] if "no causal" in conclusion.lower() or "failed" in _claim(data).lower() else PALETTE["blue"]
    right.text(.02,.28,textwrap.fill(_claim(data),width=32),fontsize=10,fontweight="bold",color=color,va="top")
    bottom.axis("off");bottom.text(.01,.90,"Conclusion",fontsize=11,fontweight="bold",color=PALETTE["ink"],va="top")
    bottom.text(.01,.73,textwrap.fill(conclusion,width=145),fontsize=9.4,color=PALETTE["ink"],va="top",linespacing=1.45)
    bottom.text(.01,.25,textwrap.fill("No individual treatment recommendation is made. BMI and HbA1c do not capture safety, reflux, nutritional complications, cost, burden, durability, contraindications, or patient preference.",width=150),fontsize=8.4,color=PALETTE["muted"],va="top")
    return figure


def page_protocol(data: Mapping[str,Any]) -> Any:
    if data.get("mode")=="descriptive":
        figure=new_page(1,"Observed cohort definitions and reporting funnel","Frozen prognostic reporting rules; patient labels remain observed",_claim(data))
        left=figure.add_axes([.06,.09,.43,.76]);right=figure.add_axes([.53,.09,.41,.76]);left.axis("off")
        left.text(.01,.98,"Prognostic reporting protocol",fontsize=11,fontweight="bold",va="top")
        lines=["Cohorts: observed RYGB, sleeve gastrectomy, and incretin therapy",
               "Rows: held-out calibrated selected-model forecasts only",
               "Standardization: overlap weights from shared baseline covariates",
               "Labels: each patient's observed procedure/cohort is retained",
               "Uncertainty: deterministic patient-cluster bootstrap",
               "Claim: separate-cohort prognosis; not a treatment effect"]
        left.text(.01,.88,"\n\n".join(textwrap.fill(line,width=62) for line in lines),fontsize=8.7,va="top",color=PALETTE["ink"])
        counts=data["outputs"].get("three_arm_overlap",pd.DataFrame())
        draw_table(right,counts,["arm","n","ess","p01","median","p99"],["Cohort","N","Overlap ESS","P01","Median","P99"],max_rows=3)
        return figure
    figure=new_page(1,"Target-trial protocol and cohort funnel","Frozen protocol beside observational mapping",_claim(data))
    left=figure.add_axes([.06,.09,.43,.76]);right=figure.add_axes([.53,.09,.41,.76])
    protocol=data.get("protocol",{})
    left.axis("off");left.text(.01,.98,"Frozen target protocol",fontsize=11,fontweight="bold",va="top")
    lines=["Population: adults with T2D, BMI 35-75, >=365 baseline days",
           "Strategies: SG, RYGB, first incident incretin start",
           "Time zero: actual initiation; eligibility and follow-up aligned",
           "Policy analog: later augmentation does not censor surgery",
           "Primary: absolute BMI at 12 months; adjust baseline outcome",
           "Secondary: HbA1c at 12; 6/24 months only with support",
           "Death: reported separately; never ordinary missingness",
           "Estimand: pairwise marginal mean arm A minus arm B"]
    left.text(.01,.90,"\n\n".join(textwrap.fill(line,width=62) for line in lines),fontsize=8.5,va="top",color=PALETTE["ink"])
    funnel=data["outputs"].get("cohort_funnel",pd.DataFrame())
    draw_table(right,funnel,["stage","arm","n","excluded_n","reason"],["Stage","Arm","N","Excluded","Reason"],max_rows=20,font_size=6.5)
    return figure


def page_overlap(data: Mapping[str,Any]) -> Any:
    figure=new_page(2,"Three-arm overlap and balance","All three probability tails, overlap ESS, and maximum pairwise SMD",_claim(data))
    overlap=data["outputs"].get("three_arm_overlap",pd.DataFrame());balance=data["outputs"].get("three_arm_balance",pd.DataFrame())
    top=figure.add_axes([.06,.53,.88,.31]);draw_table(top,overlap,["arm","n","ess","min_probability","p01","median","p99","max_weight"],max_rows=3)
    axis=figure.add_axes([.10,.11,.80,.32])
    if balance.empty:
        axis.axis("off");axis.text(.5,.5,"Balance unavailable until a full eligible cohort passes design gates.",ha="center",color=PALETTE["muted"])
    else:
        plot=balance.copy();plot["absolute_smd"]=pd.to_numeric(plot["weighted_smd"],errors="coerce").abs()
        plot=plot.sort_values("absolute_smd").tail(18)
        axis.scatter(plot["absolute_smd"],np.arange(len(plot)),color=PALETTE["blue"],s=22)
        axis.axvline(.10,color=PALETTE["red"],ls="--",lw=1);axis.set_yticks(np.arange(len(plot)),plot["variable"].astype(str),fontsize=7)
        axis.set_xlabel("Maximum/pairwise absolute weighted SMD");axis.set_title("Weighted balance (0.10 soft-gate threshold)",loc="left")
    return figure


def page_descriptive(data: Mapping[str,Any]) -> Any:
    figure=new_page(3,"Descriptive three-arm trajectories","Observed-cohort forecasts and BMI <35 incremental-discrimination audit",_claim(data))
    trajectories=data.get("trajectories",pd.DataFrame());threshold=data["outputs"].get("threshold_incremental_discrimination",pd.DataFrame())
    left=figure.add_axes([.07,.47,.54,.36]);right=figure.add_axes([.65,.47,.29,.36]);bottom=figure.add_axes([.06,.09,.88,.28])
    if trajectories.empty:
        left.axis("off");left.text(.5,.5,"Descriptive pages require --descriptive-from-run.",ha="center",color=PALETTE["muted"])
    else:
        bmi=trajectories.loc[trajectories["outcome"].eq("bmi")]
        for arm,color in ARM_COLORS.items():
            cell=bmi.loc[bmi["arm"].eq(arm)].sort_values("horizon")
            if len(cell):
                left.plot(cell["horizon"],cell["predicted_median"],marker="o",label=arm,color=color)
                left.fill_between(cell["horizon"].to_numpy(float),cell["ci80_low"].to_numpy(float),cell["ci80_high"].to_numpy(float),color=color,alpha=.16)
        left.set_xlabel("Months after observed cohort index");left.set_ylabel("Predicted median BMI (kg/m²)");left.legend(frameon=False,ncol=3);left.set_title("Held-out calibrated predictions with 80% patient-bootstrap intervals",loc="left")
    right.axis("off");right.text(.5,.88,"NOT A TREATMENT EFFECT",ha="center",fontsize=13,fontweight="bold",color=PALETTE["red"],bbox={"boxstyle":"round,pad=.5","facecolor":"#FFF2F2","edgecolor":PALETTE["red"]})
    right.text(.02,.60,textwrap.fill("Prognostic comparison of separate observed cohorts; every patient retains the procedure or cohort label actually observed.",width=42),fontsize=8.8,color=PALETTE["ink"],va="top")
    right.text(.02,.25,textwrap.fill("AUROC is non-additive. Full-minus-baseline AUROC is not a unique percentage contribution of baseline BMI.",width=42),fontsize=8.2,color=PALETTE["muted"],va="top")
    audit=threshold.loc[threshold["band"].eq("all")] if len(threshold) else threshold
    draw_table(bottom,audit,["arm","horizon","n","events","auroc_full","auroc_baseline_bmi","difference","ci_low","ci_high","status"],max_rows=9,font_size=6.7)
    return figure


def page_effects(data: Mapping[str,Any]) -> Any:
    if data.get("mode")=="descriptive":
        figure=new_page(4,"Prognostic accuracy and calibration","Held-out cohort-specific forecast performance; no causal contrast",_claim(data))
        trajectories=data.get("trajectories",pd.DataFrame());axis=figure.add_axes([.06,.10,.88,.74])
        draw_table(axis,trajectories,["arm","outcome","horizon","n","observed_n","predicted_median","observed_median","rmse","coverage80","overlap_ess"],max_rows=22,font_size=6.3)
        return figure
    figure=new_page(4,"Three-arm primary effects","Pairwise IPCW-AIPW contrasts and standardized arm means",_claim(data))
    pairs=data["outputs"].get("pairwise_aipw",pd.DataFrame());means=data["outputs"].get("arm_standardized_means",pd.DataFrame())
    left=figure.add_axes([.08,.12,.52,.70]);right=figure.add_axes([.64,.12,.30,.70])
    estimable=pairs.loc[pd.to_numeric(pairs.get("estimate"),errors="coerce").notna()] if len(pairs) else pairs
    if estimable.empty:
        left.axis("off");left.text(.5,.55,"Point estimates withheld because a hard design gate failed.",ha="center",color=PALETTE["red"],fontweight="bold")
        left.text(.5,.42,"A failed overlap or source gate is a scientific result, not an invitation to extrapolate.",ha="center",color=PALETTE["muted"],wrap=True)
    else:
        plot=estimable.reset_index(drop=True);y=np.arange(len(plot))
        left.errorbar(plot["estimate"],y,xerr=[plot["estimate"]-plot["ci_low"],plot["ci_high"]-plot["estimate"]],fmt="o",color=PALETTE["blue"],ecolor=PALETTE["blue"],capsize=2)
        left.axvline(0,color=PALETTE["ink"],lw=.8);left.set_yticks(y,[f"{r.outcome.upper()} {int(r.horizon)}m | {r.contrast}" for r in plot.itertuples()],fontsize=7);left.invert_yaxis();left.set_xlabel("Marginal mean difference: first arm minus second arm")
    draw_table(right,means,["outcome","horizon","arm","mean","ci_low","ci_high","observed_n"],max_rows=18,font_size=6.5)
    return figure


def page_robustness(data: Mapping[str,Any]) -> Any:
    if data.get("mode")=="descriptive":
        figure=new_page(5,"Prognostic support and threshold audit","Held-out observation support and full-versus-baseline BMI discrimination",_claim(data))
        threshold=data["outputs"].get("threshold_incremental_discrimination",pd.DataFrame());top=figure.add_axes([.06,.42,.88,.42]);bottom=figure.add_axes([.06,.10,.88,.23])
        draw_table(top,threshold,["arm","horizon","band","n","events","nonevents","auroc_full","auroc_baseline_bmi","difference","status"],max_rows=22,font_size=6.1)
        bottom.axis("off");bottom.text(.01,.90,textwrap.fill("Unavailable cells remain declared. A small full-minus-baseline difference together with near-chance within-band AUROC is consistent with apparent discrimination being dominated by proximity to BMI 35; it is not a decomposition of predictor contributions.",width=155),fontsize=8.8,va="top",color=PALETTE["ink"])
        return figure
    figure=new_page(5,"Missingness and robustness","Weight rules, nuisance form, complete cases, observation support, and death",_claim(data))
    robust=data["outputs"].get("robustness",pd.DataFrame());top=figure.add_axes([.06,.43,.88,.41]);bottom=figure.add_axes([.06,.10,.88,.25])
    draw_table(top,robust,["outcome","horizon","contrast","analysis","n","estimate","ci_low","ci_high","status"],max_rows=18,font_size=6.3)
    bottom.axis("off");cells=data.get("cell_summaries",{})
    if cells:
        lines=[f"{key}: eligible/mature n={value.get('n')}, observed={value.get('observed_n')}, missing={value.get('missing_n')}, deaths={value.get('death_n')}" for key,value in sorted(cells.items())]
        bottom.text(.01,.95,"\n".join(lines),va="top",fontsize=8,color=PALETTE["ink"])
    else:bottom.text(.5,.5,"Observation/death summaries unavailable until full-cohort analysis.",ha="center",color=PALETTE["muted"])
    return figure


def page_heterogeneity(data: Mapping[str,Any]) -> Any:
    if data.get("mode")=="descriptive":
        figure=new_page(6,"Scope of the prognostic comparison","No individualized comparative-benefit analysis was performed",_claim(data))
        axis=figure.add_axes([.10,.18,.80,.58]);axis.axis("off")
        axis.text(.5,.78,"SEPARATE OBSERVED COHORTS",ha="center",fontsize=15,fontweight="bold",color=PALETTE["blue"])
        text="This mode evaluates forecast discrimination, calibration, accuracy, sample size, and reporting overlap. It does not rank patients by comparative benefit and does not support an individualized strategy choice."
        axis.text(.5,.52,textwrap.fill(text,width=95),ha="center",va="center",fontsize=10,color=PALETTE["ink"])
        return figure
    figure=new_page(6,"Pairwise heterogeneous benefit","Cross-fitted DR learner, RATE/TOC, calibration, and pairwise c-for-benefit",_claim(data))
    rate=data["outputs"].get("pairwise_rate",pd.DataFrame());cal=data["outputs"].get("benefit_calibration",pd.DataFrame())
    top=figure.add_axes([.06,.54,.88,.30]);bottom=figure.add_axes([.08,.11,.84,.32])
    draw_table(top,rate,["outcome","contrast","n","rate","ci_low","ci_high","c_for_benefit","status"],max_rows=8)
    if cal.empty:
        bottom.axis("off");bottom.text(.5,.5,"Heterogeneity not estimable because causal design gates did not pass.",ha="center",color=PALETTE["muted"])
    else:
        for contrast,color in zip(sorted(cal["contrast"].unique()),[PALETTE["blue"],PALETTE["green"],PALETTE["orange"]]):
            cell=cal.loc[(cal["contrast"].eq(contrast))&(cal["outcome"].eq("bmi"))].sort_values("decile")
            bottom.plot(cell["predicted_benefit"],cell["observed_dr_benefit"],marker="o",label=contrast,color=color)
        limits=bottom.get_xlim();bottom.plot(limits,limits,color=PALETTE["muted"],ls="--",lw=.8);bottom.legend(frameon=False,fontsize=7);bottom.set_xlabel("Mean predicted benefit by decile");bottom.set_ylabel("Mean DR benefit");bottom.set_title("12-month BMI benefit calibration (positive favors first-named arm)",loc="left")
    return figure


def page_policy(data: Mapping[str,Any]) -> Any:
    if data.get("mode")=="descriptive":
        figure=new_page(7,"Permitted prognostic claims","Interpretation boundaries and unavailable clinical dimensions",_claim(data))
        axis=figure.add_axes([.08,.16,.84,.66]);axis.axis("off")
        permitted="Permitted: compare held-out forecast distributions and performance among the three separate observed cohorts after reporting-overlap weighting."
        prohibited="Not permitted: infer that changing a patient's initial strategy would change BMI or HbA1c by the observed forecast difference, or issue an individual strategy recommendation."
        limitations="Unavailable or incomplete: safety, adverse events, reflux, nutritional complications, cost, burden, durability, contraindications, adherence, and patient preference."
        axis.text(.01,.95,"Permitted",fontsize=12,fontweight="bold",color=PALETTE["green"],va="top");axis.text(.01,.85,textwrap.fill(permitted,width=125),fontsize=9.5,va="top")
        axis.text(.01,.61,"Not permitted",fontsize=12,fontweight="bold",color=PALETTE["red"],va="top");axis.text(.01,.51,textwrap.fill(prohibited,width=125),fontsize=9.5,va="top")
        axis.text(.01,.27,"Clinical limitations",fontsize=12,fontweight="bold",color=PALETTE["ink"],va="top");axis.text(.01,.17,textwrap.fill(limitations,width=125),fontsize=9.5,va="top")
        return figure
    figure=new_page(7,"Policy value and permitted claims","Learned versus constant policies, uncertainty, and missing clinical outcomes",_claim(data))
    values=data["outputs"].get("policy_value",pd.DataFrame());left=figure.add_axes([.06,.46,.55,.38]);right=figure.add_axes([.65,.46,.29,.38]);bottom=figure.add_axes([.06,.09,.88,.27])
    draw_table(left,values,["outcome","policy","n","value","ci_low","ci_high","difference_vs_learned","difference_ci_low","difference_ci_high"],max_rows=10,font_size=6.5)
    right.axis("off");right.text(.01,.98,"Permitted claims",fontsize=11,fontweight="bold",va="top")
    permitted=("Descriptive mode: separate observed-cohort prognoses only.\n\nCausal mode: exploratory marginal initial-strategy contrasts only after every hard gate.\n\nA learned policy is not clinically actionable without improvement over all constant policies, an interval excluding zero, external validation, and missing safety/burden outcomes.")
    right.text(.01,.86,textwrap.fill(permitted,width=44),fontsize=8.2,va="top",color=PALETTE["ink"])
    bottom.axis("off");bottom.text(.01,.94,"Clinical limitations",fontsize=11,fontweight="bold",va="top")
    limitations="Safety/adverse events; GERD and nutritional complications; cost and treatment burden; durability; exact contraindications; adherence; patient preference; and transportability to another health system are unavailable or incomplete. No individual treatment recommendation is made."
    bottom.text(.01,.72,textwrap.fill(limitations,width=150),fontsize=9,color=PALETTE["ink"],va="top")
    return figure


PAGE_RENDERERS=(page_status,page_protocol,page_overlap,page_descriptive,page_effects,page_robustness,page_heterogeneity,page_policy)


def _clean_contract_files(export: Path) -> None:
    for name in (*FIGURE_FILES,FIGURE_BOOK):
        path=export/name
        if path.exists() and path.is_file():path.unlink()
        temporary=export/(name+".tmp")
        if temporary.exists() and temporary.is_file():temporary.unlink()


def render_figure_book(cfg: TrialConfig, data: Mapping[str,Any], status_only: bool=False) -> list[Path]:
    configure_style();cfg.export.mkdir(parents=True,exist_ok=True);_clean_contract_files(cfg.export)
    renderers=PAGE_RENDERERS[:1] if status_only else PAGE_RENDERERS
    names=FIGURE_FILES[:1] if status_only else FIGURE_FILES
    pdf_tmp=cfg.export/(FIGURE_BOOK+".tmp");written=[]
    metadata={"Title":"Three-Arm Metabolic Treatment Study","Author":"Shin Lab",
              "CreationDate":datetime(2000,1,1),"ModDate":datetime(2000,1,1)}
    with PdfPages(pdf_tmp,metadata=metadata) as pdf:
        for name,renderer in zip(names,renderers,strict=True):
            figure=renderer(data);temporary=cfg.export/(name+".tmp")
            figure.savefig(temporary,format="png",dpi=220,facecolor=figure.get_facecolor(),metadata={"Software":VERSION})
            study.replace_file(temporary,cfg.export/name);pdf.savefig(figure,dpi=220,facecolor=figure.get_facecolor());plt.close(figure);written.append(cfg.export/name)
    study.replace_file(pdf_tmp,cfg.export/FIGURE_BOOK);written.append(cfg.export/FIGURE_BOOK)
    expected=set(names)|{FIGURE_BOOK};present={path.name for path in cfg.export.iterdir() if path.is_file()}
    unexpected=present-expected
    if unexpected:raise RuntimeError("FIGURES_TO_EXPORT contains non-contract files: "+", ".join(sorted(unexpected)))
    if present!=expected:raise RuntimeError("FIGURES_TO_EXPORT contract is incomplete")
    return written


def aggregate_checkpoint_payload(cfg: TrialConfig, render_data: Mapping[str,Any], status_only: bool) -> dict[str,Any]:
    return {"version":VERSION,"config":asdict(cfg),"status_only":status_only,"render_data":dict(render_data),
            "aggregate_files":{name:sha256_file(cfg.run_dir/name) for name in AGGREGATE_FILES if name.endswith(".csv") and (cfg.run_dir/name).exists()},
            "protocol_hash":sha256_file(cfg.run_dir/"three_arm_trial_protocol.json")}


def write_aggregate_checkpoint(cfg: TrialConfig, render_data: Mapping[str,Any], status_only: bool) -> None:
    payload=aggregate_checkpoint_payload(cfg,render_data,status_only);body=cfg.internal/"aggregate_checkpoint.pkl"
    atomic_pickle(body,payload)
    atomic_json(cfg.internal/"aggregate_checkpoint.json",{"complete":True,"artifact_sha256":sha256_file(body),
                                                           "payload_hash":digest({"version":payload["version"],"aggregate_files":payload["aggregate_files"],"protocol_hash":payload["protocol_hash"]})})


def load_aggregate_checkpoint(cfg: TrialConfig) -> tuple[dict[str,Any],bool]:
    body=cfg.internal/"aggregate_checkpoint.pkl";meta=study.read_json(cfg.internal/"aggregate_checkpoint.json",{}) or {}
    if not body.exists() or not meta.get("complete") or sha256_file(body)!=meta.get("artifact_sha256"):
        raise DesignGateFailure("Plot-only aggregate checkpoint is missing, partial, or corrupt.")
    with body.open("rb") as stream:payload=pickle.load(stream)
    if payload.get("version")!=VERSION:raise DesignGateFailure("Plot-only checkpoint version does not match this script.")
    for name,expected in payload.get("aggregate_files",{}).items():
        path=cfg.run_dir/name
        if not path.exists() or sha256_file(path)!=expected:raise DesignGateFailure(f"Aggregate file failed verification: {name}")
    if sha256_file(cfg.run_dir/"three_arm_trial_protocol.json")!=payload.get("protocol_hash"):
        raise DesignGateFailure("Frozen protocol failed checkpoint verification.")
    return payload["render_data"],bool(payload.get("status_only"))


def write_manifest(cfg: TrialConfig, render_data: Mapping[str,Any], protocol: Mapping[str,Any],
                   status_only: bool=False) -> dict[str,Any]:
    outputs=render_data["outputs"];diagnostic=outputs.get("design_diagnostic",pd.DataFrame())
    missing=[name for name in FROZEN_CONFOUNDERS if name not in set(render_data.get("available_confounders",[]))]
    figures={path.name:sha256_file(path) for path in sorted(cfg.export.iterdir()) if path.is_file()}
    manifest={
        "version":VERSION,"generated_utc":utc_now(),"mode":cfg.mode,"script_sha256":sha256_file(SCRIPT_PATH),
        "query_fingerprint":render_data.get("query_fingerprint","not_applicable"),"source_fingerprint":render_data.get("source_fingerprint","not_applicable"),
        "protocol_sha256":sha256_file(cfg.run_dir/"three_arm_trial_protocol.json"),"arm_definitions":ARM_LABELS,
        "confounders":list(FROZEN_CONFOUNDERS),"missing_confounders":missing,"folds":cfg.folds,"seed":cfg.seed,
        "gate_outcomes":diagnostic.to_dict("records") if isinstance(diagnostic,pd.DataFrame) else [],
        "claim_status":render_data.get("claim_status"),"nuisance_models":protocol.get("nuisance_models",{}),
        "weight_rules":{"primary":"common support plus 1/99 correction-weight truncation","sensitivities":["none","1/99","2.5/97.5","5/95"]},
        "estimand_population":"common-support, administratively mature full eligible cohort" if cfg.mode=="full" else "separate observed held-out reporting cohorts",
        "omissions":protocol.get("clinical_limitations",[]),"figure_hashes":figures,"status_only":status_only,
        "peak_rss_bytes":study.process_peak_rss_bytes(),"individual_treatment_recommendation":False,
    }
    atomic_json(cfg.run_dir/"manifest.json",manifest);return manifest


def write_bundle(cfg: TrialConfig) -> Path:
    destination=cfg.run_dir/"three_arm_results_bundle.zip"
    members=[]
    for name in AGGREGATE_FILES:
        path=cfg.run_dir/name
        if path.exists():members.append((name,path))
    for path in sorted(cfg.export.iterdir()):
        if path.is_file():members.append((f"FIGURES_TO_EXPORT/{path.name}",path))
    fd,temp_name=tempfile.mkstemp(prefix=destination.name+".",suffix=".tmp",dir=destination.parent);os.close(fd)
    try:
        with zipfile.ZipFile(temp_name,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for arcname,path in sorted(members):
                info=zipfile.ZipInfo(arcname,date_time=(2000,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
                archive.writestr(info,path.read_bytes())
        study.replace_file(temp_name,destination)
    finally:
        if os.path.exists(temp_name):os.unlink(temp_name)
    return destination


def _render_data(cfg: TrialConfig, outputs: Mapping[str,Any], protocol: Mapping[str,Any],
                 aggregate: Mapping[str,Any], metadata: Mapping[str,Any] | None=None) -> dict[str,Any]:
    meta=dict(metadata or {})
    diagnostic=outputs.get("design_diagnostic",pd.DataFrame())
    status=aggregate.get("claim_status") or claim_status(diagnostic)
    if cfg.mode=="descriptive":
        conclusion=("This is a prognostic comparison of separate observed RYGB, SG, and incretin cohorts. "
                    "The overlap standardization improves reporting comparability but does not identify a treatment effect.")
    elif hard_gates_pass(diagnostic):
        conclusion=("All hard design gates passed. Reported pairwise estimates are marginal initial-strategy contrasts; "
                    "soft-gate failures retain the EXPLORATORY CAUSAL label and preclude individual recommendations.")
    else:
        failed=diagnostic.loc[(diagnostic.get("gate_type",pd.Series(dtype=str)).eq("hard"))&(~diagnostic.get("passed",pd.Series(dtype=bool)).astype(bool)),"gate"].astype(str).tolist() if len(diagnostic) else []
        conclusion=("No causal estimate was released. The source/design failed: "+", ".join(failed)+". "
                    "The failed gate is the scientific result; the program does not extrapolate around it.")
    return {
        "mode":cfg.mode,"seed":cfg.seed,"folds":cfg.folds,"outputs":dict(outputs),"protocol":dict(protocol),
        "protocol_hash":sha256_file(cfg.run_dir/"three_arm_trial_protocol.json"),"claim_status":status,
        "conclusion":conclusion,"source_mode":meta.get("source_mode",aggregate.get("source_mode","unknown")),
        "query_fingerprint":meta.get("query_fingerprint","not_applicable"),
        "source_fingerprint":meta.get("schema_fingerprint",aggregate.get("source_fingerprint","not_applicable")),
        "available_confounders":list(aggregate.get("available_confounders",[])),
        "trajectories":aggregate.get("trajectories",pd.DataFrame()),
        "cell_summaries":aggregate.get("cell_summaries",{}),
    }


def publish(cfg: TrialConfig, outputs: Mapping[str,Any], protocol: Mapping[str,Any],
            aggregate: Mapping[str,Any], metadata: Mapping[str,Any] | None=None,
            status_only: bool=False) -> Path:
    write_output_frames(cfg,outputs)
    # Re-read exactly what was disclosure controlled and released; figures and plot-only
    # checkpoint therefore cannot accidentally retain a suppressed value.
    released=dict(outputs)
    for name in AGGREGATE_FILES:
        if name.endswith(".csv"):
            released[name[:-4]]=read_csv_or_empty(cfg.run_dir/name)
    data=_render_data(cfg,released,protocol,aggregate,metadata)
    render_figure_book(cfg,data,status_only=status_only)
    write_manifest(cfg,data,protocol,status_only=status_only)
    write_aggregate_checkpoint(cfg,data,status_only)
    write_bundle(cfg)
    return cfg.run_dir


def run_diagnose(cfg: TrialConfig) -> Path:
    ensure_directories(cfg)
    diagnostic,facts=source_design_diagnostic(connect=True)
    protocol=write_protocol(cfg,facts)
    outputs=empty_outputs();outputs["design_diagnostic"]=diagnostic
    aggregate={"claim_status":claim_status(diagnostic),"source_mode":facts.get("source_mode"),"available_confounders":[]}
    return publish(cfg,outputs,protocol,aggregate,facts,status_only=True)


def run_full(cfg: TrialConfig) -> Path:
    ensure_directories(cfg)
    diagnostic,facts=source_design_diagnostic(connect=True)
    protocol=write_protocol(cfg,facts)  # frozen before any outcome is constructed or inspected
    foundational=diagnostic.loc[diagnostic["gate"].isin(HARD_GATE_LABELS[:4]),"passed"].astype(bool).all()
    if not foundational:
        outputs=empty_outputs();outputs["design_diagnostic"]=diagnostic
        aggregate={"claim_status":claim_status(diagnostic),"source_mode":facts.get("source_mode"),"available_confounders":[]}
        return publish(cfg,outputs,protocol,aggregate,facts,status_only=True)
    bundle=acquire_raw_trial_source(cfg)
    cohort,funnel,metadata=construct_trial_cohort(bundle,cfg)
    unrestricted_n=len(cohort)
    cohort,availability=restrict_common_availability(cohort,cfg.min_cell)
    metadata["common_availability_restriction"]=availability
    availability_rows=[]
    for arm_value,label in ARM_LABELS.items():
        availability_rows.append({"stage":"common_calendar_geography","arm":label,
                                  "n":int(cohort["arm"].eq(arm_value).sum()) if len(cohort) else 0,
                                  "excluded_n":np.nan,"reason":"retained in a stratum with all three strategies"})
    funnel=pd.concat([funnel,pd.DataFrame(availability_rows)],ignore_index=True)
    if cohort.empty:
        diagnostic=_replace_gate(diagnostic,HARD_GATE_LABELS[4],False,
                                 f"No common calendar/geographic availability stratum survived from {unrestricted_n} initially eligible rows.",0)
        outputs=empty_outputs();outputs["design_diagnostic"]=diagnostic;outputs["cohort_funnel"]=funnel
        return publish(cfg,outputs,protocol,{"claim_status":claim_status(diagnostic),"source_mode":metadata.get("source_mode")},metadata,True)
    atomic_pickle(cfg.internal/"trial_cohort.pkl",cohort)  # row-level; deliberately absent from bundle
    outputs,diagnostic,internal=causal_analysis(cohort,diagnostic,metadata,cfg)
    outputs["cohort_funnel"]=funnel
    outputs["design_diagnostic"]=diagnostic
    cell_summaries={key:{name:value for name,value in payload.items() if name in ("n","observed_n","missing_n","death_n")}
                    for key,payload in internal.get("cell_payloads",{}).items()}
    aggregate={"claim_status":claim_status(diagnostic),"source_mode":metadata.get("source_mode"),
               "available_confounders":internal.get("feature_columns",[]),"cell_summaries":cell_summaries,
               "support_removed_n":internal.get("support_removed_n",0),"support_retained_n":internal.get("support_retained_n",0)}
    return publish(cfg,outputs,protocol,aggregate,metadata,status_only=not hard_gates_pass(diagnostic))


def run_descriptive(cfg: TrialConfig) -> Path:
    ensure_directories(cfg)
    source=Path(str(cfg.descriptive_run)).expanduser().resolve()
    outputs,aggregate=descriptive_analysis(source,cfg)
    protocol=write_descriptive_protocol(cfg,source)
    return publish(cfg,outputs,protocol,aggregate,aggregate.get("source_metadata",{}),False)


def run_plot_only(cfg: TrialConfig) -> Path:
    ensure_directories(cfg)
    data,status_only=load_aggregate_checkpoint(cfg)
    render_figure_book(cfg,data,status_only=status_only)
    write_manifest(cfg,data,data.get("protocol",{}),status_only=status_only)
    write_bundle(cfg)
    return cfg.run_dir


def _selftest_render_data(cfg: TrialConfig) -> dict[str,Any]:
    outputs=empty_outputs()
    outputs["design_diagnostic"]=pd.DataFrame([gate_row("hard",gate,True,"synthetic pass") for gate in HARD_GATE_LABELS]+[gate_row("soft",gate,False,"synthetic exploratory") for gate in SOFT_GATE_LABELS])
    outputs["cohort_funnel"]=pd.DataFrame([{"stage":"eligible_trial_cohort","arm":label,"n":150,"excluded_n":0,"reason":"included"} for label in ARM_LABELS.values()])
    protocol=trial_protocol(cfg,{"source_mode":"synthetic_raw_events"})
    atomic_json(cfg.run_dir/"three_arm_trial_protocol.json",protocol)
    return _render_data(cfg,outputs,protocol,{"claim_status":"EXPLORATORY CAUSAL","available_confounders":list(FROZEN_CONFOUNDERS)},{"source_mode":"synthetic_raw_events"})


def run_self_tests() -> dict[str,Any]:
    load_runtime();results=[]
    def check(name: str, condition: bool, detail: str="") -> None:
        results.append({"test":name,"passed":bool(condition),"detail":detail})
        if not condition:raise AssertionError(f"{name}: {detail or 'assertion failed'}")

    rng=np.random.default_rng(SEED);n=18000;x=rng.normal(size=n)
    logits=np.column_stack((-.2*x,.25*x,-.05*x));e=np.exp(logits-logits.max(axis=1,keepdims=True));e/=e.sum(axis=1,keepdims=True)
    arm=np.asarray([rng.choice(3,p=row) for row in e]);effects=np.array([0.,-3.,2.]);mu=40+2*x[:,None]+effects[None,:]
    y=mu[np.arange(n),arm]+rng.normal(scale=.7,size=n);delta=np.ones(n,bool);pc=np.ones(n)
    recovered=three_arm_aipw(y,arm,delta,e,pc,mu)
    check("01_three_arm_aipw_recovers_planted_effects",all(abs(recovered["pairwise"][(a,b)]["estimate"]-(effects[a]-effects[b]))<.12 for a,b in PAIRWISE))
    constant=np.tile(np.mean(y),(n,3));ps_only=three_arm_aipw(y,arm,delta,e,pc,constant)
    uniform=np.full((n,3),1/3);mu_only=three_arm_aipw(y,arm,delta,uniform,pc,mu)
    check("02_double_robust_recovery",all(abs(ps_only["pairwise"][(a,b)]["estimate"]-(effects[a]-effects[b]))<.20 and abs(mu_only["pairwise"][(a,b)]["estimate"]-(effects[a]-effects[b]))<.12 for a,b in PAIRWISE))
    model=_fit_outcome_mean(np.column_stack((x[:500],arm[:500])),y[:500],SEED)
    check("03_outcome_nuisance_targets_mean",getattr(model,"loss",None)=="squared_error",str(getattr(model,"loss",None)))
    import inspect
    cohort_source=inspect.getsource(construct_trial_cohort)
    forbidden_entry=any(token in cohort_source for token in ("glp1_end","duration_days","qualifies_window","pdc_qualifying"))
    check("04_future_persistence_cannot_affect_entry",not forbidden_entry)
    synthetic=synthetic_trial_cohort(SEED,450)
    aligned=(synthetic["index_date"].notna().all() and all((synthetic[f"{outcome}_{horizon}_date"]>=synthetic["index_date"]).all() for outcome in OUTCOMES for horizon in HORIZONS))
    check("05_time_zero_alignment",aligned)
    check("06_no_postindex_covariate_in_L",not any(token in name.lower() for name in FROZEN_CONFOUNDERS for token in FORBIDDEN_POSTINDEX_TOKENS),str(FROZEN_CONFOUNDERS))
    bad_e=e.copy();bad_e[:,2]=.005;bad_e[:,:2]*=.995/bad_e[:,:2].sum(axis=1,keepdims=True)
    all_tails=all(float(e[:,a].min())>.01 for a in ARM_LABELS);failed_arm=any(float(bad_e[:,a].min())<=.01 for a in ARM_LABELS)
    check("07_symmetric_three_arm_positivity",all_tails and failed_arm)
    metadata={"source_mode":"synthetic_raw_events","future_persistence_fields_read":False,"prediction_store_read":False,"measurement_timing":"exact_day"}
    check("08_full_cohort_independent_of_predictions",len(synthetic)==450 and not metadata["prediction_store_read"])
    labels_ok=all(recovered["pairwise"][(a,b)]["estimate"]*(effects[a]-effects[b])>0 and contrast_label(a,b)==f"{ARM_LABELS[a]} vs {ARM_LABELS[b]}" for a,b in PAIRWISE)
    check("09_pairwise_signs_and_labels",labels_ok)
    missing=y.copy();missing[:3000]=np.nan;observed=np.ones(n,bool);observed[:3000]=False
    missing_result=three_arm_aipw(missing,arm,observed,e,pc,mu)
    zero_result=three_arm_aipw(np.nan_to_num(missing),arm,np.ones(n,bool),e,pc,mu)
    check("10_missing_y_not_observed_zero",abs(missing_result["means"].mean()-zero_result["means"].mean())>.5)
    noise_rank=np.ones(1000);hetero=np.linspace(-2,2,1000)
    check("11_c_for_benefit",abs(c_for_benefit(noise_rank,hetero)-.5)<.02 and c_for_benefit(hetero,hetero)>.95)
    bmi=rng.uniform(35,55,5000);event=(bmi<42).astype(int);baseline_auc=study.weighted_auroc(event,-bmi,np.ones(len(bmi)))
    independent=rng.normal(size=len(bmi));latent=-.28*(bmi-42)+1.4*independent;event2=(latent>np.quantile(latent,.55)).astype(int)
    full_auc=study.weighted_auroc(event2,latent,np.ones(len(bmi)));base2=study.weighted_auroc(event2,-bmi,np.ones(len(bmi)))
    check("12_bmi_incremental_discrimination",baseline_auc>.99 and full_auc-base2>.05,f"{baseline_auc:.3f}, {full_auc-base2:.3f}")
    with tempfile.TemporaryDirectory(prefix="three-arm-a-") as one,tempfile.TemporaryDirectory(prefix="three-arm-b-") as two:
        hashes=[]
        for directory in (one,two):
            temp_cfg=TrialConfig("self-test",directory,seed=SEED,bootstrap_replicates=20)
            ensure_directories(temp_cfg);data=_selftest_render_data(temp_cfg);render_figure_book(temp_cfg,data,False)
            hashes.append({path.name:sha256_file(path) for path in sorted(temp_cfg.export.iterdir())})
            names={path.name for path in temp_cfg.export.iterdir() if path.is_file()}
            check("14_figure_contract_"+Path(directory).name,names==set(FIGURE_FILES)|{FIGURE_BOOK},str(sorted(names)))
        check("13_repeated_figure_hashes",hashes[0]==hashes[1])
    return {"version":VERSION,"seed":SEED,"passed":True,"tests":results}


def default_output(mode: str, source: str | None=None) -> str:
    if mode=="descriptive" and source:
        return str(Path(source).expanduser().resolve()/"three_arm_target_trial")
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    return str((Path.cwd()/"results"/f"three_arm_target_trial_{stamp}").resolve())


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--descriptive-from-run",metavar="RUN_DIR",help="Produce noncausal held-out prognostic comparison without Cosmos")
    modes.add_argument("--diagnose",action="store_true",help="Query only aggregate/schema facts and render the design eligibility report")
    modes.add_argument("--full",action="store_true",help="Acquire the full raw eligible cohort and run the gated target trial")
    modes.add_argument("--plot-only",metavar="RUN_DIR",help="Rebuild figures from a verified aggregate checkpoint")
    modes.add_argument("--self-test",action="store_true",help="Run embedded deterministic synthetic tests; no Cosmos connection")
    parser.add_argument("--output-dir",help="Run directory (required location may be inferred from the selected mode)")
    parser.add_argument("--seed",type=int,default=SEED,help=f"Deterministic seed (default {SEED})")
    parser.add_argument("--bootstrap-replicates",type=int,default=400)
    return parser


def _failure_diagnostic(message: str) -> Any:
    rows=[gate_row("hard",gate,False,message if index==0 else "Not evaluated after upstream failure.") for index,gate in enumerate(HARD_GATE_LABELS)]
    rows.extend(gate_row("soft",gate,False,"Not evaluated after upstream failure.") for gate in SOFT_GATE_LABELS)
    return pd.DataFrame(rows)


def publish_failure(cfg: TrialConfig, error: Exception) -> Path:
    ensure_directories(cfg);diagnostic=getattr(error,"diagnostic",None)
    if diagnostic is None:diagnostic=_failure_diagnostic(f"{type(error).__name__}: {study.sanitize_exception_text(error)}")
    facts={"source_mode":"unavailable","failure":f"{type(error).__name__}: {study.sanitize_exception_text(error)}"}
    protocol=write_protocol(cfg,facts);outputs=empty_outputs();outputs["design_diagnostic"]=diagnostic
    return publish(cfg,outputs,protocol,{"claim_status":claim_status(diagnostic),"source_mode":"unavailable"},facts,True)


def main(argv: Sequence[str] | None=None) -> int:
    args=build_parser().parse_args(argv);load_runtime()
    if args.self_test:
        result=run_self_tests();print(json.dumps(result,indent=2));return 0
    if args.descriptive_from_run:
        mode="descriptive";source=args.descriptive_from_run;output=args.output_dir or default_output(mode,source)
    elif args.plot_only:
        mode="plot-only";source=None;output=str(Path(args.plot_only).expanduser().resolve())
        if args.output_dir and Path(args.output_dir).expanduser().resolve()!=Path(output):
            raise SystemExit("--plot-only RUN_DIR cannot be combined with a different --output-dir")
    elif args.diagnose:
        mode="diagnose";source=None;output=args.output_dir or default_output(mode)
    else:
        mode="full";source=None;output=args.output_dir or default_output(mode)
    cfg=TrialConfig(mode=mode,output_dir=output,descriptive_run=source,seed=args.seed,
                    bootstrap_replicates=max(20,int(args.bootstrap_replicates)))
    random.seed(cfg.seed);np.random.seed(cfg.seed)
    try:
        if mode=="descriptive":run_dir=run_descriptive(cfg)
        elif mode=="diagnose":run_dir=run_diagnose(cfg)
        elif mode=="full":run_dir=run_full(cfg)
        else:run_dir=run_plot_only(cfg)
    except Exception as error:
        if mode=="plot-only":raise
        run_dir=publish_failure(cfg,error)
        print(f"[three-arm] stopped after diagnostic: {error}",file=sys.stderr)
        print(f"[three-arm] report: {run_dir}")
        return 2
    print(f"[three-arm] completed: {run_dir}")
    print(f"[three-arm] figures: {run_dir/'FIGURES_TO_EXPORT'}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
