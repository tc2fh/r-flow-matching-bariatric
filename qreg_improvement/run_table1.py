#!/usr/bin/env python3
"""Table 1 for the metabolic trajectory study: cohort sizes, treatments, demographics,
comorbidities, and model-input availability for the sleeve gastrectomy, RYGB, and incretin-based
therapy cohorts.

Run on the Cosmos VM after the study has finished (one command, no arguments needed):

    python qreg_improvement/run_table1.py
    python qreg_improvement/run_table1.py --bundle "Z:\\Project D33AFD\\results\\<run>\\final_study_bundle.zip"
    python qreg_improvement/run_table1.py --self-test            # deterministic checks, ~30 s
    python qreg_improvement/run_table1.py --render-from-csv table1_cohort_characteristics.csv

The table describes exactly the analytic cohort the models were developed and evaluated on. It is
read from the completed run's own verified checkpoints - INTERNAL/checkpoints/global_splits.pkl
(the cohort frame with its model partition) and the INTERNAL/rows prediction-row store (which
outcomes each held-out patient contributed) - so nothing is re-queried and every count matches the
run. Only if those checkpoints are missing or fail their integrity check does it re-acquire the
cohort from Cosmos with the study's own code (run_metabolic_trajectory_study.py, next to this file),
which applies the same cohort definition; a footnote records which source was used.

Outputs, written to <run>/table1/ and added to final_study_bundle.zip under table1/:
  table1_cohort_characteristics.png  the rendered table
  table1_cohort_characteristics.csv  every cell exactly as displayed, plus the title, column
                                     headers, and footnotes. The PNG is rendered from this CSV
                                     after it is written, so the CSV alone regenerates it.

Disclosure control (Epic Cosmos policy): every displayed patient count below 11 is shown as "<11".
Further cells are then suppressed until no count of 1 to 10 patients can be recovered, or even
narrowed to fewer than six possible values, from anything the table publishes: partition totals,
the SG + RYGB = All MBS identity, the availability rows, rounded percentages, and the bounds the
masks themselves imply. That is audited before anything is written. Nothing patient-level leaves
the run directory.

Self-contained by design: standard library plus numpy, pandas, matplotlib, and scipy (installed
with scikit-learn), all of which the study already requires. The study module is imported only on
the Cosmos fallback path.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
import textwrap
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

THIS_FILE = Path(__file__).resolve()
STUDY_SCRIPT = "run_metabolic_trajectory_study.py"
DEFAULT_BUNDLE = r"Z:\Project D33AFD\results\shared_three_arm_scenario_20260813_101808\final_study_bundle.zip"
TABLE_STEM = "table1_cohort_characteristics"
BUNDLE_FOLDER = "table1"
# run_final_study.py pins every bundle member's mtime to this so identical bundles stay identical.
BUNDLE_TIMESTAMP = (2001, 1, 1, 0, 0, 0)
MIN_CELL = 11
DPI = 300

# Mirrors of study constants. Self-test 09 asserts they still match the study script.
HELD_OUT_SPLITS = ("temporal_test", "geographic_test")
SPLIT_ORDER = ("train", "validation", "calibration", "temporal_test", "geographic_test")
SPLIT_LABELS = {
    "train": "Development: training",
    "validation": "Development: validation",
    "calibration": "Development: calibration",
    "temporal_test": "Held-out test: temporal",
    "geographic_test": "Held-out test: geographic",
}
PROCEDURE_CODES = {"43775": "sleeve", "43644": "rygb", "43645": "rygb", "43846": "rygb"}
OPTIONAL_WIDE_NUMERIC_COVARIATES = (
    "creatinine_baseline", "egfr_baseline", "gerd_baseline", "ldl_baseline", "hdl_baseline",
    "triglycerides_baseline", "glucose_fasting_baseline", "sbp_baseline", "dbp_baseline",
)
OPTIONAL_COVARIATE_THRESHOLD = 0.5
TARGET_WINDOWS = {3: (2.0, 4.5), 6: (4.5, 9.0), 12: (9.0, 18.0), 24: (18.0, 30.0),
                  36: (30.0, 42.0), 48: (42.0, 54.0), 60: (54.0, 66.0)}

# Display columns. SG and RYGB partition the surgical cohort, so All MBS = SG + RYGB exactly, an
# identity the disclosure control has to respect. IBT is the separate incretin cohort.
COLUMNS = ("sg", "rygb", "all_mbs", "ibt")
ADDITIVE = ("sg", "rygb", "all_mbs")
COLUMN_TITLES = {
    "sg": "Sleeve gastrectomy",
    "rygb": "Roux-en-Y gastric bypass",
    "all_mbs": "All MBS",
    "ibt": "Incretin-based therapy",
}
NOT_APPLICABLE = "-"
NOT_IN_SOURCE = "Not in source"
MASK_PRIMARY = "<11"
MASK_SECONDARY = "Masked\u2021"
MASK_COMPLEMENT = "All but <11"

FLAG_ROWS = (
    ("diabetes_flag", "Type 2 diabetes"),
    ("hypertension", "Hypertension"),
    ("dyslipidemia", "Dyslipidemia"),
    ("osa", "Obstructive sleep apnea"),
)
MEDICATION_ROWS = (
    ("insulin", "Insulin"),
    ("biguanide", "Metformin"),
    ("sglt2", "SGLT2 inhibitor"),
)
# (column, label, decimals) for the optional enriched baseline numerics other than GERD.
LAB_ROWS = (
    ("creatinine_baseline", "Creatinine", 2),
    ("egfr_baseline", "eGFR", 0),
    ("ldl_baseline", "LDL cholesterol", 0),
    ("hdl_baseline", "HDL cholesterol", 0),
    ("triglycerides_baseline", "Triglycerides", 0),
    ("glucose_fasting_baseline", "Fasting glucose", 0),
    ("sbp_baseline", "Systolic blood pressure", 0),
    ("dbp_baseline", "Diastolic blood pressure", 0),
)
GERD_COLUMN = "gerd_baseline"
AGENT_LEVELS = (
    "Semaglutide, injectable", "Semaglutide, oral", "Tirzepatide", "Liraglutide", "Dulaglutide",
    "Exenatide, lixisenatide, or albiglutide", "Other or unrecognized",
)
AGE_LEVELS = ("18-39", "40-54", "55-64", "\u226565", "Not recorded")
YEAR_LEVELS = ("2015 or earlier", "2016-2018", "2019-2021", "2022 or later")
BMI_LEVELS = ("30.0-34.9", "35.0-39.9", "40.0-49.9", "\u226550.0")
HBA1C_LEVELS = ("<5.7", "5.7-6.4", "\u22656.5", "Not measured")
RUCA_LEVELS = ("Metropolitan (RUCA 1-3)", "Nonmetropolitan (RUCA 4-10)", "Not recorded")

COHORT_COLUMNS = (
    "patient_id", "cohort", "treatment", "procedure", "index_ingredient", "index_route", "split",
    "index_date", "age_at_index", "sex", "race", "ethnicity", "coverage", "smoking", "svi", "ruca",
    "baseline_bmi", "baseline_hba1c", "diabetes_eligible",
    *[name for name, _ in FLAG_ROWS], *[name for name, _ in MEDICATION_ROWS],
    *OPTIONAL_WIDE_NUMERIC_COVARIATES,
)
ROW_COLUMNS = ("patient_id", "cohort", "outcome", "origin_month", "target_month", "split",
               "target_observed")


def log(message: str) -> None:
    print(f"[table1] {message}", flush=True)


class SourceUnavailable(RuntimeError):
    """The frozen run checkpoints cannot be used: missing, incomplete, or failing their hash."""


class DisclosureError(RuntimeError):
    """The table could not be made disclosure-safe, so nothing is written or bundled."""


# ======================================================================================
# 1. Category mapping. Source values are free-ish text, so every mapping is explicit and the
#    raw values feeding each displayed level are recorded in the CSV's source_values column.
# ======================================================================================

_MISSING_TOKENS = frozenset({
    "", "nan", "none", "null", "na", "n/a", "<na>", "nat", "unk", "unknown", "?", "-", "--",
})
_UNKNOWN_PATTERN = re.compile(
    r"unknown|unspecified|refus|declin|prefer not|unable to|undetermined|not assessed|"
    r"never assessed|no information|not (recorded|reported|specified|available|documented|"
    r"disclosed|provided|on file|obtained)",
    re.I,
)


def is_unknown(text: str) -> bool:
    value = str(text).strip().strip("*").strip().lower()
    return value in _MISSING_TOKENS or bool(_UNKNOWN_PATTERN.search(value))


def sex_category(text: str) -> str:
    if is_unknown(text):
        return "Unknown or not recorded"
    value = text.strip().lower()
    if value in {"f", "female", "woman", "girl"} or value.startswith("female"):
        return "Female"
    if value in {"m", "male", "man", "boy"} or value.startswith("male"):
        return "Male"
    return "Other"


RACE_LEVELS = (
    "White", "Black or African American", "Asian", "American Indian or Alaska Native",
    "Native Hawaiian or Other Pacific Islander", "Multiracial", "Other", "Unknown or not reported",
)


def race_category(text: str) -> str:
    if is_unknown(text):
        return "Unknown or not reported"
    value = text.strip().lower()
    if re.search(r"multi|two or more|more than one|mixed|biracial", value):
        return "Multiracial"
    if re.search(r"american indian|alaska", value):
        return "American Indian or Alaska Native"
    if re.search(r"hawaiian|pacific island", value):
        return "Native Hawaiian or Other Pacific Islander"
    if re.search(r"\bblack\b|african american|^african$", value):
        return "Black or African American"
    if re.search(r"\basian\b", value):
        return "Asian"
    if re.search(r"\bwhite\b|caucasian", value):
        return "White"
    return "Other"


ETHNICITY_LEVELS = ("Hispanic or Latino", "Not Hispanic or Latino", "Other", "Unknown or not reported")


def ethnicity_category(text: str) -> str:
    if is_unknown(text):
        return "Unknown or not reported"
    value = text.strip().lower()
    if re.search(r"\bnot\b|\bnon[- ]?hispanic|\bnon[- ]?latin", value):
        return "Not Hispanic or Latino"
    if re.search(r"hispanic|latin[aox]", value):
        return "Hispanic or Latino"
    return "Other"


COVERAGE_LEVELS = ("Commercial", "Medicare", "Medicaid", "Self-pay or uninsured", "Other",
                   "Unknown or not reported")


def coverage_category(text: str) -> str:
    if is_unknown(text):
        return "Unknown or not reported"
    value = text.strip().lower()
    if "medicare" in value:
        return "Medicare"
    if re.search(r"medicaid|\bchip\b", value):
        return "Medicaid"
    if re.search(r"self|uninsur|no coverage|no insurance|charity", value):
        return "Self-pay or uninsured"
    if re.search(r"commercial|private|managed|employer|\bhmo\b|\bppo\b|exchange|marketplace|blue", value):
        return "Commercial"
    return "Other"


SMOKING_LEVELS = ("Current", "Former", "Never", "Never or former", "Unknown or not recorded")


def smoking_category(text: str) -> str:
    value = str(text).strip().lower()
    if re.search(r"never assessed|not assessed", value):
        return "Unknown or not recorded"
    if "never" in value and re.search(r"former|\bex\b", value):
        return "Never or former"
    if "never" in value:
        return "Never"
    if re.search(r"former|quit|\bex[- ]", value):
        return "Former"
    if re.search(r"current|every day|some day|daily|smoker\b|heavy|light", value):
        return "Current"
    return "Unknown or not recorded"


def ruca_category(value: Any) -> str:
    """RUCA is free text in the source (nvarchar), e.g. "1", "4.2", or "1 metropolitan": the leading
    primary code decides (1-3 metropolitan, 4-10 not), and a label without a code falls back to its
    wording."""
    text = "" if value is None else str(value).strip().lower()
    match = re.match(r"(\d+(?:\.\d+)?)", text)
    if match:
        code = float(match.group(1))
        if 1 <= code < 11:
            return "Metropolitan (RUCA 1-3)" if code < 4 else "Nonmetropolitan (RUCA 4-10)"
        return "Not recorded"
    if re.search(r"non[- ]?metro|micro|small town|rural", text):
        return "Nonmetropolitan (RUCA 4-10)"
    if "metro" in text or "urban" in text:
        return "Metropolitan (RUCA 1-3)"
    return "Not recorded"


def agent_category(ingredient: Any, route: Any) -> str:
    name = str(ingredient).strip().lower()
    path = str(route).strip().lower()
    if name == "semaglutide":
        return "Semaglutide, oral" if path in {"oral", "tablet", "po"} else "Semaglutide, injectable"
    if name in {"tirzepatide", "liraglutide", "dulaglutide"}:
        return name.capitalize()
    if name in {"exenatide", "lixisenatide", "albiglutide"}:
        return "Exenatide, lixisenatide, or albiglutide"
    return "Other or unrecognized"


def age_category(age: float) -> str:
    if not math.isfinite(age):
        return "Not recorded"
    if age < 40:
        return "18-39"
    if age < 55:
        return "40-54"
    if age < 65:
        return "55-64"
    return "\u226565"


def year_category(year: float) -> str:
    if year <= 2015:
        return "2015 or earlier"
    if year <= 2018:
        return "2016-2018"
    if year <= 2021:
        return "2019-2021"
    return "2022 or later"


def bmi_category(bmi: float) -> str:
    if bmi < 35:
        return "30.0-34.9"
    if bmi < 40:
        return "35.0-39.9"
    if bmi < 50:
        return "40.0-49.9"
    return "\u226550.0"


def hba1c_category(value: float) -> str:
    if not math.isfinite(value):
        return "Not measured"
    if value < 5.7:
        return "<5.7"
    if value < 6.5:
        return "5.7-6.4"
    return "\u22656.5"


def mapped_text(series: Any, mapper: Callable[[str], str]) -> tuple[Any, dict[str, Counter]]:
    """Map a text column through ``mapper`` once per distinct value; return the mapped column and,
    per displayed level, a Counter of the raw source values (patients) it absorbed."""
    raw = series.astype("string").fillna("").str.strip().astype(object)
    lookup = {value: mapper(value) for value in pd.unique(raw)}
    mapped = raw.map(lookup)
    crosswalk: dict[str, Counter] = {}
    for value, count in raw.value_counts().items():
        crosswalk.setdefault(lookup[value], Counter())[value] = int(count)
    return mapped, crosswalk


def source_values_text(counter: Counter | None) -> str:
    """Raw values behind a level, most common first. Only values held by at least 11 patients are
    named, so the label of a rare source category is never published."""
    if not counter:
        return ""
    named = [value if value else "(blank)" for value, count in counter.most_common() if count >= MIN_CELL]
    rare = sum(1 for count in counter.values() if count < MIN_CELL)
    if len(named) > 12:
        rare += len(named) - 12
        named = named[:12]
    text = "; ".join(named)
    if rare:
        text += ("; " if text else "") + f"+{rare} other source value{'s' if rare != 1 else ''}"
    return text


# ======================================================================================
# 2. Loading the analytic cohort: frozen run checkpoints first, Cosmos re-acquire as fallback.
# ======================================================================================


@dataclass
class StudyData:
    cohorts: Any                      # one row per (patient, cohort), carrying its split
    outcome_rows: Any                 # held-out, observed, origin-0 prediction rows
    row_columns: tuple[str, ...]      # the prediction-row schema: which inputs the models received
    split_metadata: dict[str, Any]
    provenance: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def verify_checkpoint(checkpoints: Path, stage: str) -> tuple[Path, dict[str, Any]]:
    """Return (body, metadata) for a completed checkpoint whose body still matches its recorded
    sha256 - the same completion and integrity checks the study's require_checkpoint applies."""
    meta_path, body = checkpoints / f"{stage}.json", checkpoints / f"{stage}.pkl"
    if not meta_path.is_file():
        raise SourceUnavailable(f"checkpoint record {meta_path} is missing")
    metadata = read_json(meta_path)
    if metadata.get("completion_marker") != "COMPLETE":
        raise SourceUnavailable(f"checkpoint {stage} is not marked COMPLETE")
    if not body.is_file():
        raise SourceUnavailable(f"checkpoint body {body} is missing")
    if sha256_file(body) != metadata.get("artifact_sha256"):
        raise SourceUnavailable(f"checkpoint {stage} failed its sha256 integrity check")
    return body, metadata


_PARTITION_FILE = re.compile(r"^(?P<partition>.+__origin(?P<origin>\d{3}))__(?P<part>\d{4})\.(?P<ext>parquet|pkl)$")


def read_origin0_rows(root: Path, expected: Mapping[str, int] | None) -> tuple[Any, tuple[str, ...]]:
    """Read the forecast-origin-0 partitions of a prediction-row store directory without
    unpickling the store object, keeping only held-out rows with an observed outcome.

    Every partition's row count is checked against the count its checkpoint recorded, so a
    truncated or partially rewritten store is rejected rather than silently undercounted.
    """
    if not root.is_dir():
        raise SourceUnavailable(f"prediction-row store {root} is missing")
    files: dict[str, list[Path]] = {}
    for path in sorted(root.iterdir()):
        match = _PARTITION_FILE.match(path.name)
        if match and int(match["origin"]) == 0:
            files.setdefault(match["partition"], []).append(path)
    if not files:
        raise SourceUnavailable(f"prediction-row store {root} has no origin-0 partitions")
    if expected is not None:
        wanted = {name for name in expected if name.endswith("__origin000")}
        if wanted != set(files):
            raise SourceUnavailable(
                f"prediction-row store {root} partitions {sorted(files)} do not match its "
                f"checkpoint record {sorted(wanted)}")
    frames, schema = [], None
    for partition, paths in sorted(files.items()):
        rows_seen = 0
        for path in sorted(paths):
            if path.suffix == ".parquet":
                try:
                    import pyarrow.parquet as parquet
                except ImportError as error:
                    raise SourceUnavailable(f"pyarrow is needed to read {path.name}: {error}") from error
                names = tuple(parquet.read_schema(path).names)
                frame = pd.read_parquet(path, columns=[name for name in ROW_COLUMNS if name in names])
            else:
                with path.open("rb") as stream:
                    frame = pickle.load(stream)
                names = tuple(str(name) for name in frame.columns)
                frame = frame[[name for name in ROW_COLUMNS if name in names]]
            schema = schema or names
            missing = [name for name in ROW_COLUMNS if name not in names]
            if missing:
                raise SourceUnavailable(f"{path.name} lacks columns {missing}")
            rows_seen += len(frame)
            keep = (frame["split"].astype(str).isin(HELD_OUT_SPLITS).to_numpy()
                    & frame["target_observed"].fillna(False).astype(bool).to_numpy())
            kept = frame.loc[keep]
            frames.append(pd.DataFrame({column: kept[column].astype(str).to_numpy()
                                        for column in ("patient_id", "cohort", "outcome", "split")}
                                       | {"target_month": pd.to_numeric(kept["target_month"]).astype(int).to_numpy()}))
            del frame, kept
        if expected is not None and int(expected[partition]) != rows_seen:
            raise SourceUnavailable(
                f"prediction-row partition {partition} holds {rows_seen} rows but its checkpoint "
                f"recorded {expected[partition]}")
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(ROW_COLUMNS))
    return rows, tuple(schema or ())


def store_manifest_partitions(checkpoints: Path, stage: str) -> dict[str, int]:
    """Per-partition row counts the run recorded for a row-store checkpoint."""
    _, metadata = verify_checkpoint(checkpoints, stage)
    manifest = metadata.get("payload_manifest", {})
    if stage == "weights":
        manifest = manifest.get("children", {}).get("rows", {})
    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict) or not partitions:
        raise SourceUnavailable(f"checkpoint {stage} records no row-store partitions")
    return {str(name): int(count) for name, count in partitions.items()}


def slim_cohorts(frame: Any) -> Any:
    """Keep only the columns the table reads; absent optional columns become NaN."""
    present = [name for name in COHORT_COLUMNS if name in frame.columns]
    slim = frame[present].copy()
    for name in COHORT_COLUMNS:
        if name not in slim.columns:
            slim[name] = np.nan
    slim.attrs["source_columns"] = tuple(str(name) for name in frame.columns)
    return slim


def run_context(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    configuration = manifest.get("configuration", {}) or {}
    return {
        "run_name": run_dir.name,
        "study_version": manifest.get("study_version", "unknown study version"),
        "incretin_qualifying_months": int(configuration.get("incretin_qualifying_months") or 12),
    }


def load_frozen(run_dir: Path) -> StudyData:
    checkpoints = run_dir / "INTERNAL" / "checkpoints"
    body, metadata = verify_checkpoint(checkpoints, "global_splits")
    log(f"reading verified cohort checkpoint {body} (sha256 {metadata['artifact_sha256'][:12]})")
    with body.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "cohorts" not in payload:
        raise SourceUnavailable("global_splits checkpoint does not hold a cohorts frame")
    cohorts = slim_cohorts(payload["cohorts"])
    split_metadata = dict(payload.get("metadata") or {})
    del payload
    gc.collect()
    failures = []
    for stage, store in (("weights", "weighted"), ("prediction_rows", "unweighted")):
        try:
            expected = store_manifest_partitions(checkpoints, stage)
            rows, schema = read_origin0_rows(run_dir / "INTERNAL" / "rows" / store, expected)
            log(f"read held-out outcome rows from INTERNAL/rows/{store} ({len(rows):,} held-out observed rows)")
            break
        except SourceUnavailable as error:
            failures.append(f"{store}: {error}")
    else:
        raise SourceUnavailable("no usable prediction-row store: " + "; ".join(failures))
    provenance = {**run_context(run_dir), "source": "frozen run checkpoints",
                  "detail": f"global_splits sha256 {metadata['artifact_sha256'][:12]}; rows/{store}"}
    return StudyData(cohorts, rows, schema, split_metadata, provenance)


def import_study(scripts_dir: Path) -> Any:
    """Load the study script as a library, as the shared three-arm wrapper does."""
    path = scripts_dir / STUDY_SCRIPT
    if not path.is_file():
        raise RuntimeError(f"Cosmos fallback needs {STUDY_SCRIPT} in {scripts_dir}")
    spec = importlib.util.spec_from_file_location("run_metabolic_trajectory_study", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load_runtime_packages()
    return module


def load_cosmos(run_dir: Path, scripts_dir: Path, work_dir: Path) -> StudyData:
    """Rebuild the analytic cohort, its splits, and its prediction rows from Cosmos with the study's
    own acquire, split, and row builders under the run's recorded cohort settings."""
    study = import_study(scripts_dir)
    context = run_context(run_dir)
    dependencies, issues = study.dependency_manifest(require_database=True)
    if issues:
        raise RuntimeError("Cosmos fallback cannot run: " + "; ".join(issues))
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = study.RunConfig.create("production", str(work_dir), False,
                                 incretin_qualifying_months=context["incretin_qualifying_months"])
    log("re-acquiring the cohort from Cosmos with the study's own acquire (this takes a while)")
    artifacts, _ = study.stream_wide_acquire(cfg)
    cohorts, split_metadata = study.assign_global_splits(artifacts["cohorts"])
    store = study.RowStore(work_dir / "rows")
    study.build_prediction_rows(cohorts, artifacts["measurements"], store)
    expected = {name: int(count) for name, count in store.manifest()["partitions"].items()}
    rows, schema = read_origin0_rows(store.root, expected)
    slim = slim_cohorts(cohorts)
    del artifacts, cohorts
    gc.collect()
    provenance = {**context, "source": "Cosmos re-acquire",
                  "detail": f"study {study.STUDY_VERSION}, acquired "
                            f"{datetime.now(timezone.utc).date().isoformat()}"}
    return StudyData(slim, rows, schema, dict(split_metadata), provenance)


def load_study_data(run_dir: Path, source: str, scripts_dir: Path, work_dir: Path) -> StudyData:
    if source in ("auto", "checkpoint"):
        try:
            return load_frozen(run_dir)
        except SourceUnavailable as error:
            if source == "checkpoint":
                raise
            log(f"frozen checkpoints unusable ({error}); falling back to a Cosmos re-acquire")
            data = load_cosmos(run_dir, scripts_dir, work_dir)
            data.provenance["detail"] += f"; frozen checkpoints unusable: {error}"
            return data
    return load_cosmos(run_dir, scripts_dir, work_dir)


# ======================================================================================
# 3. Patient-level preparation (inside the VM only; nothing here is written out).
# ======================================================================================


def arm_of(cohort: Any, treatment: Any, procedure: Any) -> str | None:
    """incretin / rygb / sleeve, with the same rule as the shared three-arm wrapper's arm_series."""
    cohort = str(cohort).strip().lower()
    if cohort == "incretin":
        return "incretin"
    if cohort != "surgery":
        return None
    text = f"{treatment} {procedure}".lower()
    if re.search(r"rygb|roux|gastric\s*bypass", text):
        return "rygb"
    if "sleeve" in text:
        return "sleeve"
    return None


def numeric(series: Any) -> Any:
    return pd.to_numeric(series, errors="coerce").astype(float)


def prepare_patients(cohorts: Any) -> tuple[Any, dict[str, dict[str, Counter]], list[str]]:
    frame = cohorts.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["cohort"] = frame["cohort"].astype(str).str.strip().str.lower()
    frame = frame.drop_duplicates(["patient_id", "cohort"], keep="first").reset_index(drop=True)
    warnings: list[str] = []
    arms = [arm_of(c, t, p) for c, t, p in zip(frame["cohort"], frame["treatment"], frame["procedure"])]
    frame["arm"] = arms
    unassigned = int(frame["arm"].isna().sum())
    if unassigned:
        warnings.append(f"{unassigned} cohort rows have no SG/RYGB/IBT arm and are excluded")
        frame = frame.loc[frame["arm"].notna()].reset_index(drop=True)

    out = pd.DataFrame({"patient_id": frame["patient_id"], "cohort": frame["cohort"], "arm": frame["arm"],
                        "split": frame["split"].astype(str)})
    crosswalks: dict[str, dict[str, Counter]] = {}
    for column, mapper in (("sex", sex_category), ("race", race_category),
                           ("ethnicity", ethnicity_category), ("coverage", coverage_category),
                           ("smoking", smoking_category)):
        out[column], crosswalks[column] = mapped_text(frame[column], mapper)
    out["age"] = numeric(frame["age_at_index"])
    out["age_group"] = [age_category(value) for value in out["age"].to_numpy(float)]
    svi = numeric(frame["svi"])
    out_of_range = int((svi.notna() & ~svi.between(0.0, 1.0)).sum())
    if out_of_range:
        warnings.append(f"{out_of_range} SVI values outside [0, 1] treated as not recorded")
    out["svi"] = svi.where(svi.between(0.0, 1.0))
    out["ruca"] = [ruca_category(value) for value in frame["ruca"]]
    out["bmi"] = numeric(frame["baseline_bmi"])
    out["bmi_group"] = [bmi_category(value) for value in out["bmi"].to_numpy(float)]
    out["hba1c"] = numeric(frame["baseline_hba1c"])
    out["hba1c_group"] = [hba1c_category(value) for value in out["hba1c"].to_numpy(float)]
    out["eligible_hba1c"] = frame["diabetes_eligible"].fillna(False).astype(bool)
    for name, _ in (*FLAG_ROWS, *MEDICATION_ROWS):
        out[name] = numeric(frame[name]).eq(1.0)
    for name in OPTIONAL_WIDE_NUMERIC_COVARIATES:
        out[name] = numeric(frame[name])
    index_year = pd.to_datetime(frame["index_date"], errors="coerce").dt.year.astype(float)
    if index_year.isna().any():
        raise RuntimeError("cohort rows without an index date; the cohort frame is malformed")
    out["index_year"] = index_year
    out["year_group"] = [year_category(value) for value in index_year.to_numpy(float)]
    out["agent"] = [agent_category(ingredient, route) if arm == "incretin" else ""
                    for arm, ingredient, route in zip(frame["arm"], frame["index_ingredient"], frame["index_route"])]
    return out, crosswalks, warnings


def dense_enough(patients: Any, column: str, informative: Any) -> bool:
    """The study's rule for whether an optional covariate became a model feature: recorded for at
    least half of EVERY cohort and carrying more than one value. Used only when the prediction-row
    schema is unavailable; otherwise the schema itself says which inputs the models received."""
    by_cohort = informative.groupby(patients["cohort"]).mean()
    return bool(len(by_cohort) and by_cohort.min() >= OPTIONAL_COVARIATE_THRESHOLD
                and patients.loc[informative, column].nunique() > 1)


# ======================================================================================
# 4. Disclosure control: primary suppression plus audited complementary suppression.
# ======================================================================================


@dataclass
class Constraint:
    """sum(coefficient * true value of ref) == rhs, where rhs is built only from published totals."""

    terms: list[tuple[int, tuple]]
    rhs: int
    description: str


class DisclosureControl:
    """Small-cell suppression that a reader cannot undo from the published table.

    Every published count is a cell, and every identity a reader could exploit is an equality over
    cells: category partitions summing to the column total, SG + RYGB = All MBS, availability rows
    complementing their "not recorded" level, held-out outcomes within the held-out partition.
    Undisplayed complements (the patients without a comorbidity) are hidden unknowns. A reader also
    learns bounds from the masks themselves: "<11" is 0 to 10, "Masked‡" is at least 11 (it would
    otherwise read "<11"), and "All but <11" is 1 to 10 short of the column total.

    A target is protected when its true count is 1 to 10: a masked cell, a hidden complement, or the
    combined count of several small cells of one identity. After primary suppression (every count
    below 11 masked), cells are masked until, for every protected target, the smallest and largest
    counts consistent with everything published - two small linear programs over that target's
    connected block of unknowns - are at least PROTECTION_WIDTH apart. So no count of 1 to 10 can be
    recovered, or narrowed to fewer than six possible values. Zero counts are masked for display but
    are not protected:
    learning that a category is empty reveals no patient.
    """

    PROTECTION_WIDTH = 5.0

    def __init__(self) -> None:
        self.values: dict[tuple, int] = {}
        self.order: dict[tuple, int] = {}
        self.limits: dict[tuple, int] = {}
        self.complement: dict[tuple, int] = {}
        self.constraints: list[Constraint] = []
        self.masked: dict[tuple, str] = {}
        # Secondary masks published with their rounded percentage, and the counts that implies.
        self.percent_bounds: dict[tuple, tuple[int, int]] = {}
        self._hidden = 0
        # A block's program is fixed by which unknowns it holds and their bounds, so ranges are
        # reused across the many re-checks of the greedy loop and the pruning pass.
        self._ranges: dict[tuple, tuple[float, float]] = {}

    def _register(self, ref: tuple, value: int, limit: int) -> tuple:
        self.values[ref], self.limits[ref], self.order[ref] = int(value), int(limit), len(self.order)
        return ref

    def cell(self, row: int, column: str, value: int, limit: int) -> tuple:
        return self._register(("cell", row, column), value, limit)

    def hidden(self, value: int, limit: int) -> tuple:
        self._hidden += 1
        return self._register(("hidden", self._hidden), value, limit)

    def add(self, terms: Sequence[tuple[int, tuple]], rhs: int, description: str) -> None:
        actual = sum(coefficient * self.values[ref] for coefficient, ref in terms)
        if actual != int(rhs):
            raise ValueError(f"internal table identity does not hold ({description}): {actual} != {rhs}")
        self.constraints.append(Constraint(list(terms), int(rhs), description))

    def unknown(self, ref: tuple) -> bool:
        return ref[0] == "hidden" or ref in self.masked

    def shows_complement(self, ref: tuple) -> bool:
        """A secondary mask on a near-complete count reads "All but <11" rather than "Masked‡"."""
        return self.masked.get(ref) == "secondary" and 1 <= self.complement.get(ref, 0) < MIN_CELL

    def _bounds(self, ref: tuple) -> tuple[int, int]:
        limit = self.limits[ref]
        if ref[0] == "hidden":
            return 0, limit
        if self.masked[ref] == "primary":
            return 0, MIN_CELL - 1
        if self.shows_complement(ref):
            return limit - (MIN_CELL - 1), limit - 1
        if ref in self.percent_bounds:
            return self.percent_bounds[ref]
        return MIN_CELL, limit

    def _system(self) -> tuple[dict[tuple, tuple], dict[tuple, list[tuple]], list[tuple[dict[tuple, int], int]]]:
        """Each identity restricted to the unknowns (known cells moved to the right-hand side), and
        the unknowns grouped into connected blocks so every program stays small."""
        unknowns = sorted((ref for ref in self.values if self.unknown(ref)), key=self.order.__getitem__)
        parent = {ref: ref for ref in unknowns}

        def find(ref: tuple) -> tuple:
            while parent[ref] != ref:
                parent[ref] = parent[parent[ref]]
                ref = parent[ref]
            return ref

        equations: list[tuple[dict[tuple, int], int]] = []
        for constraint in self.constraints:
            coefficients: dict[tuple, int] = {}
            rhs = constraint.rhs
            for coefficient, ref in constraint.terms:
                if ref in parent:
                    coefficients[ref] = coefficients.get(ref, 0) + coefficient
                else:
                    rhs -= coefficient * self.values[ref]
            coefficients = {ref: value for ref, value in coefficients.items() if value}
            if coefficients:
                equations.append((coefficients, rhs))
                first = next(iter(coefficients))
                for ref in coefficients:
                    parent[find(ref)] = find(first)
        root_of = {ref: find(ref) for ref in unknowns}
        blocks: dict[tuple, list[tuple]] = {}
        for ref in unknowns:
            blocks.setdefault(root_of[ref], []).append(ref)
        return root_of, blocks, equations

    def _range(self, system: Any, members: Sequence[tuple]) -> tuple[float, float]:
        from scipy.optimize import linprog

        root_of, blocks, equations = system
        root = root_of[members[0]]
        variables = blocks[root]
        bounds = [self._bounds(ref) for ref in variables]
        key = (tuple(members), tuple(variables), tuple(bounds))
        if key in self._ranges:
            return self._ranges[key]
        index = {ref: position for position, ref in enumerate(variables)}
        rows = [(coefficients, rhs) for coefficients, rhs in equations if root_of[next(iter(coefficients))] == root]
        matrix = np.zeros((len(rows), len(variables)))
        rhs = np.zeros(len(rows))
        for position, (coefficients, value) in enumerate(rows):
            for ref, coefficient in coefficients.items():
                matrix[position, index[ref]] = coefficient
            rhs[position] = value
        objective = np.zeros(len(variables))
        for ref in members:
            objective[index[ref]] = 1.0
        problem = {"A_eq": matrix if rows else None, "b_eq": rhs if rows else None,
                   "bounds": bounds, "method": "highs"}
        low, high = linprog(objective, **problem), linprog(-objective, **problem)
        if low.status != 0 or high.status != 0:
            raise DisclosureError(f"disclosure audit could not bound {members}: {low.message} / {high.message}")
        self._ranges[key] = (float(low.fun), float(-high.fun))
        return self._ranges[key]

    def _targets(self) -> list[tuple[tuple, ...]]:
        """Protected targets: single unknowns with a true count of 1 to 10, then combinations of
        small unknowns within one identity whose combined count is 1 to 10."""
        targets = [(ref,) for ref in sorted(self.values, key=self.order.__getitem__)
                   if self.unknown(ref) and 1 <= self.values[ref] < MIN_CELL]
        seen: set[frozenset] = set()
        for constraint in self.constraints:
            small = sorted({ref for _, ref in constraint.terms if self.unknown(ref) and self.values[ref] < MIN_CELL},
                           key=self.order.__getitem__)
            sizes = range(2, len(small) + 1) if len(small) <= 8 else (2, len(small))
            for size in sizes:
                for group in itertools.combinations(small, size):
                    key = frozenset(group)
                    if key not in seen and 1 <= sum(self.values[ref] for ref in group) < MIN_CELL:
                        seen.add(key)
                        targets.append(group)
        return targets

    def _exposed(self, system: Any, members: Sequence[tuple]) -> bool:
        low, high = self._range(system, members)
        return high - low < self.PROTECTION_WIDTH - 1e-7

    def violations(self, first_only: bool = False) -> list[tuple[tuple, ...]]:
        system = self._system()
        found = []
        for members in self._targets():
            if self._exposed(system, members):
                found.append(members)
                if first_only:
                    break
        return found

    def _cost(self, ref: tuple) -> tuple:
        # Prefer masks that still read "All but <11", then the smallest count, so as little as
        # possible is hidden from the reader.
        reads_as_complement = 1 <= self.complement.get(ref, 0) < MIN_CELL
        return (0 if reads_as_complement else 1, self.values[ref], self.order[ref])

    def _candidates(self, members: Sequence[tuple]) -> list[tuple]:
        """Displayed cells sharing an identity with the exposed target, nearest identities first."""
        frontier, seen = set(members), set()
        for _ in range(6):
            layer = [position for position, constraint in enumerate(self.constraints)
                     if position not in seen and any(ref in frontier for _, ref in constraint.terms)]
            if not layer:
                break
            seen.update(layer)
            displayed = {ref for position in layer for _, ref in self.constraints[position].terms
                         if ref[0] == "cell" and ref not in self.masked}
            if displayed:
                return sorted(displayed, key=self._cost)
            frontier = {ref for position in layer for _, ref in self.constraints[position].terms if self.unknown(ref)}
        return []

    def run(self) -> None:
        for ref, value in self.values.items():
            if ref[0] == "cell" and value < MIN_CELL:
                self.masked[ref] = "primary"
        for _ in range(len(self.values) + 1):
            pending = self.violations(first_only=True)
            if not pending:
                self._prune()
                return
            members = pending[0]
            candidates = self._candidates(members)
            if not candidates:
                raise DisclosureError(f"cannot protect a small count ({members}); nothing was written")
            # The cheapest cell whose masking alone protects this target; failing that, the cheapest
            # cell, and the next pass finds what else is needed.
            for candidate in candidates:
                self.masked[candidate] = "secondary"
                if not self._exposed(self._system(), members):
                    break
                del self.masked[candidate]
            else:
                self.masked[candidates[0]] = "secondary"
        raise DisclosureError("complementary suppression did not converge; nothing was written")

    def _prune(self) -> None:
        """Greedy masking can over-suppress: a cell masked early may be redundant once later cells
        are masked. Reveal each secondary cell again, largest first, wherever the table stays safe."""
        secondary = [ref for ref, state in self.masked.items() if state == "secondary"]
        for ref in sorted(secondary, key=lambda item: (-self.values[item], -self.order[item])):
            del self.masked[ref]
            if self.violations(first_only=True):
                self.masked[ref] = "secondary"

    def reveal_percentages(self, proposals: Mapping[tuple, tuple[int, int]]) -> None:
        """Publish the rounded percentage of a secondary mask wherever the counts it implies still
        leave every small count protected; largest cells first, since they gain the most."""
        for ref in sorted(proposals, key=lambda item: (-self.values[item], -self.order[item])):
            low, high = proposals[ref]
            if self.masked.get(ref) != "secondary" or self.shows_complement(ref) or not low <= self.values[ref] <= high:
                continue
            self.percent_bounds[ref] = (low, high)
            if self.violations(first_only=True):
                del self.percent_bounds[ref]

    def audit(self) -> list[str]:
        """Final check of what a reader can see and what they could derive from it."""
        problems = [f"displayed count {value} below {MIN_CELL} at {ref}" for ref, value in self.values.items()
                    if ref[0] == "cell" and ref not in self.masked and value < MIN_CELL]
        problems += [f"recoverable small count: {members}" for members in self.violations()]
        return problems


# ======================================================================================
# 5. The table: rows, identities, and display text.
# ======================================================================================


@dataclass
class TableModel:
    """Everything the PNG shows. It round-trips through the CSV losslessly."""

    title: str
    subtitle: str
    column_titles: dict[str, str]
    column_ns: dict[str, str]
    body: list[dict[str, Any]]        # {"kind", "label", "indent", "cells": {column: text}, "source_values"}
    footnotes: list[str]
    footer: str


class TableBuilder:
    def __init__(self, n: Mapping[str, int]) -> None:
        self.n = {column: int(n[column]) for column in COLUMNS}
        self.rows: list[dict[str, Any]] = []
        self.dc = DisclosureControl()

    def _append(self, kind: str, label: str, indent: int = 0, **extra: Any) -> int:
        self.rows.append({"kind": kind, "label": label, "indent": indent, **extra})
        return len(self.rows) - 1

    def section(self, label: str) -> None:
        self._append("section", label)

    def heading(self, label: str) -> None:
        self._append("variable", label)

    def stat(self, label: str, texts: Mapping[str, str], indent: int = 0) -> None:
        self._append("stat", label, indent, texts=dict(texts))

    def counts(self, label: str, values: Mapping[str, int | None], indent: int = 0, pct: bool = True,
               source_values: str = "") -> int:
        row = self._append("count", label, indent, values=dict(values), pct=pct, refs={},
                           source_values=source_values)
        refs = self.rows[row]["refs"]
        for column in COLUMNS:
            value = values.get(column)
            if value is not None:
                refs[column] = self.dc.cell(row, column, int(value), self.n[column])
        if all(column in refs for column in ADDITIVE):
            self.dc.add([(1, refs["sg"]), (1, refs["rygb"]), (-1, refs["all_mbs"])], 0, f"SG + RYGB = All MBS: {label}")
        return row

    def ref(self, row: int, column: str) -> tuple | None:
        return self.rows[row]["refs"].get(column)

    def partition(self, rows: Sequence[int], label: str) -> None:
        """The rows are the complete set of levels of one variable within each column."""
        for column in COLUMNS:
            refs = [self.ref(row, column) for row in rows]
            refs = [ref for ref in refs if ref is not None]
            if refs:
                self.dc.add([(1, ref) for ref in refs], self.n[column], f"{label} partition [{column}]")

    def binary(self, row: int) -> None:
        """A single published count whose complement within the column is not shown."""
        for column in COLUMNS:
            ref = self.ref(row, column)
            if ref is not None:
                rest = self.n[column] - self.dc.values[ref]
                hidden = self.dc.hidden(rest, self.n[column])
                self.dc.add([(1, ref), (1, hidden)], self.n[column], f"{self.rows[row]['label']} [{column}]")
                self.dc.complement[ref] = rest

    def link(self, row: int, other: int) -> None:
        """``row`` plus the published ``other`` row is the column total (recorded vs not recorded)."""
        for column in COLUMNS:
            ref, partner = self.ref(row, column), self.ref(other, column)
            if ref is not None and partner is not None:
                self.dc.add([(1, ref), (1, partner)], self.n[column], f"{self.rows[row]['label']} [{column}]")
                self.dc.complement[ref] = self.dc.values[partner]

    def subset(self, row: int, totals: Sequence[int]) -> None:
        """A count of a subset of the patients counted by the published ``totals`` rows."""
        for column in COLUMNS:
            ref = self.ref(row, column)
            total_refs = [self.ref(total, column) for total in totals]
            if ref is None or any(total is None for total in total_refs):
                continue
            rest = sum(self.dc.values[total] for total in total_refs) - self.dc.values[ref]
            hidden = self.dc.hidden(rest, self.n[column])
            self.dc.add([(1, ref), (1, hidden)] + [(-1, total) for total in total_refs], 0,
                        f"{self.rows[row]['label']} within held-out [{column}]")

    def equal(self, row: int, first: str, second: str) -> None:
        self.dc.add([(1, self.ref(row, first)), (-1, self.ref(row, second))], 0, f"{self.rows[row]['label']} equality")

    def body(self) -> list[dict[str, Any]]:
        self.dc.run()
        self.dc.reveal_percentages({
            ref: percent_count_bounds(format_percent(self.dc.values[ref], self.n[column]), self.n[column])
            for row in self.rows if row["kind"] == "count" and row["pct"]
            for column, ref in row["refs"].items() if self.dc.masked.get(ref) == "secondary"})
        problems = self.dc.audit()
        if problems:
            raise DisclosureError("disclosure audit failed: " + "; ".join(problems[:5]))
        body = []
        for position, row in enumerate(self.rows):
            entry = {"kind": "value" if row["kind"] in {"count", "stat"} else row["kind"],
                     "label": row["label"], "indent": row["indent"], "cells": {},
                     "source_values": row.get("source_values", "")}
            for column in COLUMNS:
                if row["kind"] == "stat":
                    entry["cells"][column] = row["texts"].get(column, NOT_APPLICABLE)
                elif row["kind"] == "count":
                    entry["cells"][column] = self._count_text(position, column)
                else:
                    entry["cells"][column] = ""
            body.append(entry)
        return body

    def _count_text(self, row: int, column: str) -> str:
        ref = self.ref(row, column)
        if ref is None:
            return NOT_APPLICABLE
        state = self.dc.masked.get(ref)
        if state == "primary":
            return MASK_PRIMARY
        if state == "secondary":
            if self.dc.shows_complement(ref):
                return MASK_COMPLEMENT
            if ref in self.dc.percent_bounds:
                return f"{MASK_SECONDARY} ({format_percent(self.dc.values[ref], self.n[column])})"
            return MASK_SECONDARY
        value = self.dc.values[ref]
        if not self.rows[row]["pct"]:
            return f"{value:,}"
        return f"{value:,} ({format_percent(value, self.n[column])})"


def format_percent(value: int, total: int) -> str:
    share = 100.0 * value / total if total else 0.0
    if 0 < value and share < 0.05:
        return "<0.1"
    if value < total and share >= 99.95:
        return ">99.9"
    return f"{share:.1f}"


def percent_count_bounds(text: str, total: int) -> tuple[int, int]:
    """The counts a reader can infer from a published one-decimal percentage of ``total`` (closed
    interval, so the audit never credits the reader with less than they know), within the 11-or-more
    a secondary mask already implies."""
    if text == "<0.1":
        low, high = 0.0, 0.05
    elif text == ">99.9":
        low, high = 99.95, 100.0
    else:
        low, high = float(text) - 0.05, float(text) + 0.05
    return (max(MIN_CELL, math.ceil(total * low / 100.0 - 1e-9)),
            min(total, math.floor(total * high / 100.0 + 1e-9)))


def recorded_values(values: Any) -> Any:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def mean_sd_text(values: Any, decimals: int) -> str:
    data = recorded_values(values)
    if not len(data):
        return "Not recorded"
    if len(data) < MIN_CELL:
        return "<11 recorded"
    return f"{data.mean():.{decimals}f} ({data.std(ddof=1):.{decimals}f})"


def median_iqr_text(values: Any, decimals: int) -> str:
    data = recorded_values(values)
    if not len(data):
        return "Not recorded"
    if len(data) < MIN_CELL:
        return "<11 recorded"
    q1, median, q3 = np.quantile(data, [0.25, 0.5, 0.75])
    return f"{median:.{decimals}f} [{q1:.{decimals}f}-{q3:.{decimals}f}]"


def held_out_label(splits: Sequence[str]) -> str:
    present = [name for name in HELD_OUT_SPLITS if name in splits]
    kinds = " and ".join(name.split("_")[0] for name in present) or "temporal"
    return f"held-out {kinds} test"


def build_table(data: StudyData, generated_utc: str) -> tuple[TableModel, list[str]]:
    patients, crosswalks, warnings = prepare_patients(data.cohorts)
    frames = {
        "sg": patients.loc[patients["arm"].eq("sleeve")],
        "rygb": patients.loc[patients["arm"].eq("rygb")],
        "all_mbs": patients.loc[patients["arm"].isin(["sleeve", "rygb"])],
        "ibt": patients.loc[patients["arm"].eq("incretin")],
    }
    n = {column: len(frame) for column, frame in frames.items()}
    small = [COLUMN_TITLES[column] for column, count in n.items() if count < MIN_CELL]
    if small:
        raise DisclosureError(f"column(s) with fewer than {MIN_CELL} patients: {', '.join(small)}")
    b = TableBuilder(n)
    surgical = ("sg", "rygb", "all_mbs")

    def tally(test: Callable[[Any], Any], columns: Sequence[str] = COLUMNS) -> dict[str, int | None]:
        return {column: (int(np.asarray(test(frames[column]), dtype=bool).sum()) if column in columns else None)
                for column in COLUMNS}

    def categorical(heading: str, column: str, levels: Sequence[str], crosswalk: Mapping[str, Counter] | None = None,
                    columns: Sequence[str] = COLUMNS,
                    excluded_by_design: Mapping[str, Sequence[str]] | None = None) -> dict[str, int]:
        """``excluded_by_design`` maps a level to the columns whose eligibility criteria rule it out;
        there it reads "-" rather than a "<11" that would suggest a few patients exist."""
        b.heading(heading)
        rows: dict[str, int] = {}
        for level in levels:
            values = tally(lambda frame, level=level: frame[column].eq(level), columns)
            for name in (excluded_by_design or {}).get(level, ()):
                if values[name] == 0:
                    values[name] = None
            if not any(values[name] for name in columns if values[name] is not None):
                continue  # a level empty in every column is not shown
            rows[level] = b.counts(level, values, indent=1,
                                   source_values=source_values_text((crosswalk or {}).get(level)))
        unexpected = set().union(*(set(frames[name][column].unique()) for name in columns)) - set(levels)
        if unexpected:
            raise ValueError(f"internal: unmapped {column} levels {sorted(map(str, unexpected))}")
        b.partition(list(rows.values()), heading)
        return rows

    def stat_row(label: str, column: str, formatter: Callable[[Any, int], str], decimals: int) -> None:
        b.stat(label, {name: formatter(frames[name][column].to_numpy(float), decimals) for name in COLUMNS})

    def binary_row(label: str, test: Callable[[Any], Any], indent: int = 0) -> int:
        row = b.counts(label, tally(test), indent=indent)
        b.binary(row)
        return row

    # --- Analytic sample -----------------------------------------------------------------
    b.section("Analytic sample")
    b.heading("Model partition, n (%)")
    split_rows: dict[str, int] = {}
    present_splits = set(patients["split"].unique())
    unknown_splits = present_splits - set(SPLIT_ORDER)
    if unknown_splits:
        raise ValueError(f"internal: unexpected split labels {sorted(unknown_splits)}")
    for split in SPLIT_ORDER:
        if split in present_splits:
            split_rows[split] = b.counts(SPLIT_LABELS[split], tally(lambda frame, split=split: frame["split"].eq(split)), indent=1)
    b.partition(list(split_rows.values()), "model partition")
    overlap = len(set(frames["all_mbs"]["patient_id"]) & set(frames["ibt"]["patient_id"]))
    overlap_row = b.counts("Also in the other treatment cohort, n",
                           {"sg": None, "rygb": None, "all_mbs": overlap, "ibt": overlap}, pct=False)
    b.equal(overlap_row, "all_mbs", "ibt")

    # --- Treatment ------------------------------------------------------------------------
    b.section("Treatment at index, n (%)")
    sleeve_row = b.counts("Sleeve gastrectomy", {"sg": n["sg"], "rygb": None, "all_mbs": n["sg"], "ibt": None})
    rygb_row = b.counts("Roux-en-Y gastric bypass", {"sg": None, "rygb": n["rygb"], "all_mbs": n["rygb"], "ibt": None})
    b.partition([sleeve_row, rygb_row], "procedure")
    categorical("Incretin agent at index", "agent", AGENT_LEVELS, columns=("ibt",))
    stat_row("Index year, median [IQR]", "index_year", median_iqr_text, 0)
    categorical("Index year, n (%)", "year_group", YEAR_LEVELS)

    # --- Demographics ---------------------------------------------------------------------
    b.section("Demographics")
    stat_row("Age at index, years, mean (SD)", "age", mean_sd_text, 1)
    age_rows = categorical("Age group, years, n (%)", "age_group", AGE_LEVELS)
    def demographic(label: str, column: str, levels: Sequence[str], unknown_level: str) -> dict[str, int] | None:
        # A field the source tables do not carry at all (every value unknown) gets one explicit
        # row rather than a column of "<11" cells that would suggest a handful were recorded.
        if not patients[column].ne(unknown_level).any():
            b.stat(label, {name: NOT_IN_SOURCE for name in COLUMNS})
            return None
        return categorical(f"{label}, n (%)", column, levels, crosswalks[column])

    sex_rows = demographic("Sex", "sex", ("Female", "Male", "Other", "Unknown or not recorded"), "Unknown or not recorded")
    race_rows = demographic("Race", "race", RACE_LEVELS, "Unknown or not reported")
    ethnicity_rows = demographic("Ethnicity", "ethnicity", ETHNICITY_LEVELS, "Unknown or not reported")
    coverage_rows = demographic("Insurance coverage", "coverage", COVERAGE_LEVELS, "Unknown or not reported")
    smoking_rows = demographic("Smoking status", "smoking", SMOKING_LEVELS, "Unknown or not recorded")
    stat_row("Social Vulnerability Index, median [IQR]", "svi", median_iqr_text, 2)
    categorical("Rurality, n (%)", "ruca", RUCA_LEVELS)

    # --- Baseline measurements ------------------------------------------------------------
    b.section("Baseline measurements")
    stat_row("BMI, kg/m\u00b2, mean (SD)", "bmi", mean_sd_text, 1)
    # Surgical eligibility requires a baseline BMI of 35 to 75 kg/m2.
    categorical("BMI category, kg/m\u00b2, n (%)", "bmi_group", BMI_LEVELS, excluded_by_design={"30.0-34.9": surgical})
    stat_row("HbA1c, %, mean (SD)", "hba1c", mean_sd_text, 1)
    hba1c_rows = categorical("HbA1c category, %, n (%)", "hba1c_group", HBA1C_LEVELS)
    labs = [(column, label, decimals) for column, label, decimals in LAB_ROWS
            if column in data.cohorts.attrs.get("source_columns", ()) and patients[column].notna().any()]
    for column, label, decimals in labs:
        stat_row(f"{label}, median [IQR]", column, median_iqr_text, decimals)

    # --- Comorbidities and medications ----------------------------------------------------
    b.section("Comorbidities, n (%)")
    for column, label in FLAG_ROWS:
        binary_row(label, lambda frame, column=column: frame[column])
    gerd_acquired = GERD_COLUMN in data.cohorts.attrs.get("source_columns", ()) and patients[GERD_COLUMN].notna().any()
    if gerd_acquired:
        binary_row("Gastroesophageal reflux disease", lambda frame: frame[GERD_COLUMN].eq(1.0))
    binary_row("Type 2 diabetes with a baseline HbA1c", lambda frame: frame["eligible_hba1c"])
    b.section("Baseline glucose-lowering medications, n (%)")
    for column, label in MEDICATION_ROWS:
        binary_row(label, lambda frame, column=column: frame[column])

    # --- Model input availability ---------------------------------------------------------
    # Which optional inputs the models actually received is read off the prediction-row schema.
    schema = set(data.row_columns)
    if schema:
        used_optional = {name for name in (*OPTIONAL_WIDE_NUMERIC_COVARIATES, "smoking") if name in schema}
    else:
        used_optional = {name for name in OPTIONAL_WIDE_NUMERIC_COVARIATES
                         if dense_enough(patients, name, patients[name].notna())}
        if dense_enough(patients, "smoking", patients["smoking"].ne("Unknown or not recorded")):
            used_optional.add("smoking")
    unused_marker = "\u00a7"
    b.section("Model input availability, n (%) with a recorded value")

    def available(label: str, unknown_rows: Mapping[str, int] | None = None, unknown_level: str = "",
                  test: Callable[[Any], Any] | None = None) -> None:
        if unknown_level and unknown_rows is None:
            b.stat(label, {column: NOT_IN_SOURCE for column in COLUMNS})
        elif unknown_rows is not None and unknown_level in unknown_rows:
            values = {column: n[column] - b.dc.values[b.ref(unknown_rows[unknown_level], column)] for column in COLUMNS}
            row = b.counts(label, values)
            b.link(row, unknown_rows[unknown_level])
        else:
            row = b.counts(label, tally(test) if test is not None else dict(n))
            b.binary(row)

    available("Baseline BMI", test=lambda frame: frame["bmi"].notna())
    available("Baseline HbA1c", hba1c_rows, "Not measured")
    available("Age at index", age_rows, "Not recorded")
    available("Sex", sex_rows, "Unknown or not recorded")
    available("Race", race_rows, "Unknown or not reported")
    available("Ethnicity", ethnicity_rows, "Unknown or not reported")
    available("Insurance coverage", coverage_rows, "Unknown or not reported")
    unused_smoking = smoking_rows is not None and "smoking" not in used_optional
    available("Smoking status" + (unused_marker if unused_smoking else ""), smoking_rows, "Unknown or not recorded")
    available("Social Vulnerability Index", test=lambda frame: frame["svi"].notna())
    available("Index year")
    available("Treatment (procedure, or incretin agent and route)")
    available("History flags: T2D, hypertension, dyslipidemia, OSA, insulin, metformin, SGLT2\u2020")
    optional_inputs = [(column, label) for column, label, _ in labs]
    if gerd_acquired:
        optional_inputs.append((GERD_COLUMN, "Gastroesophageal reflux disease"))
    for column, label in optional_inputs:
        available(label + ("" if column in used_optional else unused_marker),
                  test=lambda frame, column=column: frame[column].notna())

    # --- Held-out outcomes ----------------------------------------------------------------
    rows = data.outcome_rows.copy()
    arm_lookup = dict(zip(patients["cohort"] + "|" + patients["patient_id"], patients["arm"]))
    rows["arm"] = (rows["cohort"].astype(str) + "|" + rows["patient_id"].astype(str)).map(arm_lookup)
    orphan = int(rows["arm"].isna().sum())
    if orphan:
        raise RuntimeError(f"{orphan} held-out prediction rows match no cohort patient; the run is inconsistent")
    heldout_rows = [split_rows[name] for name in HELD_OUT_SPLITS if name in split_rows]
    if heldout_rows and len(rows):
        b.section(f"Outcomes observed in {held_out_label(list(split_rows))} patients, n")
    for outcome, heading in (("bmi", "BMI"), ("hba1c", "HbA1c (patients with T2D and a baseline HbA1c)")):
        months = sorted(int(month) for month in rows.loc[rows["outcome"].eq(outcome), "target_month"].unique())
        if not months or not heldout_rows:
            continue
        b.heading(heading)
        subset = rows.loc[rows["outcome"].eq(outcome)]
        for month in months:
            at = subset.loc[subset["target_month"].eq(month)]
            values = {
                "sg": at.loc[at["arm"].eq("sleeve"), "patient_id"].nunique(),
                "rygb": at.loc[at["arm"].eq("rygb"), "patient_id"].nunique(),
                "all_mbs": at.loc[at["arm"].isin(["sleeve", "rygb"]), "patient_id"].nunique(),
                "ibt": at.loc[at["arm"].eq("incretin"), "patient_id"].nunique(),
            }
            row = b.counts(f"{month} months", values, indent=1, pct=False)
            b.subset(row, heldout_rows)

    body = b.body()

    # --- Titles and footnotes --------------------------------------------------------------
    months = data.provenance.get("incretin_qualifying_months", 12)
    cutoff = data.split_metadata.get("temporal_cutoff")
    geographic = "geographic_test" in split_rows
    footnotes = [
        "Values are n (%) unless otherwise indicated; percentages use the column total as the denominator. "
        "Means (SD) and medians [IQR] are among patients with a recorded value. A dash means not applicable, "
        "including a category ruled out by a cohort's eligibility criteria (surgical eligibility requires a BMI "
        "of 35 to 75 kg/m²).",
        "Disclosure control (Epic Cosmos policy): any patient count below 11, including zero, is shown as <11. "
        "Masked\u2021 counts (11 or more patients) were also suppressed so that no count below 11 can be recovered "
        "from this table, even approximately; a percentage after Masked\u2021 is the rounded share, shown only where "
        "it keeps every small count protected. All but <11 means that fewer than 11 patients lacked the value.",
        "MBS, metabolic and bariatric surgery: sleeve gastrectomy (SG; CPT 43775) or Roux-en-Y gastric bypass "
        "(RYGB; CPT 43644, 43645, or 43846). IBT, incretin-based therapy (GLP-1 receptor agonist or dual "
        f"GIP/GLP-1 receptor agonist): new users with at least {months} months of continuous recorded therapy.",
        "Model partition is assigned per patient. Development partitions were used for model fitting, "
        "calibration, and selection; the held-out test partition holds patients indexed on or after "
        f"{cutoff or 'the temporal cutoff'}"
        + (" and patients from held-out centers." if geographic else
           " (no center identifier was available, so there is no geographic holdout)."),
        "Model input availability counts patients with a recorded value (not missing and not unknown) for each "
        "baseline input of the forecasting models. \u2020 History flags come from source history fields "
        "recorded as present or absent, so they carry no missing values. \u00a7 Present in the source but not a "
        "model input, because it was recorded for fewer than half of the patients in at least one cohort.",
        "Held-out outcomes: patients in the held-out test partition with an observed value in the target window "
        "(3 months: 2.0-4.5; 6: 4.5-9; 12: 9-18; 24: 18-30; 36: 30-42; 48: 42-54; 60: 54-66 months after index), "
        "forecast from index. Counts are per horizon and are not additive across horizons.",
        "BMI, body mass index; HbA1c, glycated hemoglobin; T2D, type 2 diabetes; OSA, obstructive sleep apnea; "
        "SGLT2, sodium-glucose cotransporter 2; SVI, CDC/ATSDR Social Vulnerability Index (overall percentile "
        "rank); RUCA, rural-urban commuting area code; eGFR, estimated glomerular filtration rate; LDL and HDL, "
        "low- and high-density lipoprotein.",
        f"Source: {data.provenance['run_name']} ({data.provenance['study_version']}); "
        f"{data.provenance['source']}, {data.provenance['detail']}. Generated {generated_utc}.",
    ]
    model = TableModel(
        title="Table 1. Baseline characteristics of the analytic cohorts",
        subtitle="Metabolic and bariatric surgery (MBS) and incretin-based therapy (IBT) cohorts of the metabolic "
                 "trajectory forecasting study (Epic Cosmos): treatment, demographics, comorbidities, "
                 "model-input availability, and held-out outcomes",
        column_titles=dict(COLUMN_TITLES),
        column_ns={column: f"n = {n[column]:,}" for column in COLUMNS},
        body=body,
        footnotes=footnotes,
        footer="Aggregate, disclosure-controlled output | Counts below 11 shown as <11 | Noncausal descriptive table",
    )
    return model, warnings


# ======================================================================================
# 6. CSV round trip. The CSV is the table's source of truth; the PNG is rendered from it.
# ======================================================================================

CSV_FIELDS = ("order", "kind", "indent", "label", *COLUMNS, "source_values")


def table_records(model: TableModel) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {"kind": "title", "label": model.title},
        {"kind": "subtitle", "label": model.subtitle},
        {"kind": "column_header", "label": "Characteristic", **model.column_titles},
        {"kind": "column_n", "label": "", **model.column_ns},
    ]
    for row in model.body:
        records.append({"kind": row["kind"], "indent": row["indent"], "label": row["label"],
                        **row["cells"], "source_values": row.get("source_values", "")})
    records += [{"kind": "footnote", "label": text} for text in model.footnotes]
    records.append({"kind": "footer", "label": model.footer})
    for order, record in enumerate(records, start=1):
        record["order"] = order
        record.setdefault("indent", 0)
    return records


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    replace_with_retry(temporary, path)


def replace_with_retry(source: Path, destination: Path, attempts: int = 8) -> None:
    """os.replace, retried briefly: on Windows a scanner or viewer can hold the target for a moment."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def write_table_csv(model: TableModel, path: Path) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_FIELDS), lineterminator="\n")
    writer.writeheader()
    for record in table_records(model):
        writer.writerow({name: record.get(name, "") for name in CSV_FIELDS})
    # utf-8-sig so Excel shows the symbols (superscript 2, >=, the footnote marks) correctly.
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8-sig"))


def read_table_csv(path: Path) -> TableModel:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = sorted(csv.DictReader(stream), key=lambda record: int(record["order"]))
    by_kind = lambda kind: [record for record in records if record["kind"] == kind]
    single = lambda kind: (by_kind(kind) or [{}])[0]
    header, counts = single("column_header"), single("column_n")
    body = [{"kind": record["kind"], "label": record["label"], "indent": int(record["indent"] or 0),
             "cells": {column: record[column] for column in COLUMNS},
             "source_values": record.get("source_values", "")}
            for record in records if record["kind"] in {"section", "variable", "value"}]
    return TableModel(
        title=single("title").get("label", ""), subtitle=single("subtitle").get("label", ""),
        column_titles={column: header.get(column, "") for column in COLUMNS},
        column_ns={column: counts.get(column, "") for column in COLUMNS},
        body=body, footnotes=[record["label"] for record in by_kind("footnote")],
        footer=single("footer").get("label", ""),
    )


# ======================================================================================
# 7. PNG rendering: a journal-style table laid out in inches on one tall page.
# ======================================================================================

INK = "#1F2A37"
MUTED = "#5B6472"
RULE = "#1F2A37"
HAIRLINE = "#CBD3DE"
BAND = "#E7EDF5"
ZEBRA = "#F5F7FA"
PAPER = "#FFFFFF"
ACCENT = "#0072B2"

PAGE_WIDTH = 10.0
MARGIN_X = 0.55
MARGIN_TOP = 0.5
MARGIN_BOTTOM = 0.42
COLUMN_WIDTH = 1.3
INDENT = 0.2
BODY_SIZE = 8.0
LINE = 0.158
PAD = 0.034


def capacity(width_inches: float, size: float, factor: float = 0.55) -> int:
    """Characters of DejaVu Sans that fit a width; the factor is a slightly generous average glyph
    width in ems, so wrapped text never runs into the next column."""
    return max(8, int(width_inches * 72.0 / (size * factor)))


def balanced_wrap(text: str, width: int) -> list[str]:
    """Wrap into the fewest lines that fit, with the lines as even as possible, so a heading never
    ends on a one-word orphan."""
    lines = textwrap.wrap(text, width)
    if len(lines) <= 1:
        return lines
    for narrower in range(math.ceil(len(text) / len(lines)), width + 1):
        candidate = textwrap.wrap(text, narrower)
        if len(candidate) == len(lines):
            return candidate
    return lines


def render_table_png(model: TableModel, destination: Path) -> None:
    content = PAGE_WIDTH - 2 * MARGIN_X
    label_width = content - len(COLUMNS) * COLUMN_WIDTH
    column_centers = [MARGIN_X + label_width + COLUMN_WIDTH * (index + 0.5) for index in range(len(COLUMNS))]

    title_lines = balanced_wrap(model.title, capacity(content, 13.0, 0.6))
    subtitle_lines = balanced_wrap(model.subtitle, capacity(content, 8.6))
    header_lines = {column: textwrap.wrap(model.column_titles[column], 15) for column in COLUMNS}
    header_depth = max(len(lines) for lines in header_lines.values()) + 1

    layout = []  # (kind, top, height, row, label_lines)
    y = MARGIN_TOP + len(title_lines) * 0.25 + 0.04 + len(subtitle_lines) * 0.155 + 0.16
    header_top = y
    y += header_depth * 0.165 + 2 * PAD + 0.02
    body_top = y
    for row in model.body:
        if row["kind"] == "section":
            y += 0.05 if layout else 0.0
            height = LINE + 2 * PAD + 0.03
            layout.append(("section", y, height, row, [row["label"]]))
        else:
            indent = INDENT * int(row["indent"])
            lines = textwrap.wrap(row["label"], capacity(label_width - indent - 0.12, BODY_SIZE),
                                  subsequent_indent="  ") or [""]
            height = len(lines) * LINE + 2 * PAD
            layout.append((row["kind"], y, height, row, lines))
        y += height
    body_bottom = y
    footnote_capacity = capacity(content, 6.7, 0.53)
    footnote_lines = [textwrap.wrap(text, footnote_capacity) for text in model.footnotes]
    y += 0.12
    footnote_top = y
    y += sum(len(lines) * 0.122 + 0.035 for lines in footnote_lines)
    y += 0.16
    footer_y = y
    height_inches = y + MARGIN_BOTTOM

    figure = plt.figure(figsize=(PAGE_WIDTH, height_inches), dpi=DPI)
    figure.patch.set_facecolor(PAPER)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_xlim(0.0, PAGE_WIDTH)
    axis.set_ylim(height_inches, 0.0)
    axis.axis("off")
    def text(x: float, y: float, value: str, color: str = INK, **style: Any) -> None:
        axis.text(x, y, value, family="DejaVu Sans", color=color, **style)

    def rule(y: float, width: float, color: str = RULE) -> None:
        axis.plot([MARGIN_X, PAGE_WIDTH - MARGIN_X], [y, y], color=color, lw=width, solid_capstyle="butt")

    y = MARGIN_TOP
    axis.add_patch(Rectangle((MARGIN_X, y - 0.02), 0.06, len(title_lines) * 0.25, color=ACCENT, lw=0))
    for line in title_lines:
        text(MARGIN_X + 0.14, y, line, fontsize=13.0, fontweight="bold", va="top")
        y += 0.25
    y += 0.04
    for line in subtitle_lines:
        text(MARGIN_X + 0.14, y, line, fontsize=8.6, color=MUTED, va="top")
        y += 0.155

    rule(header_top, 1.3)
    text(MARGIN_X + 0.06, header_top + PAD + 0.01, "Characteristic", fontsize=8.6, fontweight="bold", va="top")
    for center, column in zip(column_centers, COLUMNS):
        # Bottom-aligned, so every column's title sits directly on its n line.
        row_y = header_top + PAD + 0.01 + (header_depth - 1 - len(header_lines[column])) * 0.165
        for line in header_lines[column]:
            text(center, row_y, line, fontsize=8.4, fontweight="bold", ha="center", va="top")
            row_y += 0.165
        text(center, header_top + PAD + 0.01 + (header_depth - 1) * 0.165, model.column_ns[column],
             fontsize=7.8, color=MUTED, ha="center", va="top")
    rule(body_top - 0.01, 0.8)

    group = -1
    for kind, top, height, row, lines in layout:
        if kind == "section":
            group = -1
            axis.add_patch(Rectangle((MARGIN_X, top), content, height, color=BAND, lw=0))
            text(MARGIN_X + 0.06, top + height / 2, row["label"], fontsize=8.4, fontweight="bold", va="center")
            continue
        if int(row["indent"]) == 0:
            group += 1
        if group % 2 == 1:
            axis.add_patch(Rectangle((MARGIN_X, top), content, height, color=ZEBRA, lw=0))
        x = MARGIN_X + 0.06 + INDENT * int(row["indent"])
        for index, line in enumerate(lines):
            text(x, top + PAD + LINE * (index + 0.5), line, fontsize=BODY_SIZE, va="center")
        if kind == "value":
            middle = top + PAD + LINE * 0.5
            for center, column in zip(column_centers, COLUMNS):
                value = row["cells"].get(column, "")
                muted = (value in {NOT_APPLICABLE, MASK_COMPLEMENT, NOT_IN_SOURCE}
                         or value.startswith(MASK_SECONDARY) or value.endswith("recorded"))
                text(center, middle, value, fontsize=BODY_SIZE, ha="center", va="center",
                     color=MUTED if muted else INK)
    rule(body_bottom, 1.3)

    y = footnote_top
    for lines in footnote_lines:
        for line in lines:
            text(MARGIN_X, y, line, fontsize=6.7, color=MUTED, va="top")
            y += 0.122
        y += 0.035
    rule(footer_y - 0.06, 0.5, HAIRLINE)
    text(MARGIN_X, footer_y, model.footer, fontsize=6.6, color=MUTED, va="top")

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=DPI, facecolor=PAPER, metadata={"Software": None})
    plt.close(figure)
    atomic_write_bytes(destination, buffer.getvalue())


# ======================================================================================
# 8. Bundle update and recursive file count.
# ======================================================================================


def update_bundle(bundle: Path, additions: Mapping[str, Path]) -> list[str]:
    """Rewrite the bundle with ``additions`` added (or replaced, on a rerun) and every other member
    copied through unchanged, then swap it in atomically. The original is untouched unless the new
    archive has been fully written and verified."""
    temporary = bundle.with_name(bundle.name + ".table1.tmp")
    with zipfile.ZipFile(bundle) as source:
        broken = source.testzip()
        if broken:
            raise RuntimeError(f"{bundle} is corrupt (first bad member: {broken}); it was not modified")
        kept = [info for info in source.infolist() if info.filename not in additions]
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as target:
            for info in kept:
                clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                clone.compress_type = info.compress_type
                clone.external_attr = info.external_attr
                clone.comment = info.comment
                clone.file_size = info.file_size
                if info.is_dir():
                    target.writestr(clone, b"")
                    continue
                with source.open(info) as reader, target.open(clone, "w", force_zip64=info.file_size >= zipfile.ZIP64_LIMIT) as writer:
                    shutil.copyfileobj(reader, writer, 1 << 20)
            for name, path in additions.items():
                entry = zipfile.ZipInfo(name, date_time=BUNDLE_TIMESTAMP)
                entry.compress_type = zipfile.ZIP_DEFLATED
                with path.open("rb") as reader, target.open(entry, "w") as writer:
                    shutil.copyfileobj(reader, writer, 1 << 20)
    expected = [info.filename for info in kept] + list(additions)
    with zipfile.ZipFile(temporary) as check:
        if check.testzip() is not None or check.namelist() != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("the rewritten bundle failed verification; the original was not modified")
    replace_with_retry(temporary, bundle)
    return expected


@dataclass
class FileCount:
    total: int = 0
    top_level: int = 0
    archives: int = 0
    nested: int = 0
    by_folder: Counter = field(default_factory=Counter)
    unreadable: list[str] = field(default_factory=list)


def count_bundle_files(bundle: Path) -> FileCount:
    """Every file in the bundle, counting files inside nested zip archives at any depth as well as
    each archive itself. Directory entries are not files."""
    result = FileCount()

    def walk(archive: zipfile.ZipFile, prefix: str, depth: int) -> None:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = prefix + info.filename
            result.total += 1
            if depth == 0:
                result.top_level += 1
            else:
                result.nested += 1
            result.by_folder[name.rsplit("/", 1)[0] if "/" in name else "(bundle root)"] += 1
            if info.filename.lower().endswith(".zip"):
                result.archives += 1
                with tempfile.SpooledTemporaryFile(max_size=64 << 20) as spool:
                    with archive.open(info) as reader:
                        shutil.copyfileobj(reader, spool, 1 << 20)
                    spool.seek(0)
                    try:
                        with zipfile.ZipFile(spool) as inner:
                            walk(inner, name + "/", depth + 1)
                    except zipfile.BadZipFile:
                        result.unreadable.append(name)

    with zipfile.ZipFile(bundle) as archive:
        walk(archive, "", 0)
    return result


# ======================================================================================
# 9. Self-test: deterministic checks on synthetic data. No VM, no Cosmos, no project imports.
# ======================================================================================


def synthetic_cohorts(seed: int = 20260721, sizes: Mapping[str, int] | None = None) -> tuple[Any, Any]:
    """A cohort frame shaped like the study's, with deliberately rare categories so every
    suppression path is exercised, plus matching held-out origin-0 prediction rows."""
    rng = np.random.default_rng(seed)
    sizes = dict(sizes or {"sleeve": 230, "rygb": 150, "incretin": 420})
    records = []
    serial = 0
    for arm, count in sizes.items():
        for _ in range(count):
            serial += 1
            surgery = arm != "incretin"
            ingredient = "not_applicable" if surgery else rng.choice(
                ["semaglutide", "tirzepatide", "liraglutide", "dulaglutide", "exenatide"], p=[.42, .3, .14, .12, .02])
            records.append({
                "patient_id": f"P{serial:05d}", "cohort": "surgery" if surgery else "incretin",
                "treatment": arm if surgery else ingredient, "procedure": arm if surgery else "not_applicable",
                "index_ingredient": ingredient,
                "index_route": "not_applicable" if surgery else ("oral" if rng.random() < .03 else "subcutaneous"),
                "split": rng.choice(SPLIT_ORDER[:4], p=[.55, .15, .12, .18]),
                "index_date": pd.Timestamp("2014-01-01") + pd.Timedelta(days=int(rng.integers(0, 3800))),
                "age_at_index": float(rng.integers(18, 80)) if rng.random() > .004 else np.nan,
                "sex": rng.choice(["Female", "Male", "Unknown", "X"], p=[.74, .245, .01, .005]),
                "race": rng.choice(["White", "Black or African American", "Asian", "Native Hawaiian",
                                    "Patient Refused", "Other Race"], p=[.6, .22, .06, .01, .03, .08]),
                "ethnicity": rng.choice(["Not Hispanic or Latino", "Hispanic or Latino", "Unknown"], p=[.8, .17, .03]),
                "coverage": rng.choice(["Commercial", "Medicare", "Medicaid", "Self-Pay", "Unknown"], p=[.5, .25, .2, .03, .02]),
                "smoking": rng.choice(["Never", "Former", "Current Every Day Smoker", "Unknown If Ever Smoked"], p=[.55, .25, .1, .1]),
                "svi": float(rng.random()) if rng.random() > .05 else np.nan,
                "ruca": str(int(rng.integers(1, 11))) if rng.random() > .03 else "Unknown",
                "baseline_bmi": float(rng.uniform(35, 70) if surgery else rng.uniform(30, 55)),
                "baseline_hba1c": float(rng.uniform(5.0, 10.0)) if rng.random() > .35 else np.nan,
                "diabetes_flag": float(rng.random() < .45), "hypertension": float(rng.random() < .999),
                "dyslipidemia": float(rng.random() < .4), "osa": float(rng.random() < .3),
                "insulin": float(rng.random() < .1), "biguanide": float(rng.random() < .3),
                "sglt2": float(rng.random() < .015),
                "creatinine_baseline": float(rng.uniform(.5, 1.5)) if rng.random() > .3 else np.nan,
                "egfr_baseline": float(rng.uniform(40, 120)) if rng.random() > .6 else np.nan,
            })
    cohorts = pd.DataFrame(records)
    cohorts["diabetes_eligible"] = cohorts["diabetes_flag"].eq(1.0) & cohorts["baseline_hba1c"].notna()
    cohorts.loc[cohorts["index_date"] >= pd.Timestamp("2022-06-01"), "split"] = "temporal_test"
    rows = []
    for record in cohorts.itertuples(index=False):
        for outcome, months in (("bmi", (3, 6, 12, 24, 36, 48, 60)), ("hba1c", (12, 24, 36, 48, 60))):
            if outcome == "hba1c" and not record.diabetes_eligible:
                continue
            for month in months:
                rows.append({"patient_id": record.patient_id, "cohort": record.cohort, "outcome": outcome,
                             "origin_month": 0, "target_month": month, "split": record.split,
                             "target_observed": bool(rng.random() < max(.03, .8 - month / 70))})
    return cohorts, pd.DataFrame(rows)


def synthetic_study_data(**kwargs: Any) -> StudyData:
    cohorts, rows = synthetic_cohorts(**kwargs)
    slim = slim_cohorts(cohorts)
    keep = rows["split"].isin(HELD_OUT_SPLITS) & rows["target_observed"]
    held = rows.loc[keep, ["patient_id", "cohort", "outcome", "split", "target_month"]].reset_index(drop=True)
    schema = tuple(ROW_COLUMNS) + ("creatinine_baseline", "smoking")
    return StudyData(slim, held, schema, {"temporal_cutoff": "2022-06-01"},
                     {"run_name": "synthetic_run", "study_version": "self-test", "source": "synthetic fixture",
                      "detail": "self-test", "incretin_qualifying_months": 12})


def run_self_tests() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    # 01 Category mapping, including the awkward real-world spellings.
    cases = [
        (sex_category, "F", "Female"), (sex_category, "Male", "Male"), (sex_category, "nan", "Unknown or not recorded"),
        (sex_category, "X", "Other"), (race_category, "Black or African American", "Black or African American"),
        (race_category, "Native Hawaiian or Other Pacific Islander", "Native Hawaiian or Other Pacific Islander"),
        (race_category, "American Indian or Alaska Native", "American Indian or Alaska Native"),
        (race_category, "Middle Eastern or North African", "Other"), (race_category, "Patient Refused", "Unknown or not reported"),
        (race_category, "Two or More Races", "Multiracial"), (race_category, "White or Caucasian", "White"),
        (ethnicity_category, "Not Hispanic or Latino", "Not Hispanic or Latino"),
        (ethnicity_category, "Hispanic or Latino", "Hispanic or Latino"), (ethnicity_category, "*Unspecified", "Unknown or not reported"),
        (coverage_category, "Medicare Advantage", "Medicare"), (coverage_category, "Medicaid", "Medicaid"),
        (coverage_category, "Self-Pay", "Self-pay or uninsured"), (coverage_category, "Commercial", "Commercial"),
        (coverage_category, "Worker's Comp", "Other"), (coverage_category, "None", "Unknown or not reported"),
        (smoking_category, "Never Smoker", "Never"), (smoking_category, "Former Smoker", "Former"),
        (smoking_category, "Current Some Day Smoker", "Current"), (smoking_category, "Unknown If Ever Smoked", "Unknown or not recorded"),
        (smoking_category, "Never Assessed", "Unknown or not recorded"), (smoking_category, "Never/former", "Never or former"),
        (ruca_category, "1.0", "Metropolitan (RUCA 1-3)"), (ruca_category, "10", "Nonmetropolitan (RUCA 4-10)"),
        (ruca_category, "99", "Not recorded"), (ruca_category, "Unknown", "Not recorded"),
        (ruca_category, "1 metropolitan", "Metropolitan (RUCA 1-3)"), (ruca_category, "4 micropolitan", "Nonmetropolitan (RUCA 4-10)"),
        (ruca_category, "10 rural", "Nonmetropolitan (RUCA 4-10)"), (ruca_category, "Nonmetropolitan", "Nonmetropolitan (RUCA 4-10)"),
        (ruca_category, "nan", "Not recorded"),
    ]
    wrong = [(function.__name__, raw, function(raw), want) for function, raw, want in cases if function(raw) != want]
    wrong += [("agent", pair, agent_category(*pair), want) for pair, want in (
        (("semaglutide", "oral"), "Semaglutide, oral"), (("semaglutide", "unknown"), "Semaglutide, injectable"),
        (("lixisenatide", "subcutaneous"), "Exenatide, lixisenatide, or albiglutide"),
        (("tirzepatide", "subcutaneous"), "Tirzepatide")) if agent_category(*pair) != want]
    check("01_category_mapping", not wrong, f"{len(cases) + 4} cases; wrong={wrong}")

    # 02 Primary suppression: counts 0-10 masked, 11 shown; zeros masked but never protected.
    builder = TableBuilder({"sg": 100, "rygb": 50, "all_mbs": 150, "ibt": 200})
    rows = [builder.counts("a", {"sg": 0, "rygb": 0, "all_mbs": 0, "ibt": 11}),
            builder.counts("b", {"sg": 100, "rygb": 50, "all_mbs": 150, "ibt": 189})]
    builder.partition(rows, "two-level")
    body = builder.body()
    check("02_primary_suppression", [row["cells"] for row in body] == [
        {"sg": "<11", "rygb": "<11", "all_mbs": "<11", "ibt": "11 (5.5)"},
        {"sg": "100 (100.0)", "rygb": "50 (100.0)", "all_mbs": "150 (100.0)", "ibt": "189 (94.5)"}],
        str([row["cells"] for row in body]))

    # 03 A single small level in a partition forces a second, complementary suppression.
    builder = TableBuilder({"sg": 100, "rygb": 100, "all_mbs": 200, "ibt": 100})
    rows = [builder.counts("x", {"sg": 4, "rygb": 30, "all_mbs": 34, "ibt": 40}),
            builder.counts("y", {"sg": 20, "rygb": 30, "all_mbs": 50, "ibt": 30}),
            builder.counts("z", {"sg": 76, "rygb": 40, "all_mbs": 116, "ibt": 30})]
    builder.partition(rows, "three-level")
    body = builder.body()
    sg = [row["cells"]["sg"] for row in body]
    secondary = lambda cells: sum(cell.startswith(MASK_SECONDARY) for cell in cells)
    check("03_complementary_partition", sg.count("<11") == 1 and secondary(sg) >= 1
          and not builder.dc.audit(), f"sg column={sg}")

    # 04 SG + RYGB = All MBS: a masked arm cell cannot be recovered from the other two.
    rygb = [row["cells"]["rygb"] for row in body]
    all_mbs = [row["cells"]["all_mbs"] for row in body]
    check("04_row_identity_protected", secondary(rygb) + secondary(all_mbs) >= 1,
          f"rygb={rygb}; all_mbs={all_mbs}")

    # 05 A near-complete availability row shows "All but <11" instead of an exact count.
    builder = TableBuilder({"sg": 500, "rygb": 300, "all_mbs": 800, "ibt": 900})
    row = builder.counts("recorded", {"sg": 497, "rygb": 300, "all_mbs": 797, "ibt": 900})
    builder.binary(row)
    cells = builder.body()[0]["cells"]
    check("05_all_but_complement", cells["sg"] == MASK_COMPLEMENT and cells["all_mbs"] == MASK_COMPLEMENT
          and cells["ibt"] == "900 (100.0)" and not builder.dc.audit(), str(cells))

    # 06 Randomized stress: after suppression nothing below 11 is displayed or recoverable.
    rng = np.random.default_rng(11)
    failures = []
    for trial in range(120):
        sizes = {"sg": int(rng.integers(40, 400)), "rygb": int(rng.integers(40, 400))}
        sizes["all_mbs"], sizes["ibt"] = sizes["sg"] + sizes["rygb"], int(rng.integers(40, 900))
        builder = TableBuilder(sizes)
        for variable in range(3):
            levels = int(rng.integers(2, 6))
            shares = {column: rng.multinomial(sizes[column], rng.dirichlet(np.full(levels, .5))) for column in ("sg", "rygb", "ibt")}
            rows = [builder.counts(f"v{variable}l{level}", {
                "sg": int(shares["sg"][level]), "rygb": int(shares["rygb"][level]),
                "all_mbs": int(shares["sg"][level] + shares["rygb"][level]), "ibt": int(shares["ibt"][level])})
                for level in range(levels)]
            builder.partition(rows, f"v{variable}")
            # The last level plays "not recorded"; its availability row is the column total minus it.
            last = rows[-1]
            recorded = builder.counts(f"v{variable} recorded", {
                column: sizes[column] - builder.dc.values[builder.ref(last, column)] for column in COLUMNS})
            builder.link(recorded, last)
        for flag in range(2):
            values = {column: int(sizes[column] - rng.integers(0, 14)) for column in ("sg", "rygb", "ibt")}
            values["all_mbs"] = values["sg"] + values["rygb"]
            builder.binary(builder.counts(f"f{flag}", values))
        try:
            builder.body()
        except DisclosureError as error:
            failures.append(f"trial {trial}: {error}")
            continue
        if builder.dc.audit():
            failures.append(f"trial {trial}: {builder.dc.audit()[:2]}")
    check("06_randomized_disclosure_audit", not failures, f"120 random tables; failures={failures[:3]}")

    # 12 A small group recoverable only by combining two identities: the HbA1c levels <5.7 (0)
    #    and 5.7-6.4 (8) of a column whose "not measured" level is pinned by the availability row.
    builder = TableBuilder({"sg": 101, "rygb": 300, "all_mbs": 401, "ibt": 900})
    levels = [builder.counts("<5.7", {"sg": 0, "rygb": 40, "all_mbs": 40, "ibt": 100}),
              builder.counts("5.7-6.4", {"sg": 8, "rygb": 60, "all_mbs": 68, "ibt": 200}),
              builder.counts(">=6.5", {"sg": 65, "rygb": 100, "all_mbs": 165, "ibt": 400}),
              builder.counts("not measured", {"sg": 28, "rygb": 100, "all_mbs": 128, "ibt": 200})]
    builder.partition(levels, "hba1c")
    recorded = builder.counts("recorded", {"sg": 73, "rygb": 200, "all_mbs": 273, "ibt": 700})
    builder.link(recorded, levels[-1])
    body = builder.body()
    published = {row["label"]: row["cells"]["sg"] for row in body}
    known = [published[label] for label in (">=6.5", "not measured", "recorded")]
    check("12_cross_identity_group_protected", not builder.dc.audit() and any(not value[0].isdigit() for value in known),
          f"sg={published}")

    # 13 Bounds the masks imply are used too: "<11" + "Masked‡" = 12 would pin the "<11" cell to
    #    0 or 1 (the masked cell is at least 11), so a third cell must be masked.
    builder = TableBuilder({"sg": 100, "rygb": 100, "all_mbs": 200, "ibt": 100})
    levels = [builder.counts("a", {"sg": 1, "rygb": 30, "all_mbs": 31, "ibt": 30}),
              builder.counts("b", {"sg": 11, "rygb": 30, "all_mbs": 41, "ibt": 30}),
              builder.counts("c", {"sg": 88, "rygb": 40, "all_mbs": 128, "ibt": 40})]
    builder.partition(levels, "abc")
    sg = [row["cells"]["sg"] for row in builder.body()]
    low, high = builder.dc._range(builder.dc._system(), [builder.ref(levels[0], "sg")])
    check("13_bound_implied_pinning_protected",
          sg[0] == MASK_PRIMARY and high - low >= DisclosureControl.PROTECTION_WIDTH and not builder.dc.audit(),
          f"sg={sg}; recoverable range for the 1-patient cell=[{low:g}, {high:g}]")

    # 14 A complementary mask in a large column keeps its rounded percentage (the count it implies
    #    spans dozens of patients); in a small column the same percentage would pin the small cell.
    builder = TableBuilder({"sg": 30000, "rygb": 20000, "all_mbs": 50000, "ibt": 120})
    levels = [builder.counts("Female", {"sg": 23000, "rygb": 15002, "all_mbs": 38002, "ibt": 90}),
              builder.counts("Male", {"sg": 6995, "rygb": 4998, "all_mbs": 11993, "ibt": 27}),
              builder.counts("Unknown", {"sg": 5, "rygb": 0, "all_mbs": 5, "ibt": 3})]
    builder.partition(levels, "sex")
    male = builder.body()[1]["cells"]
    check("14_percentage_kept_only_where_safe",
          male["sg"] == f"{MASK_SECONDARY} (23.3)" and male["ibt"] == MASK_SECONDARY and not builder.dc.audit(),
          f"male row={male}")

    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)

        # 07 End to end on a synthetic cohort, and the CSV alone regenerates the identical PNG.
        model, warnings = build_table(synthetic_study_data(), "2001-01-01T00:00:00Z")
        csv_path, png_direct, png_csv = workspace / "t.csv", workspace / "direct.png", workspace / "from_csv.png"
        write_table_csv(model, csv_path)
        reread = read_table_csv(csv_path)
        render_table_png(model, png_direct)
        render_table_png(reread, png_csv)
        cells = [cell for row in model.body for cell in row["cells"].values()]
        displayed = [int(match.replace(",", "")) for cell in cells
                     for match in re.findall(r"^([\d,]+)(?: \(|$)", cell)]
        check("07_end_to_end_csv_regenerates_png",
              reread == model and png_direct.read_bytes() == png_csv.read_bytes()
              and min(displayed) >= MIN_CELL and MASK_PRIMARY in cells,
              f"{len(model.body)} rows; smallest displayed count={min(displayed)}; warnings={warnings}")

        # 08 Bundle rewrite: adds the table, keeps every other member byte-identical, is
        #    idempotent on rerun, and the recursive count sees inside nested zips.
        inner_buffer = io.BytesIO()
        with zipfile.ZipFile(inner_buffer, "w") as inner:
            inner.writestr("FIGURES_TO_EXPORT/00.png", b"png")
            inner.writestr("analysis/a.csv", b"a")
            inner.writestr("analysis/", b"")
        bundle = workspace / "final_study_bundle.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(zipfile.ZipInfo("final_study_figure_book.pdf", date_time=BUNDLE_TIMESTAMP), b"pdf")
            archive.writestr(zipfile.ZipInfo("secondary/secondary_analyses_results_bundle.zip",
                                             date_time=BUNDLE_TIMESTAMP), inner_buffer.getvalue())
            archive.writestr(zipfile.ZipInfo("modeling/run_manifest.json", date_time=BUNDLE_TIMESTAMP), b"{}")
        with zipfile.ZipFile(bundle) as archive:
            before = {info.filename: archive.read(info) for info in archive.infolist()}
        additions = {f"{BUNDLE_FOLDER}/{TABLE_STEM}.png": png_csv, f"{BUNDLE_FOLDER}/{TABLE_STEM}.csv": csv_path}
        first = update_bundle(bundle, additions)
        second = update_bundle(bundle, additions)
        with zipfile.ZipFile(bundle) as archive:
            preserved = all(archive.read(name) == payload for name, payload in before.items())
            added = archive.read(f"{BUNDLE_FOLDER}/{TABLE_STEM}.csv") == csv_path.read_bytes()
        counted = count_bundle_files(bundle)
        check("08_bundle_update_and_recursive_count",
              first == second and len(second) == 5 and preserved and added
              and (counted.total, counted.top_level, counted.archives, counted.nested) == (7, 5, 1, 2),
              f"members={second}; total={counted.total}")

        # 11 The frozen loader verifies hashes and row counts instead of trusting the files.
        run = workspace / "run"
        checkpoints, rows_dir = run / "INTERNAL" / "checkpoints", run / "INTERNAL" / "rows" / "weighted"
        checkpoints.mkdir(parents=True)
        rows_dir.mkdir(parents=True)
        cohorts, outcome_rows = synthetic_cohorts(sizes={"sleeve": 60, "rygb": 40, "incretin": 80})
        (checkpoints / "global_splits.pkl").write_bytes(pickle.dumps({"cohorts": cohorts, "metadata": {"temporal_cutoff": "2022-06-01"}}))
        (checkpoints / "global_splits.json").write_text(json.dumps({
            "completion_marker": "COMPLETE", "artifact_sha256": sha256_file(checkpoints / "global_splits.pkl")}))
        partitions = {}
        for (cohort, outcome), part in outcome_rows.groupby(["cohort", "outcome"]):
            name = f"{cohort}__{outcome}__origin000"
            (rows_dir / f"{name}__0000.pkl").write_bytes(pickle.dumps(part.reset_index(drop=True)))
            partitions[name] = len(part)
        (checkpoints / "weights.pkl").write_bytes(b"store index")
        (checkpoints / "weights.json").write_text(json.dumps({
            "completion_marker": "COMPLETE", "artifact_sha256": sha256_file(checkpoints / "weights.pkl"),
            "payload_manifest": {"children": {"rows": {"partitions": partitions}}}}))
        loaded = load_frozen(run)
        expected_held = int((outcome_rows["split"].isin(HELD_OUT_SPLITS) & outcome_rows["target_observed"]).sum())
        loaded_ok = len(loaded.cohorts) == len(cohorts) and len(loaded.outcome_rows) == expected_held
        record = json.loads((checkpoints / "weights.json").read_text())
        record["payload_manifest"]["children"]["rows"]["partitions"][next(iter(partitions))] += 1
        (checkpoints / "weights.json").write_text(json.dumps(record))
        try:
            load_frozen(run)
            short_rejected = False
        except SourceUnavailable:
            short_rejected = True
        (checkpoints / "global_splits.pkl").write_bytes(b"tampered")
        try:
            load_frozen(run)
            tamper_rejected = False
        except SourceUnavailable:
            tamper_rejected = True
        check("11_frozen_loader_verifies_inputs", loaded_ok and short_rejected and tamper_rejected,
              f"loaded={loaded_ok}; row-count mismatch rejected={short_rejected}; hash mismatch rejected={tamper_rejected}")

    # 09 The constants mirrored from the study still match it (skipped where it is not present).
    study_path = THIS_FILE.parent / STUDY_SCRIPT
    if study_path.is_file():
        constants: dict[str, Any] = {}
        for node in ast.parse(study_path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        windows = {month: tuple(bounds[:2]) for month, bounds in constants.get("WINDOW_MONTHS", {}).items()}
        mirrored = (constants.get("MIN_CELL_SIZE") == MIN_CELL
                    and constants.get("PROCEDURE_CODES") == PROCEDURE_CODES
                    and constants.get("OPTIONAL_WIDE_NUMERIC_COVARIATES") == OPTIONAL_WIDE_NUMERIC_COVARIATES
                    and constants.get("OPTIONAL_COVARIATE_THRESHOLD") == OPTIONAL_COVARIATE_THRESHOLD
                    and windows == TARGET_WINDOWS)
        check("09_mirrored_study_constants", mirrored, str(study_path.name))
    else:
        check("09_mirrored_study_constants", True, "skipped: study script not beside this file")

    # 10 Single-file discipline: no project-local module imports anywhere in this file.
    imported: list[str] = []
    for node in ast.walk(ast.parse(THIS_FILE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    allowed = {"__future__", "argparse", "ast", "csv", "gc", "hashlib", "importlib", "io", "itertools", "json",
               "math", "os", "pickle", "re", "shutil", "sys", "tempfile", "textwrap", "time", "zipfile",
               "collections", "dataclasses", "datetime", "pathlib", "typing", "numpy", "pandas", "matplotlib",
               "pyarrow", "scipy"}
    outside = sorted({name for name in imported if name.split(".")[0] not in allowed})
    check("10_no_project_local_imports", not outside, f"outside allowlist={outside}")

    for name, ok, detail in sorted(results):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"SELF-TEST FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"SELF-TEST PASSED: {len(results)}/{len(results)} deterministic tests")
    return 0


# ======================================================================================
# 10. CLI
# ======================================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE,
                        help="final_study_bundle.zip to add the table to (default: the shared three-arm run on Z:)")
    parser.add_argument("--run-dir", default=None,
                        help="The completed run directory (default: the folder holding the bundle)")
    parser.add_argument("--output-dir", default=None, help="Where the PNG and CSV are written (default: <run>/table1)")
    parser.add_argument("--source", choices=("auto", "checkpoint", "cosmos"), default="auto",
                        help="auto: frozen run checkpoints, else Cosmos re-acquire; checkpoint: frozen only; "
                             "cosmos: always re-acquire")
    parser.add_argument("--scripts-dir", default=str(THIS_FILE.parent),
                        help=f"Folder holding {STUDY_SCRIPT}, used only by the Cosmos fallback")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true", help="Run the deterministic self-tests and exit")
    modes.add_argument("--render-from-csv", metavar="CSV", default=None,
                       help="Re-render the PNG from a table CSV (written beside it) and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # A redirected Windows console may use a legacy code page; never fail on a symbol in a message.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_tests()
    if args.render_from_csv:
        source = Path(args.render_from_csv).expanduser().resolve()
        destination = source.with_suffix(".png")
        render_table_png(read_table_csv(source), destination)
        log(f"rendered {destination}")
        return 0

    bundle = Path(args.bundle).expanduser().resolve()
    if not bundle.is_file():
        log(f"ERROR: bundle not found: {bundle}")
        return 2
    try:
        with zipfile.ZipFile(bundle) as archive:
            archive.infolist()
    except zipfile.BadZipFile as error:
        log(f"ERROR: {bundle} is not a readable zip archive ({error})")
        return 2
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else bundle.parent
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / BUNDLE_FOLDER
    work_dir = output_dir / "_cosmos_work"
    log(f"run directory: {run_dir}")
    try:
        data = load_study_data(run_dir, args.source, Path(args.scripts_dir).expanduser().resolve(), work_dir)
        generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        model, warnings = build_table(data, generated)
    except (SourceUnavailable, DisclosureError, RuntimeError, ValueError) as error:
        log(f"ERROR: {error}")
        log("Nothing was written and the bundle was not modified.")
        return 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    for warning in warnings:
        log(f"note: {warning}")
    del data
    gc.collect()

    csv_path = output_dir / f"{TABLE_STEM}.csv"
    png_path = output_dir / f"{TABLE_STEM}.png"
    write_table_csv(model, csv_path)
    # Rendered from the CSV just written, so the CSV is proven sufficient to regenerate the PNG.
    render_table_png(read_table_csv(csv_path), png_path)
    log(f"wrote {png_path}")
    log(f"wrote {csv_path}")

    members = update_bundle(bundle, {f"{BUNDLE_FOLDER}/{png_path.name}": png_path,
                                     f"{BUNDLE_FOLDER}/{csv_path.name}": csv_path})
    log(f"added {BUNDLE_FOLDER}/{png_path.name} and {BUNDLE_FOLDER}/{csv_path.name} to {bundle} "
        f"({len(members)} top-level members)")

    counted = count_bundle_files(bundle)
    for folder, count in sorted(counted.by_folder.items()):
        log(f"  {count:5d}  {folder}")
    for name in counted.unreadable:
        log(f"note: nested archive {name} could not be opened; counted as one file")
    log(f"{counted.top_level} top-level files, plus {counted.nested} files inside "
        f"{counted.archives} nested zip archive(s)")
    print(f"TOTAL FILES IN {bundle.name} (all folders, subfolders, and nested zip archives, "
          f"each archive counted too): {counted.total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
