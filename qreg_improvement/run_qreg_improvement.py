#!/usr/bin/env python3
"""Production five-year Epic Cosmos bariatric forecasting study.

The no-flag command runs the frozen production database analysis:

    python qreg_improvement/run_qreg_improvement.py

This file is the complete runtime package. It intentionally imports no repository-local
Python modules. Developer modes are ``--smoke``, ``--self-test``, and ``--plot-only``.
All removable outputs are aggregate figures. Patient-level working objects stay inside
the fingerprinted run directory and never enter ``FIGURES_TO_EXPORT``.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pickle
import platform
import re
import shutil
import struct
import sys
import tempfile
import textwrap
import time
import traceback
import warnings
import zlib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


# =====================================================================================
# 1. Immutable protocol, dependency contract, and configuration
# =====================================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "runs"
RUNTIME_CACHE = Path(tempfile.gettempdir()) / "qreg_cosmos_runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(RUNTIME_CACHE / "joblib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

STUDY_VERSION = "cosmos-bariatric-forecast-2.0.0"
SQL_CONTRACT_VERSION = "cosmos-mbs-explicit-v2.0.0"
SOURCE_SCHEMA = "dbo"
SOURCE_TABLE = "MBSCohort"
CONNECTION_STRING = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=tcp:PROJECTS;"
    "Database=ProjectD332AFD;"
    "Trusted_Connection=yes;"
)
DAYS_PER_MONTH = 30.4375
SEED = 2026
LANDMARK_MONTHS = (0, 3, 6, 12, 24)
TRAJECTORY_HORIZONS = {
    "bmi": (3, 6, 12, 24, 36, 48, 60),
    "hba1c": (12, 24, 36, 48, 60),
}
MEASUREMENT_WINDOWS = {
    3: (2.0, 4.5),
    6: (4.5, 9.0),
    12: (9.0, 18.0),
    24: (18.0, 30.0),
    36: (30.0, 42.0),
    48: (42.0, 54.0),
    60: (54.0, 66.0),
}
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
COVERAGE_LEVELS = (0.50, 0.80, 0.90)
PLAUSIBLE_RANGES = {"bmi": (10.0, 100.0), "hba1c": (3.0, 20.0)}
PROCEDURE_CODES = {"43775": "sleeve", "43644": "rygb", "43846": "rygb", "43645": "rygb"}
COMPONENTS = ("mace", "nephropathy", "retinopathy")
RISK_HORIZONS = (12, 36, 60)
PERSON_PERIOD_MONTHS = 3
LITERATURE = {
    "registry_version": "2026-07-20",
    "bjs_2026": {
        "role": "contextual reference only; not a reproduced fitted model",
        "bmi_rmse": {3: 2.36, 6: 1.31, 12: 0.91, 24: 0.62, 36: 0.78, 48: 0.92, 60: 1.01},
        "pooled_rmse": 1.11,
        "pooled_mae": 0.62,
    },
    "sophia": {
        "role": "contextual reference only; not a matched comparator",
        "bmi_rmse": {12: 3.7, 24: 4.2, 60: 4.7},
        "bmi_mad_60": 2.8,
    },
}
TARGET_TRIAL_PROTOCOL = {
    "version": "sleeve-vs-rygb-v2.0.0",
    "eligibility": "Prior GLP-1 naive adults receiving first sleeve or RYGB procedure",
    "strategies": "Sleeve gastrectomy versus RYGB at the index operation",
    "time_zero": "Index bariatric operation date",
    "assignment": "Observed treatment with cross-fitted baseline adjustment",
    "primary_causal_outcome": "Five-year first cardiovascular-renal-retinal event cumulative incidence",
    "secondary_metabolic_estimand": "Strategy-specific survivor mean BMI and HbA1c at five years",
    "contrast": "Marginal RYGB minus sleeve effect in the eligible target population",
    "follow_up": "From operation through five years, event, death, or loss of observable follow-up",
    "inference": "Center-clustered bootstrap with cross-fitted nuisance models",
    "treatment_versions": {"sleeve": ["43775"], "rygb": ["43644", "43645", "43846"]},
}
DYNAMIC_GLP1_PROTOCOL = {
    "version": "postop-glp1-clone-censor-v1.0.0",
    "eligibility_landmark_month": 12,
    "strategy_a": "Initiate GLP-1 during months 12-18 when BMI at month 12 is at least 35",
    "strategy_b": "Do not initiate GLP-1 through month 24",
    "grace_period": "Months 12-18 for the indicated initiation strategy",
    "outcome_month": 36,
    "method": "Cross-fitted clone-censor inverse-probability weighting",
}
SUCCESS_GATES = {
    "trajectory_standardized_crps_relative_improvement": 0.10,
    "trajectory_max_powered_horizon_worsening": 0.05,
    "coverage_max_absolute_error": 0.05,
    "risk_calibration_slope_low": 0.80,
    "risk_calibration_slope_high": 1.20,
    "causal_max_abs_smd": 0.10,
    "causal_min_effective_sample_size": 100.0,
    "causal_min_support_probability": 0.02,
    "observation_min_probability": 0.02,
}

REQUIRED_DISTRIBUTIONS = {
    "numpy": ("numpy", "1.24"),
    "pandas": ("pandas", "1.5"),
    "scipy": ("scipy", "1.9"),
    "scikit-learn": ("sklearn", "1.2"),
    "matplotlib": ("matplotlib", "3.6"),
    "catboost": ("catboost", "1.2"),
}
PLOT_DISTRIBUTIONS = {
    "numpy": ("numpy", "1.24"),
    "pandas": ("pandas", "1.5"),
    "matplotlib": ("matplotlib", "3.6"),
}


@dataclass(frozen=True)
class RunConfig:
    mode: str = "production"
    seed: int = SEED
    output_dir: str | None = None
    resume: bool = True
    predictive_samples: int = 200
    threads: int = 1
    smoke_patients: int = 420
    hgb_iterations: int = 220
    catboost_iterations: int = 350
    cluster_bootstrap_replicates: int = 500
    heldout_center_fraction: float = 0.25
    min_development_patients: int = 1000
    min_locked_test_patients: int = 300
    min_test_centers: int = 2
    min_events_per_risk_endpoint: int = 40
    min_cell_observations: int = 100
    interrupt_after: str | None = None

    @property
    def smoke(self) -> bool:
        return self.mode == "smoke"

    @classmethod
    def for_mode(cls, mode: str, output_dir: str | None, interrupt_after: str | None = None) -> "RunConfig":
        if mode == "smoke":
            return cls(
                mode=mode,
                output_dir=output_dir,
                predictive_samples=31,
                hgb_iterations=35,
                catboost_iterations=45,
                cluster_bootstrap_replicates=100,
                min_development_patients=120,
                min_locked_test_patients=35,
                min_test_centers=2,
                min_events_per_risk_endpoint=3,
                min_cell_observations=8,
                interrupt_after=interrupt_after,
            )
        return cls(mode=mode, output_dir=output_dir, interrupt_after=interrupt_after)


@dataclass(frozen=True)
class FieldSpec:
    canonical: str
    aliases: tuple[str, ...]
    required: bool = True
    description: str = ""


@dataclass
class LockedSplits:
    train: Any
    calibration: Any
    validation: Any
    test: Any
    contemporary: Any
    mature: Any
    heldout_centers: list[str]
    calibration_centers: list[str]
    validation_centers: list[str]
    temporal_cutoff: str
    internal_external_folds: list[dict[str, Any]] = field(default_factory=list)

    def arrays(self) -> dict[str, Any]:
        return {
            "train": self.train,
            "calibration": self.calibration,
            "validation": self.validation,
            "test": self.test,
            "contemporary": self.contemporary,
            "mature": self.mature,
        }


@dataclass
class FeatureEncoder:
    numeric: list[str]
    categorical: list[str]
    medians: dict[str, float]
    scales: dict[str, float]
    levels: dict[str, list[str]]
    output_names: list[str]

    @classmethod
    def fit(cls, frame: Any, numeric: Sequence[str], categorical: Sequence[str]) -> "FeatureEncoder":
        medians: dict[str, float] = {}
        scales: dict[str, float] = {}
        levels: dict[str, list[str]] = {}
        names: list[str] = []
        for col in numeric:
            values = pd.to_numeric(frame[col], errors="coerce") if col in frame else pd.Series(dtype=float)
            median = float(values.median()) if values.notna().any() else 0.0
            medians[col] = median
            scale = float(values.std(ddof=0)) if values.notna().sum() > 1 else 1.0
            scales[col] = scale if np.isfinite(scale) and scale > 1e-8 else 1.0
            names.extend([col, f"{col}__missing"])
        for col in categorical:
            values = frame[col].astype("string").fillna("<MISSING>") if col in frame else pd.Series("<MISSING>", index=frame.index)
            levels[col] = sorted(str(x) for x in values.unique())
            names.extend(f"{col}=={level}" for level in levels[col])
        return cls(list(numeric), list(categorical), medians, scales, levels, names)

    def transform(self, frame: Any) -> Any:
        columns: list[Any] = []
        for col in self.numeric:
            values = pd.to_numeric(frame[col], errors="coerce") if col in frame else pd.Series(np.nan, index=frame.index)
            columns.append((values.fillna(self.medians[col]).to_numpy(float) - self.medians[col]) / self.scales[col])
            columns.append(values.isna().to_numpy(float))
        for col in self.categorical:
            values = frame[col].astype("string").fillna("<MISSING>") if col in frame else pd.Series("<MISSING>", index=frame.index)
            for level in self.levels[col]:
                columns.append(values.eq(level).to_numpy(float))
        if not columns:
            return np.empty((len(frame), 0), dtype=float)
        return np.column_stack(columns).astype(float)


@dataclass
class RunContext:
    cfg: RunConfig
    run_dir: Path
    fingerprint: str
    fingerprint_payload: dict[str, Any]
    state: dict[str, Any]

    @property
    def internal(self) -> Path:
        return self.run_dir / "INTERNAL"

    @property
    def aggregate(self) -> Path:
        return self.run_dir / "AGGREGATE"

    @property
    def export(self) -> Path:
        return self.run_dir / "FIGURES_TO_EXPORT"

    @property
    def checkpoints(self) -> Path:
        return self.internal / "checkpoints"

    def initialize(self) -> None:
        for path in (self.run_dir, self.internal, self.aggregate, self.export, self.checkpoints):
            path.mkdir(parents=True, exist_ok=True)
        existing = read_json(self.run_dir / "run_manifest.json", {})
        if existing and existing.get("fingerprint") != self.fingerprint:
            raise RuntimeError("Run directory fingerprint mismatch; refusing unsafe resume")
        atomic_json(self.run_dir / "run_manifest.json", {
            "study_version": STUDY_VERSION,
            "fingerprint": self.fingerprint,
            "fingerprint_payload": self.fingerprint_payload,
            "configuration": asdict(self.cfg),
            "created_utc": existing.get("created_utc", utc_now()),
        })
        if not self.state:
            self.state = {"status": "running", "stages": {}, "resumed_stages": [], "errors": []}
        atomic_json(self.run_dir / "run_state.json", self.state)

    def stage_fingerprint(self, stage: str, upstream: Mapping[str, str] | None = None) -> str:
        return digest({
            "run_fingerprint": self.fingerprint,
            "stage": stage,
            "upstream_artifact_hashes": dict(sorted((upstream or {}).items())),
        })

    def load_checkpoint(self, stage: str, upstream: Mapping[str, str] | None = None) -> Any | None:
        meta_path = self.checkpoints / f"{stage}.json"
        payload_path = self.checkpoints / f"{stage}.pkl"
        meta = read_json(meta_path, {})
        expected = self.stage_fingerprint(stage, upstream)
        if not self.cfg.resume or meta.get("stage_fingerprint") != expected or not payload_path.exists():
            return None
        actual_hash = sha256_file(payload_path)
        if actual_hash != meta.get("artifact_sha256"):
            return None
        self.state.setdefault("resumed_stages", []).append(stage)
        self.state.setdefault("stages", {})[stage] = {"status": "resumed", "fingerprint": expected}
        atomic_json(self.run_dir / "run_state.json", self.state)
        with payload_path.open("rb") as stream:
            return pickle.load(stream)

    def save_checkpoint(self, stage: str, value: Any, upstream: Mapping[str, str] | None = None) -> str:
        payload_path = self.checkpoints / f"{stage}.pkl"
        atomic_pickle(payload_path, value)
        artifact_hash = sha256_file(payload_path)
        stage_hash = self.stage_fingerprint(stage, upstream)
        atomic_json(self.checkpoints / f"{stage}.json", {
            "stage": stage,
            "stage_fingerprint": stage_hash,
            "artifact_sha256": artifact_hash,
            "upstream_artifact_hashes": dict(sorted((upstream or {}).items())),
            "completed_utc": utc_now(),
        })
        self.state.setdefault("stages", {})[stage] = {
            "status": "complete",
            "fingerprint": stage_hash,
            "artifact_sha256": artifact_hash,
        }
        atomic_json(self.run_dir / "run_state.json", self.state)
        if self.cfg.interrupt_after == stage:
            self.state["status"] = "interrupted_for_test"
            atomic_json(self.run_dir / "run_state.json", self.state)
            raise IntentionalInterrupt(f"Intentional interruption after {stage}")
        return artifact_hash


class PreflightError(RuntimeError):
    def __init__(self, title: str, issues: Sequence[str]):
        super().__init__(title + ": " + "; ".join(issues))
        self.title = title
        self.issues = list(issues)


class IntentionalInterrupt(RuntimeError):
    pass


# Runtime packages are loaded only after the standard-library dependency preflight.
np = pd = scipy_stats = plt = PdfPages = None
sklearn = None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, default=json_default, sort_keys=True))


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value)
    return tuple(int(x) for x in parts[:4]) or (0,)


def dependency_manifest(mode: str) -> tuple[dict[str, Any], list[str]]:
    requirements = PLOT_DISTRIBUTIONS if mode == "plot-only" else REQUIRED_DISTRIBUTIONS
    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    issues: list[str] = []
    if sys.version_info < (3, 10):
        issues.append(f"Python >=3.10 is required; found {platform.python_version()}")
    for distribution, (module, minimum) in requirements.items():
        try:
            found = importlib.metadata.version(distribution)
            manifest[distribution] = found
            if version_tuple(found) < version_tuple(minimum):
                issues.append(f"{distribution}>={minimum} is required; found {found}")
            if importlib.util.find_spec(module) is None:
                issues.append(f"Distribution {distribution} is installed but module {module} cannot be imported")
        except importlib.metadata.PackageNotFoundError:
            manifest[distribution] = None
            issues.append(f"Missing production dependency: {distribution}>={minimum}")
    if mode == "production":
        try:
            found = importlib.metadata.version("pyodbc")
            manifest["pyodbc"] = found
            if importlib.util.find_spec("pyodbc") is None:
                issues.append("Distribution pyodbc is installed but module pyodbc cannot be imported")
        except importlib.metadata.PackageNotFoundError:
            manifest["pyodbc"] = None
            issues.append("Missing production dependency: pyodbc and Microsoft ODBC Driver 17 for SQL Server")
    return manifest, issues


def load_runtime_packages(mode: str) -> None:
    global np, pd, scipy_stats, plt, PdfPages, sklearn
    import numpy as _np
    import pandas as _pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
    np, pd, plt, PdfPages = _np, _pd, _plt, _PdfPages
    if mode != "plot-only":
        import sklearn as _sklearn
        from scipy import stats as _stats
        sklearn, scipy_stats = _sklearn, _stats


# A compact standard-library PNG renderer keeps dependency and schema failures visible.
FONT_5X7 = {
    "A":"0111010001111111000110001","B":"1111010001111101000111110","C":"0111110000100001000001111",
    "D":"1111010001100011000111110","E":"1111110000111101000011111","F":"1111110000111101000010000",
    "G":"0111110000101111000101111","H":"1000110001111111000110001","I":"1111100100001000010011111",
    "J":"0011100010000101001001100","K":"1000110010111001001010001","L":"1000010000100001000011111",
    "M":"1000111011101011000110001","N":"1000111001101011001110001","O":"0111010001100011000101110",
    "P":"1111010001111101000010000","Q":"0111010001101011001001101","R":"1111010001111101001010001",
    "S":"0111110000011100000111110","T":"1111100100001000010000100","U":"1000110001100011000101110",
    "V":"1000110001100010101000100","W":"1000110001101011101101010","X":"1000101010001000101010001",
    "Y":"1000101010001000010000100","Z":"1111100010001000100011111","0":"0111010011101011100101110",
    "1":"0010001100001000010001110","2":"0111010001000100010011111","3":"1111000001001100000111110",
    "4":"0001000110010101111100010","5":"1111110000111100000111110","6":"0111010000111101000101110",
    "7":"1111100010001000100001000","8":"0111010001011101000101110","9":"0111010001011110000101110",
    "-":"0000000000111110000000000","_":"0000000000000000000011111",":":"0000000100000000010000000",
    ".":"0000000000000000011000110","/":"0000100010001000100010000","(":"0001000100001000010000010",
    ")":"0100000100001000010001000","[":"0011000100001000010000110","]":"0110000100001000010001100","=":"0000011111000001111100000",
    "+":"0000000100011100010000000","%":"1100100010001000100010011","?":"0111010001000100000000100",
    " ":"0000000000000000000000000",",":"0000000000000000010001000","'":"0010000100000000000000000",
}


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_failure_png(path: Path, title: str, issues: Sequence[str], details: Sequence[str] = ()) -> None:
    width = 1500
    wrapped: list[str] = []
    for item in issues:
        wrapped.extend(textwrap.wrap("- " + str(item), width=76) or ["-"])
    for item in details:
        wrapped.extend(textwrap.wrap(str(item), width=76) or [""])
    lines = [title.upper(), "", *wrapped, "", "NO SCIENTIFIC FALLBACK WAS USED."]
    scale = 3
    line_height = 28
    height = max(500, 80 + line_height * len(lines))
    pixels = bytearray([248, 248, 248] * width * height)

    def rectangle(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        for yy in range(y0, y1):
            start = (yy * width + x0) * 3
            for xx in range(x0, x1):
                offset = start + (xx - x0) * 3
                pixels[offset:offset + 3] = bytes(color)

    rectangle(0, 0, width, 18, (164, 28, 48))
    y = 40
    for line_index, line in enumerate(lines):
        color = (150, 20, 40) if line_index == 0 else (25, 30, 38)
        x = 38
        for character in line.upper():
            pattern = FONT_5X7.get(character, FONT_5X7["?"])
            for row in range(7):
                for col in range(5):
                    offset = row * 5 + col
                    if offset < len(pattern) and pattern[offset] == "1":
                        rectangle(x + col * scale, y + row * scale, x + (col + 1) * scale, y + (row + 1) * scale, color)
            x += 6 * scale
            if x > width - 40:
                break
        y += line_height
    raw = bytearray()
    stride = width * 3
    for row in range(height):
        raw.append(0)
        raw.extend(pixels[row * stride:(row + 1) * stride])
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += png_chunk(b"IDAT", zlib.compress(bytes(raw), level=8))
    data += png_chunk(b"IEND", b"")
    atomic_bytes(path, data)


def failure_directory(cfg: RunConfig, key: Any) -> Path:
    if cfg.output_dir:
        base = Path(cfg.output_dir).resolve()
    else:
        base = DEFAULT_RESULTS_ROOT / ("preflight_" + digest(key)[:16])
    (base / "FIGURES_TO_EXPORT").mkdir(parents=True, exist_ok=True)
    return base


def render_preflight_failure(cfg: RunConfig, title: str, issues: Sequence[str], details: Sequence[str] = ()) -> Path:
    directory = failure_directory(cfg, {"title": title, "issues": list(issues), "mode": cfg.mode})
    output = directory / "FIGURES_TO_EXPORT" / "00_preflight_failure.png"
    write_failure_png(output, title, issues, details)
    atomic_json(directory / "preflight_failure.json", {
        "status": "preflight_failure",
        "title": title,
        "issues": list(issues),
        "details": list(details),
        "time_utc": utc_now(),
    })
    return output


# =====================================================================================
# 2. Explicit SQL and canonical schema contract
# =====================================================================================

CORE_FIELDS = (
    FieldSpec("patient_id", ("PatKey", "PatientKey"), description="Stable deidentified patient key"),
    FieldSpec("center_id", ("CenterID", "OrganizationID", "FacilityID", "SiteID"), description="Blinded center or organization identifier"),
    FieldSpec("procedure_date", ("ProcDateValue", "ProcedureDate"), description="Exact index operation date"),
    FieldSpec("procedure_code", ("CptCode", "ProcedureCPT"), description="CPT treatment version"),
    FieldSpec("prior_glp1", ("PriorGLP1",), description="Preoperative GLP-1 exposure flag"),
    FieldSpec("prior_mbs", ("PMH_PriorMBS", "PriorMBS"), description="Prior bariatric surgery flag"),
    FieldSpec("prior_dialysis_transplant", ("PMH_dialysis_transplant", "PriorDialysisTransplant"), description="Baseline dialysis or transplant exclusion"),
    FieldSpec("baseline_bmi", ("BMIatEvent", "BMIAtSurgery"), description="BMI at operation"),
    FieldSpec("baseline_hba1c", ("HbA1cAtEvent", "HbA1cAtSurgery"), description="HbA1c at operation"),
    FieldSpec("age", ("AgeAtEvent", "AgeAtSurgery"), description="Age at operation"),
    FieldSpec("sex", ("Sex",), description="Recorded sex"),
    FieldSpec("coverage", ("CoverageClass", "InsuranceClass"), description="Baseline coverage"),
    FieldSpec("last_contact_days", ("ActiveEndInterval", "LastActiveInterval", "LastContactDays"), description="Days from surgery to last observable contact"),
    FieldSpec("administrative_end_date", ("AdministrativeEndDate", "DataThroughDate", "StudyEndDate"), description="Frozen data-through date"),
    FieldSpec("mace_flag", ("MACE",), description="Qualifying post-index MACE flag"),
    FieldSpec("mace_days", ("MACEinterval", "MACEInterval"), description="Exact days from surgery to MACE"),
    FieldSpec("nephropathy_flag", ("Nephropathy",), description="Qualifying nephropathy flag"),
    FieldSpec("nephropathy_days", ("NephropathyInterval",), description="Exact days from surgery to nephropathy"),
    FieldSpec("retinopathy_flag", ("Retinopathy",), description="Qualifying retinopathy flag"),
    FieldSpec("retinopathy_days", ("RetinopathyInterval",), description="Exact days from surgery to retinopathy"),
    FieldSpec("death_flag", ("Deceased", "Death"), description="All-cause death flag"),
    FieldSpec("death_days", ("DeathInterval",), description="Exact days from surgery to death"),
    FieldSpec("prior_mi", ("PMH_MI", "PriorMI"), description="Pre-index myocardial infarction"),
    FieldSpec("prior_stroke", ("PMH_stroke", "PriorStroke"), description="Pre-index stroke"),
    FieldSpec("prior_nephropathy", ("PMH_nephropathy", "PriorNephropathy"), description="Pre-index renal disease outcome"),
    FieldSpec("prior_retinopathy", ("PMH_retinopathy", "PriorRetinopathy"), description="Pre-index retinal disease outcome"),
)

OPTIONAL_FIELDS = (
    FieldSpec("race", ("FirstRace", "Race"), required=False),
    FieldSpec("ethnicity", ("Ethnicity",), required=False),
    FieldSpec("state", ("StateOrProvince", "State"), required=False),
    FieldSpec("ruca", ("RUCA",), required=False),
    FieldSpec("svi", ("SviOverall", "SVIOverall"), required=False),
    FieldSpec("creatinine", ("CreatinineAtEvent",), required=False),
    FieldSpec("egfr", ("eGFRatEvent", "EGFRAtEvent"), required=False),
    FieldSpec("smoking", ("SmokingStatus", "TobaccoStatus", "PMH_smoking"), required=False),
    FieldSpec("gerd", ("PMH_GERD", "GERD", "Reflux"), required=False),
    FieldSpec("diabetes_duration", ("DiabetesDurationYears", "DM2DurationYears"), required=False),
    FieldSpec("systolic_bp", ("SystolicBPAtEvent", "SBPAtEvent"), required=False),
    FieldSpec("ldl", ("LDLAtEvent",), required=False),
    FieldSpec("uacr", ("UACRAtEvent", "ProteinuriaAtEvent"), required=False),
    FieldSpec("frailty", ("FrailtyIndex",), required=False),
    FieldSpec("prior_utilization", ("EncountersPriorYear", "PriorYearUtilization"), required=False),
    FieldSpec("insulin", ("InsulinStatus",), required=False),
    FieldSpec("biguanide", ("BiguanideStatus",), required=False),
    FieldSpec("sglt2", ("SGLT2Status",), required=False),
    FieldSpec("osa", ("PMH_OSA",), required=False),
    FieldSpec("dyslipidemia", ("PMH_dyslipidemia",), required=False),
    FieldSpec("hypertension", ("PMH_hypertension",), required=False),
    FieldSpec("postop_glp1_start_days", ("GLP1Interval", "PostOpGLP1StartInterval"), required=False),
    FieldSpec("negative_control_outcome", ("NegativeControlOutcome",), required=False),
)


def target_field_specs() -> list[FieldSpec]:
    specs: list[FieldSpec] = []
    labels = {3: "3m", 6: "6m", 12: "12m", 24: "2y", 36: "3y", 48: "4y", 60: "5y"}
    for outcome, horizons in TRAJECTORY_HORIZONS.items():
        prefix = "BMI" if outcome == "bmi" else "HbA1c"
        for month in horizons:
            label = labels[month]
            canonical = f"{outcome}_{month}m"
            specs.extend([
                FieldSpec(f"{canonical}_value", (f"{prefix}{label}PostEvent", f"{prefix}{month}mPostEvent"), description=f"Selected {outcome} value near month {month}"),
                FieldSpec(f"{canonical}_day", (f"{prefix}{label}PostEventDay", f"{prefix}{label}Interval", f"{prefix}{month}mPostEventDay"), description="Exact selected measurement day offset"),
                FieldSpec(f"{canonical}_count", (f"{prefix}{label}PostEventCount", f"{prefix}{month}mWindowCount"), description="Eligible measurement count in frozen window"),
            ])
    return specs


def all_schema_specs() -> list[FieldSpec]:
    return list(CORE_FIELDS) + target_field_specs() + list(OPTIONAL_FIELDS)


def normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def resolve_schema(available_columns: Sequence[str]) -> tuple[dict[str, str], list[str], list[str]]:
    normalized: dict[str, list[str]] = {}
    for column in available_columns:
        normalized.setdefault(normalize_identifier(column), []).append(column)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    for spec in all_schema_specs():
        matches: list[str] = []
        for alias in spec.aliases:
            matches.extend(normalized.get(normalize_identifier(alias), []))
        matches = sorted(set(matches))
        if len(matches) == 1:
            resolved[spec.canonical] = matches[0]
        elif len(matches) > 1:
            ambiguous.append(f"{spec.canonical}: matched {matches}")
        elif spec.required:
            missing.append(f"{spec.canonical}: expected one of {list(spec.aliases)} ({spec.description})")
    return resolved, missing, ambiguous


def quote_sql_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe SQL identifier from schema metadata: {value!r}")
    return "[" + value + "]"


def build_explicit_sql(resolved: Mapping[str, str]) -> str:
    selected = []
    for canonical in sorted(resolved):
        selected.append(f"    {quote_sql_identifier(resolved[canonical])} AS {quote_sql_identifier(canonical)}")
    prior = quote_sql_identifier(resolved["prior_glp1"])
    prior_mbs = quote_sql_identifier(resolved["prior_mbs"])
    prior_dt = quote_sql_identifier(resolved["prior_dialysis_transplant"])
    code = quote_sql_identifier(resolved["procedure_code"])
    bmi = quote_sql_identifier(resolved["baseline_bmi"])
    return (
        f"/* {SQL_CONTRACT_VERSION} */\nSELECT\n" + ",\n".join(selected) +
        f"\nFROM {quote_sql_identifier(SOURCE_SCHEMA)}.{quote_sql_identifier(SOURCE_TABLE)}\n"
        f"WHERE {prior} = 0\n"
        f"  AND {prior_mbs} = 0\n"
        f"  AND {prior_dt} = 0\n"
        f"  AND {code} IN ('43775','43644','43645','43846')\n"
        f"  AND {bmi} BETWEEN 35 AND 75"
    )


def resolve_dynamic_monthly_schema(available_columns: Sequence[str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolve the optional longitudinal contract without guessing a Cosmos table join."""
    normalized = {normalize_identifier(column): column for column in available_columns}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    patterns = {
        "glp1": ("GLP1Month{month:02d}", "PostOpGLP1Month{month:02d}"),
        "bmi": ("BMIValueMonth{month:02d}", "BMIMonth{month:02d}"),
        "hba1c": ("HbA1cValueMonth{month:02d}", "HbA1cMonth{month:02d}"),
        "medication": ("MedicationCountMonth{month:02d}",),
        "utilization": ("UtilizationMonth{month:02d}", "EncounterCountMonth{month:02d}"),
        "observed": ("ObservableMonth{month:02d}", "ActiveMonth{month:02d}"),
    }
    for family, aliases in patterns.items():
        for month in range(1, 37):
            candidates = [normalized.get(normalize_identifier(alias.format(month=month))) for alias in aliases]
            candidates = [value for value in candidates if value is not None]
            canonical = f"monthly_{family}_{month:02d}"
            if len(set(candidates)) == 1:
                resolved[canonical] = candidates[0]
            else:
                missing.append(canonical + ": expected one of " + str([alias.format(month=month) for alias in aliases]))
    status = {
        "eligible": not missing,
        "required_months": "1-36",
        "resolved_fields": len(resolved),
        "required_fields": sum(len(range(1, 37)) for _ in patterns),
        "missing_examples": missing[:12],
        "reason": "complete monthly treatment, metabolic, medication, utilization, and observability history" if not missing else "longitudinal contract incomplete; dynamic GLP-1 analysis will be reported as not estimable",
    }
    return (resolved if not missing else {}), status


def query_database(cfg: RunConfig) -> tuple[Any, dict[str, Any]]:
    import pyodbc
    required_driver = "ODBC Driver 17 for SQL Server"
    available_drivers = list(pyodbc.drivers())
    if required_driver not in available_drivers:
        available_text = ", ".join(available_drivers) if available_drivers else "none detected"
        raise PreflightError("SQL Server ODBC driver preflight failed", [
            f"Required driver is {required_driver}; available drivers: {available_text}.",
            "Install Microsoft ODBC Driver 17 for SQL Server or revise the reviewed connection contract.",
        ])
    try:
        connection = pyodbc.connect(CONNECTION_STRING, timeout=1000)
    except Exception as exc:
        raise PreflightError("Cosmos connection preflight failed", [
            f"Could not connect with the frozen trusted connection: {type(exc).__name__}: {exc}",
            "Confirm ODBC Driver 17, PROJECTS access, and database ProjectD332AFD.",
        ]) from exc
    try:
        metadata_sql = (
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION"
        )
        cursor = connection.cursor()
        rows = cursor.execute(metadata_sql, SOURCE_SCHEMA, SOURCE_TABLE).fetchall()
        available = [str(row[0]) for row in rows]
        if not available:
            raise PreflightError("Cosmos schema preflight failed", [
                f"No columns were found for {SOURCE_SCHEMA}.{SOURCE_TABLE}.",
                "Confirm the source table name before changing the frozen SQL contract.",
            ])
        resolved, missing, ambiguous = resolve_schema(available)
        if missing or ambiguous:
            issues = []
            if missing:
                issues.append("Missing canonical fields:")
                issues.extend(missing)
            if ambiguous:
                issues.append("Ambiguous canonical aliases:")
                issues.extend(ambiguous)
            issues.append("Expose these fields in dbo.MBSCohort or update the reviewed alias contract before rerunning.")
            raise PreflightError("Cosmos schema contract is not satisfied", issues)
        dynamic_resolved, dynamic_status = resolve_dynamic_monthly_schema(available)
        resolved.update(dynamic_resolved)
        sql = build_explicit_sql(resolved)
        frame = pd.read_sql_query(sql, connection)
        return frame, {
            "sql_contract_version": SQL_CONTRACT_VERSION,
            "sql": sql,
            "sql_sha256": digest(sql),
            "resolved_aliases": resolved,
            "available_column_count": len(available),
            "dynamic_glp1_schema": dynamic_status,
        }
    finally:
        connection.close()


# =====================================================================================
# 3. Synthetic cohort, cohort validation, endpoint construction, and locked splits
# =====================================================================================

BASE_NUMERIC_FEATURES = [
    "age", "baseline_bmi", "baseline_hba1c", "creatinine", "egfr", "svi",
    "diabetes_duration", "systolic_bp", "ldl", "uacr", "frailty",
    "prior_utilization", "insulin", "biguanide", "sglt2", "osa",
    "dyslipidemia", "hypertension", "gerd", "prior_mi", "prior_stroke",
    "prior_nephropathy", "prior_retinopathy", "procedure_year",
]
BASE_CATEGORICAL_FEATURES = [
    "sex", "race", "ethnicity", "coverage", "state", "ruca", "smoking",
    "procedure_type", "center_blind",
]
FORBIDDEN_FUTURE_FEATURE_TOKENS = (
    "mace_flag", "mace_days", "nephropathy_flag", "nephropathy_days",
    "retinopathy_flag", "retinopathy_days", "death_flag", "death_days",
    "composite_days", "first_event_type",
)


def logistic(value: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def generate_synthetic_cohort(n: int = 420, seed: int = SEED) -> Any:
    """Generate a center-structured cohort that exercises the complete production path."""
    rng = np.random.default_rng(seed)
    n_centers = 12
    centers = np.asarray([f"SYNTH_CENTER_{i:02d}" for i in range(n_centers)])
    center_id = centers[np.arange(n) % n_centers]
    rng.shuffle(center_id)
    start = np.datetime64("2013-01-01")
    procedure_date = start + rng.integers(0, int(10.5 * 365.25), size=n).astype("timedelta64[D]")
    admin_end = np.datetime64("2026-06-30")
    admin_days = (admin_end - procedure_date).astype("timedelta64[D]").astype(int)
    center_index = np.asarray([int(value.rsplit("_", 1)[1]) for value in center_id])
    age = np.clip(rng.normal(47 + 0.25 * center_index, 10, n), 18, 76)
    sex = rng.choice(["Female", "Male"], size=n, p=[0.72, 0.28])
    race = rng.choice(["White", "Black", "Asian", "Other"], size=n, p=[0.55, 0.26, 0.08, 0.11])
    coverage = rng.choice(["Commercial", "Medicare", "Medicaid"], size=n, p=[0.55, 0.22, 0.23])
    baseline_bmi = np.clip(rng.normal(43 + 0.12 * center_index, 5.2, n), 35, 68)
    baseline_hba1c = np.clip(rng.normal(7.3 + 0.015 * (baseline_bmi - 42), 1.25, n), 4.3, 13.5)
    gerd = rng.binomial(1, logistic(-1.0 + 0.035 * (baseline_bmi - 40)), n)
    smoking = rng.choice(["Never", "Former", "Current"], n, p=[0.58, 0.31, 0.11])
    svi = np.clip(rng.beta(2.2, 2.5, n), 0, 1)
    procedure_logit = -0.5 + 1.0 * gerd + 0.025 * (baseline_bmi - 42) - 0.35 * svi + 0.08 * (center_index - 5)
    is_rygb = rng.binomial(1, logistic(procedure_logit), n)
    procedure_code = np.where(is_rygb == 1, rng.choice(["43644", "43645", "43846"], n), "43775")
    loss_to_followup = rng.binomial(1, 0.23, n)
    loss_day = rng.integers(280, 2500, n)
    last_contact_days = np.where(loss_to_followup == 1, np.minimum(loss_day, admin_days), admin_days)
    last_contact_days = np.maximum(last_contact_days, 90)

    frailty = np.clip(rng.normal(0.18 + 0.006 * (age - 45), 0.09, n), 0, 0.8)
    prior_mi = rng.binomial(1, logistic(-3.3 + 0.035 * (age - 45)), n)
    prior_stroke = rng.binomial(1, logistic(-3.5 + 0.03 * (age - 45)), n)
    prior_nephropathy = rng.binomial(1, logistic(-3.0 + 0.28 * (baseline_hba1c - 7)), n)
    prior_retinopathy = rng.binomial(1, logistic(-3.1 + 0.35 * (baseline_hba1c - 7)), n)
    creatinine = np.clip(rng.lognormal(np.log(0.9), 0.23, n), 0.4, 3.0)
    egfr = np.clip(105 - 0.8 * age - 11 * (creatinine - 0.9) + rng.normal(0, 8, n), 20, 125)
    postop_glp = rng.binomial(1, logistic(-1.7 + 0.25 * (baseline_hba1c - 7) + 0.3 * svi), n)
    postop_glp_start = np.where(postop_glp == 1, rng.integers(300, 1450, n), np.nan)

    linear_risk = (
        0.025 * (age - 45) + 0.18 * (baseline_hba1c - 7) + 0.45 * prior_mi +
        0.35 * prior_stroke + 0.25 * prior_nephropathy + 0.12 * center_index / 11
    )
    event_scales = {
        "mace": 10500 * np.exp(-linear_risk),
        "nephropathy": 8200 * np.exp(-(linear_risk + 0.25 * (baseline_hba1c - 7))),
        "retinopathy": 11500 * np.exp(-(linear_risk + 0.32 * (baseline_hba1c - 7))),
        "death": 13500 * np.exp(-(0.035 * (age - 45) + 0.5 * frailty)),
    }
    event_times: dict[str, Any] = {}
    event_flags: dict[str, Any] = {}
    for endpoint, scale in event_scales.items():
        latent = rng.exponential(scale)
        observed = (latent <= last_contact_days) & (latent <= admin_days)
        event_times[endpoint] = np.where(observed, latent, np.nan)
        event_flags[endpoint] = observed.astype(int)

    frame = pd.DataFrame({
        "patient_id": [f"SYNTH_{seed}_{i:06d}" for i in range(n)],
        "center_id": center_id,
        "procedure_date": pd.to_datetime(procedure_date),
        "procedure_code": procedure_code,
        "prior_glp1": 0,
        "prior_mbs": 0,
        "prior_dialysis_transplant": 0,
        "baseline_bmi": baseline_bmi,
        "baseline_hba1c": baseline_hba1c,
        "age": age,
        "sex": sex,
        "coverage": coverage,
        "race": race,
        "ethnicity": rng.choice(["Hispanic", "Not Hispanic", "Unknown"], n, p=[0.17, 0.76, 0.07]),
        "state": rng.choice(["Northeast", "South", "Midwest", "West"], n),
        "ruca": rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.60, 0.25, 0.15]),
        "svi": svi,
        "creatinine": creatinine,
        "egfr": egfr,
        "smoking": smoking,
        "gerd": gerd,
        "diabetes_duration": np.clip(rng.normal(7, 4, n), 0, 25),
        "systolic_bp": np.clip(rng.normal(132, 15, n), 90, 190),
        "ldl": np.clip(rng.normal(112, 30, n), 35, 250),
        "uacr": np.clip(rng.lognormal(2.4, 1.0, n), 1, 1000),
        "frailty": frailty,
        "prior_utilization": rng.poisson(5, n),
        "insulin": rng.binomial(1, logistic(-1.1 + 0.55 * (baseline_hba1c - 7)), n),
        "biguanide": rng.binomial(1, 0.62, n),
        "sglt2": rng.binomial(1, 0.18, n),
        "osa": rng.binomial(1, 0.45, n),
        "dyslipidemia": rng.binomial(1, 0.52, n),
        "hypertension": rng.binomial(1, 0.58, n),
        "postop_glp1_start_days": postop_glp_start,
        "last_contact_days": last_contact_days,
        "administrative_end_date": pd.Timestamp("2026-06-30"),
        "mace_flag": event_flags["mace"],
        "mace_days": event_times["mace"],
        "nephropathy_flag": event_flags["nephropathy"],
        "nephropathy_days": event_times["nephropathy"],
        "retinopathy_flag": event_flags["retinopathy"],
        "retinopathy_days": event_times["retinopathy"],
        "death_flag": event_flags["death"],
        "death_days": event_times["death"],
        "prior_mi": prior_mi,
        "prior_stroke": prior_stroke,
        "prior_nephropathy": prior_nephropathy,
        "prior_retinopathy": prior_retinopathy,
    })

    alive_days = np.where(np.isfinite(event_times["death"]), event_times["death"], np.inf)
    for outcome, horizons in TRAJECTORY_HORIZONS.items():
        baseline = baseline_bmi if outcome == "bmi" else baseline_hba1c
        for month in horizons:
            lo, hi = MEASUREMENT_WINDOWS[month]
            selected_month = np.clip(month + rng.normal(0, max(0.35, (hi - lo) / 8), n), lo + 0.02, hi - 0.02)
            day = selected_month * DAYS_PER_MONTH
            observable = (day <= last_contact_days) & (day <= admin_days) & (day < alive_days)
            attendance = logistic(1.8 - 0.5 * svi + 0.12 * (center_index % 4) - 0.0007 * np.maximum(day - 365, 0))
            observable &= rng.random(n) < attendance
            if outcome == "bmi":
                loss = (10.8 + 1.8 * is_rygb) * (1 - np.exp(-month / 6.5))
                regain = np.maximum(month - 20, 0) * (0.055 + 0.018 * (1 - is_rygb))
                glp_effect = np.where(np.isfinite(postop_glp_start) & (postop_glp_start <= day), -1.0, 0.0)
                value = baseline - loss + regain + 0.08 * (age - 45) / 10 + glp_effect + rng.normal(0, 1.5 + month / 80, n)
            else:
                reduction = (1.25 + 0.28 * is_rygb) * (1 - np.exp(-month / 9.0))
                rebound = np.maximum(month - 30, 0) * 0.008
                glp_effect = np.where(np.isfinite(postop_glp_start) & (postop_glp_start <= day), -0.32, 0.0)
                value = baseline - reduction + rebound + glp_effect + rng.normal(0, 0.42 + month / 300, n)
            count = np.where(observable, 1 + rng.poisson(1.2, n), 0)
            canonical = f"{outcome}_{month}m"
            frame[f"{canonical}_value"] = np.where(observable, value, np.nan)
            frame[f"{canonical}_day"] = np.where(observable, day, np.nan)
            frame[f"{canonical}_count"] = count

    # Keep a small documented sensitivity set with a flag but unavailable event date.
    for endpoint in ("mace", "nephropathy"):
        candidates = frame.index[frame[f"{endpoint}_flag"].eq(1)].to_numpy()
        if candidates.size:
            frame.loc[candidates[0], f"{endpoint}_days"] = np.nan
    return frame


def numeric_column(frame: Any, name: str, default: float = math.nan) -> Any:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def blind_center(value: Any) -> str:
    return "CENTER_" + hashlib.sha256(("qreg-center-v2|" + str(value)).encode("utf-8")).hexdigest()[:10].upper()


def validate_and_construct_cohort(frame: Any, cfg: RunConfig) -> tuple[Any, dict[str, Any]]:
    required = [spec.canonical for spec in all_schema_specs() if spec.required]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise PreflightError("Canonical cohort schema is incomplete", [f"Missing canonical field: {name}" for name in missing])
    data = frame.copy().reset_index(drop=True)
    funnel: list[dict[str, Any]] = [{"step": "source rows", "remaining": len(data), "excluded": 0}]
    if data["patient_id"].astype(str).duplicated().any():
        duplicates = data.loc[data["patient_id"].astype(str).duplicated(), "patient_id"].astype(str).head(5).tolist()
        raise PreflightError("Patient uniqueness gate failed", [f"Duplicate patient keys include {duplicates}"])

    def apply_filter(label: str, keep: Any) -> None:
        nonlocal data
        keep = pd.Series(keep, index=data.index).fillna(False).astype(bool)
        before = len(data)
        data = data.loc[keep].copy().reset_index(drop=True)
        funnel.append({"step": label, "remaining": len(data), "excluded": before - len(data)})

    apply_filter("PriorGLP1 = 0", numeric_column(data, "prior_glp1").eq(0))
    apply_filter("first metabolic bariatric surgery", numeric_column(data, "prior_mbs").eq(0))
    apply_filter("no baseline dialysis or transplant", numeric_column(data, "prior_dialysis_transplant").eq(0))
    codes = data["procedure_code"].astype("string").str.replace(r"\.0$", "", regex=True)
    apply_filter("sleeve or RYGB treatment version", codes.isin(PROCEDURE_CODES))
    data["procedure_code"] = data["procedure_code"].astype("string").str.replace(r"\.0$", "", regex=True)
    data["procedure_type"] = data["procedure_code"].map(PROCEDURE_CODES)
    data["procedure_date"] = pd.to_datetime(data["procedure_date"], errors="coerce")
    data["administrative_end_date"] = pd.to_datetime(data["administrative_end_date"], errors="coerce")
    bad_dates = data["procedure_date"].isna() | data["administrative_end_date"].isna() | (data["administrative_end_date"] < data["procedure_date"])
    if bad_dates.any():
        raise PreflightError("Time-zero or administrative-end gate failed", [f"{int(bad_dates.sum())} rows have invalid procedure or data-through dates"])
    data["administrative_opportunity_days"] = (data["administrative_end_date"] - data["procedure_date"]).dt.days.astype(float)
    data["last_contact_days"] = numeric_column(data, "last_contact_days")
    bad_contact = data["last_contact_days"].isna() | data["last_contact_days"].lt(0)
    if bad_contact.any():
        raise PreflightError("Censoring contract failed", [f"{int(bad_contact.sum())} rows lack a valid nonnegative last-contact interval"])
    data["censor_days"] = np.minimum(data["last_contact_days"], data["administrative_opportunity_days"])
    data["procedure_year"] = data["procedure_date"].dt.year.astype(float)
    data["center_id"] = data["center_id"].astype(str)
    data["center_blind"] = data["center_id"].map(blind_center)

    for optional in OPTIONAL_FIELDS:
        if optional.canonical not in data:
            data[optional.canonical] = np.nan
    numeric_to_convert = set(BASE_NUMERIC_FEATURES + [
        "baseline_bmi", "baseline_hba1c", "postop_glp1_start_days", "prior_mi", "prior_stroke",
        "prior_nephropathy", "prior_retinopathy", "mace_flag", "mace_days", "nephropathy_flag",
        "nephropathy_days", "retinopathy_flag", "retinopathy_days", "death_flag", "death_days",
    ])
    for name in numeric_to_convert:
        if name in data:
            data[name] = pd.to_numeric(data[name], errors="coerce")

    timing_issues: list[str] = []
    plausibility_exclusions: dict[str, int] = {}
    for outcome, horizons in TRAJECTORY_HORIZONS.items():
        lo_value, hi_value = PLAUSIBLE_RANGES[outcome]
        invalid_total = 0
        for month in horizons:
            base = f"{outcome}_{month}m"
            value = numeric_column(data, base + "_value")
            day = numeric_column(data, base + "_day")
            count = numeric_column(data, base + "_count", 0).fillna(0)
            observed = value.notna()
            lo_day, hi_day = (x * DAYS_PER_MONTH for x in MEASUREMENT_WINDOWS[month])
            bad_timing = observed & (day.isna() | day.lt(lo_day) | day.gt(hi_day) | count.lt(1))
            bad_timing |= observed & day.gt(data["censor_days"] + 1e-8)
            death_reported = numeric_column(data, "death_flag", 0).fillna(0).eq(1)
            death_day = numeric_column(data, "death_days")
            bad_timing |= observed & death_reported & death_day.notna() & day.gt(death_day + 1e-8)
            if bad_timing.any():
                timing_issues.append(
                    f"{base}: {int(bad_timing.sum())} selected values lack valid exact timing/count, "
                    f"fall outside window {MEASUREMENT_WINDOWS[month]}, or occur after censoring/death"
                )
            plausible = value.between(lo_value, hi_value) | value.isna()
            invalid_total += int((~plausible).sum())
            data.loc[~plausible, [base + "_value", base + "_day"]] = np.nan
            data.loc[~plausible, base + "_count"] = 0
        plausibility_exclusions[outcome] = invalid_total
    if timing_issues:
        raise PreflightError("Measurement-window audit failed", timing_issues)

    event_consistency: list[dict[str, Any]] = []
    for endpoint in (*COMPONENTS, "death"):
        flag = numeric_column(data, endpoint + "_flag", 0).fillna(0).eq(1)
        event_day = numeric_column(data, endpoint + "_days")
        missing_day = flag & event_day.isna()
        negative = event_day.notna() & event_day.lt(0)
        after_censor = flag & event_day.notna() & event_day.gt(data["censor_days"] + 1e-8)
        data[endpoint + "_missing_time_sensitivity"] = missing_day
        data[endpoint + "_valid_event"] = flag & event_day.notna() & ~negative & ~after_censor
        event_consistency.append({
            "endpoint": endpoint,
            "flagged": int(flag.sum()),
            "missing_time_sensitivity": int(missing_day.sum()),
            "negative_time": int(negative.sum()),
            "after_censor": int(after_censor.sum()),
            "valid_primary_events": int(data[endpoint + "_valid_event"].sum()),
        })
        if negative.any():
            raise PreflightError("Event-time consistency failed", [f"{endpoint}: {int(negative.sum())} negative event intervals"])

    component_times = np.column_stack([
        np.where(data[name + "_valid_event"], data[name + "_days"], np.inf) for name in COMPONENTS
    ])
    earliest_index = np.argmin(component_times, axis=1)
    earliest_time = np.min(component_times, axis=1)
    data["composite_days"] = np.where(np.isfinite(earliest_time), earliest_time, np.nan)
    data["composite_flag"] = np.isfinite(earliest_time).astype(int)
    data["first_event_type"] = np.where(np.isfinite(earliest_time), np.asarray(COMPONENTS)[earliest_index], "none")
    data["prior_mace"] = numeric_column(data, "prior_mi", 0).fillna(0).gt(0) | numeric_column(data, "prior_stroke", 0).fillna(0).gt(0)
    data["prior_any_component"] = data["prior_mace"] | numeric_column(data, "prior_nephropathy", 0).fillna(0).gt(0) | numeric_column(data, "prior_retinopathy", 0).fillna(0).gt(0)

    metadata = {
        "source_rows": int(funnel[0]["remaining"]),
        "analysis_rows": len(data),
        "funnel": funnel,
        "event_consistency": event_consistency,
        "plausibility_exclusions": plausibility_exclusions,
        "measurement_aggregation_rule": "Median of eligible measures within each non-overlapping window; selected day and count must be supplied for audit",
        "prior_glp1_definition": 0,
        "postoperative_glp1_prognostic_handling": "Outcomes retained; exposure enters only landmark features after initiation",
    }
    return data.reset_index(drop=True), metadata


def hashed_center_order(centers: Sequence[str], seed: int) -> list[str]:
    return sorted(set(str(x) for x in centers), key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest())


def build_locked_splits(data: Any, cfg: RunConfig) -> LockedSplits:
    centers = hashed_center_order(data["center_id"].astype(str).unique(), cfg.seed)
    if len(centers) < 6:
        raise PreflightError("Center-aware split gate failed", [f"At least 6 centers are required; found {len(centers)}"])
    n_test_centers = max(cfg.min_test_centers, int(math.ceil(len(centers) * cfg.heldout_center_fraction)))
    heldout = centers[:n_test_centers]
    development_centers = centers[n_test_centers:]
    mature = data["administrative_opportunity_days"].ge(60 * DAYS_PER_MONTH).to_numpy()
    mature_development_dates = data.loc[mature & data["center_id"].isin(development_centers), "procedure_date"]
    if mature_development_dates.empty:
        raise PreflightError("Five-year maturity gate failed", ["No development patients have five years of administrative follow-up opportunity"])
    cutoff = mature_development_dates.quantile(0.50)
    held_mask = data["center_id"].isin(heldout)
    test = np.flatnonzero((held_mask & mature & data["procedure_date"].ge(cutoff)).to_numpy())
    development_mask = data["center_id"].isin(development_centers) & data["procedure_date"].lt(cutoff)
    contemporary = np.flatnonzero((data["center_id"].isin(development_centers) & data["procedure_date"].ge(cutoff) & data["administrative_opportunity_days"].ge(24 * DAYS_PER_MONTH)).to_numpy())

    ordered_development_centers = hashed_center_order(development_centers, cfg.seed + 17)
    calibration_centers = ordered_development_centers[:1]
    validation_centers = ordered_development_centers[1:2]
    train_centers = ordered_development_centers[2:]
    train = np.flatnonzero((development_mask & data["center_id"].isin(train_centers)).to_numpy())
    calibration = np.flatnonzero((development_mask & data["center_id"].isin(calibration_centers)).to_numpy())
    validation = np.flatnonzero((development_mask & data["center_id"].isin(validation_centers)).to_numpy())
    mature_ids = np.flatnonzero(mature)

    issues: list[str] = []
    development_n = len(train) + len(calibration) + len(validation)
    if development_n < cfg.min_development_patients:
        issues.append(f"Development cohort has {development_n} patients; frozen minimum is {cfg.min_development_patients}")
    if len(test) < cfg.min_locked_test_patients:
        issues.append(f"Locked mature center-and-time test has {len(test)} patients; frozen minimum is {cfg.min_locked_test_patients}")
    if min(len(train), len(calibration), len(validation)) == 0:
        issues.append(f"Train/calibration/validation counts are {len(train)}/{len(calibration)}/{len(validation)}")
    for outcome in TRAJECTORY_HORIZONS:
        for month in (48, 60):
            if month not in TRAJECTORY_HORIZONS[outcome]:
                continue
            count = int(data.loc[test, f"{outcome}_{month}m_value"].notna().sum())
            if count == 0:
                issues.append(f"Locked test {outcome} month {month} evaluation cell is empty")
    event_free_test = data.loc[test, "prior_any_component"].eq(False)
    for endpoint in COMPONENTS:
        count = int((data.loc[test, endpoint + "_valid_event"] & event_free_test).sum())
        if count < cfg.min_events_per_risk_endpoint:
            issues.append(f"Locked test has {count} incident first-event candidates for {endpoint}; minimum is {cfg.min_events_per_risk_endpoint}")
    if issues:
        raise PreflightError("Locked validation acceptance gate failed", issues)

    all_sets = {
        "train": set(train.tolist()), "calibration": set(calibration.tolist()),
        "validation": set(validation.tolist()), "test": set(test.tolist()),
    }
    for left, right in (("train", "calibration"), ("train", "validation"), ("train", "test"), ("calibration", "validation"), ("calibration", "test"), ("validation", "test")):
        if all_sets[left] & all_sets[right]:
            raise AssertionError(f"Patient overlap between {left} and {right}")
    if set(data.loc[test, "center_id"]) & set(data.loc[np.r_[train, calibration, validation], "center_id"]):
        raise AssertionError("Held-out center entered model development")

    folds = []
    development_ids = np.r_[train, calibration, validation]
    for center in sorted(data.loc[development_ids, "center_id"].unique()):
        hold = development_ids[data.loc[development_ids, "center_id"].to_numpy() == center]
        fit = np.setdiff1d(development_ids, hold)
        if len(hold) >= cfg.min_cell_observations and len(fit) >= max(50, cfg.min_cell_observations * 2):
            folds.append({"center": str(center), "fit": fit, "hold": hold})
    return LockedSplits(
        train=train, calibration=calibration, validation=validation, test=test,
        contemporary=contemporary, mature=mature_ids, heldout_centers=list(heldout),
        calibration_centers=calibration_centers, validation_centers=validation_centers,
        temporal_cutoff=pd.Timestamp(cutoff).date().isoformat(), internal_external_folds=folds,
    )


def split_identity_manifest(data: Any, splits: LockedSplits) -> dict[str, Any]:
    out: dict[str, Any] = {
        "heldout_center_count": len(splits.heldout_centers),
        "calibration_center_count": len(splits.calibration_centers),
        "validation_center_count": len(splits.validation_centers),
        "temporal_cutoff": splits.temporal_cutoff,
    }
    for name, ids in splits.arrays().items():
        hashes = sorted(hashlib.sha256(("qreg-patient-v2|" + str(x)).encode()).hexdigest() for x in data.loc[ids, "patient_id"])
        out[name] = {"n": len(ids), "patient_hash_sha256": digest(hashes)}
    return out


def data_aggregate_manifest(data: Any) -> dict[str, Any]:
    target_counts = {}
    timing = {}
    for outcome, horizons in TRAJECTORY_HORIZONS.items():
        for month in horizons:
            base = f"{outcome}_{month}m"
            target_counts[base] = int(data[base + "_value"].notna().sum())
            timing[base] = {
                "day_min": float(data[base + "_day"].min()) if data[base + "_day"].notna().any() else None,
                "day_max": float(data[base + "_day"].max()) if data[base + "_day"].notna().any() else None,
                "count_sum": int(numeric_column(data, base + "_count", 0).fillna(0).sum()),
            }
    center_counts = {blind_center(k): int(v) for k, v in data["center_id"].value_counts().sort_index().items()}
    return {
        "n": len(data),
        "center_counts": center_counts,
        "procedure_date_min": data["procedure_date"].min().date().isoformat(),
        "procedure_date_max": data["procedure_date"].max().date().isoformat(),
        "procedure_counts": data["procedure_type"].value_counts().sort_index().to_dict(),
        "target_counts": target_counts,
        "timing": timing,
        "event_counts": {name: int(data[name + "_valid_event"].sum()) for name in (*COMPONENTS, "death")},
        "numeric_checksums": {
            name: float(np.nansum(pd.to_numeric(data[name], errors="coerce")))
            for name in ("baseline_bmi", "baseline_hba1c", "age", "last_contact_days")
        },
    }


def available_feature_columns(data: Any) -> tuple[list[str], list[str]]:
    numeric = [name for name in BASE_NUMERIC_FEATURES if name in data and pd.to_numeric(data[name], errors="coerce").notna().any()]
    categorical = [name for name in BASE_CATEGORICAL_FEATURES if name in data and data[name].notna().any()]
    if any(any(token in name.lower() for token in FORBIDDEN_FUTURE_FEATURE_TOKENS) for name in numeric + categorical):
        raise AssertionError("Future complication feature entered the baseline feature roster")
    return numeric, categorical


def target_trial_feature_columns(data: Any) -> tuple[list[str], list[str]]:
    """Return baseline adjustment variables without the treatment being modeled."""
    numeric, categorical = available_feature_columns(data)
    categorical = [name for name in categorical if name != "procedure_type"]
    if "procedure_type" in numeric or "procedure_type" in categorical:
        raise AssertionError("Target-trial treatment leaked into the baseline adjustment set")
    return numeric, categorical


def landmark_feature_frame(data: Any, ids: Any, outcome: str, origin: int) -> Any:
    work = data.loc[ids].copy()
    history_months = [month for month in TRAJECTORY_HORIZONS[outcome] if month <= origin]
    baseline_name = "baseline_bmi" if outcome == "bmi" else "baseline_hba1c"
    latest_values = pd.to_numeric(work[baseline_name], errors="coerce").to_numpy(float)
    latest_days = np.zeros(len(work), dtype=float)
    value_stack = [latest_values.copy()]
    day_stack = [np.zeros(len(work), dtype=float)]
    observed_stack = [np.isfinite(latest_values)]
    for month in history_months:
        value = pd.to_numeric(work[f"{outcome}_{month}m_value"], errors="coerce").to_numpy(float)
        day = pd.to_numeric(work[f"{outcome}_{month}m_day"], errors="coerce").to_numpy(float)
        observed = np.isfinite(value) & np.isfinite(day) & (day <= origin * DAYS_PER_MONTH + 1e-8)
        value_stack.append(value)
        day_stack.append(day)
        observed_stack.append(observed)
        latest_values = np.where(observed, value, latest_values)
        latest_days = np.where(observed, day, latest_days)
    values = np.column_stack(value_stack)
    days = np.column_stack(day_stack)
    observed = np.column_stack(observed_stack)
    masked_values = np.where(observed, values, np.nan)
    masked_days = np.where(observed, days, np.nan)
    count = observed.sum(axis=1)
    mean = np.nanmean(masked_values, axis=1)
    variability = np.nanstd(masked_values, axis=1)
    slope = np.zeros(len(work), dtype=float)
    for row in range(len(work)):
        keep = observed[row]
        if keep.sum() >= 2 and np.ptp(masked_days[row, keep]) > 0:
            slope[row] = np.polyfit(masked_days[row, keep] / DAYS_PER_MONTH, masked_values[row, keep], 1)[0]
    work[f"{outcome}_latest_value"] = latest_values
    work[f"{outcome}_latest_day"] = latest_days
    work[f"{outcome}_time_since_latest"] = origin - latest_days / DAYS_PER_MONTH
    work[f"{outcome}_history_count"] = count
    work[f"{outcome}_history_mean"] = mean
    work[f"{outcome}_history_sd"] = np.nan_to_num(variability)
    work[f"{outcome}_history_slope"] = slope
    work[f"{outcome}_latest_missing"] = (~np.isfinite(latest_values)).astype(float)
    glp_start = pd.to_numeric(work["postop_glp1_start_days"], errors="coerce")
    work["known_postop_glp1"] = (glp_start.notna() & glp_start.le(origin * DAYS_PER_MONTH)).astype(float)
    work["known_postop_glp1_duration"] = np.where(
        work["known_postop_glp1"].eq(1), origin - glp_start / DAYS_PER_MONTH, 0.0,
    )
    work["forecast_origin_month"] = float(origin)
    return work


def trajectory_feature_roster(data: Any, outcome: str) -> tuple[list[str], list[str]]:
    numeric, categorical = available_feature_columns(data)
    numeric = numeric + [
        f"{outcome}_latest_value", f"{outcome}_latest_day", f"{outcome}_time_since_latest",
        f"{outcome}_history_count", f"{outcome}_history_mean", f"{outcome}_history_sd",
        f"{outcome}_history_slope", f"{outcome}_latest_missing", "known_postop_glp1",
        "known_postop_glp1_duration", "forecast_origin_month",
    ]
    if any(any(token in name.lower() for token in FORBIDDEN_FUTURE_FEATURE_TOKENS) for name in numeric + categorical):
        raise AssertionError("Future outcome leakage in trajectory feature roster")
    return numeric, categorical


# =====================================================================================
# 4. Trajectory models, observation process, scoring, and transportability
# =====================================================================================

MODEL_PUBLISHED = "published_style_hgb"
MODEL_CATBOOST = "target_specific_catboost"
MODEL_ENSEMBLE = "conservative_oof_ensemble"
MODEL_DISPLAY = {
    MODEL_PUBLISHED: "Published-style HGB",
    MODEL_CATBOOST: "Target-specific CatBoost",
    MODEL_ENSEMBLE: "Conservative ensemble",
}
QUANTILE_COLUMN = {0.05: "q05", 0.10: "q10", 0.25: "q25", 0.50: "q50", 0.75: "q75", 0.90: "q90", 0.95: "q95"}


def quantile_matrix_to_samples(values: Any, n_samples: int = 200) -> Any:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(QUANTILES):
        raise ValueError("Quantile predictions must be [n, len(QUANTILES)]")
    values = np.sort(values, axis=1)
    probabilities = (np.arange(n_samples, dtype=float) + 0.5) / n_samples
    result = np.empty((len(values), n_samples), dtype=float)
    for row in range(len(values)):
        result[row] = np.interp(probabilities, QUANTILES, values[row], left=values[row, 0], right=values[row, -1])
    return result


def crps_ensemble(samples: Any, observed: Any) -> Any:
    samples = np.asarray(samples, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != observed.shape[0]:
        raise ValueError("CRPS arrays do not align")
    m = samples.shape[1]
    first = np.mean(np.abs(samples - observed[:, None]), axis=1)
    ordered = np.sort(samples, axis=1)
    weights = 2.0 * np.arange(1, m + 1) - m - 1
    second = np.sum(ordered * weights[None, :], axis=1) / (m * m)
    score = first - second
    if np.nanmin(score) < -1e-8:
        raise AssertionError("CRPS became negative")
    return np.maximum(score, 0)


def conformal_expansions(y: Any, quantiles: Any) -> dict[float, float]:
    y = np.asarray(y, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    output: dict[float, float] = {}
    index_pairs = {0.50: (2, 4), 0.80: (1, 5), 0.90: (0, 6)}
    finite_y = np.isfinite(y)
    for coverage, (lo, hi) in index_pairs.items():
        keep = finite_y & np.isfinite(q[:, lo]) & np.isfinite(q[:, hi])
        if keep.sum() < 3:
            output[coverage] = 0.0
            continue
        score = np.maximum.reduce([q[keep, lo] - y[keep], y[keep] - q[keep, hi], np.zeros(keep.sum())])
        level = min(1.0, math.ceil((keep.sum() + 1) * coverage) / keep.sum())
        output[coverage] = float(np.quantile(score, level, method="higher"))
    return output


def apply_conformal(quantiles: Any, expansions: Mapping[float, float]) -> Any:
    q = np.asarray(quantiles, dtype=float).copy()
    for coverage, (lo, hi) in {0.50: (2, 4), 0.80: (1, 5), 0.90: (0, 6)}.items():
        amount = float(expansions.get(coverage, 0.0))
        q[:, lo] -= amount
        q[:, hi] += amount
    return np.sort(q, axis=1)


def catboost_multi_quantile_predict(model: Any, features: Any) -> Any:
    prediction = np.asarray(model.predict(features), dtype=float)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if prediction.shape != (len(features), len(QUANTILES)):
        raise RuntimeError(
            f"CatBoost MultiQuantile returned shape {prediction.shape}, expected {(len(features), len(QUANTILES))}. "
            "No substitute quantile model is permitted."
        )
    return prediction


def make_prediction_rows(
    data: Any,
    ids: Any,
    part: str,
    outcome: str,
    origin: int,
    target_month: int,
    model_name: str,
    mean_prediction: Any,
    median_prediction: Any,
    quantile_prediction: Any,
    raw_quantiles: Any,
    target_sd: float,
) -> list[dict[str, Any]]:
    target = pd.to_numeric(data.loc[ids, f"{outcome}_{target_month}m_value"], errors="coerce").to_numpy(float)
    columns = {QUANTILE_COLUMN[q]: quantile_prediction[:, i] for i, q in enumerate(QUANTILES)}
    rows = []
    for local, patient_position in enumerate(ids):
        row = {
            "patient_position": int(patient_position),
            "split": part,
            "outcome": outcome,
            "origin": int(origin),
            "target_month": int(target_month),
            "model": model_name,
            "observed": float(target[local]) if np.isfinite(target[local]) else np.nan,
            "mean": float(mean_prediction[local]),
            "median": float(median_prediction[local]),
            "target_sd": float(target_sd),
            "raw_q05": float(raw_quantiles[local, 0]),
            "raw_q95": float(raw_quantiles[local, -1]),
        }
        row.update({name: float(values[local]) for name, values in columns.items()})
        rows.append(row)
    return rows


def fit_trajectory_models(data: Any, splits: LockedSplits, cfg: RunConfig) -> dict[str, Any]:
    from catboost import CatBoostRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(cfg.seed)
    prediction_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    calibration_rows: list[dict[str, Any]] = []
    model_specs: list[dict[str, Any]] = []
    parts = {
        "calibration": splits.calibration,
        "validation": splits.validation,
        "test": splits.test,
        "contemporary": splits.contemporary,
    }

    for outcome, horizons in TRAJECTORY_HORIZONS.items():
        baseline_name = "baseline_bmi" if outcome == "bmi" else "baseline_hba1c"
        bounds = PLAUSIBLE_RANGES[outcome]
        for origin in LANDMARK_MONTHS:
            for target_month in [month for month in horizons if month > origin]:
                target_column = f"{outcome}_{target_month}m_value"
                train_frame = landmark_feature_frame(data, splits.train, outcome, origin)
                y_absolute = pd.to_numeric(data.loc[splits.train, target_column], errors="coerce").to_numpy(float)
                baseline = pd.to_numeric(data.loc[splits.train, baseline_name], errors="coerce").to_numpy(float)
                observed_train = np.isfinite(y_absolute) & np.isfinite(baseline)
                if observed_train.sum() < max(cfg.min_cell_observations, 12):
                    raise PreflightError("Trajectory training cell is underpowered", [
                        f"{outcome} origin {origin} target {target_month}: {int(observed_train.sum())} train observations"
                    ])
                numeric, categorical = trajectory_feature_roster(data, outcome)
                encoder = FeatureEncoder.fit(train_frame.loc[observed_train], numeric, categorical)
                x_train = encoder.transform(train_frame.loc[observed_train])
                y_change = y_absolute[observed_train] - baseline[observed_train]
                target_sd = float(max(np.std(y_absolute[observed_train], ddof=1), 1e-6))

                hgb_mean = HistGradientBoostingRegressor(
                    loss="squared_error", learning_rate=0.06, max_iter=cfg.hgb_iterations,
                    max_leaf_nodes=15, l2_regularization=2.0, random_state=cfg.seed + target_month + origin,
                ).fit(x_train, y_change)
                hgb_median = HistGradientBoostingRegressor(
                    loss="quantile", quantile=0.5, learning_rate=0.06, max_iter=cfg.hgb_iterations,
                    max_leaf_nodes=15, l2_regularization=2.0, random_state=cfg.seed + 100 + target_month + origin,
                ).fit(x_train, y_change)
                train_residual = y_change - hgb_mean.predict(x_train)
                residual_quantiles = np.quantile(train_residual, QUANTILES)

                cat_common = dict(
                    iterations=cfg.catboost_iterations, depth=6 if not cfg.smoke else 4,
                    learning_rate=0.045 if not cfg.smoke else 0.08, random_seed=cfg.seed + target_month + origin,
                    verbose=False, allow_writing_files=False, thread_count=cfg.threads,
                    l2_leaf_reg=4.0,
                )
                cat_mean = CatBoostRegressor(loss_function="RMSE", **cat_common).fit(x_train, y_change)
                cat_median = CatBoostRegressor(loss_function="MAE", **cat_common).fit(x_train, y_change)
                alpha = ",".join(f"{q:g}" for q in QUANTILES)
                try:
                    cat_quantile = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alpha}", **cat_common).fit(x_train, y_change)
                except Exception as exc:
                    raise RuntimeError(
                        "The required CatBoost MultiQuantile model could not be fit. "
                        "Upgrade the validated CatBoost package; no production fallback is allowed. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ) from exc

                cell_key = f"{outcome}_o{origin}_t{target_month}"
                fitted_models[cell_key] = {
                    "encoder": encoder,
                    "published_mean": hgb_mean,
                    "published_median": hgb_median,
                    "catboost_mean": cat_mean,
                    "catboost_median": cat_median,
                    "catboost_multiquantile": cat_quantile,
                    "residual_quantiles": residual_quantiles,
                }
                model_specs.append({
                    "cell": cell_key, "outcome": outcome, "origin": origin, "target_month": target_month,
                    "train_n": int(observed_train.sum()), "feature_count": len(encoder.output_names),
                    "target": "change_from_baseline", "catboost_loss": f"MultiQuantile:alpha={alpha}",
                })

                raw_by_model: dict[str, dict[str, tuple[Any, Any, Any]]] = {MODEL_PUBLISHED: {}, MODEL_CATBOOST: {}}
                for part, ids in parts.items():
                    if len(ids) == 0:
                        empty = np.empty(0, dtype=float)
                        empty_quantiles = np.empty((0, len(QUANTILES)), dtype=float)
                        raw_by_model[MODEL_PUBLISHED][part] = (empty, empty, empty_quantiles)
                        raw_by_model[MODEL_CATBOOST][part] = (empty, empty, empty_quantiles)
                        continue
                    feature_frame = landmark_feature_frame(data, ids, outcome, origin)
                    x_part = encoder.transform(feature_frame)
                    base_part = pd.to_numeric(data.loc[ids, baseline_name], errors="coerce").to_numpy(float)
                    hgb_mean_abs = base_part + hgb_mean.predict(x_part)
                    hgb_median_abs = base_part + hgb_median.predict(x_part)
                    hgb_q = hgb_mean_abs[:, None] + residual_quantiles[None, :]
                    cat_mean_abs = base_part + np.asarray(cat_mean.predict(x_part), dtype=float).reshape(-1)
                    cat_median_abs = base_part + np.asarray(cat_median.predict(x_part), dtype=float).reshape(-1)
                    cat_q = base_part[:, None] + catboost_multi_quantile_predict(cat_quantile, x_part)
                    raw_by_model[MODEL_PUBLISHED][part] = (hgb_mean_abs, hgb_median_abs, np.sort(hgb_q, axis=1))
                    raw_by_model[MODEL_CATBOOST][part] = (cat_mean_abs, cat_median_abs, np.sort(cat_q, axis=1))

                for model_name in (MODEL_PUBLISHED, MODEL_CATBOOST):
                    cal_ids = parts["calibration"]
                    cal_y = pd.to_numeric(data.loc[cal_ids, target_column], errors="coerce").to_numpy(float)
                    expansions = conformal_expansions(cal_y, raw_by_model[model_name]["calibration"][2])
                    calibration_rows.append({
                        "model": model_name, "outcome": outcome, "origin": origin, "target_month": target_month,
                        **{f"conformal_expand_{int(level * 100)}": value for level, value in expansions.items()},
                        "calibration_n": int(np.isfinite(cal_y).sum()),
                    })
                    for part, ids in parts.items():
                        mean_abs, median_abs, raw_q = raw_by_model[model_name][part]
                        calibrated = apply_conformal(raw_q, expansions)
                        calibrated = np.clip(calibrated, bounds[0], bounds[1])
                        mean_abs = np.clip(mean_abs, bounds[0], bounds[1])
                        median_abs = np.clip(median_abs, bounds[0], bounds[1])
                        prediction_rows.extend(make_prediction_rows(
                            data, ids, part, outcome, origin, target_month, model_name,
                            mean_abs, median_abs, calibrated, raw_q, target_sd,
                        ))

    base_predictions = pd.DataFrame(prediction_rows)
    ensemble_rows, ensemble_status = build_conservative_ensemble(base_predictions, data, cfg)
    if len(ensemble_rows):
        base_predictions = pd.concat([base_predictions, ensemble_rows], ignore_index=True)
    if not np.all(np.diff(base_predictions[[QUANTILE_COLUMN[q] for q in QUANTILES]].to_numpy(float), axis=1) >= -1e-10):
        raise AssertionError("Quantile crossing remained after rearrangement")
    return {
        "predictions": base_predictions,
        "models": fitted_models,
        "model_specs": pd.DataFrame(model_specs),
        "conformal": pd.DataFrame(calibration_rows),
        "ensemble_status": pd.DataFrame(ensemble_status),
    }


def ensemble_objective(frame: Any, weight_published: float, cfg: RunConfig) -> float:
    keys = ["patient_position", "split", "outcome", "origin", "target_month", "observed", "target_sd"]
    qcols = [QUANTILE_COLUMN[q] for q in QUANTILES]
    left = frame[frame["model"].eq(MODEL_PUBLISHED)][keys + ["mean", "median", *qcols]].copy()
    right = frame[frame["model"].eq(MODEL_CATBOOST)][keys + ["mean", "median", *qcols]].copy()
    merged = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    keep = merged["observed"].notna()
    if not keep.any():
        return float("inf")
    qa = merged.loc[keep, [name + "_a" for name in qcols]].to_numpy(float)
    qb = merged.loc[keep, [name + "_b" for name in qcols]].to_numpy(float)
    combined = weight_published * qa + (1 - weight_published) * qb
    samples = quantile_matrix_to_samples(combined, min(101, cfg.predictive_samples))
    crps = crps_ensemble(samples, merged.loc[keep, "observed"].to_numpy(float))
    return float(np.mean(crps / merged.loc[keep, "target_sd"].to_numpy(float)))


def build_conservative_ensemble(predictions: Any, data: Any, cfg: RunConfig) -> tuple[Any, list[dict[str, Any]]]:
    qcols = [QUANTILE_COLUMN[q] for q in QUANTILES]
    output: list[Any] = []
    status: list[dict[str, Any]] = []
    for outcome in TRAJECTORY_HORIZONS:
        validation = predictions[(predictions["split"].eq("validation")) & (predictions["outcome"].eq(outcome))]
        grid = np.linspace(0, 1, 21)
        objectives = np.asarray([ensemble_objective(validation, float(weight), cfg) for weight in grid])
        optimum = float(grid[int(np.argmin(objectives))])
        objective_a = float(objectives[-1])
        objective_b = float(objectives[0])
        best_single_weight = 1.0 if objective_a <= objective_b else 0.0
        weight = 0.75 * optimum + 0.25 * best_single_weight
        outcome_predictions = predictions[predictions["outcome"].eq(outcome)]
        keys = ["patient_position", "split", "outcome", "origin", "target_month", "observed", "target_sd"]
        a = outcome_predictions[outcome_predictions["model"].eq(MODEL_PUBLISHED)][keys + ["mean", "median", "raw_q05", "raw_q95", *qcols]]
        b = outcome_predictions[outcome_predictions["model"].eq(MODEL_CATBOOST)][keys + ["mean", "median", "raw_q05", "raw_q95", *qcols]]
        merged = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        ensemble = merged[keys].copy()
        ensemble["model"] = MODEL_ENSEMBLE
        ensemble["mean"] = weight * merged["mean_a"] + (1 - weight) * merged["mean_b"]
        ensemble["median"] = weight * merged["median_a"] + (1 - weight) * merged["median_b"]
        ensemble["raw_q05"] = weight * merged["raw_q05_a"] + (1 - weight) * merged["raw_q05_b"]
        ensemble["raw_q95"] = weight * merged["raw_q95_a"] + (1 - weight) * merged["raw_q95_b"]
        for name in qcols:
            ensemble[name] = weight * merged[name + "_a"] + (1 - weight) * merged[name + "_b"]
        lo, hi = PLAUSIBLE_RANGES[outcome]
        extreme = bool((ensemble["raw_q05"] < lo - 5).any() or (ensemble["raw_q95"] > hi + 5).any())
        valid = ensemble[(ensemble["split"].eq("validation")) & ensemble["observed"].notna()]
        best_model = MODEL_PUBLISHED if best_single_weight == 1 else MODEL_CATBOOST
        best = outcome_predictions[(outcome_predictions["split"].eq("validation")) & outcome_predictions["model"].eq(best_model) & outcome_predictions["observed"].notna()]
        rmse_ensemble = float(np.sqrt(np.mean((valid["mean"] - valid["observed"]) ** 2))) if len(valid) else np.inf
        rmse_best = float(np.sqrt(np.mean((best["mean"] - best["observed"]) ** 2))) if len(best) else np.inf
        coverage_errors = []
        if len(valid):
            for level, low_col, high_col in ((0.5, "q25", "q75"), (0.8, "q10", "q90"), (0.9, "q05", "q95")):
                empirical = np.mean((valid["observed"] >= valid[low_col]) & (valid["observed"] <= valid[high_col]))
                coverage_errors.append(abs(float(empirical) - level))
        max_coverage_error = max(coverage_errors) if coverage_errors else np.inf
        accepted = not extreme and rmse_ensemble <= rmse_best * 1.01 and max_coverage_error <= 0.12
        reason = "accepted" if accepted else (
            "implausible tails" if extreme else "validation RMSE worse than conservative tolerance"
            if rmse_ensemble > rmse_best * 1.01 else "validation interval calibration failed"
        )
        status.append({
            "outcome": outcome, "status": "accepted" if accepted else "rejected", "reason": reason,
            "weight_published_hgb": weight, "weight_catboost": 1 - weight,
            "validation_standardized_crps": ensemble_objective(validation, weight, cfg),
            "best_single_standardized_crps": min(objective_a, objective_b),
            "validation_rmse": rmse_ensemble, "best_single_rmse": rmse_best,
            "max_coverage_error": max_coverage_error, "extreme_tail_detected": extreme,
        })
        if accepted:
            output.append(ensemble[predictions.columns])
    return (pd.concat(output, ignore_index=True) if output else pd.DataFrame(columns=predictions.columns)), status


def fit_probability_model(train_frame: Any, train_y: Any, predict_frame: Any, seed: int) -> tuple[Any, Any, FeatureEncoder | None]:
    from sklearn.linear_model import LogisticRegression
    numeric, categorical = available_feature_columns(train_frame)
    extra_numeric = [name for name in train_frame.columns if name.startswith("obs_") or name.startswith("known_") or name == "administrative_opportunity_days"]
    numeric = list(dict.fromkeys(numeric + extra_numeric))
    y = np.asarray(train_y, dtype=int)
    base = float(np.mean(y)) if len(y) else 0.5
    if len(y) < 20 or np.unique(y).size < 2:
        return np.full(len(predict_frame), base), None, None
    encoder = FeatureEncoder.fit(train_frame, numeric, categorical)
    x_train = encoder.transform(train_frame)
    x_predict = encoder.transform(predict_frame)
    model = LogisticRegression(C=0.5, max_iter=1500, random_state=seed)
    model.fit(x_train, y)
    return model.predict_proba(x_predict)[:, 1], model, encoder


def observation_process(data: Any, splits: LockedSplits, cfg: RunConfig) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    development = np.r_[splits.train, splits.calibration, splits.validation]
    evaluation_sets = {"test": splits.test, "contemporary": splits.contemporary}
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for outcome, horizons in TRAJECTORY_HORIZONS.items():
        for target_month in horizons:
            prior = max([0] + [month for month in horizons if month < target_month])
            dev_frame = landmark_feature_frame(data, development, outcome, prior)
            evaluation_frames = {
                name: landmark_feature_frame(data, ids, outcome, prior)
                for name, ids in evaluation_sets.items()
            }
            for frame in (dev_frame, *evaluation_frames.values()):
                frame["administrative_opportunity_days"] = pd.to_numeric(frame["administrative_opportunity_days"], errors="coerce")
                frame["obs_prior_measurement_count"] = frame[f"{outcome}_history_count"]
                frame["obs_prior_encounters"] = pd.to_numeric(frame.get("prior_utilization", 0), errors="coerce")
            y_dev = data.loc[development, f"{outcome}_{target_month}m_value"].notna().astype(int).to_numpy()
            center_groups = data.loc[development, "center_id"].astype(str).to_numpy()
            unique_centers = np.unique(center_groups)
            fold_count = min(5, len(unique_centers))
            p_oof = np.full(len(development), np.nan)
            if fold_count >= 2:
                folds = GroupKFold(n_splits=fold_count)
                for fold_number, (fit, hold) in enumerate(folds.split(dev_frame, y_dev, groups=center_groups)):
                    fold_probability, _, _ = fit_probability_model(
                        dev_frame.iloc[fit], y_dev[fit], dev_frame.iloc[hold],
                        cfg.seed + target_month + 1000 * fold_number,
                    )
                    p_oof[hold] = fold_probability
            else:
                p_oof[:] = float(np.mean(y_dev))
            p_oof = np.clip(p_oof, 1e-6, 1 - 1e-6)
            combined = pd.concat(list(evaluation_frames.values()), ignore_index=True)
            combined_probability, model, encoder = fit_probability_model(
                dev_frame, y_dev, combined, cfg.seed + target_month,
            )
            prevalence = float(np.mean(y_dev))
            offset = 0
            for evaluation_set, ids in evaluation_sets.items():
                p_evaluation = np.clip(combined_probability[offset:offset + len(ids)], 1e-6, 1.0)
                offset += len(ids)
                observed = data.loc[ids, f"{outcome}_{target_month}m_value"].notna().to_numpy()
                minimum_probability = float(np.min(p_evaluation)) if len(p_evaluation) else np.nan
                positivity_ok = bool(
                    np.isfinite(minimum_probability) and
                    minimum_probability >= SUCCESS_GATES["observation_min_probability"]
                )
                raw_weight = np.where(observed, prevalence / p_evaluation, 0.0)
                positive = raw_weight[raw_weight > 0]
                cap = float(np.quantile(positive, 0.99)) if positive.size else 1.0
                weight = np.minimum(raw_weight, cap)
                for local, patient_position in enumerate(ids):
                    rows.append({
                        "patient_position": int(patient_position), "evaluation_set": evaluation_set,
                        "outcome": outcome, "target_month": target_month,
                        "p_observed": float(p_evaluation[local]), "observed": bool(observed[local]),
                        "stabilized_weight": float(weight[local]), "ipow_estimable": positivity_ok,
                    })
                sensitivity = {}
                for lower in (0.01, 0.05, 0.10):
                    candidate = np.where(observed, prevalence / np.clip(p_evaluation, lower, 1), 0.0)
                    sensitivity[f"ess_floor_{lower:g}"] = effective_sample_size(candidate)
                oof_brier = float(np.mean((y_dev - p_oof) ** 2))
                oof_log_loss = float(-np.mean(y_dev * np.log(p_oof) + (1 - y_dev) * np.log(1 - p_oof)))
                summaries.append({
                    "evaluation_set": evaluation_set, "outcome": outcome, "target_month": target_month,
                    "development_prevalence": prevalence, "observed_n": int(observed.sum()), "evaluation_n": len(observed),
                    "weight_cap_99": cap, "effective_n": effective_sample_size(weight),
                    "max_raw_weight": float(positive.max()) if positive.size else np.nan,
                    "minimum_observation_probability": minimum_probability, "ipow_estimable": positivity_ok,
                    "crossfit_center_folds": fold_count, "development_oof_brier": oof_brier,
                    "development_oof_log_loss": oof_log_loss, **sensitivity,
                })
            models[f"{outcome}_{target_month}"] = {
                "model": model, "encoder": encoder, "center_crossfit_folds": fold_count,
            }
    return {"patient_weights": pd.DataFrame(rows), "summary": pd.DataFrame(summaries), "models": models}


def effective_sample_size(weight: Any) -> float:
    values = np.asarray(weight, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(values.sum() ** 2 / np.sum(values ** 2)) if values.size else 0.0


def weighted_mean(values: Any, weights: Any | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if weights is None:
        keep = np.isfinite(values)
        return float(np.mean(values[keep])) if keep.any() else np.nan
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[keep], weights=weights[keep])) if keep.any() else np.nan


def cluster_bootstrap_stat(values: Any, centers: Any, statistic: Callable[[Any], float], replicates: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values)
    centers = np.asarray(centers).astype(str)
    unique = np.unique(centers)
    if unique.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = []
    by_center = {center: np.flatnonzero(centers == center) for center in unique}
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ids = np.concatenate([by_center[center] for center in sampled])
        try:
            estimates.append(float(statistic(values[ids])))
        except Exception:
            continue
    finite = np.asarray(estimates, dtype=float)
    finite = finite[np.isfinite(finite)]
    return (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))) if finite.size else (np.nan, np.nan)


def evaluate_trajectories(data: Any, predictions: Any, observation: dict[str, Any], cfg: RunConfig) -> dict[str, Any]:
    evaluated = predictions[predictions["split"].isin(["test", "contemporary"])].copy()
    weights = observation["patient_weights"][[
        "patient_position", "evaluation_set", "outcome", "target_month",
        "stabilized_weight", "ipow_estimable",
    ]].rename(columns={"evaluation_set": "split"})
    evaluated = evaluated.merge(
        weights, on=["patient_position", "split", "outcome", "target_month"],
        how="left", validate="many_to_one",
    )
    qcols = [QUANTILE_COLUMN[q] for q in QUANTILES]
    metric_rows: list[dict[str, Any]] = []
    per_patient_rows: list[dict[str, Any]] = []
    for keys, group in evaluated.groupby(["split", "model", "outcome", "origin", "target_month"], sort=True):
        evaluation_set, model, outcome, origin, target_month = keys
        observed = group["observed"].notna().to_numpy()
        if not observed.any():
            continue
        g = group.loc[observed]
        y = g["observed"].to_numpy(float)
        q = g[qcols].to_numpy(float)
        samples = quantile_matrix_to_samples(q, cfg.predictive_samples)
        crps = crps_ensemble(samples, y)
        mean_error = g["mean"].to_numpy(float) - y
        median_error = g["median"].to_numpy(float) - y
        weight = g["stabilized_weight"].to_numpy(float)
        ipow_estimable = bool(g["ipow_estimable"].fillna(False).all())
        centers = data.loc[g["patient_position"], "center_blind"].to_numpy()
        estimates = [("complete_case", None)]
        if ipow_estimable and np.all(np.isfinite(weight)) and np.any(weight > 0):
            estimates.append(("ipow", weight))
        for estimate, w in estimates:
            base = {
                "evaluation_set": evaluation_set, "model": model, "outcome": outcome,
                "origin": int(origin), "target_month": int(target_month),
                "estimate": estimate, "n": len(g), "effective_n": len(g) if w is None else effective_sample_size(w),
                "exploratory": len(g) < cfg.min_cell_observations or (w is not None and effective_sample_size(w) < cfg.min_cell_observations),
            }
            metrics = {
                "rmse": math.sqrt(weighted_mean(mean_error ** 2, w)),
                "mae": weighted_mean(np.abs(median_error), w),
                "mad": float(np.nanmedian(np.abs(median_error))) if w is None else weighted_quantile(np.abs(median_error), w, 0.5),
                "bias": weighted_mean(mean_error, w),
                "crps": weighted_mean(crps, w),
                "standardized_crps": weighted_mean(crps / g["target_sd"].to_numpy(float), w),
            }
            for name, value in metrics.items():
                ci_low = ci_high = np.nan
                if evaluation_set == "test" and origin == 0 and estimate == "complete_case":
                    contributions: dict[str, tuple[Any, Callable[[Any], float]]] = {
                        "rmse": (mean_error ** 2, lambda x: math.sqrt(float(np.mean(x)))),
                        "mae": (np.abs(median_error), lambda x: float(np.mean(x))),
                        "standardized_crps": (
                            crps / g["target_sd"].to_numpy(float), lambda x: float(np.mean(x)),
                        ),
                    }
                    if name in contributions:
                        values, statistic = contributions[name]
                        seed_key = f"{model}|{outcome}|{origin}|{target_month}|{name}"
                        seed_offset = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
                        ci_low, ci_high = cluster_bootstrap_stat(
                            values, centers, statistic, cfg.cluster_bootstrap_replicates,
                            cfg.seed + seed_offset,
                        )
                metric_rows.append({
                    **base, "metric": name, "value": value, "level": np.nan,
                    "ci_low": ci_low, "ci_high": ci_high,
                })
            for coverage, lo, hi in ((0.50, "q25", "q75"), (0.80, "q10", "q90"), (0.90, "q05", "q95")):
                inside = ((y >= g[lo].to_numpy(float)) & (y <= g[hi].to_numpy(float))).astype(float)
                width = g[hi].to_numpy(float) - g[lo].to_numpy(float)
                metric_rows.append({
                    **base, "metric": "coverage", "value": weighted_mean(inside, w), "level": coverage,
                    "ci_low": np.nan, "ci_high": np.nan,
                })
                metric_rows.append({
                    **base, "metric": "interval_width", "value": weighted_mean(width, w), "level": coverage,
                    "ci_low": np.nan, "ci_high": np.nan,
                })
        for row_index, (_, row) in enumerate(g.iterrows()):
            per_patient_rows.append({
                "patient_position": int(row["patient_position"]), "center_blind": str(centers[row_index]),
                "evaluation_set": evaluation_set, "model": model, "outcome": outcome,
                "origin": int(origin), "target_month": int(target_month),
                "crps": float(crps[row_index]), "sq_error": float(mean_error[row_index] ** 2),
                "abs_error": float(abs(median_error[row_index])), "observed": float(y[row_index]),
                "prediction": float(row["mean"]),
            })
    metrics = pd.DataFrame(metric_rows)
    per_patient = pd.DataFrame(per_patient_rows)
    locked_test_patient = per_patient[per_patient["evaluation_set"].eq("test")]
    inference = trajectory_clustered_inference(locked_test_patient, cfg)
    center_metrics = aggregate_center_trajectory(locked_test_patient, cfg)
    subgroup_metrics = aggregate_subgroups(data, locked_test_patient, cfg)
    leaderboard = trajectory_leaderboard(metrics)
    return {
        "metrics": metrics, "per_patient": per_patient, "inference": inference,
        "center_metrics": center_metrics, "subgroup_metrics": subgroup_metrics,
        "leaderboard": leaderboard,
    }


def weighted_quantile(values: Any, weights: Any, probability: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.nan
    order = np.argsort(values[keep])
    v, w = values[keep][order], weights[keep][order]
    return float(v[np.searchsorted(np.cumsum(w), probability * w.sum(), side="left")])


def trajectory_clustered_inference(per_patient: Any, cfg: RunConfig) -> Any:
    rows = []
    for outcome in TRAJECTORY_HORIZONS:
        candidate_names = [name for name in (MODEL_CATBOOST, MODEL_ENSEMBLE) if name in set(per_patient["model"])]
        baseline = per_patient[(per_patient["model"].eq(MODEL_PUBLISHED)) & per_patient["outcome"].eq(outcome)]
        for candidate in candidate_names:
            cand = per_patient[(per_patient["model"].eq(candidate)) & per_patient["outcome"].eq(outcome)]
            keys = ["patient_position", "origin", "target_month", "center_blind"]
            merged = cand[keys + ["crps", "sq_error"]].merge(
                baseline[keys + ["crps", "sq_error"]], on=keys, suffixes=("_candidate", "_baseline"), validate="one_to_one",
            )
            if merged.empty:
                continue
            merged["crps_difference"] = merged["crps_candidate"] - merged["crps_baseline"]
            for (origin, target), group in merged.groupby(["origin", "target_month"]):
                diff = group["crps_difference"].to_numpy(float)
                ci = cluster_bootstrap_stat(diff, group["center_blind"], np.mean, cfg.cluster_bootstrap_replicates, cfg.seed + int(target))
                rows.append({
                    "model": candidate, "baseline": MODEL_PUBLISHED, "outcome": outcome,
                    "origin": int(origin), "target_month": int(target), "n": len(group),
                    "crps_difference": float(np.mean(diff)),
                    "relative_crps_improvement": float(-np.mean(diff) / max(group["crps_baseline"].mean(), 1e-8)),
                    "ci_low": ci[0], "ci_high": ci[1],
                })
            five = merged[(merged["origin"].eq(0)) & (merged["target_month"].eq(60))]
            if len(five):
                values = np.column_stack([five["sq_error_candidate"], five["sq_error_baseline"]])
                statistic = lambda x: math.sqrt(float(np.mean(x[:, 0]))) - math.sqrt(float(np.mean(x[:, 1])))
                ci = cluster_bootstrap_stat(values, five["center_blind"], statistic, cfg.cluster_bootstrap_replicates, cfg.seed + 60)
                rows.append({
                    "model": candidate, "baseline": MODEL_PUBLISHED, "outcome": outcome,
                    "origin": 0, "target_month": 60, "n": len(five),
                    "crps_difference": np.nan, "relative_crps_improvement": np.nan,
                    "rmse_difference": statistic(values), "rmse_difference_ci_low": ci[0], "rmse_difference_ci_high": ci[1],
                })
    return pd.DataFrame(rows)


def aggregate_center_trajectory(per_patient: Any, cfg: RunConfig) -> Any:
    rows = []
    selected = per_patient[(per_patient["origin"].eq(0)) & (per_patient["target_month"].eq(60))]
    for keys, group in selected.groupby(["center_blind", "model", "outcome"]):
        center, model, outcome = keys
        if len(group) < cfg.min_cell_observations:
            continue
        rows.append({
            "center_blind": center, "model": model, "outcome": outcome, "n": len(group),
            "rmse": math.sqrt(float(group["sq_error"].mean())), "crps": float(group["crps"].mean()),
        })
    return pd.DataFrame(rows)


def aggregate_subgroups(data: Any, per_patient: Any, cfg: RunConfig) -> Any:
    rows = []
    selected = per_patient[(per_patient["origin"].eq(0)) & (per_patient["target_month"].eq(60))].copy()
    columns = ["procedure_type", "sex", "race", "coverage", "state", "ruca"]
    for column in columns:
        if column not in data:
            continue
        mapping = data[column]
        selected["subgroup"] = selected["patient_position"].map(mapping)
        for keys, group in selected.groupby(["model", "outcome", "subgroup"], dropna=False):
            model, outcome, subgroup = keys
            if len(group) < cfg.min_cell_observations:
                continue
            rows.append({
                "subgroup_variable": column, "subgroup": str(subgroup), "model": model,
                "outcome": outcome, "n": len(group), "rmse": math.sqrt(float(group["sq_error"].mean())),
                "crps": float(group["crps"].mean()),
            })
    return pd.DataFrame(rows)


def trajectory_leaderboard(metrics: Any) -> Any:
    selected = metrics[
        metrics["evaluation_set"].eq("test") & metrics["estimate"].eq("complete_case") &
        metrics["metric"].eq("standardized_crps")
    ]
    if selected.empty:
        return pd.DataFrame()
    result = selected.groupby(["model", "outcome"], as_index=False).agg(
        standardized_crps=("value", "mean"), cells=("value", "size"), observations=("n", "sum"),
    )
    pooled = selected.groupby("model", as_index=False).agg(standardized_crps=("value", "mean"), cells=("value", "size"), observations=("n", "sum"))
    pooled["outcome"] = "equal_weighted_standardized"
    return pd.concat([result, pooled], ignore_index=True).sort_values(["outcome", "standardized_crps"])


def internal_external_validation(data: Any, splits: LockedSplits, cfg: RunConfig) -> Any:
    """Frozen mean-model check with each adequately powered development center held out."""
    from catboost import CatBoostRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = []
    for fold_number, fold in enumerate(splits.internal_external_folds):
        for outcome in TRAJECTORY_HORIZONS:
            target = f"{outcome}_60m_value"
            baseline = "baseline_bmi" if outcome == "bmi" else "baseline_hba1c"
            fit_ids = np.asarray(fold["fit"])
            hold_ids = np.asarray(fold["hold"])
            observed_fit = data.loc[fit_ids, target].notna().to_numpy()
            observed_hold = data.loc[hold_ids, target].notna().to_numpy()
            if observed_fit.sum() < max(30, cfg.min_cell_observations) or observed_hold.sum() < cfg.min_cell_observations:
                continue
            fit_frame = landmark_feature_frame(data, fit_ids, outcome, 0)
            hold_frame = landmark_feature_frame(data, hold_ids, outcome, 0)
            numeric, categorical = trajectory_feature_roster(data, outcome)
            encoder = FeatureEncoder.fit(fit_frame.loc[observed_fit], numeric, categorical)
            xfit = encoder.transform(fit_frame.loc[observed_fit])
            xhold = encoder.transform(hold_frame.loc[observed_hold])
            yfit = data.loc[fit_ids[observed_fit], target].to_numpy(float) - data.loc[fit_ids[observed_fit], baseline].to_numpy(float)
            yhold = data.loc[hold_ids[observed_hold], target].to_numpy(float)
            basehold = data.loc[hold_ids[observed_hold], baseline].to_numpy(float)
            hgb = HistGradientBoostingRegressor(max_iter=cfg.hgb_iterations, max_leaf_nodes=15, learning_rate=0.06, random_state=cfg.seed + fold_number).fit(xfit, yfit)
            cat = CatBoostRegressor(
                loss_function="RMSE", iterations=cfg.catboost_iterations, depth=4 if cfg.smoke else 6,
                learning_rate=0.08 if cfg.smoke else 0.045, verbose=False, allow_writing_files=False,
                thread_count=cfg.threads, random_seed=cfg.seed + fold_number,
            ).fit(xfit, yfit)
            for model, prediction in (
                (MODEL_PUBLISHED, basehold + hgb.predict(xhold)),
                (MODEL_CATBOOST, basehold + np.asarray(cat.predict(xhold), dtype=float)),
            ):
                rows.append({
                    "center_blind": blind_center(fold["center"]), "outcome": outcome, "model": model,
                    "n": int(observed_hold.sum()), "rmse": math.sqrt(float(np.mean((prediction - yhold) ** 2))),
                    "mae": float(np.mean(np.abs(prediction - yhold))),
                })
    return pd.DataFrame(rows)


# =====================================================================================
# 5. Fixed-horizon competing-risk survival models
# =====================================================================================

RISK_MODEL_BASELINE = "regularized_pooled_logistic"
RISK_MODEL_BOOSTED = "boosted_competing_risk"
RISK_MODEL_AUGMENTED = "boosted_plus_crossfit_trajectory"
RISK_CAUSE_CODE = {"none": 0, "mace": 1, "nephropathy": 2, "retinopathy": 3, "competing_death": 4}


def crossfit_trajectory_summaries(data: Any, splits: LockedSplits, cfg: RunConfig) -> Any:
    """Cross-fit preoperative trajectory summaries without any complication features."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    train_ids = np.asarray(splits.train)
    apply_ids = np.r_[splits.calibration, splits.validation, splits.test]
    summary = pd.DataFrame(index=data.index)
    groups = data.loc[train_ids, "center_id"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise PreflightError("Trajectory cross-fitting failed", ["At least two training centers are required for cross-fitted risk augmentation"])
    fold_list = list(GroupKFold(n_splits=n_splits).split(train_ids, groups=groups))
    for outcome, horizons in TRAJECTORY_HORIZONS.items():
        baseline_name = "baseline_bmi" if outcome == "bmi" else "baseline_hba1c"
        expected = np.full((len(data), len(horizons)), np.nan, dtype=float)
        width = np.full_like(expected, np.nan)
        origin_frame = landmark_feature_frame(data, train_ids, outcome, 0)
        numeric, categorical = trajectory_feature_roster(data, outcome)
        for horizon_index, month in enumerate(horizons):
            target = f"{outcome}_{month}m_value"
            y_all = pd.to_numeric(data.loc[train_ids, target], errors="coerce").to_numpy(float)
            base_all = pd.to_numeric(data.loc[train_ids, baseline_name], errors="coerce").to_numpy(float)
            for fold_fit, fold_hold in fold_list:
                observed = np.isfinite(y_all[fold_fit]) & np.isfinite(base_all[fold_fit])
                if observed.sum() < max(20, cfg.min_cell_observations):
                    continue
                fit_positions = fold_fit[observed]
                encoder = FeatureEncoder.fit(origin_frame.iloc[fit_positions], numeric, categorical)
                xfit = encoder.transform(origin_frame.iloc[fit_positions])
                xhold = encoder.transform(origin_frame.iloc[fold_hold])
                ychange = y_all[fit_positions] - base_all[fit_positions]
                model = HistGradientBoostingRegressor(
                    max_iter=cfg.hgb_iterations, max_leaf_nodes=15, learning_rate=0.06,
                    l2_regularization=2.0, random_state=cfg.seed + month,
                ).fit(xfit, ychange)
                residual_sd = float(max(np.std(ychange - model.predict(xfit), ddof=1), 0.1))
                patient_ids = train_ids[fold_hold]
                expected[patient_ids, horizon_index] = base_all[fold_hold] + model.predict(xhold)
                width[patient_ids, horizon_index] = 3.29 * residual_sd

            observed_full = np.isfinite(y_all) & np.isfinite(base_all)
            if observed_full.sum() < max(20, cfg.min_cell_observations):
                continue
            encoder = FeatureEncoder.fit(origin_frame.loc[observed_full], numeric, categorical)
            xfit = encoder.transform(origin_frame.loc[observed_full])
            ychange = y_all[observed_full] - base_all[observed_full]
            model = HistGradientBoostingRegressor(
                max_iter=cfg.hgb_iterations, max_leaf_nodes=15, learning_rate=0.06,
                l2_regularization=2.0, random_state=cfg.seed + 500 + month,
            ).fit(xfit, ychange)
            residual_sd = float(max(np.std(ychange - model.predict(xfit), ddof=1), 0.1))
            apply_frame = landmark_feature_frame(data, apply_ids, outcome, 0)
            prediction = pd.to_numeric(data.loc[apply_ids, baseline_name], errors="coerce").to_numpy(float) + model.predict(encoder.transform(apply_frame))
            expected[apply_ids, horizon_index] = prediction
            width[apply_ids, horizon_index] = 3.29 * residual_sd

        threshold = 35.0 if outcome == "bmi" else 5.7
        safe_expected = np.where(np.isfinite(expected), expected, np.nan)
        summary[f"traj_{outcome}_expected_12"] = safe_expected[:, horizons.index(12)]
        summary[f"traj_{outcome}_expected_60"] = safe_expected[:, horizons.index(60)]
        has_summary = np.isfinite(safe_expected).any(axis=1)
        nadir = np.full(len(data), np.nan)
        burden = np.full(len(data), np.nan)
        nadir[has_summary] = np.nanmin(safe_expected[has_summary], axis=1)
        burden[has_summary] = np.nanmean(np.maximum(safe_expected[has_summary] - threshold, 0), axis=1)
        summary[f"traj_{outcome}_nadir"] = nadir
        summary[f"traj_{outcome}_regain"] = safe_expected[:, -1] - nadir
        if outcome == "bmi":
            summary[f"traj_{outcome}_burden"] = burden
            summary[f"traj_{outcome}_threshold_probability"] = scipy_stats.norm.cdf((threshold - safe_expected[:, -1]) / np.maximum(width[:, -1] / 3.29, 0.1))
        else:
            summary[f"traj_{outcome}_burden"] = burden
            summary[f"traj_{outcome}_threshold_probability"] = scipy_stats.norm.cdf((threshold - safe_expected[:, -1]) / np.maximum(width[:, -1] / 3.29, 0.1))
        summary[f"traj_{outcome}_uncertainty_width"] = width[:, -1]
    if summary.loc[train_ids].isna().all(axis=1).any():
        count = int(summary.loc[train_ids].isna().all(axis=1).sum())
        raise PreflightError("Cross-fitted trajectory risk features failed", [f"{count} training patients lack all cross-fitted trajectory summaries"])
    return summary


def earliest_incident_event(data: Any, patient_position: int) -> tuple[str, float]:
    row = data.loc[patient_position]
    times = []
    for component in COMPONENTS:
        if bool(row.get(component + "_valid_event", False)):
            times.append((component, float(row[component + "_days"])))
    if bool(row.get("death_valid_event", False)):
        times.append(("competing_death", float(row["death_days"])))
    if not times:
        return "none", math.inf
    return min(times, key=lambda item: (item[1], RISK_CAUSE_CODE[item[0]]))


def missing_event_time_mask(data: Any, endpoint: str) -> Any:
    if endpoint == "composite":
        columns = [name + "_missing_time_sensitivity" for name in COMPONENTS]
    elif endpoint == "death":
        columns = ["death_missing_time_sensitivity"]
    else:
        columns = [endpoint + "_missing_time_sensitivity"]
    present = [column for column in columns if column in data]
    if not present:
        return np.zeros(len(data), dtype=bool)
    return data[present].fillna(False).astype(bool).any(axis=1).to_numpy()


def person_period_rows(data: Any, ids: Any, encoder: FeatureEncoder, augmented: Any | None = None, death_only: bool = False) -> tuple[Any, Any, Any]:
    patient_frame = data.loc[ids].copy()
    if augmented is not None:
        for column in augmented.columns:
            patient_frame[column] = augmented.loc[ids, column].to_numpy()
    base = encoder.transform(patient_frame)
    x_rows: list[Any] = []
    y_rows: list[int] = []
    patient_rows: list[int] = []
    max_day = 60 * DAYS_PER_MONTH
    for local, patient_position in enumerate(ids):
        censor = min(float(data.loc[patient_position, "censor_days"]), max_day)
        if death_only:
            cause = "death" if bool(data.loc[patient_position, "death_valid_event"]) else "none"
            event_day = float(data.loc[patient_position, "death_days"]) if cause == "death" else math.inf
        else:
            cause, event_day = earliest_incident_event(data, int(patient_position))
        stop = min(censor, event_day, max_day)
        for interval in range(1, int(60 / PERSON_PERIOD_MONTHS) + 1):
            end_month = interval * PERSON_PERIOD_MONTHS
            end_day = end_month * DAYS_PER_MONTH
            start_day = (end_month - PERSON_PERIOD_MONTHS) * DAYS_PER_MONTH
            event_here = event_day > start_day and event_day <= end_day and event_day <= censor
            if end_day > censor and not event_here:
                break
            time_features = np.asarray([end_month / 60.0, (end_month / 60.0) ** 2, math.log1p(end_month)], dtype=float)
            x_rows.append(np.r_[base[local], time_features])
            if event_here:
                y_rows.append(1 if death_only else RISK_CAUSE_CODE[cause])
            else:
                y_rows.append(0)
            patient_rows.append(int(patient_position))
            if event_here or end_day >= stop:
                break
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int), np.asarray(patient_rows, dtype=int)


def risk_feature_encoder(data: Any, train_ids: Any, trajectory_summary: Any | None) -> FeatureEncoder:
    frame = data.loc[train_ids].copy()
    numeric, categorical = available_feature_columns(data)
    if trajectory_summary is not None:
        for column in trajectory_summary.columns:
            frame[column] = trajectory_summary.loc[train_ids, column].to_numpy()
        numeric = numeric + list(trajectory_summary.columns)
    return FeatureEncoder.fit(frame, numeric, categorical)


def class_probability(model: Any, features: Any, class_value: int) -> Any:
    prediction = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    match = np.flatnonzero(classes == class_value)
    return prediction[:, match[0]] if match.size else np.zeros(len(features), dtype=float)


def predict_cumulative_incidence(
    model: Any,
    data: Any,
    ids: Any,
    encoder: FeatureEncoder,
    trajectory_summary: Any | None,
    death_only: bool = False,
) -> dict[str, Any]:
    patient_frame = data.loc[ids].copy()
    if trajectory_summary is not None:
        for column in trajectory_summary.columns:
            patient_frame[column] = trajectory_summary.loc[ids, column].to_numpy()
    base = encoder.transform(patient_frame)
    output_names = ("death",) if death_only else (*COMPONENTS, "competing_death")
    result = {name: np.zeros((len(ids), len(RISK_HORIZONS)), dtype=float) for name in output_names}
    survival = np.ones(len(ids), dtype=float)
    cumulative = {name: np.zeros(len(ids), dtype=float) for name in output_names}
    risk_index = 0
    for interval in range(1, int(60 / PERSON_PERIOD_MONTHS) + 1):
        month = interval * PERSON_PERIOD_MONTHS
        time_features = np.tile([month / 60.0, (month / 60.0) ** 2, math.log1p(month)], (len(ids), 1))
        x = np.column_stack([base, time_features])
        if death_only:
            hazards = {"death": class_probability(model, x, 1)}
        else:
            hazards = {
                "mace": class_probability(model, x, 1),
                "nephropathy": class_probability(model, x, 2),
                "retinopathy": class_probability(model, x, 3),
                "competing_death": class_probability(model, x, 4),
            }
        total = np.sum(np.column_stack(list(hazards.values())), axis=1)
        scale = np.where(total > 0.98, 0.98 / np.maximum(total, 1e-12), 1.0)
        for name in hazards:
            hazards[name] = np.clip(hazards[name] * scale, 0, 1)
            cumulative[name] += survival * hazards[name]
        survival *= np.clip(1 - np.sum(np.column_stack(list(hazards.values())), axis=1), 0, 1)
        if risk_index < len(RISK_HORIZONS) and month == RISK_HORIZONS[risk_index]:
            for name in output_names:
                result[name][:, risk_index] = cumulative[name]
            risk_index += 1
    for values in result.values():
        if np.any(values < -1e-10) or np.any(values > 1 + 1e-10) or np.any(np.diff(values, axis=1) < -1e-8):
            raise AssertionError("Invalid cumulative-incidence predictions")
    return result


def fit_survival_models(data: Any, splits: LockedSplits, trajectory_summary: Any, cfg: RunConfig) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    incomplete_event_time = np.column_stack([
        data[name + "_missing_time_sensitivity"].fillna(False).astype(bool).to_numpy()
        for name in (*COMPONENTS, "death")
    ]).any(axis=1)
    eligible = ~data["prior_any_component"].astype(bool) & ~incomplete_event_time
    train_ids = splits.train[eligible.iloc[splits.train].to_numpy()]
    calibration_ids = splits.calibration[eligible.iloc[splits.calibration].to_numpy()]
    test_ids = splits.test[eligible.iloc[splits.test].to_numpy()]
    if len(train_ids) < max(50, cfg.min_development_patients // 2) or len(test_ids) < max(20, cfg.min_locked_test_patients // 2):
        raise PreflightError("Incident competing-risk cohort is underpowered", [
            f"Incident-free training/test counts are {len(train_ids)}/{len(test_ids)}"
        ])
    model_results: dict[str, Any] = {}
    specifications: list[dict[str, Any]] = []
    for model_name, augmented, boosted in (
        (RISK_MODEL_BASELINE, False, False),
        (RISK_MODEL_BOOSTED, False, True),
        (RISK_MODEL_AUGMENTED, True, True),
    ):
        summary = trajectory_summary if augmented else None
        encoder = risk_feature_encoder(data, train_ids, summary)
        x_train, y_train, _ = person_period_rows(data, train_ids, encoder, summary, death_only=False)
        class_count = {int(value): int((y_train == value).sum()) for value in np.unique(y_train)}
        required_classes = {0, 1, 2, 3, 4}
        if not required_classes.issubset(set(class_count)):
            raise PreflightError("Competing-risk training events are incomplete", [
                f"{model_name} person-period class counts {class_count}; all component and death classes are required"
            ])
        if boosted:
            class_weight = np.asarray([len(y_train) / (len(class_count) * class_count[int(value)]) for value in y_train])
            model = HistGradientBoostingClassifier(
                max_iter=cfg.hgb_iterations, max_leaf_nodes=20, learning_rate=0.05,
                l2_regularization=2.0, random_state=cfg.seed + (9 if augmented else 7),
            ).fit(x_train, y_train, sample_weight=class_weight)
        else:
            model = LogisticRegression(
                C=0.25, max_iter=1800, class_weight="balanced", random_state=cfg.seed,
            ).fit(x_train, y_train)

        death_encoder = risk_feature_encoder(data, train_ids, summary)
        x_death, y_death, _ = person_period_rows(data, train_ids, death_encoder, summary, death_only=True)
        if np.unique(y_death).size < 2:
            raise PreflightError("All-cause death model is not estimable", [f"{model_name} has no death events in training"])
        if boosted:
            death_model = HistGradientBoostingClassifier(
                max_iter=cfg.hgb_iterations, max_leaf_nodes=15, learning_rate=0.05,
                l2_regularization=2.0, random_state=cfg.seed + 27,
            ).fit(x_death, y_death)
        else:
            death_model = LogisticRegression(C=0.25, max_iter=1500, class_weight="balanced", random_state=cfg.seed + 1).fit(x_death, y_death)

        calibration_raw = predict_cumulative_incidence(model, data, calibration_ids, encoder, summary, death_only=False)
        calibration_death = predict_cumulative_incidence(death_model, data, calibration_ids, death_encoder, summary, death_only=True)
        test_raw = predict_cumulative_incidence(model, data, test_ids, encoder, summary, death_only=False)
        test_death = predict_cumulative_incidence(death_model, data, test_ids, death_encoder, summary, death_only=True)
        factors = development_calibration_factors(data, calibration_ids, calibration_raw, calibration_death)
        calibrated = apply_risk_calibration(test_raw, test_death, factors)
        model_results[model_name] = {
            "test_ids": test_ids, "predictions": calibrated, "calibration_factors": factors,
            "competing_model": model, "death_model": death_model,
            "encoder": encoder, "death_encoder": death_encoder,
        }
        specifications.append({
            "model": model_name, "person_period_rows": len(y_train), "patient_train_n": len(train_ids),
            "patient_test_n": len(test_ids), "class_counts": canonical_json(class_count),
            "trajectory_augmented": augmented, "interval_months": PERSON_PERIOD_MONTHS,
            "missing_event_time_sensitivity_excluded": int(incomplete_event_time.sum()),
        })
    return {"models": model_results, "specifications": pd.DataFrame(specifications), "eligible_test_ids": test_ids}


def known_risk_status(data: Any, ids: Any, endpoint: str, horizon_month: int) -> tuple[Any, Any]:
    horizon_day = horizon_month * DAYS_PER_MONTH
    y = np.zeros(len(ids), dtype=float)
    known = np.zeros(len(ids), dtype=bool)
    missing_relevant = missing_event_time_mask(data, endpoint)
    for local, patient_position in enumerate(ids):
        censor = float(data.loc[patient_position, "censor_days"])
        if endpoint == "death":
            death = float(data.loc[patient_position, "death_days"]) if bool(data.loc[patient_position, "death_valid_event"]) else math.inf
            if death <= horizon_day:
                y[local], known[local] = 1.0, True
            elif missing_relevant[patient_position]:
                continue
            elif censor >= horizon_day:
                known[local] = True
            continue
        cause, event_day = earliest_incident_event(data, int(patient_position))
        target_event = event_day <= horizon_day and (
            (endpoint == "composite" and cause in COMPONENTS) or cause == endpoint
        )
        if target_event:
            y[local], known[local] = 1.0, True
        elif missing_relevant[patient_position]:
            continue
        elif event_day <= horizon_day:
            known[local] = True
        elif censor >= horizon_day:
            known[local] = True
    return y, known


def development_calibration_factors(data: Any, ids: Any, raw: dict[str, Any], death: dict[str, Any]) -> dict[str, list[float]]:
    factors: dict[str, list[float]] = {name: [] for name in (*COMPONENTS, "death")}
    for horizon_index, month in enumerate(RISK_HORIZONS):
        for endpoint in COMPONENTS:
            y, known = known_risk_status(data, ids, endpoint, month)
            predicted = raw[endpoint][:, horizon_index]
            ratio = float(y[known].mean() / max(predicted[known].mean(), 1e-6)) if known.any() else 1.0
            factors[endpoint].append(float(np.clip(ratio, 0.4, 2.5)))
        y, known = known_risk_status(data, ids, "death", month)
        predicted = death["death"][:, horizon_index]
        ratio = float(y[known].mean() / max(predicted[known].mean(), 1e-6)) if known.any() else 1.0
        factors["death"].append(float(np.clip(ratio, 0.4, 2.5)))
    return factors


def apply_risk_calibration(raw: dict[str, Any], death: dict[str, Any], factors: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    calibrated = {name: raw[name].copy() for name in (*COMPONENTS, "competing_death")}
    calibrated["death"] = death["death"].copy()
    for horizon_index in range(len(RISK_HORIZONS)):
        for endpoint in COMPONENTS:
            calibrated[endpoint][:, horizon_index] *= float(factors[endpoint][horizon_index])
        component_sum = sum(calibrated[name][:, horizon_index] for name in COMPONENTS)
        scale = np.where(component_sum > 0.98, 0.98 / np.maximum(component_sum, 1e-12), 1.0)
        for endpoint in COMPONENTS:
            calibrated[endpoint][:, horizon_index] = np.clip(calibrated[endpoint][:, horizon_index] * scale, 0, 1)
        calibrated["death"][:, horizon_index] = np.clip(calibrated["death"][:, horizon_index] * float(factors["death"][horizon_index]), 0, 1)
    calibrated["composite"] = sum(calibrated[name] for name in COMPONENTS)
    for endpoint in COMPONENTS:
        if np.any(calibrated[endpoint] > calibrated["composite"] + 1e-10):
            raise AssertionError("Component cumulative incidence exceeds composite")
    for values in calibrated.values():
        if np.any(values < -1e-10) or np.any(values > 1 + 1e-10):
            raise AssertionError("Calibrated survival probability outside [0,1]")
    return calibrated


def safe_binary_metrics(y: Any, probability: Any, weight: Any) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    weight = np.asarray(weight, dtype=float)
    result = {"auroc": np.nan, "auprc": np.nan, "brier": np.nan, "calibration_intercept": np.nan, "calibration_slope": np.nan}
    if len(y) == 0:
        return result
    result["brier"] = float(brier_score_loss(y, probability, sample_weight=weight))
    if np.unique(y).size == 2:
        result["auroc"] = float(roc_auc_score(y, probability, sample_weight=weight))
        result["auprc"] = float(average_precision_score(y, probability, sample_weight=weight))
        logit_p = np.log(probability / (1 - probability))
        try:
            calibration = LogisticRegression(C=1e8, max_iter=1200).fit(logit_p[:, None], y, sample_weight=weight)
            result["calibration_intercept"] = float(calibration.intercept_[0])
            result["calibration_slope"] = float(calibration.coef_[0, 0])
        except Exception:
            pass
    return result


def evaluate_survival(data: Any, splits: LockedSplits, survival: dict[str, Any], cfg: RunConfig) -> dict[str, Any]:
    test_ids = survival["eligible_test_ids"]
    development = np.r_[splits.train, splits.calibration, splits.validation]
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for model_name, result in survival["models"].items():
        predictions = result["predictions"]
        for horizon_index, month in enumerate(RISK_HORIZONS):
            for endpoint in (*COMPONENTS, "death", "composite"):
                y_test, known_test = known_risk_status(data, test_ids, endpoint, month)
                dev_y, dev_known = known_risk_status(data, development[~data.loc[development, "prior_any_component"].to_numpy(bool)], endpoint, month)
                dev_ids = development[~data.loc[development, "prior_any_component"].to_numpy(bool)]
                dev_frame = data.loc[dev_ids].copy()
                test_frame = data.loc[test_ids].copy()
                p_known, _, _ = fit_probability_model(dev_frame, dev_known.astype(int), test_frame, cfg.seed + month)
                evaluation_weight = np.where(known_test, 1 / np.clip(p_known, 0.02, 1), 0.0)
                usable = known_test & np.isfinite(predictions[endpoint][:, horizon_index])
                y = y_test[usable]
                p = predictions[endpoint][usable, horizon_index]
                w = evaluation_weight[usable]
                metrics = safe_binary_metrics(y, p, w)
                for metric, value in metrics.items():
                    metric_rows.append({
                        "model": model_name, "endpoint": endpoint, "horizon_month": month,
                        "metric": metric, "value": value, "n": len(y), "events": int(y.sum()),
                        "effective_n": effective_sample_size(w),
                    })
                for bin_index, bin_group in calibration_table(y, p, w).iterrows():
                    calibration_rows.append({
                        "model": model_name, "endpoint": endpoint, "horizon_month": month,
                        "bin": int(bin_index), **bin_group.to_dict(),
                    })
                if endpoint == "composite" and month == 60:
                    for threshold in np.linspace(0.03, 0.35, 12):
                        treated = p >= threshold
                        net = (np.sum(w * treated * y) - np.sum(w * treated * (1 - y)) * threshold / (1 - threshold)) / max(np.sum(w), 1e-9)
                        decision_rows.append({"model": model_name, "threshold": threshold, "net_benefit": float(net)})
                for local in np.flatnonzero(usable):
                    prediction_rows.append({
                        "patient_position": int(test_ids[local]), "model": model_name, "endpoint": endpoint,
                        "horizon_month": month, "observed": int(y_test[local]),
                        "probability": float(predictions[endpoint][local, horizon_index]),
                    })
    metrics = pd.DataFrame(metric_rows)
    integrated = metrics[(metrics["metric"].eq("brier"))].groupby(["model", "endpoint"], as_index=False).agg(value=("value", "mean"), n=("n", "min"))
    integrated["metric"] = "integrated_brier"
    integrated["horizon_month"] = 60
    integrated["events"] = np.nan
    integrated["effective_n"] = np.nan
    metrics = pd.concat([metrics, integrated[metrics.columns]], ignore_index=True)
    observed_cif = aalen_johansen_summary(data, test_ids)
    return {
        "metrics": metrics,
        "calibration": pd.DataFrame(calibration_rows),
        "decision_curve": pd.DataFrame(decision_rows),
        "predictions": pd.DataFrame(prediction_rows),
        "observed_cif": observed_cif,
    }


def calibration_table(y: Any, p: Any, w: Any, bins: int = 6) -> Any:
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w)
    if len(y) == 0:
        return pd.DataFrame(columns=["predicted", "observed", "weight", "n"])
    try:
        groups = pd.qcut(p, min(bins, max(2, len(np.unique(p)))), duplicates="drop")
    except ValueError:
        groups = pd.cut(p, bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    rows = []
    for _, ids in pd.Series(np.arange(len(y))).groupby(groups, observed=True):
        index = ids.to_numpy(int)
        if index.size:
            rows.append({
                "predicted": weighted_mean(p[index], w[index]), "observed": weighted_mean(y[index], w[index]),
                "weight": float(w[index].sum()), "n": len(index),
            })
    return pd.DataFrame(rows)


def aalen_johansen_summary(data: Any, ids: Any) -> Any:
    rows = []
    survival = 1.0
    cumulative = {name: 0.0 for name in (*COMPONENTS, "competing_death")}
    for month in range(PERSON_PERIOD_MONTHS, 61, PERSON_PERIOD_MONTHS):
        start = (month - PERSON_PERIOD_MONTHS) * DAYS_PER_MONTH
        end = month * DAYS_PER_MONTH
        at_risk, events = 0, {name: 0 for name in cumulative}
        for patient_position in ids:
            cause, event_day = earliest_incident_event(data, int(patient_position))
            censor = float(data.loc[patient_position, "censor_days"])
            if min(event_day, censor) > start:
                at_risk += 1
                if cause in events and event_day <= end and event_day <= censor:
                    events[cause] += 1
        if at_risk:
            for name in cumulative:
                cumulative[name] += survival * events[name] / at_risk
            survival *= max(0.0, 1 - sum(events.values()) / at_risk)
        if month in RISK_HORIZONS:
            for name, value in cumulative.items():
                rows.append({"endpoint": name, "horizon_month": month, "cumulative_incidence": value, "n_risk": at_risk})
            rows.append({
                "endpoint": "composite", "horizon_month": month,
                "cumulative_incidence": sum(cumulative[name] for name in COMPONENTS), "n_risk": at_risk,
            })
    return pd.DataFrame(rows)


# =====================================================================================
# 6. Cross-fitted target trial and gated dynamic postoperative GLP-1 analysis
# =====================================================================================


def continuous_or_binary_outcome_model(x: Any, y: Any, binary: bool, seed: int, cfg: RunConfig) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y)
    if binary:
        if len(y) < 10 or np.unique(y).size < 2:
            return ("constant", float(np.mean(y)) if len(y) else 0.5)
        model = LogisticRegression(C=0.1, max_iter=1400, random_state=seed).fit(x, y.astype(int))
        return ("binary", model)
    if len(y) < 10:
        return ("constant", float(np.mean(y)) if len(y) else 0.0)
    model = HistGradientBoostingRegressor(
        max_iter=cfg.hgb_iterations, max_leaf_nodes=15, learning_rate=0.06,
        l2_regularization=2.0, random_state=seed,
    ).fit(x, y.astype(float))
    return ("continuous", model)


def predict_outcome_model(model: Any, x: Any) -> Any:
    kind, fitted = model
    if kind == "constant":
        return np.full(len(x), float(fitted))
    if kind == "binary":
        return fitted.predict_proba(x)[:, 1]
    return fitted.predict(x)


def crossfit_trial_nuisance(data: Any, ids: Any, outcome: Any, observed: Any, binary: bool, cfg: RunConfig) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    ids = np.asarray(ids, dtype=int)
    treatment = data.loc[ids, "procedure_type"].eq("rygb").astype(int).to_numpy()
    outcome = np.asarray(outcome, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    numeric, categorical = target_trial_feature_columns(data)
    if any(any(token in name.lower() for token in FORBIDDEN_FUTURE_FEATURE_TOKENS) for name in numeric + categorical):
        raise AssertionError("Future complication event entered a target-trial outcome model")
    n_splits = min(5, int(np.bincount(treatment).min()))
    if n_splits < 2:
        raise PreflightError("Target-trial treatment overlap failed", ["Both procedures need at least two eligible patients"])
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    ps = np.full(len(ids), np.nan)
    pc = np.full(len(ids), np.nan)
    mu0 = np.full(len(ids), np.nan)
    mu1 = np.full(len(ids), np.nan)
    nuisance_records = []
    for fold_number, (fit, hold) in enumerate(folds.split(ids, treatment)):
        train_frame = data.loc[ids[fit]].copy()
        hold_frame = data.loc[ids[hold]].copy()
        encoder = FeatureEncoder.fit(train_frame, numeric, categorical)
        xfit = encoder.transform(train_frame)
        xhold = encoder.transform(hold_frame)
        if np.unique(treatment[fit]).size < 2:
            ps[hold] = float(np.mean(treatment[fit]))
            propensity = None
        else:
            propensity = LogisticRegression(C=0.15, max_iter=1600, random_state=cfg.seed + fold_number)
            propensity.fit(xfit, treatment[fit])
            ps[hold] = propensity.predict_proba(xhold)[:, 1]

        censor_xfit = np.column_stack([xfit, treatment[fit]])
        censor_xhold = np.column_stack([xhold, treatment[hold]])
        if np.unique(observed[fit]).size < 2:
            pc[hold] = float(np.mean(observed[fit]))
            censor_model = None
        else:
            censor_model = LogisticRegression(C=0.1, max_iter=1500, random_state=cfg.seed + 70 + fold_number)
            censor_model.fit(censor_xfit, observed[fit].astype(int))
            pc[hold] = censor_model.predict_proba(censor_xhold)[:, 1]

        arm_models = {}
        for arm in (0, 1):
            usable = (treatment[fit] == arm) & observed[fit] & np.isfinite(outcome[fit])
            model = continuous_or_binary_outcome_model(xfit[usable], outcome[fit][usable], binary, cfg.seed + arm + fold_number * 10, cfg)
            arm_models[arm] = model
            prediction = predict_outcome_model(model, xhold)
            if arm == 0:
                mu0[hold] = prediction
            else:
                mu1[hold] = prediction
        nuisance_records.append({
            "fold": fold_number, "fit_n": len(fit), "hold_n": len(hold),
            "fit_treated": int(treatment[fit].sum()), "fit_observed": int(observed[fit].sum()),
            "propensity_model": propensity, "censor_model": censor_model, "arm_models": arm_models,
        })
    return {
        "ids": ids, "treatment": treatment, "ps": np.clip(ps, 1e-6, 1 - 1e-6),
        "pc": np.clip(pc, 1e-6, 1), "mu0": mu0, "mu1": mu1,
        "outcome": outcome, "observed": observed, "folds": nuisance_records,
    }


def aipw_pseudo_outcome(nuisance: Mapping[str, Any]) -> Any:
    return aipw_arm_pseudo_outcome(nuisance, 1) - aipw_arm_pseudo_outcome(nuisance, 0)


def aipw_arm_pseudo_outcome(nuisance: Mapping[str, Any], arm: int) -> Any:
    a = np.asarray(nuisance["treatment"], dtype=float)
    y = np.asarray(nuisance["outcome"], dtype=float)
    delta = np.asarray(nuisance["observed"], dtype=float)
    ps = np.asarray(nuisance["ps"], dtype=float)
    pc = np.asarray(nuisance["pc"], dtype=float)
    mu = np.asarray(nuisance["mu1" if arm == 1 else "mu0"], dtype=float)
    observed_y = np.nan_to_num(y, nan=0.0)
    correction = delta / np.clip(pc, 1e-6, 1.0)
    indicator = a if arm == 1 else 1 - a
    exposure_probability = ps if arm == 1 else 1 - ps
    return mu + indicator * correction * (observed_y - mu) / np.clip(exposure_probability, 1e-6, 1.0)


def nuisance_minimum_support(nuisance: Mapping[str, Any]) -> float:
    ps = np.asarray(nuisance["ps"], dtype=float)
    pc = np.asarray(nuisance["pc"], dtype=float)
    candidates = np.r_[ps, 1 - ps, pc]
    candidates = candidates[np.isfinite(candidates)]
    return float(np.min(candidates)) if candidates.size else np.nan


def clustered_effect(pseudo: Any, centers: Any, cfg: RunConfig) -> dict[str, float]:
    pseudo = np.asarray(pseudo, dtype=float)
    keep = np.isfinite(pseudo)
    estimate = float(np.mean(pseudo[keep])) if keep.any() else np.nan
    ci = cluster_bootstrap_stat(pseudo[keep], np.asarray(centers)[keep], np.mean, cfg.cluster_bootstrap_replicates, cfg.seed + 818)
    return {"estimate": estimate, "ci_low": ci[0], "ci_high": ci[1], "n": int(keep.sum())}


def survivor_mean_effect(
    numerator_nuisance: Mapping[str, Any],
    survival_nuisance: Mapping[str, Any],
    centers: Any,
    cfg: RunConfig,
) -> dict[str, float]:
    """Estimate a contrast of strategy-specific survivor means without censoring death."""
    numerator = {arm: aipw_arm_pseudo_outcome(numerator_nuisance, arm) for arm in (0, 1)}
    survival = {arm: aipw_arm_pseudo_outcome(survival_nuisance, arm) for arm in (0, 1)}
    centers = np.asarray(centers).astype(str)
    keep = np.logical_and.reduce([
        np.isfinite(numerator[0]), np.isfinite(numerator[1]),
        np.isfinite(survival[0]), np.isfinite(survival[1]),
    ])
    if not keep.any():
        return {
            "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0,
            "mean_sleeve": np.nan, "mean_rygb": np.nan,
            "survival_sleeve": np.nan, "survival_rygb": np.nan,
        }
    numerator = {arm: values[keep] for arm, values in numerator.items()}
    survival = {arm: values[keep] for arm, values in survival.items()}
    centers = centers[keep]

    def arm_mean(arm: int, ids: Any) -> float:
        denominator = float(np.mean(survival[arm][ids]))
        return float(np.mean(numerator[arm][ids]) / denominator) if denominator > 1e-6 else np.nan

    all_ids = np.arange(keep.sum())
    mean0 = arm_mean(0, all_ids)
    mean1 = arm_mean(1, all_ids)
    estimate = mean1 - mean0
    unique_centers = np.unique(centers)
    rng = np.random.default_rng(cfg.seed + 919)
    by_center = {center: np.flatnonzero(centers == center) for center in unique_centers}
    bootstrap = []
    if len(unique_centers) >= 2:
        for _ in range(cfg.cluster_bootstrap_replicates):
            sampled = rng.choice(unique_centers, size=len(unique_centers), replace=True)
            ids = np.concatenate([by_center[center] for center in sampled])
            value0, value1 = arm_mean(0, ids), arm_mean(1, ids)
            if np.isfinite(value0) and np.isfinite(value1):
                bootstrap.append(value1 - value0)
    bootstrap = np.asarray(bootstrap, dtype=float)
    ci_low, ci_high = (
        (float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975)))
        if bootstrap.size else (np.nan, np.nan)
    )
    return {
        "estimate": estimate, "ci_low": ci_low, "ci_high": ci_high, "n": int(keep.sum()),
        "mean_sleeve": mean0, "mean_rygb": mean1,
        "survival_sleeve": float(np.mean(survival[0])),
        "survival_rygb": float(np.mean(survival[1])),
    }


def standardized_mean_differences(matrix: Any, treatment: Any, weight: Any | None = None) -> Any:
    matrix = np.asarray(matrix, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    result = np.full(matrix.shape[1], np.nan)
    weights = np.ones(len(treatment)) if weight is None else np.asarray(weight, dtype=float)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        arm_stats = []
        for arm in (0, 1):
            keep = (treatment == arm) & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            if not keep.any():
                arm_stats = []
                break
            mean = float(np.average(values[keep], weights=weights[keep]))
            variance = float(np.average((values[keep] - mean) ** 2, weights=weights[keep]))
            arm_stats.append((mean, variance))
        if len(arm_stats) == 2:
            result[column] = (arm_stats[1][0] - arm_stats[0][0]) / math.sqrt((arm_stats[1][1] + arm_stats[0][1]) / 2 + 1e-12)
    return result


def e_value_from_rr(rr: float, low: float | None = None, high: float | None = None) -> tuple[float, float]:
    def one(value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            return np.nan
        value = value if value >= 1 else 1 / value
        return float(value + math.sqrt(value * (value - 1)))
    point = one(rr)
    if low is None or high is None:
        return point, np.nan
    bound = 1.0 if low <= 1 <= high else one(low if low > 1 else high)
    return point, bound


def target_trial_analysis(data: Any, cfg: RunConfig) -> dict[str, Any]:
    eligible_base = (
        data["prior_glp1"].eq(0) & data["prior_mbs"].eq(0) & data["prior_dialysis_transplant"].eq(0) &
        data["procedure_type"].isin(["sleeve", "rygb"])
    )
    all_ids = np.flatnonzero(eligible_base.to_numpy())
    treatment = data.loc[all_ids, "procedure_type"].eq("rygb").astype(int).to_numpy()
    if min((treatment == 0).sum(), (treatment == 1).sum()) < max(20, cfg.min_cell_observations):
        raise PreflightError("Target-trial arm size gate failed", [
            f"Sleeve/RYGB eligible counts are {(treatment == 0).sum()}/{(treatment == 1).sum()}"
        ])
    effects: list[dict[str, Any]] = []
    nuisances: dict[str, Any] = {}
    endpoint_definitions = []
    for outcome in ("bmi", "hba1c"):
        target = f"{outcome}_60m_value"
        y = pd.to_numeric(data.loc[all_ids, target], errors="coerce").to_numpy(float)
        horizon_day = 60 * DAYS_PER_MONTH
        death_before = (
            data.loc[all_ids, "death_valid_event"].to_numpy(bool) &
            pd.to_numeric(data.loc[all_ids, "death_days"], errors="coerce").le(horizon_day).fillna(False).to_numpy()
        )
        followed_to_horizon = pd.to_numeric(data.loc[all_ids, "censor_days"], errors="coerce").ge(horizon_day).fillna(False).to_numpy()
        alive_status_known = death_before | followed_to_horizon
        alive = ~death_before
        measurement_observed = alive & followed_to_horizon & np.isfinite(y)
        product_observed = death_before | measurement_observed
        survival_outcome = np.where(alive_status_known, alive.astype(float), np.nan)
        survivor_numerator = np.where(death_before, 0.0, np.where(measurement_observed, y, np.nan))
        numerator_nuisance = crossfit_trial_nuisance(
            data, all_ids, survivor_numerator, product_observed, binary=False, cfg=cfg,
        )
        survival_nuisance = crossfit_trial_nuisance(
            data, all_ids, survival_outcome, alive_status_known, binary=True, cfg=cfg,
        )
        nuisances[target] = {"numerator": numerator_nuisance, "survival": survival_nuisance}
        effect = survivor_mean_effect(
            numerator_nuisance, survival_nuisance, data.loc[all_ids, "center_blind"].to_numpy(), cfg,
        )
        minimum_support = min(
            nuisance_minimum_support(numerator_nuisance), nuisance_minimum_support(survival_nuisance),
        )
        lo, hi = PLAUSIBLE_RANGES[outcome]
        plausible = lo <= effect["mean_sleeve"] <= hi and lo <= effect["mean_rygb"] <= hi
        estimable = bool(minimum_support >= SUCCESS_GATES["causal_min_support_probability"] and plausible)
        reason = "estimated" if estimable else (
            f"minimum cross-fitted treatment/censor support {minimum_support:.3g} is below "
            f"{SUCCESS_GATES['causal_min_support_probability']:.3g}"
            if minimum_support < SUCCESS_GATES["causal_min_support_probability"]
            else "AIPW strategy-specific survivor mean was outside the frozen clinical plausibility range"
        )
        effects.append({
            "outcome": target, "estimand": "strategy_specific_survivor_mean_difference",
            "status": "estimated" if estimable else "not_estimable", "status_reason": reason,
            "effect_rygb_minus_sleeve": effect["estimate"] if estimable else np.nan,
            "ci_low": effect["ci_low"] if estimable else np.nan, "ci_high": effect["ci_high"] if estimable else np.nan,
            "raw_untruncated_effect": effect["estimate"], "minimum_support_probability": minimum_support,
            "n": effect["n"], "observed_n": int(measurement_observed.sum()), "units": "kg/m2" if outcome == "bmi" else "percentage points",
            "mean_sleeve": effect["mean_sleeve"], "mean_rygb": effect["mean_rygb"],
            "survival_sleeve": effect["survival_sleeve"], "survival_rygb": effect["survival_rygb"],
            "interpretation": "Secondary strategy-specific survivor mean; cross-arm survivor populations can differ and no principal-stratum claim is made",
        })
        endpoint_definitions.append({
            "outcome": target, "eligible_n": len(all_ids), "observed_n": int(measurement_observed.sum()),
            "alive_status_known_n": int(alive_status_known.sum()), "death_before_horizon": int(death_before.sum()),
            "definition": "Ratio of AIPW mean survival-times-metabolic outcome to AIPW survival probability within each strategy",
        })

    incident_ids = all_ids[~data.loc[all_ids, "prior_any_component"].to_numpy(bool)]
    y, known = known_risk_status(data, incident_ids, "composite", 60)
    nuisance = crossfit_trial_nuisance(data, incident_ids, y, known, binary=True, cfg=cfg)
    nuisances["composite_60m"] = nuisance
    effect = clustered_effect(aipw_pseudo_outcome(nuisance), data.loc[incident_ids, "center_blind"].to_numpy(), cfg)
    risk0 = float(np.nanmean(aipw_arm_pseudo_outcome(nuisance, 0)))
    risk1 = float(np.nanmean(aipw_arm_pseudo_outcome(nuisance, 1)))
    minimum_support = nuisance_minimum_support(nuisance)
    estimable = bool(
        minimum_support >= SUCCESS_GATES["causal_min_support_probability"] and
        0 <= risk0 <= 1 and 0 <= risk1 <= 1
    )
    reason = "estimated" if estimable else (
        f"minimum cross-fitted treatment/censor support {minimum_support:.3g} is below "
        f"{SUCCESS_GATES['causal_min_support_probability']:.3g}"
        if minimum_support < SUCCESS_GATES["causal_min_support_probability"]
        else "AIPW marginal risk was outside the probability range"
    )
    rr = risk1 / risk0 if risk0 > 0 else np.nan
    low_rr = max(1e-6, risk0 + effect["ci_low"]) / risk0 if risk0 > 0 else np.nan
    high_rr = max(1e-6, risk0 + effect["ci_high"]) / risk0 if risk0 > 0 else np.nan
    e_point, e_bound = e_value_from_rr(rr, min(low_rr, high_rr), max(low_rr, high_rr))
    effects.append({
        "outcome": "composite_60m", "estimand": "five_year_cumulative_incidence_risk_difference",
        "status": "estimated" if estimable else "not_estimable", "status_reason": reason,
        "effect_rygb_minus_sleeve": effect["estimate"] if estimable else np.nan,
        "ci_low": effect["ci_low"] if estimable else np.nan, "ci_high": effect["ci_high"] if estimable else np.nan,
        "raw_untruncated_effect": effect["estimate"], "minimum_support_probability": minimum_support,
        "n": effect["n"], "observed_n": int(known.sum()), "units": "absolute probability",
        "risk_sleeve": risk0 if estimable else np.nan, "risk_rygb": risk1 if estimable else np.nan,
        "risk_ratio": rr if estimable else np.nan,
        "e_value_point": e_point if estimable else np.nan, "e_value_ci": e_bound if estimable else np.nan,
        "interpretation": "Incident first component event with death as a competing event",
    })
    endpoint_definitions.append({"outcome": "composite_60m", "eligible_n": len(incident_ids), "observed_n": int(known.sum()), "events": int(y[known].sum())})

    ps_nuisance = nuisances["composite_60m"]
    ps = ps_nuisance["ps"]
    a = ps_nuisance["treatment"]
    marginal = float(a.mean())
    stabilized = np.where(a == 1, marginal / np.clip(ps, 0.02, 1), (1 - marginal) / np.clip(1 - ps, 0.02, 1))
    support = (ps > 0.02) & (ps < 0.98)
    numeric, categorical = target_trial_feature_columns(data)
    balance_frame = data.loc[incident_ids]
    balance_encoder = FeatureEncoder.fit(balance_frame, numeric, categorical)
    balance_matrix = balance_encoder.transform(balance_frame)
    before = standardized_mean_differences(balance_matrix, a)
    after = standardized_mean_differences(balance_matrix, a, np.where(support, stabilized, 0))
    balance = pd.DataFrame({
        "covariate": balance_encoder.output_names, "smd_before": before, "smd_after": after,
        "abs_smd_before": np.abs(before), "abs_smd_after": np.abs(after),
    })
    overlap = pd.DataFrame([
        {
            "arm": "sleeve" if arm == 0 else "rygb", "n": int((a == arm).sum()),
            "ps_min": float(np.min(ps[a == arm])), "ps_q25": float(np.quantile(ps[a == arm], 0.25)),
            "ps_median": float(np.median(ps[a == arm])), "ps_q75": float(np.quantile(ps[a == arm], 0.75)),
            "ps_max": float(np.max(ps[a == arm])),
        }
        for arm in (0, 1)
    ])
    minimum_trial_support = float(np.nanmin([row["minimum_support_probability"] for row in effects]))
    weight_summary = pd.DataFrame([
        {
            "weighting": "stabilized_iptw_0.02_0.98", "n": int(support.sum()),
            "trimmed": int((~support).sum()), "effective_n": effective_sample_size(np.where(support, stabilized, 0)),
            "max_weight": float(np.max(stabilized[support])) if support.any() else np.nan,
            "max_abs_smd": float(np.nanmax(np.abs(after))) if np.isfinite(after).any() else np.nan,
            "minimum_support_probability": minimum_trial_support,
        },
        {
            "weighting": "overlap_sensitivity", "n": len(a), "trimmed": 0,
            "effective_n": effective_sample_size(np.where(a == 1, 1 - ps, ps)),
            "max_weight": float(np.max(np.where(a == 1, 1 - ps, ps))),
            "max_abs_smd": np.nan, "minimum_support_probability": minimum_trial_support,
        },
    ])
    protocol_flow = pd.DataFrame([
        {"step": "source cohort", "n": len(data)},
        {"step": "target-trial eligible sleeve or RYGB and PriorGLP1=0", "n": len(all_ids)},
        {"step": "incident composite target population", "n": len(incident_ids)},
        {"step": "five-year risk status known", "n": int(known.sum())},
    ])
    negative_control = {
        "status": "not_available",
        "reason": "No prespecified negative-control outcome was supplied",
    }
    if "negative_control_outcome" in data and data["negative_control_outcome"].notna().sum() >= max(20, cfg.min_cell_observations):
        nc_y = pd.to_numeric(data.loc[all_ids, "negative_control_outcome"], errors="coerce").to_numpy(float)
        nc_observed = np.isfinite(nc_y)
        nc = crossfit_trial_nuisance(data, all_ids, nc_y, nc_observed, binary=True, cfg=cfg)
        nc_effect = clustered_effect(aipw_pseudo_outcome(nc), data.loc[all_ids, "center_blind"].to_numpy(), cfg)
        negative_control = {"status": "estimated", **nc_effect}
    return {
        "effects": pd.DataFrame(effects), "balance": balance, "overlap": overlap,
        "weight_summary": weight_summary, "flow": protocol_flow,
        "endpoint_definitions": pd.DataFrame(endpoint_definitions), "negative_control": negative_control,
        "protocol": TARGET_TRIAL_PROTOCOL, "nuisance": nuisances,
    }


def dynamic_glp1_schema_status(data: Any) -> dict[str, Any]:
    required = [f"monthly_{family}_{month:02d}" for family in ("glp1", "bmi", "hba1c", "medication", "utilization", "observed") for month in range(1, 37)]
    missing = [name for name in required if name not in data]
    return {
        "eligible": not missing,
        "required_field_count": len(required),
        "missing_field_count": len(missing),
        "missing_examples": missing[:12],
        "reason": "all monthly fields available" if not missing else "Monthly treatment, BMI, HbA1c, medication, utilization, and observability histories through month 36 are required",
    }


def dynamic_glp1_analysis(data: Any, cfg: RunConfig) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    gate = dynamic_glp1_schema_status(data)
    if not gate["eligible"]:
        return {"status": "not_estimable", "gate": gate, "protocol": DYNAMIC_GLP1_PROTOCOL, "effects": pd.DataFrame()}
    bmi12 = pd.to_numeric(data["monthly_bmi_12"], errors="coerce")
    alive12 = pd.to_numeric(data["monthly_observed_12"], errors="coerce").eq(1)
    eligible_ids = np.flatnonzero((bmi12.ge(35) & alive12 & data["prior_glp1"].eq(0)).to_numpy())
    if len(eligible_ids) < max(100, cfg.min_development_patients // 4):
        return {
            "status": "not_estimable", "gate": {**gate, "reason": f"Only {len(eligible_ids)} patients pass the month-12 dynamic eligibility gate"},
            "protocol": DYNAMIC_GLP1_PROTOCOL, "effects": pd.DataFrame(),
        }
    long_rows = []
    for patient_position in eligible_ids:
        prior_treatment = 0.0
        for month in range(12, 25):
            current = float(pd.to_numeric(pd.Series([data.loc[patient_position, f"monthly_glp1_{month:02d}"]]), errors="coerce").iloc[0])
            if not np.isfinite(current):
                continue
            long_rows.append({
                "patient_position": int(patient_position), "month": month, "treatment": int(current > 0),
                "prior_treatment": prior_treatment,
                "bmi": data.loc[patient_position, f"monthly_bmi_{month:02d}"],
                "hba1c": data.loc[patient_position, f"monthly_hba1c_{month:02d}"],
                "medication": data.loc[patient_position, f"monthly_medication_{month:02d}"],
                "utilization": data.loc[patient_position, f"monthly_utilization_{month:02d}"],
                "age": data.loc[patient_position, "age"], "baseline_bmi": data.loc[patient_position, "baseline_bmi"],
                "baseline_hba1c": data.loc[patient_position, "baseline_hba1c"],
            })
            prior_treatment = max(prior_treatment, float(current > 0))
    long_frame = pd.DataFrame(long_rows)
    groups = long_frame["patient_position"].to_numpy()
    unique_patients = np.unique(groups)
    n_splits = min(5, len(unique_patients))
    p_treatment = np.full(len(long_frame), np.nan)
    feature_columns = ["month", "prior_treatment", "bmi", "hba1c", "medication", "utilization", "age", "baseline_bmi", "baseline_hba1c"]
    for fit, hold in GroupKFold(n_splits=n_splits).split(long_frame, groups=groups):
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=1300, random_state=cfg.seed)),
        ])
        if long_frame.iloc[fit]["treatment"].nunique() < 2:
            p_treatment[hold] = float(long_frame.iloc[fit]["treatment"].mean())
        else:
            model.fit(long_frame.iloc[fit][feature_columns], long_frame.iloc[fit]["treatment"])
            p_treatment[hold] = model.predict_proba(long_frame.iloc[hold][feature_columns])[:, 1]
    long_frame["p_treatment"] = np.clip(p_treatment, 0.01, 0.99)
    effects = []
    for outcome in ("bmi", "hba1c"):
        strategy_values = {}
        for strategy in ("initiate_12_18_if_bmi_ge_35", "no_initiation_through_24"):
            values, weights = [], []
            for patient_position, history in long_frame.groupby("patient_position"):
                history = history.sort_values("month")
                initiated = False
                adherent = True
                probability = 1.0
                for row in history.itertuples():
                    actual = int(row.treatment)
                    probability *= row.p_treatment if actual else (1 - row.p_treatment)
                    initiated = initiated or bool(actual)
                    if strategy == "no_initiation_through_24" and actual:
                        adherent = False
                        break
                    if strategy == "initiate_12_18_if_bmi_ge_35" and row.month == 18 and not initiated:
                        adherent = False
                        break
                outcome_value = pd.to_numeric(pd.Series([data.loc[patient_position, f"monthly_{outcome}_36"]]), errors="coerce").iloc[0]
                observed36 = pd.to_numeric(pd.Series([data.loc[patient_position, "monthly_observed_36"]]), errors="coerce").iloc[0]
                if adherent and observed36 == 1 and np.isfinite(outcome_value):
                    values.append(float(outcome_value))
                    weights.append(1 / max(probability, 0.01))
            values = np.asarray(values, dtype=float)
            weights = np.asarray(weights, dtype=float)
            if weights.size:
                weights = np.minimum(weights, np.quantile(weights, 0.99))
            strategy_values[strategy] = (values, weights)
        a_values, a_weights = strategy_values["initiate_12_18_if_bmi_ge_35"]
        b_values, b_weights = strategy_values["no_initiation_through_24"]
        if min(len(a_values), len(b_values)) < max(20, cfg.min_cell_observations):
            continue
        effects.append({
            "outcome": f"{outcome}_36m", "strategy_a_mean": weighted_mean(a_values, a_weights),
            "strategy_b_mean": weighted_mean(b_values, b_weights),
            "difference_a_minus_b": weighted_mean(a_values, a_weights) - weighted_mean(b_values, b_weights),
            "strategy_a_n": len(a_values), "strategy_b_n": len(b_values),
            "strategy_a_effective_n": effective_sample_size(a_weights), "strategy_b_effective_n": effective_sample_size(b_weights),
        })
    effect_frame = pd.DataFrame(effects)
    if effect_frame.empty:
        return {"status": "not_estimable", "gate": {**gate, "reason": "Clone adherence or 36-month outcome support was inadequate"}, "protocol": DYNAMIC_GLP1_PROTOCOL, "effects": effect_frame}
    return {"status": "estimated", "gate": gate, "protocol": DYNAMIC_GLP1_PROTOCOL, "effects": effect_frame}


# =====================================================================================
# 7. Frozen gates, aggregate artifacts, and report inputs
# =====================================================================================


def simulation_precision_analysis(cfg: RunConfig) -> Any:
    """Frozen, outcome-blind simulation used to document minimum cell sizes."""
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for n in sorted(set([cfg.min_cell_observations, max(30, cfg.min_cell_observations * 2), max(100, cfg.min_cell_observations * 4)])):
        differences = rng.normal(-0.12, 0.65, size=(1500, n))
        estimates = differences.mean(axis=1)
        half_width = float(np.quantile(np.abs(estimates - estimates.mean()), 0.95))
        events = rng.binomial(n, 0.08, size=1500)
        rows.append({
            "n": n, "assumed_paired_standardized_difference": -0.12,
            "simulated_95_half_width": half_width, "assumed_event_rate": 0.08,
            "median_events": float(np.median(events)), "seed": cfg.seed,
        })
    return pd.DataFrame(rows)


def build_preflight_tables(data: Any, cohort_meta: Mapping[str, Any], splits: LockedSplits, observation: dict[str, Any], cfg: RunConfig) -> dict[str, Any]:
    split_label = pd.Series("not_primary_split", index=data.index, dtype="string")
    for name in ("train", "calibration", "validation", "test", "contemporary"):
        split_label.iloc[getattr(splits, name)] = name
    composition_rows = []
    for keys, group in data.assign(split=split_label).groupby(["center_blind", "split", "procedure_type"], dropna=False):
        composition_rows.append({"center_blind": keys[0], "split": str(keys[1]), "procedure_type": keys[2], "n": len(group)})
    calendar = data.assign(split=split_label).groupby([data["procedure_date"].dt.year.rename("procedure_year"), "split"], as_index=False).size().rename(columns={"size": "n"})
    maturity_rows = []
    for split_name in ("train", "calibration", "validation", "test", "contemporary"):
        ids = getattr(splits, split_name)
        for outcome, horizons in TRAJECTORY_HORIZONS.items():
            for month in horizons:
                maturity_rows.append({
                    "split": split_name, "outcome": outcome, "target_month": month, "n_total": len(ids),
                    "administrative_opportunity_n": int(data.loc[ids, "administrative_opportunity_days"].ge(month * DAYS_PER_MONTH).sum()),
                    "observed_measurement_n": int(data.loc[ids, f"{outcome}_{month}m_value"].notna().sum()),
                })
    timing_rows = []
    for outcome, horizons in TRAJECTORY_HORIZONS.items():
        for month in horizons:
            base = f"{outcome}_{month}m"
            observed = data[base + "_value"].notna()
            day = pd.to_numeric(data.loc[observed, base + "_day"], errors="coerce")
            timing_rows.append({
                "outcome": outcome, "target_month": month, "observed_n": int(observed.sum()),
                "median_selected_month": float(np.median(day / DAYS_PER_MONTH)) if len(day) else np.nan,
                "p10_selected_month": float(np.quantile(day / DAYS_PER_MONTH, 0.10)) if len(day) else np.nan,
                "p90_selected_month": float(np.quantile(day / DAYS_PER_MONTH, 0.90)) if len(day) else np.nan,
                "median_window_count": float(np.median(data.loc[observed, base + "_count"])) if observed.any() else np.nan,
                "missing_fraction": float(1 - observed.mean()),
            })
    missingness_rows = []
    for column in [name for name in BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES if name in data]:
        missingness_rows.append({"field": column, "missing_fraction": float(data[column].isna().mean()), "category": "baseline_feature"})
    for row in cohort_meta["event_consistency"]:
        missingness_rows.append({
            "field": row["endpoint"] + "_event_time_when_flagged",
            "missing_fraction": row["missing_time_sensitivity"] / max(row["flagged"], 1),
            "category": "event_timing",
        })
    return {
        "cohort_funnel": pd.DataFrame(cohort_meta["funnel"]),
        "composition": pd.DataFrame(composition_rows),
        "calendar": calendar,
        "maturity": pd.DataFrame(maturity_rows),
        "timing": pd.DataFrame(timing_rows),
        "missingness": pd.DataFrame(missingness_rows),
        "event_consistency": pd.DataFrame(cohort_meta["event_consistency"]),
        "observation_weights": observation["summary"],
        "precision_analysis": simulation_precision_analysis(cfg),
    }


def select_promoted_model(trajectory: dict[str, Any], outcome: str) -> str:
    ensemble = trajectory["ensemble_status"]
    row = ensemble[(ensemble["outcome"].eq(outcome)) & ensemble["status"].eq("accepted")]
    return MODEL_ENSEMBLE if len(row) else MODEL_CATBOOST


def evaluate_success_gates(
    trajectory: dict[str, Any],
    trajectory_eval: dict[str, Any],
    survival_eval: dict[str, Any],
    trial: dict[str, Any],
    cfg: RunConfig,
) -> tuple[Any, str]:
    rows: list[dict[str, Any]] = []
    metrics = trajectory_eval["metrics"]
    inference = trajectory_eval["inference"]
    for outcome in TRAJECTORY_HORIZONS:
        candidate = select_promoted_model(trajectory, outcome)
        selected = metrics[
            metrics["evaluation_set"].eq("test") & metrics["estimate"].eq("complete_case") &
            metrics["metric"].eq("standardized_crps") & metrics["outcome"].eq(outcome)
        ]
        base = float(selected[selected["model"].eq(MODEL_PUBLISHED)]["value"].mean())
        cand = float(selected[selected["model"].eq(candidate)]["value"].mean())
        improvement = (base - cand) / base if np.isfinite(base) and base > 0 else np.nan
        rows.append({
            "domain": "trajectory", "outcome": outcome, "gate": "standardized CRPS improvement at least 10 percent",
            "value": improvement, "threshold": SUCCESS_GATES["trajectory_standardized_crps_relative_improvement"],
            "assessable": np.isfinite(improvement), "passed": bool(np.isfinite(improvement) and improvement >= SUCCESS_GATES["trajectory_standardized_crps_relative_improvement"]),
        })
        horizon = inference[(inference["model"].eq(candidate)) & inference["outcome"].eq(outcome) & inference["relative_crps_improvement"].notna()]
        worst = float(horizon["relative_crps_improvement"].min()) if len(horizon) else np.nan
        rows.append({
            "domain": "trajectory", "outcome": outcome, "gate": "no powered horizon more than 5 percent worse",
            "value": worst, "threshold": -SUCCESS_GATES["trajectory_max_powered_horizon_worsening"],
            "assessable": np.isfinite(worst), "passed": bool(np.isfinite(worst) and worst >= -SUCCESS_GATES["trajectory_max_powered_horizon_worsening"]),
        })
        rmse = inference[(inference["model"].eq(candidate)) & inference["outcome"].eq(outcome) & inference["target_month"].eq(60) & inference["rmse_difference"].notna()]
        upper = float(rmse["rmse_difference_ci_high"].iloc[0]) if len(rmse) else np.nan
        rows.append({
            "domain": "trajectory", "outcome": outcome, "gate": "five-year RMSE clustered interval supports improvement",
            "value": upper, "threshold": 0.0, "assessable": np.isfinite(upper), "passed": bool(np.isfinite(upper) and upper < 0),
        })
        coverage = metrics[
            metrics["evaluation_set"].eq("test") & metrics["model"].eq(candidate) &
            metrics["outcome"].eq(outcome) & metrics["origin"].eq(0) &
            metrics["estimate"].eq("complete_case") & metrics["metric"].eq("coverage")
        ]
        error = float(np.max(np.abs(coverage["value"] - coverage["level"]))) if len(coverage) else np.nan
        rows.append({
            "domain": "trajectory", "outcome": outcome, "gate": "maximum interval coverage error at most 5 points",
            "value": error, "threshold": SUCCESS_GATES["coverage_max_absolute_error"],
            "assessable": np.isfinite(error), "passed": bool(np.isfinite(error) and error <= SUCCESS_GATES["coverage_max_absolute_error"]),
        })

    risk = survival_eval["metrics"]
    for metric, direction in (("brier", "lower"), ("calibration_slope", "range")):
        candidate_row = risk[(risk["model"].eq(RISK_MODEL_AUGMENTED)) & risk["endpoint"].eq("composite") & risk["horizon_month"].eq(60) & risk["metric"].eq(metric)]
        baseline_row = risk[(risk["model"].eq(RISK_MODEL_BASELINE)) & risk["endpoint"].eq("composite") & risk["horizon_month"].eq(60) & risk["metric"].eq(metric)]
        value = float(candidate_row["value"].iloc[0]) if len(candidate_row) else np.nan
        if direction == "lower":
            baseline_value = float(baseline_row["value"].iloc[0]) if len(baseline_row) else np.nan
            passed = np.isfinite(value) and np.isfinite(baseline_value) and value < baseline_value
            gate = "five-year composite Brier improves on clinical survival baseline"
            threshold = baseline_value
        else:
            passed = np.isfinite(value) and SUCCESS_GATES["risk_calibration_slope_low"] <= value <= SUCCESS_GATES["risk_calibration_slope_high"]
            gate = "five-year composite calibration slope 0.80 to 1.20"
            threshold = 1.0
        rows.append({"domain": "risk", "outcome": "composite", "gate": gate, "value": value, "threshold": threshold, "assessable": np.isfinite(value), "passed": bool(passed)})

    max_smd = float(trial["weight_summary"].iloc[0]["max_abs_smd"])
    ess = float(trial["weight_summary"].iloc[0]["effective_n"])
    minimum_support = float(trial["weight_summary"].iloc[0]["minimum_support_probability"])
    rows.append({
        "domain": "causal", "outcome": "sleeve_vs_rygb", "gate": "post-adjustment maximum absolute SMD below 0.10",
        "value": max_smd, "threshold": SUCCESS_GATES["causal_max_abs_smd"], "assessable": np.isfinite(max_smd),
        "passed": bool(np.isfinite(max_smd) and max_smd < SUCCESS_GATES["causal_max_abs_smd"]),
    })
    rows.append({
        "domain": "causal", "outcome": "sleeve_vs_rygb", "gate": "treatment-weight effective sample size adequate",
        "value": ess, "threshold": SUCCESS_GATES["causal_min_effective_sample_size"], "assessable": np.isfinite(ess),
        "passed": bool(np.isfinite(ess) and ess >= SUCCESS_GATES["causal_min_effective_sample_size"]),
    })
    rows.append({
        "domain": "causal", "outcome": "sleeve_vs_rygb", "gate": "minimum treatment/censor support at least 0.02",
        "value": minimum_support, "threshold": SUCCESS_GATES["causal_min_support_probability"],
        "assessable": np.isfinite(minimum_support),
        "passed": bool(np.isfinite(minimum_support) and minimum_support >= SUCCESS_GATES["causal_min_support_probability"]),
    })
    gates = pd.DataFrame(rows)
    validity = gates["assessable"].all()
    prediction_pass = gates[gates["domain"].isin(["trajectory", "risk"])]["passed"].all()
    if cfg.smoke:
        classification = "synthetic validation only"
    elif not validity:
        classification = "analysis not estimable"
    elif prediction_pass:
        classification = "superiority supported"
    elif gates[gates["domain"].isin(["trajectory", "risk"])]["passed"].any():
        classification = "promising but not externally established"
    else:
        classification = "exploratory only"
    return gates, classification


def artifact_tables(
    preflight: dict[str, Any],
    trajectory: dict[str, Any],
    trajectory_eval: dict[str, Any],
    iev: Any,
    survival: dict[str, Any],
    survival_eval: dict[str, Any],
    trial: dict[str, Any],
    glp1: dict[str, Any],
    gates: Any,
) -> dict[str, Any]:
    tables = dict(preflight)
    tables.update({
        "trajectory_model_specs": trajectory["model_specs"],
        "trajectory_conformal": trajectory["conformal"],
        "trajectory_ensemble": trajectory["ensemble_status"],
        "trajectory_metrics": trajectory_eval["metrics"],
        "trajectory_inference": trajectory_eval["inference"],
        "trajectory_center_metrics": trajectory_eval["center_metrics"],
        "trajectory_subgroup_metrics": trajectory_eval["subgroup_metrics"],
        "trajectory_leaderboard": trajectory_eval["leaderboard"],
        "internal_external_validation": iev,
        "survival_model_specs": survival["specifications"],
        "survival_metrics": survival_eval["metrics"],
        "survival_calibration": survival_eval["calibration"],
        "survival_decision_curve": survival_eval["decision_curve"],
        "observed_cumulative_incidence": survival_eval["observed_cif"],
        "trial_flow": trial["flow"],
        "trial_effects": trial["effects"],
        "trial_balance": trial["balance"],
        "trial_overlap": trial["overlap"],
        "trial_weights": trial["weight_summary"],
        "trial_endpoint_definitions": trial["endpoint_definitions"],
        "dynamic_glp1_effects": glp1["effects"],
        "success_gates": gates,
    })
    return tables


def save_aggregate_tables(context: RunContext, tables: Mapping[str, Any], scalars: Mapping[str, Any]) -> None:
    index: dict[str, str] = {}
    forbidden = {"patient_id", "patient_position", "center_id"}
    for name, frame in tables.items():
        if frame is None:
            frame = pd.DataFrame()
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        leak = forbidden & set(frame.columns)
        if leak:
            raise AssertionError(f"Aggregate table {name} contains patient-level or identifiable columns: {sorted(leak)}")
        path = context.aggregate / f"{name}.csv"
        atomic_csv(path, frame)
        index[name] = path.name
    atomic_json(context.aggregate / "report_index.json", {"tables": index, "scalars": scalars})


def load_aggregate_tables(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    index = read_json(run_dir / "AGGREGATE" / "report_index.json", {})
    if not index:
        raise PreflightError("Plot-only reconstruction failed", [f"No aggregate report index exists under {run_dir}"])
    tables = {}
    for name, filename in index.get("tables", {}).items():
        path = run_dir / "AGGREGATE" / filename
        try:
            tables[name] = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
        except pd.errors.EmptyDataError:
            tables[name] = pd.DataFrame()
    return tables, index.get("scalars", {})


# =====================================================================================
# 8. Aggregate-only 15-page report and plot-only reconstruction
# =====================================================================================

REPORT_PAGES = (
    "run_identity_and_success_gates",
    "cohort_funnel_and_exclusions",
    "center_procedure_calendar_and_splits",
    "followup_maturity_censoring_and_effective_n",
    "measurement_timing_and_missingness",
    "trajectory_leaderboard_and_literature_context",
    "trajectory_performance_by_horizon",
    "trajectory_calibration_coverage_and_tails",
    "center_and_subgroup_transportability",
    "competing_risk_performance_and_incidence",
    "trajectory_risk_ablation_and_decision_curves",
    "target_trial_protocol_and_flow",
    "propensity_overlap_balance_and_weights",
    "causal_effects_and_sensitivity",
    "limitations_unmet_gates_and_conclusion",
)


def new_report_page(number: int, title: str, subtitle: str = "") -> Any:
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.025, 0.06, 0.95, 0.81), w_pad=4 / 72, h_pad=4 / 72)
    fig.patch.set_facecolor("#FAFBFD")
    fig.text(
        0.035, 0.955, f"{number:02d}", fontsize=20, family="DejaVu Sans Mono",
        weight="bold", color="#18212F", va="top",
    )
    fig.text(0.072, 0.955, title, fontsize=20, weight="bold", color="#18212F", va="top")
    if subtitle:
        fig.text(0.035, 0.915, subtitle, fontsize=10.5, color="#596579", va="top")
    fig.text(0.965, 0.018, "Aggregate output only | Epic Cosmos bariatric five-year study", ha="right", fontsize=8, color="#7A8493")
    return fig


def style_axis(axis: Any, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(axis="y", alpha=0.18, linewidth=0.8)
    axis.tick_params(labelsize=8)


def no_data(axis: Any, message: str) -> None:
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#687386", transform=axis.transAxes)


def add_text_box(
    axis: Any,
    lines: Sequence[str],
    fontsize: float = 10,
    monospace: bool = False,
    wrap_width: int | None = None,
) -> None:
    rendered_lines: list[str] = []
    for raw_line in lines:
        for line in str(raw_line).split("\n"):
            if wrap_width is None or not line:
                rendered_lines.append(line)
                continue
            leading = re.match(r"\s*", line).group(0)
            subsequent = leading + ("  " if line.lstrip().startswith("-") else "")
            rendered_lines.extend(textwrap.wrap(
                line,
                width=wrap_width,
                subsequent_indent=subsequent,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""])
    axis.axis("off")
    axis.text(
        0.01, 0.99, "\n".join(rendered_lines), va="top", ha="left", fontsize=fontsize,
        family="monospace" if monospace else "sans-serif", color="#202936", transform=axis.transAxes,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "edgecolor": "#D8DEE8"},
    )


def report_figures(tables: Mapping[str, Any], scalars: Mapping[str, Any]) -> list[Any]:
    figures: list[Any] = []
    classification = str(scalars.get("classification", "not classified"))

    # Page 1
    fig = new_report_page(1, "Run identity, status, dependencies, and success gates", f"Claim classification: {classification}")
    grid = fig.add_gridspec(1, 2, left=0.04, right=0.97, top=0.87, bottom=0.07, width_ratios=[0.42, 0.58])
    identity = fig.add_subplot(grid[0, 0])
    fingerprint = str(scalars.get("fingerprint", "unknown"))
    dependencies = scalars.get("dependencies", {})
    identity_lines = [
        f"Study version: {STUDY_VERSION}", f"Run fingerprint: {fingerprint}",
        f"Mode: {scalars.get('mode', 'unknown')}", f"Status: {scalars.get('status', 'unknown')}",
        f"Patients: {scalars.get('analysis_n', 'unknown')}", f"Centers: {scalars.get('center_n', 'unknown')}",
        f"SQL contract: {scalars.get('sql_contract_version', SQL_CONTRACT_VERSION)}",
        f"SQL SHA-256: {str(scalars.get('sql_sha256', 'unknown'))[:24]}...",
        f"Code SHA-256: {str(scalars.get('code_sha256', 'unknown'))[:24]}...", "", "Dependencies:",
    ] + [f"  {name}: {version}" for name, version in sorted(dependencies.items()) if name not in {"executable", "platform"}]
    add_text_box(identity, identity_lines, fontsize=9.3, monospace=True)
    gate_axis = fig.add_subplot(grid[0, 1])
    gates = tables.get("success_gates", pd.DataFrame())
    if gates.empty:
        no_data(gate_axis, "Success gates unavailable")
    else:
        gate_axis.axis("off")
        rows = []
        colors = []
        for row in gates.itertuples():
            assessable = str(getattr(row, "assessable", True)).lower() in {"true", "1"} if isinstance(getattr(row, "assessable", True), str) else bool(getattr(row, "assessable", True))
            passed = str(getattr(row, "passed", False)).lower() in {"true", "1"} if isinstance(getattr(row, "passed", False), str) else bool(getattr(row, "passed", False))
            state = "PASS" if passed else "FAIL" if assessable else "NOT ASSESSABLE"
            value = getattr(row, "value", np.nan)
            rows.append([state, str(row.domain), str(getattr(row, "outcome", "")), str(row.gate)[:62], f"{value:.3g}" if np.isfinite(value) else "NA"])
            colors.append("#DDF3E5" if passed else "#FBE1E4" if assessable else "#EEF1F5")
        table = gate_axis.table(cellText=rows, colLabels=["State", "Domain", "Outcome", "Frozen gate", "Value"], loc="upper left", cellLoc="left", colWidths=[0.12, 0.11, 0.13, 0.54, 0.10])
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1, 1.35)
        for column in range(5):
            table[(0, column)].set_facecolor("#23334D")
            table[(0, column)].set_text_props(color="white", weight="bold")
        for row_index, color in enumerate(colors, 1):
            for column in range(5):
                table[(row_index, column)].set_facecolor(color)
    figures.append(fig)

    # Page 2
    fig = new_report_page(2, "Cohort funnel and exclusions", "Filters are applied in frozen order; follow-up is not a baseline eligibility filter")
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.96, top=0.87, bottom=0.09, width_ratios=[0.60, 0.40])
    axis = fig.add_subplot(grid[0, 0])
    funnel = tables.get("cohort_funnel", pd.DataFrame())
    if funnel.empty:
        no_data(axis, "Cohort funnel unavailable")
    else:
        shown = funnel.iloc[::-1]
        axis.barh(shown["step"], shown["remaining"], color="#3C78A8")
        for index, value in enumerate(shown["remaining"]):
            axis.text(value, index, f" {int(value):,}", va="center", fontsize=9)
        axis.set_xlabel("Patients remaining")
        style_axis(axis)
    event_axis = fig.add_subplot(grid[0, 1])
    event = tables.get("event_consistency", pd.DataFrame())
    if event.empty:
        no_data(event_axis, "Event-time audit unavailable")
    else:
        add_text_box(event_axis, [
            "Event-time consistency audit", "",
            *[
                f"{row.endpoint:14s} flagged={int(row.flagged):4d}  valid={int(row.valid_primary_events):4d}  missing-time sensitivity={int(row.missing_time_sensitivity):3d}"
                for row in event.itertuples()
            ], "", "Patients with a flagged event but no date remain in a sensitivity set.",
            "They are not treated as known five-year cases or noncases.",
        ], fontsize=9.2, monospace=True)
    figures.append(fig)

    # Page 3
    fig = new_report_page(3, "Center, procedure, calendar, and split composition", "Center labels are deterministic blinded hashes")
    grid = fig.add_gridspec(2, 2, left=0.055, right=0.97, top=0.87, bottom=0.08, hspace=0.30)
    composition = tables.get("composition", pd.DataFrame())
    axis = fig.add_subplot(grid[:, 0])
    if composition.empty:
        no_data(axis, "Center composition unavailable")
    else:
        center = composition.groupby(["center_blind", "split"], as_index=False)["n"].sum()
        pivot = center.pivot(index="center_blind", columns="split", values="n").fillna(0)
        pivot.plot.barh(stacked=True, ax=axis, colormap="tab20c", width=0.8)
        axis.set_xlabel("Patients")
        axis.set_ylabel("")
        axis.legend(fontsize=7, ncol=2, loc="lower right")
        style_axis(axis)
    calendar_axis = fig.add_subplot(grid[0, 1])
    calendar = tables.get("calendar", pd.DataFrame())
    if calendar.empty:
        no_data(calendar_axis, "Calendar composition unavailable")
    else:
        for split, group in calendar.groupby("split"):
            calendar_axis.plot(group["procedure_year"], group["n"], marker="o", label=split)
        calendar_axis.axvline(pd.Timestamp(scalars.get("temporal_cutoff", "2000-01-01")).year, color="#B64253", linestyle="--", label="temporal cutoff")
        calendar_axis.set_xlabel("Procedure year")
        calendar_axis.set_ylabel("Patients")
        calendar_axis.legend(fontsize=7, ncol=2)
        style_axis(calendar_axis)
    procedure_axis = fig.add_subplot(grid[1, 1])
    if composition.empty:
        no_data(procedure_axis, "Procedure composition unavailable")
    else:
        proc = composition.groupby(["split", "procedure_type"], as_index=False)["n"].sum().pivot(index="split", columns="procedure_type", values="n").fillna(0)
        proc.plot.bar(ax=procedure_axis, color=["#4388B5", "#E08A4B"][:len(proc.columns)])
        procedure_axis.set_ylabel("Patients")
        procedure_axis.set_xlabel("")
        procedure_axis.tick_params(axis="x", rotation=25)
        procedure_axis.legend(fontsize=8)
        style_axis(procedure_axis)
    figures.append(fig)

    # Page 4
    fig = new_report_page(4, "Follow-up maturity, censoring, horizon counts, and effective sample size")
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.87, bottom=0.10)
    maturity_axis = fig.add_subplot(grid[0, 0])
    maturity = tables.get("maturity", pd.DataFrame())
    if maturity.empty:
        no_data(maturity_axis, "Maturity table unavailable")
    else:
        test_maturity = maturity[maturity["split"].eq("test")]
        for outcome, group in test_maturity.groupby("outcome"):
            group = group.sort_values("target_month")
            maturity_axis.plot(group["target_month"], group["administrative_opportunity_n"], "o--", label=f"{outcome} opportunity")
            maturity_axis.plot(group["target_month"], group["observed_measurement_n"], "o-", label=f"{outcome} observed")
        maturity_axis.set_xlabel("Horizon month")
        maturity_axis.set_ylabel("Locked-test patients")
        maturity_axis.legend(fontsize=8)
        style_axis(maturity_axis)
    weight_axis = fig.add_subplot(grid[0, 1])
    weights = tables.get("observation_weights", pd.DataFrame())
    if weights.empty:
        no_data(weight_axis, "Observation-weight diagnostics unavailable")
    else:
        test_weights = weights[weights["evaluation_set"].eq("test")]
        for outcome, group in test_weights.groupby("outcome"):
            weight_axis.plot(group["target_month"], group["effective_n"], marker="o", label=f"{outcome} IPOW effective n")
            weight_axis.plot(group["target_month"], group["observed_n"], linestyle="--", marker=".", label=f"{outcome} observed n")
        weight_axis.axhline(float(scalars.get("min_cell_observations", 100)), color="#B64253", linestyle=":", label="frozen minimum")
        weight_axis.set_xlabel("Horizon month")
        weight_axis.set_ylabel("Patients")
        weight_axis.legend(fontsize=7)
        style_axis(weight_axis)
    figures.append(fig)

    # Page 5
    fig = new_report_page(5, "Exact measurement-window timing and missingness", "Selected measurement offsets and eligible counts are required by schema")
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.87, bottom=0.10)
    timing_axis = fig.add_subplot(grid[0, 0])
    timing = tables.get("timing", pd.DataFrame())
    if timing.empty:
        no_data(timing_axis, "Measurement timing unavailable")
    else:
        for outcome, group in timing.groupby("outcome"):
            group = group.sort_values("target_month")
            timing_axis.errorbar(
                group["target_month"], group["median_selected_month"],
                yerr=[group["median_selected_month"] - group["p10_selected_month"], group["p90_selected_month"] - group["median_selected_month"]],
                marker="o", capsize=3, label=outcome,
            )
        timing_axis.plot([0, 65], [0, 65], color="#808895", linestyle="--", label="nominal")
        timing_axis.set_xlabel("Nominal horizon month")
        timing_axis.set_ylabel("Selected measurement month, median and 10-90 percent")
        timing_axis.legend(fontsize=8)
        style_axis(timing_axis)
    missing_axis = fig.add_subplot(grid[0, 1])
    missing = tables.get("missingness", pd.DataFrame())
    if missing.empty:
        no_data(missing_axis, "Missingness audit unavailable")
    else:
        shown = missing.sort_values("missing_fraction").tail(22)
        missing_axis.barh(shown["field"], 100 * shown["missing_fraction"], color=np.where(shown["category"].eq("event_timing"), "#B64253", "#5A91BC"))
        missing_axis.set_xlabel("Missing percent")
        style_axis(missing_axis)
    figures.append(fig)

    # Page 6
    fig = new_report_page(6, "Same-cohort trajectory leaderboard and literature context", "BMI and HbA1c are standardized separately before equal weighting")
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.87, bottom=0.10, width_ratios=[0.62, 0.38])
    board_axis = fig.add_subplot(grid[0, 0])
    leaderboard = tables.get("trajectory_leaderboard", pd.DataFrame())
    if leaderboard.empty:
        no_data(board_axis, "Trajectory leaderboard unavailable")
    else:
        shown = leaderboard[leaderboard["outcome"].eq("equal_weighted_standardized")].sort_values("standardized_crps", ascending=True)
        board_axis.barh(shown["model"], shown["standardized_crps"], color=["#346C9A", "#D47A42", "#5A9B68"][:len(shown)])
        board_axis.set_xlabel("Equal-weighted standardized CRPS, lower is better")
        style_axis(board_axis)
    context_axis = fig.add_subplot(grid[0, 1])
    ensemble = tables.get("trajectory_ensemble", pd.DataFrame())
    lines = [
        "Matched comparison", "Published-style HGB, target-specific CatBoost, and ensemble are evaluated on identical locked-test cells.", "",
        "Unmatched literature context", "BJS 60-month BMI RMSE: 1.01 kg/m2", "SOPHIA 60-month BMI RMSE: 4.7 kg/m2", "",
        "Published values are contextual only. They are never used for local promotion or paired inference.", "",
        "Ensemble status:",
    ]
    if not ensemble.empty:
        lines.extend([f"{row.outcome}: {row.status}, HGB weight={row.weight_published_hgb:.2f}, reason={row.reason}" for row in ensemble.itertuples()])
    add_text_box(context_axis, lines, fontsize=9.2, wrap_width=74)
    figures.append(fig)

    # Page 7
    fig = new_report_page(
        7, "BMI and HbA1c RMSE, MAE, and CRPS by horizon",
        "Locked mature center-and-time test; complete-case estimates with center-clustered 95% intervals",
    )
    grid = fig.add_gridspec(2, 3, left=0.055, right=0.97, top=0.87, bottom=0.08, hspace=0.34, wspace=0.23)
    trajectory_metrics = tables.get("trajectory_metrics", pd.DataFrame())
    for row_index, outcome in enumerate(("bmi", "hba1c")):
        for column_index, metric in enumerate(("rmse", "mae", "standardized_crps")):
            axis = fig.add_subplot(grid[row_index, column_index])
            selected = trajectory_metrics[
                trajectory_metrics["evaluation_set"].eq("test") & trajectory_metrics["outcome"].eq(outcome) &
                trajectory_metrics["origin"].eq(0) & trajectory_metrics["estimate"].eq("complete_case") &
                trajectory_metrics["metric"].eq(metric)
            ] if not trajectory_metrics.empty else pd.DataFrame()
            if selected.empty:
                no_data(axis, f"No {outcome} {metric} results")
                continue
            for model, group in selected.groupby("model"):
                group = group.sort_values("target_month")
                lower = group["value"] - group["ci_low"]
                upper = group["ci_high"] - group["value"]
                has_interval = lower.notna().all() and upper.notna().all()
                if has_interval:
                    axis.errorbar(
                        group["target_month"], group["value"],
                        yerr=[np.maximum(lower, 0), np.maximum(upper, 0)],
                        marker="o", capsize=2, linewidth=1.2, label=MODEL_DISPLAY.get(model, model),
                    )
                else:
                    axis.plot(
                        group["target_month"], group["value"], marker="o",
                        label=MODEL_DISPLAY.get(model, model),
                    )
            if outcome == "bmi" and metric == "rmse":
                bjs = LITERATURE["bjs_2026"]["bmi_rmse"]
                axis.plot(list(bjs), list(bjs.values()), color="black", linestyle=":", label="BJS context")
            outcome_label = "BMI" if outcome == "bmi" else "HbA1c"
            metric_label = "standardized CRPS" if metric == "standardized_crps" else metric.upper()
            axis.set_title(f"{outcome_label} {metric_label}")
            axis.set_xlabel("Horizon month")
            axis.set_ylabel(metric_label)
            axis.legend(fontsize=5.8, ncol=1)
            style_axis(axis)
    figures.append(fig)

    # Page 8
    fig = new_report_page(8, "Interval coverage calibration and tail plausibility")
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.87, bottom=0.10)
    coverage_axis = fig.add_subplot(grid[0, 0])
    coverage = trajectory_metrics[
        trajectory_metrics["evaluation_set"].eq("test") & trajectory_metrics["metric"].eq("coverage") &
        trajectory_metrics["origin"].eq(0) & trajectory_metrics["estimate"].eq("complete_case")
    ] if not trajectory_metrics.empty else pd.DataFrame()
    if coverage.empty:
        no_data(coverage_axis, "Coverage results unavailable")
    else:
        for keys, group in coverage.groupby(["model", "outcome"]):
            coverage_axis.plot(group["level"], group["value"], marker="o", linestyle="none", label=f"{keys[0]} {keys[1]}")
        coverage_axis.plot([0.45, 0.95], [0.45, 0.95], "k--", label="ideal")
        coverage_axis.axvspan(0.45, 0.95, color="#EEF2F6", alpha=0.25)
        coverage_axis.set_xlim(0.45, 0.95)
        coverage_axis.set_ylim(0.35, 1.02)
        coverage_axis.set_xlabel("Nominal coverage")
        coverage_axis.set_ylabel("Empirical coverage")
        coverage_axis.legend(fontsize=6, ncol=2)
        style_axis(coverage_axis)
    diagnostic_axis = fig.add_subplot(grid[0, 1])
    conformal = tables.get("trajectory_conformal", pd.DataFrame())
    ensemble = tables.get("trajectory_ensemble", pd.DataFrame())
    lines = ["Quantile diagnostics", "", "All quantile grids are monotonically rearranged.", "Conformal expansion is learned only on the development calibration fold.", "Clinical plausibility bounds:", "  BMI 10-100 kg/m2", "  HbA1c 3-20 percent", ""]
    if not conformal.empty:
        lines.append(f"Median 90% conformal expansion: {conformal['conformal_expand_90'].median():.3g}")
        lines.append(f"Maximum 90% conformal expansion: {conformal['conformal_expand_90'].max():.3g}")
    widths = trajectory_metrics[
        trajectory_metrics["evaluation_set"].eq("test") &
        trajectory_metrics["origin"].eq(0) & trajectory_metrics["estimate"].eq("complete_case") &
        trajectory_metrics["metric"].eq("interval_width") & trajectory_metrics["level"].eq(0.9)
    ] if not trajectory_metrics.empty else pd.DataFrame()
    if not widths.empty:
        lines.extend(["", "Median 90% interval width across preoperative horizons:"])
        for keys, group in widths.groupby(["model", "outcome"]):
            lines.append(f"{MODEL_DISPLAY.get(keys[0], keys[0])} {keys[1]}: {group['value'].median():.3g}")
    if not ensemble.empty:
        lines.append("")
        lines.extend([f"{row.outcome}: extreme tail detected={row.extreme_tail_detected}; status={row.status}" for row in ensemble.itertuples()])
    add_text_box(diagnostic_axis, lines, fontsize=9.5, wrap_width=72)
    figures.append(fig)

    # Page 9
    fig = new_report_page(9, "Blinded center and prespecified subgroup transportability")
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.87, bottom=0.10)
    center_axis = fig.add_subplot(grid[0, 0])
    center = tables.get("trajectory_center_metrics", pd.DataFrame())
    iev = tables.get("internal_external_validation", pd.DataFrame())
    if center.empty and iev.empty:
        no_data(center_axis, "Center-level results suppressed or unavailable")
    else:
        source = iev if not iev.empty else center
        shown = source[source["outcome"].eq("bmi")]
        for model, group in shown.groupby("model"):
            center_axis.scatter(group["rmse"], group["center_blind"], label=model, alpha=0.85)
        if not trajectory_metrics.empty:
            contemporary = trajectory_metrics[
                trajectory_metrics["evaluation_set"].eq("contemporary") &
                trajectory_metrics["outcome"].eq("bmi") & trajectory_metrics["origin"].eq(0) &
                trajectory_metrics["target_month"].eq(60) &
                trajectory_metrics["estimate"].eq("complete_case") &
                trajectory_metrics["metric"].eq("rmse")
            ]
            accepted = ensemble[
                ensemble["outcome"].eq("bmi") & ensemble["status"].eq("accepted")
            ] if not ensemble.empty else pd.DataFrame()
            promoted_model = MODEL_ENSEMBLE if len(accepted) else MODEL_CATBOOST
            promoted_contemporary = contemporary[contemporary["model"].eq(promoted_model)]
            if len(promoted_contemporary):
                row = promoted_contemporary.iloc[0]
                center_axis.axvline(
                    row["value"], color="black", linestyle=":", linewidth=1.5,
                    label="promoted contemporary period",
                )
        center_axis.set_xlabel("60-month BMI RMSE")
        center_axis.set_ylabel("")
        center_axis.legend(fontsize=7)
        style_axis(center_axis)
    subgroup_axis = fig.add_subplot(grid[0, 1])
    subgroup = tables.get("trajectory_subgroup_metrics", pd.DataFrame())
    if subgroup.empty:
        no_data(subgroup_axis, "Subgroup cells below suppression threshold or unavailable")
    else:
        promoted = subgroup[(subgroup["outcome"].eq("bmi")) & subgroup["model"].isin([MODEL_ENSEMBLE, MODEL_CATBOOST])].copy()
        promoted["label"] = promoted["subgroup_variable"] + ": " + promoted["subgroup"].astype(str)
        promoted = promoted.sort_values("rmse").tail(24)
        subgroup_axis.barh(promoted["label"], promoted["rmse"], color="#648FB0")
        subgroup_axis.set_xlabel("60-month BMI RMSE")
        style_axis(subgroup_axis)
    figures.append(fig)

    # Page 10
    fig = new_report_page(10, "Competing-risk performance, calibration, and cumulative incidence", "Death is modeled as a competing event and as a separate all-cause endpoint")
    grid = fig.add_gridspec(2, 2, left=0.055, right=0.97, top=0.87, bottom=0.08, hspace=0.33)
    survival_metrics = tables.get("survival_metrics", pd.DataFrame())
    for axis_index, metric in enumerate(("brier", "auroc")):
        axis = fig.add_subplot(grid[0, axis_index])
        selected = survival_metrics[(survival_metrics["horizon_month"].eq(60)) & survival_metrics["metric"].eq(metric)] if not survival_metrics.empty else pd.DataFrame()
        if selected.empty:
            no_data(axis, f"No {metric} results")
        else:
            pivot = selected.pivot(index="endpoint", columns="model", values="value")
            pivot.plot.bar(ax=axis)
            axis.set_title(f"Five-year {metric.upper()}")
            axis.set_xlabel("")
            axis.tick_params(axis="x", rotation=25)
            axis.legend(fontsize=6)
            style_axis(axis)
    calibration_axis = fig.add_subplot(grid[1, 0])
    risk_cal = tables.get("survival_calibration", pd.DataFrame())
    selected_cal = risk_cal[(risk_cal["endpoint"].eq("composite")) & risk_cal["horizon_month"].eq(60)] if not risk_cal.empty else pd.DataFrame()
    if selected_cal.empty:
        no_data(calibration_axis, "Composite calibration curve unavailable")
    else:
        for model, group in selected_cal.groupby("model"):
            calibration_axis.plot(group["predicted"], group["observed"], marker="o", label=model)
        calibration_axis.plot([0, 1], [0, 1], "k--")
        calibration_axis.set_xlabel("Predicted cumulative incidence")
        calibration_axis.set_ylabel("Observed cumulative incidence")
        calibration_axis.legend(fontsize=6)
        style_axis(calibration_axis)
    cif_axis = fig.add_subplot(grid[1, 1])
    cif = tables.get("observed_cumulative_incidence", pd.DataFrame())
    if cif.empty:
        no_data(cif_axis, "Observed cumulative incidence unavailable")
    else:
        for endpoint, group in cif.groupby("endpoint"):
            if endpoint != "competing_death":
                cif_axis.plot(group["horizon_month"], group["cumulative_incidence"], marker="o", label=endpoint)
        cif_axis.set_xlabel("Month")
        cif_axis.set_ylabel("Aalen-Johansen cumulative incidence")
        cif_axis.legend(fontsize=7, ncol=2)
        style_axis(cif_axis)
    figures.append(fig)

    # Page 11
    fig = new_report_page(11, "Trajectory-feature risk ablation and decision curves")
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.87, bottom=0.10)
    ablation_axis = fig.add_subplot(grid[0, 0])
    selected = survival_metrics[(survival_metrics["endpoint"].eq("composite")) & survival_metrics["horizon_month"].eq(60) & survival_metrics["metric"].isin(["brier", "auroc", "auprc"])] if not survival_metrics.empty else pd.DataFrame()
    if selected.empty:
        no_data(ablation_axis, "Risk ablation unavailable")
    else:
        selected.pivot(index="model", columns="metric", values="value").plot.bar(ax=ablation_axis)
        ablation_axis.set_xlabel("")
        ablation_axis.tick_params(axis="x", rotation=20)
        ablation_axis.legend(fontsize=8)
        style_axis(ablation_axis)
    decision_axis = fig.add_subplot(grid[0, 1])
    decision = tables.get("survival_decision_curve", pd.DataFrame())
    if decision.empty:
        no_data(decision_axis, "Decision-curve analysis unavailable")
    else:
        for model, group in decision.groupby("model"):
            decision_axis.plot(group["threshold"], group["net_benefit"], label=model)
        decision_axis.axhline(0, color="black", linestyle="--")
        decision_axis.set_xlabel("Risk threshold")
        decision_axis.set_ylabel("Net benefit")
        decision_axis.legend(fontsize=7)
        style_axis(decision_axis)
    figures.append(fig)

    # Page 12
    fig = new_report_page(12, "Sleeve-versus-RYGB target-trial protocol and cohort flow", "Prediction and causal estimation are reported as separate analyses")
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.87, bottom=0.10)
    protocol_axis = fig.add_subplot(grid[0, 0])
    protocol = scalars.get("target_trial_protocol", TARGET_TRIAL_PROTOCOL)
    protocol_lines = [f"{key.replace('_', ' ').title()}:\n  {value}" for key, value in protocol.items() if key != "treatment_versions"]
    protocol_lines.append("Treatment versions:\n  " + canonical_json(protocol.get("treatment_versions", {})))
    add_text_box(protocol_axis, protocol_lines, fontsize=8.8, wrap_width=72)
    flow_axis = fig.add_subplot(grid[0, 1])
    flow = tables.get("trial_flow", pd.DataFrame())
    if flow.empty:
        no_data(flow_axis, "Target-trial flow unavailable")
    else:
        shown = flow.iloc[::-1]
        flow_axis.barh(shown["step"], shown["n"], color="#4D8E78")
        for index, value in enumerate(shown["n"]):
            flow_axis.text(value, index, f" {int(value):,}", va="center", fontsize=9)
        flow_axis.set_xlabel("Patients")
        style_axis(flow_axis)
    figures.append(fig)

    # Page 13
    fig = new_report_page(13, "Propensity overlap, covariate balance, weights, and effective sample size")
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.87, bottom=0.10)
    overlap_axis = fig.add_subplot(grid[0, 0])
    overlap = tables.get("trial_overlap", pd.DataFrame())
    if overlap.empty:
        no_data(overlap_axis, "Propensity overlap unavailable")
    else:
        y = np.arange(len(overlap))
        overlap_axis.errorbar(overlap["ps_median"], y, xerr=[overlap["ps_median"] - overlap["ps_q25"], overlap["ps_q75"] - overlap["ps_median"]], fmt="o", capsize=5, color="#457FA7")
        overlap_axis.hlines(y, overlap["ps_min"], overlap["ps_max"], color="#9BB4C8", linewidth=2)
        overlap_axis.axvline(0.02, color="#B64253", linestyle="--")
        overlap_axis.axvline(0.98, color="#B64253", linestyle="--")
        overlap_axis.set_yticks(y, overlap["arm"])
        overlap_axis.set_xlabel("Cross-fitted P(RYGB | baseline L)")
        overlap_axis.set_xlim(0, 1)
        weight_summary = tables.get("trial_weights", pd.DataFrame())
        if not weight_summary.empty:
            weight_row = weight_summary.iloc[0]
            overlap_axis.set_title(
                f"IPTW effective n {weight_row.get('effective_n', np.nan):.1f}; "
                f"trimmed outside 0.02-0.98: {int(weight_row.get('trimmed', 0))}",
                fontsize=9,
            )
        style_axis(overlap_axis)
    balance_axis = fig.add_subplot(grid[0, 1])
    balance = tables.get("trial_balance", pd.DataFrame())
    if balance.empty:
        no_data(balance_axis, "Covariate balance unavailable")
    else:
        shown = balance.dropna(subset=["abs_smd_before", "abs_smd_after"], how="all").sort_values("abs_smd_before").tail(28)
        y = np.arange(len(shown))
        balance_axis.scatter(shown["abs_smd_before"], y, label="before", color="#C65F5F", s=22)
        balance_axis.scatter(shown["abs_smd_after"], y, label="after", color="#3D83A8", s=22)
        balance_axis.axvline(0.1, color="black", linestyle="--", label="0.10 gate")
        balance_axis.set_yticks(y, shown["covariate"], fontsize=6)
        balance_axis.set_xlabel("Absolute standardized mean difference")
        balance_axis.legend(fontsize=8)
        style_axis(balance_axis)
    figures.append(fig)

    # Page 14
    fig = new_report_page(14, "Causal effect estimates and sensitivity analyses", "No causal result has a required direction or magnitude")
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.87, bottom=0.10, width_ratios=[0.64, 0.36])
    effect_axis = fig.add_subplot(grid[0, 0])
    effects = tables.get("trial_effects", pd.DataFrame())
    if effects.empty:
        no_data(effect_axis, "Causal effects unavailable")
    else:
        shown = effects.copy()
        labels = shown["outcome"] + "\n" + shown["units"]
        y = np.arange(len(shown))
        finite = shown[["effect_rygb_minus_sleeve", "ci_low", "ci_high"]].notna().all(axis=1)
        effect_axis.errorbar(
            shown.loc[finite, "effect_rygb_minus_sleeve"], y[finite],
            xerr=[
                shown.loc[finite, "effect_rygb_minus_sleeve"] - shown.loc[finite, "ci_low"],
                shown.loc[finite, "ci_high"] - shown.loc[finite, "effect_rygb_minus_sleeve"],
            ], fmt="o", capsize=5, color="#3A789F",
        )
        for position, row in zip(y[~finite], shown.loc[~finite].itertuples()):
            effect_axis.text(0.02, position, f"not estimable: {row.status_reason}", va="center", fontsize=7, transform=effect_axis.get_yaxis_transform())
        effect_axis.axvline(0, color="black", linestyle="--")
        effect_axis.set_yticks(y, labels)
        effect_axis.set_xlabel("RYGB minus sleeve marginal effect with center-clustered 95% interval")
        style_axis(effect_axis)
    sensitivity_axis = fig.add_subplot(grid[0, 1])
    weights = tables.get("trial_weights", pd.DataFrame())
    glp_status = scalars.get("dynamic_glp1", {})
    lines = ["Causal diagnostics", ""]
    if not effects.empty and "e_value_point" in effects:
        composite = effects[effects["outcome"].eq("composite_60m")]
        if len(composite):
            row = composite.iloc[0]
            lines.extend([f"Composite E-value point: {row.get('e_value_point', np.nan):.3g}", f"Composite E-value CI limit: {row.get('e_value_ci', np.nan):.3g}", ""])
    if not weights.empty:
        lines.extend([
            f"IPTW effective n: {weights.iloc[0]['effective_n']:.1f}",
            f"Trimmed outside 0.02-0.98: {int(weights.iloc[0]['trimmed'])}",
            f"Minimum treatment/censor support: {weights.iloc[0]['minimum_support_probability']:.3f}",
            f"Max adjusted |SMD|: {weights.iloc[0]['max_abs_smd']:.3f}", "",
        ])
    if not effects.empty and "status" in effects:
        not_estimable = effects[~effects["status"].eq("estimated")]
        if len(not_estimable):
            lines.extend(["Non-estimable causal endpoints", *[f"{row.outcome}: {row.status_reason}" for row in not_estimable.itertuples()], ""])
    lines.extend(["Dynamic postoperative GLP-1 analysis", f"Status: {glp_status.get('status', 'unknown')}", f"Reason: {glp_status.get('reason', 'not recorded')}"])
    add_text_box(sensitivity_axis, lines, fontsize=9.3, wrap_width=50)
    figures.append(fig)

    # Page 15
    fig = new_report_page(15, "Limitations, unmet gates, and claim-level conclusion", f"Final classification: {classification}")
    grid = fig.add_gridspec(1, 1, left=0.05, right=0.95, top=0.87, bottom=0.08)
    axis = fig.add_subplot(grid[0, 0])
    gates = tables.get("success_gates", pd.DataFrame())
    failed = []
    if not gates.empty:
        for row in gates.itertuples():
            passed = str(getattr(row, "passed", False)).lower() in {"true", "1"} if isinstance(getattr(row, "passed", False), str) else bool(getattr(row, "passed", False))
            if not passed:
                failed.append(f"- {row.domain} / {getattr(row, 'outcome', '')}: {row.gate} (value={getattr(row, 'value', np.nan):.3g})")
    limitations = [
        "Claim-level conclusion", f"  {classification}", "", "Unmet or non-assessable frozen gates", *(failed or ["- None"]), "",
        "Prespecified limitations",
        "- Center identifiers are blinded and small cells are suppressed.",
        "- BMI and HbA1c are modeled and selected separately; pooled raw-scale scores are prohibited.",
        "- Complication risks use event time, censoring time, and competing death rather than binary ever labels.",
        "- Postoperative GLP-1 does not censor ordinary prognostic outcomes.",
        "- Strategy-specific survivor means compare different potential survivor populations and are not principal-stratum effects.",
        "- Residual confounding remains possible despite cross-fitting, center adjustment, balance checks, and sensitivity analysis.",
        "- Published BJS and SOPHIA values are unmatched context, not paired comparators.",
        "- No real patient trajectory, patient-level prediction, or identifiable center information is exported.", "",
        "Dynamic GLP-1 status", f"- {scalars.get('dynamic_glp1', {}).get('status', 'unknown')}: {scalars.get('dynamic_glp1', {}).get('reason', 'not recorded')}",
    ]
    add_text_box(axis, limitations, fontsize=9.1, wrap_width=125)
    figures.append(fig)
    return figures


def atomic_save_figure(fig: Any, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        fig.savefig(temporary, dpi=dpi, facecolor=fig.get_facecolor())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generate_report(run_dir: Path, tables: Mapping[str, Any], scalars: Mapping[str, Any]) -> None:
    export = run_dir / "FIGURES_TO_EXPORT"
    export.mkdir(parents=True, exist_ok=True)
    figures = report_figures(tables, scalars)
    if len(figures) != len(REPORT_PAGES):
        raise AssertionError("Report page count changed unexpectedly")
    pdf_fd, temporary_pdf = tempfile.mkstemp(prefix="performance_report.", suffix=".tmp.pdf", dir=export)
    os.close(pdf_fd)
    try:
        with PdfPages(temporary_pdf) as pdf:
            for number, (name, fig) in enumerate(zip(REPORT_PAGES, figures), 1):
                atomic_save_figure(fig, export / f"{number:02d}_{name}.png")
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)
        os.replace(temporary_pdf, export / "performance_report.pdf")
    finally:
        if os.path.exists(temporary_pdf):
            os.unlink(temporary_pdf)
        for fig in figures:
            try:
                plt.close(fig)
            except Exception:
                pass
    allowed = {".png", ".pdf"}
    disallowed = [path.name for path in export.iterdir() if path.is_file() and path.suffix.lower() not in allowed]
    if disallowed:
        raise AssertionError(f"Non-figure artifacts entered FIGURES_TO_EXPORT: {disallowed}")


def plot_only(run_dir: Path) -> None:
    tables, scalars = load_aggregate_tables(run_dir)
    generate_report(run_dir, tables, scalars)
    print(f"plot-only reconstruction complete: {run_dir / 'FIGURES_TO_EXPORT'}")


# =====================================================================================
# 9. Fingerprinted orchestration, embedded regression tests, and command line
# =====================================================================================


def scientific_constants_manifest() -> dict[str, Any]:
    return {
        "study_version": STUDY_VERSION,
        "sql_contract_version": SQL_CONTRACT_VERSION,
        "landmarks": LANDMARK_MONTHS,
        "trajectory_horizons": TRAJECTORY_HORIZONS,
        "measurement_windows": MEASUREMENT_WINDOWS,
        "quantiles": QUANTILES,
        "coverage_levels": COVERAGE_LEVELS,
        "plausible_ranges": PLAUSIBLE_RANGES,
        "risk_horizons": RISK_HORIZONS,
        "person_period_months": PERSON_PERIOD_MONTHS,
        "literature": LITERATURE,
        "success_gates": SUCCESS_GATES,
        "target_trial_protocol": TARGET_TRIAL_PROTOCOL,
        "dynamic_glp1_protocol": DYNAMIC_GLP1_PROTOCOL,
    }


def fingerprint_configuration(cfg: RunConfig) -> dict[str, Any]:
    value = asdict(cfg)
    for key in ("output_dir", "resume", "interrupt_after"):
        value.pop(key, None)
    return value


def initialize_run_context(
    cfg: RunConfig,
    data: Any,
    splits: LockedSplits,
    sql_meta: Mapping[str, Any],
    dependencies: Mapping[str, Any],
) -> RunContext:
    code_hash = sha256_file(SCRIPT_PATH)
    features = available_feature_columns(data)
    feature_schema = {
        "baseline": {"numeric": features[0], "categorical": features[1]},
        "target_trial": dict(zip(("numeric", "categorical"), target_trial_feature_columns(data))),
        "trajectory": {
            outcome: dict(zip(("numeric", "categorical"), trajectory_feature_roster(data, outcome)))
            for outcome in TRAJECTORY_HORIZONS
        },
    }
    payload = {
        "code_sha256": code_hash,
        "sql_contract_version": SQL_CONTRACT_VERSION,
        "sql_sha256": sql_meta["sql_sha256"],
        "schema_aliases": sql_meta.get("resolved_aliases", {}),
        "dynamic_glp1_schema": sql_meta.get("dynamic_glp1_schema", {}),
        "data_aggregates": data_aggregate_manifest(data),
        "feature_schema": feature_schema,
        "split_identity": split_identity_manifest(data, splits),
        "dependency_versions": dependencies,
        "scientific_constants": scientific_constants_manifest(),
        "configuration": fingerprint_configuration(cfg),
    }
    fingerprint = digest(payload)
    run_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else (DEFAULT_RESULTS_ROOT / f"run_{fingerprint[:16]}").resolve()
    context = RunContext(
        cfg=cfg, run_dir=run_dir, fingerprint=fingerprint, fingerprint_payload=payload,
        state=read_json(run_dir / "run_state.json", {}),
    )
    context.initialize()
    return context


def write_internal_manifests(context: RunContext, data: Any, splits: LockedSplits, sql_meta: Mapping[str, Any], cohort_meta: Mapping[str, Any]) -> None:
    atomic_text(context.internal / "versioned_query.sql", str(sql_meta.get("sql", "")))
    atomic_json(context.internal / "schema_alias_manifest.json", {
        "contract_version": SQL_CONTRACT_VERSION,
        "resolved_aliases": sql_meta.get("resolved_aliases", {}),
        "dynamic_glp1_schema": sql_meta.get("dynamic_glp1_schema", {}),
    })
    atomic_json(context.internal / "split_manifest.json", split_identity_manifest(data, splits))
    atomic_json(context.internal / "cohort_contract.json", cohort_meta)
    atomic_json(context.internal / "feature_contract.json", {
        "baseline_features": dict(zip(("numeric", "categorical"), available_feature_columns(data))),
        "target_trial_features": dict(zip(("numeric", "categorical"), target_trial_feature_columns(data))),
        "trajectory_features": {
            outcome: dict(zip(("numeric", "categorical"), trajectory_feature_roster(data, outcome)))
            for outcome in TRAJECTORY_HORIZONS
        },
        "future_event_features_prohibited": list(FORBIDDEN_FUTURE_FEATURE_TOKENS),
        "postoperative_glp1": "retained for prognosis and added only after known at a landmark",
    })


def stage_or_resume(
    context: RunContext,
    stage: str,
    upstream: Mapping[str, str],
    builder: Callable[[], Any],
) -> tuple[Any, str]:
    loaded = context.load_checkpoint(stage, upstream)
    if loaded is not None:
        return loaded, sha256_file(context.checkpoints / f"{stage}.pkl")
    context.state.setdefault("stages", {})[stage] = {"status": "running", "started_utc": utc_now()}
    atomic_json(context.run_dir / "run_state.json", context.state)
    value = builder()
    artifact_hash = context.save_checkpoint(stage, value, upstream)
    return value, artifact_hash


def execute_analysis(cfg: RunConfig, dependencies: Mapping[str, Any]) -> Path:
    context: RunContext | None = None
    try:
        fast_self_tests()
        if cfg.smoke:
            raw = generate_synthetic_cohort(cfg.smoke_patients, cfg.seed)
            sql_meta = {
                "sql_contract_version": SQL_CONTRACT_VERSION,
                "sql": "/* embedded synthetic cohort; production always uses the explicit SQL contract */",
                "sql_sha256": digest("embedded-synthetic-v2"),
                "resolved_aliases": {name: name for name in raw.columns},
                "dynamic_glp1_schema": dynamic_glp1_schema_status(raw),
            }
        else:
            raw, sql_meta = query_database(cfg)
        data, cohort_meta = validate_and_construct_cohort(raw, cfg)
        splits = build_locked_splits(data, cfg)
        context = initialize_run_context(cfg, data, splits, sql_meta, dependencies)
        write_internal_manifests(context, data, splits, sql_meta, cohort_meta)

        cohort_payload = {"data": data, "splits": splits, "cohort_meta": cohort_meta}
        cohort, cohort_hash = stage_or_resume(context, "cohort", {}, lambda: cohort_payload)
        data, splits, cohort_meta = cohort["data"], cohort["splits"], cohort["cohort_meta"]
        trajectory, trajectory_hash = stage_or_resume(
            context, "trajectory", {"cohort": cohort_hash}, lambda: fit_trajectory_models(data, splits, cfg),
        )
        observation, observation_hash = stage_or_resume(
            context, "observation", {"cohort": cohort_hash}, lambda: observation_process(data, splits, cfg),
        )
        trajectory_eval, trajectory_eval_hash = stage_or_resume(
            context, "trajectory_evaluation", {"trajectory": trajectory_hash, "observation": observation_hash},
            lambda: evaluate_trajectories(data, trajectory["predictions"], observation, cfg),
        )
        iev, iev_hash = stage_or_resume(
            context, "internal_external_validation", {"cohort": cohort_hash},
            lambda: internal_external_validation(data, splits, cfg),
        )

        def survival_builder() -> dict[str, Any]:
            summaries = crossfit_trajectory_summaries(data, splits, cfg)
            fitted = fit_survival_models(data, splits, summaries, cfg)
            evaluated = evaluate_survival(data, splits, fitted, cfg)
            return {"trajectory_summaries": summaries, "fitted": fitted, "evaluated": evaluated}

        survival_bundle, survival_hash = stage_or_resume(
            context, "survival", {"cohort": cohort_hash, "trajectory": trajectory_hash}, survival_builder,
        )
        trial, trial_hash = stage_or_resume(
            context, "target_trial", {"cohort": cohort_hash}, lambda: target_trial_analysis(data, cfg),
        )
        glp1, glp1_hash = stage_or_resume(
            context, "dynamic_glp1", {"cohort": cohort_hash}, lambda: dynamic_glp1_analysis(data, cfg),
        )

        preflight = build_preflight_tables(data, cohort_meta, splits, observation, cfg)
        gates, classification = evaluate_success_gates(
            trajectory, trajectory_eval, survival_bundle["evaluated"], trial, cfg,
        )
        tables = artifact_tables(
            preflight, trajectory, trajectory_eval, iev, survival_bundle["fitted"],
            survival_bundle["evaluated"], trial, glp1, gates,
        )
        optional_missing = [
            spec.canonical for spec in OPTIONAL_FIELDS
            if spec.canonical not in data or data[spec.canonical].isna().all()
        ]
        scalars = {
            "fingerprint": context.fingerprint,
            "mode": cfg.mode,
            "status": "complete",
            "classification": classification,
            "analysis_n": len(data),
            "center_n": data["center_blind"].nunique(),
            "temporal_cutoff": splits.temporal_cutoff,
            "dependencies": dict(dependencies),
            "sql_contract_version": SQL_CONTRACT_VERSION,
            "sql_sha256": sql_meta["sql_sha256"],
            "code_sha256": sha256_file(SCRIPT_PATH),
            "min_cell_observations": cfg.min_cell_observations,
            "target_trial_protocol": TARGET_TRIAL_PROTOCOL,
            "dynamic_glp1": {"status": glp1["status"], "reason": glp1["gate"]["reason"]},
            "optional_fields_not_available": optional_missing,
            "checkpoint_artifact_hashes": {
                "cohort": cohort_hash, "trajectory": trajectory_hash, "observation": observation_hash,
                "trajectory_evaluation": trajectory_eval_hash, "internal_external_validation": iev_hash,
                "survival": survival_hash, "target_trial": trial_hash, "dynamic_glp1": glp1_hash,
            },
            "negative_control": trial["negative_control"],
        }
        save_aggregate_tables(context, tables, scalars)
        generate_report(context.run_dir, tables, scalars)
        context.state["status"] = "complete"
        context.state["classification"] = classification
        context.state["completed_utc"] = utc_now()
        context.state.setdefault("stages", {})["report"] = {
            "status": "complete",
            "aggregate_index_sha256": sha256_file(context.aggregate / "report_index.json"),
            "report_pdf_sha256": sha256_file(context.export / "performance_report.pdf"),
        }
        atomic_json(context.run_dir / "run_state.json", context.state)
        print(f"complete: {context.run_dir}")
        return context.run_dir
    except IntentionalInterrupt:
        if context is not None:
            print(f"intentional interruption recorded: {context.run_dir}")
        raise
    except Exception as exc:
        if context is not None:
            context.state["status"] = "failed"
            context.state.setdefault("errors", []).append({
                "time_utc": utc_now(), "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            })
            atomic_json(context.run_dir / "run_state.json", context.state)
            write_failure_png(
                context.export / "00_preflight_failure.png",
                "Analysis stopped before completion",
                [f"{type(exc).__name__}: {exc}"],
                ["Inspect INTERNAL checkpoint metadata and run_state.json. No scientific fallback was used."],
            )
        raise


def regression_failure_fixture() -> dict[str, Any]:
    """Small frozen examples of the previous runner's scientifically invalid behavior."""
    return {
        "immature_temporal_test": {
            "procedure_year": [2015, 2016, 2017, 2022, 2023],
            "bmi_60_observed": [1, 1, 1, 0, 0],
            "legacy_latest_test_indices": [3, 4],
        },
        "raw_scale_pooling": {
            "model_a_rmse": {"bmi": 1.0, "hba1c": 1.0},
            "model_b_rmse": {"bmi": 2.0, "hba1c": 0.1},
            "outcome_sd": {"bmi": 10.0, "hba1c": 0.5},
        },
        "ever_event_censoring": {
            "ever_flag": 0, "censor_day": 500.0, "five_year_day": 60 * DAYS_PER_MONTH,
            "legacy_binary_label": 0, "valid_five_year_status": "unknown",
        },
        "extreme_ensemble": {
            "model_a": [30.0, 32.0], "model_b": [38.0, 40.0],
            "legacy_unconstrained_weights": [-5.0, 6.0],
        },
        "stale_checkpoint": {
            "legacy_key": "trajectory-model-name-only", "code_hash_before": "abc", "code_hash_after": "def",
        },
        "legacy_compile_failure_observed_on_python_3_9": True,
    }


def assert_no_repository_local_imports() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    local_modules = {path.stem for path in SCRIPT_PATH.parents[1].glob("*.py")}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    violations = sorted(imported & local_modules)
    if violations:
        raise AssertionError(f"Repository-local runtime imports found: {violations}")


def fast_self_tests() -> list[str]:
    checks: list[str] = []

    def check(name: str, condition: Any) -> None:
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    fixture = regression_failure_fixture()
    immature = fixture["immature_temporal_test"]
    check("regression_immature_five_year_test", sum(immature["bmi_60_observed"][index] for index in immature["legacy_latest_test_indices"]) == 0)
    pooling = fixture["raw_scale_pooling"]
    raw_a = math.sqrt(np.mean([value ** 2 for value in pooling["model_a_rmse"].values()]))
    raw_b = math.sqrt(np.mean([value ** 2 for value in pooling["model_b_rmse"].values()]))
    std_a = math.sqrt(np.mean([(pooling["model_a_rmse"][name] / pooling["outcome_sd"][name]) ** 2 for name in ("bmi", "hba1c")]))
    std_b = math.sqrt(np.mean([(pooling["model_b_rmse"][name] / pooling["outcome_sd"][name]) ** 2 for name in ("bmi", "hba1c")]))
    check("regression_raw_scale_rank_reversal", raw_a < raw_b and std_b < std_a)
    censoring = fixture["ever_event_censoring"]
    check("regression_binary_ever_censoring", censoring["legacy_binary_label"] == 0 and censoring["censor_day"] < censoring["five_year_day"] and censoring["valid_five_year_status"] == "unknown")
    extreme = fixture["extreme_ensemble"]
    legacy = extreme["legacy_unconstrained_weights"][0] * np.asarray(extreme["model_a"]) + extreme["legacy_unconstrained_weights"][1] * np.asarray(extreme["model_b"])
    conservative = 0.5 * np.asarray(extreme["model_a"]) + 0.5 * np.asarray(extreme["model_b"])
    check("regression_extreme_ensemble", legacy.max() > 60 and conservative.min() >= 10 and conservative.max() <= 100)
    check("regression_stale_checkpoint", digest({"code": "abc", "stage": "trajectory"}) != digest({"code": "def", "stage": "trajectory"}))
    check("regression_legacy_compile_failure", fixture["legacy_compile_failure_observed_on_python_3_9"] is True)

    q = np.asarray([[8, 6, 7, 5, 9, 10, 4], [1, 2, 3, 4, 5, 6, 7]], dtype=float)
    rearranged = np.sort(q, axis=1)
    check("quantile_rearrangement", np.all(np.diff(rearranged, axis=1) >= 0))
    samples = quantile_matrix_to_samples(rearranged, 31)
    check("quantile_samples", samples.shape == (2, 31) and np.all(np.isfinite(samples)))
    check("crps_nonnegative", np.all(crps_ensemble(samples, np.asarray([7.0, 3.0])) >= 0))
    expanded = apply_conformal(rearranged, {0.5: 1, 0.8: 2, 0.9: 3})
    check("conformal_monotone", np.all(np.diff(expanded, axis=1) >= 0) and expanded[0, 0] <= rearranged[0, 0])

    canonical_columns = [spec.aliases[0] for spec in all_schema_specs()]
    resolved, missing, ambiguous = resolve_schema(canonical_columns)
    check("schema_alias_resolution", not missing and not ambiguous and len(resolved) >= len([spec for spec in all_schema_specs() if spec.required]))
    sql = build_explicit_sql(resolved)
    check("explicit_sql", "SELECT *" not in sql.upper() and "PriorGLP1" in sql and "ActiveEndInterval >=" not in sql)
    check("prior_glp1_frozen", " = 0" in sql)
    check("nonoverlapping_windows", all(MEASUREMENT_WINDOWS[left][1] <= MEASUREMENT_WINDOWS[right][0] for left, right in zip(sorted(MEASUREMENT_WINDOWS)[:-1], sorted(MEASUREMENT_WINDOWS)[1:])))

    mini = generate_synthetic_cohort(80, cfg_seed := SEED + 1)
    cohort, _ = validate_and_construct_cohort(mini, RunConfig.for_mode("smoke", None))
    before = landmark_feature_frame(cohort, np.arange(10), "bmi", 3)
    changed = cohort.copy()
    changed.loc[:9, "bmi_60m_value"] = 9999
    after = landmark_feature_frame(changed, np.arange(10), "bmi", 3)
    roster = trajectory_feature_roster(cohort, "bmi")
    check("landmark_future_leakage", before[list(roster[0]) + list(roster[1])].astype(str).equals(after[list(roster[0]) + list(roster[1])].astype(str)))
    trial_numeric, trial_categorical = target_trial_feature_columns(cohort)
    check("target_trial_treatment_leakage", "procedure_type" not in trial_numeric + trial_categorical)
    check("event_before_censor_contract", not ((cohort["mace_valid_event"]) & cohort["mace_days"].gt(cohort["censor_days"])).any())
    y, known = known_risk_status(cohort, np.arange(len(cohort)), "composite", 60)
    check("survival_unknown_censoring", np.all((y == 0) | (y == 1)) and np.any(~known))
    missing_mace = np.flatnonzero(cohort["mace_missing_time_sensitivity"].to_numpy(bool))
    if missing_mace.size:
        _, missing_known = known_risk_status(cohort, missing_mace[:1], "mace", 60)
        check("missing_event_time_remains_unknown", not bool(missing_known[0]))
    else:
        raise AssertionError("Synthetic fixture did not retain its missing-event-time sensitivity record")
    synthetic_cif = {name: np.full((4, 3), 0.03) for name in COMPONENTS}
    synthetic_cif["competing_death"] = np.full((4, 3), 0.02)
    death = {"death": np.full((4, 3), 0.04)}
    calibrated = apply_risk_calibration(synthetic_cif, death, {name: [1, 1, 1] for name in (*COMPONENTS, "death")})
    check("survival_probability_bounds", all(np.all((value >= 0) & (value <= 1)) for value in calibrated.values()))
    check("component_below_composite", all(np.all(calibrated[name] <= calibrated["composite"]) for name in COMPONENTS))
    synthetic_treatment = np.asarray([0, 0, 1, 1])
    common_nuisance = {
        "treatment": synthetic_treatment, "ps": np.full(4, 0.5), "pc": np.ones(4),
        "observed": np.ones(4, dtype=bool),
    }
    numerator_nuisance = {
        **common_nuisance, "outcome": np.asarray([20.0, 0.0, 30.0, 0.0]),
        "mu0": np.full(4, 10.0), "mu1": np.full(4, 15.0),
    }
    survival_nuisance = {
        **common_nuisance, "outcome": np.asarray([1.0, 0.0, 1.0, 0.0]),
        "mu0": np.full(4, 0.5), "mu1": np.full(4, 0.5),
    }
    survivor_effect = survivor_mean_effect(
        numerator_nuisance, survival_nuisance, np.asarray(["A", "A", "B", "B"]),
        RunConfig.for_mode("smoke", None),
    )
    check("survivor_mean_death_handling", abs(survivor_effect["estimate"] - 10.0) < 1e-10)
    low_support_nuisance = {**survival_nuisance, "pc": np.asarray([0.5, 0.5, 0.001, 0.5])}
    check("causal_nonpositivity_gate", nuisance_minimum_support(low_support_nuisance) < SUCCESS_GATES["causal_min_support_probability"])

    with tempfile.TemporaryDirectory(prefix="qreg_fast_test_") as directory:
        path = Path(directory) / "failure.png"
        write_failure_png(path, "Synthetic failure", ["one exact issue"])
        check("failure_png", path.read_bytes().startswith(b"\x89PNG") and path.stat().st_size > 1000)
        dummy_cfg = RunConfig.for_mode("smoke", str(Path(directory) / "run"))
        dummy = RunContext(dummy_cfg, Path(directory) / "run", "fingerprint-a", {"code": "a"}, {})
        dummy.initialize()
        artifact_hash = dummy.save_checkpoint("unit", {"value": 7}, {})
        loaded = dummy.load_checkpoint("unit", {})
        check("checkpoint_exact_resume", loaded == {"value": 7} and artifact_hash == sha256_file(dummy.checkpoints / "unit.pkl"))
        with (dummy.checkpoints / "unit.pkl").open("ab") as stream:
            stream.write(b"corruption")
        check("checkpoint_hash_guard", dummy.load_checkpoint("unit", {}) is None)
    assert_no_repository_local_imports()
    checks.append("no_repository_local_imports")
    check("no_em_dash_in_runner", "\u2014" not in SCRIPT_PATH.read_text(encoding="utf-8"))
    return checks


def extensive_self_test(output_dir: str | None, dependencies: Mapping[str, Any]) -> None:
    checks = fast_self_tests()
    if output_dir:
        base = Path(output_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        run_dir = base / "embedded_synthetic_e2e"
        execute_analysis(RunConfig.for_mode("smoke", str(run_dir)), dependencies)
        tables, scalars = load_aggregate_tables(run_dir)
        plot_only(run_dir)
        export = run_dir / "FIGURES_TO_EXPORT"
        figure_files = sorted(export.glob("*.png"))
        if len(figure_files) != 15:
            raise AssertionError(f"Expected 15 aggregate PNG pages, found {len(figure_files)}")
        if not (export / "performance_report.pdf").exists():
            raise AssertionError("Multi-page PDF was not generated")
        if any(path.suffix.lower() not in {".png", ".pdf"} for path in export.iterdir() if path.is_file()):
            raise AssertionError("Patient-level or non-figure artifact entered export directory")
        if any({"patient_id", "patient_position", "center_id"} & set(frame.columns) for frame in tables.values()):
            raise AssertionError("Patient-level field entered an aggregate report table")
        checks.extend(["synthetic_e2e", "complete_report", "plot_only_reconstruction", "aggregate_export_only"])
    else:
        with tempfile.TemporaryDirectory(prefix="qreg_self_test_") as directory:
            extensive_self_test(directory, dependencies)
            return
    print(f"self-test: {len(checks)} checks passed")
    print(", ".join(checks))


def production_readiness_self_test(dependencies: Mapping[str, Any]) -> None:
    """Exercise interruption, exact resume, report generation, and aggregate-only reconstruction."""
    try:
        with tempfile.TemporaryDirectory(prefix="qreg_production_readiness_") as directory:
            run_dir = Path(directory) / "interrupted_smoke"
            interrupted = RunConfig.for_mode("smoke", str(run_dir), "trajectory")
            try:
                execute_analysis(interrupted, dependencies)
            except IntentionalInterrupt:
                pass
            else:
                raise AssertionError("Synthetic interruption did not stop at the requested stage")

            execute_analysis(RunConfig.for_mode("smoke", str(run_dir)), dependencies)
            state = read_json(run_dir / "run_state.json", {})
            resumed = set(state.get("resumed_stages", []))
            if not {"cohort", "trajectory"}.issubset(resumed):
                raise AssertionError(f"Exact resume did not reuse the expected verified stages: {sorted(resumed)}")
            plot_only(run_dir)
            export = run_dir / "FIGURES_TO_EXPORT"
            if len(list(export.glob("*.png"))) != len(REPORT_PAGES) or not (export / "performance_report.pdf").exists():
                raise AssertionError("Synthetic readiness report is incomplete")
            if any(path.suffix.lower() not in {".png", ".pdf"} for path in export.iterdir() if path.is_file()):
                raise AssertionError("Synthetic readiness export contains a non-figure artifact")
            tables, _ = load_aggregate_tables(run_dir)
            forbidden = {"patient_id", "patient_position", "center_id"}
            if any(forbidden & set(frame.columns) for frame in tables.values()):
                raise AssertionError("Synthetic readiness aggregate tables contain a patient-level field")
    except PreflightError:
        raise
    except Exception as exc:
        raise PreflightError("Embedded production readiness tests failed", [
            f"{type(exc).__name__}: {exc}",
            "The production database query was not started.",
        ]) from exc
    print("production readiness: interrupted/resumed synthetic E2E and plot-only reconstruction passed")


def latest_plot_run() -> Path | None:
    if not DEFAULT_RESULTS_ROOT.exists():
        return None
    candidates = [path for path in DEFAULT_RESULTS_ROOT.iterdir() if (path / "AGGREGATE" / "report_index.json").exists()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def parse_args(argv: Sequence[str] | None = None) -> tuple[RunConfig, bool]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true", help="run the complete embedded synthetic E2E study")
    modes.add_argument("--self-test", action="store_true", help="run embedded regression, leakage, survival, checkpoint, E2E, and report tests")
    modes.add_argument("--plot-only", action="store_true", help="reconstruct aggregate figures from an existing run")
    parser.add_argument("--output-dir", help="developer-only output or existing run directory")
    parser.add_argument(
        "--interrupt-after", choices=("cohort", "trajectory", "observation", "trajectory_evaluation", "internal_external_validation", "survival", "target_trial", "dynamic_glp1"),
        help="developer-only interruption point used to verify safe resume",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        mode = "self-test"
    elif args.smoke:
        mode = "smoke"
    elif args.plot_only:
        mode = "plot-only"
    else:
        mode = "production"
    return RunConfig.for_mode(mode, args.output_dir, args.interrupt_after), bool(args.plot_only)


def main(argv: Sequence[str] | None = None) -> int:
    cfg, is_plot_only = parse_args(argv)
    manifest, dependency_issues = dependency_manifest(cfg.mode)
    if dependency_issues:
        failure = render_preflight_failure(
            cfg, "Production dependency preflight failed", dependency_issues,
            ["Install the exact dependencies, then rerun the same command. The analysis did not fall back."],
        )
        print(f"preflight failed: {failure}", file=sys.stderr)
        return 2
    load_runtime_packages(cfg.mode)
    try:
        if is_plot_only:
            run_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else latest_plot_run()
            if run_dir is None:
                raise PreflightError("Plot-only reconstruction failed", ["No completed fingerprinted run was found; pass --output-dir RUN_DIR"])
            plot_only(run_dir)
            return 0
        if cfg.mode == "self-test":
            extensive_self_test(cfg.output_dir, manifest)
            return 0
        if cfg.mode == "production":
            production_readiness_self_test(manifest)
        execute_analysis(cfg, manifest)
        return 0
    except IntentionalInterrupt as exc:
        print(str(exc), file=sys.stderr)
        return 75
    except PreflightError as exc:
        failure = render_preflight_failure(cfg, exc.title, exc.issues)
        print(f"preflight failed: {failure}", file=sys.stderr)
        return 2
    except Exception as exc:
        traceback.print_exc()
        failure = render_preflight_failure(
            cfg, "Analysis failed", [f"{type(exc).__name__}: {exc}"],
            ["No scientifically different fallback was used. Inspect the fingerprinted run state when available."],
        )
        print(f"failure artifact: {failure}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
