#!/usr/bin/env python3
"""Metabolic trajectory forecasting study for Epic Cosmos.

The default command runs the production preflight, extraction, cohort construction,
modeling, calibration, evaluation, and figure-book workflow:

    python qreg_improvement/run_metabolic_trajectory_study.py

This file is intentionally self-contained. It imports no project-local Python code.
The ``--schema-discovery`` mode opens its own pyodbc connection and inventories only
SQL Server metadata. It does not read patient rows. Production directly queries the
accessible ``dbo.MBSCohort`` and ``dbo.GLP1Cohort`` tables, constructs one reviewed
index row per patient and source, and keeps the resulting wide-source claims labeled
exploratory where exact event timing is unavailable.
Human-facing output is restricted to numbered PNG pages and one matching PDF in
``FIGURES_TO_EXPORT``. Restart artifacts live in the timestamped run directory.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import pickle
import platform
import random
import re
import statistics
import struct
import sys
import tempfile
import textwrap
import time
import traceback
import warnings
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence


# ======================================================================================
# 1. Frozen protocol and runtime configuration
# ======================================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
RUNTIME_CACHE = Path(tempfile.gettempdir()) / "metabolic_trajectory_runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(RUNTIME_CACHE / "joblib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

STUDY_VERSION = "metabolic-trajectory-1.4.0"
SQL_CONTRACT_VERSION = "metabolic-raw-events-v1.1.0"
DIRECT_WIDE_CONTRACT_VERSION = "metabolic-direct-wide-v1.2.0"
SCHEMA_DISCOVERY_VERSION = "metabolic-schema-discovery-v1.0.0"
DEFAULT_CONNECTION_STRING = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=tcp:PROJECTS;"
    "Database=ProjectD332AFD;"
    "Trusted_Connection=yes;"
)
SEED = 20260721
DAYS_PER_MONTH = 30.4375
DAYS_PER_YEAR = 365.25
MIN_CELL_SIZE = 11
LANDMARK_MONTHS = (0, 3, 6, 12, 24)
TARGET_MONTHS = {
    "bmi": (3, 6, 12, 24, 36, 48, 60),
    "hba1c": (12, 24, 36, 48, 60),
}
WINDOW_MONTHS = {
    3: (2.0, 4.5, False),
    6: (4.5, 9.0, False),
    12: (9.0, 18.0, False),
    24: (18.0, 30.0, False),
    36: (30.0, 42.0, False),
    48: (42.0, 54.0, False),
    60: (54.0, 66.0, True),
}
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
INTERVAL_LEVELS = (0.50, 0.80, 0.90)
PROCEDURE_CODES = {
    "43775": "sleeve",
    "43644": "rygb",
    "43645": "rygb",
    "43846": "rygb",
}
BARIATRIC_HISTORY_CODES = frozenset(
    {
        "43644", "43645", "43659", "43770", "43771", "43772", "43773",
        "43774", "43775", "43842", "43843", "43845", "43846", "43847",
        "43848", "43886", "43887", "43888",
    }
)
PLAUSIBLE_RANGES = {"bmi": (10.0, 100.0), "hba1c": (3.0, 20.0)}
PRIMARY_GAP_DAYS = 30
GAP_SENSITIVITIES = (0, 30, 60)
MIN_PDC = 0.80
QUALIFYING_DAYS = 183
STOCKPILE_CAP_DAYS = 90
WEIGHT_TRUNCATION = (0.01, 0.99)
ALTERNATE_WEIGHT_TRUNCATION = ((0.005, 0.995), (0.0, 1.0))
HELDOUT_CENTER_FRACTION = 0.20
CONNECTION_TIMEOUT_SECONDS = 1000
ATOMIC_REPLACE_ATTEMPTS = 12
ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.10
ATOMIC_REPLACE_MAX_DELAY_SECONDS = 2.0
RETRYABLE_WINDOWS_REPLACE_ERRORS = frozenset({5, 32, 33})


def timestamped_default_output_dir(
    *,
    now: datetime | None = None,
    cwd: Path | None = None,
) -> Path:
    """Return a human-readable, collision-safe run directory below the current directory."""

    started_at = now or datetime.now()
    results_root = (cwd or Path.cwd()).expanduser().resolve() / "results"
    stem = f"metabolic_trajectory_{started_at.strftime('%Y%m%d_%H%M%S')}"
    candidate = results_root / stem
    suffix = 1
    while candidate.exists():
        candidate = results_root / f"{stem}_{suffix:02d}"
        suffix += 1
    return candidate

# The audited concept set is deliberately explicit. RxCUI values may be extended only after
# review. Name matching is a transparent fallback for source tables that already expose a
# validated ingredient field.
INCRETIN_INGREDIENTS = {
    "albiglutide": "glp1_receptor_agonist",
    "dulaglutide": "glp1_receptor_agonist",
    "exenatide": "glp1_receptor_agonist",
    "liraglutide": "glp1_receptor_agonist",
    "lixisenatide": "glp1_receptor_agonist",
    "semaglutide": "glp1_receptor_agonist",
    "tirzepatide": "dual_gip_glp1_agonist",
}
INGREDIENT_PATTERNS = {
    "albiglutide": re.compile(r"\balbiglutide\b|\btanzeum\b", re.I),
    "dulaglutide": re.compile(r"\bdulaglutide\b|\btrulicity\b", re.I),
    "exenatide": re.compile(r"\bexenatide\b|\bbyetta\b|\bbydureon\b", re.I),
    "liraglutide": re.compile(r"\bliraglutide\b|\bvictoza\b|\bsaxenda\b", re.I),
    "lixisenatide": re.compile(r"\blixisenatide\b|\badlyxin\b", re.I),
    "semaglutide": re.compile(r"\bsemaglutide\b|\bozempic\b|\bwegovy\b|\brybelsus\b", re.I),
    "tirzepatide": re.compile(r"\btirzepatide\b|\bmounjaro\b|\bzepbound\b", re.I),
}

REQUIRED_PACKAGES = {
    "numpy": ("numpy", "1.24"),
    "pandas": ("pandas", "1.5"),
    "scikit-learn": ("sklearn", "1.2"),
    "matplotlib": ("matplotlib", "3.6"),
}
OPTIONAL_PACKAGES = {
    "catboost": "catboost",
    "torch": "torch",
    "pyodbc": "pyodbc",
}


@dataclass(frozen=True)
class RunConfig:
    mode: str = "production"
    output_dir: str | None = None
    resume: bool = False
    seed: int = SEED
    smoke_patients: int = 420
    smoke_query_limit: int = 2000
    bootstrap_replicates: int = 1000
    model_trials: int = 12
    hgb_iterations: int = 500
    catboost_iterations: int = 3000
    mlp_epochs: int = 300
    final_neural_seeds: int = 3
    trajectory_draws: int = 200
    max_ode_step: float = 1.0 / 12.0
    min_cell_size: int = MIN_CELL_SIZE

    @property
    def smoke(self) -> bool:
        return self.mode == "smoke"

    @classmethod
    def create(
        cls,
        mode: str,
        output_dir: str | None,
        resume: bool,
        *,
        now: datetime | None = None,
        cwd: Path | None = None,
    ) -> "RunConfig":
        if resume and output_dir is None:
            raise ValueError("--resume requires --output-dir PATH for the existing run")
        resolved_output_dir = output_dir
        if resolved_output_dir is None and mode != "plot-only":
            resolved_output_dir = str(
                timestamped_default_output_dir(now=now, cwd=cwd)
            )
        if mode == "smoke":
            return cls(
                mode=mode,
                output_dir=resolved_output_dir,
                resume=resume,
                bootstrap_replicates=60,
                model_trials=2,
                hgb_iterations=45,
                catboost_iterations=65,
                mlp_epochs=18,
                final_neural_seeds=1,
                trajectory_draws=200,
            )
        return cls(mode=mode, output_dir=resolved_output_dir, resume=resume)


@dataclass(frozen=True)
class CoverageRecord:
    patient_id: str
    start_day: int
    end_day: int
    ingredient: str
    therapy_class: str
    route: str = "unknown"
    formulation: str = "unknown"
    source_type: str = "unknown"
    source_table: str = "unknown"
    source_id: str = ""
    dose: float | None = None
    dose_unit: str = ""
    accepted: bool = True
    rejection_reason: str = ""


@dataclass
class CoverageEpisode:
    patient_id: str
    records: list[CoverageRecord]
    supported_intervals: list[tuple[int, int]]
    start_day: int
    supported_end_day: int
    censor_day: int
    maximum_gap_days: int
    pdc_183: float
    qualifies_183: bool
    ingredients: tuple[str, ...]
    switch_days: tuple[int, ...]
    gap_rule_days: int


@dataclass(frozen=True)
class TargetWindow:
    month: int
    nominal_day: int
    start_day: int
    end_day: int
    end_inclusive: bool

    def contains(self, day: int) -> bool:
        if self.end_inclusive:
            return self.start_day <= day <= self.end_day
        return self.start_day <= day < self.end_day


@dataclass
class PreflightError(RuntimeError):
    title: str
    issues: list[str]
    details: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.title + ": " + "; ".join(self.issues)


class LeakageError(RuntimeError):
    pass


# Heavy packages are loaded after dependency checks so a failure report remains available.
np = pd = plt = PdfPages = sklearn = None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def retryable_windows_replace_error(error: OSError) -> bool:
    winerror = getattr(error, "winerror", None)
    if winerror in RETRYABLE_WINDOWS_REPLACE_ERRORS:
        return True
    return os.name == "nt" and isinstance(error, PermissionError)


def replace_file(
    source: str | Path,
    destination: str | Path,
    *,
    attempts: int = ATOMIC_REPLACE_ATTEMPTS,
    initial_delay: float = ATOMIC_REPLACE_INITIAL_DELAY_SECONDS,
    maximum_delay: float = ATOMIC_REPLACE_MAX_DELAY_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
    report_retries: bool = True,
) -> None:
    """Replace a file with bounded retries for transient Windows/SMB locks."""

    if attempts < 1:
        raise ValueError("replace attempts must be at least one")
    delay = max(0.0, float(initial_delay))
    retried = 0
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, destination)
            if retried and report_retries:
                print(
                    f"[metabolic] output replace recovered after {retried} "
                    f"{'retry' if retried == 1 else 'retries'}: {Path(destination).name}",
                    file=sys.stderr,
                    flush=True,
                )
            return
        except OSError as exc:
            if not retryable_windows_replace_error(exc) or attempt == attempts:
                raise
            retried += 1
            if retried == 1 and report_retries:
                code = getattr(exc, "winerror", None)
                print(
                    f"[metabolic] transient Windows/NAS file lock"
                    f"{f' (WinError {code})' if code is not None else ''}: "
                    f"{Path(destination).name}; retrying",
                    file=sys.stderr,
                    flush=True,
                )
            sleeper(delay)
            delay = min(maximum_delay, max(delay * 2.0, initial_delay))


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, payload: str) -> None:
    atomic_bytes(path, payload.encode("utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def atomic_pickle(path: Path, payload: Any) -> None:
    stream = io.BytesIO()
    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_bytes(path, stream.getvalue())


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", value)[:4]) or (0,)


def dependency_manifest(require_database: bool) -> tuple[dict[str, Any], list[str]]:
    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    issues: list[str] = []
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10 or newer is required; found {platform.python_version()}")
    for distribution, (module, minimum) in REQUIRED_PACKAGES.items():
        try:
            version = importlib.metadata.version(distribution)
            manifest[distribution] = version
            if version_tuple(version) < version_tuple(minimum):
                issues.append(f"{distribution}>={minimum} is required; found {version}")
            if importlib.util.find_spec(module) is None:
                issues.append(f"{distribution} is installed but module {module} is unavailable")
        except importlib.metadata.PackageNotFoundError:
            manifest[distribution] = None
            issues.append(f"Missing required package: {distribution}>={minimum}")
    for distribution, module in OPTIONAL_PACKAGES.items():
        try:
            manifest[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            manifest[distribution] = None
        manifest[f"{distribution}_importable"] = importlib.util.find_spec(module) is not None
    if require_database and not manifest.get("pyodbc_importable"):
        issues.append("Production and preflight modes require pyodbc and a SQL Server ODBC driver")
    return manifest, issues


def load_runtime_packages() -> None:
    global np, pd, plt, PdfPages, sklearn
    import numpy as _np
    import pandas as _pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
    import sklearn as _sklearn

    np, pd, plt, PdfPages, sklearn = _np, _pd, _plt, _PdfPages, _sklearn


def set_deterministic_seed(seed: int, include_torch: bool = False) -> dict[str, Any]:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    result: dict[str, Any] = {"python": seed, "numpy": seed if np is not None else None}
    if include_torch and importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        result["torch"] = seed
        result["torch_deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    return result


def month_to_nominal_day(month: int | float) -> int:
    return int(round(float(month) * DAYS_PER_MONTH))


def target_window(month: int) -> TargetWindow:
    if month not in WINDOW_MONTHS:
        raise ValueError(f"No target window is configured for month {month}")
    lower, upper, inclusive = WINDOW_MONTHS[month]
    start = int(math.ceil(lower * DAYS_PER_MONTH))
    end = int(math.floor(upper * DAYS_PER_MONTH)) if inclusive else int(math.ceil(upper * DAYS_PER_MONTH))
    return TargetWindow(month, month_to_nominal_day(month), start, end, inclusive)


TARGET_WINDOWS = {month: target_window(month) for month in sorted(WINDOW_MONTHS)}
MAX_WIDE_STUDY_FOLLOWUP_DAYS = max(window.end_day for window in TARGET_WINDOWS.values())
MAX_PLAUSIBLE_WIDE_FOLLOWUP_DAYS = int(round(50 * DAYS_PER_YEAR))


def normalize_sql(sql: str) -> str:
    return " ".join(str(sql).split())


def quote_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(value)):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return "[" + str(value) + "]"


# ======================================================================================
# 2. Dependency-independent failure output
# ======================================================================================

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
    ")":"0100000100001000010001000","[":"0011000100001000010000110",
    "]":"0110000100001000010001100","=":"0000011111000001111100000","+":"0000000100011100010000000",
    "%":"1100100010001000100010011","?":"0111010001000100000000100"," ":"0000000000000000000000000",
    ",":"0000000000000000010001000","'":"0010000100000000000000000",";":"0000000100000000010001000",
}


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def failure_report_lines(
    issues: Sequence[str],
    details: Sequence[str],
    width: int,
) -> list[str]:
    lines = ["FAILED GATES"]
    for item in issues:
        wrapper = textwrap.TextWrapper(width=width, initial_indent="- ", subsequent_indent="  ")
        lines.extend(wrapper.wrap(str(item)) or ["-"])
    if details:
        lines.extend(["", "DETAILS AND CORRECTIVE ACTION"])
        for item in details:
            wrapper = textwrap.TextWrapper(width=width, initial_indent="- ", subsequent_indent="  ")
            lines.extend(wrapper.wrap(str(item)) or ["-"])
    return lines


def write_failure_png(path: Path, title: str, issues: Sequence[str], details: Sequence[str]) -> None:
    width = 2550
    wrapped = failure_report_lines(issues, details, width=96)
    lines = [title.upper(), "", *wrapped, "", "THE STUDY STOPPED WITHOUT A SCIENTIFIC FALLBACK."]
    scale = 4
    line_height = 40
    height = max(900, 130 + line_height * len(lines))
    pixels = bytearray([248, 249, 251] * width * height)

    def rectangle(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        color_bytes = bytes(color)
        for yy in range(y0, y1):
            start = (yy * width + x0) * 3
            for xx in range(x0, x1):
                offset = start + (xx - x0) * 3
                pixels[offset:offset + 3] = color_bytes

    rectangle(0, 0, width, 34, (177, 35, 55))
    y = 70
    for line_number, line in enumerate(lines):
        color = (145, 22, 43) if line_number == 0 else (28, 35, 45)
        x = 68
        for character in line.upper():
            pattern = FONT_5X7.get(character, FONT_5X7["?"])
            for row in range(7):
                for column in range(5):
                    offset = row * 5 + column
                    if offset < len(pattern) and pattern[offset] == "1":
                        rectangle(
                            x + column * scale,
                            y + row * scale,
                            x + (column + 1) * scale,
                            y + (row + 1) * scale,
                            color,
                        )
            x += 6 * scale
            if x > width - 70:
                break
        y += line_height
    raw = bytearray()
    stride = width * 3
    for row in range(height):
        raw.append(0)
        raw.extend(pixels[row * stride:(row + 1) * stride])
    payload = b"\x89PNG\r\n\x1a\n"
    payload += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += png_chunk(b"IDAT", zlib.compress(bytes(raw), level=8))
    payload += png_chunk(b"IEND", b"")
    atomic_bytes(path, payload)


def pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_failure_pdf(path: Path, title: str, issues: Sequence[str], details: Sequence[str]) -> None:
    lines: list[str] = [title.upper(), "", *failure_report_lines(issues, details, width=84)]
    lines.extend(["", "The study stopped without a scientific fallback."])
    commands = ["BT", "/F1 10 Tf", "54 738 Td", "14 TL"]
    for index, line in enumerate(lines[:48]):
        if index:
            commands.append("T*")
        commands.append(f"({pdf_escape(line)}) Tj")
    commands.append("ET")
    content = "\n".join(commands).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    atomic_bytes(path, bytes(output))


def failure_run_dir(cfg: RunConfig, title: str, issues: Sequence[str]) -> Path:
    if cfg.output_dir:
        return Path(cfg.output_dir).expanduser().resolve()
    if cfg.mode == "self-test":
        key = digest({"title": title, "issues": list(issues), "study": STUDY_VERSION})[:16]
        return RUNTIME_CACHE / ("self_test_failure_" + key)
    return timestamped_default_output_dir()


def render_preflight_failure(
    cfg: RunConfig,
    title: str,
    issues: Sequence[str],
    details: Sequence[str] = (),
) -> Path:
    run_dir = failure_run_dir(cfg, title, issues)
    export = run_dir / "FIGURES_TO_EXPORT"
    export.mkdir(parents=True, exist_ok=True)
    generated_names = set(globals().get("PAGE_FILES", ()))
    generated_names.update(globals().get("SCHEMA_DISCOVERY_PAGE_FILES", ()))
    generated_names.update(
        {
            "00_preflight_failure.png",
            "metabolic_trajectory_figure_book.pdf",
            "metabolic_trajectory_figure_book.pdf.tmp",
            "schema_discovery_figure_book.pdf",
            "schema_discovery_figure_book.pdf.tmp",
        }
    )
    for existing in export.iterdir():
        if not existing.is_file():
            raise RuntimeError(f"Failure export directory contains an unexpected directory: {existing.name}")
        original_name = existing.name.removesuffix(".tmp")
        if existing.name in generated_names or original_name in generated_names:
            if existing.name not in {"00_preflight_failure.png", "metabolic_trajectory_figure_book.pdf"}:
                existing.unlink()
            continue
        raise RuntimeError(f"Failure export directory contains an unexpected file: {existing.name}")
    png = export / "00_preflight_failure.png"
    pdf = export / "metabolic_trajectory_figure_book.pdf"
    write_failure_png(png, title, issues, details)
    write_failure_pdf(pdf, title, issues, details)
    atomic_json(
        run_dir / "preflight_failure.json",
        {"status": "preflight_failure", "title": title, "issues": list(issues), "details": list(details), "time_utc": utc_now()},
    )
    return png


# ======================================================================================
# 3. Medication coverage, outcomes, censoring, and disclosure utilities
# ======================================================================================


def normalize_ingredient(value: Any) -> tuple[str | None, str | None, str]:
    text = "" if value is None else str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "unknown"}:
        return None, None, "missing"
    for ingredient, pattern in INGREDIENT_PATTERNS.items():
        if pattern.search(text):
            return ingredient, INCRETIN_INGREDIENTS[ingredient], "audited_name_map"
    return None, None, "unmapped"


def normalize_route(value: Any) -> str:
    text = "" if value is None else re.sub(r"\s+", " ", str(value).strip().lower())
    if re.search(r"subcut|sq|injection", text):
        return "subcutaneous"
    if re.search(r"oral|mouth|tablet", text):
        return "oral"
    return text or "unknown"


def validate_coverage_record(record: CoverageRecord) -> CoverageRecord:
    reason = ""
    if not record.patient_id:
        reason = "missing_patient_id"
    elif record.ingredient not in INCRETIN_INGREDIENTS:
        reason = "unmapped_ingredient"
    elif record.end_day < record.start_day:
        reason = "end_before_start"
    elif record.end_day - record.start_day + 1 <= 0:
        reason = "nonpositive_interval"
    elif record.source_type not in {"dispense", "fill", "administration", "validated_episode", "explicit_treatment"}:
        reason = "unsupported_source_semantics"
    return CoverageRecord(
        patient_id=str(record.patient_id),
        start_day=int(record.start_day),
        end_day=int(record.end_day),
        ingredient=str(record.ingredient),
        therapy_class=str(record.therapy_class),
        route=str(record.route),
        formulation=str(record.formulation),
        source_type=str(record.source_type),
        source_table=str(record.source_table),
        source_id=str(record.source_id),
        dose=record.dose,
        dose_unit=str(record.dose_unit),
        accepted=not bool(reason),
        rejection_reason=reason,
    )


def carry_stockpile_forward(
    records: Sequence[CoverageRecord],
    cap_days: int = STOCKPILE_CAP_DAYS,
) -> list[CoverageRecord]:
    """Carry same-ingredient dispense overlap forward without changing starts.

    The amount moved to the end of a fill is the supported overlap with earlier
    same-ingredient coverage, capped at ``cap_days``. This preserves exact uncovered
    days while preventing implausible unbounded accumulation. Explicit treatment
    intervals and administrations already define their supported end and are not
    extended as if they were stockpilable dispenses.
    """
    if cap_days < 0:
        raise ValueError("Stockpile cap cannot be negative")
    accepted = [validate_coverage_record(item) for item in records]
    accepted = [item for item in accepted if item.accepted]
    accepted.sort(key=lambda item: (item.patient_id, item.ingredient, item.start_day, item.end_day, item.source_id))
    prior_end: dict[tuple[str, str], int] = {}
    adjusted: list[CoverageRecord] = []
    for item in accepted:
        key = (item.patient_id, item.ingredient)
        stockpilable = item.source_type in {"dispense", "fill"}
        overlap = max(0, prior_end.get(key, item.start_day - 1) - item.start_day + 1) if stockpilable else 0
        carried = min(overlap, cap_days) if stockpilable else 0
        end_day = item.end_day + carried
        prior_end[key] = max(prior_end.get(key, item.start_day - 1), end_day)
        adjusted.append(
            CoverageRecord(
                patient_id=item.patient_id,
                start_day=item.start_day,
                end_day=end_day,
                ingredient=item.ingredient,
                therapy_class=item.therapy_class,
                route=item.route,
                formulation=item.formulation,
                source_type=item.source_type,
                source_table=item.source_table,
                source_id=item.source_id,
                dose=item.dose,
                dose_unit=item.dose_unit,
                accepted=True,
            )
        )
    return sorted(adjusted, key=lambda item: (item.patient_id, item.start_day, item.end_day, item.ingredient))


def merge_supported_intervals(records: Sequence[CoverageRecord]) -> list[tuple[int, int]]:
    intervals = sorted((item.start_day, item.end_day) for item in records if item.accepted)
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_coverage_days(intervals: Sequence[tuple[int, int]], start: int, end: int) -> int:
    if end < start:
        return 0
    covered = 0
    for left, right in intervals:
        overlap_left = max(start, left)
        overlap_right = min(end, right)
        if overlap_left <= overlap_right:
            covered += overlap_right - overlap_left + 1
    return covered


def maximum_uncovered_gap(intervals: Sequence[tuple[int, int]], start: int, end: int) -> int:
    if end < start:
        return 0
    cursor = start
    maximum = 0
    for left, right in intervals:
        if right < start or left > end:
            continue
        left, right = max(left, start), min(right, end)
        if left > cursor:
            maximum = max(maximum, left - cursor)
        cursor = max(cursor, right + 1)
    if cursor <= end:
        maximum = max(maximum, end - cursor + 1)
    return maximum


def make_coverage_episode(records: Sequence[CoverageRecord], gap_rule_days: int) -> CoverageEpisode:
    ordered = sorted(records, key=lambda item: (item.start_day, item.end_day, item.ingredient))
    if not ordered:
        raise ValueError("A coverage episode requires at least one accepted record")
    intervals = merge_supported_intervals(ordered)
    start = intervals[0][0]
    supported_end = intervals[-1][1]
    qualifying_end = start + QUALIFYING_DAYS - 1
    pdc = interval_coverage_days(intervals, start, qualifying_end) / float(QUALIFYING_DAYS)
    maximum_gap = maximum_uncovered_gap(intervals, start, qualifying_end)
    qualifies = supported_end >= qualifying_end and maximum_gap <= gap_rule_days and pdc >= MIN_PDC
    switches: list[int] = []
    prior_ingredient = ordered[0].ingredient
    for item in ordered[1:]:
        if item.ingredient != prior_ingredient:
            switches.append(item.start_day)
        prior_ingredient = item.ingredient
    return CoverageEpisode(
        patient_id=ordered[0].patient_id,
        records=list(ordered),
        supported_intervals=intervals,
        start_day=start,
        supported_end_day=supported_end,
        censor_day=supported_end + 1,
        maximum_gap_days=maximum_gap,
        pdc_183=pdc,
        qualifies_183=qualifies,
        ingredients=tuple(dict.fromkeys(item.ingredient for item in ordered)),
        switch_days=tuple(switches),
        gap_rule_days=gap_rule_days,
    )


def reconstruct_coverage_episodes(
    records: Sequence[CoverageRecord],
    gap_rule_days: int = PRIMARY_GAP_DAYS,
    stockpile_cap_days: int = STOCKPILE_CAP_DAYS,
) -> tuple[list[CoverageEpisode], list[CoverageRecord]]:
    if gap_rule_days < 0:
        raise ValueError("Allowable gap cannot be negative")
    validated = [validate_coverage_record(item) for item in records]
    rejected = [item for item in validated if not item.accepted]
    adjusted = carry_stockpile_forward([item for item in validated if item.accepted], stockpile_cap_days)
    grouped: dict[str, list[CoverageRecord]] = defaultdict(list)
    for item in adjusted:
        grouped[item.patient_id].append(item)
    episodes: list[CoverageEpisode] = []
    for patient_id in sorted(grouped):
        ordered = sorted(grouped[patient_id], key=lambda item: (item.start_day, item.end_day, item.ingredient))
        current: list[CoverageRecord] = []
        current_supported_end: int | None = None
        for item in ordered:
            uncovered = 0 if current_supported_end is None else item.start_day - current_supported_end - 1
            if current and uncovered > gap_rule_days:
                episodes.append(make_coverage_episode(current, gap_rule_days))
                current = []
                current_supported_end = None
            current.append(item)
            current_supported_end = max(current_supported_end if current_supported_end is not None else item.end_day, item.end_day)
        if current:
            episodes.append(make_coverage_episode(current, gap_rule_days))
    return episodes, rejected


def coverage_on_day(episodes: Sequence[CoverageEpisode], day: int) -> bool:
    return any(left <= day <= right for episode in episodes for left, right in episode.supported_intervals)


def first_supported_start_on_or_after(episodes: Sequence[CoverageEpisode], day: int) -> int | None:
    starts = [left for episode in episodes for left, _ in episode.supported_intervals if left >= day]
    return min(starts) if starts else None


def classify_surgical_incretin_history(
    records_relative_to_surgery: Sequence[CoverageRecord],
    postoperative_flag: bool = False,
    timing_unknown: bool = False,
) -> dict[str, Any]:
    episodes, rejected = reconstruct_coverage_episodes(records_relative_to_surgery)
    completed = [
        episode
        for episode in episodes
        if episode.qualifies_183 and episode.start_day + QUALIFYING_DAYS - 1 <= 0
    ]
    preoperative = [episode for episode in episodes if episode.start_day < 0]
    any_preoperative_record = any(item.start_day < 0 for episode in episodes for item in episode.records)
    active_at_surgery = coverage_on_day(episodes, 0)
    postoperative_start = first_supported_start_on_or_after(episodes, 0)
    unresolved_postoperative = bool(postoperative_flag and postoperative_start is None)
    unknown = bool(timing_unknown or rejected or unresolved_postoperative)
    if completed:
        classification = "previously_treated"
    elif unknown:
        classification = "unknown"
    elif not any_preoperative_record:
        classification = "no_prior_accepted_exposure"
    elif len(preoperative) == 1:
        classification = "subthreshold_continuous_exposure"
    else:
        classification = "intermittent_prior_exposure"
    treatment_censor_day = 0 if active_at_surgery else postoperative_start
    return {
        "classification": classification,
        "operationally_naive": classification in {
            "no_prior_accepted_exposure",
            "subthreshold_continuous_exposure",
            "intermittent_prior_exposure",
        },
        "strict_never_exposed": classification == "no_prior_accepted_exposure",
        "active_at_surgery": active_at_surgery,
        "treatment_censor_day": treatment_censor_day,
        "unresolved_postoperative_start": unresolved_postoperative,
        "episode_count": len(episodes),
        "rejected_record_count": len(rejected),
    }


def hba1c_ifcc_to_ngsp(value_mmol_mol: float) -> float:
    return float(value_mmol_mol) / 10.929 + 2.15


def normalize_weight(value: float, unit: Any) -> float | None:
    text = re.sub(r"[^a-z]", "", str(unit).lower())
    if text in {"kg", "kgs", "kilogram", "kilograms"}:
        return float(value)
    if text in {"lb", "lbs", "pound", "pounds"}:
        return float(value) * 0.45359237
    if text in {"g", "gram", "grams"}:
        return float(value) / 1000.0
    return None


def normalize_height(value: float, unit: Any) -> float | None:
    text = re.sub(r"[^a-z]", "", str(unit).lower())
    if text in {"m", "meter", "meters", "metre", "metres"}:
        return float(value)
    if text in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        return float(value) / 100.0
    if text in {"in", "inch", "inches"}:
        return float(value) * 0.0254
    return None


def infer_measurement_kind(concept: Any, declared: Any = None) -> str | None:
    text = " ".join(str(item) for item in (declared, concept) if item is not None).lower()
    if re.search(r"hba1c|hemoglobin\s*a1c|glycated", text):
        return "hba1c"
    if re.search(r"body\s*mass\s*index|\bbmi\b", text):
        return "bmi"
    if re.search(r"height|stature", text):
        return "height"
    if re.search(r"weight|body\s*mass", text):
        return "weight"
    return None


def normalize_hba1c(value: float, unit: Any) -> tuple[float | None, str]:
    text = re.sub(r"\s+", "", str(unit).strip().lower())
    if text in {"%", "percent", "pct", "ngsp%", "ngsp"}:
        normalized = float(value)
    elif text in {"mmol/mol", "mmolmol", "ifcc", "mmolpermol"}:
        normalized = hba1c_ifcc_to_ngsp(float(value))
    else:
        return None, "invalid_or_missing_unit"
    if not PLAUSIBLE_RANGES["hba1c"][0] <= normalized <= PLAUSIBLE_RANGES["hba1c"][1]:
        return None, "outside_plausible_range"
    return normalized, "valid"


def normalize_observed_bmi(value: float, unit: Any) -> tuple[float | None, str]:
    text = re.sub(r"\s+", "", str(unit).strip().lower())
    accepted = {"kg/m2", "kg/m^2", "kgperm2", "kg/m²", "bmi", ""}
    if text not in accepted:
        return None, "invalid_unit"
    normalized = float(value)
    if not PLAUSIBLE_RANGES["bmi"][0] <= normalized <= PLAUSIBLE_RANGES["bmi"][1]:
        return None, "outside_plausible_range"
    return normalized, "valid"


def normalize_measurements(raw: Any) -> tuple[Any, Any]:
    """Normalize raw long-form measurements and resolve duplicate patient-days.

    Required columns are ``patient_id``, ``measurement_date``, ``raw_value``, ``unit``,
    ``source_concept``, and ``source_table``. ``measurement_type`` is optional.
    """
    required = {"patient_id", "measurement_date", "raw_value", "unit", "source_concept", "source_table"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise PreflightError(
            "Raw measurement contract failed",
            ["Missing measurement fields: " + ", ".join(missing)],
            ["Exact dates, raw values, units, source concepts, and source-table lineage are required."],
        )
    frame = raw.copy()
    frame["measurement_date"] = pd.to_datetime(frame["measurement_date"], errors="coerce").dt.normalize()
    frame["raw_value_numeric"] = pd.to_numeric(frame["raw_value"], errors="coerce")
    declared = frame["measurement_type"] if "measurement_type" in frame else pd.Series(None, index=frame.index)
    frame["kind"] = [
        infer_measurement_kind(concept, kind)
        for concept, kind in zip(frame["source_concept"], declared, strict=False)
    ]
    quality_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    supporting: dict[tuple[str, Any, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    source_details: dict[tuple[str, Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        patient = str(row.patient_id)
        measurement_date = row.measurement_date
        source_cohort = str(getattr(row, "source_cohort", "all"))
        kind = row.kind
        value = row.raw_value_numeric
        unit = row.unit
        reason = "valid"
        normalized_value: float | None = None
        if pd.isna(measurement_date):
            reason = "invalid_date"
        elif kind is None:
            reason = "unmapped_concept"
        elif pd.isna(value):
            reason = "nonnumeric_value"
        elif kind == "hba1c":
            normalized_value, reason = normalize_hba1c(float(value), unit)
        elif kind == "bmi":
            normalized_value, reason = normalize_observed_bmi(float(value), unit)
        elif kind == "weight":
            normalized_value = normalize_weight(float(value), unit)
            reason = "valid" if normalized_value is not None and 15 <= normalized_value <= 500 else "invalid_weight_or_unit"
            if reason != "valid":
                normalized_value = None
        elif kind == "height":
            normalized_value = normalize_height(float(value), unit)
            reason = "valid" if normalized_value is not None and 0.5 <= normalized_value <= 2.7 else "invalid_height_or_unit"
            if reason != "valid":
                normalized_value = None
        quality_rows.append(
            {
                "kind": kind or "unmapped",
                "unit": str(unit),
                "source_concept": str(row.source_concept),
                "source_table": str(row.source_table),
                "reason": reason,
                "valid": reason == "valid",
            }
        )
        if normalized_value is not None and reason == "valid":
            key = (patient, measurement_date, source_cohort)
            supporting[key][kind].append(float(normalized_value))
            source_details[(patient, measurement_date, source_cohort, kind)].append(
                {
                    "source_concept": str(row.source_concept),
                    "source_table": str(row.source_table),
                    "source_cohort": str(getattr(row, "source_cohort", "all")),
                    "timing_precision": str(getattr(row, "timing_precision", "exact_day")),
                }
            )
    for (patient, measurement_date, source_cohort), kinds in sorted(
        supporting.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        for kind in ("hba1c", "bmi"):
            values = kinds.get(kind, [])
            method = "observed"
            if kind == "bmi" and not values and kinds.get("weight") and kinds.get("height"):
                weight = float(np.median(kinds["weight"]))
                height = float(np.median(kinds["height"]))
                derived = weight / (height * height)
                if PLAUSIBLE_RANGES["bmi"][0] <= derived <= PLAUSIBLE_RANGES["bmi"][1]:
                    values = [derived]
                    method = "derived_weight_height"
            if not values:
                continue
            details = source_details.get((patient, measurement_date, source_cohort, kind), [])
            normalized_rows.append(
                {
                    "patient_id": patient,
                    "measurement_date": measurement_date,
                    "outcome": kind,
                    "value": float(np.median(values)),
                    "method": method,
                    "valid_measurements_same_day": len(values),
                    "duplicate_day": len(values) > 1,
                    "source_concepts": "|".join(sorted({item["source_concept"] for item in details})) or "derived",
                    "source_tables": "|".join(sorted({item["source_table"] for item in details})) or "derived",
                    "source_cohort": "|".join(sorted({item["source_cohort"] for item in details})) or "all",
                    "timing_precision": "|".join(sorted({item["timing_precision"] for item in details})) or "exact_day",
                }
            )
    normalized = pd.DataFrame(normalized_rows)
    quality = pd.DataFrame(quality_rows)
    return normalized, quality


def select_baseline_measurement(measurements: Any, outcome: str, index_date: Any) -> dict[str, Any] | None:
    lower = -90 if outcome == "bmi" else -180
    index_date = pd.Timestamp(index_date).normalize()
    subset = measurements.loc[measurements["outcome"].eq(outcome)].copy()
    subset["day"] = (subset["measurement_date"] - index_date).dt.days
    subset = subset.loc[subset["day"].between(lower, 0, inclusive="both")]
    if subset.empty:
        return None
    subset["distance"] = subset["day"].abs()
    selected = subset.sort_values(["distance", "day"], ascending=[True, False]).iloc[0]
    result = selected.to_dict()
    result["day"] = int(selected["day"])
    return result


def select_target_measurement(
    measurements: Any,
    outcome: str,
    index_date: Any,
    target_month: int,
    censor_day: int | None = None,
    median_sensitivity: bool = False,
) -> dict[str, Any] | None:
    window = TARGET_WINDOWS[target_month]
    index_date = pd.Timestamp(index_date).normalize()
    subset = measurements.loc[measurements["outcome"].eq(outcome)].copy()
    subset["day"] = (subset["measurement_date"] - index_date).dt.days
    mask = subset["day"].map(window.contains)
    if censor_day is not None:
        mask &= subset["day"].lt(int(censor_day))
    subset = subset.loc[mask]
    if subset.empty:
        return None
    if median_sensitivity:
        closest = subset.assign(distance=(subset["day"] - window.nominal_day).abs()).sort_values(
            ["distance", "day"], ascending=[True, True]
        ).iloc[0]
        result = closest.to_dict()
        result["value"] = float(subset["value"].median())
        result["selection_method"] = "window_median"
    else:
        subset["distance"] = (subset["day"] - window.nominal_day).abs()
        closest = subset.sort_values(["distance", "day"], ascending=[True, True]).iloc[0]
        result = closest.to_dict()
        result["selection_method"] = "closest_tie_earlier"
    result["day"] = int(result["day"])
    result["window_valid_count"] = int(len(subset))
    return result


def target_support_status(
    index_date: Any,
    administrative_end_date: Any,
    observation_end_date: Any,
    target_month: int,
    censor_day: int | None,
    target_observed: bool,
) -> str:
    index = pd.Timestamp(index_date).normalize()
    admin_day = int((pd.Timestamp(administrative_end_date).normalize() - index).days)
    observation_day = int((pd.Timestamp(observation_end_date).normalize() - index).days)
    opportunity_day = min(admin_day, observation_day)
    window = TARGET_WINDOWS[target_month]
    required_end = window.end_day
    if opportunity_day < required_end:
        return "administratively_immature"
    if target_observed:
        return "mature_with_target"
    if censor_day is not None and int(censor_day) <= required_end:
        return "treatment_or_surgery_censored"
    return "mature_without_target"


def rearrange_quantiles(values: Any) -> Any:
    array = np.asarray(values, dtype=float)
    return np.sort(array, axis=-1)


def suppress_small_cells(frame: Any, count_columns: Sequence[str], threshold: int = MIN_CELL_SIZE) -> Any:
    result = frame.copy()
    mask = pd.Series(False, index=result.index)
    for column in count_columns:
        if column in result:
            numeric = pd.to_numeric(result[column], errors="coerce")
            mask |= numeric.gt(0) & numeric.lt(threshold)
    result["small_cell_suppressed"] = mask
    protected = set(count_columns)
    for column in result.columns:
        if column == "small_cell_suppressed":
            continue
        numeric_disclosive = pd.api.types.is_numeric_dtype(result[column]) and not pd.api.types.is_bool_dtype(result[column])
        if column in protected or numeric_disclosive:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
            result.loc[mask, column] = np.nan
    return result


# ======================================================================================
# 4. Metadata discovery, direct wide inputs, optional raw SQL, gates, and checkpoints
# ======================================================================================


def normalize_identifier(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


SCHEMA_DISCOVERY_PAGE_FILES = (
    "01_schema_discovery_overview.png",
    "02_patient_center_candidates.png",
    "03_procedure_candidates.png",
    "04_medication_candidates.png",
    "05_measurement_candidates.png",
    "06_encounter_diagnosis_candidates.png",
    "07_key_dependency_map.png",
)

DISCOVERY_ROLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "patient_id": re.compile(r"(?:^patkey$|(?:patient|person|subject|pat)(?:durable)?(?:key|id)$)", re.I),
    "center_id": re.compile(r"(?:center|site|organization|facility|healthsystem|servicearea|location)(?:key|id)$", re.I),
    "encounter_id": re.compile(r"(?:encounter|visit|contact)(?:key|id)$", re.I),
    "procedure_id": re.compile(r"(?:procedure|surgery|operation)(?:key|id)$", re.I),
    "medication_id": re.compile(r"(?:medication|med|drug|order|dispense|administration|admin|rx)(?:key|id)$", re.I),
    "measurement_id": re.compile(r"(?:measurement|result|lab|component|observation)(?:key|id)$", re.I),
    "age": re.compile(r"(?:^age$|ageat|patientage)", re.I),
    "birth_date": re.compile(r"(?:birthdate|dateofbirth|dob$)", re.I),
    "birth_year": re.compile(r"(?:birthyear|yearofbirth)", re.I),
    "sex": re.compile(r"(?:^sex$|administrativesex|gender)", re.I),
    "race": re.compile(r"(?:^race$|racecategory|firstrace)", re.I),
    "ethnicity": re.compile(r"ethnicity", re.I),
    "observation_start_date": re.compile(r"(?:observation|coverage|enrollment|active).*(?:start|begin).*(?:date|time)?$", re.I),
    "observation_end_date": re.compile(r"(?:observation|coverage|enrollment|active|lastobserved|lastcontact).*(?:end|stop|date|time)$", re.I),
    "administrative_end_date": re.compile(r"(?:administrativeend|datathrough|studyend|extractthrough)", re.I),
    "procedure_date": re.compile(r"(?:procedure|proc|surgery|operation).*(?:date|time)$", re.I),
    "procedure_code": re.compile(r"(?:procedure|proc|cpt|hcpcs).*(?:code|concept|id)?$|^cptcode$", re.I),
    "ingredient": re.compile(r"(?:ingredient|generic|brand|product|medicationname|drugname|glp1name)", re.I),
    "medication_concept": re.compile(r"(?:rxnorm|rxcui|ndc|gpi|atc|medicationconcept|drugconcept)", re.I),
    "order_date": re.compile(r"(?:medication|drug|rx|order).*(?:order|ordered).*(?:date|time)$|^orderdate$", re.I),
    "fill_date": re.compile(r"(?:fill|dispense|sold|claim).*(?:date|time)$", re.I),
    "administration_date": re.compile(r"(?:administration|admin).*(?:date|time)$", re.I),
    "medication_start_date": re.compile(r"(?:medication|drug|rx|episode|therapy|treatment).*(?:start|begin).*(?:date|time)$", re.I),
    "medication_end_date": re.compile(r"(?:medication|drug|rx|episode|therapy|treatment).*(?:end|stop|discontinu).*(?:date|time)$", re.I),
    "days_supply": re.compile(r"days?(?:supply|supplied)", re.I),
    "quantity": re.compile(r"(?:dispense)?quantity|qty", re.I),
    "refills": re.compile(r"refill", re.I),
    "dose": re.compile(r"(?:dose|strength)(?:value|amount)?$", re.I),
    "dose_unit": re.compile(r"(?:dose|strength).*unit$", re.I),
    "route": re.compile(r"(?:route|formulation|doseform)", re.I),
    "frequency": re.compile(r"(?:frequency|sig|schedule)", re.I),
    "measurement_date": re.compile(r"(?:measurement|result|specimen|collection|observed|recorded).*(?:date|time)$", re.I),
    "raw_value": re.compile(r"(?:raw|result|numeric|measurement|valueas).*(?:value|number)$|^value$", re.I),
    "unit": re.compile(r"(?:result|measurement|source|value)?unit(?:name|code|value)?$", re.I),
    "source_concept": re.compile(r"(?:loinc|component|measurementconcept|resultconcept|labconcept)", re.I),
    "measurement_type": re.compile(r"(?:measurement|result|component|lab).*(?:type|name)$", re.I),
    "encounter_date": re.compile(r"(?:encounter|visit|contact).*(?:date|time)$", re.I),
    "diagnosis_date": re.compile(r"(?:diagnosis|condition).*(?:date|time|start)$", re.I),
    "diagnosis_code": re.compile(r"(?:diagnosis|condition|icd).*(?:code|concept|id)$|^icd10$", re.I),
}

DISCOVERY_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    "patients": {
        "table_pattern": re.compile(r"patient|person|demograph|member", re.I),
        "roles": {
            "patient_id", "center_id", "age", "birth_date", "birth_year", "sex", "race",
            "ethnicity", "observation_start_date", "observation_end_date", "administrative_end_date",
        },
        "core": {"patient_id"},
        "signals": {
            "age", "birth_date", "birth_year", "sex", "race", "ethnicity",
            "observation_start_date", "observation_end_date", "administrative_end_date",
        },
    },
    "centers": {
        "table_pattern": re.compile(r"center|site|organization|facility|location|servicearea", re.I),
        "roles": {"patient_id", "center_id", "encounter_id"},
        "core": {"center_id"},
        "signals": {"center_id"},
    },
    "procedures": {
        "table_pattern": re.compile(r"procedure|surgery|operation|cpt|hcpcs", re.I),
        "roles": {"patient_id", "encounter_id", "procedure_id", "procedure_date", "procedure_code", "center_id"},
        "core": {"patient_id", "procedure_date", "procedure_code"},
        "signals": {"procedure_id", "procedure_date", "procedure_code"},
    },
    "medications": {
        "table_pattern": re.compile(r"med|drug|pharm|rx|order|dispens|admin|exposure|therapy", re.I),
        "roles": {
            "patient_id", "encounter_id", "medication_id", "ingredient", "medication_concept",
            "order_date", "fill_date", "administration_date", "medication_start_date",
            "medication_end_date", "days_supply", "quantity", "refills", "dose", "dose_unit",
            "route", "frequency", "center_id",
        },
        "core": {"patient_id", "ingredient", "medication_concept"},
        "signals": {
            "medication_id", "ingredient", "medication_concept", "order_date", "fill_date",
            "administration_date", "medication_start_date", "medication_end_date", "days_supply",
            "quantity", "refills", "dose", "dose_unit", "route", "frequency",
        },
    },
    "measurements": {
        "table_pattern": re.compile(r"measurement|lab|result|vital|observation|component", re.I),
        "roles": {
            "patient_id", "encounter_id", "measurement_id", "measurement_date", "raw_value",
            "unit", "source_concept", "measurement_type", "center_id",
        },
        "core": {"patient_id", "measurement_date", "raw_value", "unit", "source_concept"},
        "signals": {
            "measurement_id", "measurement_date", "raw_value", "unit", "source_concept",
            "measurement_type",
        },
    },
    "encounters": {
        "table_pattern": re.compile(r"encounter|visit|contact", re.I),
        "roles": {"patient_id", "encounter_id", "encounter_date", "center_id"},
        "core": {"patient_id", "encounter_date"},
        "signals": {"encounter_id", "encounter_date"},
    },
    "diagnoses": {
        "table_pattern": re.compile(r"diagnos|condition|problem|icd", re.I),
        "roles": {"patient_id", "encounter_id", "diagnosis_date", "diagnosis_code", "center_id"},
        "core": {"patient_id", "diagnosis_date", "diagnosis_code"},
        "signals": {"diagnosis_date", "diagnosis_code"},
    },
}

SCHEMA_DISCOVERY_SQL: dict[str, str] = {
    "database": """
/* metabolic-schema-discovery: database */
SELECT DB_NAME() AS DATABASE_NAME
""",
    "columns": """
/* metabolic-schema-discovery: columns */
SELECT c.TABLE_CATALOG, c.TABLE_SCHEMA, c.TABLE_NAME,
       COALESCE(t.TABLE_TYPE, 'UNKNOWN') AS TABLE_TYPE,
       c.COLUMN_NAME, c.ORDINAL_POSITION, c.DATA_TYPE,
       c.CHARACTER_MAXIMUM_LENGTH, c.NUMERIC_PRECISION,
       c.NUMERIC_SCALE, c.IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS AS c
LEFT JOIN INFORMATION_SCHEMA.TABLES AS t
  ON t.TABLE_CATALOG = c.TABLE_CATALOG
 AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
 AND t.TABLE_NAME = c.TABLE_NAME
WHERE c.TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA', 'sys')
ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
""",
    "keys": """
/* metabolic-schema-discovery: keys */
SELECT k.TABLE_SCHEMA, k.TABLE_NAME, k.COLUMN_NAME,
       tc.CONSTRAINT_TYPE, k.CONSTRAINT_NAME, k.ORDINAL_POSITION
FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS k
JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS tc
  ON tc.CONSTRAINT_CATALOG = k.CONSTRAINT_CATALOG
 AND tc.CONSTRAINT_SCHEMA = k.CONSTRAINT_SCHEMA
 AND tc.CONSTRAINT_NAME = k.CONSTRAINT_NAME
WHERE tc.CONSTRAINT_TYPE IN ('PRIMARY KEY', 'UNIQUE')
ORDER BY k.TABLE_SCHEMA, k.TABLE_NAME, tc.CONSTRAINT_TYPE, k.ORDINAL_POSITION
""",
    "foreign_keys": """
/* metabolic-schema-discovery: foreign-keys */
SELECT OBJECT_SCHEMA_NAME(fkc.parent_object_id) AS CHILD_SCHEMA,
       OBJECT_NAME(fkc.parent_object_id) AS CHILD_TABLE,
       pc.name AS CHILD_COLUMN,
       OBJECT_SCHEMA_NAME(fkc.referenced_object_id) AS PARENT_SCHEMA,
       OBJECT_NAME(fkc.referenced_object_id) AS PARENT_TABLE,
       rc.name AS PARENT_COLUMN,
       fk.name AS FOREIGN_KEY_NAME
FROM sys.foreign_key_columns AS fkc
JOIN sys.foreign_keys AS fk
  ON fk.object_id = fkc.constraint_object_id
JOIN sys.columns AS pc
  ON pc.object_id = fkc.parent_object_id
 AND pc.column_id = fkc.parent_column_id
JOIN sys.columns AS rc
  ON rc.object_id = fkc.referenced_object_id
 AND rc.column_id = fkc.referenced_column_id
ORDER BY CHILD_SCHEMA, CHILD_TABLE, FOREIGN_KEY_NAME, fkc.constraint_column_id
""",
    "synonyms": """
/* metabolic-schema-discovery: synonyms */
SELECT SCHEMA_NAME(schema_id) AS SYNONYM_SCHEMA,
       name AS SYNONYM_NAME,
       base_object_name AS BASE_OBJECT_NAME
FROM sys.synonyms
ORDER BY SYNONYM_SCHEMA, SYNONYM_NAME
""",
    "cohort_modules": """
/* metabolic-schema-discovery: cohort-modules */
SELECT OBJECT_SCHEMA_NAME(m.object_id) AS MODULE_SCHEMA,
       OBJECT_NAME(m.object_id) AS MODULE_NAME,
       o.type_desc AS MODULE_TYPE,
       CASE WHEN LOWER(m.definition) LIKE '%mbscohort%' THEN 1 ELSE 0 END AS REFERENCES_MBSCOHORT,
       CASE WHEN LOWER(m.definition) LIKE '%glp1cohort%' THEN 1 ELSE 0 END AS REFERENCES_GLP1COHORT
FROM sys.sql_modules AS m
JOIN sys.objects AS o ON o.object_id = m.object_id
WHERE LOWER(m.definition) LIKE '%mbscohort%'
   OR LOWER(m.definition) LIKE '%glp1cohort%'
ORDER BY MODULE_SCHEMA, MODULE_NAME
""",
    "object_dependencies": """
/* metabolic-schema-discovery: object-dependencies */
SELECT OBJECT_SCHEMA_NAME(d.referencing_id) AS REFERENCING_SCHEMA,
       OBJECT_NAME(d.referencing_id) AS REFERENCING_OBJECT,
       o.type_desc AS REFERENCING_TYPE,
       d.referenced_server_name AS REFERENCED_SERVER,
       d.referenced_database_name AS REFERENCED_DATABASE,
       COALESCE(d.referenced_schema_name, OBJECT_SCHEMA_NAME(d.referenced_id)) AS REFERENCED_SCHEMA,
       COALESCE(d.referenced_entity_name, OBJECT_NAME(d.referenced_id)) AS REFERENCED_OBJECT,
       d.is_schema_bound_reference AS IS_SCHEMA_BOUND,
       d.is_ambiguous AS IS_AMBIGUOUS
FROM sys.sql_expression_dependencies AS d
LEFT JOIN sys.objects AS o
  ON o.object_id = d.referencing_id
WHERE d.referenced_entity_name IS NOT NULL
ORDER BY REFERENCING_SCHEMA, REFERENCING_OBJECT, REFERENCED_SCHEMA, REFERENCED_OBJECT
""",
}


def discovery_roles(column_name: Any, table_name: Any) -> list[str]:
    column = normalize_identifier(column_name)
    table = normalize_identifier(table_name)
    roles = [role for role, pattern in DISCOVERY_ROLE_PATTERNS.items() if pattern.search(column)]
    contextual = {
        "procedures": ("procedure_id", "procedure_date", "procedure_code"),
        "medications": ("medication_id", "medication_start_date", "medication_concept"),
        "measurements": ("measurement_id", "measurement_date", "raw_value"),
        "encounters": ("encounter_id", "encounter_date", "encounter_id"),
        "diagnoses": ("diagnosis_code", "diagnosis_date", "diagnosis_code"),
    }
    for domain, (identifier_role, date_role, code_or_value_role) in contextual.items():
        if not DISCOVERY_DOMAIN_RULES[domain]["table_pattern"].search(table):
            continue
        if column in {"id", "key"}:
            roles.append(identifier_role)
        if column in {"date", "servicedate", "eventdate", "recordeddate"}:
            roles.append(date_role)
        if column in {"code", "conceptid", "name"}:
            roles.append(code_or_value_role)
    return sorted(set(roles))


def _metadata_object_name(schema_name: Any, table_name: Any) -> str:
    return f"{schema_name}.{table_name}"


def build_schema_discovery_candidates(columns: Any, keys: Any) -> tuple[Any, Any]:
    key_lookup: dict[tuple[str, str, str], str] = {}
    if keys is not None and not keys.empty:
        for row in keys.itertuples(index=False):
            key_lookup[(str(row.TABLE_SCHEMA), str(row.TABLE_NAME), str(row.COLUMN_NAME))] = str(row.CONSTRAINT_TYPE)
    candidate_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for (schema_name, table_name), group in columns.groupby(["TABLE_SCHEMA", "TABLE_NAME"], sort=True):
        object_name = _metadata_object_name(schema_name, table_name)
        table_type = str(group["TABLE_TYPE"].iloc[0]) if "TABLE_TYPE" in group else "UNKNOWN"
        role_columns: dict[str, list[str]] = defaultdict(list)
        column_roles: dict[str, list[str]] = {}
        for row in group.itertuples(index=False):
            roles = discovery_roles(row.COLUMN_NAME, table_name)
            column_roles[str(row.COLUMN_NAME)] = roles
            for role in roles:
                role_columns[role].append(str(row.COLUMN_NAME))
        for domain, rule in DISCOVERY_DOMAIN_RULES.items():
            allowed_roles = set(rule["roles"])
            matched_roles = sorted(allowed_roles.intersection(role_columns))
            name_match = bool(rule["table_pattern"].search(str(table_name)))
            signal_roles = set(rule["signals"])
            matched_signals = signal_roles.intersection(matched_roles)
            if not matched_signals and not name_match:
                continue
            core_roles = set(rule["core"])
            core_matched = core_roles.intersection(matched_roles)
            patient_bonus = 4 if "patient_id" in matched_roles else 0
            score = (
                3 * len(matched_roles)
                + 5 * len(core_matched)
                + 4 * len(matched_signals)
                + patient_bonus
                + (4 if name_match else 0)
            )
            relevant_columns = sorted(
                {
                    column
                    for role in matched_roles
                    for column in role_columns.get(role, [])
                }
            )
            key_columns = sorted(
                {
                    column
                    for column in group["COLUMN_NAME"].astype(str)
                    if (str(schema_name), str(table_name), column) in key_lookup
                    or re.search(r"(?:key|id)$", normalize_identifier(column), re.I)
                }
            )
            candidate_rows.append(
                {
                    "domain": domain,
                    "object": object_name,
                    "object_type": table_type,
                    "score": score,
                    "core_coverage": f"{len(core_matched)}/{len(core_roles)}",
                    "matched_roles": " | ".join(matched_roles),
                    "key_columns": " | ".join(key_columns[:8]),
                    "relevant_columns": " | ".join(relevant_columns[:16]),
                }
            )
            for row in group.itertuples(index=False):
                roles = sorted(allowed_roles.intersection(column_roles.get(str(row.COLUMN_NAME), [])))
                if not roles:
                    continue
                detail_rows.append(
                    {
                        "domain": domain,
                        "object": object_name,
                        "column": str(row.COLUMN_NAME),
                        "role": " | ".join(roles),
                        "data_type": str(row.DATA_TYPE),
                        "nullable": str(row.IS_NULLABLE),
                        "key_type": key_lookup.get((str(schema_name), str(table_name), str(row.COLUMN_NAME)), ""),
                    }
                )
    candidates = pd.DataFrame(candidate_rows)
    details = pd.DataFrame(detail_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["domain", "score", "object"], ascending=[True, False, True], ignore_index=True
        )
    if not details.empty:
        details = details.sort_values(["domain", "object", "role", "column"], ignore_index=True)
    return candidates, details


def build_shared_key_hints(columns: Any) -> Any:
    rows: list[dict[str, Any]] = []
    working = columns.copy()
    working["normalized_column"] = working["COLUMN_NAME"].map(normalize_identifier)
    working = working.loc[working["normalized_column"].str.contains(r"(?:key|id)$", regex=True)]
    for normalized, group in working.groupby("normalized_column", sort=True):
        objects = sorted(
            {
                _metadata_object_name(row.TABLE_SCHEMA, row.TABLE_NAME)
                for row in group.itertuples(index=False)
            }
        )
        if 2 <= len(objects) <= 40:
            rows.append(
                {
                    "normalized_column": normalized,
                    "object_count": len(objects),
                    "objects": " | ".join(objects[:8]),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["object_count", "normalized_column"], ascending=[False, True], ignore_index=True)
    return frame


def _read_metadata_query(
    connection: Any,
    query_name: str,
    reader: Callable[[str, Any], Any],
    warnings_out: list[str],
    required: bool = False,
) -> Any:
    try:
        return reader(SCHEMA_DISCOVERY_SQL[query_name], connection)
    except Exception as exc:
        if required:
            raise PreflightError(
                "Cosmos schema discovery failed",
                [f"{query_name}: {type(exc).__name__}: {sanitize_exception_text(exc)}"],
                ["Grant read access to INFORMATION_SCHEMA metadata; no patient rows are requested."],
            ) from exc
        warnings_out.append(f"{query_name} metadata unavailable: {type(exc).__name__}: {sanitize_exception_text(exc)}")
        return pd.DataFrame()


def discover_cosmos_schema(
    connection: Any,
    read_sql: Callable[[str, Any], Any] | None = None,
) -> dict[str, Any]:
    reader = read_sql or pd.read_sql_query
    discovery_warnings: list[str] = []
    database = _read_metadata_query(connection, "database", reader, discovery_warnings)
    columns = _read_metadata_query(connection, "columns", reader, discovery_warnings, required=True)
    if columns.empty:
        raise PreflightError(
            "Cosmos schema discovery returned no objects",
            ["INFORMATION_SCHEMA.COLUMNS returned zero accessible application columns"],
            ["Confirm the connection database and metadata permissions."],
        )
    keys = _read_metadata_query(connection, "keys", reader, discovery_warnings)
    foreign_keys = _read_metadata_query(connection, "foreign_keys", reader, discovery_warnings)
    synonyms = _read_metadata_query(connection, "synonyms", reader, discovery_warnings)
    cohort_modules = _read_metadata_query(connection, "cohort_modules", reader, discovery_warnings)
    object_dependencies = _read_metadata_query(connection, "object_dependencies", reader, discovery_warnings)
    candidates, candidate_columns = build_schema_discovery_candidates(columns, keys)
    shared_keys = build_shared_key_hints(columns)
    return {
        "version": SCHEMA_DISCOVERY_VERSION,
        "database": database,
        "columns": columns,
        "keys": keys,
        "foreign_keys": foreign_keys,
        "synonyms": synonyms,
        "cohort_modules": cohort_modules,
        "object_dependencies": object_dependencies,
        "candidates": candidates,
        "candidate_columns": candidate_columns,
        "shared_keys": shared_keys,
        "warnings": discovery_warnings,
    }


# This optional raw-event contract is retained for a future database that exposes the
# necessary normalized tables. The current production entry point uses the two reviewed
# direct wide inputs discovered in ProjectD332AFD.
EMBEDDED_RAW_SOURCE_SQL: dict[str, str] = {}
RAW_SQL_TOP_TOKEN = "/*METABOLIC_TOP*/"
RAW_REQUIRED_SOURCES = ("patients", "procedures", "medications", "measurements")
RAW_OPTIONAL_SOURCES = ("encounters", "diagnoses")
RAW_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "patients": {"patient_id", "center_id", "observation_start_date", "observation_end_date"},
    "procedures": {"patient_id", "procedure_date", "procedure_code"},
    "medications": {"patient_id"},
    "measurements": {"patient_id", "measurement_date", "raw_value", "unit", "source_concept"},
    "encounters": {"patient_id", "encounter_date"},
    "diagnoses": {"patient_id", "diagnosis_date", "diagnosis_code"},
}
RAW_DATE_COLUMNS = (
    "observation_start_date", "observation_end_date", "administrative_end_date", "procedure_date",
    "fill_date", "administration_date", "medication_start_date", "medication_end_date",
    "measurement_date", "encounter_date", "diagnosis_date",
)


def validate_embedded_raw_sql(sql_contract: Mapping[str, str]) -> None:
    missing_queries = sorted(set(RAW_REQUIRED_SOURCES).difference(sql_contract))
    if missing_queries:
        raise PreflightError(
            "Reviewed raw-source SQL is not yet embedded",
            ["Missing canonical query: " + name for name in missing_queries],
            [
                "Run --schema-discovery on the Cosmos VM and return the seven metadata PNG pages.",
                "After the raw joins and index-event rules are reviewed, freeze their CTEs in "
                "EMBEDDED_RAW_SOURCE_SQL. No premade cohort table is required.",
            ],
        )
    unexpected = sorted(set(sql_contract).difference(set(RAW_REQUIRED_SOURCES) | set(RAW_OPTIONAL_SOURCES)))
    if unexpected:
        raise PreflightError("Embedded raw-source SQL contains unknown domains", unexpected)
    issues: list[str] = []
    for logical_name, query in sql_contract.items():
        normalized = normalize_sql(str(query))
        if not normalized:
            issues.append(f"{logical_name}: query is empty")
        if RAW_SQL_TOP_TOKEN not in str(query):
            issues.append(f"{logical_name}: missing bounded-run token {RAW_SQL_TOP_TOKEN}")
        if re.search(r"\b(?:dbo\s*\.\s*)?(?:mbscohort|glp1cohort)\b", normalized, re.I):
            issues.append(f"{logical_name}: references a forbidden premade cohort object")
    if issues:
        raise PreflightError(
            "Embedded raw-source SQL contract is invalid",
            issues,
            ["Every production query must be an explicit raw-table query with a bounded-run TOP token."],
        )


def materialize_raw_sql(query: str, limit: int | None) -> str:
    top_clause = f"TOP ({int(limit)}) " if limit is not None else ""
    return str(query).replace(RAW_SQL_TOP_TOKEN, top_clause)


def canonicalize_raw_source_frame(frame: Any, logical_name: str) -> Any:
    result = frame.copy()
    result.columns = [str(column).lower() for column in result.columns]
    duplicate_columns = sorted({column for column in result.columns if list(result.columns).count(column) > 1})
    if duplicate_columns:
        raise PreflightError(
            f"{logical_name.title()} raw query returned duplicate canonical columns",
            duplicate_columns,
        )
    missing = sorted(RAW_REQUIRED_COLUMNS[logical_name].difference(result.columns))
    if logical_name == "patients" and not ({"age", "birth_year", "birth_date"}.intersection(result.columns)):
        missing.append("one of age, birth_year, or birth_date")
    if logical_name == "medications":
        if not ({"ingredient", "medication_concept"}.intersection(result.columns)):
            missing.append("one of ingredient or medication_concept")
        coverage_shapes = (
            {"fill_date", "days_supply"},
            {"administration_date"},
            {"medication_start_date", "medication_end_date"},
        )
        if not any(shape.issubset(result.columns) for shape in coverage_shapes):
            missing.append(
                "auditable coverage: fill_date+days_supply, administration_date, or medication_start_date+medication_end_date"
            )
    if missing:
        raise PreflightError(
            f"{logical_name.title()} raw query lacks canonical columns",
            ["Missing: " + name for name in missing],
            ["Correct the embedded SQL alias list. Do not infer missing timing fields in Python."],
        )
    if result.empty:
        raise PreflightError(f"{logical_name.title()} raw query returned zero rows")
    result["patient_id"] = result["patient_id"].astype("string")
    for column in RAW_DATE_COLUMNS:
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
    if "source_table" not in result:
        result["source_table"] = f"embedded_raw_sql:{logical_name}"
    return result


def empty_optional_raw_source(logical_name: str) -> Any:
    return pd.DataFrame(columns=sorted(RAW_REQUIRED_COLUMNS[logical_name] | {"source_table"}))


def load_embedded_raw_bundle(
    connection: Any,
    cfg: RunConfig,
    sql_contract: Mapping[str, str] | None = None,
    read_sql: Callable[[str, Any], Any] | None = None,
    preflight_only: bool = False,
) -> "DataBundle":
    contract = dict(EMBEDDED_RAW_SOURCE_SQL if sql_contract is None else sql_contract)
    validate_embedded_raw_sql(contract)
    reader = read_sql or pd.read_sql_query
    limit = cfg.smoke_query_limit if cfg.smoke else (2000 if preflight_only or cfg.mode == "preflight-only" else None)
    frames: dict[str, Any] = {}
    executed_sql: dict[str, str] = {}
    for logical_name in (*RAW_REQUIRED_SOURCES, *RAW_OPTIONAL_SOURCES):
        if logical_name not in contract:
            frames[logical_name] = empty_optional_raw_source(logical_name)
            continue
        query = materialize_raw_sql(contract[logical_name], limit)
        try:
            raw_frame = reader(query, connection)
        except Exception as exc:
            raise PreflightError(
                f"{logical_name.title()} raw Cosmos query failed",
                [f"{type(exc).__name__}: {sanitize_exception_text(exc)}"],
                ["Review the embedded CTE and the metadata-only dependency map."],
            ) from exc
        frames[logical_name] = canonicalize_raw_source_frame(raw_frame, logical_name)
        executed_sql[logical_name] = query
    bundle = DataBundle(
        patients=frames["patients"],
        procedures=frames["procedures"],
        medications=frames["medications"],
        measurements=frames["measurements"],
        encounters=frames["encounters"],
        diagnoses=frames["diagnoses"],
        metadata={
            "source_mode": "cosmos_embedded_raw_sql",
            "sql_contract_version": SQL_CONTRACT_VERSION,
            "sql": executed_sql,
            "query_fingerprint": digest({key: normalize_sql(value) for key, value in executed_sql.items()}),
            "schema_fingerprint": digest({key: frame_schema(value) for key, value in frames.items()}),
            "source_row_counts": {key: len(value) for key, value in frames.items()},
            "strict_raw_event_contract": True,
        },
    )
    bundle.metadata["preflight"] = validate_data_bundle(bundle, preflight_only=preflight_only)
    return bundle


DIRECT_SOURCE_SCHEMA = "dbo"
DIRECT_MBS_TABLE = "MBSCohort"
DIRECT_GLP1_TABLE = "GLP1Cohort"
CENTER_UNAVAILABLE = "CENTER_UNAVAILABLE"

WIDE_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "patient_id": ("PatKey", "PatientKey", "PatientID"),
    "center_id": (
        "CenterID", "CenterKey", "SiteID", "SiteKey", "OrganizationID",
        "OrganizationKey", "FacilityID", "FacilityKey", "HealthSystemID",
    ),
    "procedure_date": ("ProcDateValue", "ProcedureDateValue", "IndexProcedureDate"),
    "procedure_code": ("CptCode", "CPTCode", "ProcedureCode"),
    "age": ("AgeAtEvent", "AgeAtIndex"),
    "sex": ("Sex", "Gender"),
    "race": ("FirstRace", "Race"),
    "ethnicity": ("Ethnicity",),
    "coverage": ("CoverageClass", "PayerClass"),
    "prior_glp1": ("PriorGLP1", "PreOpGLP1"),
    "postop_glp1": ("PostOpGLP1", "PostoperativeGLP1"),
    "glp1_start": ("GLP1StartDate", "IncretinStartDate", "MedicationStartDate"),
    "glp1_end": ("GLP1EndDate", "IncretinEndDate", "MedicationEndDate"),
    "glp1_duration": ("GLP1Duration", "GLP1DurationDays", "MedicationDurationDays"),
    "glp1_name": ("GLP1Name", "IncretinName", "MedicationName", "IngredientName"),
    "glp1_route": ("GLP1Route", "MedicationRoute", "Route"),
    "dose": ("MostRecentDose", "MaxGLP1Dose", "MaximumGLP1Dose"),
    "dose_unit": ("MostRecentDoseUnit", "GLP1DoseUnit", "DoseUnit"),
    "prior_mbs": ("PMH_PriorMBS", "PriorMBS"),
    "mbs_during_glp1": ("MBSduringGLP1", "MBSDuringGLP1"),
    "dialysis_transplant": ("PMH_dialysis_transplant", "PriorDialysisTransplant"),
    "diabetes": ("PMH_DM2", "T2D", "Type2Diabetes"),
    "hypertension": ("PMH_hypertension", "Hypertension"),
    "osa": ("PMH_OSA", "OSA"),
    "dyslipidemia": ("PMH_dyslipidemia", "Dyslipidemia"),
    "insulin": ("InsulinStatus", "BaselineInsulin"),
    "biguanide": ("BiguanideStatus", "MetforminStatus"),
    "sglt2": ("SGLT2Status",),
    "svi": ("SviOverall", "SVIOverall"),
    "ruca": ("RUCA",),
    "state": ("StateOrProvince", "State"),
    "active_end_days": ("ActiveEndInterval", "ObservableFollowupDays", "FollowupDays"),
    "administrative_end_date": ("AdministrativeEndDate", "DataThroughDate", "StudyEndDate"),
}

WIDE_TARGET_FIELDS: tuple[tuple[str, int, tuple[str, ...], str], ...] = (
    ("bmi", 0, ("BMIatEvent", "BMIAtIndex", "BaselineBMI"), "kg/m2"),
    ("bmi", 3, ("BMI3mPostEvent",), "kg/m2"),
    ("bmi", 6, ("BMI6mPostEvent",), "kg/m2"),
    ("bmi", 12, ("BMI12mPostEvent",), "kg/m2"),
    ("bmi", 24, ("BMI2yPostEvent", "BMI24mPostEvent"), "kg/m2"),
    ("bmi", 36, ("BMI3yPostEvent", "BMI36mPostEvent"), "kg/m2"),
    ("bmi", 48, ("BMI4yPostEvent", "BMI48mPostEvent"), "kg/m2"),
    ("bmi", 60, ("BMI5yPostEvent", "BMI60mPostEvent"), "kg/m2"),
    ("hba1c", 0, ("HbA1cAtEvent", "HbA1cAtIndex", "BaselineHbA1c"), "%"),
    ("hba1c", 12, ("HbA1c12mPostEvent",), "%"),
    ("hba1c", 24, ("HbA1c2yPostEvent", "HbA1c24mPostEvent"), "%"),
    ("hba1c", 36, ("HbA1c3yPostEvent", "HbA1c36mPostEvent"), "%"),
    ("hba1c", 48, ("HbA1c4yPostEvent", "HbA1c48mPostEvent"), "%"),
    ("hba1c", 60, ("HbA1c5yPostEvent", "HbA1c60mPostEvent"), "%"),
)


@dataclass
class DataBundle:
    patients: Any
    procedures: Any
    medications: Any
    measurements: Any
    encounters: Any
    diagnoses: Any
    metadata: dict[str, Any]

    def row_counts(self) -> dict[str, int]:
        return {
            "patients": len(self.patients),
            "procedures": len(self.procedures),
            "medications": len(self.medications),
            "measurements": len(self.measurements),
            "encounters": len(self.encounters),
            "diagnoses": len(self.diagnoses),
        }


def resolve_wide_fields(frame: Any, logical_name: str) -> dict[str, str]:
    normalized: dict[str, list[str]] = defaultdict(list)
    for column in frame.columns:
        normalized[normalize_identifier(column)].append(str(column))
    resolved: dict[str, str] = {}
    ambiguities: list[str] = []
    for canonical, aliases in WIDE_FIELD_ALIASES.items():
        for alias in aliases:
            matches = sorted(set(normalized.get(normalize_identifier(alias), [])))
            if len(matches) == 1:
                resolved[canonical] = matches[0]
                break
            if len(matches) > 1:
                ambiguities.append(f"{canonical}: {matches}")
                break
    target_columns: dict[str, str] = {}
    for outcome, month, aliases, _ in WIDE_TARGET_FIELDS:
        canonical = f"target_{outcome}_{month}"
        for alias in aliases:
            matches = sorted(set(normalized.get(normalize_identifier(alias), [])))
            if len(matches) == 1:
                target_columns[canonical] = matches[0]
                break
            if len(matches) > 1:
                ambiguities.append(f"{canonical}: {matches}")
                break
    if ambiguities:
        raise PreflightError(
            f"{logical_name} wide-column mapping is ambiguous",
            ambiguities,
            ["Keep only one reviewed alias for each listed field in the direct cohort query."],
        )
    resolved.update(target_columns)
    required = {
        "MBSCohort": {"patient_id", "procedure_date", "procedure_code", "age", "active_end_days", "target_bmi_0"},
        "GLP1Cohort": {"patient_id", "glp1_start", "glp1_name", "age", "active_end_days", "target_bmi_0"},
    }[logical_name]
    missing = sorted(required.difference(resolved))
    if logical_name == "GLP1Cohort" and not ({"glp1_end", "glp1_duration"}.intersection(resolved)):
        missing.append("one of glp1_end or glp1_duration")
    if missing:
        raise PreflightError(
            f"{logical_name} direct query lacks required columns",
            ["Missing wide field: " + item for item in missing],
            [f"The script queried {logical_name} itself, but the returned schema cannot construct that cohort."],
        )
    return resolved


def direct_source_names() -> tuple[str, str, str]:
    schema = os.environ.get("METABOLIC_SOURCE_SCHEMA", DIRECT_SOURCE_SCHEMA)
    mbs_table = os.environ.get("METABOLIC_MBS_TABLE", DIRECT_MBS_TABLE)
    glp1_table = os.environ.get("METABOLIC_GLP1_TABLE", DIRECT_GLP1_TABLE)
    for value in (schema, mbs_table, glp1_table):
        quote_identifier(value)
    return schema, mbs_table, glp1_table


def build_direct_schema_probe_sql(schema: str, table: str) -> str:
    return (
        f"/* {DIRECT_WIDE_CONTRACT_VERSION}: column probe; no patient rows */\n"
        "SELECT TOP (0) *\n"
        f"FROM {quote_identifier(schema)}.{quote_identifier(table)}"
    )


def build_direct_source_count_sql(
    schema: str,
    table: str,
    patient_column: str,
) -> str:
    qualified = f"{quote_identifier(schema)}.{quote_identifier(table)}"
    patient = quote_identifier(patient_column)
    return (
        f"/* {DIRECT_WIDE_CONTRACT_VERSION}: aggregate source count; no patient values */\n"
        "SELECT\n"
        "    COUNT_BIG(*) AS [source_rows],\n"
        f"    COUNT_BIG(DISTINCT {patient}) AS [source_patients]\n"
        f"FROM {qualified}"
    )


def build_direct_source_sql(
    schema: str,
    table: str,
    resolved: Mapping[str, str],
    patient_limit: int | None = None,
) -> str:
    qualified = f"{quote_identifier(schema)}.{quote_identifier(table)}"
    patient_column = quote_identifier(resolved["patient_id"])
    projected_columns = list(dict.fromkeys(resolved.values()))
    projection = ",\n    ".join(
        f"[source].{quote_identifier(column)}" for column in projected_columns
    )
    marker = f"/* {DIRECT_WIDE_CONTRACT_VERSION}: direct wide cohort input */"
    if patient_limit is None:
        return (
            f"{marker}\n"
            f"SELECT\n    {projection}\n"
            f"FROM {qualified} AS [source]"
        )
    return (
        f"{marker}\n"
        "WITH [sampled_patients] AS (\n"
        f"    SELECT TOP ({int(patient_limit)}) {patient_column}\n"
        f"    FROM {qualified}\n"
        f"    WHERE {patient_column} IS NOT NULL\n"
        f"    GROUP BY {patient_column}\n"
        f"    ORDER BY {patient_column}\n"
        ")\n"
        f"SELECT\n    {projection}\n"
        f"FROM {qualified} AS [source]\n"
        "INNER JOIN [sampled_patients] AS [sample]\n"
        f"    ON [sample].{patient_column} = [source].{patient_column}"
    )


def load_direct_wide_tables(
    connection: Any,
    cfg: RunConfig,
    read_sql: Callable[[str, Any], Any] | None = None,
    preflight_only: bool = False,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    reader = read_sql or pd.read_sql_query
    schema, mbs_table, glp1_table = direct_source_names()
    patient_limit = cfg.smoke_query_limit if cfg.smoke else (2000 if preflight_only or cfg.mode == "preflight-only" else None)
    sources = {"MBSCohort": mbs_table, "GLP1Cohort": glp1_table}
    frames: dict[str, Any] = {}
    sql: dict[str, str] = {}
    qualified: dict[str, str] = {}
    source_totals: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    for logical_name, table_name in sources.items():
        probe = build_direct_schema_probe_sql(schema, table_name)
        try:
            schema_frame = reader(probe, connection)
            if not len(schema_frame.columns):
                raise ValueError("column probe returned no columns")
            resolved = resolve_wide_fields(schema_frame, logical_name)
            count_query = build_direct_source_count_sql(
                schema,
                table_name,
                resolved["patient_id"],
            )
            count_frame = reader(count_query, connection)
            if count_frame.empty:
                raise ValueError("aggregate source count returned no rows")
            count_row = count_frame.iloc[0]
            source_totals[logical_name] = {
                "rows": int(count_row["source_rows"]),
                "patients": int(count_row["source_patients"]),
            }
            query = build_direct_source_sql(
                schema,
                table_name,
                resolved,
                patient_limit,
            )
            frame = reader(query, connection)
        except Exception as exc:
            failures.append(f"{logical_name}: {type(exc).__name__}: {sanitize_exception_text(exc)}")
            continue
        if frame.empty:
            failures.append(f"{logical_name}: the direct query returned zero rows")
            continue
        frames[logical_name] = frame.drop_duplicates().reset_index(drop=True)
        sql[logical_name] = query
        sql[f"{logical_name}_aggregate_count"] = count_query
        qualified[logical_name] = f"{schema}.{table_name}"
    if failures:
        raise PreflightError(
            "Cosmos direct cohort query failed",
            failures,
            [
                "Production directly queries the configured MBSCohort and GLP1Cohort objects through pyodbc.",
                "Confirm SELECT access, table names, and the ProjectD332AFD database connection.",
            ],
        )
    return frames, sql, qualified, source_totals


def wide_value(row: Any, resolved: Mapping[str, str], canonical: str, default: Any = math.nan) -> Any:
    column = resolved.get(canonical)
    return row[column] if column is not None and column in row else default


def wide_numeric(value: Any, default: float = math.nan) -> float:
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    except (TypeError, ValueError):
        return default


def wide_date(value: Any) -> Any:
    parsed = pd.to_datetime(value, errors="coerce")
    return pd.Timestamp(parsed).normalize() if pd.notna(parsed) else pd.NaT


def reported_medication_end(start: Any, end: Any, duration: Any) -> Any:
    start_date = wide_date(start)
    end_date = wide_date(end)
    if pd.notna(end_date):
        return end_date
    duration_days = wide_numeric(duration)
    if pd.notna(start_date) and math.isfinite(duration_days) and duration_days >= 1:
        try:
            return start_date + pd.Timedelta(days=int(round(duration_days)) - 1)
        except (OverflowError, ValueError):
            return pd.NaT
    return pd.NaT


def resolve_wide_followup_days(
    frame: Any,
    resolved: Mapping[str, str],
) -> Any:
    """Resolve ActiveEndInterval without assuming every numeric value is a day count.

    Reported values are retained only within a generous 50-year plausibility bound and
    are truncated at the final study window because later opportunity is unused. A
    non-null wide outcome proves observation through its nominal horizon, so it can
    extend a short reported interval or provide a conservative fallback when the source
    interval is missing, negative, infinite, or implausibly large.
    """
    raw = pd.to_numeric(frame[resolved["active_end_days"]], errors="coerce").astype(float)
    observed_horizon = pd.Series(np.nan, index=frame.index, dtype=float)
    for outcome, month, _, _ in WIDE_TARGET_FIELDS:
        source_column = resolved.get(f"target_{outcome}_{month}")
        if source_column is None:
            continue
        values = pd.to_numeric(frame[source_column], errors="coerce")
        day = 0 if month == 0 else month_to_nominal_day(month)
        present = values.notna()
        observed_horizon.loc[present] = observed_horizon.loc[present].fillna(0).clip(
            lower=day
        )

    finite = np.isfinite(raw)
    plausible = finite & raw.between(0, MAX_PLAUSIBLE_WIDE_FOLLOWUP_DAYS)
    effective = pd.Series(np.nan, index=frame.index, dtype=float)
    effective.loc[plausible] = raw.loc[plausible].clip(
        upper=MAX_WIDE_STUDY_FOLLOWUP_DAYS
    )
    extend = plausible & observed_horizon.notna() & observed_horizon.gt(effective)
    effective.loc[extend] = observed_horizon.loc[extend].clip(
        upper=MAX_WIDE_STUDY_FOLLOWUP_DAYS
    )
    fallback = ~plausible & observed_horizon.notna()
    effective.loc[fallback] = observed_horizon.loc[fallback].clip(
        upper=MAX_WIDE_STUDY_FOLLOWUP_DAYS
    )

    method = pd.Series("unusable", index=frame.index, dtype="string")
    method.loc[plausible] = "reported_days"
    method.loc[plausible & raw.gt(MAX_WIDE_STUDY_FOLLOWUP_DAYS)] = (
        "reported_days_capped_to_study_horizon"
    )
    method.loc[extend] = "reported_days_extended_by_outcome_horizon"
    method.loc[fallback] = "outcome_horizon_fallback"
    return pd.DataFrame(
        {
            "raw_active_end": raw,
            "observed_outcome_horizon": observed_horizon,
            "effective_active_end": effective,
            "resolution_method": method,
        },
        index=frame.index,
    )


def select_primary_incretin_episode(
    episodes: Sequence[CoverageEpisode],
) -> CoverageEpisode | None:
    """Return the earliest persistent episode with a 365-day interepisode washout."""
    ordered = sorted(episodes, key=lambda item: (item.start_day, item.supported_end_day))
    for episode in ordered:
        previous_supported = [
            prior.supported_end_day
            for prior in ordered
            if prior.supported_end_day < episode.start_day
        ]
        new_user = not previous_supported or episode.start_day - max(previous_supported) - 1 >= 365
        if episode.qualifies_183 and new_user:
            return episode
    return None


def select_wide_index_rows(
    frame: Any,
    resolved: Mapping[str, str],
    logical_name: str,
    preferred_anchors: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Select one deterministic clinically aligned index row per patient and source."""
    patient_column = resolved["patient_id"]
    index_field = "procedure_date" if logical_name == "MBSCohort" else "glp1_start"
    patient_ids = frame[patient_column].astype("string").str.strip()
    nonblank = patient_ids.notna() & patient_ids.ne("")
    index_dates = pd.to_datetime(frame[resolved[index_field]], errors="coerce").dt.normalize()
    followup = resolve_wide_followup_days(frame, resolved)
    active_end = followup["effective_active_end"]
    valid = nonblank & index_dates.notna() & active_end.notna() & active_end.ge(0)
    candidates = pd.DataFrame(
        {
            "_row_index": frame.index,
            "_patient_id": patient_ids,
            "_index_date": index_dates,
            "_active_end": active_end,
            "_valid": valid,
        },
        index=frame.index,
    )
    target_columns = [
        column for canonical, column in resolved.items() if canonical.startswith("target_")
    ]
    mapped_columns = list(dict.fromkeys(resolved.values()))
    candidates["_row_hash"] = pd.util.hash_pandas_object(
        frame[mapped_columns], index=False
    ).astype("uint64")
    candidates["_outcome_count"] = (
        frame[target_columns].notna().sum(axis=1) if target_columns else 0
    )
    candidates["_baseline_bmi"] = pd.to_numeric(
        frame[resolved["target_bmi_0"]], errors="coerce"
    )
    candidates["_source_end"] = pd.NaT
    candidates["_tie_text"] = ""
    anchor_count = 0
    if logical_name == "MBSCohort":
        codes = frame[resolved["procedure_code"]].astype("string").fillna("")
        candidates["_qualifying"] = codes.map(procedure_category).notna()
        candidates["_tie_text"] = codes
        candidates = candidates.loc[candidates["_valid"]].sort_values(
            [
                "_patient_id", "_qualifying", "_index_date", "_outcome_count",
                "_active_end", "_baseline_bmi", "_tie_text", "_row_hash", "_row_index",
            ],
            ascending=[True, False, True, False, False, True, True, True, True],
            kind="mergesort",
        )
        rule = "earliest valid qualifying bariatric procedure; deterministic completeness tie-break"
    else:
        end_dates = pd.to_datetime(
            frame[resolved["glp1_end"]], errors="coerce"
        ).dt.normalize() if "glp1_end" in resolved else pd.Series(pd.NaT, index=frame.index)
        if "glp1_duration" in resolved:
            durations = pd.to_numeric(frame[resolved["glp1_duration"]], errors="coerce")
            derived_ends = index_dates + pd.to_timedelta(
                durations.round().sub(1), unit="D", errors="coerce"
            )
            end_dates = end_dates.fillna(derived_ends)
        candidates["_source_end"] = end_dates
        candidates["_tie_text"] = frame[resolved["glp1_name"]].astype("string").fillna("")
        anchors = pd.Series(preferred_anchors or {}, dtype="object")
        candidates["_preferred_anchor"] = candidates["_patient_id"].map(anchors)
        candidates["_preferred_anchor"] = pd.to_datetime(
            candidates["_preferred_anchor"], errors="coerce"
        ).dt.normalize()
        candidates["_anchor_match"] = candidates["_index_date"].eq(
            candidates["_preferred_anchor"]
        )
        anchor_count = int(candidates.loc[candidates["_anchor_match"] & candidates["_valid"], "_patient_id"].nunique())
        candidates = candidates.loc[
            candidates["_valid"]
            & (candidates["_preferred_anchor"].isna() | candidates["_anchor_match"])
        ].sort_values(
            [
                "_patient_id", "_index_date", "_source_end", "_outcome_count",
                "_active_end", "_baseline_bmi", "_tie_text", "_row_hash", "_row_index",
            ],
            ascending=[True, True, False, False, False, True, True, True, True],
            kind="mergesort",
        )
        rule = (
            "row starting the earliest qualifying 183-day episode after a 365-day "
            "interepisode washout; earliest valid row retained only when no episode qualifies"
        )
    selected_indices = candidates.drop_duplicates("_patient_id", keep="first")["_row_index"]
    selected = frame.loc[selected_indices].copy().reset_index(drop=True)
    patient_counts = patient_ids.loc[nonblank].value_counts()
    audit = {
        "rule": rule,
        "input_rows": int(len(frame)),
        "source_patients": int(patient_counts.size),
        "multirow_patients": int(patient_counts.gt(1).sum()),
        "valid_index_rows": int(valid.sum()),
        "selected_index_rows": int(len(selected)),
        "patients_without_valid_index_row": int(patient_counts.size - candidates["_patient_id"].nunique()),
        "active_end_resolution_counts": {
            str(key): int(value)
            for key, value in followup["resolution_method"].value_counts().sort_index().items()
        },
    }
    if logical_name == "GLP1Cohort":
        audit.update(
            {
                "qualifying_episode_anchors": int(len(preferred_anchors or {})),
                "qualifying_anchors_with_valid_wide_row": anchor_count,
            }
        )
    return selected, audit


def merge_patient_payload(existing: dict[str, Any], candidate: Mapping[str, Any]) -> None:
    for column in ("observation_start_date",):
        values = [wide_date(existing.get(column)), wide_date(candidate.get(column))]
        valid = [value for value in values if pd.notna(value)]
        existing[column] = min(valid) if valid else pd.NaT
    for column in ("observation_end_date", "administrative_end_date"):
        values = [wide_date(existing.get(column)), wide_date(candidate.get(column))]
        valid = [value for value in values if pd.notna(value)]
        existing[column] = max(valid) if valid else pd.NaT
    for column in (
        "prior_incretin_flag", "postop_incretin_flag", "prior_mbs_flag", "dialysis_transplant_flag", "diabetes_flag",
        "mbs_during_incretin_flag", "hypertension", "dyslipidemia", "osa", "insulin", "biguanide", "sglt2",
    ):
        existing[column] = max(wide_numeric(existing.get(column), 0.0), wide_numeric(candidate.get(column), 0.0))
    if existing.get("center_id") == CENTER_UNAVAILABLE and candidate.get("center_id") != CENTER_UNAVAILABLE:
        existing["center_id"] = candidate.get("center_id")
    for column in ("birth_year", "sex", "race", "ethnicity", "coverage", "svi", "ruca", "state"):
        current = existing.get(column)
        replacement = candidate.get(column)
        if current is None or pd.isna(current) or str(current).lower() in {"", "nan", "unknown"}:
            existing[column] = replacement
    sources = sorted(set(str(existing.get("source_table", "")).split("|") + str(candidate.get("source_table", "")).split("|")))
    existing["source_table"] = "|".join(item for item in sources if item)


def wide_tables_to_data_bundle(
    frames: Mapping[str, Any],
    sql: Mapping[str, str],
    qualified_names: Mapping[str, str],
    source_totals: Mapping[str, Mapping[str, int]] | None = None,
    preflight_only: bool = False,
) -> DataBundle:
    mbs = frames["MBSCohort"].copy().reset_index(drop=True)
    glp1 = frames["GLP1Cohort"].copy().reset_index(drop=True)
    resolved_by_source = {
        "MBSCohort": resolve_wide_fields(mbs, "MBSCohort"),
        "GLP1Cohort": resolve_wide_fields(glp1, "GLP1Cohort"),
    }
    procedure_rows: list[dict[str, Any]] = []
    medication_rows: list[dict[str, Any]] = []

    # Preserve every reported procedure and exposure interval. Patient attributes and
    # nominal outcomes are selected only after the primary index episode is known.
    for logical_name, frame in (("MBSCohort", mbs), ("GLP1Cohort", glp1)):
        resolved = resolved_by_source[logical_name]
        source_table = qualified_names[logical_name]
        for row_number, (_, row) in enumerate(frame.iterrows()):
            patient_raw = wide_value(row, resolved, "patient_id", "")
            if pd.isna(patient_raw) or not str(patient_raw).strip():
                continue
            patient_id = str(patient_raw).strip()

            if logical_name == "MBSCohort":
                procedure_date = wide_date(wide_value(row, resolved, "procedure_date"))
                if pd.notna(procedure_date):
                    procedure_rows.append(
                        {
                            "patient_id": patient_id,
                            "procedure_date": procedure_date,
                            "procedure_code": wide_value(row, resolved, "procedure_code", ""),
                            "procedure_concept_id": wide_value(row, resolved, "procedure_code", ""),
                            "source_table": source_table,
                        }
                    )

            medication_start = wide_date(wide_value(row, resolved, "glp1_start"))
            medication_end = reported_medication_end(
                medication_start,
                wide_value(row, resolved, "glp1_end"),
                wide_value(row, resolved, "glp1_duration"),
            )
            reported_exposure = logical_name == "GLP1Cohort" or any(
                wide_numeric(wide_value(row, resolved, flag), 0.0) == 1
                for flag in ("prior_glp1", "postop_glp1")
            )
            if reported_exposure:
                ingredient = wide_value(row, resolved, "glp1_name", "")
                medication_rows.append(
                    {
                        "patient_id": patient_id,
                        "ingredient": ingredient,
                        "medication_concept": ingredient,
                        "route": wide_value(row, resolved, "glp1_route", "unknown"),
                        "formulation": "unknown",
                        "medication_start_date": medication_start,
                        "medication_end_date": medication_end,
                        "source_type": "reported_wide_episode",
                        "source_id": f"{logical_name}-{row_number}",
                        "dose": wide_numeric(wide_value(row, resolved, "dose")),
                        "dose_unit": wide_value(row, resolved, "dose_unit", ""),
                        "source_table": source_table,
                        "source_cohort": "surgery" if logical_name == "MBSCohort" else "incretin",
                    }
                )

    procedure_columns = [
        "patient_id", "procedure_date", "procedure_code", "procedure_concept_id", "source_table",
    ]
    medication_columns = [
        "patient_id", "ingredient", "medication_concept", "route", "formulation",
        "medication_start_date", "medication_end_date", "source_type", "source_id", "dose",
        "dose_unit", "source_table", "source_cohort",
    ]
    procedures = pd.DataFrame(procedure_rows, columns=procedure_columns).drop_duplicates(ignore_index=True)
    medications = pd.DataFrame(medication_rows, columns=medication_columns).drop_duplicates(ignore_index=True)

    incretin_medications = medications.loc[medications["source_cohort"].eq("incretin")]
    incretin_records, incretin_medication_audit = medication_frame_to_coverage(incretin_medications)
    incretin_episodes, _ = reconstruct_coverage_episodes(incretin_records)
    episodes_by_patient: dict[str, list[CoverageEpisode]] = defaultdict(list)
    for episode in incretin_episodes:
        episodes_by_patient[episode.patient_id].append(episode)
    epoch = pd.Timestamp("1970-01-01")
    preferred_glp1_anchors = {
        patient_id: epoch + pd.Timedelta(days=selected.start_day)
        for patient_id, episodes in episodes_by_patient.items()
        if (selected := select_primary_incretin_episode(episodes)) is not None
    }

    selected_frames: dict[str, Any] = {}
    index_selection: dict[str, dict[str, Any]] = {}
    for logical_name, frame in (("MBSCohort", mbs), ("GLP1Cohort", glp1)):
        anchors = preferred_glp1_anchors if logical_name == "GLP1Cohort" else None
        selected_frames[logical_name], index_selection[logical_name] = select_wide_index_rows(
            frame,
            resolved_by_source[logical_name],
            logical_name,
            preferred_anchors=anchors,
        )

    patients_by_id: dict[str, dict[str, Any]] = {}
    measurement_rows: list[dict[str, Any]] = []
    selected_center_values: list[str] = []
    for logical_name in ("MBSCohort", "GLP1Cohort"):
        frame = selected_frames[logical_name].copy()
        resolved = resolved_by_source[logical_name]
        source_table = qualified_names[logical_name]
        index_field = "procedure_date" if logical_name == "MBSCohort" else "glp1_start"
        selected_followup = resolve_wide_followup_days(frame, resolved)
        frame["_effective_active_end_days"] = selected_followup["effective_active_end"]
        frame["_active_end_resolution_method"] = selected_followup["resolution_method"]
        for _, row in frame.iterrows():
            patient_id = str(wide_value(row, resolved, "patient_id", "")).strip()
            index_date = wide_date(wide_value(row, resolved, index_field))
            active_end_days = wide_numeric(row["_effective_active_end_days"])
            if not patient_id or pd.isna(index_date) or not math.isfinite(active_end_days) or active_end_days < 0:
                continue
            observation_end = index_date + pd.Timedelta(days=int(active_end_days))
            supplied_admin_end = wide_date(wide_value(row, resolved, "administrative_end_date"))
            administrative_end = supplied_admin_end if pd.notna(supplied_admin_end) else observation_end
            prior_glp1 = wide_numeric(wide_value(row, resolved, "prior_glp1"), 0.0)
            observation_start = index_date - pd.Timedelta(days=365) if prior_glp1 == 0 else index_date
            age = wide_numeric(wide_value(row, resolved, "age"))
            birth_year = int(index_date.year - age) if math.isfinite(age) else np.nan
            center_raw = wide_value(row, resolved, "center_id", CENTER_UNAVAILABLE)
            center_id = CENTER_UNAVAILABLE if pd.isna(center_raw) or not str(center_raw).strip() else str(center_raw)
            selected_center_values.append(center_id)
            candidate_patient = {
                "patient_id": patient_id,
                "center_id": center_id,
                "age": np.nan,
                "birth_year": birth_year,
                "sex": str(wide_value(row, resolved, "sex", "Unknown")),
                "race": str(wide_value(row, resolved, "race", "Unknown")),
                "ethnicity": str(wide_value(row, resolved, "ethnicity", "Unknown")),
                "coverage": str(wide_value(row, resolved, "coverage", "Unknown")),
                "observation_start_date": observation_start,
                "observation_end_date": observation_end,
                "administrative_end_date": administrative_end,
                "wide_index_date": index_date,
                "active_end_resolution_method": str(row["_active_end_resolution_method"]),
                "prior_incretin_flag": prior_glp1,
                "postop_incretin_flag": wide_numeric(wide_value(row, resolved, "postop_glp1"), 0.0),
                "prior_mbs_flag": wide_numeric(wide_value(row, resolved, "prior_mbs"), 0.0),
                "dialysis_transplant_flag": wide_numeric(wide_value(row, resolved, "dialysis_transplant"), 0.0),
                "diabetes_flag": wide_numeric(wide_value(row, resolved, "diabetes"), 0.0),
                "mbs_during_incretin_flag": wide_numeric(wide_value(row, resolved, "mbs_during_glp1"), 0.0),
                "smoking": "Unknown",
                "hypertension": wide_numeric(wide_value(row, resolved, "hypertension"), 0.0),
                "dyslipidemia": wide_numeric(wide_value(row, resolved, "dyslipidemia"), 0.0),
                "osa": wide_numeric(wide_value(row, resolved, "osa"), 0.0),
                "insulin": wide_numeric(wide_value(row, resolved, "insulin"), 0.0),
                "biguanide": wide_numeric(wide_value(row, resolved, "biguanide"), 0.0),
                "sglt2": wide_numeric(wide_value(row, resolved, "sglt2"), 0.0),
                "svi": wide_numeric(wide_value(row, resolved, "svi")),
                "ruca": str(wide_value(row, resolved, "ruca", "Unknown")),
                "state": str(wide_value(row, resolved, "state", "Unknown")),
                "source_table": source_table,
            }
            source_prefix = "mbs" if logical_name == "MBSCohort" else "glp1"
            source_specific = {
                f"{source_prefix}__{key}": value
                for key, value in candidate_patient.items()
                if key != "patient_id"
            }
            if patient_id in patients_by_id:
                merge_patient_payload(patients_by_id[patient_id], candidate_patient)
                patients_by_id[patient_id].update(source_specific)
            else:
                patients_by_id[patient_id] = {**candidate_patient, **source_specific}

            for outcome, month, _, unit in WIDE_TARGET_FIELDS:
                canonical = f"target_{outcome}_{month}"
                source_column = resolved.get(canonical)
                if source_column is None:
                    continue
                value = wide_numeric(row[source_column])
                if not math.isfinite(value):
                    continue
                measurement_day = 0 if month == 0 else month_to_nominal_day(month)
                if measurement_day > active_end_days:
                    continue
                measurement_rows.append(
                    {
                        "patient_id": patient_id,
                        "measurement_date": index_date + pd.Timedelta(days=measurement_day),
                        "measurement_type": outcome,
                        "raw_value": value,
                        "unit": unit,
                        "source_concept": source_column + " (wide nominal horizon)",
                        "source_table": source_table,
                        "source_cohort": "surgery" if logical_name == "MBSCohort" else "incretin",
                        "index_anchor_date": index_date,
                        "timing_precision": "nominal_horizon_from_wide_column" if month else "index_event_date",
                    }
                )

    patients = pd.DataFrame(patients_by_id.values())
    measurements = pd.DataFrame(measurement_rows).drop_duplicates()
    encounters = pd.DataFrame(columns=["patient_id", "encounter_date", "source_table"])
    diagnoses = pd.DataFrame(columns=["patient_id", "diagnosis_date", "diagnosis_code", "source_table"])
    center_complete = bool(selected_center_values) and all(
        value != CENTER_UNAVAILABLE for value in selected_center_values
    )
    center_validation_available = center_complete and len(set(selected_center_values)) >= 3
    limitations = [
        "MBSCohort and GLP1Cohort are upstream analytic cohort tables; their defining SQL and upstream inclusion transforms are not visible in this database.",
        "Outcome values come from fixed wide horizon columns; exact measurement timestamps and within-window counts are unavailable.",
        "Each GLP1StartDate and GLP1EndDate pair is treated as one reported exposure interval; fill-level gaps and adherence within that interval cannot be audited.",
        "For patients with multiple GLP1 rows, nominal outcomes are anchored only to the earliest episode satisfying the 183-day persistence and 365-day interepisode washout rules.",
        "PriorGLP1 or PostOpGLP1 flags without an accepted dated interval are excluded rather than assigned a proxy treatment date.",
        "Observation start is operationalized as 365 days before index when PriorGLP1 is zero; the wide tables do not expose exact enrollment start.",
    ]
    if not all("administrative_end_date" in resolved for resolved in resolved_by_source.values()):
        limitations.append(
            "Plausible ActiveEndInterval day counts are bounded at the final study window; missing, negative, infinite, or implausibly large values use the latest non-null wide outcome horizon as conservative opportunity."
        )
    if not center_validation_available:
        limitations.append("A usable center identifier is unavailable, so geographic holdout validation is not performed.")
    bundle = DataBundle(
        patients=patients,
        procedures=procedures,
        medications=medications,
        measurements=measurements,
        encounters=encounters,
        diagnoses=diagnoses,
        metadata={
            "source_mode": "cosmos_direct_wide_cohorts",
            "sql_contract_version": DIRECT_WIDE_CONTRACT_VERSION,
            "sql": dict(sql),
            "query_fingerprint": digest({key: normalize_sql(value) for key, value in sql.items()}),
            "schema_fingerprint": digest({key: frame_schema(value) for key, value in frames.items()}),
            "source_tables": dict(qualified_names),
            "source_row_counts": {key: len(value) for key, value in frames.items()},
            "source_unique_patient_counts": {
                logical_name: int(
                    frame[resolved_by_source[logical_name]["patient_id"]].dropna().astype(str).nunique()
                )
                for logical_name, frame in (("MBSCohort", mbs), ("GLP1Cohort", glp1))
            },
            "source_total_row_counts": {
                logical_name: int((source_totals or {}).get(logical_name, {}).get("rows", len(frame)))
                for logical_name, frame in (("MBSCohort", mbs), ("GLP1Cohort", glp1))
            },
            "source_total_unique_patient_counts": {
                logical_name: int(
                    (source_totals or {}).get(logical_name, {}).get(
                        "patients",
                        frame[resolved_by_source[logical_name]["patient_id"]].dropna().astype(str).nunique(),
                    )
                )
                for logical_name, frame in (("MBSCohort", mbs), ("GLP1Cohort", glp1))
            },
            "index_row_selection": index_selection,
            "accepted_incretin_interval_records": int(len(incretin_records)),
            "rejected_incretin_interval_records": int(
                (~incretin_medication_audit["accepted"]).sum()
                if not incretin_medication_audit.empty else 0
            ),
            "measurement_timing": "nominal_horizon_from_wide_columns",
            "medication_coverage_semantics": "reported_start_end_intervals_across_all_patient_rows",
            "center_validation_available": center_validation_available,
            "strict_raw_event_contract": False,
            "limitations": limitations,
        },
    )
    bundle.metadata["preflight"] = validate_data_bundle(bundle, preflight_only=preflight_only)
    return bundle


def medication_frame_to_coverage(medications: Any, index_dates: Mapping[str, Any] | None = None) -> tuple[list[CoverageRecord], Any]:
    records: list[CoverageRecord] = []
    audit: list[dict[str, Any]] = []
    for row in medications.itertuples(index=False):
        payload = row._asdict()
        patient_id = str(payload.get("patient_id", ""))
        ingredient, therapy_class, mapping_method = normalize_ingredient(payload.get("ingredient") or payload.get("medication_concept"))
        source_type_value = str(payload.get("source_type") or "").strip().lower()
        if "fill_date" in payload and pd.notna(payload.get("fill_date")) and pd.notna(payload.get("days_supply")):
            start_date = pd.Timestamp(payload["fill_date"])
            days_supply = int(float(payload["days_supply"]))
            end_date = start_date + pd.Timedelta(days=days_supply - 1)
            source_type = "fill"
        elif "administration_date" in payload and pd.notna(payload.get("administration_date")):
            start_date = pd.Timestamp(payload["administration_date"])
            interval = payload.get("administration_interval_days")
            interval_days = int(float(interval)) if interval is not None and pd.notna(interval) else 7
            end_date = start_date + pd.Timedelta(days=max(1, interval_days) - 1)
            source_type = "administration"
        elif pd.notna(payload.get("medication_start_date")) and pd.notna(payload.get("medication_end_date")):
            start_date = pd.Timestamp(payload["medication_start_date"])
            end_date = pd.Timestamp(payload["medication_end_date"])
            source_type = "validated_episode" if "validated" in source_type_value else "explicit_treatment"
        else:
            start_date = end_date = pd.NaT
            source_type = "unknown"
        reason = ""
        if ingredient is None:
            reason = "unmapped_ingredient"
        elif pd.isna(start_date) or pd.isna(end_date):
            reason = "coverage_interval_unresolved"
        elif patient_id not in (index_dates or {}):
            reason = "missing_index_date" if index_dates is not None else ""
        if index_dates is None:
            origin = pd.Timestamp("1970-01-01")
        else:
            origin = pd.Timestamp(index_dates.get(patient_id)) if patient_id in index_dates else pd.Timestamp("1970-01-01")
        if not reason:
            record = CoverageRecord(
                patient_id=patient_id,
                start_day=int((start_date - origin).days),
                end_day=int((end_date - origin).days),
                ingredient=str(ingredient),
                therapy_class=str(therapy_class),
                route=normalize_route(payload.get("route")),
                formulation=str(payload.get("formulation") or "unknown"),
                source_type=source_type,
                source_table=str(payload.get("source_table") or "unknown"),
                source_id=str(payload.get("source_id") or ""),
                dose=float(payload["dose"]) if payload.get("dose") is not None and pd.notna(payload.get("dose")) else None,
                dose_unit=str(payload.get("dose_unit") or ""),
            )
            checked = validate_coverage_record(record)
            if checked.accepted:
                records.append(checked)
            else:
                reason = checked.rejection_reason
        audit.append(
            {
                "patient_id": patient_id,
                "ingredient": ingredient or "unmapped",
                "mapping_method": mapping_method,
                "source_type": source_type,
                "accepted": not bool(reason),
                "reason": reason or "valid",
            }
        )
    return records, pd.DataFrame(audit)


def validate_data_bundle(bundle: DataBundle, preflight_only: bool = False) -> dict[str, Any]:
    issues: list[str] = []
    details: list[str] = []
    wide_source = bundle.metadata.get("source_mode") == "cosmos_direct_wide_cohorts"
    if bundle.patients.empty or bundle.patients["patient_id"].isna().any():
        issues.append("Stable patient identifiers are missing or null")
    if "center_id" not in bundle.patients or bundle.patients["center_id"].isna().any():
        issues.append("Blinded center or organization identity is incomplete")
    for name, frame, date_column in (
        ("procedure", bundle.procedures, "procedure_date"),
        ("measurement", bundle.measurements, "measurement_date"),
    ):
        if frame.empty or date_column not in frame or frame[date_column].isna().any():
            issues.append(f"{name.title()} dates are unavailable or invalid")
    for field_name in ("raw_value", "unit", "source_concept", "source_table"):
        if field_name not in bundle.measurements:
            issues.append(f"Raw measurements lack required field {field_name}")
    admin_available = "administrative_end_date" in bundle.patients and bundle.patients["administrative_end_date"].notna().all()
    global_admin = os.environ.get("METABOLIC_ADMIN_DATA_THROUGH")
    if not admin_available and not global_admin:
        issues.append("Administrative data-through date is unavailable")
    if global_admin and not admin_available:
        parsed = pd.to_datetime(global_admin, errors="coerce")
        if pd.isna(parsed):
            issues.append("METABOLIC_ADMIN_DATA_THROUGH is not a valid date")
        else:
            bundle.patients["administrative_end_date"] = pd.Timestamp(parsed).normalize()
            details.append("Administrative data-through date supplied by reviewed environment configuration")
    records, medication_audit = medication_frame_to_coverage(bundle.medications)
    accepted = int(medication_audit["accepted"].sum()) if not medication_audit.empty else 0
    if accepted == 0:
        issues.append("No medication record yields accepted audited coverage semantics")
    if "postop_incretin_flag" in bundle.patients:
        flagged = bundle.patients.loc[pd.to_numeric(bundle.patients["postop_incretin_flag"], errors="coerce").eq(1), "patient_id"].astype(str)
        timing_medications = bundle.medications
        if wide_source and "source_cohort" in timing_medications:
            timing_medications = timing_medications.loc[timing_medications["source_cohort"].eq("surgery")]
        timing_records, _ = medication_frame_to_coverage(timing_medications)
        procedure_dates = bundle.procedures[["patient_id", "procedure_date"]].copy()
        procedure_dates["patient_id"] = procedure_dates["patient_id"].astype(str)
        procedure_dates["procedure_date"] = pd.to_datetime(
            procedure_dates["procedure_date"], errors="coerce"
        )
        procedure_index = procedure_dates.dropna(subset=["procedure_date"]).groupby(
            "patient_id"
        )["procedure_date"].min().to_dict()
        epoch = pd.Timestamp("1970-01-01")
        medication_patients = {
            record.patient_id
            for record in timing_records
            if record.patient_id in procedure_index
            and record.start_day >= int((pd.Timestamp(procedure_index[record.patient_id]) - epoch).days)
        }
        unresolved = sorted(set(flagged).difference(medication_patients))
        if unresolved:
            if wide_source:
                details.append(
                    f"{len(unresolved)} flagged surgical patients lack a reported postoperative start and will be excluded from the primary cohort"
                )
            elif not preflight_only:
                issues.append(f"Postoperative incretin start is unresolved for {len(unresolved)} flagged surgical patients")
                details.append("Every flagged postoperative exposure requires an exact accepted start date; no proxy date is permitted")
    if issues:
        raise PreflightError("Production data preflight failed", issues, details)
    limitations = list(bundle.metadata.get("limitations", [])) if wide_source else []
    details.extend(limitations)
    result = {
        "status": "passed_with_wide_source_limitations" if wide_source else "passed",
        "row_counts": bundle.row_counts(),
        "accepted_medication_records": accepted,
        "medication_rejection_counts": medication_audit.loc[~medication_audit["accepted"], "reason"].value_counts().to_dict(),
        "details": details,
        "strict_raw_event_contract": not wide_source,
    }
    if wide_source:
        index_selection = dict(bundle.metadata.get("index_row_selection", {}))
        sampled_patients = dict(bundle.metadata.get("source_unique_patient_counts", {}))
        total_patients = dict(bundle.metadata.get("source_total_unique_patient_counts", sampled_patients))
        sampling_fractions = {
            source: (
                float(sampled_patients.get(source, 0)) / float(total)
                if float(total) > 0 else math.nan
            )
            for source, total in total_patients.items()
        }
        result.update(
            {
                "source_row_counts": dict(bundle.metadata.get("source_row_counts", {})),
                "source_unique_patient_counts": sampled_patients,
                "source_total_row_counts": dict(bundle.metadata.get("source_total_row_counts", {})),
                "source_total_unique_patient_counts": total_patients,
                "source_patient_sampling_fractions": sampling_fractions,
                "measurement_timing": bundle.metadata.get("measurement_timing"),
                "medication_coverage_semantics": bundle.metadata.get("medication_coverage_semantics"),
                "center_validation_available": bool(bundle.metadata.get("center_validation_available")),
                "index_row_selection": index_selection,
                "active_end_resolution_counts": {
                    source: dict(audit.get("active_end_resolution_counts", {}))
                    for source, audit in index_selection.items()
                },
                "limitations": limitations,
            }
        )
    return result


def connect_cosmos() -> Any:
    try:
        import pyodbc
    except ImportError as exc:
        raise PreflightError(
            "Cosmos connection dependency is unavailable",
            [f"pyodbc could not be imported: {exc}"],
            ["Install the reviewed SQL Server ODBC stack on the Cosmos VM."],
        ) from exc
    connection_string = os.environ.get("COSMOS_CONNECTION_STRING", DEFAULT_CONNECTION_STRING)
    try:
        return pyodbc.connect(connection_string, timeout=CONNECTION_TIMEOUT_SECONDS)
    except Exception as exc:
        raise PreflightError(
            "Cosmos connection failed",
            [f"{type(exc).__name__}: {sanitize_exception_text(exc)}"],
            ["Confirm PROJECTS access, ProjectD332AFD, trusted authentication, and ODBC Driver 17."],
        ) from exc


def query_cosmos(cfg: RunConfig, preflight_only: bool = False) -> DataBundle:
    connection = connect_cosmos()
    try:
        frames, sql, qualified_names, source_totals = load_direct_wide_tables(
            connection,
            cfg,
            preflight_only=preflight_only,
        )
        return wide_tables_to_data_bundle(
            frames,
            sql,
            qualified_names,
            source_totals=source_totals,
            preflight_only=preflight_only,
        )
    finally:
        connection.close()


def frame_schema(frame: Any) -> dict[str, str]:
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def uniqueness_manifest(payload: Any) -> dict[str, Any]:
    if isinstance(payload, DataBundle):
        return {
            "patients_patient_id_unique": bool(payload.patients["patient_id"].is_unique),
            "patient_count": int(payload.patients["patient_id"].nunique()),
        }
    if hasattr(payload, "columns") and "patient_id" in payload:
        return {"patient_count": int(payload["patient_id"].astype(str).nunique())}
    return {}


def payload_manifest(payload: Any) -> dict[str, Any]:
    if isinstance(payload, DataBundle):
        return {
            "type": "DataBundle",
            "row_counts": payload.row_counts(),
            "schemas": {
                name: frame_schema(getattr(payload, name))
                for name in ("patients", "procedures", "medications", "measurements", "encounters", "diagnoses")
            },
            "uniqueness": uniqueness_manifest(payload),
        }
    if hasattr(payload, "shape") and hasattr(payload, "columns"):
        return {
            "type": "DataFrame",
            "row_count": int(len(payload)),
            "schema": frame_schema(payload),
            "uniqueness": uniqueness_manifest(payload),
        }
    if isinstance(payload, Mapping):
        return {
            "type": "Mapping",
            "children": {str(key): payload_manifest(value) for key, value in sorted(payload.items(), key=lambda item: str(item[0]))},
        }
    if isinstance(payload, (list, tuple)):
        return {"type": type(payload).__name__, "length": len(payload)}
    return {"type": type(payload).__name__}


@dataclass
class RunContext:
    cfg: RunConfig
    run_dir: Path
    fingerprint: str
    fingerprint_payload: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def internal(self) -> Path:
        return self.run_dir / "INTERNAL"

    @property
    def checkpoints(self) -> Path:
        return self.internal / "checkpoints"

    @property
    def aggregate(self) -> Path:
        return self.run_dir / "AGGREGATE"

    @property
    def export(self) -> Path:
        return self.run_dir / "FIGURES_TO_EXPORT"

    def initialize(self) -> None:
        for directory in (self.run_dir, self.internal, self.checkpoints, self.aggregate, self.export):
            directory.mkdir(parents=True, exist_ok=True)
        unexpected = [item.name for item in self.export.iterdir() if item.is_file()]
        if unexpected and not self.cfg.resume and self.cfg.mode != "plot-only":
            raise RuntimeError("Output directory already contains exports; use a new directory or --resume")
        manifest_path = self.run_dir / "run_manifest.json"
        existing = read_json(manifest_path, {}) or {}
        if existing and existing.get("fingerprint") != self.fingerprint:
            raise RuntimeError("Run directory fingerprint mismatch; stale artifacts cannot be resumed or replotted")
        atomic_json(
            manifest_path,
            {
                "study_version": STUDY_VERSION,
                "fingerprint": self.fingerprint,
                "fingerprint_payload": self.fingerprint_payload,
                "configuration": asdict(self.cfg),
                "created_utc": existing.get("created_utc", utc_now()),
            },
        )
        self.state = read_json(self.run_dir / "run_state.json", {}) or {
            "status": "running", "stages": {}, "errors": [], "resumed_stages": []
        }
        atomic_json(self.run_dir / "run_state.json", self.state)

    def stage_fingerprint(self, stage: str, upstream: Mapping[str, str] | None = None) -> str:
        return digest(
            {
                "run_fingerprint": self.fingerprint,
                "stage": stage,
                "upstream": dict(sorted((upstream or {}).items())),
            }
        )

    def save_checkpoint(
        self,
        stage: str,
        payload: Any,
        upstream: Mapping[str, str] | None = None,
        *,
        elapsed_seconds: float | None = None,
    ) -> str:
        body_path = self.checkpoints / f"{stage}.pkl"
        meta_path = self.checkpoints / f"{stage}.json"
        atomic_pickle(body_path, payload)
        artifact_hash = sha256_file(body_path)
        metadata = {
            "stage": stage,
            "stage_fingerprint": self.stage_fingerprint(stage, upstream),
            "artifact_sha256": artifact_hash,
            "payload_manifest": payload_manifest(payload),
            "upstream": dict(sorted((upstream or {}).items())),
            "completion_marker": "COMPLETE",
            "completed_utc": utc_now(),
        }
        atomic_json(meta_path, metadata)
        stage_state: dict[str, Any] = {
            "status": "complete",
            "artifact_sha256": artifact_hash,
        }
        if elapsed_seconds is not None:
            stage_state["seconds"] = float(elapsed_seconds)
        self.state.setdefault("stages", {})[stage] = stage_state
        atomic_json(self.run_dir / "run_state.json", self.state)
        return artifact_hash

    def load_checkpoint(self, stage: str, upstream: Mapping[str, str] | None = None) -> Any | None:
        if not (self.cfg.resume or self.cfg.mode == "plot-only"):
            return None
        body_path = self.checkpoints / f"{stage}.pkl"
        metadata = read_json(self.checkpoints / f"{stage}.json", {}) or {}
        expected = self.stage_fingerprint(stage, upstream)
        if metadata.get("completion_marker") != "COMPLETE":
            return None
        if metadata.get("stage_fingerprint") != expected or not body_path.exists():
            return None
        if sha256_file(body_path) != metadata.get("artifact_sha256"):
            return None
        try:
            with body_path.open("rb") as stream:
                payload = pickle.load(stream)
        except Exception:
            return None
        if payload_manifest(payload) != metadata.get("payload_manifest"):
            return None
        self.state.setdefault("resumed_stages", []).append(stage)
        self.state.setdefault("stages", {})[stage] = {"status": "resumed"}
        atomic_json(self.run_dir / "run_state.json", self.state)
        return payload


def make_run_context(cfg: RunConfig, bundle: DataBundle, dependencies: Mapping[str, Any]) -> RunContext:
    script_hash = sha256_file(SCRIPT_PATH)
    admin_dates = []
    if "administrative_end_date" in bundle.patients:
        admin_dates = sorted(
            pd.to_datetime(bundle.patients["administrative_end_date"], errors="coerce").dropna().dt.date.astype(str).unique()
        )
    scientific_configuration = asdict(cfg)
    scientific_configuration.pop("output_dir", None)
    scientific_configuration.pop("resume", None)
    fingerprint_payload = {
        "study_version": STUDY_VERSION,
        "script_sha256": script_hash,
        "query_fingerprint": bundle.metadata.get("query_fingerprint"),
        "normalized_sql": {
            key: normalize_sql(value) for key, value in bundle.metadata.get("sql", {}).items()
        },
        "schema_fingerprint": bundle.metadata.get("schema_fingerprint"),
        "configuration": scientific_configuration,
        "dependencies": dict(dependencies),
        "seed": cfg.seed,
        "administrative_data_through": admin_dates,
    }
    fingerprint = digest(fingerprint_payload)
    run_dir = (
        Path(cfg.output_dir).expanduser().resolve()
        if cfg.output_dir
        else timestamped_default_output_dir()
    )
    context = RunContext(cfg, run_dir, fingerprint, fingerprint_payload)
    context.initialize()
    return context


# ======================================================================================
# 5. Deterministic raw-event fixture, cohorts, targets, landmarks, and leakage audit
# ======================================================================================


def synthetic_data_bundle(cfg: RunConfig) -> DataBundle:
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.smoke_patients)
    patient_ids = [f"SYN{index:06d}" for index in range(n)]
    centers = [f"CENTER_{index + 1:02d}" for index in range(10)]
    patient_rows: list[dict[str, Any]] = []
    procedure_rows: list[dict[str, Any]] = []
    medication_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    encounter_rows: list[dict[str, Any]] = []
    diagnosis_rows: list[dict[str, Any]] = []
    data_through = pd.Timestamp("2026-06-30")
    surgery_count = n // 2

    for index, patient_id in enumerate(patient_ids):
        center = centers[index % len(centers)]
        is_surgery_source = index < surgery_count
        base_year = 2017 + (index % 5) if is_surgery_source else 2018 + (index % 5)
        index_date = pd.Timestamp(year=base_year, month=1 + (index * 7) % 12, day=1 + (index * 11) % 25)
        age = int(24 + (index * 13) % 54)
        diabetes = int(index % 3 != 0)
        sex = "Female" if index % 2 else "Male"
        race = ("White", "Black", "Asian", "Other")[index % 4]
        ingredient = ("semaglutide", "liraglutide", "dulaglutide", "tirzepatide")[index % 4]
        baseline_bmi = float(36.0 + (index % 17) * 0.65 + rng.normal(0, 0.55))
        if not is_surgery_source:
            baseline_bmi = float(30.5 + (index % 23) * 0.58 + rng.normal(0, 0.55))
        baseline_hba1c = float(6.1 + diabetes * (0.7 + (index % 8) * 0.18) + rng.normal(0, 0.18))
        observation_start = index_date - pd.Timedelta(days=730 + index % 500)
        observation_end = data_through - pd.Timedelta(days=index % 45)
        postop_flag = 0

        if is_surgery_source:
            code = "43775" if index % 2 == 0 else ("43644", "43645", "43846")[index % 3]
            procedure_rows.append(
                {
                    "patient_id": patient_id,
                    "procedure_date": index_date,
                    "procedure_code": code,
                    "procedure_concept_id": code,
                    "source_table": "synthetic.procedures",
                }
            )
            # A small known previously treated group is present to exercise the exclusion.
            if index % 29 == 0:
                pre_start = index_date - pd.Timedelta(days=250)
                for fill in range(7):
                    medication_rows.append(
                        {
                            "patient_id": patient_id,
                            "ingredient": ingredient,
                            "medication_concept": ingredient,
                            "route": "subcutaneous",
                            "formulation": "injection",
                            "fill_date": pre_start + pd.Timedelta(days=28 * fill),
                            "days_supply": 28,
                            "source_type": "fill",
                            "source_id": f"PRE6-{patient_id}-{fill}",
                            "dose": 1.0,
                            "dose_unit": "mg",
                            "source_table": "synthetic.dispenses",
                        }
                    )
            elif index % 17 == 0:
                # Subthreshold exposure remains active at operation and forces day-zero censoring.
                pre_start = index_date - pd.Timedelta(days=70)
                for fill in range(3):
                    medication_rows.append(
                        {
                            "patient_id": patient_id,
                            "ingredient": ingredient,
                            "medication_concept": ingredient,
                            "route": "subcutaneous",
                            "formulation": "injection",
                            "fill_date": pre_start + pd.Timedelta(days=28 * fill),
                            "days_supply": 35,
                            "source_type": "fill",
                            "source_id": f"PREACTIVE-{patient_id}-{fill}",
                            "dose": 1.0,
                            "dose_unit": "mg",
                            "source_table": "synthetic.dispenses",
                        }
                    )
            elif index % 13 == 0:
                pre_start = index_date - pd.Timedelta(days=170)
                for fill in range(3):
                    medication_rows.append(
                        {
                            "patient_id": patient_id,
                            "ingredient": ingredient,
                            "medication_concept": ingredient,
                            "route": "subcutaneous",
                            "formulation": "injection",
                            "fill_date": pre_start + pd.Timedelta(days=28 * fill),
                            "days_supply": 28,
                            "source_type": "fill",
                            "source_id": f"PRESHORT-{patient_id}-{fill}",
                            "dose": 1.0,
                            "dose_unit": "mg",
                            "source_table": "synthetic.dispenses",
                        }
                    )
            if index % 7 == 0:
                postop_flag = 1
                postoperative_start = index_date + pd.Timedelta(days=260 + index % 900)
                for fill in range(4):
                    medication_rows.append(
                        {
                            "patient_id": patient_id,
                            "ingredient": ingredient,
                            "medication_concept": ingredient,
                            "route": "subcutaneous",
                            "formulation": "injection",
                            "fill_date": postoperative_start + pd.Timedelta(days=28 * fill),
                            "days_supply": 28,
                            "source_type": "fill",
                            "source_id": f"POST-{patient_id}-{fill}",
                            "dose": 1.0,
                            "dose_unit": "mg",
                            "source_table": "synthetic.dispenses",
                        }
                    )
        else:
            # Six-month continuer source cohort. Some records intentionally fail the rule.
            fill_count = (72 if index % 4 else 28) if index % 11 else 5
            gap_extra = 32 if index % 19 == 0 else 0
            for fill in range(fill_count):
                offset = 28 * fill + (gap_extra if fill >= 3 else 0)
                current_ingredient = ingredient
                if index % 23 == 0 and fill >= 4:
                    current_ingredient = "semaglutide" if ingredient != "semaglutide" else "dulaglutide"
                medication_rows.append(
                    {
                        "patient_id": patient_id,
                        "ingredient": current_ingredient,
                        "medication_concept": current_ingredient,
                        "route": "oral" if current_ingredient == "semaglutide" and index % 9 == 0 else "subcutaneous",
                        "formulation": "tablet" if current_ingredient == "semaglutide" and index % 9 == 0 else "injection",
                        "fill_date": index_date + pd.Timedelta(days=offset),
                        "days_supply": 28,
                        "source_type": "fill",
                        "source_id": f"NEW-{patient_id}-{fill}",
                        "dose": float(0.5 + 0.25 * min(fill, 4)),
                        "dose_unit": "mg",
                        "source_table": "synthetic.dispenses",
                    }
                )
            if index % 31 == 0:
                surgery_date = index_date + pd.Timedelta(days=370)
                procedure_rows.append(
                    {
                        "patient_id": patient_id,
                        "procedure_date": surgery_date,
                        "procedure_code": "43775",
                        "procedure_concept_id": "43775",
                        "source_table": "synthetic.procedures",
                    }
                )

        patient_rows.append(
            {
                "patient_id": patient_id,
                "center_id": center,
                "age": age,
                "birth_year": index_date.year - age,
                "sex": sex,
                "race": race,
                "ethnicity": "Hispanic" if index % 6 == 0 else "Not Hispanic",
                "coverage": ("Commercial", "Medicare", "Medicaid")[index % 3],
                "observation_start_date": observation_start,
                "observation_end_date": observation_end,
                "administrative_end_date": data_through,
                "postop_incretin_flag": postop_flag,
                "prior_mbs_flag": 0,
                "dialysis_transplant_flag": int(index % 97 == 0),
                "diabetes_flag": diabetes,
                "smoking": "Current" if index % 12 == 0 else "Never/former",
                "hypertension": int(index % 3 != 1),
                "dyslipidemia": int(index % 4 != 1),
                "osa": int(index % 5 != 2),
                "insulin": int(diabetes and index % 5 == 0),
                "biguanide": int(diabetes and index % 3 != 0),
                "sglt2": int(diabetes and index % 7 == 0),
                "svi": float((index % 100) / 100),
                "ruca": str(1 + index % 10),
                "state": ("PA", "NJ", "NY", "DE")[index % 4],
                "source_table": "synthetic.patients",
            }
        )
        if diabetes:
            diagnosis_rows.append(
                {
                    "patient_id": patient_id,
                    "diagnosis_date": index_date - pd.Timedelta(days=120 + index % 200),
                    "diagnosis_code": "E11.9",
                    "source_table": "synthetic.diagnoses",
                }
            )
        for months_before in (18, 12, 6, 1):
            encounter_rows.append(
                {
                    "patient_id": patient_id,
                    "encounter_date": index_date - pd.Timedelta(days=month_to_nominal_day(months_before)),
                    "source_table": "synthetic.encounters",
                }
            )
        # Baseline BMI has an observed value; periodic derived-only BMI records exercise derivation.
        measurement_rows.append(
            {
                "patient_id": patient_id,
                "measurement_date": index_date - pd.Timedelta(days=index % 25),
                "measurement_type": "bmi",
                "raw_value": baseline_bmi,
                "unit": "kg/m2",
                "source_concept": "LOINC BMI",
                "source_table": "synthetic.vitals",
            }
        )
        if diabetes or index % 5 == 0:
            hba_unit = "mmol/mol" if index % 10 == 0 else "%"
            hba_value = (baseline_hba1c - 2.15) * 10.929 if hba_unit == "mmol/mol" else baseline_hba1c
            measurement_rows.append(
                {
                    "patient_id": patient_id,
                    "measurement_date": index_date - pd.Timedelta(days=index % 70),
                    "measurement_type": "hba1c",
                    "raw_value": hba_value,
                    "unit": hba_unit,
                    "source_concept": "LOINC 4548-4",
                    "source_table": "synthetic.labs",
                }
            )
        horizons = sorted(set(TARGET_MONTHS["bmi"] + TARGET_MONTHS["hba1c"]))
        for month in horizons:
            nominal = month_to_nominal_day(month)
            jitter = int((index * 17 + month * 5) % 45 - 22)
            measurement_date = index_date + pd.Timedelta(days=nominal + jitter)
            if measurement_date > observation_end:
                continue
            missing_probability = min(0.12 + month / 120.0, 0.55)
            if rng.random() > missing_probability and month in TARGET_MONTHS["bmi"]:
                early_loss = (10.5 if is_surgery_source else 6.2) * (1 - math.exp(-month / 6.5))
                regain = max(month - 18, 0) * (0.065 if is_surgery_source else 0.045)
                procedure_bonus = 2.0 if is_surgery_source and index % 2 else 0.0
                value = baseline_bmi - early_loss - procedure_bonus + regain + rng.normal(0, 1.25 + month / 80)
                if month % 24 == 0 and index % 8 == 0:
                    height = 1.55 + (index % 20) * 0.012
                    weight = value * height * height
                    measurement_rows.extend(
                        [
                            {
                                "patient_id": patient_id,
                                "measurement_date": measurement_date,
                                "measurement_type": "weight",
                                "raw_value": weight / 0.45359237,
                                "unit": "lb",
                                "source_concept": "LOINC body weight",
                                "source_table": "synthetic.vitals",
                            },
                            {
                                "patient_id": patient_id,
                                "measurement_date": measurement_date,
                                "measurement_type": "height",
                                "raw_value": height * 100,
                                "unit": "cm",
                                "source_concept": "LOINC body height",
                                "source_table": "synthetic.vitals",
                            },
                        ]
                    )
                else:
                    measurement_rows.append(
                        {
                            "patient_id": patient_id,
                            "measurement_date": measurement_date,
                            "measurement_type": "bmi",
                            "raw_value": value,
                            "unit": "kg/m2",
                            "source_concept": "LOINC BMI",
                            "source_table": "synthetic.vitals",
                        }
                    )
            if diabetes and rng.random() > missing_probability and month in TARGET_MONTHS["hba1c"]:
                response = (1.15 if is_surgery_source else 0.85) * (1 - math.exp(-month / 9))
                relapse = max(month - 24, 0) * 0.012
                value = baseline_hba1c - response + relapse + rng.normal(0, 0.28)
                measurement_rows.append(
                    {
                        "patient_id": patient_id,
                        "measurement_date": measurement_date,
                        "measurement_type": "hba1c",
                        "raw_value": value,
                        "unit": "%",
                        "source_concept": "LOINC 4548-4",
                        "source_table": "synthetic.labs",
                    }
                )

    patients = pd.DataFrame(patient_rows)
    procedures = pd.DataFrame(procedure_rows)
    medications = pd.DataFrame(medication_rows)
    measurements = pd.DataFrame(measurement_rows)
    encounters = pd.DataFrame(encounter_rows)
    diagnoses = pd.DataFrame(diagnosis_rows)
    pseudo_sql = {
        "patients": "SELECT explicit synthetic patient fields FROM synthetic.patients",
        "procedures": "SELECT explicit synthetic procedure fields FROM synthetic.procedures",
        "medications": "SELECT explicit synthetic dispense fields FROM synthetic.dispenses",
        "measurements": "SELECT explicit synthetic measurement fields FROM synthetic.measurements",
    }
    bundle = DataBundle(
        patients=patients,
        procedures=procedures,
        medications=medications,
        measurements=measurements,
        encounters=encounters,
        diagnoses=diagnoses,
        metadata={
            "source_mode": "deterministic_synthetic_raw_events",
            "sql_contract_version": SQL_CONTRACT_VERSION,
            "sql": pseudo_sql,
            "query_fingerprint": digest({key: normalize_sql(value) for key, value in pseudo_sql.items()}),
            "schema_fingerprint": digest(
                {
                    "patients": frame_schema(patients),
                    "procedures": frame_schema(procedures),
                    "medications": frame_schema(medications),
                    "measurements": frame_schema(measurements),
                }
            ),
        },
    )
    bundle.metadata["preflight"] = validate_data_bundle(bundle)
    return bundle


def numeric_or_default(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def age_at_index(patient: Mapping[str, Any], index_date: Any) -> float:
    age = patient.get("age")
    if age is not None and pd.notna(age):
        return float(age)
    birth_year = patient.get("birth_year")
    if birth_year is not None and pd.notna(birth_year):
        return float(pd.Timestamp(index_date).year - int(birth_year))
    return float("nan")


def patient_for_source(patient: Mapping[str, Any], source_prefix: str) -> dict[str, Any]:
    """Overlay source-specific wide fields without changing the stable person identifier."""
    result = dict(patient)
    prefix = source_prefix + "__"
    for key, value in patient.items():
        if str(key).startswith(prefix):
            result[str(key)[len(prefix):]] = value
    return result


def procedure_category(value: Any) -> str | None:
    match = re.search(r"(\d{5})", str(value))
    return PROCEDURE_CODES.get(match.group(1)) if match else None


def first_later_bariatric_day(procedures: Any, patient_id: str, index_date: Any) -> int | None:
    subset = procedures.loc[procedures["patient_id"].astype(str).eq(str(patient_id))].copy()
    subset["code"] = subset["procedure_code"].astype(str).str.extract(r"(\d{5})", expand=False)
    subset = subset.loc[subset["code"].isin(BARIATRIC_HISTORY_CODES)]
    subset["day"] = (pd.to_datetime(subset["procedure_date"]) - pd.Timestamp(index_date)).dt.days
    later = subset.loc[subset["day"].ge(0), "day"]
    return int(later.min()) if not later.empty else None


def aggregate_funnel(rows: Sequence[dict[str, Any]]) -> Any:
    return pd.DataFrame(rows, columns=["cohort", "stage", "n_patients", "status"])


def construct_cohorts(bundle: DataBundle) -> dict[str, Any]:
    patients = bundle.patients.copy()
    patients["patient_id"] = patients["patient_id"].astype(str)
    patient_lookup = patients.set_index("patient_id", drop=False).to_dict(orient="index")
    procedures = bundle.procedures.copy()
    procedures["patient_id"] = procedures["patient_id"].astype(str)
    procedures["procedure_date"] = pd.to_datetime(procedures["procedure_date"], errors="coerce").dt.normalize()
    normalized_measurements, measurement_quality = normalize_measurements(bundle.measurements)
    direct_wide = bundle.metadata.get("source_mode") == "cosmos_direct_wide_cohorts"
    if direct_wide and "source_cohort" in bundle.medications:
        surgery_medications = bundle.medications.loc[bundle.medications["source_cohort"].eq("surgery")]
        incretin_medications = bundle.medications.loc[bundle.medications["source_cohort"].eq("incretin")]
        surgery_records, surgery_audit = medication_frame_to_coverage(surgery_medications)
        incretin_records, incretin_audit = medication_frame_to_coverage(incretin_medications)
        surgery_audit["source_cohort"] = "surgery"
        incretin_audit["source_cohort"] = "incretin"
        medication_audit = pd.concat([surgery_audit, incretin_audit], ignore_index=True)
    else:
        surgery_records, medication_audit = medication_frame_to_coverage(bundle.medications)
        incretin_records = surgery_records
    records_by_patient: dict[str, list[CoverageRecord]] = defaultdict(list)
    for record in surgery_records:
        records_by_patient[record.patient_id].append(record)
    epoch = pd.Timestamp("1970-01-01")
    cohort_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []

    surgical_candidates = procedures.copy()
    surgical_candidates["procedure_type"] = surgical_candidates["procedure_code"].map(procedure_category)
    surgical_candidates = surgical_candidates.loc[surgical_candidates["procedure_type"].notna()].sort_values(
        ["patient_id", "procedure_date"]
    )
    funnel_rows.append({"cohort": "surgery", "stage": "source patients with qualifying CPT", "n_patients": surgical_candidates["patient_id"].nunique(), "status": "included"})
    for patient_id, group in surgical_candidates.groupby("patient_id", sort=True):
        candidate = group.iloc[0]
        index_date = pd.Timestamp(candidate["procedure_date"])
        patient = patient_lookup.get(str(patient_id))
        if patient is not None and direct_wide:
            patient = patient_for_source(patient, "mbs")
        exclusion = ""
        if patient is None:
            exclusion = "missing_patient_record"
        elif age_at_index(patient, index_date) < 18:
            exclusion = "age_under_18"
        elif pd.Timestamp(patient["observation_start_date"]) > index_date - pd.Timedelta(days=365):
            exclusion = "less_than_365_days_medication_history"
        elif numeric_or_default(patient, "prior_mbs_flag") == 1:
            exclusion = "prior_bariatric_surgery"
        elif numeric_or_default(patient, "dialysis_transplant_flag") == 1:
            exclusion = "dialysis_or_transplant"
        patient_measurements = normalized_measurements.loc[normalized_measurements["patient_id"].eq(str(patient_id))]
        if direct_wide and "source_cohort" in patient_measurements:
            patient_measurements = patient_measurements.loc[patient_measurements["source_cohort"].eq("surgery")]
        baseline_bmi = select_baseline_measurement(patient_measurements, "bmi", index_date)
        if not exclusion and baseline_bmi is None:
            exclusion = "missing_baseline_bmi"
        elif not exclusion and not 35 <= float(baseline_bmi["value"]) <= 75:
            exclusion = "baseline_bmi_outside_35_75"
        index_ordinal = int((index_date - epoch).days)
        relative_records = [
            CoverageRecord(
                patient_id=item.patient_id,
                start_day=item.start_day - index_ordinal,
                end_day=item.end_day - index_ordinal,
                ingredient=item.ingredient,
                therapy_class=item.therapy_class,
                route=item.route,
                formulation=item.formulation,
                source_type=item.source_type,
                source_table=item.source_table,
                source_id=item.source_id,
                dose=item.dose,
                dose_unit=item.dose_unit,
            )
            for item in records_by_patient.get(str(patient_id), [])
        ]
        unresolved_prior_flag = bool(
            direct_wide
            and numeric_or_default(patient or {}, "prior_incretin_flag") == 1
            and not any(item.start_day < 0 for item in relative_records)
        )
        history = classify_surgical_incretin_history(
            relative_records,
            postoperative_flag=bool(numeric_or_default(patient or {}, "postop_incretin_flag")),
            timing_unknown=unresolved_prior_flag,
        )
        if not exclusion and history["classification"] == "previously_treated":
            exclusion = "qualifying_six_month_prior_episode"
        if not exclusion and history["unresolved_postoperative_start"]:
            exclusion = "postop_flag_without_start"
        if not exclusion and history["classification"] == "unknown":
            exclusion = "unknown_exposure_timing"
        exposure_rows.append(
            {
                "patient_id": str(patient_id),
                "cohort": "surgery",
                "classification": history["classification"],
                "treatment_censor_day": history["treatment_censor_day"],
                "active_at_index": history["active_at_surgery"],
                "episode_count": history["episode_count"],
                "rejected_record_count": history["rejected_record_count"],
                "excluded": bool(exclusion),
                "exclusion_reason": exclusion,
            }
        )
        if exclusion:
            continue
        baseline_hba1c = select_baseline_measurement(patient_measurements, "hba1c", index_date)
        row = dict(patient)
        row.update(
            {
                "patient_id": str(patient_id),
                "cohort": "surgery",
                "index_date": index_date,
                "treatment": str(candidate["procedure_type"]),
                "therapy_class": "bariatric_procedure",
                "procedure": str(candidate["procedure_type"]),
                "index_ingredient": "not_applicable",
                "index_route": "not_applicable",
                "age_at_index": age_at_index(patient, index_date),
                "baseline_bmi": float(baseline_bmi["value"]),
                "baseline_bmi_day": int(baseline_bmi["day"]),
                "baseline_hba1c": float(baseline_hba1c["value"]) if baseline_hba1c else np.nan,
                "baseline_hba1c_day": int(baseline_hba1c["day"]) if baseline_hba1c else np.nan,
                "diabetes_eligible": bool(numeric_or_default(patient, "diabetes_flag") == 1 and baseline_hba1c is not None),
                "treatment_censor_day": history["treatment_censor_day"],
                "surgery_censor_day": None,
                "strict_never_exposed": history["strict_never_exposed"],
                "prior_exposure_stratum": history["classification"],
            }
        )
        cohort_rows.append(row)

    surgical_exclusions = pd.Series([item["exclusion_reason"] for item in exposure_rows if item["exclusion_reason"]]).value_counts()
    for reason, count in surgical_exclusions.items():
        funnel_rows.append({"cohort": "surgery", "stage": str(reason), "n_patients": int(count), "status": "excluded"})
    funnel_rows.append({"cohort": "surgery", "stage": "primary eligible", "n_patients": sum(item["cohort"] == "surgery" for item in cohort_rows), "status": "included"})

    # Medication cohort is built from the same coverage algorithm on absolute day numbers.
    all_episodes, rejected_records = reconstruct_coverage_episodes(incretin_records)
    episodes_by_patient: dict[str, list[CoverageEpisode]] = defaultdict(list)
    for episode in all_episodes:
        episodes_by_patient[episode.patient_id].append(episode)
    funnel_rows.append({"cohort": "incretin", "stage": "patients with accepted exposure", "n_patients": len(episodes_by_patient), "status": "included"})
    medication_exclusions: dict[str, int] = defaultdict(int)
    for patient_id in sorted(episodes_by_patient):
        patient = patient_lookup.get(str(patient_id))
        if patient is not None and direct_wide:
            patient = patient_for_source(patient, "glp1")
        exclusion = ""
        selected_episode = select_primary_incretin_episode(episodes_by_patient[patient_id])
        if patient is None:
            exclusion = "missing_patient_record"
        elif selected_episode is None:
            exclusion = "no_new_user_six_month_continuation_episode"
        if selected_episode is None:
            medication_exclusions[exclusion] += 1
            continue
        index_date = epoch + pd.Timedelta(days=selected_episode.start_day)
        if age_at_index(patient, index_date) < 18:
            exclusion = "age_under_18"
        elif pd.Timestamp(patient["observation_start_date"]) > index_date - pd.Timedelta(days=365):
            exclusion = "less_than_365_days_washout_observation"
        patient_procedures = procedures.loc[procedures["patient_id"].eq(str(patient_id))].copy()
        patient_procedures["code"] = patient_procedures["procedure_code"].astype(str).str.extract(r"(\d{5})", expand=False)
        patient_procedures = patient_procedures.loc[patient_procedures["code"].isin(BARIATRIC_HISTORY_CODES)]
        prior_surgery = patient_procedures["procedure_date"].lt(index_date).any()
        first_surgery_day = first_later_bariatric_day(procedures, str(patient_id), index_date)
        if not exclusion and prior_surgery:
            exclusion = "prior_bariatric_surgery"
        if (
            not exclusion
            and numeric_or_default(patient or {}, "mbs_during_incretin_flag") == 1
            and first_surgery_day is None
        ):
            exclusion = "bariatric_surgery_timing_unresolved"
        if not exclusion and first_surgery_day is not None and first_surgery_day <= 182:
            exclusion = "bariatric_surgery_during_first_183_days"
        patient_measurements = normalized_measurements.loc[normalized_measurements["patient_id"].eq(str(patient_id))]
        if direct_wide and "source_cohort" in patient_measurements:
            patient_measurements = patient_measurements.loc[patient_measurements["source_cohort"].eq("incretin")]
        baseline_bmi = select_baseline_measurement(patient_measurements, "bmi", index_date)
        if not exclusion and baseline_bmi is None:
            exclusion = "missing_baseline_bmi"
        elif not exclusion and float(baseline_bmi["value"]) < 30:
            exclusion = "baseline_bmi_below_30"
        if exclusion:
            medication_exclusions[exclusion] += 1
            continue
        baseline_hba1c = select_baseline_measurement(patient_measurements, "hba1c", index_date)
        first_record = sorted(selected_episode.records, key=lambda item: (item.start_day, item.end_day))[0]
        treatment_censor = selected_episode.censor_day - selected_episode.start_day
        row = dict(patient)
        row.update(
            {
                "patient_id": str(patient_id),
                "cohort": "incretin",
                "index_date": index_date,
                "treatment": first_record.ingredient,
                "therapy_class": first_record.therapy_class,
                "procedure": "not_applicable",
                "index_ingredient": first_record.ingredient,
                "index_route": first_record.route,
                "age_at_index": age_at_index(patient, index_date),
                "baseline_bmi": float(baseline_bmi["value"]),
                "baseline_bmi_day": int(baseline_bmi["day"]),
                "baseline_hba1c": float(baseline_hba1c["value"]) if baseline_hba1c else np.nan,
                "baseline_hba1c_day": int(baseline_hba1c["day"]) if baseline_hba1c else np.nan,
                "diabetes_eligible": bool(numeric_or_default(patient, "diabetes_flag") == 1 and baseline_hba1c is not None),
                "treatment_censor_day": int(treatment_censor),
                "surgery_censor_day": first_surgery_day,
                "strict_never_exposed": True,
                "prior_exposure_stratum": "365_day_new_user",
                "pdc_183": selected_episode.pdc_183,
                "maximum_gap_days": selected_episode.maximum_gap_days,
                "switch_count": len(selected_episode.switch_days),
            }
        )
        cohort_rows.append(row)
        exposure_rows.append(
            {
                "patient_id": str(patient_id),
                "cohort": "incretin",
                "classification": "six_month_continuer",
                "treatment_censor_day": int(treatment_censor),
                "active_at_index": True,
                "episode_count": len(episodes_by_patient[patient_id]),
                "rejected_record_count": 0,
                "excluded": False,
                "exclusion_reason": "",
                "pdc_183": selected_episode.pdc_183,
                "maximum_gap_days": selected_episode.maximum_gap_days,
                "switch_count": len(selected_episode.switch_days),
                "source_type": first_record.source_type,
            }
        )
    for reason, count in sorted(medication_exclusions.items()):
        funnel_rows.append({"cohort": "incretin", "stage": reason, "n_patients": int(count), "status": "excluded"})
    funnel_rows.append({"cohort": "incretin", "stage": "primary six-month continuers", "n_patients": sum(item["cohort"] == "incretin" for item in cohort_rows), "status": "included"})
    cohorts = pd.DataFrame(cohort_rows)
    if cohorts.empty:
        raise PreflightError("Cohort construction produced no eligible patients", ["All patients failed prespecified eligibility rules"])
    cohorts["index_date"] = pd.to_datetime(cohorts["index_date"]).dt.normalize()
    cohorts["effective_censor_day"] = cohorts[["treatment_censor_day", "surgery_censor_day"]].apply(
        lambda row: min([int(value) for value in row if pd.notna(value)], default=np.inf), axis=1
    )
    cohorts["effective_censor_day"] = cohorts["effective_censor_day"].replace(np.inf, np.nan)
    return {
        "cohorts": cohorts,
        "measurements": normalized_measurements,
        "measurement_quality": measurement_quality,
        "medication_audit": medication_audit,
        "funnel": aggregate_funnel(funnel_rows),
        "exposure": pd.DataFrame(exposure_rows),
        "rejected_coverage_records": pd.DataFrame([asdict(item) for item in rejected_records]),
    }


def stable_hash_fraction(value: str, seed: int) -> float:
    payload = f"{seed}|{value}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64 - 1)


def assign_global_splits(cohorts: Any, seed: int = SEED) -> tuple[Any, dict[str, Any]]:
    canonical = cohorts.sort_values(["patient_id", "index_date"]).drop_duplicates("patient_id", keep="first").copy()
    canonical_centers = canonical["center_id"].astype(str)
    unavailable_center = canonical_centers.str.strip().str.lower().isin(
        {"", "nan", "none", "unknown", CENTER_UNAVAILABLE.lower()}
    )
    usable_centers = sorted(canonical_centers.loc[~unavailable_center].unique())
    center_validation_available = not bool(unavailable_center.any()) and len(usable_centers) >= 3
    center_values = usable_centers if center_validation_available else []
    center_order = sorted(center_values, key=lambda value: stable_hash_fraction("center|" + value, seed), reverse=True)
    holdout_count = max(1, int(round(len(center_order) * HELDOUT_CENTER_FRACTION))) if len(center_order) >= 3 else 0
    heldout_centers = set(center_order[:holdout_count])
    development = canonical.loc[~canonical["center_id"].astype(str).isin(heldout_centers)].copy()
    if development.empty:
        raise LeakageError("No development centers remain after geographic holdout")
    temporal_cutoff = pd.Timestamp(development["index_date"].quantile(0.80)).normalize()
    labels: dict[str, str] = {}
    for row in canonical.itertuples(index=False):
        patient_id = str(row.patient_id)
        if str(row.center_id) in heldout_centers:
            label = "geographic_test"
        elif pd.Timestamp(row.index_date) >= temporal_cutoff:
            label = "temporal_test"
        else:
            fraction = stable_hash_fraction("patient|" + patient_id, seed)
            label = "train" if fraction < 0.65 else ("validation" if fraction < 0.82 else "calibration")
        labels[patient_id] = label
    manifest = cohorts.copy()
    manifest["split"] = manifest["patient_id"].astype(str).map(labels)
    if manifest["split"].isna().any():
        raise LeakageError("At least one cohort row lacks a global patient split")
    patient_split_counts = manifest.groupby("patient_id")["split"].nunique()
    if int(patient_split_counts.max()) != 1:
        raise LeakageError("A patient was assigned to more than one split")
    development_center_overlap = set(
        manifest.loc[manifest["split"].isin(["train", "validation", "calibration", "temporal_test"]), "center_id"].astype(str)
    ).intersection(heldout_centers)
    if development_center_overlap:
        raise LeakageError("A held-out center appears in development or temporal testing")
    metadata = {
        "heldout_centers": sorted(heldout_centers),
        "heldout_center_labels": [f"Held-out center {index + 1}" for index in range(len(heldout_centers))],
        "temporal_cutoff": temporal_cutoff.date().isoformat(),
        "split_counts": manifest.drop_duplicates("patient_id")["split"].value_counts().to_dict(),
        "patient_overlap_passed": True,
        "center_validation_available": center_validation_available,
        "center_holdout_passed": True if center_validation_available else None,
        "center_validation_status": (
            "geographic holdout completed"
            if center_validation_available
            else "not estimable because complete identifiers for at least three centers are unavailable"
        ),
    }
    return manifest, metadata


def robust_slope(days: Any, values: Any) -> float:
    x = np.asarray(days, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(x) < 2 or np.ptp(x) <= 0:
        return 0.0
    slopes = []
    for left in range(len(x)):
        for right in range(left + 1, len(x)):
            if x[right] != x[left]:
                slopes.append((y[right] - y[left]) / ((x[right] - x[left]) / DAYS_PER_YEAR))
    return float(np.median(slopes)) if slopes else 0.0


def build_prediction_rows(cohorts_with_splits: Any, measurements: Any) -> Any:
    rows: list[dict[str, Any]] = []
    measurement_groups = {
        patient_id: group.copy()
        for patient_id, group in measurements.groupby("patient_id", sort=False)
    }
    for patient in cohorts_with_splits.itertuples(index=False):
        payload = patient._asdict()
        patient_measurements = measurement_groups.get(str(patient.patient_id), pd.DataFrame(columns=measurements.columns))
        index_date = pd.Timestamp(patient.index_date)
        patient_measurements = patient_measurements.copy()
        if not patient_measurements.empty:
            patient_measurements["day"] = (patient_measurements["measurement_date"] - index_date).dt.days
        censor_values = [
            int(value) for value in (payload.get("treatment_censor_day"), payload.get("surgery_censor_day"))
            if value is not None and pd.notna(value)
        ]
        censor_day = min(censor_values) if censor_values else None
        for outcome in ("bmi", "hba1c"):
            if outcome == "hba1c" and not bool(payload.get("diabetes_eligible")):
                continue
            baseline_value = payload.get(f"baseline_{outcome}")
            baseline_day = payload.get(f"baseline_{outcome}_day")
            if baseline_value is None or pd.isna(baseline_value):
                continue
            outcome_history = patient_measurements.loc[patient_measurements["outcome"].eq(outcome)].copy()
            for origin_month in LANDMARK_MONTHS:
                origin_day = month_to_nominal_day(origin_month)
                if censor_day is not None and origin_day >= censor_day:
                    continue
                history = outcome_history.loc[outcome_history["day"].le(origin_day)].copy()
                if censor_day is not None:
                    history = history.loc[history["day"].lt(censor_day)]
                if history.empty:
                    continue
                last = history.sort_values(["day", "measurement_date"]).iloc[-1]
                history_days = history["day"].to_numpy(float)
                history_values = history["value"].to_numpy(float)
                last_value = float(last["value"])
                last_day = int(last["day"])
                for target_month in TARGET_MONTHS[outcome]:
                    if target_month <= origin_month:
                        continue
                    selected = select_target_measurement(
                        patient_measurements,
                        outcome,
                        index_date,
                        target_month,
                        censor_day=censor_day,
                    )
                    status = target_support_status(
                        index_date,
                        payload["administrative_end_date"],
                        payload["observation_end_date"],
                        target_month,
                        censor_day,
                        selected is not None,
                    )
                    target_value = float(selected["value"]) if selected is not None else np.nan
                    target_day = int(selected["day"]) if selected is not None else TARGET_WINDOWS[target_month].nominal_day
                    uncensored_through_target = censor_day is None or target_day < censor_day
                    row = {
                        "patient_id": str(patient.patient_id),
                        "cohort": str(patient.cohort),
                        "outcome": outcome,
                        "origin_month": origin_month,
                        "origin_day": origin_day,
                        "target_month": target_month,
                        "target_day": target_day,
                        "time_from_origin_months": target_month - origin_month,
                        "target_value": target_value,
                        "target_observed": selected is not None,
                        "support_status": status,
                        "administratively_mature": status != "administratively_immature",
                        "uncensored_through_target": uncensored_through_target,
                        "feature_max_day": int(history["day"].max()),
                        "baseline_value": float(baseline_value),
                        "baseline_measurement_day": int(baseline_day),
                        "last_value": last_value,
                        "last_measurement_day": last_day,
                        "measurement_recency_days": origin_day - last_day,
                        "change_from_baseline_at_origin": last_value - float(baseline_value),
                        "percent_change_from_baseline_at_origin": 100.0 * (last_value - float(baseline_value)) / float(baseline_value),
                        "robust_slope_per_year": robust_slope(history_days, history_values),
                        "within_patient_variability": float(np.std(history_values, ddof=0)),
                        "history_measurement_count": int(len(history)),
                        "prediction_reference_value": float(baseline_value) if origin_month == 0 else last_value,
                        "target_change": target_value - (float(baseline_value) if origin_month == 0 else last_value) if selected is not None else np.nan,
                        "window_valid_count": int(selected["window_valid_count"]) if selected is not None else 0,
                        "split": str(patient.split),
                        "center_id": str(patient.center_id),
                        "index_date": index_date,
                        "treatment": str(patient.treatment),
                        "procedure": str(patient.procedure),
                        "index_ingredient": str(patient.index_ingredient),
                        "index_route": str(patient.index_route),
                        "therapy_class": str(patient.therapy_class),
                        "age_at_index": float(patient.age_at_index),
                        "sex": str(patient.sex),
                        "race": str(patient.race),
                        "ethnicity": str(patient.ethnicity),
                        "coverage": str(patient.coverage),
                        "diabetes_flag": numeric_or_default(payload, "diabetes_flag"),
                        "hypertension": numeric_or_default(payload, "hypertension"),
                        "dyslipidemia": numeric_or_default(payload, "dyslipidemia"),
                        "osa": numeric_or_default(payload, "osa"),
                        "insulin": numeric_or_default(payload, "insulin"),
                        "biguanide": numeric_or_default(payload, "biguanide"),
                        "sglt2": numeric_or_default(payload, "sglt2"),
                        "svi": numeric_or_default(payload, "svi", np.nan),
                        "index_year": index_date.year,
                        "effective_censor_day": censor_day if censor_day is not None else np.nan,
                    }
                    rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise PreflightError("Prediction-row construction produced no rows", ["No eligible origin and future-target combinations remain"])
    return frame


def leakage_audit(rows: Any, split_metadata: Mapping[str, Any]) -> Any:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"assertion": name, "passed": bool(passed), "detail": detail})

    patient_split = rows.groupby("patient_id")["split"].nunique()
    record("Patient IDs never overlap splits", bool(patient_split.max() == 1), f"max split count per patient = {int(patient_split.max())}")
    heldout = set(split_metadata.get("heldout_centers", []))
    development_centers = set(rows.loc[rows["split"].isin(["train", "validation", "calibration"]), "center_id"])
    if split_metadata.get("center_validation_available", True):
        record(
            "Held-out centers absent from development",
            not bool(heldout.intersection(development_centers)),
            f"overlap count = {len(heldout.intersection(development_centers))}",
        )
    else:
        record(
            "Geographic validation is explicitly unavailable",
            not bool(rows["split"].eq("geographic_test").any()),
            "no usable center identifiers and no geographic-test rows",
        )
    feature_ok = bool(rows["feature_max_day"].le(rows["origin_day"]).all())
    record("Every feature timestamp is at or before origin", feature_ok, f"violations = {int((rows['feature_max_day'] > rows['origin_day']).sum())}")
    target_ok = bool(rows["target_day"].gt(rows["origin_day"]).all())
    record("Every target is strictly after origin", target_ok, f"violations = {int((rows['target_day'] <= rows['origin_day']).sum())}")
    observed = rows.loc[rows["target_observed"] & rows["effective_censor_day"].notna()]
    censor_ok = bool(observed["target_day"].lt(observed["effective_censor_day"]).all())
    record("No observed target is on or after treatment or surgery censoring", censor_ok, f"violations = {int((observed['target_day'] >= observed['effective_censor_day']).sum())}")
    record("Outcome processing is frozen before test scoring", True, "Window and plausibility constants are module-level protocol values")
    record("Calibration patients are separate from final tests", not bool(set(rows.loc[rows['split'].eq('calibration'), 'patient_id']).intersection(set(rows.loc[rows['split'].isin(['temporal_test', 'geographic_test']), 'patient_id']))), "patient-level set intersection checked")
    audit = pd.DataFrame(checks)
    if not bool(audit["passed"].all()):
        failures = audit.loc[~audit["passed"], "assertion"].tolist()
        raise LeakageError("Leakage audit failed: " + "; ".join(failures))
    return audit


# ======================================================================================
# 6. Cross-fitted censoring/observation weights and tabular model candidates
# ======================================================================================


NUMERIC_MODEL_FEATURES = (
    "baseline_value", "last_value", "age_at_index", "origin_month", "target_month",
    "time_from_origin_months", "measurement_recency_days", "change_from_baseline_at_origin",
    "percent_change_from_baseline_at_origin", "robust_slope_per_year",
    "within_patient_variability", "history_measurement_count", "diabetes_flag",
    "hypertension", "dyslipidemia", "osa", "insulin", "biguanide", "sglt2", "svi",
    "index_year",
)
CATEGORICAL_MODEL_FEATURES = (
    "treatment", "procedure", "index_ingredient", "index_route", "therapy_class",
    "sex", "race", "ethnicity", "coverage",
)


@dataclass
class TabularEncoder:
    numeric: list[str]
    categorical: list[str]
    medians: dict[str, float]
    scales: dict[str, float]
    levels: dict[str, list[str]]

    @classmethod
    def fit(
        cls,
        frame: Any,
        numeric: Sequence[str] = NUMERIC_MODEL_FEATURES,
        categorical: Sequence[str] = CATEGORICAL_MODEL_FEATURES,
    ) -> "TabularEncoder":
        medians: dict[str, float] = {}
        scales: dict[str, float] = {}
        levels: dict[str, list[str]] = {}
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(dtype=float)
            medians[column] = float(values.median()) if values.notna().any() else 0.0
            scale = float(values.std(ddof=0)) if values.notna().sum() > 1 else 1.0
            scales[column] = scale if math.isfinite(scale) and scale > 1e-8 else 1.0
        for column in categorical:
            values = frame[column].astype("string").fillna("<MISSING>") if column in frame else pd.Series("<MISSING>", index=frame.index)
            levels[column] = sorted(str(item) for item in values.unique())
        return cls(list(numeric), list(categorical), medians, scales, levels)

    def transform(self, frame: Any) -> Any:
        columns: list[Any] = []
        for column in self.numeric:
            values = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)
            missing = values.isna().to_numpy(float)
            normalized = (values.fillna(self.medians[column]).to_numpy(float) - self.medians[column]) / self.scales[column]
            columns.extend([normalized, missing])
        for column in self.categorical:
            values = frame[column].astype("string").fillna("<MISSING>") if column in frame else pd.Series("<MISSING>", index=frame.index)
            known = set(self.levels[column])
            for level in self.levels[column]:
                columns.append(values.eq(level).to_numpy(float))
            columns.append((~values.isin(known)).to_numpy(float))
        return np.column_stack(columns).astype(np.float32) if columns else np.empty((len(frame), 0), dtype=np.float32)


def quantile_column(quantile: float) -> str:
    return "q" + f"{int(round(100 * quantile)):02d}"


QUANTILE_COLUMNS = tuple(quantile_column(item) for item in QUANTILES)


def pinball_loss(y_true: Any, prediction: Any, quantile: float, weights: Any | None = None) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    error = y - pred
    losses = np.maximum(quantile * error, (quantile - 1.0) * error)
    if weights is None:
        return float(np.mean(losses))
    weight = np.asarray(weights, dtype=float)
    return float(np.sum(losses * weight) / max(np.sum(weight), 1e-12))


def quantile_crps(y_true: Any, quantile_predictions: Any, weights: Any | None = None) -> float:
    predictions = np.asarray(quantile_predictions, dtype=float)
    losses = [pinball_loss(y_true, predictions[:, index], quantile, weights) for index, quantile in enumerate(QUANTILES)]
    return float(2.0 * np.mean(losses))


def fit_probability_model(x: Any, y: Any, sample_weight: Any | None = None) -> tuple[Any | None, float, str]:
    from sklearn.linear_model import LogisticRegression

    values = np.asarray(y, dtype=int)
    marginal = float(np.mean(values)) if len(values) else 0.5
    if len(values) < 20 or len(np.unique(values)) < 2:
        return None, min(max(marginal, 0.01), 0.99), "degenerate"
    model = LogisticRegression(max_iter=500, C=0.5, solver="lbfgs")
    try:
        model.fit(x, values, sample_weight=sample_weight)
    except Exception:
        return None, min(max(marginal, 0.01), 0.99), "nonconvergent"
    return model, min(max(marginal, 0.01), 0.99), "fitted"


def predict_probability(model: Any | None, x: Any, marginal: float) -> Any:
    if model is None:
        return np.full(len(x), marginal, dtype=float)
    return np.clip(model.predict_proba(x)[:, 1], 0.01, 0.99)


def effective_sample_size(weights: Any) -> float:
    values = np.asarray(weights, dtype=float)
    denominator = float(np.sum(values * values))
    return float(np.sum(values) ** 2 / denominator) if denominator > 0 else 0.0


def max_weighted_smd(frame: Any, weight_column: str, group_column: str) -> float:
    maximum = 0.0
    for column in ("baseline_value", "age_at_index", "diabetes_flag", "index_year"):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        groups = frame[group_column].astype(bool)
        if groups.nunique() < 2:
            continue
        moments: list[tuple[float, float]] = []
        for group in (False, True):
            mask = groups.eq(group) & values.notna()
            if not mask.any():
                moments = []
                break
            weight = pd.to_numeric(frame.loc[mask, weight_column], errors="coerce").fillna(1.0).to_numpy(float)
            observed = values.loc[mask].to_numpy(float)
            mean = float(np.average(observed, weights=weight))
            variance = float(np.average((observed - mean) ** 2, weights=weight))
            moments.append((mean, variance))
        if len(moments) == 2:
            pooled = math.sqrt(max((moments[0][1] + moments[1][1]) / 2.0, 1e-12))
            maximum = max(maximum, abs(moments[1][0] - moments[0][0]) / pooled)
    return maximum


def estimate_cross_fitted_weights(rows: Any, seed: int = SEED) -> tuple[Any, Any]:
    weighted = rows.copy().reset_index(drop=True)
    weighted["row_id"] = np.arange(len(weighted), dtype=int)
    weighted["treatment_probability"] = np.nan
    weighted["observation_probability"] = np.nan
    weighted["analysis_weight_untruncated"] = np.nan
    weighted["analysis_weight"] = np.nan
    diagnostics: list[dict[str, Any]] = []
    group_columns = ["cohort", "outcome", "origin_month", "target_month"]
    for keys, group in weighted.groupby(group_columns, sort=True):
        mature = group.loc[group["administratively_mature"]].copy()
        train = mature.loc[mature["split"].eq("train")].copy()
        if train.empty:
            diagnostics.append(
                dict(zip(group_columns, keys, strict=False), status="not_estimable_no_training_rows", n_eligible=len(mature))
            )
            continue
        encoder = TabularEncoder.fit(train, numeric=("baseline_value", "age_at_index", "diabetes_flag", "index_year"), categorical=("treatment", "sex", "race"))
        x_train = encoder.transform(train)
        x_group = encoder.transform(group)
        fold_ids = train["patient_id"].map(lambda value: int(stable_hash_fraction("weight|" + str(value), seed) * 5) % 5).to_numpy()
        treatment_prob_train = np.full(len(train), np.nan)
        observation_prob_train = np.full(len(train), np.nan)
        treatment_statuses: list[str] = []
        observation_statuses: list[str] = []
        for fold in range(5):
            fit_mask = fold_ids != fold
            predict_mask = fold_ids == fold
            if not predict_mask.any():
                continue
            treatment_model, treatment_marginal, treatment_status = fit_probability_model(
                x_train[fit_mask], train.loc[fit_mask, "uncensored_through_target"].astype(int)
            )
            treatment_prob_train[predict_mask] = predict_probability(treatment_model, x_train[predict_mask], treatment_marginal)
            treatment_statuses.append(treatment_status)
            observation_fit = fit_mask & train["uncensored_through_target"].to_numpy(bool)
            observation_model, observation_marginal, observation_status = fit_probability_model(
                x_train[observation_fit], train.loc[observation_fit, "target_observed"].astype(int)
            )
            observation_prob_train[predict_mask] = predict_probability(observation_model, x_train[predict_mask], observation_marginal)
            observation_statuses.append(observation_status)
        full_treatment_model, treatment_marginal, full_treatment_status = fit_probability_model(
            x_train, train["uncensored_through_target"].astype(int)
        )
        observation_subset = train["uncensored_through_target"].to_numpy(bool)
        full_observation_model, observation_marginal, full_observation_status = fit_probability_model(
            x_train[observation_subset], train.loc[observation_subset, "target_observed"].astype(int)
        )
        all_treatment_prob = predict_probability(full_treatment_model, x_group, treatment_marginal)
        all_observation_prob = predict_probability(full_observation_model, x_group, observation_marginal)
        train_positions = {row_id: position for position, row_id in enumerate(train["row_id"].to_numpy())}
        for position, row_id in enumerate(group["row_id"].to_numpy()):
            if row_id in train_positions:
                local = train_positions[row_id]
                if math.isfinite(treatment_prob_train[local]):
                    all_treatment_prob[position] = treatment_prob_train[local]
                if math.isfinite(observation_prob_train[local]):
                    all_observation_prob[position] = observation_prob_train[local]
        raw_weight = (treatment_marginal / np.clip(all_treatment_prob, 0.01, 0.99)) * (
            observation_marginal / np.clip(all_observation_prob, 0.01, 0.99)
        )
        eligible_weight = group["target_observed"].to_numpy(bool) & group["uncensored_through_target"].to_numpy(bool)
        raw_weight = np.where(eligible_weight, raw_weight, np.nan)
        train_raw = raw_weight[group["split"].eq("train").to_numpy() & np.isfinite(raw_weight)]
        if len(train_raw):
            low, high = np.quantile(train_raw, WEIGHT_TRUNCATION)
            truncated = np.where(np.isfinite(raw_weight), np.clip(raw_weight, low, high), np.nan)
        else:
            low, high = np.nan, np.nan
            truncated = raw_weight
        weighted.loc[group.index, "treatment_probability"] = all_treatment_prob
        weighted.loc[group.index, "observation_probability"] = all_observation_prob
        weighted.loc[group.index, "analysis_weight_untruncated"] = raw_weight
        weighted.loc[group.index, "analysis_weight"] = truncated
        observed_weights = truncated[np.isfinite(truncated)]
        ess = effective_sample_size(observed_weights)
        unweighted_count = int(len(observed_weights))
        degenerate = any(status != "fitted" for status in treatment_statuses + observation_statuses + [full_treatment_status, full_observation_status])
        positivity_fail = bool(
            degenerate
            or unweighted_count == 0
            or ess < 50
            or ess < 0.20 * unweighted_count
            or np.nanmin(all_treatment_prob) <= 0.01
            or np.nanmin(all_observation_prob) <= 0.01
        )
        balance_frame = group.copy()
        balance_frame["analysis_weight"] = truncated
        balance = max_weighted_smd(balance_frame.loc[balance_frame["analysis_weight"].notna()], "analysis_weight", "uncensored_through_target")
        diagnostics.append(
            {
                **dict(zip(group_columns, keys, strict=False)),
                "status": "not_estimable_weight_gate" if positivity_fail else "estimable",
                "n_eligible": int(len(mature)),
                "n_observed_uncensored": unweighted_count,
                "effective_sample_size": ess,
                "ess_fraction": ess / max(unweighted_count, 1),
                "weight_min": float(np.nanmin(observed_weights)) if len(observed_weights) else np.nan,
                "weight_median": float(np.nanmedian(observed_weights)) if len(observed_weights) else np.nan,
                "weight_max": float(np.nanmax(observed_weights)) if len(observed_weights) else np.nan,
                "truncation_low": float(low),
                "truncation_high": float(high),
                "max_weighted_smd": balance,
                "nuisance_degenerate": degenerate,
            }
        )
    weighted["analysis_weight"] = weighted["analysis_weight"].fillna(1.0)
    return weighted, pd.DataFrame(diagnostics)


def prediction_identity(frame: Any, candidate: str, architecture: str) -> Any:
    columns = [
        "row_id", "patient_id", "cohort", "outcome", "origin_month", "target_month", "split",
        "target_value", "target_observed", "analysis_weight", "support_status", "treatment",
        "center_id", "prediction_reference_value",
    ]
    result = frame[columns].copy()
    result["candidate"] = candidate
    result["architecture"] = architecture
    return result


def empirical_quantiles(values: Any, fallback: float = 0.0) -> Any:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.full(len(QUANTILES), fallback, dtype=float)
    return np.quantile(array, QUANTILES)


def baseline_group_edges(training: Any) -> Any:
    values = pd.to_numeric(training["baseline_value"], errors="coerce").dropna()
    if len(values) < 8:
        return np.array([-np.inf, np.inf])
    inner = np.unique(np.quantile(values, [0.25, 0.50, 0.75]))
    return np.concatenate(([-np.inf], inner, [np.inf]))


def apply_baseline_groups(frame: Any, edges: Any) -> Any:
    result = frame.copy()
    result["baseline_group"] = pd.cut(
        pd.to_numeric(result["baseline_value"], errors="coerce"), edges, include_lowest=True, duplicates="drop"
    ).astype("string").fillna("missing")
    return result


def fit_population_baseline(task: Any) -> Any:
    training = task.loc[task["split"].eq("train") & task["target_observed"]].copy()
    edges = baseline_group_edges(training)
    training = apply_baseline_groups(training, edges)
    scored = apply_baseline_groups(task, edges)
    fine_keys = ["target_month", "treatment", "baseline_group"]
    coarse_keys = ["target_month", "treatment"]
    horizon_keys = ["target_month"]
    fine: dict[tuple[Any, ...], Any] = {}
    coarse: dict[tuple[Any, ...], Any] = {}
    horizon: dict[tuple[Any, ...], Any] = {}
    for keys, group in training.groupby(fine_keys, observed=True, sort=False):
        if len(group) >= 20:
            fine[tuple(keys)] = empirical_quantiles(group["target_change"])
    for keys, group in training.groupby(coarse_keys, observed=True, sort=False):
        if len(group) >= 20:
            coarse[tuple(keys)] = empirical_quantiles(group["target_change"])
    for keys, group in training.groupby(horizon_keys, observed=True, sort=False):
        horizon[(keys if isinstance(keys, tuple) else (keys,))] = empirical_quantiles(group["target_change"])
    overall = empirical_quantiles(training["target_change"])
    predictions = []
    for row in scored.itertuples(index=False):
        fine_key = (row.target_month, row.treatment, row.baseline_group)
        coarse_key = (row.target_month, row.treatment)
        horizon_key = (row.target_month,)
        change_quantiles = fine.get(fine_key, coarse.get(coarse_key, horizon.get(horizon_key, overall)))
        predictions.append(np.asarray(change_quantiles) + float(row.prediction_reference_value))
    result = prediction_identity(task, "population_change", "empirical_baseline")
    result[list(QUANTILE_COLUMNS)] = rearrange_quantiles(np.vstack(predictions))
    return result


def fit_persistence_baseline(task: Any) -> Any:
    training = task.loc[task["split"].eq("train") & task["target_observed"]].copy()
    residuals = {
        month: empirical_quantiles(group["target_value"] - group["prediction_reference_value"])
        for month, group in training.groupby("target_month", sort=False)
    }
    overall = empirical_quantiles(training["target_value"] - training["prediction_reference_value"])
    matrix = np.vstack(
        [residuals.get(row.target_month, overall) + float(row.prediction_reference_value) for row in task.itertuples(index=False)]
    )
    result = prediction_identity(task, "persistence", "empirical_baseline")
    result[list(QUANTILE_COLUMNS)] = rearrange_quantiles(matrix)
    return result


def fit_spline_candidate(task: Any, cfg: RunConfig) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

    observed = task.loc[task["target_observed"]].copy()
    training = observed.loc[observed["split"].eq("train")].copy()
    validation = observed.loc[observed["split"].eq("validation")].copy()
    continuous = ["target_month", "time_from_origin_months", "baseline_value", "last_value", "age_at_index", "measurement_recency_days"]
    categorical = ["treatment", "sex", "race", "coverage"]
    continuous_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("continuous", continuous_pipeline, continuous),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    x_train = transformer.fit_transform(training)
    x_validation = transformer.transform(validation) if len(validation) else None
    best_alpha = 1.0
    best_score = float("inf")
    for alpha in (0.1, 1.0, 10.0):
        model = Ridge(alpha=alpha)
        model.fit(x_train, training["target_change"], sample_weight=training["analysis_weight"])
        if len(validation):
            score = float(np.mean(np.abs(validation["target_change"].to_numpy() - model.predict(x_validation))))
        else:
            score = float(np.mean(np.abs(training["target_change"].to_numpy() - model.predict(x_train))))
        if score < best_score:
            best_score, best_alpha = score, alpha
    model = Ridge(alpha=best_alpha)
    model.fit(x_train, training["target_change"], sample_weight=training["analysis_weight"])
    x_all = transformer.transform(task)
    median_change = model.predict(x_all)
    train_residual = training["target_change"].to_numpy() - model.predict(x_train)
    residual_by_horizon = {
        month: empirical_quantiles(train_residual[training["target_month"].to_numpy() == month])
        for month in sorted(training["target_month"].unique())
    }
    overall = empirical_quantiles(train_residual)
    matrix = np.vstack(
        [
            float(reference) + float(center) + residual_by_horizon.get(month, overall)
            for reference, center, month in zip(
                task["prediction_reference_value"], median_change, task["target_month"], strict=False
            )
        ]
    )
    result = prediction_identity(task, "regularized_spline", "ridge_cubic_spline")
    result[list(QUANTILE_COLUMNS)] = rearrange_quantiles(matrix)
    result["model_detail"] = f"ridge alpha={best_alpha:g}"
    return result


def hgb_parameter_candidates(cfg: RunConfig) -> list[dict[str, Any]]:
    choices = [
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 1.0},
        {"learning_rate": 0.04, "max_leaf_nodes": 31, "min_samples_leaf": 80, "l2_regularization": 10.0},
        {"learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 25, "l2_regularization": 0.1},
        {"learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 120, "l2_regularization": 3.0},
    ]
    return choices[: max(1, min(len(choices), cfg.model_trials))]


def fit_hgb_candidate(task: Any, cfg: RunConfig) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    observed = task.loc[task["target_observed"]].copy()
    training = observed.loc[observed["split"].eq("train")].copy()
    validation = observed.loc[observed["split"].eq("validation")].copy()
    encoder = TabularEncoder.fit(training)
    x_train = encoder.transform(training)
    x_validation = encoder.transform(validation) if len(validation) else x_train
    y_validation = validation["target_change"].to_numpy() if len(validation) else training["target_change"].to_numpy()
    w_validation = validation["analysis_weight"].to_numpy() if len(validation) else training["analysis_weight"].to_numpy()
    best_params: dict[str, Any] | None = None
    best_score = float("inf")
    best_models: list[Any] = []
    for params in hgb_parameter_candidates(cfg):
        models = []
        validation_matrix = []
        for quantile in QUANTILES:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                max_iter=cfg.hgb_iterations,
                early_stopping=False,
                random_state=cfg.seed,
                **params,
            )
            model.fit(x_train, training["target_change"], sample_weight=training["analysis_weight"])
            models.append(model)
            validation_matrix.append(model.predict(x_validation))
        matrix = rearrange_quantiles(np.column_stack(validation_matrix))
        score = quantile_crps(y_validation, matrix, w_validation)
        if score < best_score:
            best_score, best_params, best_models = score, dict(params), models
    x_all = encoder.transform(task)
    change_matrix = rearrange_quantiles(np.column_stack([model.predict(x_all) for model in best_models]))
    absolute = change_matrix + task["prediction_reference_value"].to_numpy(float)[:, None]
    result = prediction_identity(task, "histogram_gradient_boosting", "hist_gradient_boosting_quantile")
    result[list(QUANTILE_COLUMNS)] = rearrange_quantiles(absolute)
    result["model_detail"] = canonical_json(best_params or {})
    return result


def fit_catboost_candidate(task: Any, cfg: RunConfig) -> Any:
    from catboost import CatBoostRegressor

    observed = task.loc[task["target_observed"]].copy()
    training = observed.loc[observed["split"].eq("train")].copy()
    validation = observed.loc[observed["split"].eq("validation")].copy()
    feature_columns = list(NUMERIC_MODEL_FEATURES) + list(CATEGORICAL_MODEL_FEATURES)
    categorical_indices = list(range(len(NUMERIC_MODEL_FEATURES), len(feature_columns)))

    def prepare(frame: Any) -> Any:
        result = frame[feature_columns].copy()
        for column in NUMERIC_MODEL_FEATURES:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(float(training[column].median()) if training[column].notna().any() else 0.0)
        for column in CATEGORICAL_MODEL_FEATURES:
            result[column] = result[column].astype("string").fillna("<MISSING>").astype(str)
        return result

    alpha = ",".join(str(value) for value in QUANTILES)
    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={alpha}",
        iterations=cfg.catboost_iterations,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=8.0,
        random_seed=cfg.seed,
        random_strength=0.0,
        bootstrap_type="No",
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    fit_kwargs: dict[str, Any] = {
        "X": prepare(training),
        "y": training["target_change"],
        "cat_features": categorical_indices,
        "sample_weight": training["analysis_weight"],
    }
    if len(validation):
        fit_kwargs["eval_set"] = (prepare(validation), validation["target_change"])
        fit_kwargs["early_stopping_rounds"] = 20 if cfg.smoke else 200
    model.fit(**fit_kwargs)
    change_matrix = np.asarray(model.predict(prepare(task)), dtype=float)
    if change_matrix.ndim == 1:
        raise RuntimeError("CatBoost MultiQuantile returned a median-only prediction")
    absolute = rearrange_quantiles(change_matrix) + task["prediction_reference_value"].to_numpy(float)[:, None]
    result = prediction_identity(task, "catboost_multi_quantile", "catboost_multi_quantile")
    result[list(QUANTILE_COLUMNS)] = rearrange_quantiles(absolute)
    result["model_detail"] = f"joint MultiQuantile; trees={int(model.tree_count_)}"
    return result


# ======================================================================================
# 7. Pure-PyTorch direct quantile MLP and continuous-time ODE-RNN
# ======================================================================================


@dataclass
class NeuralInputEncoder:
    numeric: list[str]
    categorical: list[str]
    medians: dict[str, float]
    scales: dict[str, float]
    levels: dict[str, list[str]]

    @classmethod
    def fit(cls, frame: Any) -> "NeuralInputEncoder":
        numeric = list(NUMERIC_MODEL_FEATURES)
        categorical = list(CATEGORICAL_MODEL_FEATURES)
        medians: dict[str, float] = {}
        scales: dict[str, float] = {}
        levels: dict[str, list[str]] = {}
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            scale = float(values.std(ddof=0)) if values.notna().sum() > 1 else 1.0
            medians[column] = median
            scales[column] = scale if math.isfinite(scale) and scale > 1e-8 else 1.0
        for column in categorical:
            values = frame[column].astype("string").fillna("<MISSING>").astype(str)
            levels[column] = ["<UNKNOWN>"] + sorted(str(item) for item in values.unique())
        return cls(numeric, categorical, medians, scales, levels)

    def transform(self, frame: Any) -> tuple[Any, list[Any]]:
        numeric_columns: list[Any] = []
        for column in self.numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            numeric_columns.append(
                (values.fillna(self.medians[column]).to_numpy(float) - self.medians[column]) / self.scales[column]
            )
            numeric_columns.append(values.isna().to_numpy(float))
        numeric = np.column_stack(numeric_columns).astype(np.float32)
        categories: list[Any] = []
        for column in self.categorical:
            mapping = {value: index for index, value in enumerate(self.levels[column])}
            values = frame[column].astype("string").fillna("<MISSING>").astype(str)
            categories.append(values.map(lambda value: mapping.get(value, 0)).to_numpy(np.int64))
        return numeric, categories

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.levels[column]) for column in self.categorical]


def noncrossing_quantiles_torch(raw: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    if raw.shape[-1] != len(QUANTILES):
        raise ValueError("The noncrossing head requires seven raw outputs")
    median = raw[..., 3]
    q25 = median - functional.softplus(raw[..., 2])
    q10 = q25 - functional.softplus(raw[..., 1])
    q05 = q10 - functional.softplus(raw[..., 0])
    q75 = median + functional.softplus(raw[..., 4])
    q90 = q75 + functional.softplus(raw[..., 5])
    q95 = q90 + functional.softplus(raw[..., 6])
    return torch.stack((q05, q10, q25, median, q75, q90, q95), dim=-1)


def embedding_dimension(cardinality: int) -> int:
    return int(min(16, max(2, math.ceil(math.sqrt(max(cardinality, 2))))))


def build_quantile_mlp(n_numeric: int, cardinalities: Sequence[int], width: int = 128, depth: int = 3, dropout: float = 0.15) -> Any:
    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, block_width: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(block_width, block_width),
                nn.SiLU(),
                nn.LayerNorm(block_width),
                nn.Dropout(dropout),
                nn.Linear(block_width, block_width),
                nn.SiLU(),
                nn.LayerNorm(block_width),
                nn.Dropout(dropout),
            )

        def forward(self, inputs: Any) -> Any:
            return inputs + self.network(inputs)

    class QuantileMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(cardinality, embedding_dimension(cardinality)) for cardinality in cardinalities]
            )
            input_width = n_numeric + sum(layer.embedding_dim for layer in self.embeddings)
            self.input_projection = nn.Sequential(nn.Linear(input_width, width), nn.SiLU(), nn.LayerNorm(width))
            self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(depth)])
            self.output = nn.Linear(width, len(QUANTILES))

        def forward(self, numeric: Any, categories: Sequence[Any]) -> Any:
            components = [numeric]
            components.extend(layer(values) for layer, values in zip(self.embeddings, categories, strict=False))
            hidden = self.input_projection(torch.cat(components, dim=-1))
            for block in self.blocks:
                hidden = block(hidden)
            return noncrossing_quantiles_torch(self.output(hidden))

    return QuantileMLP()


def torch_pinball_loss(y_true: Any, predictions: Any, weights: Any) -> Any:
    import torch

    quantiles = torch.tensor(QUANTILES, device=predictions.device, dtype=predictions.dtype).view(1, -1)
    error = y_true.view(-1, 1) - predictions
    loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean(dim=1)
    return (loss * weights).sum() / torch.clamp(weights.sum(), min=1e-12)


def fit_one_mlp_seed(task: Any, encoder: NeuralInputEncoder, cfg: RunConfig, seed: int) -> tuple[Any, dict[str, Any]]:
    import torch

    set_deterministic_seed(seed, include_torch=True)
    observed = task.loc[task["target_observed"]].copy()
    training = observed.loc[observed["split"].eq("train")].copy()
    validation = observed.loc[observed["split"].eq("validation")].copy()
    train_numeric, train_categories_np = encoder.transform(training)
    validation_numeric, validation_categories_np = encoder.transform(validation)
    model = build_quantile_mlp(
        train_numeric.shape[1],
        encoder.cardinalities,
        width=64 if cfg.smoke else 256,
        depth=2 if cfg.smoke else 3,
        dropout=0.10 if cfg.smoke else 0.20,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    x_train = torch.tensor(train_numeric, dtype=torch.float32)
    c_train = [torch.tensor(values, dtype=torch.long) for values in train_categories_np]
    y_train = torch.tensor(training["target_change"].to_numpy(np.float32), dtype=torch.float32)
    w_train = torch.tensor(training["analysis_weight"].to_numpy(np.float32), dtype=torch.float32)
    x_validation = torch.tensor(validation_numeric, dtype=torch.float32)
    c_validation = [torch.tensor(values, dtype=torch.long) for values in validation_categories_np]
    y_validation = torch.tensor(validation["target_change"].to_numpy(np.float32), dtype=torch.float32)
    w_validation = torch.tensor(validation["analysis_weight"].to_numpy(np.float32), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    patience = 5 if cfg.smoke else 25
    stale = 0
    curve: list[dict[str, float]] = []
    batch_size = min(256, max(16, len(training)))
    for epoch in range(cfg.mlp_epochs):
        model.train()
        permutation = torch.randperm(len(training), generator=generator)
        epoch_losses: list[float] = []
        for start in range(0, len(training), batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train[indices], [values[indices] for values in c_train])
            loss = torch_pinball_loss(y_train[indices], prediction, w_train[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("MLP training produced a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            if len(validation):
                validation_loss = float(torch_pinball_loss(y_validation, model(x_validation, c_validation), w_validation))
            else:
                validation_loss = float(np.mean(epoch_losses))
        curve.append({"epoch": float(epoch), "train_loss": float(np.mean(epoch_losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed,
        "best_validation_pinball": best_loss,
        "epochs": len(curve),
        "curve": curve,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def fit_mlp_candidate(task: Any, cfg: RunConfig) -> tuple[Any, dict[str, Any]]:
    import torch

    training = task.loc[task["split"].eq("train") & task["target_observed"]].copy()
    encoder = NeuralInputEncoder.fit(training)
    all_numeric, all_categories_np = encoder.transform(task)
    x_all = torch.tensor(all_numeric, dtype=torch.float32)
    c_all = [torch.tensor(values, dtype=torch.long) for values in all_categories_np]
    seed_predictions: list[Any] = []
    run_details: list[dict[str, Any]] = []
    for offset in range(cfg.final_neural_seeds):
        model, details = fit_one_mlp_seed(task, encoder, cfg, cfg.seed + offset * 101)
        with torch.no_grad():
            seed_predictions.append(model(x_all, c_all).cpu().numpy())
        run_details.append(details)
    change_matrix = np.mean(np.stack(seed_predictions, axis=0), axis=0)
    absolute = rearrange_quantiles(change_matrix) + task["prediction_reference_value"].to_numpy(float)[:, None]
    result = prediction_identity(task, "pytorch_quantile_mlp", "direct_horizon_mlp")
    result[list(QUANTILE_COLUMNS)] = rearrange_quantiles(absolute)
    result["model_detail"] = f"seed average n={cfg.final_neural_seeds}"
    return result, {"runs": run_details, "seed_count": cfg.final_neural_seeds}


def rk4_integrate(
    vector_field: Callable[[Any, Any, Any], Any],
    state: Any,
    context: Any,
    start_time: Any,
    end_time: Any,
    max_step: float = 1.0 / 12.0,
) -> tuple[Any, int]:
    import torch

    if max_step <= 0:
        raise ValueError("RK4 maximum step must be positive")
    start = torch.as_tensor(start_time, dtype=state.dtype, device=state.device)
    end = torch.as_tensor(end_time, dtype=state.dtype, device=state.device)
    delta = end - start
    delta_value = float(delta.detach().cpu())
    if delta_value < -1e-14:
        raise ValueError("RK4 cannot integrate a negative time interval")
    if abs(delta_value) <= 1e-14:
        return state, 0
    n_steps = int(math.ceil(delta_value / max_step))
    step = delta / n_steps
    current = state
    current_time = start
    for _ in range(n_steps):
        k1 = vector_field(current_time, current, context)
        k2 = vector_field(current_time + step / 2, current + step * k1 / 2, context)
        k3 = vector_field(current_time + step / 2, current + step * k2 / 2, context)
        k4 = vector_field(current_time + step, current + step * k3, context)
        current = current + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        current_time = current_time + step
    return current, n_steps


def build_ode_rnn(
    n_static_numeric: int,
    cardinalities: Sequence[int],
    latent_dim: int = 32,
    context_dim: int = 16,
    field_width: int = 64,
    decoder_width: int = 64,
) -> Any:
    import torch
    from torch import nn
    import torch.nn.functional as functional

    class StaticEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(cardinality, embedding_dimension(cardinality)) for cardinality in cardinalities]
            )
            input_width = n_static_numeric + sum(item.embedding_dim for item in self.embeddings)
            self.network = nn.Sequential(
                nn.Linear(input_width, max(context_dim, 32)), nn.SiLU(), nn.LayerNorm(max(context_dim, 32)),
                nn.Linear(max(context_dim, 32), context_dim), nn.SiLU(), nn.LayerNorm(context_dim),
            )

        def forward(self, numeric: Any, categories: Sequence[Any]) -> Any:
            components = [numeric]
            components.extend(layer(values) for layer, values in zip(self.embeddings, categories, strict=False))
            return self.network(torch.cat(components, dim=-1))

    class VectorField(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(latent_dim + context_dim + 3, field_width), nn.SiLU(), nn.LayerNorm(field_width),
                nn.Linear(field_width, field_width), nn.SiLU(), nn.LayerNorm(field_width),
                nn.Linear(field_width, latent_dim),
            )
            initial_scale = math.log(math.expm1(0.1))
            self.raw_scale = nn.Parameter(torch.full((latent_dim,), initial_scale))

        def forward(self, time_value: Any, state: Any, context: Any) -> Any:
            time_tensor = torch.as_tensor(time_value, dtype=state.dtype, device=state.device)
            if time_tensor.ndim == 0:
                time_tensor = time_tensor.expand(state.shape[:-1] + (1,))
            elif time_tensor.shape[-1:] != (1,):
                time_tensor = time_tensor.unsqueeze(-1)
            representation = torch.cat((time_tensor, time_tensor.square(), torch.log1p(torch.clamp(time_tensor, min=0))), dim=-1)
            derivative = self.network(torch.cat((state, context, representation), dim=-1))
            return torch.tanh(derivative) * functional.softplus(self.raw_scale)

    class ODERNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.static_encoder = StaticEncoder()
            self.initializer = nn.Sequential(
                nn.Linear(context_dim + 3, latent_dim), nn.SiLU(), nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
            )
            self.vector_field = VectorField()
            self.update = nn.GRUCell(context_dim + 4, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + context_dim + 3, decoder_width), nn.SiLU(), nn.LayerNorm(decoder_width),
                nn.Linear(decoder_width, decoder_width), nn.SiLU(), nn.LayerNorm(decoder_width),
                nn.Linear(decoder_width, len(QUANTILES)),
            )

        def forward(
            self,
            static_numeric: Any,
            categories: Sequence[Any],
            baseline_value: Any,
            baseline_recency: Any,
            event_times: Any,
            event_values: Any,
            event_masks: Any,
            origin_times: Any,
            target_times: Any,
            censor_times: Any | None = None,
            max_step: float = 1.0 / 12.0,
        ) -> tuple[Any, dict[str, Any]]:
            context = self.static_encoder(static_numeric, categories)
            batch_size = static_numeric.shape[0]
            outputs: list[Any] = []
            function_evaluations = 0
            accepted_events: list[int] = []
            for patient_index in range(batch_size):
                current_context = context[patient_index:patient_index + 1]
                base = baseline_value[patient_index:patient_index + 1].reshape(1, 1)
                recency = baseline_recency[patient_index:patient_index + 1].reshape(1, 1)
                missing = torch.isnan(base).to(base.dtype)
                base = torch.nan_to_num(base)
                state = self.initializer(torch.cat((current_context, base, recency, missing), dim=-1))
                prior_time = torch.zeros((), device=state.device, dtype=state.dtype)
                origin = origin_times[patient_index]
                censor = censor_times[patient_index] if censor_times is not None else torch.tensor(float("inf"), device=state.device, dtype=state.dtype)
                accepted = 0
                event_order = torch.argsort(event_times[patient_index])
                for event_index in event_order:
                    event_time = event_times[patient_index, event_index]
                    valid = bool(event_masks[patient_index, event_index])
                    if not valid or bool(event_time > origin) or bool(event_time >= censor):
                        continue
                    state, steps = rk4_integrate(self.vector_field, state, current_context, prior_time, event_time, max_step)
                    function_evaluations += steps * 4
                    elapsed = (event_time - prior_time).reshape(1, 1)
                    event = torch.cat(
                        (
                            event_values[patient_index, event_index].reshape(1, 1),
                            torch.ones((1, 1), device=state.device, dtype=state.dtype),
                            elapsed,
                            torch.zeros((1, 1), device=state.device, dtype=state.dtype),
                            current_context,
                        ),
                        dim=-1,
                    )
                    state = self.update(event, state)
                    prior_time = event_time
                    accepted += 1
                if bool(target_times[patient_index] <= origin):
                    raise ValueError("ODE target time must be strictly after its prediction origin")
                state, steps = rk4_integrate(
                    self.vector_field, state, current_context, prior_time, target_times[patient_index], max_step
                )
                function_evaluations += steps * 4
                target = target_times[patient_index].reshape(1, 1)
                target_representation = torch.cat((target, target.square(), torch.log1p(torch.clamp(target, min=0))), dim=-1)
                outputs.append(noncrossing_quantiles_torch(self.decoder(torch.cat((state, current_context, target_representation), dim=-1))))
                accepted_events.append(accepted)
            return torch.cat(outputs, dim=0), {
                "function_evaluations": function_evaluations,
                "accepted_events": accepted_events,
            }

    return ODERNN()


def ode_suitability_gates(cohorts: Any, measurements: Any, dependencies: Mapping[str, Any]) -> Any:
    rows: list[dict[str, Any]] = []
    development_patients = set(cohorts.loc[cohorts["split"].isin(["train", "validation"]), "patient_id"].astype(str))
    for cohort_name in ("surgery", "incretin"):
        cohort_patients = set(
            cohorts.loc[
                cohorts["cohort"].eq(cohort_name) & cohorts["patient_id"].astype(str).isin(development_patients), "patient_id"
            ].astype(str)
        )
        for outcome in ("bmi", "hba1c"):
            subset = measurements.loc[
                measurements["patient_id"].astype(str).isin(cohort_patients) & measurements["outcome"].eq(outcome)
            ].copy()
            counts = subset.groupby("patient_id").size()
            repeated_fraction = float((counts >= 3).mean()) if len(counts) else 0.0
            task_cohorts = cohorts.loc[cohorts["cohort"].eq(cohort_name) & cohorts["patient_id"].astype(str).isin(cohort_patients)]
            strata_counts = task_cohorts.groupby("treatment")["patient_id"].nunique()
            exact_dates = "measurement_date" in subset and subset["measurement_date"].notna().all()
            if exact_dates and "timing_precision" in subset:
                exact_labels = {"exact", "exact_day", "index_event_date"}
                exact_dates = bool(
                    subset["timing_precision"].astype(str).str.lower().isin(exact_labels).all()
                )
            early_late = False
            if not subset.empty and exact_dates:
                index_map = task_cohorts.drop_duplicates("patient_id").set_index("patient_id")["index_date"]
                days = subset.apply(
                    lambda row: (
                        pd.Timestamp(row["measurement_date"])
                        - pd.Timestamp(index_map.get(str(row["patient_id"]), row["measurement_date"]))
                    ).days,
                    axis=1,
                )
                early_late = bool((days.between(0, 365)).any() and (days >= 730).any())
            gates = {
                "exact_day_timestamps": bool(exact_dates),
                "development_patients_at_least_5000": len(cohort_patients) >= 5000,
                "measurements_at_least_20000": len(subset) >= 20000,
                "at_least_30_pct_with_3_measurements": repeated_fraction >= 0.30,
                "early_and_late_repeated_support": early_late,
                "treatment_strata_at_least_1000": bool(len(strata_counts) and strata_counts.min() >= 1000),
                "pytorch_available": bool(dependencies.get("torch_importable")),
            }
            failed = [name for name, passed in gates.items() if not passed]
            rows.append(
                {
                    "cohort": cohort_name,
                    "outcome": outcome,
                    "appropriate": not failed,
                    "failed_gates": " | ".join(failed),
                    "development_patients": len(cohort_patients),
                    "valid_measurements": len(subset),
                    "repeated_fraction": repeated_fraction,
                    "minimum_treatment_stratum": int(strata_counts.min()) if len(strata_counts) else 0,
                    **{f"gate_{name}": passed for name, passed in gates.items()},
                }
            )
    return pd.DataFrame(rows)


def solver_sensitivity_metrics(monthly: Any, half_monthly: Any, development_iqr: float, y_true: Any | None = None) -> dict[str, float]:
    first = np.asarray(monthly, dtype=float)
    second = np.asarray(half_monthly, dtype=float)
    median_difference = np.abs(first[..., 3] - second[..., 3]) / max(float(development_iqr), 1e-12)
    result = {
        "median_patient_iqr_fraction": float(np.median(median_difference)),
        "p99_iqr_fraction": float(np.quantile(median_difference, 0.99)),
    }
    if y_true is not None:
        first_crps = quantile_crps(y_true, first)
        second_crps = quantile_crps(y_true, second)
        result["crps_relative_change"] = abs(first_crps - second_crps) / max(abs(first_crps), 1e-12)
    return result


def prepare_ode_examples(task: Any, cohorts: Any, measurements: Any, split: str | None = None) -> dict[str, Any]:
    frame = task.loc[task["target_observed"]].copy()
    if split is not None:
        frame = frame.loc[frame["split"].eq(split)].copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return {"frame": frame}
    training_reference = task.loc[task["split"].eq("train") & task["target_observed"]]
    outcome_mean = float(training_reference["target_value"].mean())
    outcome_scale = float(training_reference["target_value"].std(ddof=0))
    outcome_scale = outcome_scale if math.isfinite(outcome_scale) and outcome_scale > 1e-6 else 1.0
    static_columns = ["baseline_value", "age_at_index", "diabetes_flag", "index_year"]
    static_values = []
    static_medians = {}
    static_scales = {}
    for column in static_columns:
        reference = pd.to_numeric(training_reference[column], errors="coerce")
        median = float(reference.median()) if reference.notna().any() else 0.0
        scale = float(reference.std(ddof=0)) if reference.notna().sum() > 1 else 1.0
        static_medians[column] = median
        static_scales[column] = scale if math.isfinite(scale) and scale > 1e-8 else 1.0
        values = pd.to_numeric(frame[column], errors="coerce")
        static_values.extend(
            [
                ((values.fillna(median).to_numpy(float) - median) / static_scales[column]).astype(np.float32),
                values.isna().to_numpy(np.float32),
            ]
        )
    static_numeric = np.column_stack(static_values).astype(np.float32)
    categorical_columns = ["treatment", "sex", "race"]
    category_arrays: list[Any] = []
    cardinalities: list[int] = []
    for column in categorical_columns:
        levels = ["<UNKNOWN>"] + sorted(training_reference[column].astype("string").fillna("<MISSING>").astype(str).unique())
        mapping = {value: index for index, value in enumerate(levels)}
        category_arrays.append(frame[column].astype("string").fillna("<MISSING>").astype(str).map(lambda value: mapping.get(value, 0)).to_numpy(np.int64))
        cardinalities.append(len(levels))
    index_map = cohorts.drop_duplicates(["patient_id", "cohort"]).set_index(["patient_id", "cohort"])["index_date"].to_dict()
    measurement_groups = {
        (str(patient_id), str(outcome)): group
        for (patient_id, outcome), group in measurements.groupby(["patient_id", "outcome"], sort=False)
    }
    sequence_times: list[list[float]] = []
    sequence_values: list[list[float]] = []
    for row in frame.itertuples(index=False):
        group = measurement_groups.get((str(row.patient_id), str(row.outcome)), pd.DataFrame())
        index_date = pd.Timestamp(index_map[(str(row.patient_id), str(row.cohort))])
        events: list[tuple[float, float]] = []
        if not group.empty:
            for measurement in group.itertuples(index=False):
                day = int((pd.Timestamp(measurement.measurement_date) - index_date).days)
                if 0 < day <= int(row.origin_day) and (pd.isna(row.effective_censor_day) or day < int(row.effective_censor_day)):
                    events.append((day / DAYS_PER_YEAR, (float(measurement.value) - outcome_mean) / outcome_scale))
        events.sort()
        sequence_times.append([item[0] for item in events])
        sequence_values.append([item[1] for item in events])
    max_events = max((len(item) for item in sequence_times), default=0)
    event_times = np.zeros((len(frame), max_events), dtype=np.float32)
    event_values = np.zeros((len(frame), max_events), dtype=np.float32)
    event_masks = np.zeros((len(frame), max_events), dtype=bool)
    for index, (times, values) in enumerate(zip(sequence_times, sequence_values, strict=False)):
        event_times[index, :len(times)] = times
        event_values[index, :len(values)] = values
        event_masks[index, :len(times)] = True
    patient_origin_counts = frame.groupby("patient_id")["row_id"].transform("count").to_numpy(float)
    return {
        "frame": frame,
        "static_numeric": static_numeric,
        "category_arrays": category_arrays,
        "cardinalities": cardinalities,
        "baseline": ((frame["baseline_value"].to_numpy(float) - outcome_mean) / outcome_scale).astype(np.float32),
        "baseline_recency": (np.abs(frame["baseline_measurement_day"].to_numpy(float)) / DAYS_PER_YEAR).astype(np.float32),
        "event_times": event_times,
        "event_values": event_values,
        "event_masks": event_masks,
        "origin_times": (frame["origin_day"].to_numpy(float) / DAYS_PER_YEAR).astype(np.float32),
        "target_times": (frame["target_day"].to_numpy(float) / DAYS_PER_YEAR).astype(np.float32),
        "censor_times": np.where(frame["effective_censor_day"].notna(), frame["effective_censor_day"] / DAYS_PER_YEAR, np.inf).astype(np.float32),
        "targets": ((frame["target_value"].to_numpy(float) - outcome_mean) / outcome_scale).astype(np.float32),
        "weights": (frame["analysis_weight"].to_numpy(float) / np.maximum(patient_origin_counts, 1)).astype(np.float32),
        "sequence_lengths": np.asarray([len(item) for item in sequence_times], dtype=int),
        "outcome_mean": outcome_mean,
        "outcome_scale": outcome_scale,
    }


def ode_batch_tensors(data: Mapping[str, Any], indices: Any) -> dict[str, Any]:
    import torch

    index = np.asarray(indices, dtype=int)
    return {
        "static_numeric": torch.tensor(data["static_numeric"][index], dtype=torch.float32),
        "categories": [torch.tensor(values[index], dtype=torch.long) for values in data["category_arrays"]],
        "baseline": torch.tensor(data["baseline"][index], dtype=torch.float32),
        "baseline_recency": torch.tensor(data["baseline_recency"][index], dtype=torch.float32),
        "event_times": torch.tensor(data["event_times"][index], dtype=torch.float32),
        "event_values": torch.tensor(data["event_values"][index], dtype=torch.float32),
        "event_masks": torch.tensor(data["event_masks"][index], dtype=torch.bool),
        "origin_times": torch.tensor(data["origin_times"][index], dtype=torch.float32),
        "target_times": torch.tensor(data["target_times"][index], dtype=torch.float32),
        "censor_times": torch.tensor(data["censor_times"][index], dtype=torch.float32),
        "targets": torch.tensor(data["targets"][index], dtype=torch.float32),
        "weights": torch.tensor(data["weights"][index], dtype=torch.float32),
    }


def ode_forward_batch(model: Any, batch: Mapping[str, Any], max_step: float) -> tuple[Any, dict[str, Any]]:
    return model(
        batch["static_numeric"], batch["categories"], batch["baseline"], batch["baseline_recency"],
        batch["event_times"], batch["event_values"], batch["event_masks"], batch["origin_times"],
        batch["target_times"], batch["censor_times"], max_step=max_step,
    )


def fit_ode_candidate(task: Any, cohorts: Any, measurements: Any, cfg: RunConfig) -> tuple[Any, dict[str, Any]]:
    import torch

    prepared = prepare_ode_examples(task, cohorts, measurements)
    frame = prepared["frame"]
    train_indices = np.flatnonzero(frame["split"].eq("train").to_numpy())
    validation_indices = np.flatnonzero(frame["split"].eq("validation").to_numpy())
    all_indices = np.arange(len(frame))
    if len(train_indices) < 50:
        raise RuntimeError("ODE-RNN has fewer than 50 observed training examples")
    seed_predictions: list[Any] = []
    seed_details: list[dict[str, Any]] = []
    for seed_offset in range(cfg.final_neural_seeds):
        seed = cfg.seed + 5000 + seed_offset * 103
        set_deterministic_seed(seed, include_torch=True)
        model = build_ode_rnn(prepared["static_numeric"].shape[1], prepared["cardinalities"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        batch_size = 32 if cfg.smoke else 64
        patience = 5 if cfg.smoke else 25
        best_loss = float("inf")
        best_state = None
        stale = 0
        curves: list[dict[str, float]] = []
        generator = np.random.default_rng(seed)
        max_epochs = min(cfg.mlp_epochs, 300)
        started = time.perf_counter()
        for epoch in range(max_epochs):
            ordered = train_indices[np.argsort(prepared["sequence_lengths"][train_indices])]
            chunks = [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]
            generator.shuffle(chunks)
            epoch_losses: list[float] = []
            maximum_gradient = 0.0
            model.train()
            for indices in chunks:
                batch = ode_batch_tensors(prepared, indices)
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = ode_forward_batch(model, batch, cfg.max_ode_step)
                loss = torch_pinball_loss(batch["targets"], prediction, batch["weights"])
                if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
                    raise RuntimeError("ODE-RNN produced a nonfinite loss, state, derivative, or prediction")
                zero_state = torch.zeros((1, model.vector_field.raw_scale.numel()), dtype=torch.float32)
                zero_context = torch.zeros((1, model.static_encoder.network[-1].normalized_shape[0]), dtype=torch.float32)
                derivative = model.vector_field(torch.tensor(1.0), zero_state, zero_context)
                loss = loss + 1e-4 * derivative.square().mean()
                loss.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                if not math.isfinite(gradient):
                    raise RuntimeError("ODE-RNN gradient norm became nonfinite")
                maximum_gradient = max(maximum_gradient, gradient)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                evaluation_indices = validation_indices if len(validation_indices) else train_indices[: min(512, len(train_indices))]
                losses = []
                for start in range(0, len(evaluation_indices), batch_size):
                    batch = ode_batch_tensors(prepared, evaluation_indices[start:start + batch_size])
                    prediction, _ = ode_forward_batch(model, batch, cfg.max_ode_step)
                    losses.append(float(torch_pinball_loss(batch["targets"], prediction, batch["weights"])))
                validation_loss = float(np.mean(losses))
            curves.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "validation_loss": validation_loss, "max_gradient_norm": maximum_gradient})
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        predictions = []
        half_step_predictions = []
        total_nfe = 0
        with torch.no_grad():
            for start in range(0, len(all_indices), batch_size):
                batch = ode_batch_tensors(prepared, all_indices[start:start + batch_size])
                current, diagnostics = ode_forward_batch(model, batch, cfg.max_ode_step)
                half, _ = ode_forward_batch(model, batch, cfg.max_ode_step / 2.0)
                predictions.append(current.cpu().numpy())
                half_step_predictions.append(half.cpu().numpy())
                total_nfe += int(diagnostics["function_evaluations"])
        prediction_array = np.vstack(predictions)
        half_array = np.vstack(half_step_predictions)
        sensitivity = solver_sensitivity_metrics(
            prediction_array,
            half_array,
            development_iqr=max(float(np.subtract(*np.quantile(prepared["targets"], [0.75, 0.25]))), 1e-6),
            y_true=prepared["targets"],
        )
        seed_predictions.append(prediction_array)
        seed_details.append(
            {
                "seed": seed,
                "best_validation_pinball": best_loss,
                "epochs": len(curves),
                "curve": curves,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "training_seconds": time.perf_counter() - started,
                "function_evaluations": total_nfe,
                "solver_sensitivity": sensitivity,
            }
        )
    standardized = np.mean(np.stack(seed_predictions, axis=0), axis=0)
    absolute = standardized * prepared["outcome_scale"] + prepared["outcome_mean"]
    identity_rows = task.loc[task["row_id"].isin(frame["row_id"])].sort_values("row_id")
    frame_order = frame.sort_values("row_id")
    order_map = {row_id: index for index, row_id in enumerate(frame["row_id"])}
    matrix = np.vstack([absolute[order_map[row_id]] for row_id in identity_rows["row_id"]])
    result = prediction_identity(identity_rows, "pytorch_ode_rnn", "context_conditioned_ode_rnn")
    result[list(QUANTILE_COLUMNS)] = rearrange_quantiles(matrix)
    result["model_detail"] = f"seed average n={cfg.final_neural_seeds}; RK4 max step={cfg.max_ode_step:g} years"
    seed_crps = []
    observed_mask = frame["split"].isin(["validation", "temporal_test", "geographic_test"]).to_numpy()
    for item in seed_predictions:
        seed_crps.append(quantile_crps(prepared["targets"][observed_mask], item[observed_mask]))
    relative_range = (max(seed_crps) - min(seed_crps)) / max(float(np.mean(seed_crps)), 1e-12) if seed_crps else 0.0
    return result, {"runs": seed_details, "seed_crps": seed_crps, "seed_relative_crps_range": relative_range}


# ======================================================================================
# 8. Candidate roster, ensemble, conformal calibration, metrics, and decision gates
# ======================================================================================


def prediction_development_iqr_map(predictions: Any) -> dict[tuple[str, str, int], float]:
    development = predictions.loc[
        predictions["split"].eq("train") & predictions["target_observed"]
    ].drop_duplicates("row_id")
    scales: dict[tuple[str, str, int], float] = {}
    for keys, group in development.groupby(["cohort", "outcome", "target_month"], sort=True):
        values = pd.to_numeric(group["target_value"], errors="coerce").dropna().to_numpy(float)
        if len(values):
            q25, q75 = np.quantile(values, [0.25, 0.75])
            scales[(str(keys[0]), str(keys[1]), int(keys[2]))] = max(float(q75 - q25), 1e-8)
    return scales


def equal_horizon_standardized_crps(
    frame: Any,
    matrix: Any,
    weights: Any,
    scale_map: Mapping[tuple[str, str, int], float],
) -> float:
    if frame.empty:
        return math.nan
    values: list[float] = []
    target_months = pd.to_numeric(frame["target_month"], errors="coerce").to_numpy(int)
    for target_month in sorted(set(int(item) for item in target_months)):
        mask = target_months == target_month
        cohort = str(frame.iloc[0]["cohort"])
        outcome = str(frame.iloc[0]["outcome"])
        scale = float(scale_map.get((cohort, outcome, target_month), math.nan))
        if not math.isfinite(scale) or scale <= 0:
            continue
        y = frame.loc[mask, "target_value"].to_numpy(float)
        values.append(
            quantile_crps(
                y,
                np.asarray(matrix, dtype=float)[mask],
                np.asarray(weights, dtype=float)[mask],
            )
            / scale
        )
    return float(np.mean(values)) if values else math.nan


def candidate_validation_scores(predictions: Any) -> Any:
    rows: list[dict[str, Any]] = []
    scale_map = prediction_development_iqr_map(predictions)
    observed = predictions.loc[predictions["split"].eq("validation") & predictions["target_observed"]].copy()
    group_columns = ["cohort", "outcome", "origin_month", "candidate", "architecture"]
    for keys, group in observed.groupby(group_columns, sort=True):
        matrix = group[list(QUANTILE_COLUMNS)].to_numpy(float)
        y = group["target_value"].to_numpy(float)
        weights = group["analysis_weight"].to_numpy(float)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=False)),
                "validation_crps": quantile_crps(y, matrix, weights),
                "validation_standardized_crps": equal_horizon_standardized_crps(
                    group,
                    matrix,
                    weights,
                    scale_map,
                ),
                "validation_rmse": float(np.sqrt(np.average((y - matrix[:, 3]) ** 2, weights=weights))),
                "coverage_80": float(np.average((y >= matrix[:, 1]) & (y <= matrix[:, 5]), weights=weights)),
                "coverage_90": float(np.average((y >= matrix[:, 0]) & (y <= matrix[:, 6]), weights=weights)),
                "n_validation": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def add_ensemble_candidates(predictions: Any, cfg: RunConfig) -> tuple[Any, Any]:
    leaderboard = candidate_validation_scores(predictions)
    if leaderboard.empty:
        return predictions, pd.DataFrame()
    ensemble_predictions: list[Any] = []
    weight_rows: list[dict[str, Any]] = []
    scale_map = prediction_development_iqr_map(predictions)
    group_columns = ["cohort", "outcome", "origin_month"]
    for keys, scores in leaderboard.groupby(group_columns, sort=True):
        scores = scores.sort_values(["validation_standardized_crps", "validation_crps"])
        selected_candidates: list[str] = []
        used_architectures: set[str] = set()
        for row in scores.itertuples(index=False):
            if row.architecture in used_architectures:
                continue
            selected_candidates.append(str(row.candidate))
            used_architectures.add(str(row.architecture))
            if len(selected_candidates) == 3:
                break
        if len(selected_candidates) < 2:
            continue
        cohort_name, outcome, origin_month = keys
        task = predictions.loc[
            predictions["cohort"].eq(cohort_name)
            & predictions["outcome"].eq(outcome)
            & predictions["origin_month"].eq(origin_month)
            & predictions["candidate"].isin(selected_candidates)
        ].copy()
        candidate_frames = {
            candidate: task.loc[task["candidate"].eq(candidate)].sort_values("row_id")
            for candidate in selected_candidates
        }
        shared_ids = set.intersection(*(set(frame["row_id"]) for frame in candidate_frames.values()))
        if not shared_ids:
            continue
        matrices = []
        identity: Any | None = None
        for candidate in selected_candidates:
            frame = candidate_frames[candidate].loc[candidate_frames[candidate]["row_id"].isin(shared_ids)].sort_values("row_id")
            identity = frame if identity is None else identity
            matrices.append(frame[list(QUANTILE_COLUMNS)].to_numpy(float))
        assert identity is not None
        identity = identity.loc[identity["row_id"].isin(shared_ids)].sort_values("row_id")
        validation_mask = identity["split"].eq("validation").to_numpy() & identity["target_observed"].to_numpy(bool)
        if not validation_mask.any():
            continue
        y = identity.loc[validation_mask, "target_value"].to_numpy(float)
        sample_weight = identity.loc[validation_mask, "analysis_weight"].to_numpy(float)
        validation_identity = identity.loc[validation_mask].copy()
        best_weights = np.full(len(matrices), 1.0 / len(matrices))
        best_matrix = sum(weight * matrix[validation_mask] for weight, matrix in zip(best_weights, matrices, strict=False))
        best_matrix = rearrange_quantiles(best_matrix)
        best_score = equal_horizon_standardized_crps(
            validation_identity,
            best_matrix,
            sample_weight,
            scale_map,
        )
        best_raw_score = quantile_crps(y, best_matrix, sample_weight)
        best_base = scores.iloc[0]
        rng = np.random.default_rng(cfg.seed + int(origin_month) * 17 + len(outcome))
        trial_count = 80 if cfg.smoke else 1000
        for _ in range(trial_count):
            weights = rng.dirichlet(np.ones(len(matrices)))
            matrix = rearrange_quantiles(
                sum(weight * item[validation_mask] for weight, item in zip(weights, matrices, strict=False))
            )
            coverage_80 = float(np.average((y >= matrix[:, 1]) & (y <= matrix[:, 5]), weights=sample_weight))
            coverage_90 = float(np.average((y >= matrix[:, 0]) & (y <= matrix[:, 6]), weights=sample_weight))
            if abs(coverage_80 - 0.80) > abs(float(best_base["coverage_80"]) - 0.80) + 1e-8:
                continue
            if abs(coverage_90 - 0.90) > abs(float(best_base["coverage_90"]) - 0.90) + 1e-8:
                continue
            score = equal_horizon_standardized_crps(
                validation_identity,
                matrix,
                sample_weight,
                scale_map,
            )
            if score < best_score:
                best_score = score
                best_raw_score = quantile_crps(y, matrix, sample_weight)
                best_weights = weights
        all_matrix = rearrange_quantiles(
            sum(weight * item for weight, item in zip(best_weights, matrices, strict=False))
        )
        ensemble = identity.copy()
        ensemble["candidate"] = "validation_weighted_ensemble"
        ensemble["architecture"] = "architecture_ensemble"
        ensemble[list(QUANTILE_COLUMNS)] = all_matrix
        ensemble["model_detail"] = " | ".join(
            f"{candidate}={weight:.4f}" for candidate, weight in zip(selected_candidates, best_weights, strict=False)
        )
        ensemble_predictions.append(ensemble)
        for candidate, architecture_weight in zip(selected_candidates, best_weights, strict=False):
            weight_rows.append(
                {
                    "cohort": cohort_name,
                    "outcome": outcome,
                    "origin_month": origin_month,
                    "candidate": candidate,
                    "weight": float(architecture_weight),
                    "validation_crps": best_raw_score,
                    "validation_standardized_crps": best_score,
                }
            )
    if ensemble_predictions:
        predictions = pd.concat([predictions, *ensemble_predictions], ignore_index=True)
    return predictions, pd.DataFrame(weight_rows)


def fit_candidate_roster(
    weighted_rows: Any,
    cfg: RunConfig,
    dependencies: Mapping[str, Any],
    ode_gates: Any,
    cohorts: Any | None = None,
    measurements: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    predictions: list[Any] = []
    status_rows: list[dict[str, Any]] = []
    neural_details: dict[str, Any] = {"mlp": {}, "ode": {}}
    group_columns = ["cohort", "outcome", "origin_month"]
    for keys, task in weighted_rows.groupby(group_columns, sort=True):
        cohort_name, outcome, origin_month = keys
        task_key = f"{cohort_name}|{outcome}|origin{origin_month}"
        observed_training_count = int((task["split"].eq("train") & task["target_observed"]).sum())
        if observed_training_count < MIN_CELL_SIZE:
            for candidate_name, architecture in (
                ("population_change", "empirical_baseline"),
                ("persistence", "empirical_baseline"),
                ("regularized_spline", "ridge_cubic_spline"),
                ("histogram_gradient_boosting", "hist_gradient_boosting_quantile"),
                ("catboost_multi_quantile", "catboost_multi_quantile"),
                ("pytorch_quantile_mlp", "direct_horizon_mlp"),
                ("pytorch_ode_rnn", "context_conditioned_ode_rnn"),
            ):
                status_rows.append(
                    {
                        **dict(zip(group_columns, keys, strict=False)),
                        "candidate": candidate_name,
                        "architecture": architecture,
                        "status": "not_estimable",
                        "reason": f"only {observed_training_count} observed training targets",
                    }
                )
            continue
        candidates: list[tuple[str, str, Callable[[], Any], bool]] = [
            ("population_change", "empirical_baseline", lambda task=task: fit_population_baseline(task), True),
            ("persistence", "empirical_baseline", lambda task=task: fit_persistence_baseline(task), True),
            ("regularized_spline", "ridge_cubic_spline", lambda task=task: fit_spline_candidate(task, cfg), True),
        ]
        advanced_allowed = not cfg.smoke or origin_month == 0
        candidates.append(
            (
                "histogram_gradient_boosting",
                "hist_gradient_boosting_quantile",
                lambda task=task: fit_hgb_candidate(task, cfg),
                advanced_allowed,
            )
        )
        candidates.append(
            (
                "catboost_multi_quantile",
                "catboost_multi_quantile",
                lambda task=task: fit_catboost_candidate(task, cfg),
                advanced_allowed and bool(dependencies.get("catboost_importable")),
            )
        )
        for candidate_name, architecture, fit_function, applicable in candidates:
            if not applicable:
                reason = "reduced smoke mode" if advanced_allowed is False else "dependency unavailable"
                status_rows.append(
                    {
                        **dict(zip(group_columns, keys, strict=False)),
                        "candidate": candidate_name,
                        "architecture": architecture,
                        "status": "not_run",
                        "reason": reason,
                    }
                )
                continue
            started = time.perf_counter()
            try:
                candidate_predictions = fit_function()
                predictions.append(candidate_predictions)
                status_rows.append(
                    {
                        **dict(zip(group_columns, keys, strict=False)),
                        "candidate": candidate_name,
                        "architecture": architecture,
                        "status": "fitted",
                        "reason": "",
                        "training_seconds": time.perf_counter() - started,
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        **dict(zip(group_columns, keys, strict=False)),
                        "candidate": candidate_name,
                        "architecture": architecture,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "training_seconds": time.perf_counter() - started,
                    }
                )
        if advanced_allowed and dependencies.get("torch_importable"):
            started = time.perf_counter()
            try:
                mlp_predictions, details = fit_mlp_candidate(task, cfg)
                predictions.append(mlp_predictions)
                neural_details["mlp"][task_key] = details
                status_rows.append(
                    {
                        **dict(zip(group_columns, keys, strict=False)),
                        "candidate": "pytorch_quantile_mlp",
                        "architecture": "direct_horizon_mlp",
                        "status": "fitted",
                        "reason": "",
                        "training_seconds": time.perf_counter() - started,
                        "parameter_count": details["runs"][0]["parameter_count"],
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        **dict(zip(group_columns, keys, strict=False)),
                        "candidate": "pytorch_quantile_mlp",
                        "architecture": "direct_horizon_mlp",
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "training_seconds": time.perf_counter() - started,
                    }
                )
        else:
            status_rows.append(
                {
                    **dict(zip(group_columns, keys, strict=False)),
                    "candidate": "pytorch_quantile_mlp",
                    "architecture": "direct_horizon_mlp",
                    "status": "not_run",
                    "reason": "reduced smoke mode" if not advanced_allowed else "PyTorch unavailable",
                }
            )
        gate = ode_gates.loc[ode_gates["cohort"].eq(cohort_name) & ode_gates["outcome"].eq(outcome)]
        appropriate = bool(not gate.empty and gate.iloc[0]["appropriate"])
        # The baseline implementation is present, but fitting is only permitted after every
        # repeated-measure gate passes. Smoke mode always runs its numerical forward/backward test.
        status_rows.append(
            {
                **dict(zip(group_columns, keys, strict=False)),
                "candidate": "pytorch_ode_rnn",
                "architecture": "context_conditioned_ode_rnn",
                "status": "eligible_pending_fit" if appropriate else "not_appropriate",
                "reason": "" if appropriate else (str(gate.iloc[0]["failed_gates"]) if not gate.empty else "suitability gate unavailable"),
            }
        )
    for gate_row in ode_gates.loc[ode_gates["appropriate"]].itertuples(index=False):
        task_key = f"{gate_row.cohort}|{gate_row.outcome}"
        if cohorts is None or measurements is None:
            for status in status_rows:
                if status["cohort"] == gate_row.cohort and status["outcome"] == gate_row.outcome and status["candidate"] == "pytorch_ode_rnn":
                    status["status"] = "failed"
                    status["reason"] = "eligible task lacks cohort or exact-measurement input"
            continue
        task = weighted_rows.loc[
            weighted_rows["cohort"].eq(gate_row.cohort) & weighted_rows["outcome"].eq(gate_row.outcome)
        ].copy()
        started = time.perf_counter()
        try:
            ode_predictions, details = fit_ode_candidate(task, cohorts, measurements, cfg)
            predictions.append(ode_predictions)
            neural_details["ode"][task_key] = details
            for status in status_rows:
                if status["cohort"] == gate_row.cohort and status["outcome"] == gate_row.outcome and status["candidate"] == "pytorch_ode_rnn":
                    status["status"] = "fitted"
                    status["reason"] = ""
                    status["training_seconds"] = time.perf_counter() - started
                    status["parameter_count"] = details["runs"][0]["parameter_count"]
        except Exception as exc:
            for status in status_rows:
                if status["cohort"] == gate_row.cohort and status["outcome"] == gate_row.outcome and status["candidate"] == "pytorch_ode_rnn":
                    status["status"] = "failed"
                    status["reason"] = f"{type(exc).__name__}: {exc}"
                    status["training_seconds"] = time.perf_counter() - started
    if not predictions:
        raise RuntimeError("Every model candidate failed")
    combined = pd.concat(predictions, ignore_index=True)
    combined, ensemble_weights = add_ensemble_candidates(combined, cfg)
    return combined, pd.DataFrame(status_rows), {**neural_details, "ensemble_weights": ensemble_weights}


def finite_sample_quantile(values: Any, coverage: float) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    array = array[np.isfinite(array)]
    if not len(array):
        return 0.0
    rank = min(len(array) - 1, max(0, int(math.ceil((len(array) + 1) * coverage)) - 1))
    return float(array[rank])


def conformal_calibrate(predictions: Any) -> tuple[Any, Any]:
    calibrated = predictions.copy()
    corrections: list[dict[str, Any]] = []
    group_columns = ["cohort", "outcome", "origin_month", "target_month", "candidate"]
    interval_pairs = ((0, 6, 0.90), (1, 5, 0.80), (2, 4, 0.50))
    for keys, group in calibrated.groupby(group_columns, sort=True):
        calibration = group.loc[group["split"].eq("calibration") & group["target_observed"]]
        group_corrections: dict[tuple[int, int], float] = {}
        for lower_index, upper_index, coverage in interval_pairs:
            if len(calibration) < MIN_CELL_SIZE:
                correction = 0.0
                status = "insufficient_calibration_support"
            else:
                y = calibration["target_value"].to_numpy(float)
                lower = calibration[QUANTILE_COLUMNS[lower_index]].to_numpy(float)
                upper = calibration[QUANTILE_COLUMNS[upper_index]].to_numpy(float)
                scores = np.maximum(lower - y, y - upper)
                correction = max(0.0, finite_sample_quantile(scores, coverage))
                status = "calibrated"
            group_corrections[(lower_index, upper_index)] = correction
            corrections.append(
                {
                    **dict(zip(group_columns, keys, strict=False)),
                    "coverage": coverage,
                    "correction": correction,
                    "n_calibration": int(len(calibration)),
                    "status": status,
                }
            )
        indices = group.index
        values = calibrated.loc[indices, list(QUANTILE_COLUMNS)].to_numpy(dtype=float, copy=True)
        for (lower_index, upper_index), correction in group_corrections.items():
            values[:, lower_index] -= correction
            values[:, upper_index] += correction
        calibrated.loc[indices, list(QUANTILE_COLUMNS)] = rearrange_quantiles(values)
    return calibrated, pd.DataFrame(corrections)


def pit_from_quantiles(y_true: Any, matrix: Any) -> Any:
    y = np.asarray(y_true, dtype=float)
    predictions = rearrange_quantiles(matrix)
    output = np.empty(len(y), dtype=float)
    q = np.asarray(QUANTILES, dtype=float)
    for index, value in enumerate(y):
        row = predictions[index]
        if value <= row[0]:
            slope = q[0] / max(row[0] - (row[0] - max(abs(row[1] - row[0]), 1e-3)), 1e-6)
            output[index] = max(0.0, q[0] - slope * (row[0] - value))
        elif value >= row[-1]:
            width = max(abs(row[-1] - row[-2]), 1e-3)
            output[index] = min(1.0, q[-1] + (1 - q[-1]) * (value - row[-1]) / width)
        else:
            output[index] = float(np.interp(value, row, q))
    return np.clip(output, 0, 1)


def weighted_mean(values: Any, weights: Any) -> float:
    array = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    return float(np.sum(array * weight) / max(np.sum(weight), 1e-12))


def evaluate_predictions(predictions: Any, weight_diagnostics: Any) -> tuple[Any, Any]:
    rows: list[dict[str, Any]] = []
    pit_rows: list[dict[str, Any]] = []
    tests = predictions.loc[
        predictions["split"].isin(["temporal_test", "geographic_test"]) & predictions["target_observed"]
    ].copy()
    group_columns = ["cohort", "outcome", "origin_month", "target_month", "candidate", "split"]
    for keys, group in tests.groupby(group_columns, sort=True):
        y = group["target_value"].to_numpy(float)
        matrix = rearrange_quantiles(group[list(QUANTILE_COLUMNS)].to_numpy(float))
        weight = group["analysis_weight"].to_numpy(float)
        median = matrix[:, 3]
        errors = median - y
        pit = pit_from_quantiles(y, matrix)
        outcome = str(keys[1])
        plausible_low, plausible_high = PLAUSIBLE_RANGES[outcome]
        diagnostic = weight_diagnostics.loc[
            weight_diagnostics["cohort"].eq(keys[0])
            & weight_diagnostics["outcome"].eq(keys[1])
            & weight_diagnostics["origin_month"].eq(keys[2])
            & weight_diagnostics["target_month"].eq(keys[3])
        ]
        estimability = str(diagnostic.iloc[0]["status"]) if not diagnostic.empty else "weight_diagnostic_missing"
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=False)),
                "n": int(len(group)),
                "effective_sample_size": effective_sample_size(weight),
                "crps": quantile_crps(y, matrix, weight),
                "rmse": math.sqrt(weighted_mean(errors**2, weight)),
                "mae": weighted_mean(np.abs(errors), weight),
                "mean_signed_error": weighted_mean(errors, weight),
                "median_absolute_error": float(np.median(np.abs(errors))),
                "coverage_50": weighted_mean((y >= matrix[:, 2]) & (y <= matrix[:, 4]), weight),
                "coverage_80": weighted_mean((y >= matrix[:, 1]) & (y <= matrix[:, 5]), weight),
                "coverage_90": weighted_mean((y >= matrix[:, 0]) & (y <= matrix[:, 6]), weight),
                "width_50": weighted_mean(matrix[:, 4] - matrix[:, 2], weight),
                "width_80": weighted_mean(matrix[:, 5] - matrix[:, 1], weight),
                "width_90": weighted_mean(matrix[:, 6] - matrix[:, 0], weight),
                "quantile_loss": quantile_crps(y, matrix, weight) / 2.0,
                "pit_mean": weighted_mean(pit, weight),
                "pit_variance": weighted_mean((pit - weighted_mean(pit, weight)) ** 2, weight),
                "implausible_prediction_rate": float(np.mean((matrix < plausible_low) | (matrix > plausible_high))),
                "estimability": estimability,
            }
        )
        for patient_id, pit_value in zip(group["patient_id"], pit, strict=False):
            pit_rows.append(
                {
                    "patient_id": patient_id,
                    **dict(zip(group_columns, keys, strict=False)),
                    "pit": float(pit_value),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pit_rows)


def select_models(leaderboard: Any) -> Any:
    rows: list[dict[str, Any]] = []
    group_columns = ["cohort", "outcome", "origin_month"]
    for keys, group in leaderboard.groupby(group_columns, sort=True):
        viable = group.loc[group["n_validation"].ge(MIN_CELL_SIZE)].copy()
        if viable.empty:
            rows.append({**dict(zip(group_columns, keys, strict=False)), "selected_candidate": "not_estimable", "selection_reason": "fewer than 11 validation targets"})
            continue
        viable["coverage_penalty"] = (viable["coverage_80"] - 0.80).abs() + (viable["coverage_90"] - 0.90).abs()
        acceptable = viable.loc[(viable["coverage_80"] - 0.80).abs().le(0.05) & (viable["coverage_90"] - 0.90).abs().le(0.05)]
        pool = acceptable if not acceptable.empty else viable
        selected = pool.sort_values(
            ["validation_standardized_crps", "coverage_penalty", "validation_crps", "candidate"]
        ).iloc[0]
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=False)),
                "selected_candidate": selected["candidate"],
                "validation_crps": selected["validation_crps"],
                "validation_standardized_crps": selected["validation_standardized_crps"],
                "selection_reason": (
                    "lowest equal-horizon standardized validation CRPS among candidates passing interval calibration"
                    if not acceptable.empty
                    else "lowest equal-horizon standardized validation CRPS; no candidate passed both coverage gates"
                ),
            }
        )
    return pd.DataFrame(rows)


def development_iqr_by_task(weighted_rows: Any) -> Any:
    rows = []
    for keys, group in weighted_rows.loc[weighted_rows["split"].eq("train") & weighted_rows["target_observed"]].groupby(
        ["cohort", "outcome", "target_month"], sort=True
    ):
        q25, q75 = np.quantile(group["target_value"], [0.25, 0.75])
        rows.append(
            {
                "cohort": keys[0], "outcome": keys[1], "target_month": keys[2],
                "development_iqr": max(float(q75 - q25), 1e-8),
            }
        )
    return pd.DataFrame(rows)


def apply_success_gates(
    metrics: Any,
    selected: Any,
    iqr_table: Any,
    comparisons: Any | None = None,
) -> Any:
    rows: list[dict[str, Any]] = []
    merged = metrics.merge(iqr_table, on=["cohort", "outcome", "target_month"], how="left")
    merged["standardized_crps"] = merged["crps"] / merged["development_iqr"]
    comparison_frame = comparisons if comparisons is not None else pd.DataFrame()
    for selection in selected.itertuples(index=False):
        task = merged.loc[
            merged["cohort"].eq(selection.cohort)
            & merged["outcome"].eq(selection.outcome)
            & merged["origin_month"].eq(selection.origin_month)
        ]
        candidate = task.loc[task["candidate"].eq(selection.selected_candidate)]
        baselines = task.loc[task["candidate"].isin(["population_change", "persistence"])]
        guard_by_split: dict[str, bool] = {}
        for split, split_candidate in candidate.groupby("split", sort=True):
            improvements: list[float] = []
            split_baselines = baselines.loc[baselines["split"].eq(split)]
            for target_month, current_cell in split_candidate.groupby("target_month", sort=True):
                baseline_cell = split_baselines.loc[split_baselines["target_month"].eq(target_month)]
                if current_cell.empty or baseline_cell.empty:
                    continue
                current_value = float(current_cell.iloc[0]["standardized_crps"])
                baseline_value = float(baseline_cell["standardized_crps"].min())
                if math.isfinite(current_value) and math.isfinite(baseline_value) and baseline_value > 0:
                    improvements.append((baseline_value - current_value) / baseline_value)
            guard_by_split[str(split)] = bool(improvements) and min(improvements) >= -0.05
        for (target_month, split), current in candidate.groupby(["target_month", "split"], sort=True):
            baseline_cell = baselines.loc[baselines["target_month"].eq(target_month) & baselines["split"].eq(split)]
            current_row: Any | None = None
            strongest: Any | None = None
            gate_values: dict[str, bool] = {}
            if current.empty or baseline_cell.empty:
                status = "Not estimable"
                improvement = np.nan
                reasons = "matched baseline cell unavailable"
            else:
                current_row = current.iloc[0]
                strongest = baseline_cell.sort_values("standardized_crps").iloc[0]
                improvement = (strongest["standardized_crps"] - current_row["standardized_crps"]) / max(strongest["standardized_crps"], 1e-12)
                gate_values = {
                    "crps_improvement_at_least_10_pct": improvement >= 0.10,
                    "coverage_80_within_5_points": abs(current_row["coverage_80"] - 0.80) <= 0.05,
                    "coverage_90_within_5_points": abs(current_row["coverage_90"] - 0.90) <= 0.05,
                    "weight_gate": current_row["estimability"] == "estimable",
                    "minimum_cell": int(current_row["n"]) >= MIN_CELL_SIZE,
                    "no_horizon_more_than_5_pct_worse": guard_by_split.get(str(split), False),
                }
                failed = [name for name, passed in gate_values.items() if not passed]
                status = "Supported" if not failed else ("Promising but not yet validated" if len(failed) <= 2 else "Exploratory")
                reasons = "all available gates passed" if not failed else "failed: " + " | ".join(failed)

            comparison_row: Any | None = None
            if not comparison_frame.empty:
                matched_comparison = comparison_frame.loc[
                    comparison_frame["cohort"].eq(selection.cohort)
                    & comparison_frame["outcome"].eq(selection.outcome)
                    & comparison_frame["origin_month"].eq(selection.origin_month)
                    & comparison_frame["target_month"].eq(target_month)
                    & comparison_frame["split"].eq(split)
                ]
                if not matched_comparison.empty:
                    comparison_row = matched_comparison.iloc[0]
            rows.append(
                {
                    "cohort": selection.cohort,
                    "outcome": selection.outcome,
                    "origin_month": selection.origin_month,
                    "target_month": target_month,
                    "split": split,
                    "selected_candidate": selection.selected_candidate,
                    "relative_standardized_crps_improvement": improvement,
                    "n": int(current_row["n"]) if current_row is not None else 0,
                    "effective_sample_size": (
                        float(current_row["effective_sample_size"]) if current_row is not None else np.nan
                    ),
                    "selected_crps": float(current_row["crps"]) if current_row is not None else np.nan,
                    "baseline_crps": float(strongest["crps"]) if strongest is not None else np.nan,
                    "selected_rmse": float(current_row["rmse"]) if current_row is not None else np.nan,
                    "coverage_80": float(current_row["coverage_80"]) if current_row is not None else np.nan,
                    "coverage_90": float(current_row["coverage_90"]) if current_row is not None else np.nan,
                    "weight_gate_pass": gate_values.get("weight_gate", False),
                    "minimum_cell_pass": gate_values.get("minimum_cell", False),
                    "horizon_guard_pass": gate_values.get("no_horizon_more_than_5_pct_worse", False),
                    "paired_crps_difference": (
                        float(comparison_row["mean_crps_difference"]) if comparison_row is not None else np.nan
                    ),
                    "paired_crps_difference_low": (
                        float(comparison_row["difference_low"]) if comparison_row is not None else np.nan
                    ),
                    "paired_crps_difference_high": (
                        float(comparison_row["difference_high"]) if comparison_row is not None else np.nan
                    ),
                    "paired_fdr_q_value": (
                        float(comparison_row["fdr_q_value"]) if comparison_row is not None else np.nan
                    ),
                    "source_contract_pass": True,
                    "claim_status": status,
                    "gate_detail": reasons,
                }
            )
    return pd.DataFrame(rows)


def apply_source_claim_limit(gates: Any, metadata: Mapping[str, Any]) -> Any:
    if metadata.get("source_mode") != "cosmos_direct_wide_cohorts" or gates.empty:
        return gates
    limited = gates.copy()
    estimable = limited["claim_status"].ne("Not estimable")
    limited.loc[estimable, "claim_status"] = "Exploratory"
    limited["source_contract_pass"] = False
    suffix = "source limit: nominal wide-horizon timing and no fill-level coverage audit"
    limited["gate_detail"] = limited["gate_detail"].astype(str).map(
        lambda value: value + " | " + suffix if value else suffix
    )
    return limited


def apply_smoke_claim_limit(gates: Any, cfg: RunConfig) -> Any:
    if not cfg.smoke or gates.empty:
        return gates
    limited = gates.copy()
    estimable = limited["claim_status"].ne("Not estimable")
    limited.loc[estimable, "claim_status"] = "Exploratory"
    suffix = "run limit: bounded smoke sample with reduced tuning and bootstrap replication"
    limited["gate_detail"] = limited["gate_detail"].astype(str).map(
        lambda value: value + " | " + suffix if value else suffix
    )
    return limited


def row_crps(y_true: Any, matrix: Any) -> Any:
    y = np.asarray(y_true, dtype=float)[:, None]
    predictions = np.asarray(matrix, dtype=float)
    quantiles = np.asarray(QUANTILES, dtype=float)[None, :]
    error = y - predictions
    return 2.0 * np.mean(np.maximum(quantiles * error, (quantiles - 1.0) * error), axis=1)


def bootstrap_uncertainty(predictions: Any, selected: Any, cfg: RunConfig) -> tuple[Any, Any]:
    rng = np.random.default_rng(cfg.seed + 991)
    ci_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    selected_map = {
        (row.cohort, row.outcome, row.origin_month): row.selected_candidate
        for row in selected.itertuples(index=False)
    }
    tests = predictions.loc[
        predictions["split"].isin(["temporal_test", "geographic_test"])
        & predictions["target_observed"]
    ].copy()
    for keys, group in tests.groupby(["cohort", "outcome", "origin_month", "target_month", "split"], sort=True):
        selected_candidate = selected_map.get((keys[0], keys[1], keys[2]))
        candidates = [selected_candidate, "population_change", "persistence"]
        candidates = [item for item in dict.fromkeys(candidates) if item and item != "not_estimable"]
        candidate_data: dict[str, Any] = {}
        for candidate in candidates:
            frame = group.loc[group["candidate"].eq(candidate)].copy()
            if len(frame) >= MIN_CELL_SIZE:
                matrix = frame[list(QUANTILE_COLUMNS)].to_numpy(float)
                frame["row_crps"] = row_crps(frame["target_value"], matrix)
                frame["squared_error"] = (frame["target_value"].to_numpy(float) - matrix[:, 3]) ** 2
                frame["covered_80"] = (
                    (frame["target_value"].to_numpy(float) >= matrix[:, 1])
                    & (frame["target_value"].to_numpy(float) <= matrix[:, 5])
                ).astype(float)
                candidate_data[candidate] = frame
        for candidate, frame in candidate_data.items():
            patient_ids = np.asarray(sorted(frame["patient_id"].astype(str).unique()))
            crps_samples: list[float] = []
            rmse_samples: list[float] = []
            coverage_80_samples: list[float] = []
            for _ in range(cfg.bootstrap_replicates):
                sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
                counts = pd.Series(sampled).value_counts()
                merged = frame.merge(counts.rename("bootstrap_count"), left_on="patient_id", right_index=True)
                weights = merged["analysis_weight"].to_numpy(float) * merged["bootstrap_count"].to_numpy(float)
                crps_samples.append(weighted_mean(merged["row_crps"], weights))
                rmse_samples.append(math.sqrt(weighted_mean(merged["squared_error"], weights)))
                coverage_80_samples.append(weighted_mean(merged["covered_80"], weights))
            ci_rows.append(
                {
                    "cohort": keys[0], "outcome": keys[1], "origin_month": keys[2], "target_month": keys[3],
                    "split": keys[4], "candidate": candidate, "bootstrap_type": "patient",
                    "crps_low": float(np.quantile(crps_samples, 0.025)),
                    "crps_high": float(np.quantile(crps_samples, 0.975)),
                    "rmse_low": float(np.quantile(rmse_samples, 0.025)),
                    "rmse_high": float(np.quantile(rmse_samples, 0.975)),
                    "coverage_80_low": float(np.quantile(coverage_80_samples, 0.025)),
                    "coverage_80_high": float(np.quantile(coverage_80_samples, 0.975)),
                    "replicates": cfg.bootstrap_replicates,
                }
            )
            centers = np.asarray(sorted(frame["center_id"].astype(str).unique()))
            if len(centers) >= 2:
                center_crps: list[float] = []
                for _ in range(cfg.bootstrap_replicates):
                    sampled_centers = rng.choice(centers, size=len(centers), replace=True)
                    center_counts = pd.Series(sampled_centers).value_counts()
                    merged = frame.merge(center_counts.rename("bootstrap_count"), left_on="center_id", right_index=True)
                    weights = merged["analysis_weight"].to_numpy(float) * merged["bootstrap_count"].to_numpy(float)
                    center_crps.append(weighted_mean(merged["row_crps"], weights))
                ci_rows.append(
                    {
                        "cohort": keys[0], "outcome": keys[1], "origin_month": keys[2], "target_month": keys[3],
                        "split": keys[4], "candidate": candidate, "bootstrap_type": "center_clustered",
                        "crps_low": float(np.quantile(center_crps, 0.025)),
                        "crps_high": float(np.quantile(center_crps, 0.975)),
                        "rmse_low": np.nan, "rmse_high": np.nan,
                        "coverage_80_low": np.nan, "coverage_80_high": np.nan,
                        "replicates": cfg.bootstrap_replicates,
                    }
                )
        if selected_candidate in candidate_data:
            selected_frame = candidate_data[selected_candidate][["patient_id", "row_id", "row_crps"]].rename(columns={"row_crps": "selected_crps"})
            baseline_scores = []
            for baseline in ("population_change", "persistence"):
                if baseline in candidate_data:
                    baseline_frame = candidate_data[baseline][["row_id", "row_crps"]].rename(columns={"row_crps": f"{baseline}_crps"})
                    baseline_scores.append((baseline, baseline_frame))
            if baseline_scores:
                baseline_name, baseline_frame = min(
                    baseline_scores,
                    key=lambda item: float(item[1][f"{item[0]}_crps"].mean()),
                )
                paired = selected_frame.merge(baseline_frame, on="row_id", how="inner")
                differences = paired["selected_crps"] - paired[f"{baseline_name}_crps"]
                if len(differences):
                    patient_differences = paired.assign(difference=differences).groupby("patient_id")["difference"].mean()
                    boot = []
                    values = patient_differences.to_numpy(float)
                    for _ in range(cfg.bootstrap_replicates):
                        boot.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
                    comparison_rows.append(
                        {
                            "cohort": keys[0], "outcome": keys[1], "origin_month": keys[2], "target_month": keys[3],
                            "split": keys[4], "selected_candidate": selected_candidate, "baseline": baseline_name,
                            "mean_crps_difference": float(np.mean(values)),
                            "difference_low": float(np.quantile(boot, 0.025)),
                            "difference_high": float(np.quantile(boot, 0.975)),
                            "p_value": float(2 * min(np.mean(np.asarray(boot) <= 0), np.mean(np.asarray(boot) >= 0))),
                        }
                    )
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["fdr_q_value"] = benjamini_hochberg(comparisons["p_value"].to_numpy(float))
    return pd.DataFrame(ci_rows), comparisons


def benjamini_hochberg(p_values: Any) -> Any:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_index in range(len(values) - 1, -1, -1):
        rank = reverse_index + 1
        running = min(running, ranked[reverse_index] * len(values) / rank)
        adjusted[order[reverse_index]] = min(1.0, running)
    return adjusted


def weight_sensitivity_table(predictions: Any, selected: Any) -> Any:
    selected_map = {
        (row.cohort, row.outcome, row.origin_month): row.selected_candidate
        for row in selected.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    tests = predictions.loc[predictions["split"].isin(["temporal_test", "geographic_test"]) & predictions["target_observed"]]
    for keys, group in tests.groupby(["cohort", "outcome", "origin_month", "target_month", "split"], sort=True):
        candidate = selected_map.get((keys[0], keys[1], keys[2]))
        frame = group.loc[group["candidate"].eq(candidate)].copy()
        if frame.empty:
            continue
        y = frame["target_value"].to_numpy(float)
        matrix = frame[list(QUANTILE_COLUMNS)].to_numpy(float)
        raw = frame["analysis_weight"].to_numpy(float)
        schemes = {"primary_1_99": raw, "unweighted": np.ones(len(frame), dtype=float)}
        for low, high in ALTERNATE_WEIGHT_TRUNCATION:
            name = "untruncated" if low == 0 else "alternate_0.5_99.5"
            if low == 0:
                schemes[name] = raw
            else:
                lower, upper = np.quantile(raw, [low, high])
                schemes[name] = np.clip(raw, lower, upper)
        for scheme, weights in schemes.items():
            rows.append(
                {
                    "cohort": keys[0], "outcome": keys[1], "origin_month": keys[2], "target_month": keys[3],
                    "split": keys[4], "scheme": scheme, "n": len(frame),
                    "effective_sample_size": effective_sample_size(weights),
                    "crps": quantile_crps(y, matrix, weights),
                    "coverage_80": weighted_mean((y >= matrix[:, 1]) & (y <= matrix[:, 5]), weights),
                    "coverage_90": weighted_mean((y >= matrix[:, 0]) & (y <= matrix[:, 6]), weights),
                }
            )
    return pd.DataFrame(rows)


def gap_rule_sensitivity(bundle: DataBundle) -> Any:
    medications = bundle.medications
    if bundle.metadata.get("source_mode") == "cosmos_direct_wide_cohorts" and "source_cohort" in medications:
        medications = medications.loc[medications["source_cohort"].eq("incretin")]
    absolute_records, _ = medication_frame_to_coverage(medications)
    patient_sets: dict[int, set[str]] = {}
    rows: list[dict[str, Any]] = []
    for gap in GAP_SENSITIVITIES:
        episodes, _ = reconstruct_coverage_episodes(absolute_records, gap_rule_days=gap)
        qualifiers = {episode.patient_id for episode in episodes if episode.qualifies_183}
        patient_sets[gap] = qualifiers
        rows.append(
            {
                "gap_rule_days": gap,
                "qualifying_patients": len(qualifiers),
                "reclassified_vs_primary": 0,
            }
        )
    primary = patient_sets[PRIMARY_GAP_DAYS]
    for row in rows:
        gap = int(row["gap_rule_days"])
        row["reclassified_vs_primary"] = len(primary.symmetric_difference(patient_sets[gap]))
    return pd.DataFrame(rows)


def nearest_psd_correlation(matrix: Any) -> Any:
    values = np.asarray(matrix, dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    values = (values + values.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    eigenvalues = np.clip(eigenvalues, 1e-6, None)
    rebuilt = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(rebuilt))
    return rebuilt / np.outer(scale, scale)


def gaussian_cdf(values: Any) -> Any:
    array = np.asarray(values, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(array / math.sqrt(2.0)))


def estimate_residual_correlation(predictions: Any, candidate: str, cohort: str, outcome: str, origin_month: int) -> tuple[list[int], Any]:
    calibration = predictions.loc[
        predictions["candidate"].eq(candidate)
        & predictions["cohort"].eq(cohort)
        & predictions["outcome"].eq(outcome)
        & predictions["origin_month"].eq(origin_month)
        & predictions["split"].eq("calibration")
        & predictions["target_observed"]
    ].copy()
    calibration["residual"] = calibration["target_value"] - calibration["q50"]
    pivot = calibration.pivot_table(index="patient_id", columns="target_month", values="residual", aggfunc="first")
    horizons = [int(item) for item in pivot.columns]
    if len(horizons) < 2:
        return horizons, np.eye(max(1, len(horizons)))
    rank = pivot.rank(axis=0, pct=True)
    correlation = rank.corr(min_periods=MIN_CELL_SIZE).to_numpy(float)
    return horizons, nearest_psd_correlation(correlation)


def inverse_quantile(values: Any, probabilities: Any) -> Any:
    quantile_values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    return np.interp(probabilities, np.asarray(QUANTILES), quantile_values, left=quantile_values[0], right=quantile_values[-1])


def build_synthetic_trajectory_examples(predictions: Any, selected: Any, cfg: RunConfig) -> tuple[Any, Any]:
    rng = np.random.default_rng(cfg.seed + 404)
    example_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for selection in selected.loc[selected["origin_month"].eq(0)].itertuples(index=False):
        if selection.selected_candidate == "not_estimable":
            continue
        task = predictions.loc[
            predictions["cohort"].eq(selection.cohort)
            & predictions["outcome"].eq(selection.outcome)
            & predictions["origin_month"].eq(0)
            & predictions["candidate"].eq(selection.selected_candidate)
            & predictions["split"].eq("calibration")
        ].copy()
        if task.empty:
            continue
        horizons, correlation = estimate_residual_correlation(
            predictions, selection.selected_candidate, selection.cohort, selection.outcome, 0
        )
        supported_horizons = sorted(set(int(item) for item in task["target_month"]).intersection(horizons))
        if not supported_horizons:
            supported_horizons = sorted(int(item) for item in task["target_month"].unique())
            correlation = np.eye(len(supported_horizons))
        representative = task.groupby("target_month")[list(QUANTILE_COLUMNS)].median().reindex(supported_horizons)
        if representative.isna().all(axis=None):
            continue
        normal = rng.multivariate_normal(np.zeros(len(supported_horizons)), correlation, size=cfg.trajectory_draws)
        uniforms = gaussian_cdf(normal)
        draws = np.column_stack(
            [inverse_quantile(representative.loc[horizon].to_numpy(float), uniforms[:, index]) for index, horizon in enumerate(supported_horizons)]
        )
        for horizon_index, horizon in enumerate(supported_horizons):
            values = draws[:, horizon_index]
            example_rows.append(
                {
                    "example_id": f"synthetic_{selection.cohort}_{selection.outcome}",
                    "cohort": selection.cohort,
                    "outcome": selection.outcome,
                    "target_month": horizon,
                    "q05": float(np.quantile(values, 0.05)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q50": float(np.quantile(values, 0.50)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q95": float(np.quantile(values, 0.95)),
                    "draw_count": cfg.trajectory_draws,
                    "label": "Fully synthetic example; conditional model projection; not an individual treatment effect",
                }
            )
        if draws.shape[1] >= 2:
            jumps = np.diff(draws, axis=1)
            threshold = 8.0 if selection.outcome == "bmi" else 2.0
            score_rows.append(
                {
                    "cohort": selection.cohort,
                    "outcome": selection.outcome,
                    "candidate": selection.selected_candidate,
                    "energy_score_proxy": float(np.mean(np.linalg.norm(draws - np.median(draws, axis=0), axis=1))),
                    "variogram_score_proxy": float(np.mean(np.abs(jumps))),
                    "implausible_jump_rate": float(np.mean(np.abs(jumps) > threshold)),
                    "direction_change_rate": float(np.mean(np.sign(jumps[:, 1:]) != np.sign(jumps[:, :-1]))) if jumps.shape[1] > 1 else 0.0,
                }
            )
    return pd.DataFrame(example_rows), pd.DataFrame(score_rows)


# ======================================================================================
# 9. Aggregate auditing and the figure-only collaborator output
# ======================================================================================


PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "ink": "#24303F",
    "muted": "#6B7280",
    "grid": "#D8DEE8",
    "paper": "#FBFCFE",
}

PAGE_FILES = (
    "00_executive_summary.png",
    "01_run_identity_and_status.png",
    "02_cohort_funnels.png",
    "03_exposure_continuity_and_censoring.png",
    "04_baseline_composition.png",
    "05_followup_and_target_support.png",
    "06_measurement_quality.png",
    "07_split_and_leakage_audit.png",
    "08_model_selection.png",
    "09_continuous_time_model_diagnostics.png",
    "10_surgical_bmi_performance.png",
    "11_surgical_hba1c_performance.png",
    "12_incretin_bmi_performance.png",
    "13_incretin_hba1c_performance.png",
    "14_distributional_calibration.png",
    "15_transportability_and_subgroups.png",
    "16_censoring_and_persistence_sensitivities.png",
    "17_conditional_trajectory_examples.png",
    "18_gates_limitations_and_conclusion.png",
)


def display_count(value: Any, threshold: int = MIN_CELL_SIZE) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return "Not estimable"
    return "<11" if 0 < numeric < threshold else f"{numeric:,}"


def mask_small_cell_metrics(
    frame: Any,
    metric_columns: Sequence[str],
    count_column: str = "n",
) -> Any:
    """Blank plotted or summarized metrics when their patient denominator is under 11."""
    masked = frame.copy()
    if masked.empty or count_column not in masked:
        return masked
    counts = pd.to_numeric(masked[count_column], errors="coerce")
    small = counts.gt(0) & counts.lt(MIN_CELL_SIZE)
    available_metrics = [column for column in metric_columns if column in masked]
    if available_metrics:
        masked.loc[small, available_metrics] = np.nan
    return masked


def baseline_composition_table(cohorts: Any) -> Any:
    rows: list[dict[str, Any]] = []
    for keys, group in cohorts.groupby(["cohort", "treatment"], sort=True):
        n = group["patient_id"].nunique()
        rows.append(
            {
                "cohort": keys[0],
                "treatment": keys[1],
                "n": int(n),
                "age_mean": float(group["age_at_index"].mean()),
                "baseline_bmi_mean": float(group["baseline_bmi"].mean()),
                "baseline_hba1c_mean": float(group["baseline_hba1c"].mean()),
                "female_pct": 100.0 * float(group["sex"].astype(str).str.lower().eq("female").mean()),
                "diabetes_pct": 100.0 * float(pd.to_numeric(group["diabetes_flag"], errors="coerce").eq(1).mean()),
                "index_year_median": float(group["index_date"].dt.year.median()),
                "centers": int(group["center_id"].nunique()),
            }
        )
    result = pd.DataFrame(rows)
    return suppress_small_cells(result, ["n"])


def target_support_table(weighted_rows: Any) -> Any:
    rows = []
    for keys, group in weighted_rows.groupby(["cohort", "outcome", "origin_month", "target_month"], sort=True):
        counts = group["support_status"].value_counts()
        rows.append(
            {
                "cohort": keys[0], "outcome": keys[1], "origin_month": keys[2], "target_month": keys[3],
                "eligible": int(len(group)),
                "administratively_mature": int(group["administratively_mature"].sum()),
                "mature_with_target": int(counts.get("mature_with_target", 0)),
                "mature_without_target": int(counts.get("mature_without_target", 0)),
                "censored": int(counts.get("treatment_or_surgery_censored", 0)),
                "administratively_immature": int(counts.get("administratively_immature", 0)),
                "target_availability_pct": 100.0 * float(group["target_observed"].mean()),
            }
        )
    return pd.DataFrame(rows)


def measurement_quality_table(quality: Any, normalized: Any) -> Any:
    grouped = quality.groupby(["kind", "unit", "reason", "valid"], dropna=False).size().rename("n").reset_index()
    duplicate = normalized.groupby("outcome").agg(
        valid_measurements=("value", "size"),
        duplicate_days=("duplicate_day", "sum"),
        derived_values=("method", lambda values: int((values == "derived_weight_height").sum())),
    ).reset_index()
    return {"reasons": suppress_small_cells(grouped, ["n"]), "duplicates": duplicate}


def exposure_summary(exposure: Any, medication_audit: Any) -> dict[str, Any]:
    classification = exposure.groupby(["cohort", "classification"], dropna=False).size().rename("n").reset_index()
    censoring = exposure.groupby("cohort").agg(
        n=("patient_id", "nunique"),
        censored=("treatment_censor_day", lambda values: int(pd.to_numeric(values, errors="coerce").notna().sum())),
        day_zero=("treatment_censor_day", lambda values: int(pd.to_numeric(values, errors="coerce").eq(0).sum())),
        median_censor_day=("treatment_censor_day", "median"),
    ).reset_index()
    sources = medication_audit.groupby(["source_type", "accepted", "reason"], dropna=False).size().rename("n").reset_index()
    pdc = exposure.loc[exposure["cohort"].eq("incretin"), [column for column in ("pdc_183", "maximum_gap_days", "switch_count") if column in exposure]].copy()
    return {
        "classification": suppress_small_cells(classification, ["n"]),
        "censoring": censoring,
        "sources": suppress_small_cells(sources, ["n"]),
        "continuity": pdc.describe(percentiles=[0.25, 0.5, 0.75]).reset_index() if not pdc.empty else pd.DataFrame(),
    }


def aggregate_pit_histograms(pit_values: Any, selected: Any) -> Any:
    selected_map = {
        (row.cohort, row.outcome, row.origin_month): row.selected_candidate
        for row in selected.itertuples(index=False)
    }
    rows = []
    bins = np.linspace(0, 1, 11)
    for keys, group in pit_values.groupby(["cohort", "outcome", "origin_month", "candidate", "split"], sort=True):
        if selected_map.get((keys[0], keys[1], keys[2])) != keys[3]:
            continue
        counts, edges = np.histogram(group["pit"], bins=bins)
        for index, count in enumerate(counts):
            rows.append(
                {
                    "cohort": keys[0], "outcome": keys[1], "origin_month": keys[2],
                    "candidate": keys[3], "split": keys[4], "bin_left": edges[index],
                    "bin_right": edges[index + 1], "n": int(count),
                }
            )
    return pd.DataFrame(rows)


def subgroup_performance(predictions: Any, selected: Any) -> Any:
    selected_map = {
        (row.cohort, row.outcome, row.origin_month): row.selected_candidate
        for row in selected.itertuples(index=False)
    }
    tests = predictions.loc[
        predictions["split"].isin(["temporal_test", "geographic_test"])
        & predictions["target_observed"]
        & predictions["origin_month"].eq(0)
    ].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in tests.groupby(["cohort", "outcome", "target_month", "split"], sort=True):
        candidate = selected_map.get((keys[0], keys[1], 0))
        group = group.loc[group["candidate"].eq(candidate)].copy()
        if group.empty:
            continue
        group["baseline_group"] = pd.cut(group["prediction_reference_value"], 3, labels=["lower", "middle", "higher"], duplicates="drop").astype("string")
        subgroup_specs = {
            "treatment": group["treatment"].astype(str),
            "baseline_group": group["baseline_group"].astype(str),
        }
        for subgroup_name, subgroup_values in subgroup_specs.items():
            group["subgroup_value"] = subgroup_values
            for value, cell in group.groupby("subgroup_value", sort=True):
                n = len(cell)
                if n < MIN_CELL_SIZE:
                    rows.append(
                        {
                            "cohort": keys[0], "outcome": keys[1], "target_month": keys[2], "split": keys[3],
                            "subgroup": subgroup_name, "value": value, "n": n, "crps": np.nan,
                            "coverage_80": np.nan, "suppressed": True,
                        }
                    )
                    continue
                matrix = cell[list(QUANTILE_COLUMNS)].to_numpy(float)
                y = cell["target_value"].to_numpy(float)
                weight = cell["analysis_weight"].to_numpy(float)
                rows.append(
                    {
                        "cohort": keys[0], "outcome": keys[1], "target_month": keys[2], "split": keys[3],
                        "subgroup": subgroup_name, "value": value, "n": n,
                        "crps": quantile_crps(y, matrix, weight),
                        "coverage_80": weighted_mean((y >= matrix[:, 1]) & (y <= matrix[:, 5]), weight),
                        "suppressed": False,
                    }
                )
    return pd.DataFrame(rows)


def blind_center_summary(cohorts: Any, split_metadata: Mapping[str, Any], fingerprint: str) -> Any:
    centers = sorted(str(item) for item in cohorts["center_id"].unique())
    order = sorted(centers, key=lambda value: digest({"fingerprint": fingerprint, "center": value}))
    if not split_metadata.get("center_validation_available", True):
        mapping = {center: "Center unavailable" for center in order}
    else:
        mapping = {center: f"Center {index + 1:02d}" for index, center in enumerate(order)}
    table = cohorts.drop_duplicates(["patient_id", "center_id"]).copy()
    table["center_blind"] = table["center_id"].astype(str).map(mapping)
    table["heldout"] = table["center_id"].astype(str).isin(split_metadata.get("heldout_centers", []))
    result = table.groupby(["center_blind", "heldout"]).size().rename("n").reset_index()
    return suppress_small_cells(result, ["n"])


def build_figure_data(
    context: RunContext,
    dependencies: Mapping[str, Any],
    bundle: DataBundle,
    cohort_artifacts: Mapping[str, Any],
    cohorts: Any,
    split_metadata: Mapping[str, Any],
    leakage: Any,
    weighted_rows: Any,
    weight_diagnostics: Any,
    predictions: Any,
    model_status: Any,
    neural_details: Mapping[str, Any],
    leaderboard: Any,
    selected: Any,
    calibration: Any,
    metrics: Any,
    iqr: Any,
    pit_values: Any,
    bootstrap_ci: Any,
    comparisons: Any,
    gates: Any,
    ode_gates: Any,
    sensitivity: Any,
    gap_sensitivity: Any,
    examples: Any,
    joint_scores: Any,
) -> dict[str, Any]:
    index_range = (
        cohorts["index_date"].min().date().isoformat(),
        cohorts["index_date"].max().date().isoformat(),
    )
    baseline_table = baseline_composition_table(cohorts)
    if not split_metadata.get("center_validation_available", True):
        baseline_table["centers"] = "Not available"
    return {
        "identity": {
            "study_version": STUDY_VERSION,
            "fingerprint": context.fingerprint,
            "script_sha256": context.fingerprint_payload["script_sha256"],
            "query_fingerprint": bundle.metadata.get("query_fingerprint", "unknown"),
            "schema_fingerprint": bundle.metadata.get("schema_fingerprint", "unknown"),
            "generated_utc": utc_now(),
            "source_mode": bundle.metadata.get("source_mode", "unknown"),
            "cohort_date_range": index_range,
            "status": "completed",
            "run_mode": context.cfg.mode,
            "smoke_query_limit": context.cfg.smoke_query_limit if context.cfg.smoke else None,
            "dependencies": dict(dependencies),
            "preflight": bundle.metadata.get("preflight", {}),
            "measurement_timing": bundle.metadata.get("measurement_timing", "exact_day"),
            "medication_coverage_semantics": bundle.metadata.get(
                "medication_coverage_semantics", "audited_raw_events"
            ),
            "center_validation_available": bool(
                bundle.metadata.get("center_validation_available", True)
            ),
            "source_limitations": list(bundle.metadata.get("limitations", [])),
        },
        "funnel": cohort_artifacts["funnel"],
        "cohort_counts": (
            cohorts.drop_duplicates("patient_id")
            .groupby("cohort")
            .size()
            .rename("n")
            .reset_index()
        ),
        "exposure": exposure_summary(cohort_artifacts["exposure"], cohort_artifacts["medication_audit"]),
        "baseline": baseline_table,
        "support": target_support_table(weighted_rows),
        "measurement_quality": measurement_quality_table(cohort_artifacts["measurement_quality"], cohort_artifacts["measurements"]),
        "split": {
            "metadata": dict(split_metadata),
            "counts": cohorts.drop_duplicates("patient_id")["split"].value_counts().rename_axis("split").reset_index(name="n"),
            "centers": blind_center_summary(cohorts, split_metadata, context.fingerprint),
            "leakage": leakage,
        },
        "weight_diagnostics": weight_diagnostics,
        "model_status": model_status,
        "leaderboard": leaderboard,
        "selected": selected,
        "calibration": calibration,
        "metrics": metrics,
        "iqr": iqr,
        "pit_histograms": aggregate_pit_histograms(pit_values, selected),
        "bootstrap_ci": bootstrap_ci,
        "comparisons": comparisons,
        "subgroups": subgroup_performance(predictions, selected),
        "gates": gates,
        "ode_gates": ode_gates,
        "neural_details": dict(neural_details),
        "sensitivity": sensitivity,
        "gap_sensitivity": gap_sensitivity,
        "examples": examples,
        "joint_scores": joint_scores,
    }


def configure_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": PALETTE["grid"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": "white",
            "savefig.facecolor": PALETTE["paper"],
        }
    )


def new_page(number: int, title: str, subtitle: str) -> Any:
    figure = plt.figure(figsize=(11, 8.5), constrained_layout=False)
    figure.patch.set_facecolor(PALETTE["paper"])
    figure.text(0.055, 0.947, f"{number:02d}", fontsize=22, fontweight="bold", color=PALETTE["blue"], va="top")
    figure.text(0.115, 0.947, title, fontsize=17, fontweight="bold", color=PALETTE["ink"], va="top")
    figure.text(0.115, 0.915, subtitle, fontsize=9.5, color=PALETTE["muted"], va="top")
    figure.lines.append(plt.Line2D([0.055, 0.945], [0.893, 0.893], transform=figure.transFigure, color=PALETTE["grid"], lw=1.0))
    figure.text(0.055, 0.025, "Aggregate, disclosure-controlled output | Cells n < 11 suppressed | Noncausal prognostic study", fontsize=7.5, color=PALETTE["muted"])
    return figure


def wide_source_limited(data: Mapping[str, Any]) -> bool:
    return data.get("identity", {}).get("preflight", {}).get("strict_raw_event_contract") is False


def run_badge(data: Mapping[str, Any]) -> tuple[str, str]:
    identity = data.get("identity", {})
    mode = str(identity.get("run_mode", "unknown"))
    if mode == "smoke":
        return "SMOKE - NON-INFERENTIAL", PALETTE["red"]
    if mode == "preflight-only":
        return "PREFLIGHT ONLY", PALETTE["blue"]
    if wide_source_limited(data):
        return "FULL WIDE-SOURCE - EXPLORATORY", PALETTE["orange"]
    return "FULL RAW-EVENT RUN", PALETTE["green"]


def add_run_provenance(figure: Any, data: Mapping[str, Any]) -> None:
    label, color = run_badge(data)
    figure.text(
        0.945,
        0.025,
        label,
        ha="right",
        fontsize=8.2,
        fontweight="bold",
        color=color,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": color, "alpha": 0.95},
    )


def source_aware_text(data: Mapping[str, Any], value: str) -> str:
    if not wide_source_limited(data):
        return value
    replacements = (
        ("six-month incretin continuers", "patients with a reported ≥183-day incretin interval"),
        ("six-month continuers", "reported ≥183-day interval cohort"),
        ("primary six-month continuers", "primary reported ≥183-day interval cohort"),
        ("no new user six month continuation episode", "no qualifying reported ≥183-day interval"),
        ("patients with accepted exposure", "patients with an accepted reported interval"),
        ("six month continuer", "reported ≥183-day interval"),
        ("six-month persistence", "reported interval duration"),
    )
    result = str(value)
    for original, replacement in replacements:
        result = result.replace(original, replacement)
        result = result.replace(original.replace("-", "_").replace(" ", "_"), replacement)
    return result


def executive_comparison_table(data: Mapping[str, Any]) -> Any:
    comparisons = data.get("comparisons", pd.DataFrame())
    if comparisons.empty:
        return comparisons
    baseline_metrics = selected_only_metric_frame(data)
    baseline_metrics = baseline_metrics.loc[
        baseline_metrics["origin_month"].eq(0)
        & baseline_metrics["split"].eq("temporal_test")
    ][["cohort", "outcome", "origin_month", "target_month", "split", "n"]]
    summary = comparisons.loc[
        comparisons["origin_month"].eq(0)
        & comparisons["split"].eq("temporal_test")
    ].merge(
        baseline_metrics,
        on=["cohort", "outcome", "origin_month", "target_month", "split"],
        how="left",
        validate="one_to_one",
    )
    summary["95% CI"] = summary.apply(
        lambda row: f"{row['difference_low']:.3f} to {row['difference_high']:.3f}",
        axis=1,
    )
    return summary[
        [
            "cohort",
            "outcome",
            "target_month",
            "n",
            "baseline",
            "mean_crps_difference",
            "95% CI",
            "fdr_q_value",
        ]
    ].sort_values(["cohort", "outcome", "target_month"])


def render_page_00(data: Mapping[str, Any]) -> Any:
    identity = data["identity"]
    badge, badge_color = run_badge(data)
    figure = new_page(
        0,
        "Executive summary",
        "Screenshot-ready run provenance, cohort yield, matched performance, limitations, and permitted interpretation",
    )
    figure.text(
        0.50,
        0.855,
        badge,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=badge_color,
    )
    left = figure.add_axes([0.06, 0.54, 0.42, 0.22])
    right = figure.add_axes([0.54, 0.54, 0.40, 0.22])
    middle = figure.add_axes([0.06, 0.24, 0.88, 0.22])
    bottom = figure.add_axes([0.06, 0.07, 0.88, 0.11])

    panel_label(left, "A", "Source coverage")
    preflight = identity.get("preflight", {})
    sampled = preflight.get("source_unique_patient_counts", {})
    totals = preflight.get("source_total_unique_patient_counts", sampled)
    source_rows = []
    if sampled or totals:
        for source in sorted(set(sampled) | set(totals)):
            sampled_value = int(sampled.get(source, 0))
            total_value = int(totals.get(source, sampled_value))
            source_rows.append(
                {
                    "source": source,
                    "queried patients": sampled_value,
                    "available patients": total_value,
                    "fraction": sampled_value / total_value if total_value else np.nan,
                }
            )
        source_columns = ["source", "queried patients", "available patients", "fraction"]
    else:
        for source, rows in preflight.get("row_counts", {}).items():
            source_rows.append({"source": source, "queried rows": int(rows)})
        source_columns = ["source", "queried rows"]
    draw_compact_table(left, pd.DataFrame(source_rows), source_columns, max_rows=6)

    panel_label(right, "B", "Cohorts and claim classification")
    cohort_counts = data.get("cohort_counts", pd.DataFrame()).copy()
    if not cohort_counts.empty:
        cohort_counts["category"] = "eligible cohort"
        cohort_counts = cohort_counts.rename(columns={"cohort": "item"})
    gates = data.get("gates", pd.DataFrame())
    status_counts = (
        gates["claim_status"].value_counts().rename_axis("item").reset_index(name="n")
        if not gates.empty else pd.DataFrame(columns=["item", "n"])
    )
    if not status_counts.empty:
        status_counts["category"] = "decision cells"
    combined = pd.concat(
        [
            cohort_counts[["category", "item", "n"]] if not cohort_counts.empty else cohort_counts,
            status_counts[["category", "item", "n"]] if not status_counts.empty else status_counts,
        ],
        ignore_index=True,
    )
    draw_compact_table(right, combined, ["category", "item", "n"], max_rows=8)

    panel_label(middle, "C", "Baseline-origin paired CRPS difference versus strongest simple baseline")
    comparisons = executive_comparison_table(data)
    draw_compact_table(middle, comparisons, list(comparisons.columns), max_rows=9)

    panel_label(bottom, "D", "Permitted interpretation")
    bottom.axis("off")
    if identity.get("run_mode") == "smoke":
        conclusion = (
            "This bounded smoke run validates the end-to-end pipeline only. Reduced tuning, bootstrap replication, and "
            "sample size make every performance result non-inferential. Run the same fingerprint without --smoke before "
            "interpreting any metric."
        )
        if wide_source_limited(data):
            conclusion += (
                " The wide source additionally uses nominal horizons and reported start-end intervals and lacks geographic validation."
            )
    elif wide_source_limited(data):
        conclusion = (
            "Exploratory temporal prediction results only. Follow-up values use nominal wide-column horizons; "
            "reported medication start-end intervals do not establish fill-level adherence; administrative opportunity "
            "is operationalized; and unavailable center identity prevents geographic validation. Do not claim exact-time "
            "trajectories, confirmed persistence, five-year performance without supported cells, treatment effects, or transportability."
        )
    else:
        conclusion = (
            "Claims remain prognostic, horizon-specific, treatment-policy explicit, and noncausal. Supported cells still "
            "require the displayed calibration, weighting, uncertainty, and transportability gates."
        )
    bottom.text(0.01, 0.82, textwrap.fill(conclusion, width=145), va="top", fontsize=8.7, color=PALETTE["ink"])
    return figure


def panel_label(axis: Any, label: str, title: str) -> None:
    axis.text(-0.045, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", color=PALETTE["blue"], va="bottom")
    axis.set_title(title, loc="left", fontweight="bold", color=PALETTE["ink"], pad=8)


def empty_panel(axis: Any, message: str = "Not estimable") -> None:
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color=PALETTE["muted"], wrap=True)


def draw_compact_table(axis: Any, frame: Any, columns: Sequence[str], labels: Sequence[str] | None = None, max_rows: int = 12) -> None:
    axis.axis("off")
    if frame is None or frame.empty:
        axis.text(0.5, 0.5, "Not estimable", ha="center", va="center", color=PALETTE["muted"])
        return
    source_display = frame.head(max_rows).copy()
    display = source_display.loc[:, [column for column in columns if column in source_display]].copy()
    small_cell_mask = pd.Series(False, index=source_display.index)
    denominator_columns = ("n", "n_calibration", "eligible", "qualifying_patients", "development_patients")
    small_count_masks: dict[str, Any] = {}
    for count_column in denominator_columns:
        if count_column in source_display:
            patient_counts = pd.to_numeric(source_display[count_column], errors="coerce")
            small_count_masks[count_column] = patient_counts.gt(0) & patient_counts.lt(MIN_CELL_SIZE)
            small_cell_mask |= small_count_masks[count_column]
    if "small_cell_suppressed" in source_display:
        small_cell_mask |= source_display["small_cell_suppressed"].fillna(False).astype(bool)
    structural_numeric_columns = {"coverage", "gap_rule_days", "origin_month", "target_month"}
    if small_cell_mask.any():
        for column in display.columns:
            if column in denominator_columns:
                display[column] = display[column].astype(object)
                display.loc[small_cell_mask, column] = np.nan
                own_small_count = small_count_masks.get(column, pd.Series(False, index=source_display.index))
                display.loc[own_small_count, column] = "<11"
                if column == "n" and "small_cell_suppressed" in source_display:
                    display.loc[source_display["small_cell_suppressed"].fillna(False).astype(bool), column] = "<11"
            elif column == "suppressed":
                display[column] = display[column].astype(object)
                display.loc[small_cell_mask, column] = True
            elif pd.api.types.is_bool_dtype(display[column]):
                display[column] = display[column].astype(object)
                display.loc[small_cell_mask, column] = np.nan
            elif (
                column not in structural_numeric_columns
                and pd.api.types.is_numeric_dtype(display[column])
            ):
                display.loc[small_cell_mask, column] = np.nan
        if "estimability" in display:
            display.loc[small_cell_mask, "estimability"] = "Suppressed"
    integer_columns = {
        "n",
        "accepted_medication_records",
        "administratively_immature",
        "administratively_mature",
        "censored",
        "centers",
        "day_zero",
        "development_patients",
        "eligible",
        "mature_with_target",
        "mature_without_target",
        "median_censor_day",
        "n_calibration",
        "qualifying_patients",
        "reclassified_vs_primary",
        "valid_measurements",
    }
    disclosure_count_columns = integer_columns.difference({"centers", "median_censor_day"})
    pass_fail_columns = {"appropriate", "passed"}
    value_aliases = {
        "alternate_0.5_99.5": "Alt. 0.5%-99.5%",
        "catboost_multiquantile": "CatBoost",
        "catboost_multi_quantile": "CatBoost",
        "geographic_test": "Geographic test",
        "hist_gradient_boosting": "Histogram boosting",
        "histogram_gradient_boosting": "Histogram boosting",
        "noncrossing_mlp": "Noncrossing MLP",
        "not_estimable_weight_gate": "Weight gate failed",
        "population_change": "Population change",
        "primary_1_99": "Primary 1%-99%",
        "pytorch_ode_rnn": "ODE-RNN",
        "pytorch_quantile_mlp": "Quantile MLP",
        "regularized_spline": "Regularized spline",
        "spline_ridge": "Spline ridge",
        "temporal_test": "Temporal test",
        "validation_weighted_ensemble": "Weighted ensemble",
    }
    for column in display.columns:
        def format_value(value: Any) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, (bool, np.bool_)):
                if column in pass_fail_columns:
                    return "PASS" if bool(value) else "FAIL"
                return "Yes" if bool(value) else "No"
            if column in disclosure_count_columns and isinstance(value, (int, float, np.integer, np.floating)):
                return display_count(value)
            if column in integer_columns and isinstance(value, (int, float, np.integer, np.floating)):
                return f"{int(round(float(value))):,}"
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.3f}"
            prose_columns = {"assertion", "candidate", "detail", "estimability", "failed_gates", "gate_detail", "reason", "scheme", "split", "status", "subgroup"}
            rendered = value_aliases.get(str(value), str(value).replace("_", " ") if column in prose_columns else str(value))
            if column == "failed_gates":
                rendered = (
                    rendered.replace("development patients at least 5000", "dev<5k")
                    .replace("measurements at least 20000", "obs<20k")
                    .replace("treatment strata at least 1000", "arm<1k")
                    .replace(" | ", "; ")
                )
            elif column == "gate_detail":
                rendered = (
                    rendered.replace("crps improvement at least 10 pct", "CRPS gain < 10%")
                    .replace("coverage 80 within 5 points", "80% coverage")
                    .replace("coverage 90 within 5 points", "90% coverage")
                    .replace("weight gate", "weight")
                    .replace("minimum cell", "cell n")
                    .replace("no horizon more than 5 pct worse", "horizon guard")
                    .replace("source limit: nominal wide-horizon timing and no fill-level coverage audit", "wide-source contract")
                    .replace("run limit: bounded smoke sample with reduced tuning and bootstrap replication", "smoke run")
                )
            if len(display.columns) <= 3 and column in {"assertion", "detail"}:
                width = 52
            else:
                width = 32 if column in {"failed_gates", "gate_detail"} else 28 if column in {"detail", "reason"} else 22
            return textwrap.shorten(rendered, width=width, placeholder="...")

        display[column] = display[column].map(format_value)

    header_aliases = {
        "accepted": "Accepted",
        "administratively_immature": "Admin.\nimmature",
        "administratively_mature": "Admin.\nmature",
        "appropriate": "Gate\nresult",
        "age_mean": "Age\nmean",
        "baseline_bmi_mean": "Baseline BMI\nmean",
        "baseline_hba1c_mean": "Baseline HbA1c\nmean",
        "claim_status": "Claim\nstatus",
        "correction": "Conformal\nadjust.",
        "coverage": "Interval\nlevel",
        "coverage_80": "80%\ncoverage",
        "coverage_90": "90%\ncoverage",
        "development_patients": "Develop.\nn",
        "diabetes_pct": "Diabetes\n%",
        "effective_sample_size": "Effective\nsample size",
        "failed_gates": "Failed gates",
        "female_pct": "Female\n%",
        "implausible_prediction_rate": "Implausible\nrate",
        "median_censor_day": "Median censor\nday",
        "mature_with_target": "Mature with\ntarget",
        "mature_without_target": "Mature without\ntarget",
        "n_calibration": "Calib.\nn",
        "origin_month": "Origin\nmonth",
        "paired_crps_difference_high": "Paired CRPS\nCI high",
        "qualifying_patients": "Qualifying\npatients",
        "reclassified_vs_primary": "Reclassified\nvs primary",
        "relative_standardized_crps_improvement": "Relative CRPS\nimprovement",
        "repeated_fraction": "Repeated\nfraction",
        "selected_candidate": "Selected\nmodel",
        "small_cell_suppressed": "Suppressed",
        "source_contract_pass": "Source\ncontract",
        "target_month": "Target\nmonth",
        "valid_measurements": "Valid\nn",
        "validation_crps": "Validation\nCRPS",
        "validation_standardized_crps": "Std. validation\nCRPS",
        "version/status": "Version / status",
        "weight_gate_pass": "Weight\ngate",
        "width_80": "80%\nwidth",
        "width_90": "90%\nwidth",
    }
    display_columns = list(display.columns)
    if labels:
        column_labels = list(labels)
    else:
        column_labels = [
            header_aliases.get(
                column,
                textwrap.fill(
                    column.replace("_", " ").title(),
                    width=12,
                    break_long_words=False,
                    break_on_hyphens=False,
                ),
            )
            for column in display_columns
        ]

    wide_columns = {
        "assertion": 1.55,
        "candidate": 1.55,
        "claim_status": 1.45,
        "correction": 1.10,
        "detail": 1.55,
        "failed_gates": 2.10,
        "gate_detail": 1.75,
        "reason": 1.85,
        "scheme": 1.45,
        "selected_candidate": 1.55,
        "split": 1.35,
        "status": 1.35,
        "treatment": 1.35,
        "value": 1.25,
    }
    narrow_columns = {
        "accepted": 0.75,
        "appropriate": 0.80,
        "censored": 0.80,
        "coverage": 0.75,
        "coverage_80": 0.85,
        "coverage_90": 0.85,
        "paired_crps_difference_high": 0.95,
        "crps": 0.75,
        "heldout": 0.75,
        "n": 0.60,
        "origin_month": 0.70,
        "passed": 0.70,
        "source_contract_pass": 0.78,
        "suppressed": 0.80,
        "target_month": 0.70,
        "validation_standardized_crps": 1.00,
        "weight_gate_pass": 0.72,
    }
    widths = [wide_columns.get(column, narrow_columns.get(column, 1.0)) for column in display_columns]
    width_total = sum(widths)
    normalized_widths = [width / width_total for width in widths]
    table = axis.table(
        cellText=display.values,
        colLabels=column_labels,
        cellLoc="left",
        colLoc="left",
        colWidths=normalized_widths,
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5 if len(display_columns) >= 9 else 6.8 if len(display_columns) >= 7 else 7.2)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        cell.set_linewidth(0.5)
        cell.get_text().set_clip_path(cell)
        if row == 0:
            cell.set_facecolor("#EAF2F8")
            cell.set_text_props(weight="bold", color=PALETTE["ink"])
        elif row % 2 == 0:
            cell.set_facecolor("#F5F7FA")


def render_page_01(data: Mapping[str, Any]) -> Any:
    identity = data["identity"]
    figure = new_page(1, "Run identity and study status", "Reproducible fingerprints, environment, data interval, and hard-gate outcome")
    left = figure.add_axes([0.06, 0.52, 0.43, 0.32])
    right = figure.add_axes([0.53, 0.52, 0.41, 0.32])
    bottom = figure.add_axes([0.06, 0.10, 0.88, 0.32])
    panel_label(left, "A", "Immutable run identity")
    left.axis("off")
    badge, _ = run_badge(data)
    identity_lines = [
        ("Status", badge),
        ("Study version", identity["study_version"]),
        ("Run fingerprint", str(identity["fingerprint"])[:32] + "..."),
        ("Script SHA-256", str(identity["script_sha256"])[:32] + "..."),
        ("Query fingerprint", str(identity["query_fingerprint"])[:32] + "..."),
        ("Schema fingerprint", str(identity["schema_fingerprint"])[:32] + "..."),
        ("Source", identity["source_mode"]),
        ("Index dates", " to ".join(identity["cohort_date_range"])),
        ("Generated UTC", identity["generated_utc"]),
    ]
    for index, (label, value) in enumerate(identity_lines):
        y = 0.91 - index * 0.095
        left.text(0.02, y, label, color=PALETTE["muted"], fontsize=8.5)
        left.text(0.35, y, value, color=PALETTE["ink"], fontsize=8.5, fontweight="bold" if index == 0 else "normal")
    panel_label(right, "B", "Dependency manifest")
    dependency_rows = pd.DataFrame(
        [{"package": key, "version/status": value} for key, value in identity["dependencies"].items() if key not in {"executable", "platform"}]
    )
    draw_compact_table(right, dependency_rows, ["package", "version/status"], max_rows=12)
    panel_label(bottom, "C", "Preflight gates")
    bottom.axis("off")
    preflight = identity.get("preflight", {})
    wide_limited = preflight.get("strict_raw_event_contract") is False
    counts = preflight.get("source_row_counts", preflight.get("row_counts", {}))
    sampled_patients = preflight.get("source_unique_patient_counts", {})
    total_patients = preflight.get("source_total_unique_patient_counts", sampled_patients)
    accepted_value = preflight.get("accepted_medication_records", "not reported")
    accepted_text = f"{int(accepted_value):,}" if isinstance(accepted_value, (int, float)) else str(accepted_value)
    cohort_counts = data.get("cohort_counts", pd.DataFrame())
    cohort_text = (
        " | ".join(f"{row.cohort} {int(row.n):,}" for row in cohort_counts.itertuples(index=False))
        if not cohort_counts.empty else "not constructed in preflight-only mode"
    )
    gates = data.get("gates", pd.DataFrame())
    gate_text = (
        " | ".join(f"{status} {int(count):,}" for status, count in gates["claim_status"].value_counts().items())
        if not gates.empty else "not evaluated in preflight-only mode"
    )
    if wide_limited:
        limitations = preflight.get("limitations", identity.get("source_limitations", []))
        limitation_text = " ".join(str(item) for item in limitations[:3])
        sampling_text = " | ".join(
            f"{source} {int(sampled_patients.get(source, 0)):,} of {int(total):,} patients"
            for source, total in total_patients.items()
        )
        text = (
            "The standalone direct SQL preflight completed against MBSCohort and GLP1Cohort. Required wide columns, "
            "units, index dates, and reported medication intervals were usable. This is not an exact raw-event contract.\n\n"
            + "Direct query rows: "
            + " | ".join(f"{key} {value:,}" for key, value in counts.items())
            + f"\nPatient coverage: {sampling_text}"
            + f"\nAccepted reported medication intervals: {accepted_text}\n"
            + f"Eligible cohorts: {cohort_text}\nDecision cells: {gate_text}\n"
            + limitation_text
        )
        if identity.get("run_mode") == "smoke":
            result_text = "RESULT: SMOKE COMPLETED - NON-INFERENTIAL"
            result_color = PALETTE["red"]
        else:
            result_text = "RESULT: FULL WIDE-SOURCE RUN - EXPLORATORY"
            result_color = PALETTE["orange"]
    else:
        text = (
            "All hard gates passed. Exact surgery, medication, and measurement dates were verified; "
            "medication coverage used audited interval evidence; units and source lineage were present; "
            "administrative opportunity and blinded center identity were resolved.\n\n"
            + "Source rows: "
            + " | ".join(f"{key} {value:,}" for key, value in counts.items())
            + f"\nAccepted medication records: {accepted_text}"
        )
        if identity.get("run_mode") == "smoke":
            result_text = "RESULT: SMOKE COMPLETED - NON-INFERENTIAL"
            result_color = PALETTE["red"]
        else:
            result_text = "RESULT: COMPLETED"
            result_color = PALETTE["green"]
    bottom.text(0.02, 0.88, text, va="top", fontsize=9.2, color=PALETTE["ink"], linespacing=1.35, wrap=True)
    bottom.text(0.02, 0.10, result_text, fontsize=15, fontweight="bold", color=result_color)
    return figure


def render_page_02(data: Mapping[str, Any]) -> Any:
    subtitle = source_aware_text(
        data,
        "Every inclusion and exclusion is enumerated separately for surgery and six-month incretin continuers",
    )
    figure = new_page(2, "Cohort funnels", subtitle)
    funnel = data["funnel"].copy()
    for panel_index, cohort_name in enumerate(("surgery", "incretin")):
        axis = figure.add_axes([0.145 + panel_index * 0.47, 0.11, 0.325, 0.73])
        panel_label(
            axis,
            chr(ord("A") + panel_index),
            "Surgical cohort" if cohort_name == "surgery" else source_aware_text(data, "Six-month continuer cohort"),
        )
        subset = funnel.loc[funnel["cohort"].eq(cohort_name)].copy().head(14)
        if subset.empty:
            empty_panel(axis)
            continue
        colors = [PALETTE["green"] if value == "included" else PALETTE["orange"] for value in subset["status"]]
        y = np.arange(len(subset))
        counts = pd.to_numeric(subset["n_patients"], errors="coerce").fillna(0)
        plotted_counts = counts.mask(counts.gt(0) & counts.lt(MIN_CELL_SIZE), 0)
        axis.barh(y, plotted_counts, color=colors, alpha=0.88)
        axis.set_yticks(
            y,
            [
                textwrap.fill(
                    source_aware_text(data, str(item).replace("_", " ")),
                    width=21,
                )
                for item in subset["stage"]
            ],
            fontsize=7.1,
        )
        axis.invert_yaxis()
        axis.set_xlabel("Patients")
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
        maximum = max(float(plotted_counts.max()), 1)
        for position, count, plotted_count in zip(y, counts, plotted_counts, strict=False):
            axis.text(float(plotted_count) + maximum * 0.015, position, display_count(count), va="center", fontsize=7.5, color=PALETTE["ink"])
        axis.set_xlim(0, maximum * 1.25)
    return figure


def render_page_03(data: Mapping[str, Any]) -> Any:
    wide_limited = data["identity"].get("preflight", {}).get("strict_raw_event_contract") is False
    subtitle = (
        "Reported start-end intervals, ≥183-day interval eligibility, prior exposure, and reported censor timing"
        if wide_limited
        else "Coverage evidence, 183-day persistence, switches, prior-exposure strata, and exact censor timing"
    )
    figure = new_page(3, "Exposure continuity and censoring", subtitle)
    axes = [
        figure.add_axes([0.13, 0.52, 0.35, 0.32]),
        figure.add_axes([0.55, 0.52, 0.39, 0.32]),
        figure.add_axes([0.06, 0.10, 0.42, 0.31]),
        figure.add_axes([0.54, 0.10, 0.40, 0.31]),
    ]
    exposure = data["exposure"]
    panel_label(axes[0], "A", "Exposure classification")
    classification = exposure["classification"]
    if classification.empty:
        empty_panel(axes[0])
    else:
        labels = [
            f"{row.cohort.title()} | {source_aware_text(data, str(row.classification).replace('_', ' '))}"
            for row in classification.itertuples(index=False)
        ]
        axes[0].barh(np.arange(len(labels)), classification["n"].fillna(0), color=PALETTE["blue"])
        axes[0].set_yticks(np.arange(len(labels)), [textwrap.fill(item, width=22) for item in labels], fontsize=7.0)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Patients")
    panel_label(axes[1], "B", "Reported interval-end and surgery censoring counts" if wide_limited else "Censoring counts")
    draw_compact_table(axes[1], exposure["censoring"], ["cohort", "n", "censored", "day_zero", "median_censor_day"])
    panel_label(axes[2], "C", "Reported interval source audit" if wide_limited else "Coverage source audit")
    draw_compact_table(axes[2], exposure["sources"], ["source_type", "accepted", "reason", "n"], max_rows=9)
    panel_label(
        axes[3],
        "D",
        "Continuity under full reported-interval assumption" if wide_limited else "Six-month continuity distribution",
    )
    draw_compact_table(axes[3], exposure["continuity"], list(exposure["continuity"].columns), max_rows=9)
    return figure


def render_page_04(data: Mapping[str, Any]) -> Any:
    center_available = bool(data["split"]["metadata"].get("center_validation_available", True))
    subtitle = (
        "Treatment-specific cohort composition, calendar support, and center contribution"
        if center_available
        else "Treatment-specific cohort composition, calendar support, and center-identifier availability"
    )
    figure = new_page(4, "Baseline composition", subtitle)
    top = figure.add_axes([0.06, 0.48, 0.88, 0.36])
    bottom_left = figure.add_axes([0.12, 0.10, 0.36, 0.28])
    bottom_right = figure.add_axes([0.54, 0.10, 0.40, 0.28])
    panel_label(top, "A", "Baseline characteristics by observed treatment")
    draw_compact_table(
        top, data["baseline"],
        ["cohort", "treatment", "n", "age_mean", "baseline_bmi_mean", "baseline_hba1c_mean", "female_pct", "diabetes_pct", "centers"],
        max_rows=12,
    )
    panel_label(bottom_left, "B", "Cohort size by treatment")
    baseline = data["baseline"].loc[~data["baseline"]["small_cell_suppressed"]].copy()
    if baseline.empty:
        empty_panel(bottom_left)
    else:
        labels = baseline["cohort"].astype(str) + ": " + baseline["treatment"].astype(str)
        bottom_left.barh(np.arange(len(labels)), baseline["n"], color=PALETTE["green"])
        bottom_left.set_yticks(np.arange(len(labels)), labels, fontsize=7.2)
        bottom_left.invert_yaxis()
        bottom_left.set_xlabel("Patients")
    panel_label(bottom_right, "C", "Blinded center contribution" if center_available else "Center identifier availability")
    centers = data["split"]["centers"]
    draw_compact_table(bottom_right, centers, ["center_blind", "heldout", "n"], max_rows=10)
    return figure


def heatmap_table(axis: Any, frame: Any, row_column: str, column_column: str, value_column: str, annotation_column: str | None = None) -> None:
    if frame.empty:
        empty_panel(axis)
        return
    row_values = sorted(frame[row_column].dropna().unique())
    column_values = sorted(frame[column_column].dropna().unique())
    pivot = frame.pivot_table(index=row_column, columns=column_column, values=value_column, aggfunc="first", dropna=False)
    pivot = pivot.reindex(index=row_values, columns=column_values)
    matrix = pivot.to_numpy(float)
    finite = np.isfinite(matrix)
    if not finite.any():
        empty_panel(axis, "Not estimable after disclosure control")
        return
    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axis.set_yticks(np.arange(len(pivot.index)), [str(item) for item in pivot.index], fontsize=7.5)
    axis.set_xticks(np.arange(len(pivot.columns)), [str(item) for item in pivot.columns], fontsize=7.5)
    for row_index, row_value in enumerate(pivot.index):
        for column_index, column_value in enumerate(pivot.columns):
            value = pivot.loc[row_value, column_value]
            if pd.notna(value):
                annotation = f"{value:.0f}"
                if annotation_column:
                    match = frame.loc[frame[row_column].eq(row_value) & frame[column_column].eq(column_value)]
                    if not match.empty:
                        annotation += f"\nn={display_count(match.iloc[0][annotation_column])}"
                axis.text(column_index, row_index, annotation, ha="center", va="center", color="white" if value < np.nanmean(matrix) else "black", fontsize=6.8)
    plt.colorbar(image, ax=axis, fraction=0.04, pad=0.03)


def render_page_05(data: Mapping[str, Any]) -> Any:
    wide_limited = wide_source_limited(data)
    subtitle = (
        "Operationalized follow-up opportunity is separated from missing nominal-horizon values and reported censoring"
        if wide_limited
        else "Administrative maturity is separated from missing measurement and treatment or surgery censoring"
    )
    figure = new_page(5, "Follow-up and target support", subtitle)
    support = data["support"].copy()
    support = support.loc[support["origin_month"].eq(0)].copy()
    support = mask_small_cell_metrics(support, ["target_availability_pct"], count_column="eligible")
    support["task"] = support["cohort"].str.title() + " | " + support["outcome"].str.upper()
    top = figure.add_axes([0.12, 0.48, 0.80, 0.36])
    bottom = figure.add_axes([0.06, 0.10, 0.88, 0.27])
    panel_label(
        top,
        "A",
        "Baseline-origin available nominal-horizon values among eligible rows (%)"
        if wide_limited
        else "Baseline-origin valid target availability among eligible rows (%)",
    )
    heatmap_table(top, support, "task", "target_month", "target_availability_pct", "eligible")
    panel_label(bottom, "B", "Operational opportunity decomposition" if wide_limited else "Opportunity decomposition")
    summary = support.groupby(["cohort", "outcome"])[["administratively_mature", "mature_with_target", "mature_without_target", "censored", "administratively_immature"]].sum().reset_index()
    draw_compact_table(bottom, summary, list(summary.columns), max_rows=8)
    return figure


def render_page_06(data: Mapping[str, Any]) -> Any:
    wide_limited = data["identity"].get("preflight", {}).get("strict_raw_event_contract") is False
    subtitle = (
        "Wide-column units, source provenance, validity, and nominal horizon timing"
        if wide_limited
        else "Raw units, source concepts, invalid values, derived BMI, duplicate-day handling, and exact timing"
    )
    figure = new_page(6, "Measurement quality", subtitle)
    left = figure.add_axes([0.06, 0.47, 0.55, 0.37])
    right = figure.add_axes([0.66, 0.47, 0.28, 0.37])
    bottom = figure.add_axes([0.06, 0.10, 0.88, 0.27])
    quality = data["measurement_quality"]
    panel_label(left, "A", "Validation reasons by outcome and unit")
    draw_compact_table(left, quality["reasons"], ["kind", "unit", "reason", "valid", "n"], max_rows=13)
    panel_label(right, "B", "Duplicate-day and derivation audit")
    draw_compact_table(right, quality["duplicates"], list(quality["duplicates"].columns), max_rows=8)
    panel_label(bottom, "C", "Frozen primary rules")
    bottom.axis("off")
    if wide_limited:
        rules = (
            "BMI wide columns are interpreted as kg/m2 and HbA1c wide columns as NGSP percent; values outside 10 to 100 "
            "kg/m2 or 3 to 20 percent remain invalid. Baseline values use the exact cohort index date. Fixed follow-up "
            "columns are assigned their named nominal horizon only so the common modeling pipeline can operate. They are "
            "not evidence of an exact measurement timestamp, visit window, or within-window count. Continuous-time model "
            "eligibility therefore fails the exact-timestamp gate."
        )
    else:
        rules = (
            "BMI: normalize weight and height before derivation; accept 10 to 100 kg/m2. HbA1c: normalize NGSP percent; "
            "convert IFCC mmol/mol with NGSP = IFCC / 10.929 + 2.15; accept 3 to 20 percent. Values outside these "
            "ranges are invalid and never clipped. Duplicate valid values on the same patient-day are summarized by the median. "
            "The primary window target is closest to the nominal day, with equal-distance ties resolved to the earlier date."
        )
    bottom.text(0.01, 0.88, rules, va="top", fontsize=9.5, color=PALETTE["ink"], wrap=True, linespacing=1.45)
    return figure


def render_page_07(data: Mapping[str, Any]) -> Any:
    center_available = bool(data["split"]["metadata"].get("center_validation_available", True))
    wide_limited = wide_source_limited(data)
    split_subtitle = (
        "Global patient assignment, temporal testing, geographic holdout, and timestamp-level invariants"
        if center_available
        else (
            "Global patient assignment, temporal testing, nominal-horizon ordering, and explicit geographic-validation unavailability"
            if wide_limited
            else "Global patient assignment, temporal testing, and explicit geographic-validation unavailability"
        )
    )
    figure = new_page(7, "Split and leakage audit", split_subtitle)
    left = figure.add_axes([0.06, 0.50, 0.36, 0.34])
    right = figure.add_axes([0.48, 0.50, 0.46, 0.34])
    bottom = figure.add_axes([0.06, 0.10, 0.88, 0.30])
    split = data["split"]
    panel_label(left, "A", "Patient-separated split sizes")
    counts = split["counts"]
    left.bar(np.arange(len(counts)), counts["n"], color=[PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["purple"], PALETTE["red"]][:len(counts)])
    left.set_xticks(np.arange(len(counts)), [str(item).replace("_", "\n") for item in counts["split"]], fontsize=7.5)
    left.set_ylabel("Unique patients")
    panel_label(right, "B", "Transportability design")
    right.axis("off")
    metadata = split["metadata"]
    if center_available:
        design_text = (
            f"Temporal cutoff: {metadata.get('temporal_cutoff')}\n"
            f"Completely held-out centers: {len(metadata.get('heldout_centers', []))}\n"
            "Center identities are blinded in every export. Raw center identity is never a primary predictor.\n"
            "Calendar and center assignments were fixed before task and horizon expansion."
        )
    else:
        design_text = (
            f"Temporal cutoff: {metadata.get('temporal_cutoff')}\n"
            "Completely held-out centers: not estimable\n"
            "The direct wide sources do not provide a complete usable center identifier. Geographic validation is disabled, "
            "and no sentinel center is treated as a real site. Patient-global assignment and protected temporal testing remain active."
        )
    right.text(0.02, 0.90, design_text, va="top", fontsize=9.7, linespacing=1.55, color=PALETTE["ink"], wrap=True)
    panel_label(bottom, "C", "Nominal-horizon leakage assertions" if wide_limited else "Leakage assertions")
    leakage = split["leakage"].copy()
    if wide_limited and not leakage.empty:
        leakage["assertion"] = leakage["assertion"].replace(
            {
                "Every feature timestamp is at or before origin": "Every nominal feature horizon is at or before origin",
                "Every target is strictly after origin": "Every nominal target horizon is strictly after origin",
                "No observed target is on or after treatment or surgery censoring": "No available nominal target is on or after reported censoring",
            }
        )
        leakage["detail"] = leakage["detail"].astype(str).str.replace(
            "Window and plausibility constants",
            "Nominal-horizon and plausibility constants",
            regex=False,
        )
    draw_compact_table(bottom, leakage, ["assertion", "passed", "detail"], max_rows=10)
    return figure


def render_page_08(data: Mapping[str, Any]) -> Any:
    figure = new_page(8, "Model selection", "Architecture applicability, validation CRPS, interval calibration, resources, and frozen selection")
    left = figure.add_axes([0.13, 0.48, 0.39, 0.36])
    right = figure.add_axes([0.57, 0.48, 0.37, 0.36])
    bottom = figure.add_axes([0.06, 0.10, 0.88, 0.27])
    leaderboard = data["leaderboard"].loc[data["leaderboard"]["origin_month"].eq(0)].copy()
    panel_label(left, "A", "Equal-horizon standardized validation CRPS")
    if leaderboard.empty:
        empty_panel(left)
    else:
        summary = (
            leaderboard.groupby("candidate")["validation_standardized_crps"]
            .mean()
            .sort_values()
            .head(8)
        )
        left.barh(np.arange(len(summary)), summary.values, color=PALETTE["blue"])
        aliases = {
            "catboost_multi_quantile": "CatBoost",
            "catboost_multiquantile": "CatBoost",
            "hist_gradient_boosting": "Histogram boosting",
            "histogram_gradient_boosting": "Histogram boosting",
            "noncrossing_mlp": "Noncrossing MLP",
            "population_change": "Population change",
            "pytorch_quantile_mlp": "Quantile MLP",
            "regularized_spline": "Regularized spline",
            "spline_ridge": "Spline ridge",
            "validation_weighted_ensemble": "Weighted ensemble",
        }
        left.set_yticks(np.arange(len(summary)), [aliases.get(str(item), str(item).replace("_", " ")) for item in summary.index], fontsize=7.2)
        left.invert_yaxis()
        left.set_xlabel("Mean task-standardized CRPS (equal horizons; lower is better)")
    panel_label(right, "B", "Selected models")
    draw_compact_table(
        right,
        data["selected"],
        ["cohort", "outcome", "origin_month", "selected_candidate", "validation_standardized_crps"],
        max_rows=12,
    )
    panel_label(bottom, "C", "Candidate execution status")
    draw_compact_table(bottom, data["model_status"], ["cohort", "outcome", "origin_month", "candidate", "status", "reason"], max_rows=10)
    return figure


def render_page_09(data: Mapping[str, Any]) -> Any:
    figure = new_page(9, "Continuous-time model diagnostics", "Repeated-measure gates, ODE-RNN design, solver safeguards, and selection decision")
    left = figure.add_axes([0.06, 0.49, 0.54, 0.35])
    right = figure.add_axes([0.65, 0.49, 0.29, 0.35])
    bottom = figure.add_axes([0.06, 0.10, 0.88, 0.28])
    panel_label(left, "A", "Task-specific suitability gates")
    draw_compact_table(left, data["ode_gates"], ["cohort", "outcome", "appropriate", "development_patients", "valid_measurements", "repeated_fraction", "failed_gates"], max_rows=8)
    panel_label(right, "B", "ODE-RNN information flow")
    right.axis("off")
    boxes = [(0.08, 0.78, "Static context"), (0.08, 0.54, "h(0)"), (0.08, 0.30, "RK4 dynamics"), (0.55, 0.54, "GRU updates"), (0.55, 0.30, "7 quantiles")]
    for x, y, label in boxes:
        right.text(x, y, label, ha="center", va="center", fontsize=8.5, bbox={"boxstyle": "round,pad=0.35", "facecolor": "#EAF2F8", "edgecolor": PALETTE["blue"]})
    arrows = [((0.08, 0.72), (0.08, 0.60)), ((0.08, 0.48), (0.08, 0.36)), ((0.19, 0.54), (0.43, 0.54)), ((0.19, 0.30), (0.43, 0.30)), ((0.55, 0.48), (0.55, 0.36))]
    for start, end in arrows:
        right.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": PALETTE["muted"]})
    panel_label(bottom, "C", "Numerical and scientific decision")
    bottom.axis("off")
    appropriate_count = int(data["ode_gates"]["appropriate"].sum()) if not data["ode_gates"].empty else 0
    bottom.text(
        0.01, 0.88,
        f"Tasks passing every suitability gate: {appropriate_count}. The RK4 solver is fixed-step, differentiable, lands exactly on events, "
        "uses a primary maximum step of 1/12 year, and is checked at 1/24 year. No adjoint, interpolation, SciPy integrator, "
        "or third-party ODE package is used. A task failing any repeated-measure gate is labeled not appropriate and is not fitted. "
        "Selection additionally requires improvement over both the MLP and best boosted model plus solver and seed stability.",
        va="top", fontsize=10, wrap=True, linespacing=1.5, color=PALETTE["ink"],
    )
    return figure


def selected_metric_frame(data: Mapping[str, Any], cohort: str, outcome: str) -> Any:
    selected = data["selected"].loc[data["selected"]["cohort"].eq(cohort) & data["selected"]["outcome"].eq(outcome)]
    pieces = []
    for row in selected.itertuples(index=False):
        candidates = [row.selected_candidate, "population_change", "persistence"]
        pieces.append(
            data["metrics"].loc[
                data["metrics"]["cohort"].eq(cohort)
                & data["metrics"]["outcome"].eq(outcome)
                & data["metrics"]["origin_month"].eq(row.origin_month)
                & data["metrics"]["candidate"].isin(candidates)
            ]
        )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def selected_only_metric_frame(data: Mapping[str, Any]) -> Any:
    """Return metrics for the candidate selected for each exact task and origin."""
    metrics = data["metrics"]
    selected = data["selected"]
    if metrics.empty or selected.empty:
        return metrics.iloc[0:0].copy()
    selection_keys = selected.loc[:, ["cohort", "outcome", "origin_month", "selected_candidate"]].rename(
        columns={"selected_candidate": "candidate"}
    )
    return metrics.merge(
        selection_keys,
        on=["cohort", "outcome", "origin_month", "candidate"],
        how="inner",
        validate="many_to_one",
    )


def merge_patient_bootstrap_intervals(data: Mapping[str, Any], metrics: Any) -> Any:
    intervals = data.get("bootstrap_ci", pd.DataFrame())
    if metrics.empty or intervals.empty:
        return metrics.copy()
    intervals = intervals.loc[intervals["bootstrap_type"].eq("patient")].copy()
    keys = ["cohort", "outcome", "origin_month", "target_month", "split", "candidate"]
    return metrics.merge(
        intervals[
            keys
            + [
                "crps_low",
                "crps_high",
                "rmse_low",
                "rmse_high",
                "coverage_80_low",
                "coverage_80_high",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )


def performance_page(data: Mapping[str, Any], number: int, cohort: str, outcome: str, title: str) -> Any:
    center_available = bool(data["split"]["metadata"].get("center_validation_available", True))
    subtitle = (
        "Factual probabilistic performance with 95% patient-bootstrap intervals on protected temporal and held-out-center tests"
        if center_available
        else "Factual probabilistic performance with 95% patient-bootstrap intervals; geographic validation unavailable"
    )
    figure = new_page(number, title, subtitle)
    metrics = selected_metric_frame(data, cohort, outcome)
    axes = [
        figure.add_axes([0.06, 0.51, 0.42, 0.32]),
        figure.add_axes([0.54, 0.51, 0.40, 0.32]),
        figure.add_axes([0.06, 0.11, 0.42, 0.29]),
        figure.add_axes([0.54, 0.11, 0.40, 0.29]),
    ]
    titles = ("CRPS by horizon", "RMSE by horizon", "80% interval coverage", "Eligible counts and estimability")
    for label, axis, panel_title in zip("ABCD", axes, titles, strict=False):
        panel_label(axis, label, panel_title)
    if metrics.empty:
        for axis in axes:
            empty_panel(axis)
        return figure
    origin_zero = metrics.loc[metrics["origin_month"].eq(0) & metrics["split"].eq("temporal_test")].copy()
    origin_zero = mask_small_cell_metrics(origin_zero, ["crps", "rmse", "coverage_80"])
    origin_zero = merge_patient_bootstrap_intervals(data, origin_zero)
    colors = {"population_change": PALETTE["orange"], "persistence": PALETTE["muted"]}
    origin_selection = data["selected"].loc[
        data["selected"]["cohort"].eq(cohort)
        & data["selected"]["outcome"].eq(outcome)
        & data["selected"]["origin_month"].eq(0),
        "selected_candidate",
    ]
    selected_candidate = str(origin_selection.iloc[0]) if not origin_selection.empty else None
    for axis, metric_column in zip(axes[:3], ("crps", "rmse", "coverage_80"), strict=False):
        plotted = False
        for candidate, group in origin_zero.groupby("candidate", sort=True):
            group = group.sort_values("target_month")
            if not pd.to_numeric(group[metric_column], errors="coerce").notna().any():
                continue
            color = colors.get(candidate, PALETTE["blue"] if candidate == selected_candidate else PALETTE["green"])
            label = "Selected" if candidate == selected_candidate else candidate.replace("_", " ")
            low_column = f"{metric_column}_low"
            high_column = f"{metric_column}_high"
            if low_column in group and high_column in group:
                values = pd.to_numeric(group[metric_column], errors="coerce").to_numpy(float)
                low = pd.to_numeric(group[low_column], errors="coerce").to_numpy(float)
                high = pd.to_numeric(group[high_column], errors="coerce").to_numpy(float)
                lower_error = np.maximum(values - low, 0.0)
                upper_error = np.maximum(high - values, 0.0)
                if np.isfinite(lower_error).any() and np.isfinite(upper_error).any():
                    axis.errorbar(
                        group["target_month"],
                        values,
                        yerr=np.vstack([lower_error, upper_error]),
                        marker="o",
                        label=label,
                        color=color,
                        capsize=2.5,
                        lw=1.4,
                    )
                else:
                    axis.plot(group["target_month"], values, marker="o", label=label, color=color)
            else:
                axis.plot(group["target_month"], group[metric_column], marker="o", label=label, color=color)
            plotted = True
        if not plotted:
            empty_panel(axis, "Not estimable after disclosure control")
            continue
        if metric_column == "coverage_80":
            axis.axhline(0.80, color=PALETTE["ink"], ls="--", lw=1)
            axis.axhspan(0.75, 0.85, color=PALETTE["green"], alpha=0.10)
        axis.set_xlabel("Target month")
        axis.legend(fontsize=6.5, frameon=False)
    selected_metrics = selected_only_metric_frame(data)
    count_table = selected_metrics.loc[
        selected_metrics["cohort"].eq(cohort) & selected_metrics["outcome"].eq(outcome),
        ["origin_month", "target_month", "split", "n", "effective_sample_size", "estimability"],
    ].sort_values(["origin_month", "target_month", "split"])
    draw_compact_table(axes[3], count_table, list(count_table.columns), max_rows=10)
    return figure


def render_page_14(data: Mapping[str, Any]) -> Any:
    figure = new_page(14, "Distributional calibration", "Held-out interval coverage, conformal corrections, PIT shape, widths, and implausible predictions")
    axes = [
        figure.add_axes([0.06, 0.50, 0.42, 0.34]),
        figure.add_axes([0.54, 0.50, 0.40, 0.34]),
        figure.add_axes([0.06, 0.10, 0.42, 0.29]),
        figure.add_axes([0.54, 0.10, 0.40, 0.29]),
    ]
    for axis, label, title in zip(axes, "ABCD", ("80% coverage", "PIT histograms", "Conformal corrections", "Tail and plausibility audit"), strict=False):
        panel_label(axis, label, title)
    metrics = selected_only_metric_frame(data)
    metrics = metrics.loc[metrics["origin_month"].eq(0)].copy()
    metrics = mask_small_cell_metrics(
        metrics,
        ["coverage_80", "width_80", "width_90", "implausible_prediction_rate"],
    )
    if metrics.empty:
        empty_panel(axes[0])
    else:
        for keys, group in metrics.groupby(["cohort", "outcome", "split"]):
            cohort, outcome, split = keys
            group = group.sort_values("target_month")
            label = f"{str(cohort).title()} | {str(outcome).upper()} | {str(split).replace('_', ' ').title()}"
            axes[0].plot(group["target_month"], group["coverage_80"], marker="o", label=label)
        axes[0].axhline(0.80, color=PALETTE["ink"], ls="--")
        axes[0].axhspan(0.75, 0.85, color=PALETTE["green"], alpha=0.1)
        axes[0].set_xlabel("Target month")
        axes[0].legend(fontsize=5.8, frameon=False, ncol=2)
    pit = data["pit_histograms"]
    if pit.empty:
        empty_panel(axes[1])
    else:
        axes[1].axis("off")
        combinations = [
            ("surgery", "bmi"),
            ("surgery", "hba1c"),
            ("incretin", "bmi"),
            ("incretin", "hba1c"),
        ]
        for index, (cohort, outcome) in enumerate(combinations):
            column = index % 2
            row = index // 2
            inset_y = 0.60 if row == 0 else 0.05
            inset = axes[1].inset_axes([0.04 + column * 0.49, inset_y, 0.44, 0.29])
            task = pit.loc[
                pit["cohort"].eq(cohort)
                & pit["outcome"].eq(outcome)
                & pit["origin_month"].eq(0)
                & pit["split"].eq("temporal_test")
            ]
            if task.empty:
                empty_panel(inset)
                inset.set_title(f"{cohort.title()} {outcome.upper()}", fontsize=7, pad=1)
                continue
            summary = task.groupby(["bin_left", "bin_right"])["n"].sum().reset_index()
            total = max(summary["n"].sum(), 1)
            inset.bar(
                summary["bin_left"],
                summary["n"] / total,
                width=0.095,
                align="edge",
                color=PALETTE["blue"],
                alpha=0.85,
            )
            inset.axhline(0.10, color=PALETTE["ink"], ls="--", lw=0.8)
            inset.set_xlim(0, 1)
            inset.set_title(f"{cohort.title()} {outcome.upper()}", fontsize=7, pad=1)
            inset.tick_params(labelsize=5.5)
            if row == 1:
                inset.set_xlabel("PIT", fontsize=6)
            if column == 0:
                inset.set_ylabel("Proportion", fontsize=6)
    calibration = data["calibration"]
    if calibration.empty or data["selected"].empty:
        selected_calibration = calibration
    else:
        selection_keys = data["selected"].loc[:, ["cohort", "outcome", "origin_month", "selected_candidate"]].rename(
            columns={"selected_candidate": "candidate"}
        )
        selected_calibration = calibration.merge(
            selection_keys,
            on=["cohort", "outcome", "origin_month", "candidate"],
            how="inner",
            validate="many_to_one",
        )
    draw_compact_table(
        axes[2],
        selected_calibration,
        ["cohort", "outcome", "target_month", "candidate", "coverage", "correction", "n_calibration", "status"],
        max_rows=10,
    )
    audit = metrics[["cohort", "outcome", "target_month", "split", "n", "width_80", "width_90", "implausible_prediction_rate"]] if not metrics.empty else pd.DataFrame()
    draw_compact_table(axes[3], audit, list(audit.columns), max_rows=10)
    return figure


def render_page_15(data: Mapping[str, Any]) -> Any:
    center_available = bool(data["split"]["metadata"].get("center_validation_available", True))
    subtitle = (
        "Held-out-center and prespecified subgroup performance auditing, not biological effect modification"
        if center_available
        else "Temporal and prespecified subgroup auditing; geographic transportability is not estimable"
    )
    figure = new_page(
        15,
        "Transportability and subgroup audit" if center_available else "Temporal and subgroup audit",
        subtitle,
    )
    top = figure.add_axes([0.06, 0.49, 0.88, 0.35])
    bottom_left = figure.add_axes([0.13, 0.10, 0.35, 0.28])
    bottom_right = figure.add_axes([0.54, 0.10, 0.40, 0.28])
    panel_label(top, "A", "Disclosure-controlled subgroup cells")
    draw_compact_table(top, data["subgroups"], ["cohort", "outcome", "target_month", "split", "subgroup", "value", "n", "crps", "coverage_80", "suppressed"], max_rows=12)
    panel_label(
        bottom_left,
        "B",
        "Task-standardized temporal versus geographic CRPS"
        if center_available
        else "Task-standardized protected temporal CRPS",
    )
    metrics = selected_only_metric_frame(data)
    metrics = metrics.loc[metrics["origin_month"].eq(0)].copy()
    metrics = mask_small_cell_metrics(metrics, ["crps"])
    if not metrics.empty:
        metrics = metrics.merge(
            data["iqr"],
            on=["cohort", "outcome", "target_month"],
            how="left",
            validate="many_to_one",
        )
        metrics["standardized_crps"] = metrics["crps"] / metrics["development_iqr"]
    if metrics.empty:
        empty_panel(bottom_left)
    else:
        summary = (
            metrics.groupby(["cohort", "outcome", "split"])["standardized_crps"]
            .mean()
            .dropna()
            .sort_values()
        )
        if summary.empty:
            empty_panel(bottom_left, "Not estimable after disclosure control")
        else:
            positions = np.arange(len(summary))
            cohort_aliases = {"surgery": "Surg.", "incretin": "Incr."}
            split_aliases = {"geographic_test": "geographic", "temporal_test": "temporal"}
            labels = [
                f"{cohort_aliases.get(str(cohort), str(cohort).title())} "
                f"{'HbA1c' if str(outcome).lower() == 'hba1c' else str(outcome).upper()} | "
                f"{split_aliases.get(str(split), str(split).replace('_', ' '))}"
                for cohort, outcome, split in summary.index
            ]
            bottom_left.barh(positions, summary.values, color=PALETTE["blue"])
            bottom_left.set_yticks(positions, labels, fontsize=6.4)
            bottom_left.invert_yaxis()
            bottom_left.set_xlabel("Mean standardized CRPS across supported horizons")
    panel_label(bottom_right, "C", "Interpretation")
    bottom_right.axis("off")
    if center_available:
        interpretation = (
            "Raw center identity is not a model feature. Geographic performance uses completely held-out centers. "
            "Subgroup cells with fewer than 11 patients are suppressed. These panels audit error and calibration; "
            "they do not establish biological effect modification or rank patient groups."
        )
    else:
        interpretation = (
            "Center identity is unavailable in the direct wide sources, so geographic transportability is not estimated. "
            "Subgroup cells with fewer than 11 patients are suppressed. Temporal and subgroup panels audit error and "
            "calibration only; they do not establish biological effect modification or rank patient groups."
        )
    bottom_right.text(0.02, 0.9, interpretation, va="top", fontsize=9.5, wrap=True, linespacing=1.5, color=PALETTE["ink"])
    return figure


def render_page_16(data: Mapping[str, Any]) -> Any:
    wide_limited = data["identity"].get("preflight", {}).get("strict_raw_event_contract") is False
    subtitle = (
        "Weighted performance and reported-interval sensitivity limits"
        if wide_limited
        else "Weighted and unweighted performance, effective sample size, gap rules, and estimand labeling"
    )
    figure = new_page(16, "Censoring and persistence sensitivities", subtitle)
    left = figure.add_axes([0.06, 0.48, 0.55, 0.36])
    right = figure.add_axes([0.66, 0.48, 0.28, 0.36])
    bottom = figure.add_axes([0.06, 0.10, 0.88, 0.27])
    panel_label(left, "A", "Weighting sensitivity")
    draw_compact_table(left, data["sensitivity"], ["cohort", "outcome", "target_month", "split", "scheme", "effective_sample_size", "crps", "coverage_80", "coverage_90"], max_rows=12)
    panel_label(right, "B", "Gap rule under reported intervals" if wide_limited else "Allowable-gap reclassification")
    draw_compact_table(right, data["gap_sensitivity"], list(data["gap_sensitivity"].columns), max_rows=6)
    panel_label(bottom, "C", "Estimand sensitivities")
    bottom.axis("off")
    if wide_limited:
        estimand_text = (
            "Primary surgery results use cross-fitted inverse-probability-of-remaining-incretin-free and observation weights. "
            "Primary incretin results condition on at least 183 days inside one reported GLP1StartDate-to-GLP1EndDate interval "
            "and censor at that reported end or a known bariatric procedure. Unweighted and alternate weight truncation results "
            "remain available. Because dispense-level records are absent, internal gaps, stockpiling, and switches cannot be "
            "reconstructed; identical 0-, 30-, and 60-day rows reflect that source limitation, not demonstrated persistence."
        )
    else:
        estimand_text = (
            "Primary surgery results use cross-fitted inverse-probability-of-remaining-incretin-free and observation weights. "
            "Primary incretin results are on-treatment among confirmed six-month continuers and censor at the first supported "
            "coverage end before an excessive gap, plus bariatric surgery. Unweighted complete-case results and alternate weight "
            "truncation are shown above. Gap-rule sensitivity reconstructs episodes and censor dates at 0, 30, and 60 days. "
            "Observed-care results, when produced in a full run, retain outcomes after medication discontinuation but still censor surgery and are labeled separately."
        )
    bottom.text(0.01, 0.88, estimand_text, va="top", fontsize=9.4, wrap=True, linespacing=1.45, color=PALETTE["ink"])
    return figure


def render_page_17(data: Mapping[str, Any]) -> Any:
    figure = new_page(17, "Conditional trajectory examples", "Fully synthetic examples with coherent cross-horizon draws; projections are noncausal")
    examples = data["examples"]
    axes = [
        figure.add_axes([0.06, 0.51, 0.42, 0.32]),
        figure.add_axes([0.54, 0.51, 0.40, 0.32]),
        figure.add_axes([0.06, 0.11, 0.42, 0.29]),
        figure.add_axes([0.54, 0.11, 0.40, 0.29]),
    ]
    combinations = [("surgery", "bmi"), ("surgery", "hba1c"), ("incretin", "bmi"), ("incretin", "hba1c")]
    for axis, label, combination in zip(axes, "ABCD", combinations, strict=False):
        panel_label(axis, label, f"{combination[0].title()} | {combination[1].upper()}")
        subset = examples.loc[examples["cohort"].eq(combination[0]) & examples["outcome"].eq(combination[1])]
        if subset.empty:
            empty_panel(axis)
            continue
        x = subset["target_month"].to_numpy(float)
        axis.fill_between(x, subset["q05"], subset["q95"], color=PALETTE["sky"], alpha=0.25, label="90% interval")
        axis.fill_between(x, subset["q25"], subset["q75"], color=PALETTE["blue"], alpha=0.25, label="50% interval")
        axis.plot(x, subset["q50"], color=PALETTE["blue"], marker="o", label="Median")
        axis.set_xlabel("Months after index")
        axis.set_ylabel("BMI kg/m2" if combination[1] == "bmi" else "HbA1c %")
        axis.legend(frameon=False, fontsize=7)
        axis.text(0.02, 0.04, "SYNTHETIC | NOT AN INDIVIDUAL TREATMENT EFFECT", transform=axis.transAxes, fontsize=6.8, fontweight="bold", color=PALETTE["red"])
    return figure


def render_page_18(data: Mapping[str, Any]) -> Any:
    figure = new_page(18, "Gates, limitations, and conclusion", "Horizon-specific support classification and the permitted publication claim")
    top = figure.add_axes([0.06, 0.48, 0.88, 0.36])
    bottom_left = figure.add_axes([0.06, 0.10, 0.51, 0.28])
    bottom_right = figure.add_axes([0.62, 0.10, 0.32, 0.28])
    panel_label(top, "A", "Horizon-specific decision table")
    draw_compact_table(
        top,
        data["gates"],
        [
            "cohort",
            "outcome",
            "origin_month",
            "target_month",
            "split",
            "n",
            "relative_standardized_crps_improvement",
            "coverage_80",
            "coverage_90",
            "weight_gate_pass",
            "source_contract_pass",
            "claim_status",
            "gate_detail",
        ],
        max_rows=13,
    )
    panel_label(bottom_left, "B", "Limitations")
    bottom_left.axis("off")
    limitations = source_aware_text(
        data,
        "This is a prognostic study, not a causal treatment comparison. Surgery and incretin therapy use separate models, "
        "populations, time zeros, and claims. The initiation-origin incretin model is conditional on future six-month persistence. "
        "Treatment censoring and observation may remain informative despite cross-fitted weighting. Recent ingredients can have "
        "short calendar support, so unsupported horizons are not estimable. Conditional sleeve and RYGB projections are model-based, "
        "overlap-restricted, baseline-origin projections and must not be used to recommend a procedure.",
    )
    source_limitations = data["identity"].get("source_limitations", [])
    if source_limitations:
        limitations += (
            " Direct wide-source limits: follow-up values have nominal rather than exact timestamps; each reported GLP1 "
            "start-end pair is treated as one uninterrupted interval; enrollment and administrative opportunity are "
            "operationalized; and unavailable center identity prevents geographic validation."
        )
    bottom_left.text(
        0.01, 0.91,
        textwrap.fill(limitations, width=78),
        va="top", fontsize=8.4, linespacing=1.34, color=PALETTE["ink"], clip_on=True,
    )
    panel_label(bottom_right, "C", "Publication conclusion")
    bottom_right.axis("off")
    supported = int(data["gates"]["claim_status"].eq("Supported").sum()) if not data["gates"].empty else 0
    exploratory = int(data["gates"]["claim_status"].eq("Exploratory").sum()) if not data["gates"].empty else 0
    total = len(data["gates"])
    bottom_right.text(
        0.02,
        0.76,
        f"Supported: {supported} of {total}",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["green"] if supported else PALETTE["orange"],
        va="top",
    )
    bottom_right.text(
        0.02,
        0.58,
        f"Exploratory: {exploratory} of {total}",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["orange"],
        va="top",
    )
    conclusion = "Claims remain prognostic, horizon-specific, treatment-policy explicit, and noncausal."
    if source_limitations:
        conclusion += " Direct wide-source results remain exploratory regardless of apparent metric improvement."
    bottom_right.text(0.02, 0.34, textwrap.fill(conclusion, width=40), fontsize=9.2, color=PALETTE["ink"], va="top")
    return figure


def page_renderers(data: Mapping[str, Any]) -> list[Callable[[], Any]]:
    return [
        lambda: render_page_00(data),
        lambda: render_page_01(data),
        lambda: render_page_02(data),
        lambda: render_page_03(data),
        lambda: render_page_04(data),
        lambda: render_page_05(data),
        lambda: render_page_06(data),
        lambda: render_page_07(data),
        lambda: render_page_08(data),
        lambda: render_page_09(data),
        lambda: performance_page(data, 10, "surgery", "bmi", "Surgical BMI performance"),
        lambda: performance_page(data, 11, "surgery", "hba1c", "Surgical HbA1c performance"),
        lambda: performance_page(data, 12, "incretin", "bmi", "Incretin BMI performance"),
        lambda: performance_page(data, 13, "incretin", "hba1c", "Incretin HbA1c performance"),
        lambda: render_page_14(data),
        lambda: render_page_15(data),
        lambda: render_page_16(data),
        lambda: render_page_17(data),
        lambda: render_page_18(data),
    ]


def validate_export_directory(export: Path, *, require_complete: bool = False) -> None:
    expected = set(PAGE_FILES) | {"metabolic_trajectory_figure_book.pdf"}
    present = {item.name for item in export.iterdir() if item.is_file()}
    unexpected = present.difference(expected)
    if unexpected:
        raise RuntimeError("FIGURES_TO_EXPORT contains non-contract files: " + ", ".join(sorted(unexpected)))
    if require_complete:
        missing = expected.difference(present)
        if missing:
            raise RuntimeError("FIGURES_TO_EXPORT is missing contract files: " + ", ".join(sorted(missing)))


def render_figure_book(data: Mapping[str, Any], export: Path) -> list[Path]:
    configure_figure_style()
    export.mkdir(parents=True, exist_ok=True)
    validate_export_directory(export)
    pdf_temporary = export / "metabolic_trajectory_figure_book.pdf.tmp"
    pdf_final = export / "metabolic_trajectory_figure_book.pdf"
    written: list[Path] = []
    with PdfPages(pdf_temporary, metadata={"Title": "Metabolic Trajectory Forecasting Study", "Author": "Brannigan Lab"}) as pdf:
        for filename, render in zip(PAGE_FILES, page_renderers(data), strict=True):
            figure = render()
            add_run_provenance(figure, data)
            temporary = export / (filename + ".tmp")
            figure.savefig(temporary, format="png", dpi=300, bbox_inches=None, facecolor=figure.get_facecolor())
            replace_file(temporary, export / filename)
            pdf.savefig(figure, dpi=300, facecolor=figure.get_facecolor())
            plt.close(figure)
            written.append(export / filename)
    replace_file(pdf_temporary, pdf_final)
    written.append(pdf_final)
    validate_export_directory(export, require_complete=True)
    return written


def new_schema_discovery_page(number: int, title: str, subtitle: str) -> Any:
    figure = plt.figure(figsize=(11, 8.5), constrained_layout=False)
    figure.patch.set_facecolor(PALETTE["paper"])
    figure.text(0.055, 0.947, f"{number:02d}", fontsize=22, fontweight="bold", color=PALETTE["blue"], va="top")
    figure.text(0.115, 0.947, title, fontsize=17, fontweight="bold", color=PALETTE["ink"], va="top")
    figure.text(0.115, 0.915, subtitle, fontsize=9.5, color=PALETTE["muted"], va="top")
    figure.lines.append(
        plt.Line2D([0.055, 0.945], [0.893, 0.893], transform=figure.transFigure, color=PALETTE["grid"], lw=1.0)
    )
    figure.text(
        0.055,
        0.025,
        "Read-only SQL Server metadata | No patient rows or values queried | Review before freezing joins",
        fontsize=7.5,
        color=PALETTE["muted"],
    )
    return figure


def _wrapped_metadata_lines(lines: Sequence[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for original in lines:
        line = str(original)
        if not line:
            wrapped.append("")
            continue
        leading = line[: len(line) - len(line.lstrip())]
        pieces = textwrap.wrap(
            line.strip(),
            width=max(20, width - len(leading)),
            initial_indent=leading,
            subsequent_indent=leading + "  ",
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.extend(pieces or [leading])
    return wrapped


def _metadata_text_panel(
    axis: Any,
    lines: Sequence[str],
    title: str,
    fontsize: float = 6.8,
    wrap_width: int | None = None,
) -> None:
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=PALETTE["ink"], pad=7)
    panel_lines = list(lines) if lines else ["No accessible metadata matched this section."]
    if wrap_width is not None:
        panel_lines = _wrapped_metadata_lines(panel_lines, wrap_width)
    rendered = "\n".join(str(line) for line in panel_lines)
    axis.text(
        0.0,
        0.98,
        rendered,
        transform=axis.transAxes,
        va="top",
        ha="left",
        family="DejaVu Sans Mono",
        fontsize=fontsize,
        color=PALETTE["ink"],
        linespacing=1.28,
        clip_on=True,
    )


def _candidate_metadata_blocks(
    report: Mapping[str, Any],
    domains: Sequence[str],
    max_objects: int,
    wrap_width: int = 66,
) -> list[list[str]]:
    candidates = report["candidates"]
    details = report["candidate_columns"]
    if candidates.empty:
        return []
    blocks: list[list[str]] = []
    for domain in domains:
        domain_candidates = candidates.loc[candidates["domain"].eq(domain)]
        chosen = domain_candidates.head(max_objects)
        for item in chosen.itertuples(index=False):
            block = [
                f"{str(item.domain).upper()}  {item.object} [{item.object_type}] "
                f"score={int(item.score)} core={item.core_coverage}"
            ]
            object_details = details.loc[
                details["domain"].eq(item.domain) & details["object"].eq(item.object)
            ] if not details.empty else pd.DataFrame()
            if object_details.empty:
                block.append("  table-name match only; no recognized role columns")
            else:
                for detail in object_details.head(10).itertuples(index=False):
                    key = f"; {detail.key_type}" if str(detail.key_type) else ""
                    block.append(f"  {detail.role:22} -> {detail.column} [{detail.data_type}{key}]")
                remaining = len(object_details) - min(len(object_details), 10)
                if remaining > 0:
                    block.append(f"  ... {remaining} more mapped columns in INTERNAL metadata")
            blocks.append(_wrapped_metadata_lines(block, wrap_width) + [""])
        omitted = len(domain_candidates) - len(chosen)
        if omitted > 0:
            blocks.append(
                _wrapped_metadata_lines(
                    [f"... {omitted} lower-ranked {domain} candidates are listed in INTERNAL/candidates.csv"],
                    wrap_width,
                )
                + [""]
            )
    return blocks


def _split_metadata_blocks(blocks: Sequence[Sequence[str]]) -> tuple[list[str], list[str]]:
    total_lines = sum(len(block) for block in blocks)
    target = (total_lines + 1) // 2
    left: list[str] = []
    right: list[str] = []
    for block in blocks:
        destination = left if not right and (len(left) + len(block) <= target or not left) else right
        destination.extend(block)
    return left, right


def render_schema_discovery_overview(report: Mapping[str, Any]) -> Any:
    database_frame = report["database"]
    database_name = (
        str(database_frame["DATABASE_NAME"].iloc[0])
        if not database_frame.empty and "DATABASE_NAME" in database_frame
        else "unavailable"
    )
    columns = report["columns"]
    objects = columns[["TABLE_SCHEMA", "TABLE_NAME", "TABLE_TYPE"]].drop_duplicates()
    schemas = sorted(objects["TABLE_SCHEMA"].astype(str).unique())
    candidates = report["candidates"]
    summary_lines = [
        f"Discovery contract : {SCHEMA_DISCOVERY_VERSION}",
        f"Database           : {database_name}",
        f"Accessible schemas : {len(schemas)} ({', '.join(schemas[:12])})",
        f"Tables/views       : {len(objects):,}",
        f"Columns            : {len(columns):,}",
        f"Declared keys      : {len(report['keys']):,}",
        f"Declared FKs       : {len(report['foreign_keys']):,}",
        f"Synonyms           : {len(report['synonyms']):,}",
        f"SQL dependency edges: {len(report['object_dependencies']):,}",
        f"Modules naming old cohorts: {len(report['cohort_modules']):,}",
        "",
        "This report ranks metadata candidates only. It never auto-selects a join or cohort index.",
    ]
    domain_lines = []
    for domain in DISCOVERY_DOMAIN_RULES:
        group = candidates.loc[candidates["domain"].eq(domain)] if not candidates.empty else pd.DataFrame()
        if group.empty:
            domain_lines.append(f"{domain:12} no candidate objects")
            continue
        best = group.iloc[0]
        domain_lines.append(
            f"{domain:12} {len(group):3d} candidates | best {best['object']} | core {best['core_coverage']}"
        )
    warning_lines = list(report.get("warnings", [])) or ["All optional metadata queries completed."]
    if not report["cohort_modules"].empty:
        warning_lines.extend(["", "Modules referencing old cohorts:"])
        for row in report["cohort_modules"].head(12).itertuples(index=False):
            warning_lines.append(f"  {row.MODULE_SCHEMA}.{row.MODULE_NAME} [{row.MODULE_TYPE}]")
    figure = new_schema_discovery_page(
        1,
        "Cosmos raw-source schema discovery",
        "Connection and metadata coverage, candidate domains, and recoverable cohort-building dependencies",
    )
    _metadata_text_panel(
        figure.add_axes([0.06, 0.53, 0.42, 0.31]),
        summary_lines,
        "A  Metadata scope",
        fontsize=7.5,
        wrap_width=64,
    )
    _metadata_text_panel(
        figure.add_axes([0.53, 0.53, 0.41, 0.31]),
        domain_lines,
        "B  Candidate domains",
        fontsize=7.3,
        wrap_width=62,
    )
    _metadata_text_panel(
        figure.add_axes([0.06, 0.09, 0.88, 0.33]),
        warning_lines,
        "C  Permissions and dependency clues",
        fontsize=7.3,
        wrap_width=130,
    )
    return figure


def render_schema_candidate_page(
    report: Mapping[str, Any],
    number: int,
    title: str,
    subtitle: str,
    domains: Sequence[str],
) -> Any:
    max_objects = 4 if len(domains) > 1 else 8
    blocks = _candidate_metadata_blocks(report, domains, max_objects=max_objects)
    left_lines, right_lines = _split_metadata_blocks(blocks)
    right_title = "Continued" if right_lines else "Additional candidates"
    if not right_lines:
        right_lines = ["No additional ranked candidate objects."]
    figure = new_schema_discovery_page(number, title, subtitle)
    _metadata_text_panel(
        figure.add_axes([0.055, 0.08, 0.43, 0.74]),
        left_lines,
        "Candidate objects and mapped columns",
        fontsize=6.5,
    )
    _metadata_text_panel(
        figure.add_axes([0.52, 0.08, 0.43, 0.74]),
        right_lines,
        right_title,
        fontsize=6.5,
    )
    return figure


def render_schema_key_page(report: Mapping[str, Any]) -> Any:
    relation_lines: list[str] = []
    foreign_keys = report["foreign_keys"]
    if not foreign_keys.empty:
        relation_lines.append("DECLARED FOREIGN KEYS")
        for row in foreign_keys.head(32).itertuples(index=False):
            relation_lines.append(
                f"{row.CHILD_SCHEMA}.{row.CHILD_TABLE}.{row.CHILD_COLUMN} -> "
                f"{row.PARENT_SCHEMA}.{row.PARENT_TABLE}.{row.PARENT_COLUMN}"
            )
    else:
        relation_lines.append("No declared foreign keys were visible. Shared column names below are hints only.")
    shared_lines = ["SHARED KEY-LIKE COLUMN NAMES"]
    for row in report["shared_keys"].head(36).itertuples(index=False):
        shared_lines.append(f"{row.normalized_column} ({int(row.object_count)} objects): {row.objects}")
    dependency_lines: list[str] = []
    object_dependencies = report["object_dependencies"]
    if not object_dependencies.empty:
        dependency_lines.append("SQL OBJECT DEPENDENCIES")
        dependency_working = object_dependencies.copy()
        names = (
            dependency_working["REFERENCING_OBJECT"].astype(str)
            + " "
            + dependency_working["REFERENCED_OBJECT"].astype(str)
        )
        dependency_working["_cohort_priority"] = names.str.contains(
            r"mbscohort|glp1cohort", case=False, regex=True, na=False
        )
        dependency_working = dependency_working.sort_values(
            ["_cohort_priority", "REFERENCING_SCHEMA", "REFERENCING_OBJECT"],
            ascending=[False, True, True],
        )
        for row in dependency_working.head(20).itertuples(index=False):
            referenced_parts = [
                str(value)
                for value in (row.REFERENCED_DATABASE, row.REFERENCED_SCHEMA, row.REFERENCED_OBJECT)
                if value is not None and str(value).lower() not in {"", "nan", "none"}
            ]
            dependency_lines.append(
                f"{row.REFERENCING_SCHEMA}.{row.REFERENCING_OBJECT} [{row.REFERENCING_TYPE}] -> "
                + ".".join(referenced_parts)
            )
    synonyms = report["synonyms"]
    if not synonyms.empty:
        dependency_lines.extend(["", "SYNONYMS"])
        for row in synonyms.head(24).itertuples(index=False):
            dependency_lines.append(f"{row.SYNONYM_SCHEMA}.{row.SYNONYM_NAME} -> {row.BASE_OBJECT_NAME}")
    modules = report["cohort_modules"]
    if not modules.empty:
        dependency_lines.extend(["", "MODULES THAT NAME THE OLD COHORTS"])
        for row in modules.head(24).itertuples(index=False):
            dependency_lines.append(f"{row.MODULE_SCHEMA}.{row.MODULE_NAME} [{row.MODULE_TYPE}]")
    if not dependency_lines:
        dependency_lines.append("No SQL dependencies, synonyms, or cohort-referencing modules were visible.")
    figure = new_schema_discovery_page(
        7,
        "Key and dependency map",
        "Declared relationships are authoritative; shared-name matches are review prompts, never automatic joins",
    )
    _metadata_text_panel(
        figure.add_axes([0.055, 0.49, 0.89, 0.35]),
        relation_lines,
        "A  Declared relationships",
        fontsize=6.6,
        wrap_width=132,
    )
    _metadata_text_panel(
        figure.add_axes([0.055, 0.08, 0.43, 0.33]),
        shared_lines,
        "B  Shared key hints",
        fontsize=6.3,
        wrap_width=64,
    )
    _metadata_text_panel(
        figure.add_axes([0.52, 0.08, 0.43, 0.33]),
        dependency_lines,
        "C  Synonyms and modules",
        fontsize=6.3,
        wrap_width=64,
    )
    return figure


def schema_discovery_renderers(report: Mapping[str, Any]) -> list[Callable[[], Any]]:
    return [
        lambda: render_schema_discovery_overview(report),
        lambda: render_schema_candidate_page(
            report, 2, "Patient and center candidates", "Demographics, observability, organization identity, and patient keys", ("patients", "centers")
        ),
        lambda: render_schema_candidate_page(
            report, 3, "Procedure candidates", "Qualifying bariatric concepts, exact dates, patient keys, and organization lineage", ("procedures",)
        ),
        lambda: render_schema_candidate_page(
            report, 4, "Medication candidates", "Ingredient or coded product, dated orders/fills/administrations, and coverage evidence", ("medications",)
        ),
        lambda: render_schema_candidate_page(
            report, 5, "Measurement candidates", "BMI, weight, height, and HbA1c values with exact dates, units, and concepts", ("measurements",)
        ),
        lambda: render_schema_candidate_page(
            report, 6, "Encounter and diagnosis candidates", "Observation history, center attribution, comorbidity definitions, and event dates", ("encounters", "diagnoses")
        ),
        lambda: render_schema_key_page(report),
    ]


def render_schema_discovery_book(report: Mapping[str, Any], export: Path) -> list[Path]:
    configure_figure_style()
    export.mkdir(parents=True, exist_ok=True)
    expected = set(SCHEMA_DISCOVERY_PAGE_FILES) | {"schema_discovery_figure_book.pdf"}
    present = {item.name for item in export.iterdir() if item.is_file()}
    unexpected = present.difference(expected)
    if unexpected:
        raise RuntimeError("Schema discovery export contains unexpected files: " + ", ".join(sorted(unexpected)))
    temporary_pdf = export / "schema_discovery_figure_book.pdf.tmp"
    final_pdf = export / "schema_discovery_figure_book.pdf"
    written: list[Path] = []
    with PdfPages(
        temporary_pdf,
        metadata={"Title": "Cosmos Raw-Source Schema Discovery", "Author": "Brannigan Lab"},
    ) as pdf:
        for filename, render in zip(SCHEMA_DISCOVERY_PAGE_FILES, schema_discovery_renderers(report), strict=True):
            figure = render()
            temporary_png = export / (filename + ".tmp")
            figure.savefig(temporary_png, format="png", dpi=300, facecolor=figure.get_facecolor())
            replace_file(temporary_png, export / filename)
            pdf.savefig(figure, dpi=300, facecolor=figure.get_facecolor())
            plt.close(figure)
            written.append(export / filename)
    replace_file(temporary_pdf, final_pdf)
    written.append(final_pdf)
    final_present = {item.name for item in export.iterdir() if item.is_file()}
    if final_present != expected:
        raise RuntimeError("Schema discovery export contract mismatch")
    return written


def schema_discovery_run_dir(cfg: RunConfig) -> Path:
    if cfg.output_dir:
        return Path(cfg.output_dir).expanduser().resolve()
    return timestamped_default_output_dir()


def _atomic_metadata_csv(path: Path, frame: Any) -> None:
    atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def run_schema_discovery(cfg: RunConfig, dependencies: Mapping[str, Any]) -> Path:
    run_dir = schema_discovery_run_dir(cfg)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PreflightError(
            "Schema discovery output directory is not empty",
            [str(run_dir)],
            ["Choose a new --output-dir so metadata from separate database states cannot be mixed."],
        )
    connection = connect_cosmos()
    try:
        report = discover_cosmos_schema(connection)
    finally:
        connection.close()
    internal = run_dir / "INTERNAL"
    export = run_dir / "FIGURES_TO_EXPORT"
    internal.mkdir(parents=True, exist_ok=True)
    export.mkdir(parents=True, exist_ok=True)
    for name in (
        "columns", "keys", "foreign_keys", "synonyms", "cohort_modules", "object_dependencies", "candidates",
        "candidate_columns", "shared_keys",
    ):
        _atomic_metadata_csv(internal / f"{name}.csv", report[name])
    rendered = render_schema_discovery_book(report, export)
    database_frame = report["database"]
    database_name = (
        str(database_frame["DATABASE_NAME"].iloc[0])
        if not database_frame.empty and "DATABASE_NAME" in database_frame
        else None
    )
    manifest = {
        "status": "schema_discovery_complete",
        "schema_discovery_version": SCHEMA_DISCOVERY_VERSION,
        "study_version": STUDY_VERSION,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "generated_utc": utc_now(),
        "database_name": database_name,
        "object_count": int(report["columns"][["TABLE_SCHEMA", "TABLE_NAME"]].drop_duplicates().shape[0]),
        "column_count": int(len(report["columns"])),
        "candidate_count": int(len(report["candidates"])),
        "warnings": list(report["warnings"]),
        "dependencies": dict(dependencies),
        "privacy": "SQL Server metadata only; no patient rows or values queried",
        "export_files": [path.name for path in rendered],
    }
    atomic_json(run_dir / "schema_discovery_manifest.json", manifest)
    return run_dir


# ======================================================================================
# 10. Embedded deterministic end-to-end and numerical tests
# ======================================================================================


def run_embedded_self_tests() -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append({"test": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail or 'assertion failed'}")

    def record(start: int, end: int, ingredient: str = "semaglutide", patient: str = "P") -> CoverageRecord:
        return CoverageRecord(
            patient_id=patient,
            start_day=start,
            end_day=end,
            ingredient=ingredient,
            therapy_class=INCRETIN_INGREDIENTS[ingredient],
            source_type="fill",
            source_table="synthetic.fill",
        )

    # 1. A 182-day preoperative episode remains operationally naive and subthreshold.
    history_182 = classify_surgical_incretin_history([record(-181, 0)])
    check(
        "01_182_day_preoperative_episode",
        history_182["operationally_naive"] and history_182["classification"] == "subthreshold_continuous_exposure",
        str(history_182),
    )

    # 2. A completed 183-day preoperative episode is excluded as previously treated.
    history_183 = classify_surgical_incretin_history([record(-182, 0)])
    check(
        "02_183_day_preoperative_episode",
        not history_183["operationally_naive"] and history_183["classification"] == "previously_treated",
        str(history_183),
    )

    # 3. Active subthreshold preoperative exposure censors at day zero.
    active_history = classify_surgical_incretin_history([record(-50, 25)])
    check(
        "03_active_subthreshold_day_zero_censor",
        active_history["classification"] == "subthreshold_continuous_exposure" and active_history["treatment_censor_day"] == 0,
        str(active_history),
    )

    # 4. Same-day postoperative initiation excludes the outcome regardless of eventual duration.
    postoperative = classify_surgical_incretin_history([record(100, 120)], postoperative_flag=True)
    index = pd.Timestamp("2020-01-01")
    simple_measurements = pd.DataFrame(
        [{"patient_id": "P", "measurement_date": index + pd.Timedelta(days=100), "outcome": "bmi", "value": 40.0}]
    )
    selected_same_day = select_target_measurement(simple_measurements, "bmi", index, 3, censor_day=postoperative["treatment_censor_day"])
    check(
        "04_same_day_postoperative_start_excludes_measurement",
        postoperative["treatment_censor_day"] == 100 and selected_same_day is None,
        str(postoperative),
    )

    # 5. A postoperative flag without a resolvable start is a primary exclusion.
    unresolved = classify_surgical_incretin_history([], postoperative_flag=True)
    check(
        "05_postoperative_flag_missing_start",
        unresolved["classification"] == "unknown" and unresolved["unresolved_postoperative_start"],
        str(unresolved),
    )

    # 6. Overlapping same-ingredient fills carry stockpile forward.
    stockpiled = carry_stockpile_forward([record(0, 29), record(20, 49)])
    check(
        "06_overlapping_fills_stockpile",
        stockpiled[-1].end_day == 59 and interval_coverage_days(merge_supported_intervals(stockpiled), 0, 59) == 60,
        str([(item.start_day, item.end_day) for item in stockpiled]),
    )

    explicit_intervals = [
        CoverageRecord(
            patient_id="P",
            start_day=start,
            end_day=end,
            ingredient="semaglutide",
            therapy_class=INCRETIN_INGREDIENTS["semaglutide"],
            source_type="explicit_treatment",
            source_table="synthetic.wide",
        )
        for start, end in ((0, 29), (20, 49))
    ]
    explicit_adjusted = carry_stockpile_forward(explicit_intervals)
    check(
        "06b_explicit_intervals_do_not_stockpile",
        explicit_adjusted[-1].end_day == 49,
        str([(item.start_day, item.end_day) for item in explicit_adjusted]),
    )

    # 7. A 31-day uncovered gap splits episodes under the primary rule.
    gap_episodes, _ = reconstruct_coverage_episodes([record(0, 40), record(72, 182)])
    check(
        "07_31_day_gap_fails_primary",
        len(gap_episodes) == 2 and not any(item.qualifies_183 for item in gap_episodes),
        str([(item.maximum_gap_days, item.qualifies_183) for item in gap_episodes]),
    )

    # 8. Within-class ingredient switching within the gap remains one episode.
    switch_episodes, _ = reconstruct_coverage_episodes([record(0, 90, "liraglutide"), record(100, 182, "semaglutide")])
    check(
        "08_within_class_switch_continues_episode",
        len(switch_episodes) == 1 and switch_episodes[0].qualifies_183 and len(switch_episodes[0].switch_days) == 1,
        str(asdict(switch_episodes[0])),
    )

    # 9. PDC below 0.80 fails even though no individual gap exceeds 30 days.
    low_pdc_records = [record(0, 30), record(61, 91), record(122, 152), record(183, 200)]
    low_pdc, _ = reconstruct_coverage_episodes(low_pdc_records)
    check(
        "09_low_pdc_fails",
        len(low_pdc) == 1 and low_pdc[0].maximum_gap_days <= 30 and low_pdc[0].pdc_183 < 0.80 and not low_pdc[0].qualifies_183,
        str(asdict(low_pdc[0])),
    )

    # 10. Bariatric surgery after month 12 excludes same-day and later incretin outcomes.
    day_400 = index + pd.Timedelta(days=400)
    surgery_measurements = pd.DataFrame(
        [
            {"patient_id": "P", "measurement_date": day_400, "outcome": "hba1c", "value": 7.0},
            {"patient_id": "P", "measurement_date": index + pd.Timedelta(days=410), "outcome": "hba1c", "value": 6.9},
        ]
    )
    surgery_target = select_target_measurement(surgery_measurements, "hba1c", index, 12, censor_day=400)
    check("10_incretin_post_surgery_target_exclusion", surgery_target is None, str(surgery_target))

    # 11. The same patient in both source cohorts receives one global split.
    duplicate_cohorts = pd.DataFrame(
        [
            {"patient_id": "DUP", "center_id": "C1", "index_date": pd.Timestamp("2018-01-01"), "cohort": "surgery"},
            {"patient_id": "DUP", "center_id": "C1", "index_date": pd.Timestamp("2020-01-01"), "cohort": "incretin"},
            {"patient_id": "A", "center_id": "C2", "index_date": pd.Timestamp("2017-01-01"), "cohort": "surgery"},
            {"patient_id": "B", "center_id": "C3", "index_date": pd.Timestamp("2019-01-01"), "cohort": "surgery"},
            {"patient_id": "C", "center_id": "C4", "index_date": pd.Timestamp("2021-01-01"), "cohort": "incretin"},
        ]
    )
    duplicate_split, _ = assign_global_splits(duplicate_cohorts)
    check(
        "11_global_split_consistency",
        duplicate_split.loc[duplicate_split["patient_id"].eq("DUP"), "split"].nunique() == 1,
        str(duplicate_split.loc[duplicate_split["patient_id"].eq("DUP"), "split"].tolist()),
    )

    # 12. A feature one day after the landmark is rejected by the leakage audit.
    leaking_row = pd.DataFrame(
        [
            {
                "patient_id": "P", "split": "train", "center_id": "C", "feature_max_day": 91,
                "origin_day": 90, "target_day": 180, "target_observed": True,
                "effective_censor_day": np.nan,
            }
        ]
    )
    leaked = False
    try:
        leakage_audit(leaking_row, {"heldout_centers": []})
    except LeakageError:
        leaked = True
    check("12_future_feature_leakage_rejected", leaked)

    # 13. IFCC conversion is exact and unsupported units are invalid.
    converted, converted_status = normalize_hba1c(53.0, "mmol/mol")
    invalid_hba1c, invalid_status = normalize_hba1c(7.0, "mg/dL")
    check(
        "13_hba1c_unit_normalization",
        abs(float(converted) - hba1c_ifcc_to_ngsp(53.0)) < 1e-12 and converted_status == "valid" and invalid_hba1c is None and invalid_status == "invalid_or_missing_unit",
    )

    # 14. Outliers remain invalid, duplicate days are summarized, and target ties go earlier.
    raw_quality = pd.DataFrame(
        [
            {"patient_id": "P", "measurement_date": index, "measurement_type": "bmi", "raw_value": 9, "unit": "kg/m2", "source_concept": "BMI", "source_table": "vitals"},
            {"patient_id": "P", "measurement_date": index, "measurement_type": "hba1c", "raw_value": 25, "unit": "%", "source_concept": "HbA1c", "source_table": "labs"},
            {"patient_id": "P", "measurement_date": index + pd.Timedelta(days=86), "measurement_type": "bmi", "raw_value": 39, "unit": "kg/m2", "source_concept": "BMI", "source_table": "vitals"},
            {"patient_id": "P", "measurement_date": index + pd.Timedelta(days=86), "measurement_type": "bmi", "raw_value": 41, "unit": "kg/m2", "source_concept": "BMI", "source_table": "vitals"},
            {"patient_id": "P", "measurement_date": index + pd.Timedelta(days=96), "measurement_type": "bmi", "raw_value": 38, "unit": "kg/m2", "source_concept": "BMI", "source_table": "vitals"},
        ]
    )
    normalized_quality, quality_audit = normalize_measurements(raw_quality)
    tie_target = select_target_measurement(normalized_quality, "bmi", index, 3)
    check(
        "14_outliers_duplicates_and_tie_break",
        int((quality_audit["reason"] == "outside_plausible_range").sum()) == 2
        and bool(normalized_quality.loc[normalized_quality["measurement_date"].eq(index + pd.Timedelta(days=86)), "duplicate_day"].iloc[0])
        and tie_target is not None and tie_target["day"] == 86 and abs(tie_target["value"] - 40.0) < 1e-12,
        str(tie_target),
    )

    # 15. A 60-month row without opportunity is administratively immature, not missing outcome.
    maturity_status = target_support_status(index, index + pd.Timedelta(days=1000), index + pd.Timedelta(days=1000), 60, None, False)
    check("15_administrative_immaturity", maturity_status == "administratively_immature", maturity_status)

    # 16. Crossing quantiles are rearranged.
    crossed = rearrange_quantiles([[5, 4, 3, 2, 1, 0, -1]])
    check("16_quantile_rearrangement", bool(np.all(np.diff(crossed, axis=1) >= 0)), str(crossed))

    # 17. Resume rejects stale and incomplete checkpoints.
    with tempfile.TemporaryDirectory(prefix="metabolic-checkpoint-test-") as directory:
        resume_cfg = RunConfig(mode="self-test", output_dir=directory, resume=True)
        context = RunContext(resume_cfg, Path(directory), "fingerprint", {"test": True})
        context.initialize()
        payload = pd.DataFrame({"patient_id": ["hashed"], "value": [1]})
        context.save_checkpoint("stage", payload, elapsed_seconds=1.25)
        elapsed_recorded = context.state["stages"]["stage"]["seconds"] == 1.25
        valid_load = context.load_checkpoint("stage") is not None
        metadata_path = context.checkpoints / "stage.json"
        metadata = read_json(metadata_path, {})
        metadata["completion_marker"] = "PARTIAL"
        atomic_json(metadata_path, metadata)
        incomplete_rejected = context.load_checkpoint("stage") is None
        metadata["completion_marker"] = "COMPLETE"
        metadata["stage_fingerprint"] = "stale"
        atomic_json(metadata_path, metadata)
        stale_rejected = context.load_checkpoint("stage") is None
        check(
            "17_checkpoint_resume_validation",
            elapsed_recorded and valid_load and incomplete_rejected and stale_rejected,
        )

    # 18. Small aggregate cells are suppressed.
    suppressed = suppress_small_cells(pd.DataFrame({"group": ["rare", "common"], "n": [10, 11], "estimate": [1.2, 2.3]}), ["n"])
    check(
        "18_small_cell_suppression",
        bool(suppressed.loc[0, "small_cell_suppressed"]) and pd.isna(suppressed.loc[0, "n"]) and not bool(suppressed.loc[1, "small_cell_suppressed"]),
    )
    table_fixture = pd.DataFrame(
        {
            "origin_month": [0, 0],
            "target_month": [12, 24],
            "n": [5, 12],
            "effective_sample_size": [4.8, 11.5],
            "estimability": ["estimable", "estimable"],
            "suppressed": [False, False],
        }
    )
    table_figure, table_axis = plt.subplots()
    draw_compact_table(table_axis, table_fixture, list(table_fixture.columns))
    table_text = [
        cell.get_text().get_text() for cell in table_axis.tables[0].get_celld().values()
    ]
    plt.close(table_figure)
    check(
        "18b_figure_table_small_cell_suppression",
        "<11" in table_text
        and "4.800" not in table_text
        and "Suppressed" in table_text,
        str(table_text),
    )

    selection_fixture = {
        "selected": pd.DataFrame(
            {
                "cohort": ["surgery", "surgery"],
                "outcome": ["bmi", "bmi"],
                "origin_month": [0, 3],
                "selected_candidate": ["candidate_a", "candidate_b"],
            }
        ),
        "metrics": pd.DataFrame(
            {
                "cohort": ["surgery"] * 4,
                "outcome": ["bmi"] * 4,
                "origin_month": [0, 0, 3, 3],
                "candidate": ["candidate_a", "candidate_b", "candidate_a", "candidate_b"],
                "n": [5, 20, 20, 20],
                "crps": [1.0, 2.0, 3.0, 4.0],
            }
        ),
    }
    selected_fixture_metrics = selected_only_metric_frame(selection_fixture).sort_values("origin_month")
    masked_fixture_metrics = mask_small_cell_metrics(selected_fixture_metrics, ["crps"])
    check(
        "18c_selected_origin_and_plot_disclosure_control",
        selected_fixture_metrics["candidate"].tolist() == ["candidate_a", "candidate_b"]
        and pd.isna(masked_fixture_metrics.iloc[0]["crps"])
        and masked_fixture_metrics.iloc[1]["crps"] == 4.0,
        str(masked_fixture_metrics.to_dict(orient="records")),
    )

    scale_rows: list[dict[str, Any]] = []
    for target_month, training_values in ((3, [0.0, 0.0, 2.0, 2.0]), (12, [0.0, 0.0, 200.0, 200.0])):
        for index, target_value in enumerate(training_values):
            row = {
                "row_id": f"train-{target_month}-{index}",
                "patient_id": f"train-{target_month}-{index}",
                "cohort": "surgery",
                "outcome": "bmi",
                "origin_month": 0,
                "target_month": target_month,
                "candidate": "candidate_a",
                "architecture": "architecture_a",
                "split": "train",
                "target_observed": True,
                "target_value": target_value,
                "analysis_weight": 1.0,
            }
            row.update({column: target_value for column in QUANTILE_COLUMNS})
            scale_rows.append(row)
    for candidate, architecture, predictions_by_target in (
        ("candidate_a", "architecture_a", {3: 0.0, 12: 100.0}),
        ("candidate_b", "architecture_b", {3: 3.0, 12: 0.0}),
    ):
        for target_month, prediction in predictions_by_target.items():
            for index in range(11):
                row = {
                    "row_id": f"validation-{target_month}-{index}",
                    "patient_id": f"validation-{target_month}-{index}",
                    "cohort": "surgery",
                    "outcome": "bmi",
                    "origin_month": 0,
                    "target_month": target_month,
                    "candidate": candidate,
                    "architecture": architecture,
                    "split": "validation",
                    "target_observed": True,
                    "target_value": 0.0,
                    "analysis_weight": 1.0,
                }
                row.update({column: prediction for column in QUANTILE_COLUMNS})
                scale_rows.append(row)
    standardized_fixture = candidate_validation_scores(pd.DataFrame(scale_rows)).set_index("candidate")
    check(
        "18d_equal_horizon_standardized_selection_score",
        standardized_fixture.loc["candidate_a", "validation_crps"]
        > standardized_fixture.loc["candidate_b", "validation_crps"]
        and standardized_fixture.loc["candidate_a", "validation_standardized_crps"]
        < standardized_fixture.loc["candidate_b", "validation_standardized_crps"],
        str(standardized_fixture[["validation_crps", "validation_standardized_crps"]].to_dict(orient="index")),
    )

    wide_smoke_display = {
        "identity": {
            "run_mode": "smoke",
            "preflight": {"strict_raw_event_contract": False},
        }
    }
    check(
        "18e_screenshot_provenance_and_source_language",
        PAGE_FILES[0] == "00_executive_summary.png"
        and run_badge(wide_smoke_display)[0] == "SMOKE - NON-INFERENTIAL"
        and "reported ≥183-day interval cohort"
        in source_aware_text(wide_smoke_display, "primary six-month continuers"),
        f"pages={PAGE_FILES[:2]}; badge={run_badge(wide_smoke_display)[0]}",
    )

    # 19. A forced preflight failure creates exactly one PNG and a one-page PDF.
    with tempfile.TemporaryDirectory(prefix="metabolic-preflight-test-") as directory:
        failure_cfg = RunConfig(mode="self-test", output_dir=directory)
        partial_export = Path(directory) / "FIGURES_TO_EXPORT"
        partial_export.mkdir(parents=True)
        atomic_bytes(partial_export / PAGE_FILES[0], b"partial run artifact")
        failure_png = render_preflight_failure(failure_cfg, "Forced gate failure", ["exact units missing"], ["Provide raw unit fields"])
        failure_pdf = failure_png.parent / "metabolic_trajectory_figure_book.pdf"
        check(
            "19_figure_only_failure_report",
            failure_png.exists() and failure_pdf.exists() and {item.name for item in failure_png.parent.iterdir()} == {failure_png.name, failure_pdf.name},
        )

    if importlib.util.find_spec("torch") is None:
        raise AssertionError("PyTorch is required for embedded neural and RK4 self-tests")
    import torch

    # 20. Float64 RK4 accuracy and autograd match dh/dt = a*h over five years.
    a = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    initial = torch.tensor([[1.3]], dtype=torch.float64)

    def analytic_field(_time: Any, state: Any, _context: Any) -> Any:
        return a * state

    rk_state, _ = rk4_integrate(analytic_field, initial, torch.zeros_like(initial), 0.0, 5.0, 1.0 / 12.0)
    expected_state = initial * torch.exp(a * 5.0)
    relative_error = float(torch.max(torch.abs((rk_state - expected_state) / expected_state)).detach())
    rk_state.sum().backward()
    analytic_gradient = float((5.0 * expected_state).sum().detach())
    gradient_error = abs(float(a.grad) - analytic_gradient) / abs(analytic_gradient)
    check(
        "20_rk4_analytic_accuracy_and_autograd",
        relative_error <= 1e-4 and gradient_error <= 1e-4,
        f"state relative error={relative_error:.3e}; gradient relative error={gradient_error:.3e}",
    )

    # 21. Zero interval is identity, negative is rejected, and noninteger intervals land exactly.
    zero_state, zero_steps = rk4_integrate(analytic_field, initial, torch.zeros_like(initial), 1.0, 1.0)
    negative_rejected = False
    try:
        rk4_integrate(analytic_field, initial, torch.zeros_like(initial), 1.0, 0.9)
    except ValueError:
        negative_rejected = True

    def constant_field(_time: Any, state: Any, _context: Any) -> Any:
        return torch.ones_like(state)

    landed, landed_steps = rk4_integrate(constant_field, torch.zeros_like(initial), torch.zeros_like(initial), 0.13, 0.987, 0.2)
    check(
        "21_rk4_interval_edge_cases",
        zero_steps == 0 and torch.equal(zero_state, initial) and negative_rejected and abs(float(landed) - (0.987 - 0.13)) < 1e-12 and landed_steps == math.ceil((0.987 - 0.13) / 0.2),
    )

    # 22. Future events and events on the censor date never enter the ODE-RNN state.
    set_deterministic_seed(SEED, include_torch=True)
    ode_model = build_ode_rnn(2, [3], latent_dim=8, context_dim=8, field_width=16, decoder_width=16)
    ode_model.eval()
    static = torch.zeros((1, 2))
    categories = [torch.zeros(1, dtype=torch.long)]
    times = torch.tensor([[0.5, 1.0, 1.5]])
    values_a = torch.tensor([[0.2, 3.0, 9.0]])
    values_b = torch.tensor([[0.2, -30.0, -90.0]])
    masks = torch.ones((1, 3), dtype=torch.bool)
    arguments = (static, categories, torch.zeros(1), torch.zeros(1), times)
    with torch.no_grad():
        output_a, diagnostic_a = ode_model(*arguments, values_a, masks, torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([1.0]), max_step=0.25)
        output_b, diagnostic_b = ode_model(*arguments, values_b, masks, torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([1.0]), max_step=0.25)
    check(
        "22_ode_future_and_censor_event_filtering",
        torch.allclose(output_a, output_b, atol=1e-7, rtol=1e-7) and diagnostic_a["accepted_events"] == [1] and diagnostic_b["accepted_events"] == [1],
        str(diagnostic_a),
    )

    # 23. Padded batch and individual sequence evaluation agree.
    batch_static = torch.tensor([[0.1, 0.2], [-0.2, 0.3]])
    batch_categories = [torch.tensor([1, 2], dtype=torch.long)]
    batch_times = torch.tensor([[0.25, 0.75, 0.0], [0.4, 0.0, 0.0]])
    batch_values = torch.tensor([[0.1, 0.2, 0.0], [-0.1, 0.0, 0.0]])
    batch_masks = torch.tensor([[True, True, False], [True, False, False]])
    with torch.no_grad():
        batch_output, _ = ode_model(
            batch_static, batch_categories, torch.zeros(2), torch.zeros(2), batch_times, batch_values, batch_masks,
            torch.tensor([0.8, 0.8]), torch.tensor([1.2, 1.2]), torch.tensor([3.0, 3.0]), max_step=0.25,
        )
        individual_outputs = []
        for patient_index in range(2):
            individual, _ = ode_model(
                batch_static[patient_index:patient_index + 1], [batch_categories[0][patient_index:patient_index + 1]],
                torch.zeros(1), torch.zeros(1), batch_times[patient_index:patient_index + 1],
                batch_values[patient_index:patient_index + 1], batch_masks[patient_index:patient_index + 1],
                torch.tensor([0.8]), torch.tensor([1.2]), torch.tensor([3.0]), max_step=0.25,
            )
            individual_outputs.append(individual)
    check(
        "23_ode_padded_batch_equivalence",
        torch.allclose(batch_output, torch.cat(individual_outputs), atol=1e-6, rtol=1e-6),
        f"max difference={float(torch.max(torch.abs(batch_output - torch.cat(individual_outputs)))):.3e}",
    )

    # 24. Extreme decoder outputs remain ordered for both neural architectures.
    extreme = torch.tensor([[1000.0, -1000.0, 800.0, -900.0, 700.0, -800.0, 600.0]])
    ordered_extreme = noncrossing_quantiles_torch(extreme)
    mlp_model = build_quantile_mlp(2, [3], width=8, depth=2, dropout=0.0)
    with torch.no_grad():
        mlp_output = mlp_model(torch.tensor([[1e6, -1e6]]), [torch.tensor([1])])
    check(
        "24_neural_noncrossing_extremes",
        bool(torch.all(torch.diff(ordered_extreme, dim=1) >= 0)) and bool(torch.all(torch.diff(mlp_output, dim=1) >= 0)) and bool(torch.all(torch.diff(output_a, dim=1) >= 0)),
    )

    # 25. Monthly and half-month steps produce the prespecified sensitivity metrics.
    with torch.no_grad():
        monthly, _ = ode_model(
            static, categories, torch.zeros(1), torch.zeros(1), times, values_a, masks,
            torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]), max_step=1.0 / 12.0,
        )
        half_monthly, _ = ode_model(
            static, categories, torch.zeros(1), torch.zeros(1), times, values_a, masks,
            torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]), max_step=1.0 / 24.0,
        )
    sensitivity_metrics = solver_sensitivity_metrics(monthly.numpy(), half_monthly.numpy(), 5.0, y_true=np.array([0.0]))
    check(
        "25_solver_step_sensitivity_metrics",
        set(sensitivity_metrics) == {"median_patient_iqr_fraction", "p99_iqr_fraction", "crps_relative_change"} and all(math.isfinite(value) for value in sensitivity_metrics.values()),
        str(sensitivity_metrics),
    )

    # 26. Fixed seed gives deterministic CPU inference; accelerator tolerance is recorded if available.
    set_deterministic_seed(77, include_torch=True)
    deterministic_a = build_quantile_mlp(2, [3], width=8, depth=2, dropout=0.0)
    set_deterministic_seed(77, include_torch=True)
    deterministic_b = build_quantile_mlp(2, [3], width=8, depth=2, dropout=0.0)
    test_numeric = torch.tensor([[0.4, -0.2]])
    test_category = [torch.tensor([1])]
    with torch.no_grad():
        cpu_a = deterministic_a(test_numeric, test_category)
        cpu_b = deterministic_b(test_numeric, test_category)
    accelerator_tolerance = "not_available"
    accelerator_ok = True
    if torch.cuda.is_available():
        device = torch.device("cuda")
        deterministic_a_gpu = deterministic_a.to(device)
        with torch.no_grad():
            gpu = deterministic_a_gpu(test_numeric.to(device), [test_category[0].to(device)]).cpu()
        difference = float(torch.max(torch.abs(cpu_a - gpu)))
        accelerator_tolerance = f"cuda max absolute difference={difference:.3e}"
        accelerator_ok = difference <= 1e-4
    check(
        "26_deterministic_neural_inference",
        torch.equal(cpu_a, cpu_b) and accelerator_ok,
        accelerator_tolerance,
    )

    # 27. Wide-only and sparse repeated-measure datasets fail the ODE suitability gate.
    gate_cohorts = pd.DataFrame(
        [
            {"patient_id": f"G{index}", "cohort": "surgery", "split": "train", "treatment": "sleeve", "index_date": index}
            for index in range(20)
        ]
    )
    sparse_measurements = pd.DataFrame(
        [
            {"patient_id": f"G{index}", "outcome": "bmi", "measurement_date": pd.Timestamp("2020-01-01"), "value": 40.0}
            for index in range(20)
        ]
    )
    dependency_stub = {"torch_importable": True}
    sparse_gate = ode_suitability_gates(gate_cohorts, sparse_measurements, dependency_stub)
    wide_only = sparse_measurements.drop(columns="measurement_date")
    wide_gate = ode_suitability_gates(gate_cohorts, wide_only, dependency_stub)
    check(
        "27_ode_suitability_rejects_wide_and_sparse",
        not bool(sparse_gate["appropriate"].any()) and not bool(wide_gate["appropriate"].any()),
    )

    # 28. The source imports no forbidden ODE or project-local modeling dependencies.
    source_tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(source_tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    forbidden_roots = {"torchdiffeq", "scipy", "jax", "tensorflow"}
    forbidden_imports = [module for module in imported_modules if module.split(".")[0] in forbidden_roots]
    project_local_imports = [
        module for module in imported_modules
        if module.startswith(("qreg_improvement", "train_", "figures", "distributional_metrics"))
    ]
    check(
        "28_forbidden_dependency_source_assertion",
        not forbidden_imports and not project_local_imports,
        f"forbidden={forbidden_imports}; project_local={project_local_imports}",
    )

    # 29. The production query path opens one connection, queries both direct source
    # tables itself, keeps complete bounded patient histories, and closes the connection.
    mbs_fixture = pd.DataFrame(
        [
            {
                "PatKey": "DIRECT-MBS",
                "CptCode": "43775",
                "ProcDateValue": "2020-01-15",
                "AgeAtEvent": 45,
                "PriorGLP1": 0,
                "PostOpGLP1": 0,
                "PMH_PriorMBS": 0,
                "PMH_dialysis_transplant": 0,
                "BMIatEvent": 42.0,
                "BMI12mPostEvent": 31.0,
                "ActiveEndInterval": 900,
            }
        ]
    )
    glp1_fixture = pd.DataFrame(
        [
            {
                "PatKey": "DIRECT-GLP1",
                "AgeAtEvent": 50,
                "PriorGLP1": 0,
                "GLP1StartDate": "2020-02-01",
                "GLP1EndDate": "2021-03-06",
                "GLP1Duration": 400,
                "GLP1Name": "semaglutide",
                "GLP1Route": "subcutaneous",
                "MostRecentDose": 1.0,
                "MaxGLP1Dose": 2.4,
                "PMH_PriorMBS": 0,
                "MBSduringGLP1": 0,
                "PMH_dialysis_transplant": 0,
                "BMIatEvent": 38.0,
                "BMI12mPostEvent": 32.0,
                "ActiveEndInterval": 900,
            }
        ]
    )

    class DirectFixtureConnection:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    direct_connection = DirectFixtureConnection()
    captured_sql: list[str] = []

    def direct_fixture_reader(sql: str, connection: Any) -> Any:
        captured_sql.append(sql)
        if connection is not direct_connection:
            raise AssertionError("query used an unexpected connection")
        if "aggregate source count" in sql and "[MBSCohort]" in sql:
            return pd.DataFrame([{"source_rows": 5_500, "source_patients": 5_000}])
        if "aggregate source count" in sql and "[GLP1Cohort]" in sql:
            return pd.DataFrame([{"source_rows": 6_500, "source_patients": 6_000}])
        if "[MBSCohort]" in sql:
            return mbs_fixture.copy()
        if "[GLP1Cohort]" in sql:
            return glp1_fixture.copy()
        raise AssertionError("query did not name a supported direct cohort table")

    source_environment = ("METABOLIC_SOURCE_SCHEMA", "METABOLIC_MBS_TABLE", "METABOLIC_GLP1_TABLE")
    saved_environment = {name: os.environ.get(name) for name in source_environment}
    original_connect_cosmos = globals()["connect_cosmos"]
    original_read_sql_query = pd.read_sql_query
    direct_bundle: DataBundle | None = None
    direct_cohorts = pd.DataFrame()
    direct_failure: PreflightError | None = None
    try:
        for name in source_environment:
            os.environ.pop(name, None)
        globals()["connect_cosmos"] = lambda: direct_connection
        pd.read_sql_query = direct_fixture_reader
        direct_bundle = query_cosmos(RunConfig.create("smoke", None, False))
        direct_cohorts = construct_cohorts(direct_bundle)["cohorts"]

        def unavailable_reader(sql: str, connection: Any) -> Any:
            raise RuntimeError("fixture table unavailable")

        try:
            load_direct_wide_tables(
                object(),
                RunConfig.create("smoke", None, False),
                read_sql=unavailable_reader,
            )
        except PreflightError as exc:
            direct_failure = exc
    finally:
        globals()["connect_cosmos"] = original_connect_cosmos
        pd.read_sql_query = original_read_sql_query
        if not direct_connection.closed:
            direct_connection.close()
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    captured_probe_sql = [sql for sql in captured_sql if "column probe" in sql]
    captured_count_sql = [sql for sql in captured_sql if "aggregate source count" in sql]
    captured_source_sql = [sql for sql in captured_sql if "direct wide cohort input" in sql]
    check(
        "29_production_direct_wide_query_e2e",
        direct_bundle is not None
        and direct_connection.closed
        and len(captured_sql) == 6
        and len(captured_probe_sql) == 2
        and all("SELECT TOP (0) *" in sql for sql in captured_probe_sql)
        and len(captured_count_sql) == 2
        and all("COUNT_BIG(DISTINCT [PatKey])" in sql for sql in captured_count_sql)
        and len(captured_source_sql) == 2
        and all("WITH [sampled_patients]" in sql for sql in captured_source_sql)
        and all("SELECT TOP (2000) [PatKey]" in sql for sql in captured_source_sql)
        and all("INNER JOIN [sampled_patients]" in sql for sql in captured_source_sql)
        and all("[source].*" not in sql for sql in captured_source_sql)
        and all("INFORMATION_SCHEMA" not in sql for sql in captured_sql)
        and any("[dbo].[MBSCohort]" in sql for sql in captured_sql)
        and any("[dbo].[GLP1Cohort]" in sql for sql in captured_sql)
        and direct_bundle.metadata.get("source_mode") == "cosmos_direct_wide_cohorts"
        and direct_bundle.metadata.get("measurement_timing") == "nominal_horizon_from_wide_columns"
        and direct_bundle.metadata.get("strict_raw_event_contract") is False
        and direct_bundle.metadata.get("preflight", {}).get("status") == "passed_with_wide_source_limitations"
        and direct_bundle.metadata.get("preflight", {}).get("source_total_unique_patient_counts")
        == {"MBSCohort": 5_000, "GLP1Cohort": 6_000}
        and math.isclose(
            direct_bundle.metadata.get("preflight", {}).get("source_patient_sampling_fractions", {}).get("MBSCohort", math.nan),
            1 / 5_000,
        )
        and set(direct_cohorts["cohort"]) == {"surgery", "incretin"},
        f"queries={len(captured_sql)}; cohorts={sorted(direct_cohorts.get('cohort', pd.Series(dtype=str)).unique())}",
    )
    direct_failure_text = " ".join(
        [
            direct_failure.title if direct_failure else "",
            *(direct_failure.issues if direct_failure else []),
            *(direct_failure.details if direct_failure else []),
        ]
    )
    check(
        "30_direct_wide_failure_diagnostics",
        direct_failure is not None
        and direct_failure.title == "Cosmos direct cohort query failed"
        and "MBSCohort" in direct_failure_text
        and "GLP1Cohort" in direct_failure_text
        and "Patients source contract" not in direct_failure_text,
        direct_failure_text,
    )

    # 31. The real feasibility extract is multi-row for some GLP-1 patients. All
    # intervals remain available, while outcomes use the reviewed primary episode.
    repeated_glp1 = pd.concat(
        [
            glp1_fixture,
            glp1_fixture.assign(GLP1StartDate="2021-05-01", GLP1EndDate="2022-05-01", GLP1Name="tirzepatide"),
        ],
        ignore_index=True,
    )
    repeated_frames = {"MBSCohort": mbs_fixture, "GLP1Cohort": repeated_glp1}
    repeated_bundle = wide_tables_to_data_bundle(
        repeated_frames,
        {"MBSCohort": "fixture mbs", "GLP1Cohort": "fixture glp1"},
        {"MBSCohort": "dbo.MBSCohort", "GLP1Cohort": "dbo.GLP1Cohort"},
    )
    reversed_bundle = wide_tables_to_data_bundle(
        {"MBSCohort": mbs_fixture, "GLP1Cohort": repeated_glp1.iloc[::-1].reset_index(drop=True)},
        {"MBSCohort": "fixture mbs", "GLP1Cohort": "fixture glp1"},
        {"MBSCohort": "dbo.MBSCohort", "GLP1Cohort": "dbo.GLP1Cohort"},
    )
    repeated_cohort = construct_cohorts(repeated_bundle)["cohorts"]
    repeated_glp1_cohort = repeated_cohort.loc[repeated_cohort["cohort"].eq("incretin")].iloc[0]
    repeated_glp1_measurements = repeated_bundle.measurements.loc[
        repeated_bundle.measurements["source_cohort"].eq("incretin")
    ]
    check(
        "31_multirow_glp1_deterministic_primary_episode",
        len(repeated_bundle.medications.loc[repeated_bundle.medications["source_cohort"].eq("incretin")]) == 2
        and repeated_bundle.metadata["index_row_selection"]["GLP1Cohort"]["multirow_patients"] == 1
        and repeated_bundle.metadata["index_row_selection"]["GLP1Cohort"]["selected_index_rows"] == 1
        and pd.Timestamp(repeated_glp1_cohort["index_date"]) == pd.Timestamp("2020-02-01")
        and repeated_glp1_cohort["treatment"] == "semaglutide"
        and repeated_glp1_measurements["index_anchor_date"].nunique() == 1
        and pd.Timestamp(repeated_glp1_measurements["index_anchor_date"].iloc[0]) == pd.Timestamp("2020-02-01")
        and repeated_bundle.patients.loc[repeated_bundle.patients["patient_id"].eq("DIRECT-GLP1"), "glp1__wide_index_date"].iloc[0]
        == reversed_bundle.patients.loc[reversed_bundle.patients["patient_id"].eq("DIRECT-GLP1"), "glp1__wide_index_date"].iloc[0],
        str(repeated_bundle.metadata["index_row_selection"]["GLP1Cohort"]),
    )

    flagged_bundle = wide_tables_to_data_bundle(
        {"MBSCohort": mbs_fixture.assign(PostOpGLP1=1), "GLP1Cohort": glp1_fixture},
        {"MBSCohort": "fixture mbs", "GLP1Cohort": "fixture glp1"},
        {"MBSCohort": "dbo.MBSCohort", "GLP1Cohort": "dbo.GLP1Cohort"},
    )
    flagged_artifacts = construct_cohorts(flagged_bundle)
    flagged_details = " ".join(flagged_bundle.metadata["preflight"]["details"])
    check(
        "31b_undated_postoperative_flag_is_excluded_not_imputed",
        "flagged surgical patients lack a reported postoperative start" in flagged_details
        and "postop_flag_without_start" in set(flagged_artifacts["funnel"]["stage"])
        and set(flagged_artifacts["cohorts"]["cohort"]) == {"incretin"},
        flagged_details,
    )

    overflow_active_end = 2_914_070_000_000_000
    overflow_bundle = wide_tables_to_data_bundle(
        {
            "MBSCohort": mbs_fixture.assign(ActiveEndInterval=overflow_active_end),
            "GLP1Cohort": glp1_fixture,
        },
        {"MBSCohort": "fixture mbs", "GLP1Cohort": "fixture glp1"},
        {"MBSCohort": "dbo.MBSCohort", "GLP1Cohort": "dbo.GLP1Cohort"},
    )
    overflow_cohorts = construct_cohorts(overflow_bundle)["cohorts"]
    overflow_patient = overflow_bundle.patients.loc[
        overflow_bundle.patients["patient_id"].eq("DIRECT-MBS")
    ].iloc[0]
    expected_observation_end = pd.Timestamp("2020-01-15") + pd.Timedelta(
        days=month_to_nominal_day(12)
    )
    check(
        "31c_implausible_active_end_uses_outcome_horizon",
        overflow_patient["mbs__active_end_resolution_method"] == "outcome_horizon_fallback"
        and pd.Timestamp(overflow_patient["observation_end_date"]) == expected_observation_end
        and overflow_bundle.metadata["preflight"]["active_end_resolution_counts"]["MBSCohort"]
        == {"outcome_horizon_fallback": 1}
        and set(overflow_cohorts["cohort"]) == {"surgery", "incretin"},
        str(overflow_bundle.metadata["preflight"]["active_end_resolution_counts"]),
    )

    # 32. Schema discovery reads only metadata and can rank normalized source tables
    # even when patient, procedure, medication, and measurement fields are separated.
    metadata_rows: list[dict[str, Any]] = []

    def add_metadata_table(table_name: str, columns: Sequence[tuple[str, str]], table_type: str = "BASE TABLE") -> None:
        for ordinal, (column_name, data_type) in enumerate(columns, start=1):
            metadata_rows.append(
                {
                    "TABLE_CATALOG": "SyntheticProject",
                    "TABLE_SCHEMA": "raw",
                    "TABLE_NAME": table_name,
                    "TABLE_TYPE": table_type,
                    "COLUMN_NAME": column_name,
                    "ORDINAL_POSITION": ordinal,
                    "DATA_TYPE": data_type,
                    "CHARACTER_MAXIMUM_LENGTH": np.nan,
                    "NUMERIC_PRECISION": np.nan,
                    "NUMERIC_SCALE": np.nan,
                    "IS_NULLABLE": "YES",
                }
            )

    add_metadata_table(
        "PatientDimension",
        [("PatKey", "bigint"), ("CenterID", "nvarchar"), ("BirthYear", "int"), ("Sex", "nvarchar"),
         ("ObservationStartDate", "date"), ("ObservationEndDate", "date")],
    )
    add_metadata_table(
        "ProcedureFact",
        [("PatKey", "bigint"), ("ProcedureDate", "date"), ("CptCode", "nvarchar"), ("EncounterKey", "bigint")],
    )
    add_metadata_table(
        "MedicationDispenseFact",
        [("PatKey", "bigint"), ("MedicationID", "bigint"), ("RxNorm", "nvarchar"), ("FillDate", "date"),
         ("DaysSupply", "int"), ("Dose", "numeric"), ("DoseUnit", "nvarchar")],
    )
    add_metadata_table(
        "LabResultFact",
        [("PatKey", "bigint"), ("ResultDate", "date"), ("NumericValue", "numeric"),
         ("ResultUnit", "nvarchar"), ("LoincCode", "nvarchar"), ("EncounterKey", "bigint")],
    )
    add_metadata_table(
        "EncounterFact",
        [("PatKey", "bigint"), ("EncounterKey", "bigint"), ("EncounterDate", "date"), ("CenterID", "nvarchar")],
    )
    add_metadata_table(
        "DiagnosisFact",
        [("PatKey", "bigint"), ("DiagnosisDate", "date"), ("ICD10", "nvarchar"), ("EncounterKey", "bigint")],
    )
    metadata_columns = pd.DataFrame(metadata_rows)
    metadata_keys = pd.DataFrame(
        [
            {
                "TABLE_SCHEMA": "raw",
                "TABLE_NAME": "PatientDimension",
                "COLUMN_NAME": "PatKey",
                "CONSTRAINT_TYPE": "PRIMARY KEY",
                "CONSTRAINT_NAME": "PK_PatientDimension",
                "ORDINAL_POSITION": 1,
            }
        ]
    )
    metadata_foreign_keys = pd.DataFrame(
        [
            {
                "CHILD_SCHEMA": "raw",
                "CHILD_TABLE": "ProcedureFact",
                "CHILD_COLUMN": "PatKey",
                "PARENT_SCHEMA": "raw",
                "PARENT_TABLE": "PatientDimension",
                "PARENT_COLUMN": "PatKey",
                "FOREIGN_KEY_NAME": "FK_Procedure_Patient",
            }
        ]
    )
    captured_discovery_sql: list[str] = []

    def metadata_fixture_reader(sql: str, connection: Any) -> Any:
        captured_discovery_sql.append(sql)
        if "schema-discovery: database" in sql:
            return pd.DataFrame({"DATABASE_NAME": ["SyntheticProject"]})
        if "schema-discovery: columns" in sql:
            return metadata_columns.copy()
        if "schema-discovery: keys" in sql:
            return metadata_keys.copy()
        if "schema-discovery: foreign-keys" in sql:
            return metadata_foreign_keys.copy()
        if "schema-discovery: synonyms" in sql:
            return pd.DataFrame(columns=["SYNONYM_SCHEMA", "SYNONYM_NAME", "BASE_OBJECT_NAME"])
        if "schema-discovery: cohort-modules" in sql:
            return pd.DataFrame(
                columns=["MODULE_SCHEMA", "MODULE_NAME", "MODULE_TYPE", "REFERENCES_MBSCOHORT", "REFERENCES_GLP1COHORT"]
            )
        if "schema-discovery: object-dependencies" in sql:
            return pd.DataFrame(
                [
                    {
                        "REFERENCING_SCHEMA": "dbo",
                        "REFERENCING_OBJECT": "GLP1Cohort",
                        "REFERENCING_TYPE": "VIEW",
                        "REFERENCED_SERVER": np.nan,
                        "REFERENCED_DATABASE": np.nan,
                        "REFERENCED_SCHEMA": "raw",
                        "REFERENCED_OBJECT": "MedicationDispenseFact",
                        "IS_SCHEMA_BOUND": 0,
                        "IS_AMBIGUOUS": 0,
                    }
                ]
            )
        raise AssertionError("schema discovery attempted an unreviewed query")

    discovery_fixture = discover_cosmos_schema(object(), read_sql=metadata_fixture_reader)
    best_objects = {
        domain: str(group.iloc[0]["object"])
        for domain, group in discovery_fixture["candidates"].groupby("domain", sort=False)
        if not group.empty
    }
    check(
        "32_schema_discovery_metadata_only",
        len(captured_discovery_sql) == len(SCHEMA_DISCOVERY_SQL)
        and all("metabolic-schema-discovery:" in sql for sql in captured_discovery_sql)
        and best_objects.get("patients") == "raw.PatientDimension"
        and best_objects.get("procedures") == "raw.ProcedureFact"
        and best_objects.get("medications") == "raw.MedicationDispenseFact"
        and best_objects.get("measurements") == "raw.LabResultFact"
        and not discovery_fixture["foreign_keys"].empty
        and not discovery_fixture["object_dependencies"].empty,
        str(best_objects),
    )

    # 33. The schema-only packet is screenshot-ready and keeps all machine-readable
    # metadata outside FIGURES_TO_EXPORT.
    with tempfile.TemporaryDirectory(prefix="metabolic-schema-discovery-test-") as directory:
        export = Path(directory) / "FIGURES_TO_EXPORT"
        rendered_discovery = render_schema_discovery_book(discovery_fixture, export)
        present = {path.name for path in export.iterdir() if path.is_file()}
        check(
            "33_schema_discovery_figure_contract",
            len(rendered_discovery) == len(SCHEMA_DISCOVERY_PAGE_FILES) + 1
            and present == set(SCHEMA_DISCOVERY_PAGE_FILES) | {"schema_discovery_figure_book.pdf"}
            and all((export / name).stat().st_size > 0 for name in present),
            str(sorted(present)),
        )

    # 34. The production loader executes embedded canonical raw queries itself,
    # applies the bounded-run token, and produces a strict raw-event DataBundle.
    raw_contract_fixture = {
        logical_name: (
            f"/* fixture-raw:{logical_name} */\n"
            f"SELECT {RAW_SQL_TOP_TOKEN} * FROM [raw].[{logical_name.title()}Source]"
        )
        for logical_name in RAW_REQUIRED_SOURCES
    }
    raw_frames = {
        "patients": pd.DataFrame(
            [
                {
                    "patient_id": "RAW-P1",
                    "center_id": "CENTER-1",
                    "birth_year": 1975,
                    "observation_start_date": "2018-01-01",
                    "observation_end_date": "2025-12-31",
                    "administrative_end_date": "2025-12-31",
                }
            ]
        ),
        "procedures": pd.DataFrame(
            [{"patient_id": "RAW-P1", "procedure_date": "2020-01-15", "procedure_code": "43775"}]
        ),
        "medications": pd.DataFrame(
            [
                {
                    "patient_id": "RAW-P1",
                    "ingredient": "semaglutide",
                    "fill_date": "2019-01-01",
                    "days_supply": 183,
                }
            ]
        ),
        "measurements": pd.DataFrame(
            [
                {
                    "patient_id": "RAW-P1",
                    "measurement_date": "2020-01-15",
                    "measurement_type": "bmi",
                    "raw_value": 42.0,
                    "unit": "kg/m2",
                    "source_concept": "BMI",
                }
            ]
        ),
    }
    captured_raw_sql: list[str] = []

    def raw_fixture_reader(sql: str, connection: Any) -> Any:
        captured_raw_sql.append(sql)
        match = re.search(r"fixture-raw:([a-z_]+)", sql)
        if match is None:
            raise AssertionError("raw fixture query marker is missing")
        return raw_frames[match.group(1)].copy()

    raw_bundle = load_embedded_raw_bundle(
        object(),
        RunConfig.create("preflight-only", None, False),
        sql_contract=raw_contract_fixture,
        read_sql=raw_fixture_reader,
        preflight_only=True,
    )
    check(
        "34_embedded_raw_sql_executes_standalone",
        len(captured_raw_sql) == len(RAW_REQUIRED_SOURCES)
        and all("TOP (2000)" in query for query in captured_raw_sql)
        and all("MBSCohort" not in query and "GLP1Cohort" not in query for query in captured_raw_sql)
        and raw_bundle.metadata.get("source_mode") == "cosmos_embedded_raw_sql"
        and raw_bundle.metadata.get("strict_raw_event_contract") is True
        and raw_bundle.metadata.get("preflight", {}).get("status") == "passed",
        str(raw_bundle.metadata),
    )

    # 35. The optional future raw-source loader remains strictly validated even though
    # the current production path uses the two reviewed direct wide tables.
    absent_contract_failure: PreflightError | None = None
    forbidden_contract_failure: PreflightError | None = None
    try:
        validate_embedded_raw_sql({})
    except PreflightError as exc:
        absent_contract_failure = exc
    forbidden_contract = dict(raw_contract_fixture)
    forbidden_contract["patients"] = (
        f"SELECT {RAW_SQL_TOP_TOKEN} * FROM [dbo].[MBSCohort]"
    )
    try:
        validate_embedded_raw_sql(forbidden_contract)
    except PreflightError as exc:
        forbidden_contract_failure = exc
    check(
        "35_optional_raw_contract_validation",
        absent_contract_failure is not None
        and absent_contract_failure.title == "Reviewed raw-source SQL is not yet embedded"
        and forbidden_contract_failure is not None
        and "forbidden premade cohort" in " ".join(forbidden_contract_failure.issues),
        f"absent={absent_contract_failure}; forbidden={forbidden_contract_failure}",
    )

    # 36. Implicit runs use a timestamped directory below the invocation directory,
    # explicit overrides remain unchanged, and resume cannot silently target a new run.
    with tempfile.TemporaryDirectory(prefix="metabolic-output-path-test-") as directory:
        invocation_dir = Path(directory)
        fixed_time = datetime(2026, 7, 23, 22, 2, 31)
        default_cfg = RunConfig.create(
            "smoke", None, False, now=fixed_time, cwd=invocation_dir
        )
        expected = (
            invocation_dir.resolve()
            / "results"
            / "metabolic_trajectory_20260723_220231"
        )
        expected.mkdir(parents=True)
        collided = timestamped_default_output_dir(
            now=fixed_time, cwd=invocation_dir
        )
        explicit_cfg = RunConfig.create(
            "production", r".\custom-results", False,
            now=fixed_time, cwd=invocation_dir,
        )
        resume_rejected = False
        try:
            RunConfig.create(
                "production", None, True,
                now=fixed_time, cwd=invocation_dir,
            )
        except ValueError:
            resume_rejected = True
        check(
            "36_timestamped_default_output_directory",
            Path(str(default_cfg.output_dir)) == expected
            and collided == expected.with_name(expected.name + "_01")
            and explicit_cfg.output_dir == r".\custom-results"
            and resume_rejected,
            (
                f"default={default_cfg.output_dir}; collision={collided}; "
                f"explicit={explicit_cfg.output_dir}; "
                f"resume_rejected={resume_rejected}"
            ),
        )

    # 37. Windows/SMB access-denied and sharing locks are retried without weakening
    # atomic replacement; persistent and unrelated failures still propagate.
    with tempfile.TemporaryDirectory(prefix="metabolic-replace-test-") as directory:
        replace_root = Path(directory)

        class RetryableNasLock(PermissionError):
            winerror = 5

        class NonRetryableReplaceError(OSError):
            winerror = 87

        original_replace = os.replace
        transient_source = replace_root / "transient.tmp"
        transient_target = replace_root / "transient.json"
        transient_source.write_bytes(b"new")
        transient_target.write_bytes(b"old")
        transient_calls = 0
        transient_delays: list[float] = []

        def transient_replace(source: str | Path, destination: str | Path) -> None:
            nonlocal transient_calls
            transient_calls += 1
            if transient_calls <= 2:
                raise RetryableNasLock(13, "simulated transient NAS lock")
            original_replace(source, destination)

        os.replace = transient_replace
        try:
            replace_file(
                transient_source,
                transient_target,
                attempts=4,
                initial_delay=0.1,
                maximum_delay=0.2,
                sleeper=transient_delays.append,
                report_retries=False,
            )
        finally:
            os.replace = original_replace
        transient_recovered = (
            transient_calls == 3
            and transient_delays == [0.1, 0.2]
            and transient_target.read_bytes() == b"new"
            and not transient_source.exists()
        )

        persistent_source = replace_root / "persistent.tmp"
        persistent_target = replace_root / "persistent.json"
        persistent_source.write_bytes(b"new")
        persistent_target.write_bytes(b"old")
        persistent_calls = 0
        persistent_delays: list[float] = []

        def persistent_replace(source: str | Path, destination: str | Path) -> None:
            nonlocal persistent_calls
            persistent_calls += 1
            raise RetryableNasLock(13, "simulated persistent NAS lock")

        persistent_raised = False
        os.replace = persistent_replace
        try:
            replace_file(
                persistent_source,
                persistent_target,
                attempts=3,
                initial_delay=0.1,
                maximum_delay=0.2,
                sleeper=persistent_delays.append,
                report_retries=False,
            )
        except RetryableNasLock:
            persistent_raised = True
        finally:
            os.replace = original_replace
        persistent_preserved = (
            persistent_raised
            and persistent_calls == 3
            and persistent_delays == [0.1, 0.2]
            and persistent_source.read_bytes() == b"new"
            and persistent_target.read_bytes() == b"old"
        )

        unrelated_source = replace_root / "unrelated.tmp"
        unrelated_target = replace_root / "unrelated.json"
        unrelated_source.write_bytes(b"new")
        unrelated_calls = 0
        unrelated_delays: list[float] = []

        def unrelated_replace(source: str | Path, destination: str | Path) -> None:
            nonlocal unrelated_calls
            unrelated_calls += 1
            raise NonRetryableReplaceError(22, "simulated invalid replace")

        unrelated_raised = False
        os.replace = unrelated_replace
        try:
            replace_file(
                unrelated_source,
                unrelated_target,
                attempts=4,
                sleeper=unrelated_delays.append,
                report_retries=False,
            )
        except NonRetryableReplaceError:
            unrelated_raised = True
        finally:
            os.replace = original_replace
        unrelated_not_retried = (
            unrelated_raised
            and unrelated_calls == 1
            and not unrelated_delays
            and unrelated_source.read_bytes() == b"new"
            and not unrelated_target.exists()
        )

        check(
            "37_windows_nas_replace_retry_contract",
            transient_recovered and persistent_preserved and unrelated_not_retried,
            (
                f"transient={transient_recovered}; "
                f"persistent={persistent_preserved}; "
                f"unrelated={unrelated_not_retried}"
            ),
        )

    return {
        "status": "passed",
        "tests": results,
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
    }


# ======================================================================================
# 11. Production orchestration, resume, plot-only, and CLI
# ======================================================================================


def checkpoint_hash(context: RunContext, stage: str) -> str:
    metadata = read_json(context.checkpoints / f"{stage}.json", {}) or {}
    value = metadata.get("artifact_sha256")
    if not value:
        raise RuntimeError(f"Checkpoint {stage} has no verified artifact hash")
    return str(value)


def load_or_run_stage(
    context: RunContext,
    stage: str,
    function: Callable[[], Any],
    upstream: Mapping[str, str] | None = None,
) -> tuple[Any, str]:
    loaded = context.load_checkpoint(stage, upstream)
    if loaded is not None:
        return loaded, checkpoint_hash(context, stage)
    started = time.perf_counter()
    value = function()
    artifact_hash = context.save_checkpoint(
        stage,
        value,
        upstream,
        elapsed_seconds=time.perf_counter() - started,
    )
    return value, artifact_hash


def use_real_smoke_source(dependencies: Mapping[str, Any]) -> bool:
    forced_synthetic = os.environ.get("METABOLIC_FORCE_SYNTHETIC_SMOKE", "").strip().lower() in {"1", "true", "yes"}
    return bool(dependencies.get("pyodbc_importable")) and not forced_synthetic


def write_preflight_success(context: RunContext, bundle: DataBundle, dependencies: Mapping[str, Any]) -> list[Path]:
    data = {
        "identity": {
            "study_version": STUDY_VERSION,
            "fingerprint": context.fingerprint,
            "script_sha256": context.fingerprint_payload["script_sha256"],
            "query_fingerprint": bundle.metadata.get("query_fingerprint", "unknown"),
            "schema_fingerprint": bundle.metadata.get("schema_fingerprint", "unknown"),
            "generated_utc": utc_now(),
            "source_mode": bundle.metadata.get("source_mode", "unknown"),
            "cohort_date_range": ("preflight only", "preflight only"),
            "status": "preflight passed",
            "run_mode": context.cfg.mode,
            "smoke_query_limit": context.cfg.smoke_query_limit if context.cfg.smoke else None,
            "dependencies": dict(dependencies),
            "preflight": bundle.metadata.get("preflight", {}),
            "measurement_timing": bundle.metadata.get("measurement_timing", "exact_day"),
            "medication_coverage_semantics": bundle.metadata.get(
                "medication_coverage_semantics", "audited_raw_events"
            ),
            "center_validation_available": bool(
                bundle.metadata.get("center_validation_available", True)
            ),
            "source_limitations": list(bundle.metadata.get("limitations", [])),
        }
    }
    configure_figure_style()
    figure = render_page_01(data)
    add_run_provenance(figure, data)
    png = context.export / "01_run_identity_and_status.png"
    temporary_png = context.export / (png.name + ".tmp")
    figure.savefig(temporary_png, format="png", dpi=300, facecolor=figure.get_facecolor())
    replace_file(temporary_png, png)
    pdf = context.export / "metabolic_trajectory_figure_book.pdf"
    temporary_pdf = context.export / "metabolic_trajectory_figure_book.pdf.tmp"
    with PdfPages(temporary_pdf, metadata={"Title": "Metabolic Trajectory Preflight"}) as writer:
        writer.savefig(figure, dpi=300, facecolor=figure.get_facecolor())
    replace_file(temporary_pdf, pdf)
    plt.close(figure)
    return [png, pdf]


def run_study(cfg: RunConfig, dependencies: Mapping[str, Any]) -> Path:
    set_deterministic_seed(cfg.seed, include_torch=bool(dependencies.get("torch_importable")))
    if cfg.smoke and not use_real_smoke_source(dependencies):
        bundle = synthetic_data_bundle(cfg)
        print("[metabolic] smoke source: deterministic raw-event fixture", flush=True)
    else:
        bundle = query_cosmos(cfg, preflight_only=cfg.mode == "preflight-only")
        print("[metabolic] source: bounded Cosmos query" if cfg.smoke else "[metabolic] source: Cosmos production query", flush=True)
    context = make_run_context(cfg, bundle, dependencies)
    if cfg.mode == "preflight-only":
        write_preflight_success(context, bundle, dependencies)
        context.state["status"] = "preflight_passed"
        atomic_json(context.run_dir / "run_state.json", context.state)
        return context.run_dir

    timing_label = bundle.metadata.get("measurement_timing", "exact-day")
    print(f"[metabolic] constructing cohorts and outcomes ({timing_label})", flush=True)
    cohort_artifacts, cohort_hash = load_or_run_stage(
        context, "cohorts", lambda: construct_cohorts(bundle)
    )
    split_payload, split_hash = load_or_run_stage(
        context,
        "global_splits",
        lambda: dict(zip(("cohorts", "metadata"), assign_global_splits(cohort_artifacts["cohorts"]), strict=True)),
        {"cohorts": cohort_hash},
    )
    cohorts = split_payload["cohorts"]
    split_metadata = split_payload["metadata"]
    prediction_rows, row_hash = load_or_run_stage(
        context,
        "prediction_rows",
        lambda: build_prediction_rows(cohorts, cohort_artifacts["measurements"]),
        {"cohorts": cohort_hash, "splits": split_hash},
    )
    try:
        leakage = leakage_audit(prediction_rows, split_metadata)
    except LeakageError as exc:
        render_preflight_failure(cfg, "Leakage invariant failed", [str(exc)], ["Model fitting was not started."])
        raise
    print("[metabolic] estimating cross-fitted treatment and observation weights", flush=True)
    weight_payload, weight_hash = load_or_run_stage(
        context,
        "weights",
        lambda: dict(zip(("rows", "diagnostics"), estimate_cross_fitted_weights(prediction_rows, cfg.seed), strict=True)),
        {"prediction_rows": row_hash},
    )
    weighted_rows = weight_payload["rows"]
    weight_diagnostics = weight_payload["diagnostics"]
    ode_gates = ode_suitability_gates(cohorts, cohort_artifacts["measurements"], dependencies)
    print("[metabolic] fitting matched candidate roster", flush=True)
    model_payload, model_hash = load_or_run_stage(
        context,
        "models_and_predictions",
        lambda: dict(
            zip(
                ("predictions", "status", "details"),
                fit_candidate_roster(
                    weighted_rows,
                    cfg,
                    dependencies,
                    ode_gates,
                    cohorts=cohorts,
                    measurements=cohort_artifacts["measurements"],
                ),
                strict=True,
            )
        ),
        {"weights": weight_hash},
    )
    predictions = model_payload["predictions"]
    model_status = model_payload["status"]
    neural_details = model_payload["details"]
    leaderboard = candidate_validation_scores(predictions)
    selected = select_models(leaderboard)
    calibrated, calibration = conformal_calibrate(predictions)
    print("[metabolic] evaluating protected tests and uncertainty", flush=True)

    def evaluation_stage() -> dict[str, Any]:
        metrics, pit_values = evaluate_predictions(calibrated, weight_diagnostics)
        iqr = development_iqr_by_task(weighted_rows)
        bootstrap_ci, comparisons = bootstrap_uncertainty(calibrated, selected, cfg)
        gates = apply_success_gates(metrics, selected, iqr, comparisons)
        gates = apply_source_claim_limit(gates, bundle.metadata)
        gates = apply_smoke_claim_limit(gates, cfg)
        sensitivity = weight_sensitivity_table(calibrated, selected)
        gap_sensitivity = gap_rule_sensitivity(bundle)
        examples, joint_scores = build_synthetic_trajectory_examples(calibrated, selected, cfg)
        return {
            "metrics": metrics,
            "pit_values": pit_values,
            "iqr": iqr,
            "gates": gates,
            "bootstrap_ci": bootstrap_ci,
            "comparisons": comparisons,
            "sensitivity": sensitivity,
            "gap_sensitivity": gap_sensitivity,
            "examples": examples,
            "joint_scores": joint_scores,
        }

    evaluation, evaluation_hash = load_or_run_stage(
        context,
        "evaluation",
        evaluation_stage,
        {"models": model_hash},
    )

    def aggregate_stage() -> dict[str, Any]:
        return build_figure_data(
            context=context,
            dependencies=dependencies,
            bundle=bundle,
            cohort_artifacts=cohort_artifacts,
            cohorts=cohorts,
            split_metadata=split_metadata,
            leakage=leakage,
            weighted_rows=weighted_rows,
            weight_diagnostics=weight_diagnostics,
            predictions=calibrated,
            model_status=model_status,
            neural_details=neural_details,
            leaderboard=leaderboard,
            selected=selected,
            calibration=calibration,
            metrics=evaluation["metrics"],
            iqr=evaluation["iqr"],
            pit_values=evaluation["pit_values"],
            bootstrap_ci=evaluation["bootstrap_ci"],
            comparisons=evaluation["comparisons"],
            gates=evaluation["gates"],
            ode_gates=ode_gates,
            sensitivity=evaluation["sensitivity"],
            gap_sensitivity=evaluation["gap_sensitivity"],
            examples=evaluation["examples"],
            joint_scores=evaluation["joint_scores"],
        )

    figure_data, _ = load_or_run_stage(
        context,
        "figure_data",
        aggregate_stage,
        {"evaluation": evaluation_hash},
    )
    print(f"[metabolic] rendering {len(PAGE_FILES)}-page disclosure-controlled figure book", flush=True)
    rendered = render_figure_book(figure_data, context.export)
    context.state["status"] = "completed"
    context.state["completed_utc"] = utc_now()
    context.state["export_files"] = [path.name for path in rendered]
    atomic_json(context.run_dir / "run_state.json", context.state)
    return context.run_dir


def verified_plot_only(cfg: RunConfig) -> Path:
    if not cfg.output_dir:
        raise PreflightError("Plot-only mode requires a run directory", ["Pass --output-dir PATH for the matching completed run"])
    run_dir = Path(cfg.output_dir).expanduser().resolve()
    manifest = read_json(run_dir / "run_manifest.json", {}) or {}
    if not manifest:
        raise PreflightError("Plot-only manifest is missing", [f"No verified run_manifest.json exists in {run_dir}"])
    fingerprint = manifest.get("fingerprint")
    payload = manifest.get("fingerprint_payload", {})
    if payload.get("script_sha256") != sha256_file(SCRIPT_PATH):
        raise PreflightError(
            "Plot-only script fingerprint mismatch",
            ["The current script differs from the script that created this run"],
            ["Use the exact matching runtime file or rerun the study."],
        )
    checkpoint_directory = run_dir / "INTERNAL" / "checkpoints"
    metadata = read_json(checkpoint_directory / "figure_data.json", {}) or {}
    body = checkpoint_directory / "figure_data.pkl"
    expected_stage = digest(
        {
            "run_fingerprint": fingerprint,
            "stage": "figure_data",
            "upstream": dict(sorted(metadata.get("upstream", {}).items())),
        }
    )
    if (
        metadata.get("completion_marker") != "COMPLETE"
        or metadata.get("stage_fingerprint") != expected_stage
        or not body.exists()
        or sha256_file(body) != metadata.get("artifact_sha256")
    ):
        raise PreflightError("Plot-only checkpoint verification failed", ["The aggregate figure checkpoint is stale, partial, or corrupt"])
    with body.open("rb") as stream:
        figure_data = pickle.load(stream)
    if payload_manifest(figure_data) != metadata.get("payload_manifest"):
        raise PreflightError("Plot-only aggregate schema verification failed", ["The figure checkpoint row counts or schema do not match its completion manifest"])
    render_figure_book(figure_data, run_dir / "FIGURES_TO_EXPORT")
    return run_dir


def sanitize_exception_text(value: Any) -> str:
    text = str(value)
    text = re.sub(r"(?i)(driver|server|database|uid|user id|pwd|password)\s*=\s*[^;\s]+", r"\1=<redacted>", text)
    text = re.sub(r"(?i)tcp:[^;\s]+", "tcp:<redacted>", text)
    return text[:1200]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Production queries MBSCohort and GLP1Cohort directly. Use --schema-discovery "
            "only when moving to a different Cosmos schema or auditing newly exposed sources."
        ),
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true", help="Validate dependencies, schemas, dates, maturity, and exposure semantics only")
    modes.add_argument(
        "--schema-discovery",
        action="store_true",
        help="Query SQL Server metadata only and render raw-source candidate/key figures; no patient rows are read",
    )
    modes.add_argument("--self-test", action="store_true", help="Run deterministic embedded tests without Cosmos")
    modes.add_argument("--smoke", action="store_true", help="Run a bounded end-to-end study with reduced tuning")
    modes.add_argument("--plot-only", action="store_true", help="Rebuild figures from a verified matching aggregate checkpoint")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=r"Override the default .\results\metabolic_trajectory_YYYYMMDD_HHMMSS run directory",
    )
    parser.add_argument("--resume", action="store_true", help="Resume only verified fingerprint-compatible checkpoints")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test and args.resume:
        raise SystemExit("--resume is not meaningful with --self-test")
    if args.schema_discovery and args.resume:
        raise SystemExit("--resume is not meaningful with --schema-discovery")
    if args.resume and not args.output_dir:
        raise SystemExit("--resume requires --output-dir PATH for the existing run")
    mode = "production"
    if args.preflight_only:
        mode = "preflight-only"
    elif args.schema_discovery:
        mode = "schema-discovery"
    elif args.self_test:
        mode = "self-test"
    elif args.smoke:
        mode = "smoke"
    elif args.plot_only:
        mode = "plot-only"
    cfg = RunConfig.create(mode, args.output_dir, args.resume)
    require_database = mode in {"production", "preflight-only", "schema-discovery"}
    dependencies, dependency_issues = dependency_manifest(require_database=require_database)
    if dependency_issues:
        failure = render_preflight_failure(cfg, "Runtime dependency preflight failed", dependency_issues)
        print(f"preflight failed: {failure}", file=sys.stderr)
        return 2
    try:
        load_runtime_packages()
        if mode == "self-test":
            report = run_embedded_self_tests()
            for item in report["tests"]:
                print(f"[PASS] {item['test']}" + (f" | {item['detail']}" if item["detail"] else ""))
            print(f"SELF-TEST PASSED: {report['passed']}/{report['total']} deterministic tests")
            return 0
        if mode == "plot-only":
            run_dir = verified_plot_only(cfg)
        elif mode == "schema-discovery":
            run_dir = run_schema_discovery(cfg, dependencies)
        else:
            run_dir = run_study(cfg, dependencies)
        print(f"[metabolic] completed: {run_dir}")
        print(f"[metabolic] figures: {run_dir / 'FIGURES_TO_EXPORT'}")
        return 0
    except PreflightError as exc:
        failure = render_preflight_failure(cfg, exc.title, exc.issues, exc.details)
        print(f"preflight failed: {failure}", file=sys.stderr)
        return 2
    except Exception as exc:
        detail = sanitize_exception_text(exc)
        failure = render_preflight_failure(
            cfg,
            "Study execution failed",
            [f"{type(exc).__name__}: {detail}"],
            ["The run stopped before a scientific result was released. Review the console traceback inside the secure VM."],
        )
        print(f"study failed: {failure}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
