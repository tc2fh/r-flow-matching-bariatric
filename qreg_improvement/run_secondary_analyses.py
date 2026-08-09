#!/usr/bin/env python3
"""Secondary analyses run for the metabolic trajectory study.

This is a single, self-contained runnable script that consumes a completed production run of
``run_metabolic_trajectory_study.py`` (study ``metabolic-trajectory-1.4.0``) and emits a
publication-ready figure book, CSV metrics, a provenance manifest, and one results bundle,
following the same conventions as the production run.

It adds two families of analysis on the SAME cohorts and the SAME frozen predictions:

  1. A wide-source point-intervention target-trial emulation (TTE): sleeve gastrectomy vs
     Roux-en-Y gastric bypass, doubly-robust IPCW-AIPW, with an E-value, an RCT benchmark, a
     concordance-for-benefit calibration, and positivity/balance diagnostics. This is the causal
     contrast the production run deliberately does not make; it is causal only under conditional
     exchangeability given the measured baseline covariates, and the E-value bounds residual
     confounding from the named unmeasured confounders (GERD/reflux, surgeon/center).
  2. Four tiers of subgroup, sensitivity, and clinical-reframing analyses (Tiers 1-4) that turn
     documented limitations and untapped covariates into results.

The primary command for the collaborator on the Cosmos VM is::

    python qreg_improvement/run_secondary_analyses.py --from-run RESULTS/metabolic_trajectory_YYYYMMDD_HHMMSS

Canonical invocation directory: the repository root (the directory that contains
``causal_tte.py`` and the ``qreg_improvement/`` package). The script inserts both the repository
root and ``qreg_improvement/`` onto ``sys.path`` at startup, so it also runs correctly from
inside ``qreg_improvement/`` or with an absolute path from anywhere.

Modes (mirroring the production run):
  --from-run RUN_DIR : load the frozen predictions + selected-model table from a completed run
                       and re-materialize the patient-level covariate frame via the production
                       streaming acquire (Cosmos reachable on the VM). Fast; the default path.
  --full             : no completed run available; re-run the production modeling pipeline via
                       the imported study to regenerate predictions, then analyse. Heavy.
  --smoke            : bounded end-to-end run on the production synthetic bundle (no Cosmos, no
                       torch). Synthesizes a schema-faithful prediction store from real synthetic
                       cohorts and produces a COMPLETE, contract-valid figure book + bundle.
  --self-test        : deterministic embedded unit tests on synthetic fixtures (no Cosmos, no VM).
  --plot-only        : rebuild the figure book from this run's own aggregate checkpoint.

Memory: like the production run, each analysis stage can run in its own process (``--orchestrate``,
the default for --from-run/--full) so peak RSS is bounded by the single worst stage; intermediate
frames are spilled to parquet/pickle and predictions are streamed one task partition at a time.

Determinism: fixed seeds; identical inputs and config produce byte-identical CSVs and an identical
figure hash. Only the wall-clock stamp in the manifest is nondeterministic.

Disclosure control: any displayed cell with fewer than 11 patients is suppressed, exactly as in
the production run (``study.MIN_CELL_SIZE``).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


# --------------------------------------------------------------------------------------------
# Import path bootstrap: put the repo root and qreg_improvement/ on sys.path so the two reused
# modules import regardless of the current working directory. causal_tte.py is in the repo root;
# run_metabolic_trajectory_study.py is in qreg_improvement/.
# --------------------------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_QREG_DIR = _THIS_FILE.parent
_REPO_ROOT = _QREG_DIR.parent
for _candidate in (_REPO_ROOT, _QREG_DIR):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

import run_metabolic_trajectory_study as study  # noqa: E402  (after sys.path bootstrap)
import causal_tte as tte  # noqa: E402  (after sys.path bootstrap)


# --------------------------------------------------------------------------------------------
# Frozen protocol and identity
# --------------------------------------------------------------------------------------------
SECONDARY_VERSION = "secondary-analyses-1.0.0"
SOURCE_STUDY_VERSION = study.STUDY_VERSION
SEED = study.SEED
MIN_CELL_SIZE = study.MIN_CELL_SIZE  # 11

# Adequately-powered primary-display thresholds (Section 6 of the spec).
POWERED_MIN_N = 200
POWERED_MIN_ESS = 100

# Bootstrap replicates (patient-clustered), full vs smoke.
BOOTSTRAP_FULL = 1000
BOOTSTRAP_SMOKE = 200
# Multiple-imputation count default (Tier 1.2).
MI_IMPUTATIONS_DEFAULT = 10
# Cross-fitting folds for every nuisance (ps, pc, mu).
NUISANCE_FOLDS = 5

# The two surgical arms of the TTE. A = 1 is RYGB, A = 0 is sleeve.
SURGERY_ARMS = ("sleeve", "rygb")

# Cohort / outcome vocabulary carried by the production store.
COHORTS = ("surgery", "incretin")
OUTCOMES = ("bmi", "hba1c")
QUANTILE_LEVELS = study.QUANTILES  # (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QUANTILE_COLS = list(study.QUANTILE_COLUMNS)  # ["q05","q10","q25","q50","q75","q90","q95"]

# Held-out evaluation splits for prognostic metrics (mirrors production subgroup_performance).
HELD_OUT_SPLITS = ("temporal_test", "geographic_test")

# SOPHIA per-procedure preoperative 60-month BMI RMSE anchors (Lancet Digit Health 2023,
# PMID 37652841). Drawn as dotted CONTEXT lines only; not a matched comparator. The 60-month
# horizon is not estimable on this source, so these are context, not comparisons.
SOPHIA_PROCEDURE_RMSE_60 = {"rygb": 4.5, "sleeve": 5.7}
SOPHIA_POOLED_RMSE = {12: 3.7, 24: 4.2, 60: 4.7}

# Incretin drug groups for Tier 1.3 heterogeneity, keyed off index_ingredient (+ index_route for
# semaglutide oral vs injectable). The "older" bucket pools the earlier agents.
OLDER_INCRETINS = ("exenatide", "lixisenatide", "albiglutide")

# The residual-confounding estimand caveat carried verbatim onto the TTE page and CSV headers.
ESTIMAND_NOTE = (
    "Marginal average treatment effect of RYGB vs sleeve in the eligible surgical target "
    "population, under conditional exchangeability given measured baseline L. Causal only under "
    "that unverifiable assumption; the E-value and RCT benchmark bound residual confounding. "
    "Procedure choice is a point intervention at time zero, so ITT and per-protocol coincide and "
    "no separate per-protocol contrast is estimated. "
    "Not a substitute for an RCT. Unmeasured confounders include GERD/reflux and surgeon/center."
)

# Figure book contract: numbered PNGs + one bound PDF, exactly like production's PAGE_FILES.
SECONDARY_PAGE_FILES = (
    "00_executive_summary.png",
    "01_run_identity_and_provenance.png",
    "02_tte_design_and_cohort.png",
    "03_tte_propensity_and_positivity.png",
    "04_tte_primary_results.png",
    "05_transportability_and_equity.png",
    "06_attrition_mnar_sensitivity.png",
    "07_incretin_drug_heterogeneity.png",
    "08_procedure_heterogeneity_vs_sophia.png",
    "09_fairness_subgroups.png",
    "10_clinical_subgroups.png",
    "11_obesity_class_and_calendar_era.png",
    "12_robustness_panel.png",
    "13_clinical_threshold_probabilities.png",
    "14_decision_curves.png",
    "15_predictability_map_and_glp1_vs_surgery.png",
    "16_gates_limitations_and_conclusion.png",
    "17_per_procedure_auroc.png",
    "18_cate_heterogeneous_benefit.png",
    "19_forecast_feature_sensitivity.png",
    "20_per_arm_calibration_and_accuracy.png",
)
FIGURE_BOOK_PDF = "secondary_analyses_figure_book.pdf"
FAILURE_PNG = "00_preflight_failure.png"

# The ordered analysis stage graph. Each stage is idempotent, checkpointed, and resumable.
STAGE_SEQUENCE = (
    "assemble",
    "tte",
    "tier1",
    "tier2",
    "tier3",
    "tier4",
    "render",
)

# Heavy runtime packages, populated by load_runtime() from the study module's own lazy globals.
np = pd = plt = PdfPages = None


def load_runtime() -> None:
    """Bind numpy/pandas/matplotlib/PdfPages from the study module's lazy loader.

    The study functions we reuse read the study module's own ``np``/``pd`` globals, so we must
    call the study's loader (not just import numpy) before invoking any of them. Matplotlib is
    forced onto the non-interactive Agg backend inside ``study.load_runtime_packages``. Caches
    are already redirected to a temp directory by the study module at import time; we additionally
    steer torch's cache there so a VM needs no environment exports.
    """
    global np, pd, plt, PdfPages
    os.environ.setdefault("TORCH_HOME", str(study.RUNTIME_CACHE / "torch"))
    study.load_runtime_packages()
    np, pd, plt, PdfPages = study.np, study.pd, study.plt, study.PdfPages


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SecondaryConfig:
    """Resolved configuration for one secondary-analyses run.

    Holds a production ``study.RunConfig`` (used to drive the acquire so re-materialized cohorts
    match the source run) plus the analysis toggles specific to this run. Frozen for determinism;
    ``config_hash`` is computed over the resolved, serialisable fields.
    """

    mode: str
    output_dir: str
    from_run: str | None = None
    resume: bool = False
    seed: int = SEED
    incretin_qualifying_months: int = study.INCRETIN_QUALIFYING_MONTHS
    skip_tte: bool = False
    loso_refit: bool = False
    mi_imputations: int = MI_IMPUTATIONS_DEFAULT
    neural_outcome_model: bool = False
    bootstrap_replicates: int = BOOTSTRAP_FULL
    nuisance_folds: int = NUISANCE_FOLDS
    orchestrate: bool = False

    @property
    def smoke(self) -> bool:
        return self.mode == "smoke"

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir)

    @property
    def secondary_dir(self) -> Path:
        return self.run_dir / "secondary"

    @property
    def checkpoints_dir(self) -> Path:
        return self.secondary_dir / "checkpoints"

    @property
    def export_dir(self) -> Path:
        return self.secondary_dir / "FIGURES_TO_EXPORT"

    def study_config(self) -> "study.RunConfig":
        """A production RunConfig for the acquire, matching this run's estimand knob."""
        study_mode = "smoke" if self.smoke else "production"
        return study.RunConfig.create(
            study_mode,
            self.output_dir,
            False,
            incretin_qualifying_months=self.incretin_qualifying_months,
        )


def _default_output_dir(from_run: str | None, mode: str) -> str:
    """Resolve the run directory. --from-run/--plot-only nest under the source run; others make
    a fresh timestamped directory under ./results so a smoke or full run is self-contained."""
    if from_run is not None:
        return str(Path(from_run).expanduser().resolve())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path.cwd() / "results"
    candidate = root / f"secondary_analyses_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"secondary_analyses_{stamp}_{suffix:02d}"
        suffix += 1
    return str(candidate)


def resolve_config(args: argparse.Namespace) -> SecondaryConfig:
    mode = "from-run"
    if args.full:
        mode = "full"
    elif args.smoke:
        mode = "smoke"
    elif args.self_test:
        mode = "self-test"
    elif args.plot_only:
        mode = "plot-only"

    from_run = args.from_run
    if mode in {"from-run", "plot-only"} and not from_run and not args.output_dir:
        raise SystemExit(
            f"--{mode.replace('from-run', 'from-run')} requires --from-run RUN_DIR "
            "(or --output-dir for --plot-only)"
        )
    output_dir = args.output_dir or _default_output_dir(from_run, mode)
    bootstrap = BOOTSTRAP_SMOKE if mode == "smoke" else BOOTSTRAP_FULL
    return SecondaryConfig(
        mode=mode,
        output_dir=output_dir,
        from_run=from_run,
        resume=bool(args.resume),
        seed=int(args.seed),
        incretin_qualifying_months=int(args.incretin_qualifying_months),
        skip_tte=bool(args.skip_tte),
        loso_refit=bool(args.loso_refit),
        mi_imputations=int(args.mi_imputations),
        neural_outcome_model=bool(args.neural_outcome_model),
        bootstrap_replicates=bootstrap,
        nuisance_folds=NUISANCE_FOLDS,
        orchestrate=bool(args.orchestrate),
    )


# --------------------------------------------------------------------------------------------
# Identity, hashing, small-cell disclosure (reuse study helpers wherever they exist)
# --------------------------------------------------------------------------------------------
def script_sha256() -> str:
    return study.sha256_file(_THIS_FILE)


def config_hash(cfg: SecondaryConfig) -> str:
    payload = {
        "secondary_version": SECONDARY_VERSION,
        "source_study_version": SOURCE_STUDY_VERSION,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "incretin_qualifying_months": cfg.incretin_qualifying_months,
        "skip_tte": cfg.skip_tte,
        "loso_refit": cfg.loso_refit,
        "mi_imputations": cfg.mi_imputations,
        "neural_outcome_model": cfg.neural_outcome_model,
        "bootstrap_replicates": cfg.bootstrap_replicates,
        "nuisance_folds": cfg.nuisance_folds,
        "script_sha256": script_sha256(),
    }
    return study.digest(payload)


def source_run_fingerprint(cfg: SecondaryConfig) -> dict[str, Any]:
    """A light fingerprint of the source run directory (name + manifest hash if present)."""
    run_dir = cfg.run_dir
    fingerprint: dict[str, Any] = {"run_dir": str(run_dir), "exists": run_dir.exists()}
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            fingerprint["run_manifest_sha256"] = study.sha256_file(manifest_path)
        except OSError:
            fingerprint["run_manifest_sha256"] = None
    return fingerprint


def suppress(frame: Any, count_columns: Sequence[str]) -> Any:
    """Disclosure control: reuse the production small-cell suppressor (n < 11)."""
    return study.suppress_small_cells(frame, list(count_columns))


def stable_fraction(value: str, seed: int) -> float:
    return study.stable_hash_fraction(value, seed)


def patient_folds(patient_ids: Any, seed: int, folds: int = NUISANCE_FOLDS, salt: str = "xfit") -> Any:
    """Deterministic patient-clustered fold ids, mirroring the production weighting cross-fit."""
    return np.array(
        [int(stable_fraction(f"{salt}|{pid}", seed) * folds) % folds for pid in patient_ids],
        dtype=int,
    )


# --------------------------------------------------------------------------------------------
# Progress image and failure reporting (mirror the production failure-render contract)
# --------------------------------------------------------------------------------------------
def update_progress_image(cfg: SecondaryConfig, statuses: Mapping[str, str]) -> None:
    """Write secondary/00_progress.png - a simple stage-status figure visible in the filesystem.

    Never raises: progress reporting must not be able to fail a scientific run.
    """
    try:
        cfg.secondary_dir.mkdir(parents=True, exist_ok=True)
        study.configure_figure_style()
        figure = plt.figure(figsize=(7.0, 5.0))
        figure.patch.set_facecolor(study.PALETTE["paper"])
        figure.text(0.06, 0.94, "Secondary analyses - stage status", fontsize=14,
                    fontweight="bold", color=study.PALETTE["ink"], va="top")
        figure.text(0.06, 0.89, f"mode: {cfg.mode}   |   {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    fontsize=9, color=study.PALETTE["muted"], va="top")
        colors = {
            "done": study.PALETTE["green"],
            "running": study.PALETTE["blue"],
            "skipped": study.PALETTE["muted"],
            "failed": study.PALETTE["red"],
            "pending": study.PALETTE["grid"],
        }
        marks = {"done": "PASS", "running": "....", "skipped": "SKIP", "failed": "FAIL", "pending": "  - "}
        y = 0.80
        for stage in STAGE_SEQUENCE:
            state = statuses.get(stage, "pending")
            figure.text(0.10, y, marks.get(state, "  - "), fontsize=11, fontweight="bold",
                        color=colors.get(state, study.PALETTE["grid"]), family="monospace", va="center")
            figure.text(0.24, y, stage, fontsize=11, color=study.PALETTE["ink"], va="center")
            y -= 0.095
        path = cfg.secondary_dir / "00_progress.png"
        temporary = cfg.secondary_dir / "00_progress.png.tmp"
        figure.savefig(temporary, format="png", dpi=150, facecolor=figure.get_facecolor())
        plt.close(figure)
        study.replace_file(temporary, path)
    except Exception as error:  # pragma: no cover - progress must never fail the run
        print(f"[secondary] progress image could not be written: {error!r}", file=sys.stderr)


def render_secondary_failure(cfg: SecondaryConfig, title: str, issues: Sequence[str],
                             details: Sequence[str] = ()) -> Path:
    """Emit a single failure record + a one-page failure PNG + one-page PDF under secondary/.

    Mirrors ``study.render_preflight_failure``, reusing the production hand-rolled (matplotlib-free)
    ``write_failure_png`` / ``write_failure_pdf`` so the report survives even under memory pressure.
    The machine-readable JSON is written first.
    """
    export = cfg.export_dir
    export.mkdir(parents=True, exist_ok=True)
    study.atomic_json(
        cfg.secondary_dir / "secondary_preflight_failure.json",
        {
            "status": "secondary_preflight_failure",
            "title": title,
            "issues": list(issues),
            "details": list(details),
            "time_utc": study.utc_now(),
        },
    )
    png = export / FAILURE_PNG
    pdf = export / FIGURE_BOOK_PDF
    try:
        study.write_failure_png(png, title, list(issues), list(details))
        study.write_failure_pdf(pdf, title, list(issues), list(details))
    except Exception as render_error:  # never let report rendering mask the real failure
        print(f"[secondary] failure report images could not be rendered: {render_error!r}",
              file=sys.stderr)
    return png


class SecondaryPreflightError(RuntimeError):
    """Raised when a preflight gate blocks the run; carries a title, issues, and details."""

    def __init__(self, title: str, issues: Sequence[str], details: Sequence[str] = ()) -> None:
        super().__init__(title + ": " + "; ".join(issues))
        self.title = title
        self.issues = list(issues)
        self.details = list(details)


# --------------------------------------------------------------------------------------------
# Prediction store reopen and selected-model table (from a completed run)
# --------------------------------------------------------------------------------------------
def reopen_prediction_store(root: Path) -> "study.PredictionStore":
    """Reopen an on-disk PredictionStore by rebuilding its partition index from the files present.

    A store pickles as its root path plus the partition index; constructing ``PredictionStore(root)``
    alone yields an empty index. We scan ``root`` for ``<cohort>__<outcome>__origin###__NNNN.{parquet,pkl}``
    files, group them into partitions, and recover each task key - so ``.keys()`` / ``.read(key, ...)``
    work exactly as they did in the run that wrote the store. The on-disk extension (not the local
    pyarrow flag) decides the read format so a parquet store reopens correctly.
    """
    root = Path(root)
    store = study.PredictionStore(root)
    store.partitions = {}
    extensions: set[str] = set()
    staged: dict[str, list[tuple[int, str]]] = {}
    for item in sorted(root.glob("*")):
        if not item.is_file() or item.suffix not in (".parquet", ".pkl"):
            continue
        stem = item.name[: -len(item.suffix)]
        base, separator, number = stem.rpartition("__")
        if not separator or not number.isdigit():
            continue
        parts = base.split("__")
        if len(parts) != 3:
            continue
        cohort, outcome, origin_token = parts
        digits = re.sub(r"[^0-9]", "", origin_token)
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
        key = [cohort, outcome, int(re.sub(r"[^0-9]", "", origin_token))]
        store.partitions[base] = {"key": key, "files": [name for _, name in files], "rows": 0}
    if extensions:
        store.parquet = ".parquet" in extensions
        if store.parquet and study._pa_dataset is None:
            raise SecondaryPreflightError(
                "Prediction store is parquet but pyarrow is unavailable",
                ["The frozen store was written as parquet; install pyarrow to read it."],
            )
    return store


def load_selected_and_metadata(run_dir: Path) -> dict[str, Any]:
    """Read the selected-model table + split metadata + identity from the run's figure_data.pkl.

    ``selected`` is not persisted to its own file; it lives inside the pickled figure-data payload
    under the ``"selected"`` key (columns: cohort, outcome, origin_month, selected_candidate,
    validation_crps, validation_standardized_crps, selection_reason). Falls back to the calibration
    checkpoint if the figure-data checkpoint is missing.
    """
    checkpoints = run_dir / "INTERNAL" / "checkpoints"
    for name in ("figure_data.pkl", "calibration.pkl"):
        path = checkpoints / name
        if not path.exists():
            continue
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
        if isinstance(payload, Mapping) and "selected" in payload:
            return {
                "selected": payload["selected"],
                "split": payload.get("split", {}),
                "identity": payload.get("identity", {}),
                "source_checkpoint": name,
            }
    raise SecondaryPreflightError(
        "Could not load the selected-model table",
        [f"No figure_data.pkl or calibration.pkl with a 'selected' key under {checkpoints}"],
    )


def selected_map(selected: Any) -> dict[tuple[str, str, int], str]:
    """(cohort, outcome, origin_month) -> selected_candidate, dropping not-estimable tasks."""
    mapping: dict[tuple[str, str, int], str] = {}
    for row in selected.itertuples(index=False):
        candidate = str(getattr(row, "selected_candidate"))
        if candidate and candidate != "not_estimable":
            mapping[(str(row.cohort), str(row.outcome), int(row.origin_month))] = candidate
    return mapping


# --------------------------------------------------------------------------------------------
# Covariate frame re-materialization
# --------------------------------------------------------------------------------------------
# Patient/baseline-level covariate columns kept on the join table. Trajectory-derived features are
# deliberately excluded (they are per-origin/target and are colliders/mediators at origin 0).
COVARIATE_COLUMNS = (
    "patient_id", "cohort", "treatment", "procedure", "index_ingredient", "index_route",
    "therapy_class", "age_at_index", "baseline_bmi", "baseline_bmi_day", "baseline_hba1c",
    "baseline_hba1c_day", "diabetes_eligible", "diabetes_flag", "hypertension", "dyslipidemia",
    "osa", "insulin", "biguanide", "sglt2", "svi", "ruca", "state", "sex", "race", "ethnicity",
    "coverage", "center_id", "index_year", "smoking",
# Section 4.1's enriched preoperative baselines ride on the study module's own roster, so this
# frame surfaces exactly what the acquire gated in. _dedupe_covariates keeps only the names that
# are actually columns, so an unacquired covariate never materializes as an all-NaN column.
) + study.OPTIONAL_WIDE_NUMERIC_COVARIATES
# The covariate axes whose population must be audited before an analysis that needs them runs.
AUDITED_COLUMNS = (
    "state", "svi", "ruca", "race", "sex", "ethnicity", "index_ingredient", "index_route",
    "procedure", "diabetes_flag", "hypertension", "dyslipidemia", "osa", "insulin", "biguanide",
    "sglt2", "coverage", "index_year", "age_at_index", "baseline_bmi", "baseline_hba1c",
) + study.OPTIONAL_WIDE_NUMERIC_COVARIATES


def _dedupe_covariates(cohorts: Any) -> Any:
    """Reduce a constructed cohort frame to one row per (patient_id, cohort) with covariates.

    Adds ``index_year`` (derived from index_date, since it is created only in the prediction rows
    upstream) and keeps the patient/baseline-level covariate columns that exist.
    """
    frame = cohorts.copy()
    if "index_date" in frame and "index_year" not in frame:
        frame["index_year"] = pd.to_datetime(frame["index_date"], errors="coerce").dt.year
    columns = [column for column in COVARIATE_COLUMNS if column in frame.columns]
    reduced = frame.loc[:, columns].copy()
    reduced = reduced.drop_duplicates(subset=["patient_id", "cohort"], keep="first").reset_index(drop=True)
    reduced["patient_id"] = reduced["patient_id"].astype(str)
    reduced["cohort"] = reduced["cohort"].astype(str)
    return reduced


def materialize_covariate_frame(cfg: SecondaryConfig) -> Any:
    """Re-materialize the patient-level covariate frame using the production acquire.

    --from-run / --full re-run the production streaming acquire (Cosmos on the VM) so cohorts match
    the source run byte-for-byte. --smoke uses the deterministic synthetic bundle. The result is
    persisted to secondary/covariate_frame.parquet (or .pkl) so --plot-only and resume are cheap.
    """
    study_cfg = cfg.study_config()
    if cfg.smoke:
        bundle = study.synthetic_data_bundle(study_cfg)
        artifacts = study.construct_cohorts(bundle)
    else:
        artifacts, _bundle = study.stream_wide_acquire(study_cfg)
    covariates = _dedupe_covariates(artifacts["cohorts"])
    return covariates


# --------------------------------------------------------------------------------------------
# Column-population audit (Section 5)
# --------------------------------------------------------------------------------------------
def column_population_audit(covariates: Any, columns: Sequence[str] = AUDITED_COLUMNS) -> Any:
    """Non-null fraction and distinct-value count per covariate; mark usability.

    An axis is usable only if non-null fraction >= 0.5 and n_distinct >= 2 (tertile splits need
    >= 3, reported via ``usable_tertile``). Never crashes on a missing/empty covariate.
    """
    rows: list[dict[str, Any]] = []
    total = int(len(covariates))
    for column in columns:
        if column not in covariates.columns:
            rows.append({
                "column": column, "present": False, "non_null_fraction": 0.0, "n_distinct": 0,
                "usable": False, "usable_tertile": False,
            })
            continue
        series = covariates[column]
        non_null = series.notna()
        # Treat empty strings and the "<MISSING>"/"not_applicable" sentinels as unpopulated.
        text = series.astype("string")
        blank = text.str.strip().str.lower().isin({"", "nan", "none", "unknown", "<missing>"})
        populated = non_null & ~blank.fillna(False)
        fraction = float(populated.mean()) if total else 0.0
        distinct = int(series.loc[populated].astype("string").nunique())
        rows.append({
            "column": column,
            "present": True,
            "non_null_fraction": fraction,
            "n_distinct": distinct,
            "usable": bool(fraction >= 0.5 and distinct >= 2),
            "usable_tertile": bool(fraction >= 0.5 and distinct >= 3),
        })
    return pd.DataFrame(rows)


def audit_usable(audit: Any, column: str, tertile: bool = False) -> bool:
    """Look up whether an axis passed the population gate."""
    match = audit.loc[audit["column"] == column]
    if match.empty:
        return False
    key = "usable_tertile" if tertile else "usable"
    return bool(match.iloc[0][key])


# --------------------------------------------------------------------------------------------
# Smoke fixture: a schema-faithful prediction store synthesized from real synthetic cohorts
# --------------------------------------------------------------------------------------------
def _synthesize_quantiles(reference: Any, target_value: Any, target_month: Any, seed: int) -> Any:
    """Deterministic, roughly-calibrated quantile ladder around each row's true target value.

    The center tracks the observed target with a small deterministic error; the spread widens with
    the horizon. Uses standard-normal offsets scaled per row, no RNG state, so the smoke store is
    byte-reproducible. Where the target is unobserved the reference (baseline anchor) is used.
    """
    z = _normal_ppf(np.asarray(QUANTILE_LEVELS, dtype=float))
    center = np.where(np.isfinite(target_value), target_value, reference).astype(float)
    center = np.where(np.isfinite(center), center, 0.0)
    months = np.asarray(target_month, dtype=float)
    spread = 1.5 + 0.05 * months
    # A tiny deterministic per-row shift so predictions are not exactly the target (realistic CRPS).
    shift = 0.25 * np.sin(center + months)
    matrix = center[:, None] + shift[:, None] + spread[:, None] * z[None, :]
    return matrix.astype(float)


def _normal_ppf(p: Any) -> Any:
    """Standard-normal inverse CDF via a rational approximation (no scipy dependency)."""
    p = np.asarray(p, dtype=float)
    # Acklam's algorithm, accurate to ~1e-9 over (0,1).
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.zeros_like(p)
    lower = p < plow
    upper = p > phigh
    middle = ~(lower | upper)
    if np.any(lower):
        q = np.sqrt(-2 * np.log(p[lower]))
        out[lower] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                     ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if np.any(upper):
        q = np.sqrt(-2 * np.log(1 - p[upper]))
        out[upper] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q
        out[middle] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return out


def build_smoke_predictions(cfg: SecondaryConfig) -> tuple["study.PredictionStore", Any, Any]:
    """Build a synthetic-but-schema-faithful calibrated PredictionStore + selected + covariates.

    Reuses production cohort construction, split assignment, prediction-row building, and IPCW
    weight estimation on the synthetic bundle (so target_value/target_observed/support_status/
    analysis_weight/split are all real), then attaches synthetic quantile predictions. This
    exercises the entire secondary stage graph without invoking production model training or torch.
    Returns the store, the selected-model table, and the deduplicated covariate frame - all derived
    from ONE bundle build so smoke does no redundant work.
    """
    study_cfg = cfg.study_config()
    bundle = study.synthetic_data_bundle(study_cfg)
    artifacts = study.construct_cohorts(bundle)
    cohorts_split, _split_meta = study.assign_global_splits(artifacts["cohorts"], seed=cfg.seed)
    covariates = _dedupe_covariates(cohorts_split)

    unweighted_root = cfg.checkpoints_dir / "smoke_rows_unweighted"
    weighted_root = cfg.checkpoints_dir / "smoke_rows_weighted"
    row_store = study.RowStore(unweighted_root)
    study.build_prediction_rows(cohorts_split, artifacts["measurements"], row_store)
    weighted_store = study.RowStore(weighted_root)
    study.estimate_weights_over_store(row_store, weighted_store, seed=cfg.seed)

    predictions_root = cfg.run_dir / "INTERNAL" / "predictions" / "calibrated"
    import shutil
    shutil.rmtree(predictions_root, ignore_errors=True)
    store = study.PredictionStore(predictions_root)
    selected_rows: list[dict[str, Any]] = []
    candidate = "histogram_gradient_boosting"
    architecture = "hist_gradient_boosting_quantile"
    for key in weighted_store.keys():
        rows = weighted_store.read(key)
        if rows.empty:
            continue
        reference = pd.to_numeric(rows.get("prediction_reference_value"), errors="coerce")
        if reference is None or reference.isna().all():
            reference = pd.to_numeric(rows.get("baseline_value"), errors="coerce")
        target = pd.to_numeric(rows["target_value"], errors="coerce").to_numpy(float)
        matrix = _synthesize_quantiles(
            reference.to_numpy(float), target, rows["target_month"].to_numpy(float), cfg.seed
        )
        frame = pd.DataFrame({column: rows[column].to_numpy() for column in study.PREDICTION_IDENTITY_COLUMNS
                              if column in rows.columns})
        for column in study.PREDICTION_IDENTITY_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        frame["candidate"] = candidate
        frame["architecture"] = architecture
        for index, column in enumerate(QUANTILE_COLS):
            frame[column] = matrix[:, index].astype("float32")
        frame = study.drop_training_predictions(frame)
        if not frame.empty:
            store.add(frame[list(study.STORED_PREDICTION_COLUMNS)], key=key)
        cohort, outcome, origin = key
        selected_rows.append({
            "cohort": cohort, "outcome": outcome, "origin_month": int(origin),
            "selected_candidate": candidate, "validation_crps": float("nan"),
            "validation_standardized_crps": float("nan"), "selection_reason": "smoke_synthetic",
        })
        del rows, frame
        gc.collect()
    selected = pd.DataFrame(selected_rows)
    return store, selected, covariates


# --------------------------------------------------------------------------------------------
# Data assembly stage
# --------------------------------------------------------------------------------------------
def stage_assemble(cfg: SecondaryConfig) -> dict[str, Any]:
    """Materialize covariates, load/synthesize frozen predictions + selected, join, and audit.

    Persists: secondary/covariate_frame.parquet, secondary/column_population_audit.csv, and an
    assemble checkpoint carrying the store root, the selected table, the audit, and preflight facts.
    """
    if cfg.smoke:
        store, selected, covariates = build_smoke_predictions(cfg)
        split_meta: dict[str, Any] = {}
    else:
        covariates = materialize_covariate_frame(cfg)
        loaded = load_selected_and_metadata(cfg.run_dir)
        selected = loaded["selected"]
        split_meta = loaded.get("split", {})
        store = reopen_prediction_store(cfg.run_dir / "INTERNAL" / "predictions" / "calibrated")
    _write_frame(cfg.secondary_dir / "covariate_frame.parquet", covariates)

    keys = store.keys()
    if not keys:
        raise SecondaryPreflightError(
            "The source prediction store is empty or unreadable",
            [f"No task partitions found under {store.root}"],
        )

    # Join-rate check: covariate frame must join to >= 99% of prediction patient_ids.
    join_rate, join_detail = _prediction_join_rate(store, covariates)

    audit = column_population_audit(covariates)
    _write_csv(cfg.secondary_dir / "column_population_audit.csv", audit)

    # Both surgical arms present with >= MIN_CELL_SIZE patients each (else force skip_tte).
    surgical = covariates.loc[covariates["cohort"] == "surgery"]
    arm_counts = {
        arm: int((surgical["procedure"].astype(str) == arm).sum()) for arm in SURGERY_ARMS
    }
    tte_gate_ok = all(count >= MIN_CELL_SIZE for count in arm_counts.values())
    tte_skip_reason = "" if tte_gate_ok else f"arm counts below {MIN_CELL_SIZE}: {arm_counts}"

    checkpoint = {
        "store_root": str(store.root),
        "store_parquet": bool(store.parquet),
        "store_keys": [list(key) for key in keys],
        "selected": selected,
        "split_meta": split_meta,
        "audit": audit,
        "join_rate": join_rate,
        "join_detail": join_detail,
        "arm_counts": arm_counts,
        "tte_gate_ok": bool(tte_gate_ok and not cfg.skip_tte),
        "tte_skip_reason": tte_skip_reason if not cfg.skip_tte else "forced by --skip-tte",
        "covariate_rows": int(len(covariates)),
    }
    _save_checkpoint(cfg, "assemble", checkpoint)
    return checkpoint


def _prediction_join_rate(store: "study.PredictionStore", covariates: Any) -> tuple[float, dict[str, Any]]:
    """Fraction of distinct prediction patient_ids that join to the covariate frame, per cohort."""
    covariate_keys = set(zip(covariates["patient_id"].astype(str), covariates["cohort"].astype(str)))
    seen: set[tuple[str, str]] = set()
    joined: set[tuple[str, str]] = set()
    for key in store.keys():
        frame = store.read(key, columns=["patient_id", "cohort"])
        if frame.empty:
            continue
        pairs = set(zip(frame["patient_id"].astype(str), frame["cohort"].astype(str)))
        seen |= pairs
        joined |= {pair for pair in pairs if pair in covariate_keys}
    rate = float(len(joined) / len(seen)) if seen else 0.0
    return rate, {"distinct_prediction_patients": len(seen), "joined": len(joined)}


# --------------------------------------------------------------------------------------------
# Target-trial emulation (TTE): wide-source point-intervention sleeve-vs-RYGB, IPCW-AIPW.
# Estimand: marginal ATE of RYGB vs sleeve on BMI (all eligible surgical patients) and HbA1c
# (diabetes-eligible subset only) at each supported horizon, doubly-robust and cross-fitted.
# --------------------------------------------------------------------------------------------
# Confounders L (baseline, pre-treatment ONLY). baseline_value is chosen per outcome
# (baseline_bmi or baseline_hba1c). Trajectory-derived features, the arm, treatment, and race
# are deliberately excluded (spec Section 6 Part A; causal_tte DECISION 3 excludes race from L).
TTE_NUMERIC_CONFOUNDERS = (
    "baseline_value", "age_at_index", "diabetes_flag", "hypertension", "dyslipidemia",
    "osa", "insulin", "biguanide", "sglt2", "svi", "index_year",
)
TTE_BASE_CATEGORICAL_CONFOUNDERS = ("sex", "ethnicity", "coverage")
# state / ruca and Section 4.1's enriched baselines (creatinine, eGFR, GERD, the lab panel) enter
# L only when the column-population audit marks them usable; the rest are named absent confounders.
TTE_OPTIONAL_CATEGORICAL_CONFOUNDERS = ("state", "ruca")
TTE_OPTIONAL_NUMERIC_CONFOUNDERS = study.OPTIONAL_WIDE_NUMERIC_COVARIATES

# Positivity / overlap gate thresholds (production gate, spec Section 6 Part A).
POSITIVITY_MIN_ESS = 50.0
POSITIVITY_MIN_ESS_FRACTION = 0.2
POSITIVITY_MIN_PS = 0.01
IPTW_TRIM = (0.02, 0.98)
# Minimum observed training rows before the g-computation outcome model is fit (else fall back).
TTE_MIN_MU_FIT = 20

TTE_AIPW_COLUMNS = (
    "outcome", "target_month", "n", "n_rygb", "n_sleeve", "ess_iptw", "ate", "se",
    "ci_low", "ci_high", "e_value_point", "e_value_ci", "rct_anchor", "overlaps_rct_ci",
    "c_for_benefit", "max_abs_smd_weighted", "positivity_fail", "estimand_note",
)
TTE_BALANCE_COLUMNS = ("outcome", "covariate", "smd_unweighted", "smd_weighted")


def _tte_baseline_column(outcome: str) -> str:
    """The cohort-level baseline covariate that becomes L's baseline_value for this outcome."""
    return "baseline_hba1c" if outcome == "hba1c" else "baseline_bmi"


def _tte_population_label(outcome: str) -> str:
    return "HbA1c (diabetes-eligible)" if outcome == "hba1c" else "BMI (all eligible surgical)"


def _tte_confounder_lists(audit: Any) -> tuple[list[str], list[str], list[str]]:
    """Numeric and categorical L column lists, plus the optional confounders the audit refused.

    One gate covers every optional confounder: state/ruca and the enriched baselines alike join L
    only when the population audit marks the column usable. An unpopulated one (GERD and the lab
    panel are not acquirable from this source) is OMITTED from L rather than carried as an all-NaN
    confounder, and comes back in the third list as a named absent confounder.
    """
    numeric = list(TTE_NUMERIC_CONFOUNDERS)
    categorical = list(TTE_BASE_CATEGORICAL_CONFOUNDERS)
    absent: list[str] = []
    for optional, accepted in ((TTE_OPTIONAL_NUMERIC_CONFOUNDERS, numeric),
                               (TTE_OPTIONAL_CATEGORICAL_CONFOUNDERS, categorical)):
        for column in optional:
            (accepted if audit is not None and audit_usable(audit, column) else absent).append(column)
    return numeric, categorical, sorted(absent)


def _encoded_feature_names(encoder: Any) -> list[str]:
    """Reconstruct study.TabularEncoder.transform's output column names, in emission order.

    transform emits, per numeric column, a value column then a missingness flag; per categorical
    column, one indicator per known level (in encoder.levels order) then an out-of-vocabulary
    column. Reproducing that order lets the Love plot / balance table label each encoded column.
    """
    names: list[str] = []
    for column in encoder.numeric:
        names.append(column)
        names.append(f"{column}__missing")
    for column in encoder.categorical:
        for level in encoder.levels[column]:
            names.append(f"{column}={level}")
        names.append(f"{column}__unknown")
    return names


def build_wide_L_A(surgical_covariates: Any, outcome: str, audit: Any) -> tuple[Any, Any, list[str], Any]:
    """Wide-native confounder matrix L, arm A (1=RYGB, 0=sleeve), encoded names, and the encoder.

    The population frame is pre-filtered by the caller (procedure in {sleeve, rygb}; for hba1c the
    diabetes-eligible subset). baseline_value is selected per outcome. Encoding uses
    study.TabularEncoder restricted to the L subset (numeric value + missingness flag; categorical
    one-hot + out-of-vocabulary), which the tree nuisance models consume directly. Never includes
    the arm, treatment, race, or any trajectory-derived feature.
    """
    frame = surgical_covariates.copy()
    frame["baseline_value"] = pd.to_numeric(frame[_tte_baseline_column(outcome)], errors="coerce")
    numeric, categorical, _absent = _tte_confounder_lists(audit)
    numeric = [column for column in numeric if column == "baseline_value" or column in frame.columns]
    categorical = [column for column in categorical if column in frame.columns]
    encoder = study.TabularEncoder.fit(frame, numeric=numeric, categorical=categorical)
    L = encoder.transform(frame)
    A = (frame["procedure"].astype("string").str.lower() == "rygb").astype(int).to_numpy()
    return L, A, _encoded_feature_names(encoder), encoder


def _tte_population(surgical: Any, outcome: str) -> Any:
    """The eligible surgical analysis population for one outcome (RYGB/sleeve, non-null baseline).

    BMI uses all eligible surgical patients; HbA1c restricts to the diabetes-eligible subset.
    """
    procedure = surgical["procedure"].astype("string").str.lower()
    pop = surgical.loc[procedure.isin(["sleeve", "rygb"])].copy()
    baseline_column = _tte_baseline_column(outcome)
    pop = pop.loc[pd.to_numeric(pop[baseline_column], errors="coerce").notna()]
    if outcome == "hba1c" and "diabetes_eligible" in pop.columns:
        eligible = pop["diabetes_eligible"].fillna(False).astype(bool)
        pop = pop.loc[eligible]
    return pop.reset_index(drop=True)


def _tte_outcome_regressor(seed: int) -> Any:
    """Median (absolute-error) HistGradientBoosting regressor for cross-fitted g-computation.

    torch-free and deterministic. loss='absolute_error' targets E[Y|A,L] robustly; a TypeError on
    older scikit-learn (which named the loss differently) falls back to the default squared loss.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    try:
        return HistGradientBoostingRegressor(
            loss="absolute_error", random_state=int(seed), max_depth=3, max_iter=200,
            early_stopping=False,
        )
    except TypeError:
        return HistGradientBoostingRegressor(random_state=int(seed), max_depth=3, max_iter=200)


def _crossfit_propensity(L: Any, A: Any, patient_ids: Any, seed: int, folds: int) -> tuple[Any, bool]:
    """Out-of-fold ps = P(A=1|L) via causal_tte.propensity_scores, patient-clustered folds.

    Returns (ps, degenerate); degenerate is True if any fold's classifier could not be fit (None)
    or a fold left an unscored patient (filled with the arm base rate).
    """
    L = np.asarray(L, dtype=float)
    A = np.asarray(A)
    n = int(A.shape[0])
    ps = np.full(n, np.nan)
    fold_ids = patient_folds(patient_ids, seed, folds)
    degenerate = False
    for fold in range(folds):
        test_idx = np.where(fold_ids == fold)[0]
        if test_idx.size == 0:
            continue
        train_idx = np.where(fold_ids != fold)[0]
        ps_test, clf = tte.propensity_scores(L, A, train_idx, test_idx)
        ps[test_idx] = ps_test
        if clf is None:
            degenerate = True
    if np.isnan(ps).any():
        base = float(np.clip(np.mean(A), 0.01, 0.99)) if n else 0.5
        ps = np.where(np.isnan(ps), base, ps)
        degenerate = True
    return ps, degenerate


def _crossfit_mu_pc(L_cell: Any, A_cell: Any, Y: Any, delta: Any, patient_ids: Any,
                    seed: int, folds: int) -> tuple[Any, Any, Any]:
    """Out-of-fold mu1, mu0 (g-computation with the arm forced) and pc = P(observed|L,A).

    mu is fit on observed (delta==1) training rows of [L, A] and predicted twice with A set to 1
    and 0; a fold with too few observed rows falls back to the observed training mean. pc uses
    causal_tte.censoring_model on the arm-augmented design. All predictions are strictly held-out.
    """
    L_cell = np.asarray(L_cell, dtype=float)
    A_cell = np.asarray(A_cell, dtype=float)
    Y = np.asarray(Y, dtype=float)
    delta = np.asarray(delta).astype(int)
    n = int(A_cell.shape[0])
    design = np.column_stack([L_cell, A_cell])
    mu1 = np.full(n, np.nan)
    mu0 = np.full(n, np.nan)
    pc = np.full(n, np.nan)
    fold_ids = patient_folds(patient_ids, seed, folds)
    observed_all = Y[delta == 1]
    global_fill = float(np.nanmean(observed_all)) if observed_all.size else 0.0
    if not np.isfinite(global_fill):
        global_fill = 0.0
    for fold in range(folds):
        test_idx = np.where(fold_ids == fold)[0]
        if test_idx.size == 0:
            continue
        train_idx = np.where(fold_ids != fold)[0]
        observed_train = train_idx[delta[train_idx] == 1]
        design1 = design[test_idx].copy()
        design1[:, -1] = 1.0
        design0 = design[test_idx].copy()
        design0[:, -1] = 0.0
        fitted = False
        if observed_train.size >= TTE_MIN_MU_FIT:
            model = _tte_outcome_regressor(seed)
            try:
                model.fit(design[observed_train], Y[observed_train])
                mu1[test_idx] = model.predict(design1)
                mu0[test_idx] = model.predict(design0)
                fitted = True
            except Exception:
                fitted = False
        if not fitted:
            fill = float(np.nanmean(Y[observed_train])) if observed_train.size else global_fill
            if not np.isfinite(fill):
                fill = global_fill
            mu1[test_idx] = fill
            mu0[test_idx] = fill
        p_obs, _ = tte.censoring_model(design, delta, train_idx, test_idx)
        pc[test_idx] = p_obs
    pc = np.where(np.isfinite(pc), pc, 1.0)
    mu1 = np.where(np.isfinite(mu1), mu1, global_fill)
    mu0 = np.where(np.isfinite(mu0), mu0, global_fill)
    return mu1, mu0, pc


def positivity_gate(ps: Any, arm: Any, degenerate: bool = False, trim: tuple[float, float] = IPTW_TRIM) -> dict[str, Any]:
    """Production positivity/overlap gate for one cell.

    Fails when the propensity nuisance is degenerate, the trimmed IPTW effective sample size is
    below 50 or below 0.2*n, or the minimum propensity is at or below 0.01 (near-deterministic
    assignment). The diagnostics (ESS, min PS) are always returned so a failed cell can still be
    shown; only the point estimate is later suppressed.
    """
    ps = np.asarray(ps, dtype=float)
    arm = np.asarray(arm)
    n = int(ps.size)
    weights, keep = tte.stabilized_iptw(arm, ps, trim=trim)
    kept = weights[keep] if bool(np.any(keep)) else np.asarray([], dtype=float)
    ess = tte.weighted_effective_sample_size(kept)
    finite_ps = ps[np.isfinite(ps)]
    min_ps = float(np.min(finite_ps)) if finite_ps.size else float("nan")
    fail = bool(
        bool(degenerate)
        or finite_ps.size == 0
        or not np.isfinite(ess)
        or ess < POSITIVITY_MIN_ESS
        or ess < POSITIVITY_MIN_ESS_FRACTION * n
        or not np.isfinite(min_ps)
        or min_ps <= POSITIVITY_MIN_PS
    )
    return {
        "positivity_fail": fail,
        "ess_iptw": float(ess) if np.isfinite(ess) else float("nan"),
        "min_ps": min_ps,
        "n": n,
        "degenerate": bool(degenerate),
    }


def _c_for_benefit(predicted_tau: Any, A: Any, Y: Any, ps: Any) -> dict[str, Any]:
    """Concordance-for-benefit for a predicted RYGB-vs-sleeve effect, sign-matched to the metric.

    ``tte.c_for_benefit`` forms the OBSERVED pair benefit as Y[control] - Y[treated], which is
    already higher-is-better for a lower-is-better outcome, but it consumes the PREDICTED vector
    exactly as handed over. Our predicted effect is tau = mu1 - mu0 = E[Y(RYGB)] - E[Y(sleeve)],
    where a NEGATIVE value is the benefit, so tau must be negated before it goes in: passing it raw
    scores a perfect predictor near 0.08 instead of near 0.92. Both the marginal TTE cell and the
    CATE page route through here, so the two can never disagree about the sign.
    """
    return tte.c_for_benefit(-np.asarray(predicted_tau, dtype=float), A, Y, ps, lower_is_better=True)


def _resolve_outcome_model_label(cfg: SecondaryConfig) -> str:
    """Label for the counterfactual outcome model, importing torch lazily only when requested.

    The default self-contained path is the median HistGradientBoosting g-computation. The optional
    --neural-outcome-model flag lazily imports torch; reusing the frozen selected neural model for
    counterfactual scoring is not wired in this build, so it records the request and falls back to
    the HGB g-computation (validation stays torch-free because the flag is never set there).
    """
    if not cfg.neural_outcome_model:
        return "hist_gradient_boosting_absolute_error"
    try:
        import torch  # lazy: torch is imported only inside this explicitly-requested branch

        return f"neural_requested_fallback_hgb(torch-{torch.__version__})"
    except Exception:
        return "neural_requested_unavailable_fallback_hgb"


def _tte_skipped(cfg: SecondaryConfig, reason: str, gate_ok: bool) -> dict[str, Any]:
    """Write empty contract-schema CSVs and a skip checkpoint so pages 02-04 degrade gracefully."""
    _write_csv(cfg.secondary_dir / "tte_aipw.csv",
               pd.DataFrame(columns=list(TTE_AIPW_COLUMNS)), header_note=ESTIMAND_NOTE)
    _write_csv(cfg.secondary_dir / "tte_balance.csv", pd.DataFrame(columns=list(TTE_BALANCE_COLUMNS)))
    _write_cate_csvs(cfg, {})
    payload = {
        "status": "skipped", "skip_reason": reason, "tte_gate_ok": bool(gate_ok),
        "cells": [], "balance": [], "ps_overlap": {}, "love": {}, "populations": {}, "cate": {},
        "outcome_model": "none", "estimand_note": ESTIMAND_NOTE,
    }
    _save_checkpoint(cfg, "tte", payload)
    return payload


def _write_tte_csvs(cfg: SecondaryConfig, cells_frame: Any, balance_frame: Any) -> None:
    """Emit tte_aipw.csv and tte_balance.csv: positivity-suppressed, then n<11 disclosure-blanked."""
    if cells_frame.empty:
        aipw_csv = pd.DataFrame(columns=list(TTE_AIPW_COLUMNS))
    else:
        display = cells_frame.copy()
        fail = display["positivity_fail"].fillna(False).astype(bool)
        for column in ("ate", "se", "ci_low", "ci_high", "e_value_point", "e_value_ci", "c_for_benefit"):
            display.loc[fail, column] = np.nan
        display["overlaps_rct_ci"] = display["overlaps_rct_ci"].astype(object)
        display.loc[fail, "overlaps_rct_ci"] = None
        display["estimand_note"] = ESTIMAND_NOTE
        display = display.loc[:, [column for column in TTE_AIPW_COLUMNS]]
        display = suppress(display, ["n"])
        # target_month is a structural, non-disclosive label; restore it after numeric blanking.
        display["target_month"] = cells_frame["target_month"].to_numpy()
        display["outcome"] = cells_frame["outcome"].to_numpy()
        display["estimand_note"] = ESTIMAND_NOTE
        aipw_csv = display.loc[:, [column for column in TTE_AIPW_COLUMNS]]
    _write_csv(cfg.secondary_dir / "tte_aipw.csv", aipw_csv, header_note=ESTIMAND_NOTE)
    balance_out = balance_frame if not balance_frame.empty else pd.DataFrame(columns=list(TTE_BALANCE_COLUMNS))
    _write_csv(cfg.secondary_dir / "tte_balance.csv", balance_out)


# --------------------------------------------------------------------------------------------
# Purpose-built CATE (spec Section 8): a DR-learner riding on the TTE's own cross-fitted nuisances
# --------------------------------------------------------------------------------------------
# The individualized estimand is carried by ONE cell, the headline BMI horizon: reusing that
# cell's ps / pc / mu1 / mu0 (and its patient-clustered folds) means the CATE costs exactly one
# extra regression and no new large data structure.
CATE_OUTCOME = "bmi"
CATE_TARGET_MONTH = 12
CATE_MIN_N = 100                # below this the deciles and the cross-fit folds are not usable
CATE_DECILES = 10
CATE_PERMUTATION_REPEATS = 5
CATE_TOC_GRID = tuple(round(0.05 * step, 2) for step in range(1, 21))
# Effect modifiers X (spec 8.2). The optional ones (creatinine/eGFR, GERD, the lab panel) join X
# only when the population audit marks them usable, exactly as they join L: an unpopulated modifier
# is OMITTED rather than carried as an all-NaN column, and is named as an absent modifier instead.
CATE_NUMERIC_MODIFIERS = (
    "baseline_bmi", "baseline_hba1c", "age_at_index", "diabetes_flag", "hypertension",
    "dyslipidemia", "osa", "insulin", "biguanide", "sglt2",
)
CATE_CATEGORICAL_MODIFIERS = ("sex", "hba1c_band")
CATE_OPTIONAL_NUMERIC_MODIFIERS = study.OPTIONAL_WIDE_NUMERIC_COVARIATES
CATE_CALIBRATION_COLUMNS = ("decile", "n", "mean_predicted_benefit", "aipw_benefit",
                            "ci_low", "ci_high")
CATE_IMPORTANCE_COLUMNS = ("modifier", "permutation_importance")
# Spec 8.4 is an OPTIONAL cross-check. GRF/econml is not a confirmed dependency of the production
# VM and this script imports nothing outside study/causal_tte, so the causal forest is a DOCUMENTED
# omission (it reaches the page and the ledger) rather than a silent one.
CAUSAL_FOREST_NOTE = (
    "Optional causal-forest cross-check (spec 8.4) deliberately omitted: GRF/econml is outside "
    "this script's bundled dependencies, so the cross-fitted DR-learner is the sole CATE estimator."
)
CATE_CAVEAT = (
    "Heterogeneity map, not an individualized prescription. tau_hat reads MEASURED effect "
    "modification only, and an unmeasured modifier can masquerade as heterogeneity. Treat the "
    "ranking as a hypothesis about who may benefit more, not as a treatment assignment rule, "
    "unless the RATE interval clearly excludes zero."
)


def _dr_pseudo_outcome(Y: Any, A: Any, delta: Any, ps: Any, pc: Any, mu1: Any, mu0: Any) -> Any:
    """Per-patient doubly-robust (efficient influence function) pseudo-outcome, spec 8.1.

    This is exactly the per-row term ``tte.aipw`` averages into the marginal ATE, restated so the
    DR-learner can regress on it:

      psi = (mu1 - mu0) + [A / ps] * w * (Y - mu1) - [(1 - A) / (1 - ps)] * w * (Y - mu0)

    The IPCW weight w is the one the AIPW itself applies, ``delta / pc`` from the censoring
    nuisance (NOT the store's ``analysis_weight``, which is a prognostic-evaluation weight and
    never enters this estimator), and the ps / pc clipping constants are the AIPW's own, so the
    pseudo-outcome and the marginal estimate can never drift apart. A = 1 is RYGB, so psi estimates
    Y(RYGB) - Y(sleeve) in the outcome's units and NEGATIVE psi means RYGB lowered BMI.
    """
    Y = np.asarray(Y, dtype=float)
    A = np.asarray(A, dtype=float)
    ps = np.asarray(ps, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    w_ipcw = np.where(np.asarray(delta) == 1, 1.0 / np.clip(np.asarray(pc, dtype=float), 0.05, 1.0), 0.0)
    observed = np.nan_to_num(Y)  # unobserved rows carry weight 0 through w_ipcw
    return ((mu1 - mu0)
            + A * w_ipcw * (observed - mu1) / np.clip(ps, 1e-3, 1.0)
            - (1.0 - A) * w_ipcw * (observed - mu0) / np.clip(1.0 - ps, 1e-3, 1.0))


def _cate_regressor(seed: int) -> Any:
    """Squared-loss HistGradientBoosting for the DR-learner (torch-free, seeded, deterministic).

    tau(x) = E[psi | X = x] is a conditional MEAN, so unlike the g-computation nuisance this fit
    must not be the median (absolute-error) one.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(random_state=int(seed), max_depth=3, max_iter=200,
                                         early_stopping=False)


def _cate_modifier_design(rows: Any, audit: Any) -> tuple[Any, list[tuple[str, list[int]]], list[str]]:
    """Encoded modifier design X, its per-modifier column groups, and the absent modifiers.

    Encoding is the production ``study.TabularEncoder`` (numeric value + missingness flag,
    categorical one-hot + out-of-vocabulary), the same encoder L uses. ``hba1c_band`` is derived
    from baseline HbA1c with the bands the subgroup pages already use. Columns the encoder did not
    keep (an audit-refused optional modifier, or one this source never carried) come back as named
    absent modifiers so the page and the omissions ledger can say so.
    """
    frame = rows.reset_index(drop=True).copy()
    if "baseline_hba1c" in frame.columns:
        frame["hba1c_band"] = _tier2_hba1c_band_labels(frame["baseline_hba1c"]).to_numpy()
    absent = _cate_absent_modifiers(audit)
    numeric = list(CATE_NUMERIC_MODIFIERS) + [column for column in CATE_OPTIONAL_NUMERIC_MODIFIERS
                                              if column not in absent]
    categorical = list(CATE_CATEGORICAL_MODIFIERS)
    encoder = study.TabularEncoder.fit(frame, numeric=numeric, categorical=categorical)
    kept = set(encoder.numeric) | set(encoder.categorical)
    absent.extend(column for column in numeric + categorical if column not in kept)
    return encoder.transform(frame), _cate_modifier_groups(encoder), sorted(set(absent))


def _cate_absent_modifiers(audit: Any) -> list[str]:
    """The optional modifiers (creatinine/eGFR, GERD, the lab panel) the population audit refused.

    Derived from the audit alone, so the honesty gate can name them even on a run where the CATE
    itself was not estimable, exactly as the TTE names its absent confounders whether or not it ran.
    Sorted, so the estimable and not-estimable payloads order the same names the same way.
    """
    return sorted(column for column in CATE_OPTIONAL_NUMERIC_MODIFIERS
                  if audit is None or not audit_usable(audit, column))


def _cate_modifier_groups(encoder: Any) -> list[tuple[str, list[int]]]:
    """Encoded-column indices grouped by SOURCE modifier, in ``TabularEncoder.transform`` order.

    Permuting a categorical's one-hot levels one at a time would understate it, so importance is
    reported per clinical variable: every encoded column a modifier produced moves together.
    """
    groups: list[tuple[str, list[int]]] = []
    index = 0
    for column in encoder.numeric:
        groups.append((column, [index, index + 1]))
        index += 2
    for column in encoder.categorical:
        width = len(encoder.levels[column]) + 1
        groups.append((column, list(range(index, index + width))))
        index += width
    return groups


def _crossfit_tau(X: Any, psi: Any, patient_ids: Any, seed: int, folds: int,
                  groups: Sequence[tuple[str, list[int]]]) -> tuple[Any, Any]:
    """Out-of-fold tau_hat(x) and per-modifier permutation importance, in ONE pass over the folds.

    ``patient_folds`` is called with the same seed and salt the TTE nuisances used, so the fold
    assignment is byte-identical and no patient's tau_hat is fit on that patient's own psi.
    Importance is the fold-size-weighted mean rise in held-out squared error when a modifier's
    encoded columns are permuted together. Each fold's model and its permutation block are freed
    before the next fold is fit, so only one fold of design matrix is live at a time.
    """
    X = np.asarray(X, dtype=float)
    psi = np.asarray(psi, dtype=float)
    n = int(psi.size)
    tau = np.full(n, np.nan)
    importance = np.zeros(len(groups), dtype=float)
    fold_ids = patient_folds(patient_ids, seed, folds)
    rng = np.random.default_rng(int(seed))
    for fold in range(folds):
        test_idx = np.where(fold_ids == fold)[0]
        if test_idx.size == 0:
            continue
        train_idx = np.where(fold_ids != fold)[0]
        if train_idx.size < TTE_MIN_MU_FIT:  # too thin to fit: fall back to a constant (no signal)
            tau[test_idx] = float(np.mean(psi[train_idx])) if train_idx.size else 0.0
            continue
        model = _cate_regressor(seed)
        model.fit(X[train_idx], psi[train_idx])
        # Fancy indexing already hands back a private copy of the fold, so the permutations below
        # shuffle it in place without touching X and without a second fold-sized allocation.
        block = X[test_idx]
        prediction = model.predict(block)
        tau[test_idx] = prediction
        baseline = float(np.mean((psi[test_idx] - prediction) ** 2))
        for position, (_name, columns) in enumerate(groups):
            original = block[:, columns].copy()
            total = 0.0
            for _repeat in range(CATE_PERMUTATION_REPEATS):
                block[:, columns] = original[rng.permutation(test_idx.size)]
                total += float(np.mean((psi[test_idx] - model.predict(block)) ** 2))
            block[:, columns] = original
            importance[position] += (total / CATE_PERMUTATION_REPEATS - baseline) * test_idx.size
        del model, block
    fill = float(np.mean(psi[np.isfinite(psi)])) if np.isfinite(psi).any() else 0.0
    return np.where(np.isfinite(tau), tau, fill), importance / max(n, 1)


def _toc_and_rate(rank: Any, benefit_psi: Any) -> tuple[Any, float]:
    """Yadlowsky TOC over every prefix of the benefit ranking, and the RATE (AUTOC) it integrates.

    TOC(q) is the mean doubly-robust BENEFIT among the top q fraction minus the population mean,
    so a ranking that really does put higher-benefit patients first has a positive, decaying TOC
    that reaches exactly zero at q = 1; RATE is its average over q. Ties keep source order (stable
    sort), so the curve and its bootstrap are deterministic.
    """
    order = np.argsort(-np.asarray(rank, dtype=float), kind="stable")
    cumulative = np.cumsum(np.asarray(benefit_psi, dtype=float)[order])
    prefix = np.arange(1, cumulative.size + 1, dtype=float)
    toc = cumulative / prefix - cumulative[-1] / cumulative.size
    return toc, float(np.mean(toc))


def _cate_dr_learner(cfg: SecondaryConfig, rows: Any, audit: Any, Y: Any, A: Any, delta: Any,
                     ps: Any, pc: Any, mu1: Any, mu0: Any, patient_ids: Any,
                     gate: Mapping[str, Any]) -> dict[str, Any]:
    """DR-learner CATE for one sleeve-vs-RYGB cell (spec 8.1 - 8.3), on the TTE's own nuisances.

    ps / pc / mu1 / mu0 arrive already cross-fitted by ``stage_tte`` over the same patient folds
    this regression reuses, so nothing here refits a nuisance. Orientation: tau_hat estimates
    E[Y(RYGB) - Y(sleeve) | X] in BMI units and lower BMI is better, so everything the page reports
    is stated in BENEFIT units, benefit = -tau, where a larger number means RYGB helps more.

    A cell whose positivity gate failed has no usable propensity overlap, so the pseudo-outcome is
    dominated by the clipped 1/ps term: that cell returns not-estimable rather than a fitted curve
    over noise, exactly as the TTE pages suppress a failed cell's point estimate.
    """
    n = int(np.asarray(Y).size)
    if bool(gate.get("positivity_fail")) or n < CATE_MIN_N:
        reason = (f"positivity gate failed for the {CATE_OUTCOME} {CATE_TARGET_MONTH}-month cell "
                  f"(minimum propensity {gate.get('min_ps', float('nan')):.2e}), so the "
                  "doubly-robust pseudo-outcome has no propensity overlap to stand on"
                  if bool(gate.get("positivity_fail")) else
                  # Reaches the page-16 ledger, so the count is suppressed like every other one.
                  f"cell has {study.display_count(n)} patients, below the {CATE_MIN_N} the "
                  "decile calibration needs")
        return {"status": "not_estimable", "n": n, "skipped_reason": reason,
                "outcome": CATE_OUTCOME, "target_month": CATE_TARGET_MONTH,
                "absent_modifiers": _cate_absent_modifiers(audit),
                "calibration": [], "importance": []}

    Y = np.asarray(Y, dtype=float)
    A = np.asarray(A).astype(int)
    delta = np.asarray(delta).astype(int)
    ps = np.asarray(ps, dtype=float)
    pc = np.asarray(pc, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    psi = _dr_pseudo_outcome(Y, A, delta, ps, pc, mu1, mu0)
    X, groups, absent = _cate_modifier_design(rows, audit)
    tau, importance = _crossfit_tau(X, psi, patient_ids, cfg.seed, cfg.nuisance_folds, groups)
    del X
    benefit_hat = -tau            # predicted BMI reduction from choosing RYGB
    benefit_psi = -psi            # its doubly-robust observed counterpart

    # RATE / TOC with inference: a patient-clustered bootstrap over ROW INDICES (the cell is one
    # row per patient), which never copies a frame and re-ranks the same two vectors per replicate.
    grid = np.asarray(CATE_TOC_GRID, dtype=float)
    grid_index = np.clip((grid * n).astype(int) - 1, 0, n - 1)
    toc, rate = _toc_and_rate(benefit_hat, benefit_psi)
    replicates = int(cfg.bootstrap_replicates)
    rates = np.empty(replicates, dtype=float)
    band = np.empty((replicates, grid.size), dtype=float)
    rng = np.random.default_rng([int(cfg.seed), int(CATE_TARGET_MONTH)])
    for replicate in range(replicates):
        resample = rng.integers(0, n, size=n)
        curve, rates[replicate] = _toc_and_rate(benefit_hat[resample], benefit_psi[resample])
        band[replicate] = curve[grid_index]
    finite = rates[np.isfinite(rates)]
    rate_low = rate_high = float("nan")
    if finite.size >= 2:
        rate_low, rate_high = (float(value) for value in np.percentile(finite, [2.5, 97.5]))
    toc_low, toc_high = np.percentile(band, [2.5, 97.5], axis=0)
    del band, rates

    # Benefit calibration: equal-size predicted-benefit deciles, each re-estimated with the SAME
    # doubly-robust estimator the marginal ATE uses, so the observed axis is not the predicted one.
    order = np.argsort(benefit_hat, kind="stable")
    decile = np.empty(n, dtype=int)
    decile[order] = (np.arange(n) * CATE_DECILES) // n
    calibration: list[dict[str, Any]] = []
    for index in range(CATE_DECILES):
        mask = decile == index
        estimate = tte.aipw(Y[mask], A[mask], delta[mask], ps[mask], pc[mask], mu1[mask], mu0[mask])
        low, high = estimate["ci"]
        calibration.append({
            "decile": index + 1, "n": int(mask.sum()),
            "mean_predicted_benefit": float(np.mean(benefit_hat[mask])) if bool(mask.any()) else float("nan"),
            "aipw_benefit": -float(estimate["ate"]),
            "ci_low": -float(high), "ci_high": -float(low),  # negation swaps the interval ends
        })
    observed = np.asarray([row["aipw_benefit"] for row in calibration], dtype=float)
    estimable = np.isfinite(observed)
    rho = float("nan")
    if int(estimable.sum()) >= 3:
        ranked = np.argsort(np.argsort(observed[estimable], kind="stable"), kind="stable").astype(float)
        rho = float(np.corrcoef(np.arange(CATE_DECILES, dtype=float)[estimable], ranked)[0, 1])
    steps = np.diff(observed[estimable])

    # Policy value of "operate RYGB where tau_hat predicts benefit", read off the same pseudo-
    # outcomes: mean(pi * benefit_psi) against treat-all (mean benefit_psi) and treat-none (0).
    treat = benefit_hat > 0.0
    benefit = _c_for_benefit(tau, A, Y, ps)
    return {
        "status": "done", "outcome": CATE_OUTCOME, "target_month": CATE_TARGET_MONTH, "n": n,
        "n_rygb": int((A == 1).sum()), "n_sleeve": int((A == 0).sum()),
        "rate": rate, "rate_ci_low": rate_low, "rate_ci_high": rate_high,
        "toc_grid": [float(value) for value in grid], "toc": [float(value) for value in toc[grid_index]],
        "toc_ci_low": [float(value) for value in toc_low],
        "toc_ci_high": [float(value) for value in toc_high],
        "calibration": calibration, "calibration_rho": rho,
        "monotone_fraction": float(np.mean(steps >= 0.0)) if steps.size else float("nan"),
        "importance": [{"modifier": name, "permutation_importance": float(value)}
                       for (name, _columns), value in zip(groups, importance, strict=True)],
        "absent_modifiers": absent,
        "c_for_benefit": float(benefit["c_for_benefit"]), "n_pairs": int(benefit["n_pairs"]),
        "policy_value": float(np.mean(np.where(treat, benefit_psi, 0.0))),
        "treat_all_value": float(np.mean(benefit_psi)), "treated_fraction": float(np.mean(treat)),
        "ess_iptw": float(gate.get("ess_iptw", float("nan"))),
    }


def _write_cate_csvs(cfg: SecondaryConfig, cate: Mapping[str, Any]) -> None:
    """Benefit-calibration deciles and per-modifier permutation importance, n < 11 suppressed."""
    _write_tier1_csv(cfg.secondary_dir / "cate_benefit_calibration.csv",
                     _rows_to_frame(list(cate.get("calibration") or []), CATE_CALIBRATION_COLUMNS),
                     CATE_CALIBRATION_COLUMNS, ["n"], structural=["decile"], header_note=CATE_CAVEAT)
    _write_tier1_csv(cfg.secondary_dir / "cate_modifier_importance.csv",
                     _rows_to_frame(list(cate.get("importance") or []), CATE_IMPORTANCE_COLUMNS),
                     CATE_IMPORTANCE_COLUMNS, [], header_note=CAUSAL_FOREST_NOTE)


def stage_tte(cfg: SecondaryConfig) -> dict[str, Any]:
    """Doubly-robust IPCW-AIPW sleeve-vs-RYGB TTE at origin 0 over all available frozen rows.

    Cross-fits ps, pc, and mu (5 patient-clustered folds) per outcome/horizon, estimates the
    marginal ATE with an influence-function CI, and derives balance, positivity, E-value,
    c-for-benefit, and the %TWL RCT benchmark. Honors the assemble arm gate and --skip-tte.
    """
    assemble = _load_checkpoint(cfg, "assemble")
    gate_ok = bool(assemble.get("tte_gate_ok", False))
    audit = assemble.get("audit")
    if cfg.skip_tte or not gate_ok:
        reason = assemble.get("tte_skip_reason") or (
            "forced by --skip-tte" if cfg.skip_tte else "surgical arm gate not satisfied"
        )
        return _tte_skipped(cfg, reason, gate_ok)

    outcome_model = _resolve_outcome_model_label(cfg)
    covariates = _read_frame(cfg.secondary_dir / "covariate_frame.parquet")
    store = reopen_prediction_store(Path(assemble["store_root"]))
    available = set(store.keys())
    surgical = covariates.loc[covariates["cohort"].astype(str) == "surgery"].copy()

    cells: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    cate: dict[str, Any] = {}  # filled by the one cell that carries the individualized estimand
    # What the enriched, audit-gated L actually resolved to; the encoder is the ground truth. Both
    # outcomes resolve the SAME column list (they differ only in rows and in what baseline_value
    # aliases), so the per-outcome write below restates it rather than racing. The refused
    # confounders are deliberately NOT duplicated here: _tte_absent_confounders derives them from
    # the assemble audit, so they are named even on the runs where the TTE never executes.
    confounders: dict[str, Any] = {"numeric": [], "categorical": []}
    ps_overlap: dict[str, dict[str, list[float]]] = {}
    love: dict[str, dict[str, Any]] = {}
    populations: dict[str, dict[str, Any]] = {}

    for outcome in OUTCOMES:
        key = ("surgery", outcome, 0)
        if key not in available:
            continue
        pop = _tte_population(surgical, outcome)
        if len(pop) < MIN_CELL_SIZE:
            continue
        L, A, names, encoder = build_wide_L_A(pop, outcome, audit)
        confounders.update(numeric=list(encoder.numeric), categorical=list(encoder.categorical))
        patient_ids = pop["patient_id"].astype(str).to_numpy()
        ps, degenerate = _crossfit_propensity(L, A, patient_ids, cfg.seed, cfg.nuisance_folds)

        weights, _keep = tte.stabilized_iptw(A, ps, trim=IPTW_TRIM)
        smd_unweighted = tte.standardized_mean_diff(L, A)
        smd_weighted = tte.standardized_mean_diff(L, A, w=weights)
        max_abs_smd_weighted = (
            float(np.nanmax(np.abs(smd_weighted))) if np.isfinite(smd_weighted).any() else float("nan")
        )
        love_names: list[str] = []
        love_unweighted: list[float] = []
        love_weighted: list[float] = []
        for name, unweighted, weighted in zip(names, smd_unweighted, smd_weighted):
            if name.endswith("__missing") or name.endswith("__unknown") or not np.isfinite(unweighted):
                continue
            weighted_value = float(weighted) if np.isfinite(weighted) else float("nan")
            balance_rows.append({
                "outcome": outcome, "covariate": name,
                "smd_unweighted": float(unweighted), "smd_weighted": weighted_value,
            })
            love_names.append(name)
            love_unweighted.append(float(unweighted))
            love_weighted.append(weighted_value)
        love[outcome] = {
            "names": love_names, "unweighted": love_unweighted, "weighted": love_weighted,
            "max_abs_smd_weighted": max_abs_smd_weighted,
        }
        ps_overlap[outcome] = {
            "rygb": [float(value) for value in ps[A == 1] if np.isfinite(value)],
            "sleeve": [float(value) for value in ps[A == 0] if np.isfinite(value)],
        }
        mean_baseline_bmi = float(pd.to_numeric(pop.get("baseline_bmi"), errors="coerce").mean())
        populations[outcome] = {
            "n": int(len(pop)), "n_rygb": int((A == 1).sum()), "n_sleeve": int((A == 0).sum()),
            "mean_baseline_bmi": mean_baseline_bmi,
        }

        pop_join = pd.DataFrame({
            "patient_id": patient_ids,
            "_row": np.arange(len(pop)),
            "_A": A,
            "_ps": ps,
        })
        store_rows = store.read(key, columns=["patient_id", "target_month", "target_value", "target_observed"])
        store_rows = store_rows.copy()
        store_rows["patient_id"] = store_rows["patient_id"].astype(str)
        for target_month, group in store_rows.groupby("target_month", sort=True):
            group = group.drop_duplicates("patient_id")
            merged = pop_join.merge(
                group[["patient_id", "target_value", "target_observed"]], on="patient_id", how="inner"
            )
            n = int(len(merged))
            if n == 0:
                continue
            rows_idx = merged["_row"].to_numpy()
            L_cell = L[rows_idx]
            A_cell = merged["_A"].to_numpy().astype(int)
            ps_cell = merged["_ps"].to_numpy(dtype=float)
            Y = pd.to_numeric(merged["target_value"], errors="coerce").to_numpy(dtype=float)
            delta = merged["target_observed"].fillna(False).astype(bool).to_numpy().astype(int)
            cell_ids = merged["patient_id"].astype(str).to_numpy()

            gate = positivity_gate(ps_cell, A_cell, degenerate=degenerate)
            mu1, mu0, pc = _crossfit_mu_pc(L_cell, A_cell, Y, delta, cell_ids, cfg.seed, cfg.nuisance_folds)
            estimate = tte.aipw(Y, A_cell, delta, ps_cell, pc, mu1, mu0)
            ate, se = estimate["ate"], estimate["se"]
            ci_low, ci_high = estimate["ci"]
            benefit = _c_for_benefit(mu1 - mu0, A_cell, Y, ps_cell)
            if outcome == CATE_OUTCOME and int(target_month) == CATE_TARGET_MONTH:
                # Spec 8.1: the CATE runs HERE so it consumes this cell's already-cross-fitted
                # nuisances (and their folds) rather than refitting any of them.
                cate = _cate_dr_learner(cfg, pop.iloc[rows_idx], audit, Y, A_cell, delta, ps_cell,
                                        pc, mu1, mu0, cell_ids, gate)

            observed = Y[delta == 1]
            outcome_sd = float(np.nanstd(observed, ddof=1)) if observed.size > 1 else float("nan")
            e_point = float("nan")
            e_ci = float("nan")
            if np.isfinite(ate) and np.isfinite(outcome_sd) and outcome_sd > 1e-9:
                rr = tte.smd_to_rr(ate / outcome_sd)
                lo_rr = tte.smd_to_rr(ci_low / outcome_sd) if np.isfinite(ci_low) else None
                hi_rr = tte.smd_to_rr(ci_high / outcome_sd) if np.isfinite(ci_high) else None
                evalue = tte.e_value(rr, lo_rr, hi_rr)
                if evalue["e_point"] is not None:
                    e_point = float(evalue["e_point"])
                if evalue["e_bound"] is not None:
                    e_ci = float(evalue["e_bound"])

            rct_anchor = ""
            overlaps: bool | None = None
            if outcome == "bmi" and int(target_month) in (12, 24) and np.isfinite(ate) and mean_baseline_bmi > 0:
                twl = -100.0 * ate / mean_baseline_bmi
                twl_low = -100.0 * ci_high / mean_baseline_bmi
                twl_high = -100.0 * ci_low / mean_baseline_bmi
                benchmark = tte.benchmark_vs_rct(
                    twl, (min(twl_low, twl_high), max(twl_low, twl_high)), "twl_pct_1_2y"
                )
                rct_anchor = "twl_pct_1_2y"
                overlaps = bool(benchmark["overlaps_rct_ci"])

            powered = bool(n >= POWERED_MIN_N and gate["ess_iptw"] >= POWERED_MIN_ESS)
            cells.append({
                "outcome": outcome, "target_month": int(target_month), "n": n,
                "n_rygb": int((A_cell == 1).sum()), "n_sleeve": int((A_cell == 0).sum()),
                "ess_iptw": gate["ess_iptw"], "ate": float(ate), "se": float(se),
                "ci_low": float(ci_low), "ci_high": float(ci_high),
                "e_value_point": e_point, "e_value_ci": e_ci,
                "rct_anchor": rct_anchor, "overlaps_rct_ci": overlaps,
                "c_for_benefit": float(benefit["c_for_benefit"]), "n_pairs": int(benefit["n_pairs"]),
                "max_abs_smd_weighted": max_abs_smd_weighted,
                "positivity_fail": bool(gate["positivity_fail"]), "min_ps": gate["min_ps"],
                "degenerate": bool(gate["degenerate"]), "powered": powered,
            })
        del store_rows
        gc.collect()

    cells_frame = pd.DataFrame(cells)
    balance_frame = pd.DataFrame(balance_rows, columns=list(TTE_BALANCE_COLUMNS))
    _write_tte_csvs(cfg, cells_frame, balance_frame)
    _write_cate_csvs(cfg, cate)

    payload = {
        "status": "done" if cells else "no_cells",
        "tte_gate_ok": True,
        "outcome_model": outcome_model,
        "cells": cells,
        "cate": cate,
        "balance": balance_rows,
        "confounders": confounders,
        "ps_overlap": ps_overlap,
        "love": love,
        "populations": populations,
        "trim": IPTW_TRIM,
        "arm_counts": assemble.get("arm_counts", {}),
        "estimand_note": ESTIMAND_NOTE,
    }
    _save_checkpoint(cfg, "tte", payload)
    return payload


# --------------------------------------------------------------------------------------------
# Placeholder analysis stages (filled in by later milestones). Each writes a checkpoint so the
# render stage can build every page, and resume can skip completed stages.
# --------------------------------------------------------------------------------------------
def stage_tier1(cfg: SecondaryConfig) -> dict[str, Any]:
    """Tier 1: transportability/equity (T1.1), attrition/MNAR (T1.2), and drug/procedure
    heterogeneity + SOPHIA context (T1.3). Streams the frozen selected predictions one origin-0
    partition at a time, writes six disclosure-controlled CSVs, and checkpoints everything pages
    05-08 need. Every axis degrades gracefully (recorded skipped_reason) when its column is not
    usable; nothing here crashes the book."""
    assemble = _load_checkpoint(cfg, "assemble")
    audit = assemble.get("audit")
    selected = assemble.get("selected")
    covariates = _read_frame(cfg.secondary_dir / "covariate_frame.parquet")
    store = reopen_prediction_store(Path(assemble["store_root"]))

    frames = _tier1_heldout_frames(cfg, store, selected, covariates)

    transport_rows, transport_meta = _tier1_transportability(cfg, frames, audit)
    tipping_rows, triage_rows, smd_rows, tipping_curves = _tier1_attrition(cfg, frames)
    drug_rows = _tier1_drug_heterogeneity(cfg, frames)
    which_incretin = _within_incretin_dr_read(cfg, frames, audit)
    procedure_rows = _tier1_procedure_heterogeneity(cfg, frames)

    # BH-FDR the coverage-calibration test within each horizon/subgroup family, report q-values.
    transport_frame = _apply_fdr(
        _rows_to_frame(transport_rows, TRANSPORTABILITY_COLUMNS),
        ["arm", "outcome", "axis", "target_month"], "p_value_cov80", "q_value_cov80")
    drug_frame = _apply_fdr(
        _rows_to_frame(drug_rows, HETEROGENEITY_DRUG_COLUMNS),
        ["outcome", "target_month"], "p_value_cov80", "q_value_cov80")
    procedure_frame = _apply_fdr(
        _rows_to_frame(procedure_rows, HETEROGENEITY_PROCEDURE_COLUMNS),
        ["outcome", "target_month"], "p_value_cov80", "q_value_cov80")
    tipping_frame = _rows_to_frame(tipping_rows, TIPPING_POINT_COLUMNS)
    triage_frame = _rows_to_frame(triage_rows, TRIAGE_COLUMNS)
    smd_frame = _rows_to_frame(smd_rows, ATTRITION_SMD_COLUMNS)

    transport_note = (
        "Held-out (temporal/geographic) re-validation of the frozen selected model stratified by "
        "geography/equity axes; state/SVI/RUCA substitute for the absent center id. "
        + ARM_RECUT_NOTE
    )
    tipping_note = (
        "MNAR delta tipping-point per arm. delta_star_crps is where THAT arm's model stops beating "
        "the population-mean baseline. ate_at_0 / delta_star_ate are the surgical arm against the "
        "OTHER surgical arm - a cross-arm contrast, so it is read off the shared surgical frame "
        "(both procedures present) and the sleeve row mirrors the RYGB row. " + ARM_RECUT_NOTE
    )
    _write_tier1_csv(cfg.secondary_dir / "transportability.csv", transport_frame,
                     TRANSPORTABILITY_COLUMNS, ["n"], structural=["target_month"],
                     header_note=transport_note)
    _write_tier1_csv(cfg.secondary_dir / "attrition_tipping_point.csv", tipping_frame,
                     TIPPING_POINT_COLUMNS, ATTRITION_COUNT_COLUMNS, structural=["target_month"],
                     header_note=tipping_note)
    _write_tier1_csv(cfg.secondary_dir / "attrition_estimator_triage.csv", triage_frame,
                     TRIAGE_COLUMNS, ATTRITION_COUNT_COLUMNS, structural=["target_month"],
                     header_note=ARM_RECUT_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "attrition_observed_vs_censored_smd.csv", smd_frame,
                     ATTRITION_SMD_COLUMNS, ATTRITION_COUNT_COLUMNS, structural=["target_month"],
                     header_note=ARM_RECUT_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "heterogeneity_drug.csv", drug_frame,
                     HETEROGENEITY_DRUG_COLUMNS, ["n"], structural=["target_month"])
    _write_tier1_csv(cfg.secondary_dir / "heterogeneity_procedure.csv", procedure_frame,
                     HETEROGENEITY_PROCEDURE_COLUMNS, ["n"], structural=["target_month"])
    _write_tier1_csv(cfg.secondary_dir / "which_incretin_modifier_importance.csv",
                     _rows_to_frame(list(which_incretin.get("importance") or []),
                                    CATE_IMPORTANCE_COLUMNS),
                     CATE_IMPORTANCE_COLUMNS, [], header_note=WITHIN_INCRETIN_CAVEAT)

    payload = {
        "status": "done",
        "transportability": {
            "rows": transport_frame.to_dict(orient="records"),
            "axes_usable": transport_meta["axes_usable"],
            "loso_requested": transport_meta["loso_requested"],
            "loso_note": transport_meta["loso_note"],
        },
        "attrition": {
            "tipping": tipping_frame.to_dict(orient="records"),
            "curves": tipping_curves,
            "triage": triage_frame.to_dict(orient="records"),
            "smd": smd_frame.to_dict(orient="records"),
        },
        "heterogeneity": {
            "drug": drug_frame.to_dict(orient="records"),
            "which_incretin": which_incretin,
            "procedure": procedure_frame.to_dict(orient="records"),
            "sophia_procedure": dict(SOPHIA_PROCEDURE_RMSE_60),
            "sophia_pooled": {int(k): float(v) for k, v in SOPHIA_POOLED_RMSE.items()},
        },
        "caveat": SUBGROUP_CAVEAT,
        # One record for the whole Section-10 re-cut: every tier reads these same frames.
        "arm_gaps": _arm_formation_gaps(frames),
        "cells": {
            "transportability": int(len(transport_frame)),
            "tipping": int(len(tipping_frame)),
            "triage": int(len(triage_frame)),
            "smd": int(len(smd_frame)),
            "drug": int(len(drug_frame)),
            "procedure": int(len(procedure_frame)),
        },
    }
    _save_checkpoint(cfg, "tier1", payload)
    return payload


def stage_tier2(cfg: SecondaryConfig) -> dict[str, Any]:
    """Tier 2 subgroup menu: per-cell CRPS / 80 coverage / calibration slope+intercept / n / ESS
    across sixteen axes (demographics, glycemia, comorbidity burden, concomitant medications,
    clinical obesity class, calendar era), plus a fairness calibration-equity view (coverage and
    calibration-slope gaps vs the pooled all-patients model for the same cohort/outcome/horizon).

    Reuses the M4 held-out prognostic machinery (_tier1_heldout_frames -> selected candidate at
    origin 0 on HELD_OUT_SPLITS joined to the covariate frame; _prognostic_metrics, _cell_weight,
    _effective_sample_size, _is_powered, _coverage_pvalue), BH-FDRs the coverage-vs-nominal test
    family within each (cohort, outcome, axis) horizon set, disclosure-suppresses n < 11, and
    checkpoints everything pages 09-11 need. Any not-populated axis degrades to a skipped_reason
    row plus a visible page note; nothing here crashes the book. These panels audit error and
    calibration; they do not establish biological effect modification or rank patient groups."""
    assemble = _load_checkpoint(cfg, "assemble")
    audit = assemble.get("audit")
    selected = assemble.get("selected")
    covariates = _read_frame(cfg.secondary_dir / "covariate_frame.parquet")
    store = reopen_prediction_store(Path(assemble["store_root"]))

    frames = _tier1_heldout_frames(cfg, store, selected, covariates)
    pooled = _tier2_pooled_metrics(frames)
    rows, axes_usable = _tier2_subgroup_cells(cfg, frames, audit, pooled)

    frame = _rows_to_frame(rows, SUBGROUPS_TIER2_COLUMNS)
    # BH-FDR the coverage-vs-nominal test family within each (arm, outcome, axis) horizon set.
    frame = _apply_fdr(frame, ["arm", "outcome", "axis", "target_month"],
                       "p_value_cov80", "q_value_cov80")
    _write_tier1_csv(cfg.secondary_dir / "subgroups_tier2.csv", frame, SUBGROUPS_TIER2_COLUMNS,
                     ["n"], structural=["target_month"],
                     header_note=SUBGROUP_CAVEAT + " " + ARM_RECUT_NOTE)

    scored = frame.loc[frame["skipped_reason"].astype(str) == ""]
    payload = {
        "status": "done",
        "rows": frame.to_dict(orient="records"),
        "axes_usable": axes_usable,
        "fairness_axes": list(FAIRNESS_AXES),
        "caveat": SUBGROUP_CAVEAT,
        "cells": {"scored": int(len(scored)), "total": int(len(frame))},
    }
    _save_checkpoint(cfg, "tier2", payload)
    return payload


def stage_tier3(cfg: SecondaryConfig) -> dict[str, Any]:
    """Tier 3 robustness beyond what production did: (1) a powered-only reanalysis of the pooled
    CRPS-improvement gate, (2) an incretin eligibility-threshold sweep (6/12/18 completed months -
    re-derived in-process from the synthetic bundle in --smoke, and in a per-threshold Cosmos
    re-acquire subprocess in --from-run/--full), (3) IPCW model-form sensitivity (logistic vs
    gradient-boosted censoring), (4) baseline-window sensitivity keyed off baseline_bmi_day, and
    (5) a state-cluster bootstrap of the headline CIs. Streams the frozen selected predictions one
    origin-0 partition at a time, writes five disclosure-controlled robustness_*.csv, and
    checkpoints everything page 12 needs plus an omissions list. Every sub-analysis degrades to a
    logged skipped_reason when its input is not populated; nothing here crashes the book."""
    assemble = _load_checkpoint(cfg, "assemble")
    audit = assemble.get("audit")
    selected = assemble.get("selected")
    covariates = _read_frame(cfg.secondary_dir / "covariate_frame.parquet")
    store = reopen_prediction_store(Path(assemble["store_root"]))

    frames = _tier1_heldout_frames(cfg, store, selected, covariates)
    rng = np.random.default_rng(cfg.seed)
    omissions: list[str] = []

    powered_rows = _tier3_powered_only(frames)
    eligibility_rows, eligibility_meta = _tier3_eligibility_sweep(cfg, frames)
    omissions.extend(eligibility_meta.get("omissions", []))
    ipcw_rows = _tier3_ipcw_form(cfg, frames)
    baseline_rows, baseline_meta = _tier3_baseline_window(cfg, frames, covariates)
    if baseline_meta.get("skipped_reason"):
        omissions.append(str(baseline_meta["skipped_reason"]))
    cluster_rows, cluster_meta = _tier3_cluster_bootstrap(cfg, frames, audit, rng)
    if cluster_meta.get("skipped_reason"):
        omissions.append(str(cluster_meta["skipped_reason"]))

    powered_frame = _rows_to_frame(powered_rows, ROBUSTNESS_POWERED_ONLY_COLUMNS)
    eligibility_frame = _rows_to_frame(eligibility_rows, ROBUSTNESS_ELIGIBILITY_COLUMNS)
    ipcw_frame = _rows_to_frame(ipcw_rows, ROBUSTNESS_IPCW_FORM_COLUMNS)
    baseline_frame = _rows_to_frame(baseline_rows, ROBUSTNESS_BASELINE_WINDOW_COLUMNS)
    cluster_frame = _rows_to_frame(cluster_rows, ROBUSTNESS_CLUSTER_BOOTSTRAP_COLUMNS)

    _write_tier1_csv(cfg.secondary_dir / "robustness_powered_only.csv", powered_frame,
                     ROBUSTNESS_POWERED_ONLY_COLUMNS, ["n"], header_note=POWERED_ONLY_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "robustness_eligibility_sweep.csv", eligibility_frame,
                     ROBUSTNESS_ELIGIBILITY_COLUMNS, ["n"],
                     structural=["target_month", "qualifying_months", "cohort_n"],
                     header_note=ELIGIBILITY_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "robustness_ipcw_form.csv", ipcw_frame,
                     ROBUSTNESS_IPCW_FORM_COLUMNS, ["n_observed"], structural=["target_month"],
                     header_note=IPCW_FORM_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "robustness_baseline_window.csv", baseline_frame,
                     ROBUSTNESS_BASELINE_WINDOW_COLUMNS, ["n"], structural=["target_month"],
                     header_note=BASELINE_WINDOW_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "robustness_cluster_bootstrap.csv", cluster_frame,
                     ROBUSTNESS_CLUSTER_BOOTSTRAP_COLUMNS, ["n"], structural=["target_month"],
                     header_note=CLUSTER_BOOTSTRAP_NOTE)

    payload = {
        "status": "done",
        "powered_only": powered_frame.to_dict(orient="records"),
        "eligibility": {
            "rows": eligibility_frame.to_dict(orient="records"),
            "months": eligibility_meta.get("months", []),
            "cohort_n": eligibility_meta.get("cohort_n", {}),
            "coverage_note": eligibility_meta.get("coverage_note", ""),
        },
        "ipcw_form": ipcw_frame.to_dict(orient="records"),
        "baseline_window": {
            "rows": baseline_frame.to_dict(orient="records"),
            "usable": bool(baseline_meta.get("usable", False)),
            "skipped_reason": baseline_meta.get("skipped_reason", ""),
        },
        "cluster_bootstrap": {
            "rows": cluster_frame.to_dict(orient="records"),
            "usable": bool(cluster_meta.get("usable", False)),
            "skipped_reason": cluster_meta.get("skipped_reason", ""),
        },
        "omissions": omissions,
        "cells": {
            "powered_only": int(len(powered_frame)),
            "eligibility": int(len(eligibility_frame)),
            "ipcw_form": int(len(ipcw_frame)),
            "baseline_window": int(len(baseline_frame)),
            "cluster_bootstrap": int(len(cluster_frame)),
        },
    }
    _save_checkpoint(cfg, "tier3", payload)
    return payload


def stage_tier4(cfg: SecondaryConfig) -> dict[str, Any]:
    """Tier 4 clinical value reframings (spec Section 6 Tier 4): (1) clinical threshold
    probabilities from the predictive quantile ladder with IPCW reliability / Brier / AUROC,
    (2) decision-curve net benefit for the binary threshold events, (3) a who-is-predictable map
    ranking every disclosable subgroup stratum by 80% interval width and CRPS, and (4) a strictly
    prognostic overlap-weighted GLP-1-vs-surgery predicted-trajectory contrast (NOT a treatment
    effect). Streams the frozen selected predictions one origin-0 partition at a time, writes four
    disclosure-controlled CSVs, and checkpoints everything pages 13-15 need. Every sub-analysis
    degrades gracefully to a documented skipped_reason; nothing here crashes the book."""
    assemble = _load_checkpoint(cfg, "assemble")
    audit = assemble.get("audit")
    selected = assemble.get("selected")
    covariates = _read_frame(cfg.secondary_dir / "covariate_frame.parquet")
    store = reopen_prediction_store(Path(assemble["store_root"]))

    frames = _tier1_heldout_frames(cfg, store, selected, covariates)

    threshold_rows, reliability, auroc_computable = _tier4_threshold_probabilities(cfg, frames, audit)
    decision_rows = _tier4_decision_curves(cfg, frames, audit)
    predictability_rows = _tier4_predictability_map(cfg, frames, audit)
    glp1_rows, glp1_trajectory = _tier4_glp1_overlap(cfg, frames)
    procedure_auroc_rows, procedure_auroc_curves = _procedure_auroc(cfg, frames)
    per_arm_rows = _per_arm_prognostic(cfg, frames)
    sensitivity_rows, sensitivity_sweeps = _feature_sensitivity(cfg, frames, audit)

    threshold_frame = _rows_to_frame(threshold_rows, THRESHOLD_PROB_COLUMNS)
    decision_frame = _rows_to_frame(decision_rows, DECISION_CURVE_COLUMNS)
    predictability_frame = _rows_to_frame(predictability_rows, PREDICTABILITY_MAP_COLUMNS)
    glp1_frame = _rows_to_frame(glp1_rows, GLP1_OVERLAP_COLUMNS)
    procedure_auroc_frame = _rows_to_frame(procedure_auroc_rows, PROCEDURE_AUROC_COLUMNS)
    per_arm_frame = _rows_to_frame(per_arm_rows, PER_ARM_PROGNOSTIC_COLUMNS)
    sensitivity_frame = _rows_to_frame(sensitivity_rows, FEATURE_SENSITIVITY_COLUMNS)

    _write_tier1_csv(cfg.secondary_dir / "threshold_probabilities.csv", threshold_frame,
                     THRESHOLD_PROB_COLUMNS, ["n"], structural=["target_month"],
                     header_note=THRESHOLD_PROB_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "decision_curves.csv", decision_frame,
                     DECISION_CURVE_COLUMNS, ["n"], structural=["target_month", "p_t"],
                     header_note=DECISION_CURVE_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "predictability_map.csv", predictability_frame,
                     PREDICTABILITY_MAP_COLUMNS, ["n"],
                     structural=["target_month", "rank_width", "rank_crps"],
                     header_note=PREDICTABILITY_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "glp1_vs_surgery_overlap.csv", glp1_frame,
                     GLP1_OVERLAP_COLUMNS, ["n_surgery", "n_incretin"], structural=["target_month"],
                     header_note=GLP1_CAVEAT)
    _write_tier1_csv(cfg.secondary_dir / "per_procedure_auroc.csv", procedure_auroc_frame,
                     PROCEDURE_AUROC_COLUMNS, ["n"], structural=["target_month"],
                     header_note=PROCEDURE_AUROC_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "per_arm_calibration_accuracy.csv", per_arm_frame,
                     PER_ARM_PROGNOSTIC_COLUMNS, ["n"], structural=["target_month"],
                     header_note=PER_ARM_PROGNOSTIC_NOTE)
    _write_tier1_csv(cfg.secondary_dir / "forecast_feature_sensitivity.csv", sensitivity_frame,
                     FEATURE_SENSITIVITY_COLUMNS, ["n"], structural=["rank"],
                     header_note=FEATURE_SENSITIVITY_NOTE)

    payload = {
        "status": "done",
        "threshold": {
            "rows": threshold_frame.to_dict(orient="records"),
            "reliability": reliability,
            "auroc_computable": auroc_computable,
        },
        "decision": {
            "rows": decision_frame.to_dict(orient="records"),
            "grid": [float(value) for value in DECISION_PT_GRID],
        },
        "predictability": {
            "rows": predictability_frame.to_dict(orient="records"),
            "n_cells": int(len(predictability_frame)),
        },
        "glp1": {
            "rows": glp1_frame.to_dict(orient="records"),
            "trajectory": glp1_trajectory,
            "caveat": GLP1_CAVEAT,
        },
        "procedure_auroc": {
            "rows": procedure_auroc_frame.to_dict(orient="records"),
            "curves": procedure_auroc_curves,
            "fpr_grid": [float(value) for value in ROC_FPR_GRID],
        },
        "per_arm_prognostic": {
            "rows": per_arm_frame.to_dict(orient="records"),
            "note": PER_ARM_PROGNOSTIC_NOTE,
        },
        "feature_sensitivity": {
            "rows": sensitivity_frame.to_dict(orient="records"),
            "sweeps": sensitivity_sweeps,
            "target_month": FEATURE_SENSITIVITY_MONTH,
            "note": FEATURE_SENSITIVITY_NOTE,
        },
        "cells": {
            "threshold": int(len(threshold_frame)),
            "decision": int(len(decision_frame)),
            "predictability": int(len(predictability_frame)),
            "glp1": int(len(glp1_frame)),
            "procedure_auroc": int(len(procedure_auroc_frame)),
            "per_arm_prognostic": int(len(per_arm_frame)),
            "feature_sensitivity": int(len(sensitivity_frame)),
        },
    }
    _save_checkpoint(cfg, "tier4", payload)
    return payload


# --------------------------------------------------------------------------------------------
# Checkpoint I/O and frame spill helpers
# --------------------------------------------------------------------------------------------
def _peak_rss_gb() -> float | None:
    """This process's peak RSS in GB, or None where the platform cannot report it."""
    peak = study.process_peak_rss_bytes()
    return None if peak is None else round(peak / (1024 ** 3), 3)


def _save_checkpoint(cfg: SecondaryConfig, stage: str, payload: Any) -> None:
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    study.atomic_pickle(cfg.checkpoints_dir / f"{stage}.pkl", payload)
    study.atomic_json(
        cfg.checkpoints_dir / f"{stage}.json",
        # The peak is taken HERE, inside the stage's own process, so it is the stage's real
        # high-water mark under --orchestrate too; the driver loop's own peak (which is what it
        # would measure after a subprocess returns) says nothing about what the stage needed.
        {"stage": stage, "time_utc": study.utc_now(), "config_hash": config_hash(cfg),
         "peak_rss_gb": _peak_rss_gb()},
    )


def _load_checkpoint(cfg: SecondaryConfig, stage: str) -> Any:
    path = cfg.checkpoints_dir / f"{stage}.pkl"
    with open(path, "rb") as stream:
        return pickle.load(stream)


def _checkpoint_valid(cfg: SecondaryConfig, stage: str) -> bool:
    """A stage may be skipped on resume if its pkl+json exist and the config hash matches."""
    meta_path = cfg.checkpoints_dir / f"{stage}.json"
    body_path = cfg.checkpoints_dir / f"{stage}.pkl"
    if not (meta_path.exists() and body_path.exists()):
        return False
    meta = study.read_json(meta_path, {})
    return meta.get("config_hash") == config_hash(cfg)


def _write_frame(path: Path, frame: Any) -> None:
    """Spill a frame to parquet when pyarrow is present, else pickle - both stream on read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if study._PARQUET_SPILL_ENABLED:
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, engine="pyarrow", index=False)
        study.replace_file(temporary, path)
    else:
        study.atomic_pickle(path.with_suffix(".pkl"), frame)


def _read_frame(path: Path) -> Any:
    if path.exists() and study._PARQUET_SPILL_ENABLED:
        return pd.read_parquet(path)
    pickle_path = path.with_suffix(".pkl")
    if pickle_path.exists():
        with open(pickle_path, "rb") as stream:
            return pickle.load(stream)
    raise FileNotFoundError(f"No spilled frame at {path} or {pickle_path}")


def _write_csv(path: Path, frame: Any, header_note: str | None = None) -> None:
    """Deterministic CSV write via the atomic text writer, optional leading comment header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = frame.to_csv(index=False)
    if header_note:
        body = "".join(f"# {line}\n" for line in header_note.splitlines()) + body
    study.atomic_text(path, body)


# --------------------------------------------------------------------------------------------
# Figure book: page primitives and renderers
# --------------------------------------------------------------------------------------------
def secondary_new_page(number: int, title: str, subtitle: str) -> Any:
    """A production-styled landscape page with a secondary-run disclosure footer.

    Mirrors ``study.new_page`` (same geometry, palette, header) but the footer states the secondary
    run's mixed causal/prognostic register instead of the production book's hardcoded
    "Noncausal prognostic study" line, which would misdescribe the TTE pages.
    """
    figure = plt.figure(figsize=(11, 8.5), constrained_layout=False)
    figure.patch.set_facecolor(study.PALETTE["paper"])
    figure.text(0.055, 0.947, f"{number:02d}", fontsize=22, fontweight="bold",
                color=study.PALETTE["blue"], va="top")
    figure.text(0.115, 0.947, title, fontsize=17, fontweight="bold", color=study.PALETTE["ink"], va="top")
    figure.text(0.115, 0.915, subtitle, fontsize=9.5, color=study.PALETTE["muted"], va="top")
    figure.lines.append(plt.Line2D([0.055, 0.945], [0.893, 0.893], transform=figure.transFigure,
                                    color=study.PALETTE["grid"], lw=1.0))
    figure.text(0.055, 0.025,
                "Aggregate, disclosure-controlled output | Cells n < 11 suppressed | "
                "Secondary analyses: causal claims only under stated assumptions",
                fontsize=7.5, color=study.PALETTE["muted"])
    return figure


def stamp_provenance(figure: Any, data: Mapping[str, Any]) -> None:
    """A small top-right provenance badge (mode + config hash), analogous to add_run_provenance."""
    mode = data.get("mode", "?")
    short_hash = str(data.get("config_hash", ""))[:10]
    badge = f"{SECONDARY_VERSION} | mode {mode} | {short_hash}"
    color = study.PALETTE["red"] if mode == "smoke" else study.PALETTE["muted"]
    figure.text(0.945, 0.965, badge, fontsize=7.0, color=color, ha="right", va="top")


def empty_panel(axis: Any, message: str = "Not estimable") -> None:
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", color=study.PALETTE["muted"], fontsize=10)


PAGE_TITLES = {
    0: ("Executive summary", "Secondary analyses of the metabolic trajectory study"),
    1: ("Run identity and provenance", "Fingerprints, dependencies, source run, column-population audit"),
    2: ("TTE design and cohort", "Sleeve vs RYGB point-intervention target-trial emulation"),
    3: ("TTE propensity and positivity", "PS overlap, IPTW effective sample size, trimming, positivity"),
    4: ("TTE primary results", "AIPW ATE by horizon, E-value, RCT overlap, c-for-benefit"),
    5: ("Transportability and equity", "State / SVI / RUCA re-validation of held-out performance"),
    6: ("Attrition and MNAR sensitivity", "Delta tipping-point, CC vs IPCW vs MI, observed-vs-censored"),
    7: ("Incretin drug heterogeneity", "Tirzepatide vs semaglutide vs older agents"),
    8: ("Procedure heterogeneity vs SOPHIA", "RYGB vs sleeve RMSE against SOPHIA context lines"),
    9: ("Fairness subgroups", "Sex / race / ethnicity / age: calibration, CRPS, coverage"),
    10: ("Clinical subgroups", "Diabetes / glycemic / comorbidity / concomitant medications"),
    11: ("Obesity class and calendar era", "Baseline BMI class and index-year strata"),
    12: ("Robustness panel", "Powered-only, eligibility sweep, IPCW form, baseline window, cluster bootstrap"),
    13: ("Clinical threshold probabilities", "Reliability diagrams from the predictive quantiles"),
    14: ("Decision curves", "Net benefit across threshold probabilities"),
    15: ("Predictability map and GLP-1 vs surgery", "Who is predictable; overlap-weighted contrast"),
    16: ("Gates, limitations, and conclusion", "Supported vs exploratory; residual-confounding caveat"),
    17: ("Per-procedure discrimination for P(BMI < 35)",
         "IPCW-weighted ROC at 6, 12, and 24 months: incretin, RYGB, sleeve"),
    18: ("Heterogeneous benefit of RYGB vs sleeve",
         "DR-learner CATE: RATE / TOC, benefit calibration, top effect modifiers"),
    19: ("Forecast sensitivity to baseline features",
         "Surrogate permutation importance and counterfactual sweep, per arm"),
    20: ("Per-arm calibration and accuracy",
         "CRPS / RMSE / MAE / bias / coverage / calibration slope by horizon: incretin, RYGB, sleeve"),
}
PENDING_NOTE = "Content pending - this analysis is populated by a later build milestone."


def _placeholder_page(number: int, title: str, subtitle: str, note: str = PENDING_NOTE) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    axis = figure.add_axes([0.08, 0.12, 0.84, 0.72])
    empty_panel(axis, note)
    return figure


# Page builders are registered here as milestones fill them in; missing pages fall back to a
# titled placeholder so the figure-book contract always holds. Each builder has the signature
# ``builder(cfg, data, number, title, subtitle) -> figure``.
PAGE_BUILDERS: dict[int, Callable[..., Any]] = {}


def register_page(number: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        PAGE_BUILDERS[number] = function
        return function
    return decorator


# --------------------------------------------------------------------------------------------
# Bookend pages (00 executive summary, 01 run identity + provenance, 16 gates + limitations).
# Unlike the analysis pages, these run no new computation: they AGGREGATE the SAME per-stage
# checkpoints (data["assemble"|"tte"|"tier1".."tier4"]) into a synthesized, honest, high-level
# overview. Every accessor is defensive so a partial run still renders a visible note rather than
# raising - a single bookend error degrades to a placeholder via _safe_build, never crashing the
# book. The synthesis helpers below also back the finalized manifest gate summary + omissions.
# --------------------------------------------------------------------------------------------
def _short_hash(value: Any, keep: int = 16) -> str:
    """A short representation of a hex fingerprint (or path tail) for the provenance tables."""
    text = str(value if value is not None else "")
    return text if len(text) <= keep else text[:keep] + "..."


def _json_float(value: Any) -> Any:
    """A JSON-safe float: non-finite (NaN/inf) and non-numeric values collapse to None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_floats(values: Any) -> list[float]:
    """The finite floats in an iterable, silently dropping None / non-numeric / NaN entries."""
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return out


def _tte_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    """Headline TTE facts from the tte checkpoint: cell counts, positivity outcomes, the
    adequately-powered count, and the reported E-value bound across estimable horizons."""
    tte_data = data.get("tte") or {}
    status = str(tte_data.get("status", "absent"))
    cells = list(tte_data.get("cells", []) or [])
    if status == "skipped" or not cells:
        return {
            "status": status, "skip_reason": str(tte_data.get("skip_reason", "")),
            "n_cells": 0, "n_estimable": 0, "n_positivity_fail": 0, "n_powered": 0,
            "e_value_low": float("nan"), "e_value_high": float("nan"),
            "outcome_model": str(tte_data.get("outcome_model", "none")),
        }
    fails = sum(1 for cell in cells if bool(cell.get("positivity_fail")))
    estimable = [cell for cell in cells if not bool(cell.get("positivity_fail"))]
    powered = sum(1 for cell in estimable if bool(cell.get("powered")))
    evalues = _finite_floats(cell.get("e_value_point") for cell in estimable)
    return {
        "status": status, "skip_reason": "", "n_cells": len(cells), "n_estimable": len(estimable),
        "n_positivity_fail": fails, "n_powered": powered,
        "e_value_low": float(min(evalues)) if evalues else float("nan"),
        "e_value_high": float(max(evalues)) if evalues else float("nan"),
        "outcome_model": str(tte_data.get("outcome_model", "none")),
    }


def _powered_counts(rows: Any) -> tuple[int, int]:
    """(#scored cells, #adequately powered) for a metric-row list; skipped_reason rows excluded."""
    scored = [row for row in (rows or []) if not str(row.get("skipped_reason", "")).strip()]
    return len(scored), sum(1 for row in scored if bool(row.get("powered")))


def _verdict_summary(verdicts: Sequence[str]) -> str:
    """A compact tally of the Tier-3 powered-only verdicts (e.g. '2 stable real signal, 1 real ceiling')."""
    counts: dict[str, int] = {}
    for verdict in verdicts:
        key = str(verdict)
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{count} {name.replace('_', ' ')}" for name, count in sorted(counts.items())) or "no cells"


def _analysis_family_summary(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-analysis-family scored-cell + adequately-powered counts and a short gate label,
    aggregated from EVERY tier checkpoint. Backs the page-00 census, the page-16 gate table, and
    the manifest gate summary."""
    families: list[dict[str, Any]] = []
    tte = _tte_summary(data)
    families.append({
        "family": "TTE (RYGB vs sleeve)",
        "cells": 0 if tte["status"] == "skipped" else tte["n_cells"],
        "powered": 0 if tte["status"] == "skipped" else tte["n_powered"],
        "gate": "skipped" if tte["status"] == "skipped" else "positivity gate",
    })
    tier1 = data.get("tier1") or {}
    scored, powered = _powered_counts((tier1.get("transportability") or {}).get("rows", []))
    families.append({"family": "Transportability", "cells": scored, "powered": powered, "gate": "proxy axes"})
    heterogeneity = tier1.get("heterogeneity") or {}
    scored, powered = _powered_counts(list(heterogeneity.get("drug", [])) + list(heterogeneity.get("procedure", [])))
    families.append({"family": "Heterogeneity", "cells": scored, "powered": powered, "gate": "SOPHIA context"})
    scored, powered = _powered_counts((tier1.get("attrition") or {}).get("tipping", []))
    families.append({"family": "Attrition/MNAR", "cells": scored, "powered": powered, "gate": "tipping/triage"})
    tier2 = data.get("tier2") or {}
    scored, powered = _powered_counts(tier2.get("rows", []))
    families.append({"family": "Subgroups", "cells": scored, "powered": powered, "gate": "descriptive"})
    tier3 = data.get("tier3") or {}
    verdicts = [str(row.get("verdict", "")) for row in tier3.get("powered_only", [])]
    # An arm with nothing above the disclosure floor is named in the verdict column but scored
    # nothing, so it must not inflate the census's scored-cell count.
    families.append({
        "family": "Robustness",
        "cells": sum(1 for verdict in verdicts if verdict != "no_disclosable_cells"),
        "powered": sum(1 for verdict in verdicts if verdict in {"stable_real_signal", "power_artifact"}),
        "gate": "powered vs all",
    })
    tier4 = data.get("tier4") or {}
    scored, powered = _powered_counts((tier4.get("threshold") or {}).get("rows", []))
    families.append({"family": "Threshold prob.", "cells": scored, "powered": powered, "gate": "reliability"})
    scored, powered = _powered_counts((tier4.get("decision") or {}).get("rows", []))
    families.append({"family": "Decision curves", "cells": scored, "powered": powered, "gate": "net benefit"})
    scored, powered = _powered_counts((tier4.get("predictability") or {}).get("rows", []))
    families.append({"family": "Predictability", "cells": scored, "powered": powered, "gate": "reliable flag"})
    glp1 = (tier4.get("glp1") or {}).get("rows", [])
    families.append({"family": "GLP-1 vs surgery", "cells": len(glp1), "powered": 0, "gate": "not causal"})
    return families


def _tte_absent_confounders(data: Mapping[str, Any]) -> list[str]:
    """The named residual-confounding target (Section 9): every optional confounder the audit
    refused, then the surgeon/center identifiers this source never carries. Derived from the
    assemble audit so the names are recorded whether or not the TTE ran. The order is fixed (the
    sorted audit-refused names, then the structural one), so the manifest, the omissions ledger,
    and the page text are deterministic."""
    audit = (data.get("assemble") or {}).get("audit")
    gated = _tte_confounder_lists(audit)[2] if audit is not None else []
    return gated + ["surgeon_or_center_id"]


def _collect_omissions(data: Mapping[str, Any]) -> list[str]:
    """The de-duplicated 'no silent caps' ledger: every documented skip/omission across the
    checkpoints (surfaced on page 16 and in the manifest). Surfaces the Tier-3 omissions list and
    every not-usable axis so nothing is silently dropped."""
    omissions: list[str] = []
    tte_data = data.get("tte") or {}
    if str(tte_data.get("status")) == "skipped" and tte_data.get("skip_reason"):
        omissions.append(f"TTE skipped: {tte_data['skip_reason']}")
    tier1 = data.get("tier1") or {}
    omissions.extend(f"Reporting arm not formed, dropped from every per-arm tier: {gap}"
                     for gap in (tier1.get("arm_gaps") or []))
    transportability = tier1.get("transportability") or {}
    if transportability.get("loso_note"):
        omissions.append(str(transportability["loso_note"]))
    for axis, usable in sorted((transportability.get("axes_usable") or {}).items()):
        if not usable:
            omissions.append(f"Transportability axis not populated: {axis}")
    tier2 = data.get("tier2") or {}
    for axis, usable in sorted((tier2.get("axes_usable") or {}).items()):
        if not usable:
            omissions.append(f"Subgroup axis not populated: {axis}")
    tier3 = data.get("tier3") or {}
    omissions.extend(str(item) for item in (tier3.get("omissions") or []))
    baseline_window = tier3.get("baseline_window") or {}
    if not baseline_window.get("usable", True) and baseline_window.get("skipped_reason"):
        omissions.append(str(baseline_window["skipped_reason"]))
    cluster_bootstrap = tier3.get("cluster_bootstrap") or {}
    if not cluster_bootstrap.get("usable", True) and cluster_bootstrap.get("skipped_reason"):
        omissions.append(str(cluster_bootstrap["skipped_reason"]))
    tier4 = data.get("tier4") or {}
    threshold_reasons = sorted({
        str(row.get("skipped_reason", "")).strip()
        for row in (tier4.get("threshold") or {}).get("rows", [])
        if str(row.get("skipped_reason", "")).strip()
    })
    omissions.extend(f"Threshold event disabled: {reason}" for reason in threshold_reasons)
    # Named per (arm, outcome): the reasons differ only in a row count, so an unnamed ledger entry
    # would read as four near-duplicates instead of four distinct cells.
    sensitivity_reasons = sorted({
        f"{row.get('arm')} {row.get('outcome')}, {str(row.get('skipped_reason', '')).strip()}"
        for row in (tier4.get("feature_sensitivity") or {}).get("rows", [])
        if str(row.get("skipped_reason", "")).strip()
    })
    omissions.extend(f"Feature sensitivity not estimable for {reason}" for reason in sensitivity_reasons)
    which_incretin = ((tier1.get("heterogeneity") or {}).get("which_incretin") or {})
    if which_incretin.get("skipped_reason"):
        omissions.append(f"Which-incretin contrast not estimable: {which_incretin['skipped_reason']}")
    cate = _cate_payload(data)
    if cate.get("skipped_reason"):
        omissions.append(f"CATE not estimable: {cate['skipped_reason']}")
    # Structural, every-run entries last so the run-specific skips stay at the head of the ledger.
    omissions.append(CAUSAL_FOREST_NOTE)
    omissions.extend(f"TTE confounder not populated, omitted from L and unadjusted: {name}"
                     for name in _tte_absent_confounders(data))
    unique: list[str] = []
    for item in omissions:
        if item and item not in unique:
            unique.append(item)
    return unique


def _executive_highlights(data: Mapping[str, Any]) -> list[str]:
    """Short clinical-value + subgroup highlight lines synthesized from Tier 1/2/4 checkpoints."""
    tier1 = data.get("tier1") or {}
    usable_axes = sorted(axis for axis, ok in
                         ((tier1.get("transportability") or {}).get("axes_usable") or {}).items() if ok)
    heterogeneity = tier1.get("heterogeneity") or {}
    tier2 = data.get("tier2") or {}
    scored = int((tier2.get("cells") or {}).get("scored", 0))
    usable_subgroup_axes = sum(1 for ok in (tier2.get("axes_usable") or {}).values() if ok)
    tier4 = data.get("tier4") or {}
    n_threshold = _powered_counts((tier4.get("threshold") or {}).get("rows", []))[0]
    decision_rows = (tier4.get("decision") or {}).get("rows", [])
    # Keyed on arm, not cohort: the decision rows report the three reporting arms, so dropping the
    # arm out of this dedup would silently halve the census (rygb folding onto incretin).
    n_decision = len({(row.get("arm"), row.get("outcome"), row.get("target_month"), row.get("event"))
                      for row in decision_rows})
    predictability = (tier4.get("predictability") or {}).get("rows", [])
    n_reliable = sum(1 for row in predictability if bool(row.get("reliable_flag")))
    n_glp1 = len((tier4.get("glp1") or {}).get("rows", []))
    return [
        f"- Transportability: {len(usable_axes)}/3 equity axes usable (proxy for absent center id).",
        f"- Heterogeneity: {len(heterogeneity.get('drug', []))} drug, "
        f"{len(heterogeneity.get('procedure', []))} procedure cells vs SOPHIA context.",
        f"- Subgroups: {scored} scored cells across {usable_subgroup_axes} usable axes; fairness gaps vs pooled.",
        f"- Threshold probabilities: {n_threshold} event-cells (reliability / Brier / AUROC).",
        f"- Decision curves: {n_decision} event-cells scored (net benefit).",
        f"- Predictability: {n_reliable}/{len(predictability)} strata flagged reliable.",
        f"- GLP-1 vs surgery: {n_glp1} overlap contrasts (NOT a treatment effect).",
    ]


def _supported_vs_exploratory_text(data: Mapping[str, Any], tte: Mapping[str, Any]) -> str:
    """The honest supported-vs-exploratory narrative for the executive summary."""
    families = _analysis_family_summary(data)
    total_cells = sum(int(family["cells"]) for family in families)
    total_powered = sum(int(family["powered"]) for family in families)
    if tte["status"] == "skipped":
        causal = ("The target-trial emulation was skipped (" +
                  (tte.get("skip_reason") or "gate not satisfied") + "), so no causal contrast is reported.")
    else:
        low, high = tte.get("e_value_low"), tte.get("e_value_high")
        if np.isfinite(low) and np.isfinite(high):
            e_clause = (f"E-value {low:.2f}" if abs(high - low) < 1e-9
                        else f"E-value {low:.2f} to {high:.2f}")
        else:
            e_clause = "E-value not estimable"
        causal = ("The sleeve-vs-RYGB ATE is causal ONLY under conditional exchangeability given measured L; the "
                  + e_clause + " across estimable horizons and the RCT benchmark bound residual confounding, and it "
                  "is not a substitute for an RCT.")
    return (
        f"Supported: disclosure-controlled prognostic accuracy and calibration audits on the frozen held-out "
        f"predictions (subgroups, transportability re-validation, threshold probabilities, decision curves, "
        f"predictability), with {total_powered} of {total_cells} scored cells adequately powered (n >= "
        f"{POWERED_MIN_N} and IPCW ESS >= {POWERED_MIN_ESS}). " + causal + " Exploratory: any cell below the "
        f"powered threshold, the GLP-1-vs-surgery overlap contrast (a prognostic prediction comparison, not a "
        f"treatment effect), and geographic transportability, which is approximated by state / SVI / RUCA rather "
        f"than true held-out centers."
    )


def _limitations_text(data: Mapping[str, Any], tte: Mapping[str, Any]) -> str:
    """The honest limitations paragraph: the causal-only-under-conditional-exchangeability caveat
    (ESTIMAND_NOTE) with the reported E-value bound and the absent confounders named explicitly, plus
    the transportability-approximation note and the remaining prognostic caveats already in the file."""
    absent = ", ".join(_tte_absent_confounders(data))
    if tte["status"] == "skipped":
        lead = ("The target-trial emulation was skipped in this run (" +
                (tte.get("skip_reason") or "gate not satisfied") + "), so no causal contrast is claimed; the "
                "estimand and its caveat would otherwise be as follows. ")
        e_clause = ""
    else:
        lead = ""
        low, high = tte.get("e_value_low"), tte.get("e_value_high")
        if np.isfinite(low) and np.isfinite(high):
            bound = f"{low:.2f}" if abs(high - low) < 1e-9 else f"{low:.2f} to {high:.2f}"
            e_clause = (f" The reported E-value ({bound} across estimable horizons) is read against the enriched L "
                        f"(creatinine and eGFR now adjusted): an unmeasured confounder - one of the axes ABSENT "
                        f"from this source and therefore omitted from L ({absent}) - would need at least that "
                        "association strength with both the arm and the outcome to explain the estimate away.")
        else:
            e_clause = (f" The E-value bound is not estimable in this run; the axes ABSENT from this source and "
                        f"therefore omitted from the enriched L and unadjusted are {absent}.")
    return (
        lead + ESTIMAND_NOTE + e_clause + " Geographic transportability is APPROXIMATED by state, SVI, and RUCA "
        "strata, not by true held-out centers, so out-of-center generalization is not established. Treatment "
        "censoring and observation may remain informative despite cross-fitted IPCW weighting; recent ingredients "
        "carry short calendar support, so unsupported horizons are not estimable; and the subgroup panels audit "
        "error and calibration, not biological effect modification."
    )


def _permitted_claims_text() -> str:
    """Permitted vs not-permitted claims for the conclusion panel."""
    return (
        "Permitted: prognostic, horizon-specific, disclosure-controlled forecasts and their calibration and "
        "accuracy audits; the TTE effect only under the stated exchangeability assumption with its E-value bound. "
        "Not permitted: exact-time trajectories, treatment effects beyond the assumption-bounded TTE, "
        "generalization to true held-out centers, or any causal GLP-1-vs-surgery claim."
    )


@register_page(0)
def _page_executive_summary(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                            title: str, subtitle: str) -> Any:
    """Synthesized headline summary drawing from ALL tier checkpoints: the TTE ATE headline with
    positivity/overlap and the residual-confounding caveat, the analysis-coverage census, the
    Tier-3 robustness verdict, the Tier 1/2/4 clinical-value + subgroup highlights, and an honest
    supported-vs-exploratory narrative."""
    figure = secondary_new_page(number, title, subtitle)
    mode = str(data.get("mode", cfg.mode))
    headline_color = study.PALETTE["red"] if mode == "smoke" else study.PALETTE["blue"]
    headline = ("Bounded SMOKE run - end-to-end pipeline validation only; results are non-inferential."
                if mode == "smoke" else
                "Doubly-robust sleeve-vs-RYGB TTE plus four analysis tiers on the frozen predictions.")
    figure.text(0.5, 0.862, headline, ha="center", va="center", fontsize=12.5, fontweight="bold",
                color=headline_color)

    tte = _tte_summary(data)
    families = _analysis_family_summary(data)

    ax_tte = figure.add_axes([0.055, 0.505, 0.44, 0.235])
    study.panel_label(ax_tte, "A", "Causal contrast: AIPW ATE (RYGB - sleeve)")
    tte_data = data.get("tte") or {}
    tte_cells = pd.DataFrame(tte_data.get("cells", []))
    if str(tte_data.get("status")) == "skipped" or tte_cells.empty:
        empty_panel(ax_tte, (f"TTE skipped: {tte.get('skip_reason') or 'gate not satisfied'}"
                             if tte["status"] == "skipped" else "TTE not populated in this source"))
    else:
        display = tte_cells.copy()
        fail = display["positivity_fail"].fillna(False).astype(bool)
        for column in ("ate", "ci_low", "ci_high", "e_value_point"):
            display.loc[fail, column] = np.nan
        columns = ["outcome", "target_month", "n", "ate", "ci_low", "ci_high", "e_value_point"]
        study.draw_compact_table(
            ax_tte, display.loc[:, columns], columns,
            labels=["Outcome", "Target\nmonth", "N", "ATE", "CI low", "CI high", "E-value"], max_rows=9)

    ax_rob = figure.add_axes([0.55, 0.505, 0.40, 0.235])
    study.panel_label(ax_rob, "B", "Robustness verdict (powered vs all-cells)")
    powered_only = pd.DataFrame((data.get("tier3") or {}).get("powered_only", []))
    if powered_only.empty:
        empty_panel(ax_rob, "Robustness not populated")
    else:
        columns = ["arm", "outcome", "rel_improvement_all", "rel_improvement_powered", "verdict"]
        for column in columns:
            if column not in powered_only.columns:
                powered_only[column] = np.nan
        study.draw_compact_table(
            ax_rob, powered_only.loc[:, columns], columns,
            labels=["Arm", "Outcome", "Rel.\nall", "Rel.\npowered", "Verdict"], max_rows=8)

    ax_cov = figure.add_axes([0.055, 0.185, 0.44, 0.235])
    study.panel_label(ax_cov, "C", "Analysis coverage: scored cells and powered")
    coverage = pd.DataFrame(families)
    study.draw_compact_table(
        ax_cov, coverage.loc[:, ["family", "cells", "powered"]], ["family", "cells", "powered"],
        labels=["Analysis family", "Cells", "Powered"], max_rows=12)

    ax_high = figure.add_axes([0.55, 0.185, 0.40, 0.235])
    study.panel_label(ax_high, "D", "Clinical-value and subgroup highlights")
    ax_high.axis("off")
    ax_high.text(0.0, 0.99, "\n".join(_executive_highlights(data)), va="top", ha="left", fontsize=6.9,
                 color=study.PALETTE["ink"], linespacing=1.55, transform=ax_high.transAxes)

    figure.text(0.055, 0.147, "Supported vs exploratory", fontsize=9.5, fontweight="bold",
                color=study.PALETTE["ink"], va="top")
    figure.text(0.055, 0.123, textwrap.fill(_supported_vs_exploratory_text(data, tte), 188),
                fontsize=7.1, color=study.PALETTE["muted"], va="top")
    return figure


@register_page(1)
def _page_run_identity(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                       title: str, subtitle: str) -> Any:
    """Fingerprints (script SHA-256, config hash, source-run fingerprint), dependency versions,
    the source RUN_DIR, and the FULL column-population audit (from data['assemble']['audit'])
    rendered with study.draw_compact_table."""
    figure = secondary_new_page(number, title, subtitle)
    assemble = data.get("assemble") or {}
    audit = assemble.get("audit")
    script_hash = script_sha256()
    cfg_hash = str(data.get("config_hash", config_hash(cfg)))
    fingerprint = source_run_fingerprint(cfg)

    ax_fp = figure.add_axes([0.055, 0.585, 0.42, 0.255])
    study.panel_label(ax_fp, "A", "Run fingerprints")
    fingerprint_rows = [
        {"field": "secondary version", "value": str(data.get("secondary_version", SECONDARY_VERSION))},
        {"field": "source study version", "value": str(data.get("source_study_version", SOURCE_STUDY_VERSION))},
        {"field": "mode", "value": str(data.get("mode", cfg.mode))},
        {"field": "seed", "value": int(cfg.seed)},
        {"field": "config hash", "value": _short_hash(cfg_hash)},
        {"field": "script SHA-256", "value": _short_hash(script_hash)},
        {"field": "source manifest sha", "value": _short_hash(fingerprint.get("run_manifest_sha256")) or "n/a"},
    ]
    study.draw_compact_table(ax_fp, pd.DataFrame(fingerprint_rows), ["field", "value"],
                             labels=["Field", "Value"], max_rows=8)

    ax_dep = figure.add_axes([0.55, 0.585, 0.40, 0.255])
    study.panel_label(ax_dep, "B", "Dependencies and source")
    join_rate = assemble.get("join_rate")
    dependency_rows = [
        {"field": "python", "value": platform.python_version()},
        {"field": "numpy", "value": str(getattr(np, "__version__", "n/a"))},
        {"field": "pandas", "value": str(getattr(pd, "__version__", "n/a"))},
        {"field": "parquet spill", "value": bool(study._PARQUET_SPILL_ENABLED)},
        {"field": "source run exists", "value": bool(fingerprint.get("exists", False))},
        {"field": "join rate", "value": float(join_rate) if join_rate is not None else np.nan},
        {"field": "covariate rows", "value": int(assemble.get("covariate_rows", 0))},
    ]
    study.draw_compact_table(ax_dep, pd.DataFrame(dependency_rows), ["field", "value"],
                             labels=["Field", "Value"], max_rows=8)

    run_dir = str(fingerprint.get("run_dir", cfg.output_dir))
    if len(run_dir) > 92:
        run_dir = "..." + run_dir[-89:]
    figure.text(0.055, 0.552, "script SHA-256: " + script_hash, fontsize=6.3, family="monospace",
                color=study.PALETTE["muted"], va="top")
    figure.text(0.055, 0.533, "config hash:    " + cfg_hash, fontsize=6.3, family="monospace",
                color=study.PALETTE["muted"], va="top")
    figure.text(0.055, 0.514, "source RUN_DIR: " + run_dir, fontsize=6.3, family="monospace",
                color=study.PALETTE["muted"], va="top")

    ax_audit_left = figure.add_axes([0.055, 0.085, 0.42, 0.37])
    ax_audit_right = figure.add_axes([0.55, 0.085, 0.42, 0.37])
    study.panel_label(ax_audit_left, "C", "Column-population audit (1 of 2)")
    study.panel_label(ax_audit_right, "D", "Column-population audit (2 of 2)")
    audit_columns = ["column", "non_null_fraction", "n_distinct", "usable"]
    audit_labels = ["Covariate", "Non-null\nfrac", "Distinct", "Usable"]
    if audit is None or not hasattr(audit, "empty") or audit.empty:
        empty_panel(ax_audit_left, "Column-population audit not available")
        empty_panel(ax_audit_right, "")
    else:
        audit_frame = audit.loc[:, [column for column in audit_columns if column in audit.columns]].reset_index(drop=True)
        # The enriched baselines run past draw_compact_table's 22-character cell budget, which would
        # render glucose_fasting_baseline as a bare ellipsis. The suffix is uniform and dropping it
        # collides with nothing, so shorten it here; the CSV keeps the full column names.
        audit_frame["column"] = audit_frame["column"].astype(str).str.replace("_baseline", "", regex=False)
        half = (len(audit_frame) + 1) // 2
        study.draw_compact_table(ax_audit_left, audit_frame.iloc[:half], audit_columns,
                                 labels=audit_labels, max_rows=half + 1)
        study.draw_compact_table(ax_audit_right, audit_frame.iloc[half:], audit_columns,
                                 labels=audit_labels, max_rows=len(audit_frame) - half + 1)
    return figure


@register_page(16)
def _page_gates_limitations(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                            title: str, subtitle: str) -> Any:
    """Gates, limitations, and conclusion in the production honest register: which cells are
    adequately powered vs exploratory, the causal-only-under-conditional-exchangeability caveat
    with the E-value bound and the named absent confounders (GERD/reflux, surgeon/center), the
    state/SVI/RUCA transportability-approximation note, and the permitted vs not-permitted claims."""
    figure = secondary_new_page(number, title, subtitle)
    families = _analysis_family_summary(data)
    tte = _tte_summary(data)

    ax_gate = figure.add_axes([0.055, 0.485, 0.89, 0.34])
    study.panel_label(ax_gate, "A", "Adequately powered vs exploratory, by analysis family")
    gate_frame = pd.DataFrame(families)
    gate_frame["exploratory"] = gate_frame["cells"].astype(int) - gate_frame["powered"].astype(int)
    gate_columns = ["family", "cells", "powered", "exploratory", "gate"]
    study.draw_compact_table(
        ax_gate, gate_frame.loc[:, gate_columns], gate_columns,
        labels=["Analysis family", "Scored\ncells", "Adequately\npowered", "Exploratory", "Gate basis"],
        max_rows=12)

    ax_lim = figure.add_axes([0.055, 0.10, 0.52, 0.32])
    study.panel_label(ax_lim, "B", "Limitations")
    ax_lim.axis("off")
    ax_lim.text(0.0, 0.99, textwrap.fill(_limitations_text(data, tte), 82), va="top", ha="left",
                fontsize=7.4, linespacing=1.32, color=study.PALETTE["ink"], transform=ax_lim.transAxes)

    ax_conclusion = figure.add_axes([0.62, 0.10, 0.33, 0.32])
    study.panel_label(ax_conclusion, "C", "Permitted claims and conclusion")
    ax_conclusion.axis("off")
    total_cells = int(gate_frame["cells"].astype(int).sum())
    total_powered = int(gate_frame["powered"].astype(int).sum())
    ax_conclusion.text(0.0, 0.99, f"Adequately powered: {total_powered} of {total_cells}", fontsize=11,
                       fontweight="bold", va="top", transform=ax_conclusion.transAxes,
                       color=study.PALETTE["green"] if total_powered else study.PALETTE["orange"])
    ax_conclusion.text(0.0, 0.86, f"Exploratory: {total_cells - total_powered} of {total_cells}",
                       fontsize=11, fontweight="bold", color=study.PALETTE["orange"], va="top",
                       transform=ax_conclusion.transAxes)
    ax_conclusion.text(0.0, 0.70, textwrap.fill(_permitted_claims_text(), 44), fontsize=7.2,
                       color=study.PALETTE["ink"], va="top", transform=ax_conclusion.transAxes)

    omissions = _collect_omissions(data)
    ledger = "Omissions (no silent caps): " + (" | ".join(omissions) if omissions else "none recorded.")
    # Cap the RENDERED lines, not the item count. Capping items alone does not bound the height,
    # because the entries vary in length, and an overlong block prints straight over the page
    # footer. Three lines at width 205, anchored at y=0.085, is the budget that clears it; the
    # trimmed last line still points at the manifest, which always carries the complete ledger.
    lines = textwrap.wrap(ledger, 205)
    if len(lines) > 3:
        tail = f" ... ({len(omissions)} omissions listed in full in manifest.json)"
        lines = lines[:3]
        lines[-1] = lines[-1][:205 - len(tail)].rstrip() + tail
    figure.text(0.055, 0.085, "\n".join(lines), fontsize=6.0, color=study.PALETTE["muted"], va="top")
    return figure


# --------------------------------------------------------------------------------------------
# TTE figure pages (02 design/cohort, 03 propensity/positivity, 04 primary results)
# --------------------------------------------------------------------------------------------
def _short_covariate(name: str) -> str:
    return name.replace("__", " ").replace("_", " ").replace("=", ": ")


def _feature_label(name: str) -> str:
    """A baseline feature's display name, used by EVERY page that names one.

    Every such feature is a baseline covariate, so the redundant baseline_ prefix and _baseline
    suffix the source columns carry are dropped (they cost a y axis or a table column more width
    than they carry meaning), and the clinical acronyms are cased the way the rest of the book
    writes them. This is the book's standing convention - figures name things for a reader, CSVs
    keep the source column names - so pages 07, 18, and 19 all route their modifier and feature
    names through here and cate_modifier_importance.csv,
    which_incretin_modifier_importance.csv, and forecast_feature_sensitivity.csv all keep
    baseline_bmi.
    """
    acronyms = {"bmi": "BMI", "hba1c": "HbA1c", "egfr": "eGFR", "osa": "OSA", "sglt2": "SGLT2"}
    text = str(name)
    text = text[len("baseline_"):] if text.startswith("baseline_") else text
    text = text[:-len("_baseline")] if text.endswith("_baseline") else text
    # Cased per word, not per whole name, so hba1c_band reads "HbA1c band" beside a plain "HbA1c".
    return " ".join(acronyms.get(word, word) for word in _short_covariate(text).split(" "))


def _draw_modifier_importance(axis: Any, importance: Sequence[Mapping[str, Any]],
                              max_rows: int) -> None:
    """The permutation-importance modifier ranking, shared by the CATE and which-incretin pages.

    Ranking is done on the SOURCE names so the order matches the CSV exactly; only the rendered
    label is shortened.
    """
    frame = pd.DataFrame(list(importance or []))
    if not frame.empty:
        frame = frame.sort_values(["permutation_importance", "modifier"],
                                  ascending=[False, True]).head(max_rows)
        frame = frame.assign(modifier=frame["modifier"].map(_feature_label))
    study.draw_compact_table(axis, frame, list(CATE_IMPORTANCE_COLUMNS),
                             labels=["Effect modifier", "Permutation\nimportance"],
                             max_rows=max_rows)


def _tte_confounder_text(data: Mapping[str, Any]) -> str:
    """What the enriched, audit-gated L resolved to, and which axes it could not adjust for."""
    confounders = (data.get("tte") or {}).get("confounders") or {}
    entered = list(confounders.get("numeric") or []) + list(confounders.get("categorical") or [])
    return (f"{len(entered)} audit-gated baseline confounders entered L: "
            + (", ".join(entered) if entered else "none")
            + ". Unpopulated in this source, so omitted from L rather than carried as an all-NaN "
            "column, and therefore unadjusted: " + ", ".join(_tte_absent_confounders(data)) + ".")


def _tte_skip_panel(figure: Any, tte: Mapping[str, Any]) -> Any:
    """A full-page graceful note when the TTE was skipped or not populated."""
    axis = figure.add_axes([0.08, 0.12, 0.84, 0.72])
    if str(tte.get("status")) == "skipped":
        empty_panel(axis, f"TTE skipped: {tte.get('skip_reason', 'gate not satisfied')}")
    else:
        empty_panel(axis, "TTE not populated in this source")
    return figure


def _draw_love_plot(axis: Any, love: Mapping[str, Any]) -> None:
    outcome = "bmi" if "bmi" in love else (next(iter(love)) if love else None)
    entry = love.get(outcome) if outcome else None
    if not entry or not entry.get("names"):
        empty_panel(axis, "Balance not estimable")
        return
    names = list(entry["names"])
    unweighted = np.asarray(entry["unweighted"], dtype=float)
    weighted = np.asarray(entry["weighted"], dtype=float)
    order = np.argsort(-np.abs(np.where(np.isfinite(unweighted), unweighted, 0.0)))[:14]
    names = [names[i] for i in order]
    unweighted = unweighted[order]
    weighted = weighted[order]
    y = np.arange(len(names))[::-1]
    axis.axvspan(-0.1, 0.1, color=study.PALETTE["grid"], alpha=0.5, zorder=0)
    axis.axvline(0.0, color=study.PALETTE["muted"], lw=0.8, zorder=1)
    for edge in (-0.1, 0.1):
        axis.axvline(edge, color=study.PALETTE["muted"], lw=0.6, ls=":", zorder=1)
    axis.scatter(unweighted, y, s=26, color=study.PALETTE["orange"], label="Unweighted", zorder=3)
    axis.scatter(weighted, y, s=26, color=study.PALETTE["blue"], marker="D",
                 label="IPTW-weighted", zorder=3)
    axis.set_yticks(y)
    axis.set_yticklabels([_short_covariate(name) for name in names], fontsize=6.3)
    axis.set_xlabel("Standardized mean difference", fontsize=8)
    axis.legend(fontsize=6.3, loc="lower right", frameon=False)
    axis.set_title(_tte_population_label(outcome), loc="right", fontsize=7.0, color=study.PALETTE["muted"])


def _draw_ps_overlap(axis: Any, ps_overlap: Mapping[str, Any], trim: tuple[float, float]) -> None:
    outcome = "bmi" if "bmi" in ps_overlap else (next(iter(ps_overlap)) if ps_overlap else None)
    entry = ps_overlap.get(outcome) if outcome else None
    if not entry:
        empty_panel(axis, "Propensity not estimable")
        return
    bins = np.linspace(0.0, 1.0, 21)
    sleeve = np.asarray(entry.get("sleeve", []), dtype=float)
    rygb = np.asarray(entry.get("rygb", []), dtype=float)
    if sleeve.size:
        axis.hist(sleeve, bins=bins, color=study.PALETTE["blue"], alpha=0.55, density=True,
                  label="Sleeve (A=0)")
    if rygb.size:
        axis.hist(rygb, bins=bins, color=study.PALETTE["orange"], alpha=0.55, density=True,
                  label="RYGB (A=1)")
    for edge in trim:
        axis.axvline(float(edge), color=study.PALETTE["red"], lw=0.8, ls=":")
    axis.set_xlabel("P(RYGB | L)", fontsize=8)
    axis.set_ylabel("Density", fontsize=8)
    axis.legend(fontsize=6.3, loc="upper center", frameon=False)
    axis.set_title(_tte_population_label(outcome), loc="right", fontsize=7.0, color=study.PALETTE["muted"])


def _draw_ate_forest(axis: Any, sub: Any, xlabel: str) -> None:
    if sub is None or sub.empty or not sub["ate"].notna().any():
        empty_panel(axis, "No estimate past positivity gate")
        return
    sub = sub.dropna(subset=["ate"]).sort_values("target_month")
    ate = sub["ate"].to_numpy(dtype=float)
    low = sub["ci_low"].to_numpy(dtype=float)
    high = sub["ci_high"].to_numpy(dtype=float)
    y = np.arange(len(sub))[::-1]
    axis.axvline(0.0, color=study.PALETTE["muted"], lw=0.9)
    axis.errorbar(ate, y, xerr=[np.clip(ate - low, 0, None), np.clip(high - ate, 0, None)],
                  fmt="o", color=study.PALETTE["blue"], ecolor=study.PALETTE["sky"], capsize=3, ms=5)
    axis.set_yticks(y)
    axis.set_yticklabels([f"{int(month)} mo" for month in sub["target_month"]], fontsize=7)
    axis.set_xlabel(xlabel, fontsize=8)


@register_page(2)
def _page_tte_design(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                     title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tte = data.get("tte") or {}
    if str(tte.get("status")) in {"skipped", "None", ""} or not tte.get("populations"):
        return _tte_skip_panel(figure, tte)
    ax_counts = figure.add_axes([0.06, 0.52, 0.40, 0.30])
    study.panel_label(ax_counts, "A", "Arms by outcome population")
    rows = []
    for outcome in OUTCOMES:
        population = tte["populations"].get(outcome)
        if not population:
            continue
        rows.append({
            "population": _tte_population_label(outcome), "n": int(population["n"]),
            "n_rygb": int(population["n_rygb"]), "n_sleeve": int(population["n_sleeve"]),
        })
    study.draw_compact_table(ax_counts, pd.DataFrame(rows), ["population", "n", "n_rygb", "n_sleeve"],
                             labels=["Population", "N", "RYGB", "Sleeve"])
    ax_love = figure.add_axes([0.55, 0.14, 0.40, 0.68])
    _draw_love_plot(ax_love, tte.get("love", {}))
    study.panel_label(ax_love, "B", "Covariate balance (Love plot)")
    figure.text(0.06, 0.46, textwrap.fill("Estimand. " + ESTIMAND_NOTE, 74), fontsize=7.2,
                color=study.PALETTE["muted"], va="top")
    figure.text(0.06, 0.34, textwrap.fill("Confounder set L. " + _tte_confounder_text(data), 74),
                fontsize=7.2, color=study.PALETTE["muted"], va="top")
    return figure


@register_page(3)
def _page_tte_positivity(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                         title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tte = data.get("tte") or {}
    if str(tte.get("status")) in {"skipped", "None", ""} or not tte.get("ps_overlap"):
        return _tte_skip_panel(figure, tte)
    ax_ps = figure.add_axes([0.07, 0.16, 0.40, 0.62])
    _draw_ps_overlap(ax_ps, tte.get("ps_overlap", {}), tuple(tte.get("trim", IPTW_TRIM)))
    study.panel_label(ax_ps, "A", "Propensity overlap by arm")
    ax_table = figure.add_axes([0.54, 0.14, 0.42, 0.66])
    study.panel_label(ax_table, "B", "Positivity and effective sample size by cell")
    cells = pd.DataFrame(tte.get("cells", []))
    if cells.empty:
        empty_panel(ax_table, "No cells estimable")
    else:
        columns = ["outcome", "target_month", "n", "ess_iptw", "min_ps", "positivity_fail"]
        study.draw_compact_table(
            ax_table, cells.loc[:, columns], columns,
            labels=["Outcome", "Target\nmonth", "N", "IPTW\nESS", "Min PS", "Positivity\nfail"],
            max_rows=14,
        )
    figure.text(0.07, 0.115, textwrap.fill(
        "Propensity and positivity are re-audited over the enriched L. " + _tte_confounder_text(data), 195),
        fontsize=6.5, color=study.PALETTE["muted"], va="top")
    return figure


@register_page(4)
def _page_tte_results(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                      title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tte = data.get("tte") or {}
    cells = pd.DataFrame(tte.get("cells", []))
    if str(tte.get("status")) in {"skipped", "None", ""} or cells.empty:
        return _tte_skip_panel(figure, tte)
    display = cells.copy()
    fail = display["positivity_fail"].fillna(False).astype(bool)
    for column in ("ate", "se", "ci_low", "ci_high", "e_value_point", "e_value_ci", "c_for_benefit"):
        display.loc[fail, column] = np.nan
    display["overlaps_rct_ci"] = display["overlaps_rct_ci"].astype(object)
    display.loc[fail, "overlaps_rct_ci"] = None
    ax_bmi = figure.add_axes([0.07, 0.55, 0.40, 0.24])
    study.panel_label(ax_bmi, "A", "ATE: BMI (kg/m2)")
    _draw_ate_forest(ax_bmi, display[display["outcome"] == "bmi"], "RYGB - sleeve, BMI")
    ax_hba = figure.add_axes([0.57, 0.55, 0.38, 0.24])
    study.panel_label(ax_hba, "B", "ATE: HbA1c (%)")
    _draw_ate_forest(ax_hba, display[display["outcome"] == "hba1c"], "RYGB - sleeve, HbA1c")
    figure.text(0.06, 0.495, textwrap.fill(
        "The marginal doubly-robust IPCW-AIPW ATE over the enriched, audit-gated L is the PRIMARY "
        "estimand; negative ATE favors RYGB. Procedure choice is a point intervention at time zero, "
        "so ITT and per-protocol coincide. BMI 12/24-mo ATE is benchmarked as %TWL against the "
        "SM-BOSS/SLEEVEPASS anchor (RCT overlap column). Causal only under conditional "
        "exchangeability given measured L; the E-value and RCT overlap bound confounding by the "
        "axes still absent from this source and therefore omitted from L: "
        + ", ".join(_tte_absent_confounders(data)) + ". Point estimates are suppressed "
        "where the positivity gate fails.", 168),
        fontsize=6.5, color=study.PALETTE["muted"], va="top")
    ax_table = figure.add_axes([0.06, 0.06, 0.89, 0.32])
    study.panel_label(ax_table, "C", "Doubly-robust AIPW summary")
    columns = ["outcome", "target_month", "n", "ate", "ci_low", "ci_high", "e_value_point",
               "e_value_ci", "overlaps_rct_ci", "c_for_benefit", "positivity_fail"]
    study.draw_compact_table(
        ax_table, display.loc[:, columns], columns,
        labels=["Outcome", "Target\nmonth", "N", "ATE", "CI low", "CI high", "E-value",
                "E-value\nCI", "RCT\noverlap", "c-for-\nbenefit", "Posit.\nfail"],
        max_rows=14,
    )
    return figure


# --------------------------------------------------------------------------------------------
# Tier 1 - transportability/equity (T1.1), attrition/MNAR sensitivity (T1.2), and drug/procedure
# heterogeneity + SOPHIA context (T1.3). Every prognostic metric reads only the frozen selected
# candidate at origin 0 on the held-out splits (HELD_OUT_SPLITS), exactly like
# study.subgroup_performance, streamed one task partition at a time and joined to the covariate
# frame on patient_id. These panels audit error and calibration; they do not establish biological
# effect modification.
# --------------------------------------------------------------------------------------------
# Columns streamed per origin-0 partition for the held-out prognostic evaluation.
TIER1_READ_COLUMNS = (
    "patient_id", "target_month", "split", "candidate", "target_observed", "target_value",
    "analysis_weight",
) + tuple(QUANTILE_COLS)

# Symmetric MNAR delta grids: BMI in kg/m^2 (-6..+6), HbA1c in % on a proportionate scale.
DELTA_GRID_BMI = tuple(float(value) for value in range(-6, 7))
DELTA_GRID_HBA1C = tuple(round(-1.5 + 0.25 * i, 3) for i in range(13))

# Top-N states retained by patient volume before the rest pool into "other" (no silent cap).
TRANSPORT_TOP_STATES = 10
# |SMD| above this flags a covariate as an observed-vs-censored (MNAR) imbalance risk.
MNAR_SMD_FLAG = 0.10

# Baseline covariates compared between observed and censored patients (T1.2c). Numeric only so
# tte.standardized_mean_diff reads them directly; baseline_value is outcome-appropriate.
ATTRITION_SMD_COVARIATES = (
    "baseline_value", "age_at_index", "diabetes_flag", "hypertension", "dyslipidemia", "osa",
    "insulin", "biguanide", "sglt2", "svi", "index_year",
)

# The production caveat carried verbatim onto every subgroup page.
SUBGROUP_CAVEAT = (
    "These panels audit error and calibration; they do not establish biological effect "
    "modification or rank patient groups."
)

# Every Section-10 tier is cut over the three reporting arms, so its CSV header says so once.
ARM_RECUT_NOTE = (
    "Reported per REPORTING ARM: the incretin cohort, and the shared surgical model subset by "
    "procedure into RYGB and sleeve (separately reported, not separately fitted). The arms are "
    "formed by the single shared arm primitive every per-arm read uses. No arm-cell below the "
    "n < 11 disclosure floor is scored: where this analysis emits a row per declared cell that "
    "row is blanked, and where it scores only disclosable cells the thin cell is absent."
)

TRANSPORTABILITY_COLUMNS = (
    "arm", "outcome", "axis", "stratum", "target_month", "n", "ess", "crps",
    "coverage_80", "coverage_90", "cal_slope", "cal_intercept", "powered",
    "p_value_cov80", "q_value_cov80", "skipped_reason",
)
TIPPING_POINT_COLUMNS = (
    "arm", "outcome", "target_month", "n", "n_censored", "censored_fraction",
    "model_beats_baseline_at_0", "delta_star_crps", "is_surgery", "ate_at_0",
    "delta_star_ate", "powered",
)
TRIAGE_COLUMNS = (
    "arm", "outcome", "target_month", "estimator", "n", "rmse", "rmse_ci_low",
    "rmse_ci_high", "crps", "powered",
)
ATTRITION_SMD_COLUMNS = (
    "arm", "outcome", "target_month", "covariate", "smd", "abs_smd", "mnar_flag",
    "n_observed", "n_censored",
)
# Every count the attrition CSVs publish. n_censored has to be suppressed like any other count:
# censored_fraction times n recovers it exactly, and the observed-vs-censored SMD is a statistic
# ABOUT the censored group, so a sub-11 censored count takes its whole row with it.
ATTRITION_COUNT_COLUMNS = ("n", "n_observed", "n_censored")
HETEROGENEITY_DRUG_COLUMNS = (
    "outcome", "drug_group", "target_month", "n", "ess", "crps", "rmse", "mae",
    "coverage_80", "cal_slope", "auroc_bmi35", "event_rate_bmi35", "powered",
    "p_value_cov80", "q_value_cov80",
)
DRUG_GROUP_LABELS = {
    "tirzepatide": "Tirzepatide", "semaglutide_injectable": "Semaglutide inj",
    "semaglutide_oral": "Semaglutide oral", "dulaglutide": "Dulaglutide",
    "liraglutide": "Liraglutide", "older": "Older agents",
}
# Spec 10 asks for a "which agent for whom" read on a within-incretin contrast where a defensible
# comparator exists. Tirzepatide against INJECTABLE semaglutide is the only pair that qualifies:
# same route, same weekly schedule, same prescribing era, same indications. Oral semaglutide and
# the older agents differ in administration or potency class, so they are never contrasted here.
WITHIN_INCRETIN_TREATED = "tirzepatide"
WITHIN_INCRETIN_COMPARATOR = "semaglutide_injectable"
WITHIN_INCRETIN_CAVEAT = (
    "PROGNOSTIC HETEROGENEITY, NOT A CAUSAL TREATMENT EFFECT. Incretin agent choice is "
    "prescriber-driven and never randomized, and this source carries no defensible propensity for "
    "it: formulary position, prior authorization, tolerability, and supply shortage all drive "
    "which agent a patient receives and none of them is measured here. Spec 8.5 keeps this "
    "contrast secondary and exploratory. Read tau_hat as a map of which patients' forecasts "
    "differ between agents, never as the effect of switching agent."
)
HETEROGENEITY_PROCEDURE_COLUMNS = (
    "outcome", "procedure", "target_month", "n", "ess", "crps", "rmse", "mae",
    "coverage_80", "cal_slope", "powered", "sophia_context_rmse", "p_value_cov80",
    "q_value_cov80",
)


# ----- small numeric helpers ---------------------------------------------------------------
def _delta_grid(outcome: str) -> tuple[float, ...]:
    return DELTA_GRID_HBA1C if outcome == "hba1c" else DELTA_GRID_BMI


def _normal_cdf(z: float) -> float:
    """Standard-normal CDF via erf (no scipy); used only for the coverage calibration test."""
    if not np.isfinite(z):
        return float("nan")
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _weighted_linfit(x: Any, y: Any, weight: Any) -> tuple[float, float]:
    """IPCW-weighted least squares of y on x, returning (slope, intercept); NaN if degenerate.

    Calibration regresses the observed outcome (y) on the predicted median (x); slope 1 and
    intercept 0 are ideal. Solved through the weighted normal equations so the weights are true
    statistical weights (not polyfit's sqrt-weight convention).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weight, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if x.size < 3 or float(np.ptp(x)) < 1e-9:
        return float("nan"), float("nan")
    sw = float(np.sum(w))
    swx = float(np.sum(w * x))
    swy = float(np.sum(w * y))
    swxx = float(np.sum(w * x * x))
    swxy = float(np.sum(w * x * y))
    det = sw * swxx - swx * swx
    if abs(det) < 1e-12:
        return float("nan"), float("nan")
    slope = (sw * swxy - swx * swy) / det
    intercept = (swxx * swy - swx * swxy) / det
    return float(slope), float(intercept)


def _quantile_matrix(frame: Any) -> Any:
    return frame.loc[:, list(QUANTILE_COLS)].to_numpy(dtype=float)


def _cell_weight(frame: Any) -> Any:
    """IPCW analysis weights for one cell; non-finite/non-positive dropped, all-zero -> ones."""
    w = pd.to_numeric(frame["analysis_weight"], errors="coerce").to_numpy(dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if float(np.sum(w)) <= 0.0:
        w = np.ones(w.shape[0], dtype=float)
    return w


def _effective_sample_size(weight: Any) -> float:
    return float(study.effective_sample_size(np.asarray(weight, dtype=float)))


def _is_powered(n: int, ess: float) -> bool:
    return bool(int(n) >= POWERED_MIN_N and np.isfinite(ess) and ess >= POWERED_MIN_ESS)


def _prognostic_metrics(y: Any, matrix: Any, weight: Any) -> dict[str, float]:
    """CRPS/RMSE/MAE/bias, 80 and 90 interval coverage and width, IPCW calibration slope/intercept.

    Mirrors study.subgroup_performance's scorer choices: q50 (column 3) is the median, coverage_80
    uses q10/q90 (columns 1/5), coverage_90 uses q05/q95 (columns 0/6).
    """
    y = np.asarray(y, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    weight = np.asarray(weight, dtype=float)
    median = matrix[:, 3]
    resid = y - median
    slope, intercept = _weighted_linfit(median, y, weight)
    return {
        "crps": float(study.quantile_crps(y, matrix, weight)),
        "rmse": float(math.sqrt(max(study.weighted_mean(resid * resid, weight), 0.0))),
        "mae": float(study.weighted_mean(np.abs(resid), weight)),
        "bias": float(study.weighted_mean(resid, weight)),
        "coverage_80": float(study.weighted_mean((y >= matrix[:, 1]) & (y <= matrix[:, 5]), weight)),
        "coverage_90": float(study.weighted_mean((y >= matrix[:, 0]) & (y <= matrix[:, 6]), weight)),
        "width_80": float(study.weighted_mean(matrix[:, 5] - matrix[:, 1], weight)),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }


def _bh_fdr(pvalues: Sequence[float]) -> Any:
    """Benjamini-Hochberg q-values (q = 0.05 family); NaN p-values pass through as NaN."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.where(np.isfinite(p))[0]
    m = int(valid.size)
    if m == 0:
        return q
    order = valid[np.argsort(p[valid], kind="mergesort")]
    adjusted = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def _coverage_pvalue(coverage: float, n_eff: float, target: float) -> float:
    """Two-sided normal-approximation p-value for observed interval coverage vs its nominal target."""
    if not (np.isfinite(coverage) and np.isfinite(n_eff)) or n_eff < 1.0:
        return float("nan")
    se = math.sqrt(target * (1.0 - target) / n_eff)
    if se <= 0.0:
        return float("nan")
    z = (coverage - target) / se
    return float(2.0 * (1.0 - _normal_cdf(abs(z))))


def _apply_fdr(frame: Any, group_columns: Sequence[str], p_column: str, q_column: str) -> Any:
    """BH-FDR the p_column within each group family and write q_column back onto the frame."""
    result = frame.copy()
    if result.empty:
        result[q_column] = pd.Series(dtype=float)
        return result
    result[q_column] = np.nan
    for _, index in result.groupby(list(group_columns), sort=False, dropna=False).groups.items():
        result.loc[index, q_column] = _bh_fdr(pd.to_numeric(result.loc[index, p_column], errors="coerce").to_numpy(float))
    return result


# ----- delta tipping-point cores (T1.2a) ---------------------------------------------------
def _crossing_nearest_zero(grid: Sequence[float], curve: Sequence[float]) -> float:
    """Grid location nearest delta=0 where the curve changes sign (linear interp); NaN if none."""
    g = np.asarray(grid, dtype=float)
    c = np.asarray(curve, dtype=float)
    best = float("nan")
    best_abs = float("inf")
    for i in range(g.size - 1):
        a, b = c[i], c[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        crossing = None
        if a == 0.0:
            crossing = float(g[i])
        elif a * b < 0.0:
            t = a / (a - b)
            crossing = float(g[i] + t * (g[i + 1] - g[i]))
        if crossing is not None and abs(crossing) < best_abs:
            best_abs = abs(crossing)
            best = crossing
    if not np.isfinite(best) and np.isfinite(c[-1]) and c[-1] == 0.0:
        best = float(g[-1])
    return best


def tipping_point_effect_curve(observed: Any, pred_median: Any, arm: Any, weight: Any,
                               delta_grid: Sequence[float], base_outcome: Any) -> tuple[Any, float, float]:
    """MNAR arm-contrast (ATE proxy) as censored outcomes are imputed at pred_median shifted by
    delta - RYGB (arm==1) by +delta, sleeve (arm==0) by -delta - over the grid.

    Returns (ate_curve, ate_at_zero, delta_star). The RYGB imputed mean rises and the sleeve mean
    falls with delta, so the contrast is linear and MONOTONE non-decreasing in delta; delta_star
    (where it crosses zero) therefore has the sign opposite to ate_at_zero.
    """
    observed = np.asarray(observed).astype(bool)
    pred_median = np.asarray(pred_median, dtype=float)
    arm = np.asarray(arm).astype(int)
    weight = np.asarray(weight, dtype=float)
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    base = np.where(observed, np.asarray(base_outcome, dtype=float), pred_median)
    sign = np.where(arm == 1, 1.0, -1.0)
    rygb = arm == 1
    sleeve = arm == 0
    curve: list[float] = []
    for delta in delta_grid:
        shifted = np.where(observed, base, base + sign * float(delta))
        m1 = study.weighted_mean(shifted[rygb], weight[rygb]) if bool(rygb.any()) else float("nan")
        m0 = study.weighted_mean(shifted[sleeve], weight[sleeve]) if bool(sleeve.any()) else float("nan")
        curve.append(float(m1 - m0))
    curve = np.asarray(curve, dtype=float)
    zero_index = int(np.argmin(np.abs(np.asarray(delta_grid, dtype=float))))
    return curve, float(curve[zero_index]), _crossing_nearest_zero(delta_grid, curve)


def tipping_point_crps_curve(observed: Any, y: Any, matrix: Any, weight: Any,
                             delta_grid: Sequence[float]) -> tuple[Any, float, float, bool]:
    """Model IPCW-CRPS skill margin vs a population-change baseline as censored outcomes are
    imputed at the model median + delta.

    margin(delta) = baseline_CRPS - model_CRPS (positive -> the model still beats the naive
    population-mean forecast). Returns (margin_curve, margin_at_zero, delta_star, beats_at_zero)
    with delta_star the grid crossing nearest zero (where the model stops beating the baseline).
    """
    observed = np.asarray(observed).astype(bool)
    y = np.asarray(y, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    weight = np.asarray(weight, dtype=float)
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    median = matrix[:, 3]
    baseline_const = (
        study.weighted_mean(y[observed], weight[observed]) if bool(observed.any()) else float("nan")
    )
    margins: list[float] = []
    for delta in delta_grid:
        y_full = np.where(observed, y, median + float(delta))
        model_crps = float(study.quantile_crps(y_full, matrix, weight))
        baseline_crps = float(study.weighted_mean(np.abs(y_full - baseline_const), weight))
        margins.append(baseline_crps - model_crps)
    margins = np.asarray(margins, dtype=float)
    zero_index = int(np.argmin(np.abs(np.asarray(delta_grid, dtype=float))))
    margin0 = float(margins[zero_index])
    return margins, margin0, _crossing_nearest_zero(delta_grid, margins), bool(margin0 > 0.0)


# ----- stratum label helpers ---------------------------------------------------------------
def _svi_tertile_labels(values: Any) -> Any:
    """SVI split into low/mid/high tertiles; NA where SVI is missing or collapses to < 3 bins."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=numeric.index, dtype="string")
    try:
        binned = pd.qcut(numeric, 3, duplicates="drop")
    except (ValueError, IndexError):
        return result
    categories = list(binned.cat.categories)
    names = {3: ["low", "mid", "high"], 2: ["low", "high"], 1: ["all"]}.get(
        len(categories), [f"t{i + 1}" for i in range(len(categories))]
    )
    codes = binned.cat.codes.to_numpy()
    for position, code in enumerate(codes):
        if code >= 0:
            result.iloc[position] = names[code]
    return result


def _ruca_urban_rural(values: Any) -> Any:
    """RUCA collapsed to urban (codes 1-3) vs rural (4+); NA where RUCA is missing."""
    numeric = pd.to_numeric(values, errors="coerce")
    labels = np.where(~np.isfinite(numeric.to_numpy(float)), None,
                      np.where(numeric.to_numpy(float) <= 3.0, "urban", "rural"))
    return pd.Series(labels, index=numeric.index, dtype="string")


def _state_top_n_labels(values: Any, top_n: int = TRANSPORT_TOP_STATES) -> tuple[Any, list[str]]:
    """Keep the top-N states by patient volume; pool the rest into 'other' (no silent cap)."""
    text = values.astype("string")
    valid = text[text.notna() & (text.str.strip() != "") & (text.str.lower() != "unknown")]
    counts = valid.value_counts()
    top = [str(value) for value in counts.index[:top_n]]
    labels = text.where(text.isin(top), other="other")
    labels = labels.where(text.notna() & (text.str.strip() != "") & (text.str.lower() != "unknown"))
    return labels.astype("string"), top


def _drug_group(ingredient: Any, route: Any) -> str | None:
    """Map an incretin index ingredient (+ route for semaglutide) to a Tier 1.3 drug group."""
    ingredient = str(ingredient).strip().lower()
    route = str(route).strip().lower()
    if ingredient in OLDER_INCRETINS:
        return "older"
    if ingredient == "tirzepatide":
        return "tirzepatide"
    if ingredient == "dulaglutide":
        return "dulaglutide"
    if ingredient == "liraglutide":
        return "liraglutide"
    if ingredient == "semaglutide":
        return "semaglutide_oral" if route in {"oral", "tablet", "po"} else "semaglutide_injectable"
    return None


# ----- held-out prognostic frames (streamed per task partition) ----------------------------
def _tier1_heldout_frames(cfg: SecondaryConfig, store: "study.PredictionStore", selected: Any,
                          covariates: Any) -> dict[tuple[str, str], Any]:
    """Per (cohort, outcome) held-out frame at origin 0: selected candidate, HELD_OUT_SPLITS,
    joined to the covariate frame on patient_id. Streams one origin-0 partition at a time."""
    sel = selected_map(selected)
    covariates = covariates.copy()
    covariates["patient_id"] = covariates["patient_id"].astype(str)
    covariates["cohort"] = covariates["cohort"].astype(str)
    frames: dict[tuple[str, str], Any] = {}
    for key in store.keys():
        cohort, outcome, origin = str(key[0]), str(key[1]), int(key[2])
        if origin != 0:
            continue
        candidate = sel.get((cohort, outcome, 0))
        if candidate is None:
            continue
        rows = store.read(list(key), columns=list(TIER1_READ_COLUMNS))
        if rows is None or rows.empty:
            continue
        rows = rows.loc[rows["candidate"].astype(str) == candidate]
        rows = rows.loc[rows["split"].astype(str).isin(HELD_OUT_SPLITS)]
        if rows.empty:
            continue
        rows = rows.copy()
        rows["patient_id"] = rows["patient_id"].astype(str)
        cov = covariates.loc[covariates["cohort"] == cohort].drop(columns=["cohort"])
        merged = rows.merge(cov, on="patient_id", how="left")
        merged["baseline_value"] = pd.to_numeric(merged.get(_tte_baseline_column(outcome)), errors="coerce")
        frames[(cohort, outcome)] = merged.reset_index(drop=True)
        del rows, merged
        gc.collect()
    return frames


# ----- T1.1 transportability / equity ------------------------------------------------------
def _tier1_transportability(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any],
                            audit: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    axes_plan = (
        ("state", "state", "urban/rural unavailable", bool(audit_usable(audit, "state"))),
        ("svi", "svi", "SVI not populated in this source", bool(audit_usable(audit, "svi", tertile=True))),
        ("ruca", "ruca", "RUCA not populated in this source", bool(audit_usable(audit, "ruca"))),
    )
    axes_usable = {axis: usable for axis, _column, _msg, usable in axes_plan}
    rows: list[dict[str, Any]] = []
    for arm, outcome, frame in _reporting_arm_frames(frames):
        observed = frame.loc[frame["target_observed"].fillna(False).astype(bool)].copy()
        for axis, column, message, usable in axes_plan:
            if not usable or column not in observed.columns:
                rows.append({
                    "arm": arm, "outcome": outcome, "axis": axis, "stratum": "",
                    "target_month": np.nan, "n": 0, "ess": np.nan, "crps": np.nan,
                    "coverage_80": np.nan, "coverage_90": np.nan, "cal_slope": np.nan,
                    "cal_intercept": np.nan, "powered": False, "p_value_cov80": np.nan,
                    "q_value_cov80": np.nan,
                    "skipped_reason": message if not usable else f"{column} absent from source",
                })
                continue
            if observed.empty:
                continue
            if axis == "svi":
                labels = _svi_tertile_labels(observed[column])
            elif axis == "ruca":
                labels = _ruca_urban_rural(observed[column])
            else:
                labels, _top = _state_top_n_labels(observed[column])
            work = observed.assign(_stratum=labels)
            work = work.loc[work["_stratum"].notna() & (work["_stratum"].astype(str) != "")]
            for (target_month, stratum), cell in work.groupby(["target_month", "_stratum"], sort=True):
                if cell.empty:
                    continue
                y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
                matrix = _quantile_matrix(cell)
                weight = _cell_weight(cell)
                metrics = _prognostic_metrics(y, matrix, weight)
                ess = _effective_sample_size(weight)
                n = int(len(cell))
                rows.append({
                    "arm": arm, "outcome": outcome, "axis": axis, "stratum": str(stratum),
                    "target_month": int(target_month), "n": n, "ess": ess,
                    "crps": metrics["crps"], "coverage_80": metrics["coverage_80"],
                    "coverage_90": metrics["coverage_90"], "cal_slope": metrics["cal_slope"],
                    "cal_intercept": metrics["cal_intercept"], "powered": _is_powered(n, ess),
                    "p_value_cov80": _coverage_pvalue(metrics["coverage_80"], ess, 0.80),
                    "q_value_cov80": np.nan, "skipped_reason": "",
                })
    loso_note = ""
    if cfg.loso_refit:
        loso_note = (
            "Leave-one-state-out refit requested (--loso-refit): the frozen store drops train "
            "rows, so a true refit needs the full modeling pipeline; the frozen re-validation "
            "above is reported and the refit is recorded as an omission, not silently capped."
        )
    meta = {"axes_usable": axes_usable, "loso_requested": bool(cfg.loso_refit), "loso_note": loso_note}
    return rows, meta


# ----- T1.2 attrition / MNAR sensitivity ---------------------------------------------------
def _weighted_rmse(resid: Any, weight: Any) -> float:
    resid = np.asarray(resid, dtype=float)
    weight = np.asarray(weight, dtype=float)
    return float(math.sqrt(max(study.weighted_mean(resid * resid, weight), 0.0)))


def _patient_bootstrap_rmse_ci(resid: Any, weight: Any, patient_ids: Any, rng: Any,
                               reps: int) -> tuple[float, float]:
    """Patient-clustered percentile CI of the weighted RMSE (resample patients, not rows)."""
    resid = np.asarray(resid, dtype=float)
    weight = np.asarray(weight, dtype=float)
    patient_ids = np.asarray(patient_ids)
    unique = np.unique(patient_ids)
    if unique.size < 2 or reps <= 0 or resid.size == 0:
        return float("nan"), float("nan")
    index_by_patient = {patient: np.where(patient_ids == patient)[0] for patient in unique}
    estimates = np.empty(reps, dtype=float)
    for replicate in range(reps):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([index_by_patient[patient] for patient in drawn])
        estimates[replicate] = _weighted_rmse(resid[idx], weight[idx])
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi)


def _impute_censored_outcome(cell: Any, y: Any, observed: Any, median: Any, seed: int) -> Any:
    """One MI draw: impute the censored target via sklearn IterativeImputer over baseline
    covariates + the predicted median; fall back to the predicted median if the imputer fails."""
    y = np.asarray(y, dtype=float)
    observed = np.asarray(observed).astype(bool)
    median = np.asarray(median, dtype=float)
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        numeric_columns = [column for column in ATTRITION_SMD_COVARIATES if column in cell.columns]
        feature_blocks = []
        for column in numeric_columns:
            values = pd.to_numeric(cell[column], errors="coerce").to_numpy(float)
            if np.isfinite(values).any():
                feature_blocks.append(values)
        feature_blocks.append(median)
        features = np.column_stack(feature_blocks)
        target = y.copy()
        target[~observed] = np.nan
        design = np.column_stack([features, target.reshape(-1, 1)])
        imputer = IterativeImputer(random_state=int(seed), sample_posterior=True, max_iter=5)
        completed = imputer.fit_transform(design)
        imputed = completed[:, -1]
        if not np.isfinite(imputed).all():
            raise ValueError("non-finite imputation")
        # Restrict draws to the observed support (expanded by half its range) so posterior draws
        # stay physically plausible - a standard bounded-MI guard that still reflects uncertainty.
        observed_values = y[observed]
        if observed_values.size:
            lo = float(np.min(observed_values))
            hi = float(np.max(observed_values))
            span = hi - lo
            imputed = np.clip(imputed, lo - 0.5 * span, hi + 0.5 * span)
        return np.where(observed, y, imputed)
    except Exception:
        return np.where(observed, y, median)


def _estimator_triage(arm: str, outcome: str, target_month: int, cell: Any, observed: Any,
                      y: Any, matrix: Any, weight: Any, cfg: SecondaryConfig, rng: Any,
                      powered: bool) -> list[dict[str, Any]]:
    """Complete-case vs IPCW vs multiple-imputation headline RMSE/CRPS with patient-clustered CIs."""
    observed = np.asarray(observed).astype(bool)
    y = np.asarray(y, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    weight = np.asarray(weight, dtype=float)
    median = matrix[:, 3]
    patient_ids = cell["patient_id"].astype(str).to_numpy()
    reps = int(cfg.bootstrap_replicates)
    obs_idx = np.where(observed)[0]
    n_obs = int(obs_idx.size)
    rows: list[dict[str, Any]] = []

    def _record(estimator: str, rmse: float, ci: tuple[float, float], crps: float, n: int) -> None:
        rows.append({
            "arm": arm, "outcome": outcome, "target_month": int(target_month),
            "estimator": estimator, "n": int(n), "rmse": float(rmse),
            "rmse_ci_low": float(ci[0]), "rmse_ci_high": float(ci[1]), "crps": float(crps),
            "powered": bool(powered),
        })

    # Complete-case: observed rows only, unweighted.
    if n_obs > 0:
        resid_cc = y[obs_idx] - median[obs_idx]
        ones = np.ones(n_obs, dtype=float)
        cc_rmse = _weighted_rmse(resid_cc, ones)
        cc_ci = _patient_bootstrap_rmse_ci(resid_cc, ones, patient_ids[obs_idx], rng, reps)
        cc_crps = float(study.quantile_crps(y[obs_idx], matrix[obs_idx], ones))
        _record("complete_case", cc_rmse, cc_ci, cc_crps, n_obs)
    else:
        _record("complete_case", float("nan"), (float("nan"), float("nan")), float("nan"), 0)

    # IPCW: observed rows, production analysis weights.
    if n_obs > 0:
        resid_ipcw = y[obs_idx] - median[obs_idx]
        w_obs = weight[obs_idx]
        ipcw_rmse = _weighted_rmse(resid_ipcw, w_obs)
        ipcw_ci = _patient_bootstrap_rmse_ci(resid_ipcw, w_obs, patient_ids[obs_idx], rng, reps)
        ipcw_crps = float(study.quantile_crps(y[obs_idx], matrix[obs_idx], w_obs))
        _record("ipcw", ipcw_rmse, ipcw_ci, ipcw_crps, n_obs)
    else:
        _record("ipcw", float("nan"), (float("nan"), float("nan")), float("nan"), 0)

    # Multiple imputation: pool across cfg.mi_imputations completed datasets; CI from a
    # patient-bootstrap of the first completed imputation (bounded and patient-clustered).
    n_total = int(len(cell))
    if n_obs >= MIN_CELL_SIZE and int((~observed).sum()) > 0:
        ones_full = np.ones(n_total, dtype=float)
        mi_rmse_draws: list[float] = []
        mi_crps_draws: list[float] = []
        first_completed: Any | None = None
        for draw in range(int(cfg.mi_imputations)):
            completed = _impute_censored_outcome(cell, y, observed, median, cfg.seed + draw)
            if first_completed is None:
                first_completed = completed
            mi_rmse_draws.append(_weighted_rmse(completed - median, ones_full))
            mi_crps_draws.append(float(study.quantile_crps(completed, matrix, ones_full)))
        mi_rmse = float(np.mean(mi_rmse_draws))
        mi_crps = float(np.mean(mi_crps_draws))
        mi_ci = _patient_bootstrap_rmse_ci(first_completed - median, ones_full, patient_ids, rng, reps)
        _record("multiple_imputation", mi_rmse, mi_ci, mi_crps, n_total)
    else:
        _record("multiple_imputation", float("nan"), (float("nan"), float("nan")), float("nan"), n_total)
    return rows


def _observed_vs_censored_smd(arm: str, outcome: str, target_month: int, cell: Any,
                              observed: Any) -> list[dict[str, Any]]:
    """Standardized mean difference of baseline covariates between observed and censored patients."""
    observed = np.asarray(observed).astype(bool)
    columns = [column for column in ATTRITION_SMD_COVARIATES if column in cell.columns]
    if not columns:
        return []
    L = np.column_stack([pd.to_numeric(cell[column], errors="coerce").to_numpy(float) for column in columns])
    smd = tte.standardized_mean_diff(L, observed.astype(int))
    n_obs = int(observed.sum())
    n_cens = int((~observed).sum())
    rows: list[dict[str, Any]] = []
    for column, value in zip(columns, np.asarray(smd, dtype=float)):
        abs_value = float(abs(value)) if np.isfinite(value) else float("nan")
        rows.append({
            "arm": arm, "outcome": outcome, "target_month": int(target_month),
            "covariate": column, "smd": float(value) if np.isfinite(value) else float("nan"),
            "abs_smd": abs_value,
            "mnar_flag": bool(np.isfinite(value) and abs_value > MNAR_SMD_FLAG),
            "n_observed": n_obs, "n_censored": n_cens,
        })
    return rows


def _surgical_ate_tipping(parent: Any, arm: str, target_month: int,
                          grid: Sequence[float]) -> tuple[list[float] | None, float, float]:
    """This surgical arm minus the OTHER surgical arm, stress-tested over the MNAR delta grid.

    RYGB-vs-sleeve is a CONTRAST, not a within-arm read, so it is computed on the shared surgical
    frame - where both procedures are present - rather than on the arm subset, which carries a
    single procedure and could never form a contrast. The sleeve row is therefore the mirror of the
    RYGB row by construction, and every field of an arm's row still describes that arm. Returns
    (curve, ate_at_0, delta*), all blank when the contrast cannot be formed.
    """
    blank: tuple[list[float] | None, float, float] = (None, float("nan"), float("nan"))
    if parent is None or "procedure" not in parent.columns:
        return blank
    cell = parent.loc[pd.to_numeric(parent["target_month"], errors="coerce") == int(target_month)]
    cell = cell.drop_duplicates("patient_id")
    indicator = (cell["procedure"].astype("string").str.lower() == arm).fillna(False).astype(int).to_numpy()
    if not (int(indicator.sum()) and int((indicator == 0).sum())):
        return blank
    matrix = _quantile_matrix(cell)
    curve, ate0, delta_star = tipping_point_effect_curve(
        cell["target_observed"].fillna(False).astype(bool).to_numpy(), matrix[:, 3], indicator,
        _cell_weight(cell), grid, pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float))
    return [float(value) for value in curve], float(ate0), float(delta_star)


def _tier1_attrition(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any]) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(cfg.seed)
    tipping_rows: list[dict[str, Any]] = []
    triage_rows: list[dict[str, Any]] = []
    smd_rows: list[dict[str, Any]] = []
    tipping_curves: list[dict[str, Any]] = []
    for arm, outcome, frame in _reporting_arm_frames(frames):
        is_surgery = arm != "incretin"
        # Uncopied identity view of the shared surgical frame; the ATE contrast needs both
        # procedures, which an arm subset by construction never has.
        parent = _arm_view(frames, "surgery", outcome, "")[0] if is_surgery else None
        grid = _delta_grid(outcome)
        for target_month, raw_cell in frame.groupby("target_month", sort=True):
            cell = raw_cell.drop_duplicates("patient_id").reset_index(drop=True)
            n = int(len(cell))
            if n < MIN_CELL_SIZE:
                continue
            observed = cell["target_observed"].fillna(False).astype(bool).to_numpy()
            y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
            matrix = _quantile_matrix(cell)
            weight = _cell_weight(cell)
            ess = _effective_sample_size(weight)
            powered = _is_powered(n, ess)
            n_censored = int((~observed).sum())

            margins, margin0, delta_star_crps, beats0 = tipping_point_crps_curve(
                observed, y, matrix, weight, grid)
            ate_curve, ate0, delta_star_ate = _surgical_ate_tipping(
                parent, arm, int(target_month), grid)

            tipping_rows.append({
                "arm": arm, "outcome": outcome, "target_month": int(target_month), "n": n,
                "n_censored": n_censored, "censored_fraction": float(n_censored / n) if n else np.nan,
                "model_beats_baseline_at_0": bool(beats0), "delta_star_crps": float(delta_star_crps),
                "is_surgery": bool(is_surgery), "ate_at_0": float(ate0),
                "delta_star_ate": float(delta_star_ate), "powered": bool(powered),
            })
            tipping_curves.append({
                "arm": arm, "outcome": outcome, "target_month": int(target_month),
                "grid": [float(value) for value in grid], "margins": [float(value) for value in margins],
                "ate_curve": ate_curve, "delta_star_crps": float(delta_star_crps),
                "delta_star_ate": float(delta_star_ate), "is_surgery": bool(is_surgery),
                "powered": bool(powered), "n": n, "n_censored": n_censored,
            })
            triage_rows.extend(_estimator_triage(
                arm, outcome, int(target_month), cell, observed, y, matrix, weight, cfg, rng, powered))
            smd_rows.extend(_observed_vs_censored_smd(arm, outcome, int(target_month), cell, observed))
    return tipping_rows, triage_rows, smd_rows, tipping_curves


# ----- T1.3 drug-level and procedure-level heterogeneity -----------------------------------
def _heterogeneity_metric_row(outcome: str, group_key: str, group_value: str, target_month: int,
                              cell: Any) -> dict[str, Any]:
    y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
    matrix = _quantile_matrix(cell)
    weight = _cell_weight(cell)
    metrics = _prognostic_metrics(y, matrix, weight)
    ess = _effective_sample_size(weight)
    n = int(len(cell))
    return {
        group_key: group_value, "outcome": outcome, "target_month": int(target_month), "n": n,
        "ess": ess, "crps": metrics["crps"], "rmse": metrics["rmse"], "mae": metrics["mae"],
        "coverage_80": metrics["coverage_80"], "cal_slope": metrics["cal_slope"],
        "powered": _is_powered(n, ess),
        "p_value_cov80": _coverage_pvalue(metrics["coverage_80"], ess, 0.80),
        "q_value_cov80": np.nan,
    }


def _tier1_drug_heterogeneity(cfg: SecondaryConfig,
                              frames: Mapping[tuple[str, str], Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (cohort, outcome), frame in sorted(frames.items()):
        if cohort != "incretin":
            continue
        observed = frame.loc[frame["target_observed"].fillna(False).astype(bool)].copy()
        if observed.empty or "index_ingredient" not in observed.columns:
            continue
        route = observed["index_route"] if "index_route" in observed.columns else pd.Series(
            [""] * len(observed), index=observed.index)
        observed = observed.assign(_group=[
            _drug_group(ingredient, route_value)
            for ingredient, route_value in zip(observed["index_ingredient"], route)
        ])
        observed = observed.loc[observed["_group"].notna()]
        for (group, target_month), cell in observed.groupby(["_group", "target_month"], sort=True):
            if cell.empty:
                continue
            record = _heterogeneity_metric_row(outcome, "drug_group", str(group), int(target_month), cell)
            # Spec 10's per-drug discrimination read. P(BMI < 35) is a BMI event, so the HbA1c rows
            # carry it as not estimable rather than as a second scale silently pooled into it.
            if outcome == "bmi":
                record["auroc_bmi35"], record["event_rate_bmi35"] = _cell_threshold_auroc(cell)
            rows.append(record)
    return rows


def _within_incretin_dr_read(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any],
                             audit: Any) -> dict[str, Any]:
    """Spec 10's "which agent for whom": the DR-learner read on ONE within-incretin contrast.

    Tirzepatide (A=1) against injectable semaglutide (A=0) on the held-out incretin BMI cell at the
    CATE horizon. Estimator, gating, and payload are the sleeve-vs-RYGB CATE's unchanged - the same
    doubly-robust pseudo-outcome, the same cross-fitted tau, the same positivity gate - so a cell
    without agent-choice overlap comes back not-estimable rather than as a curve fitted over a
    clipped propensity ratio. The nuisances CANNOT be borrowed from the target trial (that one is
    surgical and its arm is the procedure), so ps / pc / mu are cross-fit here over the same
    patient folds, on the modifier design that also serves as L.

    Nothing in this contrast is randomized. Spec 8.5 makes it secondary and exploratory, and every
    surface that renders it carries WITHIN_INCRETIN_CAVEAT: this is prognostic heterogeneity, not a
    causal treatment effect. Benefit is oriented as for the CATE, so a positive number means the
    tirzepatide forecast is the lower BMI.
    """
    frame = frames.get(("incretin", "bmi"))
    if frame is None or frame.empty or "index_ingredient" not in frame.columns:
        return {"status": "not_estimable", "n": 0, "calibration": [], "importance": [],
                "outcome": CATE_OUTCOME, "target_month": CATE_TARGET_MONTH,
                "absent_modifiers": _cate_absent_modifiers(audit),
                "treated_agent": WITHIN_INCRETIN_TREATED, "n_treated": 0,
                "comparator_agent": WITHIN_INCRETIN_COMPARATOR, "n_comparator": 0,
                "caveat": WITHIN_INCRETIN_CAVEAT,
                "skipped_reason": "no held-out incretin BMI predictions with a recorded index agent"}
    route = (frame["index_route"] if "index_route" in frame.columns
             else pd.Series([""] * len(frame), index=frame.index))
    agent = pd.Series([_drug_group(ingredient, value) for ingredient, value
                       in zip(frame["index_ingredient"], route)], index=frame.index)
    keep = (agent.isin([WITHIN_INCRETIN_TREATED, WITHIN_INCRETIN_COMPARATOR])
            & (pd.to_numeric(frame["target_month"], errors="coerce") == CATE_TARGET_MONTH))
    rows = frame.loc[keep].copy()          # the two-agent horizon cell only, never the whole frame
    rows["_agent"] = agent.loc[keep].to_numpy()
    rows = rows.drop_duplicates("patient_id").reset_index(drop=True)
    A = (rows["_agent"].to_numpy() == WITHIN_INCRETIN_TREATED).astype(int)
    Y = pd.to_numeric(rows["target_value"], errors="coerce").to_numpy(float)
    delta = rows["target_observed"].fillna(False).astype(bool).astype(int).to_numpy()
    patient_ids = rows["patient_id"].to_numpy()
    n, treated = int(A.size), int(A.sum())
    if n < CATE_MIN_N or treated in {0, n}:
        # One agent absent (or the cell too thin to cross-fit): skip the nuisances entirely and let
        # the CATE's own gate write the refusal. min_ps 0 is the literal truth for a one-armed cell.
        blank = np.zeros(n)
        read = _cate_dr_learner(cfg, rows, audit, Y, A, delta, np.full(n, 0.5), np.ones(n), blank,
                                blank, patient_ids,
                                {"positivity_fail": treated in {0, n}, "min_ps": 0.0,
                                 "ess_iptw": float("nan")})
    else:
        X, _groups, _absent = _cate_modifier_design(rows, audit)
        ps, degenerate = _crossfit_propensity(X, A, patient_ids, cfg.seed, cfg.nuisance_folds)
        gate = positivity_gate(ps, A, degenerate=degenerate)
        mu1, mu0, pc = _crossfit_mu_pc(X, A, Y, delta, patient_ids, cfg.seed, cfg.nuisance_folds)
        del X
        gc.collect()
        read = _cate_dr_learner(cfg, rows, audit, Y, A, delta, ps, pc, mu1, mu0, patient_ids, gate)
    # The CATE payload names its arms rygb/sleeve; this contrast has different ones, so they are
    # replaced rather than left to be misread.
    read.pop("n_rygb", None)
    read.pop("n_sleeve", None)
    read.update({"treated_agent": WITHIN_INCRETIN_TREATED, "n_treated": treated,
                 "comparator_agent": WITHIN_INCRETIN_COMPARATOR, "n_comparator": n - treated,
                 "caveat": WITHIN_INCRETIN_CAVEAT})
    return read


def _tier1_procedure_heterogeneity(cfg: SecondaryConfig,
                                   frames: Mapping[tuple[str, str], Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (cohort, outcome), frame in sorted(frames.items()):
        if cohort != "surgery":
            continue
        observed = frame.loc[frame["target_observed"].fillna(False).astype(bool)].copy()
        if observed.empty or "procedure" not in observed.columns:
            continue
        observed = observed.assign(_group=observed["procedure"].astype("string").str.lower())
        observed = observed.loc[observed["_group"].isin(["rygb", "sleeve"])]
        for (group, target_month), cell in observed.groupby(["_group", "target_month"], sort=True):
            if cell.empty:
                continue
            record = _heterogeneity_metric_row(outcome, "procedure", str(group), int(target_month), cell)
            record["sophia_context_rmse"] = float(SOPHIA_POOLED_RMSE.get(int(target_month), float("nan")))
            rows.append(record)
    return rows


# ----- Tier 1 CSV writer (disclosure suppression + structural-label restore) ---------------
def _rows_to_frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Any:
    frame = pd.DataFrame(list(rows))
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan if column not in {"skipped_reason"} else ""
    return frame.loc[:, list(columns)] if not frame.empty else pd.DataFrame(columns=list(columns))


def _write_tier1_csv(path: Path, frame: Any, columns: Sequence[str], count_columns: Sequence[str],
                     structural: Sequence[str] = (), header_note: str | None = None) -> None:
    """Write a Tier-1 CSV: suppress n < 11 cells, then restore structural label columns
    (target_month) that suppression blanks, exactly like the production TTE CSV path."""
    if frame is None or frame.empty:
        _write_csv(path, pd.DataFrame(columns=list(columns)), header_note=header_note)
        return
    display = frame.copy()
    for column in columns:
        if column not in display.columns:
            display[column] = np.nan
    display = display.loc[:, list(columns)]
    suppressed = suppress(display, list(count_columns))
    for column in structural:
        if column in frame.columns:
            suppressed[column] = frame[column].to_numpy()
    _write_csv(path, suppressed.loc[:, list(columns)], header_note=header_note)


# ----- Tier 1 figure pages (05 transportability, 06 attrition, 07 drug, 08 procedure) ------
def _tier1_note_page(figure: Any, message: str) -> Any:
    axis = figure.add_axes([0.08, 0.12, 0.84, 0.72])
    empty_panel(axis, message)
    return figure


def _tier1_payload(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    tier1 = data.get("tier1")
    if not isinstance(tier1, Mapping) or str(tier1.get("status")) != "done":
        return None
    return tier1


def _bar_by_group(axis: Any, labels: Sequence[str], values: Sequence[float], ylabel: str,
                  color: str) -> None:
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    if not labels or not np.isfinite(values).any():
        empty_panel(axis, "Not estimable")
        return
    x = np.arange(len(labels))
    axis.bar(x, np.where(np.isfinite(values), values, 0.0), color=color, alpha=0.85)
    axis.set_xticks(x)
    axis.set_xticklabels([str(item) for item in labels], rotation=30, ha="right", fontsize=6.6)
    axis.set_ylabel(ylabel, fontsize=8)


@register_page(5)
def _page_transportability(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                           title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier1 = _tier1_payload(data)
    if tier1 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    transport = tier1.get("transportability", {})
    table = pd.DataFrame(transport.get("rows", []))
    usable = transport.get("axes_usable", {})
    if table.empty:
        return _tier1_note_page(figure, "Transportability not estimable in this source")
    scored = table.loc[table["stratum"].astype(str) != ""].copy() if "stratum" in table else table.iloc[0:0]
    ax_table = figure.add_axes([0.06, 0.30, 0.52, 0.52])
    study.panel_label(ax_table, "A", "Held-out performance by geography / equity stratum")
    if scored.empty:
        empty_panel(ax_table, "No populated strata")
    else:
        ordered = scored.sort_values(["powered", "axis", "arm", "outcome", "target_month", "stratum"],
                                     ascending=[False, True, True, True, True, True])
        columns = ["axis", "arm", "outcome", "stratum", "target_month", "n", "crps", "coverage_80",
                   "cal_slope"]
        study.draw_compact_table(
            ax_table, ordered.loc[:, columns], columns,
            labels=["Axis", "Arm", "Outcome", "Stratum", "Target\nmonth", "N", "CRPS", "Cov 80",
                    "Cal\nslope"], max_rows=16)
    ax_bar = figure.add_axes([0.66, 0.30, 0.29, 0.52])
    study.panel_label(ax_bar, "B", "CRPS by stratum")
    # Disclosure: the bar plots a statistic rather than a count, so sub-11 strata are dropped from
    # it outright (the same n >= 11 guard _tier2_single_axis_slice applies before its bar).
    bar_source = scored.loc[pd.to_numeric(scored["n"], errors="coerce") >= MIN_CELL_SIZE]
    if not bar_source.empty:
        pick_axis = bar_source["axis"].iloc[0]
        subset = bar_source.loc[bar_source["axis"] == pick_axis]
        pick = subset.loc[(subset["target_month"] == subset["target_month"].iloc[0])
                          & (subset["arm"] == subset["arm"].iloc[0])]
        _bar_by_group(ax_bar, pick["stratum"].tolist(), pick["crps"].tolist(),
                      f"CRPS ({pick_axis}, {pick['arm'].iloc[0]})", study.PALETTE["blue"])
    else:
        empty_panel(ax_bar, "Not estimable")
    degraded = [axis for axis, ok in usable.items() if not ok]
    lines = [f"Axis population: " + ", ".join(
        f"{axis} {'usable' if ok else 'not populated'}" for axis, ok in sorted(usable.items()))]
    if degraded:
        lines.append("Degraded axes render as documented omissions in transportability.csv.")
    if transport.get("loso_note"):
        lines.append(str(transport["loso_note"]))
    lines.append(SUBGROUP_CAVEAT)
    figure.text(0.06, 0.24, textwrap.fill("  ".join(lines), 150), fontsize=7.0,
                color=study.PALETTE["muted"], va="top")
    return figure


@register_page(6)
def _page_attrition(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                    title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier1 = _tier1_payload(data)
    if tier1 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    attrition = tier1.get("attrition", {})
    curves = attrition.get("curves", [])
    triage = pd.DataFrame(attrition.get("triage", []))
    smd = pd.DataFrame(attrition.get("smd", []))

    ax_tip = figure.add_axes([0.07, 0.55, 0.40, 0.28])
    study.panel_label(ax_tip, "A", "Delta tipping-point (MNAR)")
    if curves:
        chosen = next((item for item in curves if item.get("ate_curve")), curves[0])
        grid = np.asarray(chosen["grid"], dtype=float)
        margins = np.asarray(chosen["margins"], dtype=float)
        ax_tip.axhline(0.0, color=study.PALETTE["muted"], lw=0.8)
        ax_tip.plot(grid, margins, color=study.PALETTE["blue"], marker="o", ms=3,
                    label="CRPS skill margin")
        if np.isfinite(chosen.get("delta_star_crps", float("nan"))):
            ax_tip.axvline(float(chosen["delta_star_crps"]), color=study.PALETTE["red"], ls=":",
                           lw=1.0, label="flip (CRPS)")
        if chosen.get("ate_curve"):
            twin = ax_tip.twinx()
            twin.plot(grid, np.asarray(chosen["ate_curve"], dtype=float),
                      color=study.PALETTE["orange"], marker="s", ms=3, label="ATE")
            twin.set_ylabel(f"ATE ({chosen['arm']} - other surgical arm)", fontsize=7,
                            color=study.PALETTE["orange"])
            if np.isfinite(chosen.get("delta_star_ate", float("nan"))):
                twin.axvline(float(chosen["delta_star_ate"]), color=study.PALETTE["orange"], ls="--", lw=0.9)
        ax_tip.set_xlabel("delta (imputed shift on censored)", fontsize=8)
        ax_tip.set_ylabel("baseline CRPS - model CRPS", fontsize=8)
        ax_tip.legend(fontsize=6.0, loc="best", frameon=False)
        ax_tip.set_title(f"{chosen['arm']}/{chosen['outcome']} at {chosen['target_month']} mo",
                         loc="right", fontsize=6.8, color=study.PALETTE["muted"])
    else:
        empty_panel(ax_tip, "No powered cells to stress-test")

    ax_triage = figure.add_axes([0.55, 0.52, 0.40, 0.31])
    study.panel_label(ax_triage, "B", "Estimator triage (CC vs IPCW vs MI)")
    if triage.empty:
        empty_panel(ax_triage, "Not estimable")
    else:
        columns = ["arm", "outcome", "target_month", "estimator", "n", "rmse", "rmse_ci_low", "rmse_ci_high"]
        study.draw_compact_table(
            ax_triage, triage.loc[:, columns], columns,
            labels=["Arm", "Outcome", "Target\nmonth", "Estimator", "N", "RMSE", "CI low", "CI high"],
            max_rows=14)

    ax_smd = figure.add_axes([0.07, 0.13, 0.55, 0.30])
    study.panel_label(ax_smd, "C", "Observed vs censored baseline SMD")
    _draw_smd_heatmap(ax_smd, smd)
    figure.text(0.66, 0.42, textwrap.fill(
        "Delta shifts the censored outcomes (imputed at the model median) to probe MNAR, per "
        "reporting arm: the flip point marks where that arm's model stops beating the "
        "population-mean baseline or, for a surgical arm, where its ATE against the other surgical "
        "arm crosses zero (a cross-arm contrast, so it is read off the shared surgical frame). "
        "|SMD| > 0.1 (panel C) flags observed-vs-censored imbalance and MNAR risk. "
        + SUBGROUP_CAVEAT, 60), fontsize=6.8, color=study.PALETTE["muted"], va="top")
    return figure


def _draw_smd_heatmap(axis: Any, smd: Any) -> None:
    if smd is None or smd.empty:
        empty_panel(axis, "Not estimable")
        return
    frame = smd.copy()
    frame["cell"] = frame["arm"].astype(str) + "/" + frame["outcome"].astype(str) + " " + \
        frame["target_month"].astype("Int64").astype(str) + "mo"
    pivot = frame.pivot_table(index="covariate", columns="cell", values="abs_smd", aggfunc="mean")
    if pivot.empty:
        empty_panel(axis, "Not estimable")
        return
    matrix = pivot.to_numpy(dtype=float)
    image = axis.imshow(np.where(np.isfinite(matrix), matrix, np.nan), aspect="auto",
                        cmap="magma", vmin=0.0, vmax=max(0.3, float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 0.3))
    axis.set_xticks(np.arange(pivot.shape[1]))
    axis.set_xticklabels([str(item) for item in pivot.columns], rotation=40, ha="right", fontsize=5.6)
    axis.set_yticks(np.arange(pivot.shape[0]))
    axis.set_yticklabels([str(item) for item in pivot.index], fontsize=5.8)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    colorbar.ax.tick_params(labelsize=5.6)
    colorbar.set_label("|SMD|", fontsize=6.5)


def _drug_horizon_rows(drug: Any) -> tuple[Any, int]:
    """Every agent's rows at ONE horizon: the CATE horizon when this source reaches it, else the
    earliest one it does.

    Pinning the page to a single horizon lets the table show every agent at a comparable point
    instead of an alphabetical truncation of the full agent-by-horizon grid, drops the now-constant
    target_month column (which buys the agent names the width they were being clipped at), and
    makes the bar panels read exactly the rows the table lists. heterogeneity_drug.csv keeps the
    complete grid, so nothing here is a silent cap.
    """
    months = pd.to_numeric(drug["target_month"], errors="coerce")
    available = sorted({int(value) for value in months.dropna().tolist()})
    if not available:
        return drug.iloc[0:0], 0
    month = CATE_TARGET_MONTH if CATE_TARGET_MONTH in available else available[0]
    return drug.loc[months == month].sort_values(["outcome", "drug_group"]), month


def _draw_drug_bars(axis: Any, pick: Any, column: str, xlabel: str, color: str,
                    labelled: bool, chance: float | None = None) -> None:
    """Horizontal bars over the drug groups. Long agent names read far better on the y axis than
    rotated under a narrow panel, and a shared row order lets the second panel go label-free.

    The checkpointed rows are unsuppressed (the CSV writer and the table renderer each suppress
    their own copy), so a bar panel reading them directly has to apply the disclosure floor itself:
    an under-floor agent gets a labelled row and an explicit "n < 11" in place of its bar, never a
    plotted value and never a silent gap.
    """
    values = pd.to_numeric(pick.get(column), errors="coerce").to_numpy(float) if len(pick) else np.zeros(0)
    counts = pd.to_numeric(pick.get("n"), errors="coerce").to_numpy(float) if len(pick) else np.zeros(0)
    disclosable = np.isfinite(counts) & (counts >= MIN_CELL_SIZE)
    values = np.where(disclosable, values, np.nan)
    if values.size == 0 or not np.isfinite(values).any():
        empty_panel(axis, "Not estimable")
        return
    y = np.arange(values.size)[::-1]
    axis.barh(y, values, color=color, alpha=0.88, height=0.66)
    axis.set_ylim(-0.6, values.size - 0.4)   # room for the suppression markers on the end rows
    axis.set_yticks(y)
    axis.set_yticklabels([DRUG_GROUP_LABELS.get(str(name), str(name))
                          for name in pick["drug_group"]] if labelled else [""] * values.size,
                         fontsize=6.8)
    if chance is not None:
        axis.axvline(chance, color=study.PALETTE["muted"], lw=0.9, ls=":")
        axis.set_xlim(min(0.4, float(np.nanmin(values))), 1.0)
    for row in np.where(~disclosable)[0]:
        # On the paper ground, so a marker that lands on the AUROC chance line stays legible.
        axis.text(0.02, y[row], "n < 11", transform=axis.get_yaxis_transform(), ha="left",
                  va="center", fontsize=6.0, color=study.PALETTE["muted"],
                  bbox={"facecolor": study.PALETTE["paper"], "edgecolor": "none", "pad": 0.8})
    axis.set_xlabel(xlabel, fontsize=8)


def _which_incretin_summary_lines(read: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The within-incretin readout, in the order the summary column prints it."""
    return [
        ("Agent scored", DRUG_GROUP_LABELS.get(str(read.get("treated_agent")), "?")),
        ("Comparator", DRUG_GROUP_LABELS.get(str(read.get("comparator_agent")), "?")),
        ("N on agent", study.display_count(int(read.get("n_treated") or 0))),
        ("N on comparator", study.display_count(int(read.get("n_comparator") or 0))),
        ("RATE", _cate_number(read.get("rate"), 3)),
        ("c-for-benefit", _cate_number(read.get("c_for_benefit"), 3)),
        ("IPTW effective N", _cate_number(read.get("ess_iptw"), 1)),
    ]


@register_page(7)
def _page_drug_heterogeneity(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                             title: str, subtitle: str) -> Any:
    """Spec 10's which-incretin science page: per-agent calibrated CRPS and P(BMI < 35)
    discrimination (panels A-C), then the "which agent for whom" DR-learner read on the one
    within-incretin contrast with a defensible comparator (panels D-F). The contrast is not
    randomized, so panel F carries the prognostic-not-causal gate and a cell without agent-choice
    overlap renders not-estimable exactly as the CATE page does."""
    figure = secondary_new_page(number, title, subtitle)
    tier1 = _tier1_payload(data)
    if tier1 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    heterogeneity = tier1.get("heterogeneity", {})
    drug = pd.DataFrame(heterogeneity.get("drug", []))
    if drug.empty:
        return _tier1_note_page(figure, "Incretin drug groups not populated in this source")
    horizon, month = _drug_horizon_rows(drug)
    ax_table = figure.add_axes([0.055, 0.455, 0.505, 0.355])
    study.panel_label(ax_table, "A", f"Calibrated performance by agent at {month} months")
    columns = ["drug_group", "outcome", "n", "crps", "rmse", "coverage_80", "auroc_bmi35"]
    labelled = horizon.copy()
    labelled["drug_group"] = labelled["drug_group"].map(
        lambda value: DRUG_GROUP_LABELS.get(str(value), str(value)))
    study.draw_compact_table(
        ax_table, labelled.loc[:, columns], columns,
        labels=["Agent", "Outcome", "N", "CRPS", "RMSE", "Cov 80", "AUROC\nBMI<35"], max_rows=12)
    pick = horizon.loc[horizon["outcome"].astype(str) == "bmi"]
    ax_rmse = figure.add_axes([0.620, 0.455, 0.150, 0.355])
    _narrow_panel_label(ax_rmse, "B", f"RMSE at {month} mo")
    _draw_drug_bars(ax_rmse, pick, "rmse", "RMSE, BMI (kg/m2)", study.PALETTE["green"], True)
    ax_auroc = figure.add_axes([0.800, 0.455, 0.150, 0.355])
    _narrow_panel_label(ax_auroc, "C", f"AUROC at {month} mo")
    _draw_drug_bars(ax_auroc, pick, "auroc_bmi35", "P(BMI < 35) AUROC", study.PALETTE["blue"],
                    False, chance=0.5)

    read = heterogeneity.get("which_incretin") or {}
    ax_modifiers = figure.add_axes([0.055, 0.135, 0.245, 0.225])
    _narrow_panel_label(ax_modifiers, "D", "Agent-contrast modifiers")
    if str(read.get("status")) == "done":
        _draw_modifier_importance(ax_modifiers, read.get("importance"), 6)
    else:
        empty_panel(ax_modifiers, textwrap.fill(
            f"Not estimable: {read.get('skipped_reason')}" if read.get("skipped_reason")
            else "Within-incretin contrast not populated in this source", 34))
    _draw_summary_column(figure.add_axes([0.375, 0.135, 0.185, 0.225]), "E",
                         "Which agent for whom", _which_incretin_summary_lines(read),
                         "Benefit is BMI reduction under the scored\nagent rather than the "
                         "comparator.")
    # The caveat is read off the payload, not the constant, so the page states what the archived
    # estimate itself carries - on the not-estimable branches as much as the estimable one.
    _cate_honesty_panel(figure.add_axes([0.630, 0.135, 0.320, 0.225]), "F", read, 66,
                        caveat=str(read.get("caveat", WITHIN_INCRETIN_CAVEAT)),
                        structural="prescriber_or_formulary", tail="")
    figure.text(0.055, 0.088, textwrap.fill(
        "Groups derived from index_ingredient (+ index_route for oral vs injectable semaglutide); "
        "heterogeneity_drug.csv carries every agent at every horizon. CRPS, RMSE, and coverage are "
        "IPCW-weighted over held-out observed rows of the calibrated selected model, and the AUROC "
        "reads P(BMI < 35) off the same calibrated quantile ladder the headline per-procedure page "
        "uses. " + SUBGROUP_CAVEAT, 172),
        fontsize=6.8, color=study.PALETTE["muted"], va="top")
    return figure


@register_page(8)
def _page_procedure_heterogeneity(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                                  title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier1 = _tier1_payload(data)
    if tier1 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    heterogeneity = tier1.get("heterogeneity", {})
    procedure = pd.DataFrame(heterogeneity.get("procedure", []))
    if procedure.empty:
        return _tier1_note_page(figure, "Surgery procedure groups not populated in this source")
    ax_rmse = figure.add_axes([0.07, 0.30, 0.52, 0.52])
    study.panel_label(ax_rmse, "A", "RYGB vs sleeve RMSE by horizon vs SOPHIA context")
    bmi = procedure.loc[procedure["outcome"] == "bmi"] if "outcome" in procedure else procedure.iloc[0:0]
    plotted = False
    colors = {"rygb": study.PALETTE["orange"], "sleeve": study.PALETTE["blue"]}
    for group in ("rygb", "sleeve"):
        subset = bmi.loc[bmi["procedure"] == group].sort_values("target_month")
        if subset.empty:
            continue
        ax_rmse.plot(subset["target_month"].to_numpy(float), subset["rmse"].to_numpy(float),
                     marker="o", ms=4, color=colors[group], label=f"{group} (this source)")
        plotted = True
    sophia_procedure = heterogeneity.get("sophia_procedure", SOPHIA_PROCEDURE_RMSE_60)
    for group, value in sophia_procedure.items():
        ax_rmse.axhline(float(value), color=colors.get(group, study.PALETTE["muted"]), ls=":", lw=1.0)
        ax_rmse.text(0.98, float(value), f"SOPHIA {group} 60mo {value}", transform=ax_rmse.get_yaxis_transform(),
                     ha="right", va="bottom", fontsize=5.8, color=colors.get(group, study.PALETTE["muted"]))
    sophia_pooled = heterogeneity.get("sophia_pooled", SOPHIA_POOLED_RMSE)
    pooled_x = sorted(int(k) for k in sophia_pooled)
    ax_rmse.plot(pooled_x, [sophia_pooled[k] for k in pooled_x], ls=":", marker="D", ms=3,
                 color=study.PALETTE["muted"], label="SOPHIA pooled (context)")
    if not plotted:
        empty_panel(ax_rmse, "No powered BMI horizons on this source")
    else:
        ax_rmse.set_xlabel("Target month", fontsize=8)
        ax_rmse.set_ylabel("BMI RMSE (kg/m2)", fontsize=8)
        ax_rmse.legend(fontsize=6.0, loc="best", frameon=False)
    ax_table = figure.add_axes([0.64, 0.30, 0.31, 0.52])
    study.panel_label(ax_table, "B", "Procedure metrics")
    columns = ["procedure", "outcome", "target_month", "n", "rmse", "crps", "coverage_80"]
    study.draw_compact_table(
        ax_table, procedure.sort_values(["outcome", "procedure", "target_month"]).loc[:, columns], columns,
        labels=["Procedure", "Outcome", "Target\nmonth", "N", "RMSE", "CRPS", "Cov 80"], max_rows=16)
    figure.text(0.07, 0.24, textwrap.fill(
        "SOPHIA per-procedure (RYGB 4.5, sleeve 5.7 at 60 mo) and pooled (3.7/4.2/4.7 at 12/24/60 mo) "
        "lines are drawn as CONTEXT only. The 60-month horizon is not estimable on this source, so "
        "these are not matched comparators. " + SUBGROUP_CAVEAT, 150), fontsize=7.0,
        color=study.PALETTE["muted"], va="top")
    return figure


# --------------------------------------------------------------------------------------------
# Tier 2 - subgroup menu (descriptive audit, NOT effect-modification claims). Sixteen axes span
# demographics, glycemia, comorbidity burden, concomitant medications, clinical obesity class,
# and calendar era. Every cell is scored with the SAME held-out prognostic machinery Tier 1 uses
# (selected candidate at origin 0 on HELD_OUT_SPLITS, target_observed, joined to the covariate
# frame on patient_id), streamed one origin-0 partition at a time. Fairness axes additionally
# carry a calibration-equity view: coverage and calibration-slope gaps vs the pooled all-patients
# model for the same (arm, outcome, horizon). Suppression of n < 11 is the single hardest
# invariant here (many small cells), enforced on the CSV via suppress(...) and on the pages via
# study.draw_compact_table.
# --------------------------------------------------------------------------------------------
SUBGROUPS_TIER2_COLUMNS = (
    "arm", "outcome", "target_month", "axis", "subgroup", "n", "ess", "crps", "coverage_80",
    "calibration_slope", "calibration_intercept", "coverage_gap_vs_pooled",
    "calibration_slope_gap_vs_pooled", "powered", "p_value_cov80", "q_value_cov80",
    "skipped_reason",
)

# Fairness axes get the calibration-equity gap view; the other axes do not.
FAIRNESS_AXES = ("sex", "race", "ethnicity", "age_band")


# ----- subgroup label helpers (deterministic, NaN-preserving) ------------------------------
def _tier2_categorical_labels(values: Any) -> Any:
    """As-is categorical subgroup labels (sex/race/ethnicity); blank/unknown/missing -> NA."""
    text = pd.Series(values).astype("string").str.strip()
    lowered = text.str.lower()
    blank = lowered.isin(["", "nan", "none", "unknown", "<missing>", "not_applicable"])
    return text.where(text.notna() & ~blank.fillna(True))


def _tier2_flag_series(values: Any) -> Any:
    """Boolean Series from a 0/1, boolean, or yes/no flag column; NA where unpopulated."""
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    result[numeric > 0] = True
    result[numeric == 0] = False
    text = series.astype("string").str.strip().str.lower()
    result[text.isin(["yes", "true", "y", "t"])] = True
    result[text.isin(["no", "false", "n", "f"])] = False
    return result


def _tier2_flag_labels(values: Any, positive: str, negative: str) -> Any:
    """Two-level flag labels (e.g. yes/no, on/off); NA preserved."""
    return _tier2_flag_series(values).map({True: positive, False: negative}).astype("string")


def _tier2_age_band_labels(values: Any) -> Any:
    """Age bands from age_at_index: <40, 40-54, 55-64, 65+."""
    age = pd.to_numeric(pd.Series(values), errors="coerce")
    result = pd.Series(pd.NA, index=age.index, dtype="string")
    result[age < 40] = "<40"
    result[(age >= 40) & (age < 55)] = "40-54"
    result[(age >= 55) & (age < 65)] = "55-64"
    result[age >= 65] = "65+"
    return result


def _tier2_hba1c_band_labels(values: Any) -> Any:
    """Baseline-HbA1c bands: <5.7, 5.7-6.4, >=6.5."""
    hba1c = pd.to_numeric(pd.Series(values), errors="coerce")
    result = pd.Series(pd.NA, index=hba1c.index, dtype="string")
    result[hba1c < 5.7] = "<5.7"
    result[(hba1c >= 5.7) & (hba1c < 6.5)] = "5.7-6.4"
    result[hba1c >= 6.5] = ">=6.5"
    return result


def _tier2_obesity_class_labels(values: Any) -> Any:
    """Fixed clinical obesity classes (spec Section 6): 35-40, 40-50, 50+ (super-obesity). BMI < 35
    is outside the requested clinical classes and maps to NA - these are fixed clinical bins, not
    data-driven tertiles, for interpretability and literature comparability."""
    bmi = pd.to_numeric(pd.Series(values), errors="coerce")
    result = pd.Series(pd.NA, index=bmi.index, dtype="string")
    result[(bmi >= 35) & (bmi < 40)] = "35-40"
    result[(bmi >= 40) & (bmi < 50)] = "40-50"
    result[bmi >= 50] = "50+"
    return result


def _tier2_calendar_era_labels(values: Any) -> Any:
    """Calendar era from index_year: pre-2021 vs 2021+."""
    year = pd.to_numeric(pd.Series(values), errors="coerce")
    result = pd.Series(pd.NA, index=year.index, dtype="string")
    result[year < 2021] = "pre-2021"
    result[year >= 2021] = "2021+"
    return result


def _tier2_covid_flag_labels(values: Any) -> Any:
    """COVID index flag: index_year in {2020, 2021} -> yes, otherwise no; NA where year missing."""
    year = pd.to_numeric(pd.Series(values), errors="coerce")
    result = pd.Series(pd.NA, index=year.index, dtype="string")
    covid = year.isin([2020, 2021])
    result[year.notna() & covid] = "yes"
    result[year.notna() & ~covid] = "no"
    return result


def _tier2_comorbidity_count_labels(frame: Any) -> Any:
    """Comorbidity count (0/1/2/3) = hypertension + dyslipidemia + osa; NA if any flag missing."""
    total = None
    for column in ("hypertension", "dyslipidemia", "osa"):
        flag = _tier2_flag_series(frame[column]).map({True: 1, False: 0}).astype("Int64")
        total = flag if total is None else total + flag
    if total is None:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    return total.astype("string")


def _tier2_axis_specs() -> list[dict[str, Any]]:
    """The sixteen Tier-2 axes: name, required source columns (audited), the reporting arms they
    apply to, whether they are a fairness axis, a labeler(frame) -> subgroup Series, and the
    not-populated message. biguanide/sglt2 are the incretin arm only (concomitant meds)."""
    both = {arm for arm, _cohort, _procedure in PROCEDURE_AUROC_ARMS}
    incretin_only = {"incretin"}
    return [
        {"axis": "sex", "columns": ("sex",), "arms": both, "fairness": True,
         "labeler": lambda f: _tier2_categorical_labels(f["sex"]),
         "message": "sex not populated in this source"},
        {"axis": "race", "columns": ("race",), "arms": both, "fairness": True,
         "labeler": lambda f: _tier2_categorical_labels(f["race"]),
         "message": "race not populated in this source"},
        {"axis": "ethnicity", "columns": ("ethnicity",), "arms": both, "fairness": True,
         "labeler": lambda f: _tier2_categorical_labels(f["ethnicity"]),
         "message": "ethnicity not populated in this source"},
        {"axis": "age_band", "columns": ("age_at_index",), "arms": both, "fairness": True,
         "labeler": lambda f: _tier2_age_band_labels(f["age_at_index"]),
         "message": "age_at_index not populated in this source"},
        {"axis": "diabetes_flag", "columns": ("diabetes_flag",), "arms": both, "fairness": False,
         "labeler": lambda f: _tier2_flag_labels(f["diabetes_flag"], "yes", "no"),
         "message": "diabetes_flag not populated in this source"},
        {"axis": "hba1c_band", "columns": ("baseline_hba1c",), "arms": both, "fairness": False,
         "labeler": lambda f: _tier2_hba1c_band_labels(f["baseline_hba1c"]),
         "message": "baseline_hba1c not populated in this source"},
        {"axis": "insulin", "columns": ("insulin",), "arms": both, "fairness": False,
         "labeler": lambda f: _tier2_flag_labels(f["insulin"], "on", "off"),
         "message": "insulin not populated in this source"},
        {"axis": "hypertension", "columns": ("hypertension",), "arms": both, "fairness": False,
         "labeler": lambda f: _tier2_flag_labels(f["hypertension"], "yes", "no"),
         "message": "hypertension not populated in this source"},
        {"axis": "dyslipidemia", "columns": ("dyslipidemia",), "arms": both, "fairness": False,
         "labeler": lambda f: _tier2_flag_labels(f["dyslipidemia"], "yes", "no"),
         "message": "dyslipidemia not populated in this source"},
        {"axis": "osa", "columns": ("osa",), "arms": both, "fairness": False,
         "labeler": lambda f: _tier2_flag_labels(f["osa"], "yes", "no"),
         "message": "osa not populated in this source"},
        {"axis": "comorbidity_count", "columns": ("hypertension", "dyslipidemia", "osa"),
         "arms": both, "fairness": False, "labeler": _tier2_comorbidity_count_labels,
         "message": "comorbidity flags not populated in this source"},
        {"axis": "biguanide", "columns": ("biguanide",), "arms": incretin_only, "fairness": False,
         "labeler": lambda f: _tier2_flag_labels(f["biguanide"], "yes", "no"),
         "message": "biguanide not populated in this source"},
        {"axis": "sglt2", "columns": ("sglt2",), "arms": incretin_only, "fairness": False,
         "labeler": lambda f: _tier2_flag_labels(f["sglt2"], "yes", "no"),
         "message": "sglt2 not populated in this source"},
        {"axis": "obesity_class", "columns": ("baseline_bmi",), "arms": both, "fairness": False,
         "labeler": lambda f: _tier2_obesity_class_labels(f["baseline_bmi"]),
         "message": "baseline_bmi not populated in this source"},
        {"axis": "calendar_era", "columns": ("index_year",), "arms": both, "fairness": False,
         "labeler": lambda f: _tier2_calendar_era_labels(f["index_year"]),
         "message": "index_year not populated in this source"},
        {"axis": "covid_flag", "columns": ("index_year",), "arms": both, "fairness": False,
         "labeler": lambda f: _tier2_covid_flag_labels(f["index_year"]),
         "message": "index_year not populated in this source"},
    ]


def _tier2_skip_row(arm: str, outcome: str, axis: str, reason: str) -> dict[str, Any]:
    """A not-populated axis row: labels kept, metrics NA, carrying the skipped_reason (n = 0 so it
    survives disclosure suppression, which only blanks 0 < n < 11)."""
    return {
        "arm": arm, "outcome": outcome, "target_month": np.nan, "axis": axis,
        "subgroup": "", "n": 0, "ess": np.nan, "crps": np.nan, "coverage_80": np.nan,
        "calibration_slope": np.nan, "calibration_intercept": np.nan,
        "coverage_gap_vs_pooled": np.nan, "calibration_slope_gap_vs_pooled": np.nan,
        "powered": False, "p_value_cov80": np.nan, "q_value_cov80": np.nan,
        "skipped_reason": reason,
    }


def _tier2_pooled_metrics(frames: Mapping[tuple[str, str], Any]) -> dict[tuple[str, str, int], dict[str, float]]:
    """Pooled (all-patients) held-out metrics per (arm, outcome, target_month) - the reference the
    fairness calibration-equity gaps are measured against, so each arm is judged against its OWN
    pooled model rather than against a cohort its subgroups are not drawn from."""
    pooled: dict[tuple[str, str, int], dict[str, float]] = {}
    for arm, outcome, frame in _reporting_arm_frames(frames):
        observed = frame.loc[frame["target_observed"].fillna(False).astype(bool)]
        if observed.empty:
            continue
        for target_month, cell in observed.groupby("target_month", sort=True):
            y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
            matrix = _quantile_matrix(cell)
            weight = _cell_weight(cell)
            metrics = _prognostic_metrics(y, matrix, weight)
            pooled[(arm, outcome, int(target_month))] = {
                "coverage_80": float(metrics["coverage_80"]),
                "cal_slope": float(metrics["cal_slope"]),
                "n": int(len(cell)),
                "ess": _effective_sample_size(weight),
            }
    return pooled


def _tier2_subgroup_cells(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any], audit: Any,
                          pooled: Mapping[tuple[str, str, int], Mapping[str, float]]) -> tuple[
                              list[dict[str, Any]], dict[str, bool]]:
    """Per (arm, outcome, target_month) x subgroup-cell CRPS / 80 coverage / calibration
    slope+intercept / n / ESS across every usable axis, with fairness gaps vs the pooled model.
    Each axis is gated through the column-population audit and degrades to a skipped_reason row
    when its column(s) are absent or not usable. Never crashes on a missing/empty covariate."""
    specs = _tier2_axis_specs()
    axes_usable: dict[str, bool] = {spec["axis"]: False for spec in specs}
    rows: list[dict[str, Any]] = []
    for arm, outcome, frame in _reporting_arm_frames(frames):
        observed = frame.loc[frame["target_observed"].fillna(False).astype(bool)].copy()
        for spec in specs:
            axis = spec["axis"]
            if arm not in spec["arms"]:
                continue
            present = all(column in observed.columns for column in spec["columns"])
            if not present:
                rows.append(_tier2_skip_row(arm, outcome, axis,
                                            f"{axis}: required columns absent from source"))
                continue
            usable = audit is not None and all(audit_usable(audit, column) for column in spec["columns"])
            axes_usable[axis] = bool(axes_usable[axis] or usable)
            if not usable:
                rows.append(_tier2_skip_row(arm, outcome, axis, spec["message"]))
                continue
            if observed.empty:
                continue
            work = observed.assign(_subgroup=spec["labeler"](observed))
            work["_subgroup"] = work["_subgroup"].astype("string")
            work = work.loc[work["_subgroup"].notna() & (work["_subgroup"].str.len() > 0)]
            if work.empty:
                continue
            for (target_month, subgroup), cell in work.groupby(["target_month", "_subgroup"], sort=True):
                if cell.empty:
                    continue
                y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
                matrix = _quantile_matrix(cell)
                weight = _cell_weight(cell)
                metrics = _prognostic_metrics(y, matrix, weight)
                ess = _effective_sample_size(weight)
                n = int(len(cell))
                coverage_gap = np.nan
                slope_gap = np.nan
                if spec["fairness"]:
                    reference = pooled.get((arm, outcome, int(target_month)))
                    if reference is not None:
                        if np.isfinite(metrics["coverage_80"]) and np.isfinite(reference["coverage_80"]):
                            coverage_gap = float(metrics["coverage_80"] - reference["coverage_80"])
                        if np.isfinite(metrics["cal_slope"]) and np.isfinite(reference["cal_slope"]):
                            slope_gap = float(metrics["cal_slope"] - reference["cal_slope"])
                rows.append({
                    "arm": arm, "outcome": outcome, "target_month": int(target_month),
                    "axis": axis, "subgroup": str(subgroup), "n": n, "ess": ess,
                    "crps": metrics["crps"], "coverage_80": metrics["coverage_80"],
                    "calibration_slope": metrics["cal_slope"],
                    "calibration_intercept": metrics["cal_intercept"],
                    "coverage_gap_vs_pooled": coverage_gap,
                    "calibration_slope_gap_vs_pooled": slope_gap,
                    "powered": _is_powered(n, ess),
                    "p_value_cov80": _coverage_pvalue(metrics["coverage_80"], ess, 0.80),
                    "q_value_cov80": np.nan, "skipped_reason": "",
                })
    return rows, axes_usable


# ----- Tier 2 figure pages (09 fairness, 10 clinical, 11 obesity class + calendar era) ------
_TIER2_TABLE_COLUMNS = ["axis", "arm", "outcome", "subgroup", "target_month", "n", "crps",
                        "coverage_80", "calibration_slope", "powered"]
_TIER2_TABLE_LABELS = ["Axis", "Arm", "Outcome", "Subgroup", "Target\nmonth", "N", "CRPS",
                       "Cov 80", "Cal\nslope", "Powered"]
_TIER2_COMPACT_COLUMNS = ["arm", "outcome", "subgroup", "target_month", "n", "crps",
                          "coverage_80", "calibration_slope"]
_TIER2_COMPACT_LABELS = ["Arm", "Outcome", "Subgroup", "Target\nmonth", "N", "CRPS", "Cov 80",
                         "Cal\nslope"]


def _tier2_payload(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    tier2 = data.get("tier2")
    if not isinstance(tier2, Mapping) or str(tier2.get("status")) != "done":
        return None
    return tier2


def _tier2_scored_frame(tier2: Mapping[str, Any]) -> Any:
    """The scored (non-skipped) subgroup cells as a frame, ready for study.draw_compact_table
    (which re-applies the n < 11 display suppression, rendering small cells as '<11')."""
    frame = pd.DataFrame(tier2.get("rows", []))
    if frame.empty:
        return frame
    if "skipped_reason" in frame.columns:
        frame = frame.loc[frame["skipped_reason"].astype(str).fillna("") == ""]
    if "subgroup" in frame.columns:
        frame = frame.loc[frame["subgroup"].astype(str) != ""]
    return frame.reset_index(drop=True)


def _arm_balanced_head(ordered: Any, max_rows: int) -> Any:
    """The top ``max_rows`` rows shared out ACROSS the arms, in the caller's existing row order.

    A table that sorts by arm and then truncates fills up with whichever arm sorts first, so a
    three-arm re-cut would render as one arm. Taking a per-arm head first keeps every arm visible
    (thin arms included - they render as '<11', which is the honest read). ``groupby.head`` is a
    row filter, so the caller's powered-first ordering and determinism both survive it.
    """
    if ordered is None or ordered.empty or "arm" not in ordered.columns:
        return ordered
    arms = max(1, int(ordered["arm"].nunique()))
    return ordered.groupby("arm", sort=False).head(max(1, int(max_rows) // arms))


def _tier2_axis_table(axis_obj: Any, frame: Any, axes: Sequence[str], max_rows: int = 14,
                      columns: Sequence[str] | None = None, labels: Sequence[str] | None = None) -> None:
    """Compact per-cell table for one axis group; powered cells sort first, small cells show '<11'."""
    columns = list(columns) if columns is not None else _TIER2_TABLE_COLUMNS
    labels = list(labels) if labels is not None else _TIER2_TABLE_LABELS
    subset = frame.loc[frame["axis"].isin(list(axes))] if frame is not None and not frame.empty else frame
    if subset is None or subset.empty:
        empty_panel(axis_obj, "Not populated in this source")
        return
    ordered = subset.sort_values(
        ["powered", "axis", "arm", "outcome", "target_month", "subgroup"],
        ascending=[False, True, True, True, True, True])
    ordered = _arm_balanced_head(ordered, max_rows)
    study.draw_compact_table(axis_obj, ordered.loc[:, columns], columns, labels=labels, max_rows=max_rows)


def _tier2_single_axis_slice(frame: Any, axis_name: str) -> Any:
    """The disclosure-safe (n >= 11) (arm, outcome, target_month) group with the most subgroups
    for one axis - the slice a bar panel plots so no suppressed-cell statistic is drawn."""
    subset = frame.loc[frame["axis"] == axis_name] if frame is not None and not frame.empty else frame
    if subset is None or subset.empty:
        return pd.DataFrame()
    subset = subset.loc[pd.to_numeric(subset["n"], errors="coerce") >= MIN_CELL_SIZE]
    if subset.empty:
        return subset
    counts = subset.groupby(["arm", "outcome", "target_month"], sort=False).size()
    arm_v, outcome_v, month_v = counts.idxmax()
    chosen = subset.loc[(subset["arm"] == arm_v) & (subset["outcome"] == outcome_v)
                        & (subset["target_month"] == month_v)]
    return chosen.sort_values("subgroup")


def _tier2_fairness_gap_slice(fair_frame: Any) -> Any:
    """The disclosure-safe fairness (axis, arm, outcome, target_month) group with the most
    gap-scored cells, for the coverage-gap bar."""
    if fair_frame is None or fair_frame.empty:
        return pd.DataFrame()
    subset = fair_frame.loc[fair_frame["coverage_gap_vs_pooled"].notna()
                            & (pd.to_numeric(fair_frame["n"], errors="coerce") >= MIN_CELL_SIZE)]
    if subset.empty:
        return subset
    counts = subset.groupby(["axis", "arm", "outcome", "target_month"], sort=False).size()
    axis_v, arm_v, outcome_v, month_v = counts.idxmax()
    chosen = subset.loc[(subset["axis"] == axis_v) & (subset["arm"] == arm_v)
                        & (subset["outcome"] == outcome_v) & (subset["target_month"] == month_v)]
    return chosen.sort_values("subgroup")


def _tier2_gap_bar(axis_obj: Any, subset: Any, value_column: str, ylabel: str) -> None:
    """Signed gap bar centered at zero (blue >= 0, red < 0)."""
    values = pd.to_numeric(subset[value_column], errors="coerce").to_numpy(float) \
        if subset is not None and not subset.empty else np.asarray([])
    if not values.size or not np.isfinite(values).any():
        empty_panel(axis_obj, "No pooled reference")
        return
    labels = subset["subgroup"].astype(str).tolist()
    x = np.arange(len(labels))
    colors = [study.PALETTE["blue"] if (not np.isfinite(value) or value >= 0) else study.PALETTE["red"]
              for value in values]
    axis_obj.axhline(0.0, color=study.PALETTE["muted"], lw=0.8)
    axis_obj.bar(x, np.where(np.isfinite(values), values, 0.0), color=colors, alpha=0.85)
    axis_obj.set_xticks(x)
    axis_obj.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.6)
    axis_obj.set_ylabel(ylabel, fontsize=8)


def _tier2_gap_table(axis_obj: Any, fair_frame: Any) -> None:
    """Calibration-equity table: coverage and calibration-slope gaps vs the pooled model."""
    subset = fair_frame.loc[fair_frame["coverage_gap_vs_pooled"].notna()
                            | fair_frame["calibration_slope_gap_vs_pooled"].notna()] \
        if fair_frame is not None and not fair_frame.empty else fair_frame
    if subset is None or subset.empty:
        empty_panel(axis_obj, "No pooled reference")
        return
    ordered = subset.sort_values(["axis", "arm", "outcome", "target_month", "subgroup"])
    columns = ["axis", "arm", "subgroup", "outcome", "target_month", "n", "coverage_gap_vs_pooled",
               "calibration_slope_gap_vs_pooled"]
    study.draw_compact_table(
        axis_obj, ordered.loc[:, columns], columns,
        labels=["Axis", "Arm", "Subgroup", "Outcome", "Target\nmonth", "N", "Cov 80\ngap",
                "Cal slope\ngap"], max_rows=14)


def _tier2_axis_population_note(usable: Mapping[str, bool], axes: Sequence[str]) -> str:
    return "Axis population: " + ", ".join(
        f"{axis} {'usable' if usable.get(axis, False) else 'not populated'}" for axis in axes) + "."


@register_page(9)
def _page_fairness_subgroups(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                             title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier2 = _tier2_payload(data)
    if tier2 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    frame = _tier2_scored_frame(tier2)
    fair_frame = frame.loc[frame["axis"].isin(list(FAIRNESS_AXES))] if not frame.empty else frame
    if fair_frame is None or fair_frame.empty:
        return _tier1_note_page(figure, "Fairness axes not populated in this source")
    ax_table = figure.add_axes([0.06, 0.50, 0.88, 0.32])
    study.panel_label(ax_table, "A", "Held-out performance by fairness stratum")
    _tier2_axis_table(ax_table, fair_frame, FAIRNESS_AXES, max_rows=12)
    ax_bar = figure.add_axes([0.07, 0.15, 0.36, 0.25])
    study.panel_label(ax_bar, "B", "80% coverage gap vs pooled")
    _tier2_gap_bar(ax_bar, _tier2_fairness_gap_slice(fair_frame), "coverage_gap_vs_pooled",
                   "Coverage - pooled")
    ax_gap = figure.add_axes([0.52, 0.14, 0.43, 0.27])
    study.panel_label(ax_gap, "C", "Calibration-equity: gaps vs pooled model")
    _tier2_gap_table(ax_gap, fair_frame)
    usable = tier2.get("axes_usable", {})
    degraded = [axis for axis in FAIRNESS_AXES if not usable.get(axis, False)]
    note = _tier2_axis_population_note(usable, FAIRNESS_AXES) + " "
    if degraded:
        note += "Not-populated axes are documented omissions in subgroups_tier2.csv. "
    note += SUBGROUP_CAVEAT
    figure.text(0.06, 0.085, textwrap.fill(note, 156), fontsize=7.0, color=study.PALETTE["muted"], va="top")
    return figure


@register_page(10)
def _page_clinical_subgroups(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                             title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier2 = _tier2_payload(data)
    if tier2 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    frame = _tier2_scored_frame(tier2)
    glycemic = ("diabetes_flag", "hba1c_band", "insulin")
    comorbid = ("hypertension", "dyslipidemia", "osa", "comorbidity_count", "biguanide", "sglt2")
    if frame is None or frame.empty or not frame["axis"].isin(list(glycemic + comorbid)).any():
        return _tier1_note_page(figure, "Clinical subgroups not populated in this source")
    ax_gly = figure.add_axes([0.06, 0.49, 0.88, 0.33])
    study.panel_label(ax_gly, "A", "Diabetes and glycemic strata (diabetes / HbA1c band / insulin)")
    _tier2_axis_table(ax_gly, frame, glycemic, max_rows=13)
    ax_com = figure.add_axes([0.06, 0.13, 0.88, 0.29])
    study.panel_label(ax_com, "B", "Comorbidity burden and concomitant medications")
    _tier2_axis_table(ax_com, frame, comorbid, max_rows=12)
    usable = tier2.get("axes_usable", {})
    note = _tier2_axis_population_note(usable, glycemic + comorbid) + \
        " biguanide/sglt2 are scored on the incretin arm only. " + SUBGROUP_CAVEAT
    figure.text(0.06, 0.075, textwrap.fill(note, 158), fontsize=6.8, color=study.PALETTE["muted"], va="top")
    return figure


@register_page(11)
def _page_obesity_calendar(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                           title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier2 = _tier2_payload(data)
    if tier2 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    frame = _tier2_scored_frame(tier2)
    obesity = ("obesity_class",)
    calendar = ("calendar_era", "covid_flag")
    if frame is None or frame.empty or not frame["axis"].isin(list(obesity + calendar)).any():
        return _tier1_note_page(figure, "Obesity-class / calendar-era strata not populated in this source")
    ax_ob = figure.add_axes([0.06, 0.47, 0.52, 0.35])
    study.panel_label(ax_ob, "A", "Clinical obesity class (35-40 / 40-50 / 50+)")
    _tier2_axis_table(ax_ob, frame, obesity, max_rows=14, columns=_TIER2_COMPACT_COLUMNS,
                      labels=_TIER2_COMPACT_LABELS)
    ax_bar = figure.add_axes([0.66, 0.47, 0.29, 0.35])
    study.panel_label(ax_bar, "B", "CRPS by obesity class")
    obesity_slice = _tier2_single_axis_slice(frame, "obesity_class")
    if obesity_slice is None or obesity_slice.empty:
        empty_panel(ax_bar, "Not estimable")
    else:
        _bar_by_group(ax_bar, obesity_slice["subgroup"].tolist(), obesity_slice["crps"].tolist(),
                      "CRPS (kg/m2)", study.PALETTE["green"])
    ax_cal = figure.add_axes([0.06, 0.13, 0.88, 0.26])
    study.panel_label(ax_cal, "C", "Calendar era (pre-2021 vs 2021+) and COVID index flag")
    _tier2_axis_table(ax_cal, frame, calendar, max_rows=12)
    usable = tier2.get("axes_usable", {})
    note = _tier2_axis_population_note(usable, obesity + calendar) + \
        " Obesity classes are fixed clinical bins, not data-driven tertiles. " + SUBGROUP_CAVEAT
    figure.text(0.06, 0.075, textwrap.fill(note, 158), fontsize=6.8, color=study.PALETTE["muted"], va="top")
    return figure


# --------------------------------------------------------------------------------------------
# Tier 3 - robustness beyond what production did. Five sub-analyses, one disclosure-controlled
# robustness_*.csv each, all reading the SAME frozen held-out prognostic frames Tier 1/2 use
# (selected candidate at origin 0 on HELD_OUT_SPLITS, joined to the covariate frame on
# patient_id, streamed one origin-0 partition at a time):
#   1. powered-only reanalysis of the pooled CRPS-improvement gate (all cells vs n>=200 & ESS>=100)
#   2. incretin eligibility-threshold sweep at 6/12/18 completed months (estimand change, not just N)
#   3. IPCW model-form sensitivity (logistic vs gradient-boosted censoring)
#   4. baseline-window sensitivity keyed off baseline_bmi_day recency
#   5. state-cluster bootstrap of the headline CIs vs the patient bootstrap
# Every sub-analysis degrades to a logged skipped_reason; no silent caps (omissions are recorded
# on the checkpoint, in the manifest via the render stage, and visibly on page 12).
# --------------------------------------------------------------------------------------------
ROBUSTNESS_POWERED_ONLY_COLUMNS = (
    "arm", "outcome", "n", "n_cells_all", "n_cells_powered", "pooled_model_crps_all",
    "pooled_baseline_crps_all", "rel_improvement_all", "pooled_model_crps_powered",
    "pooled_baseline_crps_powered", "rel_improvement_powered", "verdict",
)
ROBUSTNESS_ELIGIBILITY_COLUMNS = (
    "qualifying_months", "outcome", "target_month", "cohort_n", "n", "ess", "crps",
    "coverage_80", "cal_slope", "powered", "skipped_reason",
)
ROBUSTNESS_IPCW_FORM_COLUMNS = (
    "arm", "outcome", "target_month", "form", "n", "n_observed", "ess", "weight_mean",
    "weight_max", "weight_p50", "weight_p90", "weight_p99", "crps", "coverage_80", "cal_slope",
    "powered",
)
ROBUSTNESS_BASELINE_WINDOW_COLUMNS = (
    "arm", "outcome", "target_month", "window", "n", "ess", "crps", "coverage_80",
    "cal_slope", "powered", "skipped_reason",
)
ROBUSTNESS_CLUSTER_BOOTSTRAP_COLUMNS = (
    "arm", "outcome", "target_month", "n", "n_states", "crps", "patient_ci_low",
    "patient_ci_high", "patient_ci_width", "state_ci_low", "state_ci_high", "state_ci_width",
    "width_ratio_state_over_patient", "powered", "skipped_reason",
)

# The incretin continuer eligibility thresholds swept (completed months of treatment). 6 and 12
# are the knob's choices; 18 extends the knob (re-derived directly via the cohort-construction
# path, since 18 is outside study.INCRETIN_QUALIFYING_MONTH_CHOICES - a documented widening, not a
# silent cap). month_word(18) -> "18" upstream, so construction does not crash on 18.
ELIGIBILITY_SWEEP_MONTHS = (6, 12, 18)

# IPCW model-form sensitivity: the confounder set the censoring model conditions on (the same
# leak-free baseline covariates the production weighting path uses), and the two propensity forms.
IPCW_FORM_NUMERIC = ("baseline_value", "age_at_index", "diabetes_flag", "index_year")
IPCW_FORM_CATEGORICAL = ("sex",)
IPCW_FORMS = ("logistic", "gradient_boosted")

POWERED_ONLY_NOTE = (
    "Powered-only reanalysis. The pooled CRPS-improvement gate (model CRPS vs the population-change "
    "baseline CRPS = IPCW-weighted MAE around the weighted-mean observed outcome) is computed over "
    "ALL disclosable cells and, separately, over powered-only cells (n >= 200 AND IPCW ESS >= 100), "
    "per (arm, outcome). Reporting both prevents a pooled gain from hiding a failed powered "
    "horizon. verdict: real_ceiling = the model does not beat baseline even where powered (a real "
    "ceiling, not a power artifact); power_artifact = the powered-only improvement exceeds the "
    "all-cells pooled value by >= 0.02 (underpowered cells were masking a real gain); "
    "stable_real_signal = the two agree; no_powered_cells = no cell reached the powered threshold; "
    "no_disclosable_cells = the arm has no cell above the n < 11 floor, so nothing is scored. "
    + ARM_RECUT_NOTE
)
ELIGIBILITY_NOTE = (
    "Incretin eligibility-threshold sweep. Changing the qualifying-months threshold changes the "
    "ESTIMAND - which continuation-defined population is being forecast - not merely N. Cohorts are "
    "re-derived at 6, 12, and 18 completed months (18 is outside the {6,12} knob and is re-derived "
    "directly through the cohort-construction path). Metrics are computed on the frozen predictions "
    "subset to the re-derived eligible patient_ids; because frozen forecasts exist only for the "
    "production cohort, a looser threshold cannot add unmodeled patients (their newly-eligible "
    "members have no frozen forecast), so those cells reflect the intersection with the frozen store."
)
IPCW_FORM_NOTE = (
    "IPCW model-form sensitivity. The censoring/observation weight P(observed | L) is re-estimated "
    "two ways - a logistic form (the production propensity family) and a gradient-boosted form - "
    "cross-fitted over 5 patient-clustered folds and truncated at the production (0.01, 0.99) weight "
    "quantiles. Weight distribution, IPCW effective sample size, and headline-metric stability are "
    "reported for each form, extending production's truncation-only weight sensitivity. "
    + ARM_RECUT_NOTE
)
BASELINE_WINDOW_NOTE = (
    "Baseline-window sensitivity. Held-out accuracy is stratified by baseline-measurement recency "
    "|baseline_bmi_day| (days from index), to show how far-from-index baselines affect accuracy. "
    "Degrades to a logged skipped_reason when baseline_bmi_day is not populated. " + ARM_RECUT_NOTE
)
CLUSTER_BOOTSTRAP_NOTE = (
    "State-cluster bootstrap. Headline CRPS confidence intervals are recomputed resampling whole "
    "states (then all of their patients) instead of resampling patients, and the CI widths are "
    "compared. Wider state-cluster intervals expose within-state correlation the patient bootstrap "
    "ignores. Resampling stays index-based, so no frame is copied per replicate. Skipped with a "
    "logged reason when state is not usable. " + ARM_RECUT_NOTE
)


# ----- (1) powered-only reanalysis ---------------------------------------------------------
def _population_baseline_crps(y: Any, weight: Any) -> float:
    """Population-change baseline CRPS = IPCW-weighted MAE around the weighted-mean observed
    outcome (a degenerate point forecast at the population mean). This is the same baseline the
    M4 tipping-point uses and mirrors the production population_change candidate's role."""
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if y.size == 0:
        return float("nan")
    center = study.weighted_mean(y, weight)
    return float(study.weighted_mean(np.abs(y - center), weight))


def _powered_only_verdict(rel_all: float, rel_powered: float, n_cells_powered: int) -> str:
    if n_cells_powered == 0 or not np.isfinite(rel_powered):
        return "no_powered_cells"
    if rel_powered <= 0.0:
        return "real_ceiling"
    if np.isfinite(rel_all) and (rel_powered - rel_all) >= 0.02:
        return "power_artifact"
    return "stable_real_signal"


def _tier3_powered_only(frames: Mapping[tuple[str, str], Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, outcome, frame in _reporting_arm_frames(frames):
        observed = frame.loc[frame["target_observed"].fillna(False).astype(bool)]
        if observed.empty:
            continue
        n_all = n_cells_all = n_cells_powered = 0
        model_num_all = base_num_all = wsum_all = 0.0
        model_num_pw = base_num_pw = wsum_pw = 0.0
        for target_month, raw_cell in observed.groupby("target_month", sort=True):
            cell = raw_cell.drop_duplicates("patient_id")
            n = int(len(cell))
            if n < MIN_CELL_SIZE:
                continue
            y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
            matrix = _quantile_matrix(cell)
            weight = _cell_weight(cell)
            wsum = float(np.sum(weight))
            model_crps = float(study.quantile_crps(y, matrix, weight))
            base_crps = _population_baseline_crps(y, weight)
            ess = _effective_sample_size(weight)
            n_cells_all += 1
            n_all += n
            model_num_all += model_crps * wsum
            base_num_all += base_crps * wsum
            wsum_all += wsum
            if _is_powered(n, ess):
                n_cells_powered += 1
                model_num_pw += model_crps * wsum
                base_num_pw += base_crps * wsum
                wsum_pw += wsum
        if n_cells_all == 0:
            # Named, not dropped: a thin arm (the sleeve arm is the one this fires for) would
            # otherwise vanish from the headline robustness table with no record.
            rows.append({"arm": arm, "outcome": outcome, "n": 0, "n_cells_all": 0,
                         "n_cells_powered": 0, "verdict": "no_disclosable_cells"})
            continue
        pooled_model_all = model_num_all / wsum_all if wsum_all > 0 else float("nan")
        pooled_base_all = base_num_all / wsum_all if wsum_all > 0 else float("nan")
        rel_all = ((pooled_base_all - pooled_model_all) / pooled_base_all
                   if np.isfinite(pooled_base_all) and pooled_base_all > 0 else float("nan"))
        if n_cells_powered > 0 and wsum_pw > 0:
            pooled_model_pw = model_num_pw / wsum_pw
            pooled_base_pw = base_num_pw / wsum_pw
            rel_pw = ((pooled_base_pw - pooled_model_pw) / pooled_base_pw
                      if pooled_base_pw > 0 else float("nan"))
        else:
            pooled_model_pw = pooled_base_pw = rel_pw = float("nan")
        rows.append({
            "arm": arm, "outcome": outcome, "n": int(n_all),
            "n_cells_all": int(n_cells_all), "n_cells_powered": int(n_cells_powered),
            "pooled_model_crps_all": pooled_model_all, "pooled_baseline_crps_all": pooled_base_all,
            "rel_improvement_all": rel_all, "pooled_model_crps_powered": pooled_model_pw,
            "pooled_baseline_crps_powered": pooled_base_pw, "rel_improvement_powered": rel_pw,
            "verdict": _powered_only_verdict(rel_all, rel_pw, n_cells_powered),
        })
    return rows


# ----- (2) incretin eligibility-threshold sweep --------------------------------------------
def _incretin_eligible_ids_from_bundle(bundle: Any, months: int) -> set[str]:
    """Re-derive the incretin cohort at a qualifying-months threshold and return its patient_ids.

    Cohort construction reads the entry window from bundle.metadata['incretin_qualifying_days'] and
    re-derives the incretin continuers from the raw medication records, so overriding that metadata
    (and calling construct_cohorts on a fresh bundle) re-derives the cohort at any threshold - 18
    included, since month_word(18) -> '18' upstream rather than raising.
    """
    bundle.metadata["incretin_qualifying_days"] = study.qualifying_days_for_months(int(months))
    bundle.metadata["incretin_qualifying_months"] = int(months)
    artifacts = study.construct_cohorts(bundle)
    cohorts = artifacts["cohorts"]
    incretin = cohorts.loc[cohorts["cohort"].astype(str) == "incretin"]
    return set(incretin["patient_id"].astype(str))


def _eligibility_ids_smoke(cfg: SecondaryConfig, months_list: Sequence[int]) -> tuple[
        dict[int, set[str] | None], list[str]]:
    """--smoke path: re-derive each threshold in-process from the synthetic bundle (no Cosmos)."""
    study_cfg = cfg.study_config()
    ids_by_threshold: dict[int, set[str] | None] = {}
    omissions: list[str] = []
    for months in months_list:
        try:
            bundle = study.synthetic_data_bundle(study_cfg)
            ids_by_threshold[int(months)] = _incretin_eligible_ids_from_bundle(bundle, int(months))
            del bundle
            gc.collect()
        except Exception as error:  # never crash the sweep; log the omission
            ids_by_threshold[int(months)] = None
            omissions.append(
                f"Eligibility sweep: {months}-month synthetic re-derivation failed ({error!r}); omitted.")
    return ids_by_threshold, omissions


def _eligibility_ids_subprocess(cfg: SecondaryConfig, months_list: Sequence[int]) -> tuple[
        dict[int, set[str] | None], list[str]]:
    """--from-run/--full path: run each threshold's memory-heavy Cosmos re-acquire in its own
    subprocess (mirrors _run_stage_subprocess) so peak RSS is reclaimed on exit; the subprocess
    writes the eligible patient_ids to a small json this parent reads."""
    ids_by_threshold: dict[int, set[str] | None] = {}
    omissions: list[str] = []
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for months in months_list:
        out_path = cfg.checkpoints_dir / f"eligibility_ids_{int(months)}.json"
        if out_path.exists():
            out_path.unlink()
        command = [
            sys.executable, str(_THIS_FILE), "--eligibility-worker-months", str(int(months)),
            "--worker-mode", cfg.mode, "--output-dir", cfg.output_dir, "--seed", str(cfg.seed),
            "--incretin-qualifying-months", str(cfg.incretin_qualifying_months),
        ]
        if cfg.from_run:
            command += ["--from-run", cfg.from_run]
        try:
            result = subprocess.run(command, cwd=str(_REPO_ROOT))
            if result.returncode != 0 or not out_path.exists():
                raise RuntimeError(f"exit code {result.returncode}")
            payload = study.read_json(out_path, {})
            patient_ids = payload.get("patient_ids")
            ids_by_threshold[int(months)] = (
                {str(pid) for pid in patient_ids} if patient_ids is not None else None)
            if ids_by_threshold[int(months)] is None:
                omissions.append(
                    f"Eligibility sweep: {months}-month re-acquire returned no ids; omitted.")
        except Exception as error:
            ids_by_threshold[int(months)] = None
            omissions.append(
                f"Eligibility sweep: {months}-month Cosmos re-acquire subprocess failed ({error!r}); omitted.")
    return ids_by_threshold, omissions


def _eligibility_skip_row(months: int, reason: str, cohort_n: int | None = None) -> dict[str, Any]:
    return {
        "qualifying_months": int(months), "outcome": "", "target_month": np.nan,
        "cohort_n": int(cohort_n) if cohort_n is not None else np.nan, "n": 0, "ess": np.nan,
        "crps": np.nan, "coverage_80": np.nan, "cal_slope": np.nan, "powered": False,
        "skipped_reason": reason,
    }


def _tier3_eligibility_sweep(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any]) -> tuple[
        list[dict[str, Any]], dict[str, Any]]:
    months_list = [int(m) for m in ELIGIBILITY_SWEEP_MONTHS]
    if cfg.smoke:
        ids_by_threshold, omissions = _eligibility_ids_smoke(cfg, months_list)
    else:
        ids_by_threshold, omissions = _eligibility_ids_subprocess(cfg, months_list)
    if 18 in months_list:
        omissions.append(
            "Eligibility sweep includes an 18-month cohort re-derived directly via the "
            "cohort-construction path (18 is outside the --incretin-qualifying-months {6,12} knob).")
    coverage_note = (
        f"Frozen predictions exist only for the production incretin cohort "
        f"(qualifying = {cfg.incretin_qualifying_months} months); looser thresholds cannot add "
        f"unmodeled patients, so metric cells reflect the intersection with the frozen store."
    )
    # The one Section-10 robustness read that is NOT cut over the three arms: the qualifying-months
    # threshold defines the incretin continuer cohort, so a surgical arm has no eligibility knob to
    # sweep and intersecting a re-derived incretin id set with it is empty by construction. It still
    # forms its arm through the shared primitive, so the incretin arm is defined identically here.
    incretin_frames = {outcome: _arm_view(frames, "incretin", outcome, "")[0] for outcome in OUTCOMES}
    rows: list[dict[str, Any]] = []
    cohort_n_map: dict[int, int | None] = {}
    for months in months_list:
        ids = ids_by_threshold.get(int(months))
        cohort_n_map[int(months)] = int(len(ids)) if ids is not None else None
        if ids is None:
            rows.append(_eligibility_skip_row(months, "eligible cohort could not be re-derived at this threshold"))
            continue
        cohort_n = int(len(ids))
        any_cell = False
        for outcome in OUTCOMES:
            frame = incretin_frames.get(outcome)
            if frame is None or frame.empty:
                continue
            observed = frame.loc[frame["target_observed"].fillna(False).astype(bool)]
            observed = observed.loc[observed["patient_id"].astype(str).isin(ids)]
            if observed.empty:
                continue
            for target_month, raw_cell in observed.groupby("target_month", sort=True):
                cell = raw_cell.drop_duplicates("patient_id")
                n = int(len(cell))
                if n == 0:
                    continue
                y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
                matrix = _quantile_matrix(cell)
                weight = _cell_weight(cell)
                metrics = _prognostic_metrics(y, matrix, weight)
                ess = _effective_sample_size(weight)
                rows.append({
                    "qualifying_months": int(months), "outcome": outcome,
                    "target_month": int(target_month), "cohort_n": cohort_n, "n": n, "ess": ess,
                    "crps": metrics["crps"], "coverage_80": metrics["coverage_80"],
                    "cal_slope": metrics["cal_slope"], "powered": _is_powered(n, ess),
                    "skipped_reason": "",
                })
                any_cell = True
        if not any_cell:
            rows.append(_eligibility_skip_row(
                months, "no frozen incretin predictions for the eligible set", cohort_n=cohort_n))
    meta = {"months": months_list, "cohort_n": cohort_n_map, "omissions": omissions,
            "coverage_note": coverage_note}
    return rows, meta


# ----- (3) IPCW model-form sensitivity -----------------------------------------------------
def _gbm_classifier(seed: int) -> Any:
    """Gradient-boosted (torch-free) censoring/propensity classifier for the IPCW-form contrast."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    try:
        return HistGradientBoostingClassifier(
            random_state=int(seed), max_depth=3, max_iter=150, early_stopping=False)
    except TypeError:
        return HistGradientBoostingClassifier(random_state=int(seed), max_depth=3, max_iter=150)


def _tier3_encode_cell(cell: Any) -> Any | None:
    """Encode one cell's leak-free baseline confounders with the production TabularEncoder."""
    numeric = [column for column in IPCW_FORM_NUMERIC if column in cell.columns]
    categorical = [column for column in IPCW_FORM_CATEGORICAL if column in cell.columns]
    if not numeric and not categorical:
        return None
    encoder = study.TabularEncoder.fit(cell, numeric=numeric, categorical=categorical)
    return encoder.transform(cell)


def _crossfit_observation_prob(x: Any, delta: Any, patient_ids: Any, form: str, seed: int,
                               folds: int) -> tuple[Any, float]:
    """Out-of-fold P(observed | L) via a logistic or gradient-boosted form, patient-clustered folds.

    The logistic form reuses study.fit_probability_model / predict_probability (the production
    propensity family); the gradient-boosted form fits a HistGradientBoostingClassifier out-of-fold.
    Either form falls back to the fold's marginal when a fold cannot fit (too few rows or one class),
    so a degenerate fold never raises. Returns (probabilities, overall marginal)."""
    x = np.asarray(x, dtype=float)
    delta = np.asarray(delta).astype(int)
    n = int(delta.shape[0])
    prob = np.full(n, np.nan)
    marginal = float(np.mean(delta)) if n else 0.5
    fold_ids = patient_folds(patient_ids, seed, folds, salt="ipcw")
    for fold in range(folds):
        test_idx = np.where(fold_ids == fold)[0]
        if test_idx.size == 0:
            continue
        train_idx = np.where(fold_ids != fold)[0]
        y_train = delta[train_idx]
        if form == "logistic":
            model, fold_marginal, _status = study.fit_probability_model(x[train_idx], y_train)
            prob[test_idx] = study.predict_probability(model, x[test_idx], fold_marginal)
        else:
            fold_marginal = float(np.mean(y_train)) if y_train.size else marginal
            if y_train.size < 20 or np.unique(y_train).size < 2:
                prob[test_idx] = float(min(max(fold_marginal, 0.01), 0.99))
            else:
                try:
                    classifier = _gbm_classifier(seed)
                    classifier.fit(x[train_idx], y_train)
                    prob[test_idx] = np.clip(classifier.predict_proba(x[test_idx])[:, 1], 0.01, 0.99)
                except Exception:
                    prob[test_idx] = float(min(max(fold_marginal, 0.01), 0.99))
    prob = np.where(np.isfinite(prob), prob, float(min(max(marginal, 0.01), 0.99)))
    return prob, marginal


def _tier3_ipcw_form(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, outcome, frame in _reporting_arm_frames(frames):
        for target_month, raw_cell in frame.groupby("target_month", sort=True):
            cell = raw_cell.drop_duplicates("patient_id").reset_index(drop=True)
            n = int(len(cell))
            if n < MIN_CELL_SIZE:
                continue
            delta = cell["target_observed"].fillna(False).astype(bool).to_numpy()
            n_observed = int(delta.sum())
            if n_observed < MIN_CELL_SIZE:
                continue
            x = _tier3_encode_cell(cell)
            if x is None:
                continue
            patient_ids = cell["patient_id"].astype(str).to_numpy()
            y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
            matrix = _quantile_matrix(cell)
            obs_idx = np.where(delta)[0]
            for form in IPCW_FORMS:
                prob, marginal = _crossfit_observation_prob(
                    x, delta.astype(int), patient_ids, form, cfg.seed, cfg.nuisance_folds)
                raw_weight = marginal / np.clip(prob, 0.01, 0.99)
                obs_weight_raw = raw_weight[obs_idx]
                finite = obs_weight_raw[np.isfinite(obs_weight_raw)]
                if finite.size:
                    low, high = np.quantile(finite, study.WEIGHT_TRUNCATION)
                    weight = np.clip(obs_weight_raw, low, high)
                else:
                    weight = np.ones(obs_idx.size, dtype=float)
                metrics = _prognostic_metrics(y[obs_idx], matrix[obs_idx], weight)
                ess = _effective_sample_size(weight)
                rows.append({
                    "arm": arm, "outcome": outcome, "target_month": int(target_month),
                    "form": form, "n": n, "n_observed": n_observed, "ess": ess,
                    "weight_mean": float(np.mean(weight)), "weight_max": float(np.max(weight)),
                    "weight_p50": float(np.percentile(weight, 50)),
                    "weight_p90": float(np.percentile(weight, 90)),
                    "weight_p99": float(np.percentile(weight, 99)),
                    "crps": metrics["crps"], "coverage_80": metrics["coverage_80"],
                    "cal_slope": metrics["cal_slope"], "powered": _is_powered(n_observed, ess),
                })
    return rows


# ----- (4) baseline-window sensitivity -----------------------------------------------------
def _baseline_window_labels(values: Any) -> Any:
    """Baseline-recency windows from |baseline_bmi_day|: <=30, 31-90, 91-180, >180; NA if missing."""
    day = pd.to_numeric(pd.Series(values), errors="coerce").abs()
    result = pd.Series(pd.NA, index=day.index, dtype="string")
    result[day <= 30] = "<=30"
    result[(day > 30) & (day <= 90)] = "31-90"
    result[(day > 90) & (day <= 180)] = "91-180"
    result[day > 180] = ">180"
    return result


def _tier3_baseline_window(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any],
                           covariates: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    local_audit = column_population_audit(covariates, ["baseline_bmi_day"])
    if not audit_usable(local_audit, "baseline_bmi_day"):
        reason = "baseline_bmi_day not populated in this source; baseline-window sensitivity skipped"
        row = {
            "arm": "", "outcome": "", "target_month": np.nan, "window": "", "n": 0, "ess": np.nan,
            "crps": np.nan, "coverage_80": np.nan, "cal_slope": np.nan, "powered": False,
            "skipped_reason": reason,
        }
        return [row], {"usable": False, "skipped_reason": reason}
    rows: list[dict[str, Any]] = []
    for arm, outcome, frame in _reporting_arm_frames(frames):
        observed = frame.loc[frame["target_observed"].fillna(False).astype(bool)].copy()
        if observed.empty or "baseline_bmi_day" not in observed.columns:
            continue
        work = observed.assign(_window=_baseline_window_labels(observed["baseline_bmi_day"]))
        work = work.loc[work["_window"].notna()]
        for (target_month, window), raw_cell in work.groupby(["target_month", "_window"], sort=True):
            cell = raw_cell.drop_duplicates("patient_id")
            n = int(len(cell))
            if n == 0:
                continue
            y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
            matrix = _quantile_matrix(cell)
            weight = _cell_weight(cell)
            metrics = _prognostic_metrics(y, matrix, weight)
            ess = _effective_sample_size(weight)
            rows.append({
                "arm": arm, "outcome": outcome, "target_month": int(target_month),
                "window": str(window), "n": n, "ess": ess, "crps": metrics["crps"],
                "coverage_80": metrics["coverage_80"], "cal_slope": metrics["cal_slope"],
                "powered": _is_powered(n, ess), "skipped_reason": "",
            })
    return rows, {"usable": True, "skipped_reason": ""}


# ----- (5) state-cluster bootstrap ---------------------------------------------------------
def _cluster_bootstrap_crps_ci(y: Any, matrix: Any, weight: Any, cluster_ids: Any, rng: Any,
                               reps: int) -> tuple[float, float]:
    """Percentile CI of the IPCW-weighted CRPS resampling whole clusters (patients or states)."""
    y = np.asarray(y, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    weight = np.asarray(weight, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    unique = np.unique(cluster_ids)
    if unique.size < 2 or reps <= 0 or y.size == 0:
        return float("nan"), float("nan")
    index_by_cluster = {cluster: np.where(cluster_ids == cluster)[0] for cluster in unique}
    estimates = np.empty(reps, dtype=float)
    for replicate in range(reps):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([index_by_cluster[cluster] for cluster in drawn])
        estimates[replicate] = float(study.quantile_crps(y[idx], matrix[idx], weight[idx]))
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi)


def _cluster_bootstrap_skip(reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "arm": "", "outcome": "", "target_month": np.nan, "n": 0, "n_states": 0, "crps": np.nan,
        "patient_ci_low": np.nan, "patient_ci_high": np.nan, "patient_ci_width": np.nan,
        "state_ci_low": np.nan, "state_ci_high": np.nan, "state_ci_width": np.nan,
        "width_ratio_state_over_patient": np.nan, "powered": False, "skipped_reason": reason,
    }
    return [row], {"usable": False, "skipped_reason": reason}


def _tier3_cluster_bootstrap(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any], audit: Any,
                             rng: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (audit is not None and audit_usable(audit, "state")):
        return _cluster_bootstrap_skip("state not usable (audit gate); state-cluster bootstrap skipped")
    reps = int(cfg.bootstrap_replicates)
    rows: list[dict[str, Any]] = []
    for arm, outcome, frame in _reporting_arm_frames(frames):
        observed = frame.loc[frame["target_observed"].fillna(False).astype(bool)].copy()
        if observed.empty or "state" not in observed.columns:
            continue
        observed["_state"] = observed["state"].astype("string")
        observed = observed.loc[observed["_state"].notna() & (observed["_state"].str.strip() != "")]
        for target_month, raw_cell in observed.groupby("target_month", sort=True):
            cell = raw_cell.drop_duplicates("patient_id")
            n = int(len(cell))
            if n < MIN_CELL_SIZE:
                continue
            y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
            matrix = _quantile_matrix(cell)
            weight = _cell_weight(cell)
            ess = _effective_sample_size(weight)
            patient_ids = cell["patient_id"].astype(str).to_numpy()
            states = cell["_state"].astype(str).to_numpy()
            n_states = int(np.unique(states).size)
            crps = float(study.quantile_crps(y, matrix, weight))
            p_lo, p_hi = _cluster_bootstrap_crps_ci(y, matrix, weight, patient_ids, rng, reps)
            s_lo, s_hi = _cluster_bootstrap_crps_ci(y, matrix, weight, states, rng, reps)
            p_width = p_hi - p_lo if np.isfinite(p_lo) and np.isfinite(p_hi) else float("nan")
            s_width = s_hi - s_lo if np.isfinite(s_lo) and np.isfinite(s_hi) else float("nan")
            ratio = (s_width / p_width
                     if np.isfinite(s_width) and np.isfinite(p_width) and p_width > 0 else float("nan"))
            rows.append({
                "arm": arm, "outcome": outcome, "target_month": int(target_month), "n": n,
                "n_states": n_states, "crps": crps, "patient_ci_low": p_lo, "patient_ci_high": p_hi,
                "patient_ci_width": p_width, "state_ci_low": s_lo, "state_ci_high": s_hi,
                "state_ci_width": s_width, "width_ratio_state_over_patient": ratio,
                "powered": _is_powered(n, ess), "skipped_reason": "",
            })
    if not rows:
        return _cluster_bootstrap_skip("no usable state-labeled cells for the cluster bootstrap")
    return rows, {"usable": True, "skipped_reason": ""}


# ----- eligibility re-acquire worker (one threshold, isolated process) ----------------------
def _run_eligibility_acquire_worker(args: argparse.Namespace) -> int:
    """One-threshold incretin eligibility re-acquire, isolated in its own process so the whole-bundle
    Cosmos load's peak RSS is reclaimed on exit. Loads the bundle via the production query_cosmos,
    overrides the qualifying-months metadata to the requested threshold, re-derives the incretin
    cohort, and writes the eligible patient_ids to secondary/checkpoints/eligibility_ids_<months>.json
    for the parent stage_tier3 to consume. Never invoked in --smoke (that path stays in-process)."""
    load_runtime()
    months = int(args.eligibility_worker_months)
    cfg = SecondaryConfig(
        mode=args.worker_mode or "from-run", output_dir=args.output_dir, from_run=args.from_run,
        seed=int(args.seed), incretin_qualifying_months=int(args.incretin_qualifying_months),
    )
    study_cfg = cfg.study_config()
    bundle = study.query_cosmos(study_cfg)
    ids = _incretin_eligible_ids_from_bundle(bundle, months)
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    study.atomic_json(
        cfg.checkpoints_dir / f"eligibility_ids_{months}.json",
        {"qualifying_months": months, "cohort_n": int(len(ids)), "patient_ids": sorted(ids)},
    )
    return 0


# --------------------------------------------------------------------------------------------
# Tier 3 figure page (12 robustness panel): five panels + visible omission notes. Exception-safe.
# --------------------------------------------------------------------------------------------
def _tier3_payload(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    tier3 = data.get("tier3")
    if not isinstance(tier3, Mapping) or str(tier3.get("status")) != "done":
        return None
    return tier3


def _scored_rows(frame: Any) -> Any:
    """Drop skipped_reason rows so a panel plots only real cells (suppression handled downstream)."""
    if frame is None or frame.empty:
        return frame
    if "skipped_reason" in frame.columns:
        return frame.loc[frame["skipped_reason"].astype(str) == ""].copy()
    return frame


def _tier3_powered_bar(axis: Any, frame: Any) -> None:
    if frame is None or frame.empty:
        empty_panel(axis, "Not estimable")
        return
    # Three arms put six bars in this panel, so the verdict wraps instead of running into its
    # neighbour's label.
    labels = [f"{row.arm}/{row.outcome}\n" + textwrap.fill(str(row.verdict).replace("_", " "), 14)
              for row in frame.itertuples()]
    all_vals = pd.to_numeric(frame["rel_improvement_all"], errors="coerce").to_numpy(float)
    powered_vals = pd.to_numeric(frame["rel_improvement_powered"], errors="coerce").to_numpy(float)
    x = np.arange(len(labels))
    width = 0.38
    axis.axhline(0.0, color=study.PALETTE["muted"], lw=0.8)
    axis.bar(x - width / 2, np.where(np.isfinite(all_vals), all_vals, 0.0), width=width,
             color=study.PALETTE["blue"], label="All cells")
    axis.bar(x + width / 2, np.where(np.isfinite(powered_vals), powered_vals, 0.0), width=width,
             color=study.PALETTE["orange"], label="Powered only")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=5.6)
    axis.set_ylabel("Rel. CRPS improvement", fontsize=8)
    axis.legend(fontsize=6.0, loc="best", frameon=False)


def _tier3_eligibility_panel(axis: Any, eligibility: Mapping[str, Any]) -> None:
    frame = pd.DataFrame(eligibility.get("rows", []) if isinstance(eligibility, Mapping) else [])
    if frame.empty:
        empty_panel(axis, "Not estimable")
        return
    scored = _scored_rows(frame)
    show = scored if scored is not None and not scored.empty else frame
    columns = ["qualifying_months", "outcome", "target_month", "cohort_n", "n", "crps", "coverage_80"]
    for column in columns:
        if column not in show.columns:
            show[column] = np.nan
    ordered = show.sort_values(["outcome", "qualifying_months", "target_month"])
    study.draw_compact_table(
        axis, ordered.loc[:, columns], columns,
        labels=["Qual.\nmonths", "Outcome", "Target\nmonth", "Cohort\nN", "N", "CRPS", "Cov 80"],
        max_rows=12)


def _tier3_ipcw_panel(axis: Any, frame: Any) -> None:
    if frame is None or frame.empty:
        empty_panel(axis, "Not estimable")
        return
    frame = frame.copy()
    columns = ["arm", "outcome", "target_month", "form", "ess", "weight_max", "crps", "coverage_80"]
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    ordered = frame.sort_values(["arm", "outcome", "target_month", "form"])
    study.draw_compact_table(
        axis, ordered.loc[:, columns], columns,
        labels=["Arm", "Outcome", "Target\nmonth", "Form", "IPCW\nESS", "Weight\nmax", "CRPS", "Cov 80"],
        max_rows=10)


def _tier3_baseline_panel(axis: Any, baseline: Mapping[str, Any]) -> None:
    if not isinstance(baseline, Mapping) or not baseline.get("usable", False):
        message = str(baseline.get("skipped_reason") or "baseline_bmi_day not populated") \
            if isinstance(baseline, Mapping) else "Not estimable"
        empty_panel(axis, message)
        return
    scored = _scored_rows(pd.DataFrame(baseline.get("rows", [])))
    if scored is None or scored.empty:
        empty_panel(axis, "Not estimable")
        return
    columns = ["arm", "outcome", "window", "target_month", "n", "crps", "coverage_80"]
    for column in columns:
        if column not in scored.columns:
            scored[column] = np.nan
    ordered = scored.sort_values(["arm", "outcome", "target_month", "window"])
    study.draw_compact_table(
        axis, ordered.loc[:, columns], columns,
        labels=["Arm", "Outcome", "Window\n(days)", "Target\nmonth", "N", "CRPS", "Cov 80"],
        max_rows=10)


def _tier3_cluster_panel(axis: Any, cluster: Mapping[str, Any]) -> None:
    if not isinstance(cluster, Mapping) or not cluster.get("usable", False):
        message = str(cluster.get("skipped_reason") or "state not usable; cluster bootstrap skipped") \
            if isinstance(cluster, Mapping) else "Not estimable"
        empty_panel(axis, message)
        return
    scored = _scored_rows(pd.DataFrame(cluster.get("rows", [])))
    if scored is None or scored.empty:
        empty_panel(axis, "Not estimable")
        return
    columns = ["arm", "outcome", "target_month", "n", "n_states", "patient_ci_width",
               "state_ci_width", "width_ratio_state_over_patient"]
    for column in columns:
        if column not in scored.columns:
            scored[column] = np.nan
    ordered = scored.sort_values(["arm", "outcome", "target_month"])
    study.draw_compact_table(
        axis, ordered.loc[:, columns], columns,
        labels=["Arm", "Outcome", "Target\nmonth", "N", "States", "Patient\nCI width",
                "State\nCI width", "Width\nratio"],
        max_rows=8)


@register_page(12)
def _page_robustness(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                     title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier3 = _tier3_payload(data)
    if tier3 is None:
        return _tier1_note_page(figure, PENDING_NOTE)

    ax_a = figure.add_axes([0.06, 0.56, 0.42, 0.27])
    study.panel_label(ax_a, "A", "Pooled CRPS improvement: all cells vs powered-only")
    _tier3_powered_bar(ax_a, pd.DataFrame(tier3.get("powered_only", [])))

    ax_b = figure.add_axes([0.55, 0.56, 0.40, 0.27])
    study.panel_label(ax_b, "B", "Eligibility-threshold sweep (changes the estimand)")
    _tier3_eligibility_panel(ax_b, tier3.get("eligibility", {}))

    ax_c = figure.add_axes([0.06, 0.31, 0.42, 0.18])
    study.panel_label(ax_c, "C", "IPCW form: logistic vs gradient-boosted")
    _tier3_ipcw_panel(ax_c, pd.DataFrame(tier3.get("ipcw_form", [])))

    ax_d = figure.add_axes([0.55, 0.31, 0.40, 0.18])
    study.panel_label(ax_d, "D", "Baseline-recency window accuracy")
    _tier3_baseline_panel(ax_d, tier3.get("baseline_window", {}))

    ax_e = figure.add_axes([0.06, 0.09, 0.50, 0.14])
    study.panel_label(ax_e, "E", "State-cluster vs patient bootstrap CI widths")
    _tier3_cluster_panel(ax_e, tier3.get("cluster_bootstrap", {}))

    omissions = tier3.get("omissions", []) or []
    note = "Omissions: " + (" | ".join(str(item) for item in omissions) if omissions else "none recorded.")
    coverage_note = ""
    eligibility = tier3.get("eligibility", {})
    if isinstance(eligibility, Mapping):
        coverage_note = str(eligibility.get("coverage_note", ""))
    text = note + (("  " + coverage_note) if coverage_note else "")
    figure.text(0.60, 0.225, textwrap.fill(text, 66), fontsize=6.2, color=study.PALETTE["muted"], va="top")
    return figure


# --------------------------------------------------------------------------------------------
# Tier 4 - clinical value reframings (spec Section 6 Tier 4). Four sub-analyses, one
# disclosure-controlled CSV each, all reading the SAME frozen held-out prognostic frames Tiers
# 1/2/3 use (selected candidate at origin 0 on HELD_OUT_SPLITS, joined to the covariate frame on
# patient_id, streamed one origin-0 partition at a time):
#   1. Threshold probabilities from the predictive quantile ladder (per-row CDF) with an
#      IPCW-weighted reliability diagram, Brier score, and AUROC for each clinical event.
#   2. Decision-curve net benefit for the binary threshold events vs treat-all / treat-none.
#   3. A "who is predictable" map ranking every disclosable subgroup stratum by 80% interval
#      width and CRPS, flagging reliable vs unreliable strata.
#   4. A strictly prognostic overlap-weighted GLP-1-vs-surgery predicted-trajectory contrast on
#      the shared covariate space. This is NOT a treatment effect (populations, time-zeros, and
#      forecasting models all differ); the caveat is carried onto the page and the CSV header.
# --------------------------------------------------------------------------------------------
# Clinical thresholds for the threshold-probability events.
BMI_THRESHOLD_BELOW_35 = 35.0            # below Class II obesity
HBA1C_THRESHOLD_NONDIABETIC = 5.7        # back into the non-diabetic range
HBA1C_THRESHOLD_REMISSION = 6.5          # remission-proxy glycemic ceiling (with insulin off)
TWL_TARGETS = (0.05, 0.10, 0.15)         # >=5 / 10 / 15% total weight loss
EWL_REFERENCE_BMI = 25.0                 # ideal BMI anchor for %excess-weight-loss

# Decision-curve threshold-probability grid: clinically sensible and bounded away from 1.0 so the
# p_t / (1 - p_t) odds term does not blow up.
DECISION_PT_GRID = tuple(round(0.05 + 0.05 * index, 2) for index in range(15))  # 0.05 .. 0.75

# Shared (cohort-agnostic) covariate space for the GLP-1-vs-surgery overlap propensity. Uses only
# covariates present in BOTH cohorts; every cohort-specific field (procedure, index_ingredient,
# index_route, therapy_class, treatment) is deliberately excluded.
GLP1_SHARED_NUMERIC = (
    "baseline_bmi", "baseline_hba1c", "age_at_index", "diabetes_flag", "hypertension",
    "dyslipidemia", "osa", "insulin", "svi", "index_year",
)
GLP1_SHARED_CATEGORICAL = ("sex", "ethnicity", "coverage")

THRESHOLD_PROB_COLUMNS = (
    "arm", "outcome", "target_month", "event", "event_label", "n", "ess", "brier", "auroc",
    "cal_in_large", "pred_mean", "obs_mean", "powered", "skipped_reason",
)
DECISION_CURVE_COLUMNS = (
    "arm", "outcome", "target_month", "event", "event_label", "p_t", "n", "net_benefit",
    "nb_treat_all", "nb_treat_none", "powered",
)
PREDICTABILITY_MAP_COLUMNS = (
    "arm", "outcome", "target_month", "axis", "subgroup", "n", "ess", "mean_width_80", "crps",
    "rank_width", "rank_crps", "reliable_flag", "powered",
)
GLP1_OVERLAP_COLUMNS = (
    "outcome", "target_month", "n_surgery", "n_incretin", "overlap_weighted_mean_pred_surgery",
    "overlap_weighted_mean_pred_incretin", "standardized_diff", "ess_overlap",
)

# The three reporting arms of the headline per-procedure AUROC (spec 7.2 / 3): the incretin
# cohort, plus the shared surgical cohort subset by the stored `procedure` column. Each entry is
# (arm, store cohort, required procedure or "" for the whole cohort).
PROCEDURE_AUROC_ARMS = (
    ("incretin", "incretin", ""),
    ("rygb", "surgery", "rygb"),
    ("sleeve", "surgery", "sleeve"),
)
PROCEDURE_ARM_LABELS = {"incretin": "Incretin", "rygb": "RYGB", "sleeve": "Sleeve"}
PROCEDURE_AUROC_MONTHS = (6, 12, 24)
# ROC curves are stored as true-positive rates on this fixed false-positive grid, so the
# checkpoint payload stays small and byte-identical across runs regardless of cohort size.
ROC_FPR_GRID = tuple(round(0.02 * index, 2) for index in range(51))  # 0.00 .. 1.00
PROCEDURE_AUROC_COLUMNS = (
    "arm", "cohort", "procedure", "target_month", "n", "ess", "event_rate", "auroc",
    "auroc_ci_low", "auroc_ci_high", "brier", "cal_in_large", "powered", "suppressed",
    "skipped_reason",
)
PROCEDURE_AUROC_NOTE = (
    "Headline per-procedure discrimination for P(BMI < 35) at 6, 12, and 24 months, reported "
    "separately for the incretin cohort and for the shared surgical model subset by procedure "
    "(RYGB, sleeve). Per-row probabilities are read off the calibrated predictive quantile ladder; "
    "AUROC, Brier, and calibration-in-the-large (mean predicted minus mean observed event rate) "
    "are IPCW-weighted over held-out observed rows. The 95% interval is a patient-clustered "
    "percentile bootstrap. Arm-horizon cells with n < 11 are suppressed, not dropped. Prognostic "
    "discrimination, not a causal effect."
)
PER_ARM_PROGNOSTIC_COLUMNS = (
    "arm", "cohort", "outcome", "target_month", "n", "ess", "crps", "rmse", "mae", "bias",
    "coverage_80", "coverage_90", "width_80", "cal_slope", "cal_intercept", "powered",
    "suppressed", "skipped_reason",
)
PER_ARM_PROGNOSTIC_NOTE = (
    "Per-arm calibration and accuracy of the forecast (spec 7.1), reported for the same three "
    "reporting arms as the headline discrimination: the incretin cohort, and the shared surgical "
    "model subset by procedure into RYGB and sleeve. Every metric is IPCW-weighted over held-out "
    "observed rows at the declared horizon: CRPS against the full 7-quantile ladder; RMSE / MAE / "
    "bias against the predicted median; 80% and 90% coverage against the q10-q90 and q05-q95 "
    "intervals; the calibration slope and intercept from a weighted regression of observed on "
    "predicted median (slope 1 and intercept 0 are perfect). Every declared arm-outcome-horizon "
    "cell yields a row: cells with n < 11 are suppressed, not dropped. The two surgical arms "
    "subset one shared model, so RYGB and sleeve are separately calibrated but not separately "
    "fitted. Prognostic performance, not a causal effect."
)

# Spec 10's feature-sensitivity science page. The secondary script consumes the FROZEN prediction
# store, never the trained model, so the forecast cannot be re-run with perturbed inputs. The
# model-agnostic read that IS available is a surrogate: regress the stored median forecast on the
# same baseline covariates the model saw, then perturb the surrogate. Everything it reports is a
# property of the SURROGATE's approximation of the forecast, which is why surrogate_r2 is carried
# beside every importance and why the page states the read as association, never causation.
FEATURE_SENSITIVITY_MONTH = 12
# Floor for fitting at all: every cross-fit fold has to leave a training set above TTE_MIN_MU_FIT
# (20 rows, the codebase's own too-thin-to-fit threshold), and a design this wide needs at least
# twice that. It guards against a degenerate fit, not against low power - surrogate_r2 is the
# honest power read and it is carried beside every importance.
FEATURE_SENSITIVITY_MIN_N = 50
FEATURE_SENSITIVITY_TOP_N = 7            # source features the page ranks per outcome
FEATURE_SWEEP_ROWS = 2000                # seeded subsample the counterfactual sweep averages over
FEATURE_SWEEP_GRID = tuple(round(0.1 * step, 2) for step in range(1, 10))  # 0.10 .. 0.90
FEATURE_SENSITIVITY_COLUMNS = (
    "arm", "cohort", "outcome", "feature", "permutation_importance", "importance_share", "rank",
    "n", "surrogate_r2", "skipped_reason",
)
FEATURE_SENSITIVITY_NOTE = (
    "Sensitivity of the FORECAST to baseline features, per reporting arm. The frozen store carries "
    "stored quantiles, not the trained model, so the forecast cannot be re-run with perturbed "
    "inputs. Instead a seeded, cross-fitted gradient-boosted SURROGATE maps the baseline "
    f"covariates to the stored median forecast at {FEATURE_SENSITIVITY_MONTH} months on held-out "
    "rows; importance is the fold-weighted rise in held-out squared error when a source feature's "
    "encoded columns are permuted together, and surrogate_r2 is the out-of-fold share of the "
    "forecast the surrogate reproduces, which bounds the read. This is an association between the "
    "baseline features and the model's own output: NOT a causal feature effect, and NOT a "
    "statement about the patient's biology."
)
# Page-only: the CSV carries the held-out importance rows, not the sweep, so this clause belongs
# on the page beside panel C rather than in the CSV header note above.
FEATURE_SWEEP_NOTE = (
    "Panel C sweeps that same fitted surrogate and is therefore IN-SAMPLE: a partial-dependence "
    "curve describes a fitted response surface, not a held-out forecast."
)

THRESHOLD_PROB_NOTE = (
    "Clinical threshold probabilities from the predictive quantile ladder. Each per-row CDF is a "
    "monotone piecewise-linear interpolation of the 7 stored quantiles (value=q_j, prob=level_j); "
    "thresholds outside the ladder clamp to the nearest endpoint. Reliability (predicted vs "
    "observed by decile), Brier score, and AUROC are IPCW-weighted over held-out observed rows; "
    "AUROC is reported only where the observed label carries both classes. Reported per reporting "
    "arm: incretin, and the shared surgical model subset by procedure into RYGB and sleeve. "
    "Arm-event cells with n < 11 are suppressed. Prognostic, not causal."
)
DECISION_CURVE_NOTE = (
    "Decision-curve net benefit for the binary threshold events: NB(p_t) = TP/n - (FP/n) * "
    "p_t/(1-p_t) with IPCW-weighted counts, against treat-all and treat-none reference lines. A "
    "prognostic clinical-utility reframing of the forecasts, not a causal effect. Reported per "
    "reporting arm: incretin, and the shared surgical model subset by procedure into RYGB and "
    "sleeve."
)
PREDICTABILITY_NOTE = (
    "Who-is-predictable map: every disclosable subgroup stratum (the Tier-2 axes) ranked WITHIN "
    "its outcome family by mean 80% predictive-interval width (q90-q10) and by CRPS (BMI and HbA1c "
    "are never pooled - their scales differ). reliable_flag marks strata at or below the "
    "within-outcome median on BOTH precision (width) and accuracy (CRPS). This audits "
    "predictability; it does not establish biological effect modification or rank patient groups. "
    + ARM_RECUT_NOTE
)
GLP1_CAVEAT = (
    "NOT A TREATMENT EFFECT. Overlap-weighted contrast of the model's predicted median "
    "trajectories between the surgery and incretin cohorts on the shared baseline covariate space, "
    "weighted by a cohort-membership propensity P(cohort=surgery | shared L) (overlap weight "
    "p*(1-p), cross-fitted tree classifier, patient-clustered folds). The two cohorts differ in "
    "population, time-zero, and forecasting model, so this is a strictly prognostic/descriptive "
    "comparison of predictions - never a causal GLP-1-vs-surgery effect."
)


# ----- threshold-probability core (quantile-ladder CDF, weighted AUROC, reliability) --------
def _quantile_ladder_cdf(matrix: Any, thresholds: Any) -> Any:
    """P(X <= threshold) per row from the stored quantile ladder.

    Builds each row's CDF by monotone piecewise-linear interpolation of the 7 points
    (value=q_j, prob=QUANTILE_LEVELS[j]). Quantile values are de-crossed to be non-decreasing;
    thresholds below/above the ladder clamp to the nearest endpoint probability; the result is
    clamped to [0, 1]. Fully vectorised over rows.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    n, width = matrix.shape
    levels = np.asarray(QUANTILE_LEVELS, dtype=float)
    values = np.maximum.accumulate(matrix, axis=1)  # de-cross to non-decreasing values per row
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.full(n, float(thresholds))
    rows = np.arange(n)
    k = np.sum(values <= thresholds[:, None], axis=1)  # count of ladder points at or below t
    below = k == 0
    above = k == width
    upper = np.clip(k, 1, width - 1)
    lower = upper - 1
    v_lo = values[rows, lower]
    v_hi = values[rows, upper]
    denom = v_hi - v_lo
    frac = np.where(denom > 0, (thresholds - v_lo) / np.where(denom > 0, denom, 1.0), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    prob = levels[lower] + frac * (levels[upper] - levels[lower])
    prob = np.where(below, levels[0], prob)
    prob = np.where(above, levels[-1], prob)
    return np.clip(prob, 0.0, 1.0)


def _weighted_auroc(label: Any, pred: Any, weight: Any) -> float:
    """IPCW-weighted AUROC via the weighted Mann-Whitney U statistic (tie-aware). NaN when either
    class is empty. O(n log n): negatives are sorted once, then every positive is placed by a pair
    of searchsorted calls."""
    label = np.asarray(label, dtype=float)
    pred = np.asarray(pred, dtype=float)
    weight = np.asarray(weight, dtype=float)
    mask = np.isfinite(label) & np.isfinite(pred) & np.isfinite(weight) & (weight > 0)
    label, pred, weight = label[mask], pred[mask], weight[mask]
    positive = label >= 0.5
    weight_pos, score_pos = weight[positive], pred[positive]
    weight_neg, score_neg = weight[~positive], pred[~positive]
    total_pos = float(np.sum(weight_pos))
    total_neg = float(np.sum(weight_neg))
    if score_pos.size == 0 or score_neg.size == 0 or total_pos <= 0 or total_neg <= 0:
        return float("nan")
    order = np.argsort(score_neg, kind="mergesort")
    score_neg_sorted = score_neg[order]
    cumulative_neg = np.concatenate([[0.0], np.cumsum(weight_neg[order])])
    below = np.searchsorted(score_neg_sorted, score_pos, side="left")
    at_or_below = np.searchsorted(score_neg_sorted, score_pos, side="right")
    weight_below = cumulative_neg[below]
    weight_equal = cumulative_neg[at_or_below] - weight_below
    numerator = float(np.sum(weight_pos * (weight_below + 0.5 * weight_equal)))
    return float(numerator / (total_pos * total_neg))


def _cell_threshold_auroc(cell: Any) -> tuple[float, float]:
    """IPCW-weighted P(BMI < 35) AUROC and observed event rate for one held-out observed cell.

    The same read the headline per-procedure page makes (calibrated quantile ladder -> per-row
    CDF -> weighted Mann-Whitney), reduced to the two numbers a heterogeneity row carries. Returns
    (nan, nan) when the cell has no scorable row; a single-class cell yields a nan AUROC from
    ``_weighted_auroc`` itself, so it is reported as not estimable rather than as chance.
    """
    target = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
    pred = _quantile_ladder_cdf(_quantile_matrix(cell), np.full(len(cell), BMI_THRESHOLD_BELOW_35))
    weight = _cell_weight(cell)
    keep = np.isfinite(target) & np.isfinite(pred) & np.isfinite(weight) & (weight > 0)
    if not bool(keep.any()):
        return float("nan"), float("nan")
    label = (target[keep] < BMI_THRESHOLD_BELOW_35).astype(float)
    return (_unit_clamp(_weighted_auroc(label, pred[keep], weight[keep])),
            float(study.weighted_mean(label, weight[keep])))


def _reliability_deciles(pred: Any, label: Any, weight: Any, bins: int = 10) -> list[dict[str, float]]:
    """IPCW-weighted reliability points: within each predicted-probability decile, the weighted
    mean predicted probability (x) and weighted mean observed frequency (y), plus the decile n."""
    pred = np.asarray(pred, dtype=float)
    label = np.asarray(label, dtype=float)
    weight = np.asarray(weight, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(label) & np.isfinite(weight) & (weight > 0)
    pred, label, weight = pred[mask], label[mask], weight[mask]
    if pred.size == 0:
        return []
    quantiles = int(min(bins, max(1, np.unique(pred).size)))
    try:
        codes = pd.qcut(pred, quantiles, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        codes = np.zeros(pred.size, dtype=float)
    codes = np.asarray(codes, dtype=float)
    points: list[dict[str, float]] = []
    for code in np.unique(codes[np.isfinite(codes)]):
        selected = codes == code
        cell_weight = weight[selected]
        points.append({
            "pred": float(study.weighted_mean(pred[selected], cell_weight)),
            "obs": float(study.weighted_mean(label[selected], cell_weight)),
            "n": int(np.sum(selected)),
        })
    return points


# ----- the three reporting arms (one re-cut of the held-out frames, shared by every per-arm read)
def _arm_view(frames: Mapping[tuple[str, str], Any], cohort: str, outcome: str,
              procedure: str) -> tuple[Any, str]:
    """One reporting arm's view of a (cohort, outcome) held-out frame, plus a not-estimable reason.

    A cohort-level arm (procedure "") passes its frame through UNCOPIED, so the 177k-row incretin
    frame is never duplicated; a procedure arm is an index-based subset of the shared surgical
    frame. The reason separates an arm absent from the source from an unpopulated procedure column,
    so a structural gap is never reported as a small-cell suppression. Returns (None, reason) when
    the arm cannot be formed at all, and (frame, "") otherwise - including an empty subset, which
    the callers report per horizon rather than as a whole-arm gap.
    """
    frame = frames.get((cohort, outcome))
    if frame is None or frame.empty:
        return None, f"no held-out {cohort} {outcome} predictions in this source"
    if not procedure:
        return frame, ""
    if "procedure" not in frame.columns:
        return None, "procedure not populated in this source"
    # fillna(False) so rows with an unrecorded procedure drop out instead of raising on a
    # nullable-boolean mask; they are neither a RYGB nor a sleeve patient.
    recorded = frame["procedure"].astype("string").str.lower()
    return frame.loc[recorded.eq(procedure).fillna(False)], ""


def _reporting_arm_frames(frames: Mapping[tuple[str, str], Any]) -> Iterator[tuple[str, str, Any]]:
    """The held-out frames re-cut to the three reporting arms, in a fixed arm-then-outcome order.

    Yields (arm, outcome, frame) one arm at a time so at most one procedure subset is alive; arms
    absent from the source are skipped here, and callers that must document that gap read the
    reason from ``_arm_view`` directly.
    """
    for arm, cohort, procedure in PROCEDURE_AUROC_ARMS:
        for outcome in OUTCOMES:
            frame, _reason = _arm_view(frames, cohort, outcome, procedure)
            if frame is not None and not frame.empty:
                yield arm, outcome, frame


def _arm_formation_gaps(frames: Mapping[tuple[str, str], Any]) -> list[str]:
    """Reporting arms this source cannot form at all, named once for the omissions ledger.

    Every Section-10 tier cut over the arms reads the SAME held-out frames, so one sweep documents
    the gap for all of them: an arm ``_arm_view`` cannot form is skipped by ``_reporting_arm_frames``
    and would otherwise vanish from five tiers with no record. An arm that forms but is EMPTY is not
    a gap - that is a small-cell story the tiers already report per cell.
    """
    return sorted({f"{arm} arm ({outcome}): {reason}"
                   for arm, cohort, procedure in PROCEDURE_AUROC_ARMS for outcome in OUTCOMES
                   if (reason := _arm_view(frames, cohort, outcome, procedure)[1])})


# ----- headline per-procedure AUROC (spec 7.2) ---------------------------------------------
def _weighted_roc_curve(label: Any, pred: Any, weight: Any) -> list[float]:
    """IPCW-weighted ROC as true-positive rates sampled on the fixed ROC_FPR_GRID.

    Scores are swept high to low; the weighted cumulative positives / negatives give the empirical
    ROC, which is then interpolated onto the shared grid so every stored curve is the same small
    fixed length. Empty when either class carries no weight.
    """
    label = np.asarray(label, dtype=float)
    pred = np.asarray(pred, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if label.size == 0:
        return []
    order = np.argsort(-pred, kind="mergesort")
    ordered_label, ordered_weight = label[order], weight[order]
    true_positive = np.cumsum(ordered_weight * ordered_label)
    false_positive = np.cumsum(ordered_weight * (1.0 - ordered_label))
    total_positive, total_negative = float(true_positive[-1]), float(false_positive[-1])
    if total_positive <= 0.0 or total_negative <= 0.0:
        return []
    tpr = np.concatenate([[0.0], true_positive / total_positive])
    fpr = np.concatenate([[0.0], false_positive / total_negative])
    return [float(value) for value in np.interp(ROC_FPR_GRID, fpr, tpr)]


def _patient_bootstrap_auroc_ci(label: Any, pred: Any, weight: Any, rng: Any,
                                reps: int) -> tuple[float, float]:
    """Patient-clustered percentile CI of the IPCW-weighted AUROC.

    The cell is deduplicated to one row per patient upstream, so resampling row INDICES is exactly
    the patient-clustered bootstrap; only an index vector is drawn per replicate and no frame is
    ever copied (spec Section 11).
    """
    n = int(np.asarray(label).size)
    if n < MIN_CELL_SIZE or reps <= 0:
        return float("nan"), float("nan")
    estimates = np.empty(int(reps), dtype=float)
    for replicate in range(int(reps)):
        idx = rng.integers(0, n, size=n)
        estimates[replicate] = _weighted_auroc(label[idx], pred[idx], weight[idx])
    finite = estimates[np.isfinite(estimates)]
    if finite.size < 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def _unit_clamp(value: Any) -> float:
    """Clamp a probability-scale statistic into [0, 1]. The weighted Mann-Whitney ratio and its
    bootstrap percentiles can land a few ULP outside the unit interval; np.clip passes NaN through
    so a not-estimable cell stays not estimable."""
    return float(np.clip(value, 0.0, 1.0))


def _procedure_auroc_row(arm: str, cohort: str, procedure: str, target_month: int, n: int,
                         suppressed: bool, reason: str) -> dict[str, Any]:
    """An unscored arm-horizon cell: the row is always emitted, every metric blank.

    ``suppressed`` separates a disclosure suppression (n < MIN_CELL_SIZE) from an arm whose
    predictions are absent from the source, so the page never explains a structural gap as a
    small-cell suppression.
    """
    return {
        "arm": arm, "cohort": cohort, "procedure": procedure, "target_month": int(target_month),
        "n": int(n), "ess": np.nan, "event_rate": np.nan, "auroc": np.nan,
        "auroc_ci_low": np.nan, "auroc_ci_high": np.nan, "brier": np.nan, "cal_in_large": np.nan,
        "powered": False, "suppressed": bool(suppressed), "skipped_reason": reason,
    }


def _procedure_auroc(cfg: SecondaryConfig,
                     frames: Mapping[tuple[str, str], Any]) -> tuple[list[dict[str, Any]],
                                                                     list[dict[str, Any]]]:
    """Per-arm discrimination for P(BMI < 35) at 6 / 12 / 24 months (spec 7.2, the headline).

    Scores the three reporting arms - incretin, and the shared surgical held-out frame subset by
    ``procedure`` into rygb and sleeve - on held-out observed rows only. Predicted probabilities
    come from the calibrated quantile ladder; AUROC, Brier, and calibration-in-the-large are
    IPCW-weighted; the interval is a patient-clustered index-based bootstrap whose seed is derived
    from the arm and horizon, so a cell's CI never depends on iteration order. Every one of the
    nine arm-horizon cells yields a row: cells below MIN_CELL_SIZE are emitted suppressed rather
    than hidden. Returns (csv_rows, roc_curves).
    """
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    reps = int(cfg.bootstrap_replicates)
    for arm, cohort, procedure in PROCEDURE_AUROC_ARMS:
        frame, reason = _arm_view(frames, cohort, "bmi", procedure)
        for month in PROCEDURE_AUROC_MONTHS:
            if reason:
                rows.append(_procedure_auroc_row(arm, cohort, procedure, month, 0, False, reason))
                continue
            cell = frame.loc[(pd.to_numeric(frame["target_month"], errors="coerce") == month)
                             & frame["target_observed"].fillna(False).astype(bool)]
            cell = cell.drop_duplicates("patient_id")
            target = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
            pred = _quantile_ladder_cdf(_quantile_matrix(cell),
                                        np.full(len(cell), BMI_THRESHOLD_BELOW_35))
            weight = _cell_weight(cell)
            keep = np.isfinite(target) & np.isfinite(pred) & np.isfinite(weight) & (weight > 0)
            label = (target < BMI_THRESHOLD_BELOW_35).astype(float)[keep]
            pred, weight = pred[keep], weight[keep]
            n = int(label.size)
            del cell, target, keep
            if n < MIN_CELL_SIZE:
                rows.append(_procedure_auroc_row(
                    arm, cohort, procedure, month, n, True,
                    "cell below the n < 11 disclosure floor" if n
                    else "no held-out observed rows at this horizon"))
                continue
            auroc = _unit_clamp(_weighted_auroc(label, pred, weight))
            seed = int(stable_fraction(f"procedure_auroc|{arm}|{month}", cfg.seed) * (2**32))
            ci_low, ci_high = (_unit_clamp(bound) for bound in _patient_bootstrap_auroc_ci(
                label, pred, weight, np.random.default_rng(seed), reps))
            ess = _effective_sample_size(weight)
            rows.append({
                "arm": arm, "cohort": cohort, "procedure": procedure, "target_month": month,
                "n": n, "ess": ess, "event_rate": float(study.weighted_mean(label, weight)),
                "auroc": auroc, "auroc_ci_low": ci_low, "auroc_ci_high": ci_high,
                "brier": float(study.weighted_mean((pred - label) ** 2, weight)),
                "cal_in_large": float(study.weighted_mean(pred, weight)
                                      - study.weighted_mean(label, weight)),
                "powered": _is_powered(n, ess), "suppressed": False, "skipped_reason": "",
            })
            curves.append({
                "arm": arm, "target_month": month, "n": n, "auroc": auroc,
                "auroc_ci_low": ci_low, "auroc_ci_high": ci_high,
                "event_rate": rows[-1]["event_rate"],
                "tpr": _weighted_roc_curve(label, pred, weight),
            })
    return rows, curves


# ----- per-arm calibration and accuracy (spec 7.1) -----------------------------------------
def _per_arm_prognostic_row(arm: str, cohort: str, outcome: str, target_month: int, n: int,
                            suppressed: bool, reason: str) -> dict[str, Any]:
    """An unscored arm-outcome-horizon cell: the row is always emitted, every metric blank."""
    return {"arm": arm, "cohort": cohort, "outcome": outcome, "target_month": int(target_month),
            "n": int(n), "ess": np.nan, "powered": False, "suppressed": bool(suppressed),
            "skipped_reason": reason}


def _per_arm_prognostic(cfg: SecondaryConfig,
                        frames: Mapping[tuple[str, str], Any]) -> list[dict[str, Any]]:
    """Calibration and accuracy per reporting arm and horizon (spec 7.1 / spec 13's per-arm table).

    Scores the same three arms as the headline discrimination - incretin, and the shared surgical
    held-out frame subset by ``procedure`` into rygb and sleeve - on held-out observed rows, and
    reports the FULL prognostic metric set ``_prognostic_metrics`` computes: CRPS, RMSE, MAE, bias,
    80% and 90% interval coverage, mean 80% width, and the IPCW calibration slope and intercept.
    Every declared arm-outcome-horizon cell in ``study.TARGET_MONTHS`` yields a row, so a horizon
    with no held-out rows is documented rather than silently missing; cells below MIN_CELL_SIZE are
    emitted suppressed. One horizon cell is materialized at a time, never a whole-arm copy.
    """
    rows: list[dict[str, Any]] = []
    for arm, cohort, procedure in PROCEDURE_AUROC_ARMS:
        for outcome in OUTCOMES:
            frame, reason = _arm_view(frames, cohort, outcome, procedure)
            for month in study.TARGET_MONTHS[outcome]:
                if reason:
                    rows.append(_per_arm_prognostic_row(arm, cohort, outcome, month, 0, False, reason))
                    continue
                cell = frame.loc[(pd.to_numeric(frame["target_month"], errors="coerce") == month)
                                 & frame["target_observed"].fillna(False).astype(bool)]
                cell = cell.drop_duplicates("patient_id")
                y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
                matrix = _quantile_matrix(cell)
                weight = _cell_weight(cell)
                keep = (np.isfinite(y) & np.isfinite(weight) & (weight > 0)
                        & np.isfinite(matrix).all(axis=1))
                n = int(np.sum(keep))
                del cell
                if n < MIN_CELL_SIZE:
                    rows.append(_per_arm_prognostic_row(
                        arm, cohort, outcome, month, n, True,
                        "cell below the n < 11 disclosure floor" if n
                        else "no held-out observed rows at this horizon"))
                    continue
                weight = weight[keep]
                ess = _effective_sample_size(weight)
                rows.append({
                    "arm": arm, "cohort": cohort, "outcome": outcome, "target_month": int(month),
                    "n": n, "ess": ess, "powered": _is_powered(n, ess), "suppressed": False,
                    "skipped_reason": "", **_prognostic_metrics(y[keep], matrix[keep], weight),
                })
    return rows


# ----- feature sensitivity of the forecast (spec 10, surrogate read on the frozen store) ----
def _feature_sensitivity_row(arm: str, cohort: str, outcome: str, n: int,
                             reason: str) -> dict[str, Any]:
    """An unscored arm-outcome cell: the row is always emitted, every metric blank."""
    return {"arm": arm, "cohort": cohort, "outcome": outcome, "feature": "",
            "permutation_importance": np.nan, "importance_share": np.nan, "rank": np.nan,
            "n": int(n), "surrogate_r2": np.nan, "skipped_reason": reason}


def _feature_sweep(cfg: SecondaryConfig, X: Any, y: Any, arm: str, name: str,
                   column: int) -> dict[str, Any]:
    """Counterfactual sweep of ONE baseline feature across its own deciles, over the surrogate.

    The surrogate is refit once on the whole cell (a partial-dependence sweep describes a fitted
    response surface, so it is in-sample by construction) and evaluated on a seeded subsample with
    the swept feature's value column overwritten at each grid point. Only that subsample block is
    copied, so the sweep never allocates a second full design matrix, and the block is restored and
    freed before returning.
    """
    model = _cate_regressor(cfg.seed)
    model.fit(X, y)
    n = int(X.shape[0])
    rng = np.random.default_rng(int(cfg.seed))
    index = (np.arange(n) if n <= FEATURE_SWEEP_ROWS
             else np.sort(rng.choice(n, FEATURE_SWEEP_ROWS, replace=False)))
    block = X[index]                       # fancy indexing already hands back a private copy
    values = np.nanpercentile(X[:, column], [100.0 * point for point in FEATURE_SWEEP_GRID])
    original = block[:, column].copy()
    forecast = []
    for value in values:
        block[:, column] = value
        forecast.append(float(np.mean(model.predict(block))))
    block[:, column] = original
    del model, block
    return {"arm": arm, "feature": name, "n_swept": int(index.size),
            "grid": [float(point) for point in FEATURE_SWEEP_GRID],
            "value": [float(value) for value in values], "forecast": forecast}


def _feature_sensitivity(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any],
                         audit: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Spec 10's feature-sensitivity science read, per reporting arm and outcome.

    For each of the three arms the headline AUROC page reports, regress the STORED MEDIAN FORECAST
    at ``FEATURE_SENSITIVITY_MONTH`` on the baseline covariates with the DR-learner's own
    cross-fitted surrogate (``_crossfit_tau``, which returns out-of-fold predictions and per-SOURCE
    permutation importance in one pass over the folds), then sweep the top numeric feature over the
    surrogate for the BMI arms. Every held-out row with a finite median counts: the sensitivity of
    a FORECAST does not depend on that patient's target being observed, so no observed-only filter
    applies here. Cells too thin to cross-fit are emitted as documented skips, never dropped.
    Returns (csv_rows, sweeps).
    """
    numeric_features = set(CATE_NUMERIC_MODIFIERS) | set(CATE_OPTIONAL_NUMERIC_MODIFIERS)
    rows: list[dict[str, Any]] = []
    sweeps: list[dict[str, Any]] = []
    for arm, cohort, procedure in PROCEDURE_AUROC_ARMS:
        for outcome in OUTCOMES:
            frame = frames.get((cohort, outcome))
            if frame is None or frame.empty:
                rows.append(_feature_sensitivity_row(
                    arm, cohort, outcome, 0, f"no held-out {cohort} {outcome} predictions"))
                continue
            if procedure and "procedure" not in frame.columns:
                rows.append(_feature_sensitivity_row(
                    arm, cohort, outcome, 0, "procedure not populated in this source"))
                continue
            select = pd.to_numeric(frame["target_month"], errors="coerce") == FEATURE_SENSITIVITY_MONTH
            if procedure:
                select &= frame["procedure"].astype("string").str.lower().eq(procedure).fillna(False)
            cell = frame.loc[select].drop_duplicates("patient_id")
            median = _quantile_matrix(cell)[:, 3] if len(cell) else np.zeros(0)
            keep = np.isfinite(median)
            cell, median = cell.loc[keep], median[keep]
            n = int(len(cell))
            if n < FEATURE_SENSITIVITY_MIN_N:
                rows.append(_feature_sensitivity_row(
                    arm, cohort, outcome, n,
                    # The reason reaches the page-16 ledger and the CSV, so the count it names is
                    # disclosure-suppressed exactly like the n column beside it.
                    f"cell has {study.display_count(n)} held-out forecasts, below the "
                    f"{FEATURE_SENSITIVITY_MIN_N} the cross-fitted surrogate needs"))
                continue
            X, groups, _absent = _cate_modifier_design(cell, audit)
            surrogate, importance = _crossfit_tau(X, median, cell["patient_id"].to_numpy(),
                                                  cfg.seed, cfg.nuisance_folds, groups)
            spread = float(np.var(median))
            r2 = float(1.0 - np.mean((median - surrogate) ** 2) / spread) if spread > 0 else float("nan")
            positive = np.clip(importance, 0.0, None)
            total = float(positive.sum())
            order = sorted(range(len(groups)), key=lambda item: (-float(positive[item]), groups[item][0]))
            rows.extend({
                "arm": arm, "cohort": cohort, "outcome": outcome, "feature": groups[position][0],
                "permutation_importance": float(importance[position]),
                "importance_share": float(positive[position] / total) if total > 0 else float("nan"),
                "rank": rank + 1, "n": n, "surrogate_r2": r2, "skipped_reason": "",
            } for rank, position in enumerate(order))
            if outcome == "bmi":
                top = next((position for position in order
                            if groups[position][0] in numeric_features), None)
                if top is not None:
                    sweeps.append(_feature_sweep(cfg, X, median, arm, groups[top][0],
                                                 groups[top][1][0]))
            del X, cell, median, surrogate
            gc.collect()
    return rows, sweeps


def _tier4_cell_events(outcome: str, cell: Any, matrix: Any, enabled: set[str]) -> list[dict[str, Any]]:
    """Per-row predicted probability, observed value, and validity mask for every enabled clinical
    event on one (cohort, outcome, target_month) cell. Binary events carry a 0/1 observed label;
    the %EWL event is a continuous summary (Brier/AUROC/reliability do not apply to it)."""
    n = int(matrix.shape[0])
    target = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
    finite_target = np.isfinite(target)
    events: list[dict[str, Any]] = []
    if outcome == "bmi":
        baseline = (pd.to_numeric(cell["baseline_bmi"], errors="coerce").to_numpy(float)
                    if "baseline_bmi" in cell.columns else np.full(n, np.nan))
        if "bmi_lt_35" in enabled:
            pred = _quantile_ladder_cdf(matrix, np.full(n, BMI_THRESHOLD_BELOW_35))
            events.append({"event": "bmi_lt_35", "event_label": "P(BMI < 35)", "kind": "binary",
                           "pred": pred, "obs": (target < BMI_THRESHOLD_BELOW_35).astype(float),
                           "valid": finite_target})
        for tau in TWL_TARGETS:
            key = f"twl_ge_{int(round(tau * 100))}pct"
            if key not in enabled:
                continue
            threshold = baseline * (1.0 - tau)  # target_bmi <= baseline*(1-tau) == >= tau TWL
            pred = _quantile_ladder_cdf(matrix, threshold)
            events.append({"event": key, "event_label": f"P(>={int(round(tau * 100))}% TWL)",
                           "kind": "binary", "pred": pred, "obs": (target <= threshold).astype(float),
                           "valid": finite_target & np.isfinite(baseline) & (baseline > 0)})
        if "ewl_pct" in enabled:
            denom = baseline - EWL_REFERENCE_BMI
            median = matrix[:, 3]
            pred_ewl = np.where(denom > 0, 100.0 * (baseline - median) / denom, np.nan)
            obs_ewl = np.where(denom > 0, 100.0 * (baseline - target) / denom, np.nan)
            events.append({"event": "ewl_pct", "event_label": "%EWL (summary)", "kind": "summary",
                           "pred": pred_ewl, "obs": obs_ewl,
                           "valid": np.isfinite(pred_ewl) & np.isfinite(obs_ewl)})
    else:  # hba1c
        if "insulin" in cell.columns:
            insulin = _tier2_flag_series(cell["insulin"]).map({True: 1.0, False: 0.0}).astype(float).to_numpy()
        else:
            insulin = np.full(n, np.nan)
        if "hba1c_lt_5.7" in enabled:
            pred = _quantile_ladder_cdf(matrix, np.full(n, HBA1C_THRESHOLD_NONDIABETIC))
            events.append({"event": "hba1c_lt_5.7", "event_label": "P(HbA1c < 5.7)", "kind": "binary",
                           "pred": pred, "obs": (target < HBA1C_THRESHOLD_NONDIABETIC).astype(float),
                           "valid": finite_target})
        if "remission_proxy" in enabled:
            off_insulin = insulin == 0.0
            pred_rem = np.where(off_insulin,
                                _quantile_ladder_cdf(matrix, np.full(n, HBA1C_THRESHOLD_REMISSION)), 0.0)
            label_rem = ((target < HBA1C_THRESHOLD_REMISSION) & off_insulin).astype(float)
            events.append({"event": "remission_proxy", "event_label": "P(HbA1c<6.5 & insulin off)",
                           "kind": "binary", "pred": pred_rem, "obs": label_rem,
                           "valid": finite_target & np.isfinite(insulin)})
    return events


def _tier4_enabled_events(audit: Any) -> tuple[set[str], dict[str, str]]:
    """Which threshold events are enabled given the column-population audit, plus a
    skipped_reason per disabled event. P(BMI<35) and P(HbA1c<5.7) need only target_value, so they
    are always enabled; TWL/%EWL need baseline_bmi and the remission proxy needs insulin."""
    enabled = {"bmi_lt_35", "hba1c_lt_5.7"}
    disabled: dict[str, str] = {}
    if audit is None or audit_usable(audit, "baseline_bmi"):
        enabled |= {"twl_ge_5pct", "twl_ge_10pct", "twl_ge_15pct", "ewl_pct"}
    else:
        for key in ("twl_ge_5pct", "twl_ge_10pct", "twl_ge_15pct", "ewl_pct"):
            disabled[key] = "baseline_bmi not populated in this source"
    if audit is None or audit_usable(audit, "insulin"):
        enabled.add("remission_proxy")
    else:
        disabled["remission_proxy"] = "insulin not populated in this source"
    return enabled, disabled


_EVENT_OUTCOME = {
    "bmi_lt_35": "bmi", "twl_ge_5pct": "bmi", "twl_ge_10pct": "bmi", "twl_ge_15pct": "bmi",
    "ewl_pct": "bmi", "hba1c_lt_5.7": "hba1c", "remission_proxy": "hba1c",
}
_EVENT_LABELS = {
    "bmi_lt_35": "P(BMI < 35)", "twl_ge_5pct": "P(>=5% TWL)", "twl_ge_10pct": "P(>=10% TWL)",
    "twl_ge_15pct": "P(>=15% TWL)", "ewl_pct": "%EWL (summary)",
    "hba1c_lt_5.7": "P(HbA1c < 5.7)", "remission_proxy": "P(HbA1c<6.5 & insulin off)",
}


def _tier4_threshold_probabilities(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any],
                                   audit: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                        dict[str, bool]]:
    """Per (arm, outcome, target_month, event) threshold-probability metrics on held-out observed
    rows: IPCW Brier, weighted AUROC, calibration-in-the-large, and the reliability deciles (stored
    for the page). Keyed by the three reporting arms, not by the two source cohorts, so RYGB and
    sleeve are reported separately. Returns (csv_rows, reliability_entries, auroc_computable)."""
    enabled, disabled = _tier4_enabled_events(audit)
    rows: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    auroc_computable: dict[str, bool] = {}
    arm_outcomes: list[tuple[str, str]] = []
    for arm, outcome, frame in _reporting_arm_frames(frames):
        arm_outcomes.append((arm, outcome))
        for target_month, raw_cell in frame.groupby("target_month", sort=True):
            cell = raw_cell.drop_duplicates("patient_id").reset_index(drop=True)
            if cell.empty:
                continue
            matrix = _quantile_matrix(cell)
            observed_mask = cell["target_observed"].fillna(False).astype(bool).to_numpy()
            weight_all = _cell_weight(cell)
            for event in _tier4_cell_events(outcome, cell, matrix, enabled):
                pred = np.asarray(event["pred"], dtype=float)
                obs = np.asarray(event["obs"], dtype=float)
                selected = np.asarray(event["valid"]) & observed_mask & np.isfinite(pred) & np.isfinite(obs)
                n = int(np.sum(selected))
                if n == 0:
                    continue
                weight = weight_all[selected]
                p = pred[selected]
                o = obs[selected]
                ess = _effective_sample_size(weight)
                pred_mean = float(study.weighted_mean(p, weight))
                obs_mean = float(study.weighted_mean(o, weight))
                if event["kind"] == "binary":
                    brier = float(study.weighted_mean((p - o) ** 2, weight))
                    auroc = _weighted_auroc(o, p, weight)
                    key = f"{arm}|{outcome}|{int(target_month)}|{event['event']}"
                    auroc_computable[key] = bool(np.isfinite(auroc))
                else:
                    brier = float("nan")
                    auroc = float("nan")
                powered = _is_powered(n, ess)
                rows.append({
                    "arm": arm, "outcome": outcome, "target_month": int(target_month),
                    "event": event["event"], "event_label": event["event_label"], "n": n,
                    "ess": ess, "brier": brier, "auroc": auroc,
                    "cal_in_large": float(pred_mean - obs_mean), "pred_mean": pred_mean,
                    "obs_mean": obs_mean, "powered": powered, "skipped_reason": "",
                })
                # Reliability points are a per-decile aggregate; only keep them for disclosable
                # cells (n >= 11), and bin so each decile itself clears n >= 11 (up to 10 deciles)
                # so no sub-11 stratum is ever plotted.
                if event["kind"] == "binary" and n >= MIN_CELL_SIZE:
                    decile_bins = int(min(10, max(1, n // MIN_CELL_SIZE)))
                    points = _reliability_deciles(p, o, weight, bins=decile_bins)
                    if points:
                        reliability.append({
                            "arm": arm, "outcome": outcome, "target_month": int(target_month),
                            "event": event["event"], "event_label": event["event_label"], "n": n,
                            "ess": ess, "brier": brier, "auroc": auroc,
                            "cal_in_large": float(pred_mean - obs_mean), "powered": bool(powered),
                            "points": points,
                        })
    # Document disabled events (not-populated covariate) as skipped rows so the CSV is honest.
    for event_key, reason in sorted(disabled.items()):
        target_outcome = _EVENT_OUTCOME[event_key]
        for (arm, outcome) in arm_outcomes:
            if outcome != target_outcome:
                continue
            rows.append({
                "arm": arm, "outcome": outcome, "target_month": np.nan, "event": event_key,
                "event_label": _EVENT_LABELS[event_key], "n": 0, "ess": np.nan, "brier": np.nan,
                "auroc": np.nan, "cal_in_large": np.nan, "pred_mean": np.nan, "obs_mean": np.nan,
                "powered": False, "skipped_reason": reason,
            })
    return rows, reliability, auroc_computable


def _tier4_decision_curves(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any],
                           audit: Any) -> list[dict[str, Any]]:
    """Decision-curve net benefit for the binary threshold events across DECISION_PT_GRID, with
    treat-all and treat-none reference lines, on held-out observed rows (n >= 11 cells only).
    Keyed by the three reporting arms, not by the two source cohorts."""
    enabled, _disabled = _tier4_enabled_events(audit)
    grid = np.asarray(DECISION_PT_GRID, dtype=float)
    rows: list[dict[str, Any]] = []
    for arm, outcome, frame in _reporting_arm_frames(frames):
        for target_month, raw_cell in frame.groupby("target_month", sort=True):
            cell = raw_cell.drop_duplicates("patient_id").reset_index(drop=True)
            if cell.empty:
                continue
            matrix = _quantile_matrix(cell)
            observed_mask = cell["target_observed"].fillna(False).astype(bool).to_numpy()
            weight_all = _cell_weight(cell)
            for event in _tier4_cell_events(outcome, cell, matrix, enabled):
                if event["kind"] != "binary":
                    continue
                pred = np.asarray(event["pred"], dtype=float)
                obs = np.asarray(event["obs"], dtype=float)
                selected = np.asarray(event["valid"]) & observed_mask & np.isfinite(pred) & np.isfinite(obs)
                n = int(np.sum(selected))
                if n < MIN_CELL_SIZE:
                    continue
                weight = weight_all[selected]
                total_weight = float(np.sum(weight))
                if total_weight <= 0:
                    continue
                p = pred[selected]
                o = obs[selected]
                prevalence = float(np.sum(weight * o) / total_weight)
                powered = _is_powered(n, _effective_sample_size(weight))
                for p_t in grid:
                    odds = float(p_t) / (1.0 - float(p_t))
                    treat = p >= p_t
                    true_positive = float(np.sum(weight[treat] * o[treat])) if bool(np.any(treat)) else 0.0
                    false_positive = float(np.sum(weight[treat] * (1.0 - o[treat]))) if bool(np.any(treat)) else 0.0
                    net_benefit = true_positive / total_weight - (false_positive / total_weight) * odds
                    rows.append({
                        "arm": arm, "outcome": outcome, "target_month": int(target_month),
                        "event": event["event"], "event_label": event["event_label"],
                        "p_t": float(round(float(p_t), 4)), "n": n, "net_benefit": float(net_benefit),
                        "nb_treat_all": float(prevalence - (1.0 - prevalence) * odds),
                        "nb_treat_none": 0.0, "powered": bool(powered),
                    })
    return rows


def _tier4_predictability_map(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any],
                              audit: Any) -> list[dict[str, Any]]:
    """Rank every disclosable subgroup stratum (reusing the Tier-2 axes / labelers) by mean 80%
    interval width and by CRPS, WITHIN each outcome family (BMI and HbA1c scales are not pooled),
    flagging strata that are both precise and accurate as reliable."""
    specs = _tier2_axis_specs()
    raw: list[dict[str, Any]] = []
    for arm, outcome, frame in _reporting_arm_frames(frames):
        observed = frame.loc[frame["target_observed"].fillna(False).astype(bool)].copy()
        if observed.empty:
            continue
        for spec in specs:
            axis = spec["axis"]
            if arm not in spec["arms"]:
                continue
            if not all(column in observed.columns for column in spec["columns"]):
                continue
            if not (audit is not None and all(audit_usable(audit, column) for column in spec["columns"])):
                continue
            work = observed.assign(_subgroup=spec["labeler"](observed))
            work["_subgroup"] = work["_subgroup"].astype("string")
            work = work.loc[work["_subgroup"].notna() & (work["_subgroup"].str.len() > 0)]
            if work.empty:
                continue
            for (target_month, subgroup), raw_cell in work.groupby(["target_month", "_subgroup"], sort=True):
                cell = raw_cell.drop_duplicates("patient_id")
                n = int(len(cell))
                if n < MIN_CELL_SIZE:
                    continue
                y = pd.to_numeric(cell["target_value"], errors="coerce").to_numpy(float)
                matrix = _quantile_matrix(cell)
                weight = _cell_weight(cell)
                metrics = _prognostic_metrics(y, matrix, weight)
                ess = _effective_sample_size(weight)
                raw.append({
                    "arm": arm, "outcome": outcome, "target_month": int(target_month),
                    "axis": axis, "subgroup": str(subgroup), "n": n, "ess": ess,
                    "mean_width_80": float(metrics["width_80"]), "crps": float(metrics["crps"]),
                    "powered": _is_powered(n, ess),
                })
    if not raw:
        return []
    frame = pd.DataFrame(raw)
    frame["rank_width"] = 0
    frame["rank_crps"] = 0
    frame["reliable_flag"] = False
    for _outcome, block in frame.groupby("outcome", sort=False):
        index = block.index
        width = pd.to_numeric(block["mean_width_80"], errors="coerce")
        crps = pd.to_numeric(block["crps"], errors="coerce")
        frame.loc[index, "rank_width"] = width.rank(method="min", ascending=True).astype(int)
        frame.loc[index, "rank_crps"] = crps.rank(method="min", ascending=True).astype(int)
        reliable = (width <= float(width.median())) & (crps <= float(crps.median()))
        frame.loc[index, "reliable_flag"] = reliable.fillna(False).to_numpy()
    frame = frame.sort_values(["outcome", "reliable_flag", "rank_width"],
                              ascending=[True, False, True]).reset_index(drop=True)
    return frame.to_dict(orient="records")


def _overlap_horizon_stats(frame: Any, month: int, weight_by_patient: Mapping[tuple[str, int], float],
                           arm: int) -> dict[str, Any]:
    """Overlap-weighted mean/variance of the predicted median (q50) at one horizon for one cohort,
    using the patient-level overlap weights. Returns n, mean, var, and the contributing weights."""
    months = pd.to_numeric(frame["target_month"], errors="coerce")
    sub = frame.loc[months == int(month)].drop_duplicates("patient_id")
    empty = {"n": 0, "mean": float("nan"), "var": float("nan"), "weights": np.asarray([], dtype=float)}
    if sub.empty:
        return empty
    median = _quantile_matrix(sub)[:, 3]
    patient_ids = sub["patient_id"].astype(str).to_numpy()
    weight = np.asarray([weight_by_patient.get((pid, arm), 0.0) for pid in patient_ids], dtype=float)
    keep = np.isfinite(median) & np.isfinite(weight) & (weight > 0)
    median, weight = median[keep], weight[keep]
    if median.size == 0 or float(np.sum(weight)) <= 0:
        return {"n": int(median.size), "mean": float("nan"), "var": float("nan"), "weights": weight}
    mean = float(study.weighted_mean(median, weight))
    variance = float(study.weighted_mean((median - mean) ** 2, weight))
    return {"n": int(median.size), "mean": mean, "var": variance, "weights": weight}


def _tier4_glp1_overlap(cfg: SecondaryConfig, frames: Mapping[tuple[str, str], Any]) -> tuple[
        list[dict[str, Any]], dict[str, Any]]:
    """Strictly prognostic overlap-weighted GLP-1-vs-surgery predicted-trajectory contrast. Fits a
    cohort-membership propensity P(surgery | shared L) on the OVERLAPPING covariate space
    (cross-fitted tree classifier, patient-clustered folds), forms overlap weights p*(1-p), and
    compares the overlap-weighted mean predicted median by horizon per cohort. NOT a treatment
    effect - the caveat is carried in the CSV header and on page 15."""
    rows: list[dict[str, Any]] = []
    trajectory: dict[str, Any] = {}
    for outcome in OUTCOMES:
        surgery = frames.get(("surgery", outcome))
        incretin = frames.get(("incretin", outcome))
        if surgery is None or incretin is None or surgery.empty or incretin.empty:
            continue
        surgery_patients = surgery.drop_duplicates("patient_id")
        incretin_patients = incretin.drop_duplicates("patient_id")
        numeric = [column for column in GLP1_SHARED_NUMERIC
                   if column in surgery_patients.columns and column in incretin_patients.columns]
        categorical = [column for column in GLP1_SHARED_CATEGORICAL
                       if column in surgery_patients.columns and column in incretin_patients.columns]
        if not numeric and not categorical:
            continue
        combined = pd.concat([surgery_patients.assign(_is_surgery=1),
                              incretin_patients.assign(_is_surgery=0)], ignore_index=True)
        try:
            encoder = study.TabularEncoder.fit(combined, numeric=numeric, categorical=categorical)
            L = encoder.transform(combined)
            A = combined["_is_surgery"].to_numpy().astype(int)
            patient_ids = combined["patient_id"].astype(str).to_numpy()
            ps, _degenerate = _crossfit_propensity(L, A, patient_ids, cfg.seed, cfg.nuisance_folds)
        except Exception:  # a degenerate cohort split must not crash the book
            continue
        overlap = ps * (1.0 - ps)
        weight_by_patient = {
            (str(pid), int(arm)): float(w) for pid, w, arm in zip(patient_ids, overlap, A)
        }
        traj_surgery: dict[int, float] = {}
        traj_incretin: dict[int, float] = {}
        months = sorted(
            set(pd.to_numeric(surgery["target_month"], errors="coerce").dropna().astype(int))
            | set(pd.to_numeric(incretin["target_month"], errors="coerce").dropna().astype(int))
        )
        for month in months:
            surgery_stats = _overlap_horizon_stats(surgery, month, weight_by_patient, 1)
            incretin_stats = _overlap_horizon_stats(incretin, month, weight_by_patient, 0)
            n_surgery = int(surgery_stats["n"])
            n_incretin = int(incretin_stats["n"])
            if n_surgery >= MIN_CELL_SIZE and np.isfinite(surgery_stats["mean"]):
                traj_surgery[int(month)] = float(surgery_stats["mean"])
            if n_incretin >= MIN_CELL_SIZE and np.isfinite(incretin_stats["mean"]):
                traj_incretin[int(month)] = float(incretin_stats["mean"])
            if n_surgery < MIN_CELL_SIZE or n_incretin < MIN_CELL_SIZE:
                continue  # disclosure: only emit a contrast when both arms clear n >= 11
            mean_surgery = float(surgery_stats["mean"])
            mean_incretin = float(incretin_stats["mean"])
            pooled_sd = math.sqrt(max((surgery_stats["var"] + incretin_stats["var"]) / 2.0, 0.0)) \
                if (np.isfinite(surgery_stats["var"]) and np.isfinite(incretin_stats["var"])) else float("nan")
            standardized_diff = (float((mean_surgery - mean_incretin) / pooled_sd)
                                 if np.isfinite(pooled_sd) and pooled_sd > 1e-9 else float("nan"))
            combined_weights = np.concatenate([surgery_stats["weights"], incretin_stats["weights"]])
            ess_overlap = (float(tte.weighted_effective_sample_size(combined_weights))
                           if combined_weights.size else float("nan"))
            rows.append({
                "outcome": outcome, "target_month": int(month), "n_surgery": n_surgery,
                "n_incretin": n_incretin, "overlap_weighted_mean_pred_surgery": mean_surgery,
                "overlap_weighted_mean_pred_incretin": mean_incretin,
                "standardized_diff": standardized_diff, "ess_overlap": ess_overlap,
            })
        trajectory[outcome] = {"surgery": traj_surgery, "incretin": traj_incretin}
    return rows, trajectory


# ----- Tier 4 figure pages (13 threshold probabilities, 14 decision curves, 15 map + GLP-1) ---
def _tier4_payload(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    tier4 = data.get("tier4")
    if not isinstance(tier4, Mapping) or str(tier4.get("status")) != "done":
        return None
    return tier4


def _draw_reliability(axis: Any, entry: Mapping[str, Any]) -> None:
    """One IPCW reliability diagram: predicted-vs-observed by decile with the identity line and a
    Brier / AUROC / n annotation. Only decile points with n >= 11 are drawn (disclosure control)."""
    points = [point for point in entry.get("points", []) if int(point.get("n", 0)) >= MIN_CELL_SIZE]
    axis.plot([0, 1], [0, 1], color=study.PALETTE["muted"], lw=0.8, ls=":")
    if len(points) >= 2:
        pred = np.asarray([point["pred"] for point in points], dtype=float)
        obs = np.asarray([point["obs"] for point in points], dtype=float)
        axis.plot(pred, obs, marker="o", ms=4, color=study.PALETTE["blue"])
    else:
        axis.text(0.5, 0.32, "deciles suppressed (n<11)", ha="center", va="center",
                  fontsize=6.0, color=study.PALETTE["muted"])
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Predicted probability", fontsize=7.5)
    axis.set_ylabel("Observed frequency", fontsize=7.5)
    brier = entry.get("brier")
    auroc = entry.get("auroc")
    brier_text = f"Brier {brier:.3f}" if (brier is not None and np.isfinite(brier)) else "Brier n/a"
    auroc_text = f"AUROC {auroc:.3f}" if (auroc is not None and np.isfinite(auroc)) else "AUROC n/a"
    # An arm with no disclosable cell carries no metric annotation at all: the suppression note in
    # the middle of the frame is the whole read, and an "n=0" corner would misdescribe an arm whose
    # cells exist but sit under the floor.
    if entry:
        axis.text(0.03, 0.97, f"{brier_text} | {auroc_text}\nn={study.display_count(entry.get('n', 0))}",
                  transform=axis.transAxes, va="top", ha="left", fontsize=6.2, color=study.PALETTE["ink"])


@register_page(13)
def _page_threshold_probabilities(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                                  title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier4 = _tier4_payload(data)
    if tier4 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    threshold = tier4.get("threshold", {})
    reliability = list(threshold.get("reliability", []))
    table = pd.DataFrame(threshold.get("rows", []))
    if not reliability and table.empty:
        return _tier1_note_page(figure, "Threshold probabilities not estimable in this source")
    positions = [[0.065, 0.52, 0.25, 0.29], [0.385, 0.52, 0.25, 0.29], [0.705, 0.52, 0.25, 0.29]]
    letters = ["A", "B", "C"]
    # One reliability panel per reporting arm: that arm's largest disclosable binary event cell.
    # ``reliability`` is built in a fixed arm/outcome/horizon/event order, so max() breaks ties on
    # the first such cell and the panel choice is deterministic.
    for index, (arm, _cohort, _procedure) in enumerate(PROCEDURE_AUROC_ARMS):
        axis = figure.add_axes(positions[index])
        entry = max((item for item in reliability if str(item.get("arm")) == arm),
                    key=lambda item: (int(item.get("n", 0)), bool(item.get("powered", False))),
                    default=None)
        study.panel_label(axis, letters[index], PROCEDURE_ARM_LABELS[arm] + (
            f", {entry['event_label']} @ {int(entry['target_month'])}mo" if entry else ""))
        # An arm with no disclosable cell keeps its frame and identity line and carries the
        # suppression marker, so the gap reads as suppression, not as a missing panel (page 17).
        _draw_reliability(axis, entry or {})
    ax_table = figure.add_axes([0.065, 0.10, 0.885, 0.32])
    study.panel_label(ax_table, "D", "Threshold-probability metrics (IPCW; n<11 suppressed)")
    if table.empty:
        empty_panel(ax_table, "Not estimable")
    else:
        scored = table.loc[table.get("skipped_reason", pd.Series("", index=table.index)).astype(str) == ""].copy()
        scored = scored if not scored.empty else table
        # Arm-major, largest cell first, capped at five rows per arm so all three reporting arms
        # reach the 15-row window instead of the largest arm filling it on its own.
        order = {arm: index for index, (arm, _cohort, _procedure) in enumerate(PROCEDURE_AUROC_ARMS)}
        scored = scored.assign(arm_order=scored["arm"].map(order)).sort_values(
            ["arm_order", "powered", "n"], ascending=[True, False, False])
        scored = scored.groupby("arm_order", sort=True).head(5)
        columns = ["arm", "outcome", "event", "target_month", "n", "brier", "auroc",
                   "cal_in_large", "pred_mean", "obs_mean"]
        study.draw_compact_table(
            ax_table, scored.loc[:, columns], columns,
            labels=["Arm", "Outcome", "Event", "Target\nmonth", "N", "Brier", "AUROC",
                    "Cal-in-\nlarge", "Pred\nmean", "Obs\nmean"], max_rows=15)
    figure.text(0.065, 0.075, textwrap.fill(
        "Per-row P(event) is read off the predictive quantile ladder (monotone piecewise-linear "
        "CDF). Reliability, Brier, and AUROC are IPCW-weighted over held-out observed rows; the "
        "dotted line is perfect calibration. AUROC is shown only where the observed label has both "
        "classes. Panels A-C are one reporting arm each - incretin, and the shared surgical model "
        "subset by procedure into RYGB and sleeve - showing that arm's largest disclosable event "
        "cell. Prognostic clinical reframing, not a causal effect.", 185),
        fontsize=6.8, color=study.PALETTE["muted"], va="top")
    return figure


def _draw_decision_curve(axis: Any, sub: Any) -> None:
    """Net-benefit curve for one event cell vs the treat-all / treat-none references."""
    sub = sub.sort_values("p_t")
    p_t = pd.to_numeric(sub["p_t"], errors="coerce").to_numpy(float)
    net_benefit = pd.to_numeric(sub["net_benefit"], errors="coerce").to_numpy(float)
    nb_all = pd.to_numeric(sub["nb_treat_all"], errors="coerce").to_numpy(float)
    axis.axhline(0.0, color=study.PALETTE["muted"], lw=0.9, label="Treat none")
    axis.plot(p_t, nb_all, color=study.PALETTE["orange"], lw=1.0, ls="--", label="Treat all")
    axis.plot(p_t, net_benefit, color=study.PALETTE["blue"], lw=1.5, marker="o", ms=3, label="Model")
    axis.set_xlabel("Threshold probability p_t", fontsize=8)
    axis.set_ylabel("Net benefit", fontsize=8)
    finite = np.concatenate([net_benefit[np.isfinite(net_benefit)], nb_all[np.isfinite(nb_all)], [0.0]])
    if finite.size:
        low = float(min(finite.min(), 0.0))
        high = float(max(finite.max(), 0.0))
        pad = 0.06 * (high - low + 1e-6)
        axis.set_ylim(low - pad, high + pad)
    axis.legend(fontsize=6.2, loc="best", frameon=False)


@register_page(14)
def _page_decision_curves(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                          title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier4 = _tier4_payload(data)
    if tier4 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    frame = pd.DataFrame(tier4.get("decision", {}).get("rows", []))
    if frame.empty:
        return _tier1_note_page(figure, "Decision curves not estimable in this source")
    keys = frame[["arm", "target_month", "event", "event_label", "n"]].drop_duplicates()
    # Same three-column geometry as page 13, so the two Tier-4 arm pages read as one spread.
    positions = [[0.065, 0.30, 0.25, 0.52], [0.385, 0.30, 0.25, 0.52], [0.705, 0.30, 0.25, 0.52]]
    letters = ["A", "B", "C"]
    # One net-benefit panel per reporting arm: that arm's largest decision-curve cell. The sort
    # keys after n are the rest of the cell key, so a tie on n resolves deterministically.
    for index, (arm, _cohort, _procedure) in enumerate(PROCEDURE_AUROC_ARMS):
        axis = figure.add_axes(positions[index])
        block = keys.loc[keys["arm"] == arm].sort_values(["n", "event", "target_month"],
                                                         ascending=[False, True, True])
        study.panel_label(axis, letters[index], PROCEDURE_ARM_LABELS[arm] + (
            f", {block.iloc[0]['event_label']} @ {int(block.iloc[0]['target_month'])}mo"
            if not block.empty else ""))
        if block.empty:
            empty_panel(axis, "No cell clears n >= 11")
            continue
        key = block.iloc[0]
        sub = frame.loc[(frame["arm"] == arm) & (frame["target_month"] == key["target_month"])
                        & (frame["event"] == key["event"])]
        _draw_decision_curve(axis, sub)
    figure.text(0.065, 0.22, textwrap.fill(
        "Net benefit NB(p_t) = TP/n - (FP/n) * p_t/(1-p_t) with IPCW-weighted counts. The model "
        "curve is clinically useful over the range of threshold probabilities p_t where it sits "
        "above BOTH the treat-all (dashed) and treat-none (solid grey) reference lines. One panel "
        "per reporting arm - incretin, and the shared surgical model subset by procedure into RYGB "
        "and sleeve - each showing that arm's largest disclosable event cell (n >= 11). This is a "
        "prognostic clinical-utility reframing of the forecasts, not a causal effect.", 185),
        fontsize=6.8, color=study.PALETTE["muted"], va="top")
    return figure


def _draw_glp1_trajectory(axis: Any, trajectory: Mapping[str, Any], outcome: str) -> bool:
    """Overlap-weighted predicted-median trajectory, surgery vs incretin, for one outcome."""
    traj = trajectory.get(outcome) if isinstance(trajectory, Mapping) else None
    if not traj:
        empty_panel(axis, "Not estimable")
        return False
    surgery = traj.get("surgery", {})
    incretin = traj.get("incretin", {})
    plotted = False
    if surgery:
        months = sorted(int(month) for month in surgery)
        axis.plot(months, [surgery[month] for month in months], marker="o", ms=4,
                  color=study.PALETTE["orange"], label="Surgery")
        plotted = True
    if incretin:
        months = sorted(int(month) for month in incretin)
        axis.plot(months, [incretin[month] for month in months], marker="s", ms=4,
                  color=study.PALETTE["blue"], label="Incretin")
        plotted = True
    if not plotted:
        empty_panel(axis, "Not estimable")
        return False
    axis.set_xlabel("Target month", fontsize=8)
    axis.set_ylabel(f"Overlap-wtd predicted {'BMI' if outcome == 'bmi' else 'HbA1c'} median", fontsize=7.2)
    axis.legend(fontsize=6.2, loc="best", frameon=False)
    return True


@register_page(15)
def _page_predictability_and_glp1(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                                  title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier4 = _tier4_payload(data)
    if tier4 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    predictability = pd.DataFrame(tier4.get("predictability", {}).get("rows", []))
    glp1 = tier4.get("glp1", {})
    glp1_frame = pd.DataFrame(glp1.get("rows", []))
    trajectory = glp1.get("trajectory", {})

    ax_map = figure.add_axes([0.055, 0.135, 0.44, 0.685])
    study.panel_label(ax_map, "A", "Who is predictable: strata ranked by 80% width and CRPS")
    if predictability.empty:
        empty_panel(ax_map, "No disclosable subgroup strata")
    else:
        columns = ["arm", "outcome", "axis", "subgroup", "target_month", "n", "mean_width_80",
                   "crps", "reliable_flag"]
        study.draw_compact_table(
            ax_map, predictability.loc[:, columns], columns,
            labels=["Arm", "Outcome", "Axis", "Subgroup", "Target\nmonth", "N", "Width\n80", "CRPS",
                    "Reliable"], max_rows=20)

    ax_traj = figure.add_axes([0.57, 0.54, 0.385, 0.28])
    traj_outcome = "bmi" if (isinstance(trajectory, Mapping) and trajectory.get("bmi")) else (
        next((key for key in OUTCOMES if isinstance(trajectory, Mapping) and trajectory.get(key)), "bmi"))
    study.panel_label(ax_traj, "B", "Overlap-weighted predicted trajectory")
    _draw_glp1_trajectory(ax_traj, trajectory, traj_outcome)

    ax_glp1 = figure.add_axes([0.57, 0.20, 0.385, 0.22])
    study.panel_label(ax_glp1, "C", "GLP-1 vs surgery overlap (not causal)")
    if glp1_frame.empty:
        empty_panel(ax_glp1, "Not estimable")
    else:
        columns = ["outcome", "target_month", "overlap_weighted_mean_pred_surgery",
                   "overlap_weighted_mean_pred_incretin", "standardized_diff", "ess_overlap"]
        show = glp1_frame.copy()
        for column in columns:
            if column not in show.columns:
                show[column] = np.nan
        study.draw_compact_table(
            ax_glp1, show.sort_values(["outcome", "target_month"]).loc[:, columns], columns,
            labels=["Outcome", "Target\nmonth", "Surgery\npred", "Incretin\npred", "Std\ndiff",
                    "Overlap\nESS"], max_rows=10)
    figure.text(0.57, 0.155, textwrap.fill(glp1.get("caveat", GLP1_CAVEAT), 76), fontsize=6.0,
                color=study.PALETTE["muted"], va="top")
    figure.text(0.055, 0.11, textwrap.fill(
        "Strata (Tier-2 axes) ranked WITHIN each outcome family by mean 80% interval width and CRPS "
        "(BMI and HbA1c are never pooled). Reliable = at or below the within-outcome median on BOTH "
        "precision and accuracy. Audits predictability, not effect modification.", 118),
        fontsize=6.2, color=study.PALETTE["muted"], va="top")
    return figure


# ----- Page 17: headline per-procedure AUROC (spec 7.2) ------------------------------------
def _narrow_panel_label(axis: Any, letter: str, title: str) -> None:
    """The book's panel label, with the title nudged clear of the hanging letter.

    study.panel_label offsets the letter by a fraction of the AXES width; on a narrow panel (the
    per-procedure ROC grid, the CATE summary and honesty panels) that offset is smaller than the
    glyph, so the title would start underneath it. The letter keeps the shared styling and only
    the title's x is restated.
    """
    study.panel_label(axis, letter, "")
    axis.set_title(title, loc="left", x=0.11, pad=8, fontweight="bold", color=study.PALETTE["ink"])


def _draw_procedure_roc(axis: Any, curve: Mapping[str, Any] | None, row: Mapping[str, Any]) -> None:
    """One IPCW-weighted ROC panel: chance diagonal, the AUROC with its bootstrap interval inside
    the axes, and a small right-aligned context label. A suppressed cell keeps its frame and
    diagonal and carries a visible marker instead of a curve, so the gap is legible as suppression
    rather than as a missing panel; its n is rendered through the disclosure formatter."""
    axis.plot([0.0, 1.0], [0.0, 1.0], color=study.PALETTE["grid"], lw=0.9, ls=":")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.03, 1.07)  # headroom so a perfect ROC reads as a curve, not as the top spine
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.tick_params(labelsize=7.0)
    tpr = list(curve.get("tpr", [])) if curve else []
    if tpr:
        axis.plot(ROC_FPR_GRID, tpr, color=study.PALETTE["blue"], lw=1.4)
    else:
        axis.text(0.5, 0.55, "Suppressed (n < 11)" if bool(row.get("suppressed")) else "Not estimable",
                  transform=axis.transAxes, ha="center", va="center", fontsize=7.0,
                  color=study.PALETTE["muted"])
    auroc = row.get("auroc")
    if auroc is not None and np.isfinite(auroc):
        low, high = row.get("auroc_ci_low"), row.get("auroc_ci_high")
        interval = (f" ({low:.3f} to {high:.3f})" if low is not None and high is not None
                    and np.isfinite(low) and np.isfinite(high) else "")
        axis.text(0.97, 0.10, f"AUROC {auroc:.3f}{interval}", transform=axis.transAxes,
                  ha="right", va="bottom", fontsize=7.0, color=study.PALETTE["ink"])
    rate = row.get("event_rate")
    context = f"n = {study.display_count(int(row.get('n') or 0))}"
    if rate is not None and np.isfinite(rate) and not bool(row.get("suppressed")):
        context += f" | events {rate:.0%}"
    axis.text(0.97, 0.03, context, transform=axis.transAxes, ha="right", va="bottom",
              fontsize=6.5, color=study.PALETTE["muted"])


@register_page(17)
def _page_per_procedure_auroc(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                              title: str, subtitle: str) -> Any:
    figure = secondary_new_page(number, title, subtitle)
    tier4 = _tier4_payload(data)
    if tier4 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    payload = tier4.get("procedure_auroc", {})
    table = pd.DataFrame(payload.get("rows", []))
    if table.empty:
        return _tier1_note_page(figure, "Per-procedure discrimination not estimable in this source")
    cells = {(str(row["arm"]), int(row["target_month"])): row for row in payload.get("rows", [])}
    curves = {(str(item["arm"]), int(item["target_month"])): item for item in payload.get("curves", [])}
    # ROC grid left, metric table right. The panels are square in PHYSICAL size (the page is
    # 11 x 8.5 in, so 11 * width = 8.5 * height) to keep the chance diagonal at a true 45 degrees.
    # Beside the table the grid is width-bound rather than height-bound, so the square is the
    # widest one that still leaves the right margin a legible six-column table; columns are
    # gutter-spaced at 28% of a panel.
    panel_w = 0.1559
    panel_h = round(panel_w * 11.0 / 8.5, 4)
    gutter = round(0.28 * panel_w, 4)
    columns_x = tuple(round(0.115 + index * (panel_w + gutter), 4) for index in range(3))
    rows_y = tuple(round(0.847 - panel_h - index * (panel_h + 0.038), 4) for index in range(3))
    letters = "ABCDEFGHI"
    for row_index, month in enumerate(PROCEDURE_AUROC_MONTHS):
        for column_index, (arm, _cohort, _procedure) in enumerate(PROCEDURE_AUROC_ARMS):
            axis = figure.add_axes([columns_x[column_index], rows_y[row_index], panel_w, panel_h])
            _narrow_panel_label(axis, letters[row_index * len(PROCEDURE_AUROC_ARMS) + column_index],
                                   f"{PROCEDURE_ARM_LABELS[arm]}, {month} months")
            _draw_procedure_roc(axis, curves.get((arm, month)), cells.get((arm, month), {}))
            if row_index == len(PROCEDURE_AUROC_MONTHS) - 1:
                axis.set_xlabel("False positive rate", fontsize=8.0)
            else:  # shared axis: only the bottom row is annotated, so the rows can sit closer
                axis.set_xticklabels([])
                axis.tick_params(axis="x", length=0)
            if column_index == 0:
                axis.set_ylabel("True positive rate", fontsize=8.0)
            else:
                axis.set_yticklabels([])
    ax_table = figure.add_axes([0.700, 0.377, 0.245, 0.470])
    study.panel_label(ax_table, "J", "Per-arm metrics by horizon")
    # The six columns spec 7.2 names. ess / powered / the CI bounds stay in per_procedure_auroc.csv;
    # the CI is annotated in every panel and suppression still reads as "<11" in N.
    columns = ["arm", "target_month", "n", "auroc", "brier", "cal_in_large"]
    ordered = table.sort_values(["arm", "target_month"]).loc[:, columns].copy()
    ordered["arm"] = ordered["arm"].map(lambda value: PROCEDURE_ARM_LABELS.get(str(value), str(value)))
    study.draw_compact_table(
        ax_table, ordered, columns,
        # "Mo" not "Target month": draw_compact_table gives target_month a narrow fixed share and
        # clips overflow, and renaming the column to win a wider share would cost the structural
        # label its small-cell protection. The panel J title names the horizon.
        labels=["Arm", "Mo", "N", "AUROC", "Brier", "Cal-in-\nlarge"],
        max_rows=len(PROCEDURE_AUROC_ARMS) * len(PROCEDURE_AUROC_MONTHS))
    figure.text(0.700, 0.347, textwrap.fill(
        "P(BMI < 35) is read off the calibrated predictive quantile ladder. AUROC, Brier, and "
        "calibration-in-the-large (mean predicted minus mean observed) are IPCW-weighted over "
        "held-out observed rows; the interval in each panel is a patient-clustered percentile "
        "bootstrap. The two surgical arms subset one shared model by procedure, so RYGB and sleeve "
        "are separately calibrated but not separately fitted. Prognostic discrimination, not a "
        "causal effect.", 62),
        fontsize=6.5, color=study.PALETTE["muted"], va="top")
    return figure


def _cate_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """The CATE block the TTE stage attached to its checkpoint (empty when the stage never ran)."""
    return (data.get("tte") or {}).get("cate") or {}


def _draw_toc_curve(axis: Any, cate: Mapping[str, Any]) -> None:
    """The primary CATE validation figure: TOC with its patient-clustered bootstrap band, RATE."""
    grid = np.asarray(cate.get("toc_grid") or [], dtype=float)
    toc = np.asarray(cate.get("toc") or [], dtype=float)
    if grid.size == 0 or toc.size != grid.size:
        empty_panel(axis, "TOC not estimable")
        return
    low = np.asarray(cate.get("toc_ci_low") or [], dtype=float)
    high = np.asarray(cate.get("toc_ci_high") or [], dtype=float)
    if low.size == grid.size and high.size == grid.size:
        axis.fill_between(grid, low, high, color=study.PALETTE["blue"], alpha=0.16, lw=0,
                          label="95% patient-clustered bootstrap")
    axis.axhline(0.0, color=study.PALETTE["muted"], lw=0.8, ls=":")
    axis.plot(grid, toc, color=study.PALETTE["blue"], lw=1.6, label="TOC")
    axis.set_xlabel("Fraction treated, highest predicted benefit first", fontsize=8)
    axis.set_ylabel("Benefit above population mean (BMI)", fontsize=8)
    axis.legend(fontsize=6.3, loc="upper right", frameon=False)
    rate, low_rate, high_rate = cate.get("rate"), cate.get("rate_ci_low"), cate.get("rate_ci_high")
    if rate is not None and np.isfinite(rate):
        interval = (f" ({low_rate:.3f} to {high_rate:.3f})"
                    if low_rate is not None and high_rate is not None
                    and np.isfinite(low_rate) and np.isfinite(high_rate) else "")
        axis.text(0.03, 0.05, f"RATE {rate:.3f}{interval} BMI units", transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=7.2, color=study.PALETTE["ink"])


def _draw_benefit_calibration(axis: Any, cate: Mapping[str, Any]) -> None:
    """Predicted-benefit decile against the AIPW effect actually observed in that decile.

    Both axes are BMI benefit units, so the identity line is the calibration target; deciles below
    the disclosure floor are dropped rather than plotted.
    """
    rows = [row for row in (cate.get("calibration") or [])
            if int(row.get("n") or 0) >= MIN_CELL_SIZE
            and np.isfinite(row.get("aipw_benefit", np.nan))]
    if len(rows) < 2:
        empty_panel(axis, "Benefit calibration not estimable")
        return
    predicted = np.asarray([row["mean_predicted_benefit"] for row in rows], dtype=float)
    observed = np.asarray([row["aipw_benefit"] for row in rows], dtype=float)
    low = np.asarray([row["ci_low"] for row in rows], dtype=float)
    high = np.asarray([row["ci_high"] for row in rows], dtype=float)
    error = np.vstack([np.nan_to_num(observed - low), np.nan_to_num(high - observed)])
    axis.errorbar(predicted, observed, yerr=np.clip(error, 0.0, None), fmt="o", ms=4.5, lw=0.0,
                  elinewidth=0.9, capsize=2, color=study.PALETTE["blue"],
                  ecolor=study.PALETTE["grid"], zorder=3)
    span = [float(np.nanmin([predicted.min(), low.min()])), float(np.nanmax([predicted.max(), high.max()]))]
    axis.plot(span, span, color=study.PALETTE["muted"], lw=0.8, ls=":", zorder=1)
    axis.set_xlabel("Predicted benefit, decile mean (BMI)", fontsize=8)
    axis.set_ylabel("Observed AIPW benefit (BMI)", fontsize=8)
    rho, monotone = cate.get("calibration_rho"), cate.get("monotone_fraction")
    read = f"{len(rows)} of {CATE_DECILES} deciles estimable"
    if rho is not None and np.isfinite(rho):
        read += f"\nrank correlation {rho:+.2f}"
    if monotone is not None and np.isfinite(monotone):
        read += f"\n{monotone:.0%} of steps increase"
    # Inside the axes at lower right: the panel title already owns the space above the frame, and
    # the deciles run bottom-left to top-right, so this corner is the one that is always clear.
    axis.text(0.97, 0.04, read, transform=axis.transAxes, ha="right", va="bottom", fontsize=6.6,
              linespacing=1.4, color=study.PALETTE["muted"])


def _cate_summary_lines(cate: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The c-for-benefit and policy-value read, in the order the summary panel prints it."""
    policy, treat_all = cate.get("policy_value"), cate.get("treat_all_value")
    gain = (policy - treat_all if policy is not None and treat_all is not None
            and np.isfinite(policy) and np.isfinite(treat_all) else float("nan"))
    return [
        ("Patients scored", study.display_count(int(cate.get("n") or 0))),
        ("c-for-benefit", _cate_number(cate.get("c_for_benefit"), 3)),
        ("Matched pairs", study.display_count(int(cate.get("n_pairs") or 0))),
        ("Treat-all value", _cate_number(treat_all, 3)),
        ("Treat-none value", "0.000"),
        ("Policy value", _cate_number(policy, 3)),
        ("Policy gain vs treat-all", _cate_number(gain, 3)),
        ("Fraction sent to RYGB", _cate_number(cate.get("treated_fraction"), 2)),
    ]


def _cate_number(value: Any, digits: int) -> str:
    return f"{float(value):.{digits}f}" if value is not None and np.isfinite(value) else "not estimable"


def _draw_summary_column(axis: Any, letter: str, title: str, lines: Sequence[tuple[str, str]],
                         footnote: str) -> None:
    """A label/value readout column: muted labels left, bold values right, footnote at the foot."""
    _narrow_panel_label(axis, letter, title)
    axis.axis("off")
    for position, (label, value) in enumerate(lines):
        y = 0.96 - 0.105 * position   # eight rows, ending clear of the two-line unit note below
        axis.text(0.0, y, label, fontsize=7.0, color=study.PALETTE["muted"], va="top",
                  transform=axis.transAxes)
        axis.text(1.0, y, value, fontsize=7.6, fontweight="bold", color=study.PALETTE["ink"],
                  ha="right", va="top", transform=axis.transAxes)
    axis.text(0.0, 0.0, footnote, fontsize=6.2, color=study.PALETTE["muted"], va="bottom",
              transform=axis.transAxes)


def _draw_cate_summary(axis: Any, cate: Mapping[str, Any]) -> None:
    """Policy value of treating the predicted benefiters, against treat-all and treat-none."""
    _draw_summary_column(axis, "D", "Concordance and policy", _cate_summary_lines(cate),
                         "Values are mean BMI reduction per patient in the\npopulation, from the "
                         "same pseudo-outcomes.")


def _cate_honesty_panel(axis: Any, letter: str, cate: Mapping[str, Any], width: int,
                        caveat: str = CATE_CAVEAT, structural: str = "surgeon_or_center_id",
                        tail: str = CAUSAL_FOREST_NOTE) -> None:
    """The spec 8.3 honesty gate, carried on the page: what this map is not, which modifiers were
    unavailable, and why the optional causal-forest cross-check is absent. The which-incretin read
    on page 07 reuses it with its own caveat, its own structural residual, and no forest note."""
    # This panel is narrow beside the summary and full width in the not-estimable layout, and only
    # the narrow one needs its title nudged clear of the hanging letter.
    (_narrow_panel_label if axis.get_position().width < 0.30 else study.panel_label)(
        axis, letter, "Honesty gate")
    axis.axis("off")
    # The same naming M5 uses for L: whatever the audit refused, then the structural identifier
    # this source never carries. Each is omitted from X rather than encoded as an empty column.
    absent = list(cate.get("absent_modifiers") or []) + [structural]
    named = ("Unmeasured or unpopulated here, so absent from X and able to masquerade as effect "
             "modification: " + ", ".join(absent) + ".")
    body = " ".join(part for part in (caveat, named, tail) if part)
    axis.text(0.0, 0.99, textwrap.fill(body, width),
              va="top", ha="left", fontsize=6.6, linespacing=1.32, color=study.PALETTE["ink"],
              transform=axis.transAxes)


@register_page(18)
def _page_cate_heterogeneous_benefit(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                                     title: str, subtitle: str) -> Any:
    """Spec 8.3: RATE / TOC with inference, benefit calibration against the observed AIPW effect,
    the permutation-importance modifier ranking, c-for-benefit and policy value, under the
    heterogeneity-map honesty gate. A cell without propensity overlap renders the graceful
    not-estimable state the TTE pages use, never a curve fitted through a degenerate ratio."""
    figure = secondary_new_page(number, title, subtitle)
    cate = _cate_payload(data)
    if str(cate.get("status")) != "done":
        axis = figure.add_axes([0.075, 0.42, 0.87, 0.42])
        empty_panel(axis, textwrap.fill(
            f"CATE not estimable: {cate['skipped_reason']}" if cate.get("skipped_reason")
            else "CATE not populated in this source", 78))
        _cate_honesty_panel(figure.add_axes([0.075, 0.11, 0.87, 0.22]), "A", cate, 178)
        return figure

    ax_toc = figure.add_axes([0.075, 0.505, 0.375, 0.325])
    study.panel_label(ax_toc, "A", "RATE / TOC: do the top-ranked really benefit more")
    _draw_toc_curve(ax_toc, cate)
    ax_cal = figure.add_axes([0.565, 0.505, 0.375, 0.325])
    study.panel_label(ax_cal, "B", "Benefit calibration by predicted-benefit decile")
    _draw_benefit_calibration(ax_cal, cate)

    ax_modifiers = figure.add_axes([0.075, 0.115, 0.300, 0.295])
    study.panel_label(ax_modifiers, "C", "Modifiers driving tau_hat")
    _draw_modifier_importance(ax_modifiers, cate.get("importance"), 8)
    _draw_cate_summary(figure.add_axes([0.430, 0.115, 0.190, 0.295]), cate)
    _cate_honesty_panel(figure.add_axes([0.700, 0.115, 0.245, 0.295]), "E", cate, 56)
    figure.text(0.075, 0.086, textwrap.fill(
        "DR-learner: the doubly-robust pseudo-outcome built from the target trial's own "
        f"cross-fitted propensity, censoring, and outcome nuisances for the {cate.get('outcome')} "
        f"{int(cate.get('target_month') or 0)}-month cell, regressed on the measured modifiers with "
        "a cross-fitted gradient-boosted model over the same patient folds. Benefit is stated as "
        "BMI reduction from RYGB rather than sleeve, so a positive number favors RYGB.", 178),
        fontsize=6.5, color=study.PALETTE["muted"], va="top")
    return figure


ARM_COLORS = (study.PALETTE["blue"], study.PALETTE["sky"], study.PALETTE["green"])


def _draw_sensitivity_bars(axis: Any, rows: Sequence[Mapping[str, Any]], outcome: str) -> None:
    """Top source features by surrogate permutation-importance share, one bar group per arm.

    Features are ranked by their share POOLED over the arms so all three read the same rows, and
    an arm that never scored a feature shows an absent bar rather than a fabricated zero rank.
    """
    scored = [row for row in rows if str(row.get("outcome")) == outcome
              and str(row.get("feature")) and np.isfinite(row.get("importance_share", np.nan))]
    if not scored:
        empty_panel(axis, "Surrogate not estimable")
        return
    pooled: dict[str, float] = {}
    for row in scored:
        pooled[str(row["feature"])] = pooled.get(str(row["feature"]), 0.0) + float(row["importance_share"])
    features = [name for name, _share in
                sorted(pooled.items(), key=lambda item: (-item[1], item[0]))][:FEATURE_SENSITIVITY_TOP_N]
    share = {(str(row["arm"]), str(row["feature"])): float(row["importance_share"]) for row in scored}
    # Only arms this outcome actually scored get a bar series: an arm whose cell fell below the
    # surrogate floor is absent from the panel, never drawn as a row of honest-looking zeros.
    arms = [(position, arm) for position, (arm, _cohort, _procedure) in enumerate(PROCEDURE_AUROC_ARMS)
            if any(key[0] == arm for key in share)]
    y = np.arange(len(features))[::-1].astype(float)
    height = 0.78 / len(arms)
    for offset, (position, arm) in enumerate(arms):
        axis.barh(y + (offset - (len(arms) - 1) / 2.0) * height,
                  [share.get((arm, name), np.nan) for name in features], height=height,
                  color=ARM_COLORS[position], alpha=0.9, label=PROCEDURE_ARM_LABELS[arm])
    axis.set_yticks(y)
    axis.set_yticklabels([_feature_label(name) for name in features], fontsize=6.8)
    axis.set_xlabel("Share of surrogate permutation importance", fontsize=8)
    axis.legend(fontsize=6.3, loc="lower right", frameon=False)


def _draw_sensitivity_sweep(axis: Any, sweeps: Sequence[Mapping[str, Any]]) -> None:
    """Counterfactual sweep: the mean surrogate-predicted BMI as each arm's most influential
    baseline feature is moved across its own deciles, every other covariate held as observed.

    The arms swing different features over different units, so the shared x axis is the swept
    feature's own percentile and the legend names the feature each line belongs to.
    """
    arms = {arm: position for position, (arm, _cohort, _procedure) in enumerate(PROCEDURE_AUROC_ARMS)}
    drawn = 0
    for item in sorted(sweeps, key=lambda entry: arms.get(str(entry.get("arm")), 99)):
        grid = np.asarray(item.get("grid") or [], dtype=float)
        forecast = np.asarray(item.get("forecast") or [], dtype=float)
        arm = str(item.get("arm"))
        if grid.size == 0 or forecast.size != grid.size or arm not in arms:
            continue
        axis.plot(grid, forecast, marker="o", ms=3.2, lw=1.5, color=ARM_COLORS[arms[arm]],
                  label=f"{PROCEDURE_ARM_LABELS[arm]}: {_feature_label(str(item.get('feature')))}")
        drawn += 1
    if not drawn:
        empty_panel(axis, "Counterfactual sweep not estimable")
        return
    axis.set_xlabel("Percentile of the swept baseline feature", fontsize=8)
    axis.set_ylabel(f"Mean predicted BMI at {FEATURE_SENSITIVITY_MONTH} mo", fontsize=8)
    axis.legend(fontsize=6.3, loc="best", frameon=False)


def _sensitivity_top_rows(rows: Sequence[Mapping[str, Any]]) -> Any:
    """One row per (arm, outcome): its top-ranked feature, or a visible not-estimable marker.

    The full reason a cell was not scored stays in forecast_feature_sensitivity.csv and in the
    page-16 omissions ledger; truncating it into a table cell would render it unreadable, so the
    table says only that the cell is not estimable and points at the sources that say why.
    """
    best: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("skipped_reason")) or int(row.get("rank") or 0) == 1:
            best[(str(row.get("arm")), str(row.get("outcome")))] = row
    ordered = []
    for arm, _cohort, _procedure in PROCEDURE_AUROC_ARMS:
        for outcome in OUTCOMES:
            row = best.get((arm, outcome))
            if row is None:
                continue
            ordered.append({
                "arm": PROCEDURE_ARM_LABELS.get(arm, arm), "outcome": outcome,
                "n": int(row.get("n") or 0), "surrogate_r2": row.get("surrogate_r2"),
                "feature": _feature_label(str(row.get("feature"))) if row.get("feature") else "not estimable",
                "importance_share": row.get("importance_share"),
            })
    return pd.DataFrame(ordered)


@register_page(19)
def _page_feature_sensitivity(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                              title: str, subtitle: str) -> Any:
    """Spec 10's feature-sensitivity science page. The frozen store has no trained model to
    re-run, so sensitivity is read off a cross-fitted surrogate of the stored median forecast:
    permutation importance per source feature (A, B) and a counterfactual sweep of the top feature
    over that surrogate (C), with the surrogate's own out-of-fold fidelity beside every read (D)."""
    figure = secondary_new_page(number, title, subtitle)
    tier4 = _tier4_payload(data)
    if tier4 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    payload = tier4.get("feature_sensitivity", {})
    rows = list(payload.get("rows", []))
    if not rows:
        return _tier1_note_page(figure, "Forecast feature sensitivity not estimable in this source")
    month = int(payload.get("target_month") or FEATURE_SENSITIVITY_MONTH)
    ax_bmi = figure.add_axes([0.115, 0.500, 0.340, 0.295])
    study.panel_label(ax_bmi, "A", f"BMI forecast at {month} months")
    _draw_sensitivity_bars(ax_bmi, rows, "bmi")
    ax_hba1c = figure.add_axes([0.605, 0.500, 0.340, 0.295])
    study.panel_label(ax_hba1c, "B", f"HbA1c forecast at {month} months")
    _draw_sensitivity_bars(ax_hba1c, rows, "hba1c")

    ax_sweep = figure.add_axes([0.115, 0.165, 0.335, 0.245])
    study.panel_label(ax_sweep, "C", "Counterfactual sweep of the top feature")
    _draw_sensitivity_sweep(ax_sweep, list(payload.get("sweeps", [])))
    ax_table = figure.add_axes([0.530, 0.165, 0.415, 0.245])
    study.panel_label(ax_table, "D", "Top feature and surrogate fidelity by arm")
    columns = ["arm", "outcome", "n", "feature", "importance_share", "surrogate_r2"]
    study.draw_compact_table(
        ax_table, _sensitivity_top_rows(rows), columns,
        labels=["Arm", "Outcome", "N", "Top feature", "Share", "Surrogate\nR2"], max_rows=6)
    # One line taller than the note it replaced, so it starts one line higher and keeps the same
    # clearance above the page footer.
    figure.text(0.075, 0.108, textwrap.fill(FEATURE_SENSITIVITY_NOTE + " " + FEATURE_SWEEP_NOTE, 188),
                fontsize=6.5, color=study.PALETTE["muted"], va="top")
    return figure


# ----- Page 20: per-arm calibration and accuracy (spec 7.1) --------------------------------
# Single-line headers: a full-height 22-row table gives the header row the same height as a data
# row, which is too short for the book's usual two-line column labels.
PER_ARM_TABLE_COLUMNS = ("arm", "target_month", "n", "ess", "crps", "rmse", "mae", "bias",
                         "coverage_80", "coverage_90", "width_80", "cal_slope")
PER_ARM_TABLE_LABELS = ["Arm", "Mo", "N", "ESS", "CRPS", "RMSE", "MAE", "Bias", "80% cov.",
                        "90% cov.", "80% width", "Cal. slope"]


def _draw_per_arm_table(axis: Any, frame: Any, outcome: str) -> None:
    """The full per-arm metric set for one outcome, arm-major and horizon-ordered. n < 11 cells
    keep their row: draw_compact_table renders the count as "<11" and blanks that row's metrics."""
    sub = frame.loc[frame["outcome"].astype(str) == outcome].copy()
    if sub.empty:
        empty_panel(axis, "Not estimable")
        return
    sub["arm_order"] = sub["arm"].map({arm: index for index, (arm, _cohort, _procedure)
                                       in enumerate(PROCEDURE_AUROC_ARMS)})
    sub = sub.sort_values(["arm_order", "target_month"])
    sub["arm"] = sub["arm"].map(lambda value: PROCEDURE_ARM_LABELS.get(str(value), str(value)))
    study.draw_compact_table(axis, sub.loc[:, list(PER_ARM_TABLE_COLUMNS)],
                             list(PER_ARM_TABLE_COLUMNS), labels=PER_ARM_TABLE_LABELS,
                             max_rows=len(PROCEDURE_AUROC_ARMS) * len(study.TARGET_MONTHS[outcome]))


@register_page(20)
def _page_per_arm_calibration(cfg: SecondaryConfig, data: Mapping[str, Any], number: int,
                              title: str, subtitle: str) -> Any:
    """Spec 7.1's per-arm calibration and accuracy table: the complete prognostic metric set for
    each of the three reporting arms at every declared horizon, one table per outcome. The pooled
    two-cohort read is deliberately not shown - RYGB and sleeve share one fitted model but are
    separately calibrated, and pooling them hides exactly that."""
    figure = secondary_new_page(number, title, subtitle)
    tier4 = _tier4_payload(data)
    if tier4 is None:
        return _tier1_note_page(figure, PENDING_NOTE)
    frame = pd.DataFrame(tier4.get("per_arm_prognostic", {}).get("rows", []))
    if frame.empty:
        return _tier1_note_page(figure, "Per-arm calibration not estimable in this source")
    ax_bmi = figure.add_axes([0.045, 0.450, 0.915, 0.390])
    study.panel_label(ax_bmi, "A", "BMI: calibration and accuracy by arm and horizon")
    _draw_per_arm_table(ax_bmi, frame, "bmi")
    ax_hba1c = figure.add_axes([0.045, 0.135, 0.915, 0.275])
    study.panel_label(ax_hba1c, "B", "HbA1c: calibration and accuracy by arm and horizon")
    _draw_per_arm_table(ax_hba1c, frame, "hba1c")
    figure.text(0.045, 0.108, textwrap.fill(PER_ARM_PROGNOSTIC_NOTE, 188),
                fontsize=6.5, color=study.PALETTE["muted"], va="top")
    return figure


def page_renderers(cfg: SecondaryConfig, data: Mapping[str, Any]) -> list[Callable[[], Any]]:
    """One renderer per PAGE_FILES entry, order-locked. Registered builders replace placeholders."""
    renderers: list[Callable[[], Any]] = []
    for number in range(len(SECONDARY_PAGE_FILES)):
        title, subtitle = PAGE_TITLES[number]
        builder = PAGE_BUILDERS.get(number)
        if builder is None:
            renderers.append(lambda number=number, title=title, subtitle=subtitle:
                             _placeholder_page(number, title, subtitle))
        else:
            renderers.append(
                lambda builder=builder, number=number, title=title, subtitle=subtitle:
                _safe_build(builder, cfg, data, number, title, subtitle)
            )
    return renderers


def _safe_build(builder: Callable[..., Any], cfg: SecondaryConfig, data: Mapping[str, Any],
                number: int, title: str, subtitle: str) -> Any:
    """Render a page, degrading to a visible note (never crashing the book) on any page error."""
    try:
        return builder(cfg, data, number, title, subtitle)
    except Exception as error:  # a single page must never break the whole figure book
        traceback.print_exc()
        return _placeholder_page(number, title, subtitle, f"Page could not be rendered: {error!r}")


def _validate_export(export: Path, require_complete: bool = False) -> None:
    expected = set(SECONDARY_PAGE_FILES) | {FIGURE_BOOK_PDF}
    present = {item.name for item in export.iterdir() if item.is_file()}
    unexpected = present.difference(expected | {FAILURE_PNG})
    if unexpected:
        raise RuntimeError("FIGURES_TO_EXPORT contains non-contract files: " + ", ".join(sorted(unexpected)))
    if require_complete:
        missing = expected.difference(present)
        if missing:
            raise RuntimeError("FIGURES_TO_EXPORT is missing contract files: " + ", ".join(sorted(missing)))


def render_figure_book(cfg: SecondaryConfig, data: Mapping[str, Any]) -> list[Path]:
    """Render the numbered PNGs + one bound PDF (one page per PNG, in order), enforcing the contract.

    Mirrors ``study.render_figure_book``: validate before, save each PNG atomically while binding
    the same figure into the open PdfPages, then validate completeness after.
    """
    study.configure_figure_style()
    export = cfg.export_dir
    export.mkdir(parents=True, exist_ok=True)
    # Clear any stale failure PNG from a prior aborted attempt so the contract check is clean.
    stale = export / FAILURE_PNG
    if stale.exists():
        stale.unlink()
    _validate_export(export)
    pdf_temporary = export / (FIGURE_BOOK_PDF + ".tmp")
    pdf_final = export / FIGURE_BOOK_PDF
    written: list[Path] = []
    with PdfPages(pdf_temporary, metadata={"Title": "Metabolic Trajectory Secondary Analyses",
                                           "Author": "Brannigan Lab"}) as pdf:
        for filename, render in zip(SECONDARY_PAGE_FILES, page_renderers(cfg, data), strict=True):
            figure = render()
            stamp_provenance(figure, data)
            temporary = export / (filename + ".tmp")
            figure.savefig(temporary, format="png", dpi=300, facecolor=figure.get_facecolor())
            study.replace_file(temporary, export / filename)
            pdf.savefig(figure, dpi=300, facecolor=figure.get_facecolor())
            plt.close(figure)
            written.append(export / filename)
    study.replace_file(pdf_temporary, pdf_final)
    written.append(pdf_final)
    _validate_export(export, require_complete=True)
    return written


def stage_render(cfg: SecondaryConfig) -> dict[str, Any]:
    """Assemble figure data from stage checkpoints, render the book, write manifest + bundle."""
    data = _collect_figure_data(cfg)
    render_figure_book(cfg, data)
    write_manifest(cfg, data)
    write_bundle(cfg)
    _save_checkpoint(cfg, "render", {"status": "done", "pages": list(SECONDARY_PAGE_FILES)})
    return {"status": "done"}


def _collect_figure_data(cfg: SecondaryConfig) -> dict[str, Any]:
    """Gather everything the pages need from the per-stage checkpoints."""
    data: dict[str, Any] = {
        "mode": cfg.mode,
        "config_hash": config_hash(cfg),
        "secondary_version": SECONDARY_VERSION,
        "source_study_version": SOURCE_STUDY_VERSION,
    }
    for stage in ("assemble", "tte", "tier1", "tier2", "tier3", "tier4"):
        path = cfg.checkpoints_dir / f"{stage}.pkl"
        if path.exists():
            with open(path, "rb") as stream:
                data[stage] = pickle.load(stream)
    return data


# --------------------------------------------------------------------------------------------
# Manifest and results bundle
# --------------------------------------------------------------------------------------------
def _stage_peak_rss(cfg: SecondaryConfig) -> dict[str, float]:
    """Per-stage peak RSS for the manifest, read back from each stage's checkpoint metadata.

    Section 11 wants the probe in the manifest so a memory regression is visible in the returned
    artifacts rather than only in a console log nobody kept. Every stage already wrote its own
    peak beside its checkpoint, from inside its own process, so this holds whether the run was
    orchestrated or in-process. ``render`` is measured live: it is the stage writing this file,
    and by now it has done the whole figure book, which is its expensive part.
    """
    peaks: dict[str, float] = {}
    for stage in STAGE_SEQUENCE:
        value = (study.read_json(cfg.checkpoints_dir / f"{stage}.json", {}) or {}).get("peak_rss_gb")
        if value is not None:
            peaks[stage] = float(value)
    live = _peak_rss_gb()
    if live is not None:
        peaks["render"] = live
    return peaks


def write_manifest(cfg: SecondaryConfig, data: Mapping[str, Any]) -> None:
    assemble = data.get("assemble", {})
    audit = assemble.get("audit")
    audit_summary = []
    if audit is not None:
        audit_summary = [
            {k: (float(v) if isinstance(v, float) else v) for k, v in row.items()}
            for row in audit.to_dict(orient="records")
        ]

    # Per-analysis gate-outcome summary pulled from every tier checkpoint (tte + tier1..tier4):
    # positivity outcomes, the powered-vs-all robustness verdicts, and which axes were usable or
    # skipped. Plus the aggregated omissions ledger (M6 stores an omissions list; surface it here).
    tte_summary = _tte_summary(data)
    cate = _cate_payload(data)
    tier1 = data.get("tier1") or {}
    tier2 = data.get("tier2") or {}
    tier3 = data.get("tier3") or {}
    verdicts = [str(row.get("verdict", "")) for row in tier3.get("powered_only", [])]
    gate_summary = {
        "families": _analysis_family_summary(data),
        "tte": {
            "status": tte_summary["status"],
            "skip_reason": tte_summary["skip_reason"],
            "n_cells": tte_summary["n_cells"],
            "n_estimable": tte_summary["n_estimable"],
            "n_positivity_fail": tte_summary["n_positivity_fail"],
            "n_powered": tte_summary["n_powered"],
            "e_value_low": _json_float(tte_summary["e_value_low"]),
            "e_value_high": _json_float(tte_summary["e_value_high"]),
            "outcome_model": tte_summary["outcome_model"],
            "confounders": (data.get("tte") or {}).get("confounders") or {},
            "absent_confounders": _tte_absent_confounders(data),
        },
        "cate": {
            "status": str(cate.get("status", "absent")), "n": cate.get("n"),
            "rate": _json_float(cate.get("rate", float("nan"))),
            "absent_modifiers": list(cate.get("absent_modifiers") or []),
            "skipped_reason": str(cate.get("skipped_reason", "")),
        },
        "robustness_verdicts": [
            {"arm": row.get("arm"), "outcome": row.get("outcome"), "verdict": row.get("verdict")}
            for row in tier3.get("powered_only", [])
        ],
        "robustness_verdict_summary": _verdict_summary(verdicts),
        "transportability_axes_usable": (tier1.get("transportability") or {}).get("axes_usable", {}),
        "subgroup_axes_usable": tier2.get("axes_usable", {}),
        "tier3_baseline_window_usable": bool((tier3.get("baseline_window") or {}).get("usable", False)),
        "tier3_cluster_bootstrap_usable": bool((tier3.get("cluster_bootstrap") or {}).get("usable", False)),
        "eligibility_cohort_n": (tier3.get("eligibility") or {}).get("cohort_n", {}),
    }
    omissions = _collect_omissions(data)

    manifest = {
        "secondary_version": SECONDARY_VERSION,
        "source_study_version": SOURCE_STUDY_VERSION,
        "script_sha256": script_sha256(),
        "config_hash": config_hash(cfg),
        "mode": cfg.mode,
        "seed": cfg.seed,
        "source_run_fingerprint": source_run_fingerprint(cfg),
        "dependencies": {
            "python": platform.python_version(),
            "numpy": getattr(np, "__version__", None),
            "pandas": getattr(pd, "__version__", None),
            "parquet_spill_enabled": study._PARQUET_SPILL_ENABLED,
        },
        "column_population_audit": audit_summary,
        "join_rate": assemble.get("join_rate"),
        "arm_counts": assemble.get("arm_counts"),
        "tte_gate_ok": assemble.get("tte_gate_ok"),
        "tte_skip_reason": assemble.get("tte_skip_reason"),
        "analysis_status": {stage: (data.get(stage, {}) or {}).get("status", "absent")
                            for stage in ("tte", "tier1", "tier2", "tier3", "tier4")},
        "tte_outcome_model": (data.get("tte", {}) or {}).get("outcome_model"),
        "estimand_note": ESTIMAND_NOTE,
        "gate_summary": gate_summary,
        "omissions": omissions,
        "upstream_changes_requested": [],
        "peak_rss_gb_by_stage": _stage_peak_rss(cfg),
        "wall_clock_utc": study.utc_now(),
    }
    study.atomic_json(cfg.secondary_dir / "manifest.json", manifest)


def write_bundle(cfg: SecondaryConfig) -> None:
    """Zip the figure book, all CSVs, the manifest, and logs into one returnable bundle.

    Omits the large per-row frames and the covariate parquet, per the spec. The file list is
    collected BEFORE the archive is opened, and the temporary archive is written under the
    (excluded) checkpoints directory, so the growing bundle can never scan or add itself.
    """
    bundle_path = cfg.secondary_dir / "secondary_analyses_results_bundle.zip"
    exclude_names = {bundle_path.name, "covariate_frame.parquet", "covariate_frame.pkl"}
    members: list[tuple[Path, str]] = []
    for item in sorted(cfg.secondary_dir.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(cfg.secondary_dir)
        if relative.parts[0] == "checkpoints":
            continue
        if item.name in exclude_names or item.name.endswith(".tmp"):
            continue
        members.append((item, str(relative)))
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    temporary = cfg.checkpoints_dir / "results_bundle.building.zip"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item, arcname in members:
            archive.write(item, arcname=arcname)
    study.replace_file(temporary, bundle_path)


# --------------------------------------------------------------------------------------------
# Stage orchestration, resume, and process-per-stage execution
# --------------------------------------------------------------------------------------------
STAGE_FUNCTIONS: dict[str, Callable[[SecondaryConfig], Any]] = {
    "assemble": stage_assemble,
    "tte": stage_tte,
    "tier1": stage_tier1,
    "tier2": stage_tier2,
    "tier3": stage_tier3,
    "tier4": stage_tier4,
    "render": stage_render,
}


def run_stage(cfg: SecondaryConfig, stage: str) -> None:
    STAGE_FUNCTIONS[stage](cfg)


def run_all_stages(cfg: SecondaryConfig, dependencies: Mapping[str, Any]) -> Path:
    """Run the full stage graph either in-process or one-subprocess-per-stage.

    Orchestrated (the default for --from-run/--full) bounds peak RSS to the single worst stage by
    running each stage in its own worker process and freeing memory between stages, mirroring the
    production pattern. --smoke stays single-process for speed unless --orchestrate is passed.
    """
    cfg.secondary_dir.mkdir(parents=True, exist_ok=True)
    statuses = {stage: "pending" for stage in STAGE_SEQUENCE}
    orchestrate = cfg.orchestrate or (cfg.mode in {"from-run", "full"})
    update_progress_image(cfg, statuses)
    for stage in STAGE_SEQUENCE:
        if cfg.resume and _checkpoint_valid(cfg, stage) and stage != "render":
            statuses[stage] = "done"
            update_progress_image(cfg, statuses)
            print(f"[secondary] resume: skipping completed stage {stage}", flush=True)
            continue
        statuses[stage] = "running"
        update_progress_image(cfg, statuses)
        print(f"[secondary] stage: {stage}", flush=True)
        if orchestrate:
            _run_stage_subprocess(cfg, stage)
        else:
            run_stage(cfg, stage)
        statuses[stage] = "done"
        update_progress_image(cfg, statuses)
        study.log_peak_rss(f"secondary:{stage}")
        gc.collect()
    return cfg.run_dir


def _run_stage_subprocess(cfg: SecondaryConfig, stage: str) -> None:
    """Invoke this script as a single-stage worker so the stage's peak RSS is reclaimed on exit."""
    command = [
        sys.executable, str(_THIS_FILE), "--worker", stage,
        "--worker-mode", cfg.mode, "--output-dir", cfg.output_dir,
        "--seed", str(cfg.seed),
        "--incretin-qualifying-months", str(cfg.incretin_qualifying_months),
        "--mi-imputations", str(cfg.mi_imputations),
    ]
    if cfg.from_run:
        command += ["--from-run", cfg.from_run]
    if cfg.skip_tte:
        command += ["--skip-tte"]
    if cfg.loso_refit:
        command += ["--loso-refit"]
    if cfg.neural_outcome_model:
        command += ["--neural-outcome-model"]
    result = subprocess.run(command, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"secondary worker stage {stage} failed with exit code {result.returncode}")


# --------------------------------------------------------------------------------------------
# Plot-only
# --------------------------------------------------------------------------------------------
def run_plot_only(cfg: SecondaryConfig) -> Path:
    """Rebuild the figure book from this run's own stage checkpoints."""
    if not cfg.checkpoints_dir.exists():
        raise SecondaryPreflightError(
            "No secondary checkpoints to plot from",
            [f"Expected stage checkpoints under {cfg.checkpoints_dir}"],
        )
    stage_render(cfg)
    return cfg.run_dir


# --------------------------------------------------------------------------------------------
# Self-test harness (deterministic, torch-free, no Cosmos)
# --------------------------------------------------------------------------------------------
def run_self_tests() -> dict[str, Any]:
    """Deterministic embedded unit tests on synthetic fixtures.

    Uses the production numbered-check convention (fail-fast). Unlike the production self-test,
    this one does NOT require torch, so it runs on the minimal local interpreter.
    """
    load_runtime()
    results: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append({"test": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail or 'assertion failed'}")

    # 01 - normal ppf approximation is close to a known value.
    check("01_normal_ppf", abs(float(_normal_ppf(np.array([0.975]))[0]) - 1.959964) < 1e-3,
          "Phi^{-1}(0.975) ~ 1.96")

    # 05 - small-cell suppression blanks n < 11 cells.
    frame = pd.DataFrame({"cell": ["a", "b"], "n": [5, 50], "metric": [1.0, 2.0]})
    suppressed = suppress(frame, ["n"])
    small_metric = suppressed.loc[suppressed["cell"] == "a", "metric"].iloc[0]
    check("05_small_cell_suppression", bool(pd.isna(small_metric)),
          "metric for n=5 cell is blanked")

    # 06 - an all-null covariate is marked not usable by the population audit.
    covariates = pd.DataFrame({
        "patient_id": [str(i) for i in range(20)],
        "cohort": ["surgery"] * 20,
        "state": [None] * 20,
        "svi": [0.1 * (i % 5) for i in range(20)],
    })
    audit = column_population_audit(covariates, ["state", "svi"])
    check("06_audit_all_null_not_usable",
          (not audit_usable(audit, "state")) and audit_usable(audit, "svi"),
          "all-null state not usable; varied svi usable")

    # 07 - figure-book contract: emitted PNG set == PAGE_FILES and the PDF binds one page per PNG.
    with tempfile.TemporaryDirectory(prefix="secondary-selftest-") as directory:
        cfg = SecondaryConfig(mode="self-test", output_dir=directory)
        data = {"mode": "self-test", "config_hash": config_hash(cfg)}
        written = render_figure_book(cfg, data)
        names = {item.name for item in cfg.export_dir.iterdir() if item.is_file()}
        pdf_pages = _count_pdf_pages(cfg.export_dir / FIGURE_BOOK_PDF)
        check("07_figure_book_contract",
              names == set(SECONDARY_PAGE_FILES) | {FIGURE_BOOK_PDF}
              and pdf_pages == len(SECONDARY_PAGE_FILES),
              f"{len(written)} artifacts, {pdf_pages} pdf pages")

    # 08 - E-value and SMD->RR match hand-computed values on a fixture.
    rr = tte.smd_to_rr(0.25)
    ev = tte.e_value(rr)
    expected_rr = math.exp(0.91 * 0.25)
    expected_ev = expected_rr + math.sqrt(expected_rr * (expected_rr - 1.0))
    check("08_e_value_and_smd_to_rr",
          abs(rr - expected_rr) < 1e-9 and abs(ev["e_point"] - expected_ev) < 1e-9,
          f"rr={rr:.4f} e={ev['e_point']:.4f}")

    # 10 - reopen_prediction_store round-trips a synthesized pickle store.
    with tempfile.TemporaryDirectory(prefix="secondary-store-") as directory:
        root = Path(directory) / "calibrated"
        store = study.PredictionStore(root)
        frame = _fixture_prediction_frame()
        store.add(frame, key=["surgery", "bmi", 0])
        reopened = reopen_prediction_store(root)
        back = reopened.read(["surgery", "bmi", 0], columns=["patient_id", "target_value"])
        check("10_store_reopen_roundtrip",
              reopened.keys() == [("surgery", "bmi", 0)] and len(back) == len(frame),
              f"{reopened.keys()} rows={len(back)}")

    # 12 - config hash is stable across identical configs (determinism precondition).
    with tempfile.TemporaryDirectory(prefix="secondary-hash-") as directory:
        a = SecondaryConfig(mode="smoke", output_dir=directory, seed=7)
        b = SecondaryConfig(mode="smoke", output_dir=directory, seed=7)
        check("12_config_hash_deterministic", config_hash(a) == config_hash(b),
              "identical configs share a config hash")

    # 02 - AIPW recovers a planted ATE under known confounding, and stays correct when EITHER the
    # propensity model OR the outcome model is deliberately misspecified (double robustness).
    generator = np.random.default_rng(20240807)
    n_dgp = 6000
    confounder = generator.normal(size=n_dgp)
    ps_true = 1.0 / (1.0 + np.exp(-0.8 * confounder))
    arm_dgp = (generator.random(n_dgp) < ps_true).astype(int)
    tau = -3.0  # true ATE: RYGB lowers the outcome by 3 units
    outcome_dgp = 40.0 + tau * arm_dgp + 2.0 * confounder + generator.normal(scale=0.5, size=n_dgp)
    delta_dgp = np.ones(n_dgp)
    pc_true = np.ones(n_dgp)
    mu1_true = 40.0 + tau + 2.0 * confounder
    mu0_true = 40.0 + 2.0 * confounder
    both = tte.aipw(outcome_dgp, arm_dgp, delta_dgp, ps_true, pc_true, mu1_true, mu0_true)["ate"]
    constant_mu = np.full(n_dgp, float(np.mean(outcome_dgp)))
    ps_only = tte.aipw(outcome_dgp, arm_dgp, delta_dgp, ps_true, pc_true, constant_mu, constant_mu)["ate"]
    mu_only = tte.aipw(outcome_dgp, arm_dgp, delta_dgp, np.full(n_dgp, 0.5), pc_true, mu1_true, mu0_true)["ate"]
    check("02_aipw_recovers_ate_double_robust",
          abs(both - tau) < 0.25 and abs(ps_only - tau) < 0.4 and abs(mu_only - tau) < 0.25,
          f"both={both:.3f} ps_only={ps_only:.3f} mu_only={mu_only:.3f} (tau={tau})")

    # 03 - the positivity gate passes on healthy overlap and fires on a near-degenerate arm where
    # one procedure dominates each propensity extreme (min PS <= 0.01, trimmed ESS collapses).
    healthy = positivity_gate(np.full(400, 0.5), np.tile([0, 1], 200))
    degenerate_ps = np.concatenate([np.full(120, 0.005), np.full(80, 0.99)])
    degenerate_arm = np.concatenate([np.zeros(120, dtype=int), np.ones(80, dtype=int)])
    degenerate = positivity_gate(degenerate_ps, degenerate_arm)
    check("03_positivity_gate_fires",
          (not healthy["positivity_fail"]) and degenerate["positivity_fail"],
          f"healthy ess={healthy['ess_iptw']:.1f} fail={healthy['positivity_fail']}; "
          f"degenerate min_ps={degenerate['min_ps']:.3f} fail={degenerate['positivity_fail']}")

    # 04 - the delta tipping-point arm-contrast is MONOTONE in delta and delta_star has the
    # correct sign (spec self-test 3). Controlled fixture: RYGB observed 38, sleeve observed 40,
    # censored (half of each arm) sit at the predicted median 40, so ATE(0) = -1 (RYGB favored)
    # and, since RYGB-censored are shifted +delta and sleeve-censored -delta, the crossing must be
    # at a POSITIVE delta = +1. Monotone non-decreasing by construction.
    grid = tuple(float(value) for value in range(-6, 7))
    arm_fixture = np.array([1] * 20 + [0] * 20)
    observed_fixture = np.array([True] * 10 + [False] * 10 + [True] * 10 + [False] * 10)
    median_fixture = np.full(40, 40.0)
    base_y_fixture = np.where(arm_fixture == 1, 38.0, 40.0)
    weight_fixture = np.ones(40)
    curve, ate0, delta_star = tipping_point_effect_curve(
        observed_fixture, median_fixture, arm_fixture, weight_fixture, grid, base_y_fixture)
    monotone = bool(np.all(np.diff(curve) >= -1e-9))
    sign_ok = bool(ate0 < 0.0 and delta_star > 0.0 and abs(delta_star - 1.0) < 1e-6)
    check("04_tipping_point_monotone_sign", monotone and sign_ok,
          f"ate0={ate0:.3f} delta_star={delta_star:.3f} monotone={monotone}")

    # 13 - threshold-probability calibration is ~identity on a synthetic WELL-CALIBRATED generator
    # and the Brier score sits at its theoretical floor (spec self-test 4). Draw true outcomes from
    # per-row normals, set the 7 predicted quantiles to each row's TRUE quantiles, then read
    # P(Y < c) off the quantile-ladder CDF at a fixed threshold. A perfectly calibrated forecaster
    # has reliability curve = identity and Brier = mean p(1-p) (the irreducible Bernoulli variance).
    calibration_rng = np.random.default_rng(13131313)
    n_calibration = 20000
    sigma = 1.0
    threshold = 40.0
    mu = calibration_rng.uniform(threshold - 2.0, threshold + 2.0, size=n_calibration)
    y_true = calibration_rng.normal(mu, sigma)
    z = _normal_ppf(np.asarray(QUANTILE_LEVELS, dtype=float))
    ladder = mu[:, None] + sigma * z[None, :]  # each row's TRUE quantiles
    predicted = _quantile_ladder_cdf(ladder, np.full(n_calibration, threshold))
    label = (y_true < threshold).astype(float)
    ones = np.ones(n_calibration)
    reliability_points = _reliability_deciles(predicted, label, ones)
    reliability_gap = (float(np.mean([abs(point["pred"] - point["obs"]) for point in reliability_points]))
                       if reliability_points else 1.0)
    brier = float(study.weighted_mean((predicted - label) ** 2, ones))
    p_true = np.array([0.5 * (1.0 + math.erf(float(arg))) for arg in (threshold - mu) / (sigma * math.sqrt(2.0))])
    brier_floor = float(np.mean(p_true * (1.0 - p_true)))
    auroc = _weighted_auroc(label, predicted, ones)
    check("13_threshold_probability_calibration",
          reliability_gap < 0.03 and abs(brier - brier_floor) < 0.02
          and len(reliability_points) >= 8 and 0.5 < auroc <= 1.0,
          f"reliability_gap={reliability_gap:.4f} brier={brier:.4f} floor={brier_floor:.4f} "
          f"deciles={len(reliability_points)} auroc={auroc:.3f}")

    # 09 - Determinism (spec self-test 9): the same seed yields identical metric AND figure hashes.
    # Metric determinism seeds a patient-clustered bootstrap twice; figure determinism renders the
    # whole book via render_figure_book into ONE temp dir twice and compares the PNG byte hashes.
    # Rendering into the same directory keeps every path-dependent provenance string (page 01) fixed,
    # so a byte difference is a true nondeterminism, not a temp-path artifact; the bound PDF embeds a
    # creation timestamp, so only the scored PNG pages are hashed (fast; no full smoke is run here).
    resid_fixture = np.linspace(-2.0, 2.0, 60)
    weight_fixture = np.ones(60)
    ids_fixture = np.array([f"P{i // 2}" for i in range(60)])  # two rows per patient
    ci_first = _patient_bootstrap_rmse_ci(resid_fixture, weight_fixture, ids_fixture,
                                          np.random.default_rng(SEED), 128)
    ci_second = _patient_bootstrap_rmse_ci(resid_fixture, weight_fixture, ids_fixture,
                                           np.random.default_rng(SEED), 128)
    metric_hash_first = hashlib.sha256(repr(tuple(round(v, 12) for v in ci_first)).encode()).hexdigest()
    metric_hash_second = hashlib.sha256(repr(tuple(round(v, 12) for v in ci_second)).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="secondary-determinism-") as directory:
        cfg = SecondaryConfig(mode="self-test", output_dir=directory)
        figure_data = {"mode": "self-test", "config_hash": config_hash(cfg)}

        def _png_bundle_hash(export: Path) -> str:
            joined = b"".join((export / name).read_bytes() for name in SECONDARY_PAGE_FILES)
            return hashlib.sha256(joined).hexdigest()

        render_figure_book(cfg, figure_data)
        figure_hash_first = _png_bundle_hash(cfg.export_dir)
        render_figure_book(cfg, figure_data)
        figure_hash_second = _png_bundle_hash(cfg.export_dir)
    check("09_determinism_metric_and_figure",
          ci_first == ci_second and metric_hash_first == metric_hash_second
          and figure_hash_first == figure_hash_second,
          f"metric={metric_hash_first[:10]} figure={figure_hash_first[:10]}")

    # 11 - Join integrity (spec self-test 10): _prediction_join_rate reaches the planted overlap
    # rate on a synthetic (store, covariate-frame) pair. A 30-patient surgery store; the covariate
    # frame covers exactly 24 of them (plus two non-store patients), so the rate must be 24/30 = 0.8.
    with tempfile.TemporaryDirectory(prefix="secondary-join-") as directory:
        root = Path(directory) / "calibrated"
        store = study.PredictionStore(root)
        store.add(_fixture_prediction_frame(), key=["surgery", "bmi", 0])
        covered = [f"P{i}" for i in range(24)]
        covariates = pd.DataFrame({
            "patient_id": covered + ["Z0", "Z1"],
            "cohort": (["surgery"] * 24) + ["incretin", "surgery"],
        })
        join_rate, join_detail = _prediction_join_rate(store, covariates)
        check("11_join_integrity_patient_id",
              abs(join_rate - 24.0 / 30.0) < 1e-9
              and int(join_detail["distinct_prediction_patients"]) == 30
              and int(join_detail["joined"]) == 24,
              f"rate={join_rate:.4f} joined={join_detail['joined']}/{join_detail['distinct_prediction_patients']}")

    # 14 - the headline per-procedure AUROC matches a HAND-COMPUTED value, subsets the three arms
    # correctly, and suppresses every n < 11 arm-horizon cell. The ladder is built so the BMI<35
    # threshold lands exactly on a stored quantile, making each row's predicted probability an
    # exact QUANTILE_LEVELS entry: baseline medians 33/34/35/36/37 give 0.90/0.75/0.50/0.25/0.10.
    # Twelve unit-weight patients per estimable arm (six with target 34 -> label 1, six with 36 ->
    # label 0) give the weighted Mann-Whitney sum 6+6+6+6+5.5+4 = 33.5 over 6*6 discordant pairs.
    def _auroc_fixture(medians: Sequence[float], targets: Sequence[float],
                       procedures: Sequence[str] | None) -> Any:
        ladder = (np.asarray(medians, dtype=float)[:, None]
                  + np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])[None, :])
        frame = pd.DataFrame({
            "patient_id": [f"F{index}" for index in range(len(medians))],
            "target_month": [12] * len(medians), "target_observed": [True] * len(medians),
            "target_value": list(targets), "analysis_weight": [1.0] * len(medians),
        })
        for index, column in enumerate(QUANTILE_COLS):
            frame[column] = ladder[:, index]
        if procedures is not None:
            frame["procedure"] = list(procedures)
        return frame

    scored_medians = [33.0, 33.0, 34.0, 34.0, 35.0, 36.0, 35.0, 36.0, 36.0, 37.0, 37.0, 37.0]
    scored_targets = ([34.0] * 6) + ([36.0] * 6)  # below / above the BMI 35 threshold
    auroc_cfg = SecondaryConfig(mode="self-test", output_dir=".", bootstrap_replicates=64)
    auroc_rows, auroc_curves = _procedure_auroc(auroc_cfg, {
        ("incretin", "bmi"): _auroc_fixture(scored_medians, scored_targets, None),
        ("surgery", "bmi"): _auroc_fixture(
            scored_medians + [33.0, 34.0, 35.0, 36.0, 37.0],
            scored_targets + [34.0, 34.0, 36.0, 36.0, 36.0],
            (["rygb"] * 12) + (["sleeve"] * 5)),
    })
    repeat_rows, _repeat_curves = _procedure_auroc(auroc_cfg, {
        ("incretin", "bmi"): _auroc_fixture(scored_medians, scored_targets, None),
        ("surgery", "bmi"): _auroc_fixture(
            scored_medians + [33.0, 34.0, 35.0, 36.0, 37.0],
            scored_targets + [34.0, 34.0, 36.0, 36.0, 36.0],
            (["rygb"] * 12) + (["sleeve"] * 5)),
    })
    absent_rows, _absent_curves = _procedure_auroc(auroc_cfg, {})  # no predictions for any arm
    by_cell = {(row["arm"], row["target_month"]): row for row in auroc_rows}
    incretin_cell, rygb_cell, sleeve_cell = (by_cell[("incretin", 12)], by_cell[("rygb", 12)],
                                             by_cell[("sleeve", 12)])
    small = [row for row in auroc_rows if int(row["n"]) < MIN_CELL_SIZE]
    check("14_per_procedure_auroc_and_suppression",
          len(auroc_rows) == len(PROCEDURE_AUROC_ARMS) * len(PROCEDURE_AUROC_MONTHS)
          and abs(incretin_cell["auroc"] - 33.5 / 36.0) < 1e-9
          and abs(rygb_cell["auroc"] - 33.5 / 36.0) < 1e-9
          and int(incretin_cell["n"]) == 12 and int(rygb_cell["n"]) == 12
          and abs(incretin_cell["brier"] - 1.3625 / 12.0) < 1e-9
          and abs(incretin_cell["cal_in_large"] - (5.35 / 12.0 - 0.5)) < 1e-9
          and int(sleeve_cell["n"]) == 5 and bool(sleeve_cell["suppressed"])
          and not np.isfinite(sleeve_cell["auroc"])
          and len(small) == 7 and all(bool(row["suppressed"]) for row in small)
          and not any(np.isfinite(row["auroc"]) for row in small)
          and len(auroc_curves) == 2 and len(auroc_curves[0]["tpr"]) == len(ROC_FPR_GRID)
          # an arm absent from the source is reported as not estimable, never as a suppression
          and len(absent_rows) == 9 and not any(bool(row["suppressed"]) for row in absent_rows)
          and all(str(row["skipped_reason"]) for row in absent_rows)
          and repr(repeat_rows) == repr(auroc_rows),
          f"auroc={incretin_cell['auroc']:.6f} (=33.5/36) rygb_n={int(rygb_cell['n'])} "
          f"sleeve_n={int(sleeve_cell['n'])} suppressed={len(small)}/9 curves={len(auroc_curves)}")

    # 15 - the enriched confounder set is audit-gated (Section 9). Creatinine/eGFR are populated so
    # they join L and the balance/positivity re-audit covers them; GERD and the lab panel are absent
    # from this source, so they are OMITTED from L (never an all-NaN confounder) and come back as
    # named absent confounders. The fixture keeps sex non-collinear with the arm so the re-audited
    # positivity gate is a real read rather than the smoke bundle's degenerate one.
    n = 120
    enriched = pd.DataFrame({
        "patient_id": [f"E{index}" for index in range(n)],
        "cohort": ["surgery"] * n,
        "procedure": ["rygb" if index % 2 else "sleeve" for index in range(n)],
        "baseline_bmi": [40.0 + (index % 7) for index in range(n)],
        "age_at_index": [35.0 + (index % 20) for index in range(n)],
        "sex": ["F" if index % 3 else "M" for index in range(n)],
        "creatinine_baseline": [0.7 + 0.01 * (index % 9) for index in range(n)],
        "egfr_baseline": [90.0 - (index % 25) for index in range(n)],
        "gerd_baseline": [None] * n,
    })
    enriched_audit = column_population_audit(enriched)
    numeric_L, categorical_L, absent_L = _tte_confounder_lists(enriched_audit)
    L, A, encoded, _enc = build_wide_L_A(enriched, "bmi", enriched_audit)
    ps, degenerate = _crossfit_propensity(L, A, enriched["patient_id"].to_numpy(), SEED, NUISANCE_FOLDS)
    gate = positivity_gate(ps, A, degenerate=degenerate)
    smd = dict(zip(encoded, tte.standardized_mean_diff(L, A)))
    all_nan = [name for name, column in zip(encoded, np.asarray(L, dtype=float).T)
               if not np.isfinite(column).any()]
    named = _tte_absent_confounders({"assemble": {"audit": enriched_audit}})
    check("15_enriched_L_audit_gated",
          {"creatinine_baseline", "egfr_baseline"}.issubset(numeric_L)
          and {"creatinine_baseline", "egfr_baseline"}.issubset(encoded)
          and "gerd_baseline" not in numeric_L and "gerd_baseline" not in encoded
          # exactly the optional confounders this fixture leaves unpopulated, in a fixed order
          and absent_L == sorted(set(TTE_OPTIONAL_NUMERIC_CONFOUNDERS
                                     + TTE_OPTIONAL_CATEGORICAL_CONFOUNDERS)
                                 - {"creatinine_baseline", "egfr_baseline"})
          and not all_nan
          and named == absent_L + ["surgeon_or_center_id"]
          and np.isfinite(smd["creatinine_baseline"]) and np.isfinite(smd["egfr_baseline"])
          and len(smd) == L.shape[1] and int(gate["n"]) == n and np.isfinite(gate["ess_iptw"]),
          f"L={len(numeric_L)}n+{len(categorical_L)}c -> {L.shape[1]} encoded; absent={len(absent_L)}; "
          f"ess={gate['ess_iptw']:.1f} min_ps={gate['min_ps']:.4f}")

    # 16 - the DR-learner recovers a KNOWN CATE (spec 12). The fixture randomizes the arm inside
    # each (BMI level, fold) stratum (ps = 0.5, exact overlap) and plants tau(x) =
    # -(1 + 0.3 (BMI - 40)), so RYGB helps the heavier patient more, while the outcome nuisance is
    # handed the MARGINAL effect only (mu1 - mu0 = mean tau for everyone) and so contributes no
    # effect modification at all: nothing but the augmentation term can carry the heterogeneity.
    # Recovery is judged against the planted tau, the modifier that drives it must top the
    # permutation importance, an unpopulated modifier must be named absent rather than encoded,
    # and a cell without propensity overlap must refuse to fit at all.
    fixture = _cate_fixture()
    cate_cfg = SecondaryConfig(mode="self-test", output_dir=".", bootstrap_replicates=200)
    psi = _dr_pseudo_outcome(fixture["Y"], fixture["A"], fixture["delta"], fixture["ps"],
                             fixture["pc"], fixture["mu1"], fixture["mu0"])
    design, groups, absent_modifiers = _cate_modifier_design(fixture["rows"], fixture["audit"])
    tau_hat, importance = _crossfit_tau(design, psi, fixture["patient_ids"], SEED, NUISANCE_FOLDS,
                                        groups)
    recovery = float(np.corrcoef(tau_hat, fixture["tau"])[0, 1])
    top_modifier = groups[int(np.argmax(importance))][0]
    arguments = (cate_cfg, fixture["rows"], fixture["audit"], fixture["Y"], fixture["A"],
                 fixture["delta"], fixture["ps"], fixture["pc"], fixture["mu1"], fixture["mu0"],
                 fixture["patient_ids"])
    planted = _cate_dr_learner(*arguments, {"positivity_fail": False, "min_ps": 0.5, "ess_iptw": 800.0})
    blocked = _cate_dr_learner(*arguments, {"positivity_fail": True, "min_ps": 1e-6, "ess_iptw": 3.0})
    check("16_dr_learner_recovers_known_cate",
          planted["status"] == "done" and recovery > 0.7
          and abs(float(np.mean(tau_hat)) - float(np.mean(fixture["tau"]))) < 0.5
          and top_modifier == "baseline_bmi"
          and absent_modifiers == sorted(CATE_OPTIONAL_NUMERIC_MODIFIERS)
          and planted["absent_modifiers"] == absent_modifiers
          and not np.isnan(np.asarray(design, dtype=float)).all(axis=0).any()
          and blocked["status"] == "not_estimable" and "positivity" in blocked["skipped_reason"],
          f"corr(tau_hat, tau)={recovery:.3f} mean={np.mean(tau_hat):.3f} (planted "
          f"{np.mean(fixture['tau']):.3f}) top={top_modifier} absent={len(absent_modifiers)}")

    # 17 - the RATE / TOC curve is monotone on that planted signal: the patients the DR-learner
    # ranks high-benefit really do benefit more, the curve decays to exactly zero once everyone is
    # treated, the bootstrap interval excludes zero, and the benefit-calibration deciles rise.
    toc = np.asarray(planted["toc"], dtype=float)
    toc_grid = np.asarray(planted["toc_grid"], dtype=float)
    decile_benefit = np.asarray([row["aipw_benefit"] for row in planted["calibration"]], dtype=float)
    check("17_rate_and_toc_monotone_on_planted_signal",
          planted["rate"] > 0.0 and planted["rate_ci_low"] > 0.0
          and float(np.corrcoef(toc_grid, toc)[0, 1]) < -0.9
          and toc[0] > toc[-1] and abs(toc[-1]) < 1e-9
          and len(planted["toc_ci_low"]) == toc_grid.size == len(planted["toc_ci_high"])
          and planted["calibration_rho"] > 0.8 and decile_benefit[-1] > decile_benefit[0]
          and len(planted["calibration"]) == CATE_DECILES
          and planted["policy_value"] >= planted["treat_all_value"],
          f"rate={planted['rate']:.3f} ({planted['rate_ci_low']:.3f} to "
          f"{planted['rate_ci_high']:.3f}) toc {toc[0]:.3f} -> {toc[-1]:.3f} "
          f"rho={planted['calibration_rho']:.2f} deciles {decile_benefit[0]:.2f} -> "
          f"{decile_benefit[-1]:.2f}")

    # 18 - c-for-benefit is sign-matched to the metric on BOTH pages. tte.c_for_benefit builds the
    # observed pair benefit as Y[control] - Y[treated] (higher is better) but takes the predicted
    # vector as handed over, so feeding it tau = mu1 - mu0 raw (negative IS the benefit) scores a
    # PERFECT predictor near 0.08 rather than near 0.92. This fixture plants a perfect predictor
    # with an X-informative propensity, so nearest-PS matching pairs like with like and the metric
    # can actually see the effect; the raw-tau orientation must stay on the losing side of 0.5.
    size = 200
    row = np.arange(size)
    band_bmi = 35.0 + (row % 20) * 0.75
    planted_tau = -(1.0 + 0.30 * (band_bmi - 40.0))
    band_ps = 0.30 + 0.40 * (band_bmi - 35.0) / 14.25
    band_arm = np.zeros(size, dtype=int)
    for band in range(20):
        members = np.where((row % 20) == band)[0]
        band_arm[members[1::2]] = 1
    band_y = 0.85 * band_bmi + band_arm * planted_tau + 0.40 * np.sin(row.astype(float))
    corrected = _c_for_benefit(planted_tau, band_arm, band_y, band_ps)["c_for_benefit"]
    raw_tau = tte.c_for_benefit(planted_tau, band_arm, band_y, band_ps,
                                lower_is_better=True)["c_for_benefit"]
    check("18_c_for_benefit_orientation",
          corrected > 0.8 and raw_tau < 0.2 and corrected > raw_tau
          # the CATE payload routes through the same helper, so its value moves with this one
          and planted["c_for_benefit"] > 0.5,
          f"perfect predictor: corrected={corrected:.4f} vs raw-tau={raw_tau:.4f}; "
          f"planted-fixture CATE c-for-benefit={planted['c_for_benefit']:.4f}")

    # 19 - the which-incretin contrast (spec 10) is gated exactly like the CATE and flagged
    # prognostic. The CATE fixture's stratified-randomized arm is relabelled as the agent, so the
    # healthy cell has real overlap and must FIT; collapsing it onto one agent must refuse on
    # positivity rather than fit over a clipped ratio; a source without incretin predictions must
    # refuse structurally. Every branch carries the agent names and the prognostic caveat, the
    # rygb/sleeve arm counts are gone, and two identical calls agree byte for byte.
    incretin_frame = _incretin_heldout_fixture()
    incretin_audit = column_population_audit(incretin_frame)
    which_cfg = SecondaryConfig(mode="self-test", output_dir=".", bootstrap_replicates=64)
    healthy = _within_incretin_dr_read(which_cfg, {("incretin", "bmi"): incretin_frame},
                                       incretin_audit)
    repeated = _within_incretin_dr_read(which_cfg, {("incretin", "bmi"): incretin_frame},
                                        incretin_audit)
    one_armed = _within_incretin_dr_read(
        which_cfg, {("incretin", "bmi"): incretin_frame.assign(index_ingredient="tirzepatide")},
        incretin_audit)
    no_source = _within_incretin_dr_read(which_cfg, {}, incretin_audit)
    branches = (healthy, one_armed, no_source)
    check("19_which_incretin_gated_and_prognostic",
          str(healthy["status"]) == "done"
          and int(healthy["n_treated"]) + int(healthy["n_comparator"]) == int(healthy["n"]) == 800
          and "n_rygb" not in healthy and "n_sleeve" not in healthy
          and bool(healthy["importance"]) and np.isfinite(healthy["c_for_benefit"])
          # a one-agent cell has no propensity overlap: refused, never fitted
          and str(one_armed["status"]) == "not_estimable"
          and "positivity" in str(one_armed["skipped_reason"])
          and int(one_armed["n_comparator"]) == 0
          and str(no_source["status"]) == "not_estimable" and int(no_source["n"]) == 0
          # the prognostic-not-causal flag rides on EVERY branch, estimable or not
          and all(str(item.get("treated_agent")) == WITHIN_INCRETIN_TREATED
                  and str(item.get("comparator_agent")) == WITHIN_INCRETIN_COMPARATOR
                  for item in branches)
          # no default: the key itself has to be on every branch, since page 07 renders it
          and all("NOT A CAUSAL TREATMENT EFFECT" in str(item["caveat"]) for item in branches)
          and repr(repeated) == repr(healthy),
          f"done n={int(healthy['n'])} ({int(healthy['n_treated'])} vs "
          f"{int(healthy['n_comparator'])}) rate={healthy['rate']:.4f} "
          f"c-for-benefit={healthy['c_for_benefit']:.4f}; one-agent and no-source both refused")

    # 20 - the feature-sensitivity surrogate (spec 10) ranks a PLANTED dominant feature first. The
    # stored median is made an almost-pure function of baseline_bmi, while every other covariate
    # cycles on a coprime modulus and so cannot stand in for it. The surrogate must recover it as
    # rank 1 with almost all the importance share and near-perfect out-of-fold fidelity, the
    # counterfactual sweep must climb with it, arms absent from the source must come back as
    # documented skips rather than vanish, and two identical calls must agree byte for byte.
    sensitivity_frames = {("incretin", "bmi"): incretin_frame}
    sensitivity_rows, sweeps = _feature_sensitivity(which_cfg, sensitivity_frames, incretin_audit)
    sensitivity_repeat, _repeat_sweeps = _feature_sensitivity(which_cfg, sensitivity_frames,
                                                              incretin_audit)
    scored = [row for row in sensitivity_rows if not row["skipped_reason"]]
    skipped = [row for row in sensitivity_rows if row["skipped_reason"]]
    top = next(row for row in scored if int(row["rank"]) == 1)
    sweep_forecast = np.asarray(sweeps[0]["forecast"], dtype=float) if sweeps else np.zeros(0)
    check("20_feature_sensitivity_ranks_planted_feature",
          str(top["feature"]) == "baseline_bmi" and float(top["importance_share"]) > 0.5
          and float(top["surrogate_r2"]) > 0.95 and int(top["n"]) == 800
          and sorted(int(row["rank"]) for row in scored) == list(range(1, len(scored) + 1))
          # the five (arm, outcome) cells this fixture cannot score are documented, not dropped
          and len(skipped) == 5 and all(not row["feature"] for row in skipped)
          and all(not np.isfinite(row["permutation_importance"]) for row in skipped)
          and len(sweeps) == 1 and str(sweeps[0]["feature"]) == "baseline_bmi"
          and sweep_forecast.size == len(FEATURE_SWEEP_GRID)
          and bool(np.all(np.diff(sweep_forecast) >= 0.0))
          and float(sweep_forecast[-1] - sweep_forecast[0]) > 5.0
          and repr(sensitivity_repeat) == repr(sensitivity_rows),
          f"rank 1 = {top['feature']} share={float(top['importance_share']):.3f} "
          f"r2={float(top['surrogate_r2']):.4f}; sweep {sweep_forecast[0]:.2f} -> "
          f"{sweep_forecast[-1]:.2f}; {len(skipped)} documented skips")

    # 21 - the PRIMARY prognostic reporting re-cut from the two source cohorts to the THREE
    # reporting arms. The incretin frame passes through UNCOPIED (identity, not equality - the
    # 177k-row arm must never be duplicated), the shared surgical frame splits into rygb + sleeve
    # whose counts add back up to it, every declared arm-outcome-horizon cell yields a row, and
    # the RYGB arm carries the full metric set at hand-computed values. The 12 residuals are
    # chosen ORTHOGONAL to the predicted median, so the weighted calibration slope is exactly 1
    # and its intercept is exactly the bias; a 5-row sleeve arm exercises the suppression floor.
    arm_medians = [30.0, 35.0, 31.0, 34.0, 32.0, 33.0, 30.0, 35.0, 31.0, 34.0, 32.0, 33.0]
    arm_targets = [32.0, 37.0, 33.0, 36.0, 31.0, 32.0, 29.0, 34.0, 34.0, 37.0, 28.0, 29.0]
    incretin_frame = _auroc_fixture(arm_medians, arm_targets, None)
    surgical_frame = _auroc_fixture(arm_medians + arm_medians[:5], arm_targets + arm_targets[:5],
                                    (["rygb"] * 12) + (["sleeve"] * 5))
    arm_frames = {("incretin", "bmi"): incretin_frame, ("surgery", "bmi"): surgical_frame}
    arm_cfg = SecondaryConfig(mode="self-test", output_dir=".")
    per_arm = _per_arm_prognostic(arm_cfg, arm_frames)
    per_arm_cells = {(row["arm"], row["outcome"], row["target_month"]): row for row in per_arm}
    rygb_cell, sleeve_cell = per_arm_cells[("rygb", "bmi", 12)], per_arm_cells[("sleeve", "bmi", 12)]
    spec_metrics = ("crps", "rmse", "mae", "bias", "coverage_80", "coverage_90", "cal_slope")
    rygb_view, sleeve_view = (_arm_view(arm_frames, "surgery", "bmi", "rygb")[0],
                              _arm_view(arm_frames, "surgery", "bmi", "sleeve")[0])
    threshold_rows, _arm_rel, _arm_computable = _tier4_threshold_probabilities(arm_cfg, arm_frames, None)
    decision_rows = _tier4_decision_curves(arm_cfg, arm_frames, None)
    threshold_n = {(str(row["arm"]), str(row["event"])): int(row["n"]) for row in threshold_rows}
    # The page-00 census must dedup decision cells on the ARM dimension. Keying it on the dropped
    # `cohort` returns None for every row and silently folds rygb onto incretin, so assert the
    # consumer's own output: two arms scoring one event-horizon must read as two cells, not one.
    highlights = _executive_highlights({"tier4": {"threshold": {"rows": threshold_rows},
                                                  "decision": {"rows": decision_rows}}})
    decision_line = next(line for line in highlights if line.startswith("- Decision curves:"))
    check("21_per_arm_recut_forms_three_arms",
          # the arms: incretin passes through uncopied; the surgical frame splits and adds back up
          _arm_view(arm_frames, "incretin", "bmi", "")[0] is incretin_frame
          and len(rygb_view) == 12 and len(sleeve_view) == 5
          and len(rygb_view) + len(sleeve_view) == len(surgical_frame)
          # every declared arm-outcome-horizon cell yields a row; the absent hba1c arms are named
          and len(per_arm) == len(PROCEDURE_AUROC_ARMS) * sum(len(study.TARGET_MONTHS[outcome])
                                                              for outcome in OUTCOMES)
          and all(str(row["skipped_reason"]) for row in per_arm if str(row["outcome"]) == "hba1c")
          # the per-arm metric set is complete, and hand-computed for the RYGB arm
          and all(column in PER_ARM_PROGNOSTIC_COLUMNS for column in spec_metrics)
          and all(np.isfinite(rygb_cell[column]) for column in spec_metrics)
          and int(rygb_cell["n"]) == 12 and abs(rygb_cell["bias"] - 1.0 / 6.0) < 1e-9
          and abs(rygb_cell["mae"] - 26.0 / 12.0) < 1e-9
          and abs(rygb_cell["rmse"] - math.sqrt(70.0 / 12.0)) < 1e-9
          and abs(rygb_cell["coverage_80"] - 8.0 / 12.0) < 1e-9
          and abs(rygb_cell["coverage_90"] - 10.0 / 12.0) < 1e-9
          and abs(rygb_cell["cal_slope"] - 1.0) < 1e-9
          and abs(rygb_cell["cal_intercept"] - 1.0 / 6.0) < 1e-9
          # the thin sleeve arm-horizon cell is suppressed, not dropped, and never scored
          and int(sleeve_cell["n"]) == 5 and bool(sleeve_cell["suppressed"])
          and not np.isfinite(sleeve_cell.get("crps", np.nan))
          # threshold + decision reporting is keyed by arm, and the sub-11 arm gets no curve
          and sorted({str(row["arm"]) for row in threshold_rows}) == ["incretin", "rygb", "sleeve"]
          and not any("cohort" in row for row in threshold_rows + decision_rows)
          and threshold_n[("sleeve", "bmi_lt_35")] == 5
          and sorted({str(row["arm"]) for row in decision_rows}) == ["incretin", "rygb"]
          # the page-00 census keeps the arm dimension (2 arms x 1 event-horizon, not 1 pooled cell)
          and decision_line == "- Decision curves: 2 event-cells scored (net benefit).",
          f"rygb={len(rygb_view)} + sleeve={len(sleeve_view)} = {len(surgical_frame)}; "
          f"{len(per_arm)} arm-horizon cells; rygb bias={rygb_cell['bias']:.4f} "
          f"mae={rygb_cell['mae']:.4f} slope={rygb_cell['cal_slope']:.4f}; "
          f"threshold arms={len({str(row['arm']) for row in threshold_rows})}; "
          f"census '{decision_line.strip('- ')}'")

    # 22 - the SECTION-10 tiers re-cut over those SAME three arms. Transportability stands in for
    # the whole tier family: they all iterate the one `_reporting_arm_frames` primitive, so forming
    # the arms is tested once. The incretin arm is scored whole, the surgical frame splits into
    # rygb + sleeve whose stratum counts add back up to it, the 5-patient sleeve stratum is blanked
    # by the disclosure floor rather than published, and no row still carries the dropped `cohort`
    # key. The GLP-1-vs-surgery overlap is the one read that must NOT be re-cut - it is inherently
    # a cross-cohort contrast - so it is asserted on frames whose surgical ARMS cannot be formed at
    # all: the per-arm tier loses the surgical read and records the gap, the overlap still scores.
    # One state per arm, so each arm contributes exactly one stratum and the ONLY sub-11 cell in
    # the tier is the 5-patient sleeve arm.
    recut_frames = {
        ("incretin", "bmi"): incretin_frame.assign(state="DE", baseline_bmi=arm_medians),
        ("surgery", "bmi"): surgical_frame.assign(state=(["NJ"] * 12) + (["PA"] * 5),
                                                  baseline_bmi=arm_medians + arm_medians[:5]),
    }
    recut_audit = column_population_audit(recut_frames[("surgery", "bmi")], ["state"])
    transport_rows, _recut_meta = _tier1_transportability(arm_cfg, recut_frames, recut_audit)
    n_by_arm: dict[str, int] = {}
    for row in (item for item in transport_rows if str(item["stratum"])):
        n_by_arm[str(row["arm"])] = n_by_arm.get(str(row["arm"]), 0) + int(row["n"])
    published = suppress(_rows_to_frame(transport_rows, TRANSPORTABILITY_COLUMNS), ["n"])
    thin = published.loc[published["small_cell_suppressed"]]
    armless = {key: value.drop(columns=["procedure"]) if "procedure" in value else value
               for key, value in recut_frames.items()}
    armless_rows, _armless_meta = _tier1_transportability(arm_cfg, armless, recut_audit)
    glp1_rows, glp1_trajectory = _tier4_glp1_overlap(arm_cfg, armless)
    # A truncated subgroup table must still show every arm, and n_censored is a count like any
    # other: censored_fraction times n recovers it, so a sub-11 censored cell takes its row.
    wide = pd.DataFrame({"arm": (["incretin"] * 10) + (["rygb"] * 10) + (["sleeve"] * 10),
                         "axis": ["sex"] * 30, "n": [50] * 30})
    censored = _rows_to_frame([{"arm": "rygb", "outcome": "bmi", "target_month": 12, "n": 54,
                                "n_censored": 7, "censored_fraction": 7 / 54},
                               {"arm": "incretin", "outcome": "bmi", "target_month": 12, "n": 54,
                                "n_censored": 20, "censored_fraction": 20 / 54}],
                              TIPPING_POINT_COLUMNS)
    censored = suppress(censored, list(ATTRITION_COUNT_COLUMNS))
    check("22_section10_tiers_recut_over_three_arms",
          # the tier sees exactly the three arms, and the surgical arms partition their source frame
          sorted(n_by_arm) == ["incretin", "rygb", "sleeve"]
          and n_by_arm["incretin"] == len(incretin_frame) and n_by_arm["sleeve"] == 5
          and n_by_arm["rygb"] + n_by_arm["sleeve"] == len(surgical_frame)
          and not any("cohort" in row for row in transport_rows)
          # the one sub-11 arm-cell is blanked - neither its count nor its statistic is published
          and len(thin) == 1 and not np.isfinite(float(thin["n"].iloc[0]))
          and not np.isfinite(float(thin["crps"].iloc[0]))
          # an unformable arm is dropped from the per-arm tiers but NAMED in the omissions ledger
          and {str(row["arm"]) for row in armless_rows} == {"incretin"}
          and sum(1 for gap in _arm_formation_gaps(armless)
                  if "procedure not populated" in gap) == 2
          # the GLP-1 overlap stays the 2-cohort contrast: it scores with no arm formable at all
          and bool(glp1_rows) and all(set(row) == set(GLP1_OVERLAP_COLUMNS) for row in glp1_rows)
          and sorted(glp1_trajectory["bmi"]) == ["incretin", "surgery"]
          # a 12-row truncated subgroup table still reaches all three arms, 4 rows each
          and sorted(_arm_balanced_head(wide, 12)["arm"].unique()) == ["incretin", "rygb", "sleeve"]
          and len(_arm_balanced_head(wide, 12)) == 12
          # the sub-11 censored cell is suppressed; the 20-censored cell is published intact
          and bool(censored["small_cell_suppressed"].iloc[0])
          and not np.isfinite(float(censored["censored_fraction"].iloc[0]))
          and not bool(censored["small_cell_suppressed"].iloc[1])
          and int(censored["n_censored"].iloc[1]) == 20,
          f"arms {n_by_arm}; rygb+sleeve={n_by_arm['rygb'] + n_by_arm['sleeve']}"
          f"={len(surgical_frame)}; {len(thin)} sub-11 cell blanked; armless tier arms="
          f"{sorted({str(row['arm']) for row in armless_rows})} with "
          f"{len(_arm_formation_gaps(armless))} gaps named; GLP-1 overlap {len(glp1_rows)} "
          f"cross-cohort contrasts; 12-row table reaches "
          f"{_arm_balanced_head(wide, 12)['arm'].nunique()} arms; n_censored=7 row suppressed")

    return {"status": "passed", "tests": results,
            "passed": sum(item["passed"] for item in results), "total": len(results)}


def _incretin_heldout_fixture() -> Any:
    """A held-out incretin BMI frame with EXACT agent overlap and two independent plants.

    Shaped like what ``_tier1_heldout_frames`` hands the Tier-1 analyses: one row per patient at
    the CATE horizon, the index-agent columns the drug grouping reads, and a quantile ladder at the
    same +/- 3 spacing the AUROC fixture uses. Nothing draws from an RNG.

    Each row's covariates are fixed by two INDEPENDENT coordinates of its profile: ``bmi_level``
    sets baseline BMI and ``block`` sets everything else, so no other covariate can stand in for
    BMI. The agent alternates inside every (profile, cross-fit fold) stratum exactly as the CATE
    fixture's arm does, so P(agent | covariates) is 0.5 in every training subset and no classifier
    can beat the base rate out of fold: the overlap this cell needs is a property of the fixture,
    not of the fit. The two plants ride on separate channels - the observed target carries an
    agent-by-BMI interaction for the DR read, the stored median is an almost-pure function of
    baseline BMI for the surrogate read - so neither test can pass on the other's signal.
    """
    n, profiles = 800, 40
    index = np.arange(n)
    profile = index % profiles
    bmi = 34.0 + 2.0 * (profile % 8)
    block = profile // 8
    patient_ids = np.array([f"I{item}" for item in index])
    fold_ids = patient_folds(patient_ids, SEED, NUISANCE_FOLDS)
    arm = np.zeros(n, dtype=int)
    for stratum in range(profiles * NUISANCE_FOLDS):
        members = np.where((profile == stratum // NUISANCE_FOLDS)
                           & (fold_ids == stratum % NUISANCE_FOLDS))[0]
        arm[members[1::2]] = 1
    tau = -(1.0 + 0.30 * (bmi - 40.0))
    ripple = np.sin(index.astype(float))
    rows = pd.DataFrame({
        "patient_id": patient_ids,
        "cohort": ["incretin"] * n,
        "index_ingredient": np.where(arm == 1, "tirzepatide", "semaglutide"),
        "index_route": ["injection"] * n,
        "baseline_bmi": bmi,
        "baseline_hba1c": 5.4 + 0.4 * block,
        "age_at_index": 32.0 + 4.0 * (block % 4),
        "sex": np.where(block % 2 == 0, "M", "F"),
        "diabetes_flag": (block % 3 == 0).astype(float),
        "hypertension": (block % 4 == 1).astype(float),
        "dyslipidemia": (block % 5 == 2).astype(float),
        "osa": (block % 3 == 1).astype(float),
        "insulin": (block % 4 == 2).astype(float),
        "biguanide": (block % 5 == 3).astype(float),
        "sglt2": (block % 3 == 2).astype(float),
        "target_month": CATE_TARGET_MONTH,
        "target_observed": True,
        "target_value": 0.85 * bmi + arm * tau + 0.15 * ripple,
        "analysis_weight": 1.0,
    })
    ladder = ((0.9 * bmi + 0.05 * ripple)[:, None]
              + np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])[None, :])
    for position, column in enumerate(QUANTILE_COLS):
        rows[column] = ladder[:, position]
    return rows


def _cate_fixture() -> dict[str, Any]:
    """A planted-CATE cell: stratified-randomized arm, perfect overlap, and a known tau(x).

    Y = 0.85 * BMI + A * tau(BMI) + a deterministic ripple, with tau(BMI) = -(1 + 0.3 (BMI - 40)):
    RYGB lowers BMI more for heavier patients, so the planted BENEFIT (-tau) rises from about -0.5
    to +3.8 kg/m^2 across the BMI range. The outcome nuisance is handed the MARGINAL effect only
    (mu1 - mu0 = mean tau for everyone), so it contributes no effect modification at all and the
    augmentation term has to supply every bit of the heterogeneity the DR-learner recovers.

    The arm is randomized WITHIN each (BMI level, cross-fit fold) stratum rather than at random, so
    every fold sees the same arm mix at every X. Fold-level arm-fraction noise would otherwise leak
    into an out-of-fold prediction (a fold rich in treated rows depresses the prediction its own
    members receive), which is a small-fixture artifact rather than anything the estimator does.
    Every covariate cycles on a modulus coprime to the BMI modulus, so no other modifier can stand
    in for the real one, and nothing draws from an RNG: each value is a closed form of the row index.
    """
    n = 800
    index = np.arange(n)
    level = index % 20
    bmi = 35.0 + level * 0.75
    tau = -(1.0 + 0.30 * (bmi - 40.0))
    patient_ids = np.array([f"C{item}" for item in index])
    fold_ids = patient_folds(patient_ids, SEED, NUISANCE_FOLDS)
    arm = np.zeros(n, dtype=int)
    for stratum in range(20 * NUISANCE_FOLDS):
        members = np.where((level == stratum // NUISANCE_FOLDS) & (fold_ids == stratum % NUISANCE_FOLDS))[0]
        arm[members[1::2]] = 1
    mu0 = 0.85 * bmi
    mu1 = mu0 + float(np.mean(tau))
    rows = pd.DataFrame({
        "patient_id": patient_ids,
        "cohort": ["surgery"] * n,
        "procedure": np.where(arm == 1, "rygb", "sleeve"),
        "baseline_bmi": bmi,
        "baseline_hba1c": 5.5 + (index % 7) * 0.3,
        "age_at_index": 30.0 + (index % 13),
        "sex": np.where(index % 3 == 0, "M", "F"),
        "diabetes_flag": (index % 9 == 0).astype(float),
        "hypertension": (index % 11 == 0).astype(float),
        "dyslipidemia": (index % 17 == 0).astype(float),
        "osa": (index % 19 == 0).astype(float),
        "insulin": (index % 21 == 0).astype(float),
        "biguanide": (index % 23 == 0).astype(float),
        "sglt2": (index % 3 == 1).astype(float),
    })
    return {
        "rows": rows, "audit": column_population_audit(rows), "tau": tau, "mu1": mu1, "mu0": mu0,
        "Y": mu0 + arm * tau + 0.15 * np.sin(index.astype(float)),
        "A": arm, "delta": np.ones(n, dtype=int), "ps": np.full(n, 0.5), "pc": np.ones(n),
        "patient_ids": patient_ids,
    }


def _fixture_prediction_frame() -> Any:
    """A tiny schema-faithful stored-prediction frame for self-tests."""
    n = 30
    base = {
        "row_id": list(range(n)),
        "patient_id": [f"P{i}" for i in range(n)],
        "cohort": ["surgery"] * n,
        "outcome": ["bmi"] * n,
        "origin_month": [0] * n,
        "target_month": [12] * n,
        "split": (["temporal_test"] * 20) + (["train"] * 10),
        "target_value": [40.0 - 0.1 * i for i in range(n)],
        "target_observed": [True] * n,
        "analysis_weight": [1.0] * n,
        "support_status": ["mature_with_target"] * n,
        "treatment": (["rygb"] * 15) + (["sleeve"] * 15),
        "center_id": ["c0"] * n,
        "prediction_reference_value": [42.0] * n,
        "candidate": ["histogram_gradient_boosting"] * n,
        "architecture": ["hist_gradient_boosting_quantile"] * n,
    }
    for index, column in enumerate(QUANTILE_COLS):
        offset = (index - 3) * 1.5
        base[column] = [40.0 - 0.1 * i + offset for i in range(n)]
    return pd.DataFrame(base)[list(study.STORED_PREDICTION_COLUMNS)]


def _count_pdf_pages(path: Path) -> int:
    """Count PDF pages without a PDF library: count '/Type /Page' occurrences (not /Pages)."""
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--from-run", metavar="RUN_DIR", default=None,
                       help="Consume a completed production run directory (the default path)")
    modes.add_argument("--full", action="store_true",
                       help="Re-run the production modeling pipeline to regenerate predictions, then analyse")
    modes.add_argument("--smoke", action="store_true",
                       help="Bounded end-to-end run on the synthetic bundle (no Cosmos, no torch)")
    modes.add_argument("--self-test", action="store_true",
                       help="Run deterministic embedded tests without Cosmos or torch")
    modes.add_argument("--plot-only", action="store_true",
                       help="Rebuild the figure book from this run's own checkpoints")
    parser.add_argument("--output-dir", default=None,
                        help="Run directory (defaults to the --from-run dir, else a timestamped ./results dir)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages whose checkpoint exists and whose config hash matches")
    parser.add_argument("--incretin-qualifying-months", type=int,
                        choices=list(study.INCRETIN_QUALIFYING_MONTH_CHOICES),
                        default=study.INCRETIN_QUALIFYING_MONTHS,
                        help="Completed months of incretin treatment required for cohort entry (Tier 3 sweep)")
    parser.add_argument("--skip-tte", action="store_true", help="Skip the target-trial emulation")
    parser.add_argument("--loso-refit", action="store_true",
                        help="Tier 1: true leave-one-state-out refit (heavy) instead of frozen re-validation")
    parser.add_argument("--mi-imputations", type=int, default=MI_IMPUTATIONS_DEFAULT,
                        help="Tier 1.2 multiple-imputation count (default 10)")
    parser.add_argument("--neural-outcome-model", action="store_true",
                        help="TTE: reuse the frozen selected neural model for counterfactual mu (optional)")
    parser.add_argument("--orchestrate", action="store_true",
                        help="Force process-per-stage orchestration (default for --from-run/--full)")
    parser.add_argument("--seed", type=int, default=SEED, help=argparse.SUPPRESS)
    parser.add_argument("--worker", choices=STAGE_SEQUENCE, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eligibility-worker-months", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # A finer-grained worker: re-acquire the incretin cohort at one eligibility threshold and exit,
    # so the whole-bundle Cosmos load's peak RSS is reclaimed (Tier 3 eligibility sweep).
    if args.eligibility_worker_months is not None:
        try:
            return _run_eligibility_acquire_worker(args)
        except Exception as exc:  # surfaced to the parent via a nonzero exit code
            traceback.print_exc()
            print(f"[secondary] eligibility worker ({args.eligibility_worker_months} mo) failed: {exc}",
                  file=sys.stderr)
            return 1

    # A worker subprocess runs exactly one stage of an already-resolved run and exits.
    if args.worker:
        args.smoke = args.worker_mode == "smoke"
        args.full = args.worker_mode == "full"
        args.self_test = False
        args.plot_only = False
        if args.worker_mode in {"from-run", "full", "smoke"} and not args.smoke and not args.full:
            pass
        cfg = resolve_config(args)
        cfg = replace(cfg, mode=args.worker_mode or cfg.mode)
        try:
            load_runtime()
            run_stage(cfg, args.worker)
            return 0
        except Exception as exc:  # a worker failure is surfaced to the parent via exit code
            traceback.print_exc()
            print(f"[secondary] worker stage {args.worker} failed: {exc}", file=sys.stderr)
            return 1

    if args.self_test:
        try:
            report = run_self_tests()
        except AssertionError as exc:
            print(f"SELF-TEST FAILED: {exc}", file=sys.stderr)
            return 1
        for item in report["tests"]:
            print(f"[PASS] {item['test']}" + (f" | {item['detail']}" if item["detail"] else ""))
        print(f"SELF-TEST PASSED: {report['passed']}/{report['total']} deterministic tests")
        return 0

    cfg = resolve_config(args)
    try:
        load_runtime()
        study.set_deterministic_seed(cfg.seed)
        if cfg.mode == "plot-only":
            run_dir = run_plot_only(cfg)
        else:
            run_dir = run_all_stages(cfg, {})
        print(f"[secondary] completed: {run_dir}")
        print(f"[secondary] figures: {cfg.export_dir}")
        print(f"[secondary] bundle: {cfg.secondary_dir / 'secondary_analyses_results_bundle.zip'}")
        return 0
    except SecondaryPreflightError as exc:
        render_secondary_failure(cfg, exc.title, exc.issues, exc.details)
        print(f"[secondary] preflight failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        traceback.print_exc()
        try:
            render_secondary_failure(cfg, "Secondary run execution failed",
                                     [f"{type(exc).__name__}: {exc}"],
                                     ["The run stopped before a result was released."])
        except Exception as report_error:
            print(f"[secondary] failure report could not be written: {report_error!r}", file=sys.stderr)
        print(f"[secondary] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
