"""freeze_run.py - one command that freezes a fully reproducible digital-twin run.

Pinned to ONE config, this thin orchestrator runs, in sequence, into
``runs/frozen/<timestamp>/``:

  1. ``debug_attrition.run``               -> ``attrition/``  (CONSORT / missingness report)
  2. ``train_twin_pipeline.run_pipeline``  -> ``twin_pipeline/`` (calibrated composite-MACE
                                             GBM + event-conditioned twin-flow Optuna sweep;
                                             a FRESH twin is trained every run)
  3. ``evaluate_twin.evaluate``            -> ``evaluation/`` (GBM + flow + simulator metrics,
                                             incl. W5 conformal calibration + threshold probs)
  4. ``evaluate_twin.compare_trajectory_models`` -> ``evaluation/`` (W4 four-arm event-
                                             conditioning ablation CSVs; guarded)
  5. ``make_table_one.generate``           -> ``table_one/``  (Table 1, same split as the models)
  6. ``figures.build_all`` (subprocess)    -> ``figures/``    (W6 main + supplement journal
                                             figures; guarded so a figure crash never loses the run)

then writes a top-level ``RUN_MANIFEST.json`` capturing everything needed to
reproduce the run and to make the backend unmistakable: git SHA, all seeds,
``split_strategy``, the actual GBM backend (xgboost vs HistGradientBoosting) +
xgboost / scikit-learn / torch versions, a full ``uv pip freeze``, the input
CSV path + SHA-256, the resolved GBM and twin configs, and the pinned
patient-feature width (``len(fm.PATIENT_FEATURES)``).

Reproducibility guard (why the width is pinned)
-----------------------------------------------
The twin net's encoder input width is ``surgery_emb_dim + event_emb_dim +
len(fm.PATIENT_FEATURES)``. A committed checkpoint trained at an older feature
width load-fails against newer code (the dim-30-vs-32 bug that motivated this
harness). freeze_run therefore ALWAYS trains a fresh twin via ``run_pipeline`` and
never loads a committed checkpoint; the manifest records both the code-side width
and the freshly-trained checkpoint's actual encoder input width so any future
silent mismatch is caught by a diff.

NOTE: results across GBM backends (HistGradientBoosting locally vs XGBoost on the
Cosmos VM) are NOT comparable. Install xgboost in the venv so local smoke == the VM
backend; the manifest records whichever backend actually ran.

Run (Cosmos VM, real cohort - ``--db`` is the DEFAULT source, so a bare run does
the real run; add ``--split-strategy temporal`` for the temporal-validation fold)::

    python freeze_run.py                            # == --db (Cosmos MBSCohort)
    python freeze_run.py --split-strategy temporal

Run (fake-cohort smoke, minimal but real - ``--smoke`` with no source uses the
bundled fake cohort and never touches the live DB)::

    python freeze_run.py --smoke
    python freeze_run.py --csv fake_data/fake_mbs_cohort.csv --smoke
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import traceback
from pathlib import Path

# --------------------------------------------------------------------------- #
# macOS OpenMP guard -- MUST run before importing torch / xgboost.
#
# On macOS, torch ships its own bundled libomp while xgboost loads Homebrew's
# libomp; two OpenMP runtimes in one process SIGSEGV the moment xgboost fits
# after torch has been imported (reproduced: exit 139). Pinning
# OMP_NUM_THREADS=1 makes both use a single thread team and avoids the crash.
# This is macOS-only: on the Linux Cosmos VM torch and xgboost share libgomp, so
# the guard is skipped and the run stays fully multi-threaded. A user-provided
# OMP_NUM_THREADS is always respected (setdefault).
# --------------------------------------------------------------------------- #
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

import train_flow_matching as fm
import gbm_mace_baseline as gb
import train_flow_matching_twin as tw
import tune_flow_matching_twin_optuna as tune_twin
import train_twin_pipeline as pipeline
import evaluate_flow_matching as ev
import evaluate_twin as ev_twin
import make_table_one as t1
import debug_attrition as attrition


DEFAULT_OUTPUT_ROOT = fm.REPO_ROOT / "runs" / "frozen"


# --------------------------------------------------------------------------- #
# The ONE pinned config. CLI flags override individual fields; ``--smoke`` swaps
# in minimal-but-real values for the 52-row fake cohort. The manifest records the
# RESOLVED config (post-override), so what actually ran is never ambiguous.
# --------------------------------------------------------------------------- #
@dataclass
class FreezeConfig:
    csv_path: str | None = None          # local CSV export (post-SQL); None + use_db -> Cosmos
    use_db: bool = False                 # True -> load from Cosmos MBSCohort (VM only)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    # Shared split (GBM, twin, and Table 1 line up patient-for-patient at this seed).
    split_strategy: str = "surgery"
    split_seed: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    # Model seeds.
    seed: int = 0                        # GBM + twin trainer seed (and Optuna sampler seed)
    eval_seed: int = 0
    deterministic: bool = False          # -> fm.enable_determinism (global torch determinism)
    # Twin-flow Optuna sweep.
    n_trials: int = tune_twin.N_TRIALS
    twin_num_steps: int | None = None    # None -> tune_twin.BASE_CONFIG.num_steps
    final_train_best: bool = True        # required True: evaluation needs a final twin run dir
    # Evaluation.
    n_samples: int = 200
    n_steps: int = 50
    n_boot: int = 1000
    device: str = "cpu"
    # Table 1.
    table_continuous: str = "median"


# --------------------------------------------------------------------------- #
# Manifest helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(dist_name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(dist_name)
    except Exception:  # noqa: BLE001 - a missing package is not fatal for the manifest
        return None


def collect_pip_freeze() -> dict:
    """Full dependency snapshot: prefer ``uv pip freeze`` (uv-managed venv), else
    fall back to ``importlib.metadata`` so the manifest is never empty."""
    try:
        proc = subprocess.run(
            ["uv", "pip", "freeze", "--python", sys.executable],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return {"tool": "uv pip freeze",
                    "packages": [ln for ln in proc.stdout.splitlines() if ln.strip()]}
    except Exception:  # noqa: BLE001 - uv may not be on PATH on the VM
        pass
    try:
        import importlib.metadata as md

        pkgs = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in md.distributions()
            if d.metadata and d.metadata["Name"]
        )
        return {"tool": "importlib.metadata", "packages": pkgs}
    except Exception as exc:  # noqa: BLE001
        return {"tool": "unavailable", "error": str(exc), "packages": []}


def git_info() -> dict:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(fm.REPO_ROOT),
                capture_output=True, text=True, timeout=30,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    status = _git("status", "--porcelain")
    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


def twin_checkpoint_in_features(twin_run_dir: Path) -> int | None:
    """Actual encoder input width baked into the freshly-trained checkpoint.

    Read the first ``encoder.<n>.weight`` in the saved state dict: its shape[1] is
    the encoder's ``in_features`` (== surgery_emb_dim + event_emb_dim +
    patient_feature_width). Compared against the code-side width in the manifest,
    this is the concrete guard against a checkpoint/code mismatch. Best-effort.
    """
    try:
        try:
            state = torch.load(twin_run_dir / "model.pt", map_location="cpu", weights_only=True)
        except TypeError:  # older torch without weights_only
            state = torch.load(twin_run_dir / "model.pt", map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    best: tuple[int, int] | None = None
    for key, tensor in state.items():
        if key.startswith("encoder.") and key.endswith(".weight") and getattr(tensor, "ndim", 0) == 2:
            match = re.search(r"encoder\.(\d+)\.weight", key)
            idx = int(match.group(1)) if match else 0
            if best is None or idx < best[0]:
                best = (idx, int(tensor.shape[1]))
    return best[1] if best else None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"could not read {path}: {exc}"}


def _list_files(path: Path, pattern: str = "*") -> list[str]:
    """Sorted basenames of the files directly under ``path`` (for the artifact inventory)."""
    p = Path(path)
    if not p.is_dir():
        return []
    return sorted(f.name for f in p.glob(pattern) if f.is_file())


def temporal_era_stats(cfg: FreezeConfig, dataset: fm.FlowDataset) -> dict:
    """Per-era (train/val/test) descriptive stats for a TEMPORAL freeze.

    Emitted into RUN_MANIFEST.json ONLY when ``split_strategy=='temporal'``. Recomputes
    the SAME partition the models used (``fm.make_temporal_splits`` at the freeze's
    split_seed + fractions), then documents the three structural properties of an
    out-of-time fold that MUST be reported (not hidden, not worked around):

      * surgery-date range per era -- train earliest ... test latest, non-overlapping;
      * follow-up MATURITY -- per BMI/HbA1c horizon, the count/fraction of each era's
        patients with an OBSERVED (unmasked, GLP-1-uncensored) cell. Later eras have
        shorter follow-up before the ProcDateValue<=2023-05-01 cutoff, so the Test era's
        5-6yr cells shrink to small/zero n; this is exactly why long-horizon Test metrics
        are underpowered, and the numbers make that explicit;
      * postoperative GLP-1 exposure per era (censors BMI/HbA1c cells) + median active
        follow-up (ActiveEndInterval, days) -- later eras are more GLP-1-selected and
        less follow-up-mature.

    GLP-1-naive fraction: the cohort is ``PriorGLP1==0`` by SQL construction, so every era
    is 100% GLP-1-naive at BASELINE. The PRE-filter surgical/GLP-1-naive fraction per era
    (the shrinking-eligible-population narrative) needs the pre-SQL denominator
    (``debug_attrition.db_prefilter_counts``, ``--db`` only) and is deferred to a VM run;
    here we emit the derivable post-filter per-era counts + postoperative-initiation frac.
    """
    split_cfg = fm.TrainConfig(
        split_seed=cfg.split_seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
    )
    splits = fm.make_temporal_splits(dataset, split_cfg)
    proc = pd.to_datetime(dataset.frame["ProcDateValue"], errors="coerce").to_numpy()
    active = gb.frame_feature(dataset, "ActiveEndInterval")  # follow-up window (days) or None
    postop = gb.frame_feature(dataset, "PostOpGLP1")         # 0/1 post-op incretin flag or None
    horizons = [
        (m["name"], m["horizon_months"], int(m["dim"]))
        for m in dataset.target_metadata
        if m["group"] in {"bmi", "hba1c"} and m["horizon_months"] is not None
    ]

    eras: dict[str, dict] = {}
    for era in ("train", "val", "test"):
        idx = splits[era]
        n = int(idx.size)
        dates = pd.Series(proc[idx])
        dmin, dmax, dmed = dates.min(), dates.max(), dates.median()
        maturity = [
            {
                "target": name,
                "horizon_months": horizon_months,
                "n_observed": int(dataset.mask[idx, dim].sum()),
                "frac_observed": (float(dataset.mask[idx, dim].sum()) / n) if n else float("nan"),
            }
            for name, horizon_months, dim in horizons
        ]
        if postop is not None and n:
            init_n = int((np.nan_to_num(postop[idx], nan=0.0) == 1).sum())
            glp1_frac = init_n / n
        else:
            init_n, glp1_frac = None, None
        med_fu = None
        if active is not None:
            av = active[idx]
            av = av[np.isfinite(av)]
            med_fu = float(np.median(av)) if av.size else None
        eras[era] = {
            "n": n,
            "surgery_date_min": None if pd.isna(dmin) else str(pd.Timestamp(dmin).date()),
            "surgery_date_max": None if pd.isna(dmax) else str(pd.Timestamp(dmax).date()),
            "surgery_date_median": None if pd.isna(dmed) else str(pd.Timestamp(dmed).date()),
            "postop_glp1_initiated_n": init_n,
            "postop_glp1_initiated_frac": glp1_frac,
            "median_active_followup_days": med_fu,
            "followup_maturity": maturity,
        }

    return {
        "split_seed": cfg.split_seed,
        "fractions": {"train": cfg.train_frac, "val": cfg.val_frac, "test": cfg.test_frac},
        "date_ranges_nonoverlapping_by_construction": True,
        "eras": eras,
        "glp1_naive_note": (
            "Cohort is PriorGLP1==0 by SQL construction, so every era is 100% GLP-1-naive at "
            "BASELINE. 'postop_glp1_initiated_*' is POSTOPERATIVE incretin initiation (censors "
            "that era's BMI/HbA1c cells); later eras are more GLP-1-selected. The PRE-filter "
            "surgical/GLP-1-naive fraction per era (shrinking-eligible-population narrative) needs "
            "the pre-SQL denominator (debug_attrition.db_prefilter_counts, --db only) and is a "
            "VM-time addition."
        ),
        "maturity_note": (
            "followup_maturity[*].n_observed counts each era's patients with an unmasked "
            "BMI/HbA1c cell at that horizon; later eras (Test) have shorter follow-up before "
            "ProcDateValue<=2023-05-01, so long-horizon (5-6yr) cells shrink to small/zero n -- "
            "expected and reported, not worked around. median_active_followup_days = per-era "
            "median ActiveEndInterval."
        ),
    }


def build_manifest(cfg: FreezeConfig, frozen_dir: Path, dataset: fm.FlowDataset,
                   csv_path: Path | None, attrition_report: Path, pipeline_dir: Path,
                   pipe_manifest: dict, gbm_run_dir: Path, twin_final_run_dir: Path,
                   eval_summary: dict, table_dir: Path,
                   traj_result: dict | None = None, figures_result: dict | None = None) -> dict:
    gbm_config = _read_json(gbm_run_dir / "config.json")
    twin_config = _read_json(twin_final_run_dir / "config.json")
    traj_result = traj_result or {}
    figures_result = figures_result or {}

    backend = "xgboost" if gb.xgboost_available() else "hist_gradient_boosting"

    patient_features = list(fm.PATIENT_FEATURES)
    ckpt_in_features = twin_checkpoint_in_features(twin_final_run_dir)
    expected_static = None
    if isinstance(twin_config, dict) and {"surgery_emb_dim", "event_emb_dim"} <= twin_config.keys():
        expected_static = (int(twin_config["surgery_emb_dim"])
                           + int(twin_config["event_emb_dim"])
                           + len(patient_features))

    manifest = {
        "schema_version": 1,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "frozen_run_dir": str(frozen_dir),
        "git": git_info(),
        # --- reproducibility guard: code SHA is paired with the feature width ----
        "code": {
            "patient_feature_width": len(patient_features),
            "patient_features": patient_features,
            "twin_x_cont_dim": int(tw.X_CONT_DIM),
            "twin_encoder_static_dim_expected": expected_static,
            "twin_checkpoint_encoder_in_features": ckpt_in_features,
            "checkpoint_width_matches_code": (
                None if (ckpt_in_features is None or expected_static is None)
                else ckpt_in_features == expected_static
            ),
            "fresh_twin_trained": True,
        },
        # --- backend must be UNMISTAKABLE (HistGB smoke != XGBoost result) -------
        "backend": {
            "gbm_backend": backend,
            "gbm_backend_from_pipeline": pipe_manifest.get("gbm_backend"),
            "xgboost_available": gb.xgboost_available(),
            "versions": {
                "python": platform.python_version(),
                "xgboost": pkg_version("xgboost"),
                "scikit_learn": pkg_version("scikit-learn"),
                "torch": pkg_version("torch"),
                "numpy": pkg_version("numpy"),
                "pandas": pkg_version("pandas"),
                "scipy": pkg_version("scipy"),
                "optuna": pkg_version("optuna"),
            },
        },
        "seeds": {
            "model_seed": cfg.seed,
            "split_seed": cfg.split_seed,
            "eval_seed": cfg.eval_seed,
            "table_split_seed": cfg.split_seed,
            "deterministic": cfg.deterministic,
        },
        "split": {
            "split_strategy": cfg.split_strategy,
            "train_frac": cfg.train_frac,
            "val_frac": cfg.val_frac,
            "test_frac": cfg.test_frac,
            "shared_split": pipe_manifest.get("shared_split"),
            "split_sizes": eval_summary.get("split_sizes"),
            "test_prevalence": eval_summary.get("test_prevalence"),
            "table_one_split_note": (
                "make_table_one routes on split_strategy (fm.make_stratified_splits for 'surgery', "
                "fm.make_temporal_splits for 'temporal'), so Table 1 aligns patient-for-patient with "
                "the GBM and twin for BOTH folds (all consume the same split_seed + fractions)."
            ),
            # Per-era surgery-date ranges, follow-up maturity, and GLP-1 exposure for the
            # out-of-time fold; None for the (random) surgery/outcome splits where eras are
            # not calendar-ordered. This is the honest record of the temporal fold's limits.
            "per_era_temporal": (
                temporal_era_stats(cfg, dataset) if cfg.split_strategy == "temporal" else None
            ),
        },
        "input": {
            "use_db": cfg.use_db,
            "source": None if csv_path is None else str(csv_path),
            "source_abspath": None if csv_path is None else str(Path(csv_path).resolve()),
            "sha256": None if csv_path is None else sha256_file(Path(csv_path)),
            "source_label": getattr(dataset, "source_label", None),
            "n_patients": int(len(dataset.subject_ids)),
        },
        "resolved_config": {
            "freeze": asdict(cfg),
            "gbm": gbm_config,
            "twin": twin_config,
        },
        "artifacts": {
            "attrition_report": str(attrition_report),
            "pipeline_dir": str(pipeline_dir),
            "pipeline_manifest": str(pipeline_dir / "manifest.json"),
            "gbm_run_dir": str(gbm_run_dir),
            "gbm_pickle": pipe_manifest.get("gbm_pickle"),
            "twin_study_dir": pipe_manifest.get("twin_study_dir"),
            "twin_final_run_dir": str(twin_final_run_dir),
            "twin_checkpoint": str(twin_final_run_dir / "model.pt"),
            "evaluation_dir": eval_summary.get("output_dir"),
            "eval_summary_json": (
                str(Path(eval_summary["output_dir"]) / "eval_twin_summary.json")
                if eval_summary.get("output_dir") else None
            ),
            "table_one_dir": str(table_dir),
            # --- W4 event-conditioning trajectory ablation (written into evaluation/) ---
            "trajectory_comparison": {
                "status": traj_result.get("status"),
                "dir": traj_result.get("output_dir"),
                "arms": traj_result.get("arms"),
                "metrics_csv": traj_result.get("metrics_csv"),
                "paired_tests_csv": traj_result.get("paired_tests_csv"),
                "summary_json": (
                    str(Path(traj_result["output_dir"]) / "trajectory_comparison_summary.json")
                    if traj_result.get("output_dir") else None
                ),
                "noevent_twin_run": traj_result.get("noevent_twin_run"),
                "error": traj_result.get("error"),
            },
            # --- W6 journal figures (main + supplement) ---
            "figures": {
                "status": figures_result.get("status"),
                "dir": figures_result.get("output_dir"),
                "returncode": figures_result.get("returncode"),
                "main": _list_files(frozen_dir / "figures" / "main"),
                "supplement": _list_files(frozen_dir / "figures" / "supplement"),
                "error": figures_result.get("error"),
            },
        },
        "environment": {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "pip_freeze": collect_pip_freeze(),
        },
    }
    return manifest


def run_trajectory_comparison(cfg: FreezeConfig, frozen_dir: Path, twin_final_run_dir: Path,
                              csv_path: Path | None) -> dict:
    """W4 four-arm event-conditioning ablation (event-flow vs no-event-flow vs XGB vs Ridge).

    Calls the existing ``evaluate_twin.compare_trajectory_models`` entrypoint and points its
    ``output_dir`` at the run's ``evaluation/`` dir, so ``trajectory_comparison_{metrics,
    paired_tests,summary}`` land beside the rest of the eval artifacts where
    ``figures.figure_ablation`` (and the before/after delta report) look for them. The
    controlled no-event arm is trained INLINE from the event arm's config with only
    ``use_event=False`` flipped and the SAME ``num_steps`` (a budget-matched, fair ablation)
    under ``evaluation/noevent_twin/``.

    Guarded: any failure is logged (with traceback) and returned as ``status='failed'`` so the
    freeze still finishes and writes its manifest; ``figure_ablation`` then skips gracefully.
    Returns the compare_trajectory_models summary dict augmented with a ``status`` field.
    """
    eval_dir = frozen_dir / "evaluation"
    try:
        summary = ev_twin.compare_trajectory_models(
            event_twin_run=twin_final_run_dir,
            csv_path=csv_path,
            output_dir=eval_dir,
            noevent_twin_run=None,
            pipeline=None,
            n_samples=cfg.n_samples,
            n_steps=cfg.n_steps,
            seed=cfg.eval_seed,
            device_name=cfg.device,
            include_baselines=True,
        )
        return {"status": "ok", **summary}
    except Exception as exc:  # noqa: BLE001 - a failed ablation must never lose the run
        traceback.print_exc()
        print(f"[freeze] WARNING: trajectory comparison FAILED ({type(exc).__name__}: {exc}); "
              "continuing without the ablation CSVs (figure_ablation will skip).", file=sys.stderr)
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "output_dir": str(eval_dir)}


def build_figures(cfg: FreezeConfig, frozen_dir: Path, csv_path: Path | None) -> dict:
    """Render every W6 journal figure (main + supplement) via the documented
    ``python -m figures.build_all`` entrypoint, in a CHILD PROCESS.

    A subprocess is deliberate: figure rendering re-loads the twin and re-samples, and
    running it isolated means even a hard crash (segfault / OOM / dual-OpenMP) in figures
    cannot lose the GBM / twin / evaluation / table / ablation artifacts already on disk,
    nor stop RUN_MANIFEST.json from being written. ``build_all`` already guards each figure
    individually (one figure failing -> SKIP/FAIL, the rest still render) and returns exit
    code 1 iff at least one figure FAILED, else 0; that is surfaced here as the status
    (``ok`` / ``partial``). The subprocess call is itself wrapped so a launch failure or
    timeout is logged and returned as ``status='failed'`` without aborting the freeze.
    """
    figures_dir = frozen_dir / "figures"
    cmd = [
        sys.executable, "-m", "figures.build_all",
        "--run", str(frozen_dir),
        "--out", str(figures_dir),
        "--device", cfg.device,
        "--n-samples", str(cfg.n_samples),
        "--n-steps", str(cfg.n_steps),
        "--n-boot", str(cfg.n_boot),
        "--seed", str(cfg.eval_seed),
    ]
    # CONSORT + re-sampling need a data source: pass the same one the freeze used.
    if cfg.use_db:
        cmd.append("--use-db")
    elif csv_path is not None:
        cmd += ["--csv", str(csv_path)]
    env = dict(os.environ)
    if sys.platform == "darwin":
        env.setdefault("OMP_NUM_THREADS", "1")  # macOS torch+xgboost dual-OpenMP guard
    print(f"[freeze] figures cmd: {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=str(fm.REPO_ROOT), env=env, timeout=7200)
        status = "ok" if proc.returncode == 0 else "partial"
        if status == "partial":
            print(f"[freeze] WARNING: figures.build_all exited {proc.returncode} (>=1 figure "
                  "FAILED; see the build summary above). The rest of the run is intact.",
                  file=sys.stderr)
        return {"status": status, "returncode": proc.returncode,
                "output_dir": str(figures_dir), "command": " ".join(cmd)}
    except Exception as exc:  # noqa: BLE001 - figures must never lose the run
        traceback.print_exc()
        print(f"[freeze] WARNING: figures build FAILED to run ({type(exc).__name__}: {exc}); "
              "the rest of the run is intact.", file=sys.stderr)
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "output_dir": str(figures_dir), "command": " ".join(cmd)}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def freeze(cfg: FreezeConfig) -> Path:
    if not cfg.use_db and not cfg.csv_path:
        raise RuntimeError("no input: pass csv_path (local) or use_db=True (Cosmos VM).")
    if not cfg.final_train_best:
        raise RuntimeError("final_train_best must be True: the evaluator needs a final twin run dir.")

    # Reproducibility: seed the module RNGs up front. With --deterministic we also
    # flip on torch.use_deterministic_algorithms (may raise on an unsupported op, so
    # it stays opt-in); without it we still pin the seeds, which is harmless.
    if cfg.deterministic:
        fm.enable_determinism(cfg.seed)
    else:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    frozen_dir = Path(cfg.output_root) / timestamp
    frozen_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(cfg.csv_path) if cfg.csv_path else None
    print(f"[freeze] frozen run dir -> {frozen_dir}")
    print(f"[freeze] backend = {'xgboost' if gb.xgboost_available() else 'hist_gradient_boosting'} "
          f"| split={cfg.split_strategy} seed={cfg.seed}/{cfg.split_seed} "
          f"| n_trials={cfg.n_trials} twin_steps={cfg.twin_num_steps} deterministic={cfg.deterministic}")

    # --- 1. attrition / CONSORT + missingness --------------------------------- #
    print("\n[freeze 1/6] debug_attrition ...", flush=True)
    attrition_report = attrition.run(
        csv_path=csv_path, use_db=cfg.use_db, output_dir=frozen_dir / "attrition",
    )

    # Load the cohort ONCE so the pipeline and Table 1 see identical data.
    dataset = fm.load_dataset_from_database() if cfg.use_db else fm.load_dataset_from_csv(csv_path)

    # --- 2. GBM + event-conditioned twin-flow sweep (FRESH twin) -------------- #
    print("\n[freeze 2/6] train_twin_pipeline.run_pipeline ...", flush=True)
    pipe = pipeline.run_pipeline(
        dataset,
        output_dir=str(frozen_dir / "twin_pipeline"),
        split_strategy=cfg.split_strategy,
        split_seed=cfg.split_seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        n_trials=cfg.n_trials,
        twin_num_steps=cfg.twin_num_steps,
        seed=cfg.seed,
        final_train_best=cfg.final_train_best,
    )
    pipeline_dir = Path(pipe["pipeline_dir"])
    pipe_manifest = pipe["manifest"]
    gbm_run_dir = Path(pipe_manifest["gbm_run_dir"])
    twin_final = pipe_manifest.get("twin_final_run_dir")
    if not twin_final:
        raise RuntimeError(
            "twin pipeline produced no final twin run dir (best-config training did not run); "
            "cannot evaluate. Check the Optuna sweep output."
        )
    twin_final_run_dir = Path(twin_final)

    # --- 3. evaluate GBM + flow + simulator (incl. W5 calibration + thresholds) - #
    print("\n[freeze 3/6] evaluate_twin.evaluate ...", flush=True)
    eval_summary = ev_twin.evaluate(
        pipeline=pipeline_dir, gbm_run=None, twin_run=None,
        csv_path=csv_path, output_dir=frozen_dir / "evaluation",
        n_samples=cfg.n_samples, n_steps=cfg.n_steps, seed=cfg.eval_seed,
        n_show=ev.N_SHOW_PER_PROCEDURE, max_lines=ev.MAX_SAMPLE_LINES,
        n_boot=cfg.n_boot, device_name=cfg.device, compare_predictions=None,
    )

    # --- 4. W4 four-arm event-conditioning trajectory ablation (guarded) ------ #
    #     Writes trajectory_comparison_*.csv INTO evaluation/ so figure_ablation and
    #     the before/after delta report find them beside the rest of the eval artifacts.
    print("\n[freeze 4/6] evaluate_twin.compare_trajectory_models (event-conditioning ablation) ...", flush=True)
    traj_result = run_trajectory_comparison(cfg, frozen_dir, twin_final_run_dir, csv_path)

    # --- 5. Table 1 (same split_strategy + seed -> aligns with the models) ---- #
    print("\n[freeze 5/6] make_table_one.generate ...", flush=True)
    table_dir = t1.generate(
        dataset,
        t1.TableConfig(
            output_dir=str(frozen_dir / "table_one"),
            split_seed=cfg.split_seed,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            test_frac=cfg.test_frac,
            continuous=cfg.table_continuous,
            split_strategy=cfg.split_strategy,
        ),
    )

    # --- 6. W6 journal figures: main + supplement (subprocess, guarded) ------- #
    print("\n[freeze 6/6] figures.build_all ...", flush=True)
    figures_result = build_figures(cfg, frozen_dir, csv_path)

    # --- RUN_MANIFEST.json ----------------------------------------------------- #
    manifest = build_manifest(
        cfg, frozen_dir, dataset, csv_path, attrition_report, pipeline_dir,
        pipe_manifest, gbm_run_dir, twin_final_run_dir, eval_summary, table_dir,
        traj_result, figures_result,
    )
    manifest_path = frozen_dir / "RUN_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"\n[freeze] wrote {manifest_path}")
    print(f"[freeze] backend that ran: {manifest['backend']['gbm_backend']} "
          f"(xgboost=={manifest['backend']['versions']['xgboost']})")
    print(f"[freeze] patient_feature_width={manifest['code']['patient_feature_width']} "
          f"checkpoint_in_features={manifest['code']['twin_checkpoint_encoder_in_features']} "
          f"match={manifest['code']['checkpoint_width_matches_code']}")
    fig_art = manifest["artifacts"]["figures"]
    print(f"[freeze] trajectory ablation: {traj_result.get('status')} "
          f"(arms={traj_result.get('arms')}) | figures: {fig_art['status']} "
          f"(main={len(fig_art['main'])} supplement={len(fig_art['supplement'])})")
    print(f"[freeze] DONE. All artifacts under {frozen_dir}")
    return frozen_dir


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--csv", "--csv-path", dest="csv_path", type=str, default=None,
                        help="Local CSV export (post-SQL). Omit and pass --db to query Cosmos.")
    source.add_argument("--db", action="store_true",
                        help="Load from Cosmos MBSCohort (VM). This is the DEFAULT when neither "
                             "--csv nor --smoke is given, so a bare `python freeze_run.py` runs it.")
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--split-strategy", type=str, default="surgery", choices=["surgery", "temporal", "outcome"])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="Enable global torch determinism via fm.enable_determinism (opt-in).")
    p.add_argument("--n-trials", type=int, default=tune_twin.N_TRIALS)
    p.add_argument("--twin-num-steps", type=int, default=None,
                   help="Override the twin sweep's per-trial training steps (smoke tests).")
    p.add_argument("--no-final-train", action="store_true",
                   help="Skip final best-config training (then evaluation cannot run; not for a freeze).")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--continuous", dest="table_continuous", choices=["median", "mean"], default="median")
    p.add_argument("--smoke", action="store_true",
                   help="Minimal-but-real settings for a fast fake-cohort smoke test.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Source resolution. Explicit --csv / --db always win. With neither given:
    #   --smoke   -> the bundled fake cohort (a smoke test must never hit the live DB)
    #   otherwise -> --db, so a bare `python freeze_run.py` runs the real Cosmos cohort.
    if not args.csv_path and not args.db:
        if args.smoke:
            args.csv_path = str(fm.REPO_ROOT / "fake_data" / "fake_mbs_cohort.csv")
            print(f"[freeze] --smoke with no source: using fake cohort {args.csv_path}", flush=True)
        else:
            args.db = True
            print("[freeze] no source given: defaulting to --db (Cosmos MBSCohort).", flush=True)

    cfg = FreezeConfig(
        csv_path=args.csv_path,
        use_db=bool(args.db),
        output_root=args.output_root,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        eval_seed=args.eval_seed,
        deterministic=bool(args.deterministic),
        n_trials=args.n_trials,
        twin_num_steps=args.twin_num_steps,
        final_train_best=not args.no_final_train,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        n_boot=args.n_boot,
        device=args.device,
        table_continuous=args.table_continuous,
    )
    if args.smoke:
        cfg = replace(
            cfg,
            n_trials=min(cfg.n_trials, 2),
            twin_num_steps=cfg.twin_num_steps or 40,
            n_samples=min(cfg.n_samples, 64),
            n_steps=min(cfg.n_steps, 20),
            n_boot=min(cfg.n_boot, 100),
        )

    try:
        freeze(cfg)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
