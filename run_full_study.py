#!/usr/bin/env python3
"""Run the paired internal/temporal study on the Cosmos VM.

Normal run::

    python run_full_study.py

Resume a legacy no-Git study after steps 1-3 completed::

    python run_full_study.py --resume-study runs/full_study/study_<timestamp> \
        --allow-missing-code-identity

New freezes record deterministic source fingerprints, so future paired runs do
not need Git and do not need the legacy override.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG: Path | None = None


def banner(message: str) -> None:
    print(f"\n{'=' * 70}\n>> {message}\n{'=' * 70}", flush=True)


def fail(message: str) -> "NoReturn":
    suffix = f"\n(full log: {LOG})" if LOG else ""
    raise SystemExit(f"\nERROR: {message}{suffix}")


def run(*args: object) -> None:
    command = [str(arg) for arg in args]
    print("  running:", " ".join(command), flush=True)
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    if process.wait():
        fail(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def python_has_dependencies(python: str) -> bool:
    check = "import numpy,pandas,sklearn,xgboost,torch"
    return subprocess.run(
        [python, "-c", check], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def select_python() -> str:
    candidates = [
        os.environ.get("PYTHON"),
        str(ROOT.parent / "mbsaqip_flow/.venv/Scripts/python.exe"),
        str(ROOT / ".venv/Scripts/python.exe"),
        sys.executable,
        "python.exe",
        "python3",
        "python",
    ]
    for candidate in filter(None, candidates):
        try:
            if python_has_dependencies(candidate):
                resolved = shutil.which(candidate) or candidate
                return str(Path(resolved).resolve())
        except OSError:
            pass
    fail(
        "Could not find Python with numpy/pandas/sklearn/xgboost/torch. "
        r"Activate the project venv or set PYTHON=C:\path\to\.venv\Scripts\python.exe."
    )


def ensure_project_python() -> None:
    """Re-launch under the project venv when Windows started another Python."""
    if os.environ.get("MBSAQIP_PYTHON_SELECTED"):
        return
    python = select_python()
    if Path(python) != Path(sys.executable).resolve():
        env = os.environ.copy()
        env["MBSAQIP_PYTHON_SELECTED"] = "1"
        completed = subprocess.run([python, str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
        raise SystemExit(completed.returncode)


class Tee:
    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def newest(pattern: str) -> Path | None:
    matches = [Path(path) for path in glob.glob(pattern)]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def validated_freeze(output_root: Path, expected_split: str) -> tuple[Path, dict]:
    """Resolve one isolated freeze output and verify the full-study components succeeded."""
    manifests = sorted(output_root.glob("*/RUN_MANIFEST.json"))
    if len(manifests) != 1:
        fail(
            f"Expected exactly one {expected_split} RUN_MANIFEST.json under {output_root}, "
            f"found {len(manifests)}."
        )

    manifest_path = manifests[0]
    frozen = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Could not read {manifest_path}: {exc}")

    actual_split = (manifest.get("split") or {}).get("split_strategy")
    if actual_split != expected_split:
        fail(
            f"Expected split_strategy={expected_split!r}, but {manifest_path} records "
            f"{actual_split!r}."
        )

    causal = manifest.get("causal_distributional")
    if not isinstance(causal, dict) or causal.get("status") == "failed":
        detail = causal.get("error") if isinstance(causal, dict) else "manifest block missing"
        fail(f"The {expected_split} causal/distributional pass failed: {detail}")
    tte = causal.get("causal_tte")
    if not isinstance(tte, dict) or tte.get("status") in {"failed", "skipped"}:
        detail = tte.get("error") if isinstance(tte, dict) else "causal_tte block missing"
        fail(f"The {expected_split} target-trial-emulation pass did not complete: {detail}")

    required_tte = {
        "tte_marginal_effects",
        "tte_propensity_overlap",
        "tte_covariate_balance_love",
        "tte_weights_summary",
    }
    tte_artifacts = tte.get("artifacts") or {}
    missing_tte = sorted(required_tte - set(tte_artifacts))
    missing_tte_files = sorted(
        key for key in required_tte
        if key in tte_artifacts and not Path(tte_artifacts[key]).is_file()
    )
    if missing_tte or missing_tte_files:
        fail(
            f"The {expected_split} target-trial-emulation artifacts are incomplete. "
            f"Missing manifest keys={missing_tte}; missing files={missing_tte_files}."
        )

    distributional_csv = frozen / "evaluation" / "dist_calibration_slope_citl.csv"
    if not distributional_csv.is_file():
        fail(f"Cross-run comparison input is missing: {distributional_csv}")

    figures = ((manifest.get("artifacts") or {}).get("figures") or {})
    if figures.get("status") != "ok":
        fail(
            f"The {expected_split} standard figure build did not complete successfully: "
            f"{figures.get('error') or figures.get('status') or 'status missing'}"
        )

    fairness = manifest.get("fairness")
    if not isinstance(fairness, dict) or fairness.get("status") == "failed":
        detail = fairness.get("error") if isinstance(fairness, dict) else "manifest block missing"
        fail(f"The {expected_split} fairness audit failed: {detail}")

    sizes = (manifest.get("split") or {}).get("split_sizes") or {}
    print(f"  verified frozen run: {frozen}")
    print(
        "  split sizes: "
        f"train={sizes.get('train', '?')} | validation={sizes.get('val', '?')} | "
        f"test={sizes.get('test', '?')}"
    )
    print(
        f"  target-trial artifacts: {len(tte_artifacts)} | "
        f"standard figures: {len(figures.get('main') or [])} main + "
        f"{len(figures.get('supplement') or [])} supplement"
    )
    return frozen, manifest


def _source_sha(manifest: dict) -> str | None:
    fingerprint = (manifest.get("code") or {}).get("source_fingerprint") or {}
    return fingerprint.get("sha256")


def validate_paired_runs(internal: dict, temporal: dict, *,
                         allow_missing_code_identity: bool = False) -> dict:
    """Confirm that split strategy, rather than cohort or code, defines the comparison."""
    internal_hash = (internal.get("input") or {}).get("sha256")
    temporal_hash = (temporal.get("input") or {}).get("sha256")
    if not internal_hash or internal_hash != temporal_hash:
        fail(
            "The internal and temporal runs do not record the same cohort SHA-256; "
            "a cross-run drift comparison would not be valid."
        )
    print(f"  paired-run cohort SHA-256: {internal_hash}")

    internal_source = _source_sha(internal)
    temporal_source = _source_sha(temporal)
    if internal_source or temporal_source:
        if not internal_source or not temporal_source:
            fail(
                "Only one paired run records a source fingerprint; code identity cannot be verified."
            )
        if internal_source != temporal_source:
            fail(
                "The internal and temporal source fingerprints differ; a cross-run drift "
                "comparison would not be valid."
            )
        for label, manifest in (("internal", internal), ("temporal", temporal)):
            stable = (manifest.get("code") or {}).get("source_stable_during_run")
            if stable is False:
                fail(f"The {label} run records source changes during its freeze; it cannot be paired.")
        identity = {"kind": "source_sha256", "value": internal_source, "verified": True}
        print(f"  paired-run source SHA-256: {internal_source}")
    else:
        internal_sha = (internal.get("git") or {}).get("sha")
        temporal_sha = (temporal.get("git") or {}).get("sha")
        if internal_sha or temporal_sha:
            if not internal_sha or not temporal_sha or internal_sha != temporal_sha:
                fail(
                    "The internal and temporal Git revisions differ or one is missing; "
                    "a cross-run drift comparison would not be valid."
                )
            identity = {"kind": "git", "value": internal_sha, "verified": True}
            print(f"  paired-run Git revision: {internal_sha}")
        elif allow_missing_code_identity:
            identity = {"kind": "legacy_unverified", "value": None, "verified": False}
            print(
                "  WARNING: legacy manifests contain no Git revision or source fingerprint.\n"
                "  Continuing only because --allow-missing-code-identity was explicitly supplied.\n"
                "  This override does not permit a known code or cohort mismatch."
            )
        else:
            fail(
                "Both paired runs have no Git revision or source fingerprint. For legacy no-Git "
                "VM results, resume with --allow-missing-code-identity only after confirming that "
                "no source files were transferred between the two freezes."
            )
    if identity["verified"]:
        print("  comparison validity check passed: same cohort and code, different time split")
    else:
        print("  legacy comparison override accepted: same cohort; code identity remains unverified")
    return identity


def find_pipeline(frozen: Path) -> Path:
    pipeline = newest(str(frozen / "twin_pipeline/pipeline_*"))
    if not pipeline:
        fail(f"Could not find the trained twin pipeline under {frozen / 'twin_pipeline'}.")
    return pipeline


def validate_quantile_comparison(output_dir: Path, label: str) -> None:
    required = [
        output_dir / "quantile_comparison_summary.json",
        output_dir / "quantile_comparison_metrics.csv",
        output_dir / "quantile_comparison_calibration.csv",
        output_dir / "quantile_comparison_paired_tests.csv",
        output_dir / "quantile_comparison_scorecard.csv",
        output_dir / "quantile_comparison.png",
        output_dir / "quantile_comparison_transfer.png",
        output_dir / "quantile_comparison_transfer.jpg",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        fail(f"The {label} quantile comparison did not write required outputs: {missing}")
    print(f"  verified {label} quantile comparison: {output_dir}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_resume_study(study_dir: str | Path) -> tuple[Path, dict, Path, dict, Path]:
    """Load the two completed freezes and their unchanged shared cohort for post-processing."""
    study_dir = Path(study_dir).expanduser().resolve()
    if not study_dir.is_dir():
        fail(f"Resume study directory does not exist: {study_dir}")

    internal_frozen, internal_manifest = validated_freeze(
        study_dir / "internal_validation", "surgery"
    )
    temporal_frozen, temporal_manifest = validated_freeze(
        study_dir / "temporal_validation", "temporal"
    )

    source = os.environ.get("STUDY_CSV")
    if source:
        csv_path = Path(source).expanduser().resolve()
    else:
        input_block = internal_manifest.get("input") or {}
        source = input_block.get("source_abspath") or input_block.get("source")
        if not source:
            fail(
                "The internal manifest has no cohort CSV path. Set STUDY_CSV to the exact CSV "
                "used by both freezes."
            )
        csv_path = Path(source).expanduser().resolve()
    if not csv_path.is_file():
        fail(f"The shared cohort CSV needed for resume does not exist: {csv_path}")

    expected_hash = (internal_manifest.get("input") or {}).get("sha256")
    actual_hash = sha256_file(csv_path)
    if not expected_hash or actual_hash != expected_hash:
        fail(
            f"The resume cohort CSV does not match the frozen runs: expected {expected_hash}, "
            f"found {actual_hash} at {csv_path}."
        )
    return internal_frozen, internal_manifest, temporal_frozen, temporal_manifest, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-study",
        type=Path,
        default=None,
        help="Reuse completed internal/temporal freezes in this study directory and run steps 4-6 only.",
    )
    parser.add_argument(
        "--allow-missing-code-identity",
        action="store_true",
        help="Resume legacy no-Git manifests that predate source fingerprints. Never permits a known mismatch.",
    )
    parser.add_argument(
        "--force-comparisons",
        action="store_true",
        help="Rebuild comparison outputs even when complete artifacts already exist.",
    )
    args = parser.parse_args()
    if args.allow_missing_code_identity and not args.resume_study:
        parser.error("--allow-missing-code-identity is only valid with --resume-study")
    return args


def _quantile_outputs_complete(output_dir: Path) -> bool:
    return all((output_dir / name).is_file() for name in (
        "quantile_comparison_summary.json",
        "quantile_comparison_metrics.csv",
        "quantile_comparison_calibration.csv",
        "quantile_comparison_paired_tests.csv",
        "quantile_comparison_scorecard.csv",
        "quantile_comparison.png",
        "quantile_comparison_transfer.png",
        "quantile_comparison_transfer.jpg",
    ))


def run_postprocessing(*, study_dir: Path, csv_path: Path,
                       internal_frozen: Path, internal_manifest: dict, internal_pipeline: Path,
                       temporal_frozen: Path, temporal_manifest: dict, temporal_pipeline: Path,
                       smoke: bool, allow_missing_code_identity: bool = False,
                       force_comparisons: bool = False) -> Path:
    """Run paired drift and baseline comparisons against two completed freezes."""
    compare_args = ["--n-samples", "100", "--n-steps", "25"] if smoke else []

    banner("Step 4/6: paired internal-versus-temporal distributional comparison")
    code_identity = validate_paired_runs(
        internal_manifest,
        temporal_manifest,
        allow_missing_code_identity=allow_missing_code_identity,
    )
    cross_run_figures = study_dir / "cross_run_comparison"
    expected_comparison = [
        cross_run_figures / f"figC2_distributional.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ]
    if force_comparisons or not all(path.is_file() for path in expected_comparison):
        run(
            sys.executable, "-m", "figures.figure_distributional",
            "--eval-dir", internal_frozen / "evaluation",
            "--temporal-eval-dir", temporal_frozen / "evaluation",
            "--out", cross_run_figures,
        )
    else:
        print("  existing cross-run distributional figures are complete; skipping rebuild")
    missing_comparison = [str(path) for path in expected_comparison if not path.is_file()]
    if missing_comparison:
        fail(f"Cross-run comparison figure build did not write: {missing_comparison}")
    print("  cross-run figure overlays internal and temporal calibration slope/CITL in panel A")
    print(f"  comparison figures: {cross_run_figures}")

    banner("Step 5/6: internal-fold flow versus quantile baselines")
    print("  Baselines inherit the internal pipeline's exact patient split.")
    internal_comparison = internal_frozen / "quantile_comparison"
    if force_comparisons or not _quantile_outputs_complete(internal_comparison):
        run(
            sys.executable, ROOT / "compare_quantile_baselines.py",
            "--csv", csv_path, "--pipeline", internal_pipeline, "--with-point",
            "--output-dir", internal_comparison, *compare_args,
        )
    else:
        print("  existing internal-fold comparison is complete; skipping rebuild")
    validate_quantile_comparison(internal_comparison, "internal-fold")

    banner("Step 6/6: temporal-fold flow versus quantile baselines")
    print("  Baselines inherit the temporal pipeline's exact later-era test split.")
    temporal_comparison = temporal_frozen / "quantile_comparison"
    if force_comparisons or not _quantile_outputs_complete(temporal_comparison):
        run(
            sys.executable, ROOT / "compare_quantile_baselines.py",
            "--csv", csv_path, "--pipeline", temporal_pipeline, "--with-point",
            "--output-dir", temporal_comparison, *compare_args,
        )
    else:
        print("  existing temporal-fold comparison is complete; skipping rebuild")
    validate_quantile_comparison(temporal_comparison, "temporal-fold")

    paired_manifest_path = study_dir / "PAIRED_STUDY_MANIFEST.json"
    paired_manifest = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "smoke": smoke,
        "shared_cohort": {
            "csv": str(csv_path),
            "sha256": (internal_manifest.get("input") or {}).get("sha256"),
            "n_patients": (internal_manifest.get("input") or {}).get("n_patients"),
        },
        "code_identity": code_identity,
        "git_revision": (internal_manifest.get("git") or {}).get("sha"),
        "internal_validation": {
            "split_strategy": "surgery",
            "frozen_run": str(internal_frozen),
            "run_manifest": str(internal_frozen / "RUN_MANIFEST.json"),
            "evaluation": str(internal_frozen / "evaluation"),
            "figures": str(internal_frozen / "figures"),
            "quantile_comparison": str(internal_comparison),
            "quantile_transfer_figure": str(internal_comparison / "quantile_comparison_transfer.png"),
            "quantile_transfer_jpg": str(internal_comparison / "quantile_comparison_transfer.jpg"),
        },
        "temporal_validation": {
            "split_strategy": "temporal",
            "frozen_run": str(temporal_frozen),
            "run_manifest": str(temporal_frozen / "RUN_MANIFEST.json"),
            "evaluation": str(temporal_frozen / "evaluation"),
            "figures": str(temporal_frozen / "figures"),
            "quantile_comparison": str(temporal_comparison),
            "quantile_transfer_figure": str(temporal_comparison / "quantile_comparison_transfer.png"),
            "quantile_transfer_jpg": str(temporal_comparison / "quantile_comparison_transfer.jpg"),
        },
        "cross_run_comparison": {
            "figures": [str(path) for path in expected_comparison],
            "internal_input": str(internal_frozen / "evaluation" / "dist_calibration_slope_citl.csv"),
            "temporal_input": str(temporal_frozen / "evaluation" / "dist_calibration_slope_citl.csv"),
        },
        "log": str(LOG),
    }
    paired_manifest_path.write_text(json.dumps(paired_manifest, indent=2), encoding="utf-8")
    print(f"  wrote paired-study manifest: {paired_manifest_path}")

    banner(f"DONE - paired full-study results are under: {study_dir}")
    print(f"""  Key outputs:
    Paired study manifest:       {paired_manifest_path}

    Internal run manifest:       {internal_frozen / 'RUN_MANIFEST.json'}
    Internal evaluation:         {internal_frozen / 'evaluation'}
    Internal journal figures:    {internal_frozen / 'figures'}
    Internal quantile comparison: {internal_comparison}
    Internal transfer card:      {internal_comparison / 'quantile_comparison_transfer.jpg'}

    Temporal run manifest:       {temporal_frozen / 'RUN_MANIFEST.json'}
    Temporal evaluation:         {temporal_frozen / 'evaluation'}
    Temporal journal figures:    {temporal_frozen / 'figures'}
    Temporal quantile comparison: {temporal_comparison}
    Temporal transfer card:      {temporal_comparison / 'quantile_comparison_transfer.jpg'}

    Cross-run drift figures:     {cross_run_figures}
    Shared cohort CSV:           {csv_path}
    Full log:                    {LOG}
  finished: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}""")
    return paired_manifest_path


def main() -> None:
    global LOG
    ensure_project_python()
    args = parse_args()
    os.chdir(ROOT)

    started = datetime.now()
    if args.resume_study:
        study_dir = args.resume_study.expanduser().resolve()
        run_kind = "resume"
    else:
        study_dir = ROOT / "runs/full_study" / f"study_{started:%Y%m%d_%H%M%S}"
        study_dir.mkdir(parents=True, exist_ok=False)
        run_kind = "full"
    log_dir = ROOT / "runs/study_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    LOG = log_dir / f"run_full_study_{run_kind}_{started:%Y%m%d_%H%M%S}.log"
    log_file = LOG.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"MBSAQIP digital-twin study - {run_kind} run")
    print(f"started: {started.astimezone():%Y-%m-%d %H:%M:%S %Z}")
    print(f"logging to: {LOG}")
    print(f"paired study output: {study_dir}")
    print("validation plan:")
    print("  1. Internal validation: random split stratified by surgery type")
    print("  2. Temporal validation: earliest surgeries train, latest surgeries test")
    print("  3. Cross-run calibration-drift figure using both held-out folds")

    banner("Preflight: checking the Python environment")
    print(f"  using Python: {sys.executable}")
    code = (
        "import sklearn,xgboost,torch,numpy,scipy; "
        "print(f'  sklearn {sklearn.__version__} | xgboost {xgboost.__version__} | '"
        "f'torch {torch.__version__} | numpy {numpy.__version__} | scipy {scipy.__version__}'); "
        "assert int(xgboost.__version__.split('.')[0]) >= 2"
    )
    check = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    print(check.stdout, end="")
    if check.returncode:
        fail("xgboost >= 2.0 and the project dependencies are required.")

    smoke = bool(os.environ.get("STUDY_SMOKE"))
    smoke_args = ["--smoke"] if smoke else []
    if smoke:
        print("  STUDY_SMOKE set: using fast sanity settings.")

    if args.resume_study:
        banner("Resume preflight: loading completed internal and temporal freezes")
        (internal_frozen, internal_manifest,
         temporal_frozen, temporal_manifest, csv_path) = resolve_resume_study(study_dir)
        internal_pipeline = find_pipeline(internal_frozen)
        temporal_pipeline = find_pipeline(temporal_frozen)
        print("  completed model freezes found; steps 1-3 will not be rerun")
        print(f"  shared cohort CSV: {csv_path}")
        run_postprocessing(
            study_dir=study_dir,
            csv_path=csv_path,
            internal_frozen=internal_frozen,
            internal_manifest=internal_manifest,
            internal_pipeline=internal_pipeline,
            temporal_frozen=temporal_frozen,
            temporal_manifest=temporal_manifest,
            temporal_pipeline=temporal_pipeline,
            smoke=smoke,
            allow_missing_code_identity=args.allow_missing_code_identity,
            force_comparisons=args.force_comparisons,
        )
        return

    banner("Step 1/6: preparing one shared cohort CSV for both validation runs")
    csv_path = Path(os.environ.get("STUDY_CSV", ROOT / "data/cosmos_mbs_flow_input.csv"))
    csv_path = csv_path.expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    import train_flow_matching as fm

    if os.environ.get("STUDY_CSV"):
        print(f"  using STUDY_CSV: {csv_path}")
        dataset = fm.load_dataset_from_csv(str(csv_path))
    else:
        print("  querying MBSCohort via pyodbc (live VM database access required)...")
        frame = fm.load_mbs_from_database()
        frame.to_csv(csv_path, index=False)
        dataset = fm.load_dataset_from_csv(str(csv_path))
        print(f"  exported {len(frame)} database rows -> {csv_path}")
    print(f"  cohort OK: {dataset.x.shape[0]} patients, {len(fm.PATIENT_FEATURES)} features")

    banner("Step 2/6: internal validation freeze (random, surgery-stratified split)")
    print("  Purpose: estimate performance on held-out patients from the same overall era.")
    internal_root = study_dir / "internal_validation"
    run(
        sys.executable, ROOT / "freeze_run.py", "--csv", csv_path,
        "--output-root", internal_root, "--split-strategy", "surgery",
        "--causal", "--fairness", *smoke_args,
    )
    internal_frozen, internal_manifest = validated_freeze(internal_root, "surgery")
    internal_pipeline = find_pipeline(internal_frozen)

    banner("Step 3/6: temporal validation freeze (earlier train, later test)")
    print("  Purpose: test transport to a later clinical era and expose temporal drift.")
    print("  Note: long-horizon outcomes may have smaller n because later patients have less follow-up.")
    temporal_root = study_dir / "temporal_validation"
    run(
        sys.executable, ROOT / "freeze_run.py", "--csv", csv_path,
        "--output-root", temporal_root, "--split-strategy", "temporal",
        "--causal", "--fairness", *smoke_args,
    )
    temporal_frozen, temporal_manifest = validated_freeze(temporal_root, "temporal")
    temporal_pipeline = find_pipeline(temporal_frozen)
    run_postprocessing(
        study_dir=study_dir,
        csv_path=csv_path,
        internal_frozen=internal_frozen,
        internal_manifest=internal_manifest,
        internal_pipeline=internal_pipeline,
        temporal_frozen=temporal_frozen,
        temporal_manifest=temporal_manifest,
        temporal_pipeline=temporal_pipeline,
        smoke=smoke,
        force_comparisons=args.force_comparisons,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Interrupted by user.")
    except Exception as exc:
        fail(str(exc))
