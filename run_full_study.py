#!/usr/bin/env python3
'''
full study on the cosmos VM
'''
from __future__ import annotations

import glob
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
        completed = subprocess.run([python, str(Path(__file__).resolve())], env=env)
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


def validate_paired_runs(internal: dict, temporal: dict) -> None:
    """Confirm that split strategy, rather than cohort or code, defines the comparison."""
    internal_hash = (internal.get("input") or {}).get("sha256")
    temporal_hash = (temporal.get("input") or {}).get("sha256")
    if not internal_hash or internal_hash != temporal_hash:
        fail(
            "The internal and temporal runs do not record the same cohort SHA-256; "
            "a cross-run drift comparison would not be valid."
        )
    internal_sha = (internal.get("git") or {}).get("sha")
    temporal_sha = (temporal.get("git") or {}).get("sha")
    if not internal_sha or internal_sha != temporal_sha:
        fail(
            "The internal and temporal runs do not record the same git revision; "
            "a cross-run drift comparison would not be valid."
        )
    print(f"  paired-run cohort SHA-256: {internal_hash}")
    print(f"  paired-run git revision:  {internal_sha}")
    print("  comparison validity check passed: same cohort and code, different time split")


def find_pipeline(frozen: Path) -> Path:
    pipeline = newest(str(frozen / "twin_pipeline/pipeline_*"))
    if not pipeline:
        fail(f"Could not find the trained twin pipeline under {frozen / 'twin_pipeline'}.")
    return pipeline


def validate_quantile_comparison(output_dir: Path, label: str) -> None:
    required = [
        output_dir / "quantile_comparison_summary.json",
        output_dir / "quantile_comparison_metrics.csv",
        output_dir / "quantile_comparison_paired_tests.csv",
        output_dir / "quantile_comparison.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        fail(f"The {label} quantile comparison did not write required outputs: {missing}")
    print(f"  verified {label} quantile comparison: {output_dir}")


def main() -> None:
    global LOG
    ensure_project_python()
    os.chdir(ROOT)

    started = datetime.now()
    study_dir = ROOT / "runs/full_study" / f"study_{started:%Y%m%d_%H%M%S}"
    study_dir.mkdir(parents=True, exist_ok=False)
    log_dir = ROOT / "runs/study_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    LOG = log_dir / f"run_full_study_{started:%Y%m%d_%H%M%S}.log"
    log_file = LOG.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print("MBSAQIP digital-twin study - full run")
    print(f"started: {started.astimezone():%Y-%m-%d %H:%M:%S %Z}")
    print(f"logging to: {LOG.relative_to(ROOT)}")
    print(f"paired study output: {study_dir.relative_to(ROOT)}")
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
    compare_args = ["--n-samples", "100", "--n-steps", "25"] if smoke else []
    if smoke:
        print("  STUDY_SMOKE set: using fast sanity settings.")

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

    banner("Step 4/6: paired internal-versus-temporal distributional comparison")
    validate_paired_runs(internal_manifest, temporal_manifest)
    cross_run_figures = study_dir / "cross_run_comparison"
    run(
        sys.executable, "-m", "figures.figure_distributional",
        "--eval-dir", internal_frozen / "evaluation",
        "--temporal-eval-dir", temporal_frozen / "evaluation",
        "--out", cross_run_figures,
    )
    expected_comparison = [
        cross_run_figures / f"figC2_distributional.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ]
    missing_comparison = [str(path) for path in expected_comparison if not path.is_file()]
    if missing_comparison:
        fail(f"Cross-run comparison figure build did not write: {missing_comparison}")
    print("  cross-run figure overlays internal and temporal calibration slope/CITL in panel A")
    print(f"  comparison figures: {cross_run_figures}")

    banner("Step 5/6: internal-fold flow versus quantile baselines")
    print("  Baselines inherit the internal pipeline's exact patient split.")
    internal_comparison = internal_frozen / "quantile_comparison"
    run(
        sys.executable, ROOT / "compare_quantile_baselines.py",
        "--csv", csv_path, "--pipeline", internal_pipeline, "--with-point",
        "--output-dir", internal_comparison, *compare_args,
    )
    validate_quantile_comparison(internal_comparison, "internal-fold")

    banner("Step 6/6: temporal-fold flow versus quantile baselines")
    print("  Baselines inherit the temporal pipeline's exact later-era test split.")
    temporal_comparison = temporal_frozen / "quantile_comparison"
    run(
        sys.executable, ROOT / "compare_quantile_baselines.py",
        "--csv", csv_path, "--pipeline", temporal_pipeline, "--with-point",
        "--output-dir", temporal_comparison, *compare_args,
    )
    validate_quantile_comparison(temporal_comparison, "temporal-fold")

    paired_manifest_path = study_dir / "PAIRED_STUDY_MANIFEST.json"
    paired_manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "smoke": smoke,
        "shared_cohort": {
            "csv": str(csv_path),
            "sha256": (internal_manifest.get("input") or {}).get("sha256"),
            "n_patients": (internal_manifest.get("input") or {}).get("n_patients"),
        },
        "git_revision": (internal_manifest.get("git") or {}).get("sha"),
        "internal_validation": {
            "split_strategy": "surgery",
            "frozen_run": str(internal_frozen),
            "run_manifest": str(internal_frozen / "RUN_MANIFEST.json"),
            "evaluation": str(internal_frozen / "evaluation"),
            "figures": str(internal_frozen / "figures"),
            "quantile_comparison": str(internal_comparison),
        },
        "temporal_validation": {
            "split_strategy": "temporal",
            "frozen_run": str(temporal_frozen),
            "run_manifest": str(temporal_frozen / "RUN_MANIFEST.json"),
            "evaluation": str(temporal_frozen / "evaluation"),
            "figures": str(temporal_frozen / "figures"),
            "quantile_comparison": str(temporal_comparison),
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

    Temporal run manifest:       {temporal_frozen / 'RUN_MANIFEST.json'}
    Temporal evaluation:         {temporal_frozen / 'evaluation'}
    Temporal journal figures:    {temporal_frozen / 'figures'}
    Temporal quantile comparison: {temporal_comparison}

    Cross-run drift figures:     {cross_run_figures}
    Shared cohort CSV:           {csv_path}
    Full log:                    {LOG}
  finished: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Interrupted by user.")
    except Exception as exc:
        fail(str(exc))
