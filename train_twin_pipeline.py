"""Train orchestration for the modular digital twin (command 2 of 3).

One command that:

  1. fits the calibrated composite-MACE GBM (``gbm_mace_baseline``) on the SHARED
     train split and saves it (as a config-deterministic run dir + a joblib pickle
     of the fitted unweighted+calibrated model); then
  2. launches the event-conditioned twin-flow Optuna sweep
     (``tune_flow_matching_twin_optuna``) on the *same* patient-for-patient split.

Why sequence them in one script but train them INDEPENDENTLY
------------------------------------------------------------
The twin flow is trained by teacher-forcing the TRUE observed event label (see
``train_flow_matching_twin.py``); it never consumes the GBM at train or tune time.
So there is no modelling dependency between the two -- the only reasons to run them
together are (a) to guarantee they share one split/seed/fracs so risk and
trajectory are sampled coherently for the *same* held-out patients, and (b) to have
a leak-free, train-only GBM sitting on disk so the evaluator/simulator can draw
held-out events e ~ Bernoulli(p_GBM(x)) without refitting on test data.

Both models are pinned to the same ``split_strategy`` / ``split_seed`` /
``train_frac`` / ``val_frac`` / ``test_frac`` (asserted before the sweep starts).
A manifest is written before the sweep and updated after it (save-as-you-go), so an
interrupted run still points at the finished GBM.

Run (local smoke test)::

    python train_twin_pipeline.py --csv fake_data/fake_mbs_cohort.csv \
        --n-trials 3 --twin-num-steps 60

Run (standalone against Cosmos)::

    python train_twin_pipeline.py --n-trials 500
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import warnings

import train_flow_matching as fm
import gbm_mace_baseline as gb
import tune_flow_matching_twin_optuna as tune_twin

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


DEFAULT_OUTPUT_DIR = fm.REPO_ROOT / "runs" / "twin_pipeline"

# Split knobs that the GBM and the twin flow MUST agree on for a shared,
# patient-for-patient split. Asserted before the flow sweep launches.
SHARED_SPLIT_KEYS = ("split_strategy", "split_seed", "train_frac", "val_frac", "test_frac")


def report_saved(path: Path, description: str = "") -> Path:
    tag = f" {description}" if description else ""
    print(f"  [saved]{tag} -> {path}", flush=True)
    return path


def make_pipeline_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir = output_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    return pipeline_dir


def persist_gbm_model(dataset: fm.FlowDataset, gbm_cfg: gb.GBMConfig, gbm_run_dir: Path) -> str | None:
    """Best-effort joblib pickle of the fitted unweighted + calibrated GBM.

    The run dir's ``config.json`` already reproduces this model deterministically
    (trees are seeded), so the evaluator can always retrain from it; this pickle is
    a convenience so it doesn't have to. Never fatal: returns the path or None.
    """
    if joblib is None:
        warnings.warn("joblib unavailable; skipping GBM pickle (config.json still reproduces it).", stacklevel=2)
        return None
    try:
        x, feature_names, y = gb.assemble_features(dataset)
        splits = gb.make_splits(dataset, gbm_cfg)
        result = gb.fit_predict_variant(gbm_cfg, balanced=False, x=x, y=y, splits=splits)
        payload = {
            "estimator": result["estimator"],
            "backend": result["backend"],
            "feature_names": feature_names,
            "variant": "unweighted",
            "split_strategy": gbm_cfg.split_strategy,
            "split_seed": gbm_cfg.split_seed,
        }
        path = gbm_run_dir / "gbm_model.joblib"
        joblib.dump(payload, path)
        report_saved(path, "fitted unweighted GBM (joblib)")
        return str(path)
    except Exception as exc:  # noqa: BLE001 - convenience artifact only
        warnings.warn(f"GBM pickle skipped: {exc}", stacklevel=2)
        return None


def run_pipeline(
    dataset: fm.FlowDataset,
    output_dir: str | Path,
    split_strategy: str,
    split_seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    n_trials: int,
    twin_num_steps: int | None,
    seed: int,
    final_train_best: bool,
) -> dict:
    pipeline_dir = make_pipeline_dir(output_dir)
    print(f"Digital-twin training pipeline -> {pipeline_dir}")
    print(f"Shared split: strategy={split_strategy} seed={split_seed} fracs=({train_frac},{val_frac},{test_frac})")

    # --- 1. GBM (calibrated composite-MACE risk) on the shared split ---------- #
    gbm_cfg = gb.GBMConfig(
        output_dir=str(pipeline_dir / "gbm"),
        seed=seed,
        split_seed=split_seed,
        split_strategy=split_strategy,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    print("\n[1/2] Fitting the composite-MACE GBM on the shared train split ...")
    gbm_result = gb.run(dataset, gbm_cfg)
    gbm_run_dir = Path(gbm_result["run_dir"])
    gbm_pickle = persist_gbm_model(dataset, gbm_cfg, gbm_run_dir)

    # --- twin-flow sweep base config on the SAME split ------------------------ #
    base_cfg = replace(
        tune_twin.BASE_CONFIG,
        output_dir=str(pipeline_dir / "twin_optuna"),
        seed=seed,
        split_seed=split_seed,
        split_strategy=split_strategy,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    if twin_num_steps is not None:
        base_cfg = replace(base_cfg, num_steps=twin_num_steps)

    # Hard guarantee: both models see the same patient-for-patient split.
    for key in SHARED_SPLIT_KEYS:
        gbm_value = getattr(gbm_cfg, key)
        twin_value = getattr(base_cfg, key)
        assert gbm_value == twin_value, f"split mismatch on {key!r}: GBM={gbm_value} twin={twin_value}"
    print("Shared-split invariant asserted: GBM and twin flow agree on", ", ".join(SHARED_SPLIT_KEYS))

    # --- manifest written BEFORE the sweep (save-as-you-go) ------------------- #
    manifest = {
        "pipeline_dir": str(pipeline_dir),
        "source_label": dataset.source_label,
        "n_patients": int(len(dataset.subject_ids)),
        "shared_split": {key: getattr(gbm_cfg, key) for key in SHARED_SPLIT_KEYS},
        "gbm_run_dir": str(gbm_run_dir),
        "gbm_pickle": gbm_pickle,
        "gbm_backend": gb.xgboost_available() and "xgboost" or "hist_gradient_boosting",
        "twin_study_dir": None,
        "twin_final_run_dir": None,
        "status": "gbm_done_sweep_running",
    }
    manifest_path = pipeline_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_saved(manifest_path, "pipeline manifest (pre-sweep)")

    # --- 2. twin-flow Optuna sweep on the shared split ------------------------ #
    print(f"\n[2/2] Launching the twin-flow Optuna sweep ({n_trials} trials) on the same split ...")
    twin_result = tune_twin.tune_dataset(
        dataset,
        base_cfg=base_cfg,
        n_trials=n_trials,
        output_dir=str(pipeline_dir / "twin_optuna"),
        final_train_best=final_train_best,
    )
    twin_study_dir = Path(twin_result["study_dir"])
    twin_final = twin_result.get("final_result")
    twin_final_run_dir = str(twin_final["run_dir"]) if twin_final else None

    # --- manifest updated AFTER the sweep ------------------------------------- #
    manifest["twin_study_dir"] = str(twin_study_dir)
    manifest["twin_final_run_dir"] = twin_final_run_dir
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_saved(manifest_path, "pipeline manifest (complete)")

    print("\nPipeline complete.")
    print(f"  GBM run dir       : {gbm_run_dir}")
    print(f"  twin study dir    : {twin_study_dir}")
    print(f"  twin best run dir : {twin_final_run_dir}")
    print(f"  manifest          : {manifest_path}")
    print(
        "\nNext: evaluate with\n"
        f"    python evaluate_twin.py --pipeline {pipeline_dir} --csv <cohort.csv>"
    )
    return {"pipeline_dir": pipeline_dir, "manifest": manifest, "gbm_result": gbm_result, "twin_result": twin_result}


def run_from_csv(csv_path: str | Path, **kwargs) -> dict:
    return run_pipeline(fm.load_dataset_from_csv(csv_path), **kwargs)


def run_from_database(**kwargs) -> dict:
    try:
        dataset = fm.load_dataset_from_database()
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc
    return run_pipeline(dataset, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split-strategy", type=str, default="surgery", choices=["surgery", "outcome"])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--twin-num-steps", type=int, default=None, help="Override base sweep num_steps (smoke tests).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-final-train", action="store_true")
    args = parser.parse_args()

    kwargs = dict(
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        n_trials=args.n_trials,
        twin_num_steps=args.twin_num_steps,
        seed=args.seed,
        final_train_best=not args.no_final_train,
    )
    try:
        if args.csv_path:
            run_from_csv(args.csv_path, **kwargs)
        else:
            run_from_database(**kwargs)
    except RuntimeError as exc:
        print(f"ERROR: {exc}\n\nPass --csv <path> to run from a saved CSV export.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
