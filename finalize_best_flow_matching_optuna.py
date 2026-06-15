"""Retrain the best Optuna flow-matching trial and save evaluator-ready artifacts.

Use this when ``tune_flow_matching_optuna.py`` completed some trials but did not
finish the final best-model retraining step, for example because an HPC job hit
its walltime limit.

Examples:
    python finalize_best_flow_matching_optuna.py --study runs/python_flow_matching_optuna/study_YYYYMMDD_HHMMSS
    python finalize_best_flow_matching_optuna.py --study <study_dir> --csv data/cosmos_mbs_flow_input.csv --device cpu

After this finishes, evaluate with:
    python evaluate_flow_matching.py --run <study_dir>
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import torch

import train_flow_matching as fm
import tune_flow_matching_optuna as tune


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def is_run_dir(path: Path) -> bool:
    return (
        (path / "config.json").exists()
        and (path / "preprocessing.json").exists()
        and (path / "model.pt").exists()
    )


def resolve_recorded_run(study_dir: Path, run_dir_value: str) -> Path | None:
    saved = Path(run_dir_value)
    candidates = [saved]
    if not saved.is_absolute():
        candidates.append(study_dir / saved)
        candidates.append(study_dir.parent / saved)
    for candidate in candidates:
        if is_run_dir(candidate):
            return candidate
    return None


def existing_final_run(study_dir: Path) -> Path | None:
    final_model = study_dir / "final_model.json"
    if not final_model.exists():
        return None
    payload = json.loads(final_model.read_text(encoding="utf-8"))
    run_dir_value = payload.get("run_dir")
    if not isinstance(run_dir_value, str):
        return None
    return resolve_recorded_run(study_dir, run_dir_value)


def train_config_from_dict(raw: dict[str, Any]) -> fm.TrainConfig:
    valid = {field.name for field in fields(fm.TrainConfig)}
    return fm.TrainConfig(**{key: value for key, value in raw.items() if key in valid})


def load_tuning_metadata(study_dir: Path) -> tuple[fm.TrainConfig, dict[str, Any], dict[str, Any]]:
    path = study_dir / "tuning_config.json"
    if not path.exists():
        warnings.warn(
            "tuning_config.json was not found; falling back to current defaults from "
            "tune_flow_matching_optuna.py.",
            stacklevel=2,
        )
        return tune.BASE_CONFIG, dict(tune.FINAL_CONFIG_OVERRIDES), {}

    metadata = json.loads(path.read_text(encoding="utf-8"))
    base_config = metadata.get("base_config", {})
    if not isinstance(base_config, dict):
        raise SystemExit(f"Invalid base_config in {path}")
    final_overrides = metadata.get("final_config_overrides", tune.FINAL_CONFIG_OVERRIDES)
    if not isinstance(final_overrides, dict):
        raise SystemExit(f"Invalid final_config_overrides in {path}")
    return train_config_from_dict(base_config), final_overrides, metadata


def load_best_trial_from_json(study_dir: Path) -> dict[str, Any] | None:
    path = study_dir / "best_trial.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "number" not in payload or "params" not in payload:
        raise SystemExit(f"Invalid best_trial.json in {study_dir}")
    return {
        "number": payload["number"],
        "value": payload.get("value"),
        "params": payload["params"],
        "user_attrs": payload.get("user_attrs", {}),
        "source": str(path),
    }


def load_best_trial_from_sqlite(study_dir: Path, study_name: str | None) -> dict[str, Any]:
    sqlite_path = study_dir / "optuna_study.sqlite3"
    if not sqlite_path.exists():
        raise SystemExit(
            f"No best_trial.json or optuna_study.sqlite3 found in {study_dir}. "
            "There is no saved Optuna best trial to finalize."
        )

    optuna_module = tune.require_optuna()
    storage_url = f"sqlite:///{sqlite_path}"
    if study_name is None:
        summaries = optuna_module.get_all_study_summaries(storage=storage_url)
        if not summaries:
            raise SystemExit(f"No Optuna studies found in {sqlite_path}")
        if len(summaries) > 1:
            names = ", ".join(summary.study_name for summary in summaries)
            raise SystemExit(f"Multiple studies found in {sqlite_path}; pass --study-name. Studies: {names}")
        study_name = summaries[0].study_name

    study = optuna_module.load_study(study_name=study_name, storage=storage_url)
    try:
        best = study.best_trial
    except ValueError as exc:
        raise SystemExit(f"No completed Optuna trials found in {sqlite_path}") from exc
    return {
        "number": best.number,
        "value": best.value,
        "params": dict(best.params),
        "user_attrs": dict(best.user_attrs),
        "source": str(sqlite_path),
        "study_name": study_name,
    }


def load_best_trial(study_dir: Path, study_name: str | None) -> dict[str, Any]:
    payload = load_best_trial_from_json(study_dir)
    if payload is not None:
        return payload
    payload = load_best_trial_from_sqlite(study_dir, study_name)
    write_json(
        study_dir / "best_trial.json",
        {
            "number": payload["number"],
            "value": payload["value"],
            "params": payload["params"],
            "user_attrs": payload["user_attrs"],
        },
    )
    return payload


def choose_device(requested: str | None, cfg: fm.TrainConfig) -> str:
    if requested is None:
        return cfg.device
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def config_for_best_trial(
    study_dir: Path,
    best_trial: dict[str, Any],
    base_cfg: fm.TrainConfig,
    final_overrides: dict[str, Any],
    device: str | None,
) -> fm.TrainConfig:
    cfg = replace(
        base_cfg,
        output_dir=str(study_dir / "best_model"),
        seed=base_cfg.seed,
        conditioning="adaln",
    )
    for name, value in best_trial["params"].items():
        cfg = replace(cfg, **{name: value})
    for name, value in final_overrides.items():
        cfg = replace(cfg, **{name: value})
    cfg = replace(cfg, device=choose_device(device, cfg), conditioning="adaln")
    return cfg


def load_dataset(csv_path: Path | None) -> fm.FlowDataset:
    if csv_path is not None:
        return fm.load_dataset_from_csv(csv_path)
    try:
        return fm.load_dataset_from_database()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\n\nPass --csv <path> to finalize from a saved CSV export.") from exc


def finalize_best_model(
    study_dir: Path,
    csv_path: Path | None,
    device: str | None,
    study_name: str | None,
    force: bool,
) -> Path:
    study_dir = Path(study_dir)
    if not study_dir.exists():
        raise SystemExit(f"Study directory does not exist: {study_dir}")

    current = existing_final_run(study_dir)
    if current is not None and not force:
        print(f"Final model already exists: {current}")
        print(f"Evaluate it with: python evaluate_flow_matching.py --run {study_dir}")
        return current

    base_cfg, final_overrides, tuning_metadata = load_tuning_metadata(study_dir)
    best_trial = load_best_trial(study_dir, study_name)
    cfg = config_for_best_trial(study_dir, best_trial, base_cfg, final_overrides, device)

    write_json(study_dir / "best_final_config.json", asdict(cfg))
    dataset = load_dataset(csv_path)
    result = fm.train_model(dataset, cfg)
    run_dir = Path(result["run_dir"])

    write_json(
        study_dir / "final_model.json",
        {
            "run_dir": str(run_dir),
            "best_trial_number": int(best_trial["number"]),
            "best_trial_value": best_trial["value"],
            "best_trial_source": best_trial["source"],
            "source_label": dataset.source_label,
            "n_patients": int(len(dataset.subject_ids)),
            "target_names": fm.TARGET_NAMES,
            "x_dim": fm.X_DIM,
            "tuning_source_label": tuning_metadata.get("source_label"),
        },
    )
    print(f"Best trial {best_trial['number']}: value={best_trial['value']}")
    print(f"Final best-model run saved to {run_dir}")
    print(f"Evaluate it with: python evaluate_flow_matching.py --run {study_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True, help="Optuna study directory.")
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Override saved device: auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--study-name", default=None, help="Optuna study name if sqlite contains multiple studies.")
    parser.add_argument("--force", action="store_true", help="Train a new final run even if final_model.json is valid.")
    args = parser.parse_args()
    try:
        finalize_best_model(
            study_dir=args.study,
            csv_path=args.csv_path,
            device=args.device,
            study_name=args.study_name,
            force=args.force,
        )
    except KeyboardInterrupt:
        warnings.warn("Interrupted by user; partial run artifacts may remain in best_model/.", stacklevel=1)
        sys.exit(130)


if __name__ == "__main__":
    main()
