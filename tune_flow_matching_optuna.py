"""AdaLN-only Optuna tuner for the MBSCohort flow-matching model.

Default standalone behavior:
    python tune_flow_matching_optuna.py

This loads Cosmos MBSCohort through train_flow_matching.py, tunes model and
optimizer hyperparameters on validation flow loss with conditioning fixed to
AdaLN, then retrains one final model with the best hyperparameters and evaluates
it on the held-out test split.

For local smoke tests without Cosmos access, import this file and call
tune_from_csv(...).
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

try:
    import optuna
except ImportError:  # pragma: no cover - exercised only in environments without Optuna.
    optuna = None

import train_flow_matching as fm


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "python_flow_matching_optuna"
DEFAULT_CSV_PATH = REPO_ROOT / "fake_data" / "fake_mbs_cohort.csv"

N_TRIALS = 500
TIMEOUT_SECONDS = None
N_JOBS = 1
FINAL_TRAIN_BEST = True
MIN_TRIAL_STOP_STEP = 2000

BASE_CONFIG = fm.TrainConfig(
    output_dir=str(DEFAULT_OUTPUT_DIR),
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=0,
    split_seed=0,
    conditioning="adaln",
    num_steps=3000,
    batch_size=64,
    val_every=150,
    val_repeats=8,
    early_stop_patience=6,
    early_stop_min_delta=0.002,
    sample_steps=50,
    n_samples_per_patient=250,
)

FINAL_CONFIG_OVERRIDES = {
    "num_steps": 6000,
    "val_every": 250,
    "val_repeats": 8,
    "early_stop_patience": 6,
    "n_samples_per_patient": 250,
}


def require_optuna():
    if optuna is None:
        raise RuntimeError(
            "Optuna is required for hyperparameter tuning, but it is not installed "
            "in this Python environment."
        )
    return optuna


def make_study_dir(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_dir = output_dir / f"study_{time.strftime('%Y%m%d_%H%M%S')}"
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "trial_logs").mkdir(parents=True, exist_ok=True)
    return study_dir


def suggest_config(trial, base_cfg: fm.TrainConfig) -> fm.TrainConfig:
    return replace(
        base_cfg,
        seed=base_cfg.seed + trial.number,
        conditioning="adaln",
        hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        num_hidden_layers=trial.suggest_int("num_hidden_layers", 2, 5),
        time_emb_dim=trial.suggest_categorical("time_emb_dim", [64, 128]),
        time_scale=trial.suggest_categorical("time_scale", [1.0, 3.0, 10.0, 30.0]),
        surgery_emb_dim=trial.suggest_categorical("surgery_emb_dim", [8, 16]),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        batch_size=trial.suggest_categorical("batch_size", [2048]),
    )


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_validation_trial(
    dataset: fm.FlowDataset,
    cfg: fm.TrainConfig,
    trial,
    study_dir: Path,
) -> float:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    splits = fm.make_stratified_splits(dataset, cfg)
    preprocessing = fm.fit_preprocessing(dataset, splits["train"])
    arrays = fm.split_arrays(dataset, splits, preprocessing)

    model = fm.VectorFieldNet(cfg, fm.X_DIM, len(fm.PATIENT_FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    batch_size = min(cfg.batch_size, max(1, arrays["train"]["x"].shape[0]))

    best_score = float("inf")
    best_step = -1
    evals_since_improve = 0
    logs = []

    for step in range(1, cfg.num_steps + 1):
        model.train()
        batch = fm.batch_sample(arrays["train"], batch_size, rng)
        x1 = fm.as_tensor(batch["x"], device)
        mask = fm.as_tensor(batch["mask"], device)
        surgery_idx = fm.as_tensor(batch["surgery_idx"], device, torch.long)
        patient_features = fm.as_tensor(batch["patient_features"], device)
        x_t, t, u_t = fm.sample_conditional_path(x1)

        optimizer.zero_grad()
        loss = fm.flow_matching_loss(model, x_t, t, surgery_idx, patient_features, u_t, mask)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu())

        should_eval = step == 1 or step % cfg.val_every == 0 or step == cfg.num_steps
        if not should_eval:
            continue

        val_loss = fm.evaluate_flow_loss(model, arrays["val"], cfg, device)
        score = train_loss if np.isnan(val_loss) else val_loss
        improved = score < best_score - cfg.early_stop_min_delta
        if improved:
            best_score = score
            best_step = step
            evals_since_improve = 0
        else:
            evals_since_improve += 1

        logs.append(
            {
                "trial": trial.number,
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val": best_score,
                "best_step": best_step,
            }
        )
        trial.report(score, step)

        if step >= MIN_TRIAL_STOP_STEP and trial.should_prune():
            pd.DataFrame(logs).to_csv(study_dir / "trial_logs" / f"trial_{trial.number:04d}.csv", index=False)
            raise optuna.TrialPruned()

        if step >= MIN_TRIAL_STOP_STEP and not np.isnan(val_loss) and evals_since_improve >= cfg.early_stop_patience:
            break

    pd.DataFrame(logs).to_csv(study_dir / "trial_logs" / f"trial_{trial.number:04d}.csv", index=False)
    trial.set_user_attr("best_step", best_step)
    trial.set_user_attr("device", cfg.device)
    return float(best_score)


def best_config_from_trial(best_trial, base_cfg: fm.TrainConfig, study_dir: Path) -> fm.TrainConfig:
    cfg = replace(
        base_cfg,
        output_dir=str(study_dir / "best_model"),
        seed=base_cfg.seed,
        conditioning="adaln",
    )
    for name, value in best_trial.params.items():
        cfg = replace(cfg, **{name: value})
    for name, value in FINAL_CONFIG_OVERRIDES.items():
        cfg = replace(cfg, **{name: value})
    cfg = replace(cfg, conditioning="adaln")
    return cfg


def save_study_tables(study, study_dir: Path) -> None:
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials.to_csv(study_dir / "optuna_trials.csv", index=False)
    write_json(
        study_dir / "best_trial.json",
        {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs,
        },
    )


def tune_dataset(
    dataset: fm.FlowDataset,
    base_cfg: fm.TrainConfig = BASE_CONFIG,
    n_trials: int = N_TRIALS,
    timeout_seconds: int | None = TIMEOUT_SECONDS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    final_train_best: bool = FINAL_TRAIN_BEST,
) -> dict:
    optuna_module = require_optuna()
    base_cfg = replace(base_cfg, conditioning="adaln")
    study_dir = make_study_dir(output_dir)
    write_json(
        study_dir / "tuning_config.json",
        {
            "base_config": asdict(base_cfg),
            "forced_conditioning": "adaln",
            "final_config_overrides": FINAL_CONFIG_OVERRIDES,
            "n_trials": n_trials,
            "timeout_seconds": timeout_seconds,
            "min_trial_stop_step": MIN_TRIAL_STOP_STEP,
            "target_names": fm.TARGET_NAMES,
            "x_dim": fm.X_DIM,
            "source_label": dataset.source_label,
            "n_patients": int(len(dataset.subject_ids)),
        },
    )

    sampler = optuna_module.samplers.TPESampler(seed=base_cfg.seed)
    pruner = optuna_module.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=MIN_TRIAL_STOP_STEP)
    storage_url = f"sqlite:///{study_dir / 'optuna_study.sqlite3'}"
    study = optuna_module.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"flow_matching_{study_dir.name}",
        storage=storage_url,
    )

    def objective(trial) -> float:
        cfg = suggest_config(trial, base_cfg)
        trial.set_user_attr("conditioning", cfg.conditioning)
        print(f"Trial {trial.number}: {trial.params}")
        return train_validation_trial(dataset, cfg, trial, study_dir)

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, n_jobs=N_JOBS, gc_after_trial=True)
    save_study_tables(study, study_dir)

    final_result = None
    if final_train_best:
        best_cfg = best_config_from_trial(study.best_trial, base_cfg, study_dir)
        write_json(study_dir / "best_final_config.json", asdict(best_cfg))
        final_result = fm.train_model(dataset, best_cfg)
        write_json(
            study_dir / "final_model.json",
            {"run_dir": str(final_result["run_dir"]), "best_trial_number": study.best_trial.number},
        )

    print(f"Best trial {study.best_trial.number}: value={study.best_trial.value:.6f}")
    print(f"Study artifacts saved to {study_dir}")
    if final_result is not None:
        print(f"Final best-model run saved to {final_result['run_dir']}")
    return {"study": study, "study_dir": study_dir, "final_result": final_result}


def tune_from_csv(csv_path: str | Path = DEFAULT_CSV_PATH, **kwargs) -> dict:
    return tune_dataset(fm.load_dataset_from_csv(csv_path), **kwargs)


def tune_from_database(**kwargs) -> dict:
    try:
        dataset = fm.load_dataset_from_database()
    except RuntimeError as exc:
        raise RuntimeError(str(exc).replace("call train_from_csv(...)", "call tune_from_csv(...)")) from exc
    return tune_dataset(dataset, **kwargs)


if __name__ == "__main__":
    try:
        tune_from_database()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        warnings.warn("Interrupted by user; completed trials remain saved in the study directory.", stacklevel=1)
        sys.exit(130)
