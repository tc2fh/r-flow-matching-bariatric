"""Optuna tuner for the digital-twin (event-conditioned) flow-matching model.

Mirrors ``tune_flow_matching_multitask_optuna.py`` but simpler: the twin flow has
NO classification head, so the objective is just the validation flow-matching loss.
There is no fixed-weight juggling and no BCE term to balance -- Optuna minimizes a
single, unambiguous quantity::

    objective = val_flow_loss   (event teacher-forced with the TRUE label)

Everything else (TPE sampler, median pruner, per-trial CSV logs written as we go,
a resumable SQLite study, an optional final full-length train of the best config)
is kept. Output goes to ``runs/python_flow_matching_twin_optuna/`` so it never
collides with the multi-task study under
``runs/python_flow_matching_multitask_optuna/`` or the Cosmos study.

Resume an interrupted study by pointing ``--study-dir`` at an existing study dir:
its SQLite DB is reopened (``load_if_exists=True``) and new trials append.

Local smoke test without Cosmos::

    python tune_flow_matching_twin_optuna.py --csv fake_data/fake_mbs_cohort.csv \
        --n-trials 3 --num-steps 60
"""

from __future__ import annotations

import argparse
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
except ImportError:  # pragma: no cover - exercised only without Optuna.
    optuna = None

import train_flow_matching as fm
import train_flow_matching_multitask as mt
import train_flow_matching_twin as tw


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "python_flow_matching_twin_optuna"
DEFAULT_CSV_PATH = REPO_ROOT / "fake_data" / "fake_mbs_cohort.csv"

# Sweep budget trimmed for time-boxed VM runs (was N_TRIALS=500, num_steps=3000,
# MIN_TRIAL_STOP_STEP=1500). TPE finds a near-optimal config well inside ~150 trials
# for the reduced 5-HP space below, the median pruner now bites at step 800 instead
# of 1500, and the sweep trains at 2400 steps while the winner is still retrained at
# 6000 (FINAL_CONFIG_OVERRIDES) -- so lower sweep fidelity costs ranking accuracy, not
# the shipped model. Restore the originals for an exhaustive sweep when not time-boxed.
N_TRIALS = 150
TIMEOUT_SECONDS = None
N_JOBS = 1
FINAL_TRAIN_BEST = True
MIN_TRIAL_STOP_STEP = 800

BASE_CONFIG = tw.TwinConfig(
    output_dir=str(DEFAULT_OUTPUT_DIR),
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=0,
    split_seed=0,
    num_steps=2400,
    batch_size=2048,
    val_every=150,
    val_repeats=8,
    early_stop_patience=6,
    early_stop_min_delta=0.002,
    sample_steps=50,
    n_samples_per_patient=250,
    # Low-impact architecture HPs pinned OUT of the Optuna search so the trimmed trial
    # budget concentrates on the high-impact ones (learning_rate, weight_decay,
    # hidden_dim, num_hidden_layers, time_scale). They are pinned HERE, not just inside
    # suggest_config, because best_config_from_trial rebuilds the final model from
    # BASE_CONFIG + the searched params -- anything absent from both would silently
    # revert to a TwinConfig default. Values: modest embeddings for the 2-category
    # surgery/event inputs, a slightly richer time embedding (cheap, helps the velocity
    # field), and the mid-range conditioning MLP.
    surgery_emb_dim=16,
    event_emb_dim=8,
    time_emb_dim=128,
    cond_hidden_dim=64,
    cond_num_layers=2,
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


def suggest_config(trial, base_cfg: tw.TwinConfig) -> tw.TwinConfig:
    # Search only the high-impact HPs; the low-impact architecture dims and batch_size
    # are pinned in BASE_CONFIG (see the note there) so the trimmed budget is spent
    # where it moves the val flow loss.
    return replace(
        base_cfg,
        seed=base_cfg.seed + trial.number,
        hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        num_hidden_layers=trial.suggest_int("num_hidden_layers", 2, 5),
        time_scale=trial.suggest_categorical("time_scale", [1.0, 3.0, 10.0, 30.0]),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
    )


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tw.report_saved(path)


def train_validation_trial(dataset: fm.FlowDataset, cfg: tw.TwinConfig, trial, study_dir: Path) -> float:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    splits = tw.make_splits(dataset, cfg)
    pre = mt.fit_preprocessing(dataset, splits["train"])
    arrays = mt.split_arrays(dataset, splits, pre)

    model = tw.TwinNet(cfg, tw.X_CONT_DIM, len(fm.PATIENT_FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    batch_size = min(cfg.batch_size, max(1, arrays["train"]["x"].shape[0]))

    best_score = float("inf")
    best_step = -1
    evals_since_improve = 0
    logs = []

    for step in range(1, cfg.num_steps + 1):
        model.train()
        batch = mt.batch_sample(arrays["train"], batch_size, rng)
        x1 = fm.as_tensor(batch["x"], device)
        mask = fm.as_tensor(batch["mask"], device)
        surgery_idx = fm.as_tensor(batch["surgery_idx"], device, torch.long)
        patient_features = fm.as_tensor(batch["patient_features"], device)
        event = fm.as_tensor(batch["y_mace"], device, torch.long)  # teacher-force TRUE event

        cond = model.encode(surgery_idx, patient_features, event)
        x_t, t, u_t = fm.sample_conditional_path(x1)
        pred = model.velocity(x_t, t, cond)
        flow_loss = mt.flow_matching_loss(pred, u_t, mask)

        optimizer.zero_grad()
        flow_loss.backward()
        optimizer.step()
        train_flow = float(flow_loss.detach().cpu())

        should_eval = step == 1 or step % cfg.val_every == 0 or step == cfg.num_steps
        if not should_eval:
            continue

        val_flow = tw.evaluate_flow_loss(model, arrays["val"], cfg, device)
        score = train_flow if np.isnan(val_flow) else val_flow

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
                "train_flow": train_flow,
                "val_flow": val_flow,
                "score": score,
                "best_score": best_score,
                "best_step": best_step,
            }
        )
        trial.report(score, step)

        if step >= MIN_TRIAL_STOP_STEP and trial.should_prune():
            pd.DataFrame(logs).to_csv(study_dir / "trial_logs" / f"trial_{trial.number:04d}.csv", index=False)
            raise optuna.TrialPruned()

        if step >= MIN_TRIAL_STOP_STEP and evals_since_improve >= cfg.early_stop_patience:
            break

    pd.DataFrame(logs).to_csv(study_dir / "trial_logs" / f"trial_{trial.number:04d}.csv", index=False)
    trial.set_user_attr("best_step", best_step)
    trial.set_user_attr("device", cfg.device)
    return float(best_score)


def best_config_from_trial(best_trial, base_cfg: tw.TwinConfig, study_dir: Path) -> tw.TwinConfig:
    cfg = replace(base_cfg, output_dir=str(study_dir / "best_model"), seed=base_cfg.seed)
    for name, value in best_trial.params.items():
        cfg = replace(cfg, **{name: value})
    for name, value in FINAL_CONFIG_OVERRIDES.items():
        cfg = replace(cfg, **{name: value})
    return cfg


def save_study_tables(study, study_dir: Path) -> None:
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials.to_csv(study_dir / "optuna_trials.csv", index=False)
    tw.report_saved(study_dir / "optuna_trials.csv", "Optuna trials table")
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
    base_cfg: tw.TwinConfig = BASE_CONFIG,
    n_trials: int = N_TRIALS,
    timeout_seconds: int | None = TIMEOUT_SECONDS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    final_train_best: bool = FINAL_TRAIN_BEST,
    study_dir: str | Path | None = None,
) -> dict:
    optuna_module = require_optuna()
    study_dir = make_study_dir(output_dir) if study_dir is None else Path(study_dir)
    (study_dir / "trial_logs").mkdir(parents=True, exist_ok=True)
    write_json(
        study_dir / "tuning_config.json",
        {
            "base_config": asdict(base_cfg),
            "objective": "minimize val_flow_loss (event teacher-forced; no cls head)",
            "final_config_overrides": FINAL_CONFIG_OVERRIDES,
            "n_trials": n_trials,
            "timeout_seconds": timeout_seconds,
            "min_trial_stop_step": MIN_TRIAL_STOP_STEP,
            "cont_names": tw.CONT_NAMES,
            "x_cont_dim": tw.X_CONT_DIM,
            "mace_label": tw.MACE_LABEL_NAME,
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
        study_name=f"flow_matching_twin_{study_dir.name}",
        storage=storage_url,
        load_if_exists=True,  # resumable: re-open the same study dir to append trials
    )

    def objective(trial) -> float:
        cfg = suggest_config(trial, base_cfg)
        print(f"Trial {trial.number}: {trial.params}")
        return train_validation_trial(dataset, cfg, trial, study_dir)

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, n_jobs=N_JOBS, gc_after_trial=True)
    save_study_tables(study, study_dir)

    final_result = None
    if final_train_best:
        best_cfg = best_config_from_trial(study.best_trial, base_cfg, study_dir)
        write_json(study_dir / "best_final_config.json", asdict(best_cfg))
        final_result = tw.train_model(dataset, best_cfg)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--study-dir", type=str, default=None, help="Resume an existing study dir (reopens its SQLite DB).")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--split-strategy", type=str, default=None, choices=["surgery", "temporal", "outcome"])
    parser.add_argument("--no-final-train", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_cfg = replace(BASE_CONFIG, output_dir=args.output_dir, seed=args.seed)
    if args.num_steps is not None:
        base_cfg = replace(base_cfg, num_steps=args.num_steps)
    if args.split_strategy is not None:
        base_cfg = replace(base_cfg, split_strategy=args.split_strategy)

    kwargs = dict(
        base_cfg=base_cfg,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        final_train_best=not args.no_final_train,
        study_dir=args.study_dir,
    )
    try:
        if args.csv_path:
            tune_from_csv(args.csv_path, **kwargs)
        else:
            tune_from_database(**kwargs)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        warnings.warn("Interrupted by user; completed trials remain saved in the study directory.", stacklevel=1)
        sys.exit(130)


if __name__ == "__main__":
    main()
