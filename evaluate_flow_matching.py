"""Plot factual and counterfactual examples from a trained PyTorch flow run.

Run against the Optuna final model directory:
    python evaluate_flow_matching.py --run runs/python_flow_matching_optuna/<study>/best_model/<run>

If the model was trained from Cosmos, the evaluator reloads Cosmos by default.
For a CSV export, pass:
    python evaluate_flow_matching.py --run <run> --csv data/cosmos_mbs_flow_input.csv

Outputs in the run directory by default:
    eval_selected_test_patients.csv
    eval_bmi_factual_counterfactual_examples_test.png
    eval_hba1c_factual_counterfactual_examples_test.png
    eval_mace_factual_counterfactual_histograms_test.png
    eval_summary.json
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
from typing import Any
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import train_flow_matching as fm


N_SHOW_PER_PROCEDURE = 3
MAX_SAMPLE_LINES = 50
INDEX_TO_SURGERY = {idx: name for name, idx in fm.SURGERY_TO_INDEX.items()}


def find_latest_run(log_dir: Path) -> Path | None:
    """Return the newest PyTorch run directory under a log or study directory."""
    if not log_dir.exists():
        return None
    if is_run_dir(log_dir):
        return log_dir
    candidates = [
        path
        for path in log_dir.rglob("run_*")
        if path.is_dir() and is_run_dir(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def is_run_dir(path: Path) -> bool:
    return (
        (path / "config.json").exists()
        and (path / "preprocessing.json").exists()
        and (path / "model.pt").exists()
    )


def resolve_run_dir(path: Path | None, log_dir: Path) -> Path:
    """Accept a run dir, Optuna study dir, best_model dir, or log root."""
    search_root = log_dir if path is None else Path(path)
    if is_run_dir(search_root):
        return search_root

    final_model = search_root / "final_model.json"
    if final_model.exists():
        payload = json.loads(final_model.read_text(encoding="utf-8"))
        saved = Path(payload["run_dir"])
        candidates = [saved]
        if not saved.is_absolute():
            candidates.append(search_root / saved)
            candidates.append(search_root.parent / saved)
        for candidate in candidates:
            if is_run_dir(candidate):
                return candidate

    latest = find_latest_run(search_root)
    if latest is not None:
        return latest

    raise SystemExit(
        f"No PyTorch flow run with config.json, preprocessing.json, and model.pt found under {search_root}"
    )


def load_config(run_dir: Path) -> tuple[fm.TrainConfig, dict[str, Any]]:
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    valid = {field.name for field in fields(fm.TrainConfig)}
    cfg = fm.TrainConfig(**{key: value for key, value in raw.items() if key in valid})
    return cfg, raw


def load_preprocessing(run_dir: Path) -> fm.Preprocessing:
    raw = json.loads((run_dir / "preprocessing.json").read_text(encoding="utf-8"))
    return fm.Preprocessing(
        target_mean=np.asarray(raw["target_mean"], dtype=np.float32),
        target_std=np.asarray(raw["target_std"], dtype=np.float32),
        static_mean=np.asarray(raw["static_mean"], dtype=np.float32),
        static_std=np.asarray(raw["static_std"], dtype=np.float32),
        static_continuous_idx=np.asarray(raw["static_continuous_idx"], dtype=np.int64),
        patient_feature_names=list(raw["patient_feature_names"]),
        target_metadata=list(raw["target_metadata"]),
    )


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU.", stacklevel=2)
        return torch.device("cpu")
    return device


def restore_model(run_dir: Path, cfg: fm.TrainConfig, device: torch.device, x_dim: int) -> fm.VectorFieldNet:
    model = fm.VectorFieldNet(cfg, x_dim, len(fm.PATIENT_FEATURES)).to(device)
    try:
        state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_dataset(csv_path: Path | None) -> fm.FlowDataset:
    if csv_path is not None:
        return fm.load_dataset_from_csv(csv_path)
    try:
        return fm.load_dataset_from_database()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\n\nPass --csv <path> to evaluate from a saved CSV export.") from exc


def target_dims(dataset: fm.FlowDataset, group: str) -> tuple[np.ndarray, np.ndarray]:
    items = [item for item in dataset.target_metadata if item["group"] == group]
    items = sorted(items, key=lambda item: float(item["horizon_months"]))
    if not items:
        raise ValueError(f"No target metadata found for group {group!r}")
    dims = np.asarray([int(item["dim"]) for item in items], dtype=np.int64)
    months = np.asarray([float(item["horizon_months"]) for item in items], dtype=np.float32)
    return dims, months


def target_dim_by_name(dataset: fm.FlowDataset, name: str) -> int:
    for item in dataset.target_metadata:
        if item["name"] == name:
            return int(item["dim"])
    raise ValueError(f"No target metadata found for target {name!r}")


def select_display_patients(
    dataset: fm.FlowDataset,
    test_idx: np.ndarray,
    rng: np.random.Generator,
    n_per_procedure: int,
) -> np.ndarray:
    """Pick balanced test examples with observed BMI/HbA1c follow-up."""
    bmi_dims, _ = target_dims(dataset, "bmi")
    hba1c_dims, _ = target_dims(dataset, "hba1c")
    score_dims = np.concatenate([bmi_dims, hba1c_dims])
    selected: list[int] = []

    for proc_idx in (fm.SURGERY_TO_INDEX["sleeve"], fm.SURGERY_TO_INDEX["rnygb"]):
        local = np.where(dataset.surgery_idx[test_idx] == proc_idx)[0]
        if local.size == 0:
            warnings.warn(f"No {INDEX_TO_SURGERY[proc_idx]} patients found in the test split.", stacklevel=2)
            continue
        obs_count = dataset.mask[test_idx[local]][:, score_dims].sum(axis=1)
        tie_breaker = rng.random(local.size)
        order = np.lexsort((tie_breaker, -obs_count))
        n_take = min(n_per_procedure, local.size)
        if n_take < n_per_procedure:
            warnings.warn(
                f"Only found {n_take} {INDEX_TO_SURGERY[proc_idx]} patients in the test split.",
                stacklevel=2,
            )
        selected.extend(test_idx[local[order[:n_take]]].tolist())

    if not selected:
        raise SystemExit("Could not select any display patients from the test split.")
    return np.asarray(selected, dtype=np.int64)


def sample_with_surgery(
    model: fm.VectorFieldNet,
    surgery_idx: np.ndarray,
    patient_features: np.ndarray,
    initial_noise: torch.Tensor,
    n_steps: int,
    device: torch.device,
) -> np.ndarray:
    n_patients, n_samples, x_dim = initial_noise.shape
    total = n_patients * n_samples
    tiled = np.repeat(np.arange(n_patients), n_samples)

    surgery = torch.as_tensor(surgery_idx[tiled], dtype=torch.long, device=device)
    features = torch.as_tensor(patient_features[tiled], dtype=torch.float32, device=device)
    x = initial_noise.reshape(total, x_dim).clone()
    dt = 1.0 / n_steps

    model.eval()
    with torch.no_grad():
        for step in range(n_steps):
            t = torch.full((total,), step * dt, dtype=torch.float32, device=device)
            x = x + dt * model(x, t, surgery, features)
    return x.detach().cpu().numpy().reshape(n_patients, n_samples, x_dim)


def sample_factual_counterfactual(
    model: fm.VectorFieldNet,
    dataset: fm.FlowDataset,
    patient_idx: np.ndarray,
    preprocessing: fm.Preprocessing,
    n_samples: int,
    n_steps: int,
    x_dim: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    patient_features = fm.transform_patient_features(
        dataset.patient_features_raw[patient_idx],
        preprocessing,
    )
    actual = dataset.surgery_idx[patient_idx].astype(np.int64)
    counterfactual = (1 - actual).astype(np.int64)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    initial_noise = torch.randn(
        (len(patient_idx), n_samples, x_dim),
        dtype=torch.float32,
        device=device,
    )

    factual_std = sample_with_surgery(
        model=model,
        surgery_idx=actual,
        patient_features=patient_features,
        initial_noise=initial_noise,
        n_steps=n_steps,
        device=device,
    )
    counterfactual_std = sample_with_surgery(
        model=model,
        surgery_idx=counterfactual,
        patient_features=patient_features,
        initial_noise=initial_noise,
        n_steps=n_steps,
        device=device,
    )
    return (
        fm.unstandardize(factual_std, preprocessing).astype(np.float32),
        fm.unstandardize(counterfactual_std, preprocessing).astype(np.float32),
    )


def _format_month(month: float) -> str:
    return str(int(month)) if float(month).is_integer() else f"{month:g}"


def plot_timecourse_factual_counterfactual(
    dataset: fm.FlowDataset,
    patient_idx: np.ndarray,
    factual_samples: np.ndarray,
    counterfactual_samples: np.ndarray,
    group: str,
    y_label: str,
    output_path: Path,
    title: str,
    max_sample_lines: int,
) -> None:
    dims, months = target_dims(dataset, group)
    n_rows = len(patient_idx)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(11, max(2.65 * n_rows, 4)),
        sharex=True,
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )

    def panel(ax, samples, color, panel_title, truth=None, obs=None):
        values = samples[:, dims]
        n_lines = min(max_sample_lines, values.shape[0])
        for sample_path in values[:n_lines]:
            ax.plot(months, sample_path, color=color, alpha=0.10, lw=0.8)
        q05, q25, q50, q75, q95 = np.quantile(
            values,
            [0.05, 0.25, 0.50, 0.75, 0.95],
            axis=0,
        )
        ax.fill_between(months, q05, q95, color=color, alpha=0.12, label="5-95%")
        ax.fill_between(months, q25, q75, color=color, alpha=0.28, label="25-75%")
        ax.plot(months, q50, color=color, lw=2.2, label="median")
        if truth is not None and obs is not None and obs.any():
            ax.scatter(months[obs], truth[obs], color="black", s=32, zorder=5, label="observed")
        ax.set_title(panel_title)
        ax.set_xlabel("Months post-op")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)

    for row, global_idx in enumerate(patient_idx):
        actual = INDEX_TO_SURGERY[int(dataset.surgery_idx[global_idx])]
        cf = INDEX_TO_SURGERY[int(1 - dataset.surgery_idx[global_idx])]
        subject_id = dataset.subject_ids[global_idx]
        truth = dataset.x[global_idx, dims]
        obs = dataset.mask[global_idx, dims].astype(bool)

        panel(
            axes[row, 0],
            factual_samples[row],
            "tab:blue",
            f"{subject_id}: factual {actual}",
            truth=truth,
            obs=obs,
        )
        panel(
            axes[row, 1],
            counterfactual_samples[row],
            "tab:orange",
            f"{subject_id}: counterfactual {cf}",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(title, y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_mace_histograms(
    dataset: fm.FlowDataset,
    patient_idx: np.ndarray,
    factual_samples: np.ndarray,
    counterfactual_samples: np.ndarray,
    output_path: Path,
) -> None:
    mace_dim = target_dim_by_name(dataset, "mace_ever")
    n_rows = len(patient_idx)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(9, max(2.15 * n_rows, 4)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    bins = np.linspace(0.0, 1.0, 31)

    for row, global_idx in enumerate(patient_idx):
        ax = axes[row, 0]
        actual = INDEX_TO_SURGERY[int(dataset.surgery_idx[global_idx])]
        cf = INDEX_TO_SURGERY[int(1 - dataset.surgery_idx[global_idx])]
        subject_id = dataset.subject_ids[global_idx]
        factual = np.clip(factual_samples[row, :, mace_dim], 0.0, 1.0)
        counterfactual = np.clip(counterfactual_samples[row, :, mace_dim], 0.0, 1.0)

        ax.hist(factual, bins=bins, density=True, alpha=0.52, color="tab:blue", label=f"factual {actual}")
        ax.hist(
            counterfactual,
            bins=bins,
            density=True,
            alpha=0.52,
            color="tab:orange",
            label=f"counterfactual {cf}",
        )
        ax.axvline(np.median(factual), color="tab:blue", lw=2.0)
        ax.axvline(np.median(counterfactual), color="tab:orange", lw=2.0)
        if dataset.mask[global_idx, mace_dim] == 1:
            observed = int(round(float(dataset.x[global_idx, mace_dim])))
            ax.axvline(observed, color="black", lw=1.6, ls="--", label=f"observed {observed}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylabel("Density")
        ax.set_title(f"{subject_id}: predicted MACE")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Predicted MACE probability")
    fig.suptitle("Test MACE factual and counterfactual predictive distributions", y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def selected_patient_frame(dataset: fm.FlowDataset, patient_idx: np.ndarray) -> pd.DataFrame:
    bmi_dims, _ = target_dims(dataset, "bmi")
    hba1c_dims, _ = target_dims(dataset, "hba1c")
    mace_dim = target_dim_by_name(dataset, "mace_ever")
    return pd.DataFrame(
        {
            "subject_id": dataset.subject_ids[patient_idx],
            "procedure": dataset.surgery_type[patient_idx],
            "bmi_observed_points": dataset.mask[patient_idx][:, bmi_dims].sum(axis=1).astype(int),
            "hba1c_observed_points": dataset.mask[patient_idx][:, hba1c_dims].sum(axis=1).astype(int),
            "observed_mace": dataset.x[patient_idx, mace_dim].astype(int),
        }
    )


def evaluate_run(
    run_dir: Path,
    output_dir: Path | None,
    csv_path: Path | None,
    n_samples: int,
    n_steps: int,
    seed: int,
    n_show_per_procedure: int,
    max_sample_lines: int,
    device_name: str,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = run_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, raw_config = load_config(run_dir)
    preprocessing = load_preprocessing(run_dir)
    x_dim = int(len(preprocessing.target_mean))
    config_x_dim = raw_config.get("x_dim")
    if config_x_dim is not None and int(config_x_dim) != x_dim:
        warnings.warn(
            f"config.json x_dim={config_x_dim}, but preprocessing has {x_dim} targets; "
            "using preprocessing dimension.",
            stacklevel=2,
        )
    device = choose_device(device_name)
    model = restore_model(run_dir, cfg, device, x_dim)
    dataset = load_dataset(csv_path)
    if dataset.x.shape[1] != x_dim:
        raise SystemExit(
            f"Dataset target dimension ({dataset.x.shape[1]}) does not match saved model dimension ({x_dim}). "
            "Use the same data-preparation code and CSV/database source used for training."
        )
    splits = fm.make_stratified_splits(dataset, cfg)
    test_idx = splits["test"]
    selected_idx = select_display_patients(
        dataset=dataset,
        test_idx=test_idx,
        rng=np.random.default_rng(seed),
        n_per_procedure=n_show_per_procedure,
    )

    factual_samples, counterfactual_samples = sample_factual_counterfactual(
        model=model,
        dataset=dataset,
        patient_idx=selected_idx,
        preprocessing=preprocessing,
        n_samples=n_samples,
        n_steps=n_steps,
        x_dim=x_dim,
        seed=seed,
        device=device,
    )

    selected_path = output_dir / "eval_selected_test_patients.csv"
    bmi_path = output_dir / "eval_bmi_factual_counterfactual_examples_test.png"
    hba1c_path = output_dir / "eval_hba1c_factual_counterfactual_examples_test.png"
    mace_path = output_dir / "eval_mace_factual_counterfactual_histograms_test.png"

    selected_patient_frame(dataset, selected_idx).to_csv(selected_path, index=False)
    plot_timecourse_factual_counterfactual(
        dataset=dataset,
        patient_idx=selected_idx,
        factual_samples=factual_samples,
        counterfactual_samples=counterfactual_samples,
        group="bmi",
        y_label="BMI",
        output_path=bmi_path,
        title=(
            "Test BMI factual and counterfactual predictive distributions "
            f"({n_show_per_procedure} sleeve, {n_show_per_procedure} RNYGB)"
        ),
        max_sample_lines=max_sample_lines,
    )
    plot_timecourse_factual_counterfactual(
        dataset=dataset,
        patient_idx=selected_idx,
        factual_samples=factual_samples,
        counterfactual_samples=counterfactual_samples,
        group="hba1c",
        y_label="HbA1c",
        output_path=hba1c_path,
        title=(
            "Test HbA1c factual and counterfactual predictive distributions "
            f"({n_show_per_procedure} sleeve, {n_show_per_procedure} RNYGB)"
        ),
        max_sample_lines=max_sample_lines,
    )
    plot_mace_histograms(
        dataset=dataset,
        patient_idx=selected_idx,
        factual_samples=factual_samples,
        counterfactual_samples=counterfactual_samples,
        output_path=mace_path,
    )

    target_names = raw_config.get("target_names") or [
        str(item["name"]) for item in preprocessing.target_metadata
    ]
    summary_path = output_dir / "eval_summary.json"
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "csv_path": None if csv_path is None else str(csv_path),
        "device": str(device),
        "n_samples": int(n_samples),
        "n_steps": int(n_steps),
        "seed": int(seed),
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "target_names": target_names,
        "selected_subject_ids": dataset.subject_ids[selected_idx].tolist(),
        "selected_procedures": dataset.surgery_type[selected_idx].tolist(),
        "outputs": {
            "selected_patients": str(selected_path),
            "bmi": str(bmi_path),
            "hba1c": str(hba1c_path),
            "mace": str(mace_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None, help="Run dir, Optuna study dir, or best_model dir.")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/python_flow_matching_optuna"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-show-per-procedure", type=int, default=N_SHOW_PER_PROCEDURE)
    parser.add_argument("--max-sample-lines", type=int, default=MAX_SAMPLE_LINES)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>.")
    args = parser.parse_args()
    if args.n_samples < 1:
        raise SystemExit("--n-samples must be at least 1.")
    if args.n_steps < 1:
        raise SystemExit("--n-steps must be at least 1.")

    run_dir = resolve_run_dir(args.run, args.log_dir)
    summary = evaluate_run(
        run_dir=run_dir,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        seed=args.seed,
        n_show_per_procedure=args.n_show_per_procedure,
        max_sample_lines=args.max_sample_lines,
        device_name=args.device,
    )

    print(f"Run: {summary['run_dir']}")
    print(f"Split sizes: {summary['split_sizes']}")
    print("Selected test patients:")
    for subject_id, procedure in zip(summary["selected_subject_ids"], summary["selected_procedures"]):
        print(f"  {subject_id}: {procedure}")
    print("Saved outputs:")
    for name, path in summary["outputs"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
