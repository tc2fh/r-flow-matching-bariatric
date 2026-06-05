"""Plot factual and counterfactual examples from a trained PyTorch flow run.

Run against the Optuna final model directory:
    python evaluate_flow_matching.py --run runs/python_flow_matching_optuna/<study>/best_model/<run>

If the model was trained from Cosmos, the evaluator reloads Cosmos by default.
For a CSV export, pass:
    python evaluate_flow_matching.py --run <run> --csv data/cosmos_mbs_flow_input.csv

Outputs in the run directory by default:
    eval_selected_test_patients.csv
    eval_timepoint_metrics_test.csv
    eval_bmi_counterfactual_welch_ttests_test.csv
    eval_bmi_counterfactual_welch_ttests_test.png
    eval_hba1c_counterfactual_welch_ttests_test.csv
    eval_hba1c_counterfactual_welch_ttests_test.png
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
from typing import Any, Callable
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import train_flow_matching as fm

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - evaluation reports a clear error before t-tests.
    scipy_stats = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps Cosmos execution portable if tqdm is absent.
    tqdm = None


N_SHOW_PER_PROCEDURE = 3
MAX_SAMPLE_LINES = 50
INDEX_TO_SURGERY = {idx: name for name, idx in fm.SURGERY_TO_INDEX.items()}


class NullProgress:
    def update(self, n: int = 1) -> None:
        return None

    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def progress_bar(total: int, desc: str, unit: str, disabled: bool):
    if disabled or tqdm is None:
        return NullProgress()
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


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
    progress_callback: Callable[[int], None] | None = None,
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
            if progress_callback is not None:
                progress_callback(1)
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
    show_progress: bool,
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

    with progress_bar(
        total=2 * n_steps,
        desc="Sampling selected factual/counterfactual plots",
        unit="step",
        disabled=not show_progress,
    ) as pbar:
        factual_std = sample_with_surgery(
            model=model,
            surgery_idx=actual,
            patient_features=patient_features,
            initial_noise=initial_noise,
            n_steps=n_steps,
            device=device,
            progress_callback=pbar.update,
        )
        counterfactual_std = sample_with_surgery(
            model=model,
            surgery_idx=counterfactual,
            patient_features=patient_features,
            initial_noise=initial_noise,
            n_steps=n_steps,
            device=device,
            progress_callback=pbar.update,
        )
    return (
        fm.unstandardize(factual_std, preprocessing).astype(np.float32),
        fm.unstandardize(counterfactual_std, preprocessing).astype(np.float32),
    )


def timepoint_metric_table(
    dataset: fm.FlowDataset,
    patient_idx: np.ndarray,
    point_predictions: np.ndarray,
) -> pd.DataFrame:
    rows = []
    labels = {"bmi": "BMI", "hba1c": "HbA1c"}
    for group in ("bmi", "hba1c"):
        dims, months = target_dims(dataset, group)
        for dim, month in zip(dims, months):
            observed = dataset.mask[patient_idx, dim] == 1
            n_observed = int(observed.sum())
            if n_observed == 0:
                mad = np.nan
                rmse = np.nan
            else:
                truth = dataset.x[patient_idx[observed], dim]
                pred = point_predictions[observed, dim]
                diff = pred - truth
                mad = float(np.median(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff**2)))
            rows.append(
                {
                    "outcome": labels[group],
                    "timepoint": f"{_format_month(float(month))}m",
                    "n_observed": n_observed,
                    "MAD": mad,
                    "RMSE": rmse,
                }
            )
    return pd.DataFrame(rows)


def format_metric_table(table: pd.DataFrame) -> str:
    display = table.copy()
    for column in ("MAD", "RMSE"):
        display[column] = display[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.3f}")
    return display.to_string(index=False)


def benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg FDR rejection mask for finite p-values."""
    p = np.asarray(p_values, dtype=float)
    reject = np.zeros(p.shape, dtype=bool)
    finite = np.isfinite(p)
    if not finite.any():
        return reject
    flat_idx = np.flatnonzero(finite)
    p_flat = p.flat[flat_idx]
    order = np.argsort(p_flat)
    sorted_p = p_flat[order]
    m = len(sorted_p)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = sorted_p <= thresholds
    if not passed.any():
        return reject
    cutoff = np.max(np.where(passed)[0])
    reject_flat = np.zeros(m, dtype=bool)
    reject_flat[order[: cutoff + 1]] = True
    reject.flat[flat_idx] = reject_flat
    return reject


def init_ttest_state(dataset: fm.FlowDataset, n_patients: int, group: str, outcome_label: str) -> dict[str, Any]:
    dims, months = target_dims(dataset, group)
    shape = (n_patients, len(dims))
    return {
        "group": group,
        "outcome_label": outcome_label,
        "dims": dims,
        "months": months,
        "t_stat": np.full(shape, np.nan, dtype=np.float32),
        "p_value": np.full(shape, np.nan, dtype=np.float32),
        "mean_diff": np.full(shape, np.nan, dtype=np.float32),
        "cohen_d": np.full(shape, np.nan, dtype=np.float32),
        "records": [],
    }


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def update_ttest_state(
    state: dict[str, Any],
    dataset: fm.FlowDataset,
    batch_idx: np.ndarray,
    batch_positions: np.ndarray,
    factual_samples: np.ndarray,
    counterfactual_samples: np.ndarray,
) -> None:
    if scipy_stats is None:
        raise RuntimeError("scipy is required for Welch t-tests. Install scipy or rerun without this analysis.")

    dims = state["dims"]
    months = state["months"]
    outcome_label = state["outcome_label"]
    factual_values = factual_samples[:, :, dims]
    counterfactual_values = counterfactual_samples[:, :, dims]

    for local_pos, global_idx in enumerate(batch_idx):
        patient_pos = int(batch_positions[local_pos])
        factual_proc = INDEX_TO_SURGERY[int(dataset.surgery_idx[global_idx])]
        cf_proc = INDEX_TO_SURGERY[int(1 - dataset.surgery_idx[global_idx])]
        for month_pos, month in enumerate(months):
            factual = factual_values[local_pos, :, month_pos]
            counterfactual = counterfactual_values[local_pos, :, month_pos]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                test = scipy_stats.ttest_ind(
                    factual,
                    counterfactual,
                    equal_var=False,
                    alternative="two-sided",
            )
            diff = float(np.mean(factual) - np.mean(counterfactual))
            if factual.size > 1 and counterfactual.size > 1:
                pooled = np.sqrt(0.5 * (np.var(factual, ddof=1) + np.var(counterfactual, ddof=1)))
            else:
                pooled = np.nan
            effect = diff / pooled if pooled > 0 else np.nan

            state["t_stat"][patient_pos, month_pos] = float(test.statistic)
            state["p_value"][patient_pos, month_pos] = float(test.pvalue)
            state["mean_diff"][patient_pos, month_pos] = diff
            state["cohen_d"][patient_pos, month_pos] = effect
            state["records"].append(
                {
                    "_patient_pos": patient_pos,
                    "_month_pos": int(month_pos),
                    "subject_id": str(dataset.subject_ids[global_idx]),
                    "actual_procedure": factual_proc,
                    "counterfactual_procedure": cf_proc,
                    "outcome": outcome_label,
                    "month": float(month),
                    "factual_mean": float(np.mean(factual)),
                    "counterfactual_mean": float(np.mean(counterfactual)),
                    "mean_diff_factual_minus_counterfactual": diff,
                    "welch_t": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "cohen_d": float(effect) if np.isfinite(effect) else np.nan,
                }
            )


def finalize_ttest_state(state: dict[str, Any], alpha: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    p_value = state["p_value"]
    t_stat = state["t_stat"]
    mean_diff = state["mean_diff"]
    cohen_d = state["cohen_d"]
    months = state["months"]
    outcome_label = state["outcome_label"]
    raw_reject = p_value < alpha
    fdr_reject = benjamini_hochberg(p_value, alpha=alpha)

    records = []
    for record in state["records"]:
        patient_pos = int(record.pop("_patient_pos"))
        month_pos = int(record.pop("_month_pos"))
        record["significant_raw"] = bool(raw_reject[patient_pos, month_pos])
        record["significant_bh_fdr"] = bool(fdr_reject[patient_pos, month_pos])
        records.append(record)

    month_summary = {}
    for month_pos, month in enumerate(months):
        month_key = _format_month(float(month))
        month_summary[month_key] = {
            "n_patients": int(p_value.shape[0]),
            "raw_significant_n": int(raw_reject[:, month_pos].sum()),
            "raw_significant_frac": float(raw_reject[:, month_pos].mean()),
            "bh_fdr_significant_n": int(fdr_reject[:, month_pos].sum()),
            "bh_fdr_significant_frac": float(fdr_reject[:, month_pos].mean()),
            "median_p_value": _finite_median(p_value[:, month_pos]),
            "median_abs_cohen_d": _finite_median(np.abs(cohen_d[:, month_pos])),
            "mean_abs_difference": _finite_mean(np.abs(mean_diff[:, month_pos])),
        }

    summary = {
        "test": "Welch independent two-sample t-test",
        "alpha": float(alpha),
        "multiple_testing": f"Benjamini-Hochberg FDR over all patient-month {outcome_label} tests",
        "same_initial_noise_used_for_factual_and_counterfactual": True,
        "outcome": outcome_label,
        "months": [float(month) for month in months],
        "month_summary": month_summary,
        "p_value": p_value,
        "t_stat": t_stat,
        "mean_diff_factual_minus_counterfactual": mean_diff,
        "cohen_d": cohen_d,
        "significant_raw": raw_reject,
        "significant_bh_fdr": fdr_reject,
    }
    return pd.DataFrame(records), summary


def sample_test_factual_counterfactual_analysis(
    model: fm.VectorFieldNet,
    dataset: fm.FlowDataset,
    patient_idx: np.ndarray,
    preprocessing: fm.Preprocessing,
    n_samples: int,
    n_steps: int,
    x_dim: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    alpha: float,
    show_progress: bool,
) -> tuple[np.ndarray, dict[str, tuple[pd.DataFrame, dict[str, Any]]]]:
    """Sample full test factual/counterfactual distributions for metrics and t-tests."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if scipy_stats is None:
        raise RuntimeError("scipy is required for Welch t-tests. Install scipy or rerun without this analysis.")

    point_predictions = np.zeros((len(patient_idx), x_dim), dtype=np.float32)
    states = {
        "bmi": init_ttest_state(dataset, len(patient_idx), "bmi", "BMI"),
        "hba1c": init_ttest_state(dataset, len(patient_idx), "hba1c", "HbA1c"),
    }

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    batch_starts = range(0, len(patient_idx), batch_size)
    n_batches = len(range(0, len(patient_idx), batch_size))
    with progress_bar(
        total=n_batches * 2 * n_steps,
        desc=f"Sampling test factual/counterfactual analyses ({len(patient_idx)} patients)",
        unit="step",
        disabled=not show_progress,
    ) as pbar:
        for start in batch_starts:
            end = min(start + batch_size, len(patient_idx))
            batch_idx = patient_idx[start:end]
            patient_features = fm.transform_patient_features(
                dataset.patient_features_raw[batch_idx],
                preprocessing,
            )
            initial_noise = torch.randn(
                (len(batch_idx), n_samples, x_dim),
                dtype=torch.float32,
                device=device,
            )
            factual_std = sample_with_surgery(
                model=model,
                surgery_idx=dataset.surgery_idx[batch_idx].astype(np.int64),
                patient_features=patient_features,
                initial_noise=initial_noise,
                n_steps=n_steps,
                device=device,
                progress_callback=pbar.update,
            )
            counterfactual_std = sample_with_surgery(
                model=model,
                surgery_idx=(1 - dataset.surgery_idx[batch_idx]).astype(np.int64),
                patient_features=patient_features,
                initial_noise=initial_noise,
                n_steps=n_steps,
                device=device,
                progress_callback=pbar.update,
            )
            factual_samples = fm.unstandardize(factual_std, preprocessing).astype(np.float32)
            counterfactual_samples = fm.unstandardize(counterfactual_std, preprocessing).astype(np.float32)
            point_predictions[start:end] = np.median(factual_samples, axis=1)
            batch_positions = np.arange(start, end, dtype=np.int64)
            for state in states.values():
                update_ttest_state(
                    state=state,
                    dataset=dataset,
                    batch_idx=batch_idx,
                    batch_positions=batch_positions,
                    factual_samples=factual_samples,
                    counterfactual_samples=counterfactual_samples,
                )

    return point_predictions, {
        group: finalize_ttest_state(state, alpha=alpha)
        for group, state in states.items()
    }


def plot_outcome_ttest_summary(
    ttest_summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot test-set factual-vs-counterfactual Welch t-test results."""
    outcome = str(ttest_summary["outcome"])
    months = np.asarray(ttest_summary["months"], dtype=float)
    raw_sig = np.asarray(ttest_summary["significant_raw"], dtype=bool)
    fdr_sig = np.asarray(ttest_summary["significant_bh_fdr"], dtype=bool)
    p_value = np.asarray(ttest_summary["p_value"], dtype=float)
    cohen_d = np.asarray(ttest_summary["cohen_d"], dtype=float)
    mean_diff = np.asarray(
        ttest_summary["mean_diff_factual_minus_counterfactual"],
        dtype=float,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sort_score = np.nanmean(np.abs(mean_diff), axis=1)
    sort_score = np.nan_to_num(sort_score, nan=-np.inf)
    sort_order = np.argsort(sort_score)
    x_pos = np.arange(len(months))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        gridspec_kw={"height_ratios": [1, 2]},
        constrained_layout=True,
    )

    width = 0.36
    axes[0, 0].bar(
        x_pos - width / 2,
        raw_sig.mean(axis=0),
        width=width,
        label="raw p<alpha",
        color="tab:blue",
    )
    axes[0, 0].bar(
        x_pos + width / 2,
        fdr_sig.mean(axis=0),
        width=width,
        label="BH-FDR significant",
        color="tab:orange",
    )
    axes[0, 0].set_xticks(x_pos, [_format_month(m) for m in months])
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xlabel("Months post-op")
    axes[0, 0].set_ylabel("Fraction of test patients")
    axes[0, 0].set_title("Welch t-test significance rate")
    axes[0, 0].legend(fontsize=8)

    effect_data = []
    for month_pos in range(len(months)):
        values = np.abs(cohen_d[:, month_pos])
        values = values[np.isfinite(values)]
        effect_data.append(values if values.size else np.asarray([np.nan]))
    axes[0, 1].boxplot(
        effect_data,
        tick_labels=[_format_month(m) for m in months],
        showfliers=False,
    )
    axes[0, 1].set_xlabel("Months post-op")
    axes[0, 1].set_ylabel("|Cohen's d|")
    axes[0, 1].set_title("Per-patient factual vs counterfactual effect size")

    heat = -np.log10(np.clip(p_value[sort_order], 1e-300, 1.0))
    im = axes[1, 0].imshow(heat, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[1, 0].set_xticks(x_pos, [_format_month(m) for m in months])
    axes[1, 0].set_xlabel("Months post-op")
    axes[1, 0].set_ylabel(f"Test patients sorted by mean |{outcome} diff|")
    axes[1, 0].set_title("-log10 p-value")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(
        fdr_sig[sort_order].astype(float),
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_xticks(x_pos, [_format_month(m) for m in months])
    axes[1, 1].set_xlabel("Months post-op")
    axes[1, 1].set_ylabel(f"Test patients sorted by mean |{outcome} diff|")
    axes[1, 1].set_title("BH-FDR significant cells")
    cbar = fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["no", "yes"])

    fig.suptitle(
        f"Test-set {outcome} factual vs counterfactual Welch independent-sample t-tests",
        y=1.03,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def ttest_json_summary(ttest_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "test": ttest_summary["test"],
        "alpha": ttest_summary["alpha"],
        "multiple_testing": ttest_summary["multiple_testing"],
        "same_initial_noise_used_for_factual_and_counterfactual": (
            ttest_summary["same_initial_noise_used_for_factual_and_counterfactual"]
        ),
        "outcome": ttest_summary["outcome"],
        "month_summary": ttest_summary["month_summary"],
    }


def format_ttest_month_summary(ttest_summary: dict[str, Any]) -> str:
    rows = []
    for month, values in ttest_summary["month_summary"].items():
        rows.append(
            {
                "outcome": ttest_summary["outcome"],
                "timepoint": f"{month}m",
                "n_patients": values["n_patients"],
                "raw_sig_frac": values["raw_significant_frac"],
                "bh_fdr_sig_frac": values["bh_fdr_significant_frac"],
                "median_p": values["median_p_value"],
                "median_abs_d": values["median_abs_cohen_d"],
            }
        )
    table = pd.DataFrame(rows)
    for column in ("raw_sig_frac", "bh_fdr_sig_frac", "median_p", "median_abs_d"):
        table[column] = table[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.3f}")
    return table.to_string(index=False)


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
    y_limits: tuple[float, float] | None = None,
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
        if y_limits is not None:
            ax.set_ylim(*y_limits)
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
        ax.set_title(f"{subject_id}: predicted composite event")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Predicted MACE/nephropathy/retinopathy probability")
    fig.suptitle("Test composite event factual and counterfactual predictive distributions", y=1.01)
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
    metric_batch_size: int,
    alpha: float,
    device_name: str,
    show_progress: bool,
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
        show_progress=show_progress,
    )

    selected_path = output_dir / "eval_selected_test_patients.csv"
    metrics_path = output_dir / "eval_timepoint_metrics_test.csv"
    bmi_ttest_csv_path = output_dir / "eval_bmi_counterfactual_welch_ttests_test.csv"
    bmi_ttest_path = output_dir / "eval_bmi_counterfactual_welch_ttests_test.png"
    hba1c_ttest_csv_path = output_dir / "eval_hba1c_counterfactual_welch_ttests_test.csv"
    hba1c_ttest_path = output_dir / "eval_hba1c_counterfactual_welch_ttests_test.png"
    bmi_path = output_dir / "eval_bmi_factual_counterfactual_examples_test.png"
    hba1c_path = output_dir / "eval_hba1c_factual_counterfactual_examples_test.png"
    mace_path = output_dir / "eval_mace_factual_counterfactual_histograms_test.png"

    selected_patient_frame(dataset, selected_idx).to_csv(selected_path, index=False)
    test_point_predictions, ttest_results = sample_test_factual_counterfactual_analysis(
        model=model,
        dataset=dataset,
        patient_idx=test_idx,
        preprocessing=preprocessing,
        n_samples=n_samples,
        n_steps=n_steps,
        x_dim=x_dim,
        seed=seed + 1,
        device=device,
        batch_size=metric_batch_size,
        alpha=alpha,
        show_progress=show_progress,
    )
    timepoint_metrics = timepoint_metric_table(dataset, test_idx, test_point_predictions)
    timepoint_metrics.to_csv(metrics_path, index=False)

    ttest_paths = {
        "bmi": {"csv": bmi_ttest_csv_path, "figure": bmi_ttest_path},
        "hba1c": {"csv": hba1c_ttest_csv_path, "figure": hba1c_ttest_path},
    }
    ttest_json = {}
    for group, (ttest_df, ttest_summary) in ttest_results.items():
        ttest_df.to_csv(ttest_paths[group]["csv"], index=False)
        plot_outcome_ttest_summary(
            ttest_summary=ttest_summary,
            output_path=ttest_paths[group]["figure"],
        )
        ttest_json[group] = {
            **ttest_json_summary(ttest_summary),
            "csv": str(ttest_paths[group]["csv"]),
            "figure": str(ttest_paths[group]["figure"]),
        }

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
        y_limits=(15, 90.0),
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
        y_limits=(3.0, 15.0),
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
        "metric_batch_size": int(metric_batch_size),
        "alpha": float(alpha),
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "target_names": target_names,
        "selected_subject_ids": dataset.subject_ids[selected_idx].tolist(),
        "selected_procedures": dataset.surgery_type[selected_idx].tolist(),
        "timepoint_metrics": timepoint_metrics.to_dict(orient="records"),
        "counterfactual_welch_ttests_test": ttest_json,
        "outputs": {
            "selected_patients": str(selected_path),
            "timepoint_metrics": str(metrics_path),
            "bmi_welch_ttests": str(bmi_ttest_path),
            "bmi_welch_ttests_csv": str(bmi_ttest_csv_path),
            "hba1c_welch_ttests": str(hba1c_ttest_path),
            "hba1c_welch_ttests_csv": str(hba1c_ttest_csv_path),
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
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for Welch t-tests and BH-FDR.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()
    if args.n_samples < 1:
        raise SystemExit("--n-samples must be at least 1.")
    if args.n_steps < 1:
        raise SystemExit("--n-steps must be at least 1.")
    if args.metric_batch_size < 1:
        raise SystemExit("--metric-batch-size must be at least 1.")
    if not (0.0 < args.alpha < 1.0):
        raise SystemExit("--alpha must be between 0 and 1.")
    show_progress = not args.no_progress
    if show_progress and tqdm is None:
        warnings.warn("tqdm is not installed; progress bars are disabled.", stacklevel=2)
    if scipy_stats is None:
        raise SystemExit("scipy is required for Welch t-test evaluation. Install scipy in this environment.")

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
        metric_batch_size=args.metric_batch_size,
        alpha=args.alpha,
        device_name=args.device,
        show_progress=show_progress,
    )

    print(f"Run: {summary['run_dir']}")
    print(f"Split sizes: {summary['split_sizes']}")
    print("Selected test patients:")
    for subject_id, procedure in zip(summary["selected_subject_ids"], summary["selected_procedures"]):
        print(f"  {subject_id}: {procedure}")
    print("Test timepoint metrics (median factual prediction vs observed):")
    print(format_metric_table(pd.DataFrame(summary["timepoint_metrics"])))
    for group, ttest_summary in summary["counterfactual_welch_ttests_test"].items():
        print(f"{ttest_summary['outcome']} counterfactual Welch t-test month summary:")
        print(format_ttest_month_summary(ttest_summary))
    print("Saved outputs:")
    for name, path in summary["outputs"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
