"""Evaluate a trained joint multi-task (flow + MACE) run.

Produces two families of outputs so results line up with the Cosmos flow model
(``evaluate_flow_matching.py``):

Continuous BMI/HbA1c outcomes (same file names/format as the Cosmos evaluator,
via a thin 17-dim adapter around the multi-task flow head):
    eval_timepoint_metrics_test.csv
    eval_bmi_counterfactual_welch_ttests_test.{csv,png}
    eval_hba1c_counterfactual_welch_ttests_test.{csv,png}
    eval_bmi_factual_counterfactual_examples_test.png
    eval_hba1c_factual_counterfactual_examples_test.png

Composite MACE risk (from the classification head -- the Cosmos flow model has
no calibrated head, so these are new, comparable risk metrics):
    eval_mace_metrics_test.csv
    eval_mace_operating_points_test.csv
    eval_mace_predictions_test.csv
    eval_mace_curves_test.png
    eval_mace_counterfactual_test.png
    eval_summary.json

Run against an Optuna best-model dir or a single run dir::

    python evaluate_multitask.py --run runs/python_flow_matching_multitask/<run> \
        --csv fake_data/fake_mbs_cohort.csv
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
from torch import nn

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, roc_curve

import train_flow_matching as fm
import train_flow_matching_multitask as mt
import evaluate_flow_matching as ev

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


DEFAULT_LOG_DIR = Path("runs/python_flow_matching_multitask_optuna")


# --------------------------------------------------------------------------- #
# Loading (reusable by the comparison script)
# --------------------------------------------------------------------------- #
def load_config(run_dir: Path) -> mt.MultiTaskConfig:
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    valid = {field.name for field in fields(mt.MultiTaskConfig)}
    return mt.MultiTaskConfig(**{key: value for key, value in raw.items() if key in valid})


def load_preprocessing(run_dir: Path) -> mt.Preprocessing:
    raw = json.loads((run_dir / "preprocessing.json").read_text(encoding="utf-8"))
    return mt.Preprocessing(
        target_mean=np.asarray(raw["target_mean"], dtype=np.float32),
        target_std=np.asarray(raw["target_std"], dtype=np.float32),
        static_mean=np.asarray(raw["static_mean"], dtype=np.float32),
        static_std=np.asarray(raw["static_std"], dtype=np.float32),
        static_continuous_idx=np.asarray(raw["static_continuous_idx"], dtype=np.int64),
        patient_feature_names=list(raw["patient_feature_names"]),
        cont_names=list(raw["cont_names"]),
    )


def restore_model(run_dir: Path, cfg: mt.MultiTaskConfig, device: torch.device) -> mt.MultiTaskNet:
    model = mt.MultiTaskNet(cfg, mt.X_CONT_DIM, len(fm.PATIENT_FEATURES)).to(device)
    try:
        state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_run(run_dir: Path, device: torch.device) -> tuple[mt.MultiTaskConfig, mt.Preprocessing, mt.MultiTaskNet]:
    cfg = load_config(run_dir)
    pre = load_preprocessing(run_dir)
    if len(pre.cont_names) != mt.X_CONT_DIM:
        warnings.warn(
            f"preprocessing has {len(pre.cont_names)} continuous dims but module expects {mt.X_CONT_DIM}.",
            stacklevel=2,
        )
    model = restore_model(run_dir, cfg, device)
    return cfg, pre, model


# --------------------------------------------------------------------------- #
# MACE classification head (reusable by the comparison script)
# --------------------------------------------------------------------------- #
def mace_arrays(dataset: fm.FlowDataset, idx: np.ndarray, pre: mt.Preprocessing, flip_surgery: bool) -> dict:
    surgery = dataset.surgery_idx[idx].astype(np.int64)
    if flip_surgery:
        surgery = (1 - surgery).astype(np.int64)
    return {
        "patient_features": mt.transform_patient_features(dataset.patient_features_raw[idx], pre),
        "surgery_idx": surgery,
    }


def predict_mace(
    model: mt.MultiTaskNet,
    dataset: fm.FlowDataset,
    idx: np.ndarray,
    pre: mt.Preprocessing,
    device: torch.device,
    flip_surgery: bool = False,
) -> np.ndarray:
    return mt.predict_mace_proba(model, mace_arrays(dataset, idx, pre, flip_surgery), device)


# --------------------------------------------------------------------------- #
# 17-dim adapter so the Cosmos continuous-outcome machinery can drive this model
# --------------------------------------------------------------------------- #
class FlowAdapter(nn.Module):
    """Expose the multi-task flow head with VectorFieldNet's call signature.

    The Cosmos evaluator integrates a full ``fm.X_DIM`` ODE and reads only the
    BMI/HbA1c dims. We evolve exactly the continuous dims via the multi-task
    velocity and leave the (unused) MACE dims at zero velocity.
    """

    def __init__(self, model: mt.MultiTaskNet, cont_dims: np.ndarray, full_dim: int):
        super().__init__()
        self.model = model
        self.register_buffer("cont_dims", torch.as_tensor(cont_dims, dtype=torch.long))
        self.full_dim = full_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, surgery_idx: torch.Tensor, patient_features: torch.Tensor) -> torch.Tensor:
        dim = x.dim() - 1
        cond = self.model.encode(surgery_idx, patient_features)
        v_cont = self.model.velocity(x.index_select(dim, self.cont_dims), t, cond)
        v = torch.zeros_like(x)
        v.index_copy_(dim, self.cont_dims, v_cont)
        return v


def expand_preprocessing(pre: mt.Preprocessing) -> fm.Preprocessing:
    """Place the 15 continuous target stats into a full 17-dim fm.Preprocessing."""
    target_mean = np.zeros(fm.X_DIM, dtype=np.float32)
    target_std = np.ones(fm.X_DIM, dtype=np.float32)
    target_mean[mt.CONT_DIMS] = pre.target_mean
    target_std[mt.CONT_DIMS] = pre.target_std
    return fm.Preprocessing(
        target_mean=target_mean,
        target_std=target_std,
        static_mean=pre.static_mean,
        static_std=pre.static_std,
        static_continuous_idx=pre.static_continuous_idx,
        patient_feature_names=pre.patient_feature_names,
        target_metadata=fm.target_metadata(),
    )


# --------------------------------------------------------------------------- #
# MACE plots
# --------------------------------------------------------------------------- #
def plot_mace_curves(y_true: np.ndarray, prob: np.ndarray, output_path: Path) -> None:
    prevalence = float(y_true.mean()) if y_true.size else float("nan")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    try:
        fpr, tpr, _ = roc_curve(y_true, prob)
        axes[0].plot(fpr, tpr)
    except Exception:
        pass
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    try:
        prec, rec, _ = precision_recall_curve(y_true, prob)
        axes[1].plot(rec, prec)
    except Exception:
        pass
    axes[1].axhline(prevalence, color="k", ls="--", alpha=0.4)
    axes[1].set(title=f"Precision-Recall (test, prevalence={prevalence:.3f})", xlabel="Recall", ylabel="Precision")
    table = ev_reliability(y_true, prob)
    if not table.empty:
        axes[2].plot(table["mean_pred"], table["frac_pos"], marker="o")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[2].set(title="Reliability (test)", xlabel="Mean predicted", ylabel="Observed frequency")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    ev.report_saved(output_path)


def ev_reliability(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            rows.append({"mean_pred": float(prob[sel].mean()), "frac_pos": float(y_true[sel].mean()), "count": int(sel.sum())})
    return pd.DataFrame(rows)


def plot_mace_counterfactual(
    surgery_type: np.ndarray, factual: np.ndarray, counterfactual: np.ndarray, output_path: Path
) -> dict[str, Any]:
    delta = counterfactual - factual
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"sleeve": "tab:blue", "rnygb": "tab:orange"}
    for proc in ("sleeve", "rnygb"):
        sel = surgery_type == proc
        if sel.any():
            axes[0].scatter(factual[sel], counterfactual[sel], s=14, alpha=0.5, color=colors.get(proc, "tab:gray"), label=f"factual {proc}")
    lim = [0, 1]
    axes[0].plot(lim, lim, "k--", alpha=0.5)
    axes[0].set(title="Per-patient MACE risk: factual vs counterfactual surgery", xlabel="Factual risk", ylabel="Counterfactual risk", xlim=lim, ylim=lim)
    axes[0].legend(fontsize=8)
    axes[1].hist(delta, bins=40, color="tab:green", alpha=0.8)
    axes[1].axvline(0.0, color="k", lw=1.0)
    axes[1].axvline(float(np.median(delta)), color="tab:red", lw=2.0, label=f"median delta={np.median(delta):.4f}")
    axes[1].set(title="Counterfactual - factual MACE risk", xlabel="Delta risk", ylabel="Patients")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    ev.report_saved(output_path)

    summary: dict[str, Any] = {
        "median_counterfactual_minus_factual": float(np.median(delta)),
        "mean_counterfactual_minus_factual": float(np.mean(delta)),
        "frac_increased_risk": float(np.mean(delta > 0)),
    }
    if scipy_stats is not None and np.any(delta != 0):
        try:
            wilcoxon = scipy_stats.wilcoxon(counterfactual, factual)
            summary["wilcoxon_statistic"] = float(wilcoxon.statistic)
            summary["wilcoxon_p_value"] = float(wilcoxon.pvalue)
        except Exception:
            summary["wilcoxon_p_value"] = None
    return summary


# --------------------------------------------------------------------------- #
# MACE classification evaluation (reusable)
# --------------------------------------------------------------------------- #
def evaluate_mace_head(
    model: mt.MultiTaskNet,
    dataset: fm.FlowDataset,
    splits: dict[str, np.ndarray],
    pre: mt.Preprocessing,
    cfg: mt.MultiTaskConfig,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    val_idx, test_idx = splits["val"], splits["test"]
    y_val = dataset.x[val_idx, mt.MACE_DIM].astype(np.int64)
    y_test = dataset.x[test_idx, mt.MACE_DIM].astype(np.int64)

    val_prob = predict_mace(model, dataset, val_idx, pre, device)
    test_prob = predict_mace(model, dataset, test_idx, pre, device)
    test_prob_cf = predict_mace(model, dataset, test_idx, pre, device, flip_surgery=True)

    test_prob_cal = test_prob
    calibrated = False
    if cfg.recalibrate and y_val.size >= 10 and len(np.unique(y_val)) == 2:
        try:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(val_prob, y_val)
            test_prob_cal = iso.transform(test_prob)
            calibrated = True
        except Exception as exc:
            warnings.warn(f"Isotonic recalibration skipped: {exc}", stacklevel=2)

    metric_rows = [{"split": "test", **mt.discrimination_metrics(y_test, test_prob)}]
    if calibrated:
        metric_rows.append({"split": "test_calibrated", **mt.discrimination_metrics(y_test, test_prob_cal)})
    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "eval_mace_metrics_test.csv"
    metrics.to_csv(metrics_path, index=False)
    ev.report_saved(metrics_path, "MACE discrimination metrics")
    ev.render_table(metrics, output_dir / "eval_mace_metrics_test.png", "Composite MACE discrimination (test)")

    thr_youden = mt.youden_threshold(y_val, val_prob) if y_val.size else 0.5
    thr_spec = mt.threshold_at_specificity(y_val, val_prob, cfg.target_specificity) if y_val.size else 0.5
    operating_points = pd.DataFrame(
        [
            mt.operating_point(y_test, test_prob, 0.5, "default_0.5"),
            mt.operating_point(y_test, test_prob, thr_youden, "youden_val"),
            mt.operating_point(y_test, test_prob, thr_spec, f"spec>={cfg.target_specificity:g}_val"),
        ]
    )
    op_path = output_dir / "eval_mace_operating_points_test.csv"
    operating_points.to_csv(op_path, index=False)
    ev.report_saved(op_path, "MACE operating points")
    ev.render_table(operating_points, output_dir / "eval_mace_operating_points_test.png", "MACE operating points (test)")

    predictions = pd.DataFrame(
        {
            "subject_id": dataset.subject_ids[test_idx],
            "surgery": dataset.surgery_type[test_idx],
            "y_true": y_test,
            "prob_factual": test_prob,
            "prob_counterfactual": test_prob_cf,
        }
    )
    if calibrated:
        predictions["prob_factual_calibrated"] = test_prob_cal
    pred_path = output_dir / "eval_mace_predictions_test.csv"
    predictions.to_csv(pred_path, index=False)
    ev.report_saved(pred_path, "MACE per-patient predictions")

    plot_mace_curves(y_test, test_prob, output_dir / "eval_mace_curves_test.png")
    cf_summary = plot_mace_counterfactual(
        dataset.surgery_type[test_idx], test_prob, test_prob_cf, output_dir / "eval_mace_counterfactual_test.png"
    )

    return {
        "metrics": metrics.to_dict(orient="records"),
        "operating_points": operating_points.to_dict(orient="records"),
        "calibrated": calibrated,
        "counterfactual_surgery": cf_summary,
    }


# --------------------------------------------------------------------------- #
# Continuous-outcome evaluation (delegates to the Cosmos evaluator)
# --------------------------------------------------------------------------- #
def evaluate_continuous(
    adapter: FlowAdapter,
    dataset: fm.FlowDataset,
    splits: dict[str, np.ndarray],
    expanded_pre: fm.Preprocessing,
    output_dir: Path,
    n_samples: int,
    n_steps: int,
    seed: int,
    n_show_per_procedure: int,
    max_sample_lines: int,
    metric_batch_size: int,
    alpha: float,
    device: torch.device,
    show_progress: bool,
) -> dict[str, Any]:
    test_idx = splits["test"]
    selected_idx = ev.select_display_patients(dataset, test_idx, np.random.default_rng(seed), n_show_per_procedure)
    factual_samples, counterfactual_samples = ev.sample_factual_counterfactual(
        model=adapter, dataset=dataset, patient_idx=selected_idx, preprocessing=expanded_pre,
        n_samples=n_samples, n_steps=n_steps, x_dim=fm.X_DIM, seed=seed, device=device, show_progress=show_progress,
    )
    selected_path = output_dir / "eval_selected_test_patients.csv"
    ev.selected_patient_frame(dataset, selected_idx).to_csv(selected_path, index=False)
    ev.report_saved(selected_path, "selected test patients")

    point_predictions, _mace_prob, ttest_results = ev.sample_test_factual_counterfactual_analysis(
        model=adapter, dataset=dataset, patient_idx=test_idx, preprocessing=expanded_pre,
        n_samples=n_samples, n_steps=n_steps, x_dim=fm.X_DIM, seed=seed + 1, device=device,
        batch_size=metric_batch_size, alpha=alpha, show_progress=show_progress,
    )
    timepoint_metrics = ev.timepoint_metric_table(dataset, test_idx, point_predictions)
    timepoint_path = output_dir / "eval_timepoint_metrics_test.csv"
    timepoint_metrics.to_csv(timepoint_path, index=False)
    ev.report_saved(timepoint_path, "BMI/HbA1c timepoint metrics")
    ev.render_table(timepoint_metrics, output_dir / "eval_timepoint_metrics_test.png", "BMI/HbA1c timepoint metrics (test)")

    ttest_paths = {
        "bmi": output_dir / "eval_bmi_counterfactual_welch_ttests_test",
        "hba1c": output_dir / "eval_hba1c_counterfactual_welch_ttests_test",
    }
    ttest_json = {}
    for group, (ttest_df, ttest_summary) in ttest_results.items():
        ttest_df.to_csv(ttest_paths[group].with_suffix(".csv"), index=False)
        ev.report_saved(ttest_paths[group].with_suffix(".csv"), f"{group} counterfactual Welch t-tests")
        ev.plot_outcome_ttest_summary(ttest_summary, ttest_paths[group].with_suffix(".png"))
        ttest_json[group] = ev.ttest_json_summary(ttest_summary)

    ev.plot_timecourse_factual_counterfactual(
        dataset, selected_idx, factual_samples, counterfactual_samples, "bmi", "BMI",
        output_dir / "eval_bmi_factual_counterfactual_examples_test.png",
        f"Test BMI factual/counterfactual ({n_show_per_procedure} sleeve, {n_show_per_procedure} RNYGB)",
        max_sample_lines, y_limits=(15.0, 90.0),
    )
    ev.plot_timecourse_factual_counterfactual(
        dataset, selected_idx, factual_samples, counterfactual_samples, "hba1c", "HbA1c",
        output_dir / "eval_hba1c_factual_counterfactual_examples_test.png",
        f"Test HbA1c factual/counterfactual ({n_show_per_procedure} sleeve, {n_show_per_procedure} RNYGB)",
        max_sample_lines, y_limits=(3.0, 15.0),
    )
    return {
        "selected_subject_ids": dataset.subject_ids[selected_idx].tolist(),
        "timepoint_metrics": timepoint_metrics.to_dict(orient="records"),
        "counterfactual_welch_ttests_test": ttest_json,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
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

    device = ev.choose_device(device_name)
    cfg, pre, model = load_run(run_dir, device)
    dataset = ev.load_dataset(csv_path)

    splits = mt.make_splits(dataset, cfg)

    mace_summary = evaluate_mace_head(model, dataset, splits, pre, cfg, device, output_dir)

    adapter = FlowAdapter(model, mt.CONT_DIMS, fm.X_DIM).to(device)
    expanded_pre = expand_preprocessing(pre)
    continuous_summary = evaluate_continuous(
        adapter, dataset, splits, expanded_pre, output_dir, n_samples, n_steps, seed,
        n_show_per_procedure, max_sample_lines, metric_batch_size, alpha, device, show_progress,
    )

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "csv_path": None if csv_path is None else str(csv_path),
        "device": str(device),
        "model": "flow_matching_multitask",
        "split_strategy": cfg.split_strategy,
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "cont_names": pre.cont_names,
        "n_samples": int(n_samples),
        "n_steps": int(n_steps),
        "seed": int(seed),
        "mace_classification_test": mace_summary,
        **continuous_summary,
    }
    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ev.report_saved(summary_path, "evaluation summary")

    print(f"Run: {run_dir}")
    print(f"Split sizes: {summary['split_sizes']}")
    print("MACE classification (test):")
    print(pd.DataFrame(mace_summary["metrics"]).to_string(index=False))
    print("Continuous timepoint metrics (test):")
    print(ev.format_metric_table(pd.DataFrame(continuous_summary["timepoint_metrics"])))
    print(f"Saved outputs to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="Multi-task run dir, study dir, or best_model dir.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-show-per-procedure", type=int, default=ev.N_SHOW_PER_PROCEDURE)
    parser.add_argument("--max-sample-lines", type=int, default=ev.MAX_SAMPLE_LINES)
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if scipy_stats is None:
        raise SystemExit("scipy is required for Welch t-test evaluation. Install scipy in this environment.")

    run_dir = ev.resolve_run_dir(args.run, args.log_dir)
    evaluate_run(
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
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
