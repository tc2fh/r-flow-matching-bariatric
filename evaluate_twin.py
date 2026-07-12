"""Monolithic evaluator for the modular digital twin (command 3 of 3).

ONE self-contained script (deliberately monolithic for cluster portability -- it
may import existing eval helpers since the whole repo moves together) that emits
EVERY plot/table for the twin, with explicit model -> results tagging in the
filenames:

  GBM  (owns composite-event risk)
    eval_gbm_probability_histogram.png     risk score, stratified by true event
    eval_gbm_calibration_curve.png         reliability curve + Brier
    eval_gbm_curves_test.png               ROC / PR / reliability
    eval_gbm_discrimination_test.csv/.png  AUROC/AUPRC/Brier (+ bootstrap 95% CIs)
    eval_gbm_per_component_discrimination.csv/.png
                                           composite vs MACE-only / nephropathy /
                                           retinopathy, each with bootstrap CIs
    eval_gbm_auroc_delong.csv              DeLong 95% CI + p for AUROC deltas

  FLOW  (owns BMI/HbA1c trajectories, conditioned on the event)
    eval_flow_bmi_factual_counterfactual_examples_test.png
    eval_flow_hba1c_factual_counterfactual_examples_test.png
    eval_flow_timepoint_metrics_test.csv/.png   (Mode-A true-event MAD/RMSE)

  SIMULATOR  (the joint GBM -> Bernoulli -> flow checks; Modes A/B/C)
    eval_sim_modeA_vs_modeB_metrics.csv/.png    oracle vs deployable point pred
    eval_sim_event_marginal.csv/.png            sim prevalence vs observed 13.3%
    eval_sim_trajectory_marginals.csv/.png      per-timepoint KS + quantiles
    eval_sim_event_stratified_contrast.csv/.png sim vs data event contrast
    eval_sim_surgery_counterfactual.csv/.png    flip surgery: risk & traj coherence
    eval_twin_summary.json

Point it at a pipeline dir from ``train_twin_pipeline.py`` (reads ``manifest.json``
for the GBM run, the twin run, and the shared split), or pass ``--gbm-run`` and
``--twin-run`` explicitly::

    python evaluate_twin.py --pipeline runs/twin_pipeline/pipeline_<ts> \
        --csv fake_data/fake_mbs_cohort.csv

The GBM is refit deterministically from its saved config on the shared TRAIN split
(leak-free) and isotonically calibrated on val -- calibration is a *simulation*
requirement, not a nicety, because the event marginal is only correct if the GBM
is calibrated (see MACE_MODELING_DECISIONS.md).
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
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

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import train_flow_matching as fm
import train_flow_matching_multitask as mt
import train_flow_matching_twin as tw
import gbm_mace_baseline as gb
import evaluate_flow_matching as ev
import baselines_trajectory as bt  # W4: per-horizon XGB/Ridge arms + scoring primitives
import calibration_twin as cal  # W5: PIT/coverage/CRPS + split-conformal for trajectory intervals

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


# Per-component labels for the composite endpoint (neuropathy is NOT in the data).
COMPONENT_COLUMNS = {"MACE": "MACE", "nephropathy": "Nephropathy", "retinopathy": "Retinopathy"}
OBSERVED_COMPOSITE_PREVALENCE = 0.133  # test-set composite prevalence on the real cohort

# Where train_twin_pipeline.py drops its pipelines; a bare run auto-discovers the
# newest complete one so no args are needed on the cluster.
DEFAULT_PIPELINE_ROOT = fm.REPO_ROOT / "runs" / "twin_pipeline"


def find_latest_pipeline(root: Path = DEFAULT_PIPELINE_ROOT) -> Path | None:
    """Newest ``pipeline_*`` dir under ``root`` that has a manifest, else None."""
    if not root.exists():
        return None
    candidates = [p for p in root.glob("pipeline_*") if (p / "manifest.json").exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_from_pipeline(pipeline_dir: Path) -> dict:
    manifest = json.loads((pipeline_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest


def load_gbm_config(gbm_run_dir: Path) -> gb.GBMConfig:
    raw = json.loads((gbm_run_dir / "config.json").read_text(encoding="utf-8"))
    valid = {f.name for f in fields(gb.GBMConfig)}
    return gb.GBMConfig(**{k: v for k, v in raw.items() if k in valid})


def load_twin_config(twin_run_dir: Path) -> tw.TwinConfig:
    raw = json.loads((twin_run_dir / "config.json").read_text(encoding="utf-8"))
    valid = {f.name for f in fields(tw.TwinConfig)}
    return tw.TwinConfig(**{k: v for k, v in raw.items() if k in valid})


def load_twin_preprocessing(twin_run_dir: Path) -> mt.Preprocessing:
    raw = json.loads((twin_run_dir / "preprocessing.json").read_text(encoding="utf-8"))
    return mt.Preprocessing(
        target_mean=np.asarray(raw["target_mean"], dtype=np.float32),
        target_std=np.asarray(raw["target_std"], dtype=np.float32),
        static_mean=np.asarray(raw["static_mean"], dtype=np.float32),
        static_std=np.asarray(raw["static_std"], dtype=np.float32),
        static_continuous_idx=np.asarray(raw["static_continuous_idx"], dtype=np.int64),
        patient_feature_names=list(raw["patient_feature_names"]),
        cont_names=list(raw["cont_names"]),
        conformal=raw.get("conformal"),  # W5: None for pre-W5 runs (backward compatible)
    )


def restore_twin(twin_run_dir: Path, cfg: tw.TwinConfig, device: torch.device) -> tw.TwinNet:
    model = tw.TwinNet(cfg, tw.X_CONT_DIM, len(fm.PATIENT_FEATURES)).to(device)
    try:
        state = torch.load(twin_run_dir / "model.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(twin_run_dir / "model.pt", map_location=device)
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


# --------------------------------------------------------------------------- #
# Statistics: bootstrap CIs + DeLong for AUROC deltas
# --------------------------------------------------------------------------- #
def bootstrap_ci(y: np.ndarray, prob: np.ndarray, metric_fn, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile bootstrap CI for a discrimination/calibration metric.

    Returns (point, lo, hi); (point, nan, nan) when the sample is too small or
    single-class to resample meaningfully.
    """
    y = np.asarray(y)
    prob = np.asarray(prob)
    point = gb.safe(metric_fn, y, prob)
    n = y.size
    if n < 3 or np.unique(y).size < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        value = gb.safe(metric_fn, y[idx], prob[idx])
        if np.isfinite(value):
            stats.append(value)
    if len(stats) < 20:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranked = x[order]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = T
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """Fast DeLong (Sun & Xu 2014). ``preds_sorted`` is [k, n] with the m positive
    cases first. Returns (aucs[k], covariance[k, k])."""
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    positive = preds_sorted[:, :m]
    negative = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_auroc_delta(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """DeLong test for the AUROC delta between two correlated scores on the SAME
    labels. Returns aucs, delta, se, z, two-sided p, and a 95% CI for the delta.
    """
    y_true = np.asarray(y_true).astype(int)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    result = {"auc_a": float("nan"), "auc_b": float("nan"), "delta": float("nan"),
              "se": float("nan"), "z": float("nan"), "p_value": float("nan"),
              "ci_lo": float("nan"), "ci_hi": float("nan"), "n_pos": n_pos, "n_neg": n_neg}
    if n_pos < 2 or n_neg < 2:
        return result
    order = np.argsort(-y_true, kind="mergesort")  # positives (label 1) first
    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, cov = _fast_delong(preds, n_pos)
    delta = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(var)) if var > 0 else 0.0
    result.update(auc_a=float(aucs[0]), auc_b=float(aucs[1]), delta=delta, se=se)
    if se > 0:
        z = delta / se
        result["z"] = float(z)
        if scipy_stats is not None:
            result["p_value"] = float(2 * scipy_stats.norm.sf(abs(z)))
        result["ci_lo"] = float(delta - 1.96 * se)
        result["ci_hi"] = float(delta + 1.96 * se)
    elif delta == 0.0:
        result["p_value"] = 1.0
        result["ci_lo"] = result["ci_hi"] = 0.0
    return result


# --------------------------------------------------------------------------- #
# GBM predictions (leak-free: refit on the shared TRAIN split, calibrate on val)
# --------------------------------------------------------------------------- #
def compute_gbm_predictions(gbm_cfg: gb.GBMConfig, dataset: fm.FlowDataset, splits: dict[str, np.ndarray]) -> dict:
    """Deterministically refit the unweighted GBM on train, calibrate on val, and
    return factual + surgery-counterfactual, raw + calibrated probabilities for
    every split. Calibrated test probs are the twin's event marginal p_GBM(x)."""
    x, feature_names, y = gb.assemble_features(dataset)
    surgery_col = feature_names.index("surgery_idx")
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    n_pos = int(y[train_idx].sum())
    n_neg = int((y[train_idx] == 0).sum())
    backend, estimator = gb.make_estimator(gbm_cfg, balanced=False, n_pos=n_pos, n_neg=n_neg)
    estimator.fit(x[train_idx], y[train_idx])

    def proba(matrix: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return estimator.predict_proba(matrix[idx])[:, 1] if idx.size else np.zeros(0)

    x_cf = x.copy()
    x_cf[:, surgery_col] = 1.0 - x_cf[:, surgery_col]

    raw = {split: proba(x, idx) for split, idx in splits.items()}
    raw_cf_test = proba(x_cf, test_idx)

    calibrated = False
    iso = None
    if gbm_cfg.recalibrate and val_idx.size >= 10 and np.unique(y[val_idx]).size == 2:
        try:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw["val"], y[val_idx])
            calibrated = True
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Isotonic recalibration skipped: {exc}", stacklevel=2)

    def cal(p: np.ndarray) -> np.ndarray:
        return iso.transform(p) if (calibrated and iso is not None and p.size) else p

    return {
        "feature_names": feature_names,
        "backend": backend,
        "estimator": estimator,
        "y": y,
        "calibrated": calibrated,
        "test_raw": raw["test"],
        "test_cal": cal(raw["test"]),
        "test_cf_raw": raw_cf_test,
        "test_cf_cal": cal(raw_cf_test),
        "val_raw": raw["val"],
    }


# --------------------------------------------------------------------------- #
# Twin sampling helpers (17-dim, original units, so ev plotting machinery works)
# --------------------------------------------------------------------------- #
def arrays_for(dataset: fm.FlowDataset, idx: np.ndarray, pre: mt.Preprocessing) -> dict:
    return {
        "patient_features": mt.transform_patient_features(dataset.patient_features_raw[idx], pre),
        "surgery_idx": dataset.surgery_idx[idx].astype(np.int64),
        "y_mace": dataset.x[idx, tw.MACE_DIM].astype(np.float32),
    }


def twin_samples_15(model, arrays: dict, event: np.ndarray, cfg: tw.TwinConfig, pre: mt.Preprocessing,
                    device: torch.device, flip_surgery: bool = False,
                    initial_noise: np.ndarray | torch.Tensor | None = None,
                    solver: str | None = None, bound_output: bool = False) -> np.ndarray:
    """[n, n_samples, 15] BMI/HbA1c samples in ORIGINAL units, conditioned on event."""
    if flip_surgery:
        arrays = {**arrays, "surgery_idx": (1 - arrays["surgery_idx"]).astype(np.int64)}
    std = tw.sample_trajectories(
        model, arrays, cfg, device, tw.X_CONT_DIM, event=event,
        initial_noise=initial_noise, solver=solver,
    )
    original = mt.unstandardize(std, pre)
    if bound_output:
        # Presentation safety only. Raw samples remain the source for calibration,
        # scoring, and counterfactual diagnostics so clipping can never hide failure.
        for dim, group in enumerate(tw.CONT_GROUPS):
            lo, hi = bt.PHYSIOLOGIC_BOUNDS[group]
            original[:, :, dim] = np.clip(original[:, :, dim], lo, hi)
    return original


def scatter_to_full(samples_15: np.ndarray) -> np.ndarray:
    """Place the 15 continuous dims into a full fm.X_DIM array (MACE dims = 0), so
    ev's trajectory plotting/metrics (which index dataset target dims) can drive it."""
    n, s, _ = samples_15.shape
    full = np.zeros((n, s, fm.X_DIM), dtype=np.float32)
    full[:, :, tw.CONT_DIMS] = samples_15
    return full


# --------------------------------------------------------------------------- #
# GBM section
# --------------------------------------------------------------------------- #
def evaluate_gbm(gbm: dict, y_test: np.ndarray, dataset: fm.FlowDataset, test_idx: np.ndarray,
                 output_dir: Path, n_boot: int, seed: int, compare_predictions: Path | None) -> dict:
    prob = gbm["test_cal"] if gbm["calibrated"] else gbm["test_raw"]
    tag = "calibrated" if gbm["calibrated"] else "raw"

    # -- discrimination with bootstrap CIs (raw + calibrated) --
    rows = []
    for name, p in [("raw", gbm["test_raw"]), ("calibrated", gbm["test_cal"])]:
        if name == "calibrated" and not gbm["calibrated"]:
            continue
        auroc = bootstrap_ci(y_test, p, roc_auc_score, n_boot, seed)
        auprc = bootstrap_ci(y_test, p, average_precision_score, n_boot, seed + 1)
        brier = bootstrap_ci(y_test, p, brier_score_loss, n_boot, seed + 2)
        rows.append({
            "score": name, "n": int(y_test.size), "n_pos": int(y_test.sum()),
            "prevalence": float(y_test.mean()) if y_test.size else float("nan"),
            "auroc": auroc[0], "auroc_lo": auroc[1], "auroc_hi": auroc[2],
            "auprc": auprc[0], "auprc_lo": auprc[1], "auprc_hi": auprc[2],
            "brier": brier[0], "brier_lo": brier[1], "brier_hi": brier[2],
        })
    disc = pd.DataFrame(rows)
    disc.to_csv(output_dir / "eval_gbm_discrimination_test.csv", index=False)
    ev.report_saved(output_dir / "eval_gbm_discrimination_test.csv", "GBM discrimination (bootstrap CIs)")
    ev.render_table(disc.round(3), output_dir / "eval_gbm_discrimination_test.png", "GBM composite-MACE discrimination (test)")

    # -- per-component discrimination (composite score vs each component label) --
    comp_rows = [_component_row("composite", y_test, prob, n_boot, seed)]
    for label_name, column in COMPONENT_COLUMNS.items():
        matched = fm.find_compatible_column(list(dataset.frame.columns), column)
        if matched is None:
            continue
        y_comp = fm.binary_event(dataset.frame[matched]).to_numpy(dtype=np.int64)[test_idx]
        comp_rows.append(_component_row(label_name, y_comp, prob, n_boot, seed))
    components = pd.DataFrame(comp_rows)
    components.to_csv(output_dir / "eval_gbm_per_component_discrimination.csv", index=False)
    ev.report_saved(output_dir / "eval_gbm_per_component_discrimination.csv", "per-component discrimination")
    ev.render_table(components.round(3), output_dir / "eval_gbm_per_component_discrimination.png",
                    "Composite risk score vs each component (test)")

    # -- DeLong AUROC deltas --
    delong_rows = []
    if gbm["calibrated"]:
        d = delong_auroc_delta(y_test, gbm["test_raw"], gbm["test_cal"])
        delong_rows.append({"comparison": "GBM_raw - GBM_calibrated", **d})
    if compare_predictions is not None:
        other = _load_compare_predictions(compare_predictions, dataset, test_idx)
        if other is not None:
            d = delong_auroc_delta(y_test, prob, other)
            delong_rows.append({"comparison": f"GBM_{tag} - {compare_predictions.name}", **d})
    if delong_rows:
        delong = pd.DataFrame(delong_rows)
        delong.to_csv(output_dir / "eval_gbm_auroc_delong.csv", index=False)
        ev.report_saved(output_dir / "eval_gbm_auroc_delong.csv", "DeLong AUROC-delta tests")

    # -- plots --
    _plot_probability_histogram(y_test, prob, tag, output_dir / "eval_gbm_probability_histogram.png")
    _plot_calibration_curve(y_test, prob, output_dir / "eval_gbm_calibration_curve.png")
    _plot_gbm_curves(y_test, prob, output_dir / "eval_gbm_curves_test.png")

    return {
        "discrimination": disc.to_dict(orient="records"),
        "per_component": components.to_dict(orient="records"),
        "auroc_delong": delong_rows,
        "calibrated": gbm["calibrated"],
        "score_used": tag,
    }


def _component_row(name: str, y: np.ndarray, prob: np.ndarray, n_boot: int, seed: int) -> dict:
    auroc = bootstrap_ci(y, prob, roc_auc_score, n_boot, seed)
    auprc = bootstrap_ci(y, prob, average_precision_score, n_boot, seed + 1)
    brier = bootstrap_ci(y, prob, brier_score_loss, n_boot, seed + 2)
    return {
        "endpoint": name, "n_pos": int(np.asarray(y).sum()), "prevalence": float(np.asarray(y).mean()) if y.size else float("nan"),
        "auroc": auroc[0], "auroc_lo": auroc[1], "auroc_hi": auroc[2],
        "auprc": auprc[0], "auprc_lo": auprc[1], "auprc_hi": auprc[2],
        "brier": brier[0], "brier_lo": brier[1], "brier_hi": brier[2],
    }


def _load_compare_predictions(path: Path, dataset: fm.FlowDataset, test_idx: np.ndarray) -> np.ndarray | None:
    try:
        df = pd.read_csv(path)
        col = next((c for c in ("prob", "mace_prob", "prob_unweighted", "pred_mean_mace_ever") if c in df.columns), None)
        id_col = next((c for c in ("subject_id", "PatKey") if c in df.columns), None)
        if col is None or id_col is None:
            warnings.warn(f"--compare-predictions {path} missing a prob/subject_id column; skipping DeLong.", stacklevel=2)
            return None
        lookup = dict(zip(df[id_col].astype(str), df[col].astype(float)))
        ids = dataset.subject_ids[test_idx]
        aligned = np.array([lookup.get(str(sid), np.nan) for sid in ids])
        return aligned if np.isfinite(aligned).all() else None
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not load --compare-predictions: {exc}", stacklevel=2)
        return None


def _plot_probability_histogram(y: np.ndarray, prob: np.ndarray, tag: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(prob[y == 0], bins=bins, alpha=0.6, label="no event", color="tab:blue", density=True)
    ax.hist(prob[y == 1], bins=bins, alpha=0.6, label="event", color="tab:red", density=True)
    ax.set(title=f"GBM composite-event risk ({tag}), stratified by outcome",
           xlabel="Predicted risk", ylabel="Density")
    ax.legend()
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    ev.report_saved(path, "GBM probability histogram")


def _plot_calibration_curve(y: np.ndarray, prob: np.ndarray, path: Path) -> None:
    table = gb.reliability_table(y, prob)
    brier = gb.safe(brier_score_loss, y, prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
    if not table.empty:
        ax.plot(table["mean_pred"], table["frac_pos"], marker="o", label="GBM")
    ax.set(title=f"GBM calibration (test, Brier={brier:.3f})", xlabel="Mean predicted risk",
           ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    ev.report_saved(path, "GBM calibration curve")


def _plot_gbm_curves(y: np.ndarray, prob: np.ndarray, path: Path) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve
    prevalence = float(y.mean()) if y.size else float("nan")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    try:
        fpr, tpr, _ = roc_curve(y, prob); axes[0].plot(fpr, tpr)
    except Exception:
        pass
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4); axes[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    try:
        prec, rec, _ = precision_recall_curve(y, prob); axes[1].plot(rec, prec)
    except Exception:
        pass
    axes[1].axhline(prevalence, color="k", ls="--", alpha=0.4)
    axes[1].set(title=f"PR (test, prevalence={prevalence:.3f})", xlabel="Recall", ylabel="Precision")
    table = gb.reliability_table(y, prob)
    if not table.empty:
        axes[2].plot(table["mean_pred"], table["frac_pos"], marker="o")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[2].set(title="Reliability (test)", xlabel="Mean predicted", ylabel="Observed frequency")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    ev.report_saved(path, "GBM ROC/PR/reliability curves")


# --------------------------------------------------------------------------- #
# Flow section: factual + counterfactual trajectory plots (reuse ev machinery)
# --------------------------------------------------------------------------- #
def _persist_conformal(twin_run: Path | None, pre: mt.Preprocessing) -> bool:
    """Re-save the run's preprocessing.json with the fitted conformal calibrator attached
    (additive: only adds the ``conformal`` key), then reload and verify the per-horizon Q
    round-trips through Preprocessing save/load. Returns the round-trip PASS/FAIL bool.

    Writing back into the twin RUN dir is what makes the calibrator "saved with the run":
    the twin persists ``mt.Preprocessing`` (aliased ``tw.Preprocessing``) to
    preprocessing.json, so that file -- not ``fm.Preprocessing`` -- is the container the
    run loads. Any future load_twin_preprocessing() therefore reconstructs the calibrator.
    """
    if twin_run is None:
        return False
    pp_path = Path(twin_run) / "preprocessing.json"
    try:
        pp_path.write_text(json.dumps(pre.to_jsonable(), indent=2), encoding="utf-8")
        reloaded = load_twin_preprocessing(Path(twin_run))
        q_before = cal.conformal_q_array(pre.conformal, pre.cont_names)
        q_after = cal.conformal_q_array(reloaded.conformal, reloaded.cont_names)
        ok = reloaded.conformal is not None and np.allclose(q_before, q_after, equal_nan=True)
        ev.report_saved(pp_path, "preprocessing.json (+conformal calibrator, saved with the run)")
        print(f"  conformal round-trip through Preprocessing save/load: {'PASS' if ok else 'FAIL'}")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not persist conformal calibrator into {pp_path}: {exc}", stacklevel=2)
        return False


def evaluate_flow(model, dataset: fm.FlowDataset, splits: dict, pre: mt.Preprocessing, twin_cfg: tw.TwinConfig,
                  twin_run: Path | None, output_dir: Path, n_samples: int, n_steps: int, seed: int, n_show: int,
                  max_lines: int, device: torch.device) -> dict:
    test_idx = splits["test"]
    sample_cfg = replace(twin_cfg, n_samples_per_patient=n_samples, sample_steps=n_steps)

    # Display patients: factual (true event, true surgery) vs surgery-flip counterfactual.
    selected = ev.select_display_patients(dataset, test_idx, np.random.default_rng(seed), n_show)
    sel_arrays = arrays_for(dataset, selected, pre)
    true_event_sel = sel_arrays["y_mace"]
    display_noise = np.random.default_rng(seed + 17).standard_normal(
        (selected.size, n_samples, tw.X_CONT_DIM)
    ).astype(np.float32)
    factual = scatter_to_full(twin_samples_15(
        model, sel_arrays, true_event_sel, sample_cfg, pre, device,
        initial_noise=display_noise, bound_output=True,
    ))
    counterfactual = scatter_to_full(
        twin_samples_15(
            model, sel_arrays, true_event_sel, sample_cfg, pre, device,
            flip_surgery=True, initial_noise=display_noise, bound_output=True,
        )
    )
    ev.selected_patient_frame(dataset, selected).to_csv(output_dir / "eval_flow_selected_test_patients.csv", index=False)
    ev.report_saved(output_dir / "eval_flow_selected_test_patients.csv", "selected display patients")

    # Mode-A timepoint metrics over the full test set (true-event conditioning).
    test_arrays = arrays_for(dataset, test_idx, pre)
    full = scatter_to_full(twin_samples_15(model, test_arrays, test_arrays["y_mace"], sample_cfg, pre, device))

    # --- W5: trajectory-distribution calibration, computed from the FULL predictive
    # samples BEFORE the median collapse below. Keep the 15 continuous dims; score PIT /
    # coverage / CRPS against the observed targets, diagnose each horizon's PIT regime,
    # then fit split-conformal on val and apply to test. We sample val under the SAME
    # Mode-A (true-event) conditioning so the conformal residuals match the test band.
    cont_test = full[:, :, tw.CONT_DIMS]  # (n_test, n_samples, 15) original units
    split_data = mt.split_arrays(dataset, splits, pre)
    test_split, val_split = split_data["test"], split_data["val"]
    val_idx = splits["val"]
    val_arrays = arrays_for(dataset, val_idx, pre)
    cont_val = twin_samples_15(model, val_arrays, val_arrays["y_mace"], sample_cfg, pre, device)
    calibration = cal.calibrate_flow_predictions(
        cont_test=cont_test, obs_test=test_split["original_x"], mask_test=test_split["original_mask"],
        cont_val=cont_val, obs_val=val_split["original_x"], mask_val=val_split["original_mask"],
        cont_names=list(tw.CONT_NAMES), cont_groups=list(tw.CONT_GROUPS),
        subject_ids_test=test_split["subject_ids"], output_dir=output_dir, report_saved=ev.report_saved,
    )
    # Attach the fitted calibrator to Preprocessing and re-save it WITH the run so it is
    # reloaded on every future load; prove it round-trips through save/load.
    pre.conformal = calibration["conformal"]
    roundtrip_ok = _persist_conformal(twin_run, pre)

    # W6 Figure spec A: the per-patient trajectory figures, EXTENDED from 3 to 5 columns
    # (columns 4-5 add per-timepoint P(threshold) under factual & counterfactual surgery)
    # and written as .pdf + .svg + .png. Rendered HERE (after calibration) so non-calibrated
    # horizons can be flagged from the PIT regime just diagnosed above.
    _name_to_month = {item[0]: float(item[2]) for item in fm.TARGET_SPECS}

    def _trust_months(grp: str) -> set[float]:
        return {_name_to_month[n] for n, d in calibration["regime_by_horizon"].items()
                if d.get("group") == grp and d.get("regime") == "calibrated" and n in _name_to_month}

    ev.plot_timecourse_factual_counterfactual(
        dataset, selected, factual, counterfactual, "bmi", "BMI",
        output_dir / "eval_flow_bmi_factual_counterfactual_examples_test.png",
        f"Twin BMI factual vs surgery-counterfactual ({n_show}/arm; +P(BMI<35) per timepoint)",
        max_lines, y_limits=(15.0, 90.0),
        threshold=35.0, threshold_label="BMI < 35", trustworthy_months=_trust_months("bmi"),
        vector_stem=output_dir / "eval_flow_bmi_factual_counterfactual_examples_test",
    )
    ev.plot_timecourse_factual_counterfactual(
        dataset, selected, factual, counterfactual, "hba1c", "HbA1c",
        output_dir / "eval_flow_hba1c_factual_counterfactual_examples_test.png",
        f"Twin HbA1c factual vs surgery-counterfactual ({n_show}/arm; +P(HbA1c<5.7) per timepoint)",
        max_lines, y_limits=(3.0, 15.0),
        threshold=5.7, threshold_label="HbA1c < 5.7", trustworthy_months=_trust_months("hba1c"),
        vector_stem=output_dir / "eval_flow_hba1c_factual_counterfactual_examples_test",
    )

    point_predictions = np.median(full, axis=1)
    timepoint = ev.timepoint_metric_table(dataset, test_idx, point_predictions)
    n_excl = int(timepoint["n_excluded"].sum()) if "n_excluded" in timepoint else 0
    if n_excl:
        print(f"[eval] physiologic QC excluded {n_excl} implausible observed value(s) "
              f"from Mode-A MAD/RMSE (bounds {bt.PHYSIOLOGIC_BOUNDS})")
    timepoint.to_csv(output_dir / "eval_flow_timepoint_metrics_test.csv", index=False)
    ev.report_saved(output_dir / "eval_flow_timepoint_metrics_test.csv", "flow Mode-A timepoint metrics")
    ev.render_table(timepoint.round(3), output_dir / "eval_flow_timepoint_metrics_test.png",
                    "Flow Mode-A (true-event) timepoint MAD/RMSE (test)")
    return {"selected_subject_ids": dataset.subject_ids[selected].tolist(),
            "timepoint_metrics_modeA": timepoint.to_dict(orient="records"),
            "calibration": {"regime_by_horizon": calibration["regime_by_horizon"],
                            "coverage": calibration["coverage"], "crps": calibration["crps"],
                            "pit": calibration["pit"], "conformal": calibration["conformal"],
                            "conformal_roundtrip_ok": roundtrip_ok, "artifacts": calibration["artifacts"]}}


# --------------------------------------------------------------------------- #
# Simulator: Modes A / B / C (the joint GBM -> Bernoulli -> flow checks)
# --------------------------------------------------------------------------- #
def cont_observed(test_arrays_original_x: np.ndarray, mask: np.ndarray, dim: int) -> np.ndarray:
    sel = mask[:, dim] == 1
    return test_arrays_original_x[sel, dim]


def evaluate_simulator(model, dataset: fm.FlowDataset, splits: dict, pre: mt.Preprocessing, twin_cfg: tw.TwinConfig,
                       gbm: dict, output_dir: Path, n_samples: int, n_steps: int, seed: int,
                       device: torch.device, pit_regimes: dict | None = None) -> dict:
    test_idx = splits["test"]
    sample_cfg = replace(twin_cfg, n_samples_per_patient=n_samples, sample_steps=n_steps)
    arrays = mt.split_arrays(dataset, splits, pre)["test"]
    y_test = arrays["y_mace"].astype(np.int64)
    original_x, mask = arrays["original_x"], arrays["original_mask"]
    p = gbm["test_cal"] if gbm["calibrated"] else gbm["test_raw"]
    test_arrays = arrays_for(dataset, test_idx, pre)

    summary: dict[str, Any] = {}

    # --- Mode B: deployable risk-weighted point prediction vs Mode A oracle ---
    mu1 = twin_samples_15(model, test_arrays, np.ones(test_idx.size), sample_cfg, pre, device).mean(axis=1)
    mu0 = twin_samples_15(model, test_arrays, np.zeros(test_idx.size), sample_cfg, pre, device).mean(axis=1)
    mu_true = twin_samples_15(model, test_arrays, test_arrays["y_mace"], sample_cfg, pre, device).mean(axis=1)
    yhat_B = p[:, None] * mu1 + (1.0 - p)[:, None] * mu0
    ab_rows = []
    for group in ["overall", "bmi", "hba1c"]:
        dims = _group_dims(group)
        obs = mask[:, dims] == 1
        n_obs = int(obs.sum())
        mad_A = _masked_mad(mu_true[:, dims], original_x[:, dims], obs)
        mad_B = _masked_mad(yhat_B[:, dims], original_x[:, dims], obs)
        ab_rows.append({"group": group, "n_observed": n_obs, "modeA_true_event_mad": mad_A,
                        "modeB_risk_weighted_mad": mad_B, "A_to_B_gap": mad_B - mad_A})
    ab = pd.DataFrame(ab_rows)
    ab.to_csv(output_dir / "eval_sim_modeA_vs_modeB_metrics.csv", index=False)
    ev.report_saved(output_dir / "eval_sim_modeA_vs_modeB_metrics.csv", "Mode A (oracle) vs Mode B (deployable)")
    ev.render_table(ab.round(3), output_dir / "eval_sim_modeA_vs_modeB_metrics.png",
                    "Mode A (true event) vs Mode B (risk-weighted) MAD")
    summary["modeA_vs_modeB"] = ab.to_dict(orient="records")

    # --- Mode C: full twin simulation (draw event ~ Bernoulli(p)) ---
    rng = np.random.default_rng(seed)
    drawn = (rng.random(test_idx.size) < p).astype(np.int64)
    sim = twin_samples_15(model, test_arrays, drawn, sample_cfg, pre, device)  # [n, s, 15]

    # (C1) event marginal
    marg = pd.DataFrame([{
        "observed_prevalence": float(y_test.mean()) if y_test.size else float("nan"),
        "mean_gbm_risk": float(np.mean(p)) if p.size else float("nan"),
        "simulated_prevalence": float(drawn.mean()) if drawn.size else float("nan"),
        "reference_real_cohort_prevalence": OBSERVED_COMPOSITE_PREVALENCE,
        "n_test": int(test_idx.size),
    }])
    marg.to_csv(output_dir / "eval_sim_event_marginal.csv", index=False)
    ev.report_saved(output_dir / "eval_sim_event_marginal.csv", "Mode C event marginal")
    _plot_event_marginal(y_test, p, drawn, output_dir / "eval_sim_event_marginal.png")
    summary["event_marginal"] = marg.to_dict(orient="records")[0]

    # (C2) trajectory marginals: per-timepoint sim vs observed (KS + quantiles).
    # W5 cross-reference: a Mode-C marginal mismatch (KS rejects, e.g. BMI biased high)
    # and a LOCATION-SHIFTED PIT at the same horizon are the SAME sampler bias, not two
    # findings. We surface the PIT regime here and flag the shared root cause so nobody
    # "calibrates over" it -- conformal widening cannot fix a biased center; the sampler
    # must be fixed. (The PIT is Mode-A/true-event; Mode-C draws the event, but both ride
    # the same conditional flow, so a biased conditional shows up in both.)
    pit_regimes = pit_regimes or {}
    marg_rows = []
    shared_root_cause_dims = []
    for dim, name in enumerate(tw.CONT_NAMES):
        observed = cont_observed(original_x, mask, dim)
        simulated = sim[:, :, dim].reshape(-1)
        ks_stat, ks_p = _ks(simulated, observed)
        reg = pit_regimes.get(name, {})
        regime = reg.get("regime", "unknown")
        ks_rejects = bool(np.isfinite(ks_p) and ks_p < 0.05)
        same_root_cause = bool(ks_rejects and regime == "location-shift")
        if same_root_cause:
            shared_root_cause_dims.append(name)
        marg_rows.append({
            "outcome": name, "n_observed": int(observed.size),
            "obs_p10": _q(observed, 0.10), "sim_p10": _q(simulated, 0.10),
            "obs_p50": _q(observed, 0.50), "sim_p50": _q(simulated, 0.50),
            "obs_p90": _q(observed, 0.90), "sim_p90": _q(simulated, 0.90),
            "ks_stat": ks_stat, "ks_p": ks_p,
            "pit_regime": regime, "pit_mean": reg.get("mean_pit", float("nan")),
            "same_root_cause_as_pit": same_root_cause,
        })
    traj_marg = pd.DataFrame(marg_rows)
    traj_marg.to_csv(output_dir / "eval_sim_trajectory_marginals.csv", index=False)
    ev.report_saved(output_dir / "eval_sim_trajectory_marginals.csv", "Mode C trajectory marginals (KS/quantiles + PIT cross-ref)")
    ev.render_table(traj_marg.round(3), output_dir / "eval_sim_trajectory_marginals.png",
                    "Simulated vs observed per-timepoint marginals (KS/quantiles + PIT regime)")
    if shared_root_cause_dims:
        print("  Mode-C <-> PIT cross-reference: KS mismatch AND location-shifted PIT (SAME sampler bias, "
              f"do NOT calibrate over it) at: {shared_root_cause_dims}")
    summary["trajectory_marginals"] = traj_marg.to_dict(orient="records")
    summary["mode_c_pit_crossref"] = {
        "shared_root_cause_dims": shared_root_cause_dims,
        "note": ("Horizons where the Mode-C marginal KS rejects (p<0.05) and the Mode-A PIT is location-shifted "
                 "share one root cause: a biased flow sampler. Conformal will not fix it; retrain/debias the sampler."),
    }

    # (C3) event-stratified contrast: sim (drawn event) vs data (true event)
    sim_mean_by_event = {e: sim[drawn == e].reshape(-1, 15).mean(axis=0) if (drawn == e).any() else np.full(15, np.nan)
                         for e in (0, 1)}
    contrast_rows = []
    for dim, name in enumerate(tw.CONT_NAMES):
        obs_dim = original_x[:, dim]; m = mask[:, dim] == 1
        data1 = obs_dim[(y_test == 1) & m]; data0 = obs_dim[(y_test == 0) & m]
        data_contrast = (_finite_mean(data1) - _finite_mean(data0))
        sim_contrast = float(sim_mean_by_event[1][dim] - sim_mean_by_event[0][dim])
        contrast_rows.append({"outcome": name, "data_event1_minus_event0": data_contrast,
                              "sim_event1_minus_event0": sim_contrast,
                              "abs_gap": abs(sim_contrast - data_contrast) if np.isfinite(data_contrast) else float("nan")})
    contrast = pd.DataFrame(contrast_rows)
    contrast.to_csv(output_dir / "eval_sim_event_stratified_contrast.csv", index=False)
    ev.report_saved(output_dir / "eval_sim_event_stratified_contrast.csv", "Mode C event-stratified contrast")
    ev.render_table(contrast.round(3), output_dir / "eval_sim_event_stratified_contrast.png",
                    "Event-stratified trajectory contrast: sim vs data")
    summary["event_stratified_contrast"] = contrast.to_dict(orient="records")

    # (C4) surgery counterfactual coherence: flip surgery -> risk delta & traj delta
    p_cf = gbm["test_cf_cal"] if gbm["calibrated"] else gbm["test_cf_raw"]
    risk_delta = p_cf - p
    # Use common random numbers so the contrast isolates the surgery clamp instead
    # of subtracting two independent Monte Carlo samples.
    cf_noise = np.random.default_rng(seed + 4104).standard_normal(
        (test_idx.size, n_samples, tw.X_CONT_DIM)
    ).astype(np.float32)
    mu_fac = twin_samples_15(
        model, test_arrays, test_arrays["y_mace"], sample_cfg, pre, device,
        initial_noise=cf_noise,
    ).mean(axis=1)
    mu_cf = twin_samples_15(
        model, test_arrays, test_arrays["y_mace"], sample_cfg, pre, device,
        flip_surgery=True, initial_noise=cf_noise,
    ).mean(axis=1)
    bmi_dims = _group_dims("bmi")
    traj_delta_bmi = (mu_cf[:, bmi_dims] - mu_fac[:, bmi_dims]).mean(axis=1)  # mean BMI shift per patient
    coherence = float("nan")
    if scipy_stats is not None and risk_delta.size >= 3 and np.std(risk_delta) > 0 and np.std(traj_delta_bmi) > 0:
        coherence = float(scipy_stats.pearsonr(risk_delta, traj_delta_bmi)[0])
    cf = pd.DataFrame([{
        "median_risk_delta_cf_minus_factual": float(np.median(risk_delta)) if risk_delta.size else float("nan"),
        "median_bmi_delta_cf_minus_factual": float(np.median(traj_delta_bmi)) if traj_delta_bmi.size else float("nan"),
        "pearson_risk_delta_vs_bmi_delta": coherence,
        "n_test": int(test_idx.size),
    }])
    cf.to_csv(output_dir / "eval_sim_surgery_counterfactual.csv", index=False)
    ev.report_saved(output_dir / "eval_sim_surgery_counterfactual.csv", "Mode C surgery counterfactual coherence")
    _plot_surgery_counterfactual(risk_delta, traj_delta_bmi, output_dir / "eval_sim_surgery_counterfactual.png")
    summary["surgery_counterfactual"] = cf.to_dict(orient="records")[0]

    # Detailed paired-noise, support, physiological-bound, and solver audit. This is
    # guarded so a diagnostic failure cannot discard the core evaluator artifacts.
    try:
        import counterfactual_audit
        summary["counterfactual_audit"] = counterfactual_audit.run(
            dataset=dataset, splits=splits, pre=pre, model=model, twin_cfg=twin_cfg,
            device=device, output_dir=output_dir, prefix="eval_counterfactual",
            n_samples=min(n_samples, 64), n_steps=n_steps, seed=seed,
        )
        print("  [saved] paired counterfactual safety dashboard + audit tables")
    except Exception as exc:  # noqa: BLE001 - diagnostic must not lose the evaluation
        import traceback
        traceback.print_exc()
        warnings.warn(
            f"counterfactual safety audit FAILED ({type(exc).__name__}: {exc})",
            stacklevel=2,
        )
        summary["counterfactual_audit"] = {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}"
        }

    return summary


def _group_dims(group: str) -> np.ndarray:
    if group == "overall":
        return np.arange(tw.X_CONT_DIM)
    return np.asarray([i for i, g in enumerate(tw.CONT_GROUPS) if g == group], dtype=np.int64)


def _masked_mad(pred: np.ndarray, obs: np.ndarray, mask_bool: np.ndarray) -> float:
    if not mask_bool.any():
        return float("nan")
    return float(np.mean(np.abs(pred[mask_bool] - obs[mask_bool])))


def _finite_mean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _q(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if a.size else float("nan")


def _ks(sim: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    sim = np.asarray(sim, dtype=float); sim = sim[np.isfinite(sim)]
    obs = np.asarray(obs, dtype=float); obs = obs[np.isfinite(obs)]
    if scipy_stats is None or sim.size < 2 or obs.size < 2:
        return float("nan"), float("nan")
    try:
        res = scipy_stats.ks_2samp(sim, obs)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def _plot_event_marginal(y: np.ndarray, p: np.ndarray, drawn: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["observed\nprevalence", "mean GBM\nrisk", "simulated\nprevalence"]
    values = [float(y.mean()) if y.size else np.nan, float(np.mean(p)) if p.size else np.nan,
              float(drawn.mean()) if drawn.size else np.nan]
    axes[0].bar(labels, values, color=["tab:gray", "tab:blue", "tab:red"], alpha=0.8)
    axes[0].axhline(OBSERVED_COMPOSITE_PREVALENCE, color="k", ls="--", alpha=0.5, label=f"real cohort {OBSERVED_COMPOSITE_PREVALENCE}")
    axes[0].set(title="Event marginal", ylabel="Prevalence"); axes[0].legend(fontsize=8)
    table = gb.reliability_table(y, p)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    if not table.empty:
        axes[1].plot(table["mean_pred"], table["frac_pos"], marker="o")
    axes[1].set(title="GBM reliability (drives the marginal)", xlabel="Mean predicted", ylabel="Observed freq",
                xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    ev.report_saved(path, "Mode C event-marginal plot")


def _plot_surgery_counterfactual(risk_delta: np.ndarray, bmi_delta: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(risk_delta, bmi_delta, s=18, alpha=0.6, color="tab:purple")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Surgery counterfactual coherence",
           xlabel="GBM risk delta (flip - factual)", ylabel="Mean BMI delta (flip - factual)")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    ev.report_saved(path, "Mode C surgery-counterfactual plot")


# --------------------------------------------------------------------------- #
# W5: threshold-probability readout (frozen-run artifact feeding the W6 figures)
# --------------------------------------------------------------------------- #
def _plot_threshold_probabilities(summ: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = [(axes[0], "bmi", "P(BMI < 35)"), (axes[1], "hba1c", "P(HbA1c < 5.7)")]
    for ax, group, title in panels:
        sub = summ[summ["group"] == group]
        tps = list(dict.fromkeys(sub["timepoint"]))
        xpos = {tp: i for i, tp in enumerate(tps)}
        for arm, colour in [("factual", "tab:blue"), ("counterfactual", "tab:orange")]:
            s = sub[sub["surgery_arm"] == arm]
            if s.empty:
                continue
            ax.plot([xpos[tp] for tp in s["timepoint"]], s["cohort_p_risk_mean"], marker="o", label=arm, color=colour)
        ax.set_xticks(range(len(tps))); ax.set_xticklabels(tps, rotation=45, ha="right", fontsize=8)
        ax.set(title=f"{title} (cohort mean, risk-weighted)", ylabel="probability", ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    ev.report_saved(path, "threshold-probability curves (factual vs counterfactual)")


def evaluate_threshold_probabilities(model, dataset: fm.FlowDataset, splits: dict, pre: mt.Preprocessing,
                                     twin_cfg: tw.TwinConfig, gbm: dict, output_dir: Path, n_samples: int,
                                     n_steps: int, seed: int, device: torch.device,
                                     pit_regimes: dict | None = None) -> dict:
    """Frozen-run artifact: per-patient x per-timepoint threshold probabilities.

    P(BMI < 35) at BMI 3m..6y and P(HbA1c < 5.7) at HbA1c 12m..6y, for each patient's
    FACTUAL surgery and its surgery COUNTERFACTUAL, risk-weighted over the composite
    event (deployable marginal). Reuses the twin sampling machinery (twin_samples_15 with
    surgery clamped factual / flipped) and the GBM's factual / flipped calibrated risk.
    These feed the W6 figures.

    CALIBRATION-DEPENDENCE (encoded on purpose): a threshold probability is the sample
    tail mass past the cut, so it inherits the flow's tail calibration. It is trustworthy
    only where the PIT/coverage pass calls the horizon calibrated; the summary carries each
    timepoint's pit_regime + a calibration_trustworthy flag so W6 never plots an interval
    the sampler has not earned.
    """
    import bmi_threshold_probability as thr  # lazy: thr imports evaluate_twin (avoid an import cycle)

    test_idx = splits["test"]
    sample_cfg = replace(twin_cfg, n_samples_per_patient=n_samples, sample_steps=n_steps)
    torch.manual_seed(seed); np.random.seed(seed)
    base_arrays = arrays_for(dataset, test_idx, pre)
    subject_ids = dataset.subject_ids[test_idx]
    targets = thr.default_threshold_targets()
    p_fac = gbm["test_cal"] if gbm["calibrated"] else gbm["test_raw"]
    p_cf = gbm["test_cf_cal"] if gbm["calibrated"] else gbm["test_cf_raw"]

    table = thr.threshold_probability_table(model, base_arrays, subject_ids, targets,
                                            p_fac, p_cf, sample_cfg, pre, device)
    per_patient_path = output_dir / "eval_flow_threshold_probabilities.csv"
    table.to_csv(per_patient_path, index=False)
    ev.report_saved(per_patient_path, "per-patient x per-timepoint threshold probabilities (factual/counterfactual)")

    # Cohort summary per timepoint x arm, annotated with the PIT regime + trust flag.
    pit_regimes = pit_regimes or {}
    summ = (table.groupby(["group", "timepoint", "threshold", "direction", "surgery_arm"], sort=False)["p_risk"]
            .agg(["mean", "std", "count"]).reset_index()
            .rename(columns={"mean": "cohort_p_risk_mean", "std": "cohort_p_risk_std", "count": "n_test"}))
    summ["pit_regime"] = summ["timepoint"].map(lambda tp: pit_regimes.get(tp, {}).get("regime", "unknown"))
    summ["calibration_trustworthy"] = summ["pit_regime"] == "calibrated"
    summ["calibration_dependent"] = True
    summary_path = output_dir / "eval_flow_threshold_probabilities_summary.csv"
    summ.to_csv(summary_path, index=False)
    ev.report_saved(summary_path, "threshold-probability cohort summary (+PIT regime / trust flag)")
    ev.render_table(summ.round(3), output_dir / "eval_flow_threshold_probabilities_summary.png",
                    "Threshold probabilities (cohort mean, factual vs counterfactual) + PIT regime")
    _plot_threshold_probabilities(summ, output_dir / "eval_flow_threshold_probabilities.png")

    n_trust = int(summ["calibration_trustworthy"].sum())
    print(f"\nThreshold probabilities: {len(table)} patient-timepoint-arm rows, {len(targets)} timepoints x "
          f"{{factual,counterfactual}} x {{BMI<35,HbA1c<5.7}}. Calibration-dependent: {n_trust}/{len(summ)} "
          f"timepoint-arm rows sit on a 'calibrated' PIT (the rest are reported but flagged not-yet-trustworthy).")
    return {
        "per_patient_csv": str(per_patient_path),
        "summary_csv": str(summary_path),
        "thresholds": {"bmi": thr.BMI_THRESHOLD, "hba1c": thr.HBA1C_THRESHOLD},
        "n_timepoints": len(targets),
        "calibration_dependent": True,
        "note": ("Threshold probabilities inherit the flow's tail calibration; trust only where "
                 "pit_regime=='calibrated' (see eval_flow_calibration_pit.csv)."),
        "summary": summ.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------- #
# W4: trajectory ablation (does event conditioning actually help?) + quantile yard-stick
# --------------------------------------------------------------------------- #
# The arms scored on the held-out test set, all on the same patients/horizons:
#   event_flow    -- the coupled twin (event-conditioned flow), Mode-A true event
#   no_event_flow -- same architecture, event embedding dropped (use_event=False)
#   qgbm / qreg   -- quantile GBM + linear quantile regression (predictive ensembles)
#   xgb / ridge   -- one point regressor per horizon on the SAME conditioning
# This table is what decides whether the coupling claim survives, and whether the flow's
# predictive distribution beats a strong quantile model's.
# qgbm / qreg are the quantile arms (baselines_trajectory.fit_quantile_baselines): unlike the
# point xgb/ridge arms they predict a whole quantile grid, i.e. a real predictive ensemble, so
# they carry a spread (NLL is defined) and are the fair distributional yard-stick for the flow.
TRAJECTORY_ARMS = ("event_flow", "no_event_flow", "qgbm", "qreg", "xgb", "ridge")
DENSITY_ARMS = {"event_flow", "no_event_flow", "qgbm", "qreg"}  # arms with a predictive spread -> NLL
# Paired comparisons written to the table. event_flow is the reference; the first row
# is the decisive coupling test (event-conditioned vs unconditioned flow).
TRAJECTORY_COMPARISONS = (
    ("event_flow", "no_event_flow"),
    ("event_flow", "qgbm"),
    ("event_flow", "qreg"),
    ("event_flow", "xgb"),
    ("event_flow", "ridge"),
    ("no_event_flow", "xgb"),
    ("no_event_flow", "ridge"),
)


def _sample_twin_arm(model, dataset: fm.FlowDataset, test_idx: np.ndarray, pre: mt.Preprocessing,
                     cfg: tw.TwinConfig, n_samples: int, n_steps: int, device: torch.device) -> np.ndarray:
    """[n_test, n_samples, 15] original-unit BMI/HbA1c draws (Mode-A, true event)."""
    arrays = arrays_for(dataset, test_idx, pre)
    sample_cfg = replace(cfg, n_samples_per_patient=n_samples, sample_steps=n_steps)
    return twin_samples_15(model, arrays, arrays["y_mace"], sample_cfg, pre, device)


def _resolve_noevent_arm(dataset: fm.FlowDataset, event_cfg: tw.TwinConfig, noevent_twin_run: Path | None,
                         noevent_num_steps: int | None, output_dir: Path, device: torch.device):
    """Load a no-event twin run, or train the controlled ablation inline.

    Inline training clones the EVENT arm's config (identical architecture/optimiser/
    split) and only flips ``use_event=False``, so the contrast isolates event
    conditioning and nothing else. On the VM, pass a pre-trained (ideally
    independently tuned) no-event run via ``noevent_twin_run`` to skip retraining.
    """
    if noevent_twin_run is not None:
        run_dir = Path(noevent_twin_run)
        cfg = load_twin_config(run_dir)
        if cfg.use_event:
            warnings.warn(f"--noevent-twin-run {run_dir} has use_event=True (expected the no-event arm).", stacklevel=2)
        return restore_twin(run_dir, cfg, device), cfg, run_dir
    steps = noevent_num_steps if noevent_num_steps is not None else event_cfg.num_steps
    cfg = replace(event_cfg, use_event=False, num_steps=steps, output_dir=str(output_dir / "noevent_twin"))
    print(f"Training the no-event ablation arm inline (use_event=False, num_steps={steps}) ...")
    result = tw.train_model(dataset, cfg)
    run_dir = Path(result["run_dir"])
    cfg = load_twin_config(run_dir)  # reload persisted config (use_event=False baked in)
    return restore_twin(run_dir, cfg, device), cfg, run_dir


def _pool(per_patient: dict, arm: str, dims: list[int], key: str) -> np.ndarray:
    if not dims:
        return np.array([], dtype=np.float64)
    return np.concatenate([per_patient[arm][h][key] for h in dims])


def _patient_mean_crps(per_patient: dict, arm: str, dims: list[int]) -> np.ndarray:
    """Mean observed-horizon CRPS per patient for cluster-correct pooled tests."""
    if not dims:
        return np.array([], dtype=np.float64)
    block = np.column_stack([per_patient[arm][h]["crps_by_patient"] for h in dims])
    observed = np.isfinite(block)
    count = observed.sum(axis=1)
    total = np.nansum(block, axis=1)
    return np.divide(total, count, out=np.full(block.shape[0], np.nan), where=count > 0)


def compare_trajectory_models(
    event_twin_run: Path | None,
    csv_path: Path | None,
    output_dir: Path | None = None,
    noevent_twin_run: Path | None = None,
    pipeline: Path | None = None,
    n_samples: int = 200,
    n_steps: int = 50,
    noevent_num_steps: int | None = None,
    seed: int = 0,
    device_name: str = "auto",
    include_baselines: bool = True,
    baseline_use_event: bool = True,
    include_quantile: bool = True,
    quantiles: "tuple[float, ...] | None" = None,
) -> dict:
    """Score the trajectory arms on the shared test split and write the table.

    Arms: the event / no-event flow, the two quantile arms (``qgbm`` = quantile GBM, ``qreg``
    = linear quantile regression; on by default via ``include_quantile``), and the ``xgb`` /
    ``ridge`` point arms. Emits, per horizon (and pooled overall/bmi/hba1c rows), MAD / RMSE /
    CRPS and (for the spread-carrying flow + quantile arms) Gaussian predictive NLL, plus a
    paired-test CSV (Wilcoxon signed-rank + paired t) on per-patient CRPS for every arm pair.
    Everything is aligned patient-for-patient off ``splits['test']``. ``evaluate()`` is untouched.
    """
    if event_twin_run is None:
        if pipeline is None:
            pipeline = find_latest_pipeline()
        if pipeline is None:
            raise SystemExit("Provide --twin-run (the event arm), or --pipeline.")
        manifest = resolve_from_pipeline(Path(pipeline))
        event_twin_run = Path(manifest["twin_final_run_dir"])
        print(f"Event arm resolved from pipeline: {event_twin_run}")
    event_twin_run = Path(event_twin_run)
    output_dir = (event_twin_run / "trajectory_comparison") if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(device_name)
    dataset = load_dataset(csv_path)
    event_cfg = load_twin_config(event_twin_run)
    if not event_cfg.use_event:
        warnings.warn(f"--twin-run {event_twin_run} has use_event=False; expected the event-conditioned arm.", stacklevel=2)

    # One shared split + preprocessing drive every arm (deterministic from the split).
    splits = tw.make_splits(dataset, event_cfg)
    test_idx = splits["test"]
    pre = mt.fit_preprocessing(dataset, splits["train"])
    test_arrays = mt.split_arrays(dataset, splits, pre)["test"]
    obs, mask = test_arrays["original_x"], test_arrays["original_mask"]  # [n, 15] original units

    # Arm 1: event-conditioned flow (the coupled twin).
    event_model = restore_twin(event_twin_run, event_cfg, device)
    arm_samples: dict[str, np.ndarray] = {
        "event_flow": _sample_twin_arm(event_model, dataset, test_idx, pre, event_cfg, n_samples, n_steps, device)
    }
    # Arm 2: no-event (unconditioned) flow.
    noevent_model, noevent_cfg, noevent_run = _resolve_noevent_arm(
        dataset, event_cfg, noevent_twin_run, noevent_num_steps, output_dir, device)
    arm_samples["no_event_flow"] = _sample_twin_arm(
        noevent_model, dataset, test_idx, pre, noevent_cfg, n_samples, n_steps, device)
    # Arms qgbm & qreg: fit nonuniform quantile grids, then interpolate the rearranged quantile
    # functions onto uniform-probability nodes before sample-based CRPS/NLL scoring. Guarded: a
    # quantile-fit failure degrades to the other arms.
    quantile_baselines = None
    if include_quantile:
        q_grid = bt.DEFAULT_QUANTILES if quantiles is None else tuple(quantiles)
        try:
            quantile_baselines = bt.fit_quantile_baselines(
                dataset, splits, quantiles=q_grid, use_event=baseline_use_event, seed=seed)
            arm_samples["qgbm"] = bt.quantile_grid_to_ensemble(
                quantile_baselines["qgbm_pred"], quantile_baselines["quantiles"], n_samples=n_samples)
            arm_samples["qreg"] = bt.quantile_grid_to_ensemble(
                quantile_baselines["qreg_pred"], quantile_baselines["quantiles"], n_samples=n_samples)
        except Exception as exc:  # noqa: BLE001 - never let a baseline sink the flow comparison
            warnings.warn(f"quantile arms skipped ({type(exc).__name__}: {exc})", stacklevel=2)
    # Arms xgb & ridge: per-horizon point regressors (degenerate one-sample "ensembles").
    baselines = None
    if include_baselines:
        baselines = bt.fit_trajectory_baselines(dataset, splits, use_event=baseline_use_event, seed=seed)
        arm_samples["xgb"] = baselines["xgb_pred"][:, None, :]
        arm_samples["ridge"] = baselines["ridge_pred"][:, None, :]
    present = [a for a in TRAJECTORY_ARMS if a in arm_samples]

    cont_names, cont_groups = tw.CONT_NAMES, tw.CONT_GROUPS
    per_patient = {a: {} for a in present}
    metric_rows: list[dict] = []
    for h, name in enumerate(cont_names):
        for arm in present:
            score = bt.horizon_score(arm_samples[arm][:, :, h], obs[:, h], mask[:, h],
                                     has_density=arm in DENSITY_ARMS,
                                     obs_bounds=bt.PHYSIOLOGIC_BOUNDS.get(cont_groups[h]))
            per_patient[arm][h] = score
            metric_rows.append({"horizon": name, "group": cont_groups[h], "arm": arm, "n_obs": score["n_obs"],
                                 "mad": score["mad"], "rmse": score["rmse"], "crps": score["crps"], "nll": score["nll"]})

    group_dims = {"overall": list(range(len(cont_names))),
                  "bmi": [i for i, g in enumerate(cont_groups) if g == "bmi"],
                  "hba1c": [i for i, g in enumerate(cont_groups) if g == "hba1c"]}
    for gname, dims in group_dims.items():
        for arm in present:
            abs_err, sq_err = _pool(per_patient, arm, dims, "abs_err"), _pool(per_patient, arm, dims, "sq_err")
            crps_pp, nll_pp = _pool(per_patient, arm, dims, "crps_pp"), _pool(per_patient, arm, dims, "nll_pp")
            metric_rows.append({"horizon": f"__{gname}__", "group": gname, "arm": arm, "n_obs": int(abs_err.size),
                                "mad": float(np.median(abs_err)) if abs_err.size else float("nan"),
                                "rmse": float(np.sqrt(np.mean(sq_err))) if sq_err.size else float("nan"),
                                "crps": float(np.mean(crps_pp)) if crps_pp.size else float("nan"),
                                "nll": float(np.nanmean(nll_pp)) if np.isfinite(nll_pp).any() else float("nan")})
    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "trajectory_comparison_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    ev.report_saved(metrics_path, "per-horizon trajectory metrics (flow / quantile / point arms)")

    # Paired tests on per-patient CRPS (the proper score defined for every arm).
    paired_rows: list[dict] = []
    for h, name in enumerate(cont_names):
        for a, b in TRAJECTORY_COMPARISONS:
            if a in present and b in present:
                pt = bt.paired_test(per_patient[a][h]["crps_pp"], per_patient[b][h]["crps_pp"])
                paired_rows.append({"horizon": name, "group": cont_groups[h],
                                    "comparison": f"{a}_vs_{b}", "metric": "crps",
                                    "unit": "patient", **pt})
    for gname, dims in group_dims.items():
        for a, b in TRAJECTORY_COMPARISONS:
            if a in present and b in present:
                pt = bt.paired_test(
                    _patient_mean_crps(per_patient, a, dims),
                    _patient_mean_crps(per_patient, b, dims),
                )
                paired_rows.append({"horizon": f"__{gname}__", "group": gname,
                                    "comparison": f"{a}_vs_{b}", "metric": "crps",
                                    "unit": "patient", **pt})
    paired = pd.DataFrame(paired_rows)
    paired_path = output_dir / "trajectory_comparison_paired_tests.csv"
    paired.to_csv(paired_path, index=False)
    ev.report_saved(paired_path, "paired CRPS tests across arms (Wilcoxon + paired t)")

    print("\nTrajectory comparison (pooled group rows, test split):")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(metrics[metrics["horizon"].str.startswith("__")].round(4).to_string(index=False))
        print("\nPaired CRPS tests (overall; mean_diff<0 => first arm better):")
        print(paired[paired["group"] == "overall"].round(4).to_string(index=False))

    summary = {
        "output_dir": str(output_dir),
        "event_twin_run": str(event_twin_run),
        "noevent_twin_run": str(noevent_run),
        "arms": present,
        "n_test": int(test_idx.size),
        "n_samples": n_samples,
        "n_steps": n_steps,
        "baseline_use_event": baseline_use_event,
        "baseline_feature_names": baselines["feature_names"] if baselines else None,
        "quantile_arms": ["qgbm", "qreg"] if quantile_baselines else [],
        "n_quantiles": int(np.asarray(quantile_baselines["quantiles"]).size) if quantile_baselines else 0,
        "n_quantile_ensemble_samples": n_samples if quantile_baselines else 0,
        "metrics_csv": str(metrics_path),
        "paired_tests_csv": str(paired_path),
        "paired_test": "Wilcoxon signed-rank + paired t with patient as the unit; pooled rows average observed horizons within patient",
        "note_mad": "MAD = median absolute deviation (median_i|mean-pred_i - y_i|); Saux et al./SOPHIA definition, robust to outliers",
        "note_nll": "density arms: Gaussian predictive NLL moment-matched from flow draws or uniform-probability quantile nodes (exact CNF NLL deferred to VM)",
        "note_quantile_scoring": "nonuniform fitted quantile levels are interpolated onto equal-probability midpoint nodes before sample-based scoring",
        "note_ridge": "Ridge per-horizon stands in for a linear mixed model (statsmodels not installed here)",
    }
    summary_path = output_dir / "trajectory_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    ev.report_saved(summary_path, "trajectory comparison summary")
    print(f"\nSaved trajectory comparison ({len(present)} arms: {', '.join(present)}) to {output_dir}")
    return summary


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate(pipeline: Path | None, gbm_run: Path | None, twin_run: Path | None, csv_path: Path | None,
             output_dir: Path | None, n_samples: int, n_steps: int, seed: int, n_show: int, max_lines: int,
             n_boot: int, device_name: str, compare_predictions: Path | None,
             with_causal: bool = False, with_fairness: bool = False) -> dict:
    # No explicit run pointers -> auto-discover the newest pipeline (so a bare run
    # works on the cluster right after train_twin_pipeline.py).
    if pipeline is None and gbm_run is None and twin_run is None:
        pipeline = find_latest_pipeline()
        if pipeline is None:
            raise SystemExit(
                f"No pipeline found under {DEFAULT_PIPELINE_ROOT}. Run train_twin_pipeline.py first, "
                "or pass --pipeline / (--gbm-run and --twin-run)."
            )
        print(f"Auto-discovered latest pipeline: {pipeline}")

    if pipeline is not None:
        manifest = resolve_from_pipeline(pipeline)
        gbm_run = Path(manifest["gbm_run_dir"]) if gbm_run is None else gbm_run
        twin_run = Path(manifest["twin_final_run_dir"]) if twin_run is None else twin_run
        default_out = pipeline / "evaluation"
    else:
        if gbm_run is None or twin_run is None:
            raise SystemExit("Provide --pipeline, or BOTH --gbm-run and --twin-run.")
        default_out = Path(twin_run) / "twin_evaluation"
    gbm_run, twin_run = Path(gbm_run), Path(twin_run)
    output_dir = default_out if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(device_name)
    gbm_cfg = load_gbm_config(gbm_run)
    twin_cfg = load_twin_config(twin_run)
    pre = load_twin_preprocessing(twin_run)
    model = restore_twin(twin_run, twin_cfg, device)
    dataset = load_dataset(csv_path)

    # Shared split: assert the two configs agree, then use the twin's split.
    for key in ("split_strategy", "split_seed", "train_frac", "val_frac", "test_frac"):
        gv, tv = getattr(gbm_cfg, key), getattr(twin_cfg, key)
        if gv != tv:
            warnings.warn(f"GBM and twin disagree on {key!r}: GBM={gv} twin={tv} (evaluation uses the twin split).", stacklevel=2)
    splits = tw.make_splits(dataset, twin_cfg)
    test_idx = splits["test"]

    gbm = compute_gbm_predictions(gbm_cfg, dataset, splits)
    y_test = gbm["y"][test_idx].astype(np.int64)

    print(f"Twin evaluation | test n={test_idx.size} prevalence={float(y_test.mean()):.4f} "
          f"| GBM backend={gbm['backend']} calibrated={gbm['calibrated']}")

    gbm_summary = evaluate_gbm(gbm, y_test, dataset, test_idx, output_dir, n_boot, seed, compare_predictions)
    flow_summary = evaluate_flow(model, dataset, splits, pre, twin_cfg, twin_run, output_dir, n_samples, n_steps, seed, n_show, max_lines, device)
    pit_regimes = flow_summary.get("calibration", {}).get("regime_by_horizon")
    sim_summary = evaluate_simulator(model, dataset, splits, pre, twin_cfg, gbm, output_dir, n_samples, n_steps, seed, device,
                                     pit_regimes=pit_regimes)
    threshold_summary = evaluate_threshold_probabilities(model, dataset, splits, pre, twin_cfg, gbm, output_dir,
                                                         n_samples, n_steps, seed, device, pit_regimes)

    # --- Additive causal (target-trial emulation) + distributional + equity-fairness passes.
    #     Guarded so a failure here NEVER loses the core GBM/flow/simulator/threshold artifacts
    #     already written. run_tte / fairness_audit are imported lazily (they import evaluate_twin
    #     at module top, which is already loaded here) to avoid an import cycle. Both reuse the
    #     objects already built above, so nothing is reloaded and the split is identical. run_tte
    #     reads split_strategy from gbm_cfg, so a temporal freeze carries through automatically.
    causal_distributional_summary = None
    if with_causal:
        try:
            import run_tte
            causal_distributional_summary = run_tte.run(
                dataset=dataset, splits=splits, model=model, twin_cfg=twin_cfg, pre=pre,
                gbm=gbm, gbm_cfg=gbm_cfg, output_dir=output_dir, device=device,
                n_samples=n_samples, n_steps=n_steps, seed=seed, with_causal=True,
            )
        except Exception as exc:  # noqa: BLE001 - the causal layer must never lose the eval
            import traceback
            traceback.print_exc()
            warnings.warn(f"causal/distributional evaluation FAILED ({type(exc).__name__}: {exc}); "
                          "core twin evaluation is intact.", stacklevel=2)
            causal_distributional_summary = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    fairness_summary = None
    if with_fairness:
        try:
            import fairness_audit
            fairness_summary = fairness_audit.run(
                dataset=dataset, splits=splits, model=model, twin_cfg=twin_cfg, pre=pre,
                gbm=gbm, gbm_cfg=gbm_cfg, output_dir=output_dir, device=device,
                n_samples=n_samples, n_steps=n_steps, seed=seed, n_boot=n_boot,
            )
        except Exception as exc:  # noqa: BLE001 - the fairness audit must never lose the eval
            import traceback
            traceback.print_exc()
            warnings.warn(f"fairness audit FAILED ({type(exc).__name__}: {exc}); "
                          "core twin evaluation is intact.", stacklevel=2)
            fairness_summary = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    summary = {
        "pipeline_dir": None if pipeline is None else str(pipeline),
        "gbm_run_dir": str(gbm_run),
        "twin_run_dir": str(twin_run),
        "output_dir": str(output_dir),
        "csv_path": None if csv_path is None else str(csv_path),
        "device": str(device),
        "split_strategy": twin_cfg.split_strategy,
        "split_sizes": {k: int(v.size) for k, v in splits.items()},
        "test_prevalence": float(y_test.mean()) if y_test.size else float("nan"),
        "gbm": gbm_summary,
        "flow": flow_summary,
        "simulator": sim_summary,
        "threshold_probabilities": threshold_summary,
        "causal_distributional": causal_distributional_summary,
        "fairness": fairness_summary,
    }
    summary_path = output_dir / "eval_twin_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    ev.report_saved(summary_path, "twin evaluation summary")
    print(f"\nSaved all twin evaluation outputs to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", type=Path, default=None, help="train_twin_pipeline.py pipeline dir (reads manifest.json).")
    parser.add_argument("--gbm-run", type=Path, default=None, help="GBM run dir (if not using --pipeline).")
    parser.add_argument("--twin-run", type=Path, default=None, help="Twin flow run dir (if not using --pipeline).")
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-show-per-procedure", dest="n_show", type=int, default=ev.N_SHOW_PER_PROCEDURE)
    parser.add_argument("--max-sample-lines", dest="max_lines", type=int, default=ev.MAX_SAMPLE_LINES)
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for CIs.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compare-predictions", type=Path, default=None,
                        help="Optional CSV (subject_id, prob) for a DeLong AUROC delta vs the GBM.")
    # W4 trajectory ablation (separate entrypoint; leaves evaluate() intact).
    parser.add_argument("--trajectory-comparison", dest="trajectory_comparison", action="store_true",
                        help="Run the trajectory comparison {event/no-event flow, quantile GBM + regression, "
                             "XGB, Ridge} instead of evaluate().")
    parser.add_argument("--noevent-twin-run", type=Path, default=None,
                        help="Pre-trained no-event twin run dir; if omitted it is trained inline from the event arm's config.")
    parser.add_argument("--noevent-num-steps", type=int, default=None,
                        help="num_steps for the inline no-event arm (defaults to the event arm's).")
    parser.add_argument("--no-baselines", dest="no_baselines", action="store_true",
                        help="Trajectory comparison: flow arms only (skip XGB/Ridge).")
    parser.add_argument("--with-causal", action="store_true",
                        help="Also run the target-trial-emulation + distributional pass (run_tte.run): "
                             "writes tte_* and dist_* artifacts. Off by default here (freeze_run turns it on).")
    parser.add_argument("--with-fairness", action="store_true",
                        help="Also run the SVI/RUCA/race equity-fairness subgroup audit (fairness_audit.run): "
                             "writes fairness_* artifacts. Off by default here (freeze_run turns it on).")
    args = parser.parse_args()

    if args.trajectory_comparison:
        compare_trajectory_models(
            event_twin_run=args.twin_run, csv_path=args.csv_path, output_dir=args.output_dir,
            noevent_twin_run=args.noevent_twin_run, pipeline=args.pipeline, n_samples=args.n_samples,
            n_steps=args.n_steps, noevent_num_steps=args.noevent_num_steps, seed=args.seed,
            device_name=args.device, include_baselines=not args.no_baselines,
        )
        return

    evaluate(
        pipeline=args.pipeline, gbm_run=args.gbm_run, twin_run=args.twin_run, csv_path=args.csv_path,
        output_dir=args.output_dir, n_samples=args.n_samples, n_steps=args.n_steps, seed=args.seed,
        n_show=args.n_show, max_lines=args.max_lines, n_boot=args.n_boot, device_name=args.device,
        compare_predictions=args.compare_predictions,
        with_causal=args.with_causal, with_fairness=args.with_fairness,
    )


if __name__ == "__main__":
    main()
