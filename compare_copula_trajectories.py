"""Coherent BMI/HbA1c trajectories from quantile models, benchmarked against flow matching.

This is intentionally a single transferable script.  It trains the existing quantile GBM and
linear quantile-regression marginals, estimates a Gaussian residual-rank copula on validation
patients, samples exact-marginal independent and copula-coupled trajectories, restores an
optional trained event-conditioned flow, scores every arm, and writes tables plus figures.

The coupling step uses rank permutations of the same equal-probability marginal quantile nodes.
Consequently an independent and a copula arm have exactly the same per-patient marginal samples;
only their cross-horizon dependence differs.  Marginal CRPS/coverage should therefore be
identical up to floating-point arithmetic, while energy/variogram scores can distinguish them.

Examples::

    OMP_NUM_THREADS=1 python compare_copula_trajectories.py \
        --twin-run runs/twin_pipeline/<run> --csv data/cohort.csv

    OMP_NUM_THREADS=1 python compare_copula_trajectories.py \
        --csv fake_data/fake_mbs_cohort.csv --no-flow --self-test

Optional paired sleeve/RYGB samples reuse the same latent ranks in both worlds::

    ... --counterfactual

When event conditioning is enabled, that counterfactual holds the observed event feature fixed.
It is a paired predictive sensitivity analysis, not an identified causal treatment effect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-copula")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import baselines_trajectory as bt
import compare_quantile_baselines as qb
import evaluate_twin as et
import train_flow_matching as fm
import train_flow_matching_multitask as mt
import train_flow_matching_twin as tw


ARM_FLOW = "event_flow"
ARM_QGBM_INDEPENDENT = "qgbm_independent"
ARM_QGBM_COPULA = "qgbm_copula"
ARM_QREG_INDEPENDENT = "qreg_independent"
ARM_QREG_COPULA = "qreg_copula"
ARM_ORDER = [ARM_FLOW, ARM_QGBM_INDEPENDENT, ARM_QGBM_COPULA,
             ARM_QREG_INDEPENDENT, ARM_QREG_COPULA]
ARM_LABELS = {
    ARM_FLOW: "flow matching",
    ARM_QGBM_INDEPENDENT: "qGBM independent",
    ARM_QGBM_COPULA: "qGBM + rank copula",
    ARM_QREG_INDEPENDENT: "qReg independent",
    ARM_QREG_COPULA: "qReg + rank copula",
}
ARM_COLORS = {
    ARM_FLOW: "#1b6ca8",
    ARM_QGBM_INDEPENDENT: "#e6a6ae",
    ARM_QGBM_COPULA: "#d1495b",
    ARM_QREG_INDEPENDENT: "#f5d58c",
    ARM_QREG_COPULA: "#edae49",
}


@dataclass
class MarginalSuite:
    """Fitted horizon models and fallbacks for both quantile model families."""

    quantiles: np.ndarray
    feature_names: list[str]
    use_event: bool
    models: dict[str, list[object | None]]
    fallbacks: dict[str, list[np.ndarray]]
    train_n_per_horizon: list[int]


def fit_marginal_suite(
    dataset: fm.FlowDataset,
    splits: dict[str, np.ndarray],
    quantiles: np.ndarray,
    use_event: bool,
    seed: int,
    gbm_params: dict | None = None,
    qr_alpha: float = 0.0,
) -> MarginalSuite:
    """Train qGBM and qReg once per horizon and retain models for val/test prediction."""
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 1 or q.size < 2 or not np.all(np.diff(q) > 0):
        raise ValueError("quantiles must be a strictly increasing one-dimensional grid")
    features, feature_names = bt.build_feature_matrix(dataset, use_event)
    targets = dataset.x[:, tw.CONT_DIMS].astype(np.float64)
    masks = dataset.mask[:, tw.CONT_DIMS]
    train_idx = np.asarray(splits["train"])
    models: dict[str, list[object | None]] = {"qgbm": [], "qreg": []}
    fallbacks: dict[str, list[np.ndarray]] = {"qgbm": [], "qreg": []}
    train_n: list[int] = []

    for h in range(tw.X_CONT_DIM):
        observed = masks[:, h] == 1
        tr = train_idx[observed[train_idx]]
        y = targets[tr, h]
        train_n.append(int(tr.size))
        fallback = np.nanquantile(y, q) if y.size else np.full(q.size, np.nan)
        for family in ("qgbm", "qreg"):
            fallbacks[family].append(np.asarray(fallback, dtype=np.float64))
        if tr.size < 2:
            models["qgbm"].append(None)
            models["qreg"].append(None)
            continue

        # Match baselines_trajectory exactly: every horizon receives the same configured seed.
        gbm = bt.make_quantile_gbm(q, seed=seed, params=gbm_params)
        gbm.fit(features[tr], targets[tr, h])
        models["qgbm"].append(gbm)

        qr_models = []
        for qi in q:
            model = bt.make_quantile_regressor(float(qi), alpha=qr_alpha)
            model.fit(features[tr], targets[tr, h])
            qr_models.append(model)
        models["qreg"].append(qr_models)

    return MarginalSuite(q, feature_names, use_event, models, fallbacks, train_n)


def predict_marginals(suite: MarginalSuite, features: np.ndarray, family: str) -> np.ndarray:
    """Predict a rearranged quantile grid shaped [patient, quantile, horizon]."""
    if family not in suite.models:
        raise ValueError(f"unknown marginal family: {family}")
    features = np.asarray(features, dtype=np.float64)
    out = np.full((features.shape[0], suite.quantiles.size, tw.X_CONT_DIM), np.nan)
    for h, model in enumerate(suite.models[family]):
        if model is None:
            raw = np.broadcast_to(suite.fallbacks[family][h], (features.shape[0], suite.quantiles.size))
        elif family == "qgbm":
            raw = np.asarray(model.predict(features), dtype=np.float64)
            raw = raw.reshape(features.shape[0], suite.quantiles.size)
        else:
            raw = np.column_stack([item.predict(features) for item in model])
        out[:, :, h] = bt.monotone_rearrange(raw)
    return out


def conditional_pit(observed: np.ndarray, mask: np.ndarray, predicted: np.ndarray,
                    quantiles: np.ndarray) -> np.ndarray:
    """Invert each fitted quantile function to obtain residual percentile ranks."""
    observed = np.asarray(observed, dtype=np.float64)
    mask = np.asarray(mask)
    predicted = np.asarray(predicted, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    pit = np.full(observed.shape, np.nan, dtype=np.float64)
    for i in range(observed.shape[0]):
        for h in range(observed.shape[1]):
            if mask[i, h] != 1 or not np.isfinite(observed[i, h]):
                continue
            values = predicted[i, :, h]
            if not np.all(np.isfinite(values)):
                continue
            unique, inverse = np.unique(values, return_inverse=True)
            if unique.size == 1:
                pit[i, h] = 0.5
                continue
            q_at_unique = np.array([q[inverse == j].mean() for j in range(unique.size)])
            pit[i, h] = np.interp(observed[i, h], unique, q_at_unique,
                                  left=q_at_unique[0], right=q_at_unique[-1])
    return np.clip(pit, 1e-4, 1.0 - 1e-4)


def nearest_correlation(matrix: np.ndarray, eigen_floor: float = 1e-6) -> np.ndarray:
    """Project a symmetric estimate to a positive-definite correlation matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    projected = (vectors * np.maximum(values, eigen_floor)) @ vectors.T
    scale = np.sqrt(np.maximum(np.diag(projected), eigen_floor))
    projected = projected / np.outer(scale, scale)
    np.fill_diagonal(projected, 1.0)
    return (projected + projected.T) / 2.0


def estimate_rank_correlation(pit: np.ndarray, min_pairs: int = 10,
                              shrinkage: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Gaussian-rank correlation with shrinkage and PSD repair."""
    if min_pairs < 3:
        raise ValueError("min_pairs must be at least 3")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    z = norm.ppf(np.asarray(pit, dtype=np.float64))
    d = z.shape[1]
    corr = np.eye(d, dtype=np.float64)
    counts = np.zeros((d, d), dtype=int)
    for i in range(d):
        counts[i, i] = int(np.isfinite(z[:, i]).sum())
        for j in range(i + 1, d):
            keep = np.isfinite(z[:, i]) & np.isfinite(z[:, j])
            counts[i, j] = counts[j, i] = int(keep.sum())
            if keep.sum() < min_pairs or np.std(z[keep, i]) < 1e-10 or np.std(z[keep, j]) < 1e-10:
                value = 0.0
            else:
                value = float(np.corrcoef(z[keep, i], z[keep, j])[0, 1])
                if not np.isfinite(value):
                    value = 0.0
            corr[i, j] = corr[j, i] = value
    corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(d)
    return nearest_correlation(corr), counts


def latent_rank_orders(correlation: np.ndarray, n_patients: int, n_samples: int,
                       seed: int, independent: bool = False) -> np.ndarray:
    """Return sample indices ordered by independent or correlated latent Gaussian ranks."""
    rng = np.random.default_rng(seed)
    d = correlation.shape[0]
    if independent:
        latent = rng.normal(size=(n_patients, n_samples, d))
    else:
        latent = rng.multivariate_normal(np.zeros(d), correlation, size=(n_patients, n_samples))
    return np.argsort(latent, axis=1)


def apply_rank_orders(marginal_ensemble: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """Permute sorted marginal nodes according to latent ranks without changing marginals."""
    base = np.sort(np.asarray(marginal_ensemble, dtype=np.float64), axis=1)
    if orders.shape != base.shape:
        raise ValueError(f"orders {orders.shape} must match ensemble {base.shape}")
    out = np.empty_like(base)
    for i in range(base.shape[0]):
        for h in range(base.shape[2]):
            out[i, orders[i, :, h], h] = base[i, :, h]
    return out


def coupled_samples(predicted: np.ndarray, quantiles: np.ndarray, correlation: np.ndarray,
                    n_samples: int, seed: int, independent: bool = False,
                    orders: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Create exact-marginal trajectory samples and return them with their latent orders."""
    marginal = bt.quantile_grid_to_ensemble(predicted, quantiles, n_samples=n_samples)
    if orders is None:
        orders = latent_rank_orders(correlation, marginal.shape[0], n_samples, seed, independent)
    return apply_rank_orders(marginal, orders), orders


def sample_paired_counterfactuals(
    suite: MarginalSuite,
    base_features: np.ndarray,
    family: str,
    correlation: np.ndarray,
    n_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Sample sleeve and RYGB predictions with identical patient-level latent rank vectors."""
    surgery_col = suite.feature_names.index("surgery_idx")
    names = list(fm.SURGERY_TO_INDEX)
    if len(names) != 2:
        raise ValueError(f"paired sampling expects two surgery categories, found {names}")
    orders = latent_rank_orders(correlation, base_features.shape[0], n_samples, seed)
    result: dict[str, np.ndarray] = {}
    for name in names:
        features = np.array(base_features, copy=True)
        features[:, surgery_col] = fm.SURGERY_TO_INDEX[name]
        pred = predict_marginals(suite, features, family)
        result[name], _ = coupled_samples(pred, suite.quantiles, correlation, n_samples, seed, orders=orders)
    return result


def training_scale(dataset: fm.FlowDataset, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = dataset.x[:, tw.CONT_DIMS].astype(np.float64)
    mask = dataset.mask[:, tw.CONT_DIMS]
    mean = np.zeros(tw.X_CONT_DIM)
    sd = np.ones(tw.X_CONT_DIM)
    for h in range(tw.X_CONT_DIM):
        values = targets[train_idx, h][mask[train_idx, h] == 1]
        values = values[np.isfinite(values)]
        if values.size:
            mean[h] = values.mean()
        if values.size > 1 and values.std(ddof=1) > 1e-8:
            sd[h] = values.std(ddof=1)
    return mean, sd


def trajectory_scores(samples: np.ndarray, observed: np.ndarray, mask: np.ndarray,
                      mean: np.ndarray, sd: np.ndarray, seed: int) -> dict[str, np.ndarray | float]:
    """Standardised multivariate energy, variogram, jump error, and validity scores."""
    rng = np.random.default_rng(seed)
    per_energy = np.full(observed.shape[0], np.nan)
    per_variogram = np.full(observed.shape[0], np.nan)
    per_jump = np.full(observed.shape[0], np.nan)
    invalid_n = total_n = 0
    for h, group in enumerate(tw.CONT_GROUPS):
        lo, hi = bt.PHYSIOLOGIC_BOUNDS[group]
        finite = np.isfinite(samples[:, :, h])
        invalid_n += int(np.sum(finite & ((samples[:, :, h] < lo) | (samples[:, :, h] > hi))))
        total_n += int(finite.sum())

    for patient in range(observed.shape[0]):
        dims = np.flatnonzero((mask[patient] == 1) & np.isfinite(observed[patient]))
        dims = dims[np.all(np.isfinite(samples[patient, :, dims]), axis=1)]
        if dims.size < 2:
            continue
        x = (samples[patient][:, dims] - mean[dims]) / sd[dims]
        y = (observed[patient, dims] - mean[dims]) / sd[dims]
        first = np.linalg.norm(x - y[None, :], axis=1).mean()
        pair_terms = []
        for _ in range(4):
            pair_terms.append(np.linalg.norm(x - x[rng.permutation(x.shape[0])], axis=1).mean())
        per_energy[patient] = first - 0.5 * float(np.mean(pair_terms))

        variogram_terms = []
        jump_terms = []
        for a in range(dims.size):
            for b in range(a + 1, dims.size):
                i, j = int(dims[a]), int(dims[b])
                if tw.CONT_GROUPS[i] != tw.CONT_GROUPS[j]:
                    continue
                observed_increment = abs(y[a] - y[b])
                predicted_increment = np.mean(np.abs(x[:, a] - x[:, b]))
                gap = abs(qb.horizon_months(tw.CONT_NAMES[i]) - qb.horizon_months(tw.CONT_NAMES[j]))
                weight = 1.0 / np.sqrt(max(gap, 1.0))
                variogram_terms.append(weight * (np.sqrt(observed_increment) -
                                                   np.mean(np.sqrt(np.abs(x[:, a] - x[:, b])))) ** 2)
                if b == a + 1:
                    jump_terms.append(abs(predicted_increment - observed_increment))
        if variogram_terms:
            per_variogram[patient] = float(np.mean(variogram_terms))
        if jump_terms:
            per_jump[patient] = float(np.mean(jump_terms))
    return {
        "energy": per_energy,
        "variogram": per_variogram,
        "adjacent_jump_error": per_jump,
        "mean_energy": float(np.nanmean(per_energy)) if np.isfinite(per_energy).any() else np.nan,
        "mean_variogram": float(np.nanmean(per_variogram)) if np.isfinite(per_variogram).any() else np.nan,
        "mean_adjacent_jump_error": float(np.nanmean(per_jump)) if np.isfinite(per_jump).any() else np.nan,
        "invalid_fraction": float(invalid_n / total_n) if total_n else np.nan,
    }


def trajectory_tables(bundle: dict, dataset: fm.FlowDataset, splits: dict[str, np.ndarray],
                      seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    mean, sd = training_scale(dataset, np.asarray(splits["train"]))
    details = {}
    rows = []
    for arm in bundle["present"]:
        # Reuse the same Monte Carlo pair permutations across arms so differences in the
        # energy score come from the predictive samples, not from score-estimator noise.
        result = trajectory_scores(bundle["arm_samples"][arm], bundle["obs"], bundle["mask"],
                                   mean, sd, seed)
        details[arm] = result
        rows.append({
            "arm": arm, "label": ARM_LABELS[arm],
            "n_energy": int(np.isfinite(result["energy"]).sum()),
            "energy_score": result["mean_energy"],
            "n_variogram": int(np.isfinite(result["variogram"]).sum()),
            "variogram_score": result["mean_variogram"],
            "adjacent_jump_error": result["mean_adjacent_jump_error"],
            "invalid_sample_fraction": result["invalid_fraction"],
        })
    paired_rows = []
    comparisons = []
    if ARM_FLOW in details:
        comparisons.extend((ARM_FLOW, arm) for arm in bundle["present"] if arm != ARM_FLOW)
    comparisons.extend([(ARM_QGBM_COPULA, ARM_QGBM_INDEPENDENT),
                        (ARM_QREG_COPULA, ARM_QREG_INDEPENDENT)])
    for arm_a, arm_b in comparisons:
        if arm_a not in details or arm_b not in details:
            continue
        for metric in ("energy", "variogram", "adjacent_jump_error"):
            test = bt.paired_test(details[arm_a][metric], details[arm_b][metric])
            paired_rows.append({"arm_a": arm_a, "arm_b": arm_b, "metric": metric, **test})
    return pd.DataFrame(rows), pd.DataFrame(paired_rows), details


def make_summary_figure(bundle: dict, marginal: pd.DataFrame, trajectory: pd.DataFrame,
                        correlations: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    pooled = marginal[marginal["horizon"] == "__overall__"].set_index("arm")
    arms = [arm for arm in bundle["present"] if arm in pooled.index]
    axes[0, 0].bar(np.arange(len(arms)), [pooled.loc[a, "crps"] for a in arms],
                   color=[ARM_COLORS[a] for a in arms])
    axes[0, 0].set_title("Marginal CRPS (lower is better)")
    axes[0, 0].set_ylabel("CRPS, pooled original units")
    axes[0, 0].set_xticks(np.arange(len(arms)), [ARM_LABELS[a] for a in arms], rotation=28, ha="right")

    traj = trajectory.set_index("arm")
    for ax, column, title in ((axes[0, 1], "energy_score", "Multivariate energy score"),
                              (axes[0, 2], "variogram_score", "Temporal variogram score")):
        shown = [arm for arm in bundle["present"] if arm in traj.index]
        ax.bar(np.arange(len(shown)), [traj.loc[a, column] for a in shown],
               color=[ARM_COLORS[a] for a in shown])
        ax.set_title(f"{title} (lower is better)")
        ax.set_xticks(np.arange(len(shown)), [ARM_LABELS[a] for a in shown], rotation=28, ha="right")

    for ax, family, title in ((axes[1, 0], "qgbm", "qGBM validation rank correlation"),
                              (axes[1, 1], "qreg", "qReg validation rank correlation")):
        image = ax.imshow(correlations[family], vmin=-1, vmax=1, cmap="coolwarm")
        ax.axvline(8.5, color="black", lw=1)
        ax.axhline(8.5, color="black", lw=1)
        ax.set_title(title)
        ax.set_xticks(range(tw.X_CONT_DIM), tw.CONT_NAMES, rotation=90, fontsize=6)
        ax.set_yticks(range(tw.X_CONT_DIM), tw.CONT_NAMES, fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    example = int(np.argmax(np.sum(bundle["mask"], axis=1)))
    bmi_dims = [h for h, group in enumerate(tw.CONT_GROUPS) if group == "bmi"]
    months = np.array([qb.horizon_months(tw.CONT_NAMES[h]) for h in bmi_dims])
    obs = bundle["obs"][example, bmi_dims]
    obs_mask = bundle["mask"][example, bmi_dims] == 1
    ax.plot(months[obs_mask], obs[obs_mask], "ko-", lw=2, label="observed")
    for arm in [ARM_FLOW, ARM_QGBM_INDEPENDENT, ARM_QGBM_COPULA]:
        if arm not in bundle["arm_samples"]:
            continue
        values = bundle["arm_samples"][arm][example, :, :][:, bmi_dims]
        for row in values[:8]:
            ax.plot(months, row, color=ARM_COLORS[arm], alpha=0.10, lw=0.8)
        ax.plot(months, np.median(values, axis=0), color=ARM_COLORS[arm], lw=2,
                label=ARM_LABELS[arm])
    ax.set_title("Example BMI trajectory draws")
    ax.set_xlabel("months since surgery")
    ax.set_ylabel("BMI")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    for ax in axes[0]:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Flow matching vs quantile rank-copula trajectories (test n={bundle['n_test']})",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def counterfactual_outputs(suite: MarginalSuite, features: np.ndarray,
                           correlations: dict[str, np.ndarray], args, output_dir: Path) -> dict:
    rows = []
    surgery_names = list(fm.SURGERY_TO_INDEX)
    for family in ("qgbm", "qreg"):
        paired = sample_paired_counterfactuals(suite, features, family, correlations[family],
                                               args.n_samples, args.seed + 500)
        difference = paired[surgery_names[1]] - paired[surgery_names[0]]
        for h, horizon in enumerate(tw.CONT_NAMES):
            patient_effect = difference[:, :, h].mean(axis=1)
            rows.append({
                "family": family, "horizon": horizon, "group": tw.CONT_GROUPS[h],
                "contrast": f"{surgery_names[1]} minus {surgery_names[0]}",
                "mean": float(np.mean(patient_effect)),
                "median": float(np.median(patient_effect)),
                "q025": float(np.quantile(patient_effect, 0.025)),
                "q975": float(np.quantile(patient_effect, 0.975)),
            })
    table = pd.DataFrame(rows)
    csv_path = output_dir / "paired_counterfactual_summary.csv"
    table.to_csv(csv_path, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, group in zip(axes, ("bmi", "hba1c")):
        for family, color in (("qgbm", ARM_COLORS[ARM_QGBM_COPULA]),
                              ("qreg", ARM_COLORS[ARM_QREG_COPULA])):
            block = table[(table.family == family) & (table.group == group)]
            x = np.array([qb.horizon_months(name) for name in block.horizon])
            ax.plot(x, block["mean"], marker="o", color=color, label=family)
            ax.fill_between(x, block["q025"], block["q975"], color=color, alpha=0.15)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(f"Paired predictive contrast: {group.upper()}")
        ax.set_xlabel("months since surgery")
        ax.set_ylabel(f"{surgery_names[1]} minus {surgery_names[0]}")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig_path = output_dir / "paired_counterfactual.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {"csv": str(csv_path), "figure": str(fig_path),
            "event_feature_held_fixed": bool(suite.use_event)}


def self_test() -> None:
    """Fast invariance and correlation checks that require no cohort or fitted model."""
    rng = np.random.default_rng(123)
    n, s, d = 5, 100, 4
    base = np.sort(rng.normal(size=(n, s, d)), axis=1)
    target = np.array([[1.0, 0.7, 0.0, -0.2], [0.7, 1.0, 0.3, 0.0],
                       [0.0, 0.3, 1.0, 0.5], [-0.2, 0.0, 0.5, 1.0]])
    target = nearest_correlation(target)
    orders = latent_rank_orders(target, n, s, seed=2)
    coupled = apply_rank_orders(base, orders)
    if not np.allclose(np.sort(coupled, axis=1), base):
        raise AssertionError("copula coupling changed marginal samples")
    pit = norm.cdf(rng.multivariate_normal(np.zeros(d), target, size=2000))
    estimated, _ = estimate_rank_correlation(pit, min_pairs=10, shrinkage=0.0)
    if np.max(np.abs(estimated - target)) > 0.08:
        raise AssertionError("rank-correlation recovery failed")
    if np.linalg.eigvalsh(estimated).min() <= 0:
        raise AssertionError("correlation repair did not produce positive definiteness")
    print("Self-test: marginal invariance, correlation recovery, and PSD checks passed.")


def resolve_flow_run(args) -> Path:
    if args.twin_run:
        return Path(args.twin_run)
    if args.pipeline:
        return Path(et.resolve_from_pipeline(Path(args.pipeline))["twin_final_run_dir"])
    latest = et.find_latest_pipeline()
    if latest is None:
        raise SystemExit("Provide --twin-run or --pipeline, or use --no-flow.")
    return Path(et.resolve_from_pipeline(latest)["twin_final_run_dir"])


def run(args) -> dict:
    if args.self_test:
        self_test()
    dataset = et.load_dataset(Path(args.csv) if args.csv else None)
    flow_run = None
    if args.no_flow:
        cfg = tw.TwinConfig(split_strategy=args.split_strategy, split_seed=args.split_seed,
                            train_frac=args.train_frac, val_frac=args.val_frac,
                            test_frac=args.test_frac)
    else:
        flow_run = resolve_flow_run(args)
        cfg = et.load_twin_config(flow_run)
    splits = tw.make_splits(dataset, cfg)
    if len(splits["val"]) == 0:
        raise ValueError("a non-empty validation split is required to estimate the residual-rank copula")

    print("Training qGBM and qReg marginal models...", flush=True)
    suite = fit_marginal_suite(
        dataset, splits, np.asarray(bt.DEFAULT_QUANTILES), not args.baseline_no_event,
        args.seed, gbm_params={"n_estimators": args.gbm_estimators}, qr_alpha=args.qr_alpha,
    )
    features, _ = bt.build_feature_matrix(dataset, suite.use_event)
    targets = dataset.x[:, tw.CONT_DIMS].astype(np.float64)
    masks = dataset.mask[:, tw.CONT_DIMS]
    val_idx, test_idx = np.asarray(splits["val"]), np.asarray(splits["test"])

    correlations: dict[str, np.ndarray] = {}
    pair_counts: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for family in ("qgbm", "qreg"):
        val_pred = predict_marginals(suite, features[val_idx], family)
        pit = conditional_pit(targets[val_idx], masks[val_idx], val_pred, suite.quantiles)
        correlations[family], pair_counts[family] = estimate_rank_correlation(
            pit, min_pairs=args.min_copula_pairs, shrinkage=args.copula_shrinkage)
        test_predictions[family] = predict_marginals(suite, features[test_idx], family)

    samples: dict[str, np.ndarray] = {}
    for offset, family in enumerate(("qgbm", "qreg")):
        independent, _ = coupled_samples(test_predictions[family], suite.quantiles,
                                         correlations[family], args.n_samples,
                                         args.seed + 100 + offset, independent=True)
        copula, _ = coupled_samples(test_predictions[family], suite.quantiles,
                                    correlations[family], args.n_samples,
                                    args.seed + 200 + offset)
        samples[f"{family}_independent"] = independent
        samples[f"{family}_copula"] = copula
        if not np.allclose(np.sort(independent, axis=1), np.sort(copula, axis=1), equal_nan=True):
            raise AssertionError(f"{family} copula did not preserve exact marginal samples")

    pre = mt.fit_preprocessing(dataset, splits["train"])
    test_arrays = mt.split_arrays(dataset, splits, pre)["test"]
    if not args.no_flow:
        device = et.choose_device(args.device)
        flow_model = et.restore_twin(flow_run, cfg, device)
        samples[ARM_FLOW] = et._sample_twin_arm(flow_model, dataset, test_idx, pre, cfg,
                                                args.n_samples, args.n_steps, device)

    present = [arm for arm in ARM_ORDER if arm in samples]
    bundle = {
        "dataset_source": dataset.source_label, "event_twin_run": str(flow_run) if flow_run else None,
        "split_strategy": cfg.split_strategy, "split_seed": cfg.split_seed,
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "obs": test_arrays["original_x"], "mask": test_arrays["original_mask"],
        "cont_names": list(tw.CONT_NAMES), "cont_groups": list(tw.CONT_GROUPS),
        "arm_samples": samples, "present": present, "quantiles": suite.quantiles,
        "n_quantile_ensemble_samples": args.n_samples,
        "baseline_feature_names": suite.feature_names, "n_test": int(test_idx.size),
    }

    scored = qb.score_all(bundle)
    marginal = scored["metrics"]
    calibration = scored["calibration"]
    marginal_paired = qb.paired_tests(bundle, scored["per_patient"])
    trajectory, paired, _ = trajectory_tables(bundle, dataset, splits, args.seed + 300)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "marginal_metrics": output_dir / "copula_marginal_metrics.csv",
        "calibration": output_dir / "copula_calibration.csv",
        "trajectory_metrics": output_dir / "copula_trajectory_metrics.csv",
        "marginal_paired_tests": output_dir / "copula_marginal_paired_tests.csv",
        "trajectory_paired_tests": output_dir / "copula_trajectory_paired_tests.csv",
        "figure": output_dir / "copula_benchmark.png",
        "correlations": output_dir / "copula_correlations.npz",
        "summary": output_dir / "copula_benchmark_summary.json",
    }
    marginal.to_csv(paths["marginal_metrics"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    trajectory.to_csv(paths["trajectory_metrics"], index=False)
    marginal_paired.to_csv(paths["marginal_paired_tests"], index=False)
    paired.to_csv(paths["trajectory_paired_tests"], index=False)
    np.savez_compressed(paths["correlations"], qgbm=correlations["qgbm"],
                        qreg=correlations["qreg"], qgbm_pair_counts=pair_counts["qgbm"],
                        qreg_pair_counts=pair_counts["qreg"])
    make_summary_figure(bundle, marginal, trajectory, correlations, paths["figure"])

    counterfactual = None
    if args.counterfactual:
        if suite.use_event:
            warnings.warn("Counterfactual sampling holds the observed event feature fixed and is not causal.",
                          stacklevel=2)
        counterfactual = counterfactual_outputs(suite, features[test_idx], correlations, args, output_dir)

    pooled = marginal[marginal.horizon == "__overall__"]
    summary = {
        "dataset_source": dataset.source_label,
        "flow_run": str(flow_run) if flow_run else None,
        "split_strategy": cfg.split_strategy,
        "split_seed": cfg.split_seed,
        "split_sizes": bundle["split_sizes"],
        "arms": present,
        "n_samples": args.n_samples,
        "n_steps_flow": args.n_steps if flow_run else None,
        "copula_training_partition": "validation",
        "copula_min_pairs": args.min_copula_pairs,
        "copula_shrinkage": args.copula_shrinkage,
        "minimum_pair_count": {family: int(pair_counts[family][np.triu_indices(tw.X_CONT_DIM, 1)].min())
                               for family in pair_counts},
        "marginal_crps_overall": {row.arm: float(row.crps) for row in pooled.itertuples()},
        "trajectory_scores": trajectory.set_index("arm").to_dict(orient="index"),
        "counterfactual": counterfactual,
        "files": {key: str(value) for key, value in paths.items() if key != "summary"},
        "notes": {
            "dependence_isolation": "independent and copula variants use identical marginal sample multisets",
            "energy_variogram": "computed after horizon-wise standardisation using training-only mean and SD",
            "missingness": "each patient score uses that patient's observed finite horizons",
            "causal_warning": "paired surgery predictions are predictive contrasts, not identified causal effects",
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print("\nMarginal pooled CRPS (lower is better):")
    print(pooled[["arm", "crps"]].round(4).to_string(index=False))
    print("\nTrajectory scores (lower is better):")
    print(trajectory[["arm", "energy_score", "variogram_score", "adjacent_jump_error",
                      "invalid_sample_fraction"]].round(4).to_string(index=False))
    print(f"\nSaved benchmark artifacts to {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--twin-run", type=str, default=None)
    parser.add_argument("--pipeline", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(fm.REPO_ROOT / "runs" / "copula_trajectory"))
    parser.add_argument("--no-flow", action="store_true")
    parser.add_argument("--baseline-no-event", action="store_true")
    parser.add_argument("--counterfactual", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--gbm-estimators", type=int, default=300)
    parser.add_argument("--qr-alpha", type=float, default=0.0)
    parser.add_argument("--min-copula-pairs", type=int, default=10)
    parser.add_argument("--copula-shrinkage", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-strategy", choices=["surgery", "temporal", "outcome"], default="surgery")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
