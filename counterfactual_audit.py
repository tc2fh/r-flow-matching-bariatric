"""Paired-noise counterfactual safety and numerical-stability audit.

The audit is intentionally separate from the causal estimand. It asks whether the
frozen trajectory generator behaves numerically and physiologically when surgery is
flipped for the same patient and the same latent draw. It writes machine-readable
patient/horizon rows plus a compact dashboard and headline table for collaborators.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import baselines_trajectory as bt
import causal_tte as ct
import evaluate_twin as ev
import train_flow_matching_multitask as mt
import train_flow_matching_twin as tw


EFFECT_LIMITS = {"bmi": 15.0, "hba1c": 5.0}
PS_TRIM = (0.05, 0.95)


def paired_noise(n_patients: int, n_samples: int, n_dims: int, seed: int) -> np.ndarray:
    """Return patient-major latent draws reusable by both treatment arms."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_patients, n_samples, n_dims)).astype(np.float32)


def _support_diagnostics(dataset, splits, pre) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leak-free propensity and opposite-arm nearest-neighbour support diagnostics."""
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    train_idx = np.asarray(splits["train"], dtype=np.int64)
    L, treatment, _ = ct.build_L_A(dataset)
    propensity, _ = ct.propensity_scores(L, treatment, train_idx, test_idx)

    features = mt.transform_patient_features(dataset.patient_features_raw, pre)
    train_x = np.asarray(features[train_idx], dtype=float)
    test_x = np.asarray(features[test_idx], dtype=float)
    train_a = np.asarray(treatment[train_idx], dtype=int)
    test_a = np.asarray(treatment[test_idx], dtype=int)
    distances = np.full(test_idx.size, np.nan, dtype=float)
    thresholds = np.full(test_idx.size, np.nan, dtype=float)
    from sklearn.neighbors import NearestNeighbors

    for arm in (0, 1):
        arm_train = train_x[train_a == arm]
        if arm_train.shape[0] < 2:
            continue
        within = NearestNeighbors(n_neighbors=2).fit(arm_train)
        within_dist = within.kneighbors(arm_train, return_distance=True)[0][:, 1]
        threshold = float(np.quantile(within_dist, 0.99))
        q = test_a != arm
        if q.any():
            nn = NearestNeighbors(n_neighbors=1).fit(arm_train)
            distances[q] = nn.kneighbors(test_x[q], return_distance=True)[0][:, 0]
            thresholds[q] = threshold

    cf_probability = np.where(test_a == 0, propensity, 1.0 - propensity)
    overlap = (
        np.isfinite(propensity)
        & (propensity > PS_TRIM[0])
        & (propensity < PS_TRIM[1])
        & (cf_probability >= PS_TRIM[0])
    )
    distance_supported = ~np.isfinite(distances) | ~np.isfinite(thresholds) | (distances <= thresholds)
    return propensity, distances, overlap & distance_supported


def _patient_rows(dataset, test_idx, factual, counterfactual, propensity, nn_distance, supported):
    rows = []
    factual_surgery = np.asarray(dataset.surgery_idx[test_idx], dtype=int)
    for dim, (name, group) in enumerate(zip(tw.CONT_NAMES, tw.CONT_GROUPS)):
        lo, hi = bt.PHYSIOLOGIC_BOUNDS[group]
        fac = np.asarray(factual[:, :, dim], dtype=float)
        cf = np.asarray(counterfactual[:, :, dim], dtype=float)
        delta = cf - fac
        for i, subject_id in enumerate(dataset.subject_ids[test_idx]):
            fac_bad = ~np.isfinite(fac[i]) | (fac[i] < lo) | (fac[i] > hi)
            cf_bad = ~np.isfinite(cf[i]) | (cf[i] < lo) | (cf[i] > hi)
            d = delta[i][np.isfinite(delta[i])]
            d_med = float(np.median(d)) if d.size else np.nan
            rows.append({
                "subject_id": str(subject_id),
                "horizon": name,
                "group": group,
                "factual_surgery": "rnygb" if factual_surgery[i] == 1 else "sleeve",
                "counterfactual_surgery": "sleeve" if factual_surgery[i] == 1 else "rnygb",
                "event": int(dataset.x[test_idx[i], tw.MACE_DIM]),
                "contrast_type": "event-fixed predictive diagnostic (not total causal effect)",
                "n_samples": int(fac.shape[1]),
                "factual_p01": float(np.nanquantile(fac[i], 0.01)),
                "factual_median": float(np.nanmedian(fac[i])),
                "factual_p99": float(np.nanquantile(fac[i], 0.99)),
                "counterfactual_p01": float(np.nanquantile(cf[i], 0.01)),
                "counterfactual_median": float(np.nanmedian(cf[i])),
                "counterfactual_p99": float(np.nanquantile(cf[i], 0.99)),
                "delta_p01": float(np.nanquantile(d, 0.01)) if d.size else np.nan,
                "delta_median": d_med,
                "delta_p99": float(np.nanquantile(d, 0.99)) if d.size else np.nan,
                "factual_bound_violation_rate": float(fac_bad.mean()),
                "counterfactual_bound_violation_rate": float(cf_bad.mean()),
                "propensity_rygb": float(propensity[i]),
                "opposite_arm_nn_distance": float(nn_distance[i]),
                "counterfactual_supported": bool(supported[i]),
                "extreme_effect": bool(np.isfinite(d_med) and abs(d_med) > EFFECT_LIMITS[group]),
            })
    return pd.DataFrame(rows)


def _horizon_summary(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon, group), part in patient.groupby(["horizon", "group"], sort=False):
        d = part["delta_median"].to_numpy(dtype=float)
        rows.append({
            "horizon": horizon,
            "group": group,
            "n_patients": int(len(part)),
            "delta_patient_p01": float(np.nanquantile(d, 0.01)),
            "delta_patient_median": float(np.nanmedian(d)),
            "delta_patient_p99": float(np.nanquantile(d, 0.99)),
            "factual_draw_bound_violation_pct": 100.0 * float(part["factual_bound_violation_rate"].mean()),
            "counterfactual_draw_bound_violation_pct": 100.0 * float(part["counterfactual_bound_violation_rate"].mean()),
            "unsupported_patient_pct": 100.0 * float((~part["counterfactual_supported"]).mean()),
            "extreme_patient_pct": 100.0 * float(part["extreme_effect"].mean()),
        })
    return pd.DataFrame(rows)


def _solver_audit(model, arrays, pre, device, base_cfg, noise, patient_indices) -> pd.DataFrame:
    if patient_indices.size == 0:
        return pd.DataFrame()
    local = {key: np.asarray(value)[patient_indices] for key, value in arrays.items()}
    event = local["y_mace"]
    local_noise = noise[patient_indices, : min(8, noise.shape[1])]
    specs = [("euler", base_cfg.sample_steps), ("heun", base_cfg.sample_steps),
             ("heun", base_cfg.sample_steps * 2)]
    effects = {}
    violations = {}
    for solver, steps in specs:
        cfg = replace(base_cfg, n_samples_per_patient=local_noise.shape[1], sample_steps=steps)
        fac = ev.twin_samples_15(
            model, local, event, cfg, pre, device, initial_noise=local_noise, solver=solver,
        )
        cf = ev.twin_samples_15(
            model, local, event, cfg, pre, device, flip_surgery=True,
            initial_noise=local_noise, solver=solver,
        )
        effects[(solver, steps)] = np.nanmedian(cf - fac, axis=1)
        bad = np.zeros_like(fac, dtype=bool)
        for dim, group in enumerate(tw.CONT_GROUPS):
            lo, hi = bt.PHYSIOLOGIC_BOUNDS[group]
            bad[:, :, dim] = (~np.isfinite(fac[:, :, dim]) | (fac[:, :, dim] < lo)
                              | (fac[:, :, dim] > hi) | ~np.isfinite(cf[:, :, dim])
                              | (cf[:, :, dim] < lo) | (cf[:, :, dim] > hi))
        violations[(solver, steps)] = float(bad.mean())

    reference_key = ("heun", base_cfg.sample_steps * 2)
    reference = effects[reference_key]
    rows = []
    for key, effect in effects.items():
        error = np.abs(effect - reference)
        for group in ("bmi", "hba1c"):
            dims = np.asarray([g == group for g in tw.CONT_GROUPS])
            values = error[:, dims].reshape(-1)
            rows.append({
                "solver": key[0], "steps": key[1], "reference": "heun_2x",
                "group": group, "n_patients": int(patient_indices.size),
                "median_abs_effect_difference": float(np.nanmedian(values)),
                "p95_abs_effect_difference": float(np.nanquantile(values, 0.95)),
                "max_abs_effect_difference": float(np.nanmax(values)),
                "combined_bound_violation_pct": 100.0 * violations[key],
            })
    return pd.DataFrame(rows)


def _save_dashboard(horizon: pd.DataFrame, solver: pd.DataFrame, headline: pd.DataFrame,
                    stem: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, group, title, unit in [
        (axes[0, 0], "bmi", "A  Paired surgery effect by BMI horizon", "BMI kg/m²"),
        (axes[0, 1], "hba1c", "B  Paired surgery effect by HbA1c horizon", "HbA1c points"),
    ]:
        part = horizon[horizon["group"] == group]
        x = np.arange(len(part))
        lower = part["delta_patient_p01"].to_numpy(dtype=float)
        upper = part["delta_patient_p99"].to_numpy(dtype=float)
        median = part["delta_patient_median"].to_numpy(dtype=float)
        ax.fill_between(x, lower, upper, alpha=0.18)
        ax.plot(x, median, marker="o", lw=1.6)
        ax.axhline(0, color="0.45", lw=0.8)
        ax.set_xticks(x, part["horizon"], rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(f"counterfactual - factual ({unit})")
        ax.set_title(title, loc="left", fontweight="bold")

    ax = axes[1, 0]
    x = np.arange(len(horizon))
    ax.bar(x - 0.18, horizon["factual_draw_bound_violation_pct"], 0.36, label="factual")
    ax.bar(x + 0.18, horizon["counterfactual_draw_bound_violation_pct"], 0.36, label="counterfactual")
    ax.set_xticks(x, horizon["horizon"], rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("draws outside physiological bounds (%)")
    ax.set_title("C  Physiological validity", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    shown = solver[solver["reference"] == "heun_2x"].copy()
    labels = [f"{r.solver}-{int(r.steps)}\n{r.group}" for r in shown.itertuples()]
    ax.bar(np.arange(len(shown)), shown["p95_abs_effect_difference"], color="tab:purple", alpha=0.75)
    ax.set_xticks(np.arange(len(shown)), labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("p95 absolute difference from Heun 2x")
    ax.set_title("D  Solver convergence", loc="left", fontweight="bold")
    values = dict(zip(headline["metric"], headline["value"]))
    note = (
        f"n={values['test patients']} | unsupported={values['unsupported counterfactuals']} | "
        f"any counterfactual bound violation={values['patients with any CF bound violation']} | "
        f"extreme median effect={values['patients with an extreme median effect']}\n"
        f"p95 effect error vs Heun 2x: Euler={values['Euler p95 error vs Heun 2x']}, "
        f"Heun={values['Heun p95 error vs Heun 2x']}"
    )
    fig.suptitle("Counterfactual trajectory safety audit", fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.935, note, ha="center", va="top", fontsize=9)
    fig.subplots_adjust(top=0.85, bottom=0.12, hspace=0.48, wspace=0.25)

    written = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _save_headline_table(headline: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.axis("off")
    table = ax.table(cellText=headline.values, colLabels=headline.columns,
                     cellLoc="left", colLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    ax.set_title("Counterfactual safety audit: collaborator headline", fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(*, dataset, splits, pre, model, twin_cfg, device, output_dir: Path,
        prefix: str = "eval_counterfactual", n_samples: int = 64,
        n_steps: int = 50, seed: int = 0) -> dict:
    """Run the paired counterfactual audit and write CSV plus visual artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    arrays = ev.arrays_for(dataset, test_idx, pre)
    cfg = replace(twin_cfg, n_samples_per_patient=n_samples, sample_steps=n_steps)
    noise = paired_noise(test_idx.size, n_samples, tw.X_CONT_DIM, seed)
    factual = ev.twin_samples_15(
        model, arrays, arrays["y_mace"], cfg, pre, device,
        initial_noise=noise, solver="heun",
    )
    counterfactual = ev.twin_samples_15(
        model, arrays, arrays["y_mace"], cfg, pre, device, flip_surgery=True,
        initial_noise=noise, solver="heun",
    )
    propensity, nn_distance, supported = _support_diagnostics(dataset, splits, pre)
    patient = _patient_rows(
        dataset, test_idx, factual, counterfactual, propensity, nn_distance, supported,
    )
    horizon = _horizon_summary(patient)

    severity = patient.groupby("subject_id").agg(
        any_bound=("counterfactual_bound_violation_rate", lambda x: bool(np.any(np.asarray(x) > 0))),
        max_effect=("delta_median", lambda x: float(np.nanmax(np.abs(x)))),
    )
    ranked_ids = severity.sort_values(["any_bound", "max_effect"], ascending=False).head(128).index
    lookup = {str(s): i for i, s in enumerate(dataset.subject_ids[test_idx])}
    diagnostic_idx = [lookup[s] for s in ranked_ids if s in lookup]
    rng = np.random.default_rng(seed + 1)
    remaining = np.setdiff1d(np.arange(test_idx.size), np.asarray(diagnostic_idx, dtype=int))
    if remaining.size:
        diagnostic_idx.extend(rng.choice(remaining, size=min(128, remaining.size), replace=False).tolist())
    solver = _solver_audit(
        model, arrays, pre, device, cfg, noise, np.asarray(diagnostic_idx, dtype=int),
    )

    patient_path = output_dir / f"{prefix}_patient_horizon.csv"
    horizon_path = output_dir / f"{prefix}_horizon_summary.csv"
    solver_path = output_dir / f"{prefix}_solver_convergence.csv"
    patient.to_csv(patient_path, index=False)
    horizon.to_csv(horizon_path, index=False)
    solver.to_csv(solver_path, index=False)

    heun_row = solver[(solver["solver"] == "heun") & (solver["steps"] == n_steps)]
    euler_row = solver[(solver["solver"] == "euler") & (solver["steps"] == n_steps)]
    headline = pd.DataFrame([
        {"metric": "test patients", "value": str(test_idx.size)},
        {"metric": "contrast", "value": "paired-noise, event-fixed diagnostic"},
        {"metric": "unsupported counterfactuals", "value": f"{100 * (~supported).mean():.1f}%"},
        {"metric": "patients with any CF bound violation",
         "value": f"{100 * severity['any_bound'].mean():.1f}%"},
        {"metric": "patients with an extreme median effect",
         "value": f"{100 * patient.groupby('subject_id')['extreme_effect'].any().mean():.1f}%"},
        {"metric": "Euler p95 error vs Heun 2x",
         "value": f"{euler_row['p95_abs_effect_difference'].max():.3f}" if not euler_row.empty else "NA"},
        {"metric": "Heun p95 error vs Heun 2x",
         "value": f"{heun_row['p95_abs_effect_difference'].max():.3f}" if not heun_row.empty else "NA"},
    ])
    headline_path = output_dir / f"{prefix}_headline.csv"
    headline_png = output_dir / f"{prefix}_headline.png"
    headline.to_csv(headline_path, index=False)
    _save_headline_table(headline, headline_png)
    dashboard_stem = output_dir / f"{prefix}_dashboard"
    dashboard = _save_dashboard(horizon, solver, headline, dashboard_stem)
    return {
        "patient_horizon_csv": str(patient_path),
        "horizon_summary_csv": str(horizon_path),
        "solver_convergence_csv": str(solver_path),
        "headline_csv": str(headline_path),
        "headline_png": str(headline_png),
        "dashboard": [str(path) for path in dashboard],
        "n_test": int(test_idx.size),
        "n_samples": int(n_samples),
        "paired_noise": True,
        "primary_solver": "heun",
        "physiologic_bounds": bt.PHYSIOLOGIC_BOUNDS,
        "effect_limits": EFFECT_LIMITS,
    }
