"""fairness_audit.py - SVI / RUCA / race equity-fairness subgroup audit of the twin.

Wave-2 orchestrator (Agent D) for the TTE + distributional + fairness build. It answers
the manuscript's equity question directly:

  Does the twin's calibration / accuracy and the GBM's composite-event discrimination
  DEGRADE for socially vulnerable subgroups (high SVI, rural, minority, Medicaid /
  uninsured)?

It reports per-subgroup performance on the TEST set plus the GAP between the most- and
least-vulnerable strata. It reuses the Wave-1 pure metric modules (``distributional_metrics``,
``calibration_twin``, ``baselines_trajectory``) and the existing evaluator helpers
(``evaluate_twin``); it writes ``fairness_*`` artifacts and returns a JSON-native manifest
block. It NEVER edits any existing file.

Design rules honored (see INTEGRATION_CONTRACT.md):
  * PURE, unit-testable subgroup-label helpers (svi_quartiles / ruca_rural_urban /
    race_bucket / insurance_bucket / compute_gap) map raw columns -> subgroup labels with no
    side effects; NaN / unparseable -> "unknown" and is excluded from every statistic.
  * EFFICIENCY (Section 7): the FACTUAL twin block is sampled ONCE (n_test, s, 15) and every
    horizon / threshold is read off it. The optional counterfactual effect samples the twin 4
    more times TOTAL (2 arms x 2 events), never per-horizon.
  * Small-sample discipline: every statistic is guarded at ``MIN_N`` (=10) observations and
    degrades to NaN rather than raising. On the 52-row fake cohort the test split is tiny, so
    almost every statistic is NaN by design - the audit must still run and write every file.
  * RACE IS AUDIT-ONLY. FirstRace is bucketed for equity reporting and is NEVER a model input
    (the flow conditions on ``fm.PATIENT_FEATURES`` only; the PS model excludes race via
    ``causal_tte.INCLUDE_RACE_IN_PS = False``).

Subgroup axes (built on the TEST rows of ``dataset.frame`` via gb.frame_feature /
gb.frame_categorical). Raw-level mappings are DOCUMENTED here because registry spellings vary;
matching is case-insensitive / substring so real-world variants are absorbed:

  * SVI  (``SviOverall``, numeric 0-1): quartiles Q1 (least vulnerable) .. Q4 (most vulnerable)
    by within-test quartile. Gap = Q4 - Q1. Guard: < 8 finite values -> all "unknown".
  * RUCA (``RUCA``, e.g. "1 metropolitan", "7 small town", "10 rural", or coded "1.0"/"7.2"):
    leading integer code 1-3 -> "urban", 4-10 -> "rural". Gap = rural - urban. Unparseable ->
    "unknown".
  * RACE (``FirstRace``, AUDIT ONLY): White / Black / Hispanic / Asian / Other. Substring rules:
    "hispanic"|"latino" -> Hispanic (ethnicity wins); "white"|"caucasian" -> White;
    "black"|"african" -> Black; "asian" -> Asian; everything else present (American Indian /
    Alaska Native, Native Hawaiian / Pacific Islander, Multiracial, ...) -> Other. Gaps = each
    minority bucket - White. Blank / "unknown" / "not reported" -> "unknown".
  * INSURANCE (``CoverageClass``, optional): Medicare / Medicaid / Commercial / Uninsured /
    Other. Substring rules: "medicaid" -> Medicaid; "medicare" -> Medicare;
    "commercial"|"private"|"hmo"|"ppo" -> Commercial; "self"|"uninsured"|"no insurance" ->
    Uninsured; else Other. Gaps = Medicaid - Commercial and Uninsured - Commercial.

An ADVERSE gap (worse calibration / higher error / lower AUROC / higher ECE for the vulnerable
stratum) is the finding; the ``adverse_direction`` column of ``fairness_gaps.csv`` records
which sign is adverse per metric.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import evaluate_twin as ev
import evaluate_flow_matching as evfm
import gbm_mace_baseline as gb
import train_flow_matching as fm
import calibration_twin as ct
import baselines_trajectory as bt
import distributional_metrics as dm

# Below this many usable (subgroup + observed) rows a statistic is too noisy to trust -> NaN.
MIN_N = 10

BMI_THRESHOLD = 35.0
HBA1C_THRESHOLD = 5.7

# Headline horizons (name, group) reported individually; everything else is averaged.
HEADLINE_HORIZONS = [("bmi_12m", "bmi"), ("hba1c_12m", "hba1c")]
# Threshold-reliability targets: (label, horizon_name, threshold).
THRESHOLD_TARGETS = [("bmi35_12m", "bmi_12m", BMI_THRESHOLD),
                     ("hba1c57_12m", "hba1c_12m", HBA1C_THRESHOLD)]
# Representative follow-up horizons for the differential-attrition columns.
FOLLOWUP_HORIZONS = ["bmi_12m", "bmi_2y", "hba1c_12m", "hba1c_2y"]

# Subgroup order + gap reference pairs (most_vulnerable, least_vulnerable) per axis.
AXIS_SUBGROUPS = {
    "svi": ["Q1", "Q2", "Q3", "Q4"],
    "ruca": ["urban", "rural"],
    "race": ["White", "Black", "Hispanic", "Asian", "Other"],
    "insurance": ["Commercial", "Medicare", "Medicaid", "Uninsured", "Other"],
}
GAP_PAIRS = {
    "svi": [("Q4", "Q1")],
    "ruca": [("rural", "urban")],
    "race": [("Black", "White"), ("Hispanic", "White"), ("Asian", "White"), ("Other", "White")],
    "insurance": [("Medicaid", "Commercial"), ("Uninsured", "Commercial")],
}
# Metrics compared as gaps + the sign that is ADVERSE for the vulnerable stratum.
GAP_METRICS = ["coverage_avg", "mad_avg", "crps_avg", "ece_bmi35_12m", "gbm_auroc"]
ADVERSE_DIRECTION = {
    "coverage_avg": "negative",   # lower coverage for vulnerable = adverse
    "mad_avg": "positive",        # higher error = adverse
    "crps_avg": "positive",       # higher CRPS = adverse
    "ece_bmi35_12m": "positive",  # higher calibration error = adverse
    "gbm_auroc": "negative",      # lower discrimination = adverse
}

_NAN = float("nan")


# --------------------------------------------------------------------------- #
# PURE subgroup-label helpers (no side effects; NaN / unparseable -> "unknown")
# --------------------------------------------------------------------------- #
def svi_quartiles(svi_values) -> np.ndarray:
    """Map a per-patient SviOverall vector to quartile labels Q1 (least) .. Q4 (most vulnerable).

    Quartiles are cut within the passed vector (the test slice). SVI is higher = more
    vulnerable, so Q4 is the top quartile. Fewer than 8 finite values -> all "unknown"
    (quartiles are meaningless). Non-finite entries stay "unknown". Returns an object array
    the same length as the input.
    """
    v = np.asarray(svi_values, dtype=float)
    out = np.full(v.shape[0], "unknown", dtype=object)
    finite = np.isfinite(v)
    if int(finite.sum()) < 8:
        return out
    edges = np.nanquantile(v[finite], [0.25, 0.50, 0.75])
    names = np.array(["Q1", "Q2", "Q3", "Q4"], dtype=object)
    codes = np.clip(np.digitize(v[finite], edges, right=False), 0, 3)
    out[finite] = names[codes]
    return out


_RUCA_LEAD = re.compile(r"\s*(\d+)")


def ruca_rural_urban(ruca_values) -> np.ndarray:
    """Map raw RUCA strings to "urban" (leading code 1-3) / "rural" (4-10) / "unknown".

    Parses the INTEGER part of the leading number so both descriptive ("1 metropolitan",
    "10 rural") and coded ("1.0", "7.2") spellings work. Missing / unparseable / out-of-range
    -> "unknown". Returns an object array aligned to the input.
    """
    vals = list(ruca_values)
    out = np.full(len(vals), "unknown", dtype=object)
    for i, raw in enumerate(vals):
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none"):
            continue
        m = _RUCA_LEAD.match(s)
        if not m:
            continue
        code = int(m.group(1))
        if 1 <= code <= 3:
            out[i] = "urban"
        elif 4 <= code <= 10:
            out[i] = "rural"
    return out


def race_bucket(race_values) -> np.ndarray:
    """Bucket raw FirstRace levels into White / Black / Hispanic / Asian / Other / "unknown".

    AUDIT ONLY - race is never a model input. Case-insensitive substring matching absorbs
    registry spelling variants; ethnicity ("hispanic"/"latino") takes precedence. Small /
    unlisted levels (American Indian, Pacific Islander, Multiracial, ...) collapse to "Other".
    Blank / "unknown" / "not reported" -> "unknown". Returns an object array aligned to input.
    """
    vals = list(race_values)
    out = np.full(len(vals), "unknown", dtype=object)
    for i, raw in enumerate(vals):
        if raw is None:
            continue
        s = str(raw).strip().lower()
        if not s or s in ("nan", "none", "unknown", "not reported", "not recorded", "declined"):
            continue
        if "hispanic" in s or "latino" in s:
            out[i] = "Hispanic"
        elif "white" in s or "caucasian" in s:
            out[i] = "White"
        elif "black" in s or "african" in s:
            out[i] = "Black"
        elif "asian" in s:
            out[i] = "Asian"
        else:
            out[i] = "Other"
    return out


def insurance_bucket(coverage_values) -> np.ndarray:
    """Bucket raw CoverageClass into Medicare / Medicaid / Commercial / Uninsured / Other / unknown.

    Case-insensitive substring rules (Medicaid tested before Medicare so neither shadows the
    other). "self"/"self-pay"/"uninsured"/"no insurance" -> Uninsured. Blank / "unknown" ->
    "unknown". Returns an object array aligned to input.
    """
    vals = list(coverage_values)
    out = np.full(len(vals), "unknown", dtype=object)
    for i, raw in enumerate(vals):
        if raw is None:
            continue
        s = str(raw).strip().lower()
        if not s or s in ("nan", "none", "unknown", "not reported"):
            continue
        if "medicaid" in s:
            out[i] = "Medicaid"
        elif "medicare" in s:
            out[i] = "Medicare"
        elif "commercial" in s or "private" in s or "hmo" in s or "ppo" in s:
            out[i] = "Commercial"
        elif "self" in s or "uninsured" in s or "no insurance" in s:
            out[i] = "Uninsured"
        else:
            out[i] = "Other"
    return out


def compute_gap(values_by_subgroup: dict, most_vuln: str, least_vuln: str) -> float:
    """gap = value(most_vulnerable) - value(least_vulnerable); NaN if either is missing/NaN.

    PURE. ``values_by_subgroup`` maps subgroup label -> scalar metric value.
    """
    a = _to_float(values_by_subgroup.get(most_vuln, _NAN))
    b = _to_float(values_by_subgroup.get(least_vuln, _NAN))
    if not (np.isfinite(a) and np.isfinite(b)):
        return _NAN
    return float(a - b)


def build_axes(dataset, test_idx: np.ndarray) -> dict:
    """Return {axis_name: label_array_aligned_to_test_rows} for every axis whose column exists.

    Slices each raw column to ``test_idx`` and applies the matching pure helper. An axis whose
    source column is absent (gb.frame_* -> None) is omitted.
    """
    test_idx = np.asarray(test_idx)
    axes = {}
    svi = gb.frame_feature(dataset, "SviOverall")
    if svi is not None:
        axes["svi"] = svi_quartiles(np.asarray(svi)[test_idx])
    ruca = gb.frame_categorical(dataset, "RUCA")
    if ruca is not None:
        axes["ruca"] = ruca_rural_urban(np.asarray(ruca, dtype=object)[test_idx])
    race = gb.frame_categorical(dataset, "FirstRace")
    if race is not None:
        axes["race"] = race_bucket(np.asarray(race, dtype=object)[test_idx])
    cov = gb.frame_categorical(dataset, "CoverageClass")
    if cov is not None:
        axes["insurance"] = insurance_bucket(np.asarray(cov, dtype=object)[test_idx])
    return axes


# --------------------------------------------------------------------------- #
# Internal numeric helpers
# --------------------------------------------------------------------------- #
def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def _sanitize(obj):
    """Recursively convert numpy scalars / arrays to JSON-native Python types (NaN kept)."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _horizon_stats(fac_dim: np.ndarray, obs_dim: np.ndarray, obs_mask: np.ndarray,
                   sub: np.ndarray) -> dict:
    """MAD / CRPS / 90% coverage for one horizon over subgroup rows with an observed outcome.

    ``fac_dim`` (n_test, s) predictive samples; ``obs_dim`` (n_test,) observed value;
    ``obs_mask`` (n_test,) bool 1=observed; ``sub`` (n_test,) bool subgroup membership.
    Guards < MIN_N observed rows -> NaN. Coverage is the empirical 90% predictive-band coverage.
    """
    rows = sub & obs_mask & np.isfinite(obs_dim)
    n = int(rows.sum())
    if n < MIN_N:
        return {"mad": _NAN, "crps": _NAN, "coverage": _NAN, "n": n}
    s = fac_dim[rows]
    y = obs_dim[rows]
    med = np.median(s, axis=1)
    mad = float(np.mean(np.abs(med - y)))
    crps = float(np.mean(bt.crps_ensemble(s, y)))
    lo, hi = ct.predictive_band(s, alpha=0.10)
    cov, _ = ct.coverage_from_band(lo, hi, y)
    return {"mad": mad, "crps": crps, "coverage": float(cov), "n": n}


def _pit_regime(fac_dim: np.ndarray, obs_dim: np.ndarray, obs_mask: np.ndarray,
                sub: np.ndarray) -> str:
    """PIT calibration regime label (calibrated / location-shift / under- / over-dispersion)."""
    rows = sub & obs_mask & np.isfinite(obs_dim)
    if int(rows.sum()) < MIN_N:
        return "insufficient-n"
    pit = ct.pit_values(fac_dim[rows], obs_dim[rows])
    return str(ct.classify_pit(pit).get("regime", "unknown"))


def _counterfactual_deltas(model, base_arrays, dataset, splits, gbm, sample_cfg, pre, device):
    """Per-patient risk-weighted RYGB-minus-SG threshold-prob delta for the headline targets.

    Samples the twin 4 times TOTAL (2 surgery arms x 2 event values) and reads every target off
    the cache (INTEGRATION_CONTRACT Section 7: 4 passes, NEVER per-horizon). Mirrors
    ``bmi_threshold_probability.cohort_probability``'s risk-weighting
    ``p_gbm*P(cross|e=1) + (1-p_gbm)*P(cross|e=0)`` at each clamped arm. Returns
    {horizon_name: per-patient delta ndarray over the test rows}, or None on any failure (this
    is the optional effect-heterogeneity analysis and must never break the audit).
    """
    try:
        test_idx = np.asarray(splits["test"])
        n = int(test_idx.size)
        estimator = gbm["estimator"]
        x, feature_names, _ = gb.assemble_features(dataset)
        surgery_col = feature_names.index("surgery_idx")
        blocks, p_arm = {}, {}
        for sidx in (0, 1):
            arrays = {**base_arrays,
                      "surgery_idx": np.full_like(base_arrays["surgery_idx"], sidx)}
            e0 = ev.twin_samples_15(model, arrays, np.zeros(n, dtype=np.float32),
                                    sample_cfg, pre, device)
            e1 = ev.twin_samples_15(model, arrays, np.ones(n, dtype=np.float32),
                                    sample_cfg, pre, device)
            blocks[sidx] = (e0, e1)
            xc = x.copy()
            xc[:, surgery_col] = float(sidx)
            p_arm[sidx] = estimator.predict_proba(xc[test_idx])[:, 1]
        deltas = {}
        for name, thr in (("bmi_12m", BMI_THRESHOLD), ("hba1c_12m", HBA1C_THRESHOLD)):
            dim = fm.TARGET_NAMES.index(name)
            frac = {}
            for sidx in (0, 1):
                e0, e1 = blocks[sidx]
                f0 = (e0[:, :, dim] < thr).mean(axis=1)
                f1 = (e1[:, :, dim] < thr).mean(axis=1)
                frac[sidx] = p_arm[sidx] * f1 + (1.0 - p_arm[sidx]) * f0
            deltas[name] = np.asarray(frac[1] - frac[0], dtype=float)  # RYGB - SG
        return deltas
    except Exception:  # noqa: BLE001  - optional analysis; degrade to None, never raise
        return None


# --------------------------------------------------------------------------- #
# Optional gap bar charts (guarded)
# --------------------------------------------------------------------------- #
def _plot_axis_gaps(axis: str, gap_rows: list, path: Path) -> bool:
    """Horizontal bar chart of an axis' metric gaps. Returns True if a PNG was written.

    Guarded: needs matplotlib and >= 1 finite gap. Units differ per metric (annotated), so the
    chart is a quick-look; ``fairness_gaps.csv`` is the source of truth.
    """
    finite = [r for r in gap_rows if np.isfinite(_to_float(r["gap"]))]
    if not finite:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    try:
        labels = [f"{r['metric']}\n({r['most_vuln']} - {r['least_vuln']})" for r in finite]
        values = [_to_float(r["gap"]) for r in finite]
        colors = ["tab:red" if (
            (ADVERSE_DIRECTION.get(r["metric"]) == "positive" and _to_float(r["gap"]) > 0) or
            (ADVERSE_DIRECTION.get(r["metric"]) == "negative" and _to_float(r["gap"]) < 0)
        ) else "tab:gray" for r in finite]
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.6 * len(finite) + 1.5)))
        ax.barh(range(len(finite)), values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(finite)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set(title=f"Fairness gaps: {axis} (red = adverse for vulnerable; units differ)",
               xlabel="gap = most-vulnerable - least-vulnerable")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Programmatic entrypoint (the Wave-3 evaluate_twin hook calls this EXACTLY)
# --------------------------------------------------------------------------- #
def run(*, dataset, splits, model, twin_cfg, pre, gbm, gbm_cfg, output_dir, device,
        n_samples=200, n_steps=50, seed=0, n_boot=1000) -> dict:
    """Run the subgroup equity audit on the TEST split and write the ``fairness_*`` artifacts.

    Reuses the already-loaded pipeline objects (so the freeze does not reload the twin). Samples
    the FACTUAL twin block ONCE, computes per-subgroup calibration / accuracy / discrimination
    across the SVI / RUCA / race / insurance axes, derives most-vs-least-vulnerable gaps, and
    returns a JSON-native manifest block with the headline gaps. Every statistic is guarded at
    MIN_N and degrades to NaN; a tiny test split (fake cohort) yields mostly NaN by design.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    test_idx = np.asarray(splits["test"])
    n_test = int(test_idx.size)

    # Observed outcomes / masks on the test set, ORIGINAL units (contract Section 6).
    X_test = np.asarray(dataset.x)[test_idx]        # (n_test, 17)
    M_test = np.asarray(dataset.mask)[test_idx]     # (n_test, 17), 1 = observed

    # Factual twin block, sampled ONCE (n_test, s, 15); continuous dims are 0..14 contiguous.
    base_arrays = ev.arrays_for(dataset, test_idx, pre)
    sample_cfg = replace(twin_cfg, n_samples_per_patient=n_samples, sample_steps=n_steps)
    fac = ev.twin_samples_15(model, base_arrays, base_arrays["y_mace"], sample_cfg, pre, device)

    # GBM composite-event predictions + labels, test-aligned.
    p_gbm = np.asarray(gbm["test_cal"] if gbm.get("calibrated") else gbm["test_raw"], dtype=float)
    y_gbm = np.asarray(gbm["y"])[test_idx]

    # Optional per-patient counterfactual effect (4 extra twin passes total), guarded.
    cf_deltas = _counterfactual_deltas(model, base_arrays, dataset, splits, gbm,
                                       sample_cfg, pre, device)

    axes = build_axes(dataset, test_idx)

    perf_rows, thr_rows, repr_rows, effect_rows = [], [], [], []
    perf_lookup: dict = {}          # axis -> subgroup -> {metric: value}
    subgroup_counts: dict = {}      # axis -> subgroup -> n

    def _gbm_disc(sub: np.ndarray) -> dict:
        ys, ps = y_gbm[sub], p_gbm[sub]
        n = int(ys.size)
        if n < MIN_N:
            return {"gbm_auroc": _NAN, "gbm_auroc_lo": _NAN, "gbm_auroc_hi": _NAN,
                    "gbm_auprc": _NAN, "gbm_brier": _NAN, "n": n}
        auroc, lo, hi = ev.bootstrap_ci(ys, ps, roc_auc_score, n_boot, seed)
        auprc, _, _ = ev.bootstrap_ci(ys, ps, average_precision_score, n_boot, seed + 1)
        brier, _, _ = ev.bootstrap_ci(ys, ps, brier_score_loss, n_boot, seed + 2)
        return {"gbm_auroc": _to_float(auroc), "gbm_auroc_lo": _to_float(lo),
                "gbm_auroc_hi": _to_float(hi), "gbm_auprc": _to_float(auprc),
                "gbm_brier": _to_float(brier), "n": n}

    for axis, labels in axes.items():
        perf_lookup[axis] = {}
        subgroup_counts[axis] = {}
        present = [s for s in AXIS_SUBGROUPS[axis] if np.any(labels == s)]
        # include any unexpected-but-present label (defensive; should not happen)
        present += [s for s in np.unique(labels) if s not in AXIS_SUBGROUPS[axis] and s != "unknown"]

        for subgroup in present:
            sub = np.asarray(labels == subgroup)
            n_sub = int(sub.sum())
            subgroup_counts[axis][subgroup] = n_sub
            vals: dict = {}

            # --- representation + differential attrition (descriptive; always reported) ---
            repr_row = {"axis": axis, "subgroup": subgroup, "n": n_sub,
                        "frac_of_test": float(n_sub / n_test) if n_test else _NAN}
            for h in FOLLOWUP_HORIZONS:
                d = fm.TARGET_NAMES.index(h)
                col = M_test[sub, d]
                repr_row[f"followup_{h}"] = float(np.mean(col)) if col.size else _NAN
            repr_rows.append(repr_row)

            # --- twin trajectory: headline horizons ---
            for hname, _group in HEADLINE_HORIZONS:
                d = fm.TARGET_NAMES.index(hname)
                st = _horizon_stats(fac[:, :, d], X_test[:, d], M_test[:, d] == 1, sub)
                for stat in ("coverage", "mad", "crps"):
                    metric = f"{stat}_{hname}"
                    vals[metric] = st[stat]
                    perf_rows.append({"axis": axis, "subgroup": subgroup, "metric": metric,
                                      "value": st[stat], "n": st["n"], "detail": ""})
                regime = _pit_regime(fac[:, :, d], X_test[:, d], M_test[:, d] == 1, sub)
                perf_rows.append({"axis": axis, "subgroup": subgroup,
                                  "metric": f"pit_regime_{hname}", "value": _NAN,
                                  "n": st["n"], "detail": regime})

            # --- twin trajectory: averaged over all 15 continuous horizons ---
            mad_h, crps_h, cov_h = [], [], []
            for d in range(fm.X_DIM):
                if d >= 15:  # 0..14 are the continuous BMI/HbA1c dims
                    break
                st = _horizon_stats(fac[:, :, d], X_test[:, d], M_test[:, d] == 1, sub)
                if np.isfinite(st["mad"]):
                    mad_h.append(st["mad"])
                if np.isfinite(st["crps"]):
                    crps_h.append(st["crps"])
                if np.isfinite(st["coverage"]):
                    cov_h.append(st["coverage"])
            for stat, bucket in (("coverage", cov_h), ("mad", mad_h), ("crps", crps_h)):
                metric = f"{stat}_avg"
                value = float(np.mean(bucket)) if bucket else _NAN
                vals[metric] = value
                perf_rows.append({"axis": axis, "subgroup": subgroup, "metric": metric,
                                  "value": value, "n": n_sub, "detail": ""})

            # --- threshold reliability (ECE / Brier) for the headline threshold targets ---
            for label, hname, thr in THRESHOLD_TARGETS:
                d = fm.TARGET_NAMES.index(hname)
                p_pred = (fac[:, :, d] < thr).mean(axis=1)
                y_cross = (X_test[:, d] < thr).astype(float)
                rows_obs = sub & (M_test[:, d] == 1) & np.isfinite(X_test[:, d])
                n_obs = int(rows_obs.sum())
                res = dm.threshold_calibration(p_pred[rows_obs], y_cross[rows_obs], w=None)
                ece, brier = _to_float(res["ece"]), _to_float(res["brier"])
                thr_rows.append({"axis": axis, "subgroup": subgroup, "target": label,
                                 "ece": ece, "brier": brier, "n": n_obs})
                vals[f"ece_{label}"] = ece
                vals[f"brier_{label}"] = brier
                perf_rows.append({"axis": axis, "subgroup": subgroup, "metric": f"ece_{label}",
                                  "value": ece, "n": n_obs, "detail": ""})
                perf_rows.append({"axis": axis, "subgroup": subgroup, "metric": f"brier_{label}",
                                  "value": brier, "n": n_obs, "detail": ""})

            # --- GBM composite-event discrimination ---
            disc = _gbm_disc(sub)
            for metric in ("gbm_auroc", "gbm_auroc_lo", "gbm_auroc_hi", "gbm_auprc", "gbm_brier"):
                vals[metric] = disc[metric]
                perf_rows.append({"axis": axis, "subgroup": subgroup, "metric": metric,
                                  "value": disc[metric], "n": disc["n"], "detail": ""})

            # --- optional per-subgroup counterfactual effect (RYGB - SG) ---
            if cf_deltas is not None:
                for hname, thr in (("bmi_12m", BMI_THRESHOLD), ("hba1c_12m", HBA1C_THRESHOLD)):
                    d_arr = cf_deltas.get(hname)
                    if d_arr is None:
                        continue
                    sub_vals = d_arr[sub]
                    mean_delta = float(np.mean(sub_vals)) if n_sub >= MIN_N else _NAN
                    effect_rows.append({"axis": axis, "subgroup": subgroup,
                                        "target": f"P({hname}<{thr:g}) rygb-minus-sg",
                                        "mean_rygb_minus_sg_prob": mean_delta, "n": n_sub})

            perf_lookup[axis][subgroup] = vals

    # --- gaps: most-vulnerable minus least-vulnerable, per axis / metric / reference pair ---
    gap_rows = []
    gaps_by_axis: dict = {}
    for axis in perf_lookup:
        gaps_by_axis[axis] = []
        for most, least in GAP_PAIRS.get(axis, []):
            for metric in GAP_METRICS:
                mv = {s: perf_lookup[axis][s].get(metric, _NAN) for s in perf_lookup[axis]}
                gap = compute_gap(mv, most, least)
                row = {"axis": axis, "metric": metric, "most_vuln": most, "least_vuln": least,
                       "gap": gap, "adverse_direction": ADVERSE_DIRECTION.get(metric, "")}
                gap_rows.append(row)
                gaps_by_axis[axis].append(row)

    # --- write artifacts (write FIRST, then announce via report_saved) ---
    artifacts = []

    def _write(df: pd.DataFrame, name: str, desc: str):
        path = output_dir / name
        df.to_csv(path, index=False)
        evfm.report_saved(path, desc)
        artifacts.append(str(path))

    perf_cols = ["axis", "subgroup", "metric", "value", "n", "detail"]
    _write(pd.DataFrame(perf_rows, columns=perf_cols),
           "fairness_subgroup_performance.csv", "per-subgroup performance (long)")
    _write(pd.DataFrame(gap_rows, columns=["axis", "metric", "most_vuln", "least_vuln",
                                           "gap", "adverse_direction"]),
           "fairness_gaps.csv", "most-vs-least-vulnerable gaps")
    _write(pd.DataFrame(thr_rows, columns=["axis", "subgroup", "target", "ece", "brier", "n"]),
           "fairness_threshold_reliability.csv", "per-subgroup threshold reliability (ECE/Brier)")
    repr_cols = ["axis", "subgroup", "n", "frac_of_test"] + [f"followup_{h}" for h in FOLLOWUP_HORIZONS]
    _write(pd.DataFrame(repr_rows, columns=repr_cols),
           "fairness_representation.csv", "per-subgroup representation + follow-up completeness")
    if cf_deltas is not None:
        _write(pd.DataFrame(effect_rows, columns=["axis", "subgroup", "target",
                                                  "mean_rygb_minus_sg_prob", "n"]),
               "fairness_effect_by_subgroup.csv", "per-subgroup counterfactual RYGB-vs-SG effect")

    # --- optional gap bar charts (guarded) ---
    for axis, rows in gaps_by_axis.items():
        png = output_dir / f"fairness_gaps_{axis}.png"
        if _plot_axis_gaps(axis, rows, png):
            evfm.report_saved(png, f"fairness gap bar chart ({axis})")
            artifacts.append(str(png))

    # --- headline gaps for the manifest ---
    def _gap(axis, metric, most, least):
        if axis not in perf_lookup:
            return _NAN
        mv = {s: perf_lookup[axis][s].get(metric, _NAN) for s in perf_lookup[axis]}
        return compute_gap(mv, most, least)

    # race headline: the minority bucket with the largest n vs White (most-supported comparison)
    race_minority = None
    if "race" in subgroup_counts:
        minorities = {s: c for s, c in subgroup_counts["race"].items() if s != "White"}
        if minorities:
            race_minority = max(minorities, key=minorities.get)
    race_auroc_gap = _gap("race", "gbm_auroc", race_minority, "White") if race_minority else _NAN

    headline_gaps = {
        "svi_coverage_gap": _gap("svi", "coverage_avg", "Q4", "Q1"),
        "svi_auroc_gap": _gap("svi", "gbm_auroc", "Q4", "Q1"),
        "svi_ece_gap": _gap("svi", "ece_bmi35_12m", "Q4", "Q1"),
        "ruca_coverage_gap": _gap("ruca", "coverage_avg", "rural", "urban"),
        "ruca_ece_gap": _gap("ruca", "ece_bmi35_12m", "rural", "urban"),
        "ruca_auroc_gap": _gap("ruca", "gbm_auroc", "rural", "urban"),
        "race_auroc_gap": race_auroc_gap,
        "race_auroc_gap_bucket": race_minority,
        "insurance_auroc_gap": _gap("insurance", "gbm_auroc", "Medicaid", "Commercial"),
    }

    manifest = {
        "axes": list(perf_lookup.keys()),
        "n_test": n_test,
        "n_samples": int(n_samples),
        "n_steps": int(n_steps),
        "min_n_guard": MIN_N,
        "subgroup_counts": subgroup_counts,
        "headline_gaps": headline_gaps,
        "gap_metrics": GAP_METRICS,
        "adverse_direction": ADVERSE_DIRECTION,
        "counterfactual_effect_included": cf_deltas is not None,
        "artifacts": artifacts,
        "axis_definitions": {
            "svi": "SviOverall within-test quartiles Q1(least)..Q4(most vulnerable); gap Q4-Q1",
            "ruca": "RUCA leading code 1-3=urban, 4-10=rural; gap rural-urban",
            "race": "FirstRace -> White/Black/Hispanic/Asian/Other (AUDIT ONLY); gaps minority-White",
            "insurance": "CoverageClass -> Medicare/Medicaid/Commercial/Uninsured/Other; gaps Medicaid|Uninsured - Commercial",
        },
        "note": ("audit only; race never a model input (FirstRace bucketed for equity reporting "
                 "only). Every statistic guarded at n<%d -> NaN; a tiny test split yields mostly "
                 "NaN by design." % MIN_N),
    }
    return _sanitize(manifest)


# --------------------------------------------------------------------------- #
# CLI (mirrors bmi_threshold_probability.main)
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", type=str, default=None,
                        help="pipeline dir with manifest.json (gives gbm + twin run dirs)")
    parser.add_argument("--twin-run", type=str, default=None)
    parser.add_argument("--gbm-run", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.pipeline:
        manifest = ev.resolve_from_pipeline(Path(args.pipeline))
        gbm_run_dir = Path(args.gbm_run or manifest["gbm_run_dir"])
        twin_run_dir = Path(args.twin_run or manifest["twin_final_run_dir"])
    else:
        if not (args.twin_run and args.gbm_run):
            raise SystemExit("Provide --pipeline, or both --twin-run and --gbm-run.")
        gbm_run_dir, twin_run_dir = Path(args.gbm_run), Path(args.twin_run)

    device = ev.choose_device(args.device)
    dataset = ev.load_dataset(Path(args.csv) if args.csv else None)

    gbm_cfg = ev.load_gbm_config(gbm_run_dir)
    splits = gb.make_splits(dataset, gbm_cfg)

    twin_cfg = ev.load_twin_config(twin_run_dir)
    pre = ev.load_twin_preprocessing(twin_run_dir)
    model = ev.restore_twin(twin_run_dir, twin_cfg, device)
    gbm = ev.compute_gbm_predictions(gbm_cfg, dataset, splits)

    result = run(dataset=dataset, splits=splits, model=model, twin_cfg=twin_cfg, pre=pre,
                 gbm=gbm, gbm_cfg=gbm_cfg, output_dir=Path(args.output_dir), device=device,
                 n_samples=args.n_samples, n_steps=args.n_steps, seed=args.seed, n_boot=args.n_boot)

    print("\n=== fairness audit manifest ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
