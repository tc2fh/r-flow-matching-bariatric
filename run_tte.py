"""run_tte.py - orchestrator for the twin's TTE (causal) + distributional evaluation.

This is PART C of the TTE + distributional build (see TTE_DISTRIBUTIONAL_RUN_PLAN.md
and the INTEGRATION_CONTRACT). It is a THIN orchestrator that ties the two Wave-1
pure-metric modules to the trained digital twin and emits every ``dist_*``
(distributional) and ``tte_*`` (causal target-trial-emulation) artifact. Both families
are consolidated here so the Wave-3 evaluate_twin hook is a one-liner
(``run_tte.run(...)``).

What it does NOT do (by contract):
  * It never edits an existing file - it only imports/reuses them.
  * It never retrains the twin. The twin is the g-computation outcome model E[Y|A=a,L];
    the per-arm mu1/mu0 are built here from the surgery clamp (the
    ``bmi_threshold_probability.cohort_probability`` pattern), refactored to sample the
    twin ONLY 5 times total (4 arm x event blocks + 1 factual block) and read every
    horizon/threshold/reducer off the cache.
  * The causal estimators (propensity / IPTW / IPCW / AIPW / E-value / c-for-benefit)
    and the distributional scores stay PURE in ``causal_tte`` / ``distributional_metrics``;
    this file only wires data into them and writes CSVs/PNGs.

Nuisance models (propensity, censoring) are fit on TRAIN and applied to TEST, inheriting
the frozen ``split_strategy`` via ``gb.make_splits(dataset, gbm_cfg)`` so a temporal run
carries through for free. Every statistic degrades to NaN below ~10 usable observations
(the Wave-1 functions guard this); PNGs are guarded so a plotting failure never loses the
run. See the module docstrings of ``causal_tte`` / ``distributional_metrics`` for the
ABSENT-confounder caveat (GERD, surgeon/center, smoking) and the DECISIONS.

CLI (mirrors bmi_threshold_probability.main)::

    OMP_NUM_THREADS=1 python run_tte.py --pipeline runs/twin_pipeline/<dir> \
        --csv fake_data/fake_mbs_cohort.csv --output-dir out/ --n-samples 200 --n-steps 50

Programmatic entrypoint (the Wave-3 hook calls this with already-loaded objects)::

    run(*, dataset, splits, model, twin_cfg, pre, gbm, gbm_cfg, output_dir, device,
        n_samples=200, n_steps=50, seed=0, with_causal=True) -> dict
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import pandas as pd
import torch

import evaluate_twin as ev
import evaluate_flow_matching as evfm  # report_saved (write-then-announce)
import gbm_mace_baseline as gb
import train_flow_matching as fm
import causal_tte as ct2
import distributional_metrics as dm

try:  # AUC is a nice-to-have for the PS-model summary; guard the import defensively.
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


# ------------------------------------------------------------------------------------- #
# Module constants (easy to expand). Horizons auto-reduce when the test set is tiny.
# ------------------------------------------------------------------------------------- #
BMI_THRESHOLD = 35.0        # BMI < 35 = below Class-II obesity
HBA1C_THRESHOLD = 5.7       # HbA1c < 5.7% = back into the non-diabetic range
AUTO_REDUCE_N = 20          # below this many test patients, collapse each horizon family

# Continuous ATE contrasts (group, horizon). mean reducer; Y = observed BMI/HbA1c.
CONT_ATE_TARGETS = [
    ("bmi", "bmi_12m"), ("bmi", "bmi_2y"), ("bmi", "bmi_5y"),
    ("hba1c", "hba1c_12m"), ("hba1c", "hba1c_2y"),
]
# Threshold risk-difference contrasts (group, horizon, threshold). Y = 1{crossed}.
THRESHOLD_RD_TARGETS = [
    ("bmi", "bmi_12m", BMI_THRESHOLD), ("bmi", "bmi_2y", BMI_THRESHOLD),
    ("hba1c", "hba1c_12m", HBA1C_THRESHOLD), ("hba1c", "hba1c_2y", HBA1C_THRESHOLD),
]
# Distributional per-horizon scoring families (full clinical families).
DIST_BMI_HORIZONS = ["bmi_3m", "bmi_6m", "bmi_9m", "bmi_12m", "bmi_2y",
                     "bmi_3y", "bmi_4y", "bmi_5y", "bmi_6y"]
DIST_HBA1C_HORIZONS = ["hba1c_12m", "hba1c_2y", "hba1c_3y", "hba1c_4y", "hba1c_5y", "hba1c_6y"]
COVERAGE_HORIZONS = ["bmi_12m", "bmi_2y", "bmi_5y", "hba1c_12m", "hba1c_2y"]
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
THRESHOLD_CAL_FAMILIES = [
    ("bmi35", BMI_THRESHOLD, ["bmi_12m", "bmi_2y"]),
    ("hba1c57", HBA1C_THRESHOLD, ["hba1c_12m", "hba1c_2y"]),
]
ATTRITION_HORIZONS = ["bmi_12m", "bmi_2y", "hba1c_12m", "hba1c_2y"]

ABSENT_CONFOUNDERS = ["GERD", "surgeon_or_center_id", "smoking"]
# Sourced from the causal_tte module constants (single source of truth for the coding).
TREATMENT_CODING = {"sleeve": sorted(ct2.SLEEVE_CPTS), "rygb": sorted(ct2.RYGB_CPTS)}
PS_TRIM = (0.02, 0.98)
SEEDS = {"ps": 0, "censoring": 0}


# ------------------------------------------------------------------------------------- #
# Small IO / plotting helpers
# ------------------------------------------------------------------------------------- #
def _save_csv(df: pd.DataFrame, path: Path, desc: str) -> str:
    df.to_csv(path, index=False)
    evfm.report_saved(path, desc)
    return str(path)


def _save_json(obj: dict, path: Path, desc: str) -> str:
    path.write_text(json.dumps(obj, indent=2, default=_jsonify), encoding="utf-8")
    evfm.report_saved(path, desc)
    return str(path)


def _jsonify(o):
    """Fallback for json.dumps -> make numpy scalars/arrays / paths JSON-native."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return float(o)


def _safe_plot(fn, *args, **kwargs) -> bool:
    """Run a plotting function; a failure logs and continues (never loses the run)."""
    try:
        fn(*args, **kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"[run_tte] plot skipped ({getattr(fn, '__name__', fn)}): {exc}", stacklevel=2)
        return False


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _first_per_group(items, key):
    seen, out = set(), []
    for it in items:
        g = key(it)
        if g not in seen:
            out.append(it)
            seen.add(g)
    return out


def _dim(name: str) -> int:
    return fm.TARGET_NAMES.index(name)


# ------------------------------------------------------------------------------------- #
# Programmatic entrypoint
# ------------------------------------------------------------------------------------- #
def run(*, dataset, splits, model, twin_cfg, pre, gbm, gbm_cfg, output_dir, device,
        n_samples: int = 200, n_steps: int = 50, seed: int = 0, with_causal: bool = True) -> dict:
    """Emit dist_* (always, naive) + tte_* (when with_causal) artifacts; return a manifest.

    Parameters mirror the locals live inside ``evaluate_twin.evaluate`` at the hook point
    (contract Section 5), so the Wave-3 call is a one-liner. ``splits`` is used AS-IS (the
    twin's shared split); ``gbm`` is the ``ev.compute_gbm_predictions`` dict.

    Returns a JSON-native dict with two blocks: ``distributional`` (artifact paths +
    headline coverage/CRPS/ECE) and ``causal_tte`` (treatment coding, PS-model summary,
    weight ESS, marginal-effect headlines, absent confounders, seeds). When
    ``with_causal=False`` the ``causal_tte`` block is ``{"status": "skipped"}`` but the
    naive dist_* artifacts still emit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    test_idx = np.asarray(splits["test"], dtype=np.int64)
    n_test = int(test_idx.shape[0])

    sample_cfg = replace(twin_cfg, n_samples_per_patient=n_samples, sample_steps=n_steps)
    base_arrays = ev.arrays_for(dataset, test_idx, pre)

    # GBM design matrix + treatment column (for the per-arm outcome-model clamp).
    x, feat, _ = gb.assemble_features(dataset)
    x = np.asarray(x, dtype=float)
    surgery_col = feat.index("surgery_idx")
    estimator = gbm["estimator"]

    # --- sample the twin FIVE times total (contract Section 7), cache, reuse everywhere.
    blk = {}
    for arm in (0, 1):
        arrays_arm = {**base_arrays, "surgery_idx": np.full_like(base_arrays["surgery_idx"], arm)}
        for ev_val in (0, 1):
            event = np.full(n_test, ev_val, dtype=np.float32)
            blk[(arm, ev_val)] = ev.twin_samples_15(model, arrays_arm, event, sample_cfg, pre, device)
    fac = ev.twin_samples_15(model, base_arrays, base_arrays["y_mace"], sample_cfg, pre, device)
    print(f"[run_tte] twin sampled: 4 arm x event blocks + 1 factual block, "
          f"shape (n_test={n_test}, s={n_samples}, 15); with_causal={with_causal}")

    # Auto-reduce horizon sets on a tiny test cohort (keeps the smoke fast; every file
    # still gets rows). Twin sampling above is unaffected (already 5 passes total).
    reduce = n_test < AUTO_REDUCE_N
    cont_targets = _first_per_group(CONT_ATE_TARGETS, lambda t: t[0]) if reduce else CONT_ATE_TARGETS
    thr_targets = _first_per_group(THRESHOLD_RD_TARGETS, lambda t: t[0]) if reduce else THRESHOLD_RD_TARGETS
    bmi_dist = ["bmi_12m"] if reduce else DIST_BMI_HORIZONS
    hba1c_dist = ["hba1c_12m"] if reduce else DIST_HBA1C_HORIZONS
    cov_horizons = ["bmi_12m", "hba1c_12m"] if reduce else COVERAGE_HORIZONS
    thr_cal_families = ([("bmi35", BMI_THRESHOLD, ["bmi_12m"]),
                         ("hba1c57", HBA1C_THRESHOLD, ["hba1c_12m"])] if reduce else THRESHOLD_CAL_FAMILIES)
    attrition_horizons = ["bmi_12m", "hba1c_12m"] if reduce else ATTRITION_HORIZONS
    modec_horizons = (["bmi_12m", "hba1c_12m"] if reduce
                      else (DIST_BMI_HORIZONS + DIST_HBA1C_HORIZONS))

    # --- causal nuisance models (fit TRAIN, apply TEST). Cached per-horizon censoring.
    L = A = ps = None
    cens_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if with_causal:
        L, A, L_names = ct2.build_L_A(dataset)
        ps, _ps_clf = ct2.propensity_scores(L, A, splits["train"], splits["test"])
    else:
        L_names = []

    def _censoring(dim: int):
        """P(observed at horizon | L) + IPCW on TEST, memoized by dim (with_causal only)."""
        if dim not in cens_cache:
            cens_cache[dim] = ct2.censoring_model(
                L, dataset.mask[:, dim].astype(int), splits["train"], splits["test"])
        return cens_cache[dim]

    def _mu_arm(dim: int, reducer):
        """Risk-weighted per-arm outcome model {0: mu_SG, 1: mu_RYGB} from the twin cache."""
        out = {}
        for arm in (0, 1):
            xc = x.copy()
            xc[:, surgery_col] = float(arm)
            p_gbm = estimator.predict_proba(xc[test_idx])[:, 1]
            red1 = reducer(blk[(arm, 1)][:, :, dim])
            red0 = reducer(blk[(arm, 0)][:, :, dim])
            out[arm] = p_gbm * red1 + (1.0 - p_gbm) * red0
        return out[0], out[1]

    ctx = dict(
        dataset=dataset, splits=splits, test_idx=test_idx, n_test=n_test, x=x,
        surgery_col=surgery_col, estimator=estimator, blk=blk, fac=fac, gbm=gbm,
        gbm_cfg=gbm_cfg, with_causal=with_causal, output_dir=output_dir,
        L=L, A=A, ps=ps, L_names=L_names, censoring=_censoring, mu_arm=_mu_arm,
        cont_targets=cont_targets, thr_targets=thr_targets, bmi_dist=bmi_dist,
        hba1c_dist=hba1c_dist, cov_horizons=cov_horizons, thr_cal_families=thr_cal_families,
        attrition_horizons=attrition_horizons, modec_horizons=modec_horizons,
    )

    # follow-up completeness (for attrition-stratified calibration) computed once.
    cont_dims_all = [_dim(nm) for nm in (DIST_BMI_HORIZONS + DIST_HBA1C_HORIZONS)]
    completeness = dataset.mask[test_idx][:, cont_dims_all].sum(axis=1)
    med = float(np.median(completeness)) if completeness.size else 0.0
    ctx["has_long_followup"] = completeness >= med

    dist_block = _run_distributional(ctx)
    causal_block = _run_causal(ctx) if with_causal else {"status": "skipped"}

    manifest = {"distributional": dist_block, "causal_tte": causal_block}
    _save_json(manifest, output_dir / "run_tte_manifest.json", "run_tte manifest (dist_* + tte_*)")
    return manifest


# ------------------------------------------------------------------------------------- #
# DISTRIBUTIONAL (dist_*) - naive always; IPCW only when with_causal
# ------------------------------------------------------------------------------------- #
def _run_distributional(ctx) -> dict:
    dataset, test_idx, fac = ctx["dataset"], ctx["test_idx"], ctx["fac"]
    out_dir, with_causal = ctx["output_dir"], ctx["with_causal"]
    art: dict[str, str] = {}
    headline: dict[str, float] = {}

    def mask_at(dim):
        return dataset.mask[test_idx, dim].astype(bool)

    def obs_at(dim):
        """Observed outcome with the loader's zero-filled missing values restored to NaN."""
        values = dataset.x[test_idx, dim].astype(float)
        return np.where(mask_at(dim), values, np.nan)

    def obs_block_at(dims):
        """Observed outcome block with any unobserved cells represented as NaN."""
        values = dataset.x[test_idx][:, dims].astype(float)
        observed = dataset.mask[test_idx][:, dims].astype(bool)
        return np.where(observed, values, np.nan)

    def ipcw_at(dim):
        if not with_causal:
            return None
        p_obs, _ = ctx["censoring"](dim)
        return dm.ipcw_from_model(p_obs, mask_at(dim))

    # ---- dist_proper_scores_test.csv : per-horizon CRPS / log / interval / pinball /
    #      sharpness (naive + IPCW) + block energy/variogram per group.
    rows = []
    for group, horizons in (("bmi", ctx["bmi_dist"]), ("hba1c", ctx["hba1c_dist"])):
        dims = [_dim(nm) for nm in horizons]
        block = fac[:, :, dims]                                   # (n, s, k)
        obs_block = obs_block_at(dims)                            # (n, k), NaN if unobserved
        es_naive, _ = dm.energy_score_block(block, obs_block, w=None)
        vs_naive, _ = dm.variogram_score_block(block, obs_block, w=None)
        es_ipcw = vs_ipcw = float("nan")
        if with_causal:
            # Block IPCW: weight by the LONGEST horizon's censoring (the binding attrition
            # constraint for a complete-case block); approximate, documented as such.
            long_dim = dims[-1]
            p_obs_long, _ = ctx["censoring"](long_dim)
            all_obs = dataset.mask[test_idx][:, dims].all(axis=1)
            w_blk = dm.ipcw_from_model(p_obs_long, all_obs)
            es_ipcw, _ = dm.energy_score_block(block, obs_block, w=w_blk)
            vs_ipcw, _ = dm.variogram_score_block(block, obs_block, w=w_blk)
        for nm, dim in zip(horizons, dims):
            s_h, obs, msk = fac[:, :, dim], obs_at(dim), mask_at(dim)
            rep_n = dm.proper_scores_report(s_h, obs, mask=msk, w=None)
            rep_w = (dm.proper_scores_report(s_h, obs, mask=msk, w=ipcw_at(dim))
                     if with_causal else None)

            def _pin_mean(rep):
                vals = [v for v in rep["pinball"].values() if np.isfinite(v)]
                return float(np.mean(vals)) if vals else float("nan")

            rows.append({
                "group": group, "horizon": nm, "n_obs": int(rep_n["n"]),
                "crps_naive": rep_n["crps"], "crps_ipcw": (rep_w["crps"] if rep_w else float("nan")),
                "logscore_naive": rep_n["log_score"],
                "logscore_ipcw": (rep_w["log_score"] if rep_w else float("nan")),
                "interval_naive": rep_n["interval_score"],
                "interval_ipcw": (rep_w["interval_score"] if rep_w else float("nan")),
                "pinball_mean_naive": _pin_mean(rep_n),
                "pinball_mean_ipcw": (_pin_mean(rep_w) if rep_w else float("nan")),
                "sharpness_sd": rep_n["sharpness"]["mean_sd"],
                "sharpness_width90": rep_n["sharpness"]["mean_width90"],
                "energy_block_naive": es_naive, "energy_block_ipcw": es_ipcw,
                "variogram_block_naive": vs_naive, "variogram_block_ipcw": vs_ipcw,
            })
            if nm in ("bmi_12m", "hba1c_12m"):
                headline[f"crps_{nm}"] = rep_n["crps"]
    art["dist_proper_scores_test"] = _save_csv(
        pd.DataFrame(rows), out_dir / "dist_proper_scores_test.csv",
        "distributional proper scores (per horizon; naive + IPCW)")

    # ---- dist_coverage_curve_test.csv (+png)
    cov_rows = []
    cov_by_h = {}
    for nm in ctx["cov_horizons"]:
        dim = _dim(nm)
        s_h, obs = fac[:, :, dim], obs_at(dim)
        naive = dm.coverage_curve(s_h, obs, levels=COVERAGE_LEVELS, w=None)
        wtd = (dm.coverage_curve(s_h, obs, levels=COVERAGE_LEVELS, w=ipcw_at(dim))
               if with_causal else None)
        cov_by_h[nm] = naive
        for i, r in enumerate(naive):
            row = {"horizon": nm, "nominal": r["nominal"],
                   "empirical_naive": r["empirical"], "mean_width_naive": r["mean_width"]}
            if wtd is not None:
                row["empirical_ipcw"] = wtd[i]["empirical"]
                row["mean_width_ipcw"] = wtd[i]["mean_width"]
            cov_rows.append(row)
            if nm in ("bmi_12m", "hba1c_12m") and abs(r["nominal"] - 0.90) < 1e-9:
                headline[f"coverage90_{nm}"] = r["empirical"]
    art["dist_coverage_curve_test"] = _save_csv(
        pd.DataFrame(cov_rows), out_dir / "dist_coverage_curve_test.csv",
        "distributional coverage curve (nominal vs empirical)")
    png = out_dir / "dist_coverage_curve_test.png"
    if _safe_plot(_plot_coverage, cov_by_h, png):
        evfm.report_saved(png, "coverage curve plot")
        art["dist_coverage_curve_png"] = str(png)

    # ---- dist_threshold_calibration_{bmi35,hba1c57}_test.csv  +  dist_calibration_slope_citl.csv
    split_strategy = getattr(ctx["gbm_cfg"], "split_strategy", "unknown")
    slope_rows = []
    for family, thr, horizons in ctx["thr_cal_families"]:
        tc_rows = []
        for nm in horizons:
            dim = _dim(nm)
            s_h = fac[:, :, dim]
            p_pred = (s_h < thr).mean(axis=1)                    # per-patient factual P(cross)
            msk = mask_at(dim)
            xv = dataset.x[test_idx, dim].astype(float)
            with np.errstate(invalid="ignore"):
                y_obs = np.where(msk, (xv < thr).astype(float), np.nan)
            variants = [("naive", None)]
            if with_causal:
                variants.append(("ipcw", ipcw_at(dim)))
            for wtag, w in variants:
                cal = dm.threshold_calibration(p_pred, y_obs, w=w)
                slope = dm.calibration_slope_intercept(p_pred, y_obs, w=w)
                n_used = int(np.sum(np.isfinite(y_obs)))
                slope_rows.append({
                    "family": family, "horizon": nm, "weighting": wtag, "threshold": thr,
                    "slope": slope["slope"], "citl": slope["citl"], "n_obs": n_used,
                    "split_strategy": split_strategy})
                if cal["table"]:
                    for b in cal["table"]:
                        tc_rows.append({
                            "family": family, "horizon": nm, "weighting": wtag, "threshold": thr,
                            "n": int(np.sum(np.isfinite(y_obs))), "ece": cal["ece"],
                            "mce": cal["mce"], "brier": cal["brier"], "bin": b["bin"],
                            "bin_pred": b["pred"], "bin_obs": b["obs"], "bin_n": b["n"],
                            "bin_weight": b["weight"]})
                else:                                            # guarded (tiny n): summary row
                    tc_rows.append({
                        "family": family, "horizon": nm, "weighting": wtag, "threshold": thr,
                        "n": int(np.sum(np.isfinite(y_obs))), "ece": cal["ece"], "mce": cal["mce"],
                        "brier": cal["brier"], "bin": np.nan, "bin_pred": np.nan,
                        "bin_obs": np.nan, "bin_n": np.nan, "bin_weight": np.nan})
                if wtag == "naive" and nm in ("bmi_12m", "hba1c_12m"):
                    headline[f"ece_{family}_{nm}"] = cal["ece"]
        art[f"dist_threshold_calibration_{family}_test"] = _save_csv(
            pd.DataFrame(tc_rows), out_dir / f"dist_threshold_calibration_{family}_test.csv",
            f"threshold calibration reliability + ECE/MCE/Brier ({family})")
    art["dist_calibration_slope_citl"] = _save_csv(
        pd.DataFrame(slope_rows), out_dir / "dist_calibration_slope_citl.csv",
        "calibration slope + CITL (current fold; drift is a cross-run delta computed later)")

    # ---- dist_attrition_sensitivity.csv : naive-vs-IPCW pair per headline metric +
    #      follow-up-stratified threshold calibration.
    at_rows = []
    for nm in ctx["attrition_horizons"]:
        dim = _dim(nm)
        s_h, obs, msk = fac[:, :, dim], obs_at(dim), mask_at(dim)
        w = ipcw_at(dim)
        crps_n = dm.proper_scores_report(s_h, obs, mask=msk, w=None)["crps"]
        crps_w = dm.proper_scores_report(s_h, obs, mask=msk, w=w)["crps"] if with_causal else float("nan")
        int_n, _ = dm.interval_score(s_h, np.where(msk, obs, np.nan), w=None)
        int_w = (dm.interval_score(s_h, np.where(msk, obs, np.nan), w=w)[0]
                 if with_causal else float("nan"))
        for metric, vn, vw in (("crps", crps_n, crps_w), ("interval_score", int_n, int_w)):
            at_rows.append({"horizon": nm, "metric": metric, "value_naive": vn, "value_ipcw": vw,
                            "naive_minus_ipcw_gap": (vn - vw) if np.isfinite(vn) and np.isfinite(vw) else np.nan,
                            "long_fu": np.nan, "short_fu": np.nan, "long_minus_short_gap": np.nan,
                            "note": "attrition sensitivity: naive (observed-only) vs IPCW-weighted"})
        # threshold ECE naive vs IPCW + stratified by follow-up completeness.
        thr = BMI_THRESHOLD if nm.startswith("bmi") else HBA1C_THRESHOLD
        p_pred = (s_h < thr).mean(axis=1)
        xv = dataset.x[test_idx, dim].astype(float)
        with np.errstate(invalid="ignore"):
            y_obs = np.where(msk, (xv < thr).astype(float), np.nan)
        ece_n = dm.threshold_calibration(p_pred, y_obs, w=None)["ece"]
        ece_w = dm.threshold_calibration(p_pred, y_obs, w=w)["ece"] if with_causal else float("nan")
        at_rows.append({"horizon": nm, "metric": "threshold_ece", "value_naive": ece_n,
                        "value_ipcw": ece_w,
                        "naive_minus_ipcw_gap": (ece_n - ece_w) if np.isfinite(ece_n) and np.isfinite(ece_w) else np.nan,
                        "long_fu": np.nan, "short_fu": np.nan, "long_minus_short_gap": np.nan,
                        "note": "threshold ECE; naive vs IPCW"})
        strat = dm.stratified_calibration(p_pred, y_obs, ctx["has_long_followup"], w=w)
        el, es = strat["long_fu"]["ece"], strat["short_fu"]["ece"]
        at_rows.append({"horizon": nm, "metric": "threshold_ece_followup_strata",
                        "value_naive": np.nan, "value_ipcw": np.nan, "naive_minus_ipcw_gap": np.nan,
                        "long_fu": el, "short_fu": es,
                        "long_minus_short_gap": (el - es) if np.isfinite(el) and np.isfinite(es) else np.nan,
                        "note": "ECE for patients WITH vs WITHOUT long follow-up (informative-attrition diagnostic)"})
    art["dist_attrition_sensitivity"] = _save_csv(
        pd.DataFrame(at_rows), out_dir / "dist_attrition_sensitivity.csv",
        "attrition sensitivity (naive vs IPCW + follow-up-stratified calibration)")

    # ---- dist_modeC_marginal_distance.csv : pooled factual samples vs observed marginal.
    md_rows = []
    for nm in ctx["modec_horizons"]:
        dim = _dim(nm)
        observed = mask_at(dim)
        sim_h = ctx["fac"][observed, :, dim].reshape(-1)        # same observed-patient subset
        obs_h = obs_at(dim)[observed]
        md = dm.marginal_distance(sim_h, obs_h)
        md_rows.append({"horizon": nm, "n_obs": int(observed.sum()),
                        "wasserstein1": md["wasserstein1"], "ks_stat": md["ks_stat"],
                        "median_shift": md["median_shift"]})
    art["dist_modeC_marginal_distance"] = _save_csv(
        pd.DataFrame(md_rows), out_dir / "dist_modeC_marginal_distance.csv",
        "Mode-C marginal distance (Wasserstein-1 / KS / median shift)")

    block = {"artifacts": art, "headline": headline, "n_test": ctx["n_test"],
             "ipcw_applied": bool(with_causal), "split_strategy": split_strategy}
    return block


# ------------------------------------------------------------------------------------- #
# CAUSAL (tte_*) - only when with_causal
# ------------------------------------------------------------------------------------- #
def _run_causal(ctx) -> dict:
    dataset, splits, test_idx = ctx["dataset"], ctx["splits"], ctx["test_idx"]
    out_dir, x, surgery_col = ctx["output_dir"], ctx["x"], ctx["surgery_col"]
    estimator, gbm, n_test = ctx["estimator"], ctx["gbm"], ctx["n_test"]
    L, A, ps = ctx["L"], ctx["A"], ctx["ps"]
    A_test = A[test_idx].astype(int)
    art: dict[str, str] = {}

    mean_red = lambda col: np.nanmean(np.asarray(col, dtype=float), axis=1)

    def below_red(thr):
        return lambda col: (np.asarray(col, dtype=float) < thr).mean(axis=1)

    # ---------- B1-B3: propensity overlap + IPTW + covariate balance ----------
    sw, keep = ct2.stabilized_iptw(A_test, ps, trim=PS_TRIM)
    n_trimmed = int(np.sum(~keep))
    ps_auc = float("nan")
    if roc_auc_score is not None:
        ps_auc = gb.safe(roc_auc_score, A_test, ps)
    iptw_ess = ct2.weighted_effective_sample_size(np.where(keep, sw, 0.0))

    ov_rows = []
    for arm_name, aval in (("sleeve", 0), ("rnygb", 1)):
        p = _finite(ps[A_test == aval])
        ov_rows.append({
            "arm": arm_name, "arm_code": aval, "n": int((A_test == aval).sum()),
            "ps_min": float(p.min()) if p.size else np.nan,
            "ps_p25": float(np.percentile(p, 25)) if p.size else np.nan,
            "ps_median": float(np.median(p)) if p.size else np.nan,
            "ps_p75": float(np.percentile(p, 75)) if p.size else np.nan,
            "ps_max": float(p.max()) if p.size else np.nan,
            "trim_lo": PS_TRIM[0], "trim_hi": PS_TRIM[1], "n_trimmed": n_trimmed,
            "ps_auc": ps_auc, "n_test": n_test})
    art["tte_propensity_overlap"] = _save_csv(
        pd.DataFrame(ov_rows), out_dir / "tte_propensity_overlap.csv",
        "propensity overlap by arm (+ trim / AUC)")
    png = out_dir / "tte_propensity_overlap.png"
    if _safe_plot(_plot_ps_overlap, ps, A_test, PS_TRIM, png):
        evfm.report_saved(png, "propensity overlap plot")
        art["tte_propensity_overlap_png"] = str(png)

    smd_before = ct2.standardized_mean_diff(L[test_idx], A_test, w=None)
    smd_after = ct2.standardized_mean_diff(L[test_idx], A_test, w=np.where(keep, sw, 0.0))
    love_rows = []
    for j, nm in enumerate(ctx["L_names"]):
        b = float(smd_before[j]) if j < smd_before.size else np.nan
        a = float(smd_after[j]) if j < smd_after.size else np.nan
        love_rows.append({"covariate": nm, "smd_before": b, "smd_after": a,
                          "abs_smd_before": abs(b) if np.isfinite(b) else np.nan,
                          "abs_smd_after": abs(a) if np.isfinite(a) else np.nan,
                          "balanced_after": bool(np.isfinite(a) and abs(a) < 0.1)})
    love_df = pd.DataFrame(love_rows)
    art["tte_covariate_balance_love"] = _save_csv(
        love_df, out_dir / "tte_covariate_balance_love.csv",
        "covariate balance (SMD before vs after IPTW)")
    png = out_dir / "tte_covariate_balance_love.png"
    if _safe_plot(_plot_love, love_df, png):
        evfm.report_saved(png, "Love plot")
        art["tte_covariate_balance_love_png"] = str(png)

    # ---------- B5: AIPW marginal effects (continuous ATE, threshold RD, composite RD) ----------
    me_rows = []
    results: dict[str, dict] = {}      # keyed for reuse in RCT / E-value / c-for-benefit
    ipcw_ess_by_h: dict[str, float] = {}

    def _aipw_record(kind, group, name, thr, dim, Y, delta, mu1, mu0, exploratory, note, p_obs):
        res = ct2.aipw(Y, A_test, delta, ps, p_obs, mu1, mu0)
        lo, hi = res["ci"]
        me_rows.append({
            "outcome": name, "group": group, "horizon": name, "estimand": kind,
            "threshold": (thr if thr is not None else np.nan), "ate": res["ate"], "se": res["se"],
            "ci_lo": lo, "ci_hi": hi, "mu1_mean": float(np.nanmean(mu1)),
            "mu0_mean": float(np.nanmean(mu0)), "n_test": n_test,
            "n_observed": int(np.asarray(delta).sum()), "exploratory": bool(exploratory),
            "note": note})
        results[f"{kind}:{name}"] = {"res": res, "mu1": mu1, "mu0": mu0, "dim": dim,
                                     "group": group, "thr": thr}
        return res

    # continuous ATE
    for group, name in ctx["cont_targets"]:
        dim = _dim(name)
        Y = dataset.x[test_idx, dim].astype(float)
        delta = dataset.mask[test_idx, dim].astype(int)
        mu0, mu1 = ctx["mu_arm"](dim, mean_red)
        p_obs, ipcw = ctx["censoring"](dim)
        ipcw_ess_by_h[name] = ct2.weighted_effective_sample_size(ipcw)
        _aipw_record("continuous_ate", group, name, None, dim, Y, delta, mu1, mu0, False,
                     "AIPW ATE (kg/m^2 or %); negative = RYGB lower", p_obs)

    # threshold RD
    for group, name, thr in ctx["thr_targets"]:
        dim = _dim(name)
        delta = dataset.mask[test_idx, dim].astype(int)
        xv = dataset.x[test_idx, dim].astype(float)
        with np.errstate(invalid="ignore"):
            Y = np.where(delta == 1, (xv < thr).astype(float), 0.0)
        mu0, mu1 = ctx["mu_arm"](dim, below_red(thr))
        p_obs, ipcw = ctx["censoring"](dim)
        ipcw_ess_by_h[name] = ct2.weighted_effective_sample_size(ipcw)
        _aipw_record("threshold_rd", group, name, thr, dim, Y, delta, mu1, mu0, False,
                     f"AIPW risk difference for P({group} < {thr}); positive = RYGB more likely to cross", p_obs)

    # composite complication RD (exploratory; no RCT anchor; widest E-value)
    y_comp = np.asarray(gbm["y"])[test_idx].astype(float)
    delta_comp = np.ones(n_test, dtype=int)
    mu_comp = {}
    for arm in (0, 1):
        xc = x.copy()
        xc[:, surgery_col] = float(arm)
        mu_comp[arm] = estimator.predict_proba(xc[test_idx])[:, 1]
    _aipw_record("composite_rd", "composite", "composite_complication", None, None, y_comp,
                 delta_comp, mu_comp[1], mu_comp[0], True,
                 "EXPLORATORY: GBM composite-complication risk difference; no RCT anchor, widest E-value caveat",
                 np.ones(n_test))
    art["tte_marginal_effects"] = _save_csv(
        pd.DataFrame(me_rows), out_dir / "tte_marginal_effects.csv",
        "AIPW marginal effects (continuous ATE / threshold RD / composite RD)")

    # ---------- B7: RCT benchmark (weight + glycemia only) ----------
    bmi_base = gb.frame_feature(dataset, "BMIatEvent")
    baseline_bmi_mean = (float(np.nanmean(bmi_base[test_idx]))
                         if bmi_base is not None and np.isfinite(np.nanmean(bmi_base[test_idx])) else np.nan)
    rct_rows = []
    # (a) %TWL from a BMI ATE (prefer bmi_2y, else the first available BMI continuous horizon)
    bmi_key = next((f"continuous_ate:{n}" for _, n in ctx["cont_targets"] if n == "bmi_2y"), None)
    if bmi_key is None:
        bmi_key = next((f"continuous_ate:{n}" for g, n in ctx["cont_targets"] if g == "bmi"), None)
    if bmi_key is not None:
        res = results[bmi_key]["res"]
        conv = _twl_from_bmi_ate(res, baseline_bmi_mean)
        if conv is not None:
            est, ci = conv
            bench = ct2.benchmark_vs_rct(est, ci, "twl_pct_1_2y")
            rct_rows.append({
                "anchor": "twl_pct_1_2y", "source_horizon": bmi_key.split(":")[1],
                "emulated_estimate": est, "emulated_ci_lo": ci[0], "emulated_ci_hi": ci[1],
                "rct_point": bench["rct"]["delta_rygb_minus_sg"], "rct_ci_lo": bench["rct"]["ci"][0],
                "rct_ci_hi": bench["rct"]["ci"][1], "overlaps_rct_ci": bench["overlaps_rct_ci"],
                "comparison_valid": True,
                "unit_note": "BMI-pt ATE -> %TWL via 100*(BMI0 - BMI_t)/BMI0 (fixed-height weight proxy)",
                "caveat": "verify-before-quote; %TWL proxy assumes constant height", "verify_before_quote": True})
    # (b) t2d_remission vs P(HbA1c<5.7) RD (definition mismatch; RD vs RR)
    h_key = next((f"threshold_rd:{n}" for g, n, _ in ctx["thr_targets"] if n == "hba1c_12m"), None)
    if h_key is None:
        h_key = next((f"threshold_rd:{n}" for g, n, _ in ctx["thr_targets"] if g == "hba1c"), None)
    if h_key is not None:
        res = results[h_key]["res"]
        risk0 = float(np.nanmean(results[h_key]["mu0"]))
        risk1 = float(np.nanmean(results[h_key]["mu1"]))
        emulated_rr = (risk1 / risk0) if risk0 > 0 else np.nan
        bench = ct2.benchmark_vs_rct(res["ate"], res["ci"], "t2d_remission")
        rct_rows.append({
            "anchor": "t2d_remission", "source_horizon": h_key.split(":")[1],
            "emulated_estimate": res["ate"], "emulated_ci_lo": res["ci"][0], "emulated_ci_hi": res["ci"][1],
            "rct_point": bench["rct"]["rr_rygb_vs_sg"], "rct_ci_lo": bench["rct"]["ci"][0],
            "rct_ci_hi": bench["rct"]["ci"][1], "overlaps_rct_ci": bench["overlaps_rct_ci"],
            "comparison_valid": False, "emulated_rr_context": emulated_rr,
            "unit_note": "emulated is a RISK DIFFERENCE (prob units); RCT anchor is a RISK RATIO - NOT directly comparable",
            "caveat": "definition mismatch: P(HbA1c<5.7) vs trial remission off-meds; verify-before-quote",
            "verify_before_quote": True})
    art["tte_rct_benchmark"] = _save_csv(
        pd.DataFrame(rct_rows), out_dir / "tte_rct_benchmark.csv",
        "emulated vs RCT anchor (weight/glycemia only; verify-before-quote)")

    # ---------- B6: E-value per primary contrast ----------
    ev_rows = []
    for key, info in results.items():
        kind, name = key.split(":", 1)
        res = info["res"]
        if kind == "continuous_ate":
            dim = info["dim"]
            obs = _finite(dataset.x[test_idx, dim])
            pooled_sd = float(np.std(obs, ddof=1)) if obs.size >= dm.MIN_N else float("nan")
            rr, ep, eb = _evalue_continuous(res, pooled_sd)
            basis = "smd_to_rr(ate / pooled_outcome_sd)"
        else:                                                    # threshold_rd / composite_rd
            risk0 = float(np.nanmean(info["mu0"]))
            rr, ep, eb = _evalue_rd(res, risk0)
            basis = "RR of arm risks: (risk_SG + ATE) / risk_SG"
        ev_rows.append({"contrast": name, "estimand": kind, "rr": rr, "e_point": ep, "e_bound": eb,
                        "basis": basis,
                        "note": ("EXPLORATORY (widest caveat); no RCT anchor" if kind == "composite_rd"
                                 else "primary contrast")})
    art["tte_evalue"] = _save_csv(
        pd.DataFrame(ev_rows), out_dir / "tte_evalue.csv",
        "E-value per contrast (point + CI-bound)")

    # ---------- B8: c-for-benefit (validate the individualized contrast) ----------
    cfb_rows = []
    for group, name in ctx["cont_targets"]:
        info = results.get(f"continuous_ate:{name}")
        if info is None:
            continue
        dim = info["dim"]
        Y = dataset.x[test_idx, dim].astype(float)              # observed continuous outcome
        pred_ite = info["mu0"] - info["mu1"]                    # oriented "higher = more benefit" (BMI/HbA1c lower better)
        cfb = ct2.c_for_benefit(pred_ite, A_test, Y, ps, lower_is_better=True)
        cfb_rows.append({"outcome": name, "group": group, "c_for_benefit": cfb["c_for_benefit"],
                         "n_pairs": cfb["n_pairs"], "lower_is_better": True,
                         "pred_ite": "mu0 - mu1 (RYGB benefit = BMI/HbA1c reduction)"})
    art["tte_c_for_benefit"] = _save_csv(
        pd.DataFrame(cfb_rows), out_dir / "tte_c_for_benefit.csv",
        "c-for-benefit (individualized RYGB-vs-SG effect)")

    # ---------- tte_weights_summary.json ----------
    sw_keep = sw[keep] if keep.any() else np.array([])
    weights_summary = {
        "iptw": {"min": float(np.min(sw_keep)) if sw_keep.size else float("nan"),
                 "max": float(np.max(sw_keep)) if sw_keep.size else float("nan"),
                 "mean": float(np.mean(sw_keep)) if sw_keep.size else float("nan"),
                 "ess": iptw_ess, "n_kept": int(keep.sum()), "n_trimmed": n_trimmed,
                 "trim": list(PS_TRIM)},
        "ipcw_ess_by_horizon": ipcw_ess_by_h,
        "ps_model": {"backend": "xgboost" if gb.xgboost_available() else "histgb",
                     "auc": ps_auc, "n_features": len(ctx["L_names"])},
    }
    art["tte_weights_summary"] = _save_json(
        weights_summary, out_dir / "tte_weights_summary.json", "IPTW/IPCW/PS weight summary")

    # ---------- manifest block ----------
    headline_effects = {}
    for key, info in results.items():
        res = info["res"]
        headline_effects[key.replace(":", "__")] = {"ate": res["ate"], "ci": list(res["ci"])}
    block = {
        "artifacts": art,
        "treatment_coding": TREATMENT_CODING,
        "ps_model": {"backend": "xgboost" if gb.xgboost_available() else "histgb",
                     "auc": ps_auc, "n_features": len(ctx["L_names"]),
                     "include_race": ct2.INCLUDE_RACE_IN_PS, "trim": list(PS_TRIM),
                     "n_trimmed": n_trimmed},
        "weights": {"iptw_ess": iptw_ess, "ipcw_ess_by_horizon": ipcw_ess_by_h},
        "rct_anchors": "verify-before-quote",
        "absent_confounders": ABSENT_CONFOUNDERS,
        "seeds": SEEDS,
        "marginal_effects_headline": headline_effects,
    }
    return block


# ------------------------------------------------------------------------------------- #
# RCT / E-value conversions (approximations documented in the CSV notes)
# ------------------------------------------------------------------------------------- #
def _twl_from_bmi_ate(res: dict, baseline_bmi_mean: float):
    """Convert a BMI-point ATE (RYGB - SG) to a %TWL delta (RYGB - SG).

    %TWL_arm = 100*(BMI0 - BMI_t_arm)/BMI0, so delta%TWL = -100*ATE/BMI0 (fixed-height
    weight proxy). Returns (estimate, (ci_lo, ci_hi)) or None if not computable.
    """
    ate = res["ate"]
    lo, hi = res["ci"]
    if not (np.isfinite(ate) and np.isfinite(baseline_bmi_mean) and baseline_bmi_mean > 0):
        return None
    est = -100.0 * ate / baseline_bmi_mean
    b1 = -100.0 * lo / baseline_bmi_mean
    b2 = -100.0 * hi / baseline_bmi_mean
    return est, (min(b1, b2), max(b1, b2))


def _evalue_continuous(res: dict, pooled_sd: float):
    ate = res["ate"]
    lo, hi = res["ci"]
    if not (np.isfinite(ate) and np.isfinite(pooled_sd) and pooled_sd > 0):
        return float("nan"), float("nan"), float("nan")
    rr = ct2.smd_to_rr(ate / pooled_sd)
    rr_lo = ct2.smd_to_rr(lo / pooled_sd)
    rr_hi = ct2.smd_to_rr(hi / pooled_sd)
    e = ct2.e_value(rr, min(rr_lo, rr_hi), max(rr_lo, rr_hi))
    return rr, e["e_point"], e["e_bound"]


def _evalue_rd(res: dict, risk0: float):
    ate = res["ate"]
    lo, hi = res["ci"]
    if not (np.isfinite(ate) and np.isfinite(risk0) and risk0 > 0):
        return float("nan"), float("nan"), float("nan")
    rr = (risk0 + ate) / risk0
    rr_lo = (risk0 + lo) / risk0
    rr_hi = (risk0 + hi) / risk0
    if rr <= 0:
        return rr, float("nan"), float("nan")
    e = ct2.e_value(rr, min(rr_lo, rr_hi), max(rr_lo, rr_hi))
    return rr, e["e_point"], e["e_bound"]


# ------------------------------------------------------------------------------------- #
# Plotting (all guarded by _safe_plot)
# ------------------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_coverage(cov_by_h: dict, path: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
    for nm, rows in cov_by_h.items():
        xs = [r["nominal"] for r in rows]
        ys = [r["empirical"] for r in rows]
        if any(np.isfinite(y) for y in ys):
            ax.plot(xs, ys, marker="o", label=nm)
    ax.set(title="Coverage calibration curve", xlabel="Nominal coverage",
           ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ps_overlap(ps: np.ndarray, A_test: np.ndarray, trim, path: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 21)
    for aval, label, color in ((0, "sleeve (SG)", "tab:blue"), (1, "rnygb (RYGB)", "tab:red")):
        p = _finite(ps[A_test == aval])
        if p.size:
            ax.hist(p, bins=bins, alpha=0.6, label=label, color=color, density=True)
    for t in trim:
        ax.axvline(t, ls="--", color="k", alpha=0.5)
    ax.set(title="Propensity overlap by arm", xlabel="P(RYGB | L)", ylabel="density")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_love(love_df: pd.DataFrame, path: Path) -> None:
    plt = _mpl()
    d = love_df.dropna(subset=["abs_smd_before", "abs_smd_after"], how="all").copy()
    if d.empty:
        raise ValueError("no finite SMDs to plot")
    d = d.sort_values("abs_smd_before", ascending=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(d))))
    ax.scatter(d["abs_smd_before"], y, label="before IPTW", color="tab:red", s=25)
    ax.scatter(d["abs_smd_after"], y, label="after IPTW", color="tab:blue", s=25)
    ax.axvline(0.1, ls="--", color="k", alpha=0.5, label="|SMD|=0.1")
    ax.set_yticks(y)
    ax.set_yticklabels(d["covariate"], fontsize=6)
    ax.set(title="Covariate balance (Love plot)", xlabel="|standardized mean difference|")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------- #
# CLI (mirrors bmi_threshold_probability.main)
# ------------------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", type=str, default=None,
                        help="pipeline dir with manifest.json (gives gbm + twin run dirs)")
    parser.add_argument("--twin-run", type=str, default=None)
    parser.add_argument("--gbm-run", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-causal", dest="no_causal", action="store_true",
                        help="skip the tte_* causal pass; emit only the naive dist_* artifacts")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = run(dataset=dataset, splits=splits, model=model, twin_cfg=twin_cfg, pre=pre,
                   gbm=gbm, gbm_cfg=gbm_cfg, output_dir=output_dir, device=device,
                   n_samples=args.n_samples, n_steps=args.n_steps, seed=args.seed,
                   with_causal=not args.no_causal)

    n_test = int(splits["test"].shape[0])
    print(f"\nrun_tte complete  (n_test={n_test}, samples/patient={args.n_samples}, "
          f"with_causal={not args.no_causal})")
    print(f"  distributional artifacts: {len(manifest['distributional'].get('artifacts', {}))}")
    if isinstance(manifest["causal_tte"], dict) and manifest["causal_tte"].get("status") != "skipped":
        print(f"  causal_tte artifacts:     {len(manifest['causal_tte'].get('artifacts', {}))}")
    else:
        print("  causal_tte: skipped (--no-causal)")


if __name__ == "__main__":
    main()
