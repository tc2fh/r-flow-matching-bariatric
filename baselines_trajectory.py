"""Per-horizon trajectory baselines for the digital-twin flow ablation (W4).

The event-conditioned twin flow claims that modelling BMI/HbA1c trajectories as a
*conditional generative* process, coupled to the composite event, beats simpler
predictors. To decide whether that claim survives we need honest, patient-for-patient
aligned baselines that see the SAME conditioning information the flow sees. This
module provides two:

  * ``xgb``   -- one XGBoost regressor per BMI/HbA1c output target/horizon (15 of
                them: 9 BMI + 6 HbA1c), missing features routed the tree-native way
                (NaNs kept, XGBoost picks a default direction per split).
  * ``ridge`` -- one sklearn ``Ridge`` regressor per horizon, a LINEAR stand-in for
                a linear mixed model. statsmodels is not installed in this env, so a
                true per-patient random-effects LMM (lme4/statsmodels MixedLM) is
                deferred to the Cosmos VM; per-horizon Ridge is the honest local
                proxy (independent horizons, no random effect). Ridge cannot take
                NaNs, so features are median-imputed (train statistics) then
                standardised inside a Pipeline; the imputer keeps all-NaN columns
                (possible on the 52-row fake CSV) as a constant.

Both baselines regress the ORIGINAL-unit target (BMI value / HbA1c %) and are fit on
the shared TRAIN split only (the flow's gradients likewise only touch train; val is
early-stop selection), on the rows where that horizon is observed. Held-out TEST
predictions come back in ``[n_test, 15]`` original units, in ``splits['test']`` order,
so they line up one-for-one with the flow's per-patient samples and the observed
targets from ``mt.split_arrays``.

Conditioning feature set (the SAME information the twin encoder consumes):
  * the 8 shared ``fm.PATIENT_FEATURES`` (raw; may contain NaN),
  * ``surgery_idx`` (sleeve/rnygb), and
  * ``event`` -- the binary composite event -- when ``use_event=True`` (default), so
    the baselines match the event-conditioned flow's Mode-A (true-event) information.
    ``use_event=False`` drops it to mirror the no-event flow arm.

This module also owns the proper-scoring primitives (CRPS, Gaussian predictive NLL)
and the paired-test helper used by ``evaluate_twin.compare_trajectory_models`` so the
four arms are scored by identical code. It is deliberately importable with no side
effects; run it directly to fit the two baselines and dump a per-horizon metric CSV::

    OMP_NUM_THREADS=1 python baselines_trajectory.py --csv fake_data/fake_mbs_cohort.csv
"""

from __future__ import annotations

import os
import sys

# macOS dual-OpenMP guard: this script imports BOTH xgboost and (transitively, via
# fm/mt/tw) torch, which SIGSEGVs on darwin unless OMP threading is pinned to 1
# before torch loads. Harmless on Linux. The smoke commands also export it, but this
# keeps a standalone ``python baselines_trajectory.py`` safe on macOS on its own.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_flow_matching as fm
import train_flow_matching_twin as tw

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


ARM_XGB = "xgb"
ARM_RIDGE = "ridge"


# Physiologic-plausibility bounds for OBSERVED outcomes, applied as data QC before
# scoring. A value outside these is a data-entry error (e.g. a raw weight logged as a
# BMI) and is excluded from MAD/RMSE/CRPS so one garbage record cannot detonate RMSE
# (Saux et al./SOPHIA QC their outcomes similarly; RMSE is outlier-sensitive by design).
# Bounds are deliberately wide -- they catch gross errors, not real post-op extremes.
PHYSIOLOGIC_BOUNDS = {"bmi": (10.0, 100.0), "hba1c": (3.0, 20.0)}


def plausible_mask(obs: np.ndarray, bounds: "tuple[float, float] | None") -> np.ndarray:
    """Boolean mask of physiologically-plausible observed values (finite AND in range).
    ``bounds`` is a ``(lo, hi)`` tuple or ``None`` (None -> finiteness check only)."""
    obs = np.asarray(obs, dtype=float)
    ok = np.isfinite(obs)
    if bounds is not None:
        lo, hi = bounds
        ok = ok & (obs >= lo) & (obs <= hi)
    return ok


# --------------------------------------------------------------------------- #
# Proper scoring primitives (shared with the four-arm comparison)
# --------------------------------------------------------------------------- #
def crps_ensemble(samples: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Sample-based CRPS per observation (lower is better).

    ``samples`` is ``[n, m]`` (m predictive draws per observation), ``obs`` is
    ``[n]``. Uses the energy-form estimator with the O(m log m) sorted identity for
    the pairwise term::

        CRPS = mean_i|x_i - y| - 1/m^2 * sum_i (2i - m - 1) * x_(i)

    where ``x_(i)`` are the samples sorted ascending. For a degenerate one-sample
    ensemble (m=1, i.e. a point forecast) the pairwise term is exactly 0, so this
    returns ``|x - y|`` -- meaning a point predictor's CRPS equals its absolute
    error, so its per-horizon mean CRPS is the mean absolute error. That identity lets
    the flow arms (real predictive spread) and the point baselines be scored by one
    function. (MAD elsewhere is reported as the MEDIAN absolute deviation -- the Saux
    et al./SOPHIA definition -- so it is robust and is NOT this mean quantity.)
    """
    samples = np.asarray(samples, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("samples must be [n, m]")
    n, m = samples.shape
    term1 = np.mean(np.abs(samples - obs[:, None]), axis=1)
    if m == 1:
        return term1
    s = np.sort(samples, axis=1)
    weights = (2.0 * np.arange(1, m + 1) - m - 1)
    term2 = (s * weights[None, :]).sum(axis=1) / (m * m)
    return term1 - term2


def gaussian_predictive_nll(samples: np.ndarray, obs: np.ndarray, var_floor: float = 1e-6) -> np.ndarray:
    """Negative log predictive density under a Gaussian moment-matched to the samples.

    Per observation, fit ``N(mean, var)`` to the ``m`` predictive draws and score the
    observed value. This is the standard deep-ensemble predictive NLL; it rewards a
    calibrated spread, unlike MAD/RMSE. It is NOT the exact continuous-normalising-flow
    NLL (which needs an ODE solve of the instantaneous change-of-variables); that is
    heavier and is deferred to the VM. Requires m>=2 (a point forecast has no spread,
    so NLL is undefined and returns NaN).
    """
    samples = np.asarray(samples, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    n, m = samples.shape
    if m < 2:
        return np.full(n, np.nan)
    mu = samples.mean(axis=1)
    var = np.maximum(samples.var(axis=1, ddof=1), var_floor)
    return 0.5 * np.log(2.0 * np.pi * var) + (obs - mu) ** 2 / (2.0 * var)


def paired_test(score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank + paired t on two arms' per-patient scores.

    ``score_a``/``score_b`` are aligned per-patient scores (e.g. CRPS) for two arms on
    the SAME patients where lower is better. ``mean_diff = mean(a - b)``: negative
    means arm A scores lower (better) than arm B. Returns NaNs where a test is not
    computable (fewer than 2 finite pairs, scipy missing, or all-zero differences for
    Wilcoxon, which is reported as p=1).
    """
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    out = {"n_pairs": int(a.size), "mean_diff": float("nan"),
           "wilcoxon_stat": float("nan"), "wilcoxon_p": float("nan"),
           "ttest_stat": float("nan"), "ttest_p": float("nan")}
    if a.size == 0:
        return out
    out["mean_diff"] = float(np.mean(a - b))
    if a.size < 2 or scipy_stats is None:
        return out
    diff = a - b
    try:
        if np.allclose(diff, 0.0):
            out["wilcoxon_stat"], out["wilcoxon_p"] = 0.0, 1.0
        else:
            w = scipy_stats.wilcoxon(a, b, zero_method="wilcox")
            out["wilcoxon_stat"], out["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
    except Exception:  # noqa: BLE001 - degenerate paired samples
        pass
    try:
        t = scipy_stats.ttest_rel(a, b)
        out["ttest_stat"], out["ttest_p"] = float(t.statistic), float(t.pvalue)
    except Exception:  # noqa: BLE001
        pass
    return out


def horizon_score(samples_h: np.ndarray, obs_h: np.ndarray, mask_h: np.ndarray, has_density: bool,
                  obs_bounds: "tuple[float, float] | None" = None) -> dict:
    """Score one arm at one horizon on the observed test patients.

    ``samples_h`` is ``[n, m]`` predictive draws (m=1 for a point baseline), ``obs_h``
    the observed original-unit target ``[n]``, ``mask_h`` the 1/0 observed indicator.
    ``obs_bounds`` (lo, hi) excludes physiologically-implausible observed values from the
    scored set (data QC; pass ``PHYSIOLOGIC_BOUNDS[group]``); None keeps every finite obs.
    The point estimate is the ensemble MEAN. MAD is the MEDIAN absolute deviation
    (median_i |point_i - y_i|) -- the Saux et al./SOPHIA definition, robust to outliers
    and directly comparable to their reported MAD; RMSE (mean of squares) is the
    outlier-sensitive companion (SOPHIA note this too). Returns both the scalar
    summaries and the per-patient arrays (over the observed subset, in mask order) so
    callers can pool across horizons for group rows and run paired tests.
    """
    obs_h = np.asarray(obs_h, dtype=np.float64)
    sel = (np.asarray(mask_h) == 1) & plausible_mask(obs_h, obs_bounds)
    empty = np.array([], dtype=np.float64)
    if not sel.any():
        return {"n_obs": 0, "mad": float("nan"), "rmse": float("nan"), "crps": float("nan"),
                "nll": float("nan"), "abs_err": empty, "sq_err": empty, "crps_pp": empty, "nll_pp": empty}
    s = np.asarray(samples_h, dtype=np.float64)[sel]
    y = obs_h[sel]
    point = s.mean(axis=1)
    err = point - y
    abs_err = np.abs(err)
    sq_err = err ** 2
    crps_pp = crps_ensemble(s, y)
    nll_pp = gaussian_predictive_nll(s, y) if has_density else np.full(y.size, np.nan)
    nll = float(np.nanmean(nll_pp)) if np.isfinite(nll_pp).any() else float("nan")
    return {"n_obs": int(sel.sum()), "mad": float(np.median(abs_err)),
            "rmse": float(np.sqrt(np.mean(sq_err))), "crps": float(np.mean(crps_pp)), "nll": nll,
            "abs_err": abs_err, "sq_err": sq_err, "crps_pp": crps_pp, "nll_pp": nll_pp}


# --------------------------------------------------------------------------- #
# Feature matrix + estimators
# --------------------------------------------------------------------------- #
def build_feature_matrix(dataset: fm.FlowDataset, use_event: bool) -> tuple[np.ndarray, list[str]]:
    """Conditioning matrix mirroring the twin encoder's inputs.

    Columns: the 8 shared ``fm.PATIENT_FEATURES`` (raw, NaNs kept) + ``surgery_idx`` +
    (when ``use_event``) the binary composite ``event``. No standardisation here -- the
    Ridge pipeline standardises internally; XGBoost is scale-invariant.
    """
    patient = dataset.patient_features_raw.astype(np.float64)
    surgery = dataset.surgery_idx.astype(np.float64).reshape(-1, 1)
    columns = [patient, surgery]
    names = list(dataset.patient_feature_names) + ["surgery_idx"]
    if use_event:
        event = dataset.x[:, tw.MACE_DIM].astype(np.float64).reshape(-1, 1)
        columns.append(event)
        names.append("event")
    return np.hstack(columns), names


def make_xgb_regressor(seed: int, params: dict | None = None) -> "xgb.XGBRegressor":
    defaults = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=1.0,
        tree_method="hist", objective="reg:squarederror",
        random_state=seed, n_jobs=1,
    )
    if params:
        defaults.update(params)
    # missing defaults to np.nan -> XGBoost routes NaNs natively (no imputation).
    return xgb.XGBRegressor(**defaults)


def make_ridge_regressor(alpha: float = 1.0) -> Pipeline:
    # keep_empty_features=True: on the 52-row fake CSV a lab column can be entirely
    # NaN within a horizon's train rows; keep it as a constant rather than erroring.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def fit_trajectory_baselines(
    dataset: fm.FlowDataset,
    splits: dict[str, np.ndarray],
    use_event: bool = True,
    seed: int = 0,
    xgb_params: dict | None = None,
    ridge_alpha: float = 1.0,
) -> dict:
    """Fit one XGB and one Ridge regressor per horizon; return TEST predictions.

    Predictions are ``[n_test, 15]`` original-unit arrays in ``splits['test']`` order.
    A horizon with <2 observed train rows (no fittable signal, only on tiny CSVs)
    falls back to the observed-train target mean (NaN if none), so the arm always
    yields a number and the metric layer decides observability from the test mask.
    """
    features, feature_names = build_feature_matrix(dataset, use_event)
    x_cont = dataset.x[:, tw.CONT_DIMS].astype(np.float64)
    mask_cont = dataset.mask[:, tw.CONT_DIMS]
    train_idx, test_idx = splits["train"], splits["test"]

    n_test = int(test_idx.size)
    n_h = tw.X_CONT_DIM
    xgb_pred = np.full((n_test, n_h), np.nan, dtype=np.float64)
    ridge_pred = np.full((n_test, n_h), np.nan, dtype=np.float64)
    train_n = []

    for h in range(n_h):
        y = x_cont[:, h]
        obs = mask_cont[:, h] == 1
        tr = train_idx[obs[train_idx]]  # train patients with this horizon observed
        train_n.append(int(tr.size))
        obs_train_vals = y[tr]
        fallback = float(np.mean(obs_train_vals)) if obs_train_vals.size else float("nan")
        if tr.size < 2:
            xgb_pred[:, h] = fallback
            ridge_pred[:, h] = fallback
            continue
        xgb_model = make_xgb_regressor(seed, xgb_params)
        xgb_model.fit(features[tr], y[tr])
        xgb_pred[:, h] = xgb_model.predict(features[test_idx])
        ridge_model = make_ridge_regressor(ridge_alpha)
        ridge_model.fit(features[tr], y[tr])
        ridge_pred[:, h] = ridge_model.predict(features[test_idx])

    return {
        "feature_names": feature_names,
        "use_event": use_event,
        "test_idx": np.asarray(test_idx),
        "xgb_pred": xgb_pred,
        "ridge_pred": ridge_pred,
        "train_n_per_horizon": train_n,
    }


# --------------------------------------------------------------------------- #
# Standalone per-horizon metric table (XGB + Ridge only)
# --------------------------------------------------------------------------- #
def baseline_metric_table(dataset: fm.FlowDataset, splits: dict[str, np.ndarray], baselines: dict) -> pd.DataFrame:
    """Per-horizon MAD/RMSE/CRPS for the two point baselines on the test split.

    NLL is left NaN: point predictors are not density models (see the module CRPS/NLL
    notes). This is the standalone artefact; the four-arm table with the flow arms and
    paired tests is assembled by ``evaluate_twin.compare_trajectory_models``.
    """
    x_cont = dataset.x[:, tw.CONT_DIMS].astype(np.float64)
    mask_cont = dataset.mask[:, tw.CONT_DIMS]
    test_idx = splits["test"]
    obs = x_cont[test_idx]
    mask = mask_cont[test_idx]
    arm_pred = {ARM_XGB: baselines["xgb_pred"], ARM_RIDGE: baselines["ridge_pred"]}
    rows = []
    for h, name in enumerate(tw.CONT_NAMES):
        for arm, pred in arm_pred.items():
            score = horizon_score(pred[:, h][:, None], obs[:, h], mask[:, h], has_density=False,
                                  obs_bounds=PHYSIOLOGIC_BOUNDS.get(tw.CONT_GROUPS[h]))
            rows.append({"horizon": name, "group": tw.CONT_GROUPS[h], "arm": arm,
                         "n_obs": score["n_obs"], "mad": score["mad"], "rmse": score["rmse"],
                         "crps": score["crps"], "nll": score["nll"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(fm.REPO_ROOT / "runs" / "baselines_trajectory"))
    parser.add_argument("--split-strategy", type=str, default="surgery", choices=["surgery", "outcome"])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--no-event", dest="no_event", action="store_true",
                        help="Drop the event feature (mirror the no-event flow arm).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.csv_path:
        raise SystemExit("Pass --csv <path>; the database path needs pyodbc (VM only).")

    dataset = fm.load_dataset_from_csv(args.csv_path)
    split_cfg = tw.TwinConfig(
        split_strategy=args.split_strategy, split_seed=args.split_seed,
        train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac,
    )
    splits = tw.make_splits(dataset, split_cfg)
    baselines = fit_trajectory_baselines(dataset, splits, use_event=not args.no_event, seed=args.seed)
    table = baseline_metric_table(dataset, splits, baselines)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "noevent" if args.no_event else "event"
    out_csv = output_dir / f"baseline_trajectory_metrics_{tag}.csv"
    table.to_csv(out_csv, index=False)
    print(f"Trajectory baselines ({tag} features={baselines['feature_names']})")
    print(f"Train rows per horizon: {baselines['train_n_per_horizon']}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.round(4).to_string(index=False))
    print(f"  [saved] per-horizon baseline metrics -> {out_csv}")


if __name__ == "__main__":
    main()
