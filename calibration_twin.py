"""Trajectory-distribution calibration for the digital-twin flow (W5).

The event-conditioned flow emits a full predictive *distribution* per patient/horizon
(``n_samples`` draws). Collapsing to the median throws away the only thing that makes
a generative twin worth more than a point regressor: its uncertainty. Before we trust
any interval or threshold probability we have to ask two separate questions and never
conflate them:

  1. Is the predictive CENTER unbiased? (a location question)
  2. Is the predictive SPREAD right? (a dispersion question)

The Probability Integral Transform (PIT) answers both at once and, crucially, tells us
WHICH one is broken -- so we do not "calibrate over" a biased sampler:

  * PIT_i = rank of the observed value among that patient's samples, in [0, 1].
    Under a perfectly calibrated predictive distribution the PIT is Uniform(0, 1).
  * A PIT with mass piled at ONE end (mean far from 0.5, or one tail heavy and the
    other empty) is a LOCATION SHIFT: the median is biased. The fix is the SAMPLER
    (retrain / debias). Split-conformal widening will NOT fix it -- it only balloons
    the interval around a still-wrong center, buying nominal coverage dishonestly.
  * A symmetric U-SHAPE (both tails heavy, mean ~ 0.5) is UNDER-DISPERSION: the center
    is fine but the intervals are too narrow. THIS is what conformal fixes.
  * A central hump (tails depleted) is OVER-DISPERSION: intervals too wide.

This module owns the PIT / coverage / CRPS computation, the regime classifier, the
per-horizon split-conformal calibrator (symmetric CQR score -- it widens the interval
about the median and NEVER moves the center, which is exactly why it cannot paper over
a location shift), and the three artifact writers. It reuses ``baselines_trajectory``'s
CRPS so proper scores are defined identically across every milestone. It is importable
with no side effects and never imports ``evaluate_twin`` (no cycle).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import baselines_trajectory as bt  # reuse crps_ensemble so CRPS is defined once, repo-wide

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


# --------------------------------------------------------------------------- #
# Tunable thresholds for the PIT-regime classifier (explicit, auditable).
# --------------------------------------------------------------------------- #
NOMINAL_COVERAGE = 0.90          # central interval we quote (p5..p95)
ALPHA = 1.0 - NOMINAL_COVERAGE   # 0.10 miscoverage budget
PIT_N_MIN = 8                    # below this the PIT is too noisy to classify (W3: thin late-era horizons)
CONFORMAL_N_MIN = 5              # below this we cannot fit a per-horizon conformal Q
LOC_THRESH = 0.07                # |mean(PIT) - 0.5| above this => location shift
ASYM_THRESH = 0.15               # |frac_low - frac_high| above this => one-sided pile => location shift
UNDER_THRESH = 0.30              # frac_low + frac_high above this (uniform=0.20) & symmetric => under-dispersion
OVER_THRESH = 0.12               # frac_low + frac_high below this => over-dispersion (central hump)

# Regime -> (short label for plots, matplotlib colour).
_REGIME_STYLE = {
    "location-shift": ("LOC-SHIFT", "tab:red"),
    "under-dispersion": ("UNDER-DISP", "tab:orange"),
    "over-dispersion": ("OVER-DISP", "tab:blue"),
    "calibrated": ("CALIB", "tab:green"),
    "insufficient-n": ("N/A", "tab:gray"),
}


# --------------------------------------------------------------------------- #
# Core per-horizon primitives
# --------------------------------------------------------------------------- #
def pit_values(samples_h: np.ndarray, obs_h: np.ndarray) -> np.ndarray:
    """PIT (empirical predictive CDF at the observation) for one horizon.

    ``samples_h`` is ``[n, m]`` predictive draws, ``obs_h`` the ``[n]`` observed
    values (already restricted to observed patients). Returns ``[n]`` values in
    [0, 1]: ``(#samples < obs + 0.5 * #samples == obs) / m`` -- the mid-rank PIT,
    which is Uniform(0, 1) under a calibrated predictive distribution even with
    ties. Empty input -> empty output.
    """
    samples_h = np.asarray(samples_h, dtype=np.float64)
    obs_h = np.asarray(obs_h, dtype=np.float64)
    if samples_h.ndim != 2 or samples_h.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    m = samples_h.shape[1]
    less = (samples_h < obs_h[:, None]).sum(axis=1)
    equal = (samples_h == obs_h[:, None]).sum(axis=1)
    return (less + 0.5 * equal) / float(m)


def predictive_band(samples_h: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Per-patient central predictive interval [lo, hi] at level ``1 - alpha``.

    ``samples_h`` is ``[n, m]``. Returns ``(lo[n], hi[n])`` = the (100*alpha/2) and
    (100*(1-alpha/2)) sample percentiles per patient (5th / 95th for alpha=0.10).
    """
    samples_h = np.asarray(samples_h, dtype=np.float64)
    lo = np.percentile(samples_h, 100.0 * alpha / 2.0, axis=1)
    hi = np.percentile(samples_h, 100.0 * (1.0 - alpha / 2.0), axis=1)
    return lo, hi


def coverage_from_band(lo: np.ndarray, hi: np.ndarray, obs: np.ndarray, q: float = 0.0) -> tuple[float, float]:
    """Empirical coverage and mean width of the band [lo - q, hi + q] vs ``obs``.

    ``q`` is the symmetric split-conformal adjustment (0 = raw band). Returns
    ``(coverage, mean_width)``; ``(nan, nan)`` if no observed points.
    """
    obs = np.asarray(obs, dtype=np.float64)
    finite = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    if not finite.any():
        return float("nan"), float("nan")
    lo_c, hi_c, y = lo[finite] - q, hi[finite] + q, obs[finite]
    inside = (y >= lo_c) & (y <= hi_c)
    return float(inside.mean()), float(np.mean(hi_c - lo_c))


def classify_pit(pit: np.ndarray) -> dict:
    """Classify a horizon's PIT into a calibration regime (diagnose BEFORE fixing).

    Priority order matters: an under-dispersed PIT and a location-shifted PIT can
    both put mass near a boundary, so we test the location/asymmetry signal FIRST
    and only call "under-dispersion" for a symmetric, mean-centred U. Returns a dict
    with the regime, the recommended fix (sampler vs conformal), the shift direction,
    and the summary statistics behind the call.
    """
    pit = np.asarray(pit, dtype=np.float64)
    pit = pit[np.isfinite(pit)]
    n = int(pit.size)
    out = {"n": n, "mean_pit": float("nan"), "frac_low": float("nan"), "frac_high": float("nan"),
           "tail_mass": float("nan"), "ks_uniform_p": float("nan"),
           "regime": "insufficient-n", "root_cause_fix": "n/a (too few observed points)", "direction": ""}
    if n < PIT_N_MIN:
        return out
    mean_pit = float(pit.mean())
    frac_low = float(np.mean(pit < 0.10))
    frac_high = float(np.mean(pit > 0.90))
    tail_mass = frac_low + frac_high
    loc = mean_pit - 0.5
    asym = frac_low - frac_high  # > 0 => pile near 0 (obs below samples => samples biased HIGH)
    ks_p = float("nan")
    if scipy_stats is not None and n >= PIT_N_MIN:
        try:
            ks_p = float(scipy_stats.kstest(np.clip(pit, 1e-9, 1 - 1e-9), "uniform").pvalue)
        except Exception:  # noqa: BLE001
            ks_p = float("nan")
    out.update(mean_pit=mean_pit, frac_low=frac_low, frac_high=frac_high, tail_mass=tail_mass, ks_uniform_p=ks_p)

    if abs(loc) > LOC_THRESH or abs(asym) > ASYM_THRESH:
        out["regime"] = "location-shift"
        out["root_cause_fix"] = "SAMPLER (retrain/debias; conformal will NOT fix a biased center)"
        if loc < 0 or asym > 0:  # PIT piled near 0: observed falls BELOW the samples
            out["direction"] = "samples biased HIGH (observed below prediction); center must shift DOWN"
        else:                     # PIT piled near 1: observed ABOVE the samples
            out["direction"] = "samples biased LOW (observed above prediction); center must shift UP"
    elif tail_mass > UNDER_THRESH:
        out["regime"] = "under-dispersion"
        out["root_cause_fix"] = "CONFORMAL (symmetric widening; dispersion is fixable)"
    elif tail_mass < OVER_THRESH:
        out["regime"] = "over-dispersion"
        out["root_cause_fix"] = "intervals too wide; symmetric conformal only widens, so flag (not auto-fixed)"
    else:
        out["regime"] = "calibrated"
        out["root_cause_fix"] = "none (PIT ~ uniform)"
    return out


# --------------------------------------------------------------------------- #
# Split-conformal (per horizon, VAL residuals -> TEST intervals)
# --------------------------------------------------------------------------- #
def _conformal_q(lo_val: np.ndarray, hi_val: np.ndarray, y_val: np.ndarray, alpha: float = ALPHA) -> float:
    """Symmetric CQR conformal adjustment from one horizon's VAL residuals.

    Conformity score E_i = max(lo_i - y_i, y_i - hi_i) (Romano/CQR): positive when the
    observation is OUTSIDE the raw band, negative when comfortably inside. Q is the
    finite-sample-corrected (1 - alpha) quantile of {E_i}. The calibrated interval is
    [lo - Q, hi + Q] -- a SYMMETRIC widening that leaves the median untouched, which is
    precisely why it cannot mask a location shift. Returns nan when val is too thin.
    """
    finite = np.isfinite(y_val) & np.isfinite(lo_val) & np.isfinite(hi_val)
    lo_val, hi_val, y_val = lo_val[finite], hi_val[finite], y_val[finite]
    n = int(y_val.size)
    if n < CONFORMAL_N_MIN:
        return float("nan")
    scores = np.maximum(lo_val - y_val, y_val - hi_val)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)  # conformal finite-sample correction
    return float(np.quantile(scores, level, method="higher"))


def fit_conformal(val_samples: np.ndarray, val_obs: np.ndarray, val_mask: np.ndarray,
                  cont_names: list[str], alpha: float = ALPHA) -> dict:
    """Fit a per-horizon symmetric-CQR conformal calibrator on the VAL split.

    ``val_samples`` is ``[n_val, m, H]`` predictive draws, ``val_obs``/``val_mask`` are
    ``[n_val, H]`` original-unit targets / 1-0 observed indicators. Returns a JSON-native
    dict (lists of floats) suitable for storing in Preprocessing and reloading verbatim.
    """
    H = len(cont_names)
    q = np.full(H, np.nan)
    n_val = np.zeros(H, dtype=np.int64)
    for h in range(H):
        sel = np.asarray(val_mask[:, h]) == 1
        n_val[h] = int(sel.sum())
        if not sel.any():
            continue
        lo, hi = predictive_band(val_samples[sel, :, h], alpha)
        q[h] = _conformal_q(lo, hi, np.asarray(val_obs)[sel, h], alpha)
    return {
        "method": "split_conformal_cqr_symmetric",
        "score": "E = max(lo - y, y - hi); Q = finite-sample (1-alpha) quantile of E",
        "alpha": float(alpha),
        "nominal_coverage": float(1.0 - alpha),
        "cont_names": list(cont_names),
        "q": [None if not np.isfinite(v) else float(v) for v in q],
        "n_val": [int(v) for v in n_val],
        "fit_on": "val",
        "applied_to": "test",
        "note": ("Symmetric widening of the p5..p95 band about the (unchanged) median. Fixes "
                 "UNDER-DISPERSION only; for a location-shift horizon post-coverage is bought by "
                 "ballooning the interval around a biased center and must NOT be trusted -- fix the sampler."),
    }


def conformal_q_array(conformal: dict | None, cont_names: list[str]) -> np.ndarray:
    """Extract the per-horizon Q vector (0.0 where unavailable) from a stored calibrator."""
    H = len(cont_names)
    if not conformal or "q" not in conformal:
        return np.zeros(H)
    q = np.array([np.nan if v is None else float(v) for v in conformal["q"]], dtype=np.float64)
    if q.size != H:
        return np.zeros(H)
    return np.where(np.isfinite(q), q, 0.0)


# --------------------------------------------------------------------------- #
# Orchestrator: compute everything, write artifacts, print the PIT diagnosis
# --------------------------------------------------------------------------- #
def calibrate_flow_predictions(
    cont_test: np.ndarray, obs_test: np.ndarray, mask_test: np.ndarray,
    cont_val: np.ndarray, obs_val: np.ndarray, mask_val: np.ndarray,
    cont_names: list[str], cont_groups: list[str], subject_ids_test: np.ndarray,
    output_dir: Path, report_saved, alpha: float = ALPHA,
) -> dict:
    """Produce the three calibration artifacts (coverage / crps / pit) + conformal.

    ``cont_test``/``cont_val`` are ``[n, m, H]`` original-unit predictive draws (the 15
    BMI/HbA1c dims), ``obs_*``/``mask_*`` the aligned ``[n, H]`` targets/indicators.
    Diagnoses each horizon's PIT regime FIRST, fits split-conformal on val, applies to
    test, writes CSV+PNG for coverage / CRPS / PIT (+ raw PIT values + calibrator JSON),
    prints the per-horizon regime, and returns a summary dict (incl. ``regime_by_horizon``
    and the JSON-native ``conformal`` calibrator) for the caller to persist / cross-link.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conformal = fit_conformal(cont_val, obs_val, mask_val, cont_names, alpha)
    q_vec = conformal_q_array(conformal, cont_names)

    cov_rows, crps_rows, pit_rows, pit_value_rows = [], [], [], []
    pit_by_h: dict[int, np.ndarray] = {}
    regime_by_horizon: dict[str, dict] = {}
    for h, (name, group) in enumerate(zip(cont_names, cont_groups)):
        y = np.asarray(obs_test)[:, h]
        sel = (np.asarray(mask_test[:, h]) == 1) & np.isfinite(y)
        n_obs = int(sel.sum())
        samples_h = cont_test[sel, :, h]
        y_obs = y[sel]

        # -- CRPS (reuse the shared proper score) --
        crps = float(np.mean(bt.crps_ensemble(samples_h, y_obs))) if n_obs else float("nan")
        crps_rows.append({"horizon": name, "group": group, "n_obs_test": n_obs, "crps": crps})

        # -- Coverage pre/post conformal --
        if n_obs:
            lo, hi = predictive_band(samples_h, alpha)
            cov_pre, w_pre = coverage_from_band(lo, hi, y_obs, q=0.0)
            q_h = q_vec[h]
            cov_post, w_post = coverage_from_band(lo, hi, y_obs, q=q_h)
        else:
            cov_pre = w_pre = cov_post = w_post = float("nan")
            q_h = q_vec[h]
        cov_rows.append({"horizon": name, "group": group, "n_obs_test": n_obs,
                         "nominal_coverage": float(1.0 - alpha),
                         "coverage_pre": cov_pre, "coverage_post": cov_post,
                         "mean_width_pre": w_pre, "mean_width_post": w_post,
                         "conformal_q": float(q_h), "n_val": int(conformal["n_val"][h])})

        # -- PIT + regime diagnosis --
        pit = pit_values(samples_h, y_obs)
        pit_by_h[h] = pit
        diag = classify_pit(pit)
        regime_by_horizon[name] = {"group": group, **diag}
        pit_rows.append({"horizon": name, "group": group, "n_obs_test": n_obs,
                         "mean_pit": diag["mean_pit"], "frac_low_lt_0.1": diag["frac_low"],
                         "frac_high_gt_0.9": diag["frac_high"], "tail_mass": diag["tail_mass"],
                         "ks_uniform_p": diag["ks_uniform_p"], "regime": diag["regime"],
                         "root_cause_fix": diag["root_cause_fix"], "direction": diag["direction"]})
        sids = np.asarray(subject_ids_test)[sel]
        for sid, pv in zip(sids, pit):
            pit_value_rows.append({"horizon": name, "group": group, "subject_id": sid, "pit": float(pv)})

    coverage_df = pd.DataFrame(cov_rows)
    crps_df = pd.DataFrame(crps_rows)
    pit_df = pd.DataFrame(pit_rows)
    pit_values_df = pd.DataFrame(pit_value_rows)

    # -- write CSVs --
    cov_path = output_dir / "eval_flow_calibration_coverage.csv"
    crps_path = output_dir / "eval_flow_calibration_crps.csv"
    pit_path = output_dir / "eval_flow_calibration_pit.csv"
    pitv_path = output_dir / "eval_flow_calibration_pit_values.csv"
    cal_path = output_dir / "eval_flow_conformal_calibrator.json"
    coverage_df.to_csv(cov_path, index=False); report_saved(cov_path, "flow calibration coverage (pre/post conformal)")
    crps_df.to_csv(crps_path, index=False); report_saved(crps_path, "flow calibration CRPS per horizon")
    pit_df.to_csv(pit_path, index=False); report_saved(pit_path, "flow calibration PIT summary + regime diagnosis")
    pit_values_df.to_csv(pitv_path, index=False); report_saved(pitv_path, "flow calibration raw PIT values (for W6 histograms)")
    cal_path.write_text(json.dumps(conformal, indent=2), encoding="utf-8"); report_saved(cal_path, "split-conformal calibrator")

    # -- write PNGs --
    _plot_pit_hist(pit_by_h, cont_names, regime_by_horizon, output_dir / "eval_flow_calibration_pit.png", report_saved)
    _plot_coverage(coverage_df, output_dir / "eval_flow_calibration_coverage.png", float(1.0 - alpha), report_saved)
    _plot_crps(crps_df, output_dir / "eval_flow_calibration_crps.png", report_saved)

    # -- printed diagnosis (PIT FIRST) --
    print("\nFlow trajectory calibration -- PIT regime diagnosis (diagnose BEFORE fixing):")
    print(f"{'horizon':>10} | {'n':>3} | {'meanPIT':>7} | {'regime':>16} | fix")
    print("-" * 92)
    for name in cont_names:
        d = regime_by_horizon[name]
        mp = d["mean_pit"]
        mp_s = "  NA  " if not np.isfinite(mp) else f"{mp:6.3f}"
        print(f"{name:>10} | {d['n']:>3} | {mp_s:>7} | {d['regime']:>16} | {d['root_cause_fix']}")
    _print_regime_rollup(regime_by_horizon)

    return {
        "conformal": conformal,
        "regime_by_horizon": regime_by_horizon,
        "coverage": coverage_df.to_dict(orient="records"),
        "crps": crps_df.to_dict(orient="records"),
        "pit": pit_df.to_dict(orient="records"),
        "artifacts": {"coverage_csv": str(cov_path), "crps_csv": str(crps_path), "pit_csv": str(pit_path),
                      "pit_values_csv": str(pitv_path), "calibrator_json": str(cal_path),
                      "coverage_png": str(output_dir / "eval_flow_calibration_coverage.png"),
                      "crps_png": str(output_dir / "eval_flow_calibration_crps.png"),
                      "pit_png": str(output_dir / "eval_flow_calibration_pit.png")},
    }


def _print_regime_rollup(regime_by_horizon: dict) -> None:
    counts: dict[str, int] = {}
    for d in regime_by_horizon.values():
        counts[d["regime"]] = counts.get(d["regime"], 0) + 1
    parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    loc = [n for n, d in regime_by_horizon.items() if d["regime"] == "location-shift"]
    print(f"  regime roll-up: {parts}")
    if loc:
        print(f"  ** location-shift horizons (SAMPLER bias -- conformal will NOT fix): {loc}")
    print("  NOTE: threshold probabilities & intervals are trustworthy ONLY where the PIT is not location-shifted.")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plot_pit_hist(pit_by_h: dict, cont_names: list[str], regimes: dict, path: Path, report_saved) -> None:
    H = len(cont_names)
    ncols = 5
    nrows = int(np.ceil(H / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows), squeeze=False)
    bins = np.linspace(0, 1, 11)
    for h, name in enumerate(cont_names):
        ax = axes[h // ncols][h % ncols]
        pit = pit_by_h.get(h, np.zeros(0))
        pit = pit[np.isfinite(pit)]
        d = regimes[name]
        short, colour = _REGIME_STYLE.get(d["regime"], ("?", "tab:gray"))
        if pit.size:
            ax.hist(pit, bins=bins, color=colour, alpha=0.75, edgecolor="white")
            ax.axhline(pit.size / (len(bins) - 1), color="k", ls="--", lw=0.8, alpha=0.6)  # uniform expectation
        ax.set_title(f"{name}\n{short} (n={d['n']})", fontsize=8)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.tick_params(labelsize=7)
    for k in range(H, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("PIT per horizon (uniform = calibrated; pile at an end = LOCATION SHIFT / sampler bias; "
                 "symmetric U = under-dispersion)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    report_saved(path, "flow PIT histograms per horizon")


def _plot_coverage(coverage_df: pd.DataFrame, path: Path, nominal: float, report_saved) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(coverage_df)), 5))
    x = np.arange(len(coverage_df))
    ax.bar(x - 0.2, coverage_df["coverage_pre"], width=0.4, label="pre-conformal", color="tab:blue", alpha=0.8)
    ax.bar(x + 0.2, coverage_df["coverage_post"], width=0.4, label="post-conformal", color="tab:orange", alpha=0.8)
    ax.axhline(nominal, color="k", ls="--", alpha=0.7, label=f"nominal {nominal:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(coverage_df["horizon"], rotation=45, ha="right", fontsize=8)
    ax.set(title="Central-interval coverage pre/post split-conformal (test)", ylabel="Empirical coverage", ylim=(0, 1.05))
    ax.legend(fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    report_saved(path, "flow coverage bar chart")


def _plot_crps(crps_df: pd.DataFrame, path: Path, report_saved) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(crps_df)), 5))
    colours = ["tab:blue" if g == "bmi" else "tab:green" for g in crps_df["group"]]
    ax.bar(np.arange(len(crps_df)), crps_df["crps"], color=colours, alpha=0.8)
    ax.set_xticks(np.arange(len(crps_df))); ax.set_xticklabels(crps_df["horizon"], rotation=45, ha="right", fontsize=8)
    ax.set(title="CRPS per horizon (lower = sharper + calibrated; bmi=blue, hba1c=green)", ylabel="CRPS (original units)")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    report_saved(path, "flow CRPS bar chart")
