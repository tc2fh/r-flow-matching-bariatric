"""distributional_metrics.py - probabilistic scores for the twin's predictive samples.

Part A of the TTE + distributional-evaluation build (see TTE_DISTRIBUTIONAL_RUN_PLAN.md,
sections A1-A6, and the INTEGRATION_CONTRACT). This module adds the proper-scoring /
calibration functions the twin's per-horizon sample block needs that are NOT already
provided by ``calibration_twin`` (PIT / coverage / conformal) or ``baselines_trajectory``
(CRPS / moment-matched Gaussian NLL).

Design rules (from the contract):
  * CRPS is NOT redefined here - import ``baselines_trajectory.crps_ensemble`` (the one CRPS,
    repo-wide). The moment-matched Gaussian NLL is likewise reused from
    ``baselines_trajectory.gaussian_predictive_nll`` rather than re-derived, so the log score
    stays a thin, weighted, guard-wrapped wrapper over the canonical repo primitive.
  * PIT / predictive-band / coverage primitives live in ``calibration_twin``; this module
    imports them (``ct``) for the combined report and never reimplements them.
  * Every function takes sample arrays ``(n, m)`` or blocks ``(n, m, k)``, observed values
    ``(n,)`` or ``(n, k)``, and an OPTIONAL per-patient weight ``w`` for IPCW/IPTW-weighted
    scoring (Part A5). Weighting affects the reported mean ONLY (weighted mean); the scores
    themselves are unchanged. A weight of 0 (e.g. an unobserved patient under IPCW) drops
    that patient from the mean.
  * Pure numpy / scipy (+ a lazy sklearn import for the calibration slope). No torch is used
    in this module's own logic and there are no side effects / file writes, so every function
    is testable with synthetic arrays. (Importing the mandated reuse targets does transitively
    load torch + xgboost via ``baselines_trajectory``; run under ``OMP_NUM_THREADS=1``.)
  * Small-sample discipline: every function guards fewer than ``MIN_N`` usable (finite,
    complete-case, positive-weight) observations by returning a NaN structure rather than
    raising. Headline-number functions that the plan writes as ``(scalar, per_patient_array)``
    keep that shape so a caller can compute IPCW-vs-naive gaps from the per-patient array.

Return-shape summary (a downstream orchestrator must know these):
  * ``log_score_gaussian``, ``energy_score_block``, ``variogram_score_block``,
    ``interval_score``  -> ``(scalar_float, per_patient_ndarray)``  (per-patient array is
    length ``n_patients``; NaN where the patient/row is not usable).
  * ``coverage_curve``  -> ``list[dict]`` rows ``{"nominal","empirical","mean_width"}``.
  * ``pinball_loss``    -> ``dict{quantile_float: loss_float}``.
  * ``sharpness``       -> ``dict{"mean_sd","mean_width90"}``.
  * ``threshold_calibration`` -> ``dict{"ece","mce","brier","table"}`` (table = list of rows).
  * ``brier_decomposition``   -> ``dict{"reliability","resolution","uncertainty","brier"}``.
  * ``calibration_slope_intercept`` -> ``dict{"slope","citl"}``.
  * ``ipcw_from_model`` -> per-patient weight ``ndarray`` (0 for unobserved).
  * ``stratified_calibration`` -> ``dict{"long_fu","short_fu"}`` of ``threshold_calibration``.
  * ``marginal_distance`` -> ``dict{"wasserstein1","ks_stat","median_shift"}``.
  * ``proper_scores_report`` -> ``dict`` bundling CRPS + log score + interval score + pinball
    + sharpness (+ ``n``) for one horizon.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from baselines_trajectory import crps_ensemble, gaussian_predictive_nll  # the one CRPS + NLL
import calibration_twin as ct                                            # PIT / band / coverage

# Below this many usable observations a headline statistic is too noisy to trust; return NaN
# rather than a misleading number (matches calibration_twin's guard philosophy).
MIN_N = 10


# --------------------------------------------------------------------------- #
# Weighting helpers
# --------------------------------------------------------------------------- #
def _wmean(v, w=None):
    """NaN-aware (optionally weighted) mean of a per-patient vector.

    Unweighted: equals ``np.nanmean(v)``. Weighted: ``sum(w*v)/sum(w)`` over entries with a
    finite value and a finite, strictly-positive weight (so a weight of 0 drops that entry -
    exactly the IPCW convention for an unobserved patient). Empty / all-NaN -> NaN, no warning.
    """
    v = np.asarray(v, dtype=float)
    if w is None:
        finite = np.isfinite(v)
        return float(np.mean(v[finite])) if finite.any() else float("nan")
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def _reduce(per_patient, w=None, min_n=MIN_N):
    """Guarded weighted mean: NaN if fewer than ``min_n`` finite per-patient entries."""
    arr = np.asarray(per_patient, dtype=float)
    if int(np.isfinite(arr).sum()) < min_n:
        return float("nan")
    return _wmean(arr, w)


def _weighted_coverage(lo, hi, obs, w):
    """Weighted analogue of ``calibration_twin.coverage_from_band`` (raw band, q=0).

    Kept in THIS module so ``calibration_twin`` is not edited (contract). Returns
    ``(coverage, mean_width)`` over patients with a finite obs / band and a positive weight;
    ``(nan, nan)`` if none qualify.
    """
    obs = np.asarray(obs, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    finite = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    if not finite.any():
        return float("nan"), float("nan")
    y, lo_c, hi_c = obs[finite], lo[finite], hi[finite]
    inside = ((y >= lo_c) & (y <= hi_c)).astype(float)
    width = hi_c - lo_c
    if w is None:
        return float(inside.mean()), float(width.mean())
    ww = np.asarray(w, dtype=float)[finite]
    ok = np.isfinite(ww) & (ww > 0)
    if not ok.any():
        return float("nan"), float("nan")
    denom = ww[ok].sum()
    return (float(np.sum(ww[ok] * inside[ok]) / denom),
            float(np.sum(ww[ok] * width[ok]) / denom))


# --------------------------------------------------------------------------- #
# A1. Proper scores (log score + multivariate energy / variogram over a block)
# --------------------------------------------------------------------------- #
def log_score_gaussian(samples_h, obs_h, w=None):
    """Moment-matched Gaussian log score for one horizon -> ``(scalar, per_patient_nll)``.

    Fits ``N(mean, var)`` to each patient's ``m`` predictive draws and scores the observed
    value; the per-patient array is the negative log predictive density (lower = better). The
    exact continuous-normalising-flow NLL (an ODE change-of-variables solve) is the "right"
    log score but expensive; this is the cheap, tail-sensitive stand-in and punishes
    overconfident tails far harder than CRPS. Delegates the per-patient NLL to the repo's
    canonical ``baselines_trajectory.gaussian_predictive_nll`` (single source of truth;
    var-floor 1e-6, ddof=1; NaN where ``obs`` is unobserved or ``m < 2``).
    """
    samples_h = np.asarray(samples_h, dtype=float)
    obs_h = np.asarray(obs_h, dtype=float)
    n = obs_h.shape[0] if obs_h.ndim >= 1 else 0
    if samples_h.ndim != 2 or samples_h.shape[0] == 0:
        return float("nan"), np.full(n, np.nan)
    nll = gaussian_predictive_nll(samples_h, obs_h)   # (n,), NaN where obs NaN or m<2
    return _reduce(nll, w), nll


def energy_score_block(samples_block, obs_block, w=None, max_samples=100):
    """Multivariate CRPS (energy score) over a horizon block ``(n, m, k)``.

    ``ES = E||X - y|| - 0.5 E||X - X'||`` (Euclidean over the k horizons). Grades the JOINT
    distribution across horizons - the cross-horizon correlation the per-horizon CRPS is blind
    to. Complete-case rows only: a patient with ANY non-finite horizon in ``obs`` (or in its
    samples) yields NaN and is skipped from the mean. The O(m^2) pairwise term is subsampled to
    the first ``max_samples`` draws. Returns ``(scalar, per_patient_es)``.
    """
    S = np.asarray(samples_block, dtype=float)
    Y = np.asarray(obs_block, dtype=float)
    n = S.shape[0] if S.ndim >= 1 else 0
    if S.ndim != 3 or Y.ndim != 2 or n == 0:
        return float("nan"), np.full(n, np.nan)
    S = S[:, :max_samples, :]
    n = S.shape[0]
    es = np.full(n, np.nan)
    for i in range(n):
        y = Y[i]
        Si = S[i]
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(Si))):
            continue                                   # joint score needs the full block
        d_sy = np.linalg.norm(Si - y, axis=1).mean()
        diff = Si[:, None, :] - Si[None, :, :]
        d_ss = np.linalg.norm(diff, axis=2).mean()
        es[i] = d_sy - 0.5 * d_ss
    return _reduce(es, w), es


def variogram_score_block(samples_block, obs_block, p=0.5, w=None, max_samples=100):
    """Variogram score of order ``p`` over a horizon block ``(n, m, k)``.

    Sensitive to the DEPENDENCY structure (pairwise differences ``|Y_a - Y_b|^p``) rather than
    the marginals - catches a model that gets each horizon's marginal right but the trajectory
    SHAPE / correlation wrong. Complements the energy score. Complete-case rows only; O(m^2)
    pairwise term subsampled to ``max_samples``. Returns ``(scalar, per_patient_vs)``.
    """
    S = np.asarray(samples_block, dtype=float)
    Y = np.asarray(obs_block, dtype=float)
    n = S.shape[0] if S.ndim >= 1 else 0
    if S.ndim != 3 or Y.ndim != 2 or n == 0:
        return float("nan"), np.full(n, np.nan)
    S = S[:, :max_samples, :]
    n = S.shape[0]
    vs = np.full(n, np.nan)
    for i in range(n):
        y = Y[i]
        Si = S[i]
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(Si))):
            continue
        Ey = np.abs(y[:, None] - y[None, :]) ** p                        # (k, k)
        ES = (np.abs(Si[:, :, None] - Si[:, None, :]) ** p).mean(0)      # (k, k)
        vs[i] = float(np.sum((Ey - ES) ** 2))
    return _reduce(vs, w), vs


# --------------------------------------------------------------------------- #
# A2. Interval calibration, coverage curve, pinball, sharpness
# --------------------------------------------------------------------------- #
def interval_score(samples_h, obs_h, alpha=0.10, w=None):
    """Winkler / interval score for the central ``1 - alpha`` band -> ``(scalar, per_patient)``.

    Rewards correct coverage AND narrow width jointly, so (unlike PICP) it cannot be gamed by
    simply widening the interval. Lower = better. Unobserved patients (non-finite obs) get NaN
    and are dropped from the mean.
    """
    samples_h = np.asarray(samples_h, dtype=float)
    obs = np.asarray(obs_h, dtype=float)
    if samples_h.ndim != 2 or samples_h.shape[0] == 0:
        n = obs.shape[0] if obs.ndim >= 1 else 0
        return float("nan"), np.full(n, np.nan)
    lo = np.nanquantile(samples_h, alpha / 2.0, axis=1)
    hi = np.nanquantile(samples_h, 1.0 - alpha / 2.0, axis=1)
    below = obs < lo
    above = obs > hi
    s = (hi - lo) + (2.0 / alpha) * (lo - obs) * below + (2.0 / alpha) * (obs - hi) * above
    s = np.where(np.isfinite(obs), s, np.nan)
    return _reduce(s, w), s


def coverage_curve(samples_h, obs_h, levels=(0.5, 0.8, 0.9, 0.95), w=None):
    """Empirical coverage at several nominal levels -> a coverage-calibration curve.

    Reuses ``calibration_twin.predictive_band`` + ``coverage_from_band`` per level so the band
    convention matches the W5 session; when ``w`` is given the coverage/width are reweighted in
    THIS module (``_weighted_coverage``) so ``calibration_twin`` stays untouched. Returns rows
    ``{"nominal","empirical","mean_width"}``; NaN rows if fewer than ``MIN_N`` observed points.
    """
    samples_h = np.asarray(samples_h, dtype=float)
    obs = np.asarray(obs_h, dtype=float)
    n_obs = int(np.isfinite(obs).sum())
    guarded = samples_h.ndim != 2 or samples_h.shape[0] == 0 or n_obs < MIN_N
    rows = []
    for c in levels:
        if guarded:
            rows.append({"nominal": float(c), "empirical": float("nan"), "mean_width": float("nan")})
            continue
        lo, hi = ct.predictive_band(samples_h, alpha=1.0 - c)
        if w is None:
            cov, width = ct.coverage_from_band(lo, hi, obs)
        else:
            cov, width = _weighted_coverage(lo, hi, obs, w)
        rows.append({"nominal": float(c), "empirical": float(cov), "mean_width": float(width)})
    return rows


def pinball_loss(samples_h, obs_h, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9), w=None):
    """Mean pinball / quantile loss over a grid -> ``dict{quantile: loss}``.

    The quantile analog of CRPS and the natural readout for the W5 conformal step. Each loss is
    non-negative. Unobserved patients are dropped. NaN dict if fewer than ``MIN_N`` observed.
    """
    samples_h = np.asarray(samples_h, dtype=float)
    obs = np.asarray(obs_h, dtype=float)
    n_obs = int(np.isfinite(obs).sum())
    if samples_h.ndim != 2 or samples_h.shape[0] == 0 or n_obs < MIN_N:
        return {float(q): float("nan") for q in quantiles}
    out = {}
    for q in quantiles:
        pred_q = np.nanquantile(samples_h, q, axis=1)
        d = obs - pred_q
        loss = np.where(d >= 0, q * d, (q - 1.0) * d)
        loss = np.where(np.isfinite(obs), loss, np.nan)
        out[float(q)] = _wmean(loss, w)
    return out


def sharpness(samples_h, w=None):
    """Predictive sharpness -> ``dict{"mean_sd","mean_width90"}`` (report next to every
    calibration number; Gneiting: maximize sharpness subject to calibration). NaN if fewer than
    ``MIN_N`` patient rows.
    """
    samples_h = np.asarray(samples_h, dtype=float)
    if samples_h.ndim != 2 or samples_h.shape[0] < MIN_N:
        return {"mean_sd": float("nan"), "mean_width90": float("nan")}
    sd = np.nanstd(samples_h, axis=1)
    lo = np.nanquantile(samples_h, 0.05, axis=1)
    hi = np.nanquantile(samples_h, 0.95, axis=1)
    return {"mean_sd": _wmean(sd, w), "mean_width90": _wmean(hi - lo, w)}


# --------------------------------------------------------------------------- #
# A3. Threshold-probability calibration (the actual deliverable)
# --------------------------------------------------------------------------- #
def threshold_calibration(p_pred, y_obs, n_bins=10, w=None):
    """Reliability of a predicted threshold probability against observed crossings.

    ``p_pred``: predicted P(cross) per patient (fraction of samples past the cut). ``y_obs``:
    observed 1{crossed} at that horizon (observed patients only). Pass ``w`` = IPCW weights so
    informative attrition does not bias the curve; a weight of 0 drops that patient. Returns
    ``{"ece","mce","brier","table"}`` (Brier is the raw mean squared error, in [0, 1]).
    """
    p = np.asarray(p_pred, dtype=float)
    y = np.asarray(y_obs, dtype=float)
    ww = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
    finite = np.isfinite(p) & np.isfinite(y) & np.isfinite(ww) & (ww > 0)
    p, y, ww = p[finite], y[finite], ww[finite]
    if p.size < MIN_N:
        return {"ece": float("nan"), "mce": float("nan"), "brier": float("nan"), "table": []}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    total_w = ww.sum()
    ece = 0.0
    rows, gaps = [], []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        wb = ww[sel].sum()
        conf = float(np.average(p[sel], weights=ww[sel]))
        acc = float(np.average(y[sel], weights=ww[sel]))
        ece += (wb / total_w) * abs(acc - conf)
        gaps.append(abs(acc - conf))
        rows.append({"bin": int(b), "pred": conf, "obs": acc, "weight": float(wb), "n": int(sel.sum())})
    brier = float(np.average((p - y) ** 2, weights=ww))
    return {"ece": float(ece), "mce": float(max(gaps)) if gaps else float("nan"),
            "brier": brier, "table": rows}


def brier_decomposition(p_pred, y_obs, n_bins=10, w=None):
    """Murphy decomposition ``BS = reliability - resolution + uncertainty``.

    Separates calibration (reliability, lower better) from discrimination (resolution, higher
    better) for the threshold event; ``uncertainty = ybar*(1-ybar)`` is the irreducible base
    rate. ``brier`` is the reconstructed (binned) score ``rel - res + unc``. NaN structure if
    fewer than ``MIN_N`` usable observations.
    """
    p = np.asarray(p_pred, dtype=float)
    y = np.asarray(y_obs, dtype=float)
    ww = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
    finite = np.isfinite(p) & np.isfinite(y) & np.isfinite(ww) & (ww > 0)
    p, y, ww = p[finite], y[finite], ww[finite]
    if p.size < MIN_N:
        return {"reliability": float("nan"), "resolution": float("nan"),
                "uncertainty": float("nan"), "brier": float("nan")}
    ybar = float(np.average(y, weights=ww))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    total_w = ww.sum()
    rel = res = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        wb = ww[sel].sum()
        pb = float(np.average(p[sel], weights=ww[sel]))
        ob = float(np.average(y[sel], weights=ww[sel]))
        rel += (wb / total_w) * (pb - ob) ** 2
        res += (wb / total_w) * (ob - ybar) ** 2
    unc = ybar * (1.0 - ybar)
    return {"reliability": float(rel), "resolution": float(res),
            "uncertainty": float(unc), "brier": float(rel - res + unc)}


# --------------------------------------------------------------------------- #
# A4. Calibration drift - slope + calibration-in-the-large
# --------------------------------------------------------------------------- #
def calibration_slope_intercept(p_pred, y_obs, w=None):
    """Cox calibration for a binary / threshold probability -> ``dict{"slope","citl"}``.

      slope = coefficient of ``logit(p)`` in an unregularised logistic fit
              ``logit(P(y=1)) ~ logit(p)``   [1 = ideal; <1 = overfit/optimistic, the SOPHIA
              out-of-sample failure mode].
      citl  = calibration-in-the-large = ``mean(y) - mean(p)`` (weighted)  [0 = ideal]. This is
              the standard, clean CITL proxy (the plan's messy offset-GLM line is replaced by
              this - see deviations note). Computed from the RAW predicted probabilities.

    Uses an unregularised ``sklearn.linear_model.LogisticRegression`` (spelled
    ``l1_ratio=0, C=1e12`` for sklearn 1.8+ warning-free compatibility; identical to the
    task's ``penalty=None`` fit - see the inline note). Guards fewer than ``MIN_N``
    observations and a single-class outcome by returning NaN for both quantities.
    """
    p = np.asarray(p_pred, dtype=float)
    y = np.asarray(y_obs, dtype=float)
    finite = np.isfinite(p) & np.isfinite(y)
    if w is not None:
        wv = np.asarray(w, dtype=float)
        finite = finite & np.isfinite(wv) & (wv > 0)
    p, y = p[finite], y[finite]
    ww = None if w is None else np.asarray(w, dtype=float)[finite]
    if p.size < MIN_N or np.unique(y).size < 2:
        return {"slope": float("nan"), "citl": float("nan")}
    citl = _wmean(y, ww) - _wmean(p, ww)                      # clean, weighted CITL proxy
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    lp = np.log(pc / (1.0 - pc))
    try:
        from sklearn.linear_model import LogisticRegression
        # Unregularised logistic fit for the Cox calibration slope. The plan/task specify
        # LogisticRegression(penalty=None), but `penalty` is deprecated in sklearn 1.8
        # (removed in 1.10) and emits a FutureWarning; `l1_ratio=0` (pure L2) with a very
        # large finite `C` is the sklearn-1.8+ spelling of the IDENTICAL unregularised fit
        # (slope matches penalty=None to machine precision) and is warning-free +
        # forward-compatible. `C=np.inf` is avoided because it re-triggers a UserWarning.
        clf = LogisticRegression(l1_ratio=0.0, C=1e12, solver="lbfgs", max_iter=1000)
        clf.fit(lp[:, None], y.astype(int), sample_weight=ww)
        slope = float(clf.coef_[0, 0])
    except Exception:                                         # non-convergence / degenerate fit
        slope = float("nan")
    return {"slope": slope, "citl": float(citl)}


# --------------------------------------------------------------------------- #
# A5. Attrition robustness - IPCW weights + follow-up-stratified calibration
# --------------------------------------------------------------------------- #
def ipcw_from_model(p_observed, observed_mask, clip=0.05):
    """Per-patient inverse-probability-of-censoring weight at one horizon.

    ``p_observed``: P(observed at horizon | L) from the censoring model (fit on TRAIN).
    Observed patients get ``1 / clip(p_observed, clip, 1)``; unobserved patients get weight 0
    (they contribute through the model term, not the empirical term). Returns the weight array.
    """
    p = np.asarray(p_observed, dtype=float)
    m = np.asarray(observed_mask).astype(bool)
    return np.where(m, 1.0 / np.clip(p, clip, 1.0), 0.0)


def stratified_calibration(p_pred, y_obs, has_long_followup, **kw):
    """``threshold_calibration`` split by follow-up completeness -> ``dict{"long_fu","short_fu"}``.

    A large gap between the strata is the direct diagnostic that attrition is informative. A
    per-patient ``w`` (passed via ``**kw``) is SUBSET to each stratum before delegating, so the
    weighted path does not raise a shape mismatch (the plan's snippet passed the full-length
    weight to a subset of patients - fixed here).
    """
    p = np.asarray(p_pred, dtype=float)
    y = np.asarray(y_obs, dtype=float)
    sel_long = np.asarray(has_long_followup).astype(bool)
    w = kw.pop("w", None)

    def _sub(sel):
        wsel = None if w is None else np.asarray(w, dtype=float)[sel]
        return threshold_calibration(p[sel], y[sel], w=wsel, **kw)

    return {"long_fu": _sub(sel_long), "short_fu": _sub(~sel_long)}


# --------------------------------------------------------------------------- #
# A6. Distribution distance for Mode-C marginals
# --------------------------------------------------------------------------- #
def marginal_distance(sim_h, obs_h):
    """Magnitude (not just a p-value) of the gap between a simulated and observed marginal.

    With n in the tens of thousands the KS p-value collapses to 0 and is uninformative;
    Wasserstein-1 says HOW FAR the simulated marginal sits from observed in real units.
    Returns ``{"wasserstein1","ks_stat","median_shift"}`` (median_shift = median(sim) -
    median(obs)); finite values only. NaN structure if either side has fewer than ``MIN_N``
    finite values.
    """
    sim = np.asarray(sim_h, dtype=float)
    obs = np.asarray(obs_h, dtype=float)
    sim = sim[np.isfinite(sim)]
    obs = obs[np.isfinite(obs)]
    if sim.size < MIN_N or obs.size < MIN_N:
        return {"wasserstein1": float("nan"), "ks_stat": float("nan"), "median_shift": float("nan")}
    return {"wasserstein1": float(stats.wasserstein_distance(sim, obs)),
            "ks_stat": float(stats.ks_2samp(sim, obs).statistic),
            "median_shift": float(np.median(sim) - np.median(obs))}


# --------------------------------------------------------------------------- #
# Convenience bundle (one horizon) - used by run_tte / fairness_audit
# --------------------------------------------------------------------------- #
def proper_scores_report(samples_h, obs_h, mask=None, w=None):
    """Bundle the per-horizon proper scores into one JSON-native dict.

    CRPS (via ``crps_ensemble`` on the observed rows) + log score + interval score + pinball +
    sharpness, all restricted to the SAME observed set: ``mask`` (if given) intersected with
    finite ``obs``. Pure (no writes). NaN structure if fewer than ``MIN_N`` observed patients.
    ``w`` (per-patient, full length) reweights every component; it is subset to the observed
    rows for the CRPS term.
    """
    samples_h = np.asarray(samples_h, dtype=float)
    obs = np.asarray(obs_h, dtype=float)
    if mask is None:
        obs_mask = np.isfinite(obs)
    else:
        obs_mask = np.asarray(mask).astype(bool) & np.isfinite(obs)
    n_obs = int(obs_mask.sum())
    nan_pin = {float(q): float("nan") for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    if samples_h.ndim != 2 or samples_h.shape[0] == 0 or n_obs < MIN_N:
        return {"n": n_obs, "crps": float("nan"), "log_score": float("nan"),
                "interval_score": float("nan"), "pinball": nan_pin,
                "sharpness": {"mean_sd": float("nan"), "mean_width90": float("nan")}}
    obs_eff = np.where(obs_mask, obs, np.nan)             # common observed set for every score
    crps_pp = crps_ensemble(samples_h[obs_mask], obs[obs_mask])
    w_masked = None if w is None else np.asarray(w, dtype=float)[obs_mask]
    crps = _wmean(crps_pp, w_masked)
    ls, _ = log_score_gaussian(samples_h, obs_eff, w)
    isc, _ = interval_score(samples_h, obs_eff, w=w)
    pin = pinball_loss(samples_h, obs_eff, w=w)
    shp = sharpness(samples_h, w=w)
    return {"n": n_obs, "crps": float(crps), "log_score": float(ls),
            "interval_score": float(isc), "pinball": pin, "sharpness": shp}
