"""causal_tte.py - target-trial-emulation causal estimators for the sleeve-vs-RYGB contrast.

This module is PART B of the TTE + distributional build (see TTE_DISTRIBUTIONAL_RUN_PLAN.md
and the INTEGRATION_CONTRACT). It holds PURE estimators + nuisance-model fitters only:

  * It NEVER writes files (the orchestrator run_tte.py owns artifacts).
  * It NEVER samples the digital twin. The per-arm outcome model E[Y|A=a,L] (mu1/mu0) is
    built in run_tte.py from the surgery clamp (bmi_threshold_probability.cohort_probability
    pattern); ``aipw`` RECEIVES mu1/mu0 arrays as inputs so it stays unit-testable without a
    trained twin.
  * Nuisance models (propensity, censoring) are fit on TRAIN and applied to TEST only - never
    cross-fit on the test set. Callers pass the SHARED split (gb.make_splits inherits the
    frozen split_strategy), so temporal runs carry through automatically.

The emulated trial (protocol - full writeup in METHODS_causal_distributional.md):
  eligibility  = the existing cohort filters (GLP-1-naive, T2D, first MBS, SG or RYGB)
  strategies   = SG (CptCode 43775) vs RYGB (43644 / 43846 / 43645)  [see SLEEVE_CPTS/RYGB_CPTS]
  time zero    = ProcDateValue (point intervention: no immortal time; ITT == per-protocol)
  assignment   = observational -> assume conditional exchangeability given L, adjust by PS
  outcomes     = BMI / HbA1c trajectory, threshold probabilities, composite complication
  estimand     = marginal ATE (AIPW, doubly robust) + individual CATE (the twin)
  analysis     = doubly-robust IPCW-AIPW; benchmark vs RCT; E-value; c-for-benefit

ABSENT confounders - the exchangeability threat, documented prominently (see manifest too):
  * GERD / reflux / hiatal hernia - NOT in the cohort. The single biggest clinical driver of
    sleeve-vs-RYGB choice (reflux pushes toward RYGB). This is the primary residual-confounding
    threat and the main target of the E-value.
  * Surgeon / center / facility ID - NOT in the cohort. Provider practice pattern (often the
    dominant determinant of which operation a patient gets) is unadjustable and unclusterable.
  * Smoking / tobacco, weight history, patient preference, frailty - absent.

Key DECISIONS encoded here (see PART D of the plan):
  * DECISION 1 (CPT -> arm): 43775 = SG; 43644 / 43846 / 43645 = RYGB. The 43645 handling and the
    actual CPT -> surgery_idx mapping are UPSTREAM in the loader (dataset.surgery_idx is already
    built). RYGB_CPTS / SLEEVE_CPTS below are DOCUMENTED constants only; this module does NOT
    modify the loader and consumes dataset.surgery_idx as-is. 43645 handling MUST be verified in
    the loader separately.
  * DECISION 3 (race): INCLUDE_RACE_IN_PS = False -> race enters the FAIRNESS audit only, never L.
  * DECISION 4 (censoring vs competing risk): death (DeathInterval) censors the metabolic
    trajectory - you cannot measure BMI/HbA1c post-mortem - so ``censoring_model`` models
    observation-at-horizon directly (death is a censoring event for the metabolic outcomes). The
    competing-risk framing for the complication endpoint is noted as future work.
  * DECISION 6 (RCT backbone): RCT_ANCHORS exist ONLY for weight and glycemia (the trials
    measured them). The complication contrast is NOT anchored (reported as exploratory).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Statistics that are meaningless below this many observations degrade to NaN (never raise).
_MIN_N = 10

# ---------------------------------------------------------------------------------------------
# Module constants (DOCUMENTED; the actual CPT -> surgery_idx mapping is UPSTREAM in the loader)
# ---------------------------------------------------------------------------------------------
# DECISION 3 (PI): race is available but contested as a *predictor*. SVI/RUCA enter L as the
# social-determinant mechanism; race is used for the fairness audit only, NOT the PS model,
# unless the PI explicitly opts in.
INCLUDE_RACE_IN_PS = False

# DECISION 1: CPT -> surgery arm. These are DOCUMENTED reference constants only. The loader
# already builds ``dataset.surgery_idx`` (sleeve=0, rnygb=1); this module NEVER re-derives the
# arm from CptCode and NEVER touches the loader. The 43645 -> RYGB decision and the full
# CPT -> surgery_idx mapping MUST be verified in the loader separately.
RYGB_CPTS = {"43644", "43846", "43645"}
SLEEVE_CPTS = {"43775"}


# ---------------------------------------------------------------------------------------------
# Confounders L + treatment A
# ---------------------------------------------------------------------------------------------
def build_L_A(dataset):
    """Confounder matrix ``L`` (pre-op only) and treatment ``A`` (1 = RYGB, 0 = SG).

    Starts from ``gb.assemble_features`` (default ndarray path: the shared patient features +
    surgery_idx + the GBM extras - comorbidity flags, eGFR, baseline diabetes-drug flags, and the
    5 SVI social-determinant percentiles - are ALL already there, plus any ``*_ismissing``
    companions). It DROPS the treatment column, then APPENDS the two SES/geo categoricals not in
    the numeric matrix (RUCA, CoverageClass) as NaN-preserving integer codes so the tree nuisance
    models can split on them while routing missing cells natively.

    Never calls ``frame_feature("*_num")`` - those columns do not exist (return None). Race is
    excluded (INCLUDE_RACE_IN_PS = False).

    Returns
    -------
    L : np.ndarray [n, p]   float64, NaNs preserved (trees route them natively)
    A : np.ndarray [n]      int (1 = RYGB, 0 = SG), taken from dataset.surgery_idx as-is
    L_names : list[str]     column names of L, in order; NEVER contains "surgery_idx"
    """
    import gbm_mace_baseline as gb  # lazy: keeps the pure estimators importable without the GBM stack

    x, feat, _ = gb.assemble_features(dataset)
    if not isinstance(x, np.ndarray):
        # The GBM categorical path (GBM_CATEGORICAL_FRAME_FEATURES) is OFF by default, so
        # assemble_features returns a float64 ndarray. If it is ever enabled x is a DataFrame;
        # take the numeric block (this build appends its own categorical codes below).
        x = np.asarray(x, dtype=float)

    a_col = feat.index("surgery_idx")
    A = x[:, a_col].astype(int)  # 1 = RYGB, 0 = SG (dataset.surgery_idx, upstream mapping)
    keep = [j for j in range(x.shape[1]) if j != a_col]
    L, L_names = x[:, keep].astype(float), [feat[j] for j in keep]

    # Append SES / geo confounders not already in the numeric GBM matrix. Integer-encode via a
    # NaN-preserving factorize so missing stays NaN (never becomes a spurious level). Race is
    # deliberately absent from this list.
    for canon in ("RUCA", "CoverageClass"):
        vals = gb.frame_categorical(dataset, canon)
        if vals is not None:
            codes, _ = pd.factorize(pd.Series(vals), use_na_sentinel=True)
            codes = codes.astype(float)
            codes[codes < 0] = np.nan  # factorize sentinel (-1) -> NaN stays NaN for the trees
            L = np.column_stack([L, codes])
            L_names.append(canon + "_code")
    return L, A, L_names


# ---------------------------------------------------------------------------------------------
# Nuisance-model backend
# ---------------------------------------------------------------------------------------------
def _tree_classifier():
    """Gradient-boosted tree classifier with native NaN routing.

    XGBoost on the VM (matches the GBM backend); HistGradientBoosting fallback locally. Both
    route missing values natively - confounders are passed UN-imputed and we NEVER resample for
    class imbalance. ``random_state=0`` is set for determinism (the manifest records ps/censoring
    seeds = 0); this is the sole addition over the plan's parameter list.
    """
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=0,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0)


def _fit_predict_proba(x_train, y_train, x_test):
    """Fit a tree classifier on TRAIN, return P(class=1) on TEST, with graceful guards.

    Degrades (never raises) when the training data cannot support a real fit: a train set below
    ``_MIN_N`` or a single-class train label falls back to the constant base rate. Returns
    ``(p1_test, clf_or_None)``.
    """
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    y_train = np.asarray(y_train)
    n_train = y_train.shape[0]
    n_test = x_test.shape[0]

    base = float(np.clip(np.nanmean(y_train), 0.0, 1.0)) if n_train else 0.5
    if not np.isfinite(base):
        base = 0.5
    if n_train < _MIN_N or np.unique(y_train).size < 2:
        return np.full(n_test, base, dtype=float), None

    clf = _tree_classifier()
    try:
        clf.fit(x_train, y_train)
        proba = np.asarray(clf.predict_proba(x_test), dtype=float)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            p1 = proba[:, 1]
        else:  # single-column proba (degenerate) -> constant base rate
            p1 = np.full(n_test, base, dtype=float)
    except Exception:
        return np.full(n_test, base, dtype=float), None
    return np.asarray(p1, dtype=float), clf


def propensity_scores(L, A, train_idx, test_idx):
    """Propensity score P(RYGB | L). Fit on TRAIN, predict TEST (leak-free).

    Returns ``(ps_test, clf)`` where ``ps_test`` is aligned with ``test_idx``. ``clf`` is None
    when the training data was too small / single-class (ps falls back to the base rate).
    """
    L = np.asarray(L, dtype=float)
    A = np.asarray(A)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    return _fit_predict_proba(L[train_idx], A[train_idx].astype(int), L[test_idx])


def stabilized_iptw(A, ps, trim=(0.02, 0.98)):
    """Stabilized inverse-probability-of-treatment weights + common-support trim.

    A and ps are TEST-aligned (same length). ``keep`` is the positivity mask (PS strictly inside
    the trim band and finite). Returns ``(weights, keep_mask)``; weights are always finite.
    """
    A = np.asarray(A)
    ps = np.asarray(ps, dtype=float)
    lo, hi = trim
    keep = np.isfinite(ps) & (ps > lo) & (ps < hi)
    pt = float(np.mean(A == 1)) if A.size else float("nan")
    sw = np.where(
        A == 1,
        pt / np.clip(ps, 1e-3, 1.0),
        (1.0 - pt) / np.clip(1.0 - ps, 1e-3, 1.0),
    )
    return sw.astype(float), keep


def weighted_effective_sample_size(w):
    """Kish effective sample size ESS = (sum w)^2 / sum(w^2). ESS <= n for positive weights.

    Convenience for the run_tte manifest (IPTW / IPCW ESS). Ignores non-finite / non-positive
    weights; returns NaN if nothing is left.
    """
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return float("nan")
    denom = float(np.sum(w ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w) ** 2 / denom)


def standardized_mean_diff(L, A, w=None):
    """Per-covariate standardized mean difference between arms (Love-plot data).

    ``(mean_RYGB - mean_SG) / sqrt((var_RYGB + var_SG) / 2)`` per column, optionally weighted
    (pass IPTW weights to read balance AFTER weighting). |SMD| < 0.1 is the usual balance rule.
    NaN cells are dropped per column; a column empty in either arm -> NaN. Below ``_MIN_N`` rows
    the whole vector is NaN.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim != 2:
        L = L.reshape(L.shape[0], -1)
    A = np.asarray(A)
    p = L.shape[1]
    if L.shape[0] < _MIN_N or p == 0:
        return np.full(p, np.nan)
    w_arr = None if w is None else np.asarray(w, dtype=float)

    a1 = A == 1
    a0 = A == 0
    out = np.full(p, np.nan)
    for j in range(p):
        col = L[:, j]
        fin = np.isfinite(col)
        s1 = a1 & fin
        s0 = a0 & fin
        if s1.sum() == 0 or s0.sum() == 0:
            continue

        def _mean_var(sel):
            c = col[sel]
            ww = None
            if w_arr is not None:
                ww = w_arr[sel]
                ww = np.where(np.isfinite(ww) & (ww > 0), ww, 0.0)
                if ww.sum() <= 0:
                    return np.nan, np.nan
            mu = np.average(c, weights=ww)
            var = np.average((c - mu) ** 2, weights=ww)
            return mu, var

        m1, v1 = _mean_var(s1)
        m0, v0 = _mean_var(s0)
        out[j] = (m1 - m0) / np.sqrt((v1 + v0) / 2.0 + 1e-12)
    return out


# ---------------------------------------------------------------------------------------------
# Censoring model (shared with the distributional IPCW weighting)
# ---------------------------------------------------------------------------------------------
def censoring_model(L, observed_mask, train_idx, test_idx, clip=0.05):
    """P(observed at horizon h | L). Fit on TRAIN, apply to TEST. One call per horizon.

    ``observed_mask`` is a full-length per-patient 0/1 indicator of a NON-null outcome at this
    horizon (the same mask the flow eval uses). DECISION 4: death censors the metabolic
    trajectory (you cannot measure BMI/HbA1c post-mortem), so this models observation-at-horizon
    directly - death is folded into "not observed".

    Returns ``(p_obs_test, ipcw_test)``: the predicted observation probability on TEST and the
    per-patient inverse-probability-of-censoring weight ``delta / clip(p_obs, clip, 1)``
    (unobserved TEST patients get weight 0 - they contribute through the model, not the empirical
    term).
    """
    L = np.asarray(L, dtype=float)
    observed_mask = np.asarray(observed_mask)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    y = observed_mask.astype(int)
    p_obs, _ = _fit_predict_proba(L[train_idx], y[train_idx], L[test_idx])
    obs_test = observed_mask[test_idx].astype(bool)
    ipcw = np.where(obs_test, 1.0 / np.clip(p_obs, clip, 1.0), 0.0)
    return p_obs, ipcw


# ---------------------------------------------------------------------------------------------
# Doubly-robust IPCW-AIPW (the marginal effect; twin = outcome model, passed in as mu1/mu0)
# ---------------------------------------------------------------------------------------------
def aipw(Y, A, delta, ps, pc, mu1, mu0):
    """Doubly-robust IPCW-augmented IPW ATE for ONE outcome at ONE horizon.

    All inputs are TEST-aligned 1-D arrays of the SAME length n:

      Y     observed outcome (n,). NaN where unobserved is fine - it is gated out by ``delta``.
      A     treatment (n,), 1 = RYGB, 0 = SG.
      delta observed-at-horizon indicator (n,), 1 / 0.
      ps    P(A=1 | L) (n,)               -- propensity model (propensity_scores).
      pc    P(observed | L) (n,)          -- censoring model (censoring_model, p_obs_test).
      mu1   E[Y | A=1, L] (n,)            -- twin per-arm outcome model (built in run_tte).
      mu0   E[Y | A=0, L] (n,).

    Consistent if EITHER (ps AND pc) OR (mu1, mu0) is correct (double robustness). Returns
    ``{"ate", "se", "ci"}`` with an influence-function SE and 95% Wald CI.

    Units / dual use:
      * CONTINUOUS outcome  -> pass Y = observed BMI/HbA1c at the horizon and mu = per-arm MEAN
        of the twin samples. ATE is in the outcome's units (kg/m^2 or %); negative = RYGB lower.
      * BINARY THRESHOLD    -> pass Y = 1{crossed} and mu = per-arm threshold PROBABILITY
        (fraction of twin samples past the cut). ATE is the causal RISK DIFFERENCE ("X% vs Y%").
    """
    Y = np.asarray(Y, dtype=float)
    A = np.asarray(A, dtype=float)
    delta = np.asarray(delta, dtype=float)
    ps = np.asarray(ps, dtype=float)
    pc = np.asarray(pc, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)

    n = Y.shape[0]
    nan_result = {"ate": float("nan"), "se": float("nan"), "ci": (float("nan"), float("nan"))}
    if n < _MIN_N:
        return nan_result

    r = np.where(delta == 1, 1.0 / np.clip(pc, 0.05, 1.0), 0.0)  # IPCW factor (delta / pc)
    Yf = np.nan_to_num(Y)  # NaN outcomes are zeroed; they carry weight 0 via r / (1 - A)
    m1 = mu1 + A * r * (Yf - mu1) / np.clip(ps, 1e-3, 1.0)
    m0 = mu0 + (1.0 - A) * r * (Yf - mu0) / np.clip(1.0 - ps, 1e-3, 1.0)
    psi = m1 - m0
    psi = psi[np.isfinite(psi)]
    if psi.size < _MIN_N:
        return nan_result
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(psi.size))
    return {"ate": ate, "se": se, "ci": (ate - 1.96 * se, ate + 1.96 * se)}


# ---------------------------------------------------------------------------------------------
# E-value (unmeasured-confounding sensitivity) + SMD -> RR bridge
# ---------------------------------------------------------------------------------------------
def e_value(rr, lo=None, hi=None):
    """VanderWeele-Ding E-value for a risk ratio, and (optionally) the CI limit nearest the null.

    ``e_point`` is the E-value for the point estimate; ``e_bound`` is the E-value for the CI limit
    closest to 1 (or 1.0 when the CI already crosses the null, i.e. no confounding is required to
    explain the finding away). For a standardized mean difference d, convert first with
    ``smd_to_rr(d)``.
    """

    def _ev1(x):
        x = float(x)
        if not np.isfinite(x) or x <= 0:
            return float("nan")
        x = x if x >= 1.0 else 1.0 / x
        return x + np.sqrt(x * (x - 1.0))

    point = _ev1(rr)
    bound = None
    if lo is not None and hi is not None:
        if lo <= 1.0 <= hi:
            bound = 1.0
        else:
            bound = _ev1(lo if lo > 1.0 else hi)
    return {"e_point": point, "e_bound": bound}


def smd_to_rr(d):
    """Approximate risk ratio from a standardized mean difference (VanderWeele): exp(0.91 * d)."""
    return float(np.exp(0.91 * float(d)))


# ---------------------------------------------------------------------------------------------
# Borrowed RCT backbone (empirical calibration of the causal design; RCT-DUPLICATE logic)
# ---------------------------------------------------------------------------------------------
# DECISION 6: the RCT backbone exists ONLY for weight and glycemia (the trials measured them).
# There is NO head-to-head RCT powered for the MACE / nephropathy / retinopathy composite, so the
# complication contrast stays observational / exploratory and is NOT anchored here.
#
# VERIFY-BEFORE-QUOTE: every number below is an ILLUSTRATIVE anchor pending a source check against
# the cited papers. Do NOT quote these in the manuscript until verified; also reconcile UNITS
# first (DECISION 7): convert the twin's BMI-point difference to %TWL using baseline weight, and
# state the P(HbA1c < 5.7) threshold against the trial's remission definition (often < 6.0 or
# < 6.5 off meds). Do NOT compare mismatched endpoints.
RCT_ANCHORS = {
    "twl_pct_1_2y": {  # %TWL, RYGB greater  (verify-before-quote)
        "delta_rygb_minus_sg": +5.0,
        "ci": (+3.0, +7.0),
        "src": "SM-BOSS / SLEEVEPASS 5-10y; convert your BMI-pts to %TWL first",
    },
    "t2d_remission": {  # RR of remission, RYGB vs SG  (verify-before-quote)
        "rr_rygb_vs_sg": 1.4,
        "ci": (1.15, 1.7),
        "src": "Oseberg 1y remission 74% RYGB vs 48% SG; define remission to match",
    },
}


def benchmark_vs_rct(emulated_estimate, emulated_ci, anchor_key):
    """Does the emulated marginal effect land in the RCT confidence region?

    Overlap = the observational design reproduced the randomized answer (empirical calibration of
    the causal design, RCT-DUPLICATE logic). Weight / glycemia anchors only (DECISION 6).
    """
    a = RCT_ANCHORS[anchor_key]
    lo_e, hi_e = emulated_ci
    lo_r, hi_r = a["ci"]
    overlap = not (hi_e < lo_r or hi_r < lo_e)
    return {
        "anchor": anchor_key,
        "emulated": emulated_estimate,
        "emulated_ci": (lo_e, hi_e),
        "rct": a,
        "overlaps_rct_ci": bool(overlap),
    }


# ---------------------------------------------------------------------------------------------
# Calibration-for-benefit / c-for-benefit (validate the individualized CONTRAST, not the level)
# ---------------------------------------------------------------------------------------------
def c_for_benefit(pred_ite, A, Y, ps, lower_is_better=True):
    """Van Klaveren concordance-for-benefit for the twin's per-patient RYGB-vs-SG effect.

    Match each treated patient to the nearest-PS control (without replacement), form pairs, then
    compute per-pair the OBSERVED benefit (outcome_control - outcome_treated, reversed when higher
    is better) and the PREDICTED benefit (mean of the pair's predicted ITEs). The concordance
    between observed and predicted pair benefits ~ 0.5 means no benefit discrimination; higher
    means the twin ranks who-benefits. Requires observed outcomes in BOTH pair members.

    Guards: an empty arm or fewer than two valid pairs -> NaN. The O(pairs^2) concordance loop is
    acceptable at report scale (n ~ 4k pairs); vectorize only if it becomes a bottleneck.
    """
    pred_ite = np.asarray(pred_ite, dtype=float)
    A = np.asarray(A)
    Y = np.asarray(Y, dtype=float)
    ps = np.asarray(ps, dtype=float)

    treated = np.where(A == 1)[0]
    control = np.where(A == 0)[0]
    if treated.size == 0 or control.size == 0:
        return {"c_for_benefit": float("nan"), "n_pairs": 0}

    # nearest-PS matching without replacement
    ctrl_ps = ps[control]
    used = np.zeros(control.size, dtype=bool)
    pairs = []
    for t in treated:
        d = np.abs(ctrl_ps - ps[t])
        d = np.where(used, np.inf, d)
        j = int(np.argmin(d))
        if np.isfinite(d[j]):
            used[j] = True
            pairs.append((int(t), int(control[j])))

    obs_ben, pred_ben = [], []
    for t, c in pairs:
        if not (np.isfinite(Y[t]) and np.isfinite(Y[c]) and np.isfinite(pred_ite[t]) and np.isfinite(pred_ite[c])):
            continue
        b = (Y[c] - Y[t]) if lower_is_better else (Y[t] - Y[c])
        obs_ben.append(b)
        pred_ben.append(0.5 * (pred_ite[t] + pred_ite[c]))
    obs_ben = np.asarray(obs_ben, dtype=float)
    pred_ben = np.asarray(pred_ben, dtype=float)
    if obs_ben.size < 2:
        return {"c_for_benefit": float("nan"), "n_pairs": int(obs_ben.size)}

    # concordance over pairs-of-pairs with discordant observed benefit
    conc = disc = 0
    m = obs_ben.size
    for i in range(m):
        for k in range(i + 1, m):
            if obs_ben[i] == obs_ben[k]:
                continue
            hi_obs = i if obs_ben[i] > obs_ben[k] else k
            hi_pred = i if pred_ben[i] > pred_ben[k] else k
            conc += int(hi_obs == hi_pred)
            disc += int(hi_obs != hi_pred)
    c = conc / (conc + disc) if (conc + disc) else float("nan")
    return {"c_for_benefit": float(c), "n_pairs": int(obs_ben.size)}
