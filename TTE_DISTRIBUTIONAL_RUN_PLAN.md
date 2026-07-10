# Run Plan - Target-Trial Emulation + Expanded Distributional Evaluation

Engineering plan for a coding agent to implement, on the next Cosmos-VM run, (1) a
target-trial-emulation (TTE) causal layer for the sleeve-vs-RYGB contrast and (2) the
expanded distributional-evaluation metrics motivated by the advisor's review questions.

**Relationship to the other planning docs**
- `NEXT_RUN_PLAN.md` (W1-W7) is the *prior* iteration and is owned by a separate session.
  This plan sits ON TOP of it and assumes W1 (freeze/manifest), W3 (temporal split), W4
  (twin counterfactual sampling + `baselines_trajectory.crps_ensemble`), and W5
  (`calibration_twin.py`: PIT/coverage/CRPS/conformal) have landed. Where this plan needs
  a W-item, it says so.
- `MACE_MODELING_DECISIONS.md` holds the modeling rationale. Add a dated entry there when
  this lands.
- The companion `METHODS_causal_distributional.md` is the manuscript-facing writeup of the
  same analysis. Keep the two in sync: every estimator here has a paragraph there.

**This document is a specification, not a merge.** Implement it in a fresh session. The
embedded code is reference implementation to adapt against the live signatures on the VM,
not a drop-in patch. Verify every `ev.*` / `gb.*` / `fm.*` call against the current file
before trusting it - the other session may have moved things.

---

## 0. Hard constraints (carry over the existing build discipline)

- **Do NOT modify the pristine Cosmos core** (`train_flow_matching.py`,
  `tune_flow_matching_optuna.py`) except the one narrow additive split helper already
  planned in `NEXT_RUN_PLAN.md` W3. New work = new modules + additive hooks.
- **Do NOT re-implement CRPS / PIT / coverage.** `baselines_trajectory.crps_ensemble` is
  the one CRPS; `calibration_twin.py` owns PIT/coverage/conformal. Import and reuse.
- **Feature/confounder additions route through `gb.assemble_features` /
  `gb.frame_feature`**, never through the shared `fm.PATIENT_FEATURES` (that would leak
  the confounders into the flow's conditioning and change the twin).
- **Leakage discipline is strict.** Every propensity/confounder variable must be a
  pre-operative baseline value. The outcome and any `PostOp*` / interval field is off
  limits as a confounder.
- **Trees route NaN natively.** Pass confounders un-imputed to the PS/censoring models;
  never resample for imbalance.
- **Shared split is sacred.** The PS model, censoring model, twin, and GBM must all be
  fit/read on the SAME `split_strategy` + seed + fracs (the `SHARED_SPLIT_KEYS` invariant
  in `train_twin_pipeline.py`). Fit nuisance models on TRAIN, apply to TEST - leak-free,
  mirroring how the twin and GBM already work. Do not cross-fit on the test set.
- **Smoke-test on `fake_data/fake_mbs_cohort.csv`** with `mbsaqip_flow/.venv` before
  declaring done. The 52-row cohort makes every stat degrade gracefully (guard n<10);
  real numbers appear on the VM (test ~4.3k). Install xgboost in the venv (W1) so the PS
  backend matches the VM.

---

## 1. What the data supports (checked against `table structure.txt`)

**Confounders AVAILABLE for the propensity model L** (all pre-op, already reachable via
`gb.assemble_features` + `gb.frame_feature`):

| Group | Columns |
|---|---|
| Demographics | `AgeAtEvent`, `Sex`, `FirstRace`/`SecondRace`/`MultiRacial` (race = decision, see below) |
| Baseline clinical | `BMIatEvent`, `WeightAtEvent`, `HbA1cAtEvent`, `CreatinineAtEvent`, `eGFRatEvent` |
| Comorbidity (PMH_*) | `DM2`, `hypertension`, `OSA`, `dyslipidemia`, `MI`, `stroke`, `AFib`, `VTE` |
| Baseline meds | `InsulinStatus`, `BiguanideStatus`, `SGLT2Status` (confirm baseline, not post-op - W2 leakage guard) |
| SES / geography | `CoverageClass`, `RUCA`, `SviOverall`/`SviHousehold`/`SviTransportation`/`SviMinority`/`SviSES`, `StateorProvince` |
| Time / treatment | `ProcDateValue` (time zero), `CptCode` (treatment) |
| Censoring / competing risk | `DeathInterval`, `ActiveEndInterval`, `MACEinterval`, `NephropathyInterval`, etc. |

**Confounders ABSENT - the exchangeability threat, document prominently:**
- **GERD / reflux / hiatal hernia** - NOT in the cohort. This is the single biggest
  clinical driver of sleeve-vs-RYGB choice (reflux pushes toward RYGB). Its absence is the
  primary residual-confounding threat and the main target of the E-value.
- **Surgeon / center / facility ID** - NOT in the cohort. No provider clustering variable,
  so surgeon/center practice pattern (often the dominant determinant of which operation a
  patient gets) is unadjustable and unclusterable.
- **Smoking / tobacco** - absent. Weight-history, patient preference, frailty - absent.

**Decision (PI):** race is available but contested as a *predictor*. Default: SVI/RUCA
enter L as the social-determinant mechanism; race is used for the fairness audit only, NOT
added to the PS model, unless the PI explicitly opts in. Encode this as a config flag
`INCLUDE_RACE_IN_PS = False` with a comment pointing to this decision.

---

## 2. Architecture (3 new modules + additive hooks + 1 orchestrator)

```
distributional_metrics.py   NEW  pure-numpy scores; reuses baselines_trajectory.crps_ensemble
                                  + calibration_twin PIT/coverage. No side effects, no torch.
causal_tte.py               NEW  PS + IPTW + IPCW + AIPW + E-value + RCT benchmark +
                                  c-for-benefit. Reuses ev.twin_samples_15 (outcome model)
                                  + gb.assemble_features/frame_feature (L, A).
run_tte.py                  NEW  thin orchestrator: frozen pipeline -> L/A/weights ->
                                  marginal AIPW per outcome -> benchmark/E-value ->
                                  c-for-benefit -> tte_* artifacts + manifest block.
evaluate_twin.py            HOOK additive evaluate_distributional(...) + optional run_tte
                                  call. Insert in the fresh session; do NOT edit the W5
                                  session's evaluate_flow/evaluate_simulator bodies.
RUN_MANIFEST.json           HOOK add ps_model, weight summaries, rct_anchors, seeds.
```

Rationale for new files over editing `calibration_twin.py`: that file is owned by the W5
session. A sibling `distributional_metrics.py` that *imports* its PIT/coverage keeps a
single source of truth without a merge conflict. If the sessions later merge, fold
`distributional_metrics.py` into `calibration_twin.py` - the function boundaries are
designed for it.

---

## PART A - Distributional metrics (`distributional_metrics.py`)

Every function operates on the sample array the twin already produces:
`ev.scatter_to_full(ev.twin_samples_15(...))[:, :, dim]` -> `(n_patients, n_samples)` for
one horizon, original units. `obs`/`mask` come from the same place `calibration_twin`
reads them. Every calibration metric is reported WITH a sharpness number next to it
(Gneiting's principle: maximize sharpness subject to calibration).

### A1. Proper scores you are missing (advisor Q1: "calibration beyond MAD/RMSE")

CRPS already exists (reuse). Add:

```python
"""distributional_metrics.py - probabilistic scores for the twin's predictive samples.

Design rules:
  * CRPS is NOT redefined here - import baselines_trajectory.crps_ensemble (one CRPS,
    repo-wide). PIT/coverage live in calibration_twin; this module imports them for the
    combined report and adds the scores W5 does not cover.
  * All functions take sample arrays (n, m) or blocks (n, m, k), observed (n,) or (n, k),
    an optional boolean mask, and optional per-patient weights w (for IPCW/IPTW-weighted
    scores, Part-A5). Weighted mean is the only weighting; scores themselves are unchanged.
  * Pure numpy/scipy. No torch, no side effects, importable anywhere.
"""
import numpy as np
from scipy import stats
from baselines_trajectory import crps_ensemble          # the one CRPS
import calibration_twin as ct                            # PIT, coverage primitives

def _wmean(v, w=None):
    v = np.asarray(v, float)
    if w is None:
        return float(np.nanmean(v))
    w = np.asarray(w, float)
    ok = np.isfinite(v) & (w > 0)
    return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok])) if ok.any() else float("nan")

def log_score_gaussian(samples_h, obs_h, w=None):
    """Gaussian-kernel (moment-matched) log score per horizon.

    The flow's exact density via the probability-flow ODE is the *right* log score but is
    expensive; this moment-matched surrogate (fit N(mean,var) to each patient's samples,
    score the observed) is the cheap, tail-sensitive stand-in. Report the exact NLL only if
    the ODE likelihood is wired up. Lower = better; punishes overconfident tails (the 12m
    spike) far harder than CRPS.
    """
    mu = np.nanmean(samples_h, axis=1)
    sd = np.nanstd(samples_h, axis=1) + 1e-6
    ll = stats.norm.logpdf(obs_h, loc=mu, scale=sd)      # (n,)
    return -_wmean(ll, w), -ll

def energy_score_block(samples_block, obs_block, w=None, max_samples=100):
    """Multivariate CRPS over a horizon block (n, m, k). Complete cases in the block only.

    ES = E||X - y|| - 0.5 E||X - X'||, Euclidean over the k horizons. Grades the JOINT
    distribution across horizons (and across BMI+HbA1c if the block mixes them) - the
    cross-horizon correlation the per-horizon CRPS is blind to. Subsample to max_samples
    for the O(m^2) pairwise term.
    """
    S = samples_block[:, :max_samples, :]
    n, m, k = S.shape
    es = np.full(n, np.nan)
    for i in range(n):
        y = obs_block[i]
        if not np.all(np.isfinite(y)):
            continue                                     # joint score needs the full block
        Si = S[i]
        d_sy = np.linalg.norm(Si - y, axis=1).mean()
        diff = Si[:, None, :] - Si[None, :, :]
        d_ss = np.linalg.norm(diff, axis=2).mean()
        es[i] = d_sy - 0.5 * d_ss
    return _wmean(es, w), es

def variogram_score_block(samples_block, obs_block, p=0.5, w=None, max_samples=100):
    """Variogram score of order p over a horizon block - sensitive to the DEPENDENCY
    structure (pairwise differences) rather than the marginals. Complements the energy
    score; catches a model that gets each horizon's marginal right but the trajectory
    SHAPE / correlation wrong.
    """
    S = samples_block[:, :max_samples, :]
    n, m, k = S.shape
    vs = np.full(n, np.nan)
    for i in range(n):
        y = obs_block[i]
        if not np.all(np.isfinite(y)):
            continue
        Ey = np.abs(y[:, None] - y[None, :]) ** p                       # (k, k)
        ES = (np.abs(S[i][:, :, None] - S[i][:, None, :]) ** p).mean(0) # (k, k)
        vs[i] = np.sum((Ey - ES) ** 2)
    return _wmean(vs, w), vs
```

### A2. Interval calibration done right (advisor Q1; prevents gaming coverage with width)

```python
def interval_score(samples_h, obs_h, alpha=0.10, w=None):
    """Winkler/interval score for the central (1-alpha) band. Rewards correct coverage AND
    narrow width jointly - PICP alone is gamed by widening. Lower = better.
    """
    lo = np.nanquantile(samples_h, alpha / 2, axis=1)
    hi = np.nanquantile(samples_h, 1 - alpha / 2, axis=1)
    below, above = obs_h < lo, obs_h > hi
    s = (hi - lo) + (2 / alpha) * (lo - obs_h) * below + (2 / alpha) * (obs_h - hi) * above
    return _wmean(s, w), s

def coverage_curve(samples_h, obs_h, levels=(0.5, 0.8, 0.9, 0.95), w=None):
    """Empirical coverage at several nominal levels -> a coverage-calibration curve. Uses
    calibration_twin.coverage_from_band per level so the band convention matches W5.
    Returns rows (nominal, empirical, mean_width)."""
    rows = []
    for c in levels:
        lo, hi = ct.predictive_band(samples_h, alpha=1 - c)
        cov, width = ct.coverage_from_band(lo, hi, obs_h)   # extend ct fn to accept w, or reweight here
        rows.append({"nominal": c, "empirical": cov, "mean_width": width})
    return rows

def pinball_loss(samples_h, obs_h, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9), w=None):
    """Mean pinball/quantile loss over a grid - the quantile analog of CRPS and the natural
    readout for the W5 conformal step."""
    out = {}
    for q in quantiles:
        pred_q = np.nanquantile(samples_h, q, axis=1)
        d = obs_h - pred_q
        loss = np.where(d >= 0, q * d, (q - 1) * d)
        out[q] = _wmean(loss, w)
    return out

def sharpness(samples_h, w=None):
    """Report next to every calibration number. Mean predictive SD and mean 90% width."""
    sd = np.nanstd(samples_h, axis=1)
    lo, hi = np.nanquantile(samples_h, 0.05, axis=1), np.nanquantile(samples_h, 0.95, axis=1)
    return {"mean_sd": _wmean(sd, w), "mean_width90": _wmean(hi - lo, w)}
```

### A3. Calibrate the ACTUAL deliverable - threshold probabilities (advisor Q3, highest value)

The threshold probability `P(BMI_t<35)`, `P(HbA1c_t<5.7)` is the number that goes in front
of a clinician. CRPS/PIT grade the whole distribution, NOT that specific functional. Grade
it directly.

```python
def threshold_calibration(p_pred, y_obs, n_bins=10, w=None):
    """Reliability of a predicted threshold probability against observed crossings.

    p_pred: predicted P(cross) per patient (fraction of samples past the cut, from
            bmi_threshold_probability.default_threshold_targets machinery).
    y_obs : observed 1{crossed} at that horizon, OBSERVED patients only (join on mask).
    Returns ECE, MCE, Brier, and the reliability table (predicted vs observed per bin).
    Pass w = IPCW weights (Part A5) so informative attrition does not bias the curve.
    """
    p, y = np.asarray(p_pred, float), np.asarray(y_obs, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    W, ece, rows, gaps = w.sum(), 0.0, [], []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        wb = w[sel].sum()
        conf = np.average(p[sel], weights=w[sel])
        acc = np.average(y[sel], weights=w[sel])
        ece += (wb / W) * abs(acc - conf)
        gaps.append(abs(acc - conf))
        rows.append({"bin": b, "pred": conf, "obs": acc, "weight": wb, "n": int(sel.sum())})
    brier = np.average((p - y) ** 2, weights=w)
    return {"ece": ece, "mce": max(gaps) if gaps else float("nan"), "brier": float(brier),
            "table": rows}

def brier_decomposition(p_pred, y_obs, n_bins=10, w=None):
    """Murphy decomposition BS = reliability - resolution + uncertainty. Separates
    calibration (reliability) from discrimination (resolution) for the threshold event."""
    p, y = np.asarray(p_pred, float), np.asarray(y_obs, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    ybar = np.average(y, weights=w)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rel = res = 0.0
    W = w.sum()
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        wb = w[sel].sum()
        pb = np.average(p[sel], weights=w[sel])
        ob = np.average(y[sel], weights=w[sel])
        rel += (wb / W) * (pb - ob) ** 2
        res += (wb / W) * (ob - ybar) ** 2
    unc = ybar * (1 - ybar)
    return {"reliability": rel, "resolution": res, "uncertainty": unc, "brier": rel - res + unc}
```

### A4. Calibration drift - the actual SOPHIA lesson (advisor Q1)

The point is not "report calibration once," it is "calibration collapsed out-of-sample."
Report the drift by recomputing on the temporal test fold (W3) and showing the delta.

```python
def calibration_slope_intercept(p_pred, y_obs, w=None):
    """Cox calibration for a binary/threshold probability.
      calibration slope     = coef of logit(p) in  logit(P(y=1)) ~ logit(p)   [1 = ideal;
                              <1 = overfit/optimistic, the SOPHIA failure mode].
      calibration-in-large  = intercept of  logit(P(y=1)) ~ offset(logit(p))  [0 = ideal].
    Unregularized logistic fits (C huge). Compute on INTERNAL and on the TEMPORAL fold;
    the manifest stores both plus the delta.
    """
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p_pred, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_obs, int)
    lp = np.log(p / (1 - p))
    slope = LogisticRegression(penalty=None, solver="lbfgs").fit(
        lp[:, None], y, sample_weight=w).coef_[0, 0]
    # CITL: slope fixed at 1 via offset -> fit intercept only.
    citl = LogisticRegression(penalty=None, solver="lbfgs", fit_intercept=True).fit(
        np.zeros((len(y), 1)), y, sample_weight=w).intercept_[0] + np.mean(lp - lp)  # see note
    return {"slope": float(slope), "citl": float(np.average(y, weights=w) - np.average(p, weights=w))}
    # NOTE: implement CITL cleanly with statsmodels GLM offset if available; the mean(y)-mean(p)
    # form above is the simple calibration-in-the-large proxy and is fine for the report.
```

### A5. Make every metric attrition-robust (advisor Q2: 3468 -> 711)

Every score above is, by default, computed on patients OBSERVED at horizon t - the
non-dropouts. If attrition is informative the numbers are optimistically biased. Fix it
inside the metric with IPCW weights (the SAME censoring model `causal_tte.censoring_model`
uses), and additionally report calibration stratified by follow-up completeness.

```python
def ipcw_from_model(p_observed, observed_mask, clip=0.05):
    """Per-patient inverse-probability-of-censoring weight at one horizon.
    p_observed: P(observed at h | L) from causal_tte.censoring_model (fit on TRAIN).
    Unobserved patients get weight 0 (they contribute via the model, not the empirical term).
    """
    w = np.where(observed_mask, 1.0 / np.clip(p_observed, clip, 1.0), 0.0)
    return w

def stratified_calibration(p_pred, y_obs, has_long_followup, **kw):
    """Run threshold_calibration separately for patients WITH vs WITHOUT long follow-up.
    A large gap between strata is the direct diagnostic that attrition is informative."""
    return {"long_fu": threshold_calibration(p_pred[has_long_followup], y_obs[has_long_followup], **kw),
            "short_fu": threshold_calibration(p_pred[~has_long_followup], y_obs[~has_long_followup], **kw)}
```

Wire IPCW into the proper scores by passing `w = ipcw_from_model(...)` to any A1-A3
function. Report each headline metric twice: naive (observed-only) and IPCW-weighted. The
gap between them IS the quantitative answer to "does attrition bias our numbers."

### A6. Distribution distance for Mode-C marginals (advisor Q1 "discuss the distribution")

```python
def marginal_distance(sim_h, obs_h):
    """Upgrade the Mode-C KS check from a p-value to a magnitude. With n~28k, KS p -> 0 is
    uninformative; Wasserstein-1 says HOW FAR the simulated BMI marginal sits from observed,
    in real units (quantifies the known BMI-high bias)."""
    sim, obs = sim_h[np.isfinite(sim_h)], obs_h[np.isfinite(obs_h)]
    return {"wasserstein1": float(stats.wasserstein_distance(sim, obs)),
            "ks_stat": float(stats.ks_2samp(sim, obs).statistic),
            "median_shift": float(np.median(sim) - np.median(obs))}
```

### A-artifacts (tag `dist_`)

| File | Content |
|---|---|
| `dist_proper_scores_test.csv` | per-horizon CRPS, log-score, interval-score, energy/variogram (block); naive + IPCW columns |
| `dist_coverage_curve_test.{csv,png}` | nominal vs empirical coverage at 50/80/90/95 + mean width + sharpness |
| `dist_threshold_calibration_{bmi35,hba1c57}_test.{csv,png}` | reliability table + ECE/MCE/Brier per horizon, IPCW-weighted |
| `dist_calibration_drift.csv` | slope + CITL on internal vs temporal fold, with delta |
| `dist_attrition_sensitivity.csv` | every headline metric naive vs IPCW-weighted; stratified-by-follow-up calibration |
| `dist_modeC_marginal_distance.csv` | Wasserstein/KS/median-shift per horizon (replaces bare KS p) |

---

## PART B - Target-trial emulation (`causal_tte.py`)

The twin's per-arm predictions (`mu1`/`mu0` in `evaluate_twin.evaluate_simulator`, or
`ev.twin_samples_15(..., flip_surgery=)`) ARE the g-computation outcome model
`E[Y|A=a,L]`. AIPW combines them with a propensity model for double robustness. Nothing
here retrains the twin.

### B0. Treatment, time-zero, eligibility

```python
"""causal_tte.py - target-trial emulation for the sleeve-vs-RYGB contrast.

The emulated trial (protocol in METHODS_causal_distributional.md):
  eligibility  = the existing cohort filters (GLP-1-naive, T2D, first MBS, SG or RYGB)
  strategies   = SG (CptCode 43775) vs RYGB (43644/43846); 43645 -> RYGB (see DECISION 1)
  time zero    = ProcDateValue (point intervention: no immortal time; ITT == per-protocol)
  assignment   = observational -> assume conditional exchangeability given L, adjust by PS
  outcomes     = BMI/HbA1c trajectory, threshold probs, composite complication
  estimand     = marginal ATE (AIPW) + individual CATE (the twin)
  analysis     = doubly-robust IPCW-AIPW; benchmark vs RCT; E-value; c-for-benefit

Reuses ev.twin_samples_15 (outcome model), gb.assemble_features/frame_feature (L, A),
gb.make_splits (shared split). Nuisance models fit on TRAIN, applied to TEST (leak-free).
"""
import numpy as np
from scipy import stats
import train_flow_matching as fm
import gbm_mace_baseline as gb
import evaluate_twin as ev

INCLUDE_RACE_IN_PS = False   # DECISION (PI): race -> fairness audit only, not the PS model.
RYGB_CPTS = {"43644", "43846", "43645"}   # DECISION 1: 43645 (the flagged variant) -> RYGB.
SLEEVE_CPTS = {"43775"}

def build_L_A(dataset):
    """Confounder matrix L (pre-op only) and treatment A (1=RYGB, 0=SG).

    Starts from gb.assemble_features (surgery_idx + GBM extras already there), DROPS the
    treatment column, and optionally appends SES/geo confounders not already in the GBM
    matrix via gb.frame_feature. Verify feature_names against the live gbm_mace_baseline.
    """
    x, feat, _ = gb.assemble_features(dataset)
    a_col = feat.index("surgery_idx")
    A = x[:, a_col].astype(int)
    keep = [j for j in range(x.shape[1]) if j != a_col]
    L, L_names = x[:, keep], [feat[j] for j in keep]
    # append confounders not in the GBM design (guard duplicates); trees take NaN.
    for canon in ["CoverageClass_num", "RUCA_num"]:   # numeric-encoded; see gb.frame_feature
        col = gb.frame_feature(dataset, canon)
        if col is not None and canon not in L_names:
            L = np.column_stack([L, col]); L_names.append(canon)
    return L, A, L_names
```

### B1-B3. Propensity, stabilized weights, overlap, balance

```python
def _tree_classifier():
    """XGBoost on the VM (matches the GBM backend), HistGB fallback locally. Native NaN."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.8, eval_metric="logloss", tree_method="hist")
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05)

def propensity_scores(L, A, train_idx, test_idx):
    """P(RYGB | L). Fit on TRAIN, predict TEST (leak-free). Returns ps on TEST rows."""
    clf = _tree_classifier().fit(L[train_idx], A[train_idx])
    return clf.predict_proba(L[test_idx])[:, 1], clf

def stabilized_iptw(A, ps, trim=(0.02, 0.98)):
    """Stabilized IPTW + common-support trim. Returns (weights, keep_mask)."""
    lo, hi = trim
    keep = (ps > lo) & (ps < hi)
    pt = A.mean()
    sw = np.where(A == 1, pt / np.clip(ps, 1e-3, 1), (1 - pt) / np.clip(1 - ps, 1e-3, 1))
    return sw, keep

def standardized_mean_diff(L, A, w=None):
    """Per-covariate SMD before/after weighting (Love-plot data). |SMD|<0.1 = balanced."""
    out = []
    for j in range(L.shape[1]):
        col = L[:, j]; m = np.isfinite(col)
        def stat(sel):
            c = col[sel & m]; ww = None if w is None else w[sel & m]
            mu = np.average(c, weights=ww); var = np.average((c - mu) ** 2, weights=ww)
            return mu, var
        m1, v1 = stat(A == 1); m0, v0 = stat(A == 0)
        out.append((m1 - m0) / np.sqrt((v1 + v0) / 2 + 1e-12))
    return np.array(out)
```

### B4. IPCW censoring weights (shared with Part A5)

```python
def censoring_model(L, observed_mask, train_idx, test_idx, clip=0.05):
    """P(observed at horizon h | L). One model per horizon. observed_mask = per-horizon
    non-null outcome indicator (same mask the flow eval uses). DeathInterval defines the
    competing risk; for the metabolic outcomes death is a censoring event (you cannot
    measure BMI post-mortem) - document as DECISION 4. Returns p_observed on TEST + the IPCW.
    """
    clf = _tree_classifier().fit(L[train_idx], observed_mask[train_idx].astype(int))
    p_obs = clf.predict_proba(L[test_idx])[:, 1]
    ipcw = np.where(observed_mask[test_idx], 1.0 / np.clip(p_obs, clip, 1.0), 0.0)
    return p_obs, ipcw
```

### B5. Doubly-robust IPCW-AIPW (the marginal effect; twin = outcome model)

```python
def aipw(Y, A, delta, ps, pc, mu1, mu0):
    """Doubly-robust IPCW-augmented IPW ATE for one outcome at one horizon.

      Y    observed outcome (n,), NaN where unobserved (gated out by delta).
      A    treatment (1=RYGB, 0=SG).
      delta observed-at-horizon indicator (1/0).
      ps   P(A=1|L)          -- propensity model.
      pc   P(observed|L)     -- censoring model.
      mu1  E[Y|A=1,L], mu0   -- the twin's per-arm predictions (g-computation).

    Consistent if EITHER (ps AND pc) OR (mu1,mu0) is right. Returns ATE, SE (influence
    function), 95% CI. For a binary threshold outcome pass Y=1{crossed}, mu=twin threshold
    prob -> ATE is the causal RISK DIFFERENCE ('X% vs Y%').
    """
    r = np.where(delta == 1, 1.0 / np.clip(pc, 0.05, 1.0), 0.0)   # IPCW factor
    Yf = np.nan_to_num(Y)
    m1 = mu1 + A * r * (Yf - mu1) / np.clip(ps, 1e-3, 1)
    m0 = mu0 + (1 - A) * r * (Yf - mu0) / np.clip(1 - ps, 1e-3, 1)
    psi = m1 - m0
    ate = float(psi.mean()); se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    return {"ate": ate, "se": se, "ci": (ate - 1.96 * se, ate + 1.96 * se)}
```

The `mu1`/`mu0` come straight from the twin, per outcome/horizon:
```python
# continuous BMI/HbA1c at dim d:
s1 = ev.scatter_to_full(ev.twin_samples_15(model, arrays1, event, cfg, pre, dev))[:, :, d]
mu1 = s1.mean(axis=1)                       # E[BMI_d | A=1, L]
# threshold outcome P(BMI_d < 35):
mu1 = (s1 < 35.0).mean(axis=1)              # E[1{cross} | A=1, L]
```
where `arrays1` clamps `surgery_idx = 1` (and `arrays0 = 0`) via the
`bmi_threshold_probability.cohort_probability` pattern. Reuse that function; do not rewrite
the clamp.

### B6. E-value (advisor Q4/Q5: quantify the unmeasured-confounding threat)

```python
def e_value(rr, lo=None, hi=None):
    """VanderWeele-Ding E-value for a risk ratio and (optionally) the CI limit nearest the
    null. For a standardized mean difference d, convert first: rr = exp(0.91*d)."""
    def ev1(x):
        x = x if x >= 1 else 1.0 / x
        return x + np.sqrt(x * (x - 1))
    point = ev1(rr)
    bound = None
    if lo is not None and hi is not None:
        bound = 1.0 if lo <= 1 <= hi else ev1(lo if lo > 1 else hi)
    return {"e_point": point, "e_bound": bound}

def smd_to_rr(d):
    return float(np.exp(0.91 * d))          # continuous-effect -> approximate RR
```

### B7. Borrowed RCT backbone (advisor Q6; the defensible novelty framing)

```python
# DECISION 6: the RCT backbone exists ONLY for weight and glycemia (the trials measured
# them). There is NO head-to-head RCT powered for the MACE/nephropathy/retinopathy
# composite -> the complication contrast stays observational/exploratory, NOT anchored.
# VERIFY these numbers against the papers before quoting; they are illustrative anchors.
RCT_ANCHORS = {
    "twl_pct_1_2y":   {"delta_rygb_minus_sg": +5.0, "ci": (+3.0, +7.0),   # %TWL, RYGB greater
                       "src": "SM-BOSS/SLEEVEPASS 5-10y; convert your BMI-pts to %TWL first"},
    "t2d_remission":  {"rr_rygb_vs_sg": 1.4, "ci": (1.15, 1.7),
                       "src": "Oseberg 1y remission 74% RYGB vs 48% SG; define remission to match"},
}
def benchmark_vs_rct(emulated_estimate, emulated_ci, anchor_key):
    """Does the emulated marginal effect land in the RCT confidence region? Overlap = the
    observational design reproduced the randomized answer (empirical calibration of the
    causal design, RCT-DUPLICATE logic)."""
    a = RCT_ANCHORS[anchor_key]
    lo_e, hi_e = emulated_ci
    lo_r, hi_r = a["ci"]
    overlap = not (hi_e < lo_r or hi_r < lo_e)
    return {"anchor": anchor_key, "emulated": emulated_estimate, "emulated_ci": emulated_ci,
            "rct": a, "overlaps_rct_ci": overlap}
```

**Unit reconciliation (DECISION 7):** the trials report `%TWL`/`%EWL` and remission with
their own HbA1c cut; the twin outputs BMI (kg/m^2) and HbA1c (%). Before `benchmark_vs_rct`,
convert the emulated BMI-point difference to `%TWL` using baseline weight, and state your
`P(HbA1c<5.7)` threshold vs the trial's remission definition (often <6.0 or <6.5 off meds).
Do NOT compare mismatched endpoints.

### B8. Calibration-for-benefit / c-for-benefit (advisor Q3: validate the CONTRAST)

```python
def c_for_benefit(pred_ite, A, Y, ps, lower_is_better=True):
    """Van Klaveren concordance-for-benefit: validates the INDIVIDUALIZED treatment-effect
    estimate (the per-patient RYGB-vs-SG difference the twin produces), which is the actual
    product. Match each treated patient to the nearest-PS control, form pairs, compute the
    observed pair benefit (outcome_control - outcome_treated, or reversed if higher=better)
    and the predicted pair benefit (mean of the pair's predicted ITEs), then the concordance
    between them. ~0.5 = no benefit discrimination; higher = the twin ranks who-benefits.
    Requires observed outcomes in both pair members (join on mask).
    """
    treated = np.where(A == 1)[0]; control = np.where(A == 0)[0]
    if len(treated) == 0 or len(control) == 0:
        return {"c_for_benefit": float("nan"), "n_pairs": 0}
    # nearest-PS matching without replacement
    ctrl_ps = ps[control]; used = np.zeros(len(control), bool); pairs = []
    for t in treated:
        d = np.abs(ctrl_ps - ps[t]); d[used] = np.inf
        j = int(np.argmin(d))
        if np.isfinite(d[j]):
            used[j] = True; pairs.append((t, control[j]))
    obs_ben, pred_ben = [], []
    for t, c in pairs:
        if not (np.isfinite(Y[t]) and np.isfinite(Y[c])):
            continue
        b = (Y[c] - Y[t]) if lower_is_better else (Y[t] - Y[c])
        obs_ben.append(b); pred_ben.append(0.5 * (pred_ite[t] + pred_ite[c]))
    obs_ben, pred_ben = np.array(obs_ben), np.array(pred_ben)
    # concordance over pairs of pairs with discordant observed benefit
    conc = disc = 0
    for i in range(len(obs_ben)):
        for k in range(i + 1, len(obs_ben)):
            if obs_ben[i] == obs_ben[k]:
                continue
            hi_obs = i if obs_ben[i] > obs_ben[k] else k
            hi_pred = i if pred_ben[i] > pred_ben[k] else k
            conc += (hi_obs == hi_pred); disc += (hi_obs != hi_pred)
    c = conc / (conc + disc) if (conc + disc) else float("nan")
    return {"c_for_benefit": float(c), "n_pairs": len(obs_ben)}
```
The O(pairs^2) loop is fine for n~4k pairs at report time; vectorize only if slow.

### B-artifacts (tag `tte_`)

| File | Content |
|---|---|
| `tte_propensity_overlap.{csv,png}` | PS distribution by arm + trimmed n (positivity) |
| `tte_covariate_balance_love.{csv,png}` | SMD per covariate, before vs after IPTW |
| `tte_marginal_effects.csv` | AIPW ATE + 95% CI per outcome/horizon (BMI, HbA1c, threshold RD, composite RD) |
| `tte_rct_benchmark.csv` | emulated vs RCT anchor, overlap flag (weight/glycemia only) |
| `tte_evalue.csv` | E-value (point + CI bound) per primary contrast |
| `tte_c_for_benefit.csv` | c-for-benefit + n_pairs for BMI/HbA1c benefit |
| `tte_weights_summary.json` | IPTW/IPCW min/max/ESS, trim count, PS AUC |

---

## PART C - Orchestration & integration

### C1. `run_tte.py` (thin, mirrors `bmi_threshold_probability.main` boilerplate)

```
1. resolve frozen pipeline (ev.resolve_from_pipeline) -> gbm_run, twin_run, shared split cfg
2. dataset = ev.load_dataset(csv|None); splits = gb.make_splits(dataset, gbm_cfg)
3. L, A, names = ct2.build_L_A(dataset)          # ct2 = causal_tte
4. ps, ps_clf = ct2.propensity_scores(L, A, splits['train'], splits['test'])
5. sw, keep   = ct2.stabilized_iptw(A[splits['test']], ps)
6. model/pre/cfg = ev.restore_twin(...); for each (outcome dim, horizon):
      mu1, mu0 = twin per-arm predictions (reuse bmi_threshold_probability.cohort_probability)
      p_obs, ipcw = ct2.censoring_model(L, mask_h, train, test)
      res = ct2.aipw(Y_h, A_te, delta_h, ps, p_obs, mu1, mu0)
7. benchmark_vs_rct (weight/glycemia), e_value, c_for_benefit
8. write tte_* artifacts + a manifest block (below). report_saved() each file (VM-friendly).
```
Fit PS/censoring with `split_strategy` from the pipeline config so temporal runs (W3) carry
through automatically.

### C2. `evaluate_twin.py` additive hook (insert in the fresh session; do not touch W5 bodies)

Add a top-level `evaluate_distributional(model, dataset, splits, pre, twin_cfg, gbm, out, ...)`
that: builds the test sample block once, calls the Part-A functions per horizon and per
threshold family, and writes the `dist_*` artifacts. Call it from `evaluate(...)` after
`evaluate_flow`. Optionally shell out to `run_tte.py` (or import `run_tte.main_programmatic`)
so one `evaluate_twin` run emits `dist_*` and `tte_*` together. Guard behind `--with-causal`
so a quick eval can skip the heavier TTE pass.

### C3. `RUN_MANIFEST.json` additions (extends W1)

```json
"causal_tte": {
  "treatment_coding": {"sleeve": ["43775"], "rygb": ["43644","43846","43645"]},
  "ps_model": {"backend": "xgboost", "auc": 0.71, "features": ["...L names..."],
               "include_race": false, "trim": [0.02, 0.98], "n_trimmed": 118},
  "weights": {"iptw_ess": 3910, "ipcw_ess_by_horizon": {"bmi_12m": 3402, "bmi_6y": 690}},
  "rct_anchors": { "...": "verify-before-quote" },
  "absent_confounders": ["GERD", "surgeon_or_center_id", "smoking"],
  "seeds": {"ps": 0, "censoring": 0}
}
```

---

## PART D - Decisions to encode explicitly in code (comments/docstrings)

1. **CPT -> treatment:** 43775 = SG; 43644/43846 = RYGB; **43645 -> RYGB** (the flagged
   variant). One place: `causal_tte.RYGB_CPTS`. This also fixes the "unrecognized CPT"
   drop flagged in `MACE_MODELING_DECISIONS.md`.
2. **Confounder set L:** GBM design matrix minus `surgery_idx`, plus SES/geo. List it in
   the manifest. Name the ABSENT confounders (GERD, surgeon/center, smoking) in the code
   and the manifest - they are the exchangeability caveat.
3. **Race:** `INCLUDE_RACE_IN_PS = False` (fairness audit only) unless the PI opts in.
4. **Censoring vs competing risk:** death (`DeathInterval`) censors the metabolic
   trajectory (cannot measure BMI post-mortem); model observation-at-horizon directly.
   Note the competing-risk framing for the complication endpoint as future work.
5. **Nuisance models fit on TRAIN, applied to TEST** (leak-free; no test-set cross-fit),
   honoring the shared `split_strategy`.
6. **RCT backbone is weight/glycemia only** - the complication contrast is NOT anchored and
   is reported as exploratory with the widest E-value caveat.
7. **Unit/endpoint reconciliation** before any RCT benchmark (BMI-pts -> %TWL;
   `P(HbA1c<5.7)` vs the trial remission definition).
8. **Estimand:** primary = marginal ATE (AIPW, doubly robust); secondary = individual CATE
   (twin), validated by c-for-benefit. State ITT == per-protocol (point intervention).

---

## PART E - Smoke test + acceptance criteria (fake cohort, `mbsaqip_flow/.venv`)

- `build_L_A` returns `L` with NO `surgery_idx` column and `A in {0,1}`; assert.
- `propensity_scores` fit/predict runs; on 52 rows PS AUC is meaningless - assert only
  shape + finite. `stabilized_iptw` weights finite, ESS <= n.
- `aipw` on a tiny synthetic where mu1/mu0 equal the truth returns ATE ~ truth (DR sanity).
- `energy_score_block`/`variogram_score_block` finite on complete-case blocks; skip
  patients with a missing horizon (assert NaN handling).
- `threshold_calibration` ECE in [0,1]; `calibration_slope_intercept` returns finite slope
  on n>=10, guarded NaN below.
- `e_value(1.0)` == 1.0; `e_value(2.0)` == 2+sqrt(2) ~ 3.41 (unit test).
- `c_for_benefit` returns 0.5 +/- noise on randomized (ps==const) synthetic; n_pairs>0.
- Full `run_tte.py --csv fake_data/fake_mbs_cohort.csv --pipeline <smoke pipeline>` writes
  all `tte_*` files without raising; every stat guarded for n<10.
- `evaluate_twin --with-causal` emits `dist_*` and `tte_*` together.

Acceptance: on the VM, `tte_marginal_effects.csv` shows RYGB BMI/HbA1c differences in the
expected direction (negative = RYGB lower), `tte_rct_benchmark.csv` overlap flag is TRUE
for weight/glycemia, E-values are reported, and `dist_attrition_sensitivity.csv` shows the
naive-vs-IPCW gap. None of these gate the build - they are the evidence for the methods.

---

## Suggested order

```
E-unit-tests (e_value, aipw DR sanity, crps reuse)      # fastest, no data
-> A1-A3 proper scores + threshold calibration          # reuses existing sample block
-> A5 IPCW wiring + A4 drift (needs W3 temporal fold)
-> B0-B5 PS/IPTW/IPCW/AIPW marginal effects
-> B6-B8 E-value + RCT benchmark + c-for-benefit
-> C1 run_tte.py -> C2 evaluate_twin hook -> C3 manifest
-> A6 Mode-C distance (fold into evaluate_simulator output)
```

B0-B5 and A1-A3 are each under ~150 lines and independently shippable. c-for-benefit (B8)
and the categorical SES encoding are the chunkier pieces. Start with the unit tests and the
proper-score additions - they unlock the "distributional evaluation" half of the advisor
response immediately, before the causal layer lands.
