# Next-run plan - model improvements for the next set of runs

Build plan for the next iteration of the MBSAQIP / Cosmos digital-twin. Turns the
six-item improvement list into concrete, sequenced code changes with file/line
anchors and the traps to avoid. Rationale for the modeling framing lives in
`MACE_MODELING_DECISIONS.md`; this doc is the engineering plan that sits on top of it.

Target: a submission to a Lancet-family journal (Lancet Diabetes & Endocrinology or
Lancet Digital Health). Temporal validation is the agreed pre-submission minimum.

---

## Hard constraints (carry over from the existing build discipline)

- Do NOT modify the pristine Cosmos core: `train_flow_matching.py` and
  `tune_flow_matching_optuna.py` stay reusable-by-import. New work is new scripts or
  edits to already-forked scripts. The ONE exception this iteration is a *narrow,
  additive* split helper in `train_flow_matching.py` (W3) - a new function, no change
  to existing behavior or the default split path.
- Feature additions go to the GBM ONLY, via `assemble_features` pulling from
  `dataset.frame`. Do NOT edit the shared `fm.PATIENT_FEATURES` (it feeds the flow +
  twin too). See `GBM_EXTRA_FRAME_FEATURES`.
- Keep the GLP-1 filter (`PriorGLP1 = 0`). It is a deliberate design choice: it
  isolates a clean GLP-1-naive surgical baseline. Report it in the attrition flow;
  do not silently drop it.
- Leakage discipline is strict. Any new feature must be a pre-operative / baseline
  value. Verify before adding (see W2).
- Trees route missing values natively; pass continuous/binary features un-imputed
  (NaN preserved). Never resample for imbalance; handle it at the loss level only.
- Smoke-test every change on `fake_data/fake_mbs_cohort.csv` with the
  `mbsaqip_flow/.venv` before declaring it done. Real cohort + xgboost + DB run on the
  Cosmos VM.

---

## Assessment of the six suggestions (and the sequencing call)

All six are correct and in roughly the right order. Two adjustments:

1. **Do the freeze (W1) twice** - a "before" baseline now so any gain is provable, and
   an "after" once the changes land.
2. **Move the flow ablation (W4) earlier.** The Mode A -> B gap is already ~0, which
   predicts event conditioning barely helps the trajectory. If the ablation confirms
   that, the "coupled digital twin" claim weakens and we should honestly simplify to an
   unconditional flow + independent GBM. That is a headline-level decision, so run it
   before investing in twin calibration and figures.

Key caveats folded into the work items below:

- **Backend parity.** HistGradientBoosting locally vs XGBoost on the VM are NOT
  comparable. Install xgboost in the local venv and pin backend + version in the run
  manifest (W1).
- **Leakage audit** for `BiguanideStatus` / `SGLT2Status` / `InsulinStatus` - confirm
  baseline, not ever/post-op (W2).
- **eGFR is derived from creatinine + age + sex** (collinear). Fine for trees; do not
  over-read their split importances (W2).
- **Race as a predictor is contested.** Preference: SVI/RUCA as predictors (the
  social-determinant mechanism), race for the fairness audit only. Decision for the PI
  (W2).
- **Temporal test fold is more GLP-1-selected and less follow-up-mature** by
  construction. Report both; they are real and reportable (W3).
- **Trajectory calibration: diagnose with PIT before calibrating.** A location-shifted
  PIT means a biased median (conformal will NOT fix it; same root cause as the Mode-C
  BMI-high bias). A U-shaped PIT means under-dispersion (conformal / quantile
  calibration fixes it) (W5).

---

## Work items

Each item is independently shippable. Suggested order at the bottom.

### W1 - Freeze a reproducible run (do first)

Goal: one command produces every artifact for one run, plus a manifest that makes the
run reproducible and the backend explicit.

- New `freeze_run.py` (thin orchestrator): calls `debug_attrition` ->
  `train_twin_pipeline.run_pipeline` (`train_twin_pipeline.py` line 108) ->
  `evaluate_twin.evaluate` -> `make_table_one`, all pinned to one config, into
  `runs/frozen/<timestamp>/`.
- Write a top-level `RUN_MANIFEST.json`: git SHA (`git rev-parse HEAD`), all seeds,
  `split_strategy`, backend name + `xgboost.__version__` / sklearn / torch versions,
  `pip freeze`, input source + SHA-256 of the CSV/extract, and the resolved GBM/twin
  configs.
- Determinism: in the `train_flow_matching.py` training entry set `torch.manual_seed`,
  `np.random.seed`, and `torch.use_deterministic_algorithms(True)` behind a
  `--deterministic` flag (cuBLAS may need `CUBLAS_WORKSPACE_CONFIG=:4096:8`).
- Install xgboost into `mbsaqip_flow/.venv` so local smoke == VM backend.
- Pin the code SHA AND the patient-feature width in the manifest. Found 2026-07-09: the
  committed fake twin checkpoint has input dim 30 but current code builds dim 32 (the
  6 -> 8 patient-feature expansion, osa + dyslipidemia, commit `d721388`), so it
  load-fails with a size mismatch. A checkpoint must never be silently mismatched to
  code; the manifest is how we catch it. (Also: every locally trained model is on the
  52-row fake cohort - the real twin lives only on the Cosmos VM.)

Gotcha: results across backends are not comparable. The manifest must make the backend
unmistakable so a HistGB smoke run is never mistaken for an XGBoost result.

### W2 - GBM feature expansion (`gbm_mace_baseline.py`)

Goal: raise the risk-model ceiling with features already in the cohort. GBM-only;
`fm.PATIENT_FEATURES` untouched.

**Numeric tier (ship first, ~1 line):**
- Extend `GBM_EXTRA_FRAME_FEATURES` (`gbm_mace_baseline.py` line 76) with:
  `eGFRatEvent`, `BiguanideStatus`, `SGLT2Status`, `SviOverall`, `SviHousehold`,
  `SviTransportation`, `SviMinority`, `SviSES`.
- No other change needed: `frame_feature()` (line 128) + `fm.numeric` already resolve
  these, NaNs route natively, and they propagate to `feature_importances.csv` +
  `config.json` through `assemble_features` (line 146) automatically.

**Missingness indicators:**
- In `assemble_features` (line 146), for each column with >5% missingness (eGFR is
  ~10.3%), append a `<name>_ismissing` 0/1 column (informative missingness).

**Categorical path (`RUCA`, `CoverageClass`, race - chunkier, ship second):**
- `assemble_features` currently `np.hstack`es float64 only. Add a parallel
  `GBM_CATEGORICAL_FRAME_FEATURES` list, build a `pandas.DataFrame` with `category`
  dtype for those columns, and thread the DataFrame (not ndarray) through
  `fit`/`predict`/`permutation_importance`.
- In `make_estimator` (line ~251) pass `enable_categorical=True` (XGBoost >= 2.0) or
  `categorical_features=` (HistGradientBoosting).

**Leakage guard:**
- Before adding `BiguanideStatus` / `SGLT2Status` / `InsulinStatus`, confirm they are
  baseline (pre-op), not ever/post-op. The naming suggests baseline (vs the explicitly
  named `PostOpGLP1`), but verify against the `MBSCohort` table definition. If it can
  not be confirmed, exclude them and note it.

**Race decision (PI):** default to SVI/RUCA as predictors + race for fairness audit
only. Do not add race as a model input without an explicit decision.

Free win: once SVI/RUCA/race columns are loaded, the equity/fairness subgroup analysis
(discrimination + calibration by SVI, RUCA, race, sex) comes almost for free - add it
to `evaluate_twin`.

### W3 - Temporal validation split (`train_flow_matching.py` + `gbm_mace_baseline.py`)

Goal: train on earlier surgery dates, test on later. Feasible now because
`ProcDateValue` is retained in `dataset.frame` (via `required_columns`,
`train_flow_matching.py` line 235) and is row-aligned with `dataset.x`.

- Add `make_temporal_splits(dataset, cfg)` next to `make_stratified_splits`
  (`train_flow_matching.py` line 481):
  `order = np.argsort(pd.to_datetime(dataset.frame["ProcDateValue"]).to_numpy())`,
  then slice earliest `train_frac` / next `val_frac` / latest `test_frac`
  POSITIONALLY (frame is row-aligned with `x`, so use `.to_numpy()` positionally; do
  not rely on the frame index). Optionally stratify-shuffle within the train block only.
- Add a `"temporal"` branch to `gb.make_splits` (`gbm_mace_baseline.py` line 180)
  delegating to `fm.make_temporal_splits`. Because `train_twin_pipeline` pins
  `SHARED_SPLIT_KEYS` (`train_twin_pipeline.py` line 60), the GBM and twin stay
  patient-for-patient aligned automatically once both honor `split_strategy="temporal"`.
- Emit the GLP-1-naive fraction and follow-up maturity per era into the manifest (feeds
  the "shrinking surgical-naive population" narrative and flags thin long-horizon n in
  the late test fold).

Gotcha: one row per patient in `MBSCohort` after filters, so row == patient; the
positional slice is a clean patient split. Later-era patients have shorter follow-up
before the `ProcDateValue <= 2023-05-01` cutoff, so 5-6yr horizon metrics get small n in
the test fold - expected, report it.

### W4 - Flow ablations (`train_flow_matching_twin.py` + new baseline script)

Goal: quantify whether event conditioning actually helps the trajectory. Decision-
relevant (see sequencing note).

- No-event arm: add `use_event: bool = True` to `TwinConfig` (`train_flow_matching_twin.py`
  line ~118, next to `event_emb_dim` line 122). In `TwinNet.encode` (line 216), when
  false, drop the event embedding from the `cat` and shrink `static_dim` (line 197)
  accordingly. One flag gives conditioned vs unconditioned.
- Baselines: new `baselines_trajectory.py` - a per-horizon XGBoost regressor (one per
  BMI/HbA1c target, same conditioning features) and a linear-mixed / ridge baseline.
  Reuse `fm.load_dataset_*` + the shared split.
- Comparison: extend `evaluate_twin` to report held-out per-horizon MAD / RMSE / CRPS +
  flow NLL for {event-flow, no-event-flow, per-horizon XGB, LMM}, with paired tests.
  This table decides whether the coupling claim survives.

### W5 - Trajectory distribution calibration (`evaluate_twin.py`)

Goal: report and fix trajectory calibration. Raw material already exists - `evaluate_flow`
(`evaluate_twin.py` line 490) builds the full sample array `full`
(n_patients x n_samples x 17) and currently collapses it to the median at line ~527.

- Before the median collapse, compute from `full[:, :, dim]`: interval coverage
  (empirical fraction of observed inside [p5, p95] vs nominal 0.90), CRPS per horizon,
  and PIT values (rank of observed among samples). New
  `eval_flow_calibration_{coverage,crps,pit}` CSV/PNG artifacts.
- Diagnose with PIT FIRST (location shift -> biased median, fix the sampler;
  U-shaped -> under-dispersion, calibrate spread).
- Split-conformal: fit per-horizon conformal quantile adjustments on the VAL residuals,
  apply to test, report pre/post coverage. Store the calibrator in `Preprocessing`
  (`train_flow_matching.py` line 503 region) so it is saved with the run.
- Shares a root cause with the Mode-C marginal mismatch (`evaluate_simulator`,
  `evaluate_twin.py` line 541, KS p ~ 0 with BMI biased high) - if PIT shows a location
  shift, fix the sampler bias there too.

Threshold-probability readout (new; `bmi_threshold_probability.py`, added 2026-07-09):
- The same predictive sample array yields clinical threshold probabilities the
  SOPHIA-style point predictor cannot: `P(BMI_t < 35)` and `P(HbA1c_t < 5.7)` = the
  fraction of a patient's samples past the cutoff. `bmi_threshold_probability.py`
  already computes this as a per-surgery counterfactual clamp (the whole test cohort
  forced to each surgery), with e=0 / e=1 / risk-weighted event handling; smoke-tested
  on the fake cohort.
- Generalize it to every output timepoint and to the `HbA1c < 5.7` threshold, and FOLD
  it into `evaluate_twin` so per-patient / per-timepoint threshold probabilities become
  a frozen-run artifact (they feed the W6 figures). These probabilities inherit the
  flow's tail calibration, so only report them after the coverage/PIT pass above.

### W6 - Journal figures (last; results must be frozen first)

Goal: sparse, clinical, vector main figures. Diagnostic plots become supplements.

- New `figures/style.py` with shared matplotlib rcParams (vector PDF/SVG, consistent
  fonts/sizes). Build the main figures: (1) cohort / CONSORT funnel, (2) GBM
  discrimination + calibration + decision-curve, (3) RYGB-vs-sleeve counterfactual,
  (4) trajectory fit with calibrated intervals.
- Route the existing table-screenshots and multi-patient panels to a `supplement/` tag.
- Load the `dataviz` skill when building these.

**Figure spec A - extend the per-patient trajectory figures (BMI and HbA1c) from 3
columns to 5.** The existing 3 columns are: factual band, counterfactual band,
delta-median. Add:
- Column 4: threshold probability under the FACTUAL surgery, plotted at every output
  timepoint (y-axis 0-1).
- Column 5: threshold probability under the COUNTERFACTUAL (flipped) surgery, same
  timepoints.
- Threshold: BMI `< 35`; HbA1c `< 5.7` (normoglycemia / stringent remission - the cohort
  is 100% T2D, so this reads as "probability of returning to a normal A1c").
- Per-patient `P(threshold at t)` = fraction of that patient's samples below the cutoff
  at timepoint t. BMI timepoints: 3m, 6m, 9m, 12m, 2y..6y; HbA1c: 12m, 2y..6y.
- Computation reuses `twin_samples_15` + `scatter_to_full` with surgery clamped to the
  factual value and to the flipped value - the same machinery as
  `bmi_threshold_probability.py`, generalized across timepoints in W5.

**Figure spec B - NEW cohort comparative-effectiveness figure.** Median difference
(rnygb - sleeve) in ABSOLUTE units - BMI in kg/m^2, HbA1c in %-points - at each timepoint
of interest, across the ENTIRE held-out test set, for BMI and HbA1c (two panels or two
lines).
- Clamp the whole test cohort to rnygb, then to sleeve; per patient take the sample
  median at each timepoint; compute the signed difference `rnygb - sleeve` in native
  units (kg/m^2 for BMI, %-points for HbA1c); take the MEDIAN across the test set per
  timepoint; plot vs time, with a bootstrap / IQR band around the median. A negative
  value means RYGB is lower (the expected direction, given the lower-A1c / lower-BMI
  finding).
- This is the RYGB-vs-sleeve headline (the lower-A1c / lower-BMI finding) as one clean
  figure - a direct LDE deliverable.
- UNITS DECIDED (2026-07-09, user): absolute native units (BMI kg/m^2, HbA1c %-points),
  NOT relative percent. ("Percent difference" was the original phrasing; the manuscript
  reports absolute differences.)
- Both figures are counterfactual clamps, so they inherit the same tail-calibration
  caveat: build them AFTER the W5 coverage/PIT calibration, or the threshold crossings
  and differences will carry the flow's tail bias.

### W7 - Re-freeze

Re-run W1 with all changes. Diff the two `RUN_MANIFEST.json` + metric tables for a
before/after delta. This becomes the methods-section ablation evidence.

---

## Suggested order

W1 -> W2 (numeric tier) -> W3 -> W4 -> W5 -> W2 (categoricals) -> W6 -> W7

W2-numeric, W3, and W4-no-event are each under ~50 lines and can land in the first pass.
The categorical path (W2) and the calibration work (W5) are the two chunkier pieces.

## Fast-start recommendation

Start with **W1 + W2-numeric**: low risk, unlocks the before/after comparison, and gives
the AUROC-movement signal fastest.
