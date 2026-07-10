# MACE Modeling — Decisions & History

Running log of modeling decisions for composite‑MACE risk prediction on the
MBSAQIP / Cosmos `MBSCohort` (bariatric surgery outcomes). **Newest entries on top.**

---

## Context

- Cohort: Cosmos `MBSCohort`, ~28.7k patients after filters
  (train 20,085 / val 4,303 / test 4,306).
- Outcome: composite = **MACE OR Nephropathy OR Retinopathy**. Retinopathy +
  nephropathy were added deliberately to raise the event rate above the ~6%
  MACE‑only rate. Test‑set composite prevalence **13.3%** (571 / 4,306).
- Framing decision: **MACE is per‑patient risk prediction on a fixed‑horizon
  binary target, not a generated flow dimension.**
- Scripts:
  - `train_flow_matching.py` — Cosmos flow model (BMI/HbA1c trajectories; originally
    generated MACE as a flow dim). **Left untouched.**
  - `gbm_mace_baseline.py` — GBM (xgboost, sklearn HistGB fallback) MACE risk baseline.
  - `train_flow_matching_multitask.py` — shared encoder → flow head (15 continuous
    BMI/HbA1c dims) + sigmoid MACE head (weighted‑BCE / focal).
  - `evaluate_flow_matching.py`, `evaluate_multitask.py`,
    `evaluate_gbm_mace_baseline.py`, `compare_mace_models.py`,
    `tune_flow_matching_multitask_optuna.py`.
  - All new models default to `split_strategy="surgery"` → they share the flow
    model's exact train/val/test split for **patient‑for‑patient** comparison.

---

## 2026-07-10 (session 6) - Target-trial-emulation causal layer, expanded distributional metrics, equity-fairness audit (all wired into freeze_run)

Build session. Implemented the TTE causal layer + expanded distributional evaluation (spec:
`TTE_DISTRIBUTIONAL_RUN_PLAN.md`; manuscript methods: `METHODS_causal_distributional.md`) and a
new SVI/RUCA/race equity-fairness subgroup audit, all reusing the frozen twin and wired into
`freeze_run.py` so `python freeze_run.py` (and `--split-strategy temporal`) run the full study.

**New modules (all new files; the pristine Cosmos core and W5 `calibration_twin.py` were NOT
touched).**
- `distributional_metrics.py` - pure proper scores + calibration: log-score, energy/variogram
  (joint cross-horizon), interval/Winkler, pinball, sharpness, threshold-probability reliability
  (ECE/MCE/Brier + Murphy decomposition), calibration slope/CITL, IPCW weights + attrition
  stratification, Wasserstein/KS marginal distance. Reuses `baselines_trajectory.crps_ensemble`
  (the one CRPS) and `calibration_twin` PIT/coverage; never reimplements them.
- `causal_tte.py` - pure estimators + nuisance fitters: `build_L_A`, propensity (XGBoost, NaN
  native), stabilized IPTW + trim + SMD/Love, IPCW censoring model, doubly-robust IPCW-AIPW,
  E-value, RCT benchmark, c-for-benefit. No twin sampling, no file writes (unit-testable).
- `run_tte.py` - orchestrator: samples the twin 5x total (4 arm x event blocks + 1 factual),
  emits `tte_*` (causal) and `dist_*` (distributional) artifacts, returns manifest blocks.
- `fairness_audit.py` - per-subgroup twin calibration/error + GBM discrimination + gaps, `fairness_*`.
- `test_{distributional_metrics,causal_tte,fairness_audit}.py` (47 tests). Integration hook added
  to `evaluate_twin.evaluate` (guarded; `--with-causal`/`--with-fairness`) and to `freeze_run.py`
  (`FreezeConfig.with_causal`/`with_fairness`, default ON; `--no-causal`/`--no-fairness` opt out).

**Decisions encoded (plan Part D, as actually implemented).**
1. CPT -> arm: 43775 = SG; 43644 / 43846 / **43645 = RYGB**. Documented constants only
   (`causal_tte.RYGB_CPTS`); the actual `CptCode -> dataset.surgery_idx` mapping is UPSTREAM in the
   loader. TODO: verify 43645 is mapped (not dropped) in the loader - this also fixes the
   "unrecognized CPT" drop noted earlier.
2. Confounder set L = the GBM design matrix minus `surgery_idx` (the 5 SVI percentiles + comorbidity
   flags + eGFR + baseline drug flags are ALREADY in it) plus RUCA + CoverageClass integer codes.
   ABSENT confounders - GERD/reflux, surgeon/center, smoking - are named in code + manifest as the
   exchangeability caveat and the E-value target.
3. Race: `INCLUDE_RACE_IN_PS = False` - race enters the fairness audit ONLY, never the flow, GBM, or
   PS model.
4. Death (`DeathInterval`) censors the metabolic trajectory (cannot measure BMI/HbA1c post-mortem);
   the censoring model targets observation-at-horizon directly. Competing-risk framing for the
   complication endpoint is future work.
5. Nuisance models fit on TRAIN, applied to TEST (no test-set cross-fit). The PS/censoring models
   read the SAME split as the twin/GBM via the frozen `gbm_cfg`, so `--split-strategy temporal`
   carries through with no extra flag.
6. RCT backbone (`causal_tte.RCT_ANCHORS`) covers weight + glycemia ONLY; the composite-complication
   contrast has no randomized anchor and is reported as exploratory with the widest E-value caveat.
   **The anchor numbers are VERIFY-BEFORE-QUOTE** (illustrative pending a source check against
   SLEEVEPASS / SM-BOSS / Oseberg / STAMPEDE).
7. Unit reconciliation before any RCT benchmark: BMI-point ATE -> %TWL proxy (baseline BMI);
   P(HbA1c < 5.7) stated against each trial's remission definition (a definition mismatch is flagged
   in `tte_rct_benchmark.csv`, not silently compared).
8. Estimand: marginal ATE (doubly-robust AIPW) primary; individual CATE (the twin) secondary,
   validated by c-for-benefit. Point intervention -> ITT == per-protocol.

**Corrections to the plan's reference code (verified against the live signatures first).** The
twin's per-EVENT `mu1`/`mu0` in `evaluate_simulator` are NOT per-surgery-arm; the RYGB-vs-SG outcome
model is built from the surgery clamp (the `bmi_threshold_probability.cohort_probability` pattern),
risk-weighted over the event. `frame_feature("*_num")` does not exist (returns None); SVI is already
in `assemble_features`, and RUCA/CoverageClass come via `frame_categorical` + factorize.

**Distributional drift is a CROSS-RUN delta.** A single freeze emits calibration slope/CITL on its
own fold (tagged `split_strategy`); the SOPHIA-style drift is the delta between a `surgery` run and a
`--split-strategy temporal` run, computed from the two `RUN_MANIFEST.json`s. Run both.

**How to run (collaborator, Cosmos VM).** `python freeze_run.py` = full study incl. tte_*/dist_*/
fairness_* (causal + fairness default ON); add `--split-strategy temporal` for the out-of-time fold.
Artifacts land in `runs/frozen/<ts>/evaluation/`; headline blocks (`causal_distributional`,
`fairness`) go into `RUN_MANIFEST.json`. Local smoke: `python freeze_run.py --smoke` (fake cohort;
tiny test n so most stats are NaN by design, but every artifact is written and the wiring runs).

---

## 2026-07-09 (session 5) - Lancet targeting, cohort-funnel understanding, clinical threshold readouts, reproducibility

Strategy + tooling session (no model retrain). Full roadmap lives in a shared artifact;
the engineering plan is `NEXT_RUN_PLAN.md` (W1-W7).

**Journal strategy (benchmarked against SOPHIA).** Saux et al., *Lancet Digital Health*
2023 (CART/LASSO, 7 pre-op features, 5-yr weight-trajectory calculator): external MAD
2.8 / RMSE 4.7 kg/m^2 for BMI. Our trajectory MAD is competitive at matched timepoints;
RMSE runs ~1 kg/m^2 worse with a 12m tail spike. We cannot win as a weight-trajectory
calculator (SOPHIA owns it). **Decision: reframe around the gap SOPHIA concedes** -
cardiometabolic-renal complications + a coupled risk+trajectory twin with
counterfactuals. Primary target **Lancet Diabetes & Endocrinology** (clinical: risk tool
+ RYGB-vs-sleeve comparative effectiveness + diabetes remission); secondary **Lancet
Digital Health** (methods: the twin, gated on fixing the Mode-C marginals). Same body of
work converts to either.

**Keep the GLP-1 filter** (user decision). `PriorGLP1 = 0` is a deliberate inclusion
criterion (a clean GLP-1-naive surgical baseline, analogous to SOPHIA excluding prior
bariatric). Report it in the attrition / CONSORT flow; do not remove. GLP-1-exposed
patients live in the companion `GLP1Cohort`.

**Temporal validation is the pre-submission minimum** (user). Feasible now: `ProcDateValue`
is retained in `dataset.frame` and row-aligned, so a `split_strategy="temporal"` is a
small change (NEXT_RUN_PLAN W3). Later-era test folds are more GLP-1-selected and less
follow-up-mature - reportable, not disqualifying.

**Cohort funnel understood (answers "why only ~28.7k?").** SlicerDicer ~500k bariatric
patients -> `MBSCohort` table 91,357 (no WHERE) -> ~32k exported -> ~28.7k final. The
~5.5x table-build cut is UPSTREAM of the visible query: narrow CPT set
({43775 sleeve, 43644/43846 rnygb}), T2D-only (Table 1 is 100% DM2), and a structured-data
+ longitudinal-presence requirement. WHERE cuts are dominated by `PriorGLP1 = 0`
(-47,832) and `ProcDateValue <= '2023-05-01'` (-28,712). ACTIONS: pull the `MBSCohort`
build DDL; reconcile the query discrepancy (the R loader in `importing in r.txt` has
`AND PMH_DM2 = 1`, but the Python `MBS_SQL` does not, yet Table 1 is 100% DM2 -> the T2D
restriction lives in the table build, and `debug_attrition` replays the Python query so
its per-clause list never shows the diabetes cut); render the funnel as a CONSORT figure.

**New capability - clinical threshold probabilities** (`bmi_threshold_probability.py`,
new). The flow's predictive sample distribution yields threshold probabilities a
point predictor cannot: `P(BMI_t < 35)`, `P(HbA1c_t < 5.7)` = the fraction of a patient's
samples past the cutoff. The script computes this as a per-surgery counterfactual clamp
(the whole test cohort forced to each arm), with e=0 / e=1 / risk-weighted event
handling; smoke-tested on the fake cohort (plumbing only - values meaningless, and the
event columns coincide, consistent with the ~0 Mode A->B coupling). On the real VM twin
this gives the per-surgery percent likelihoods (RYGB vs sleeve) the user asked for. Only
trustworthy once the flow tail is calibrated (NEXT_RUN_PLAN W5). **Two new figures
planned** (NEXT_RUN_PLAN W6): threshold-probability columns added to the per-patient
trajectory figures (factual + counterfactual surgery, every timepoint), and a
cohort-level median difference (rnygb - sleeve) curve for BMI and HbA1c in absolute
units (BMI kg/m^2, HbA1c %-points; user decision 2026-07-09, not relative percent).

**Reproducibility finding.** The committed fake twin checkpoint is stale: input dim 30 vs
current 32 (the 6->8 patient-feature expansion, osa + dyslipidemia, commit `d721388`) -
it load-fails with a size mismatch. Motivates the freeze harness (NEXT_RUN_PLAN W1): pin
the code SHA + patient-feature width in the run manifest.

**Docs added this session:** `NEXT_RUN_PLAN.md` (W1-W7 build plan),
`bmi_threshold_probability.py`. All uncommitted.

---

## 2026-07-07 (session 4) - Feature expansion EXECUTED: risk features to the GBM, comorbidity conditioning to the flow

Acted on the session-2 feature-expansion decision ("features are the ceiling, not the model"). This is
the FIRST change to the shared conditioning vector: a hard constraint in sessions 2-3 was to leave
`fm.PATIENT_FEATURES` untouched, and that constraint is now **deliberately lifted** - we are
intentionally enriching the flow's *trajectory* conditioning, not only the GBM's risk features.

**Placement (clinically motivated split):**
- **GBM risk model** (`GBM_EXTRA_FRAME_FEATURES`): added `PMH_MI`, `PMH_stroke`, `PMH_AFib`, `PMH_VTE`
  (strong macrovascular / thrombotic MACE predictors). Extras list is now DM2, hypertension, MI, stroke,
  AFib, VTE. These are the Tier-A features; DM2 + hypertension (session 3) were the pre-registered *weak*
  ones, so this is the first real test of the ceiling thesis.
- **Flow / twin / multi-task conditioning** (`fm.PATIENT_FEATURES`, 6 -> 8): added `osa` (`PMH_OSA`) and
  `dyslipidemia` (`PMH_dyslipidemia`) - metabolic-syndrome-adjacent, plausibly shaping the BMI/HbA1c
  trajectory. `dyslipidemia` is in BOTH models: it reaches the GBM through the shared vector, so it is
  intentionally NOT duplicated in `GBM_EXTRA_FRAME_FEATURES`.

**Decisions made:**
- **OSA reaches the GBM too** (user call): the GBM is built on top of `patient_features_raw`, so anything
  in the shared vector is visible to it. Rather than a strict flow-only exclusion, we let OSA ride into
  the GBM - trees are robust, it is a real comorbidity, and the risk model wants more signal.
- **Binary flags, NaN preserved in `make_patient_features`** (no `fillna`, unlike `insulin_status`): the
  GBM routes native NaN; the dense flow/MT nets 0-fill them in `transform_patient_features` (they are NOT
  in `CONTINUOUS_PATIENT_FEATURES`, so never standardized). Added `PMH_OSA` / `PMH_dyslipidemia` to
  `required_columns()` so they canonicalize + assert (a fixed-width conditioning vector cannot gracefully
  drop a column, unlike the GBM's optional frame features).

**Consequence - full retrain required.** Conditioning width 6 -> 8 invalidates every existing
flow/twin/multi-task checkpoint and the 6-feature GBM. Re-run `train_twin_pipeline.py` ->
`evaluate_twin.py` (plus the multi-task / base sweeps if the head-to-head is being refreshed).

**Verified (fake cohort, HistGB fallback):** 8-dim conditioning propagates to all three model
constructors (no hardcoded width); binary flags are un-standardized and finite after transform; the GBM
trains end-to-end with the 15-feature matrix captured in `config.json`; forward passes pass for base
flow / twin / multi-task.

**Also:** `debug_attrition.py` now groups `osa`/`dyslipidemia` under *conditioning* features (and
de-dups `InsulinStatus`, which was previously double-listed as a candidate).

**NOT acted on this session** (still pending): creatinine demotion, CPT `43645` -> `rnygb` mapping, and
adding `eGFRatEvent` to the GBM (awaiting the real-cohort eGFR missingness - it sits below the fold in
the 09:28:44 attrition report).

---

## 2026-07-06 (session 3) — Digital-twin build EXECUTED (all six items shipped + smoke-tested)

All six items from `NEXT_SESSION_BUILD_PROMPT.md` are implemented and smoke-tested on
`fake_data/fake_mbs_cohort.csv` with the `mbsaqip_flow/.venv` (py3.13, HistGB fallback — no local
xgboost). **Net UX = three commands:** `debug_attrition.py` → `train_twin_pipeline.py` →
`evaluate_twin.py`. The pristine Cosmos core (`train_flow_matching.py`, `tune_flow_matching_optuna.py`)
is byte-for-byte unmodified (verified via `git status`); the only edit to an existing file is
`gbm_mace_baseline.py` (Item 2, +46/−6).

**Files:**
- `gbm_mace_baseline.py` (edited) — `assemble_features` now hstacks `PMH_DM2` + `PMH_hypertension`
  pulled from `dataset.frame` via a new `frame_feature()` helper (uses `fm.find_compatible_column`,
  tolerant of Cosmos casing/join suffixes; NaNs preserved). GBM-only — `fm.PATIENT_FEATURES` untouched.
  Feature matrix is now 9 cols; both appear in `feature_importances.csv` (confirmed).
- `debug_attrition.py` (new, read-only) — `--csv` / `--db`. Emits a terse `.txt` + 1-line stdout
  headline. Sections: (a) per-feature missingness for conditioning + candidate features, (b) LOO on
  `REQUIRED_PATIENT_FEATURES`, (c) CPT 43645 + other unrecognized codes, (d) stage decomposition.
  Replays fm's filter predicates by import to *count* each drop (loader returns no counts); stages
  reconcile exactly (raw − Σdrops = final N). `creatinine` NOT demoted — reported only.
- `train_flow_matching_twin.py` (new) — fork of the multi-task trainer: cls head + 2 MACE dims
  DROPPED; generates only the 15 continuous dims (`CONT_DIMS`); binary event added as a conditioning
  input via `nn.Embedding(2, event_emb_dim)` into the shared encoder (mirrors surgery), teacher-forced
  with the TRUE label. Reuses mt's preprocessing/split/loss by import. Module docstring documents the
  chain-rule factorization, event-not-score, leak-free teacher-forcing, and GBM→Bernoulli→flow.
- `tune_flow_matching_twin_optuna.py` (new) — twin sweep; objective = val flow loss only (no
  cls/fixed-weight juggling). Separate dir `runs/python_flow_matching_twin_optuna/`. Save-as-you-go
  per-trial logs + resumable SQLite (`load_if_exists=True` + `--study-dir` to resume).
- `train_twin_pipeline.py` (new, command 2) — fits the GBM on the shared train split (saves a
  config-deterministic run dir + a best-effort joblib pickle) → asserts the shared-split invariant →
  launches the twin sweep on the SAME split. Manifest written before AND after the sweep.
- `evaluate_twin.py` (new, command 3, monolithic) — GBM (histogram/calibration/discrimination +
  bootstrap CIs + per-component + DeLong), Flow (factual + surgery-counterfactual BMI/HbA1c plots via
  ev's model-agnostic machinery), Simulator (Modes A/B/C). 23 tagged artifacts (`gbm_`/`flow_`/`sim_`).

**Design decisions made during the build (things the prompt left open):**
- **Event conditioning = embedding, not raw scalar** (mirrors surgery exactly; verified live —
  clamping event 0→1 shifts trajectories ~1.7 BMI/HbA1c units).
- **Attrition stages include the two intermediate fm drops** (bad-event-interval, bad-glp1) between
  the prompt's [CPT] and [missing-required] stages so counts reconcile to final N exactly.
- **GBM persistence keeps the repo's retrain-from-config convention** (leak-free, portable) as the
  source of truth; the joblib pickle is an added convenience. The evaluator refits the GBM
  deterministically on the train split and calibrates on val (calibration is a *simulation*
  requirement — the event marginal is only correct if calibrated).
- **Flow plots reuse ev machinery model-agnostically**: the twin's own event-aware sampler produces
  15-dim samples scattered into a full 17-dim array (MACE dims = 0), then handed to
  `ev.plot_timecourse_factual_counterfactual` / `ev.timepoint_metric_table` (which take sample
  arrays, not a model) — "keeps existing machinery" without an event-blind adapter.
- **DeLong** validated to match sklearn AUC exactly; primary comparison is GBM-raw vs GBM-calibrated
  (both free), with `--compare-predictions` for GBM-vs-other (aligned on subject_id).
- On the 52-row fake cohort, val=7 (<10) ⇒ isotonic calibration and DeLong (n_pos=1) gracefully
  degrade to skip/NaN; every stat is guarded. Real numbers appear on the real cohort (val ~4.3k,
  test 571 events). **This build did NOT act on any attrition finding** (creatinine stays REQUIRED).

---

## 2026-07 (session 2) — Reorientation to a modular digital twin: build plan, feature & validation decisions

Committing the modular hybrid (calibrated GBM risk + event-conditioned flow) from an idea to the **active build plan**. Target UX: the research group runs **three commands** — (1) an attrition/missingness debug script, (2) a train script that fits the GBM then launches the flow Optuna sweep, (3) one monolithic evaluate script that emits every plot/table/output. The pristine Cosmos core (`train_flow_matching.py`, `tune_flow_matching_optuna.py`) stays untouched; all new work is new scripts or edits to the already-forked multi-task scripts. **All six items are implemented in one iteration — the build does NOT pause on the debug findings.** The debug output is informational: it decides whether we later *act* on optional data changes (e.g. demoting `creatinine`, adding more features), not whether the scripts get built.

### Feature decision (this iteration) + concerns for the next one
- **Doing now:** add `PMH_DM2` and `PMH_hypertension` to the **GBM only**, via its own `assemble_features` pulling from `dataset.frame` — NOT by editing the shared `PATIENT_FEATURES` list (that would push them into the flow + multi-task too). Then re-check discrimination, calibration, and permutation importances.
- **Concern (documented so we don't misread the result):** DM2 and hypertension are two of the *weaker* Tier-1 features — high-prevalence, individually less discriminating, and DM2 partly overlaps the `hba1c_at_surgery` + `insulin_status` already in the model. If this pass shows little AUROC movement, that is most likely *these two features*, NOT a refutation of the "features are the ceiling, not the model" finding.
- **Recommended next features** (cheap, all numeric/binary, NaN-native for trees), rough priority: `PMH_MI`, `PMH_stroke`, `PMH_AFib` (strongest MACE predictors), `eGFRatEvent` (renal — drives both MACE and the nephropathy component), `PMH_dyslipidemia`; then `SGLT2Status`, `BiguanideStatus`, `PMH_VTE`, `PMH_OSA`; then social-determinant fields (`Svi*` numeric scores, `RUCA`, `CoverageClass`, race) which also enable fairness reporting but need categorical encoding. Source-column names are in `table structure.txt`. Do NOT add filter-zeroed columns (`PMH_retinopathy`, `PMH_dialysis_transplant`, `PMH_PriorMBS`, `PriorGLP1` — all constant by the SQL WHERE) or any `PostOp*`/GLP1 field (post-baseline → leakage).

### Train/simulate correction — GBM and flow train INDEPENDENTLY
An earlier idea ("hand the trained GBM to the flow's Optuna so every trial conditions on it") was **wrong and is retracted.** The flow is trained by **teacher-forcing the TRUE observed event label** as conditioning — legitimate joint density estimation, no leakage, no out-of-fold. So **no Optuna trial needs the GBM**; GBM and flow are trained/tuned independently. The GBM enters only at **generation/simulation**: draw `event ~ Bernoulli(p_GBM(x))`, then `traj ~ flow(x, event)`. The train orchestration (script 2) sequences GBM → flow sweep purely for a **shared train/val/test split + one-command convenience**, and to have a leak-free (train-only) GBM available to draw held-out events at eval — NOT because the flow consumes it during training.

### Validation of the simulator — answers "should GBM-drawn trajectories match true-event trajectories?"
**No — not per patient**, and the eval must not assume so. True-event conditioning is an *oracle* (uses the outcome you would not have at prediction time); GBM-drawn conditioning marginalizes over it. They diverge exactly for patients whose realized outcome differs from their typical risk. What must match is the **ensemble**, and only if the GBM is **calibrated** — which is why calibration is a *simulation* requirement, not merely a risk-reading nicety. Three eval modes, each answering a different question:
- **Mode A — flow intrinsic (true event, per-patient):** condition on `x` + true `e`; compare to the observed trajectory (MAD, interval calibration). "Is the trajectory model good *given* we know the event?" Independent of the GBM.
- **Mode B — deployable point prediction (risk-weighted expectation, per-patient):** `ŷ(x) = p·μ(x,1) + (1−p)·μ(x,0)`, with `μ` = flow mean, `p` = GBM. The real per-patient prediction (no oracle). Expect it *worse* than Mode A; the A→B gap measures how much the event actually couples with the trajectory.
- **Mode C — full twin simulation (Bernoulli draw, distributional):** draw `e ~ Bernoulli(p)`, sample the trajectory; validate at the cohort level — (1) event marginal ≈ observed prevalence (13.3%) + reliability on the diagonal; (2) per-timepoint simulated BMI/HbA1c marginals ≈ observed (KS/quantiles); (3) event-stratified trajectory contrast in sim ≈ data; (4) counterfactual coherence (flip surgery → GBM risk AND flow trajectory move consistently).

### Composite endpoint (clarified)
The label is ONE aggregate binary `mace_ever = MACE OR Nephropathy OR Retinopathy` — there are **no per-component outputs**, and **neuropathy is not in the data at all** (`MBSCohort` has no such column). It mixes macrovascular (MACE) with two of three microvascular complications. Keeping the aggregate for now, but the evaluator must **report per-component discrimination** (MACE-only vs nephropathy vs retinopathy) so the composite AUROC isn't oversold. Per-complication risk heads would be a genuinely different (multi-label) model — not currently planned.

### Attrition/missingness debug script (script 1)
Read-only. On CSV locally it sees Python-side attrition only; on the VM it also gets the true denominator. Its numbers are **informational** — all six items are built regardless; the debug output only informs whether we later *act* on optional data changes. **Creatinine stays REQUIRED this iteration (NOT demoted).** Demoting it (from `REQUIRED_PATIENT_FEATURES` to optional/mean-imputed) is the **leading candidate change pending further information** — the leave-one-out below quantifies exactly how many rows it would recover so we can decide later; likely it dominates the ~3,311 dropped rows because age/sex/BMI are near-universal in the SQL-filtered cohort. Must produce (terse `.txt` + a one-line stdout headline): (a) per-feature missingness (count + %); (b) a **leave-one-out on `REQUIRED_PATIENT_FEATURES`** — rows recovered if each required field alone were demoted (expect `creatinine` to dominate, since age/sex/BMI are near-universal in the SQL-filtered cohort); (c) the **CPT `43645` count** surfaced directly (bypass variant currently dropped as unrecognized) plus any other unrecognized CPTs; (d) an attrition **decomposition framed as stages** — SQL-filter losses [VM-only, via a `SELECT COUNT(*)` without the WHERE, and ideally per-clause] → CPT-unrecognized drops → missing-required-conditioning drops → final N.

### Evaluate script (script 5)
One **monolithic** script (user preference — portability to the cluster; it may import existing eval helpers since the repo moves together), with explicit **model → results tagging** in filenames/titles: the GBM owns the MACE probability histogram + calibration; the flow owns the factual/counterfactual BMI/HbA1c-over-time plots; the simulator owns the joint checks (Modes A/B/C above). Keep the existing factual/counterfactual trajectory plots + MACE yes/no probability histograms; ADD per-component discrimination, calibration curves, and bootstrap CIs (DeLong for AUROC deltas).

### Env
Smoke-test with `/Users/tien/Work/BranniganLab/mbsaqip_flow/.venv/bin/python` (py3.13; torch 2.12, sklearn 1.8, optuna 4.8; no local xgboost) on `fake_data/fake_mbs_cohort.csv`. Real cohort runs + xgboost + DB `COUNT(*)` happen on the Cosmos VM. (Note: the `../mbsaqip/` path referenced in discussion does not exist — `mbsaqip_flow` is the real venv dir.)

---

## 2026‑07 — First real‑data head‑to‑head (comparison run `run_20260629_160246`)

Head‑to‑head composite‑MACE, shared test split (n = 4,306, 571 events, prevalence 0.133):

| model | variant | AUROC | AUPRC | Brier |
|---|---|---|---|---|
| baseline | predict prevalence | 0.500 | 0.133 | 0.115 |
| GBM | unweighted | 0.721 | 0.320 | **0.104** |
| GBM | unweighted + calibrated | 0.720 | 0.300 | 0.105 |
| GBM | balanced | 0.720 | 0.323 | 0.196 |
| GBM | balanced + calibrated | 0.717 | 0.297 | 0.105 |
| multitask | raw | **0.728** | **0.329** | 0.202 |
| multitask | calibrated | 0.725 | 0.305 | 0.104 |

AUPRC baseline (prevalence) = 0.133. Backend = xgboost (works on the Cosmos VM).
Continuous outcomes (multitask flow head): BMI MAD 1.8–3.7, HbA1c MAD 0.33–0.46,
degrading with horizon (expected).

### Findings

1. **GBM and the multi‑task NN are statistically tied on MACE.** The NN's 0.007
   AUROC edge is ~0.5 SE (SE(AUROC) ≈ 0.013 at 571 events) — noise. Two very
   different model families converging to ~0.72 means the **feature set, not the
   model, is the ceiling.**
2. **Class weighting / balancing gives no discrimination gain and wrecks
   calibration.** Balanced GBM (Brier 0.196) and raw multitask (0.202) are *worse
   than predicting the base rate* (0.115) as probabilities; recalibration recovers
   them (~0.104).
3. **Unweighted GBM is calibrated out of the box** (Brier 0.104, no post‑hoc step).
4. **Clinically moderate.** At 90% specificity, sensitivity ~34% (PPV 36%); at 0.5,
   sensitivity 66% (PPV 25%). Triage‑grade enrichment (~2.7× base rate), not rule‑out.

### Data‑quality flags (from run logs)

- **3,311 rows dropped** for missing required core conditioning fields (~10%). Trees
  don't need complete conditioning — dropped only for split‑comparability with the
  flow model. Check whether the missingness is informative (sicker patients?).
- **CPT `43645` excluded** as "unrecognized" — it is a gastric‑bypass variant. Decide
  whether to map it to `rnygb` (recovers patients) rather than drop it silently.

### Decisions

1. **Ship the unweighted GBM as the MACE risk model of record** (tied discrimination,
   best calibration, simplest). Keep the multi‑task model for **joint trajectories**,
   not to beat the GBM on risk.
2. **Do not use class weighting/balancing** for the deliverable. For a specific
   operating point, threshold the calibrated unweighted model.
3. **Highest‑leverage next step: expand the feature set** (saturated at 6 covariates).
   Add the unused comorbidity/labs already in the source table: `PMH_MI`,
   `PMH_stroke`, `PMH_AFib`, `PMH_hypertension`, `PMH_dyslipidemia`, `PMH_DM2`,
   `eGFRatEvent`, SGLT2/biguanide status, SVI social‑determinant fields.
4. **Investigate data attrition** (3,311 dropped; CPT `43645`). A GBM‑only run on the
   fuller cohort (relaxed conditioning, native NaN handling) is worth trying.
5. **Report bootstrap CIs** (DeLong for AUROC deltas) and **per‑component
   discrimination** (MACE‑only vs nephropathy vs retinopathy) so the composite AUROC
   isn't oversold.

---

## Decision — one overarching model (risk + trajectories)?

**A single monolithic model adds no value for _risk_.** The multi‑task NN already
tested exactly that design (shared encoder → risk head + trajectory head) and **tied
a simpler GBM** on MACE while needing post‑hoc recalibration. GBMs win on tabular
risk; folding risk into the joint net forfeits that advantage plus native
missingness handling and free calibration. Risk (binary, tabular → trees) and
trajectories (continuous, temporal, uncertainty → flow) want different inductive
biases; one architecture compromises both, and couples debugging/iteration/calibration.

**The only genuine reason to unify is a _coherent joint distribution_** — sampling
risk and trajectories that co‑vary correctly per patient (digital‑twin / scenario
simulation). If that is the goal, the right design is **modular, not monolithic**: a
dedicated calibrated **GBM for risk** + a **flow model for trajectories conditioned
on the risk score**. That buys coherence without sacrificing either task.

**Decision rule (by downstream use):**

- Need **independent outputs** (a risk number *and* trajectory predictions): keep them
  **separate** — GBM + flow. No value in unification.
- Need **coherent joint samples** (correlated risk + trajectory per patient): use the
  **modular hybrid** (GBM risk → conditions the flow), not a single monolith.

The multi‑task NN stays useful as a trajectory model and as a way to obtain a coherent
joint if wanted — but it is **not** the risk model of record.

---

## Digital‑twin / scenario simulation — modular hybrid (how to build it)

Goal: sample a coherent joint `(MACE event, BMI/HbA1c trajectory)` per patient so the
marginals **and** their correlation are correct — for scenario simulation / patient‑
specific what‑ifs.

### Factorization — why the marginals are correct by construction
Model the joint by the chain rule:

```
p(event, trajectory | x) = p(event | x) · p(trajectory | event, x)
                           └── GBM ──┘     └──────── flow ────────┘
```

- GBM owns the **event marginal** `p(event | x)`.
- Flow owns **trajectory‑given‑event**.
- Event marginal is *exactly* the GBM. Trajectory marginal falls out as the mixture
  `p(traj|x) = p·flow(x, e=1) + (1−p)·flow(x, e=0)` — correct if both factors are.

### The one requirement: calibration
"Correct event marginal" ⟺ the GBM is **calibrated**. Generation emits a draw
`event ~ Bernoulli(p_GBM(x))`; those draws reproduce the true prevalence and the true
risk‑stratified rates *only if calibrated*. The unweighted GBM already is (Brier 0.104);
the balanced/weighted model would over‑produce events (Brier ~0.20) → wrong marginal.
This is why calibration matters for **simulation**, not just for reading a risk number —
and why we ship the unweighted GBM.

### Sampling a coherent twin
```
p     = GBM(x)              # calibrated risk  -> correct event marginal
event ~ Bernoulli(p)        # concrete draw
traj  ~ flow(x, event)      # trajectory conditioned on the drawn event
```

### Condition on the EVENT, not the risk score (subtle, important)
Conditioning the flow on the GBM score `p(x)` adds nothing for coupling: `p(x)` is a
deterministic function of `x`, so `p(traj | x, p(x)) = p(traj | x)` → risk ⟂ trajectory
given `x` (correct marginals but **no residual dependence**). To capture residual
coupling (same `x`, but the patient who regains weight is likelier to have an event),
condition the flow on the realized **binary event** `e`, so `p(traj | e, x) ≠ p(traj | x)`.
For a twin you want this.

### What to build (modest change)
- **Flow:** add the binary event as a conditioning input (concat or adaLN, like surgery
  type) and **drop the generated MACE dims**. ≈ current multi‑task flow head, but
  conditioned on the event flag instead of predicting it.
- **Training is legitimate:** conditioning on the *true* event label at train time is
  joint density estimation, not prediction (both are sampled at generation) → **no
  leakage, no out‑of‑fold needed.** (OOF cross‑fitting is only needed if you instead
  condition on the GBM's *predicted* score.) Teacher‑force with the true label at train;
  sample the event from the GBM at generation.
- **Scenario / counterfactual modes come free:** flip surgery → GBM risk and flow
  trajectory shift coherently; or clamp `event=1` to ask "what do the trajectories of
  patients who go on to have an event look like?"

### Verify the simulator (marginal + joint calibration)
Simulate the whole cohort and check:
1. **Event marginal:** simulated prevalence ≈ observed (13.3%); reliability curve diagonal.
2. **Trajectory marginals:** per‑timepoint simulated BMI/HbA1c distributions ≈ observed (KS / quantiles).
3. **Joint coupling:** event‑stratified trajectory differences in the sim ≈ data.
4. **Counterfactual coherence:** flipping surgery moves risk + trajectories consistently.

Suggested artifacts if implemented: an event‑conditioned flow (variant of
`train_flow_matching_multitask.py` — event flag in the conditioning, MACE dims removed)
+ a `simulate_twins.py` doing GBM → Bernoulli → flow sampling and the four checks above.

---

## Clinical benchmarking & publishability

*Based on articles retrieved from PubMed.*

**Is AUROC ~0.72 "good"?** Moderate by convention (0.70–0.80 = "acceptable"), but
**competitive for a routine‑variable MACE model**:

- Classic risk‑factor MACE model (ASPREE, ~18.5k healthy elderly, ~9 predictors):
  **AUC 0.68 internal, 0.64 external**, and underestimated risk externally —
  Neumann et al., GeroScience 2021, https://doi.org/10.1007/s11357-021-00486-z
- Biomarker/metabolomics‑enhanced at biobank scale (UK Biobank, 229k):
  **C‑index ~0.75–0.82**, beating traditional ASCVD / Age+Sex baselines —
  Zhang et al., Cardiovasc Diabetol 2025, https://doi.org/10.1186/s12933-025-02711-x
- Bariatric + MACE is currently studied by *applying generic calculators* (Taiwan MACE,
  China‑PAR), not bespoke models — Pan et al., Int J Surg 2024,
  https://doi.org/10.1097/JS9.0000000000001631 → **real gap for a dedicated tool.**

Landscape: classic routine‑variable models ~0.64–0.72; biomarker‑enhanced ~0.75–0.82.
Our 0.72 from 6 pre‑op variables **tops the routine tier**, and the gap to ~0.78 is
**features** (labs/comorbidities) — consistent with both the ML tie (GBM ≈ NN) and the
biomarker literature.

**Would it fly?** Yes, as risk **stratification** in the bariatric population, if reported
to TRIPOD standard. **Not** as a standalone high‑accuracy decision tool (operating points
too weak: 34% sensitivity at 90% specificity). The hook is the **population gap + the
digital‑twin / counterfactual methodology**, not the AUROC.

### Validation & reporting checklist (before any submission)
- [ ] **External or temporal validation** — a single internal split is the weakest tier;
      expect a 0.03–0.05 AUROC drop, so plan for ~0.67–0.69.
- [ ] **Calibration** on the validation set (reliability curve + Brier); recheck externally
      (published models often lose calibration out‑of‑sample).
- [ ] **Decision‑curve / net‑benefit analysis** — clinical utility beyond discrimination.
- [ ] **Justify the composite endpoint** and report **per‑component discrimination**
      (MACE‑only vs nephropathy vs retinopathy) so the composite AUROC isn't oversold.
- [ ] **AUPRC** (vs prevalence baseline) + **bootstrap CIs** (≈ ±0.025 on AUROC at 571
      events); **DeLong** test for model‑vs‑model AUROC deltas.
- [ ] **Sample‑size / EPV** justification (Riley criteria) and **missing‑data handling**
      (the 3,311 dropped rows — is the missingness informative?).
- [ ] Report per **TRIPOD‑AI** (prediction‑model reporting guideline).
