# Next-session build prompt — MBSAQIP digital-twin implementation

Paste the block below into a fresh session (or point the session at this file). It implements all
six items in one pass and smoke-tests them; it does NOT pause for discussion. Rationale for every
decision lives in `MACE_MODELING_DECISIONS.md` (top entry).

---

I'm continuing the MBSAQIP bariatric-outcomes project in /Users/tien/Work/BranniganLab/R_project.
We're reorienting it into a modular "digital twin": a calibrated GBM for composite-MACE risk +
a flow-matching model for BMI/HbA1c trajectories conditioned on the event, sampled jointly as
GBM -> Bernoulli(event) -> flow. Implement ALL of the items below in one pass and smoke-test each;
do NOT pause for discussion between items.

READ FIRST (do not skip — the rationale lives here, not in this prompt):
- MACE_MODELING_DECISIONS.md — read the TOP entry, "2026-07 (session 2) — Reorientation to a
  modular digital twin", in full. It has the feature decisions, the train/simulate correction,
  the 3-mode validation protocol, and the composite-endpoint clarification.
- The two auto-memory files listed in the ~/.claude memory MEMORY.md (mbsaqip-mace-modeling,
  mbsaqip-eval-conventions).
- Code: train_flow_matching.py (data layer + feature defs, lines ~42-118), gbm_mace_baseline.py
  (assemble_features ~line 109), train_flow_matching_multitask.py (the fork the twin flow is based
  on), evaluate_flow_matching.py / evaluate_multitask.py / evaluate_gbm_mace_baseline.py /
  compare_mace_models.py, tune_flow_matching_multitask_optuna.py.

HARD CONSTRAINTS:
- Do NOT modify the pristine Cosmos core: train_flow_matching.py and tune_flow_matching_optuna.py.
  Reuse them by import. All new work = new scripts or edits to the already-forked multi-task scripts.
- Feature additions go to the GBM ONLY, via its own assemble_features pulling from dataset.frame.
  Do NOT edit the shared PATIENT_FEATURES list.
- The twin flow trains by teacher-forcing the TRUE observed event label as conditioning. It does
  NOT use the GBM at train/tune time. The GBM is used only at simulation/eval.
- Do NOT demote creatinine from REQUIRED_PATIENT_FEATURES. Leave the required set as-is. (The debug
  script only *reports* how many rows demoting it would recover; acting on that is a later decision.)
- Env for smoke testing: /Users/tien/Work/BranniganLab/mbsaqip_flow/.venv/bin/python
  (py3.13, torch 2.12, sklearn 1.8, optuna 4.8; xgboost/lightgbm NOT available locally -> sklearn
  HistGradientBoosting fallback). Smoke data: fake_data/fake_mbs_cohort.csv. runs/ and *.pt are
  gitignored. Real cohort + xgboost + DB access are on the Cosmos VM, not local.

=== ITEM 1 — attrition/missingness debug script ===
New read-only script (e.g. debug_attrition.py). Reuse fm's loaders/filters by import; don't
duplicate the SQL/filter logic.
  --csv <path>: local mode, Python-side attrition only.
  --db:         VM mode; additionally run `SELECT COUNT(*) FROM MBSCohort` WITHOUT the WHERE clause
                (and, if feasible, per-WHERE-clause counts) using CONNECTION_STRING from
                train_flow_matching.py, to get the true pre-filter denominator.
Output = terse .txt file + a one-line stdout headline pointing at it. Must contain:
  a. Per-feature missingness (count + %) for every conditioning feature AND every candidate feature
     (PMH_*, eGFRatEvent, InsulinStatus/BiguanideStatus/SGLT2Status).
  b. Leave-one-out on REQUIRED_PATIENT_FEATURES: for each required field, how many rows would be
     recovered if it alone were demoted to optional. (Expect creatinine to dominate. Report only;
     do NOT change the required set.)
  c. CPT 43645 count surfaced directly (currently dropped as unrecognized — a gastric-bypass
     variant), plus any other unrecognized CPT codes with counts.
  d. Attrition decomposition framed as STAGES with count + % at each step:
     [SQL-filter losses — VM only] -> [CPT-unrecognized drops] -> [missing-required-conditioning
     drops] -> final N.

=== ITEM 2 — add PMH_DM2 + PMH_hypertension to the GBM ===
In gbm_mace_baseline.py assemble_features, pull PMH_DM2 and PMH_hypertension from dataset.frame
(VERIFY the exact canonicalized column names first — canonicalize_columns / CSV casing may differ)
and hstack them as GBM features. GBM only; do NOT touch PATIENT_FEATURES. Keep NaNs (trees handle
them). Confirm both appear in the feature_importances output and that calibration (Brier) holds.

=== ITEM 3 — reorient the flow into the twin flow (new script, e.g. train_flow_matching_twin.py) ===
Based on train_flow_matching_multitask.py, but:
  - DROP the MACE classification head entirely (the GBM owns risk).
  - DROP the 2 MACE target dims — the flow generates ONLY the 15 continuous BMI/HbA1c dims
    (reuse CONT_DIMS).
  - ADD the binary event (mace_ever, i.e. dataset.x[:, MACE_DIM]) as a CONDITIONING input, handled
    like surgery type (concat or adaLN). Teacher-force the TRUE label at train.
  - Module docstring MUST document: the chain-rule factorization
    p(event,traj|x) = p(event|x)[GBM] * p(traj|event,x)[flow]; why we condition on the binary EVENT
    not the score; why teacher-forcing the true label is leak-free (both are sampled at generation);
    and the GBM->Bernoulli->flow sampling path — i.e. why this makes it a digital twin.

=== ITEM 4 — Optuna sweep for the twin flow (new script) ===
Like tune_flow_matching_multitask_optuna.py but simpler: no cls head, so the objective is just the
val flow-matching loss (no fixed-weight juggling). Separate runs/ output dir. Keep the
save-as-you-go / resumable behavior.

=== ITEM 5 — single MONOLITHIC evaluate script ===
One script producing ALL outputs (one file for cluster portability; it may import existing eval
helpers since the repo moves together). Explicit model->results tagging in filenames/titles:
  - GBM:        MACE probability histogram + calibration curve + discrimination.
  - Flow:       factual AND counterfactual BMI/HbA1c-over-time plots (keep existing machinery).
  - Simulator:  the joint checks (Modes A/B/C from the doc) — event marginal vs prevalence,
                trajectory marginals (KS/quantile), event-stratified contrast, surgery
                counterfactual coherence.
ADD for publishability: per-component discrimination (MACE-only vs nephropathy vs retinopathy),
calibration curves, bootstrap CIs (DeLong for AUROC deltas).

=== ITEM 6 — train orchestration script ===
One script: fit the GBM on the train split -> save it -> launch the twin-flow Optuna sweep. Both
MUST use the same split_strategy/seed/fracs (shared patient-for-patient split). The GBM is trained
here so a leak-free (train-only) GBM is available to draw held-out events at eval; the flow sweep
does NOT consume it. Save-as-you-go.

NET UX = three commands: (1) debug_attrition, (2) train (GBM -> flow sweep), (3) evaluate.

Smoke-test EVERY new script on fake_data/fake_mbs_cohort.csv with the venv above before declaring it
done. When finished, report: what you built, smoke-test results (pass/fail + key output lines), and
any point where you had to make a design decision.
