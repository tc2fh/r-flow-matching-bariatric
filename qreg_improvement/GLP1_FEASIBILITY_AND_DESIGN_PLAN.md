# Incretin Therapy Feasibility and Design Plan

Date: 2026-07-20

Status: Awaiting a full Cosmos diagnostic run. Do not change the production cohort or causal claims until the aggregate outputs described below have been reviewed.

## Bottom line

The project should not create a GLP-1 arm by globally deleting `PriorGLP1 = 0` and merging every row into one analysis. Prior incretin exposure, postoperative initiation, and nonsurgical incretin initiation answer different clinical questions and require different time-zero definitions.

The preferred architecture is:

1. Expand the surgical prediction cohort to include otherwise eligible prior incretin users.
2. Preserve a GLP-naive surgical sensitivity cohort and the sleeve-versus-RYGB target trial.
3. Make postoperative incretin rescue the preferred additional causal question if longitudinal treatment and confounder histories are available.
4. Consider a perioperative continue-or-restart study among prior incretin users if separate preoperative and postoperative medication episodes can be identified.
5. Treat a direct surgery-versus-incretin comparison as a secondary benchmark because similar direct comparisons now exist.
6. Add per-patient conditional flow matching only as a gated secondary outcome model. Treatment clamping in a predictive flow is not causal identification.

Use the broader term `incretin-based therapy` when tirzepatide is included. Tirzepatide should be recorded separately as a dual GIP/GLP-1 agonist rather than silently pooled with GLP-1 receptor agonists.

## Diagnostic script

The repository-root script is:

`debug_glp1_feasibility.py`

It is read-only. It queries the raw `MBSCohort` and `GLP1Cohort` tables without applying the historical model `WHERE` clauses, performs no model fitting, and writes aggregate outputs only. Grouped counts below 11 are suppressed by default.

### First run the synthetic plumbing check on the Cosmos VM

From the repository root:

```bash
python debug_glp1_feasibility.py \
  --mbs-csv fake_data/fake_mbs_cohort.csv \
  --glp1-csv fake_data/fake_glp1_cohort.csv \
  --output-root runs/debug_glp1_feasibility_smoke
```

Confirm that the command finishes and writes a timestamped directory containing `manifest.json`, `glp1_feasibility_report.txt`, and the CSV artifacts.

### Then run the full Cosmos diagnostic

```bash
python debug_glp1_feasibility.py
```

The default output location is:

`runs/debug_glp1_feasibility/<timestamp>/`

The default connection string is imported from `train_flow_matching.py`. It can instead be supplied through the `COSMOS_CONNECTION_STRING` environment variable. If the tables are not in `dbo`, use `--schema`. If table names differ, use `--mbs-table` and `--glp1-table`.

Do not use `--max-rows` for final feasibility decisions. That option exists only for query-plumbing checks and produces a non-random truncated sample.

Do not disable small-cell suppression unless local Cosmos policy explicitly permits it. The script never writes `PatKey` or another patient identifier.

The built-in small-cell rule is a diagnostic safeguard, not a substitute for formal Cosmos disclosure review. Keep the output on the VM unless the complete bundle has been reviewed under the applicable data-export policy.

## What the diagnostic produces

### Schema and source-table discovery

- `schema_inventory.csv`: every field in the two analytic cohort tables, its SQL type, and inferred role.
- `database_medication_schema_candidates.csv`: medication-related fields discovered across the database schema, including possible RxNorm, RxCUI, NDC, ingredient, product, strength, route, frequency, order, fill, administration, days-supply, refill, and indication fields.
- `concept_inventory.csv`: canonical study concepts and the exact cohort-table field matched to each concept.
- `column_profiles.csv`: missingness, placeholder frequency, distinct-value counts, numeric parse rates, and date ranges.

The database-wide schema artifact is important. If a medication order, administration, or dispensing table exists outside `GLP1Cohort`, it may contain the episode detail needed for causal analyses even when the current wide cohort table does not.

### Medication tags and doses

- `medication_tag_values.csv`: all exact raw medication-name, route, and unit tags, plus the most frequent values from higher-cardinality discovered coding fields, with a proposed ingredient and therapy-class mapping.
- `medication_field_coverage_by_agent.csv`: conditional completeness for names, routes, doses, units, dates, duration, and timing by agent.
- `dose_profiles.csv`: dose distributions kept separate by ingredient, route, unit, and source dose field.
- `dose_value_frequencies.csv`: frequent recorded dose values without converting across units.
- `episode_quality.csv`: end-before-start errors, negative duration, disagreement between recorded duration and dates, recent-dose-above-max inconsistencies, and dose-without-unit counts.
- `exposure_reconciliation.csv`: disagreements among prior/postoperative flags, dates, and surgery-relative intervals, plus a date-derived active-at-surgery diagnostic.

Medication normalization is intentionally conservative. Review every row where `needs_manual_review = True`. Brand mappings are suggestions, not a replacement for an Epic/RxNorm data dictionary.

The minimum desired medication representation is:

- Generic ingredient and product or brand, if available
- RxNorm concept and RxCUI, and NDC when available
- Route and formulation, including oral versus injectable semaglutide
- Strength, administered or ordered dose, and dose unit
- Frequency, quantity, days supply, and refill information
- Order, administration, dispensing, and discontinuation timestamps
- Indication or an analyzable proxy for diabetes versus obesity treatment
- Episode start, end, restart, and switching dates
- A method to distinguish order existence from actual administration or continued coverage

Do not combine doses across units. Do not assume `MostRecentDose` is a maintenance dose or that `MaxGLP1Dose` describes exposure at a particular outcome horizon.

### Cohort and design support

- `cohort_funnels.csv`: intersection-aware structural, T2D, strict incident-outcome, and selected legacy restriction funnels.
- `design_arm_counts.csv`: candidate counts for expanded surgical prognosis, baseline surgery-versus-incretin comparison, postoperative rescue, perioperative prior-user strategies, and observed sequences.
- `postoperative_timing.csv`: surgery-relative incretin initiation bins.
- `rescue_trigger_counts.csv`: month-12 support for BMI at least 35, total weight loss below 15 percent, HbA1c at least 6.5, and a composite rescue trigger.
- `calendar_counts.csv`: arm counts by index year to expose nonoverlapping treatment eras.
- `baseline_overlap_summary.csv`: common baseline-variable distributions across candidate baseline treatment arms.
- `patient_arm_overlap.csv`: duplicate-patient and pairwise cross-arm overlap, including overlap between the surgical and medication tables.

The historical marginal exclusion count for `PriorGLP1 = 0` is not the number of patients that will be recovered. Exclusions overlap. Use the new funnel and arm tables to determine the actual intersection-aware gain.

### Outcomes and causal-data gates

- `outcome_support.csv`: BMI, weight, and HbA1c availability at each wide horizon, both overall and among patients with apparent follow-up opportunity.
- `event_support.csv`: raw support for known composite-event status at 12, 24, 36, and 60 months.
- `design_requirements.csv`: field-level requirements, matched columns, completeness, and blockers.
- `design_readiness.csv`: a schema-only readiness summary for each design.

`design_readiness.csv` is not a causal green light. It must be considered together with arm counts, calendar overlap, outcome support, effective sample size after weighting, and whether clinically important confounders were measured.

## Candidate designs and decision rules

### 1. Expanded surgical prognosis

Question: What BMI, HbA1c, and complication outcomes should be expected after sleeve or RYGB under observed real-world care, including incretin use?

This is the lowest-risk extension and the most likely to be supported immediately.

Proposed exposure categories:

- Never recorded before surgery
- Remote prior use
- Active at surgery, only if an episode spans the surgery date
- Early postoperative start or restart
- Later postoperative initiation
- Unknown timing

Requirements:

- Reliable `PriorGLP1` and `PostOpGLP1`
- Start date or surgery-relative interval among recorded postoperative users
- Sufficiently complete agent tags for meaningful subgroup analyses
- Landmark models must use only exposure known by the forecast origin

For preoperative forecasts, retain postoperative outcomes after later incretin initiation because the estimand is usual-care prognosis. For landmark forecasts, include only treatment history observed before the landmark.

Keep the GLP-naive surgical cohort as a locked sensitivity analysis. If prior-use timing cannot be separated into remote versus active use, label it as a coarse prior-exposure flag rather than active treatment.

### 2. Baseline surgery-versus-incretin comparison

Question: Among patients eligible for both strategies, what are outcomes after sleeve, RYGB, or initiation of a specified incretin treatment?

Time zero must be the operation date for surgery and the first qualifying medication initiation date for the medication arms. The two cohorts must be harmonized and stacked with an explicit arm variable. Do not reuse the historical `full_join` on baseline and outcome fields.

Minimum requirements:

- A defensible new-user washout period based on dated medication history, not only a binary prior-use flag
- Agent-specific treatment versions, with semaglutide and tirzepatide separated
- Common calendar eras and clinical eligibility
- Switching dates for subsequent surgery, medication discontinuation, and agent switching
- Exact or sufficiently granular censoring and follow-up
- Comparable baseline confounders, including utilization and access-to-care measures

Primary horizons should be 12, 24, and 36 months for modern therapies. Five-year comparisons should be restricted to treatment versions and calendar periods that genuinely have five-year opportunity.

If only `GLP1StartDate`, `GLP1EndDate`, `GLP1Duration`, and a single latest or maximum dose are available, an intention-to-treat association may be possible, but continuous-use and dose-response claims are not.

### 3. Postoperative incretin rescue

Preferred causal question: Among GLP-naive patients with inadequate response at a prespecified postoperative landmark, what is the effect of initiating incretin therapy during a grace period versus not initiating through a prespecified window?

Candidate protocol already represented in the new runner:

- Landmark: month 12
- Trigger: BMI at month 12 at least 35, with alternative trigger sensitivity analyses
- Initiation strategy: start during months 12 through 18
- Comparator: no initiation through month 24
- Initial outcome horizon: month 36

Do not run this as a causal analysis unless the diagnostic finds or a new extract supplies:

- Longitudinal incretin exposure and switching
- Time-varying BMI and HbA1c before treatment decisions
- Time-varying diabetes and weight-management medications
- Utilization or encounter history
- Monthly observability or exact censoring information
- Adequate counts in both strategies and adequate month-36 outcome support

The present `dynamic_glp1_analysis()` implementation should be replaced before use. A valid clone-censor-weight analysis must create a clone for every eligible patient under every strategy, artificially censor clones at deviation, estimate stabilized artificial-censoring and natural-censoring weights, diagnose balance and positivity, and provide appropriate clustered inference.

If monthly histories are unavailable but exact dated histories exist, consider sequential nested trials or longitudinal TMLE. If only month-12 and month-36 snapshots exist, report a landmark association and do not label it a dynamic causal effect.

### 4. Perioperative strategy among prior incretin users

Question: Among patients receiving incretin therapy before surgery, what are outcomes under prompt continuation or restart versus stopping or delaying treatment?

This is the design most directly capable of recovering the previously excluded prior users. It requires separate preoperative and postoperative episodes. A single `PriorGLP1` flag plus one GLP start date cannot reliably distinguish active-at-surgery use, perioperative interruption, restart, and a later new episode.

Potential strategies should be finalized only after inspecting empirical restart timing. Examples include restart within 90 days versus no restart through 180 days. The grace period must be chosen clinically before examining comparative effects.

### 5. Treatment sequences

Potential sequence categories include:

- Incretin only
- Incretin followed by surgery
- Surgery only
- Surgery followed by incretin rescue
- Preoperative incretin followed by surgery and postoperative restart

This is scientifically attractive but likely better treated as a separate analysis or paper. It requires multi-episode longitudinal data and careful handling of time-varying eligibility and switching.

## Per-patient conditional flow matching

A per-patient flow model can be added alongside the published-style HGB, target-specific CatBoost, and conservative ensemble after the tabular production models have been validated.

### Prognostic use

Condition on baseline variables, procedure, and treatment history known at the forecast origin. Generate a joint distribution of future BMI and HbA1c trajectories. Evaluate factual predictions using standardized CRPS, coverage, PIT calibration, energy score, variogram score, RMSE, and MAE on locked center-time test data.

### Point-treatment causal use

For an aligned baseline target trial, a conditional flow can model the joint outcome distribution under each arm. Causal identification still requires propensity, censoring, and observation adjustment, cross-fitting, overlap diagnostics, and a doubly robust population estimator. Per-patient treatment contrasts should remain exploratory because individual counterfactual effects cannot be directly validated.

### Dynamic-treatment causal use

For postoperative rescue, use a sequential conditional flow only as an outcome-regression component within g-computation or longitudinal TMLE, or use a month-12 landmark flow after clone-censor weighting. Do not obtain a claimed causal effect by directly changing the treatment input while ignoring treatment-confounder feedback.

Do not condition any trajectory model on a future complication event. Keep competing-risk event prediction separate from continuous metabolic trajectory generation.

Promotion criteria should remain prespecified. Flow matching should not be promoted unless it improves locked-test joint distributional accuracy and calibration without implausible tails or clinically important horizon regressions.

## Review sequence after the Cosmos run

1. Open `glp1_feasibility_report.txt` and `manifest.json`. Confirm both source tables were loaded in full and the run was not sampled.
2. Review `database_medication_schema_candidates.csv`. Identify any source medication tables with richer order, fill, administration, RxNorm, dose, or days-supply fields.
3. Review every medication tag requiring manual mapping and confirm treatment versions with the Cosmos data dictionary.
4. Review route and unit combinations before interpreting dose distributions.
5. Inspect episode inconsistencies and determine whether start/end fields describe one episode, the first episode, the latest episode, or an aggregate episode.
6. Compare actual arm counts, calendar overlap, follow-up maturity, and outcome support.
7. Review baseline distributions across candidate direct-comparison arms. Do not proceed if clinical or calendar nonpositivity is severe.
8. Review `design_requirements.csv` and classify each blocker as extractable, unavailable in Cosmos, or requiring a revised estimand.
9. Select the supported design before modifying `run_qreg_improvement.py`.
10. Freeze eligibility, treatment versions, grace periods, outcomes, estimands, and sensitivity analyses before inspecting comparative effects.

## Likely decisions by data scenario

### Only the current wide cohort fields are available

- Proceed with expanded surgical prognosis.
- Keep agent and timing subgroups descriptive.
- Treat a direct surgery-versus-incretin comparison as associational unless a credible new-user definition and censoring strategy can be constructed.
- Do not claim a dynamic postoperative rescue effect.
- A factual per-patient flow challenger remains possible, but not a dynamic causal flow.

### Dated medication episodes are available, but time-varying confounders are not

- A baseline active-comparator target trial may be possible.
- The perioperative prior-user design may be possible if both preoperative and postoperative episodes are identifiable.
- A month-12 rescue landmark association may be possible.
- A full longitudinal rescue strategy remains unsupported.

### Dated episodes, time-varying confounders, and observability are available

- Proceed to a fully specified postoperative rescue target trial.
- Implement genuine clone-censor weighting, the parametric g-formula, or longitudinal TMLE.
- Add the conditional flow as a secondary outcome model within that design.
- Report weight distributions, effective sample size, treatment and censoring balance, calendar and center overlap, and sensitivity to unmeasured confounding.

## Production-code changes after a design is selected

Do not remove `PriorGLP1 = 0` globally. Instead:

- Build separate named cohorts for expanded prognosis and each causal trial.
- Move eligibility criteria out of global SQL into protocol-specific cohort builders.
- Stack harmonized surgery and medication cohorts for direct comparisons rather than joining on outcomes.
- Preserve exact dates and episode identifiers in the analysis data contract.
- Replace follow-up-based baseline eligibility such as `ActiveEndInterval >= 700` with post-entry opportunity and censoring logic.
- Retain future postoperative treatment as part of usual care for prognosis.
- Rewrite the postoperative strategy estimator before enabling causal output.
- Add flow matching under a separate model protocol and acceptance gate.

## Planning checkpoint template

After the full diagnostic has run, record the following before resuming implementation:

- Full path to the timestamped diagnostic directory:
- Full-table row and unique-patient counts:
- Prior-user surgical patients recovered under structural eligibility:
- Postoperative initiation counts by timing window:
- Raw and normalized agent tags:
- Route and unit combinations:
- Dose completeness by agent:
- Evidence that episodes reflect orders, fills, administrations, or another source:
- New-user washout support:
- Center identifier availability:
- Calendar overlap across candidate arms:
- Outcome support at 12, 24, 36, and 60 months:
- Event counts and known-status support:
- Postoperative rescue longitudinal-data blockers:
- Perioperative episode-data blockers:
- Supported primary design:
- Supported secondary designs:
- Revised extraction request, if needed:
