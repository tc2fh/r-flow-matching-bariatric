# Methods - Target-Trial Emulation and Distributional Evaluation of a Bariatric Digital-Twin Model

Manuscript-facing methods draft for the MBSAQIP/Cosmos bariatric digital-twin analysis.
Written to be read and revised with the research group; the engineering plan that
implements it is `TTE_DISTRIBUTIONAL_RUN_PLAN.md`, and the modeling rationale log is
`MACE_MODELING_DECISIONS.md`. Effect sizes and citations flagged "[verify]" must be
confirmed against source before submission.

---

## 1. Study design and data source

We used the Cosmos `MBSCohort` registry extract of adults undergoing a first bariatric
operation. The analytic cohort was restricted to GLP-1-naive patients with type-2 diabetes
undergoing either laparoscopic sleeve gastrectomy (SG; CPT 43775) or Roux-en-Y gastric
bypass (RYGB; CPT 43644/43846, and the reconstruction variant 43645 mapped to RYGB), with a
baseline BMI of 35-75 kg/m^2 and a procedure date on or before 2023-05-01. The GLP-1-naive
restriction (`PriorGLP1 = 0`) is a deliberate inclusion criterion isolating a clean
GLP-1-naive surgical baseline, analogous to trials excluding prior bariatric surgery;
GLP-1-exposed patients are held in a companion cohort. The cohort funnel is reported as a
CONSORT-style diagram.

The model is a modular digital twin: a calibrated gradient-boosted risk model (GBM) for a
composite cardiometabolic-renal complication endpoint (MACE OR nephropathy OR retinopathy),
and a conditional flow-matching model that generates the joint predictive distribution of
BMI and HbA1c trajectories across follow-up horizons (3 months to 6 years), conditioned on
pre-operative covariates, the operation, and the complication event. Because the flow
emits a full predictive distribution per patient rather than a point estimate, it supports
two products a point calculator cannot: (i) clinically thresholded probabilities, e.g.
P(BMI < 35) and P(HbA1c < 5.7%) at a given horizon, and (ii) a per-patient counterfactual
contrast between SG and RYGB. Products (i) and (ii) are the basis of the analyses below.

---

## 2. Target-trial specification and emulation

Because the operation was chosen in routine care rather than randomized, the SG-versus-RYGB
contrast is a causal quantity and is treated as such using the target-trial-emulation
framework (Hernan and Robins, 2016 [verify]). We specify the protocol of the hypothetical
randomized trial and emulate each component with the observational data.

| Protocol element | Target trial | Emulation |
|---|---|---|
| Eligibility | GLP-1-naive T2D adults, first bariatric operation, eligible for SG or RYGB | The cohort filters above; stated as the enrolled population |
| Treatment strategies | "Undergo SG" vs "Undergo RYGB" at time zero | `CptCode` (43775 = SG; 43644/43846/43645 = RYGB) |
| Assignment | Randomization | Not randomized; conditional exchangeability given measured confounders L, enforced by a propensity model |
| Time zero | Enrollment = treatment = start of follow-up, aligned | Date of surgery (`ProcDateValue`); a one-time (point) intervention, so there is no immortal-time window and the intention-to-treat and per-protocol contrasts coincide |
| Outcome | BMI and HbA1c trajectory; target-threshold attainment; composite complication | Continuous BMI/HbA1c at each horizon, threshold probabilities, and the composite endpoint |
| Causal contrast | Marginal average treatment effect (ATE) and conditional average treatment effect (CATE) | Marginal ATE by doubly-robust estimation; CATE by the twin's per-patient counterfactual |
| Analysis | Adjusted comparison | Doubly-robust IPCW-augmented inverse-probability weighting; see Section 3 |

### 2.1 Estimands

The primary estimand is the marginal ATE of RYGB versus SG on each outcome at each horizon:
for continuous outcomes the difference in mean BMI (kg/m^2) and HbA1c (%-points), and for
the thresholded outcomes the risk difference in P(BMI < 35) and P(HbA1c < 5.7%). The
secondary estimand is the individualized CATE, the per-patient difference produced by the
twin under each operation, which underlies the individualized decision-support readout.

---

## 3. Confounding control

### 3.1 Propensity model and weighting

The measured confounder set L comprises pre-operative variables available in the cohort:
age, sex, baseline BMI and weight, baseline HbA1c, serum creatinine and eGFR, the
comorbidity history (diabetes, hypertension, obstructive sleep apnea, dyslipidemia, prior
myocardial infarction, stroke, atrial fibrillation, venous thromboembolism), baseline
antidiabetic medication classes (insulin, biguanide, SGLT2 inhibitor; confirmed
pre-operative), social-determinant and geographic measures (insurance coverage class, RUCA
rurality, the CDC Social Vulnerability Index subscores), and surgery year. A gradient-boosted
propensity model estimates P(RYGB | L), fit on the training split and applied to the test
split to preserve the leak-free split discipline used throughout the pipeline. Overlap
(positivity) is assessed by the estimated-propensity distribution within each arm; patients
outside common support are trimmed and the number trimmed is reported. Confounding is
addressed with stabilized inverse-probability-of-treatment weights, and covariate balance
is summarized by standardized mean differences before and after weighting (a Love plot;
target |SMD| < 0.1).

### 3.2 Censoring and loss to follow-up

Follow-up completeness declines with horizon, a known feature of registry bariatric
follow-up. To prevent this attrition from biasing outcome comparisons, we model the
probability of an observed outcome at each horizon given L and apply
inverse-probability-of-censoring weights; the analytic weight at a horizon is the product of
the treatment and censoring weights. Death (`DeathInterval`) is treated as a censoring event
for the metabolic trajectory (post-mortem BMI is undefined); the competing-risk framing for
the complication endpoint is noted as an extension.

### 3.3 Doubly-robust estimation with the twin as the outcome model

The marginal ATE is estimated with an IPCW-augmented inverse-probability-weighted (AIPW)
estimator. The twin supplies the outcome-regression (g-computation) component E[Y | A, L] as
its per-arm predicted mean (for continuous outcomes) or predicted threshold probability (for
the thresholded outcomes); the propensity and censoring models supply the weighting
component. The AIPW estimator is consistent if either the outcome model or the
weighting models are correctly specified, and its influence-function variance yields
95% confidence intervals. This construction reframes the twin's counterfactual surgery-flip
as a formal g-computation estimator embedded in a target trial, rather than an informal
what-if.

---

## 4. Borrowing a randomized backbone

Two clinically decisive confounders are absent from the cohort: gastroesophageal reflux
disease (the principal indication that steers operation choice toward RYGB) and
surgeon/center identity (a dominant driver of which operation a patient receives). No
propensity adjustment can recover unmeasured variables, so the SG-versus-RYGB contrast
cannot rest on the observational design alone. We therefore anchor it to randomized
evidence in three graded ways.

First, and primarily, we benchmark the emulated marginal effect against the head-to-head
randomized trials of SG versus RYGB - SLEEVEPASS (Salminen et al., 2018; 10-year 2022
[verify]), SM-BOSS (Peterli et al., 2018 [verify]), the Oseberg diabetes-remission trial
(Hofso et al., 2019 [verify]), and STAMPEDE (Schauer et al., 2017 [verify]) - together with
their meta-analyses. If the doubly-robust emulated effect on weight loss and diabetes
remission falls within the randomized confidence region, that agreement is an empirical
calibration of the causal design: evidence that confounding control succeeded despite the
missing variables (the logic of randomized-trial emulation benchmarking; Wang et al.,
RCT-DUPLICATE [verify]). This benchmark applies only to weight and glycemia, which the
trials measured; endpoints are reconciled to matching definitions and units before
comparison (the model's BMI-point difference converted to percent total weight loss, and
the P(HbA1c < 5.7%) threshold stated against each trial's remission definition).

Second, the trials establish the direction and approximate magnitude of the effect (RYGB
gives somewhat greater weight loss and diabetes remission, at the cost of more early
complications and greater reflux control), so our contribution is positioned as the
individualized, calibrated magnitude conditional on covariates rather than a rediscovery of
the average effect - directly addressing the observation that "RYGB outperforms SG" is
established knowledge and a validity check rather than a novel finding.

Third, as a sensitivity analysis, an informative-prior formulation places a prior on the
average effect centered on the randomized meta-analytic estimate with the trials'
uncertainty, letting the observational data estimate covariate-driven deviations from it; in
this formulation residual confounding can bend the individualized surface but cannot move the
population mean away from the randomized result.

Crucially, no randomized head-to-head trial is powered for the composite
cardiometabolic-renal endpoint. The complication contrast therefore has no randomized
backbone and is reported as hypothesis-generating, with the widest sensitivity caveat.

---

## 5. Sensitivity analyses

Robustness to unmeasured confounding (chiefly reflux and surgeon/center) is quantified with
the E-value (VanderWeele and Ding, 2017 [verify]) for each primary contrast and its
confidence limit, reporting how strong an unmeasured confounder would have to be, on the
risk-ratio scale, to explain away the estimate. Negative-control outcomes (outcomes not
plausibly affected by operation choice) are used to detect residual confounding. The
attrition sensitivity analysis reports every outcome estimate with and without
inverse-probability-of-censoring weighting, and stratifies model calibration by follow-up
completeness; the gap between the weighted and unweighted results is the quantitative measure
of how much informative attrition moves the conclusions.

---

## 6. Distributional evaluation of the twin

Because the twin's value proposition is a calibrated predictive distribution - the property
whose absence degraded prior weight-trajectory calculators on external patients - the
evaluation grades the distribution and the specific quantities delivered, not only the point
error. The organizing principle is to maximize sharpness subject to calibration (Gneiting et
al., 2007 [verify]): every calibration metric is reported alongside a sharpness metric so
that neither a diffuse-but-calibrated nor a sharp-but-miscalibrated model is mistaken for a
good one. Point accuracy (mean absolute deviation, RMSE) is retained for comparability but is
not the primary evidence.

Predictive distributions are graded with strictly proper scoring rules: the continuous ranked
probability score per horizon; a tail-sensitive logarithmic score; and, over blocks of
horizons, the energy score and the variogram score, which grade the joint distribution and
its cross-horizon dependency structure rather than each marginal in isolation. Interval
calibration is summarized by empirical coverage at multiple nominal levels and by the
interval (Winkler) score, which rewards correct coverage and narrow width jointly and cannot
be improved by widening intervals alone; quantile calibration is summarized by pinball loss
across a quantile grid. Distributional calibration of the sampler is diagnosed by the
probability-integral-transform histogram, which separates a biased center (a location shift,
requiring a sampler fix) from incorrect spread (dispersion, addressable by conformal
calibration).

The clinically delivered quantities - the threshold probabilities P(BMI < 35) and
P(HbA1c < 5.7%) at each horizon - are calibrated directly: predicted probabilities are binned
against observed attainment frequencies (a reliability curve), summarized by expected and
maximum calibration error and by the Brier score with its reliability-resolution-uncertainty
decomposition. Because prior calculators failed specifically under distribution shift,
calibration is recomputed on a temporal validation fold (training on earlier and testing on
later surgery dates), reporting the calibration slope and calibration-in-the-large on the
internal and temporal folds and the drift between them.

All distributional metrics are, by default, computed on patients with observed follow-up at
each horizon and are therefore susceptible to the same attrition bias as the outcome
analysis; each is additionally reported with inverse-probability-of-censoring weighting, and
the naive-versus-weighted gap is reported. Simulated cohort marginals (from the twin's full
generative sampling) are compared to observed marginals by the Wasserstein-1 distance and
median shift per horizon, rather than by a null-hypothesis test whose p-value is
uninformative at this sample size. Finally, the individualized SG-versus-RYGB benefit is
validated with the concordance-for-benefit statistic (van Klaveren et al., 2018 [verify]),
which assesses whether the twin's predicted per-patient benefit ordering matches the observed
benefit in matched patient pairs, and clinical usefulness is summarized by decision-curve
(net-benefit) analysis for the risk and threshold readouts.

---

## 7. Handling of attrition and missing data

Missing outcomes over time are addressed by the inverse-probability-of-censoring weighting in
Sections 3.2 and 6; the trajectory model additionally uses all available observations, so a
patient contributes at each horizon they are observed rather than being dropped for
incomplete follow-up. Primary conclusions are anchored at horizons with adequate sample size,
with the longest-horizon estimates reported as exploratory with wide intervals and explicit
per-horizon effective sample sizes. Missing baseline covariates are routed natively by the
tree-based propensity and censoring models (no imputation), and informative missingness is
tested by adding missingness indicators.

---

## 8. Reporting and limitations

The prediction model is reported to the TRIPOD+AI guideline (Collins et al., 2024 [verify])
and the causal analysis to the target-trial-emulation reporting checklist. Discrimination is
reported with bootstrap confidence intervals and DeLong tests for model-versus-model
differences, and per-component discrimination is reported for the composite endpoint so its
aggregate performance is not oversold.

Principal limitations: (1) the operation was not randomized and two decisive indication
variables (reflux disease, surgeon/center) are unmeasured, so the surgery contrast is
decision-support conditional on stated assumptions rather than an established causal effect,
defended by the randomized benchmark and E-value; (2) follow-up attrition is substantial at
long horizons and is addressed but not eliminated by censoring weights; (3) the complication
endpoint has no randomized backbone and is hypothesis-generating; (4) the analysis is
single-registry, and temporal rather than fully external validation is the pre-submission
standard, with external validation as an explicit next step; (5) later-era patients are more
GLP-1-selected and less follow-up-mature by construction, which is reported per era.

---

## References (verify all before submission)

- Hernan MA, Robins JM. Using big data to emulate a target trial when a randomized trial is not available. Am J Epidemiol. 2016;183(8):758-764.
- VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value. Ann Intern Med. 2017;167(4):268-274.
- van Klaveren D, et al. The proposed 'concordance-statistic for benefit' ... J Clin Epidemiol. 2018;94:59-68.
- Gneiting T, Balabdaoui F, Raftery AE. Probabilistic forecasts, calibration and sharpness. J R Stat Soc B. 2007;69(2):243-268.
- Gneiting T, Raftery AE. Strictly proper scoring rules, prediction, and estimation. J Am Stat Assoc. 2007;102(477):359-378.
- Austin PC, Stuart EA. Moving towards best practice when using inverse probability of treatment weighting (IPTW). Stat Med. 2015;34(28):3661-3679.
- Bang H, Robins JM. Doubly robust estimation in missing data and causal inference models. Biometrics. 2005;61(4):962-973.
- Salminen P, et al. SLEEVEPASS 5-year (JAMA 2018;319:241-254) and 10-year (JAMA Surg 2022) results.
- Peterli R, et al. SM-BOSS. JAMA. 2018;319(3):255-265.
- Hofso D, et al. Oseberg (gastric bypass vs sleeve, diabetes remission). Lancet Diabetes Endocrinol. 2019;7(12):912-924.
- Schauer PR, et al. STAMPEDE 5-year outcomes. N Engl J Med. 2017;376(7):641-651.
- Wang SV, et al. RCT-DUPLICATE: emulation of randomized trials with real-world data.
- Collins GS, et al. TRIPOD+AI statement. BMJ. 2024;385:e078378.
