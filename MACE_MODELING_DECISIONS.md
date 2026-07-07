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
