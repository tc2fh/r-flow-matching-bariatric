"""Per-surgery probability that a patient reaches a BMI threshold at a timepoint.

The twin's flow sampler already produces the full predictive distribution
(n_samples trajectories per patient), so a threshold probability is just the
fraction of a patient's samples below the cutoff:

    P(BMI_t < c | x, surgery, event) = mean_over_samples( sample_BMI_t < c )

This script answers "what is the likelihood a patient has BMI < 35 at 12 months,
for RYGB vs sleeve?" as a COUNTERFACTUAL contrast: it clamps the whole test cohort
to each surgery in turn (same patients, same baseline covariates) and reports the
cohort-average probability. That is the causal question the twin is built for, and
it is something a point-prediction calculator (e.g. SOPHIA) cannot produce.

Event handling is shown three ways so the (weak) event coupling is visible:
  * e=0        : condition on "no complication"
  * e=1        : condition on "complication"
  * risk-wtd   : p_GBM * P(.|e=1) + (1 - p_GBM) * P(.|e=0)  -- the deployable marginal
The risk-weighted column is the headline; with near-zero coupling the three agree.

Usage (against a frozen pipeline)::

    python bmi_threshold_probability.py \
        --pipeline runs/twin_pipeline/<pipeline_dir> \
        --csv fake_data/fake_mbs_cohort.csv \
        --threshold 35 --timepoint bmi_12m --n-samples 200

On the Cosmos VM drop --csv to read MBSCohort through pyodbc.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_twin as ev
import gbm_mace_baseline as gb
import train_flow_matching as fm

SURGERY_ARMS = [("sleeve", 0), ("rnygb", 1)]

# Clinically meaningful cutoffs, one family per outcome. BMI < 35 = below Class II
# obesity; HbA1c < 5.7% = back into the non-diabetic range. "below" throughout (the
# patient reaching a healthier value). W5 generalizes the readout to EVERY output
# timepoint (BMI 3m..6y, HbA1c 12m..6y), not just bmi_12m.
BMI_THRESHOLD = 35.0
HBA1C_THRESHOLD = 5.7


def default_threshold_targets() -> list[tuple[str, float, str]]:
    """(timepoint, threshold, direction) for every continuous output dim."""
    targets = [(item[0], BMI_THRESHOLD, "below") for item in fm.BMI_TARGETS]
    targets += [(item[0], HBA1C_THRESHOLD, "below") for item in fm.HBA1C_TARGETS]
    return targets


def _hit_fraction(col: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    """Per-patient P(threshold crossed) = fraction of that patient's samples past the cut."""
    hit = (col < threshold) if direction == "below" else (col > threshold)
    return hit.mean(axis=1)


def threshold_probability_table(model, base_arrays, subject_ids, targets,
                                p_gbm_factual, p_gbm_cf, sample_cfg, pre, device) -> "pd.DataFrame":
    """Per-patient threshold probabilities for FACTUAL vs COUNTERFACTUAL surgery.

    ``base_arrays`` are ``arrays_for(dataset, test_idx, pre)`` at the patients' factual
    surgery; ``p_gbm_factual`` / ``p_gbm_cf`` are the calibrated composite-event risks at
    the factual / flipped surgery (from compute_gbm_predictions: test_cal / test_cf_cal).
    For each arm we sample the full 15-dim trajectory ONCE per event value (e=0/e=1) and
    read every timepoint off it, then risk-weight per patient:

        P(cross) = p_GBM * P(cross | e=1) + (1 - p_GBM) * P(cross | e=0)   [deployable marginal]

    Returns a tidy long frame: one row per (patient, timepoint, {factual,counterfactual}).
    BMI<35 vs HbA1c<5.7 are distinguished by each row's ``group`` / ``threshold``. These
    probabilities inherit the flow's tail calibration, so they are only trustworthy where
    the PIT/coverage pass says the horizon is calibrated (see the ``calibration_dependent``
    flag the evaluator attaches to the cohort summary).
    """
    import pandas as pd

    n = len(subject_ids)
    arms = [("factual", False, np.asarray(p_gbm_factual)), ("counterfactual", True, np.asarray(p_gbm_cf))]
    rows = []
    for arm_name, flip, p_gbm in arms:
        samp = {}
        for event_val in (0, 1):
            event = np.full(n, event_val, dtype=np.float32)
            samp[event_val] = ev.twin_samples_15(model, base_arrays, event, sample_cfg, pre, device,
                                                 flip_surgery=flip)  # (n, n_samples, 15)
        surgery_idx = (1 - base_arrays["surgery_idx"]) if flip else base_arrays["surgery_idx"]
        for name, threshold, direction in targets:
            dim = fm.TARGET_NAMES.index(name)  # BMI/HbA1c dims are 0..14 == the 15-dim sample order
            group = fm.TARGET_GROUPS[dim]
            frac0 = _hit_fraction(samp[0][:, :, dim], threshold, direction)
            frac1 = _hit_fraction(samp[1][:, :, dim], threshold, direction)
            p_risk = p_gbm * frac1 + (1.0 - p_gbm) * frac0
            for i, sid in enumerate(subject_ids):
                rows.append({
                    "subject_id": sid, "surgery_arm": arm_name, "surgery_idx": int(surgery_idx[i]),
                    "timepoint": name, "group": group, "threshold": threshold, "direction": direction,
                    "p_event0": float(frac0[i]), "p_event1": float(frac1[i]),
                    "p_risk": float(p_risk[i]), "p_gbm": float(p_gbm[i]),
                })
    return pd.DataFrame(rows)


def cohort_probability(model, base_arrays, dataset, splits, gbm, x, surgery_col,
                       dim, threshold, direction, sample_cfg, pre, device):
    """Return per-arm per-patient probabilities for e=0, e=1, and risk-weighted."""
    test_idx = splits["test"]
    estimator = gbm["estimator"]
    results = {}
    for name, sidx in SURGERY_ARMS:
        arrays = {**base_arrays,
                  "surgery_idx": np.full_like(base_arrays["surgery_idx"], sidx)}
        frac = {}
        for event_val in (0, 1):
            event = np.full(test_idx.shape[0], event_val, dtype=np.float32)
            samples_15 = ev.twin_samples_15(model, arrays, event, sample_cfg, pre, device)
            col = ev.scatter_to_full(samples_15)[:, :, dim]  # (n_patients, n_samples)
            hit = (col < threshold) if direction == "below" else (col > threshold)
            frac[event_val] = hit.mean(axis=1)  # per-patient probability
        x_clamped = x.copy()
        x_clamped[:, surgery_col] = float(sidx)
        p_gbm = estimator.predict_proba(x_clamped[test_idx])[:, 1]
        risk_weighted = p_gbm * frac[1] + (1.0 - p_gbm) * frac[0]
        results[name] = {"e0": frac[0], "e1": frac[1], "risk": risk_weighted, "p_gbm": p_gbm}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", type=str, default=None,
                        help="pipeline dir with manifest.json (gives gbm + twin run dirs)")
    parser.add_argument("--twin-run", type=str, default=None)
    parser.add_argument("--gbm-run", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=35.0)
    parser.add_argument("--timepoint", type=str, default="bmi_12m",
                        help="a BMI/HbA1c target name, e.g. bmi_12m, bmi_2y, hba1c_12m")
    parser.add_argument("--direction", choices=["below", "above"], default="below")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=None, help="optional per-patient CSV")
    parser.add_argument("--all-timepoints", dest="all_timepoints", action="store_true",
                        help="Emit per-patient P(BMI<35) & P(HbA1c<5.7) at EVERY output timepoint "
                             "(factual vs counterfactual surgery) instead of the single-timepoint arm contrast.")
    args = parser.parse_args()

    if args.pipeline:
        manifest = ev.resolve_from_pipeline(Path(args.pipeline))
        gbm_run_dir = Path(args.gbm_run or manifest["gbm_run_dir"])
        twin_run_dir = Path(args.twin_run or manifest["twin_final_run_dir"])
    else:
        if not (args.twin_run and args.gbm_run):
            raise SystemExit("Provide --pipeline, or both --twin-run and --gbm-run.")
        gbm_run_dir, twin_run_dir = Path(args.gbm_run), Path(args.twin_run)

    device = torch.device(args.device)
    dataset = ev.load_dataset(Path(args.csv) if args.csv else None)

    gbm_cfg = ev.load_gbm_config(gbm_run_dir)
    splits = gb.make_splits(dataset, gbm_cfg)

    twin_cfg = ev.load_twin_config(twin_run_dir)
    pre = ev.load_twin_preprocessing(twin_run_dir)
    model = ev.restore_twin(twin_run_dir, twin_cfg, device)
    sample_cfg = replace(twin_cfg, n_samples_per_patient=args.n_samples, sample_steps=args.n_steps)

    if args.timepoint not in fm.TARGET_NAMES:
        raise SystemExit(f"Unknown timepoint {args.timepoint!r}. Choices: {fm.TARGET_NAMES}")
    dim = fm.TARGET_NAMES.index(args.timepoint)

    gbm = ev.compute_gbm_predictions(gbm_cfg, dataset, splits)
    x, feature_names, _ = gb.assemble_features(dataset)
    surgery_col = feature_names.index("surgery_idx")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base_arrays = ev.arrays_for(dataset, splits["test"], pre)

    if args.all_timepoints:
        targets = default_threshold_targets()
        p_fac = gbm["test_cal"] if gbm["calibrated"] else gbm["test_raw"]
        p_cf = gbm["test_cf_cal"] if gbm["calibrated"] else gbm["test_cf_raw"]
        subject_ids = dataset.subject_ids[splits["test"]]
        table = threshold_probability_table(model, base_arrays, subject_ids, targets,
                                            p_fac, p_cf, sample_cfg, pre, device)
        out_path = args.out or "threshold_probabilities_all_timepoints.csv"
        table.to_csv(out_path, index=False)
        print(f"\nPer-timepoint threshold probabilities (factual vs counterfactual surgery)  "
              f"n_test={subject_ids.size}, samples/patient={args.n_samples}, "
              f"gbm_calibrated={gbm['calibrated']}")
        summary = (table.groupby(["group", "timepoint", "threshold", "surgery_arm"], sort=False)["p_risk"]
                   .mean().reset_index().rename(columns={"p_risk": "cohort_mean_p_risk"}))
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(summary.round(4).to_string(index=False))
        print(f"  [saved] per-patient x per-timepoint threshold probabilities -> {out_path}")
        return

    results = cohort_probability(model, base_arrays, dataset, splits, gbm, x, surgery_col,
                                 dim, args.threshold, args.direction, sample_cfg, pre, device)

    n_test = splits["test"].shape[0]
    op = "<" if args.direction == "below" else ">"
    print(f"\nP({args.timepoint} {op} {args.threshold:g})  -- test cohort clamped to each surgery"
          f"  (n_test={n_test}, samples/patient={args.n_samples}, backend={gbm['backend']}, "
          f"gbm_calibrated={gbm['calibrated']})")
    print(f"{'surgery':>10} | {'e=0':>8} | {'e=1':>8} | {'risk-wtd':>9}")
    print("-" * 46)
    for name, _ in SURGERY_ARMS:
        r = results[name]
        print(f"{name:>10} | {100*r['e0'].mean():7.1f}% | {100*r['e1'].mean():7.1f}% | "
              f"{100*r['risk'].mean():8.1f}%")
    delta = 100 * (results["rnygb"]["risk"].mean() - results["sleeve"]["risk"].mean())
    print("-" * 46)
    print(f"counterfactual delta (rnygb - sleeve), risk-weighted: {delta:+.1f} percentage points\n")

    if args.out:
        rows = pd.DataFrame({
            "subject_id": dataset.subject_ids[splits["test"]],
            "p_sleeve_risk": results["sleeve"]["risk"],
            "p_rnygb_risk": results["rnygb"]["risk"],
            "p_sleeve_e0": results["sleeve"]["e0"], "p_sleeve_e1": results["sleeve"]["e1"],
            "p_rnygb_e0": results["rnygb"]["e0"], "p_rnygb_e1": results["rnygb"]["e1"],
        })
        rows.to_csv(args.out, index=False)
        print(f"  [saved] per-patient probabilities -> {args.out}")


if __name__ == "__main__":
    main()
