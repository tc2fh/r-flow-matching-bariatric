"""Render every W6 journal figure against a frozen / eval run directory.

    OMP_NUM_THREADS=1 <venv>/bin/python -m figures.build_all --run runs/frozen/<ts>

Main figures land in ``<out>/main/`` (sparse, journal-ready, .pdf + .svg + .png each);
the existing table screenshots and multi-patient diagnostic panels are routed to
``<out>/supplement/`` so the main set stays clean. The twin model is loaded once and
shared by the two figures that must re-sample (Figure B and Figure A).
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # macOS torch+xgboost dual-OpenMP guard

import argparse
import shutil
import traceback
from pathlib import Path

from . import artifacts as A
from . import figure_consort, figure_gbm, figure_trajectory, figure_counterfactual
from . import figure_perpatient, figure_ablation, figure_causal, figure_distributional
from . import figure_counterfactual_diagnostics


# Existing artifacts routed to supplement/ (screenshots + diagnostic panels). (src name, dest name)
SUPPLEMENT_EVAL = [
    ("eval_flow_threshold_probabilities_summary.png", "supp_threshold_probabilities_table.png"),
    ("eval_flow_threshold_probabilities.png", "supp_threshold_probabilities_curves.png"),
    ("eval_flow_calibration_pit.png", "supp_calibration_pit.png"),
    ("eval_flow_calibration_coverage.png", "supp_calibration_coverage.png"),
    ("eval_flow_calibration_crps.png", "supp_calibration_crps.png"),
    ("eval_flow_timepoint_metrics_test.png", "supp_flow_timepoint_metrics.png"),
    ("eval_gbm_per_component_discrimination.png", "supp_gbm_per_component_discrimination.png"),
    ("eval_gbm_curves_test.png", "supp_gbm_curves.png"),
    ("eval_sim_trajectory_marginals.png", "supp_sim_trajectory_marginals.png"),
    ("eval_sim_event_marginal.png", "supp_sim_event_marginal.png"),
    ("eval_sim_surgery_counterfactual.png", "supp_sim_surgery_counterfactual.png"),
    ("eval_sim_modeA_vs_modeB_metrics.png", "supp_sim_modeA_vs_modeB.png"),
]


def _route_supplement(art: A.RunArtifacts, supp_dir: Path) -> list[str]:
    routed = []
    for src, dest in SUPPLEMENT_EVAL:
        p = art.eval_dir / src
        if p.exists():
            shutil.copy2(p, supp_dir / dest)
            routed.append(dest)
    # Table-one screenshot(s)
    if art.root and (art.root / "table_one").is_dir():
        for png in sorted((art.root / "table_one").glob("table_one_*/table_one.png")):
            shutil.copy2(png, supp_dir / "supp_table_one.png")
            routed.append("supp_table_one.png")
            break
    return routed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="frozen run dir, evaluation dir, or pipeline dir")
    ap.add_argument("--out", default=None, help="output dir (default <run>/figures)")
    ap.add_argument("--csv", default=None, help="override source cohort CSV (for CONSORT + re-sampling)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-show", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-db", action="store_true", help="CONSORT: pull SQL per-clause counts (VM only)")
    args = ap.parse_args()

    art = A.resolve(args.run)
    out = Path(args.out) if args.out else (art.root / "figures")
    main_dir, supp_dir = out / "main", out / "supplement"
    main_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resolved run: eval_dir={art.eval_dir}\n  twin_run={art.twin_run_dir}\n  out={out}")

    results: list[tuple[str, str, str]] = []

    def _run(name: str, fn):
        try:
            written = fn()
            paths = ", ".join(Path(p).name for p in written) if written else "(skipped)"
            results.append((name, "PASS" if written else "SKIP", paths))
        except Exception as exc:  # noqa: BLE001 - one figure failing must not kill the rest
            results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()

    # --- CSV-only main figures (no model load) --- #
    _run("Fig1 CONSORT", lambda: figure_consort.build(
        art, main_dir / "fig1_consort", use_db=args.use_db, csv_path=args.csv))
    _run("Fig2 GBM (ROC/reliability/DCA)", lambda: figure_gbm.build(art, main_dir / "fig2_gbm"))
    _run("Fig4 Calibrated trajectory", lambda: figure_trajectory.build(art, main_dir / "fig4_trajectory_calibrated"))
    # tte_*/dist_* CSV -> figure (skips cleanly when a run has no causal/distributional artifacts)
    _run("Fig5 Causal effects (TTE)", lambda: figure_causal.build(art, main_dir / "fig5_causal_effects"))
    _run("Fig6 Distributional validation",
         lambda: figure_distributional.build(art, main_dir / "fig6_distributional"))

    # --- model-backed figures (load the twin once, share) --- #
    bundle = None
    try:
        from . import sampling as S
        bundle = S.load_frozen(art, device=args.device, csv_path=args.csv)
    except Exception as exc:  # noqa: BLE001
        print(f"[build_all] could not load twin bundle ({type(exc).__name__}: {exc}); "
              "Figure B and Figure A will be skipped.")
        traceback.print_exc()

    _run("Fig3/B RYGB-vs-sleeve", lambda: figure_counterfactual.build(
        art, main_dir / "fig3_rygb_vs_sleeve", bundle=bundle, n_samples=args.n_samples,
        n_steps=args.n_steps, seed=args.seed, n_boot=args.n_boot, device=args.device, csv_path=args.csv))
    _run("FigA per-patient 5-col", lambda: figure_perpatient.build(
        art, main_dir, prefix="figA", bundle=bundle, n_show=args.n_show, n_samples=args.n_samples,
        n_steps=args.n_steps, seed=args.seed, device=args.device, csv_path=args.csv))

    # --- supplement --- #
    _run("Supp ablation", lambda: figure_ablation.build(art, supp_dir / "supp_ablation_event_conditioning"))
    if bundle is not None:
        _run("Supp counterfactual safety", lambda: figure_counterfactual_diagnostics.build(
            bundle, supp_dir / "supp_counterfactual_safety", n_samples=args.n_samples,
            n_steps=args.n_steps, seed=args.seed))
    routed = _route_supplement(art, supp_dir)
    print(f"\nRouted {len(routed)} existing artifacts to supplement/.")

    print("\n=== figure build summary ===")
    for name, status, detail in results:
        print(f"  [{status}] {name:30s} {detail}")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\nMain+supplement dir: {out}")
    print("RESULT:", "ALL OK" if n_fail == 0 else f"{n_fail} FAILED")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
