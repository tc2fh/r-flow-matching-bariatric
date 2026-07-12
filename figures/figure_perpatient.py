"""Figure spec A - per-patient 5-column trajectory figures (BMI and HbA1c), standalone.

Renders the SAME 5-column figure the twin evaluator now emits, but as a figure builder
that runs against a frozen run dir (re-samples the display patients from the saved twin
and calls the shared renderer, so there is one source of truth for the layout). Columns:
  1 factual band | 2 counterfactual band | 3 delta-median |
  4 P(threshold) factual surgery | 5 P(threshold) counterfactual surgery
Threshold BMI < 35 / HbA1c < 5.7. Non-calibrated horizons (flow PIT regime != calibrated,
read from eval_flow_calibration_pit.csv) are drawn with open markers.

Uses figures.sampling.display_patient_samples + evaluate_flow_matching's extended
plot_timecourse_factual_counterfactual.
"""

from __future__ import annotations

from pathlib import Path

from . import style
from . import artifacts as A
from .artifacts import RunArtifacts
from . import sampling as S


def _trust_months(art: RunArtifacts, group: str) -> set[float]:
    return {float(r["months"]) for r in art.trust_table(group) if r["trustworthy"]}


def build(art: RunArtifacts, out_dir: Path, *, prefix: str = "figA", bundle: S.TwinBundle | None = None,
          n_show: int = 3, n_samples: int = 200, n_steps: int = 50, seed: int = 0,
          device: str = "cpu", csv_path: str | None = None) -> list[Path]:
    style.apply_rcparams()
    import evaluate_flow_matching as efm

    if bundle is None:
        bundle = S.load_frozen(art, device=device, csv_path=csv_path)
    selected, factual, counterfactual = S.display_patient_samples(
        bundle, n_show=n_show, n_samples=n_samples, n_steps=n_steps, seed=seed)

    out_dir = Path(out_dir)
    written: list[Path] = []
    specs = [
        ("bmi", "BMI", (15.0, 90.0), 35.0, "BMI < 35"),
        ("hba1c", "HbA1c", (3.0, 15.0), 5.7, "HbA1c < 5.7"),
    ]
    for group, y_label, ylim, thr, thr_label in specs:
        stem = out_dir / f"{prefix}_{group}_perpatient_5col"
        efm.plot_timecourse_factual_counterfactual(
            bundle.dataset, selected, factual, counterfactual, group, y_label,
            stem.with_suffix(".png"),
            f"Per-patient {y_label}: factual vs surgery-counterfactual + P({thr_label}) "
            f"({n_show}/arm; bounded display, raw violations in safety supplement)",
            max_sample_lines=50, y_limits=ylim,
            threshold=thr, threshold_label=thr_label,
            trustworthy_months=_trust_months(art, group),
            vector_stem=stem,
        )
        written += [stem.with_suffix(".pdf"), stem.with_suffix(".svg"), stem.with_suffix(".png")]
    return written
