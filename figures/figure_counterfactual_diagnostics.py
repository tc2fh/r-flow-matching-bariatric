"""Supplement: paired-noise counterfactual safety and solver dashboard."""

from __future__ import annotations

from pathlib import Path

import counterfactual_audit as audit

from .sampling import TwinBundle


def build(bundle: TwinBundle, out_stem: Path, *, n_samples: int = 64,
          n_steps: int = 50, seed: int = 0) -> list[Path]:
    result = audit.run(
        dataset=bundle.dataset,
        splits=bundle.splits,
        pre=bundle.pre,
        model=bundle.model,
        twin_cfg=bundle.twin_cfg,
        device=bundle.device,
        output_dir=out_stem.parent,
        prefix=out_stem.name,
        n_samples=min(n_samples, 64),
        n_steps=n_steps,
        seed=seed,
    )
    written = [Path(path) for path in result["dashboard"]]
    written.extend([
        Path(result["headline_csv"]),
        Path(result["headline_png"]),
        Path(result["patient_horizon_csv"]),
        Path(result["horizon_summary_csv"]),
        Path(result["solver_convergence_csv"]),
    ])
    return written
