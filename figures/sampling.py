"""Shared twin sampling for the two figures that must draw from the model.

Figure B/3 (cohort RYGB-vs-sleeve counterfactual) and Figure A (per-patient 5-column,
standalone builder) both need fresh predictive samples that are not persisted to any
CSV (the bands/quantiles come from the model). This module loads the frozen twin once
and exposes the two sampling primitives, reusing evaluate_twin's public helpers so the
sampling matches the evaluator exactly.

Imports the torch stack (via evaluate_twin); OMP_NUM_THREADS is pinned to 1 here as a
belt-and-suspenders guard for the macOS torch+xgboost dual-OpenMP segfault (the launch
command should ALSO set it in the environment before the process starts).
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

import evaluate_twin as ev
import train_flow_matching_twin as tw
import evaluate_flow_matching as efm
from .artifacts import RunArtifacts

SLEEVE_IDX, RNYGB_IDX = 0, 1


@dataclass
class TwinBundle:
    dataset: object
    splits: dict
    pre: object
    model: object
    twin_cfg: object
    device: torch.device
    test_idx: np.ndarray


def _resolve_csv(art: RunArtifacts, csv_path: str | None) -> Path | None:
    src = csv_path or art.source_csv
    if src is None:
        return None
    p = Path(src)
    if not p.exists() and not p.is_absolute():
        p = Path.cwd() / src
    return p


def load_frozen(art: RunArtifacts, *, device: str = "cpu", csv_path: str | None = None) -> TwinBundle:
    if art.twin_run_dir is None:
        raise FileNotFoundError("twin_run_dir unresolved; cannot load the frozen twin for sampling.")
    dev = ev.choose_device(device)
    twin_cfg = ev.load_twin_config(art.twin_run_dir)
    pre = ev.load_twin_preprocessing(art.twin_run_dir)
    model = ev.restore_twin(art.twin_run_dir, twin_cfg, dev)
    dataset = ev.load_dataset(_resolve_csv(art, csv_path))
    splits = tw.make_splits(dataset, twin_cfg)
    return TwinBundle(dataset, splits, pre, model, twin_cfg, dev, splits["test"])


def cohort_surgery_medians(bundle: TwinBundle, *, n_samples: int = 200, n_steps: int = 50,
                           seed: int = 0) -> tuple[list[str], dict[str, np.ndarray]]:
    """Clamp the whole test cohort to sleeve, then rnygb; return per-patient sample
    MEDIAN at each of the 15 continuous timepoints (original units) for each arm.

    Returns (cont_names, {"sleeve": (n_test,15), "rnygb": (n_test,15)}), Mode-A
    (true-event) conditioned; event coupling is near-zero so the surgery contrast is
    what moves the trajectory.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    base = ev.arrays_for(bundle.dataset, bundle.test_idx, bundle.pre)
    event = base["y_mace"].astype(np.float32)
    cfg = replace(bundle.twin_cfg, n_samples_per_patient=n_samples, sample_steps=n_steps)
    noise = np.random.default_rng(seed).standard_normal(
        (bundle.test_idx.size, n_samples, tw.X_CONT_DIM)
    ).astype(np.float32)
    medians: dict[str, np.ndarray] = {}
    for name, sidx in (("sleeve", SLEEVE_IDX), ("rnygb", RNYGB_IDX)):
        arms = {**base, "surgery_idx": np.full_like(base["surgery_idx"], sidx)}
        samp15 = ev.twin_samples_15(
            bundle.model, arms, event, cfg, bundle.pre, bundle.device,
            initial_noise=noise, bound_output=True,
        )  # (n,S,15)
        medians[name] = np.median(samp15, axis=1)  # (n,15) native units
    return list(tw.CONT_NAMES), medians


def display_patient_samples(bundle: TwinBundle, *, n_show: int = 3, n_samples: int = 200,
                            n_steps: int = 50, seed: int = 0):
    """Selected display patients + factual and surgery-counterfactual FULL-dim samples
    (Mode-A, true event), matching evaluate_flow. Returns (selected_idx, factual_full,
    counterfactual_full) where the sample arrays are (n_show*2, n_samples, fm.X_DIM)."""
    rng = np.random.default_rng(seed)
    selected = efm.select_display_patients(bundle.dataset, bundle.test_idx, rng, n_show)
    arrays = ev.arrays_for(bundle.dataset, selected, bundle.pre)
    event = arrays["y_mace"]
    cfg = replace(bundle.twin_cfg, n_samples_per_patient=n_samples, sample_steps=n_steps)
    torch.manual_seed(seed); np.random.seed(seed)
    noise = np.random.default_rng(seed).standard_normal(
        (selected.size, n_samples, tw.X_CONT_DIM)
    ).astype(np.float32)
    factual = ev.scatter_to_full(ev.twin_samples_15(
        bundle.model, arrays, event, cfg, bundle.pre, bundle.device,
        initial_noise=noise, bound_output=True,
    ))
    counterfactual = ev.scatter_to_full(
        ev.twin_samples_15(
            bundle.model, arrays, event, cfg, bundle.pre, bundle.device,
            flip_surgery=True, initial_noise=noise, bound_output=True,
        ))
    return selected, factual, counterfactual
