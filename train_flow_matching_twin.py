"""Digital-twin trainer: event-CONDITIONED flow matching for BMI/HbA1c trajectories.

This is the "flow half" of the modular digital twin (see MACE_MODELING_DECISIONS.md,
2026-07 session 2). It is a fork of ``train_flow_matching_multitask.py`` with two
deliberate changes:

  * the MACE classification head is DROPPED -- the calibrated GBM
    (``gbm_mace_baseline.py``) owns composite-event risk, not this model; and
  * the 2 MACE target dims are DROPPED -- the flow generates ONLY the 15
    continuous BMI/HbA1c dims (``train_flow_matching_multitask.CONT_DIMS``);

and one addition:

  * the binary composite event ``mace_ever`` (``dataset.x[:, MACE_DIM]``) is added
    as a CONDITIONING input, handled exactly like surgery type (its own embedding,
    concatenated into the shared encoder). At train time we teacher-force the TRUE
    observed label.

Why this is a digital twin -- the chain-rule factorization
----------------------------------------------------------
We model the joint distribution of (event, trajectory) given pre-op covariates x
by the chain rule::

    p(event, trajectory | x) = p(event | x) . p(trajectory | event, x)
                               \\___ GBM ___/   \\_______ flow (this) ______/

The GBM owns the event marginal p(event | x); this flow owns the trajectory
conditioned on the event. The trajectory marginal then falls out as the mixture
p(traj | x) = p . flow(x, e=1) + (1-p) . flow(x, e=0), correct by construction if
both factors are (and if the GBM is calibrated -- that is the one requirement).

Why condition on the binary EVENT, not the GBM's risk SCORE
-----------------------------------------------------------
The score p(x) is a deterministic function of x, so p(traj | x, p(x)) = p(traj | x)
-- conditioning on it adds no coupling. To capture *residual* dependence (two
patients with identical x, but the one who goes on to have an event tends to have a
different trajectory) we condition on the realized binary event e, giving
p(traj | e, x) != p(traj | x). That residual coupling is exactly what a twin needs.

Why teacher-forcing the TRUE label at train time is leak-free
-------------------------------------------------------------
Conditioning on the *observed* event label during training is ordinary conditional
density estimation, NOT prediction: at generation time the event is itself sampled
(from the GBM), never read from the future. Because BOTH the event and the
trajectory are sampled at generation, feeding the true label at train time leaks
nothing and needs no out-of-fold cross-fitting. (OOF would only be required if we
instead conditioned on the GBM's predicted score.) The GBM is NOT consumed here at
train or tune time; the two models are fit independently and meet only at sampling.

The GBM -> Bernoulli -> flow sampling path (done in the evaluator/simulator)
---------------------------------------------------------------------------
    p     = GBM(x)            # calibrated composite-event risk  (correct marginal)
    event ~ Bernoulli(p)      # a concrete drawn event
    traj  ~ flow(x, event)    # trajectory conditioned on the drawn event  (this model)

Data loading/preprocessing/splitting and the continuous-dim plumbing are reused by
import from ``train_flow_matching`` (pristine core, never modified) and
``train_flow_matching_multitask`` (the sibling fork: CONT_DIMS, Preprocessing,
fit_preprocessing, split_arrays, flow_matching_loss, ...). Only the model, the
event-conditioned training objective, and event-conditioned sampling are new here.

Run (local smoke test from the fake CSV)::

    python train_flow_matching_twin.py --csv fake_data/fake_mbs_cohort.csv --num-steps 200

Run (standalone against Cosmos via the imported pyodbc path)::

    python train_flow_matching_twin.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import train_flow_matching as fm
import train_flow_matching_multitask as mt


DEFAULT_OUTPUT_DIR = fm.REPO_ROOT / "runs" / "python_flow_matching_twin"

# The twin generates exactly the continuous BMI/HbA1c dims and conditions on the
# binary composite event. Reuse the sibling fork's dimension bookkeeping so the
# split, preprocessing, and continuous-dim ordering line up patient-for-patient.
CONT_DIMS = mt.CONT_DIMS
CONT_NAMES = mt.CONT_NAMES
CONT_GROUPS = mt.CONT_GROUPS
X_CONT_DIM = mt.X_CONT_DIM
MACE_DIM = mt.MACE_DIM
MACE_LABEL_NAME = mt.MACE_LABEL_NAME

# Reuse the sibling fork's preprocessing container + report helper verbatim.
Preprocessing = mt.Preprocessing
report_saved = mt.report_saved


@dataclass
class TwinConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "cpu"
    seed: int = 0
    split_seed: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    # "surgery" reproduces the Cosmos flow split so test patients line up
    # one-for-one with the GBM and the pristine flow; "temporal" is the out-of-time
    # fold (fm.make_temporal_splits: earliest surgeries -> train, latest -> test),
    # shared patient-for-patient with the GBM and Table 1; "outcome" stratifies
    # jointly by surgery and the binary event.
    split_strategy: str = "surgery"
    # Shared encoder (surgery embedding + EVENT embedding + patient features).
    surgery_emb_dim: int = 8
    event_emb_dim: int = 8  # binary composite-event conditioning, handled like surgery
    # W4 ablation toggle. True = event-CONDITIONED trajectory (the coupled twin);
    # False = drop the event embedding entirely so the flow is UNCONDITIONED on the
    # event (baseline that quantifies how much the coupling actually buys). One flag
    # flips the whole model between the two arms; the encoder width shrinks by
    # ``event_emb_dim`` when this is False (see TwinNet.__init__).
    use_event: bool = True
    cond_hidden_dim: int = 64
    cond_num_layers: int = 2
    # Flow head.
    time_emb_dim: int = 64
    time_scale: float = 10.0
    hidden_dim: int = 64
    num_hidden_layers: int = 2
    # Optimization.
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    num_steps: int = 6000
    batch_size: int = 64
    early_stop_patience: int = 8
    early_stop_min_delta: float = 0.0
    log_every: int = 100
    val_every: int = 250
    val_repeats: int = 8
    # Sampling / evaluation.
    sample_steps: int = 50
    n_samples_per_patient: int = 50
    # Maximum number of patient-sample trajectories evaluated by the flow at once.
    # Cohort figures may request millions of trajectories; bounding the flattened
    # inference batch prevents native allocator crashes while preserving every draw.
    sample_batch_size: int = 65536


# --------------------------------------------------------------------------- #
# Splitting + preprocessing (reused from the multi-task fork)
# --------------------------------------------------------------------------- #
def make_splits(dataset: fm.FlowDataset, cfg: TwinConfig) -> dict[str, np.ndarray]:
    """Dispatch on ``cfg.split_strategy``.

    "surgery" delegates to ``fm.make_stratified_splits`` (the exact split used by
    the pristine flow, the GBM, and the multi-task fork), so with a matching
    split_seed + fractions every model in the twin shares its test patients
    patient-for-patient. "temporal" delegates to ``fm.make_temporal_splits`` (the
    out-of-time fold: earliest surgeries -> train, latest -> test); it shares the same
    delegate, split_seed, and fractions as the GBM and Table 1, so under
    ``split_strategy=="temporal"`` all three still see the identical patient-for-patient
    partition (asserted in train_twin_pipeline via SHARED_SPLIT_KEYS). "outcome"
    stratifies jointly by surgery and the binary event via the multi-task fork's
    stratifier.
    """
    if cfg.split_strategy == "surgery":
        return fm.make_stratified_splits(
            dataset,
            fm.TrainConfig(
                split_seed=cfg.split_seed,
                train_frac=cfg.train_frac,
                val_frac=cfg.val_frac,
                test_frac=cfg.test_frac,
            ),
        )
    if cfg.split_strategy == "temporal":
        return fm.make_temporal_splits(
            dataset,
            fm.TrainConfig(
                split_seed=cfg.split_seed,
                train_frac=cfg.train_frac,
                val_frac=cfg.val_frac,
                test_frac=cfg.test_frac,
            ),
        )
    if cfg.split_strategy != "outcome":
        raise ValueError(f"Unknown split_strategy: {cfg.split_strategy!r} (expected 'surgery', 'temporal', or 'outcome')")
    y = dataset.x[:, MACE_DIM].astype(np.int64)
    proxy = mt.MultiTaskConfig(
        split_seed=cfg.split_seed, train_frac=cfg.train_frac, val_frac=cfg.val_frac, test_frac=cfg.test_frac
    )
    return mt.stratified_splits_by_outcome(dataset.surgery_type, y, proxy)


# --------------------------------------------------------------------------- #
# Model: shared encoder (surgery + EVENT + patient features) -> flow head
# --------------------------------------------------------------------------- #
class TwinNet(nn.Module):
    """Event-conditioned flow vector field for the 15 continuous BMI/HbA1c dims.

    Mirrors ``MultiTaskNet`` but (a) drops the classification head and (b) adds a
    binary-event embedding into the shared encoder, so the flow is conditioned on
    (surgery, event, patient features).

    W4 ablation -- ``cfg.use_event`` selects the arm:
      * True  (default): encoder input = surgery_emb (8) + event_emb (8) + patient
        features (8) = 24; this is the coupled, event-conditioned twin.
      * False: the event embedding is not created and not concatenated, so the
        encoder input = surgery_emb (8) + patient features (8) = 16, i.e. the event
        arm width minus ``event_emb_dim``. A no-event checkpoint therefore carries
        no ``event_emb.*`` tensors and a narrower ``encoder.0`` layer, so save/load
        round-trips at exactly the right width in each mode.
    """

    def __init__(self, cfg: TwinConfig, x_cont_dim: int, patient_feature_dim: int, num_surgery_types: int = 2):
        super().__init__()
        if cfg.time_emb_dim % 2 != 0:
            raise ValueError("time_emb_dim must be even")
        self.x_cont_dim = x_cont_dim
        self.time_emb_dim = cfg.time_emb_dim
        self.time_scale = cfg.time_scale
        self.use_event = cfg.use_event
        self.surgery_emb = nn.Embedding(num_surgery_types, cfg.surgery_emb_dim)
        # The event embedding exists ONLY in the conditioned arm. Creating it
        # conditionally (rather than building it and skipping it) keeps the checkpoint
        # width honest: a no-event model has no event_emb weights and a narrower
        # encoder, so it cannot silently load into an event-conditioned model.
        event_dim = 0
        if self.use_event:
            self.event_emb = nn.Embedding(2, cfg.event_emb_dim)
            event_dim = cfg.event_emb_dim

        static_dim = cfg.surgery_emb_dim + event_dim + patient_feature_dim
        encoder_layers: list[nn.Module] = []
        in_dim = static_dim
        for _ in range(cfg.cond_num_layers):
            encoder_layers.append(nn.Linear(in_dim, cfg.cond_hidden_dim))
            encoder_layers.append(nn.SiLU())
            in_dim = cfg.cond_hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.cond_repr_dim = cfg.cond_hidden_dim if cfg.cond_num_layers > 0 else static_dim

        self.cond_flow_dim = cfg.time_emb_dim + self.cond_repr_dim
        in_dim = x_cont_dim + self.cond_flow_dim
        flow_layers: list[nn.Module] = []
        for _ in range(cfg.num_hidden_layers):
            flow_layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            in_dim = cfg.hidden_dim + self.cond_flow_dim
        self.flow_hidden = nn.ModuleList(flow_layers)
        self.flow_out = nn.Linear(in_dim, x_cont_dim)

    def encode(self, surgery_idx: torch.Tensor, patient_features: torch.Tensor, event_idx: torch.Tensor) -> torch.Tensor:
        surgery = self.surgery_emb(surgery_idx.long())
        if self.use_event:
            event = self.event_emb(event_idx.long())
            static = torch.cat([surgery, event, patient_features], dim=-1)
        else:
            # No-event arm: ``event_idx`` is still accepted so every trainer/sampler
            # keeps one call signature, but it is deliberately dropped here.
            static = torch.cat([surgery, patient_features], dim=-1)
        return self.encoder(static)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, cond_repr: torch.Tensor) -> torch.Tensor:
        t_emb = fm.sinusoidal_time_embedding(t, self.time_emb_dim, self.time_scale)
        cond = torch.cat([t_emb, cond_repr], dim=-1)
        h = torch.cat([x_t, cond], dim=-1)
        for layer in self.flow_hidden:
            h = F.silu(layer(h))
            h = torch.cat([h, cond], dim=-1)
        return self.flow_out(h)


# --------------------------------------------------------------------------- #
# Flow loss / eval / sampling (event-conditioned)
# --------------------------------------------------------------------------- #
def evaluate_flow_loss(model: TwinNet, arrays: dict, cfg: TwinConfig, device: torch.device) -> float:
    if arrays["x"].shape[0] == 0:
        return float("nan")
    x1 = fm.as_tensor(arrays["x"], device)
    mask = fm.as_tensor(arrays["mask"], device)
    surgery_idx = fm.as_tensor(arrays["surgery_idx"], device, torch.long)
    patient_features = fm.as_tensor(arrays["patient_features"], device)
    event = fm.as_tensor(arrays["y_mace"], device, torch.long)  # teacher-forced TRUE event
    losses = []
    model.eval()
    with torch.no_grad():
        cond = model.encode(surgery_idx, patient_features, event)
        for _ in range(cfg.val_repeats):
            x_t, t, u_t = fm.sample_conditional_path(x1)
            pred = model.velocity(x_t, t, cond)
            losses.append(float(mt.flow_matching_loss(pred, u_t, mask).detach().cpu()))
    model.train()
    return float(np.mean(losses))


def sample_trajectories(
    model: TwinNet,
    arrays: dict,
    cfg: TwinConfig,
    device: torch.device,
    x_cont_dim: int,
    event: np.ndarray | None = None,
) -> np.ndarray:
    """Sample the 15 continuous dims for each patient, conditioned on ``event``.

    ``event`` is a per-patient 0/1 array. Default = the true observed label
    (``arrays["y_mace"]``, i.e. Mode-A oracle conditioning). The simulator passes
    a Bernoulli draw (Mode C) or a clamped value (counterfactual) instead.
    """
    n_patients = arrays["patient_features"].shape[0]
    if n_patients == 0:
        return np.zeros((0, cfg.n_samples_per_patient, x_cont_dim), dtype=np.float32)
    if event is None:
        event = arrays["y_mace"]
    event = np.asarray(event).reshape(-1).astype(np.int64)
    n_samples = cfg.n_samples_per_patient
    batch_size = cfg.sample_batch_size
    if n_samples <= 0:
        raise ValueError(f"n_samples_per_patient must be positive, got {n_samples}.")
    if cfg.sample_steps <= 0:
        raise ValueError(f"sample_steps must be positive, got {cfg.sample_steps}.")
    if batch_size <= 0:
        raise ValueError(f"sample_batch_size must be positive, got {batch_size}.")

    total = n_patients * n_samples
    samples = np.empty((total, x_cont_dim), dtype=np.float32)
    if total > batch_size:
        n_batches = (total + batch_size - 1) // batch_size
        print(
            f"[sampling] {total:,} trajectories in {n_batches} batches "
            f"(at most {batch_size:,} per batch).",
            flush=True,
        )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dt = 1.0 / cfg.sample_steps
            # Draw noise once with the historical cohort-wide shape. PyTorch's
            # seeded random stream depends on each requested tensor shape, so
            # drawing per batch would silently change results when the memory cap
            # changes. This tensor is small compared with the repeated hidden-layer
            # activations that batching bounds (about 52 MB for the full VM run).
            initial_noise = torch.randn(total, x_cont_dim, device=device)
            for start in range(0, total, batch_size):
                stop = min(start + batch_size, total)
                # Flattened output is patient-major, with all draws for patient 0
                # first. Derive only this batch's patient indices instead of
                # materializing a cohort-wide repeated index array.
                patient_idx = np.arange(start, stop, dtype=np.int64) // n_samples
                surgery_idx = fm.as_tensor(
                    arrays["surgery_idx"][patient_idx], device, torch.long
                )
                patient_features = fm.as_tensor(
                    arrays["patient_features"][patient_idx], device
                )
                event_idx = fm.as_tensor(event[patient_idx], device, torch.long)
                cond = model.encode(surgery_idx, patient_features, event_idx)
                x = initial_noise[start:stop]
                for step in range(cfg.sample_steps):
                    t = torch.full(
                        (stop - start,), step * dt, dtype=torch.float32, device=device
                    )
                    x = x + dt * model.velocity(x, t, cond)
                samples[start:stop] = x.detach().cpu().numpy()
    finally:
        model.train(was_training)

    return samples.reshape(n_patients, n_samples, x_cont_dim)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(dataset: fm.FlowDataset, cfg: TwinConfig) -> dict:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    splits = make_splits(dataset, cfg)
    pre = mt.fit_preprocessing(dataset, splits["train"])
    arrays = mt.split_arrays(dataset, splits, pre)
    run_dir = make_run_dir(cfg.output_dir)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {**asdict(cfg), "x_cont_dim": X_CONT_DIM, "cont_names": CONT_NAMES, "mace_label": MACE_LABEL_NAME},
            f,
            indent=2,
        )
    report_saved(run_dir / "config.json", "run config")
    with (run_dir / "preprocessing.json").open("w", encoding="utf-8") as f:
        json.dump(pre.to_jsonable(), f, indent=2)
    report_saved(run_dir / "preprocessing.json", "preprocessing stats")

    model = TwinNet(cfg, X_CONT_DIM, len(fm.PATIENT_FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    batch_size = min(cfg.batch_size, max(1, arrays["train"]["x"].shape[0]))

    logs: list[dict] = []
    best_score = float("inf")
    best_step = -1
    best_state = None
    evals_since_improve = 0
    early_stopped = False

    train_prev = float(arrays["train"]["y_mace"].mean())
    print(
        f"Patients: {len(dataset.subject_ids)} "
        f"(train={splits['train'].size}, val={splits['val'].size}, test={splits['test'].size}, "
        f"x_cont_dim={X_CONT_DIM})"
    )
    print(f"Composite-event train prevalence (teacher-forced conditioning)={train_prev:.4f}")

    for step in range(1, cfg.num_steps + 1):
        model.train()
        batch = mt.batch_sample(arrays["train"], batch_size, rng)
        x1 = fm.as_tensor(batch["x"], device)
        mask = fm.as_tensor(batch["mask"], device)
        surgery_idx = fm.as_tensor(batch["surgery_idx"], device, torch.long)
        patient_features = fm.as_tensor(batch["patient_features"], device)
        event = fm.as_tensor(batch["y_mace"], device, torch.long)  # teacher-force TRUE event

        cond = model.encode(surgery_idx, patient_features, event)
        x_t, t, u_t = fm.sample_conditional_path(x1)
        pred = model.velocity(x_t, t, cond)
        flow_loss = mt.flow_matching_loss(pred, u_t, mask)

        optimizer.zero_grad()
        flow_loss.backward()
        optimizer.step()
        flow_scalar = float(flow_loss.detach().cpu())

        should_eval = step == 1 or step % cfg.val_every == 0 or step == cfg.num_steps
        if should_eval:
            val_flow = evaluate_flow_loss(model, arrays["val"], cfg, device)
            score = flow_scalar if np.isnan(val_flow) else val_flow
            improved = score < best_score - cfg.early_stop_min_delta
            if improved:
                best_score = score
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                evals_since_improve = 0
            else:
                evals_since_improve += 1
            logs.append(
                {
                    "step": step,
                    "train_flow": flow_scalar,
                    "val_flow": val_flow,
                    "score": score,
                    "best_score": best_score,
                }
            )
            pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)
            print(
                f"Step {step}/{cfg.num_steps} flow={flow_scalar:.4f} "
                f"val_flow={val_flow:.4f} best={best_score:.4f}@{best_step}"
            )
            if not np.isnan(val_flow) and evals_since_improve >= cfg.early_stop_patience:
                early_stopped = True
                print(f"Early stopping at step {step}")
                break
        elif step % cfg.log_every == 0:
            print(f"Step {step}/{cfg.num_steps} flow={flow_scalar:.4f}")

    report_saved(run_dir / "training_log.csv", "training log")
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / "model.pt")
    report_saved(run_dir / "model.pt", "model checkpoint")

    return finalize_and_evaluate(model, arrays, pre, cfg, run_dir, best_step, early_stopped, device)


def finalize_and_evaluate(
    model: TwinNet,
    arrays: dict,
    pre: Preprocessing,
    cfg: TwinConfig,
    run_dir: Path,
    best_step: int,
    early_stopped: bool,
    device: torch.device,
) -> dict:
    """Mode-A intrinsic check: condition on the TRUE event, score the trajectory.

    This is the flow's *oracle* quality (does the trajectory model fit given we
    know the event?), independent of the GBM. The deployable (Mode B) and full
    simulation (Mode C) checks live in the monolithic evaluator, which brings the
    GBM in to draw the event.
    """
    test_arrays = arrays["test"]
    samples_std = sample_trajectories(model, test_arrays, cfg, device, X_CONT_DIM, event=test_arrays["y_mace"])
    samples_original = mt.unstandardize(samples_std, pre)
    pred_mean, p10, p90 = mt.summarize_samples(samples_original)
    flow_table = mt.flow_metrics(pred_mean, test_arrays["original_x"], test_arrays["original_mask"])
    flow_table["split"] = "test"
    flow_table["conditioning"] = "true_event(mode_A)"
    flow_table["best_step"] = best_step
    flow_table["early_stopped"] = early_stopped
    flow_table.to_csv(run_dir / "test_flow_metrics.csv", index=False)
    report_saved(run_dir / "test_flow_metrics.csv", "continuous (BMI/HbA1c) Mode-A test metrics")

    predictions = pd.DataFrame({"subject_id": test_arrays["subject_ids"]})
    predictions["mace_true"] = test_arrays["y_mace"].astype(np.int64)
    for dim, name in enumerate(CONT_NAMES):
        predictions[f"pred_mean_{name}"] = pred_mean[:, dim]
        predictions[f"pred_p10_{name}"] = p10[:, dim]
        predictions[f"pred_p90_{name}"] = p90[:, dim]
        observed = test_arrays["original_x"][:, dim].copy()
        observed[test_arrays["original_mask"][:, dim] == 0] = np.nan
        predictions[f"observed_{name}"] = observed
        predictions[f"observed_mask_{name}"] = test_arrays["original_mask"][:, dim]
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    report_saved(run_dir / "test_predictions.csv", "per-patient Mode-A predictions (true-event conditioned)")

    print("\nContinuous-outcome (flow, Mode-A true-event) test metrics:")
    print(flow_table.to_string(index=False))
    print(f"\nSaved run artifacts to {run_dir}")
    return {"run_dir": run_dir, "flow_metrics": flow_table}


def make_run_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def train_from_csv(csv_path: str | Path, cfg: TwinConfig | None = None) -> dict:
    cfg = cfg or TwinConfig()
    return train_model(fm.load_dataset_from_csv(csv_path), cfg)


def train_from_database(cfg: TwinConfig | None = None) -> dict:
    cfg = cfg or TwinConfig()
    return train_model(fm.load_dataset_from_database(), cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--split-strategy", type=str, default=None, choices=["surgery", "temporal", "outcome"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-event", dest="no_event", action="store_true",
                        help="W4 ablation: train the UNCONDITIONED arm (drop the event embedding).")
    args = parser.parse_args()

    cfg = TwinConfig(output_dir=args.output_dir, device=args.device, seed=args.seed)
    if args.split_strategy is not None:
        cfg.split_strategy = args.split_strategy
    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
    if args.no_event:
        cfg.use_event = False

    try:
        if args.csv_path:
            train_from_csv(args.csv_path, cfg)
        else:
            train_from_database(cfg)
    except RuntimeError as exc:
        print(f"ERROR: {exc}\n\nPass --csv <path> to train from a saved CSV export.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
