"""Joint multi-task trainer: flow matching for continuous outcomes + a MACE head.

This mirrors the structure of ``train_flow_matching.py`` (one large standalone
script) but changes the modeling so that the rare composite MACE label is no
longer a *generated* flow dimension. Instead:

    patient features + surgery type
                 |
          [ shared encoder ]
            /            \\
   [ classification ]   [ flow vector field ]
     head -> P(MACE)      head -> velocity for the 15 continuous
                          BMI/HbA1c dimensions

Rationale (see project discussion): a binary 6-20%-prevalence outcome is a
classification problem, and treating it as a Gaussian-transported flow dimension
both smears the Bernoulli and lets 15 continuous dims drown its gradient. A
dedicated sigmoid head trained with weighted BCE (or focal loss) handles the
imbalance directly, while the flow still models the joint continuous-outcome
distribution. The shared encoder lets the two tasks regularize each other.

Data loading/preprocessing is reused from ``train_flow_matching`` (imported,
never modified). All multi-task modeling logic lives in this file.

Run (local smoke test from the fake CSV)::

    python train_flow_matching_multitask.py --csv fake_data/fake_mbs_cohort.csv \
        --num-steps 200

Run (standalone against Cosmos via the imported pyodbc path)::

    python train_flow_matching_multitask.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import train_flow_matching as fm


DEFAULT_OUTPUT_DIR = fm.REPO_ROOT / "runs" / "python_flow_matching_multitask"
MACE_LABEL_NAME = "mace_ever"

# The flow models only the continuous BMI/HbA1c outcomes; MACE goes to the head.
CONT_DIMS = np.asarray(
    [i for i, group in enumerate(fm.TARGET_GROUPS) if group in {"bmi", "hba1c"}],
    dtype=np.int64,
)
CONT_NAMES = [fm.TARGET_NAMES[i] for i in CONT_DIMS]
CONT_GROUPS = [fm.TARGET_GROUPS[i] for i in CONT_DIMS]
X_CONT_DIM = int(CONT_DIMS.size)
MACE_DIM = fm.TARGET_NAMES.index(MACE_LABEL_NAME)


@dataclass
class MultiTaskConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "cpu"
    seed: int = 0
    split_seed: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    # Shared encoder.
    surgery_emb_dim: int = 8
    cond_hidden_dim: int = 64
    cond_num_layers: int = 2
    # Flow head.
    time_emb_dim: int = 64
    time_scale: float = 10.0
    hidden_dim: int = 64
    num_hidden_layers: int = 2
    # Classification head.
    cls_hidden_dim: int = 64
    cls_num_layers: int = 2
    # Loss weighting / imbalance handling.
    cls_loss_weight: float = 1.0
    auto_pos_weight: bool = True
    pos_weight: float | None = None  # explicit override; None + auto -> n_neg/n_pos
    focal_gamma: float = 0.0  # >0 enables focal loss on the classification head
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
    select_metric: str = "combined"  # one of {"combined", "flow", "auprc"}
    # Sampling / evaluation.
    sample_steps: int = 50
    n_samples_per_patient: int = 50
    target_specificity: float = 0.90
    recalibrate: bool = True


@dataclass
class Preprocessing:
    target_mean: np.ndarray
    target_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    static_continuous_idx: np.ndarray
    patient_feature_names: list[str]
    cont_names: list[str]

    def to_jsonable(self) -> dict:
        return {
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "static_mean": self.static_mean.tolist(),
            "static_std": self.static_std.tolist(),
            "static_continuous_idx": self.static_continuous_idx.tolist(),
            "patient_feature_names": self.patient_feature_names,
            "cont_names": self.cont_names,
        }


# --------------------------------------------------------------------------- #
# Splitting + preprocessing (outcome-stratified, tolerant of sparse dims)
# --------------------------------------------------------------------------- #
def stratified_splits_by_outcome(
    surgery_type: np.ndarray, y: np.ndarray, cfg: MultiTaskConfig
) -> dict[str, np.ndarray]:
    """Stratify jointly by surgery type and the binary MACE outcome.

    Identical logic (and default seed) to the GBM baseline so the two models are
    evaluated on the same test patients.
    """
    if not np.isclose(cfg.train_frac + cfg.val_frac + cfg.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    rng = np.random.default_rng(cfg.split_seed)
    train_parts, val_parts, test_parts = [], [], []
    for surgery in sorted(set(surgery_type.tolist())):
        for label in (0, 1):
            idx = np.where((surgery_type == surgery) & (y == label))[0]
            if idx.size == 0:
                continue
            rng.shuffle(idx)
            n_train = int(np.floor(idx.size * cfg.train_frac))
            n_val = int(np.floor(idx.size * cfg.val_frac))
            train_parts.append(idx[:n_train])
            val_parts.append(idx[n_train : n_train + n_val])
            test_parts.append(idx[n_train + n_val :])
    splits = {
        "train": np.concatenate(train_parts).astype(np.int64),
        "val": np.concatenate(val_parts).astype(np.int64),
        "test": np.concatenate(test_parts).astype(np.int64),
    }
    for key in splits:
        rng.shuffle(splits[key])
    return splits


def fit_preprocessing(dataset: fm.FlowDataset, train_idx: np.ndarray) -> Preprocessing:
    x = dataset.x[train_idx][:, CONT_DIMS].astype(np.float64)
    mask = dataset.mask[train_idx][:, CONT_DIMS].astype(np.float64)
    observed = mask.sum(axis=0)
    if np.any(observed == 0):
        empty = [CONT_NAMES[i] for i in np.where(observed == 0)[0]]
        warnings.warn(
            f"Continuous target dims with no train observations (mean=0,std=1 fallback): {empty}",
            stacklevel=2,
        )
    safe_obs = np.maximum(observed, 1.0)
    mean = np.where(observed > 0, (x * mask).sum(axis=0) / safe_obs, 0.0)
    var = np.where(observed > 0, (((x - mean) ** 2) * mask).sum(axis=0) / safe_obs, 1.0)
    std = np.sqrt(var)
    std = np.where((std < 1e-8) | ~np.isfinite(std), 1.0, std)

    raw = dataset.patient_features_raw[train_idx].astype(np.float64)
    continuous_idx = np.asarray(
        [fm.PATIENT_FEATURES.index(name) for name in fm.CONTINUOUS_PATIENT_FEATURES],
        dtype=np.int64,
    )
    static_mean = np.zeros(raw.shape[1], dtype=np.float64)
    static_std = np.ones(raw.shape[1], dtype=np.float64)
    static_mean[continuous_idx] = np.nan_to_num(np.nanmean(raw[:, continuous_idx], axis=0))
    spread = np.nanstd(raw[:, continuous_idx], axis=0)
    static_std[continuous_idx] = np.where((spread < 1e-8) | ~np.isfinite(spread), 1.0, spread)
    return Preprocessing(
        target_mean=mean.astype(np.float32),
        target_std=std.astype(np.float32),
        static_mean=static_mean.astype(np.float32),
        static_std=static_std.astype(np.float32),
        static_continuous_idx=continuous_idx,
        patient_feature_names=fm.PATIENT_FEATURES.copy(),
        cont_names=CONT_NAMES.copy(),
    )


def transform_targets(x_cont: np.ndarray, mask_cont: np.ndarray, pre: Preprocessing) -> np.ndarray:
    return (((x_cont - pre.target_mean) / pre.target_std) * mask_cont).astype(np.float32)


def transform_patient_features(raw: np.ndarray, pre: Preprocessing) -> np.ndarray:
    out = raw.copy().astype(np.float32)
    idx = pre.static_continuous_idx
    missing_continuous = np.isnan(out[:, idx])
    if missing_continuous.any():
        out[:, idx] = np.where(missing_continuous, pre.static_mean[idx], out[:, idx])
    missing_other = np.isnan(out)
    if missing_other.any():
        out = np.where(missing_other, 0.0, out)
    out[:, idx] = (out[:, idx] - pre.static_mean[idx]) / pre.static_std[idx]
    return out


def split_arrays(
    dataset: fm.FlowDataset, splits: dict[str, np.ndarray], pre: Preprocessing
) -> dict[str, dict]:
    x_cont = dataset.x[:, CONT_DIMS]
    mask_cont = dataset.mask[:, CONT_DIMS]
    x_std = transform_targets(x_cont, mask_cont, pre)
    p_std = transform_patient_features(dataset.patient_features_raw, pre)
    y = dataset.x[:, MACE_DIM].astype(np.float32)

    out = {}
    for name, idx in splits.items():
        out[name] = {
            "x": x_std[idx],
            "mask": mask_cont[idx],
            "surgery_idx": dataset.surgery_idx[idx],
            "patient_features": p_std[idx],
            "y_mace": y[idx],
            "subject_ids": dataset.subject_ids[idx],
            "original_x": x_cont[idx],
            "original_mask": mask_cont[idx],
        }
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MultiTaskNet(nn.Module):
    def __init__(self, cfg: MultiTaskConfig, x_cont_dim: int, patient_feature_dim: int, num_surgery_types: int = 2):
        super().__init__()
        if cfg.time_emb_dim % 2 != 0:
            raise ValueError("time_emb_dim must be even")
        self.x_cont_dim = x_cont_dim
        self.time_emb_dim = cfg.time_emb_dim
        self.time_scale = cfg.time_scale
        self.surgery_emb = nn.Embedding(num_surgery_types, cfg.surgery_emb_dim)

        static_dim = cfg.surgery_emb_dim + patient_feature_dim
        encoder_layers: list[nn.Module] = []
        in_dim = static_dim
        for _ in range(cfg.cond_num_layers):
            encoder_layers.append(nn.Linear(in_dim, cfg.cond_hidden_dim))
            encoder_layers.append(nn.SiLU())
            in_dim = cfg.cond_hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.cond_repr_dim = cfg.cond_hidden_dim if cfg.cond_num_layers > 0 else static_dim

        cls_layers: list[nn.Module] = []
        in_dim = self.cond_repr_dim
        for _ in range(cfg.cls_num_layers):
            cls_layers.append(nn.Linear(in_dim, cfg.cls_hidden_dim))
            cls_layers.append(nn.SiLU())
            in_dim = cfg.cls_hidden_dim
        cls_layers.append(nn.Linear(in_dim, 1))
        self.cls_head = nn.Sequential(*cls_layers)

        self.cond_flow_dim = cfg.time_emb_dim + self.cond_repr_dim
        in_dim = x_cont_dim + self.cond_flow_dim
        flow_layers: list[nn.Module] = []
        for _ in range(cfg.num_hidden_layers):
            flow_layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            in_dim = cfg.hidden_dim + self.cond_flow_dim
        self.flow_hidden = nn.ModuleList(flow_layers)
        self.flow_out = nn.Linear(in_dim, x_cont_dim)

    def encode(self, surgery_idx: torch.Tensor, patient_features: torch.Tensor) -> torch.Tensor:
        surgery = self.surgery_emb(surgery_idx.long())
        static = torch.cat([surgery, patient_features], dim=-1)
        return self.encoder(static)

    def classify(self, cond_repr: torch.Tensor) -> torch.Tensor:
        return self.cls_head(cond_repr).squeeze(-1)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, cond_repr: torch.Tensor) -> torch.Tensor:
        t_emb = fm.sinusoidal_time_embedding(t, self.time_emb_dim, self.time_scale)
        cond = torch.cat([t_emb, cond_repr], dim=-1)
        h = torch.cat([x_t, cond], dim=-1)
        for layer in self.flow_hidden:
            h = F.silu(layer(h))
            h = torch.cat([h, cond], dim=-1)
        return self.flow_out(h)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def flow_matching_loss(pred: torch.Tensor, u_t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (mask * (pred - u_t).pow(2)).sum() / (mask.sum() + 1e-8)


def classification_loss(
    logit: torch.Tensor, y: torch.Tensor, pos_weight: float | None, focal_gamma: float
) -> torch.Tensor:
    if focal_gamma and focal_gamma > 0.0:
        ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        p = torch.sigmoid(logit)
        p_t = torch.where(y > 0.5, p, 1.0 - p)
        loss = (1.0 - p_t).pow(focal_gamma) * ce
        if pos_weight is not None:
            weight = torch.where(y > 0.5, torch.as_tensor(pos_weight, device=logit.device), torch.ones((), device=logit.device))
            loss = weight * loss
        return loss.mean()
    pw = None if pos_weight is None else torch.as_tensor(float(pos_weight), device=logit.device)
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)


def resolve_pos_weight(cfg: MultiTaskConfig, y_train: np.ndarray) -> float | None:
    if cfg.pos_weight is not None:
        return float(cfg.pos_weight)
    if not cfg.auto_pos_weight:
        return None
    n_pos = float(y_train.sum())
    n_neg = float((y_train == 0).sum())
    if n_pos <= 0:
        return None
    return n_neg / n_pos


# --------------------------------------------------------------------------- #
# Sampling / prediction
# --------------------------------------------------------------------------- #
def batch_sample(arrays: dict, batch_size: int, rng: np.random.Generator) -> dict:
    n = arrays["x"].shape[0]
    idx = rng.choice(n, size=batch_size, replace=batch_size > n)
    return {key: value[idx] for key, value in arrays.items() if isinstance(value, np.ndarray)}


def evaluate_flow_loss(model: MultiTaskNet, arrays: dict, cfg: MultiTaskConfig, device: torch.device) -> float:
    if arrays["x"].shape[0] == 0:
        return float("nan")
    x1 = fm.as_tensor(arrays["x"], device)
    mask = fm.as_tensor(arrays["mask"], device)
    surgery_idx = fm.as_tensor(arrays["surgery_idx"], device, torch.long)
    patient_features = fm.as_tensor(arrays["patient_features"], device)
    losses = []
    model.eval()
    with torch.no_grad():
        cond = model.encode(surgery_idx, patient_features)
        for _ in range(cfg.val_repeats):
            x_t, t, u_t = fm.sample_conditional_path(x1)
            pred = model.velocity(x_t, t, cond)
            losses.append(float(flow_matching_loss(pred, u_t, mask).detach().cpu()))
    model.train()
    return float(np.mean(losses))


def predict_mace_proba(model: MultiTaskNet, arrays: dict, device: torch.device) -> np.ndarray:
    if arrays["patient_features"].shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    surgery_idx = fm.as_tensor(arrays["surgery_idx"], device, torch.long)
    patient_features = fm.as_tensor(arrays["patient_features"], device)
    model.eval()
    with torch.no_grad():
        logit = model.classify(model.encode(surgery_idx, patient_features))
        prob = torch.sigmoid(logit).detach().cpu().numpy()
    model.train()
    return prob.astype(np.float64)


def sample_trajectories(
    model: MultiTaskNet, arrays: dict, cfg: MultiTaskConfig, device: torch.device, x_cont_dim: int
) -> np.ndarray:
    n_patients = arrays["patient_features"].shape[0]
    n_samples = cfg.n_samples_per_patient
    total = n_patients * n_samples
    tiled = np.repeat(np.arange(n_patients), n_samples)
    surgery_idx = fm.as_tensor(arrays["surgery_idx"][tiled], device, torch.long)
    patient_features = fm.as_tensor(arrays["patient_features"][tiled], device)
    model.eval()
    with torch.no_grad():
        cond = model.encode(surgery_idx, patient_features)
        x = torch.randn(total, x_cont_dim, device=device)
        dt = 1.0 / cfg.sample_steps
        for step in range(cfg.sample_steps):
            t = torch.full((total,), step * dt, dtype=torch.float32, device=device)
            x = x + dt * model.velocity(x, t, cond)
    model.train()
    return x.detach().cpu().numpy().reshape(n_patients, n_samples, x_cont_dim)


def unstandardize(samples: np.ndarray, pre: Preprocessing) -> np.ndarray:
    return samples * pre.target_std.reshape(1, 1, -1) + pre.target_mean.reshape(1, 1, -1)


def summarize_samples(samples_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = samples_original.mean(axis=1)
    p10 = np.quantile(samples_original, 0.10, axis=1)
    p90 = np.quantile(samples_original, 0.90, axis=1)
    return mean, p10, p90


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def flow_metrics(pred_mean: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    rows = []
    for group in ["overall", "bmi", "hba1c"]:
        if group == "overall":
            dims = np.arange(observed.shape[1])
        else:
            dims = np.asarray([i for i, g in enumerate(CONT_GROUPS) if g == group], dtype=np.int64)
        obs = mask[:, dims] == 1
        n_obs = int(obs.sum())
        if n_obs == 0:
            mae, rmse = np.nan, np.nan
        else:
            diff = pred_mean[:, dims][obs] - observed[:, dims][obs]
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
        rows.append({"group": group, "n_observed": n_obs, "mae": mae, "rmse": rmse})
    return pd.DataFrame(rows)


def safe(metric_fn, *args) -> float:
    try:
        value = float(metric_fn(*args))
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def discrimination_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    prevalence = float(y_true.mean()) if y_true.size else float("nan")
    return {
        "n": int(y_true.size),
        "n_pos": int(y_true.sum()),
        "prevalence": prevalence,
        "auroc": safe(roc_auc_score, y_true, y_prob),
        "auprc": safe(average_precision_score, y_true, y_prob),
        "auprc_baseline": prevalence,
        "brier": safe(brier_score_loss, y_true, y_prob),
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        return float(thr[int(np.argmax(tpr - fpr))])
    except Exception:
        return 0.5


def threshold_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, target_spec: float) -> float:
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        feasible = (1.0 - fpr) >= target_spec
        if not feasible.any():
            return 1.0
        return float(thr[int(np.argmax(np.where(feasible, tpr, -np.inf)))])
    except Exception:
        return 0.5


def operating_point(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, label: str) -> dict:
    pred = (y_prob >= threshold).astype(np.int64)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    except Exception:
        tn = fp = fn = tp = 0
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "operating_point": label,
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def selection_score(cfg: MultiTaskConfig, flow_val: float, cls_val: float, auprc: float) -> float:
    """Lower is better."""
    combined = (0.0 if np.isnan(flow_val) else flow_val) + cfg.cls_loss_weight * (
        0.0 if np.isnan(cls_val) else cls_val
    )
    if cfg.select_metric == "flow" and not np.isnan(flow_val):
        return flow_val
    if cfg.select_metric == "auprc" and not np.isnan(auprc):
        return -auprc
    return combined


def train_model(dataset: fm.FlowDataset, cfg: MultiTaskConfig) -> dict:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    y_all = dataset.x[:, MACE_DIM].astype(np.int64)
    splits = stratified_splits_by_outcome(dataset.surgery_type, y_all, cfg)
    pre = fit_preprocessing(dataset, splits["train"])
    arrays = split_arrays(dataset, splits, pre)
    run_dir = make_run_dir(cfg.output_dir)

    pos_weight = resolve_pos_weight(cfg, arrays["train"]["y_mace"])

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {**asdict(cfg), "x_cont_dim": X_CONT_DIM, "cont_names": CONT_NAMES, "resolved_pos_weight": pos_weight},
            f,
            indent=2,
        )
    with (run_dir / "preprocessing.json").open("w", encoding="utf-8") as f:
        json.dump(pre.to_jsonable(), f, indent=2)

    model = MultiTaskNet(cfg, X_CONT_DIM, len(fm.PATIENT_FEATURES)).to(device)
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
    print(
        f"Composite MACE train prevalence={train_prev:.4f}  pos_weight={pos_weight}  "
        f"focal_gamma={cfg.focal_gamma}  cls_loss_weight={cfg.cls_loss_weight}"
    )

    for step in range(1, cfg.num_steps + 1):
        model.train()
        batch = batch_sample(arrays["train"], batch_size, rng)
        x1 = fm.as_tensor(batch["x"], device)
        mask = fm.as_tensor(batch["mask"], device)
        surgery_idx = fm.as_tensor(batch["surgery_idx"], device, torch.long)
        patient_features = fm.as_tensor(batch["patient_features"], device)
        y_mace = fm.as_tensor(batch["y_mace"], device)

        cond = model.encode(surgery_idx, patient_features)
        x_t, t, u_t = fm.sample_conditional_path(x1)
        pred = model.velocity(x_t, t, cond)
        flow_loss = flow_matching_loss(pred, u_t, mask)
        logit = model.classify(cond)
        cls_loss = classification_loss(logit, y_mace, pos_weight, cfg.focal_gamma)
        total_loss = flow_loss + cfg.cls_loss_weight * cls_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        flow_scalar = float(flow_loss.detach().cpu())
        cls_scalar = float(cls_loss.detach().cpu())
        total_scalar = float(total_loss.detach().cpu())

        should_eval = step == 1 or step % cfg.val_every == 0 or step == cfg.num_steps
        if should_eval:
            val_flow = evaluate_flow_loss(model, arrays["val"], cfg, device)
            val_prob = predict_mace_proba(model, arrays["val"], device)
            y_val = arrays["val"]["y_mace"]
            if val_prob.size:
                val_cls = float(
                    F.binary_cross_entropy(
                        torch.as_tensor(np.clip(val_prob, 1e-6, 1 - 1e-6)),
                        torch.as_tensor(y_val.astype(np.float64)),
                    )
                )
            else:
                val_cls = float("nan")
            val_metrics = discrimination_metrics(y_val.astype(np.int64), val_prob) if val_prob.size else {}
            val_auprc = val_metrics.get("auprc", float("nan"))
            val_auroc = val_metrics.get("auroc", float("nan"))
            val_brier = val_metrics.get("brier", float("nan"))

            score = selection_score(cfg, val_flow, val_cls, val_auprc)
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
                    "train_total": total_scalar,
                    "train_flow": flow_scalar,
                    "train_cls": cls_scalar,
                    "val_flow": val_flow,
                    "val_cls": val_cls,
                    "val_auroc": val_auroc,
                    "val_auprc": val_auprc,
                    "val_brier": val_brier,
                    "score": score,
                    "best_score": best_score,
                }
            )
            pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)
            print(
                f"Step {step}/{cfg.num_steps} flow={flow_scalar:.4f} cls={cls_scalar:.4f} "
                f"val_flow={val_flow:.4f} val_auroc={val_auroc:.3f} val_auprc={val_auprc:.3f} "
                f"score={score:.4f} best={best_score:.4f}@{best_step}"
            )
            if evals_since_improve >= cfg.early_stop_patience:
                early_stopped = True
                print(f"Early stopping at step {step}")
                break
        elif step % cfg.log_every == 0:
            print(
                f"Step {step}/{cfg.num_steps} flow={flow_scalar:.4f} cls={cls_scalar:.4f} "
                f"total={total_scalar:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / "model.pt")

    return finalize_and_evaluate(model, arrays, pre, cfg, run_dir, best_step, early_stopped, device)


def finalize_and_evaluate(
    model: MultiTaskNet,
    arrays: dict,
    pre: Preprocessing,
    cfg: MultiTaskConfig,
    run_dir: Path,
    best_step: int,
    early_stopped: bool,
    device: torch.device,
) -> dict:
    test_arrays = arrays["test"]

    # --- Continuous outcomes (flow) ---
    samples_std = sample_trajectories(model, test_arrays, cfg, device, X_CONT_DIM)
    samples_original = unstandardize(samples_std, pre)
    pred_mean, p10, p90 = summarize_samples(samples_original)
    flow_table = flow_metrics(pred_mean, test_arrays["original_x"], test_arrays["original_mask"])
    flow_table["split"] = "test"
    flow_table["best_step"] = best_step
    flow_table["early_stopped"] = early_stopped
    flow_table.to_csv(run_dir / "test_flow_metrics.csv", index=False)

    # --- MACE classification head ---
    val_prob = predict_mace_proba(model, arrays["val"], device)
    test_prob = predict_mace_proba(model, test_arrays, device)
    y_val = arrays["val"]["y_mace"].astype(np.int64)
    y_test = test_arrays["y_mace"].astype(np.int64)

    test_prob_cal = test_prob
    calibrated = False
    if cfg.recalibrate and y_val.size >= 10 and len(np.unique(y_val)) == 2:
        try:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(val_prob, y_val)
            test_prob_cal = iso.transform(test_prob)
            calibrated = True
        except Exception as exc:
            warnings.warn(f"Isotonic recalibration skipped: {exc}", stacklevel=2)

    mace_rows = [{"split": "test", **discrimination_metrics(y_test, test_prob)}]
    if calibrated:
        mace_rows.append({"split": "test_calibrated", **discrimination_metrics(y_test, test_prob_cal)})
    mace_metrics = pd.DataFrame(mace_rows)
    mace_metrics.to_csv(run_dir / "test_mace_metrics.csv", index=False)

    thr_youden = youden_threshold(y_val, val_prob) if y_val.size else 0.5
    thr_spec = threshold_at_specificity(y_val, val_prob, cfg.target_specificity) if y_val.size else 0.5
    operating_points = pd.DataFrame(
        [
            operating_point(y_test, test_prob, 0.5, "default_0.5"),
            operating_point(y_test, test_prob, thr_youden, "youden_val"),
            operating_point(y_test, test_prob, thr_spec, f"spec>={cfg.target_specificity:g}_val"),
        ]
    )
    operating_points.to_csv(run_dir / "test_operating_points.csv", index=False)

    # --- Per-patient predictions (continuous + MACE) ---
    predictions = pd.DataFrame({"subject_id": test_arrays["subject_ids"]})
    for dim, name in enumerate(CONT_NAMES):
        predictions[f"pred_mean_{name}"] = pred_mean[:, dim]
        predictions[f"pred_p10_{name}"] = p10[:, dim]
        predictions[f"pred_p90_{name}"] = p90[:, dim]
        observed = test_arrays["original_x"][:, dim].copy()
        observed[test_arrays["original_mask"][:, dim] == 0] = np.nan
        predictions[f"observed_{name}"] = observed
        predictions[f"observed_mask_{name}"] = test_arrays["original_mask"][:, dim]
    predictions["mace_true"] = y_test
    predictions["mace_prob"] = test_prob
    if calibrated:
        predictions["mace_prob_calibrated"] = test_prob_cal
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    print("\nContinuous-outcome (flow) test metrics:")
    print(flow_table.to_string(index=False))
    print("\nMACE classification test metrics:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(mace_metrics.to_string(index=False))
    print(f"\nSaved run artifacts to {run_dir}")
    return {
        "run_dir": run_dir,
        "flow_metrics": flow_table,
        "mace_metrics": mace_metrics,
        "operating_points": operating_points,
    }


def make_run_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def train_from_csv(csv_path: str | Path, cfg: MultiTaskConfig | None = None) -> dict:
    cfg = cfg or MultiTaskConfig()
    return train_model(fm.load_dataset_from_csv(csv_path), cfg)


def train_from_database(cfg: MultiTaskConfig | None = None) -> dict:
    cfg = cfg or MultiTaskConfig()
    return train_model(fm.load_dataset_from_database(), cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--cls-loss-weight", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--select-metric", type=str, default=None, choices=["combined", "flow", "auprc"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = MultiTaskConfig(output_dir=args.output_dir, device=args.device, seed=args.seed)
    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
    if args.cls_loss_weight is not None:
        cfg.cls_loss_weight = args.cls_loss_weight
    if args.focal_gamma is not None:
        cfg.focal_gamma = args.focal_gamma
    if args.select_metric is not None:
        cfg.select_metric = args.select_metric

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
