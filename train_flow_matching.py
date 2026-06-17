"""Standalone PyTorch flow-matching trainer for Cosmos MBSCohort data.

Default standalone behavior:
    python train_flow_matching.py

This queries MBSCohort through pyodbc, trains a concat-conditioned flow matching
model, evaluates on a held-out test set, and saves artifacts under
runs/python_flow_matching/.

The training/modeling functions can also be imported for local smoke tests from
CSV without touching Cosmos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = REPO_ROOT / "fake_data" / "fake_mbs_cohort.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "python_flow_matching"

CONNECTION_STRING = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=tcp:PROJECTS;"
    "Database=ProjectD332AFD;"
    "Trusted_Connection=yes;"
)

MBS_SQL = """
SELECT *
FROM MBSCohort
WHERE Sex NOT IN (N'#Masked', N'*Unspecified', N'Unknown', N'other')
  AND CoverageClass NOT IN (N'*Not Applicable', N'*Unspecified')
  AND BMIatEvent < WeightAtEvent
  AND BMIatEvent BETWEEN 35 AND 75
  AND (eGFRatEvent IS NULL OR eGFRatEvent >= 20)
  AND PMH_dialysis_transplant = 0
  AND (NephropathyInterval IS NULL OR NephropathyInterval >= 0)
  AND (RetinopathyInterval IS NULL OR RetinopathyInterval >= 0)
  AND (MACEinterval IS NULL OR MACEinterval >= 0)
  AND PMH_PriorMBS = 0
  AND PriorGLP1 = 0
  AND PMH_retinopathy = 0
  AND ActiveEndInterval >= 700
  AND ProcDateValue <= '2023-05-01'
"""

DAYS_PER_MONTH = 30.4375
SURGERY_TO_INDEX = {"sleeve": 0, "rnygb": 1}
CPT_TO_SURGERY = {"43775": "sleeve", "43644": "rnygb", "43846": "rnygb"}

BMI_TARGETS = [
    ("bmi_3m", "bmi", 3.0, "BMI3mPostEvent"),
    ("bmi_6m", "bmi", 6.0, "BMI6mPostEvent"),
    ("bmi_9m", "bmi", 9.0, "BMI9mPostEvent"),
    ("bmi_12m", "bmi", 12.0, "BMI12mPostEvent"),
    ("bmi_2y", "bmi", 24.0, "BMI2yPostEvent"),
    ("bmi_3y", "bmi", 36.0, "BMI3yPostEvent"),
    ("bmi_4y", "bmi", 48.0, "BMI4yPostEvent"),
    ("bmi_5y", "bmi", 60.0, "BMI5yPostEvent"),
    ("bmi_6y", "bmi", 72.0, "BMI6yPostEvent"),
]

HBA1C_TARGETS = [
    ("hba1c_12m", "hba1c", 12.0, "HbA1c12mPostEvent"),
    ("hba1c_2y", "hba1c", 24.0, "HbA1c2yPostEvent"),
    ("hba1c_3y", "hba1c", 36.0, "HbA1c3yPostEvent"),
    ("hba1c_4y", "hba1c", 48.0, "HbA1c4yPostEvent"),
    ("hba1c_5y", "hba1c", 60.0, "HbA1c5yPostEvent"),
    ("hba1c_6y", "hba1c", 72.0, "HbA1c6yPostEvent"),
]

MACE_TARGETS = [
    ("mace_ever", "mace", math.nan, "MACE/Nephropathy/Retinopathy composite"),
    ("mace_interval_months", "mace", math.nan, "Earliest MACE/Nephropathy/Retinopathy interval"),
]

TARGET_SPECS = BMI_TARGETS + HBA1C_TARGETS + MACE_TARGETS
TARGET_NAMES = [item[0] for item in TARGET_SPECS]
TARGET_GROUPS = [item[1] for item in TARGET_SPECS]
X_DIM = len(TARGET_SPECS)

PATIENT_FEATURES = [
    "age_at_surgery",
    "sex_male",
    "creatinine_at_surgery",
    "hba1c_at_surgery",
    "bmi_at_surgery",
    "insulin_status",
]
CONTINUOUS_PATIENT_FEATURES = [
    "age_at_surgery",
    "creatinine_at_surgery",
    "hba1c_at_surgery",
    "bmi_at_surgery",
]
REQUIRED_PATIENT_FEATURES = [
    "age_at_surgery",
    "sex_male",
    "creatinine_at_surgery",
    "bmi_at_surgery",
]

COMPOSITE_EVENT_COLUMNS = ["MACE", "Nephropathy", "Retinopathy"]
COMPOSITE_INTERVAL_COLUMNS = ["MACEinterval", "NephropathyInterval", "RetinopathyInterval"]


@dataclass
class TrainConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "cpu"
    seed: int = 0
    split_seed: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    time_emb_dim: int = 64
    time_scale: float = 10.0
    surgery_emb_dim: int = 8
    hidden_dim: int = 64
    num_hidden_layers: int = 2
    conditioning: str = "concat"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    num_steps: int = 6000
    batch_size: int = 64
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.005
    log_every: int = 100
    val_every: int = 250
    val_repeats: int = 8
    sample_steps: int = 50
    n_samples_per_patient: int = 50


@dataclass
class FlowDataset:
    source_label: str
    frame: pd.DataFrame
    subject_ids: np.ndarray
    surgery_type: np.ndarray
    surgery_idx: np.ndarray
    patient_features_raw: np.ndarray
    patient_feature_names: list[str]
    x: np.ndarray
    mask: np.ndarray
    target_metadata: list[dict]


@dataclass
class Preprocessing:
    target_mean: np.ndarray
    target_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    static_continuous_idx: np.ndarray
    patient_feature_names: list[str]
    target_metadata: list[dict]

    def to_jsonable(self) -> dict:
        return {
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "static_mean": self.static_mean.tolist(),
            "static_std": self.static_std.tolist(),
            "static_continuous_idx": self.static_continuous_idx.tolist(),
            "patient_feature_names": self.patient_feature_names,
            "target_metadata": self.target_metadata,
        }


def target_metadata() -> list[dict]:
    out = []
    for dim, (name, group, horizon_months, source_col) in enumerate(TARGET_SPECS):
        out.append(
            {
                "dim": dim,
                "name": name,
                "group": group,
                "horizon_months": None if math.isnan(horizon_months) else horizon_months,
                "source_column": source_col,
            }
        )
    return out


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_compatible_column(existing_names: list[str], canonical: str) -> str | None:
    if canonical in existing_names:
        return canonical
    normalized = {normalize_name(name): name for name in existing_names}
    direct = normalized.get(normalize_name(canonical))
    if direct is not None:
        return direct
    for suffix in (".y", "_y", ".mbs", "_mbs", ".x", "_x"):
        found = normalized.get(normalize_name(f"{canonical}{suffix}"))
        if found is not None:
            return found
    return None


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = required_columns()
    current_names = list(df.columns)
    rename = {}
    for canonical in required:
        matched = find_compatible_column(current_names, canonical)
        if matched is not None and matched != canonical:
            rename[matched] = canonical
            current_names[current_names.index(matched)] = canonical
    if rename:
        df = df.rename(columns=rename)
    return df


def required_columns() -> list[str]:
    cols = [
        "PatKey",
        "CptCode",
        "AgeAtEvent",
        "Sex",
        "CreatinineAtEvent",
        "HbA1cAtEvent",
        "BMIatEvent",
        "InsulinStatus",
        "PostOpGLP1",
        "GLP1Interval",
        "GLP1StartDate",
        "ProcDateValue",
        "MACE",
        "MACEinterval",
        "Nephropathy",
        "NephropathyInterval",
        "Retinopathy",
        "RetinopathyInterval",
    ]
    cols.extend(item[3] for item in BMI_TARGETS + HBA1C_TARGETS)
    return sorted(set(cols))


def assert_required_columns(df: pd.DataFrame, source_label: str) -> None:
    missing = sorted(set(required_columns()) - set(df.columns))
    if missing:
        raise ValueError(f"Required columns missing from {source_label}: {missing}")


def normalize_cpt_code(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out == "")


def map_surgery_type(cpt_code: pd.Series) -> pd.Series:
    return normalize_cpt_code(cpt_code).map(CPT_TO_SURGERY)


def encode_sex_male(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[text.isin(["male", "m"])] = 1.0
    out[text.isin(["female", "f"])] = 0.0
    return out


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def binary_event(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return ((values == 1) & values.notna()).astype("float32")


def composite_event(df: pd.DataFrame) -> pd.Series:
    events = [binary_event(df[column]).astype(bool) for column in COMPOSITE_EVENT_COLUMNS]
    return pd.concat(events, axis=1).any(axis=1).astype("float32")


def composite_interval_months(df: pd.DataFrame) -> pd.Series:
    event_matrix = pd.concat([binary_event(df[column]).astype(bool) for column in COMPOSITE_EVENT_COLUMNS], axis=1)
    interval_matrix = pd.concat([numeric(df[column]) for column in COMPOSITE_INTERVAL_COLUMNS], axis=1)
    interval_matrix.columns = COMPOSITE_INTERVAL_COLUMNS
    interval_matrix = interval_matrix.mask(~event_matrix.to_numpy())
    interval_matrix = interval_matrix.mask(interval_matrix < 0)
    earliest_days = interval_matrix.min(axis=1, skipna=True)
    return (earliest_days / DAYS_PER_MONTH).astype("float32")


def compute_glp1_start_month(df: pd.DataFrame) -> pd.Series:
    post_op = numeric(df["PostOpGLP1"]).fillna(0).eq(1)
    glp1_days = numeric(df["GLP1Interval"])

    needs_date_fallback = post_op & glp1_days.isna()
    if needs_date_fallback.any():
        glp1_start = pd.to_datetime(df.loc[needs_date_fallback, "GLP1StartDate"], errors="coerce")
        proc_date = pd.to_datetime(df.loc[needs_date_fallback, "ProcDateValue"], errors="coerce")
        fallback_days = (glp1_start - proc_date).dt.days
        glp1_days.loc[needs_date_fallback] = fallback_days

    glp1_months = glp1_days / DAYS_PER_MONTH
    glp1_months.loc[~post_op] = np.nan
    return glp1_months


def build_target_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    metadata = target_metadata()
    x = np.zeros((len(df), X_DIM), dtype=np.float32)
    mask = np.zeros((len(df), X_DIM), dtype=np.float32)
    glp1_months = compute_glp1_start_month(df)

    for dim, item in enumerate(metadata):
        group = item["group"]
        name = item["name"]
        source_col = item["source_column"]

        if name == "mace_ever":
            values = composite_event(df).to_numpy(dtype=np.float32)
            observed = np.ones(len(df), dtype=bool)
        elif name == "mace_interval_months":
            event = composite_event(df).to_numpy(dtype=np.float32)
            interval_months = composite_interval_months(df).to_numpy(dtype=np.float32)
            values = np.where(event == 1.0, interval_months, 0.0).astype(np.float32)
            observed = np.ones(len(df), dtype=bool)
        else:
            values = numeric(df[source_col]).to_numpy(dtype=np.float32)
            observed = ~np.isnan(values)
            horizon = float(item["horizon_months"])
            glp1_mask = glp1_months.notna().to_numpy() & (horizon >= glp1_months.to_numpy())
            if group in {"bmi", "hba1c"}:
                observed = observed & ~glp1_mask

        values = np.nan_to_num(values, nan=0.0)
        x[observed, dim] = values[observed]
        mask[observed, dim] = 1.0

    return x, mask, metadata


def make_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age_at_surgery": numeric(df["AgeAtEvent"]),
            "sex_male": encode_sex_male(df["Sex"]),
            "creatinine_at_surgery": numeric(df["CreatinineAtEvent"]),
            "hba1c_at_surgery": numeric(df["HbA1cAtEvent"]),
            "bmi_at_surgery": numeric(df["BMIatEvent"]),
            "insulin_status": numeric(df["InsulinStatus"]).fillna(0.0),
        },
        index=df.index,
    )


def prepare_flow_dataset(df: pd.DataFrame, source_label: str) -> FlowDataset:
    df = canonicalize_columns(df)
    assert_required_columns(df, source_label)
    df = df.copy()
    df["PatKey"] = df["PatKey"].astype("string")
    df["cpt_code_normalized"] = normalize_cpt_code(df["CptCode"])
    df["surgery_type"] = map_surgery_type(df["CptCode"])

    unknown_codes = sorted(df.loc[df["surgery_type"].isna(), "cpt_code_normalized"].dropna().unique())
    if unknown_codes:
        warnings.warn(f"Excluding rows with unrecognized CptCode values: {unknown_codes}", stacklevel=2)
    df = df.loc[df["surgery_type"].notna()].copy()

    if df["PatKey"].duplicated().any():
        examples = df.loc[df["PatKey"].duplicated(), "PatKey"].head(10).tolist()
        raise ValueError(f"Duplicate PatKey rows found in wide patient input: {examples}")

    event = composite_event(df)
    event_interval_months = composite_interval_months(df)
    bad_event_interval = event.eq(1) & event_interval_months.isna()
    if bad_event_interval.any():
        warnings.warn(
            "Dropping "
            f"{int(bad_event_interval.sum())} rows with a composite MACE/nephropathy/retinopathy "
            "event and no valid nonnegative event interval.",
            stacklevel=2,
        )
        df = df.loc[~bad_event_interval].copy()

    post_op_glp1 = numeric(df["PostOpGLP1"]).fillna(0).eq(1)
    glp1_months = compute_glp1_start_month(df)
    bad_glp1 = post_op_glp1 & (glp1_months.isna() | (glp1_months < 0))
    if bad_glp1.any():
        warnings.warn(
            f"Dropping {int(bad_glp1.sum())} rows with PostOpGLP1 == 1 and unavailable/negative GLP1 start interval.",
            stacklevel=2,
        )
        df = df.loc[~bad_glp1].copy()

    patient_features = make_patient_features(df)
    complete_conditioning = patient_features[REQUIRED_PATIENT_FEATURES].notna().all(axis=1)
    if (~complete_conditioning).any():
        warnings.warn(
            f"Dropping {int((~complete_conditioning).sum())} rows with missing required core conditioning fields.",
            stacklevel=2,
        )
        df = df.loc[complete_conditioning].copy()
        patient_features = patient_features.loc[complete_conditioning].copy()

    x, mask, metadata = build_target_matrix(df)
    surgery_idx = df["surgery_type"].map(SURGERY_TO_INDEX).to_numpy(dtype=np.int64)

    return FlowDataset(
        source_label=source_label,
        frame=df,
        subject_ids=df["PatKey"].astype(str).to_numpy(),
        surgery_type=df["surgery_type"].astype(str).to_numpy(),
        surgery_idx=surgery_idx,
        patient_features_raw=patient_features[PATIENT_FEATURES].to_numpy(dtype=np.float32),
        patient_feature_names=PATIENT_FEATURES.copy(),
        x=x,
        mask=mask,
        target_metadata=metadata,
    )


def load_mbs_from_database() -> pd.DataFrame:
    try:
        import pyodbc  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyodbc is required for standalone Cosmos DB execution, but it is not installed. "
            "Install/enable pyodbc in the Cosmos Python environment, or import this script and "
            "call train_from_csv(...) for local CSV smoke testing."
        ) from exc

    try:
        with pyodbc.connect(CONNECTION_STRING, timeout=1000) as connection:
            return pd.read_sql(MBS_SQL, connection).drop_duplicates()
    except Exception as exc:
        raise RuntimeError(
            "Unable to query Cosmos MBSCohort through pyodbc. Check that the SQL Server "
            "ODBC driver is available, Cosmos database access is active, and the connection "
            "string constants near the top of this script match the environment."
        ) from exc


def load_dataset_from_database() -> FlowDataset:
    return prepare_flow_dataset(load_mbs_from_database(), "Cosmos MBSCohort")


def load_dataset_from_csv(csv_path: str | Path) -> FlowDataset:
    path = Path(csv_path)
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    return prepare_flow_dataset(df, str(path))


def make_stratified_splits(dataset: FlowDataset, cfg: TrainConfig) -> dict[str, np.ndarray]:
    if not np.isclose(cfg.train_frac + cfg.val_frac + cfg.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    rng = np.random.default_rng(cfg.split_seed)
    train_parts, val_parts, test_parts = [], [], []
    for surgery in sorted(set(dataset.surgery_type.tolist())):
        idx = np.where(dataset.surgery_type == surgery)[0]
        rng.shuffle(idx)
        n_train = int(np.floor(len(idx) * cfg.train_frac))
        n_val = int(np.floor(len(idx) * cfg.val_frac))
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train : n_train + n_val])
        test_parts.append(idx[n_train + n_val :])
    train_idx = np.concatenate(train_parts).astype(np.int64)
    val_idx = np.concatenate(val_parts).astype(np.int64)
    test_idx = np.concatenate(test_parts).astype(np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def fit_preprocessing(dataset: FlowDataset, train_idx: np.ndarray) -> Preprocessing:
    x_train = dataset.x[train_idx]
    mask_train = dataset.mask[train_idx]
    observed_count = mask_train.sum(axis=0)
    if np.any(observed_count == 0):
        empty = [TARGET_NAMES[i] for i in np.where(observed_count == 0)[0]]
        raise ValueError(f"Target dimensions have no train observations: {empty}")
    target_mean = (x_train * mask_train).sum(axis=0) / observed_count
    target_var = (((x_train - target_mean) ** 2) * mask_train).sum(axis=0) / observed_count
    target_std = np.sqrt(target_var)
    target_std = np.where((target_std < 1e-8) | np.isnan(target_std), 1.0, target_std).astype(np.float32)

    raw = dataset.patient_features_raw[train_idx].astype(np.float32)
    static_mean = np.zeros(raw.shape[1], dtype=np.float32)
    static_std = np.ones(raw.shape[1], dtype=np.float32)
    continuous_idx = np.asarray(
        [PATIENT_FEATURES.index(name) for name in CONTINUOUS_PATIENT_FEATURES],
        dtype=np.int64,
    )
    static_mean[continuous_idx] = np.nanmean(raw[:, continuous_idx], axis=0)
    static_mean[continuous_idx] = np.where(np.isnan(static_mean[continuous_idx]), 0.0, static_mean[continuous_idx])
    std = np.nanstd(raw[:, continuous_idx], axis=0)
    static_std[continuous_idx] = np.where((std < 1e-8) | np.isnan(std), 1.0, std)
    return Preprocessing(
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
        static_mean=static_mean,
        static_std=static_std,
        static_continuous_idx=continuous_idx,
        patient_feature_names=PATIENT_FEATURES.copy(),
        target_metadata=dataset.target_metadata,
    )


def transform_targets(x: np.ndarray, mask: np.ndarray, preprocessing: Preprocessing) -> np.ndarray:
    return (((x - preprocessing.target_mean) / preprocessing.target_std) * mask).astype(np.float32)


def transform_patient_features(x: np.ndarray, preprocessing: Preprocessing) -> np.ndarray:
    out = x.copy().astype(np.float32)
    idx = preprocessing.static_continuous_idx
    missing_continuous = np.isnan(out[:, idx])
    if missing_continuous.any():
        out[:, idx] = np.where(missing_continuous, preprocessing.static_mean[idx], out[:, idx])
    missing_other = np.isnan(out)
    if missing_other.any():
        out = np.where(missing_other, 0.0, out)
    out[:, idx] = (out[:, idx] - preprocessing.static_mean[idx]) / preprocessing.static_std[idx]
    return out


def split_arrays(dataset: FlowDataset, splits: dict[str, np.ndarray], preprocessing: Preprocessing) -> dict[str, dict]:
    x_std = transform_targets(dataset.x, dataset.mask, preprocessing)
    p_std = transform_patient_features(dataset.patient_features_raw, preprocessing)

    out = {}
    for name, idx in splits.items():
        out[name] = {
            "x": x_std[idx],
            "mask": dataset.mask[idx],
            "surgery_idx": dataset.surgery_idx[idx],
            "patient_features": p_std[idx],
            "subject_ids": dataset.subject_ids[idx],
            "original_x": dataset.x[idx],
            "original_mask": dataset.mask[idx],
        }
    return out


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, time_scale: float) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().view(-1, 1) * time_scale * freqs.view(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def modulate_layer_norm(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def zero_init_linear(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class VectorFieldNet(nn.Module):
    def __init__(self, cfg: TrainConfig, x_dim: int, patient_feature_dim: int, num_surgery_types: int = 2):
        super().__init__()
        if cfg.time_emb_dim % 2 != 0:
            raise ValueError("time_emb_dim must be even")
        conditioning = cfg.conditioning.lower()
        if conditioning not in {"concat", "adaln"}:
            raise ValueError(f"Unknown conditioning style: {cfg.conditioning!r}")
        self.x_dim = x_dim
        self.time_emb_dim = cfg.time_emb_dim
        self.time_scale = cfg.time_scale
        self.conditioning = conditioning
        self.surgery_emb = nn.Embedding(num_surgery_types, cfg.surgery_emb_dim)

        cond_dim = cfg.time_emb_dim + cfg.surgery_emb_dim + patient_feature_dim
        self.cond_dim = cond_dim
        if conditioning == "concat":
            self._init_concat_layers(cfg, x_dim, cond_dim)
        else:
            self._init_adaln_layers(cfg, x_dim, cond_dim)

    def _init_concat_layers(self, cfg: TrainConfig, x_dim: int, cond_dim: int) -> None:
        in_dim = x_dim + cond_dim
        layers = []
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            in_dim = cfg.hidden_dim + cond_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_dim, x_dim)

    def _init_adaln_layers(self, cfg: TrainConfig, x_dim: int, cond_dim: int) -> None:
        self.cond_in = nn.Linear(cond_dim, cfg.hidden_dim)
        self.cond_out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.x_in = nn.Linear(x_dim, cfg.hidden_dim)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False) for _ in range(cfg.num_hidden_layers)]
        )
        self.adaln_modulators = nn.ModuleList(
            [zero_init_linear(nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim)) for _ in range(cfg.num_hidden_layers)]
        )
        self.block_layers = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.num_hidden_layers)]
        )
        self.out_adaln = zero_init_linear(nn.Linear(cfg.hidden_dim, 2 * cfg.hidden_dim))
        self.out_ln = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False)
        self.out = zero_init_linear(nn.Linear(cfg.hidden_dim, x_dim))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, surgery_idx: torch.Tensor, patient_features: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim, self.time_scale)
        surgery_emb = self.surgery_emb(surgery_idx.long())
        cond = torch.cat([t_emb, surgery_emb, patient_features], dim=-1)
        if self.conditioning == "adaln":
            return self.forward_adaln(x_t, cond)
        return self.forward_concat(x_t, cond)

    def forward_concat(self, x_t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_t, cond], dim=-1)
        for layer in self.hidden_layers:
            h = F.silu(layer(h))
            h = torch.cat([h, cond], dim=-1)
        return self.out(h)

    def forward_adaln(self, x_t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_h = F.silu(self.cond_in(cond))
        cond_h = self.cond_out(cond_h)

        h = F.silu(self.x_in(x_t))
        for layer_norm, modulator, block_layer in zip(self.layer_norms, self.adaln_modulators, self.block_layers):
            shift, scale, gate = modulator(cond_h).chunk(3, dim=-1)
            y = layer_norm(h)
            y = modulate_layer_norm(y, shift, scale)
            y = F.silu(block_layer(y))
            h = h + gate * y

        shift, scale = self.out_adaln(cond_h).chunk(2, dim=-1)
        h = modulate_layer_norm(self.out_ln(h), shift, scale)
        return self.out(h)


def as_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def sample_conditional_path(x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = x1.shape[0]
    t = torch.rand(batch, device=x1.device)
    x0 = torch.randn_like(x1)
    t_expanded = t[:, None]
    x_t = (1.0 - t_expanded) * x0 + t_expanded * x1
    u_t = x1 - x0
    return x_t, t, u_t


def flow_matching_loss(
    model: VectorFieldNet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    surgery_idx: torch.Tensor,
    patient_features: torch.Tensor,
    u_t: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred = model(x_t, t, surgery_idx, patient_features)
    return (mask * (pred - u_t).pow(2)).sum() / (mask.sum() + 1e-8)


def batch_sample(arrays: dict, batch_size: int, rng: np.random.Generator) -> dict:
    n = arrays["x"].shape[0]
    replace = batch_size > n
    idx = rng.choice(n, size=batch_size, replace=replace)
    return {key: value[idx] for key, value in arrays.items() if isinstance(value, np.ndarray)}


def evaluate_flow_loss(model: VectorFieldNet, arrays: dict, cfg: TrainConfig, device: torch.device) -> float:
    if arrays["x"].shape[0] == 0:
        return float("nan")
    x1 = as_tensor(arrays["x"], device)
    mask = as_tensor(arrays["mask"], device)
    surgery_idx = as_tensor(arrays["surgery_idx"], device, torch.long)
    patient_features = as_tensor(arrays["patient_features"], device)
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(cfg.val_repeats):
            x_t, t, u_t = sample_conditional_path(x1)
            loss = flow_matching_loss(model, x_t, t, surgery_idx, patient_features, u_t, mask)
            losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses))


def sample_trajectories(
    model: VectorFieldNet,
    arrays: dict,
    cfg: TrainConfig,
    device: torch.device,
    x_dim: int,
) -> np.ndarray:
    n_patients = arrays["patient_features"].shape[0]
    n_samples = cfg.n_samples_per_patient
    total = n_patients * n_samples
    tiled = np.repeat(np.arange(n_patients), n_samples)
    surgery_idx = as_tensor(arrays["surgery_idx"][tiled], device, torch.long)
    patient_features = as_tensor(arrays["patient_features"][tiled], device)
    x = torch.randn(total, x_dim, device=device)
    dt = 1.0 / cfg.sample_steps
    model.eval()
    with torch.no_grad():
        for step in range(cfg.sample_steps):
            t = torch.full((total,), step * dt, dtype=torch.float32, device=device)
            x = x + dt * model(x, t, surgery_idx, patient_features)
    return x.detach().cpu().numpy().reshape(n_patients, n_samples, x_dim)


def unstandardize(samples: np.ndarray, preprocessing: Preprocessing) -> np.ndarray:
    return samples * preprocessing.target_std.reshape(1, 1, -1) + preprocessing.target_mean.reshape(1, 1, -1)


def summarize_samples(samples_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = samples_original.mean(axis=1)
    p10 = np.quantile(samples_original, 0.10, axis=1)
    p90 = np.quantile(samples_original, 0.90, axis=1)
    for arr in (mean, p10, p90):
        arr[:, TARGET_NAMES.index("mace_ever")] = np.clip(arr[:, TARGET_NAMES.index("mace_ever")], 0.0, 1.0)
        arr[:, TARGET_NAMES.index("mace_interval_months")] = np.maximum(
            arr[:, TARGET_NAMES.index("mace_interval_months")], 0.0
        )
    return mean, p10, p90


def compute_metrics(pred_mean: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    rows = []
    for group in ["overall", "bmi", "hba1c", "mace"]:
        if group == "overall":
            dims = np.arange(observed.shape[1])
        else:
            dims = np.asarray([i for i, g in enumerate(TARGET_GROUPS) if g == group], dtype=np.int64)
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


def prediction_frame(subject_ids: np.ndarray, pred_mean: np.ndarray, p10: np.ndarray, p90: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"subject_id": subject_ids})
    for dim, name in enumerate(TARGET_NAMES):
        out[f"pred_mean_{name}"] = pred_mean[:, dim]
        out[f"pred_p10_{name}"] = p10[:, dim]
        out[f"pred_p90_{name}"] = p90[:, dim]
        obs = observed[:, dim].copy()
        obs[mask[:, dim] == 0] = np.nan
        out[f"observed_{name}"] = obs
        out[f"observed_mask_{name}"] = mask[:, dim]
    return out


def make_run_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_model_state(run_dir: Path, state: dict[str, torch.Tensor], step: int, best_score: float) -> None:
    tmp_path = run_dir / "model.pt.tmp"
    torch.save(state, tmp_path)
    tmp_path.replace(run_dir / "model.pt")
    with (run_dir / "checkpoint.json").open("w", encoding="utf-8") as f:
        json.dump({"model_path": "model.pt", "step": step, "best_score": best_score}, f, indent=2)


def train_model(dataset: FlowDataset, cfg: TrainConfig) -> dict:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    splits = make_stratified_splits(dataset, cfg)
    preprocessing = fit_preprocessing(dataset, splits["train"])
    arrays = split_arrays(dataset, splits, preprocessing)
    run_dir = make_run_dir(cfg.output_dir)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "x_dim": X_DIM, "target_names": TARGET_NAMES}, f, indent=2)
    with (run_dir / "preprocessing.json").open("w", encoding="utf-8") as f:
        json.dump(preprocessing.to_jsonable(), f, indent=2)

    model = VectorFieldNet(cfg, X_DIM, len(PATIENT_FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    batch_size = min(cfg.batch_size, max(1, arrays["train"]["x"].shape[0]))
    logs = []
    best_score = float("inf")
    best_step = -1
    best_state = None
    evals_since_improve = 0
    early_stopped = False

    print(
        f"Patients: {len(dataset.subject_ids)} "
        f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}, x_dim={X_DIM})"
    )
    print(f"Training for up to {cfg.num_steps} steps with batch_size={batch_size}")

    for step in range(1, cfg.num_steps + 1):
        model.train()
        batch = batch_sample(arrays["train"], batch_size, rng)
        x1 = as_tensor(batch["x"], device)
        mask = as_tensor(batch["mask"], device)
        surgery_idx = as_tensor(batch["surgery_idx"], device, torch.long)
        patient_features = as_tensor(batch["patient_features"], device)
        x_t, t, u_t = sample_conditional_path(x1)

        optimizer.zero_grad()
        loss = flow_matching_loss(model, x_t, t, surgery_idx, patient_features, u_t, mask)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu())

        should_eval = step == 1 or step % cfg.val_every == 0 or step == cfg.num_steps
        if should_eval:
            val_loss = evaluate_flow_loss(model, arrays["val"], cfg, device)
            score = train_loss if np.isnan(val_loss) else val_loss
            improved = score < best_score - cfg.early_stop_min_delta
            if improved:
                best_score = score
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                save_model_state(run_dir, best_state, best_step, best_score)
                evals_since_improve = 0
            else:
                evals_since_improve += 1
            logs.append({"step": step, "train_loss": train_loss, "val_loss": val_loss, "best_val": best_score})
            pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)
            print(
                f"Step {step}/{cfg.num_steps} train={train_loss:.4f} "
                f"val={val_loss:.4f} best={best_score:.4f}@{best_step}"
            )
            if not np.isnan(val_loss) and evals_since_improve >= cfg.early_stop_patience:
                early_stopped = True
                print(f"Early stopping at step {step}")
                break
        elif step % cfg.log_every == 0:
            print(f"Step {step}/{cfg.num_steps} train={train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        save_model_state(run_dir, best_state, best_step, best_score)
    else:
        save_model_state(run_dir, {key: value.detach().cpu() for key, value in model.state_dict().items()}, -1, float("nan"))

    test_arrays = arrays["test"]
    samples_std = sample_trajectories(model, test_arrays, cfg, device, X_DIM)
    samples_original = unstandardize(samples_std, preprocessing)
    pred_mean, p10, p90 = summarize_samples(samples_original)
    metrics = compute_metrics(pred_mean, test_arrays["original_x"], test_arrays["original_mask"])
    metrics["split"] = "test"
    metrics["best_step"] = best_step
    metrics["early_stopped"] = early_stopped
    metrics.to_csv(run_dir / "test_metrics.csv", index=False)
    prediction_frame(
        test_arrays["subject_ids"],
        pred_mean,
        p10,
        p90,
        test_arrays["original_x"],
        test_arrays["original_mask"],
    ).to_csv(run_dir / "test_predictions.csv", index=False)
    print(f"Saved run artifacts to {run_dir}")
    return {"run_dir": run_dir, "metrics": metrics, "dataset": dataset, "preprocessing": preprocessing}


def train_from_csv(csv_path: str | Path = DEFAULT_CSV_PATH, cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    return train_model(load_dataset_from_csv(csv_path), cfg)


def train_from_database(cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    return train_model(load_dataset_from_database(), cfg)


if __name__ == "__main__":
    try:
        train_from_database()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
