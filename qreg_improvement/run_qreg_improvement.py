#!/usr/bin/env python3
"""Single-file five-year trajectory and MACE improvement study.

This runner is intentionally self-contained at runtime. It reuses the repository's
canonical MBSCohort loader and tested score primitives, but requires no literature
PDFs or external configuration files. Published targets and their provenance are
frozen below as an audited registry. They are reference values, never locally paired
comparators and never representations of reproduced published models.

Examples
--------
python qreg_improvement/run_qreg_improvement.py --run full \
  --output-dir qreg_improvement/results/full_run
python qreg_improvement/run_qreg_improvement.py --run smoke
python qreg_improvement/run_qreg_improvement.py --run plot-only \
  --output-dir qreg_improvement/results/full_run
python qreg_improvement/run_qreg_improvement.py --run self-test
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import pickle
import sys
import tempfile
import time
import traceback
import warnings
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = SCRIPT_DIR / ".cache"
for cache_dir in (CACHE_ROOT, CACHE_ROOT / "matplotlib", CACHE_ROOT / "xdg",
                  CACHE_ROOT / "torch", CACHE_ROOT / "joblib"):
    cache_dir.mkdir(parents=True, exist_ok=True)
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(CACHE_ROOT / "joblib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_flow_matching as fm
import baselines_trajectory as bt
import distributional_metrics as dm
import gbm_mace_baseline as gb


# Frozen, audited extraction. Runtime PDF access is deliberately unnecessary.
LITERATURE = {
    "registry_version": "2026-07-16",
    "bjs_2026": {
        "role": "published reference only; fitted model not reproduced",
        "bmi": {"pooled_rmse": 1.11, "pooled_mae": 0.62,
                "rmse_by_month": {3: 2.36, 6: 1.31, 12: 0.91, 24: 0.62,
                                  36: 0.78, 48: 0.92, 60: 1.01}},
        "remission": {"auroc": 0.99, "macro_f1": 0.88, "precision": 0.87,
                      "recall": 0.88, "log_loss": 0.07},
        "caveat": "Reported MAPE is excluded because its scale is inconsistent with MAE and RMSE.",
    },
    "sophia": {
        "role": "published reference only; CART model not trained or reproduced",
        "bmi": {"rmse_by_month": {12: 3.7, 24: 4.2, 60: 4.7}, "mad_60": 2.8,
                "normalized_rmse_pct": {12: 12.0, 24: 14.0, 60: 14.7},
                "rmse_60_by_procedure": {"rnygb": 4.5, "sleeve": 5.7}},
    },
}
LITERATURE_SHA256 = "a8424371f8e3202c130733fe01378ecadcd4076e09259946704ea0a56167c17b"

BMI = [("bmi_3m", 3), ("bmi_6m", 6), ("bmi_9m", 9), ("bmi_12m", 12),
       ("bmi_2y", 24), ("bmi_3y", 36), ("bmi_4y", 48), ("bmi_5y", 60)]
HBA1C = [("hba1c_12m", 12), ("hba1c_2y", 24), ("hba1c_3y", 36),
         ("hba1c_4y", 48), ("hba1c_5y", 60)]
TARGETS = BMI + HBA1C
ORIGINS = (0, 3, 6, 9, 12, 24, 36, 48)
QUANTILES = np.array((0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975))
COVERAGES = (0.50, 0.80, 0.90, 0.95)
POSTOP_TOKENS = ("postevent", "postop", "interval", "mace", "nephropathy", "retinopathy")
SEARCH_CAPS = {"risk": 60, "conventional": 50, "tfm": 30}
PRIMARY_BASELINE = "current_qreg_copula"
ROLLING_BASELINES = ("persistence", "population_trajectory")
_CORRELATION_CACHE: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class Settings:
    mode: str
    output_dir: str
    csv: str | None
    seed: int = 2026
    resume: bool = True
    n_samples: int = 101
    max_configs: int = 140
    device: str = "cpu"


@dataclass
class Study:
    dataset: Any
    split: dict[str, np.ndarray]
    X: np.ndarray
    Y: np.ndarray
    M: np.ndarray
    names: list[str]
    groups: list[str]
    months: np.ndarray
    target_dims: list[int]
    feature_names: list[str]
    input_hash: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def write_prediction_table(path: Path, frame: pd.DataFrame, require_parquet: bool) -> dict[str, Any]:
    """Prefer Parquet; retain a CSV plus an explicit status when its engine is absent."""
    try:
        tmp = path.with_suffix(".tmp.parquet")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return {"status": "ok", "path": str(path), "format": "parquet"}
    except (ImportError, ModuleNotFoundError) as exc:
        fallback = path.with_suffix(".csv")
        atomic_csv(fallback, frame)
        status = {"status": "dependency_missing", "dependency": "pyarrow or fastparquet",
                  "fallback": str(fallback), "error": str(exc)}
        if require_parquet:
            warnings.warn("Parquet engine unavailable; CSV retained and limitation recorded.", stacklevel=2)
        return status


class Run:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.out = Path(cfg.output_dir).resolve()
        for name in ("figures", "metrics", "predictions", "samples", "models", "search", "logs"):
            (self.out / name).mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "run_state.json"
        self.state = self.read_json(self.state_path, {"status": "initializing", "stages": {}, "errors": []})
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.FileHandler(self.out / "logs" / "study.log"), logging.StreamHandler()],
                            force=True)
        self.log = logging.getLogger("qreg-improvement")

    @staticmethod
    def read_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def done(self, stage: str, key: str, files: Iterable[Path] = ()) -> bool:
        row = self.state.get("stages", {}).get(stage, {})
        return self.cfg.resume and row.get("status") == "complete" and row.get("hash") == key and all(p.exists() for p in files)

    def mark(self, stage: str, status: str, key: str = "", **details: Any) -> None:
        self.state.setdefault("stages", {})[stage] = {
            "status": status, "hash": key, "time": time.time(), **details,
        }
        self.state["status"] = "failed" if status == "failed" else "running"
        atomic_json(self.state_path, self.state)
        self.log.info("stage=%s status=%s",stage,status)
        progress_figure(self)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def library_manifest() -> dict[str, bool]:
    return {name: module_available(name) for name in ("torch", "xgboost", "catboost", "lightgbm", "pyarrow", "fastparquet")}


def load_study(cfg: Settings) -> Study:
    dataset = fm.load_dataset_from_csv(cfg.csv) if cfg.csv else fm.load_dataset_from_database()
    by_name = {m["name"]: int(m["dim"]) for m in dataset.target_metadata}
    missing = [name for name, _ in TARGETS if name not in by_name]
    if missing:
        raise ValueError(f"Missing required five-year targets: {missing}")
    dims = [by_name[name] for name, _ in TARGETS]
    names = [name for name, _ in TARGETS]
    groups = ["bmi" if name.startswith("bmi") else "hba1c" for name in names]
    months = np.array([month for _, month in TARGETS], dtype=int)
    split_cfg = fm.TrainConfig(split_seed=cfg.seed, train_frac=0.70, val_frac=0.15, test_frac=0.15)
    split = fm.make_temporal_splits(dataset, split_cfg)
    id_sets = {key: set(dataset.subject_ids[idx].tolist()) for key, idx in split.items()}
    if any(id_sets[a] & id_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("Temporal split contains patient overlap")
    feature_names = list(dataset.patient_feature_names) + ["surgery_idx"]
    if any(any(token in name.lower() for token in POSTOP_TOKENS) for name in feature_names):
        raise AssertionError("Pre-op feature roster contains postoperative data")
    X = np.column_stack([dataset.patient_features_raw.astype(float), dataset.surgery_idx.astype(float)])
    Y = dataset.x[:, dims].astype(float)
    M = dataset.mask[:, dims].astype(bool)
    dates = pd.to_datetime(dataset.frame["ProcDateValue"], errors="coerce").astype(str).tolist()
    input_hash = digest({"ids": dataset.subject_ids.tolist(), "dates": dates,
                         "X": np.nan_to_num(X, nan=-9999).round(7).tolist(),
                         "Y": np.where(M, Y, np.nan).round(7).tolist()})
    return Study(dataset, split, X, Y, M, names, groups, months, dims, feature_names, input_hash)


def nearest_correlation(matrix: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    a = np.asarray(matrix, float)
    a = np.where(np.isfinite(a), a, 0.0)
    a = (a + a.T) / 2
    np.fill_diagonal(a, 1.0)
    values, vectors = np.linalg.eigh(a)
    a = (vectors * np.maximum(values, floor)) @ vectors.T
    scale = np.sqrt(np.diag(a))
    a = a / np.outer(scale, scale)
    np.fill_diagonal(a, 1.0)
    return a


def residual_correlation(study: Study) -> np.ndarray:
    """Rank-Gaussian correlation of cross-fitted qReg median residuals."""
    if study.input_hash in _CORRELATION_CACHE:
        return _CORRELATION_CACHE[study.input_hash].copy()
    residual = np.full_like(study.Y, np.nan)
    tr = study.split["train"]
    folds = KFold(n_splits=min(5, max(2, len(tr) // 10)), shuffle=True, random_state=2026)
    for h in range(study.Y.shape[1]):
        observed = tr[study.M[tr, h]]
        prediction = np.full(len(tr), np.median(study.Y[observed, h]) if len(observed) else 0.0)
        for fit, hold in folds.split(tr):
            fit_ids = tr[fit]; fit_ids = fit_ids[study.M[fit_ids, h]]
            if len(fit_ids) >= max(10, study.X.shape[1] + 2):
                model = quantile_estimator("current_qreg", 0.5, {"alpha": 0.01}, 2026 + h)
                model.fit(study.X[fit_ids], study.Y[fit_ids, h])
                prediction[hold] = model.predict(study.X[tr[hold]])
        residual[tr[study.M[tr, h]], h] = study.Y[tr[study.M[tr, h]], h] - prediction[study.M[tr, h]]
        finite = np.isfinite(residual[:, h])
        if finite.sum() > 1:
            ranks = stats.rankdata(residual[finite, h]) / (finite.sum() + 1)
            residual[finite, h] = stats.norm.ppf(ranks)
    corr = np.eye(study.Y.shape[1])
    for i in range(corr.shape[0]):
        for j in range(i):
            keep = np.isfinite(residual[:, i]) & np.isfinite(residual[:, j])
            value = np.corrcoef(residual[keep, i], residual[keep, j])[0, 1] if keep.sum() >= 8 else 0.0
            corr[i, j] = corr[j, i] = value
    corr = nearest_correlation(corr)
    _CORRELATION_CACHE[study.input_hash] = corr
    return corr.copy()


def qgrid_to_samples(qpred: np.ndarray, n_samples: int, corr: np.ndarray, seed: int) -> np.ndarray:
    """Map marginal quantile functions through a rank-Gaussian copula."""
    rng = np.random.default_rng(seed)
    n, _, k = qpred.shape
    z = rng.multivariate_normal(np.zeros(k), corr, size=(n, n_samples))
    u = np.clip(stats.norm.cdf(z), QUANTILES[0], QUANTILES[-1])
    out = np.empty((n, n_samples, k), dtype=np.float32)
    for i in range(n):
        for h in range(k):
            out[i, :, h] = np.interp(u[i, :, h], QUANTILES, qpred[i, :, h])
    return out


def pooled_rows(study: Study, patients: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, h = np.where(study.M[patients])
    patient = patients[p]
    static = study.X[patient]
    month = study.months[h, None] / 60.0
    group = np.array([study.groups[x] == "hba1c" for x in h], float)[:, None]
    design = np.column_stack([static, month, group, month * group])
    return design, study.Y[patient, h], h


def prediction_design(study: Study, patients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.repeat(patients, len(study.names))
    h = np.tile(np.arange(len(study.names)), len(patients))
    month = study.months[h, None] / 60.0
    group = np.array([study.groups[x] == "hba1c" for x in h], float)[:, None]
    return np.column_stack([study.X[p], month, group, month * group]), h


def quantile_estimator(family: str, q: float, params: dict[str, Any], seed: int):
    if family == "current_qreg":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("model", QuantileRegressor(quantile=q, alpha=params.get("alpha", 0.01), solver="highs"))])
    if family == "pooled_spline_qreg":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("spline", SplineTransformer(n_knots=params.get("n_knots", 4), degree=2)),
                         ("model", QuantileRegressor(quantile=q, alpha=params.get("alpha", 0.02), solver="highs"))])
    if family == "pooled_xgboost_qreg":
        import xgboost as xgb
        return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q,
                                n_estimators=params.get("n_estimators", 160), max_depth=params.get("max_depth", 4),
                                learning_rate=params.get("learning_rate", 0.05), subsample=0.85,
                                colsample_bytree=0.85, tree_method="hist", n_jobs=1, random_state=seed)
    if family == "pooled_catboost_qreg":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(loss_function=f"Quantile:alpha={q}", iterations=params.get("iterations", 180),
                                 depth=params.get("depth", 6), learning_rate=params.get("learning_rate", 0.05),
                                 random_seed=seed, verbose=False, allow_writing_files=False)
    raise ValueError(f"Unknown quantile family: {family}")


def fit_quantile_candidate(study: Study, family: str, params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Fit only on training patients and predict validation and sealed test patients."""
    outputs = {part: np.full((len(study.split[part]), len(QUANTILES), len(study.names)), np.nan)
               for part in ("val", "test")}
    models: list[Any] = []
    if family == "current_qreg":
        for h in range(len(study.names)):
            tr = study.split["train"][study.M[study.split["train"], h]]
            fallback = np.quantile(study.Y[tr, h], QUANTILES) if len(tr) else np.full(len(QUANTILES), np.nan)
            h_models = []
            for qi, q in enumerate(QUANTILES):
                if len(tr) < max(10, study.X.shape[1] + 2):
                    model = None
                else:
                    model = quantile_estimator(family, float(q), params, seed + qi)
                    model.fit(study.X[tr], study.Y[tr, h])
                h_models.append(model)
                for part in outputs:
                    outputs[part][:, qi, h] = fallback[qi] if model is None else model.predict(study.X[study.split[part]])
            models.append(h_models)
    else:
        train_x, train_y, _ = pooled_rows(study, study.split["train"])
        pred = {part: prediction_design(study, study.split[part]) for part in outputs}
        fallback = np.quantile(train_y, QUANTILES)
        for qi, q in enumerate(QUANTILES):
            model = quantile_estimator(family, float(q), params, seed + qi)
            try:
                model.fit(train_x, train_y)
            except Exception:
                model = None
            models.append(model)
            for part in outputs:
                values = np.full(len(pred[part][1]), fallback[qi]) if model is None else model.predict(pred[part][0])
                outputs[part][:, qi, :] = values.reshape(len(study.split[part]), len(study.names))
    for part in outputs:
        outputs[part] = np.sort(outputs[part], axis=1)
    return {"family": family, "params": params, "models": models, **outputs}


def mean_crps(qpred: np.ndarray, study: Study, part: str, corr: np.ndarray, seed: int) -> float:
    samples = qgrid_to_samples(qpred, 41, corr, seed)
    scores = []
    idx = study.split[part]
    for h in range(len(study.names)):
        keep = study.M[idx, h]
        if keep.any():
            scores.extend(bt.crps_ensemble(samples[keep, :, h], study.Y[idx[keep], h]).tolist())
    return float(np.mean(scores)) if scores else float("nan")


def validation_objectives(qpred:np.ndarray,study:Study,corr:np.ndarray,seed:int)->tuple[float,float,float]:
    samples=qgrid_to_samples(qpred,41,corr,seed);idx=study.split["val"];crps_values=[];sq=[];coverage_errors=[]
    for h in range(len(study.names)):
        keep=study.M[idx,h]
        if not keep.any():continue
        y=study.Y[idx[keep],h];s=samples[keep,:,h];crps_values.extend(bt.crps_ensemble(s,y));sq.extend((s.mean(1)-y)**2)
        lo,hi=np.quantile(s,[.05,.95],axis=1);coverage_errors.append(abs(np.mean((y>=lo)&(y<=hi))-.9))
    return (float(np.mean(crps_values)) if crps_values else np.nan,
            math.sqrt(float(np.mean(sq))) if sq else np.nan,
            float(np.mean(coverage_errors)) if coverage_errors else np.nan)


def conventional_search(run: Run, study: Study, corr: np.ndarray) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    smoke = run.cfg.mode == "smoke"
    grids = {
        "current_qreg": [{"alpha": 0.01}],
        "pooled_spline_qreg": [{"n_knots": n, "alpha": a} for n in ((3,) if smoke else (3, 4, 5)) for a in ((0.02,) if smoke else (0.005, 0.02))],
        "pooled_xgboost_qreg": ([{"n_estimators": 35, "max_depth": 3, "learning_rate": 0.08}] if smoke else
            [{"n_estimators": n, "max_depth": d, "learning_rate": lr} for n in (120, 240) for d in (3, 5) for lr in (0.03, 0.08)]),
        "pooled_catboost_qreg": ([{"iterations": 40, "depth": 4, "learning_rate": 0.08}] if smoke else
            [{"iterations": n, "depth": d, "learning_rate": lr} for n in (150, 300) for d in (4, 7) for lr in (0.03, 0.08)]),
    }
    availability = {"current_qreg": True, "pooled_spline_qreg": True,
                    "pooled_xgboost_qreg": module_available("xgboost"),
                    "pooled_catboost_qreg": module_available("catboost")}
    history: list[dict[str, Any]] = []
    candidates: dict[str,list[tuple[tuple[float,float,float],dict[str,Any],str]]]={}
    attempted = 0
    for family, configs in grids.items():
        if not availability[family]:
            history.append({"stage": "conventional", "family": family, "status": "dependency_missing", "score": np.nan})
            continue
        for config_i, params in enumerate(configs):
            if attempted >= min(SEARCH_CAPS["conventional"], run.cfg.max_configs):
                break
            key = digest([study.input_hash, family, params, run.cfg.seed, QUANTILES.tolist()])
            checkpoint = run.out / "models" / f"conventional_{family}_{key[:12]}.pkl"
            pred_file = run.out / "predictions" / f"conventional_{family}_{key[:12]}.npz"
            started = time.time()
            if run.done(f"candidate_{key[:12]}", key, (checkpoint, pred_file)):
                fitted = atomic_load_pickle(checkpoint)
                saved = np.load(pred_file)
                fitted["val"], fitted["test"] = saved["val"], saved["test"]
                status = "resumed"
            else:
                fitted = fit_quantile_candidate(study, family, params, run.cfg.seed + config_i)
                atomic_pickle(checkpoint, {k: v for k, v in fitted.items() if k not in ("val", "test")})
                atomic_npz(pred_file, val=fitted["val"], test=fitted["test"])
                run.mark(f"candidate_{key[:12]}", "complete", key, family=family)
                status = "fit"
            score,rmse,coverage_error=validation_objectives(fitted["val"],study,corr,run.cfg.seed)
            row = {"stage": "conventional", "family": family, "config": config_i, "params": canonical_json(params),
                   "validation_crps": score,"validation_rmse":rmse,"validation_coverage_error":coverage_error,
                   "status": status, "seconds": time.time() - started, "hash": key}
            history.append(row); attempted += 1
            atomic_csv(run.out / "search" / "search_history.csv", pd.DataFrame(history))
            candidates.setdefault(family,[]).append(((score,rmse,coverage_error),fitted,key))
    best={}
    for family,items in candidates.items():
        frontier=[]
        for i,(objective,fitted,key) in enumerate(items):
            dominated=any(j!=i and all(other<=value for other,value in zip(items[j][0],objective)) and any(other<value for other,value in zip(items[j][0],objective)) for j in range(len(items)))
            if not dominated:frontier.append((objective,fitted,key))
            for row in history:
                if row.get("hash")==key:row["pareto_frontier"]=not dominated
        best[family]=min(frontier,key=lambda item:item[0][0])
    atomic_csv(run.out/"search"/"search_history.csv",pd.DataFrame(history))
    qpred = {("current_qreg_copula" if family == "current_qreg" else family): fitted["test"]
             for family, (_, fitted, _) in best.items()}
    val_pred = {("current_qreg_copula" if family == "current_qreg" else family): fitted["val"]
                for family, (_, fitted, _) in best.items()}
    return qpred, pd.DataFrame(history), {"validation": val_pred, "availability": availability}


def atomic_load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def crossfit_autoregressive(run: Run, study: Study, corr: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Recursive HGB trained with cross-fitted prior predictions and sampled recursively."""
    tr = study.split["train"]
    order = sorted(range(len(study.names)), key=lambda h: (study.months[h], study.groups[h]))
    folds = list(KFold(n_splits=min(3 if run.cfg.mode == "smoke" else 5, max(2, len(tr) // 8)),
                       shuffle=True, random_state=run.cfg.seed).split(tr))
    oof = np.full((len(study.dataset.subject_ids), len(study.names)), np.nan)
    recursive_median = {part: np.full((len(study.split[part]), len(study.names)), np.nan) for part in ("val", "test")}
    models: dict[int, list[Any]] = {}
    previous: list[int] = []
    max_iter = 25 if run.cfg.mode == "smoke" else 180
    for h in order:
        def design(ids: np.ndarray, prior: np.ndarray) -> np.ndarray:
            return np.column_stack([study.X[ids], prior[:, previous]]) if previous else study.X[ids]
        h_models = []
        observed = tr[study.M[tr, h]]
        fallback = np.quantile(study.Y[observed, h], QUANTILES) if len(observed) else np.full(len(QUANTILES), np.nan)
        for fold_train, fold_hold in folds:
            fit_ids, hold_ids = tr[fold_train], tr[fold_hold]
            fit_ids = fit_ids[study.M[fit_ids, h]]
            if len(fit_ids) < 10:
                oof[hold_ids, h] = fallback[4]
                continue
            model = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=max_iter,
                                                   max_leaf_nodes=15, l2_regularization=1.0,
                                                   random_state=run.cfg.seed + h)
            model.fit(design(fit_ids, oof[fit_ids]), study.Y[fit_ids, h])
            oof[hold_ids, h] = model.predict(design(hold_ids, oof[hold_ids]))
        for qi, q in enumerate(QUANTILES):
            if len(observed) < 10:
                model = None
            else:
                model = HistGradientBoostingRegressor(loss="quantile", quantile=float(q), max_iter=max_iter,
                                                       max_leaf_nodes=15, l2_regularization=1.0,
                                                       random_state=run.cfg.seed + h * 17 + qi)
                model.fit(design(observed, oof[observed]), study.Y[observed, h])
            h_models.append(model)
            if qi == 4:
                for part in recursive_median:
                    ids = study.split[part]
                    recursive_median[part][:, h] = fallback[qi] if model is None else model.predict(design(ids, recursive_median[part]))
        models[h] = h_models
        previous.append(h)

    def sample(part: str) -> np.ndarray:
        ids = study.split[part]; n = len(ids); m = run.cfg.n_samples
        rng = np.random.default_rng(run.cfg.seed + (1 if part == "val" else 2))
        z = rng.multivariate_normal(np.zeros(len(study.names)), corr, size=(n, m))
        u = np.clip(stats.norm.cdf(z), QUANTILES[0], QUANTILES[-1])
        out = np.full((n, m, len(study.names)), np.nan, dtype=np.float32)
        previous_local: list[int] = []
        for h in order:
            prior = out[:, :, previous_local].reshape(n * m, len(previous_local)) if previous_local else np.empty((n*m, 0))
            x = np.repeat(study.X[ids], m, axis=0)
            d = np.column_stack([x, prior])
            qvalues = np.column_stack([
                np.full(n*m, np.nanquantile(study.Y[tr[study.M[tr,h]],h], q)) if model is None else model.predict(d)
                for q, model in zip(QUANTILES, models[h])
            ]).reshape(n, m, len(QUANTILES))
            for i in range(n):
                for j in range(m):
                    out[i, j, h] = np.interp(u[i, j, h], QUANTILES, np.sort(qvalues[i, j]))
            previous_local.append(h)
        return out
    val_samples, test_samples = sample("val"), sample("test")
    point = np.repeat(np.median(test_samples, axis=1)[:, None, :], run.cfg.n_samples, axis=1)
    checkpoint = run.out / "models" / "autoregressive_hgb.pkl"
    atomic_pickle(checkpoint, {"models": models, "order": order, "cross_fitted": True, "feature_names": study.feature_names})
    return {"autoregressive_hgb_quantile": test_samples, "autoregressive_hgb_point": point}, \
           {"autoregressive_hgb_quantile": val_samples}, {"cross_fitted": True, "order": order}


def ensemble_from_validation(val_samples: dict[str, np.ndarray], test_samples: dict[str, np.ndarray], study: Study) -> tuple[np.ndarray, dict[str, float]]:
    names = list(val_samples)
    if not names:
        raise ValueError("No validation predictions available for ensemble")
    idx = study.split["val"]
    def objective(w: np.ndarray) -> float:
        combined = sum(float(w[i]) * val_samples[name] for i, name in enumerate(names))
        scores = []
        for h in range(len(study.names)):
            keep = study.M[idx, h]
            if keep.any(): scores.extend(bt.crps_ensemble(combined[keep,:,h], study.Y[idx[keep],h]))
        return float(np.mean(scores))
    initial = np.full(len(names), 1 / len(names))
    result = optimize.minimize(objective, initial, method="SLSQP", bounds=[(0,1)]*len(names),
                               constraints={"type":"eq","fun":lambda w: np.sum(w)-1})
    weights = result.x if result.success else initial
    weights = np.maximum(weights, 0); weights /= weights.sum()
    return sum(float(weights[i])*test_samples[name] for i,name in enumerate(names)), dict(zip(names, map(float, weights)))


def observation_probabilities(study: Study) -> tuple[np.ndarray,list[Any]]:
    """Train-only censoring models, predicted once for every local patient and horizon."""
    out = np.full_like(study.Y, np.nan, dtype=float);models=[]
    tr = study.split["train"]
    for h in range(len(study.names)):
        y = study.M[:, h].astype(int)
        prevalence = float(np.mean(y[tr]))
        if np.unique(y[tr]).size < 2 or len(tr) < 20:
            out[:, h] = prevalence;models.append(None)
            continue
        model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                          ("model", LogisticRegression(C=0.2, max_iter=1000, random_state=17+h))])
        model.fit(study.X[tr], y[tr]); out[:, h] = model.predict_proba(study.X)[:,1];models.append(model)
    return np.clip(out, 0.05, 1.0),models


def preop_cells(study: Study) -> pd.DataFrame:
    te = study.split["test"]
    return pd.DataFrame({
        "patient_position": np.repeat(te, len(study.names)),
        "patient_id": np.repeat(study.dataset.subject_ids[te], len(study.names)),
        "origin": 0, "target_idx": np.tile(np.arange(len(study.names)), len(te)),
        "target": np.tile(study.names, len(te)), "group": np.tile(study.groups, len(te)),
        "month": np.tile(study.months, len(te)),
        "observed": study.Y[te].reshape(-1), "is_observed": study.M[te].reshape(-1),
        "procedure": np.repeat(study.dataset.surgery_type[te], len(study.names)),
    })


def history_features(study: Study, patient: int, origin: int, memory: int) -> list[float]:
    history: list[tuple[int, float, int]] = []
    bmi0 = study.X[patient, study.feature_names.index("bmi_at_surgery")]
    a1c0 = study.X[patient, study.feature_names.index("hba1c_at_surgery")]
    if np.isfinite(bmi0): history.append((0, bmi0, 0))
    if np.isfinite(a1c0): history.append((0, a1c0, 1))
    for h in np.where(study.months <= origin)[0]:
        if study.M[patient,h]: history.append((int(study.months[h]), float(study.Y[patient,h]), int(study.groups[h]=="hba1c")))
    history.sort(key=lambda x:x[0], reverse=True)
    values: list[float] = []
    for j in range(memory):
        if j < len(history):
            month, value, group = history[j]; values += [value, month/60, float(group), (origin-month)/60]
        else: values += [np.nan, np.nan, np.nan, np.nan]
    return values


def rolling_cells(study: Study, patients: np.ndarray, memory: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    rows, features = [], []
    for patient in patients:
        for origin in ORIGINS[1:]:
            base_history = history_features(study, int(patient), origin, memory)
            for h in np.where(study.months > origin)[0]:
                same = [j for j in np.where(study.months <= origin)[0]
                        if study.groups[j] == study.groups[h] and study.M[patient,j]]
                age = origin - study.months[same[-1]] if same else np.nan
                rows.append({"patient_position":int(patient), "patient_id":str(study.dataset.subject_ids[patient]),
                             "origin":origin, "target_idx":h, "target":study.names[h], "group":study.groups[h],
                             "month":study.months[h], "observed":study.Y[patient,h], "is_observed":study.M[patient,h],
                             "latest_measurement_age":age, "procedure":study.dataset.surgery_type[patient]})
                features.append(np.r_[study.X[patient], origin/60, study.months[h]/60,
                                      float(study.groups[h]=="hba1c"), base_history])
    return pd.DataFrame(rows), np.asarray(features, float)


def rolling_baseline_samples(study: Study, cells: pd.DataFrame, n_samples: int) -> dict[str,np.ndarray]:
    tr = study.split["train"]
    population = np.array([np.mean(study.Y[tr[study.M[tr,h]],h]) if study.M[tr,h].any() else np.nan
                           for h in range(len(study.names))])
    persistence, pop = [], []
    for row in cells.itertuples():
        same = [j for j in np.where(study.months <= row.origin)[0]
                if study.groups[j] == row.group and study.M[row.patient_position,j]]
        value = study.Y[row.patient_position,same[-1]] if same else population[row.target_idx]
        persistence.append(value); pop.append(population[row.target_idx])
    return {"persistence":np.repeat(np.asarray(persistence)[:,None],n_samples,axis=1),
            "population_trajectory":np.repeat(np.asarray(pop)[:,None],n_samples,axis=1)}


def fit_rolling_hgb(run: Run, study: Study) -> tuple[pd.DataFrame, dict[str,np.ndarray], pd.DataFrame]:
    train_cells, train_x = rolling_cells(study, study.split["train"])
    val_cells, val_x = rolling_cells(study, study.split["val"])
    test_cells, test_x = rolling_cells(study, study.split["test"])
    keep = train_cells.is_observed.to_numpy(bool)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(train_x[keep])
    tx, vx, ex = imputer.transform(train_x[keep]), imputer.transform(val_x), imputer.transform(test_x)
    y = train_cells.loc[keep,"observed"].to_numpy(float)
    models=[]; val_q=np.empty((len(val_cells),len(QUANTILES))); test_q=np.empty((len(test_cells),len(QUANTILES)))
    for qi,q in enumerate(QUANTILES):
        model=HistGradientBoostingRegressor(loss="quantile",quantile=float(q),max_iter=35 if run.cfg.mode=="smoke" else 180,
                                            max_leaf_nodes=15,learning_rate=0.06,l2_regularization=1,random_state=run.cfg.seed+qi)
        model.fit(tx,y); models.append(model); val_q[:,qi]=model.predict(vx); test_q[:,qi]=model.predict(ex)
    val_q=np.sort(val_q,axis=1); test_q=np.sort(test_q,axis=1)
    rng=np.random.default_rng(run.cfg.seed); u=rng.uniform(QUANTILES[0],QUANTILES[-1],size=(len(test_cells),run.cfg.n_samples))
    samples=np.vstack([np.interp(u[i],QUANTILES,test_q[i]) for i in range(len(test_cells))]).astype(np.float32)
    atomic_pickle(run.out/"models"/"rolling_hgb.pkl",{"imputer":imputer,"models":models,"memory":3})
    history=pd.DataFrame([{"stage":"rolling_hgb","family":"rolling_hgb","validation_crps":rolling_q_crps(val_q,val_cells),"status":"fit"}])
    return test_cells,{"rolling_hgb":samples},history


def rolling_q_crps(qpred: np.ndarray,cells:pd.DataFrame)->float:
    keep=cells.is_observed.to_numpy(bool); p=(np.arange(41)+.5)/41
    samples=np.vstack([np.interp(p,QUANTILES,qpred[i]) for i in range(len(qpred))])
    return float(np.mean(bt.crps_ensemble(samples[keep],cells.loc[keep,"observed"].to_numpy(float)))) if keep.any() else np.nan


def tfm_configs(smoke: bool) -> list[dict[str,Any]]:
    if smoke: return [{"width":24,"memory":3,"lr":1e-3,"diffusion":0.01,"solver_tolerance":0.015,"epochs":25}]
    grid=[]
    for width in (32,64,96):
        for memory in (2,3):
            for lr in (3e-4,1e-3):
                for diffusion in (0.0,0.02):
                    for tolerance in (0.01, 0.0025):
                        grid.append({"width":width,"memory":memory,"lr":lr,"diffusion":diffusion,
                                     "solver_tolerance":tolerance,"epochs":60})
    return grid[:24]


def fit_tfm_once(run:Run,study:Study,config:dict[str,Any],seed:int,part:str="test")->tuple[np.ndarray,float,dict[str,Any]]:
    """Conditional scalar flow; correlated bases yield joint future trajectories."""
    import torch
    from torch import nn
    torch.manual_seed(seed); np.random.seed(seed)
    tr_cells,tr_x=rolling_cells(study,study.split["train"],config["memory"])
    va_cells,va_x=rolling_cells(study,study.split["val"],config["memory"])
    out_cells,out_x=rolling_cells(study,study.split[part],config["memory"])
    keep=tr_cells.is_observed.to_numpy(bool)
    imputer=SimpleImputer(strategy="median",keep_empty_features=True).fit(tr_x[keep])
    scaler=StandardScaler().fit(imputer.transform(tr_x[keep]))
    xtr=scaler.transform(imputer.transform(tr_x[keep])).astype("float32")
    ytr=tr_cells.loc[keep,"observed"].to_numpy("float32")
    center,scale=float(np.mean(ytr)),float(max(np.std(ytr),1e-3)); ytr=(ytr-center)/scale
    model=nn.Sequential(nn.Linear(xtr.shape[1]+2,config["width"]),nn.SiLU(),
                        nn.Linear(config["width"],config["width"]),nn.SiLU(),nn.Linear(config["width"],1))
    device=torch.device(run.cfg.device if run.cfg.device!="auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model=model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=1e-3)
    xt=torch.tensor(xtr,device=device); yt=torch.tensor(ytr[:,None],device=device)
    batch=min(256,len(ytr)); rng=np.random.default_rng(seed)
    model.train()
    for _ in range(config["epochs"]):
        ids=torch.tensor(rng.integers(0,len(ytr),size=batch),device=device)
        y=yt[ids]; z=torch.randn_like(y); t=torch.rand((batch,1),device=device)
        path=(1-t)*z+t*y+config["diffusion"]*torch.randn_like(y)
        pred=model(torch.cat([path,t,xt[ids]],dim=1)); loss=((pred-(y-z))**2).mean()
        opt.zero_grad();loss.backward();opt.step()
    def sample(features:np.ndarray,cells:pd.DataFrame,n_samples:int)->np.ndarray:
        context=torch.tensor(scaler.transform(imputer.transform(features)).astype("float32"),device=device)
        rng2=np.random.default_rng(seed+99); z=np.empty((len(features),n_samples),dtype="float32")
        correlation=residual_correlation(study)
        for _, group_index in cells.groupby(["patient_position","origin"]).groups.items():
            ids=np.asarray(list(group_index)); targets=cells.loc[ids,"target_idx"].to_numpy(int)
            block=nearest_correlation(correlation[np.ix_(targets,targets)])
            z[ids]=rng2.multivariate_normal(np.zeros(len(ids)),block,size=n_samples).T
        state=torch.tensor(z.reshape(-1,1),device=device); c=context.repeat_interleave(n_samples,dim=0)
        steps=max(4,int(math.ceil(1/math.sqrt(config["solver_tolerance"]))));model.eval();dt=1/steps
        with torch.no_grad():
            for step in range(steps):
                t=torch.full_like(state,(step+.5)/steps)
                state=state+dt*model(torch.cat([state,t,c],dim=1))
        return (state.cpu().numpy().reshape(len(features),n_samples)*scale+center).astype(np.float32)
    val_samples=sample(va_x,va_cells,min(31,run.cfg.n_samples)); val_keep=va_cells.is_observed.to_numpy(bool)
    score=float(np.mean(bt.crps_ensemble(val_samples[val_keep],va_cells.loc[val_keep,"observed"].to_numpy(float)))) if val_keep.any() else np.nan
    out_samples=sample(out_x,out_cells,run.cfg.n_samples)
    state={"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"config":config,
           "center":center,"scale":scale,"imputer":imputer,"scaler":scaler,"context_dim":xtr.shape[1]}
    return out_samples,score,state


def tfm_search(run:Run,study:Study,test_cells:pd.DataFrame)->tuple[dict[str,np.ndarray],pd.DataFrame]:
    if not module_available("torch"):
        return {},pd.DataFrame([{"stage":"tfm","family":"trajectory_flow_matching","status":"dependency_missing"}])
    history=[];screened=[]
    for i,config in enumerate(tfm_configs(run.cfg.mode=="smoke")):
        samples,score,state=fit_tfm_once(run,study,config,run.cfg.seed+i)
        history.append({"stage":"tfm","family":"trajectory_flow_matching","config":i,"params":canonical_json(config),
                        "validation_crps":score,"status":"fit"})
        atomic_pickle(run.out/"models"/f"tfm_candidate_{i:02d}.pkl",state)
        atomic_csv(run.out/"search"/"tfm_search_history.csv",pd.DataFrame(history))
        screened.append((score,config,samples))
    if not screened:return {},pd.DataFrame(history)
    if run.cfg.mode=="smoke":best=min(screened,key=lambda item:item[0])
    else:
        promoted=[]
        for rank,(old_score,config,_) in enumerate(sorted(screened,key=lambda item:item[0])[:6]):
            full={**config,"epochs":180};samples,score,state=fit_tfm_once(run,study,full,run.cfg.seed+100+rank)
            promoted.append((score,full,samples));atomic_pickle(run.out/"models"/f"tfm_promoted_{rank:02d}.pkl",state)
            history.append({"stage":"tfm_high_fidelity","family":"trajectory_flow_matching","rank":rank,
                            "params":canonical_json(full),"validation_crps":score,"status":"fit"})
        best=min(promoted,key=lambda item:item[0])
    finalist=[]
    seeds=(run.cfg.seed,) if run.cfg.mode=="smoke" else tuple(run.cfg.seed+i for i in range(5))
    for i,seed in enumerate(seeds):
        samples,score,state=fit_tfm_once(run,study,best[1],seed)
        finalist.append(samples);atomic_pickle(run.out/"models"/f"tfm_final_seed_{seed}.pkl",state)
        history.append({"stage":"tfm_final","family":"trajectory_flow_matching","seed":seed,
                        "params":canonical_json(best[1]),"validation_crps":score,"status":"fit"})
    combined=np.concatenate(finalist,axis=1)
    select=np.linspace(0,combined.shape[1]-1,run.cfg.n_samples).astype(int)
    return {"trajectory_flow_matching":combined[:,select]},pd.DataFrame(history)


def effective_n(weights:np.ndarray)->float:
    w=np.asarray(weights,float);w=w[np.isfinite(w)&(w>0)]
    return float(w.sum()**2/np.sum(w*w)) if len(w) else 0.0


def weighted_mean(values:np.ndarray,weights:np.ndarray|None)->float:
    keep=np.isfinite(values)
    if weights is None:return float(np.mean(values[keep])) if keep.any() else np.nan
    w=np.asarray(weights,float);keep&=np.isfinite(w)&(w>0)
    return float(np.average(values[keep],weights=w[keep])) if keep.any() else np.nan


def weighted_median(values:np.ndarray,weights:np.ndarray|None)->float:
    values=np.asarray(values,float)
    if weights is None:return float(np.nanmedian(values))
    weights=np.asarray(weights,float);keep=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    if not keep.any():return np.nan
    order=np.argsort(values[keep]);v=values[keep][order];w=weights[keep][order]
    return float(v[np.searchsorted(np.cumsum(w),.5*w.sum())])


def evaluate_samples(cells:pd.DataFrame,samples_by_model:dict[str,np.ndarray],obs_prob:np.ndarray)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    metrics=[];per_cell=[];predictions=[]
    for model,samples in samples_by_model.items():
        if len(samples)!=len(cells):raise ValueError(f"{model} sample rows do not align with evaluation cells")
        for i,row in enumerate(cells.itertuples()):
            p=float(np.mean(samples[i]<(35 if row.group=="bmi" else 5.7)))
            predictions.append({"model":model,"patient_id":row.patient_id,"origin":row.origin,"target":row.target,
                                "month":row.month,"observed":row.observed if row.is_observed else np.nan,
                                "mean":float(np.mean(samples[i])),"median":float(np.median(samples[i])),
                                "p_threshold":p,"latest_measurement_age":getattr(row,"latest_measurement_age",np.nan),
                                "procedure":row.procedure})
        for (origin,target),index in cells.groupby(["origin","target"]).groups.items():
            ids=np.asarray(list(index));sub=cells.loc[ids];keep=sub.is_observed.to_numpy(bool)&np.isfinite(sub.observed)
            if not keep.any():continue
            ii=ids[keep];s=samples[ii];y=sub.loc[keep,"observed"].to_numpy(float)
            h=int(sub.target_idx.iloc[0]);patient=sub.loc[keep,"patient_position"].to_numpy(int)
            w=1/np.clip(obs_prob[patient,h],.05,1);point=np.mean(s,axis=1);cr=bt.crps_ensemble(s,y)
            pit=np.mean(s<=y[:,None],axis=1);abs_err=np.abs(point-y);sq=(point-y)**2
            threshold=35 if sub.group.iloc[0]=="bmi" else 5.7;p_cross=np.mean(s<threshold,axis=1);event=(y<threshold).astype(float)
            for estimate,weights in (("complete_case",None),("ipcw",w)):
                base={"model":model,"origin":origin,"target":target,"group":sub.group.iloc[0],"month":sub.month.iloc[0],
                      "estimate":estimate,"n":len(y),"effective_n":len(y) if weights is None else effective_n(weights)}
                vals={"crps":weighted_mean(cr,weights),"rmse":math.sqrt(weighted_mean(sq,weights)),
                      "mae":weighted_mean(abs_err,weights),"mad":weighted_median(abs_err,weights),
                      "bias":weighted_mean(point-y,weights),"pit_mean":weighted_mean(pit,weights),
                      "pit_ks":stats.kstest(pit,"uniform").statistic,"threshold_brier":weighted_mean((p_cross-event)**2,weights)}
                if sub.group.iloc[0]=="bmi":vals["normalized_rmse_pct"]=100*math.sqrt(weighted_mean(sq,weights))/max(abs(weighted_mean(y,weights)),1e-6)
                tc=dm.threshold_calibration(p_cross,event,w=weights);vals["threshold_ece"]=tc["ece"]
                for metric,value in vals.items():
                    metrics.append({**base,"metric":metric,"value":value,"exploratory":len(y)<200 or base["effective_n"]<100})
                for level in COVERAGES:
                    lo,hi=np.quantile(s,[(1-level)/2,1-(1-level)/2],axis=1)
                    metrics += [{**base,"metric":"coverage","level":level,"value":weighted_mean(((y>=lo)&(y<=hi)).astype(float),weights),
                                 "exploratory":len(y)<200 or base["effective_n"]<100},
                                {**base,"metric":"interval_width","level":level,"value":weighted_mean(hi-lo,weights),
                                 "exploratory":len(y)<200 or base["effective_n"]<100}]
            for j,cell_index in enumerate(ii):
                per_cell.append({"model":model,"patient_id":cells.loc[cell_index,"patient_id"],"origin":origin,"target":target,
                                 "month":sub.month.iloc[0],"crps":cr[j],"abs_error":abs_err[j],"sq_error":sq[j]})
        metrics.extend(trajectory_diagnostics(cells,samples,model))
        metrics.extend(joint_scores(cells,samples,model,obs_prob))
    return pd.DataFrame(metrics),pd.DataFrame(per_cell),pd.DataFrame(predictions)


def trajectory_diagnostics(cells:pd.DataFrame,samples:np.ndarray,model:str)->list[dict[str,Any]]:
    rows=[];mean=np.mean(samples,axis=1)
    work=cells.copy();work["pred"]=mean
    for (origin,group),g in work.groupby(["origin","group"]):
        jumps=[];curves=[]
        for _,p in g.sort_values("month").groupby("patient_id"):
            values=p.pred.to_numpy();
            if len(values)>=2:jumps.extend(np.abs(np.diff(values)))
            if len(values)>=3:curves.extend(np.abs(np.diff(values,n=2)))
        for metric,value in (("adjacent_jump",np.mean(jumps) if jumps else np.nan),("curvature",np.mean(curves) if curves else np.nan),
                             ("smoothness",np.mean(np.square(jumps)) if jumps else np.nan)):
            rows.append({"model":model,"origin":origin,"target":"joint","group":group,"month":np.nan,"estimate":"complete_case",
                         "n":len(jumps),"effective_n":len(jumps),"metric":metric,"value":value,"exploratory":len(jumps)<200})
    return rows


def joint_scores(cells:pd.DataFrame,samples:np.ndarray,model:str,obs_prob:np.ndarray)->list[dict[str,Any]]:
    per_es=[];per_vs=[]
    for (patient,origin,group),g in cells.groupby(["patient_position","origin","group"]):
        g=g[g.is_observed]
        if len(g)<2:continue
        ids=g.index.to_numpy();y=g.observed.to_numpy(float);s=samples[ids].T
        if not(np.all(np.isfinite(y)) and np.all(np.isfinite(s))):continue
        d1=np.linalg.norm(s-y,axis=1).mean();d2=np.linalg.norm(s[:,None,:]-s[None,:,:],axis=2).mean()
        es=d1-.5*d2;ey=np.abs(y[:,None]-y[None,:])**.5;ex=(np.abs(s[:,:,None]-s[:,None,:])**.5).mean(0)
        per_es.append((origin,group,es));per_vs.append((origin,group,float(np.sum((ey-ex)**2))))
    rows=[]
    for metric,data in (("energy_score",per_es),("variogram_score",per_vs)):
        frame=pd.DataFrame(data,columns=["origin","group","value"])
        for (origin,group),g in frame.groupby(["origin","group"]):
            rows.append({"model":model,"origin":origin,"target":"joint","group":group,"month":np.nan,"estimate":"complete_case",
                         "n":len(g),"effective_n":len(g),"metric":metric,"value":g.value.mean(),"exploratory":len(g)<100})
    return rows


def paired_inference(per_cell:pd.DataFrame)->pd.DataFrame:
    rows=[];rng=np.random.default_rng(2026)
    for origin,origin_df in per_cell.groupby("origin"):
        baseline=PRIMARY_BASELINE if origin==0 and PRIMARY_BASELINE in set(origin_df.model) else None
        if baseline is None:
            available=set(origin_df.model)&set(ROLLING_BASELINES)
            if not available:continue
            means=origin_df[origin_df.model.isin(available)].groupby("model").crps.mean();baseline=means.idxmin()
        b=origin_df[origin_df.model==baseline][["patient_id","target","crps"]].rename(columns={"crps":"base"})
        for model in sorted(set(origin_df.model)-{baseline}):
            a=origin_df[origin_df.model==model][["patient_id","target","crps"]].rename(columns={"crps":"candidate"})
            paired=a.merge(b,on=["patient_id","target"])
            patient_average=paired.groupby("patient_id")[["candidate","base"]].mean()
            diff=patient_average.candidate-patient_average.base
            if len(diff)<2:continue
            boot=np.array([np.mean(rng.choice(diff.to_numpy(),len(diff),replace=True)) for _ in range(500)])
            rows.append({"origin":origin,"target":"pooled","model":model,"baseline":baseline,"n":len(diff),
                         "mean_difference":diff.mean(),"relative_improvement":-diff.mean()/patient_average.base.mean(),
                         "wilcoxon_p":stats.wilcoxon(diff).pvalue if not np.allclose(diff,0) else 1.0,
                         "paired_t_p":stats.ttest_1samp(diff,0).pvalue,"bootstrap_low":np.quantile(boot,.025),"bootstrap_high":np.quantile(boot,.975)})
            for target,g in paired.groupby("target"):
                d=g.candidate-g.base
                if len(d)>=2:rows.append({"origin":origin,"target":target,"model":model,"baseline":baseline,"n":len(d),
                    "mean_difference":d.mean(),"relative_improvement":-d.mean()/g.base.mean(),
                    "wilcoxon_p":stats.wilcoxon(d).pvalue if not np.allclose(d,0) else 1.0,"paired_t_p":stats.ttest_1samp(d,0).pvalue})
    out=pd.DataFrame(rows)
    if len(out):
        p=out.wilcoxon_p.fillna(1).to_numpy();order=np.argsort(p);adjusted=np.empty(len(p));adjusted[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];out["fdr_p"]=np.minimum(adjusted,1)
        out["simultaneous_low"]=np.nan;out["simultaneous_high"]=np.nan
        for (origin,model,baseline),group in out[out.target!="pooled"].groupby(["origin","model","baseline"]):
            source=per_cell[per_cell.origin.eq(origin)]
            a=source[source.model.eq(model)][["patient_id","target","crps"]].rename(columns={"crps":"a"})
            b=source[source.model.eq(baseline)][["patient_id","target","crps"]].rename(columns={"crps":"b"})
            pivot=a.merge(b,on=["patient_id","target"]);pivot["difference"]=pivot.a-pivot.b
            wide=pivot.pivot(index="patient_id",columns="target",values="difference")
            if len(wide)<3:continue
            values=wide.to_numpy(float);means=np.nanmean(values,axis=0);boot=[]
            for _ in range(300):
                sampled=values[rng.integers(0,len(values),len(values))];count=np.isfinite(sampled).sum(axis=0)
                boot.append(np.divide(np.nansum(sampled,axis=0),count,out=np.full(sampled.shape[1],np.nan),where=count>0))
            boot=np.asarray(boot);se=np.nanstd(boot,axis=0,ddof=1);valid=se>0
            if not valid.any():continue
            standardized=np.abs((boot[:,valid]-means[valid])/se[valid]);tmax=np.max(np.nan_to_num(standardized,nan=0.0),axis=1);critical=np.quantile(tmax,.95)
            bounds={target:(means[j]-critical*se[j],means[j]+critical*se[j]) for j,target in enumerate(wide.columns)}
            for target,(lo,hi) in bounds.items():
                mask=(out.origin==origin)&(out.model==model)&(out.baseline==baseline)&(out.target==target)
                out.loc[mask,["simultaneous_low","simultaneous_high"]]=lo,hi
    return out


def performance_gates(metrics:pd.DataFrame,inference:pd.DataFrame,risk:pd.DataFrame)->pd.DataFrame:
    rows=[];candidate="validation_weighted_ensemble"
    powered=metrics[(metrics.origin==0)&(metrics.model==candidate)&(metrics.metric=="crps")&
                    (metrics.estimate=="complete_case")&(~metrics.exploratory.astype(bool))]
    powered_targets=set(powered.target)
    pooled=inference[(inference.origin==0)&(inference.target=="pooled")&(inference.model==candidate)]
    improvement=float(pooled.relative_improvement.iloc[0]) if len(pooled) else np.nan
    pooled_assessable=bool(len(powered_targets))
    rows.append({"domain":"trajectory","gate":"pooled patient-averaged CRPS improvement >=10%","value":improvement,"threshold":.10,"assessable":pooled_assessable,"passed":bool(improvement>=.10) if pooled_assessable and np.isfinite(improvement) else False})
    horizon=inference[(inference.origin==0)&(inference.target.isin(powered_targets))&(inference.model==candidate)]
    worst=float(horizon.relative_improvement.min()) if len(horizon) else np.nan
    rows.append({"domain":"trajectory","gate":"no adequately powered horizon >5% worse","value":worst,"threshold":-.05,"assessable":bool(len(horizon)),"passed":bool(worst>=-.05) if np.isfinite(worst) else False})
    all_ten=bool((horizon.relative_improvement>=.10).all()) if len(horizon) else False
    rows.append({"domain":"stretch","gate":"10% CRPS improvement at every adequately powered horizon","value":float(all_ten),"threshold":1,"assessable":bool(len(horizon)),"passed":all_ten})
    for metric in ("energy_score","variogram_score"):
        q=metrics[(metrics.origin==0)&(metrics.metric==metric)&(metrics.estimate=="complete_case")&(~metrics.exploratory.astype(bool))]
        c=q[q.model==candidate].value.mean();b=q[q.model==PRIMARY_BASELINE].value.mean();passed=bool(c<b) if np.isfinite(c) and np.isfinite(b) else False
        rows.append({"domain":"trajectory","gate":f"better {metric}","value":c-b,"threshold":0,"assessable":bool(len(q)),"passed":passed})
    coverage=metrics[(metrics.origin==0)&(metrics.model==candidate)&(metrics.target.isin(powered_targets))&(metrics.metric=="coverage")&(metrics.estimate=="complete_case")&metrics.level.isin([.8,.9,.95])].copy()
    error=float(np.nanmax(np.abs(coverage.value-coverage.level))) if len(coverage) else np.nan
    rows.append({"domain":"trajectory","gate":"maximum 80/90/95 coverage error <=5 points","value":error,"threshold":.05,"assessable":bool(len(coverage)),"passed":bool(error<=.05) if np.isfinite(error) else False})
    def risk_value(metric):
        q=risk[(risk.model=="stack_trajectory_features")&(risk.endpoint=="mace_ever")&(risk.metric==metric)];return float(q.value.iloc[0]) if len(q) else np.nan
    auc,ap,slope=risk_value("auroc"),risk_value("auprc"),risk_value("calibration_slope")
    existing_ap=risk[(risk.model=="existing_composite")&(risk.endpoint=="mace_ever")&(risk.metric=="auprc")].value
    mace_n=risk[(risk.model=="stack_trajectory_features")&(risk.endpoint=="mace_ever")].n.max() if len(risk) else 0;mace_assessable=bool(mace_n>=200)
    rows += [{"domain":"mace","gate":"temporal-test AUROC >=0.80","value":auc,"threshold":.80,"assessable":mace_assessable,"passed":bool(auc>=.80) if mace_assessable and np.isfinite(auc) else False},
             {"domain":"mace","gate":"AUPRC improves on existing composite","value":ap-(float(existing_ap.iloc[0]) if len(existing_ap) else np.nan),"threshold":0,"assessable":mace_assessable,"passed":bool(mace_assessable and len(existing_ap) and ap>existing_ap.iloc[0])},
             {"domain":"mace","gate":"calibration slope within 0.20 of one","value":abs(slope-1),"threshold":.20,"assessable":mace_assessable,"passed":bool(mace_assessable and abs(slope-1)<=.20) if np.isfinite(slope) else False}]
    return pd.DataFrame(rows)


def component_labels(study:Study)->dict[str,np.ndarray]:
    labels={"mace_ever":study.dataset.x[:,fm.TARGET_NAMES.index("mace_ever")].astype(int)}
    for name,column in (("mace","MACE"),("nephropathy","Nephropathy"),("retinopathy","Retinopathy")):
        labels[name]=fm.binary_event(study.dataset.frame[column]).to_numpy(int) if column in study.dataset.frame else np.zeros(len(study.Y),int)
    return labels


def make_risk_models(smoke:bool,seed:int)->dict[str,Any]:
    existing_cfg=gb.GBMConfig(seed=seed,split_seed=seed,split_strategy="temporal",
                              xgb_n_estimators=50 if smoke else 400,max_iter=50 if smoke else 400)
    _,existing=gb.make_estimator(existing_cfg,balanced=False,n_pos=1,n_neg=1)
    if "n_jobs" in existing.get_params():existing.set_params(n_jobs=1)
    models={
      "existing_composite":existing,
      "elastic_net":Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(solver="saga",l1_ratio=.5,C=.5,max_iter=1500,class_weight="balanced",random_state=seed))]),
      "extra_trees":Pipeline([("impute",SimpleImputer(strategy="median")),("model",ExtraTreesClassifier(n_estimators=50 if smoke else 400,min_samples_leaf=3,class_weight="balanced",n_jobs=1,random_state=seed))])}
    if module_available("xgboost"):
        import xgboost as xgb;models["xgboost"]=xgb.XGBClassifier(n_estimators=50 if smoke else 300,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,n_jobs=1,tree_method="hist",random_state=seed)
    if module_available("lightgbm"):
        import lightgbm as lgb;models["lightgbm"]=lgb.LGBMClassifier(n_estimators=50 if smoke else 300,num_leaves=31,learning_rate=.05,verbosity=-1,random_state=seed)
    if module_available("catboost"):
        from catboost import CatBoostClassifier;models["catboost"]=CatBoostClassifier(iterations=50 if smoke else 300,depth=6,learning_rate=.05,verbose=False,allow_writing_files=False,random_seed=seed)
    return models


def risk_variants(name:str,model:Any,smoke:bool)->list[Any]:
    if smoke or name=="existing_composite":return [model]
    variants=[model]
    if name=="elastic_net":
        variants=[clone(model).set_params(model__C=value) for value in (0.1,0.5,2.0)]
    elif name=="extra_trees":
        variants=[clone(model).set_params(model__min_samples_leaf=value) for value in (2,5,10)]
    elif name=="xgboost":
        variants=[clone(model).set_params(max_depth=depth,min_child_weight=leaf) for depth in (3,5) for leaf in (1,5)]
    elif name=="lightgbm":
        variants=[clone(model).set_params(num_leaves=leaves,min_child_samples=leaf) for leaves in (15,31) for leaf in (10,30)]
    elif name=="catboost":
        variants=[clone(model).set_params(depth=depth,l2_leaf_reg=reg) for depth in (4,7) for reg in (1,5)]
    return variants


def take_rows(X:Any,ids:np.ndarray)->Any:
    return X.iloc[ids] if isinstance(X,pd.DataFrame) else X[ids]


def crossfit_prob(model:Any,X:Any,y:np.ndarray,tr:np.ndarray,seed:int)->tuple[np.ndarray,Any]:
    oof=np.full(len(tr),np.nan);folds=KFold(n_splits=min(5,max(2,len(tr)//10)),shuffle=True,random_state=seed)
    for fit,hold in folds.split(tr):
        if np.unique(y[tr[fit]]).size<2:oof[hold]=np.mean(y[tr[fit]])
        else:
            m=clone(model);m.fit(take_rows(X,tr[fit]),y[tr[fit]]);oof[hold]=m.predict_proba(take_rows(X,tr[hold]))[:,1]
    fitted=clone(model)
    if np.unique(y[tr]).size>=2:fitted.fit(take_rows(X,tr),y[tr])
    return oof,fitted


def trajectory_stack_features(study:Study)->tuple[np.ndarray,np.ndarray]:
    tr=study.split["train"];oof=np.full((len(tr),len(study.names)),np.nan);test=np.full((len(study.split["test"]),len(study.names)),np.nan)
    folds=KFold(n_splits=min(5,max(2,len(tr)//10)),shuffle=True,random_state=study.split["train"][0] if len(tr) else 0)
    for h in range(len(study.names)):
        fallback=np.mean(study.Y[tr[study.M[tr,h]],h]) if study.M[tr,h].any() else 0
        for fit,hold in folds.split(tr):
            ids=tr[fit];ids=ids[study.M[ids,h]]
            if len(ids)<5:oof[hold,h]=fallback
            else:
                m=Pipeline([("impute",SimpleImputer(strategy="median")),("ridge",Ridge(alpha=5))]);m.fit(study.X[ids],study.Y[ids,h]);oof[hold,h]=m.predict(study.X[tr[hold]])
        ids=tr[study.M[tr,h]]
        if len(ids)<5:test[:,h]=fallback
        else:
            m=Pipeline([("impute",SimpleImputer(strategy="median")),("ridge",Ridge(alpha=5))]);m.fit(study.X[ids],study.Y[ids,h]);test[:,h]=m.predict(study.X[study.split["test"]])
    return oof,test


def ece(y:np.ndarray,p:np.ndarray,bins:int=10)->float:
    edges=np.linspace(0,1,bins+1);total=0
    for lo,hi in zip(edges[:-1],edges[1:]):
        keep=(p>=lo)&(p<(hi if hi<1 else hi+1e-9))
        if keep.any():total+=keep.mean()*abs(y[keep].mean()-p[keep].mean())
    return float(total)


def risk_metrics(model:str,endpoint:str,y:np.ndarray,p:np.ndarray)->list[dict[str,Any]]:
    if len(y)==0:return []
    values={"brier":brier_score_loss(y,p),"ece":ece(y,p)}
    if np.unique(y).size==2:values.update({"auroc":roc_auc_score(y,p),"auprc":average_precision_score(y,p)})
    if len(y)>=10 and np.unique(y).size==2:
        pc=np.clip(p,1e-6,1-1e-6);logit=np.log(pc/(1-pc))
        calibration=LogisticRegression(C=1e12,max_iter=1000).fit(logit[:,None],y)
        values.update({"calibration_slope":calibration.coef_[0,0],"calibration_intercept":calibration.intercept_[0]})
    else:values.update({"calibration_slope":np.nan,"calibration_intercept":np.nan})
    pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();values.update({"sensitivity":tp/max(tp+fn,1),"specificity":tn/max(tn+fp,1),"ppv":tp/max(tp+fp,1),"npv":tn/max(tn+fn,1)})
    values.update({"macro_f1":f1_score(y,pred,average="macro",zero_division=0),"precision":precision_score(y,pred,zero_division=0),
                   "recall":recall_score(y,pred,zero_division=0),"log_loss":log_loss(y,np.column_stack([1-p,p]),labels=[0,1])})
    return [{"model":model,"endpoint":endpoint,"metric":k,"value":float(v),"n":len(y),"n_positive":int(y.sum())} for k,v in values.items()]


def delong_pair(y:np.ndarray,a:np.ndarray,b:np.ndarray)->dict[str,float]:
    pos=y==1;neg=~pos
    def structural(scores):
        x=scores[pos][:,None];z=scores[neg][None,:];phi=(x>z)+.5*(x==z);return phi.mean(1),phi.mean(0)
    if pos.sum()<2 or neg.sum()<2:return {"difference":np.nan,"z":np.nan,"p":np.nan}
    a10,a01=structural(a);b10,b01=structural(b);diff=a10.mean()-b10.mean()
    var=np.var(a10-b10,ddof=1)/len(a10)+np.var(a01-b01,ddof=1)/len(a01);z=diff/math.sqrt(var) if var>0 else 0
    return {"difference":float(diff),"z":float(z),"p":float(2*stats.norm.sf(abs(z)))}


def risk_study(run:Run,study:Study)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    labels=component_labels(study);y=labels["mace_ever"];tr,va,te=study.split["train"],study.split["val"],study.split["test"]
    risk_X,risk_feature_names,repo_y=gb.assemble_features(study.dataset)
    if isinstance(risk_X,np.ndarray):risk_X=pd.DataFrame(risk_X,columns=risk_feature_names)
    if not np.array_equal(repo_y,y):raise AssertionError("Repository MACE label differs from study composite label")
    base=make_risk_models(run.cfg.mode=="smoke",run.cfg.seed);oof={};test_prob={};models={};search=[]
    for i,(name,model) in enumerate(base.items()):
        if np.unique(y[tr]).size<2:
            oof[name]=np.full(len(tr),y[tr].mean());test_prob[name]=np.full(len(te),y[tr].mean());models[name]=None
            search.append({"stage":"risk","family":name,"status":"fallback_single_class","validation_auroc":np.nan})
            continue
        best=None
        for config_i,candidate in enumerate(risk_variants(name,model,run.cfg.mode=="smoke")):
            candidate_oof,fitted=crossfit_prob(candidate,risk_X,y,tr,run.cfg.seed+i*10+config_i)
            val_prob=fitted.predict_proba(take_rows(risk_X,va))[:,1]
            val_auc=roc_auc_score(y[va],val_prob) if np.unique(y[va]).size==2 else .5
            val_brier=brier_score_loss(y[va],val_prob);score=val_auc-.1*val_brier
            search.append({"stage":"risk","family":name,"config":config_i,"status":"fit","validation_auroc":val_auc,
                           "validation_brier":val_brier,"selection_score":score,"params":canonical_json(candidate.get_params())})
            if best is None or score>best[0]:best=(score,candidate_oof,fitted)
        _,oof[name],models[name]=best;test_prob[name]=models[name].predict_proba(take_rows(risk_X,te))[:,1]
    for family,library in (("lightgbm","lightgbm"),("catboost","catboost")):
        if not module_available(library):search.append({"stage":"risk","family":family,"status":"dependency_missing","validation_auroc":np.nan})
    names=list(test_prob);meta=LogisticRegression(C=.5,max_iter=1000).fit(np.column_stack([oof[n] for n in names]),y[tr]);test_prob["leak_free_stack"]=meta.predict_proba(np.column_stack([test_prob[n] for n in names]))[:,1]
    component_oof=[];component_test=[]
    for endpoint in ("mace","nephropathy","retinopathy"):
        model=make_risk_models(run.cfg.mode=="smoke",run.cfg.seed)["extra_trees"];co,cm=crossfit_prob(model,risk_X,labels[endpoint],tr,run.cfg.seed+20)
        component_oof.append(co);component_test.append(cm.predict_proba(take_rows(risk_X,te))[:,1] if np.unique(labels[endpoint][tr]).size>1 else np.full(len(te),labels[endpoint][tr].mean()))
    meta_c=LogisticRegression(C=.5,max_iter=1000).fit(np.column_stack([oof[n] for n in names]+component_oof),y[tr]);test_prob["stack_component_probabilities"]=meta_c.predict_proba(np.column_stack([test_prob[n] for n in names]+component_test))[:,1]
    traj_oof,traj_test=trajectory_stack_features(study);meta_t=LogisticRegression(C=.2,max_iter=1500).fit(np.column_stack([oof[n] for n in names]+[traj_oof]),y[tr]);test_prob["stack_trajectory_features"]=meta_t.predict_proba(np.column_stack([test_prob[n] for n in names]+[traj_test]))[:,1]
    rows=[];pred=[]
    for name,p in test_prob.items():
        rows+=risk_metrics(name,"mace_ever",y[te],p);pred += [{"patient_id":study.dataset.subject_ids[te[i]],"model":name,"endpoint":"mace_ever","observed":y[te[i]],"probability":p[i]} for i in range(len(te))]
    for endpoint,p in zip(("mace","nephropathy","retinopathy"),component_test):rows+=risk_metrics("component_extra_trees",endpoint,labels[endpoint][te],p)
    remission=remission_eligibility(study)
    if remission["eligible"]:
        remission_y=fm.binary_event(study.dataset.frame[remission["exact_definition_columns"][0]]).to_numpy(int)
        model=make_risk_models(run.cfg.mode=="smoke",run.cfg.seed)["extra_trees"]
        _,fitted=crossfit_prob(model,risk_X,remission_y,tr,run.cfg.seed+77)
        remission_p=fitted.predict_proba(take_rows(risk_X,te))[:,1] if np.unique(remission_y[tr]).size>1 else np.full(len(te),remission_y[tr].mean())
        rows+=risk_metrics("remission_extra_trees","diabetes_remission",remission_y[te],remission_p)
        pred += [{"patient_id":study.dataset.subject_ids[te[i]],"model":"remission_extra_trees","endpoint":"diabetes_remission",
                  "observed":remission_y[te[i]],"probability":remission_p[i]} for i in range(len(te))]
    comparisons=[];reference=test_prob.get("existing_composite")
    rng=np.random.default_rng(run.cfg.seed)
    for name,p in test_prob.items():
        if name=="existing_composite" or reference is None:continue
        d=delong_pair(y[te],p,reference);boot=[]
        for _ in range(500):
            ids=rng.integers(0,len(te),len(te))
            if np.unique(y[te][ids]).size==2:boot.append(roc_auc_score(y[te][ids],p[ids])-roc_auc_score(y[te][ids],reference[ids]))
        comparisons.append({"model":name,"reference":"existing_composite",**d,"bootstrap_low":np.quantile(boot,.025) if boot else np.nan,"bootstrap_high":np.quantile(boot,.975) if boot else np.nan})
    dca=[]
    for name,p in test_prob.items():
        for threshold in np.linspace(.05,.5,10):
            pred_pos=p>=threshold;tp=np.sum(pred_pos&(y[te]==1));fp=np.sum(pred_pos&(y[te]==0));net=tp/len(te)-fp/len(te)*threshold/(1-threshold)
            dca.append({"model":name,"threshold":threshold,"net_benefit":net})
    atomic_pickle(run.out/"models"/"risk_models.pkl",{"base":models,"stack":meta,"component_stack":meta_c,
                  "trajectory_stack":meta_t,"feature_names":risk_feature_names})
    atomic_csv(run.out/"search"/"risk_search_history.csv",pd.DataFrame(search))
    return pd.DataFrame(rows),pd.DataFrame(pred),pd.DataFrame(comparisons+[{"model":r["model"],"threshold":r["threshold"],"net_benefit":r["net_benefit"],"kind":"decision_curve"} for r in dca])


def progress_figure(run:Run)->None:
    fig,ax=plt.subplots(figsize=(12,6.75));ax.axis("off");stages=run.state.get("stages",{})
    lines=["Five-year forecasting improvement study",f"Status: {run.state.get('status')}",""]+[f"{k:28s} {v.get('status','?')}" for k,v in stages.items()]
    ax.text(.04,.94,"\n".join(lines),va="top",family="monospace",fontsize=12)
    fig.savefig(run.out/"figures"/"00_progress.png",dpi=180,bbox_inches="tight");plt.close(fig)


def figure_report(run:Run,study:Study,metrics:pd.DataFrame,pred:pd.DataFrame,search:pd.DataFrame,
                  inference:pd.DataFrame,risk:pd.DataFrame,risk_pred:pd.DataFrame,risk_extra:pd.DataFrame,
                  gates:pd.DataFrame|None=None)->None:
    figures=run.out/"figures";pages=[]
    def add(title,draw):pages.append((title,draw))
    def text_page(text):return lambda ax:ax.text(.03,.95,text,va="top",fontsize=11,wrap=True)
    add("Run status and completed-stage dashboard",text_page(json.dumps(run.state,indent=2)[:6500]))
    dates=pd.to_datetime(study.dataset.frame["ProcDateValue"],errors="coerce")
    summary=f"Source: {study.dataset.source_label}\nPatients: {len(study.Y):,}\nTrain / validation / sealed test: "+" / ".join(str(len(study.split[x])) for x in ("train","val","test"))+f"\nSurgery dates: {dates.min().date()} to {dates.max().date()}\nInput SHA-256: {study.input_hash}"
    add("Cohort and temporal-split summary",text_page(summary))
    def maturity(ax):
        q=metrics[(metrics.metric=="crps")&(metrics.estimate=="complete_case")&(metrics.origin==0)]
        for group,g in q.groupby("group"):ax.plot(g.groupby("month").n.max(),"o-",label=group)
        ax.set(xlabel="Month",ylabel="Observed sealed-test patients");ax.legend();ax.grid(alpha=.2)
    add("Follow-up maturity and sample size",maturity)
    def search_plot(ax):
        q=search[search.validation_crps.notna()] if "validation_crps" in search else pd.DataFrame()
        for family,g in q.groupby("family"):ax.plot(np.arange(len(g)),g.validation_crps,"o-",label=family)
        ax.set(xlabel="Configuration within family",ylabel="Validation CRPS");ax.legend(fontsize=7);ax.grid(alpha=.2)
    add("Hyperparameter-search progress",search_plot)
    def leaderboard(ax):
        q=metrics[(metrics.metric=="crps")&(metrics.estimate=="complete_case")].groupby("model").value.mean().sort_values();ax.barh(q.index,q.values);ax.set(xlabel="Mean cell CRPS")
    add("Candidate leaderboard",leaderboard)
    def horizon(metric_names,title):
        def draw(ax):
            fig=ax.figure;ax.remove();axes=fig.subplots(1,2)
            for panel,(group,label) in zip(axes,(("bmi","BMI"),("hba1c","HbA1c"))):
                q=metrics[(metrics.origin==0)&(metrics.group==group)&(metrics.estimate=="complete_case")]
                for metric in metric_names:
                    for model,g in q[q.metric==metric].groupby("model"):
                        g=g.sort_values("month");label_text=model if len(metric_names)==1 else f"{model} {metric}"
                        panel.plot(g.month,g.value,"o-" if metric==metric_names[0] else "--",label=label_text,markersize=4,linewidth=1.4)
                panel.set(title=label,xlabel="Month",ylabel=" / ".join(metric_names))
                if group=="bmi" and "rmse" in metric_names:
                    bjs=LITERATURE["bjs_2026"]["bmi"]["rmse_by_month"];sophia=LITERATURE["sophia"]["bmi"]["rmse_by_month"]
                    panel.plot(list(bjs),list(bjs.values()),"k:",label="BJS RMSE reference",linewidth=1)
                    panel.scatter(list(sophia),list(sophia.values()),marker="x",color="black",label="SOPHIA RMSE reference")
                if group=="bmi" and "mae" in metric_names:panel.axhline(.62,color="gray",ls="--",label="BJS pooled MAE")
                if panel.get_legend_handles_labels()[0]:panel.legend(fontsize=5)
                panel.grid(alpha=.2)
            fig.suptitle(title,x=.03,ha="left",weight="bold")
        add(title,draw)
    horizon(("crps",),"Pre-op CRPS by all 13 horizons");horizon(("rmse","mae"),"Pre-op RMSE and MAE by horizon")
    def references(ax):
        fig=ax.figure;ax.remove();rmse,norm,mad=fig.subplots(1,3)
        q=metrics[(metrics.origin==0)&(metrics.group=="bmi")&(metrics.estimate=="complete_case")]
        for model,g in q[q.metric=="rmse"].groupby("model"):g=g.sort_values("month");rmse.plot(g.month,g.value,"o-",label=model,markersize=3)
        b=LITERATURE["bjs_2026"]["bmi"]["rmse_by_month"];s=LITERATURE["sophia"]["bmi"]["rmse_by_month"]
        rmse.plot(list(b),list(b.values()),"k--",label="BJS by month");rmse.axhline(1.11,color="k",ls=":",label="BJS pooled");rmse.scatter(list(s),list(s.values()),marker="x",s=45,label="SOPHIA")
        for model,g in q[q.metric=="normalized_rmse_pct"].groupby("model"):g=g[g.month.isin([12,24,60])].sort_values("month");norm.plot(g.month,g.value,"o-",label=model,markersize=3)
        sophia_norm=LITERATURE["sophia"]["bmi"]["normalized_rmse_pct"];norm.plot(list(sophia_norm),list(sophia_norm.values()),"kx--",label="SOPHIA")
        at60=q[(q.metric=="mad")&q.month.eq(60)].sort_values("value");mad.barh(at60.model,at60.value);mad.axvline(2.8,color="k",ls="--",label="SOPHIA MAD60")
        rmse.set(title="RMSE",xlabel="Month",ylabel="kg/m2");norm.set(title="Normalized RMSE",xlabel="Month",ylabel="Percent");mad.set(title="60-month MAD",xlabel="kg/m2")
        rmse.legend(fontsize=4);norm.legend(fontsize=4);mad.legend(fontsize=5);rmse.grid(alpha=.2);norm.grid(alpha=.2);mad.grid(alpha=.2)
        fig.suptitle("BJS and SOPHIA target comparison",x=.03,ha="left",weight="bold")
    add("BJS and SOPHIA target comparison",references)
    def heatmap(ax):
        fig=ax.figure;ax.remove();axes=fig.subplots(1,2)
        q=metrics[(metrics.metric=="crps")&(metrics.estimate=="complete_case")]
        for panel,(group,label) in zip(axes,(("bmi","BMI"),("hba1c","HbA1c"))):
            best=q[q.group==group].groupby(["origin","month"]).value.min().unstack()
            if best.empty:panel.axis("off");panel.text(.5,.5,f"No {label} cells yet",ha="center");continue
            im=panel.imshow(best,aspect="auto",cmap="viridis")
            panel.set(yticks=range(len(best)),yticklabels=best.index,xticks=range(len(best.columns)),xticklabels=best.columns,title=label,xlabel="Target month",ylabel="Origin")
            fig.colorbar(im,ax=panel,label="Best CRPS")
        fig.suptitle("Forecast-origin by target CRPS heatmap",x=.03,ha="left",weight="bold")
    add("Forecast-origin by target CRPS heatmap",heatmap)
    def lead(ax):
        q=metrics[(metrics.metric=="crps")&(metrics.estimate=="complete_case")].copy();q["lead"]=q.month-q.origin
        for model,g in q.groupby("model"):ax.plot(g.groupby("lead").value.mean(),"o-",label=model)
        ax.legend(fontsize=6);ax.set(xlabel="Lead time months",ylabel="CRPS");ax.grid(alpha=.2)
    add("Forecast lead-time performance curves",lead)
    def coverage(ax):
        q=metrics[(metrics.metric=="coverage")&(metrics.estimate=="complete_case")&(metrics.origin==0)]
        for model,g in q.groupby("model"):ax.plot(g.level,g.value,"o",label=model)
        ax.plot([.5,.95],[.5,.95],"k--");ax.legend(fontsize=6);ax.set(xlabel="Nominal",ylabel="Empirical coverage");ax.grid(alpha=.2)
    add("Coverage and calibration by horizon",coverage)
    def joint(ax):
        q=metrics[metrics.metric.isin(["energy_score","variogram_score"])];pivot=q.groupby(["model","metric"]).value.mean().unstack();pivot.plot.bar(ax=ax);ax.set_ylabel("Score, lower is better");ax.tick_params(axis="x",rotation=35)
    add("Energy and variogram comparisons",joint)
    def trajectories(ax):
        ids=pred.patient_id.drop_duplicates().head(4)
        q=pred[(pred.patient_id.isin(ids))&(pred.origin==0)&(pred.model==PRIMARY_BASELINE)]
        for patient,g in q.groupby("patient_id"):ax.plot(g.month,g["mean"],"o-",label=str(patient))
        ax.legend(fontsize=6);ax.set(xlabel="Month",ylabel="Predicted outcome, group-specific units");ax.grid(alpha=.2)
    add("BMI and HbA1c trajectory examples",trajectories)
    def procedure(ax):
        q=pred[(pred.origin==0)&pred.observed.notna()&pred.target.eq("bmi_5y")].copy();q["se"]=(q["mean"]-q.observed)**2
        table=np.sqrt(q.groupby(["model","procedure"]).se.mean()).unstack();table.plot.bar(ax=ax)
        ax.axhline(4.5,color="k",ls="--",label="SOPHIA RYGB");ax.axhline(5.7,color="gray",ls=":",label="SOPHIA sleeve")
        ax.set_ylabel("60-month BMI RMSE");ax.tick_params(axis="x",rotation=35);ax.legend(fontsize=6)
    add("Procedure-stratified results",procedure)
    def roc_pr(ax):
        fig=ax.figure;ax.remove();a,b=fig.subplots(1,2)
        for model,g in risk_pred[risk_pred.endpoint=="mace_ever"].groupby("model"):
            if g.observed.nunique()==2:
                fpr,tpr,_=roc_curve(g.observed,g.probability);precision,recall,_=precision_recall_curve(g.observed,g.probability);a.plot(fpr,tpr,label=model);b.plot(recall,precision,label=model)
        a.plot([0,1],[0,1],"k:");a.set(title="ROC",xlabel="FPR",ylabel="TPR");b.set(title="Precision-recall",xlabel="Recall",ylabel="Precision");a.legend(fontsize=5);b.legend(fontsize=5)
    add("MACE ROC and precision-recall curves",roc_pr)
    def cal_decision(ax):
        fig=ax.figure;ax.remove();cal,dca=fig.subplots(1,2)
        for model,g in risk_pred[risk_pred.endpoint.eq("mace_ever")].groupby("model"):
            bins=pd.cut(g.probability,np.linspace(0,1,6),include_lowest=True);table=g.groupby(bins,observed=True).agg(pred=("probability","mean"),obs=("observed","mean"))
            cal.plot(table.pred,table.obs,"o-",label=model)
        cal.plot([0,1],[0,1],"k--");cal.set(title="Calibration",xlabel="Predicted",ylabel="Observed")
        q=risk_extra[risk_extra.kind.eq("decision_curve")] if "kind" in risk_extra else pd.DataFrame()
        for model,g in q.groupby("model"):dca.plot(g.threshold,g.net_benefit,label=model)
        cal.legend(fontsize=5);dca.legend(fontsize=5);dca.set(title="Decision curves",xlabel="Threshold",ylabel="Net benefit");cal.grid(alpha=.2);dca.grid(alpha=.2)
    add("MACE calibration and decision curves",cal_decision)
    def components(ax):
        q=risk[(risk.metric=="auroc")];q.pivot(index="model",columns="endpoint",values="value").plot.bar(ax=ax);ax.set_ylabel("AUROC");ax.tick_params(axis="x",rotation=35)
    add("Component endpoint performance",components)
    def ablation(ax):
        q=risk[(risk.endpoint=="mace_ever")&risk.metric.isin(["auroc","auprc","brier"])];q.pivot(index="model",columns="metric",values="value").plot.bar(ax=ax);ax.tick_params(axis="x",rotation=35)
    add("Risk and trajectory-stack ablations",ablation)
    def checklist(ax):
        q=inference[(inference.origin==0)&(inference.target!="pooled")];lines=[]
        if gates is not None and len(gates):
            lines += [f"[{'PASS' if row.passed else 'FAIL' if getattr(row,'assessable',True) else 'NOT ASSESSABLE'}] {row.domain}: {row.gate} (value={row.value:.4g})" for row in gates.itertuples()]
            lines.append("")
        promoted=q[q.model.eq("validation_weighted_ensemble")]
        for row in promoted.sort_values("target").itertuples():lines.append(f"{row.model:30s} {row.target:12s} improvement={100*row.relative_improvement:6.1f}% FDR={getattr(row,'fdr_p',np.nan):.3g}")
        ax.text(.02,.98,"\n".join(lines[:45]) if lines else "No adequately paired comparisons.",va="top",family="monospace",fontsize=7)
    add("All-horizon success checklist",checklist)
    limitations=("Literature registry SHA-256: "+LITERATURE_SHA256+"\n\nPublished BJS and SOPHIA values are embedded audited references. No PDF is needed at runtime and no published fitted model is reproduced. "
                 "Local paired inference applies only to models evaluated on the same patients. Definitive literature superiority requires no-refit independent external validation. "
                 "BJS MAPE is not a success gate. Diabetes-remission targets are used only if medication-free remission can be constructed exactly. Exploratory cells remain visible.")
    add("Limitations and claim-level summary",text_page(limitations))
    tmp=run.out/"performance_report.tmp.pdf"
    with PdfPages(tmp) as pdf:
        for i,(title,draw) in enumerate(pages,1):
            fig,ax=plt.subplots(figsize=(11.69,8.27));ax.set_title(title,loc="left",weight="bold")
            try:draw(ax)
            except Exception as exc:ax.axis("off");ax.text(.03,.9,f"Figure unavailable on this incomplete run:\n{type(exc).__name__}: {exc}",va="top")
            if not ax.has_data():ax.axis("off")
            fig.subplots_adjust(left=.08,right=.96,bottom=.13,top=.88,wspace=.28)
            pdf.savefig(fig);fig.savefig(figures/f"{i:02d}_{title.lower().replace(' ','_').replace('/','_')}.png",dpi=180);plt.close(fig)
    os.replace(tmp,run.out/"performance_report.pdf")


def make_bundle(run:Run)->Path:
    path=run.out/"qreg_improvement_results_bundle.zip";tmp=path.with_suffix(".tmp")
    files=[run.out/"performance_report.pdf",run.out/"config_manifest.json",run.out/"split_manifest.json",run.out/"feature_manifest.json",run.state_path]
    files+=list((run.out/"figures").glob("*.png"))+list((run.out/"metrics").glob("*.csv"))+list((run.out/"search").glob("*.csv"))+list((run.out/"logs").glob("*.log"))
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as bundle:
        for file in files:
            if file.exists():bundle.write(file,file.relative_to(run.out))
    os.replace(tmp,path);return path


def remission_eligibility(study:Study)->dict[str,Any]:
    names={fm.normalize_name(x):x for x in study.dataset.frame.columns}
    exact_keys=("medicationfreediabetesremission","diabetesremissionmedicationfree")
    exact=[names[key] for key in exact_keys if key in names]
    eligible=bool(exact)
    return {"eligible":eligible,"exact_definition_columns":exact,
            "reason":"exact medication-free remission endpoint supplied" if eligible else
                     "no exact medication-free remission endpoint; HbA1c-only proxy is prohibited"}


def supplementary_72_month(run:Run,study:Study)->tuple[pd.DataFrame,pd.DataFrame]:
    by_name={m["name"]:int(m["dim"]) for m in study.dataset.target_metadata};names=[x for x in ("bmi_6y","hba1c_6y") if x in by_name]
    if not names:return pd.DataFrame(),pd.DataFrame()
    dims=[by_name[x] for x in names];groups=["bmi" if x.startswith("bmi") else "hba1c" for x in names]
    extra=Study(study.dataset,study.split,study.X,study.dataset.x[:,dims].astype(float),study.dataset.mask[:,dims].astype(bool),
                names,groups,np.full(len(names),72),dims,study.feature_names,study.input_hash+"-72m")
    families=[("current_qreg",{"alpha":.01}),("pooled_spline_qreg",{"n_knots":3,"alpha":.02})]
    if module_available("xgboost"):families.append(("pooled_xgboost_qreg",{"n_estimators":35 if run.cfg.mode=="smoke" else 160,"max_depth":3,"learning_rate":.06}))
    corr=np.eye(len(names));samples={};checkpoints={}
    for i,(family,params) in enumerate(families):
        fitted=fit_quantile_candidate(extra,family,params,run.cfg.seed+i);label="current_qreg_copula" if family=="current_qreg" else family
        checkpoints[label]={k:v for k,v in fitted.items() if k not in ("val","test")}
        samples[label]=qgrid_to_samples(fitted["test"],run.cfg.n_samples,corr,run.cfg.seed)
    atomic_pickle(run.out/"models"/"supplementary_72m_models.pkl",checkpoints)
    probability,_=observation_probabilities(extra);cells=preop_cells(extra);flat={name:value.transpose(0,2,1).reshape(-1,value.shape[1]) for name,value in samples.items()}
    metrics,_,pred=evaluate_samples(cells,flat,probability);return metrics,pred


def self_test()->None:
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    check("thirteen_targets",len(TARGETS)==13)
    check("target_grid",[m for _,m in BMI]==[3,6,9,12,24,36,48,60])
    check("rolling_strict_future",all(target>origin for origin in ORIGINS[1:] for target in range(origin+1,61)))
    ids={"train":set(range(7)),"val":set(range(7,9)),"test":set(range(9,12))};check("split_disjoint",not(ids["train"]&ids["test"]|ids["train"]&ids["val"]|ids["val"]&ids["test"]))
    check("preop_roster",not any(any(t in n for t in POSTOP_TOKENS) for n in ("age","bmi_at_surgery")))
    q=np.sort(np.random.default_rng(1).normal(size=(4,len(QUANTILES),13)),axis=1);corr=nearest_correlation(np.eye(13));samples=qgrid_to_samples(q,21,corr,1)
    check("monotone_quantiles",bool(np.all(np.diff(q,axis=1)>=0)))
    check("valid_correlation",bool(np.min(np.linalg.eigvalsh(corr))>-1e-9 and np.allclose(np.diag(corr),1)))
    check("joint_samples",samples.shape==(4,21,13))
    check("crps_nonnegative",bool(np.all(bt.crps_ensemble(samples[:,:,0],np.zeros(4))>=0)))
    check("literature_values",LITERATURE["bjs_2026"]["bmi"]["rmse_by_month"][60]==1.01 and LITERATURE["sophia"]["bmi"]["mad_60"]==2.8)
    check("literature_frozen",LITERATURE_SHA256==digest(LITERATURE))
    check("checkpoint_hash_sensitive",digest([1,"a"])!=digest([2,"a"]))
    check("cache_root_beside_script",CACHE_ROOT.parent==SCRIPT_DIR)
    check("cache_directories_exist",all(path.is_dir() for path in (
        CACHE_ROOT/"matplotlib",CACHE_ROOT/"xdg",CACHE_ROOT/"torch",CACHE_ROOT/"joblib")))
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"mixed.json";atomic_json(path,{"pooled":1,3:2});check("atomic_mixed_json",path.exists() and json.loads(path.read_text())["3"]==2)
        rng=np.random.default_rng(12);n=24;feature_names=list(fm.PATIENT_FEATURES)+["surgery_idx"];X=rng.normal(size=(n,len(feature_names)));X[:,feature_names.index("bmi_at_surgery")]=40;X[:,feature_names.index("hba1c_at_surgery")]=7
        Y=rng.normal(size=(n,13));M=np.ones((n,13),bool);subject=np.array([f"p{i}" for i in range(n)])
        recursive_dataset=SimpleNamespace(subject_ids=subject,surgery_type=np.array(["sleeve"]*n),frame=pd.DataFrame({"ProcDateValue":pd.date_range("2018-01-01",periods=n)}))
        recursive_study=Study(recursive_dataset,{"train":np.arange(16),"val":np.arange(16,20),"test":np.arange(20,24)},X,Y,M,[x for x,_ in TARGETS],["bmi"]*8+["hba1c"]*5,np.array([x for _,x in TARGETS]),list(range(13)),feature_names,"recursive-synthetic")
        recursive_run=Run(Settings("smoke",str(Path(directory)/"recursive_a"),None,seed=3,n_samples=5));test_a,val_a,meta=crossfit_autoregressive(recursive_run,recursive_study,np.eye(13))
        check("recursive_crossfit",meta["cross_fitted"] and test_a["autoregressive_hgb_quantile"].shape==(4,5,13))
        changed=replace(recursive_study,Y=recursive_study.Y.copy());changed.Y[changed.split["test"]]+=1000
        recursive_run_b=Run(Settings("smoke",str(Path(directory)/"recursive_b"),None,seed=3,n_samples=5));test_b,_,_=crossfit_autoregressive(recursive_run_b,changed,np.eye(13))
        check("sealed_test_outcomes",np.allclose(test_a["autoregressive_hgb_quantile"],test_b["autoregressive_hgb_quantile"]))
        changed.Y[0,study_future:=7]=999;history=history_features(changed,0,3,3);check("rolling_origin_leakage",999 not in history)
        dataset=SimpleNamespace(source_label="synthetic",subject_ids=np.array(["a","b","c","d"]),
                                frame=pd.DataFrame({"ProcDateValue":pd.date_range("2020-01-01",periods=4)}))
        fake=Study(dataset,{"train":np.array([0,1]),"val":np.array([2]),"test":np.array([3])},np.zeros((4,2)),np.zeros((4,13)),np.ones((4,13),bool),
                   [x for x,_ in TARGETS],["bmi"]*8+["hba1c"]*5,np.array([x for _,x in TARGETS]),list(range(13)),["x0","x1"],"synthetic")
        empty_run=Run(Settings("self-test",str(Path(directory)/"incomplete"),None,n_samples=5))
        figure_report(empty_run,fake,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame())
        check("incomplete_report",(empty_run.out/"performance_report.pdf").exists())
        metric_rows=[]
        for model in (PRIMARY_BASELINE,"validation_weighted_ensemble"):
            for metric,value in (("crps",1.0),("rmse",1.5),("mae",1.0),("energy_score",2.0),("variogram_score",3.0)):
                metric_rows.append({"model":model,"origin":0,"target":"bmi_3m" if "score" not in metric else "joint","group":"bmi","month":3,
                                    "estimate":"complete_case","n":4,"effective_n":4,"metric":metric,"value":value})
            metric_rows.append({"model":model,"origin":0,"target":"bmi_3m","group":"bmi","month":3,"estimate":"complete_case","n":4,"effective_n":4,"metric":"coverage","level":.9,"value":.9})
        synthetic_metrics=pd.DataFrame(metric_rows)
        synthetic_pred=pd.DataFrame([{"model":PRIMARY_BASELINE,"patient_id":x,"origin":0,"target":"bmi_5y","month":60,"observed":40.,"mean":39.,"procedure":"sleeve"} for x in "abcd"])
        synthetic_search=pd.DataFrame([{"family":"qreg","validation_crps":1.0}]);synthetic_inf=pd.DataFrame([{"origin":0,"target":"bmi_3m","model":"validation_weighted_ensemble","baseline":PRIMARY_BASELINE,"relative_improvement":.1,"fdr_p":.04}])
        synthetic_risk=pd.DataFrame([{"model":"existing_composite","endpoint":"mace_ever","metric":metric,"value":value} for metric,value in (("auroc",.8),("auprc",.4),("brier",.15))])
        synthetic_risk_pred=pd.DataFrame([{"model":"existing_composite","endpoint":"mace_ever","observed":i%2,"probability":.2+.5*(i%2)} for i in range(10)])
        synthetic_extra=pd.DataFrame([{"model":"existing_composite","kind":"decision_curve","threshold":x,"net_benefit":.1-x/10} for x in (.1,.2,.3)])
        complete_run=Run(Settings("self-test",str(Path(directory)/"complete"),None,n_samples=5));complete_gates=pd.DataFrame([{"domain":"test","gate":"synthetic","value":1.,"passed":True}])
        figure_report(complete_run,fake,synthetic_metrics,synthetic_pred,synthetic_search,synthetic_inf,synthetic_risk,synthetic_risk_pred,synthetic_extra,complete_gates)
        check("completed_report",(complete_run.out/"performance_report.pdf").exists() and len(list((complete_run.out/"figures").glob("*.png")))==20)
    print(f"self-test: {len(checks)} checks passed: "+", ".join(checks))


def execute(cfg:Settings)->None:
    if not 1<=cfg.max_configs<=140:raise ValueError("--max-configs must be between 1 and 140")
    run=Run(cfg);script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if cfg.mode=="plot-only":
        manifest=Run.read_json(run.out/"config_manifest.json",{});source=cfg.csv or manifest.get("csv");study=load_study(replace(cfg,csv=source,seed=int(manifest.get("seed",cfg.seed))))
        tables=lambda name:pd.read_csv(run.out/"metrics"/name) if (run.out/"metrics"/name).exists() else pd.DataFrame()
        combined_search=run.out/"search"/"search_history_all.csv";search=pd.read_csv(combined_search) if combined_search.exists() else pd.DataFrame()
        figure_report(run,study,tables("forecast_metrics.csv"),tables("forecast_predictions.csv"),search,tables("paired_inference.csv"),tables("risk_metrics.csv"),tables("risk_predictions.csv"),tables("risk_comparisons.csv"),tables("performance_gates.csv"));make_bundle(run);return
    study=load_study(cfg);libs=library_manifest();manifest={**asdict(cfg),"script_sha256":script_hash,"input_sha256":study.input_hash,
        "literature":LITERATURE,"literature_sha256":LITERATURE_SHA256,"libraries":libs,"search_caps":SEARCH_CAPS,
        "claim_rule":"Published values are references; no published fitted model is reproduced."}
    atomic_json(run.out/"config_manifest.json",manifest)
    atomic_json(run.out/"split_manifest.json",{k:study.dataset.subject_ids[v].tolist() for k,v in study.split.items()})
    atomic_json(run.out/"feature_manifest.json",{"features":study.feature_names,"preoperative_only":True,"target_names":study.names})
    preflight={"targets":len(study.names),"split_disjoint":True,"preoperative_only":True,"test_sealed_for_selection":True,
               "literature_registry_verified":digest(LITERATURE)==LITERATURE_SHA256,"remission":remission_eligibility(study)}
    run.mark("preflight","complete",digest([study.input_hash,preflight]),**preflight)
    corr_file=run.out/"samples"/"residual_correlation.npz";corr_key=digest(["rank_gaussian_copula",study.input_hash,run.cfg.seed])
    if run.done("rank_gaussian_copula",corr_key,(corr_file,)):
        corr=np.load(corr_file)["correlation"]
        _CORRELATION_CACHE[study.input_hash]=corr
    else:
        corr=residual_correlation(study);atomic_npz(corr_file,correlation=corr);run.mark("rank_gaussian_copula","complete",corr_key)
    qpred,conv_history,meta=conventional_search(run,study,corr)
    preop_samples={name:qgrid_to_samples(pred,run.cfg.n_samples,corr,run.cfg.seed) for name,pred in qpred.items()}
    val_samples={name:qgrid_to_samples(pred,run.cfg.n_samples,corr,run.cfg.seed+100) for name,pred in meta["validation"].items()}
    autoreg_key=digest(["autoregressive",study.input_hash,run.cfg.seed,run.cfg.n_samples,run.cfg.mode])
    autoreg_file=run.out/"samples"/"autoregressive_checkpoint.npz"
    if run.done("autoregressive",autoreg_key,(run.out/"models"/"autoregressive_hgb.pkl",autoreg_file)):
        saved=np.load(autoreg_file);autoreg_test={"autoregressive_hgb_quantile":saved["test_quantile"],"autoregressive_hgb_point":saved["test_point"]};autoreg_val={"autoregressive_hgb_quantile":saved["val_quantile"]};autoreg_meta={"cross_fitted":True}
    else:
        autoreg_test,autoreg_val,autoreg_meta=crossfit_autoregressive(run,study,corr)
        atomic_npz(autoreg_file,test_quantile=autoreg_test["autoregressive_hgb_quantile"],test_point=autoreg_test["autoregressive_hgb_point"],val_quantile=autoreg_val["autoregressive_hgb_quantile"])
        run.mark("autoregressive","complete",autoreg_key,cross_fitted=True)
    preop_samples.update(autoreg_test);val_samples.update(autoreg_val)
    ensemble_key=digest(["validation_ensemble",study.input_hash,run.cfg.seed,run.cfg.n_samples,sorted(val_samples)])
    ensemble_file=run.out/"samples"/"ensemble_checkpoint.npz";weight_file=run.out/"models"/"ensemble_weights.json"
    if run.done("validation_ensemble",ensemble_key,(ensemble_file,weight_file)):
        ensemble=np.load(ensemble_file)["samples"];weights=Run.read_json(weight_file,{})
    else:
        ensemble,weights=ensemble_from_validation(val_samples,preop_samples,study);atomic_npz(ensemble_file,samples=ensemble);atomic_json(weight_file,weights);run.mark("validation_ensemble","complete",ensemble_key,weights=weights)
    preop_samples["validation_weighted_ensemble"]=ensemble
    run.mark("trajectory_models","complete",digest([study.input_hash,weights]),models=list(preop_samples))
    observation_key=digest(["observation_models",study.input_hash,run.cfg.seed]);observation_file=run.out/"samples"/"observation_probabilities.npz";observation_model_file=run.out/"models"/"observation_models.pkl"
    if run.done("observation_models",observation_key,(observation_file,observation_model_file)):
        obs_prob=np.load(observation_file)["probability"]
    else:
        obs_prob,observation_models=observation_probabilities(study);atomic_npz(observation_file,probability=obs_prob);atomic_pickle(observation_model_file,observation_models);run.mark("observation_models","complete",observation_key)
    p_cells=preop_cells(study);p_samples={name:s.transpose(0,2,1).reshape(-1,s.shape[1]) for name,s in preop_samples.items()}
    pre_metrics,pre_per,pre_pred=evaluate_samples(p_cells,p_samples,obs_prob)
    rolling_cell,_=rolling_cells(study,study.split["test"])
    rolling_key=digest(["rolling_hgb",study.input_hash,run.cfg.seed,run.cfg.n_samples,run.cfg.mode]);rolling_file=run.out/"samples"/"rolling_hgb_checkpoint.npz"
    if run.done("rolling_hgb",rolling_key,(run.out/"models"/"rolling_hgb.pkl",rolling_file)):
        hgb_samples={"rolling_hgb":np.load(rolling_file)["samples"]};hgb_history=pd.DataFrame([{"stage":"rolling_hgb","family":"rolling_hgb","status":"resumed"}])
    else:
        rolling_cell,hgb_samples,hgb_history=fit_rolling_hgb(run,study);atomic_npz(rolling_file,samples=hgb_samples["rolling_hgb"]);run.mark("rolling_hgb","complete",rolling_key)
    rolling_samples=rolling_baseline_samples(study,rolling_cell,run.cfg.n_samples);rolling_samples.update(hgb_samples)
    tfm_key=digest(["tfm",study.input_hash,run.cfg.seed,run.cfg.n_samples,tfm_configs(run.cfg.mode=="smoke")]);tfm_file=run.out/"samples"/"tfm_checkpoint.npz"
    if run.done("tfm",tfm_key,(tfm_file,)):
        tfm_samples={"trajectory_flow_matching":np.load(tfm_file)["samples"]};path=run.out/"search"/"tfm_search_history.csv";tfm_history=pd.read_csv(path) if path.exists() else pd.DataFrame([{"stage":"tfm","family":"trajectory_flow_matching","status":"resumed"}])
    else:
        tfm_samples,tfm_history=tfm_search(run,study,rolling_cell)
        if tfm_samples:atomic_npz(tfm_file,samples=tfm_samples["trajectory_flow_matching"])
        run.mark("tfm","complete",tfm_key,models=list(tfm_samples))
    rolling_samples.update(tfm_samples)
    roll_metrics,roll_per,roll_pred=evaluate_samples(rolling_cell,rolling_samples,obs_prob)
    metrics=pd.concat([pre_metrics,roll_metrics],ignore_index=True,sort=False);per_cell=pd.concat([pre_per,roll_per],ignore_index=True,sort=False);pred=pd.concat([pre_pred,roll_pred],ignore_index=True,sort=False)
    inference=paired_inference(per_cell);risk_key=digest(["risk",study.input_hash,run.cfg.seed,run.cfg.mode,library_manifest()]);risk_file=run.out/"models"/"risk_outputs.pkl"
    if run.done("risk",risk_key,(run.out/"models"/"risk_models.pkl",risk_file)):
        risk,risk_pred,risk_extra=atomic_load_pickle(risk_file)
    else:
        risk,risk_pred,risk_extra=risk_study(run,study);atomic_pickle(risk_file,(risk,risk_pred,risk_extra));run.mark("risk","complete",risk_key,models=sorted(risk.model.unique()))
    risk_history_path=run.out/"search"/"risk_search_history.csv";risk_history=pd.read_csv(risk_history_path) if risk_history_path.exists() else pd.DataFrame()
    search=pd.concat([conv_history,hgb_history,tfm_history,risk_history],ignore_index=True,sort=False)
    atomic_csv(run.out/"metrics"/"forecast_metrics.csv",metrics);atomic_csv(run.out/"metrics"/"per_cell_scores.csv",per_cell);atomic_csv(run.out/"metrics"/"paired_inference.csv",inference)
    atomic_csv(run.out/"metrics"/"forecast_predictions.csv",pred);atomic_csv(run.out/"metrics"/"risk_metrics.csv",risk);atomic_csv(run.out/"metrics"/"risk_predictions.csv",risk_pred);atomic_csv(run.out/"metrics"/"risk_comparisons.csv",risk_extra)
    observed_pred=pred[pred.observed.notna()].copy();observed_pred["error"]=observed_pred["mean"]-observed_pred.observed
    procedure_rows=[]
    for keys,g in observed_pred.groupby(["model","origin","target","procedure"]):
        procedure_rows += [{"model":keys[0],"origin":keys[1],"target":keys[2],"procedure":keys[3],"metric":"rmse","value":math.sqrt(np.mean(g.error**2)),"n":len(g)},
                           {"model":keys[0],"origin":keys[1],"target":keys[2],"procedure":keys[3],"metric":"mae","value":np.mean(np.abs(g.error)),"n":len(g)}]
    atomic_csv(run.out/"metrics"/"procedure_metrics.csv",pd.DataFrame(procedure_rows))
    gates=performance_gates(metrics,inference,risk);atomic_csv(run.out/"metrics"/"performance_gates.csv",gates)
    supp_metric_file=run.out/"metrics"/"supplementary_72m_metrics.csv";supp_pred_file=run.out/"metrics"/"supplementary_72m_predictions.csv";supp_key=digest(["supplementary72",study.input_hash,run.cfg.seed,run.cfg.n_samples,run.cfg.mode])
    if run.done("supplementary_72m",supp_key,(run.out/"models"/"supplementary_72m_models.pkl",supp_metric_file,supp_pred_file)):
        supplementary_metrics=pd.read_csv(supp_metric_file);supplementary_predictions=pd.read_csv(supp_pred_file)
    else:
        supplementary_metrics,supplementary_predictions=supplementary_72_month(run,study);atomic_csv(supp_metric_file,supplementary_metrics);atomic_csv(supp_pred_file,supplementary_predictions);run.mark("supplementary_72m","complete",supp_key)
    atomic_csv(run.out/"search"/"search_history_all.csv",search)
    artifact_status={"forecast_predictions":write_prediction_table(run.out/"predictions"/"forecast_predictions.parquet",pred,cfg.mode=="full"),
                     "risk_predictions":write_prediction_table(run.out/"predictions"/"risk_predictions.parquet",risk_pred,cfg.mode=="full")}
    atomic_json(run.out/"artifact_manifest.json",artifact_status)
    for name,samples in preop_samples.items():atomic_npz(run.out/"samples"/f"preop_{name}.npz",samples=samples)
    for name,samples in rolling_samples.items():atomic_npz(run.out/"samples"/f"rolling_{name}.npz",samples=samples)
    run.mark("evaluation","complete",digest([len(metrics),len(risk),study.input_hash]),forecast_rows=len(metrics),risk_rows=len(risk))
    run.mark("report","complete",digest([len(metrics),len(search)]));run.state["status"]="complete";run.state["bundle"]=str(run.out/"qreg_improvement_results_bundle.zip");atomic_json(run.state_path,run.state)
    figure_report(run,study,metrics,pred,search,inference,risk,risk_pred,risk_extra,gates);make_bundle(run);progress_figure(run)
    print(f"complete: {run.out}")


def parse(argv:Iterable[str]|None=None)->Settings:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",dest="mode",choices=("full","smoke","plot-only","self-test"),default="full")
    parser.add_argument("--output-dir");parser.add_argument("--csv");parser.add_argument("--seed",type=int,default=2026);parser.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--n-samples",type=int);parser.add_argument("--max-configs",type=int,default=140);parser.add_argument("--device",default="cpu")
    args=parser.parse_args(argv);out=args.output_dir or str(ROOT/"qreg_improvement"/"results"/("smoke_run" if args.mode=="smoke" else "full_run"));csv=args.csv or (str(ROOT/"fake_data"/"fake_mbs_cohort.csv") if args.mode in ("smoke","self-test") else None)
    return Settings(args.mode,out,csv,args.seed,args.resume,args.n_samples or (31 if args.mode=="smoke" else 101),args.max_configs,args.device)


if __name__=="__main__":
    settings=parse()
    if settings.mode=="self-test":self_test()
    else:
        try:execute(settings)
        except Exception as exc:
            traceback.print_exc()
            try:
                failed=Run(settings);failed.state.setdefault("errors",[]).append({"time":time.time(),"type":type(exc).__name__,"message":str(exc)})
                failed.mark("failure","failed",digest([type(exc).__name__,str(exc)]))
            except Exception:pass
            raise SystemExit(1)
