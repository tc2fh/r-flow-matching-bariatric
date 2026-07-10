"""Locate a frozen / eval run directory and load its W4-W6 CSV artifacts.

Deliberately lightweight (json + pandas only, NO torch / sklearn import) so the
CSV-only figures (CONSORT, GBM, calibrated-trajectory, ablation) stay cheap to
render. Only ``figures.sampling`` pulls in the twin model.

Accepts, transparently:
  * a frozen run dir          runs/frozen/<ts>/            (has evaluation/, RUN_MANIFEST.json)
  * an evaluation dir         runs/frozen/<ts>/evaluation/
  * a pipeline dir            .../twin_pipeline/pipeline_<ts>/
and resolves eval_dir, attrition report, pipeline dir, gbm run dir, twin run dir and
the source cohort CSV from manifest.json / eval_twin_summary.json.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

# Static horizon -> (group, months). Mirrors train_flow_matching.BMI_TARGETS /
# HBA1C_TARGETS but hardcoded so this module never imports the torch stack.
GROUP_HORIZONS: dict[str, list[str]] = {
    "bmi": ["bmi_3m", "bmi_6m", "bmi_9m", "bmi_12m", "bmi_2y", "bmi_3y", "bmi_4y", "bmi_5y", "bmi_6y"],
    "hba1c": ["hba1c_12m", "hba1c_2y", "hba1c_3y", "hba1c_4y", "hba1c_5y", "hba1c_6y"],
}
HORIZON_MONTHS: dict[str, float] = {
    "bmi_3m": 3, "bmi_6m": 6, "bmi_9m": 9, "bmi_12m": 12, "bmi_2y": 24,
    "bmi_3y": 36, "bmi_4y": 48, "bmi_5y": 60, "bmi_6y": 72,
    "hba1c_12m": 12, "hba1c_2y": 24, "hba1c_3y": 36, "hba1c_4y": 48, "hba1c_5y": 60, "hba1c_6y": 72,
}
GROUP_LABEL = {"bmi": "BMI", "hba1c": "HbA1c"}
GROUP_UNIT = {"bmi": "kg/m$^2$", "hba1c": "%-points"}
THRESHOLD = {"bmi": 35.0, "hba1c": 5.7}
THRESHOLD_LABEL = {"bmi": "BMI < 35", "hba1c": "HbA1c < 5.7"}


@dataclass
class RunArtifacts:
    root: Path
    eval_dir: Path
    pipeline_dir: Path | None
    gbm_run_dir: Path | None
    twin_run_dir: Path | None
    attrition_txt: Path | None
    source_csv: str | None

    # ---- CSV loaders (raise a clear error if an expected artifact is absent) ---- #
    def _csv(self, name: str, required: bool = True) -> pd.DataFrame | None:
        path = self.eval_dir / name
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Expected artifact {name} not found in {self.eval_dir}. "
                    f"Run freeze_run.py / evaluate_twin.py first.")
            return None
        return pd.read_csv(path)

    def calibration_pit(self) -> pd.DataFrame:
        return self._csv("eval_flow_calibration_pit.csv")

    def calibration_coverage(self) -> pd.DataFrame:
        return self._csv("eval_flow_calibration_coverage.csv")

    def threshold_per_patient(self) -> pd.DataFrame:
        return self._csv("eval_flow_threshold_probabilities.csv")

    def threshold_summary(self) -> pd.DataFrame:
        return self._csv("eval_flow_threshold_probabilities_summary.csv")

    def gbm_discrimination(self) -> pd.DataFrame:
        return self._csv("eval_gbm_discrimination_test.csv")

    def trajectory_metrics(self) -> pd.DataFrame | None:
        return self._csv("trajectory_comparison_metrics.csv", required=False)

    def trajectory_paired(self) -> pd.DataFrame | None:
        return self._csv("trajectory_comparison_paired_tests.csv", required=False)

    def gbm_predictions(self) -> pd.DataFrame:
        if self.gbm_run_dir is None:
            raise FileNotFoundError("gbm_run_dir unresolved; cannot read GBM test_predictions.csv")
        return pd.read_csv(self.gbm_run_dir / "test_predictions.csv")

    def flow_predictions(self) -> pd.DataFrame:
        if self.twin_run_dir is None:
            raise FileNotFoundError("twin_run_dir unresolved; cannot read flow test_predictions.csv")
        return pd.read_csv(self.twin_run_dir / "test_predictions.csv")

    # ---- calibration trust map (the ONE source of the caveat) ----------------- #
    def regime_by_horizon(self) -> dict[str, str]:
        pit = self.calibration_pit()
        return dict(zip(pit["horizon"].astype(str), pit["regime"].astype(str)))

    def trust_table(self, group: str) -> list[dict]:
        """Ordered per-horizon [{horizon, months, regime, trustworthy}] for a group."""
        regime = self.regime_by_horizon()
        out = []
        for h in GROUP_HORIZONS[group]:
            r = regime.get(h, "unknown")
            out.append({"horizon": h, "months": HORIZON_MONTHS[h], "regime": r,
                        "trustworthy": r == "calibrated"})
        return out


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve(run_path: Path | str) -> RunArtifacts:
    """Resolve a frozen dir / eval dir / pipeline dir into a RunArtifacts."""
    p = Path(run_path).resolve()

    # Normalise to (root, eval_dir).
    if p.name == "evaluation" and p.is_dir():
        eval_dir, root = p, p.parent
    elif (p / "evaluation").is_dir():
        eval_dir, root = p / "evaluation", p
    elif p.name.startswith("pipeline_") and (p / "evaluation").is_dir():
        eval_dir, root = p / "evaluation", p
    elif (p / "eval_twin_summary.json").exists():
        eval_dir, root = p, p.parent
    else:
        raise FileNotFoundError(
            f"{p} is neither a frozen run dir, an evaluation dir, nor a pipeline dir with evaluation/.")

    summary = _read_json(eval_dir / "eval_twin_summary.json")

    # Pipeline dir: from summary, else the frozen layout, else None.
    pipeline_dir = None
    if summary.get("pipeline_dir"):
        pipeline_dir = Path(summary["pipeline_dir"])
    else:
        cand = sorted((root / "twin_pipeline").glob("pipeline_*")) if (root / "twin_pipeline").is_dir() else []
        pipeline_dir = cand[-1] if cand else (root if root.name.startswith("pipeline_") else None)

    manifest = _read_json(pipeline_dir / "manifest.json") if pipeline_dir else {}

    def _pick(*vals):
        for v in vals:
            if v:
                return Path(v)
        return None

    gbm_run_dir = _pick(summary.get("gbm_run_dir"), manifest.get("gbm_run_dir"))
    twin_run_dir = _pick(summary.get("twin_run_dir"), manifest.get("twin_final_run_dir"))

    # summary["csv_path"] is the authoritative source CSV -- and it is None for a
    # DB-sourced run (the cohort is queried from Cosmos, never written to disk).
    # manifest["source_label"] is only a *label*: for a CSV run it happens to equal
    # the CSV path, but for a DB run it is "Cosmos MBSCohort", which is NOT a file.
    # Only accept the label as a CSV when it actually resolves to a readable file;
    # otherwise leave source_csv None so the cohort-backed figures re-query the DB
    # instead of trying to read a nonexistent path. (Falling through to the label
    # blindly is what made CONSORT / counterfactual / per-patient crash on VM runs
    # with "No such file or directory: ...\\Cosmos MBSCohort".)
    source_csv = summary.get("csv_path")
    if not source_csv:
        label = manifest.get("source_label")
        source_csv = str(label) if (label and Path(label).exists()) else None

    attrition_dir = root / "attrition"
    attrition_txt = None
    if attrition_dir.is_dir():
        reports = sorted(attrition_dir.glob("attrition_*.txt"))
        attrition_txt = reports[-1] if reports else None

    return RunArtifacts(
        root=root, eval_dir=eval_dir, pipeline_dir=pipeline_dir,
        gbm_run_dir=gbm_run_dir, twin_run_dir=twin_run_dir,
        attrition_txt=attrition_txt, source_csv=source_csv,
    )
