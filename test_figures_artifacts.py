"""Tests for figures/artifacts.py resolve() -- specifically that a DB-sourced run
(cohort queried from Cosmos, never written to disk) does NOT fabricate a source CSV
path out of the manifest's "Cosmos MBSCohort" label.

Regression for the freeze-figures crash where CONSORT / counterfactual / per-patient
all died with "No such file or directory: ...\\Cosmos MBSCohort": resolve() had used
``summary["csv_path"] or manifest["source_label"]``, so a DB run (csv_path=None) fell
through to the label and the figures then tried to read_csv() a nonexistent file.

Pure filesystem + json -- no torch / pandas model logic.

Runnable two ways:
  * pytest:  OMP_NUM_THREADS=1 ../mbsaqip_flow/.venv/bin/python -m pytest test_figures_artifacts.py -q
  * script:  OMP_NUM_THREADS=1 ../mbsaqip_flow/.venv/bin/python test_figures_artifacts.py   -> prints "ALL OK"
"""

from __future__ import annotations

import json
from pathlib import Path

from figures import artifacts as A


def _write_run(root: Path, *, csv_path, source_label) -> Path:
    """Fabricate the minimal frozen-run tree resolve() reads: an evaluation/ dir
    with eval_twin_summary.json and a pipeline dir with manifest.json."""
    pipeline_dir = root / "twin_pipeline" / "pipeline_X"
    eval_dir = root / "evaluation"
    pipeline_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (eval_dir / "eval_twin_summary.json").write_text(json.dumps({
        "pipeline_dir": str(pipeline_dir),
        "gbm_run_dir": str(root / "gbm"),
        "twin_run_dir": str(root / "twin"),
        "csv_path": csv_path,
    }), encoding="utf-8")
    (pipeline_dir / "manifest.json").write_text(json.dumps({
        "source_label": source_label,
        "twin_final_run_dir": str(root / "twin"),
        "gbm_run_dir": str(root / "gbm"),
    }), encoding="utf-8")
    return root


def test_db_run_has_no_source_csv(tmp_path):
    """DB run: csv_path=None + label 'Cosmos MBSCohort' -> source_csv is None
    (NOT the label), so the cohort figures fall back to the database."""
    root = _write_run(tmp_path / "db", csv_path=None, source_label="Cosmos MBSCohort")
    art = A.resolve(root)
    assert art.source_csv is None


def test_csv_run_keeps_explicit_csv_path(tmp_path):
    """CSV run: summary.csv_path is authoritative even if the file is absent at
    resolve() time (consumers resolve relative paths against CWD themselves)."""
    root = _write_run(tmp_path / "csv", csv_path="/data/export.csv",
                      source_label="/data/export.csv")
    art = A.resolve(root)
    assert art.source_csv == "/data/export.csv"


def test_label_used_only_when_it_is_a_real_file(tmp_path):
    """With no csv_path, a source_label is honored only if it points to a real
    file; a bare label that is not a path stays None."""
    real = tmp_path / "cohort.csv"
    real.write_text("PatKey\n1\n", encoding="utf-8")
    root_ok = _write_run(tmp_path / "ok", csv_path=None, source_label=str(real))
    assert A.resolve(root_ok).source_csv == str(real)

    root_label = _write_run(tmp_path / "lbl", csv_path=None, source_label="Cosmos MBSCohort")
    assert A.resolve(root_label).source_csv is None


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        test_db_run_has_no_source_csv(base / "a")
        test_csv_run_keeps_explicit_csv_path(base / "b")
        test_label_used_only_when_it_is_a_real_file(base / "c")
    print("ALL OK")
