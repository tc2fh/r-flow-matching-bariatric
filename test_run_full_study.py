"""Focused reproducibility and resume tests for ``run_full_study``."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import run_full_study as study
from study_reproducibility import source_fingerprint


def _manifest(*, cohort: str = "cohort-sha", git_sha: str | None = None,
              source_sha: str | None = None, stable: bool = True) -> dict:
    code: dict = {}
    if source_sha is not None:
        code = {
            "source_fingerprint": {"algorithm": "sha256-path-and-content-v1", "sha256": source_sha},
            "source_stable_during_run": stable,
        }
    return {
        "input": {"sha256": cohort},
        "git": {"sha": git_sha},
        "code": code,
    }


def test_paired_runs_accept_matching_source_fingerprint_without_git():
    identity = study.validate_paired_runs(
        _manifest(source_sha="same-source"),
        _manifest(source_sha="same-source"),
    )
    assert identity == {
        "kind": "source_sha256",
        "value": "same-source",
        "verified": True,
    }


def test_paired_runs_reject_different_source_fingerprints_even_if_git_matches():
    with pytest.raises(SystemExit, match="source fingerprints differ"):
        study.validate_paired_runs(
            _manifest(git_sha="same-git", source_sha="source-a"),
            _manifest(git_sha="same-git", source_sha="source-b"),
        )


def test_paired_runs_legacy_no_git_requires_explicit_override():
    internal = _manifest()
    temporal = _manifest()
    with pytest.raises(SystemExit, match="no Git revision or source fingerprint"):
        study.validate_paired_runs(internal, temporal)

    identity = study.validate_paired_runs(
        internal,
        temporal,
        allow_missing_code_identity=True,
    )
    assert identity["kind"] == "legacy_unverified"
    assert identity["verified"] is False


def test_paired_runs_never_override_a_real_identity_mismatch():
    with pytest.raises(SystemExit, match="source fingerprints differ"):
        study.validate_paired_runs(
            _manifest(source_sha="source-a"),
            _manifest(source_sha="source-b"),
            allow_missing_code_identity=True,
        )


def test_resolve_resume_study_finds_both_freezes_and_shared_csv(tmp_path: Path):
    study_dir = tmp_path / "study_20260711_090459"
    internal = study_dir / "internal_validation" / "20260711_090500"
    temporal = study_dir / "temporal_validation" / "20260711_120000"
    csv_path = tmp_path / "cohort.csv"
    csv_path.write_text("id\n1\n", encoding="utf-8")
    cohort_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    for frozen, split in ((internal, "surgery"), (temporal, "temporal")):
        (frozen / "evaluation").mkdir(parents=True)
        (frozen / "twin_pipeline" / "pipeline_1").mkdir(parents=True)
        manifest = {
            "split": {"split_strategy": split, "split_sizes": {}},
            "input": {
                "source_abspath": str(csv_path),
                "source": str(csv_path),
                "sha256": cohort_sha,
            },
            "git": {"sha": None},
            "code": {},
            "causal_distributional": {"status": "ok", "causal_tte": {
                "status": "ok",
                "artifacts": {
                    "tte_marginal_effects": str(frozen / "evaluation" / "tte_marginal_effects.csv"),
                    "tte_propensity_overlap": str(frozen / "evaluation" / "tte_propensity_overlap.csv"),
                    "tte_covariate_balance_love": str(frozen / "evaluation" / "tte_covariate_balance_love.csv"),
                    "tte_weights_summary": str(frozen / "evaluation" / "tte_weights_summary.json"),
                },
            }},
            "fairness": {"status": "ok"},
            "artifacts": {"figures": {"status": "ok", "main": [], "supplement": []}},
        }
        for artifact in manifest["causal_distributional"]["causal_tte"]["artifacts"].values():
            Path(artifact).write_text("x", encoding="utf-8")
        (frozen / "evaluation" / "dist_calibration_slope_citl.csv").write_text(
            "split_strategy\nsurgery\n", encoding="utf-8"
        )
        (frozen / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    resolved = study.resolve_resume_study(study_dir)
    assert resolved[0] == internal
    assert resolved[2] == temporal
    assert resolved[4] == csv_path


def test_source_fingerprint_is_stable_cross_platform_and_tracks_content(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "b.py").write_text("y = 2\n", encoding="utf-8")
    ignored = tmp_path / "runs"
    ignored.mkdir()
    (ignored / "generated.py").write_text("ignore = True\n", encoding="utf-8")

    first = source_fingerprint(tmp_path)
    second = source_fingerprint(tmp_path)
    assert first == second
    assert first["file_count"] == 2
    assert [row["path"] for row in first["files"]] == ["a.py", "pkg/b.py"]

    (nested / "b.py").write_text("y = 3\n", encoding="utf-8")
    changed = source_fingerprint(tmp_path)
    assert changed["sha256"] != first["sha256"]
