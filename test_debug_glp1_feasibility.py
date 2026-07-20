from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import debug_glp1_feasibility as glp_debug


def test_agent_normalization_is_conservative() -> None:
    assert glp_debug.normalize_agent("Wegovy")[:2] == ("semaglutide", "glp1_receptor_agonist")
    assert glp_debug.normalize_agent("Mounjaro")[:2] == ("tirzepatide", "dual_gip_glp1_agonist")
    assert glp_debug.normalize_agent("Soliqua")[:2] == ("lixisenatide", "fixed_combination_incretin")
    assert glp_debug.normalize_agent("<missing>") == ("<missing>", "<missing>", "missing")
    assert glp_debug.normalize_agent("locally coded product") == ("unmapped", "unmapped", "manual_review")


def test_postoperative_timing_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "_glp1_interval_days": np.array([-1, 0, 12, 18, 18.1, 24, 24.1]) * glp_debug.DAYS_PER_MONTH,
            "_postop_glp1": [1, 1, 1, 1, 1, 1, 1],
        }
    )
    assert glp_debug.classify_postop_timing(frame).tolist() == [
        "recorded_before_surgery",
        "initiate_before_12m_landmark",
        "initiate_12_to_18m",
        "initiate_12_to_18m",
        "initiate_after_18_through_24m",
        "initiate_after_18_through_24m",
        "no_initiation_through_24m_then_later",
    ]


def test_small_cell_suppression_removes_counts_and_metrics() -> None:
    frame = pd.DataFrame(
        {
            "arm": ["small", "large"],
            "n_patients": [4, 20],
            "n_rows": [4, 21],
            "mean": [1.5, 2.5],
        }
    )
    protected = glp_debug.privacy_safe(frame, min_cell_size=11)
    assert protected.loc[0, "cell_size_display"] == "<11"
    assert pd.isna(protected.loc[0, "n_patients"])
    assert pd.isna(protected.loc[0, "n_rows"])
    assert pd.isna(protected.loc[0, "mean"])
    assert protected.loc[1, "n_patients"] == 20

    subcell = glp_debug.privacy_safe(
        pd.DataFrame({"n_patients": [20], "n_events": [2], "event_pct": [10.0]}),
        min_cell_size=11,
    )
    assert subcell.loc[0, "n_patients"] == 20
    assert pd.isna(subcell.loc[0, "n_events"])
    assert subcell.loc[0, "n_events_display"] == "<11"
    assert pd.isna(subcell.loc[0, "event_pct"])

    dated = glp_debug.privacy_safe(
        pd.DataFrame(
            {
                "arm": ["small"],
                "n_patients": [1],
                "index_date_min": ["2026-01-01"],
                "index_date_max": ["2026-01-01"],
            }
        ),
        min_cell_size=11,
    )
    assert dated.loc[0, "index_date_min"] == ""
    assert dated.loc[0, "index_date_max"] == ""


def test_privacy_guard_rejects_patient_level_identifier() -> None:
    with pytest.raises(RuntimeError, match="patient-level identifier"):
        glp_debug.ensure_aggregate_only("unsafe.csv", pd.DataFrame({"_patient": ["one"]}))


def test_synthetic_cohorts_run_end_to_end(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent
    args = Namespace(
        mbs_csv=root / "fake_data" / "fake_mbs_cohort.csv",
        glp1_csv=root / "fake_data" / "fake_glp1_cohort.csv",
        schema="dbo",
        mbs_table="MBSCohort",
        glp1_table="GLP1Cohort",
        connection_timeout=1000,
        max_rows=None,
        top_values=100,
        min_cell_size=11,
        output_root=tmp_path,
    )
    run_dir = glp_debug.run(args)

    expected = {
        "database_medication_schema_candidates.csv",
        "design_arm_counts.csv",
        "design_readiness.csv",
        "design_requirements.csv",
        "dose_profiles.csv",
        "exposure_reconciliation.csv",
        "glp1_feasibility_report.txt",
        "manifest.json",
        "medication_tag_values.csv",
        "outcome_support.csv",
        "patient_arm_overlap.csv",
    }
    assert expected.issubset({path.name for path in run_dir.iterdir()})

    arm_counts = pd.read_csv(run_dir / "design_arm_counts.csv")
    assert {
        "expanded_surgery_prognosis",
        "baseline_treatment_comparison",
        "postoperative_incretin_rescue",
        "perioperative_prior_user_strategy",
        "observed_treatment_sequences",
    }.issubset(set(arm_counts["design"]))
    assert not {"PatKey", "PatientID", "_patient"}.intersection(arm_counts.columns)

    tags = pd.read_csv(run_dir / "medication_tag_values.csv")
    semaglutide = tags[tags["raw_value"].astype(str).str.lower().eq("semaglutide")]
    assert set(semaglutide["proposed_ingredient"]) == {"semaglutide"}
    assert tags.loc[tags["small_cell_suppressed"].eq(True), "n_patients"].isna().all()

    readiness = pd.read_csv(run_dir / "design_readiness.csv").set_index("design")
    assert readiness.loc["expanded_surgery_prognosis", "schema_readiness"] == "minimum_fields_present"
    assert readiness.loc["postoperative_incretin_rescue", "schema_readiness"] == "blocked_by_current_extract"

    overlap = pd.read_csv(run_dir / "patient_arm_overlap.csv")
    direct_pairs = overlap[
        overlap["design"].eq("baseline_treatment_comparison") & overlap["record_type"].eq("pairwise")
    ]
    assert any(
        ("MBSCohort:" in str(left) and "GLP1Cohort:" in str(right))
        or ("GLP1Cohort:" in str(left) and "MBSCohort:" in str(right))
        for left, right in zip(direct_pairs["arm_a"], direct_pairs["arm_b"])
    )

    report = (run_dir / "glp1_feasibility_report.txt").read_text(encoding="utf-8")
    assert "<11" in report
    assert "n_patients=nan" not in report
