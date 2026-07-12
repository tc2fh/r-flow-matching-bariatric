"""Regression coverage for eligible bariatric surgery CPT mappings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import debug_attrition as attrition
import train_flow_matching as fm


FAKE_CSV = Path(__file__).resolve().parent / "fake_data" / "fake_mbs_cohort.csv"


def _eligible_43645_row() -> pd.DataFrame:
    frame = pd.read_csv(FAKE_CSV, dtype=str, keep_default_na=True)
    row = frame.loc[frame["CptCode"].eq("43644")].iloc[[0]].copy()
    row.loc[:, "PatKey"] = "eligible-cpt-43645"
    row.loc[:, "CptCode"] = "43645"
    return row


def test_cpt_43645_survives_csv_loader_as_rnygb(tmp_path: Path):
    """Reproduce the end-user path: exported row -> CSV loader -> modeled patient."""
    csv_path = tmp_path / "eligible_43645.csv"
    _eligible_43645_row().to_csv(csv_path, index=False)

    dataset = fm.load_dataset_from_csv(csv_path)

    assert dataset.subject_ids.tolist() == ["eligible-cpt-43645"]
    assert dataset.surgery_type.tolist() == ["rnygb"]
    assert dataset.surgery_idx.tolist() == [fm.SURGERY_TO_INDEX["rnygb"]]
    assert dataset.frame["cpt_code_normalized"].tolist() == ["43645"]


def test_cpt_43645_is_not_reported_as_unrecognized_attrition():
    report = attrition.python_attrition(_eligible_43645_row())

    cpt_stage = next(row for row in report["stages"] if row["stage"].startswith("CPT unrecognized"))
    assert cpt_stage["dropped"] == 0
    assert report["stages"][-1]["remaining"] == 1
    assert "43645" not in report["unrecognized_codes"].index


def test_all_supported_cpt_spellings_map_deterministically():
    codes = pd.Series(["43775", "43644", "43846", "43645", "43645.0", " 43645 "])
    assert fm.map_surgery_type(codes).tolist() == [
        "sleeve", "rnygb", "rnygb", "rnygb", "rnygb", "rnygb",
    ]
