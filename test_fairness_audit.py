"""Unit tests for fairness_audit.py (Wave-2 Agent D: SVI/RUCA/race equity subgroup audit).

Runnable two ways:
  * pytest:  OMP_NUM_THREADS=1 ../mbsaqip_flow/.venv/bin/python -m pytest test_fairness_audit.py -q
  * direct:  OMP_NUM_THREADS=1 ../mbsaqip_flow/.venv/bin/python test_fairness_audit.py   (-> "ALL OK")

These exercise the PURE, unit-testable helpers (svi_quartiles / ruca_rural_urban / race_bucket /
insurance_bucket / compute_gap) on synthetic inputs, plus the axis builders on the 52-row fake
cohort. The full run() needs a trained twin and is exercised by the end-to-end smoke, not here
(the last test imports evaluate_twin, which pulls in torch, and loads the fake FlowDataset).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import fairness_audit as fa

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
FAKE_CSV = HERE / "fake_data" / "fake_mbs_cohort.csv"

_FAKE: dict = {}


def _load_fake():
    """Load + cache the fake FlowDataset (imports evaluate_twin -> torch; fine in the venv)."""
    if "ds" not in _FAKE:
        import evaluate_twin as ev

        _FAKE["ds"] = ev.load_dataset(FAKE_CSV)
    return _FAKE["ds"]


# --------------------------------------------------------------------------------------------
# svi_quartiles
# --------------------------------------------------------------------------------------------
def test_svi_quartiles_balanced():
    # A uniform 0-1 vector splits into four roughly balanced bins Q1..Q4.
    rng = np.random.default_rng(0)
    v = rng.uniform(0.0, 1.0, size=400)
    labels = fa.svi_quartiles(v)
    assert labels.shape[0] == v.size
    counts = {q: int(np.sum(labels == q)) for q in ("Q1", "Q2", "Q3", "Q4")}
    assert sum(counts.values()) == v.size, "every finite SVI value is assigned a quartile"
    for q, c in counts.items():
        assert 80 <= c <= 120, f"{q} unbalanced: {c} (expected ~100)"
    # Ordering: Q1 is the lowest-SVI (least vulnerable), Q4 the highest (most vulnerable).
    assert v[labels == "Q1"].max() <= v[labels == "Q4"].min() + 1e-9


def test_svi_quartiles_monotone_input():
    # Deterministic ordered input: evenly spaced -> 4 contiguous, balanced bins.
    v = np.linspace(0.0, 1.0, 40)
    labels = fa.svi_quartiles(v)
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert 8 <= int(np.sum(labels == q)) <= 12


def test_svi_quartiles_guard_small_n():
    # Fewer than 8 finite values -> all "unknown" (quartiles are meaningless).
    v = np.array([0.1, 0.5, np.nan, 0.9, 0.3])
    labels = fa.svi_quartiles(v)
    assert list(np.unique(labels)) == ["unknown"]


def test_svi_quartiles_nan_excluded():
    v = np.concatenate([np.linspace(0, 1, 20), [np.nan, np.nan]])
    labels = fa.svi_quartiles(v)
    assert labels[-1] == "unknown" and labels[-2] == "unknown"
    assert set(np.unique(labels[:20])) <= {"Q1", "Q2", "Q3", "Q4"}


# --------------------------------------------------------------------------------------------
# ruca_rural_urban
# --------------------------------------------------------------------------------------------
def test_ruca_mapping():
    vals = ["1.0", "1 metropolitan", "3", "4 micropolitan", "7", "7 small town",
            "10 rural", "x", "", None, np.nan, "11"]
    labels = fa.ruca_rural_urban(vals)
    expected = ["urban", "urban", "urban", "rural", "rural", "rural",
                "rural", "unknown", "unknown", "unknown", "unknown", "unknown"]
    assert list(labels) == expected


def test_ruca_int_part_of_leading_number():
    # "int part of the leading number": 1.x -> urban, 7.x -> rural.
    assert fa.ruca_rural_urban(["1.1"])[0] == "urban"
    assert fa.ruca_rural_urban(["7.2"])[0] == "rural"
    assert fa.ruca_rural_urban(["10.6"])[0] == "rural"


# --------------------------------------------------------------------------------------------
# race_bucket  (AUDIT ONLY)
# --------------------------------------------------------------------------------------------
def test_race_bucket_representative_spellings():
    vals = ["White", "white", "Caucasian",
            "Black or African American", "African American",
            "Hispanic or Latino", "White Hispanic",
            "Asian", "Asian Indian",
            "American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander",
            "Other", "Unknown", "", None]
    labels = fa.race_bucket(vals)
    expected = ["White", "White", "White",
                "Black", "Black",
                "Hispanic", "Hispanic",  # ethnicity wins over race token
                "Asian", "Asian",
                "Other", "Other",
                "Other", "unknown", "unknown", "unknown"]
    assert list(labels) == expected


def test_race_caucasian_not_miscoded_asian():
    # "caucasian" contains the substring "asian"; the White rule must win (order matters).
    assert fa.race_bucket(["Caucasian"])[0] == "White"


# --------------------------------------------------------------------------------------------
# insurance_bucket
# --------------------------------------------------------------------------------------------
def test_insurance_bucket():
    vals = ["Commercial", "Medicare", "Medicaid", "Self-pay", "Self Pay",
            "Uninsured", "Private", "Other", "", None]
    labels = fa.insurance_bucket(vals)
    expected = ["Commercial", "Medicare", "Medicaid", "Uninsured", "Uninsured",
                "Uninsured", "Commercial", "Other", "unknown", "unknown"]
    assert list(labels) == expected


# --------------------------------------------------------------------------------------------
# compute_gap
# --------------------------------------------------------------------------------------------
def test_compute_gap_known_table():
    table = {"Q1": 0.90, "Q2": 0.80, "Q3": 0.70, "Q4": 0.60}
    # most-vulnerable (Q4) minus least-vulnerable (Q1) = 0.60 - 0.90 = -0.30 (adverse coverage).
    assert abs(fa.compute_gap(table, "Q4", "Q1") - (-0.30)) < 1e-12


def test_compute_gap_missing_or_nan():
    assert np.isnan(fa.compute_gap({"Q1": 0.9}, "Q4", "Q1"))          # Q4 missing
    assert np.isnan(fa.compute_gap({"Q1": np.nan, "Q4": 0.6}, "Q4", "Q1"))  # Q1 NaN


# --------------------------------------------------------------------------------------------
# Axis builders on the fake cohort (aligned to the test split, valid labels)
# --------------------------------------------------------------------------------------------
def _fake_splits(ds):
    import gbm_mace_baseline as gb

    try:
        return gb.make_splits(ds, gb.GBMConfig())
    except Exception:  # pragma: no cover - fallback: treat all rows as the "test" split
        n = int(np.asarray(ds.x).shape[0])
        return {"train": np.arange(0), "val": np.arange(0), "test": np.arange(n)}


def test_build_axes_aligned_and_valid():
    ds = _load_fake()
    splits = _fake_splits(ds)
    test_idx = np.asarray(splits["test"])
    axes = fa.build_axes(ds, test_idx)

    # The fake cohort carries SviOverall, RUCA, FirstRace, CoverageClass -> all four axes.
    for expected_axis in ("svi", "ruca", "race", "insurance"):
        assert expected_axis in axes, f"missing axis {expected_axis}"

    valid = {
        "svi": {"Q1", "Q2", "Q3", "Q4", "unknown"},
        "ruca": {"urban", "rural", "unknown"},
        "race": {"White", "Black", "Hispanic", "Asian", "Other", "unknown"},
        "insurance": {"Commercial", "Medicare", "Medicaid", "Uninsured", "Other", "unknown"},
    }
    for axis, labels in axes.items():
        assert labels.shape[0] == test_idx.size, f"{axis} not aligned to the test split"
        assert set(np.unique(labels)) <= valid[axis], f"{axis} produced invalid labels"
        # At least one real (non-unknown) subgroup must be present in the fake cohort.
        assert any(lbl != "unknown" for lbl in np.unique(labels)), f"{axis} all-unknown"


def test_build_axes_race_never_all_white():
    # Sanity: the fake cohort has minority representation (so gap comparisons are exercisable).
    ds = _load_fake()
    labels = fa.build_axes(ds, np.arange(int(np.asarray(ds.x).shape[0])))["race"]
    buckets = set(np.unique(labels))
    assert "White" in buckets
    assert buckets & {"Black", "Asian", "Other", "Hispanic"}, "no minority buckets found"


# --------------------------------------------------------------------------------------------
# Direct runner
# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print("ALL OK")
