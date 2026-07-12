"""End-to-end regression tests for missing outcomes in distributional evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_tte
import train_flow_matching as fm


def test_distributional_artifacts_exclude_zero_filled_missing_outcomes(tmp_path):
    """Exercise the artifact-writing path used by a frozen study evaluation.

    The loader represents an unobserved continuous outcome as ``x=0, mask=0``. Those
    sentinel zeros must never enter coverage, joint scores, or marginal-distance
    diagnostics as observed BMI values.
    """
    n_patients = 40
    n_observed = 30
    n_samples = 80
    bmi_dim = fm.TARGET_NAMES.index("bmi_12m")
    hba1c_dim = fm.TARGET_NAMES.index("hba1c_12m")

    x = np.zeros((n_patients, fm.X_DIM), dtype=np.float64)
    mask = np.zeros_like(x)
    x[:n_observed, bmi_dim] = 30.0
    x[:n_observed, hba1c_dim] = 6.0
    mask[:n_observed, [bmi_dim, hba1c_dim]] = 1.0

    fac = np.zeros((n_patients, n_samples, fm.X_DIM), dtype=np.float64)
    fac[:, :, bmi_dim] = np.linspace(29.0, 31.0, n_samples)
    fac[:, :, hba1c_dim] = np.linspace(5.5, 6.5, n_samples)

    ctx = {
        "dataset": SimpleNamespace(x=x, mask=mask),
        "test_idx": np.arange(n_patients),
        "n_test": n_patients,
        "fac": fac,
        "output_dir": tmp_path,
        "with_causal": False,
        "gbm_cfg": SimpleNamespace(split_strategy="surgery"),
        "bmi_dist": ["bmi_12m"],
        "hba1c_dist": ["hba1c_12m"],
        "cov_horizons": ["bmi_12m"],
        "thr_cal_families": [],
        "attrition_horizons": [],
        "modec_horizons": ["bmi_12m"],
    }

    run_tte._run_distributional(ctx)

    coverage = pd.read_csv(tmp_path / "dist_coverage_curve_test.csv")
    assert coverage["empirical_naive"].tolist() == [1.0] * len(run_tte.COVERAGE_LEVELS)

    proper = pd.read_csv(tmp_path / "dist_proper_scores_test.csv")
    bmi = proper.loc[proper["horizon"].eq("bmi_12m")].iloc[0]
    assert bmi["n_obs"] == n_observed
    assert bmi["energy_block_naive"] < 1.0

    marginal = pd.read_csv(tmp_path / "dist_modeC_marginal_distance.csv").iloc[0]
    assert marginal["n_obs"] == n_observed
    assert marginal["wasserstein1"] < 1.0
