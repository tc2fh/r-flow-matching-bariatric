"""Tests for distributional_metrics.py (Part A of the TTE + distributional build).

Pure synthetic arrays - no twin, no data files, no torch logic - so this runs fast and
covers the plan's Part E acceptance items for Part A plus the extra checks the task lists.

Runnable two ways:
  * pytest:  OMP_NUM_THREADS=1 ../mbsaqip_flow/.venv/bin/python -m pytest test_distributional_metrics.py -q
  * script:  OMP_NUM_THREADS=1 ../mbsaqip_flow/.venv/bin/python test_distributional_metrics.py   -> prints "ALL OK"
"""

from __future__ import annotations

import numpy as np

import distributional_metrics as dm


# --------------------------------------------------------------------------- #
# Weighting helper
# --------------------------------------------------------------------------- #
def test_wmean_uniform_equals_nanmean():
    v = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    ref = float(np.nanmean(v))
    assert abs(dm._wmean(v) - ref) < 1e-12
    assert abs(dm._wmean(v, np.ones_like(v)) - ref) < 1e-12
    # a positive but non-uniform weight still averages, and a 0 weight drops the entry
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    assert abs(dm._wmean(v, w) - float(np.nanmean(v[:4]))) < 1e-12


# --------------------------------------------------------------------------- #
# A1. log score / energy / variogram
# --------------------------------------------------------------------------- #
def test_log_score_gaussian_finite_and_masks_nan():
    rng = np.random.default_rng(0)
    n, m = 60, 200
    obs = rng.normal(size=n)
    samples = obs[:, None] + rng.normal(scale=1.0, size=(n, m))
    obs_nan = obs.copy()
    obs_nan[:5] = np.nan
    scalar, per = dm.log_score_gaussian(samples, obs_nan)
    assert np.isfinite(scalar)
    assert per.shape == (n,)
    assert np.all(np.isnan(per[:5]))          # unobserved -> NaN, dropped from the mean
    assert np.all(np.isfinite(per[5:]))
    # weighting with uniform weights reproduces the unweighted headline
    scalar_w, _ = dm.log_score_gaussian(samples, obs_nan, w=np.ones(n))
    assert abs(scalar_w - scalar) < 1e-9


def test_energy_score_block_finite_and_skips_missing():
    rng = np.random.default_rng(1)
    n, m, k = 30, 80, 3
    truth = rng.normal(size=(n, k))
    block = truth[:, None, :] + rng.normal(scale=0.5, size=(n, m, k))
    obs = truth.copy()
    obs[5, 1] = np.nan                        # one missing horizon in row 5
    scalar, es = dm.energy_score_block(block, obs)
    assert np.isfinite(scalar)
    assert np.isnan(es[5])                     # incomplete block row -> NaN, skipped
    assert np.isfinite(es[0])
    assert es.shape == (n,)


def test_variogram_score_block_finite_and_skips_missing():
    rng = np.random.default_rng(2)
    n, m, k = 25, 70, 4
    truth = rng.normal(size=(n, k))
    block = truth[:, None, :] + rng.normal(scale=0.4, size=(n, m, k))
    obs = truth.copy()
    obs[3, 2] = np.nan
    scalar, vs = dm.variogram_score_block(block, obs)
    assert np.isfinite(scalar)
    assert np.isnan(vs[3])
    assert np.isfinite(vs[0])


# --------------------------------------------------------------------------- #
# A2. interval score / coverage curve / pinball / sharpness
# --------------------------------------------------------------------------- #
def test_interval_score_tightens_around_obs():
    rng = np.random.default_rng(3)
    n, m = 50, 400
    obs = rng.normal(size=n)
    wide = obs[:, None] + rng.normal(scale=3.0, size=(n, m))
    tight = obs[:, None] + rng.normal(scale=0.5, size=(n, m))
    s_wide, _ = dm.interval_score(wide, obs)
    s_tight, _ = dm.interval_score(tight, obs)
    assert np.isfinite(s_wide) and np.isfinite(s_tight)
    assert s_tight < s_wide                    # correct tightening lowers the score


def test_pinball_loss_nonnegative():
    rng = np.random.default_rng(4)
    n, m = 80, 300
    obs = rng.normal(size=n)
    samples = obs[:, None] + rng.normal(size=(n, m))
    pin = dm.pinball_loss(samples, obs)
    assert set(pin.keys()) == {0.1, 0.25, 0.5, 0.75, 0.9}
    assert all(np.isfinite(v) and v >= 0 for v in pin.values())


def test_sharpness_positive():
    rng = np.random.default_rng(5)
    samples = rng.normal(size=(120, 200))
    sh = dm.sharpness(samples)
    assert sh["mean_sd"] > 0
    assert sh["mean_width90"] > 0


def test_coverage_curve_structure_and_calibration():
    rng = np.random.default_rng(6)
    n, m = 800, 400
    obs = rng.normal(size=n)                    # obs ~ N(0,1)
    samples = rng.normal(size=(n, m))           # predictive N(0,1) -> calibrated
    rows = dm.coverage_curve(samples, obs)
    assert len(rows) == 4
    for r in rows:
        assert 0.0 <= r["empirical"] <= 1.0
        assert r["mean_width"] > 0
    r90 = next(r for r in rows if r["nominal"] == 0.9)
    assert abs(r90["empirical"] - 0.9) < 0.1    # roughly calibrated
    # weighted path runs and stays in range
    w = rng.uniform(0.5, 1.5, n)
    rows_w = dm.coverage_curve(samples, obs, w=w)
    assert all(0.0 <= r["empirical"] <= 1.0 for r in rows_w)


# --------------------------------------------------------------------------- #
# A3. threshold calibration / brier decomposition
# --------------------------------------------------------------------------- #
def test_threshold_calibration_ranges_and_perfect():
    rng = np.random.default_rng(7)
    n = 6000
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.random(n) < p).astype(float)       # perfectly calibrated by construction
    res = dm.threshold_calibration(p, y, n_bins=10)
    assert 0.0 <= res["ece"] <= 1.0
    assert 0.0 <= res["brier"] <= 1.0
    assert res["ece"] < 0.05                     # ECE ~ 0 for a calibrated forecast
    assert len(res["table"]) > 0
    # a badly miscalibrated forecast (predict the opposite) has large ECE
    bad = dm.threshold_calibration(1.0 - p, y, n_bins=10)
    assert bad["ece"] > res["ece"]


def test_brier_decomposition_identity():
    rng = np.random.default_rng(8)
    n = 4000
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.random(n) < p).astype(float)
    bd = dm.brier_decomposition(p, y, n_bins=10)
    assert np.isfinite(bd["reliability"]) and bd["reliability"] >= 0
    assert np.isfinite(bd["resolution"]) and bd["resolution"] >= 0
    ybar = float(y.mean())
    assert abs(bd["uncertainty"] - ybar * (1 - ybar)) < 1e-9
    assert 0.0 <= bd["brier"] <= 1.0


# --------------------------------------------------------------------------- #
# A4. calibration slope + CITL
# --------------------------------------------------------------------------- #
def test_calibration_slope_intercept_finite_and_guard():
    rng = np.random.default_rng(9)
    n = 400
    x = rng.normal(size=n)
    p_true = 1.0 / (1.0 + np.exp(-1.5 * x))     # well-separated, not perfectly separable
    y = (rng.random(n) < p_true).astype(float)
    res = dm.calibration_slope_intercept(p_true, y)
    assert np.isfinite(res["slope"]) and res["slope"] > 0
    assert np.isfinite(res["citl"])
    # NaN guard below n=10
    guard = dm.calibration_slope_intercept(p_true[:5], y[:5])
    assert np.isnan(guard["slope"]) and np.isnan(guard["citl"])
    # single-class guard
    single = dm.calibration_slope_intercept(p_true, np.zeros(n))
    assert np.isnan(single["slope"])


# --------------------------------------------------------------------------- #
# A5. IPCW weights + stratified calibration
# --------------------------------------------------------------------------- #
def test_ipcw_from_model():
    p = np.array([0.5, 0.5, 0.5, 0.5])
    mask = np.array([True, True, False, True])
    w = dm.ipcw_from_model(p, mask, clip=0.05)
    assert w[0] == 2.0                           # observed, p=0.5 -> 1/0.5 = 2 (before clip)
    assert w[2] == 0.0                           # unobserved -> 0
    # clip floors tiny probabilities
    w2 = dm.ipcw_from_model(np.array([0.01]), np.array([True]), clip=0.05)
    assert abs(w2[0] - 1.0 / 0.05) < 1e-9


def test_stratified_calibration_runs_weighted():
    rng = np.random.default_rng(10)
    n = 500
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.random(n) < p).astype(float)
    long_fu = rng.random(n) < 0.6
    sc = dm.stratified_calibration(p, y, long_fu, n_bins=10)
    assert "long_fu" in sc and "short_fu" in sc
    assert "ece" in sc["long_fu"] and "ece" in sc["short_fu"]
    # weighted path must not raise a shape mismatch (the plan's snippet bug)
    w = rng.uniform(0.5, 1.5, n)
    sc_w = dm.stratified_calibration(p, y, long_fu, n_bins=10, w=w)
    assert np.isfinite(sc_w["long_fu"]["ece"]) or np.isnan(sc_w["long_fu"]["ece"])


# --------------------------------------------------------------------------- #
# A6. marginal distance
# --------------------------------------------------------------------------- #
def test_marginal_distance_identical_and_shifted():
    rng = np.random.default_rng(11)
    a = rng.normal(size=300)
    same = dm.marginal_distance(a, a.copy())
    assert abs(same["wasserstein1"]) < 1e-9      # identical -> 0
    assert abs(same["ks_stat"]) < 1e-9
    assert abs(same["median_shift"]) < 1e-9
    shifted = dm.marginal_distance(a + 3.0, a)    # sim = obs + 3
    assert shifted["wasserstein1"] > 0
    assert abs(shifted["wasserstein1"] - 3.0) < 0.3
    assert abs(shifted["median_shift"] - 3.0) < 0.3


# --------------------------------------------------------------------------- #
# Convenience bundle
# --------------------------------------------------------------------------- #
def test_proper_scores_report():
    rng = np.random.default_rng(12)
    n, m = 70, 200
    obs = rng.normal(size=n)
    samples = obs[:, None] + rng.normal(scale=1.0, size=(n, m))
    obs_nan = obs.copy()
    obs_nan[:5] = np.nan                          # 5 unobserved -> n_obs = 65
    rep = dm.proper_scores_report(samples, obs_nan)
    assert rep["n"] == n - 5
    assert np.isfinite(rep["crps"]) and rep["crps"] >= 0
    assert np.isfinite(rep["log_score"])
    assert np.isfinite(rep["interval_score"])
    assert all(np.isfinite(v) for v in rep["pinball"].values())
    assert rep["sharpness"]["mean_sd"] > 0
    # explicit mask that further restricts the observed set is honoured
    mask = np.ones(n, bool)
    mask[:20] = False
    rep2 = dm.proper_scores_report(samples, obs, mask=mask)
    assert rep2["n"] == n - 20


# --------------------------------------------------------------------------- #
# Small-n guards (never raise; return NaN structures)
# --------------------------------------------------------------------------- #
def test_small_n_guards_return_nan():
    rng = np.random.default_rng(13)
    small = rng.normal(size=(5, 30))
    obs_s = rng.normal(size=5)
    s_ls, arr_ls = dm.log_score_gaussian(small, obs_s)
    assert np.isnan(s_ls) and arr_ls.shape == (5,)
    s_is, _ = dm.interval_score(small, obs_s)
    assert np.isnan(s_is)
    blk = rng.normal(size=(5, 30, 3))
    ob = rng.normal(size=(5, 3))
    assert np.isnan(dm.energy_score_block(blk, ob)[0])
    assert np.isnan(dm.variogram_score_block(blk, ob)[0])
    assert np.isnan(dm.sharpness(small)["mean_sd"])
    pin = dm.pinball_loss(small, obs_s)
    assert all(np.isnan(v) for v in pin.values())
    rows = dm.coverage_curve(small, obs_s)
    assert all(np.isnan(r["empirical"]) for r in rows)
    tc = dm.threshold_calibration(rng.uniform(size=5), (rng.random(5) < 0.5).astype(float))
    assert np.isnan(tc["ece"]) and tc["table"] == []
    bd = dm.brier_decomposition(rng.uniform(size=5), (rng.random(5) < 0.5).astype(float))
    assert np.isnan(bd["brier"])
    md = dm.marginal_distance(rng.normal(size=5), rng.normal(size=5))
    assert np.isnan(md["wasserstein1"])
    rep = dm.proper_scores_report(small, obs_s)
    assert np.isnan(rep["crps"]) and rep["n"] == 5


# --------------------------------------------------------------------------- #
# IPCW-vs-naive gap plumbing: the per-patient arrays let a caller reweight
# --------------------------------------------------------------------------- #
def test_weighting_changes_headline_via_per_patient_array():
    rng = np.random.default_rng(14)
    n, m = 200, 200
    obs = rng.normal(size=n)
    samples = obs[:, None] + rng.normal(scale=1.0, size=(n, m))
    _, per = dm.interval_score(samples, obs)
    # emphasise the worst-scoring half; weighted headline should exceed the naive mean
    w = np.where(per > np.nanmedian(per), 2.0, 0.5)
    naive = dm._wmean(per)
    weighted = dm._wmean(per, w)
    assert weighted > naive


ALL_TESTS = [
    test_wmean_uniform_equals_nanmean,
    test_log_score_gaussian_finite_and_masks_nan,
    test_energy_score_block_finite_and_skips_missing,
    test_variogram_score_block_finite_and_skips_missing,
    test_interval_score_tightens_around_obs,
    test_pinball_loss_nonnegative,
    test_sharpness_positive,
    test_coverage_curve_structure_and_calibration,
    test_threshold_calibration_ranges_and_perfect,
    test_brier_decomposition_identity,
    test_calibration_slope_intercept_finite_and_guard,
    test_ipcw_from_model,
    test_stratified_calibration_runs_weighted,
    test_marginal_distance_identical_and_shifted,
    test_proper_scores_report,
    test_small_n_guards_return_nan,
    test_weighting_changes_headline_via_per_patient_array,
]


if __name__ == "__main__":
    failures = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {t.__name__}: {exc!r}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("ALL OK")
