"""Unit tests for causal_tte.py (PART B: target-trial-emulation estimators).

Runnable two ways:
  * pytest:   OMP_NUM_THREADS=1 ../mbsaqip_flow/.venv/bin/python -m pytest test_causal_tte.py -q
  * direct :  OMP_NUM_THREADS=1 ../mbsaqip_flow/.venv/bin/python test_causal_tte.py   (-> "ALL OK")

The synthetic tests (e_value / aipw double-robustness / c-for-benefit / iptw / smd) need no data.
The last two tests import evaluate_twin (pulls in torch) and load the 52-row fake cohort.
"""
from __future__ import annotations

import warnings
from math import sqrt
from pathlib import Path

import numpy as np

import causal_tte as ct

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
# E-value
# --------------------------------------------------------------------------------------------
def test_e_value_null():
    assert abs(ct.e_value(1.0)["e_point"] - 1.0) < 1e-12


def test_e_value_two():
    assert abs(ct.e_value(2.0)["e_point"] - (2.0 + sqrt(2.0))) < 1e-9


def test_e_value_symmetric_below_one():
    # RR and 1/RR give the same E-value (the function inverts protective RRs).
    assert abs(ct.e_value(0.5)["e_point"] - ct.e_value(2.0)["e_point"]) < 1e-12


def test_e_value_bound():
    # CI crossing the null -> no confounding needed to explain it away -> bound == 1.0
    assert ct.e_value(1.5, lo=0.8, hi=2.0)["e_bound"] == 1.0
    # CI entirely above 1 -> bound uses the limit nearest the null (the lower one)
    r = ct.e_value(2.0, lo=1.3, hi=3.0)
    assert abs(r["e_bound"] - (1.3 + sqrt(1.3 * 0.3))) < 1e-9
    # CI entirely below 1 -> bound uses the upper limit (nearest the null), inverted
    r2 = ct.e_value(0.5, lo=0.3, hi=0.8)
    inv = 1.0 / 0.8
    assert abs(r2["e_bound"] - (inv + sqrt(inv * (inv - 1.0)))) < 1e-9
    # no CI supplied -> bound is None
    assert ct.e_value(2.0)["e_bound"] is None


def test_smd_to_rr():
    assert abs(ct.smd_to_rr(0.0) - 1.0) < 1e-12
    assert abs(ct.smd_to_rr(1.0) - float(np.exp(0.91))) < 1e-12


def test_benchmark_vs_rct():
    overlap = ct.benchmark_vs_rct(1.4, (1.2, 1.6), "t2d_remission")
    assert overlap["overlaps_rct_ci"] is True
    disjoint = ct.benchmark_vs_rct(5.0, (4.5, 5.5), "t2d_remission")
    assert disjoint["overlaps_rct_ci"] is False


# --------------------------------------------------------------------------------------------
# AIPW double robustness
# --------------------------------------------------------------------------------------------
def _synthetic_causal(n=5000, seed=0, censor=True):
    """One confounder L1 drives BOTH treatment (confounding) and outcome. Correct outcome model
    is mu1=f1(L), mu0=f0(L). Returns everything aipw needs plus the finite-sample true ATE."""
    rng = np.random.default_rng(seed)
    L1 = rng.standard_normal(n)
    ps_true = 1.0 / (1.0 + np.exp(-0.8 * L1))
    A = (rng.random(n) < ps_true).astype(int)
    f0 = 1.0 + 0.5 * L1
    f1 = f0 + (2.0 + 0.3 * L1)  # heterogeneous effect; mean effect ~ 2.0
    y1 = f1 + rng.standard_normal(n)
    y0 = f0 + rng.standard_normal(n)
    y_full = np.where(A == 1, y1, y0)
    if censor:
        pc_true = 1.0 / (1.0 + np.exp(-(1.2 + 0.5 * L1)))  # mostly observed, depends on L only
        delta = (rng.random(n) < pc_true).astype(int)
    else:
        pc_true = np.ones(n)
        delta = np.ones(n, dtype=int)
    y_obs = np.where(delta == 1, y_full, np.nan)
    return {
        "L1": L1, "A": A, "Y": y_obs, "delta": delta,
        "ps_true": ps_true, "pc_true": pc_true,
        "mu1": f1.copy(), "mu0": f0.copy(),
        "true_ate": float(np.mean(f1 - f0)),
    }


def test_aipw_dr_correct_nuisances():
    d = _synthetic_causal(seed=0)
    res = ct.aipw(d["Y"], d["A"], d["delta"], d["ps_true"], d["pc_true"], d["mu1"], d["mu0"])
    tol = max(0.1, 4.0 * res["se"])
    assert abs(res["ate"] - d["true_ate"]) < tol
    lo, hi = res["ci"]
    assert lo < res["ate"] < hi


def test_aipw_dr_robust_to_wrong_ps():
    # mu1/mu0 correct but the propensity model is mildly wrong (shrunk toward 0.5).
    d = _synthetic_causal(seed=1)
    ps_wrong = np.clip(0.5 + 0.6 * (d["ps_true"] - 0.5), 0.02, 0.98)
    res = ct.aipw(d["Y"], d["A"], d["delta"], ps_wrong, d["pc_true"], d["mu1"], d["mu0"])
    tol = max(0.1, 4.0 * res["se"])
    assert abs(res["ate"] - d["true_ate"]) < tol


def test_aipw_tiny_n_guarded():
    z = np.zeros(5)
    res = ct.aipw(z, z, z, z + 0.5, z + 1.0, z, z)
    assert np.isnan(res["ate"]) and np.isnan(res["se"])
    assert all(np.isnan(v) for v in res["ci"])


# --------------------------------------------------------------------------------------------
# c-for-benefit
# --------------------------------------------------------------------------------------------
def test_c_for_benefit_no_heterogeneity():
    rng = np.random.default_rng(1)
    n = 600
    A = (rng.random(n) < 0.5).astype(int)
    ps = np.full(n, 0.5)  # randomized -> constant PS
    y0 = rng.standard_normal(n)
    y1 = y0 + 2.0  # constant benefit -> NO true heterogeneity
    Y = np.where(A == 1, y1, y0)
    pred_ite = rng.standard_normal(n)  # uninformative predicted ITE
    res = ct.c_for_benefit(pred_ite, A, Y, ps, lower_is_better=True)
    assert res["n_pairs"] > 0
    assert abs(res["c_for_benefit"] - 0.5) < 0.1


def test_c_for_benefit_empty_arm():
    n = 20
    A = np.ones(n, dtype=int)  # no controls
    res = ct.c_for_benefit(np.zeros(n), A, np.zeros(n), np.full(n, 0.5))
    assert np.isnan(res["c_for_benefit"]) and res["n_pairs"] == 0


# --------------------------------------------------------------------------------------------
# Weights + balance
# --------------------------------------------------------------------------------------------
def test_stabilized_iptw():
    rng = np.random.default_rng(0)
    n = 300
    ps = rng.uniform(0.0, 1.0, n)
    ps[0], ps[1] = 0.001, 0.999  # extremes must be trimmed
    A = (rng.random(n) < ps).astype(int)
    sw, keep = ct.stabilized_iptw(A, ps)
    assert sw.shape == (n,) and np.all(np.isfinite(sw))
    assert keep.shape == (n,)
    assert not keep[0] and not keep[1]
    ess = ct.weighted_effective_sample_size(sw[keep])
    assert 0.0 < ess <= keep.sum() + 1e-9


def test_weighted_effective_sample_size():
    assert abs(ct.weighted_effective_sample_size(np.ones(50)) - 50.0) < 1e-9
    ess = ct.weighted_effective_sample_size(np.array([1.0, 1, 1, 1, 10.0]))
    assert 0.0 < ess <= 5.0 + 1e-9
    assert np.isnan(ct.weighted_effective_sample_size(np.array([])))


def test_standardized_mean_diff():
    rng = np.random.default_rng(0)
    n = 2000
    A = (rng.random(n) < 0.5).astype(int)
    col_same = rng.standard_normal(n)  # identical distribution across arms -> SMD ~ 0
    col_diff = 2.0 * A + rng.standard_normal(n)  # ~2 SD shift between arms -> large SMD
    L = np.column_stack([col_same, col_diff])
    smd = ct.standardized_mean_diff(L, A)
    assert abs(smd[0]) < 0.15
    assert abs(smd[1]) > 0.8


def test_standardized_mean_diff_tiny_n_guard():
    L = np.zeros((4, 3))
    A = np.array([0, 1, 0, 1])
    smd = ct.standardized_mean_diff(L, A)
    assert smd.shape == (3,) and np.all(np.isnan(smd))


# --------------------------------------------------------------------------------------------
# Real fake cohort (imports evaluate_twin -> torch)
# --------------------------------------------------------------------------------------------
def test_build_L_A_fake_cohort():
    ds = _load_fake()
    L, A, names = ct.build_L_A(ds)
    assert "surgery_idx" not in names
    assert set(np.unique(A)).issubset({0, 1})
    assert L.shape[0] == len(A)
    assert L.shape[1] == len(names)
    for svi in ("SviOverall", "SviHousehold", "SviTransportation", "SviMinority", "SviSES"):
        assert svi in names, f"missing SVI confounder {svi}"
    assert "RUCA_code" in names and "CoverageClass_code" in names


def test_propensity_and_censoring_fake_cohort():
    import gbm_mace_baseline as gb

    ds = _load_fake()
    L, A, _ = ct.build_L_A(ds)
    splits = gb.make_splits(ds, gb.GBMConfig())
    n_test = len(splits["test"])

    ps, _clf = ct.propensity_scores(L, A, splits["train"], splits["test"])
    assert ps.shape == (n_test,) and np.all(np.isfinite(ps))

    sw, keep = ct.stabilized_iptw(A[splits["test"]], ps)
    assert sw.shape == (n_test,) and np.all(np.isfinite(sw))
    assert keep.shape == (n_test,)

    observed = ds.mask[:, 5] > 0.5  # some horizon's observation indicator
    p_obs, ipcw = ct.censoring_model(L, observed, splits["train"], splits["test"])
    assert p_obs.shape == (n_test,) and np.all(np.isfinite(p_obs))
    assert ipcw.shape == (n_test,) and np.all(np.isfinite(ipcw))


# --------------------------------------------------------------------------------------------
# Direct runner
# --------------------------------------------------------------------------------------------
def _run_all():
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = 0
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")
    print("ALL OK")


if __name__ == "__main__":
    _run_all()
