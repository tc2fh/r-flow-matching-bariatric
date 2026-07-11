"""Tests for the quantile trajectory arms (baselines_trajectory.fit_quantile_baselines et al.).

Covers the two models the collaborator asked for as a fair distributional yard-stick for the
flow: quantile gradient boosting (``qgbm``) and linear conditional quantile regression
(``qreg``). Mixes fast synthetic checks (estimator behaviour, monotone rearrangement, the
"quantiles-as-ensemble" coverage identity) with fake-cohort integration checks (shapes,
monotonicity, and that the arms drop into the repo's proper-scoring code with finite CRPS/NLL).

Runnable two ways:
  * pytest:  OMP_NUM_THREADS=1 ../mbsaqip_flow/.venv/bin/python -m pytest test_quantile_baselines.py -q
  * script:  OMP_NUM_THREADS=1 ../mbsaqip_flow/.venv/bin/python test_quantile_baselines.py   -> prints "ALL OK"
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from functools import lru_cache

import numpy as np

import baselines_trajectory as bt
import distributional_metrics as dm
import train_flow_matching as fm
import train_flow_matching_twin as tw

FAKE_CSV = str(fm.REPO_ROOT / "fake_data" / "fake_mbs_cohort.csv")


@lru_cache(maxsize=1)
def _fake_dataset_and_splits():
    dataset = fm.load_dataset_from_csv(FAKE_CSV)
    splits = tw.make_splits(dataset, tw.TwinConfig(split_seed=0))
    return dataset, splits


# --------------------------------------------------------------------------- #
# Quantile grid + monotone rearrangement
# --------------------------------------------------------------------------- #
def test_default_quantiles_valid_and_symmetric():
    q = np.asarray(bt.DEFAULT_QUANTILES)
    assert np.all(np.diff(q) > 0)                          # strictly increasing
    assert q.min() > 0.0 and q.max() < 1.0                 # open interval
    assert 0.5 in set(np.round(q, 6))                      # median present (point estimate)
    # symmetric grid so central bands read off exact pairs (0.05/0.95, 0.025/0.975, ...)
    assert np.allclose(np.sort(q), np.sort(1.0 - q))
    # the coverage-curve nominal levels' tails are all exact grid points
    for c in (0.5, 0.8, 0.9, 0.95):
        assert round((1 - c) / 2, 6) in set(np.round(q, 6))
        assert round(1 - (1 - c) / 2, 6) in set(np.round(q, 6))


def test_monotone_rearrange_sorts_rows_only():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(20, 7))
    out = bt.monotone_rearrange(pred)
    assert out.shape == pred.shape
    assert np.all(np.diff(out, axis=1) >= 0)               # each row non-decreasing
    # rearrangement is a per-row permutation: row multisets are preserved
    assert np.allclose(np.sort(pred, axis=1), out)


# --------------------------------------------------------------------------- #
# Estimator builders (synthetic)
# --------------------------------------------------------------------------- #
def test_quantile_gbm_predicts_all_levels_and_handles_nan():
    rng = np.random.default_rng(1)
    n, p = 200, 4
    X = rng.normal(size=(n, p))
    X[::9, 1] = np.nan                                     # native-NaN routing, like the point xgb arm
    y = 2.0 * X[:, 0] + rng.normal(scale=0.5, size=n)
    q = np.asarray(bt.DEFAULT_QUANTILES)
    model = bt.make_quantile_gbm(q, seed=0)
    model.fit(X, y)
    pred = np.asarray(model.predict(X))
    assert pred.shape == (n, q.size)                       # one column per quantile from ONE model
    assert np.isfinite(pred[::9]).all()                    # NaN rows still predicted


def test_quantile_regressor_pipeline_imputes_and_predicts():
    rng = np.random.default_rng(2)
    n, p = 150, 3
    X = rng.normal(size=(n, p))
    X[::7, 2] = np.nan                                     # QuantileRegressor can't take NaN -> imputer
    y = X[:, 0] - X[:, 1] + rng.normal(scale=0.3, size=n)
    pipe = bt.make_quantile_regressor(0.5, alpha=0.0)
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()


def test_quantiles_as_ensemble_recover_nominal_coverage():
    """The predicted quantile grid, treated as an ensemble, must yield ~nominal coverage.

    Build a perfectly-specified ensemble: for each patient the "samples" ARE the true quantiles
    of that patient's outcome distribution. dm.coverage_curve (the same scorer the comparison
    uses) should then read empirical coverage close to nominal - validating the whole
    quantiles-as-predictive-ensemble scoring path end to end."""
    rng = np.random.default_rng(3)
    n = 4000
    mu = rng.normal(size=n)
    sd = 0.5 + rng.uniform(size=n)
    q = np.asarray(bt.DEFAULT_QUANTILES)
    from scipy.stats import norm
    ens = mu[:, None] + sd[:, None] * norm.ppf(q)[None, :]  # [n, n_q] true per-patient quantiles
    obs = mu + sd * rng.normal(size=n)                      # a genuine draw from each patient
    for row in dm.coverage_curve(ens, obs, levels=(0.5, 0.8, 0.9)):
        assert abs(row["empirical"] - row["nominal"]) < 0.05


# --------------------------------------------------------------------------- #
# fit_quantile_baselines on the fake cohort (integration)
# --------------------------------------------------------------------------- #
def test_fit_quantile_baselines_shapes_and_monotone():
    dataset, splits = _fake_dataset_and_splits()
    q = bt.DEFAULT_QUANTILES
    out = bt.fit_quantile_baselines(dataset, splits, quantiles=q, use_event=True, seed=0)
    n_test = int(splits["test"].size)
    n_q, n_h = len(q), tw.X_CONT_DIM
    for key in ("qgbm_pred", "qreg_pred"):
        pred = out[key]
        assert pred.shape == (n_test, n_q, n_h), key
        # non-decreasing along the quantile axis wherever the column is populated (finite)
        finite_rows = np.all(np.isfinite(pred), axis=1)     # [n_test, n_h]
        diffs = np.diff(pred, axis=1)
        assert np.all(diffs[np.broadcast_to(finite_rows[:, None, :], diffs.shape)] >= -1e-9)
    assert np.array_equal(out["quantiles"], np.asarray(q, dtype=float))
    assert "event" in out["feature_names"]                  # event-conditioned like the flow's Mode-A


def test_no_event_drops_event_feature():
    dataset, splits = _fake_dataset_and_splits()
    out = bt.fit_quantile_baselines(dataset, splits, use_event=False, seed=0)
    assert "event" not in out["feature_names"]


def test_quantile_arms_plug_into_horizon_score_with_density():
    """A quantile arm's [n_test, n_q] slice must score through horizon_score as a real ensemble:
    finite CRPS AND finite NLL (has_density=True), unlike the point arms whose NLL is NaN."""
    dataset, splits = _fake_dataset_and_splits()
    out = bt.fit_quantile_baselines(dataset, splits, seed=0)
    x_cont = dataset.x[:, tw.CONT_DIMS].astype(float)
    mask_cont = dataset.mask[:, tw.CONT_DIMS]
    test_idx = splits["test"]
    got_density = False
    for h in range(tw.X_CONT_DIM):
        score = bt.horizon_score(out["qgbm_pred"][:, :, h], x_cont[test_idx, h], mask_cont[test_idx, h],
                                 has_density=True, obs_bounds=bt.PHYSIOLOGIC_BOUNDS.get(tw.CONT_GROUPS[h]))
        if score["n_obs"] > 0:
            assert np.isfinite(score["crps"])
            if np.isfinite(score["nll"]):
                got_density = True
    assert got_density                                       # at least one horizon yields a finite NLL


def test_quantile_metric_table_covers_both_arms_all_horizons():
    dataset, splits = _fake_dataset_and_splits()
    out = bt.fit_quantile_baselines(dataset, splits, seed=0)
    table = bt.quantile_metric_table(dataset, splits, out)
    assert set(table["arm"]) == {bt.ARM_QGBM, bt.ARM_QREG}
    assert len(table) == 2 * tw.X_CONT_DIM                   # both arms x 15 horizons
    assert {"mad", "rmse", "crps", "nll"}.issubset(table.columns)


def test_fallback_when_horizon_has_too_few_train_obs():
    """A horizon observed in <2 train rows must fall back to a flat empirical-quantile column
    (an unconditional forecast) without raising, and stay monotone/finite."""
    dataset, splits = _fake_dataset_and_splits()
    # Force a pathological split: only the first two patients train, so most horizons are sparse.
    tiny = {"train": splits["train"][:2], "val": splits["val"], "test": splits["test"]}
    out = bt.fit_quantile_baselines(dataset, tiny, seed=0)
    n_test = int(tiny["test"].size)
    assert out["qgbm_pred"].shape[0] == n_test
    # every populated column is still sorted (rearranged / flat) - no crossing, no exception
    diffs = np.diff(out["qreg_pred"], axis=1)
    finite = np.isfinite(diffs)
    assert np.all(diffs[finite] >= -1e-9)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print("ALL OK")


if __name__ == "__main__":
    _run_all()
