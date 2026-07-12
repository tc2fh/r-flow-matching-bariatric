"""Tests for the quantile trajectory arms (baselines_trajectory.fit_quantile_baselines et al.).

Covers the two models the collaborator asked for as a fair distributional yard-stick for the
flow: quantile gradient boosting (``qgbm``) and linear conditional quantile regression
(``qreg``). Mixes fast synthetic checks (estimator behaviour, monotone rearrangement, uniform-
probability quadrature of the nonuniform quantile grid) with fake-cohort integration checks
(shapes, monotonicity, and finite CRPS/NLL through the shared proper-scoring code).

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
import compare_quantile_baselines as comparison
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


def test_quantile_grid_is_interpolated_on_uniform_probability_mass():
    """A nonuniform quantile grid must not assign equal mass to every fitted level."""
    quantiles = np.array([0.10, 0.20, 0.90])
    predictions = np.array([[[0.0], [1.0], [10.0]]])
    n_samples = 200

    ensemble = bt.quantile_grid_to_ensemble(predictions, quantiles, n_samples=n_samples)

    target = (np.arange(n_samples) + 0.5) / n_samples
    expected = np.interp(target, quantiles, predictions[0, :, 0])
    assert ensemble.shape == (1, n_samples, 1)
    assert np.allclose(ensemble[0, :, 0], expected)
    assert not np.isclose(ensemble.mean(), predictions.mean())


def test_horizon_score_preserves_patient_alignment_for_pooled_tests():
    samples = np.array([[0.0, 1.0], [4.0, 5.0], [9.0, 10.0]])
    observed = np.array([0.5, 4.5, 9.5])
    mask = np.array([1, 0, 1])

    score = bt.horizon_score(samples, observed, mask, has_density=True)

    assert score["crps_by_patient"].shape == (3,)
    assert np.isfinite(score["crps_by_patient"][[0, 2]]).all()
    assert np.isnan(score["crps_by_patient"][1])


def test_pooled_crps_averages_horizons_within_patient():
    per_patient = {
        "flow": {
            0: {"crps_by_patient": np.array([1.0, 3.0, np.nan])},
            1: {"crps_by_patient": np.array([3.0, np.nan, 5.0])},
        }
    }

    pooled = comparison._patient_mean_crps(per_patient, "flow", [0, 1])

    assert np.allclose(pooled, np.array([2.0, 3.0, 5.0]), equal_nan=True)


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


def test_uniformized_quantile_grid_recovers_nominal_coverage():
    """Uniform-probability quadrature of true quantiles must yield nominal coverage.

    Build a perfectly specified quantile function for each patient, convert its nonuniform
    fitted levels to equal-probability quadrature nodes, then exercise the same sample scorer
    used by the comparison.
    """
    rng = np.random.default_rng(3)
    n = 4000
    mu = rng.normal(size=n)
    sd = 0.5 + rng.uniform(size=n)
    q = np.asarray(bt.DEFAULT_QUANTILES)
    from scipy.stats import norm
    fitted = mu[:, None] + sd[:, None] * norm.ppf(q)[None, :]
    ens = bt.quantile_grid_to_ensemble(fitted[:, :, None], q, n_samples=200)[:, :, 0]
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
    """Uniformized quantile predictions must produce finite CRPS and NLL."""
    dataset, splits = _fake_dataset_and_splits()
    out = bt.fit_quantile_baselines(dataset, splits, seed=0)
    x_cont = dataset.x[:, tw.CONT_DIMS].astype(float)
    mask_cont = dataset.mask[:, tw.CONT_DIMS]
    test_idx = splits["test"]
    ensemble = bt.quantile_grid_to_ensemble(out["qgbm_pred"], out["quantiles"])
    got_density = False
    for h in range(tw.X_CONT_DIM):
        score = bt.horizon_score(ensemble[:, :, h], x_cont[test_idx, h], mask_cont[test_idx, h],
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
