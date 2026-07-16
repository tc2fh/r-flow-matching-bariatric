# Single-Script Five-Year Forecast Improvement Study

Date recorded: 2026-07-16

## Scope

Implement the investigation in exactly one new Python source file,
`qreg_improvement/run_qreg_improvement.py`. The one-command runner owns data validation,
chronological 70/15/15 splitting, leakage checks, training and tuning, preoperative and
rolling-origin forecasts, MACE and component-risk evaluation, published-reference
comparisons, checkpointing, figures, a multi-page report, and a retrieval bundle.

Supported modes are `full`, `smoke`, `plot-only`, and `self-test`. Runs resume by default,
use configuration and input hashes, update `00_progress.png` after each stage, and write
artifacts atomically. All code and generated outputs remain under `qreg_improvement/`.
Matplotlib, XDG, Torch, and Joblib caches default to `qreg_improvement/.cache/`, so cluster
users do not need to export cache environment variables.

## Forecast protocol

The primary preoperative grid contains 13 targets: BMI at 3, 6, 9, 12, 24, 36, 48, and
60 months, and HbA1c at 12, 24, 36, 48, and 60 months. Rolling forecasts originate at 3,
6, 9, 12, 24, 36, and 48 months and predict strictly future targets through month 60 using
only information available by the origin. The most recent measurement age is recorded.
Repository-tracked 72-month forecasts are supplementary.

Patient-origin-target comparisons use identical cells. The chronological test set stays
sealed throughout model selection. Recursive pre-op features must come from cross-fitted
predictions, never observed future outcomes.

## Candidate models

The planned trajectory roster is current event-free qReg plus rank-Gaussian copula, pooled
spline quantile regression, pooled XGBoost and CatBoost quantile regression,
validation-weighted quantile ensembles, autoregressive HistGradientBoosting, rolling
Trajectory Flow Matching, persistence, and population-trajectory baselines. The
autoregressive model has separately labelled pre-op and rolling variants and propagates
uncertainty through joint sampling. TFM conditions on the three latest observations and
timestamps plus static pre-op covariates, origin, and horizon; finalists use five seeds.

The risk roster is the existing composite model, elastic-net logistic regression, XGBoost,
LightGBM, CatBoost, ExtraTrees, a leak-free stack, a component-probability stack, and a stack
with cross-fitted trajectory features. The primary endpoint remains ever-composite MACE;
MACE, nephropathy, and retinopathy are secondary components.

No CART model or SOPHIA reproduction is trained. A BJS-inspired architecture is not called
a reproduction of the published fitted model.

## Evaluation

Every forecast target and rolling cell reports observation count, IPCW effective sample
size, CRPS, RMSE, MAE, median absolute deviation, bias, energy score, variogram score, PIT,
50/80/90/95 percent coverage and width, threshold probabilities for BMI below 35 and HbA1c
below 5.7, and smoothness, curvature, and adjacent-jump diagnostics. Both complete-case and
IPCW estimates are retained. Cells with fewer than 200 observations or effective sample
size below 100 remain visible and are labelled exploratory.

Risk evaluation reports AUROC, AUPRC, Brier score, calibration intercept and slope, expected
calibration error, sensitivity, specificity, PPV, NPV, decision-curve net benefit, component
performance, paired DeLong comparisons, and bootstrap comparisons.

## Published references

BJS 2026 BMI references are pooled RMSE 1.11, pooled MAE 0.62, and RMSE values 2.36, 1.31,
0.91, 0.62, 0.78, 0.92, and 1.01 kg/m2 at 3, 6, 12, 24, 36, 48, and 60 months. There is no
9-month reference. BJS MAPE is caveated and excluded from success gates.

SOPHIA BMI references are RMSE 3.7, 4.2, and 4.7 kg/m2 at 12, 24, and 60 months; 60-month
MAD 2.8; normalized RMSE 12.0, 14.0, and 14.7 percent; and 60-month RYGB and sleeve RMSE
4.5 and 5.7 kg/m2.

BJS diabetes-remission references are evaluated only if the exact medication-free HbA1c
definition is constructible: AUROC 0.99, macro F1 0.88, precision 0.87, recall 0.88, and log
loss 0.07. MACE and HbA1c-only proxies are never compared with these values.

The audited BJS and SOPHIA values, caveats, and a registry checksum are embedded directly in
the runner. The source PDFs are not runtime inputs and do not need to be copied to the cluster.

## Gates and inference

The primary trajectory gate requires at least 10 percent lower pooled patient-averaged CRPS
than current qReg plus copula, better energy and variogram scores, acceptable interval
calibration, and no adequately powered horizon more than 5 percent worse. The stretch goal
requires at least 10 percent lower CRPS at every adequately powered horizon and rolling cell,
no worse RMSE or MAE, applicable published targets exceeded, and coverage error within five
percentage points. Pooled testing precedes horizon tests with 5 percent FDR control and
simultaneous bootstrap bands.

Primary MACE success requires temporal-test AUROC at least 0.80, improved AUPRC, and
noninferior calibration. Local paired tests apply only to models executed on the same
patients. Published values are reference benchmarks. Definitive superiority requires
no-refit independent external validation, and a pooled gain can never conceal a failed
long-horizon result.

The total search budget is capped at 140 configurations: 60 risk and stacking, 50
conventional/quantile/ensemble/autoregressive, and 30 multi-fidelity TFM configurations.

## Outputs and validation

The run produces `performance_report.pdf`, matching numbered high-resolution PNG pages,
CSV metrics, Parquet predictions, compressed joint samples, checkpoints, split and feature
manifests, search history, state, logs, and `qreg_improvement_results_bundle.zip`. The bundle
omits large predictions and checkpoints.

Preflight and synthetic validation cover all 13 targets, split overlap, pre-op and rolling
leakage, cross-fitted recursion, sealed test outcomes, monotone quantiles, valid correlation
matrices, exact literature constants, and report generation from incomplete and completed
runs.
