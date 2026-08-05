# Streaming `acquire_cohorts` - rewrite specification

Status: **ready to implement** (decisions D1-D5 resolved, Section 15). Do not implement in this session;
this is the handoff artifact for a fresh implementation session. Section 16 is the pick-up context for
that agent.

Target file: `qreg_improvement/run_metabolic_trajectory_study.py` - a single self-contained file with no
project-local imports (self-test 28 enforces this). All new code stays in this one file.

**Correctness oracle for the entire change:** the `INTERNAL/checkpoints/figure_data.pkl` payload must be
byte-identical to the pre-change baseline across four configurations (orchestrated / in-process x parquet
/ pickle-fallback). Every decision below is chosen so that holds.

---

## 1. Problem and measured evidence

`acquire_cohorts` reads both wide cohort tables in one shot and builds the entire normalized bundle in
memory before writing anything. Measured peak RSS driving the real production path (`query_cosmos` ->
`wide_tables_to_data_bundle` -> `construct_cohorts`), all 14 horizon columns populated (~14
measurements/patient):

| patients (both cohorts) | measurements | peak RSS | resident result (CUR) |
|---|---|---|---|
| 5,000   | 70k   | 0.384 GB | - |
| 20,000  | 280k  | 0.975 GB | - |
| 80,000  | 1.12M | 3.281 GB | 0.75 GB |
| 160,000 | 2.24M | 6.352 GB | 1.23 GB |

Dead-linear fit: `peak ≈ 0.2 GB + 38.4 KB × patients`. Extrapolated: **~830k patients -> 32 GB (the VM
ceiling; OOM-kill here); 1M -> ~38 GB; 2M -> ~77 GB.**

**The peak is the construction transient, not the tables.** Two measurements pin this down:

- At 160k, peak 6.35 GB but resident result only 1.23 GB - ~80% is freed-after transients.
- The raw wide tables themselves are tiny: **measured 240 B/patient (~0.4 GB at 2M), ~180 B/patient
  compacted (~0.3 GB).** Negligible.

So the entire ~38 KB/patient is Python-object intermediates built for *all* patients at once:
- `wide_tables_to_data_bundle`: `for _, row in frame.iterrows()` building `medication_rows` and
  `measurement_rows` (a 10-key dict + a boxed `Timestamp` per measurement, ~14/patient), plus
  `patients_by_id` (a dict per patient, then a parallel `mbs__`/`glp1__`-prefixed copy).
- `construct_cohorts`: `patients.set_index(...).to_dict(orient="index")` (a third full copy of patients as
  dict-of-dicts), `cohort_rows` (a ~110-key dict per eligible patient), and `normalize_measurements`'
  `source_details`/`supporting`/`normalized_rows`.
- Full-table `drop_duplicates()` on procedures, medications, measurements.

**Consequence for the design (and the answer to "does chunksize fix it?"):** No. `chunksize` on the read
only bounds the read's own `fetchall` transient (~2.4 GB at 2M) - about 1/3 of nothing, since the wide
tables are 0.4 GB anyway. The killer is the construction, which is independent of how the rows arrive.
Because the wide tables are tiny, we can hold them fully resident and **batch only the construction**,
entirely client-side, with the production SQL byte-identical (D2). `chunksize` is kept as a cheap,
optional bound on the read transient, not as the fix.

Secondary problem, same code: **wall-clock.** The 160k point took >10 min for acquire+construct alone,
all in Python per-row loops; `medication_frame_to_coverage` is walked six times over the medication
table. At 2M that is hours before the first model. Addressed in Section 11.

---

## 2. Goals and non-goals

Goals:
1. Bound `acquire_cohorts` peak RSS to `O(base + wide_tables + one_batch + resident_outputs)` instead of
   `O(all-patient transients)`. Target: **<10 GB at 2M on the 32 GB VM** (projected ~6-8 GB, Section 10).
2. Preserve `figure_data.pkl` byte-for-byte. No scientific change.
3. **Keep the production SQL byte-identical** (D2) - no `ORDER BY`, no keyset CTE, no fingerprint change.
4. Keep the single-file / no-project-local-imports constraint.
5. Cut the per-row Python loops so acquire is not the wall-clock bottleneck.

Non-goals:
- Any change to eligibility, split, measurement/coverage semantics, or the model pipeline.
- Re-architecting downstream stages (`weights` onward) - they already stream via `TaskPartitionedStore`.
- SQL-side pagination or UNPIVOT (rejected: D2 keeps SQL frozen; and it moves logic off the auditable
  single file).

---

## 3. Invariants that must hold (determinism contract)

Each is annotated with why client-side batching preserves it.

I1. **Canonical cohorts order.** Today `cohorts` = surgery arm (patients ascending by `patient_id`) then
incretin arm (patients ascending). Source: `groupby("patient_id", sort=True)` (surgical loop) and
`for patient_id in sorted(episodes_by_patient)` (incretin loop). `build_prediction_rows` iterates this
frame in order and assigns `row_id` as a running counter, so the order is load-bearing.
=> Preserved by a single stable sort of the assembled cohorts frame on `(arm_rank, patient_id)` at
finalize (each patient appears at most once per arm, so the sort is total and reproduces the append
order). `kind="mergesort"` (stable).

I2. **`row_id` values.** Running `int32` counter over `build_prediction_rows`' output in cohorts-frame
order. => Unchanged: `build_prediction_rows` is not touched and iterates the identically-ordered cohorts
frame (I1).

I3. **Per-patient completeness.** Coverage-episode reconstruction, primary-episode selection, index-row
selection, and the MBS/GLP1 attribute merge require *all* of a patient's rows from *both* tables.
=> Guaranteed: batches are patient-aligned (Section 5), so a patient's MBS+GLP1 rows are always in one
batch.

I4. **`select_wide_index_rows` tie-break, now made deterministic (D3).** Final sort keys are
`..., _row_hash, _row_index`. `_row_hash` is a content hash (batch-independent). `_row_index` is today the
frame position in *DB return order*, which has no `ORDER BY` and is therefore **non-deterministic across
runs**; it only separates rows whose entire mapped-column content is identical (which `drop_duplicates`
collapses to the same selection anyway). => D3: assign `_row_index` from a **stable global ordinal**
computed by sorting each wide table once by `(patient_id, _row_hash)` at read time. This makes the
tiebreak reproducible run-to-run and makes patient batches contiguous slices (Section 5). It cannot change
any selected row's values.

I5. **Measurement row order is a don't-care.** `build_prediction_rows` accesses measurements via
`groupby("patient_id").indices` + `.take(...)`, and every per-patient reduction is order-insensitive:
`last = history.sort_values(["day","measurement_date"]).iloc[-1]` sorts explicitly; `robust_slope` is a
median of pairwise slopes; `np.std(ddof=0)` and `measurement_quality`'s groupby are order-free. => The
per-batch-concatenated measurement frame need not match single-pass row order. This is verified against
the oracle, not merely asserted (Section 13 test 2).

I6. **Global reductions** (funnel counts, `measurement_quality`, `medication_audit`, center availability,
source counts, `rejected_coverage_records`) are computed over all patients today. => Each is a commutative
accumulation; see Section 7.

I7. **Cross-cohort patients.** A patient in both MBS and GLP1 gets one surgery cohort row + one incretin
cohort row, merged attributes via `patients_by_id`, and one split (via `assign_global_splits`'
`drop_duplicates("patient_id")`). => Patient-aligned batching keeps both source rows together.

I8. **Seeds and fingerprint inputs.** `assign_global_splits` uses `stable_hash_fraction` keyed on
`patient_id`/`center_id` (batch-independent). The fingerprint payload (`administrative_data_through`,
`source_*_counts`, schema/query fingerprints) comes from the up-front `COUNT_BIG` probes and the
unchanged SQL text, so it is identical. => No metadata change; SQL frozen (D2).

---

## 4. Design overview

Keep the read as-is (SQL byte-identical), then replace "normalize everything, construct everything" with a
client-side patient-batched construction:

```
mbs, glp1 = load_direct_wide_tables(...)          # UNCHANGED SQL; optionally chunked read (Section 5)
mbs, glp1 = sort_by_patient(mbs), sort_by_patient(glp1)   # (patient_id, _row_hash); D3 + contiguous batches
pos_mbs, pos_glp1 = patient_slices(mbs), patient_slices(glp1)   # O(patients) int offsets

accum = AcquireAccumulators()
measurements_spill = MeasurementSpill(run_dir)
cohort_row_frames = []
for batch of PATIENT_PAGE patient_ids (Section 5):
    mbs_B, glp1_B    = contiguous slices of mbs, glp1 for this batch      # tiny; wide tables are ~0.4 GB
    bundle_B         = _wide_tables_to_batch_bundle(mbs_B, glp1_B, ...)   # existing logic, per batch
    cohort_rows_B    = _construct_cohorts_batch(bundle_B, accum)          # existing logic, per batch
    cohort_row_frames.append(cohort_rows_B)
    measurements_spill.write(bundle_B.normalized_measurements)
    del bundle_B, cohort_rows_B, mbs_B, glp1_B          # one-batch transient freed each iteration

cohorts      = compact(stable_sort(concat(cohort_row_frames), by=(arm_rank, patient_id)))  # I1 + D4
measurements = measurements_spill.materialize()
artifacts    = assemble(cohorts, measurements, accum.finalize())          # same dict keys as today
# checkpoints written exactly as today: cohorts, gap_sensitivity, bundle_light
```

Nothing downstream of `acquire_cohorts` changes: the artifacts dict has the same keys and frame contents;
`global_splits`, `prediction_rows`, `models_*`, `evaluation`, `figure_data` are untouched.

---

## 5. Batching mechanism: client-side, SQL unchanged

The read stays exactly as today (`load_direct_wide_tables`, one `SELECT` per table, no `ORDER BY`). The
wide tables are tiny (~0.4 GB at 2M, measured), so they are held resident. All batching is client-side:

1. **(Optional) chunked read.** `pd.read_sql_query(sql, conn, chunksize=READ_CHUNK)` streams
   `fetchmany(READ_CHUNK)` and bounds the read's `fetchall` transient (~2.4 GB at 2M -> a few tens of MB).
   Append the (optionally compacted) chunks into the resident wide frame. This does **not** change the SQL
   text, so `query_fingerprint` is unchanged. Purely a read-transient optimization; independent of the
   memory fix.
2. **Deterministic ordering (D3).** Sort each wide table once by `(patient_id, _row_hash)` where
   `_row_hash = pd.util.hash_pandas_object(frame[mapped_columns], index=False)` (the same hash
   `select_wide_index_rows` already computes). Assign the global ordinal `_row_index` from this order. This
   (a) makes the index-selection tiebreak reproducible run-to-run, and (b) lays each patient's rows out
   contiguously so a batch is a slice, not a scatter.
3. **Patient-aligned batches.** Take the sorted union of `patient_id` across both tables; walk it in groups
   of `PATIENT_PAGE` (D1 = 50,000). Each group's rows are a contiguous slice of each sorted wide table
   (`mbs.iloc[lo:hi]`), so both tables stay aligned and a patient in both cohorts is wholly inside one
   batch (I3, I7).
4. **Per-batch construction** (Section 6) runs on the slice; its transient is bounded to `PATIENT_PAGE`
   patients (~1.9 GB at 50k), freed each iteration.

Why not `chunksize` alone: it bounds the read (a non-dominant ~2.4 GB) but leaves the ~38 KB/patient
construction transient intact (~76 GB at 2M) - the OOM stands. Why no `ORDER BY`: the client does the
patient-ordering in memory on the tiny wide frames, so the SQL is frozen (D2). Why hold the wide tables
resident rather than spill: at 0.4 GB they are trivially holdable; spilling would add a disk round-trip
for nothing. (If a future cohort makes the wide tables themselves too large - roughly >10-15M patients -
spill the sorted wide tables to a parquet dataset and slice from disk; noted as a Section 14 fallback, not
needed for 2M.)

`PATIENT_PAGE`: module constant `PATIENT_PAGE = 50_000` with an env override
(`METABOLIC_ACQUIRE_PATIENT_PAGE`), mirroring `_PARQUET_SPILL_ENABLED` (D1). It must not affect results
(Section 13 test 2 asserts invariance across page sizes).

---

## 6. Per-batch construction: refactoring the two builders

No new science; both builders are refactored so inputs are one batch and outputs append to accumulators.

### 6.1 `wide_tables_to_data_bundle` -> `_wide_tables_to_batch_bundle`
Split the current function (approx lines 2574-2835) into:
- **Once-per-run setup**: `resolve_wide_fields` (on the probe frame, already in `load_direct_wide_tables`),
  fix `resolved_by_source` / `qualified_names`, and build the `metadata` dict (depends only on the up-front
  `source_totals` and SQL/schema fingerprints, not on rows).
- **Per-batch body** `(mbs_B, glp1_B, resolved_by_source, qualified_names, incretin_qualifying_days) ->
  DataBundle`: the same two passes (Pass 1 procedures/medications + incretin episodes ->
  `preferred_glp1_anchors`; Pass 2 `select_wide_index_rows` + patient attributes + measurements). All
  per-patient, so restricting to a batch changes only scale. Batch-local `drop_duplicates` is safe
  (`patient_id` in every key; no cross-batch dupes).

### 6.2 `construct_cohorts` -> `_construct_cohorts_batch`
Refactor the body (approx lines 3752-4032) to run on one batch's `DataBundle` + an `AcquireAccumulators`:
- `patient_lookup = ... to_dict(orient="index")` becomes per batch (the single biggest transient today,
  now O(batch)).
- `normalize_measurements(bundle.measurements)` runs per batch; its `measurement_quality` merges into the
  accumulator (Section 7); its normalized rows go to the measurements spill.
- Surgical + incretin loops run per batch, appending cohort rows and their exposure/funnel/exclusion
  contributions to the accumulator.
- `effective_censor_day` (the `.apply(axis=1)` at ~line 4020) is computed vectorized on the small
  assembled cohorts frame at finalize (Section 11), not per row.

### 6.3 Finalize
- `cohorts = concat(cohort_row_frames)`; add `arm_rank` (surgery=0, incretin=1);
  `sort_values(["arm_rank","patient_id"], kind="mergesort")`; drop the helper; `reset_index(drop=True)`.
  Reproduces I1. Then the vectorized `effective_censor_day`, then compact to categoricals (D4).
- Empty-cohort guard (`PreflightError("Cohort construction produced no eligible patients")`) fires on the
  assembled frame as today.
- Emit the artifacts dict with identical keys.

### 6.4 Retained wrappers
Keep `wide_tables_to_data_bundle(...)` and `construct_cohorts(...)` as thin wrappers that run a single
"batch" = whole input and finalize. This keeps self-tests 29/34 and the in-process `run_study` path valid
for free, and gives a direct A/B (`whole` vs `batched`) oracle in tests.

---

## 7. Global reductions - exact catalog and accumulation

A missed reduction is a silent scientific change; this list must be complete.

| Output | Current computation | Accumulator |
|---|---|---|
| `funnel` per-stage `n_patients` | counts over all patients per (cohort, stage, status) | integer counters keyed by (cohort, stage, status); `aggregate_funnel` at finalize |
| surgical/medication exclusion counts | `value_counts` / `defaultdict(int)` | merge integer counters |
| `measurement_quality` | RLE groupby (`measurement_quality_counts_frame`) | sum RLE counts by full key (merge = groupby-sum; consumed at `figure_data` line ~7533 by `measurement_quality_table`, so must be exact) |
| `medication_audit` | RLE per surgery/incretin | merge RLE counts by key |
| `exposure` | one row per (patient, cohort) | append per-batch frames; concat |
| `center_validation_available` | needs all `selected_center_values`: all non-"unavailable" AND >=3 distinct | accumulate `{any_unavailable: bool, distinct_centers: set[str]}`; evaluate at finalize |
| `source_*_counts` (metadata) | `nunique`/`COUNT_BIG` | from the up-front probe `source_totals`, page-independent (I8) |
| `preferred_glp1_anchors` | per patient over incretin episodes | per batch (patient-aligned; I3) |
| `rejected_coverage_records` | `pd.DataFrame([asdict(r) ...])` | **DROP it (D5).** Verified write-only: the only occurrence is the write at ~line 4031; no stage reads `cohort_artifacts["rejected_coverage_records"]`. Replace with an empty frame (keep the key for schema stability) or remove the key if `payload_manifest`/consumers tolerate it. Confirm the manifest diff is clean. |

`assign_global_splits` (the `global_splits` stage) is unchanged: it operates on the assembled cohorts
frame (O(patients), resident), which it legitimately needs whole (temporal 0.80 quantile, distinct
centers). Cohorts is not the memory problem.

---

## 8. Measurements handling

`MeasurementSpill` - append each batch's compacted normalized-measurement frame to a numbered file under
the run dir (parquet when `_PARQUET_SPILL_ENABLED`, else pickle, matching the existing store pattern).
`materialize()` reads them back into one frame for the `measurements` artifact. Bounds acquire's
measurement memory to one batch during construction; the materialized frame (~1-3 GB at 2M) is the same
object the code holds and checkpoints today, consumed by `prediction_rows` and `figure_data`
(`measurement_quality_table` reads `cohort_artifacts["measurements"]`).

Optional future (not needed for 2M): promote measurements to a partitioned dataset checkpoint (bucketed by
patient) and stream `build_prediction_rows`' `groupby("patient_id").indices` over it, removing the last
O(patients x horizons) resident term. Only worth it beyond ~5M patients.

---

## 9. Interfaces and signatures (all inside the one file)

- `PATIENT_PAGE: int` (const) + `METABOLIC_ACQUIRE_PATIENT_PAGE` env override; `READ_CHUNK: int` for the
  optional chunked read.
- `load_direct_wide_tables(...)` - optionally add `chunksize` to the `reader(query, connection)` call and
  concat chunks; **no SQL text change**.
- `sort_wide_by_patient(frame, resolved) -> frame` - sort by `(patient_id, _row_hash)`, assign
  `_row_index` global ordinal (D3).
- `iter_patient_batches(mbs, glp1, resolved_by_source, page) -> Iterator[tuple[mbs_B, glp1_B]]` -
  contiguous slices over the sorted union of patient ids.
- `_wide_tables_to_batch_bundle(mbs_B, glp1_B, resolved_by_source, qualified_names,
  incretin_qualifying_days) -> DataBundle`.
- `wide_tables_to_data_bundle(...)` - retained wrapper (single batch), unchanged signature.
- `class AcquireAccumulators` - funnel/exclusion counters, measurement_quality RLE, medication_audit RLE,
  center set, exposure frames; `.update(...)`, `.finalize() -> dict`.
- `class MeasurementSpill` - `.write(frame)`, `.materialize() -> DataFrame`.
- `_construct_cohorts_batch(bundle_B, accum) -> DataFrame` (batch cohort rows; appends to accum).
- `construct_cohorts(bundle)` - retained wrapper (single batch + finalize), unchanged signature.
- The `acquire_cohorts` worker body (approx lines 11225-11248) and `query_cosmos` rewired to the batch
  loop; still end by writing the same three checkpoints and calling `log_peak_rss`.

---

## 10. Memory budget and page sizing

Peak(acquire) ≈ base(0.3) + wide_tables(0.4) + one_batch_transient + resident_cohorts + resident_meas.

- wide tables resident: ~0.4 GB at 2M (measured), + a one-time sort transient ~0.4 GB (before construction).
- one_batch_transient ≈ `PATIENT_PAGE × 38 KB`: 50k -> ~1.9 GB.
- resident_cohorts at 2M ≈ 2-3 GB raw, **~1-1.5 GB compacted (D4)**.
- resident_measurements at 2M ≈ 1-3 GB compacted (or spilled during construction, materialized at end).

At `PATIENT_PAGE = 50_000`: **peak ≈ 0.3 + 0.4 + 1.9 + 1.5 + 3 ≈ ~7 GB at 2M.** Comfortable on 32 GB with
room for the pyodbc driver and OS. (The sort transient and the batch transient do not coexist with the
full resident outputs at the same instant; the ~7 GB is the construction-phase high-water mark.)

`prediction_rows` (next stage) then loads measurements (~1-3 GB) + cohorts (~1.5 GB) + its 250k-row output
chunk, bounded by `TaskPartitionedStore` -> ~5-6 GB. No change needed there.

---

## 11. Wall-clock (fold into the same rewrite; output-neutral)

Covered by the same oracle (no output change):
- The `PATIENT_PAGE` cap already turns the unbounded `iterrows` passes into 50k-row passes. Additionally
  replace `iterrows()` (approx lines 2596, 2693) with `itertuples()` (which does not box an object Series
  per row) or vectorized column construction per batch.
- `medication_frame_to_coverage` is called six times over the medication table (approx 2657, 2975, 2984,
  3783, 3784, 7084). Compute coverage records once per batch and thread the result through.
- `construct_cohorts`'s `to_dict(orient="index")` (approx line 3755) is removed by per-batch scope; keep
  the existing O(1) `groupby(...).indices` position-map slicing for per-patient access.

---

## 12. Related correctness hardening to fold in (acquire-local only)

The rewrite touches these exact lines; fix here. Each must be oracle-neutral on the current fixture (they
only change real-data edge-case behavior the fixture does not exercise):
- **`patient is None` crash, incretin loop (approx line 3936).** `patient is None` is recorded as an
  exclusion but not `continue`d before `age_at_index(patient, index_date)` /
  `patient["observation_start_date"]` dereference it. Add the guard the surgical loop already has.
- **`int(NaN)` (approx lines 4304, 4306).** A `NaT` measurement date -> NaN `day` -> `int(...)` raises.
  Guard/drop NaN days in `normalize_measurements`.
- **`ZeroDivisionError` (approx line 4311).** `.../ float(baseline_value)` is raw Python division; guard
  `baseline_value == 0`.

Out of scope here (later stages; track separately): `--resume` not forwarded to workers;
`estimate_residual_correlation` `pivot_table` missing `observed=True`. A fuller non-OOM inventory exists
if wanted.

---

## 13. Validation and acceptance (all must pass)

1. **Oracle parity (the gate).** `figure_data.pkl` byte-identical to the pre-change baseline across
   {orchestrated, in-process} x {parquet, pickle-fallback}. Use the dump-and-diff harness (Section 16);
   exclude only `training_seconds`/`elapsed_seconds` and `identity.fingerprint`/`split.centers`.
2. **Batch-size invariance (proves I1-I7).** New self-test: a fixture spanning multiple pages; assert the
   assembled `cohorts` (order + content), `measurements` (as a set), and `figure_data` are identical for
   `PATIENT_PAGE in {1, 3, all}`.
3. **Cross-cohort boundary (I7).** Fixture patient present in both MBSCohort and GLP1Cohort with rows
   straddling a page boundary; assert both cohort rows, merged attributes, and the single split.
4. **Memory ceiling.** Reuse the `slope.py` harness: assert peak RSS scales with `PATIENT_PAGE`, not with
   total patients (peak at 320k/page=50k within a small constant of peak at 80k/page=50k).
5. **Existing self-tests 50/50**, including 28 (no project-local imports) and 29/34 (direct wide E2E via
   the retained wrappers).
6. **`--preflight-only`** returns identical preflight metadata (I8).

Add a per-batch `log_peak_rss(f"acquire batch {i}")` so the VM operator sees the bounded sawtooth.

---

## 14. Rollout

1. Land the change behind the retained single-batch wrappers; prove oracle parity **in-process first**
   (fastest loop), then orchestrated, then the parquet/pickle matrix.
2. **No SQL re-bless needed** (D2: SQL is byte-identical).
3. Run `--preflight-only` on the VM at the real cohort size for a measured peak before the full run.
4. Fallback only if wide tables themselves ever exceed RAM (>~10-15M patients): spill the sorted wide
   tables to a parquet dataset and slice from disk. Not needed for 2M.

---

## 15. Decisions (resolved)

- **D1 - page size:** `PATIENT_PAGE = 50_000`, module constant + `METABOLIC_ACQUIRE_PATIENT_PAGE` env
  override. **Resolved: accepted.**
- **D2 - SQL contract:** keep the production SQL byte-identical. **Resolved: no SQL change.** Achieved by
  client-side batching over the resident (tiny) wide tables; `chunksize` optional for the read transient
  only. `chunksize` alone does not fix the OOM.
- **D3 - deterministic `_row_index`:** assign from a stable global ordinal via a `(patient_id, _row_hash)`
  sort at read time. **Resolved: yes** (also yields contiguous patient batches).
- **D4 - compact `cohorts` to categoricals:** **Resolved: yes.** Roughly halves the resident cohorts frame
  (~3 GB -> ~1.5 GB). Apply at finalize; ensure fixed category rosters so downstream `groupby` order is
  unchanged (mirror the `TaskPartitionedStore` category discipline).
- **D5 - `rejected_coverage_records`:** **Resolved: drop** (verified write-only). Keep the key with an
  empty frame for schema stability unless the manifest diff shows it can be removed cleanly.

---

## 16. Context for the implementing agent (fresh session pick-up)

**What this is.** A Python metabolic-trajectory forecasting study, delivered as ONE self-contained file:
`qreg_improvement/run_metabolic_trajectory_study.py` (~11.7k lines). Run in production by a collaborator on
a secure ~32 GB Windows VM against Epic Cosmos over pyodbc. Neither the VM nor the data is reachable from
the dev machine; production behavior is validated only via `--self-test` and `--smoke` against an embedded
synthetic fixture. The VM's installed pandas version is unknown and may differ from dev - any
version-dependent pandas behavior must be pinned (e.g. `observed=True` on categorical groupbys).

**Hard constraints.**
- Single file, no project-local imports (self-test 28 asserts this; `EMBEDDED_RAW_SOURCE_SQL` is inert).
- Production SQL text is frozen (D2) - do not add `ORDER BY`/CTEs to the wide-table queries.
- Memory is the binding limit; determinism is contractual (the `figure_data` oracle).

**Interpreters (no system Python; `python` is the Windows Store stub, exits 49).**
- venv WITH torch + pyarrow (needed for `--self-test` and the parquet store path):
  `C:\Users\Tien\AppData\Local\Temp\claude\D--Work-ShinLab-r-flow-matching-bariatric\097ec4c3-703d-48bd-ae07-b6fa0ac4290d\scratchpad\venv\Scripts\python.exe`
  (pandas 3.0.5, numpy 2.5.1, torch 2.13.0+cpu). If gone, rebuild: `python -m venv` from the pixi
  interpreter below, then `pip install pandas numpy pyarrow scikit-learn matplotlib` + torch CPU wheel
  from https://download.pytorch.org/whl/cpu (PyPI is reachable).
- Fallback interpreter WITHOUT torch/pyarrow (smoke only, exercises the pickle store path):
  `D:\Work\Arciero\.pixi\envs\default\python.exe`.

**Validation commands.**
- `--self-test` (deterministic, ~1 min; hard-requires torch): expect `SELF-TEST PASSED: 50/50`.
- `--smoke --orchestrate --output-dir <dir>`: full pipeline end to end via the process-per-stage path
  production uses (default 420 synthetic patients; ~0.45 GB peak - flat, dominated by imports).
- Oracle diff: run `--smoke --orchestrate`, unpickle `<dir>/INTERNAL/checkpoints/figure_data.pkl`, dump
  every frame/key/scalar to canonical text, and diff vs a baseline captured before the change. Run all
  four configs: {orchestrated, in-process (`--single-process`)} x {parquet venv, pickle-fallback}. They
  must all match except `training_seconds`, `elapsed_seconds`, `identity.fingerprint`, `split.centers`
  (the last two are fingerprint-keyed and permute on ANY script edit - expected). A prior dump script and
  baselines may exist under the `.../097ec4c3-.../scratchpad/` dir (e.g. `dump_figure_data.py`).

**Memory-measurement harness (reuse for Section 13 test 4):**
`C:\Users\Tien\AppData\Local\Temp\claude\D--Work-ShinLab-r-flow-matching-bariatric\aed64b9c-b388-4171-a980-1b808201c891\scratchpad\slope.py`
drives the real production path with scaled synthetic wide fixtures (all 14 horizons populated) at N
patients/cohort, one process per size, printing peak RSS. It monkeypatches `study.connect_cosmos` and
`study.pd.read_sql_query`; note the module defers pandas/numpy via `load_runtime_packages()` (globals are
`None` until then), and the fixture reader must match the table by `"GLP1Cohort" in sql` (not `"GLP1"`,
which also matches the `PriorGLP1` columns). `wide_mem.py` in the same dir measures wide-frame footprint.

**Measured facts driving the design (so they need not be re-derived):**
- acquire peak `≈ 0.2 GB + 38.4 KB × patients`, linear across 5k-160k (points: 0.384/0.975/3.281/6.352
  GB). 2M -> ~77 GB; crosses 32 GB at ~830k.
- Raw wide tables: 240 B/patient (~0.4 GB at 2M); compacted ~180 B/patient. The peak is the construction
  transient, not the tables.
- Import baseline (no data): 0.296 GB (torch ~0.15 of it).

**Key source locations (function names authoritative; line numbers approximate, will drift):**
`load_direct_wide_tables` ~2257, `build_direct_source_sql` ~2222 (do not add ORDER BY), `resolve_wide_
fields` ~2140, `WIDE_TARGET_FIELDS` ~2101, `select_wide_index_rows` ~2440, `wide_tables_to_data_bundle`
~2574, `query_cosmos` ~3093, `construct_cohorts` ~3752, `normalize_measurements` ~1269, `assign_global_
splits` ~4041, `build_prediction_rows` ~4206 (do NOT change its iteration), `TaskPartitionedStore` ~6058,
`log_peak_rss` ~431 / `process_peak_rss_bytes` ~374, `run_stage_worker` ~11209 (acquire body ~11225),
`run_study_orchestrated` ~11442, `run_study` (in-process) ~10990, embedded self-tests start ~9400 with the
direct-wide E2E fixture at ~10200 (tests 29/34).

**Git state at spec time:** branch `main`; `run_metabolic_trajectory_study.py` has a large uncommitted diff
(the just-completed row-table task-partitioning work); self-test 50/50 green and orchestrated smoke clean
on the current tree. This spec file is new/untracked. Implement on a fresh branch.

**Recommended implementation order:** (1) extract `_wide_tables_to_batch_bundle` + `_construct_cohorts_
batch` with single-batch wrappers, prove in-process oracle parity (no batching yet). (2) Add
`AcquireAccumulators` + `MeasurementSpill`, prove parity. (3) Add the sort/ordinal (D3) + `iter_patient_
batches` + the loop, prove batch-size invariance and parity across all four configs. (4) Fold in D4
(compact cohorts), D5 (drop rejected records), the wall-clock vectorization, and the three acquire-local
crash fixes, re-proving parity after each. (5) Add the Section 13 tests. Land only when 50/50 + parity +
invariance all hold.
