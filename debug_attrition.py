"""Read-only attrition / missingness debug report for the MBSCohort pipeline.

This script answers *why rows disappear* between the raw table and the modelling
cohort, and *how missing* every conditioning + candidate feature is -- without
changing anything. It is purely diagnostic: it reuses ``train_flow_matching``'s
loaders, CPT mapping, event/interval logic, and filter predicates by import and
never re-implements the SQL WHERE or the Python-side filter logic. It only
re-derives the per-stage masks so it can *count* what each stage drops (the
loader emits warnings but returns no counts).

Two modes (DB is the default -- a bare run on the Cosmos VM needs no args):

    python debug_attrition.py
        Cosmos VM mode (the default when no --csv is given; equivalent to --db).
        Additionally runs ``SELECT COUNT(*) FROM MBSCohort`` WITHOUT the WHERE
        clause (and, best-effort, a per-WHERE-clause marginal count) through
        ``fm.CONNECTION_STRING`` to recover the true pre-filter denominator, so
        the SQL-filter losses become a visible stage too.

    python debug_attrition.py --csv fake_data/fake_mbs_cohort.csv
        Local mode. Sees Python-side attrition only (no true pre-filter
        denominator -- the CSV is already the post-SQL export).

Output = a terse ``.txt`` report + a one-line stdout headline pointing at it.

The report contains, per the build spec:
  (a) per-feature missingness (count + %) for every conditioning feature AND
      every candidate feature (PMH_*, eGFRatEvent, InsulinStatus /
      BiguanideStatus / SGLT2Status);
  (b) leave-one-out on ``REQUIRED_PATIENT_FEATURES`` -- rows recovered if each
      required field alone were demoted to optional (report only; the required
      set is NOT changed);
  (c) the CPT ``43645`` count surfaced directly (a gastric-bypass variant
      currently dropped as unrecognized) plus any other unrecognized CPT codes;
  (d) an attrition decomposition framed as STAGES with count + % at each step:
      [SQL-filter losses -- VM only] -> [CPT-unrecognized drops] ->
      [missing-required-conditioning drops] -> final N (with the intermediate
      fm drops that sit between them made explicit so the counts reconcile).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

import train_flow_matching as fm


DEFAULT_OUTPUT_DIR = fm.REPO_ROOT / "runs" / "debug_attrition"
GASTRIC_BYPASS_VARIANT_CPT = "43645"  # a bypass variant currently dropped as unrecognized

# Candidate features to profile (in addition to the six conditioning features).
# PMH_* columns are discovered dynamically from the frame; these are the rest.
EXPLICIT_CANDIDATE_FEATURES = ["eGFRatEvent", "InsulinStatus", "BiguanideStatus", "SGLT2Status"]

# The six conditioning features, mapped to their raw source columns + coercion so
# missingness reflects the RAW availability (mirrors fm.make_patient_features, but
# without insulin_status' fillna(0), which would otherwise hide its missingness).
CONDITIONING_SOURCES: list[tuple[str, str, str]] = [
    ("age_at_surgery", "AgeAtEvent", "numeric"),
    ("sex_male", "Sex", "sex"),
    ("creatinine_at_surgery", "CreatinineAtEvent", "numeric"),
    ("hba1c_at_surgery", "HbA1cAtEvent", "numeric"),
    ("bmi_at_surgery", "BMIatEvent", "numeric"),
    ("insulin_status", "InsulinStatus", "numeric"),
]


def pct(numerator: float, denominator: float) -> str:
    return f"{100.0 * numerator / denominator:.2f}%" if denominator else "n/a"


def coerce(series: pd.Series, kind: str) -> pd.Series:
    if kind == "sex":
        return fm.encode_sex_male(series)
    return fm.numeric(series)


# --------------------------------------------------------------------------- #
# Raw loading (read-only)
# --------------------------------------------------------------------------- #
def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Load + canonicalize a CSV export exactly as ``fm.load_dataset_from_csv``
    does, but stop *before* ``prepare_flow_dataset`` so no rows are filtered."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    df = fm.canonicalize_columns(df)
    fm.assert_required_columns(df, str(csv_path))
    return df


def load_raw_db() -> pd.DataFrame:
    """Post-SQL cohort (WHERE applied) via the fm DB loader -- the true
    pre-filter denominator is fetched separately by ``db_prefilter_counts``."""
    return fm.load_mbs_from_database()


# --------------------------------------------------------------------------- #
# (d) STAGES -- Python-side attrition decomposition
# --------------------------------------------------------------------------- #
def python_attrition(raw_df: pd.DataFrame) -> dict:
    """Replay fm.prepare_flow_dataset's filter sequence to COUNT each drop.

    Each predicate is computed with the same fm helper the real loader uses, so
    the logic is reused (not duplicated); only the masks are re-derived to count.
    Returns per-stage counts, the surviving DataFrames each stage needs, and the
    unrecognized-CPT value counts.
    """
    df = raw_df
    n_raw = len(df)
    stages: list[dict] = [{"stage": "raw loaded (post-SQL export)", "dropped": 0, "remaining": n_raw}]

    # 1. Unrecognized CPT -> not sleeve/rnygb -> dropped.
    surgery = fm.map_surgery_type(df["CptCode"])
    cpt_norm = fm.normalize_cpt_code(df["CptCode"])
    unrecognized = surgery.isna()
    unrecognized_codes = cpt_norm[unrecognized].fillna("<blank>").value_counts()
    df_cpt = df.loc[~unrecognized].copy()
    stages.append(
        {"stage": "CPT unrecognized (not sleeve/rnygb)", "dropped": int(unrecognized.sum()), "remaining": len(df_cpt)}
    )

    # (informational) duplicate PatKey -- fm RAISES on these rather than dropping.
    dup_count = int(df_cpt["PatKey"].astype("string").duplicated().sum())

    # 2. Composite event with no valid nonnegative interval -> dropped.
    event = fm.composite_event(df_cpt)
    interval = fm.composite_interval_months(df_cpt)
    bad_interval = event.eq(1) & interval.isna()
    df_interval = df_cpt.loc[~bad_interval].copy()
    stages.append(
        {"stage": "composite event w/o valid interval", "dropped": int(bad_interval.sum()), "remaining": len(df_interval)}
    )

    # 3. PostOpGLP1 == 1 with unavailable/negative start interval -> dropped.
    post_op = fm.numeric(df_interval["PostOpGLP1"]).fillna(0).eq(1)
    glp1_months = fm.compute_glp1_start_month(df_interval)
    bad_glp1 = post_op & (glp1_months.isna() | (glp1_months < 0))
    df_glp1 = df_interval.loc[~bad_glp1].copy()
    stages.append(
        {"stage": "PostOpGLP1 w/o valid start interval", "dropped": int(bad_glp1.sum()), "remaining": len(df_glp1)}
    )

    # 4. Missing required core conditioning -> dropped (this is where creatinine bites).
    pf = fm.make_patient_features(df_glp1)
    complete = pf[fm.REQUIRED_PATIENT_FEATURES].notna().all(axis=1)
    df_final = df_glp1.loc[complete].copy()
    stages.append(
        {"stage": "missing required conditioning", "dropped": int((~complete).sum()), "remaining": len(df_final)}
    )

    return {
        "n_raw": n_raw,
        "stages": stages,
        "unrecognized_codes": unrecognized_codes,
        "dup_patkey": dup_count,
        "pre_required_df": df_glp1,  # denominator for the leave-one-out
        "final_df": df_final,
    }


# --------------------------------------------------------------------------- #
# (b) Leave-one-out on REQUIRED_PATIENT_FEATURES
# --------------------------------------------------------------------------- #
def leave_one_out(pre_required_df: pd.DataFrame) -> pd.DataFrame:
    """For each required field, rows recovered if IT ALONE were demoted.

    A row is 'recovered by demoting F' iff F is missing but every *other* required
    field is present -- i.e. F is the sole reason the row is dropped. Computed on
    the cohort that reaches the required-conditioning filter. Report only.
    """
    pf = fm.make_patient_features(pre_required_df)
    required = fm.REQUIRED_PATIENT_FEATURES
    complete = pf[required].notna().all(axis=1)
    currently_dropped = int((~complete).sum())
    rows = []
    for field in required:
        others = [c for c in required if c != field]
        recovered = pf[field].isna() & pf[others].notna().all(axis=1)
        rows.append({"required_feature": field, "rows_recovered_if_demoted_alone": int(recovered.sum())})
    out = pd.DataFrame(rows).sort_values("rows_recovered_if_demoted_alone", ascending=False, ignore_index=True)
    out.attrs["currently_dropped"] = currently_dropped
    return out


# --------------------------------------------------------------------------- #
# (a) Per-feature missingness
# --------------------------------------------------------------------------- #
def missingness_report(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Missingness (count + %) for the conditioning features and the candidate
    features, on the raw (post-SQL) cohort. Returns (conditioning, candidate)."""
    n = len(raw_df)

    cond_rows = []
    for name, source_col, kind in CONDITIONING_SOURCES:
        matched = fm.find_compatible_column(list(raw_df.columns), source_col)
        if matched is None:
            cond_rows.append({"feature": name, "source": f"{source_col} (ABSENT)", "n": n, "missing": n, "missing_pct": 100.0})
            continue
        values = coerce(raw_df[matched], kind)
        miss = int(values.isna().sum())
        cond_rows.append({"feature": name, "source": matched, "n": n, "missing": miss, "missing_pct": 100.0 * miss / n if n else np.nan})

    # Candidate features: all PMH_* columns present + the explicitly named extras.
    pmh_cols = sorted(c for c in raw_df.columns if str(c).lower().startswith("pmh_"))
    candidate_names: list[str] = []
    for name in pmh_cols + EXPLICIT_CANDIDATE_FEATURES:
        if name not in candidate_names:
            candidate_names.append(name)

    cand_rows = []
    for name in candidate_names:
        matched = fm.find_compatible_column(list(raw_df.columns), name)
        if matched is None:
            cand_rows.append({"feature": name, "source": f"{name} (ABSENT)", "n": n, "missing": n, "missing_pct": 100.0})
            continue
        values = fm.numeric(raw_df[matched])
        miss = int(values.isna().sum())
        cand_rows.append({"feature": name, "source": matched, "n": n, "missing": miss, "missing_pct": 100.0 * miss / n if n else np.nan})

    return pd.DataFrame(cond_rows), pd.DataFrame(cand_rows)


# --------------------------------------------------------------------------- #
# (d, VM only) True pre-filter denominator + per-WHERE-clause marginal counts
# --------------------------------------------------------------------------- #
def parse_where_clauses(sql: str) -> list[str]:
    """Split fm.MBS_SQL's WHERE body into its individual clauses (reused, not
    re-typed), so per-clause counts don't duplicate the filter logic."""
    match = re.search(r"\bWHERE\b", sql, flags=re.IGNORECASE)
    if not match:
        return []
    body = sql[match.end():]
    parts = re.split(r"\n\s*AND\s+", body)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


def db_prefilter_counts() -> dict:
    """Best-effort true denominator + per-clause marginal removal, VM-only.

    ``true_total`` = COUNT(*) with no WHERE. Per clause we run
    ``COUNT(*) WHERE NOT (clause)`` = rows that clause alone would remove
    (NULL-sensitive; clauses overlap, so these do NOT sum to the total removed).
    Every query is guarded so a single failure never aborts the report.
    """
    try:
        import pyodbc  # type: ignore
    except ImportError as exc:
        return {"error": f"pyodbc unavailable: {exc}"}

    result: dict = {"clauses": []}
    try:
        with pyodbc.connect(fm.CONNECTION_STRING, timeout=1000) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM MBSCohort")
            result["true_total"] = int(cursor.fetchone()[0])
            for clause in parse_where_clauses(fm.MBS_SQL):
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM MBSCohort WHERE NOT ({clause})")
                    result["clauses"].append({"clause": clause, "rows_excluded_alone": int(cursor.fetchone()[0])})
                except Exception as exc:  # noqa: BLE001 - per-clause best effort
                    result["clauses"].append({"clause": clause, "rows_excluded_alone": None, "error": str(exc)})
    except Exception as exc:  # noqa: BLE001 - connection/denominator best effort
        result["error"] = str(exc)
    return result


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def build_report(source_label: str, raw_df: pd.DataFrame, db_counts: dict | None) -> tuple[str, dict]:
    attrition = python_attrition(raw_df)
    loo = leave_one_out(attrition["pre_required_df"])
    cond_miss, cand_miss = missingness_report(raw_df)

    n_raw = attrition["n_raw"]
    n_final = attrition["stages"][-1]["remaining"]
    unrecognized = attrition["unrecognized_codes"]
    cpt_43645 = int(unrecognized.get(GASTRIC_BYPASS_VARIANT_CPT, 0))
    creat_loo = int(
        loo.loc[loo["required_feature"] == "creatinine_at_surgery", "rows_recovered_if_demoted_alone"].iloc[0]
    ) if (loo["required_feature"] == "creatinine_at_surgery").any() else 0

    lines: list[str] = []
    lines.append("MBSCohort attrition / missingness debug report  (READ-ONLY)")
    lines.append(f"generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"source    : {source_label}")
    lines.append("=" * 72)

    # (d) STAGES
    lines.append("")
    lines.append(f"[STAGES] attrition decomposition (denominator = raw loaded = {n_raw})")
    if db_counts and "true_total" in db_counts:
        true_total = db_counts["true_total"]
        sql_removed = true_total - n_raw
        lines.append(f"  true DB total (no WHERE) .......... {true_total}")
        lines.append(f"  - SQL WHERE filter losses ......... -{sql_removed:<7d} -> {n_raw}   ({pct(sql_removed, true_total)} of true total)")
    else:
        lines.append("  (SQL-filter losses: VM/--db only; the CSV export is already post-SQL)")
    for stage in attrition["stages"]:
        if stage["dropped"] == 0 and stage["stage"].startswith("raw loaded"):
            lines.append(f"  {stage['stage']:.<34} {stage['remaining']}")
        else:
            lines.append(
                f"  - {stage['stage']:.<32} -{stage['dropped']:<7d} -> {stage['remaining']}"
                f"   ({pct(stage['dropped'], n_raw)} of raw)"
            )
    lines.append(f"  FINAL N .......................... {n_final}   ({pct(n_final, n_raw)} of raw retained)")
    if attrition["dup_patkey"]:
        lines.append(f"  [warn] duplicate PatKey rows: {attrition['dup_patkey']} (fm.prepare_flow_dataset RAISES on these)")

    # (c) CPT
    lines.append("")
    lines.append("[CPT] unrecognized CptCode values dropped (not mapped to sleeve/rnygb):")
    lines.append(
        f"  {GASTRIC_BYPASS_VARIANT_CPT} : {cpt_43645}"
        f"   <-- gastric-bypass variant; candidate to map -> rnygb (currently dropped)"
    )
    others = [(str(code), int(count)) for code, count in unrecognized.items() if str(code) != GASTRIC_BYPASS_VARIANT_CPT]
    if others:
        for code, count in others:
            lines.append(f"  {code} : {count}")
    else:
        lines.append("  (no other unrecognized CPT codes)")

    # (b) leave-one-out
    lines.append("")
    lines.append(
        f"[LOO] leave-one-out on REQUIRED_PATIENT_FEATURES "
        f"(rows recovered if a field ALONE were demoted; currently dropped for missing-required = {loo.attrs['currently_dropped']}):"
    )
    for _, row in loo.iterrows():
        lines.append(f"  {row['required_feature']:.<28} +{row['rows_recovered_if_demoted_alone']}")
    lines.append("  (report only -- REQUIRED_PATIENT_FEATURES is NOT changed by this script)")

    # (a) missingness
    lines.append("")
    lines.append(f"[MISS] per-feature missingness (denominator = raw loaded = {n_raw})")
    lines.append("  -- conditioning features --")
    for _, row in cond_miss.iterrows():
        lines.append(f"  {row['feature']:.<28} {int(row['missing'])} ({row['missing_pct']:.2f}%)   [{row['source']}]")
    lines.append("  -- candidate features --")
    for _, row in cand_miss.iterrows():
        lines.append(f"  {row['feature']:.<28} {int(row['missing'])} ({row['missing_pct']:.2f}%)   [{row['source']}]")

    # (d, VM) per-clause detail
    if db_counts:
        lines.append("")
        lines.append("[DB] pre-filter denominator + per-WHERE-clause marginal removal (VM only)")
        if "error" in db_counts and "true_total" not in db_counts:
            lines.append(f"  (unavailable: {db_counts['error']})")
        else:
            lines.append(f"  true total (no WHERE) : {db_counts.get('true_total', 'n/a')}")
            lines.append("  per-clause rows excluded ALONE (NULL-sensitive; clauses overlap, do not sum):")
            for entry in db_counts.get("clauses", []):
                if entry.get("rows_excluded_alone") is None:
                    lines.append(f"    [err] {entry['clause']} : {entry.get('error', 'failed')}")
                else:
                    lines.append(f"    {entry['clause']} : {entry['rows_excluded_alone']}")

    lines.append("")
    lines.append("=" * 72)
    lines.append(
        "note: stage counts reconcile as raw - (sum of drops) = FINAL N. "
        "Numbers are informational; no data or config is modified."
    )

    headline_bits = {
        "n_final": n_final,
        "n_raw": n_raw,
        "retained_pct": pct(n_final, n_raw),
        "creatinine_loo": creat_loo,
        "cpt_43645": cpt_43645,
    }
    return "\n".join(lines) + "\n", headline_bits


def write_report(text: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"attrition_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(csv_path: Path | None, use_db: bool, output_dir: Path) -> Path:
    if use_db:
        raw_df = load_raw_db()
        source_label = "Cosmos MBSCohort (DB, post-SQL WHERE)"
        db_counts = db_prefilter_counts()
    else:
        if csv_path is None:
            raise ValueError("csv_path is required when --db is not set")
        raw_df = load_raw_csv(csv_path)
        source_label = str(csv_path)
        db_counts = None

    text, headline = build_report(source_label, raw_df, db_counts)
    path = write_report(text, output_dir)
    print(
        f"[debug_attrition] wrote {path} | "
        f"final N={headline['n_final']}/{headline['n_raw']} raw ({headline['retained_pct']} retained) | "
        f"creatinine LOO=+{headline['creatinine_loo']} rows | "
        f"CPT {GASTRIC_BYPASS_VARIANT_CPT}={headline['cpt_43645']}",
        flush=True,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--csv", "--csv-path", dest="csv_path", type=Path, default=None, help="Local CSV export (post-SQL).")
    source.add_argument("--db", action="store_true", help="Query Cosmos MBSCohort (the default when no --csv is given).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # Default to DB mode when no CSV is supplied, so a bare run works on the VM.
    use_db = args.db or args.csv_path is None
    try:
        run(csv_path=args.csv_path, use_db=use_db, output_dir=args.output_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}\n\nPass --csv <path> to run locally from a saved CSV export.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
