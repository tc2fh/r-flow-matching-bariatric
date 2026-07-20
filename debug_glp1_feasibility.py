"""Read-only GLP-1 and incretin study-feasibility diagnostic for Cosmos.

The default invocation is intended for the Cosmos VM:

    python debug_glp1_feasibility.py

It reads the raw ``dbo.MBSCohort`` and ``dbo.GLP1Cohort`` tables without the
historical modelling WHERE clauses, discovers the live schema, and writes only
aggregate diagnostics. It never trains a model and never writes patient-level
records.

For a local plumbing check with the repository's synthetic cohorts:

    python debug_glp1_feasibility.py --mbs-csv fake_data/fake_mbs_cohort.csv --glp1-csv fake_data/fake_glp1_cohort.csv

The output is one timestamped directory under ``runs/debug_glp1_feasibility``.
It includes schema and tag inventories, dose and episode-quality summaries,
intersection-aware cohort funnels, treatment-timing summaries, wide-outcome
support by candidate arm, design requirement gates, and a text report.

Important boundaries:

* Medication-name normalization is deliberately conservative and is emitted as
  a reviewable proposal, not silently treated as ground truth.
* Doses are never converted or pooled across units.
* ``PostOpGLP1 == 0`` means no recorded postoperative use in these tables. It
  does not prove non-use outside the contributing health system.
* The script diagnoses whether the data can support a design. It does not
  estimate treatment effects.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

SCRIPT_VERSION = "glp1-feasibility-v1.0.0"
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "debug_glp1_feasibility"
DEFAULT_SCHEMA = "dbo"
DEFAULT_MBS_TABLE = "MBSCohort"
DEFAULT_GLP1_TABLE = "GLP1Cohort"
DAYS_PER_MONTH = 30.4375
CPT_TO_PROCEDURE = {
    "43775": "sleeve",
    "43644": "rnygb",
    "43645": "rnygb",
    "43846": "rnygb",
}

NULL_TEXT = {"", "na", "n/a", "nan", "none", "null", "nat"}
PLACEHOLDER_TEXT = {
    "#masked",
    "*masked",
    "*not applicable",
    "*unspecified",
    "masked",
    "not applicable",
    "other",
    "unknown",
    "unspecified",
}


CONCEPT_ALIASES: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        ("patient_id", ("PatKey", "PatientKey", "PatientID")),
        (
            "center_id",
            (
                "CenterID",
                "CenterKey",
                "SiteID",
                "SiteKey",
                "OrganizationID",
                "OrganizationKey",
                "ContributingOrganizationID",
                "HealthSystemID",
                "HealthSystemKey",
                "FacilityID",
                "FacilityKey",
            ),
        ),
        ("procedure_code", ("CptCode", "CPTCode", "ProcedureCode")),
        ("procedure_date", ("ProcDateValue", "ProcedureDateValue", "IndexProcedureDate")),
        ("age", ("AgeAtEvent", "AgeAtIndex")),
        ("sex", ("Sex", "Gender")),
        ("coverage", ("CoverageClass", "PayerClass")),
        ("prior_glp1", ("PriorGLP1", "PriorGLP", "PreOpGLP1")),
        ("postop_glp1", ("PostOpGLP1", "PostoperativeGLP1")),
        ("glp1_start", ("GLP1StartDate", "IncretinStartDate", "MedicationStartDate")),
        ("glp1_end", ("GLP1EndDate", "IncretinEndDate", "MedicationEndDate")),
        ("glp1_duration_days", ("GLP1Duration", "GLP1DurationDays", "MedicationDurationDays")),
        ("glp1_interval_days", ("GLP1Interval", "PostOpGLP1Interval", "DaysToGLP1")),
        ("glp1_name", ("GLP1Name", "IncretinName", "MedicationName", "GenericName", "IngredientName")),
        ("glp1_route", ("GLP1Route", "MedicationRoute", "Route")),
        ("max_glp1_dose", ("MaxGLP1Dose", "MaximumGLP1Dose", "MaxDose")),
        ("recent_glp1_dose", ("MostRecentDose", "RecentGLP1Dose", "CurrentDose")),
        ("glp1_dose_unit", ("MostRecentDoseUnit", "GLP1DoseUnit", "DoseUnit")),
        ("mbs_during_glp1", ("MBSduringGLP1", "MBSDuringGLP1")),
        ("baseline_bmi", ("BMIatEvent", "BMIAtIndex", "BaselineBMI")),
        ("baseline_weight", ("WeightAtEvent", "WeightAtIndex", "BaselineWeight")),
        ("baseline_hba1c", ("HbA1cAtEvent", "HbA1cAtIndex", "BaselineHbA1c")),
        ("baseline_creatinine", ("CreatinineAtEvent", "BaselineCreatinine")),
        ("baseline_egfr", ("eGFRatEvent", "BaselineEGFR", "eGFRAtIndex")),
        ("prior_mbs", ("PMH_PriorMBS", "PriorMBS")),
        ("diabetes", ("PMH_DM2", "T2D", "Type2Diabetes")),
        ("dialysis_transplant", ("PMH_dialysis_transplant", "PriorDialysisTransplant")),
        ("baseline_retinopathy", ("PMH_retinopathy", "PriorRetinopathy")),
        ("active_end_days", ("ActiveEndInterval", "ObservableFollowupDays", "FollowupDays")),
        ("death", ("Deceased", "Death")),
        ("death_interval_days", ("DeathInterval", "DaysToDeath")),
        ("mace", ("MACE",)),
        ("mace_interval_days", ("MACEinterval", "MACEInterval")),
        ("nephropathy", ("Nephropathy",)),
        ("nephropathy_interval_days", ("NephropathyInterval",)),
        ("retinopathy", ("Retinopathy",)),
        ("retinopathy_interval_days", ("RetinopathyInterval",)),
    ]
)


WIDE_OUTCOMES: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("bmi", 3, ("BMI3mPostEvent",)),
    ("bmi", 6, ("BMI6mPostEvent",)),
    ("bmi", 9, ("BMI9mPostEvent",)),
    ("bmi", 12, ("BMI12mPostEvent",)),
    ("bmi", 24, ("BMI2yPostEvent", "BMI24mPostEvent")),
    ("bmi", 36, ("BMI3yPostEvent", "BMI36mPostEvent")),
    ("bmi", 48, ("BMI4yPostEvent", "BMI48mPostEvent")),
    ("bmi", 60, ("BMI5yPostEvent", "BMI60mPostEvent")),
    ("bmi", 72, ("BMI6yPostEvent", "BMI72mPostEvent")),
    ("weight", 3, ("Weight3mPostEvent",)),
    ("weight", 6, ("Weight6mPostEvent",)),
    ("weight", 9, ("Weight9mPostEvent",)),
    ("weight", 12, ("Weight12mPostEvent",)),
    ("weight", 24, ("Weight2yPostEvent", "Weight24mPostEvent")),
    ("weight", 36, ("Weight3yPostEvent", "Weight36mPostEvent")),
    ("weight", 48, ("Weight4yPostEvent", "Weight48mPostEvent")),
    ("weight", 60, ("Weight5yPostEvent", "Weight60mPostEvent")),
    ("weight", 72, ("Weight6yPostEvent", "Weight72mPostEvent")),
    ("hba1c", 12, ("HbA1c12mPostEvent",)),
    ("hba1c", 24, ("HbA1c2yPostEvent", "HbA1c24mPostEvent")),
    ("hba1c", 36, ("HbA1c3yPostEvent", "HbA1c36mPostEvent")),
    ("hba1c", 48, ("HbA1c4yPostEvent", "HbA1c48mPostEvent")),
    ("hba1c", 60, ("HbA1c5yPostEvent", "HbA1c60mPostEvent")),
    ("hba1c", 72, ("HbA1c6yPostEvent", "HbA1c72mPostEvent")),
)


MEDICATION_SCHEMA_PATTERNS: OrderedDict[str, re.Pattern[str]] = OrderedDict(
    [
        (
            "agent_or_product",
            re.compile(
                r"glp|incretin|semaglut|tirzepat|liraglut|dulaglut|exenat|lixisen|albiglut|"
                r"drug|medication|ingredient|generic|brand|product",
                re.IGNORECASE,
            ),
        ),
        ("medication_code", re.compile(r"rxnorm|rxcui|ndc|gpi|atc|medicationid|drugid|conceptid", re.IGNORECASE)),
        ("dose_or_strength", re.compile(r"dose|strength|quantity|dayssupply|day_supply", re.IGNORECASE)),
        ("route_or_form", re.compile(r"route|form|formulation", re.IGNORECASE)),
        ("frequency_or_sig", re.compile(r"frequency|sig|schedule|directions", re.IGNORECASE)),
        ("medication_episode", re.compile(r"start|end|discontinu|order|fill|admin|dispens|prescri", re.IGNORECASE)),
        ("indication", re.compile(r"indication|reasonforuse|medicationdiagnosis", re.IGNORECASE)),
    ]
)


AGENT_RULES: tuple[tuple[str, str, str, re.Pattern[str]], ...] = (
    (
        "tirzepatide",
        "dual_gip_glp1_agonist",
        "ingredient_or_brand",
        re.compile(r"tirzepatide|mounjaro|zepbound", re.IGNORECASE),
    ),
    (
        "semaglutide",
        "glp1_receptor_agonist",
        "ingredient_or_brand",
        re.compile(r"semaglutide|ozempic|wegovy|rybelsus", re.IGNORECASE),
    ),
    (
        "liraglutide",
        "glp1_receptor_agonist",
        "ingredient_or_brand",
        re.compile(r"liraglutide|victoza|saxenda|xultophy", re.IGNORECASE),
    ),
    (
        "dulaglutide",
        "glp1_receptor_agonist",
        "ingredient_or_brand",
        re.compile(r"dulaglutide|trulicity", re.IGNORECASE),
    ),
    (
        "exenatide",
        "glp1_receptor_agonist",
        "ingredient_or_brand",
        re.compile(r"exenatide|byetta|bydureon", re.IGNORECASE),
    ),
    (
        "lixisenatide",
        "glp1_receptor_agonist",
        "ingredient_or_brand",
        re.compile(r"lixisenatide|adlyxin|soliqua", re.IGNORECASE),
    ),
    (
        "albiglutide",
        "glp1_receptor_agonist",
        "ingredient_or_brand",
        re.compile(r"albiglutide|tanzeum", re.IGNORECASE),
    ),
)


@dataclass
class LoadedTable:
    logical_name: str
    source_name: str
    frame: pd.DataFrame
    schema: pd.DataFrame
    concepts: dict[str, str | None]
    ambiguities: dict[str, list[str]]


@dataclass
class AnalysisGroup:
    design: str
    population: str
    arm: str
    table: str
    frame: pd.DataFrame


def normalize_identifier(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def quote_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return "[" + value + "]"


def missing_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        normalized = series.astype("string").str.strip().str.lower()
        mask = mask | normalized.isin(NULL_TEXT)
    return mask.fillna(True)


def placeholder_mask(series: pd.Series) -> pd.Series:
    if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
        return pd.Series(False, index=series.index, dtype=bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(PLACEHOLDER_TEXT).fillna(False)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.where(~missing_mask(series)), errors="coerce")


def dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.where(~missing_mask(series)), errors="coerce")


def clean_text(series: pd.Series, missing_label: str = "<missing>") -> pd.Series:
    value = series.astype("string").str.strip()
    return value.mask(missing_mask(series), missing_label).fillna(missing_label)


def resolve_column(columns: Sequence[str], aliases: Sequence[str]) -> tuple[str | None, list[str]]:
    normalized: dict[str, list[str]] = {}
    for column in columns:
        normalized.setdefault(normalize_identifier(column), []).append(str(column))
    for alias in aliases:
        matches = normalized.get(normalize_identifier(alias), [])
        if len(matches) == 1:
            return matches[0], []
        if len(matches) > 1:
            return None, matches
    return None, []


def resolve_concepts(columns: Sequence[str]) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    resolved: dict[str, str | None] = {}
    ambiguous: dict[str, list[str]] = {}
    for concept, aliases in CONCEPT_ALIASES.items():
        column, matches = resolve_column(columns, aliases)
        resolved[concept] = column
        if matches:
            ambiguous[concept] = matches
    return resolved, ambiguous


def first_column(frame: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    return resolve_column(list(frame.columns), aliases)[0]


def concept_series(table: LoadedTable, concept: str) -> pd.Series | None:
    column = table.concepts.get(concept)
    return None if column is None else table.frame[column]


def normalize_route(value: Any) -> str:
    if value is None or pd.isna(value):
        return "<missing>"
    text = str(value).strip().lower()
    if text in NULL_TEXT:
        return "<missing>"
    if re.search(r"subcut|\bsc\b|\bsq\b|injection", text):
        return "subcutaneous"
    if re.search(r"oral|mouth|\bpo\b|tablet", text):
        return "oral"
    return re.sub(r"\s+", " ", text)


def normalize_unit(value: Any) -> str:
    if value is None or pd.isna(value):
        return "<missing>"
    text = str(value).strip().lower().replace("μ", "u").replace("µ", "u")
    if text in NULL_TEXT:
        return "<missing>"
    replacements = {
        "milligram": "mg",
        "milligrams": "mg",
        "microgram": "mcg",
        "micrograms": "mcg",
        "ug": "mcg",
    }
    return replacements.get(text, re.sub(r"\s+", " ", text))


def normalize_agent(value: Any) -> tuple[str, str, str]:
    if value is None or pd.isna(value) or str(value).strip().lower() in NULL_TEXT | {"<missing>"}:
        return "<missing>", "<missing>", "missing"
    text = re.sub(r"\s+", " ", str(value).strip())
    for ingredient, therapy_class, method, pattern in AGENT_RULES:
        if pattern.search(text):
            if ingredient in {"liraglutide", "lixisenatide"} and re.search(
                r"soliqua|xultophy|insulin", text, re.IGNORECASE
            ):
                return ingredient, "fixed_combination_incretin", method
            return ingredient, therapy_class, method
    return "unmapped", "unmapped", "manual_review"


def schema_roles(column: str) -> list[str]:
    roles = [concept for concept, aliases in CONCEPT_ALIASES.items() if normalize_identifier(column) in {normalize_identifier(a) for a in aliases}]
    for role, pattern in MEDICATION_SCHEMA_PATTERNS.items():
        if pattern.search(column):
            roles.append(role)
    return sorted(set(roles))


def csv_schema(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_name": source_name,
                "ordinal_position": index + 1,
                "column_name": column,
                "data_type": str(frame[column].dtype),
                "character_maximum_length": np.nan,
                "is_nullable": "unknown_csv",
            }
            for index, column in enumerate(frame.columns)
        ]
    )


def discover_db_table(connection: Any, requested_schema: str, requested_table: str) -> tuple[str, str, pd.DataFrame]:
    query = """
SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, ORDINAL_POSITION, DATA_TYPE,
       CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE LOWER(TABLE_NAME) = LOWER(?)
ORDER BY CASE WHEN LOWER(TABLE_SCHEMA) = LOWER(?) THEN 0 ELSE 1 END,
         TABLE_SCHEMA, ORDINAL_POSITION
"""
    schema = pd.read_sql_query(query, connection, params=[requested_table, requested_schema])
    if schema.empty:
        raise RuntimeError(f"Table {requested_schema}.{requested_table} was not found in INFORMATION_SCHEMA.COLUMNS")
    candidates = schema[["TABLE_SCHEMA", "TABLE_NAME"]].drop_duplicates()
    preferred = candidates[candidates["TABLE_SCHEMA"].astype(str).str.lower().eq(requested_schema.lower())]
    chosen = preferred.iloc[0] if len(preferred) == 1 else candidates.iloc[0]
    actual_schema = str(chosen["TABLE_SCHEMA"])
    actual_table = str(chosen["TABLE_NAME"])
    schema = schema[
        schema["TABLE_SCHEMA"].astype(str).eq(actual_schema) & schema["TABLE_NAME"].astype(str).eq(actual_table)
    ].copy()
    schema = schema.rename(
        columns={
            "COLUMN_NAME": "column_name",
            "ORDINAL_POSITION": "ordinal_position",
            "DATA_TYPE": "data_type",
            "CHARACTER_MAXIMUM_LENGTH": "character_maximum_length",
            "IS_NULLABLE": "is_nullable",
        }
    )
    schema["source_name"] = actual_schema + "." + actual_table
    return actual_schema, actual_table, schema[
        ["source_name", "ordinal_position", "column_name", "data_type", "character_maximum_length", "is_nullable"]
    ]


def discover_database_schema(connection: Any) -> pd.DataFrame:
    query = """
SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, ORDINAL_POSITION, DATA_TYPE,
       CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
"""
    return pd.read_sql_query(query, connection)


def load_db_table(
    connection: Any,
    logical_name: str,
    requested_schema: str,
    requested_table: str,
    max_rows: int | None,
) -> LoadedTable:
    actual_schema, actual_table, schema = discover_db_table(connection, requested_schema, requested_table)
    top = f"TOP ({int(max_rows)}) " if max_rows is not None else ""
    sql = f"SELECT {top}* FROM {quote_identifier(actual_schema)}.{quote_identifier(actual_table)}"
    frame = pd.read_sql_query(sql, connection)
    concepts, ambiguities = resolve_concepts(list(frame.columns))
    return LoadedTable(
        logical_name=logical_name,
        source_name=f"{actual_schema}.{actual_table}",
        frame=frame,
        schema=schema,
        concepts=concepts,
        ambiguities=ambiguities,
    )


def load_csv_table(logical_name: str, path: Path) -> LoadedTable:
    frame = pd.read_csv(path, low_memory=False)
    concepts, ambiguities = resolve_concepts(list(frame.columns))
    return LoadedTable(
        logical_name=logical_name,
        source_name=str(path),
        frame=frame,
        schema=csv_schema(frame, str(path)),
        concepts=concepts,
        ambiguities=ambiguities,
    )


def load_tables(args: argparse.Namespace) -> tuple[dict[str, LoadedTable], list[str], str, pd.DataFrame]:
    warnings: list[str] = []
    if args.mbs_csv or args.glp1_csv:
        tables: dict[str, LoadedTable] = {}
        if args.mbs_csv:
            tables["MBSCohort"] = load_csv_table("MBSCohort", args.mbs_csv)
        if args.glp1_csv:
            tables["GLP1Cohort"] = load_csv_table("GLP1Cohort", args.glp1_csv)
        if not tables:
            raise RuntimeError("At least one CSV path is required in CSV mode")
        database_schema = pd.concat(
            [
                table.schema.assign(
                    TABLE_SCHEMA="csv",
                    TABLE_NAME=table.logical_name,
                    COLUMN_NAME=table.schema["column_name"],
                    ORDINAL_POSITION=table.schema["ordinal_position"],
                    DATA_TYPE=table.schema["data_type"],
                    CHARACTER_MAXIMUM_LENGTH=table.schema["character_maximum_length"],
                    IS_NULLABLE=table.schema["is_nullable"],
                )[
                    [
                        "TABLE_SCHEMA",
                        "TABLE_NAME",
                        "COLUMN_NAME",
                        "ORDINAL_POSITION",
                        "DATA_TYPE",
                        "CHARACTER_MAXIMUM_LENGTH",
                        "IS_NULLABLE",
                    ]
                ]
                for table in tables.values()
            ],
            ignore_index=True,
        )
        return tables, warnings, "CSV", database_schema

    try:
        import pyodbc  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"pyodbc is required for Cosmos DB mode: {exc}") from exc

    connection_string = os.environ.get("COSMOS_CONNECTION_STRING")
    if not connection_string:
        try:
            from train_flow_matching import CONNECTION_STRING as connection_string
        except Exception as exc:  # noqa: BLE001 - provide an environment-variable fallback
            raise RuntimeError(
                "Could not import the repository Cosmos connection string. Set "
                "COSMOS_CONNECTION_STRING or run from the repository environment."
            ) from exc

    tables = {}
    with pyodbc.connect(connection_string, timeout=args.connection_timeout) as connection:
        try:
            database_schema = discover_database_schema(connection)
        except Exception as exc:  # noqa: BLE001 - cohort diagnostics can continue
            database_schema = pd.DataFrame()
            warnings.append(f"database-wide schema discovery unavailable: {exc}")
        for logical, table_name in (("MBSCohort", args.mbs_table), ("GLP1Cohort", args.glp1_table)):
            try:
                tables[logical] = load_db_table(connection, logical, args.schema, table_name, args.max_rows)
            except Exception as exc:  # noqa: BLE001 - report one missing table without hiding the other
                warnings.append(f"{logical} unavailable: {exc}")
    if not tables:
        raise RuntimeError("Neither MBSCohort nor GLP1Cohort could be loaded")
    return tables, warnings, "Cosmos DB", database_schema


def patient_series(table: LoadedTable) -> pd.Series:
    source = concept_series(table, "patient_id")
    if source is None:
        return pd.Series([f"row-{i}" for i in range(len(table.frame))], index=table.frame.index, dtype="string")
    return clean_text(source)


def derive_table(table: LoadedTable) -> pd.DataFrame:
    frame = table.frame.copy()
    frame["_patient"] = patient_series(table)

    agent_source = concept_series(table, "glp1_name")
    if agent_source is None:
        frame["_agent_raw"] = "<missing>"
    else:
        frame["_agent_raw"] = clean_text(agent_source)
    normalized = frame["_agent_raw"].map(normalize_agent)
    frame["_agent"] = normalized.map(lambda item: item[0])
    frame["_therapy_class"] = normalized.map(lambda item: item[1])
    frame["_agent_match_method"] = normalized.map(lambda item: item[2])

    route_source = concept_series(table, "glp1_route")
    frame["_route"] = "<missing>" if route_source is None else route_source.map(normalize_route)
    unit_source = concept_series(table, "glp1_dose_unit")
    frame["_dose_unit"] = "<missing>" if unit_source is None else unit_source.map(normalize_unit)

    prior_source = concept_series(table, "prior_glp1")
    frame["_prior_glp1"] = np.nan if prior_source is None else numeric(prior_source)
    postop_source = concept_series(table, "postop_glp1")
    frame["_postop_glp1"] = np.nan if postop_source is None else numeric(postop_source)
    followup_source = concept_series(table, "active_end_days")
    frame["_active_end_days"] = np.nan if followup_source is None else numeric(followup_source)

    start_source = concept_series(table, "glp1_start")
    frame["_glp1_start"] = pd.NaT if start_source is None else dates(start_source)
    end_source = concept_series(table, "glp1_end")
    frame["_glp1_end"] = pd.NaT if end_source is None else dates(end_source)
    duration_source = concept_series(table, "glp1_duration_days")
    frame["_glp1_duration_days"] = np.nan if duration_source is None else numeric(duration_source)

    if table.logical_name == "MBSCohort":
        proc_source = concept_series(table, "procedure_code")
        if proc_source is None:
            frame["_procedure"] = "unavailable"
        else:
            normalized_cpt = proc_source.astype("string").str.extract(r"(\d{5})", expand=False)
            frame["_procedure"] = normalized_cpt.map(CPT_TO_PROCEDURE).fillna("other_or_unrecognized")
        proc_date_source = concept_series(table, "procedure_date")
        frame["_index_date"] = pd.NaT if proc_date_source is None else dates(proc_date_source)
        interval_source = concept_series(table, "glp1_interval_days")
        interval = pd.Series(np.nan, index=frame.index, dtype=float) if interval_source is None else numeric(interval_source)
        computed_interval = (frame["_glp1_start"] - frame["_index_date"]).dt.total_seconds() / 86400.0
        frame["_glp1_interval_days"] = interval.where(interval.notna(), computed_interval)
        frame["_arm"] = frame["_procedure"]
    else:
        frame["_procedure"] = "not_applicable"
        frame["_index_date"] = frame["_glp1_start"]
        frame["_glp1_interval_days"] = np.nan
        frame["_arm"] = frame["_agent"]

    return frame


def profile_columns(table: LoadedTable) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(table.frame)
    for column in table.frame.columns:
        series = table.frame[column]
        miss = missing_mask(series)
        placeholders = placeholder_mask(series)
        valid = series[~miss]
        numeric_values = pd.to_numeric(valid, errors="coerce")
        numeric_parse_pct = 100.0 * numeric_values.notna().mean() if len(valid) else np.nan
        date_values = pd.Series(dtype="datetime64[ns]")
        date_parse_pct = np.nan
        if re.search(r"date|time", column, re.IGNORECASE):
            date_values = pd.to_datetime(valid, errors="coerce")
            date_parse_pct = 100.0 * date_values.notna().mean() if len(valid) else np.nan
        is_identifier = table.concepts.get("patient_id") == column
        rows.append(
            {
                "table": table.logical_name,
                "source_name": table.source_name,
                "column": column,
                "roles": "|".join(schema_roles(column)),
                "n_rows": n,
                "n_missing_or_blank": int(miss.sum()),
                "missing_or_blank_pct": 100.0 * miss.mean() if n else np.nan,
                "n_placeholder_values": int(placeholders.sum()),
                "placeholder_pct": 100.0 * placeholders.mean() if n else np.nan,
                "n_distinct_nonmissing": int(valid.astype("string").nunique(dropna=True)),
                "numeric_parse_pct": numeric_parse_pct,
                "numeric_min": np.nan if is_identifier or numeric_values.notna().sum() == 0 else float(numeric_values.min()),
                "numeric_max": np.nan if is_identifier or numeric_values.notna().sum() == 0 else float(numeric_values.max()),
                "date_parse_pct": date_parse_pct,
                "date_min": "" if date_values.notna().sum() == 0 else date_values.min().date().isoformat(),
                "date_max": "" if date_values.notna().sum() == 0 else date_values.max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def schema_inventory(tables: Mapping[str, LoadedTable]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for table in tables.values():
        schema = table.schema.copy()
        schema.insert(0, "table", table.logical_name)
        schema["roles"] = schema["column_name"].map(lambda value: "|".join(schema_roles(str(value))))
        schema["medication_related"] = schema["roles"].str.contains(
            "agent_or_product|medication_code|dose_or_strength|route_or_form|frequency_or_sig|medication_episode|indication",
            regex=True,
        )
        rows.append(schema)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def database_medication_schema_candidates(database_schema: pd.DataFrame) -> pd.DataFrame:
    """Surface medication source fields beyond the two analytic cohort tables.

    This is schema-only. It intentionally does not query values from arbitrary
    source tables, which could be extremely large or contain free text.
    """
    if database_schema.empty:
        return pd.DataFrame()
    required = {"TABLE_SCHEMA", "TABLE_NAME", "COLUMN_NAME"}
    if not required.issubset(database_schema.columns):
        return pd.DataFrame()

    table_pattern = re.compile(r"med|drug|pharm|rx|prescri|order|dispens|admin|incretin|glp", re.IGNORECASE)
    strong_column_pattern = re.compile(
        r"glp|incretin|semaglut|tirzepat|liraglut|dulaglut|exenat|lixisen|albiglut|"
        r"rxnorm|rxcui|ndc|gpi|atc|ingredient|generic|brand|product|dose|strength|"
        r"route|formulation|dayssupply|refill|frequency|sig|indication",
        re.IGNORECASE,
    )
    frame = database_schema.copy()
    table_relevant = frame["TABLE_NAME"].astype(str).map(lambda value: bool(table_pattern.search(value)))
    column_relevant = frame["COLUMN_NAME"].astype(str).map(lambda value: bool(strong_column_pattern.search(value)))
    frame = frame.loc[table_relevant | column_relevant].copy()
    if frame.empty:
        return frame
    frame["roles"] = frame["COLUMN_NAME"].astype(str).map(lambda value: "|".join(schema_roles(value)))
    frame["table_name_medication_related"] = frame["TABLE_NAME"].astype(str).map(lambda value: bool(table_pattern.search(value)))
    frame["column_name_medication_related"] = frame["COLUMN_NAME"].astype(str).map(
        lambda value: bool(strong_column_pattern.search(value))
    )
    counts = frame.groupby(["TABLE_SCHEMA", "TABLE_NAME"])["column_name_medication_related"].transform("sum")
    frame["candidate_columns_in_table"] = counts
    return frame.sort_values(
        ["candidate_columns_in_table", "TABLE_SCHEMA", "TABLE_NAME", "ORDINAL_POSITION"],
        ascending=[False, True, True, True],
        ignore_index=True,
    )


def concept_inventory(tables: Mapping[str, LoadedTable]) -> pd.DataFrame:
    rows = []
    for table in tables.values():
        for concept, aliases in CONCEPT_ALIASES.items():
            rows.append(
                {
                    "table": table.logical_name,
                    "concept": concept,
                    "resolved_column": table.concepts.get(concept) or "",
                    "status": "ambiguous" if concept in table.ambiguities else ("present" if table.concepts.get(concept) else "absent"),
                    "ambiguous_matches": "|".join(table.ambiguities.get(concept, [])),
                    "aliases_checked": "|".join(aliases),
                }
            )
    return pd.DataFrame(rows)


def medication_tag_values(table: LoadedTable, derived: pd.DataFrame, top_values: int) -> pd.DataFrame:
    candidate_columns: set[str] = set()
    for concept in ("glp1_name", "glp1_route", "glp1_dose_unit"):
        column = table.concepts.get(concept)
        if column:
            candidate_columns.add(column)
    for column in table.frame.columns:
        roles = schema_roles(column)
        explicit_label = re.search(r"name|generic|brand|product|ingredient", str(column), re.IGNORECASE)
        if explicit_label or set(roles).intersection({"medication_code", "route_or_form", "frequency_or_sig", "indication"}):
            candidate_columns.add(column)

    rows: list[dict[str, Any]] = []
    for column in sorted(candidate_columns):
        values = clean_text(table.frame[column])
        counts = values.value_counts(dropna=False)
        complete_tag_columns = {
            table.concepts.get("glp1_name"),
            table.concepts.get("glp1_route"),
            table.concepts.get("glp1_dose_unit"),
        }
        if column not in complete_tag_columns:
            counts = counts.head(top_values)
        for rank, (raw_value, count) in enumerate(counts.items(), start=1):
            ingredient, therapy_class, method = normalize_agent(raw_value) if column == table.concepts.get("glp1_name") else ("", "", "not_agent_field")
            rows.append(
                {
                    "table": table.logical_name,
                    "column": column,
                    "rank": rank,
                    "raw_value": raw_value,
                    "proposed_ingredient": ingredient,
                    "proposed_therapy_class": therapy_class,
                    "mapping_method": method,
                    "needs_manual_review": method == "manual_review",
                    "n_rows": int(count),
                    "n_patients": int(derived.loc[values.eq(raw_value), "_patient"].nunique()),
                    "column_distinct_nonmissing": int(values[values.ne("<missing>")].nunique()),
                }
            )
    return pd.DataFrame(rows)


def coverage_by_agent(table: LoadedTable, derived: pd.DataFrame) -> pd.DataFrame:
    concepts = (
        "glp1_name",
        "glp1_route",
        "max_glp1_dose",
        "recent_glp1_dose",
        "glp1_dose_unit",
        "glp1_start",
        "glp1_end",
        "glp1_duration_days",
        "glp1_interval_days",
    )
    rows: list[dict[str, Any]] = []
    for agent, group in derived.groupby("_agent", dropna=False):
        for concept in concepts:
            column = table.concepts.get(concept)
            if column is None:
                present = pd.Series(False, index=group.index)
            else:
                present = ~missing_mask(table.frame.loc[group.index, column])
            rows.append(
                {
                    "table": table.logical_name,
                    "agent": agent,
                    "concept": concept,
                    "source_column": column or "",
                    "n_rows": len(group),
                    "n_patients": group["_patient"].nunique(),
                    "n_nonmissing": int(present.sum()),
                    "nonmissing_pct": 100.0 * present.mean() if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def dose_profiles(table: LoadedTable, derived: pd.DataFrame, top_values: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    for concept in ("max_glp1_dose", "recent_glp1_dose"):
        column = table.concepts.get(concept)
        if column is None:
            profile_rows.append(
                {
                    "table": table.logical_name,
                    "agent": "all",
                    "route": "all",
                    "dose_unit": "all",
                    "dose_field": concept,
                    "source_column": "",
                    "n_rows": len(derived),
                    "n_patients": derived["_patient"].nunique(),
                    "n_dose_nonmissing": 0,
                    "dose_nonmissing_pct": 0.0,
                    "status": "field_absent",
                }
            )
            continue
        dose = numeric(table.frame[column])
        work = derived[["_patient", "_agent", "_route", "_dose_unit"]].copy()
        work["_dose"] = dose
        for keys, group in work.groupby(["_agent", "_route", "_dose_unit"], dropna=False):
            valid = group["_dose"].dropna()
            quantiles = valid.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if len(valid) else pd.Series(dtype=float)
            profile_rows.append(
                {
                    "table": table.logical_name,
                    "agent": keys[0],
                    "route": keys[1],
                    "dose_unit": keys[2],
                    "dose_field": concept,
                    "source_column": column,
                    "n_rows": len(group),
                    "n_patients": group["_patient"].nunique(),
                    "n_dose_nonmissing": len(valid),
                    "dose_nonmissing_pct": 100.0 * len(valid) / len(group) if len(group) else np.nan,
                    "n_distinct_dose_values": valid.nunique(),
                    "n_zero": int(valid.eq(0).sum()),
                    "n_negative": int(valid.lt(0).sum()),
                    "min": valid.min() if len(valid) else np.nan,
                    "p01": quantiles.get(0.01, np.nan),
                    "p05": quantiles.get(0.05, np.nan),
                    "p25": quantiles.get(0.25, np.nan),
                    "median": quantiles.get(0.5, np.nan),
                    "p75": quantiles.get(0.75, np.nan),
                    "p95": quantiles.get(0.95, np.nan),
                    "p99": quantiles.get(0.99, np.nan),
                    "max": valid.max() if len(valid) else np.nan,
                    "status": "profiled",
                }
            )
            for rank, (value, count) in enumerate(valid.value_counts().head(top_values).items(), start=1):
                frequency_rows.append(
                    {
                        "table": table.logical_name,
                        "agent": keys[0],
                        "route": keys[1],
                        "dose_unit": keys[2],
                        "dose_field": concept,
                        "rank": rank,
                        "dose_value": value,
                        "n_rows": int(count),
                    }
                )
    return pd.DataFrame(profile_rows), pd.DataFrame(frequency_rows)


def episode_quality(table: LoadedTable, derived: pd.DataFrame) -> pd.DataFrame:
    start = derived["_glp1_start"]
    end = derived["_glp1_end"]
    duration = derived["_glp1_duration_days"]
    date_duration = (end - start).dt.total_seconds() / 86400.0
    discrepancy = duration - date_duration

    max_col = table.concepts.get("max_glp1_dose")
    recent_col = table.concepts.get("recent_glp1_dose")
    max_dose = pd.Series(np.nan, index=derived.index) if max_col is None else numeric(table.frame[max_col])
    recent_dose = pd.Series(np.nan, index=derived.index) if recent_col is None else numeric(table.frame[recent_col])

    rows = []
    for agent, group in derived.groupby("_agent", dropna=False):
        idx = group.index
        valid_discrepancy = discrepancy.loc[idx].dropna()
        rows.append(
            {
                "table": table.logical_name,
                "agent": agent,
                "n_rows": len(group),
                "n_patients": group["_patient"].nunique(),
                "start_nonmissing_pct": 100.0 * start.loc[idx].notna().mean() if len(group) else np.nan,
                "end_nonmissing_pct": 100.0 * end.loc[idx].notna().mean() if len(group) else np.nan,
                "duration_nonmissing_pct": 100.0 * duration.loc[idx].notna().mean() if len(group) else np.nan,
                "n_end_before_start": int((end.loc[idx] < start.loc[idx]).fillna(False).sum()),
                "n_negative_duration": int(duration.loc[idx].lt(0).fillna(False).sum()),
                "n_duration_date_pairs": len(valid_discrepancy),
                "median_duration_minus_dates_days": valid_discrepancy.median() if len(valid_discrepancy) else np.nan,
                "p95_abs_duration_discrepancy_days": valid_discrepancy.abs().quantile(0.95) if len(valid_discrepancy) else np.nan,
                "n_recent_dose_above_recorded_max": int(
                    (recent_dose.loc[idx].notna() & max_dose.loc[idx].notna() & recent_dose.loc[idx].gt(max_dose.loc[idx])).sum()
                ),
                "n_dose_present_unit_missing": int(
                    ((recent_dose.loc[idx].notna() | max_dose.loc[idx].notna()) & group["_dose_unit"].eq("<missing>")).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def exposure_reconciliation(table: LoadedTable, derived: pd.DataFrame) -> pd.DataFrame:
    """Cross-check flags, dates, and surgery-relative intervals without repairing them."""
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", derived)]
    groups.extend((str(agent), group) for agent, group in derived.groupby("_agent", dropna=False))

    explicit_interval_col = table.concepts.get("glp1_interval_days")
    explicit_interval = (
        pd.Series(np.nan, index=derived.index)
        if explicit_interval_col is None
        else numeric(table.frame[explicit_interval_col])
    )
    computed_interval = (derived["_glp1_start"] - derived["_index_date"]).dt.total_seconds() / 86400.0
    interval_difference = explicit_interval - computed_interval
    active_at_index = (
        derived["_index_date"].notna()
        & derived["_glp1_start"].notna()
        & derived["_glp1_start"].le(derived["_index_date"])
        & (derived["_glp1_end"].isna() | derived["_glp1_end"].ge(derived["_index_date"]))
    )
    if table.logical_name != "MBSCohort":
        active_at_index = pd.Series(False, index=derived.index)

    for label, group in groups:
        idx = group.index
        prior = group["_prior_glp1"]
        postop = group["_postop_glp1"]
        timing = group["_glp1_interval_days"]
        paired_difference = interval_difference.loc[idx].dropna()
        rows.append(
            {
                "table": table.logical_name,
                "agent": label,
                "n_rows": len(group),
                "n_patients": group["_patient"].nunique(),
                "n_prior_glp1_1": int(prior.eq(1).sum()),
                "n_prior_glp1_missing": int(prior.isna().sum()),
                "n_postop_glp1_1": int(postop.eq(1).sum()),
                "n_postop_flag_1_timing_missing": int((postop.eq(1) & timing.isna()).sum()),
                "n_postop_flag_0_nonnegative_timing": int((postop.eq(0) & timing.ge(0)).sum()),
                "n_postop_flag_1_negative_timing": int((postop.eq(1) & timing.lt(0)).sum()),
                "n_prior_flag_0_negative_timing": int((prior.eq(0) & timing.lt(0)).sum()),
                "n_episode_spans_index_date": int(active_at_index.loc[idx].sum()),
                "n_explicit_and_date_derived_interval_pairs": len(paired_difference),
                "median_explicit_minus_derived_interval_days": paired_difference.median() if len(paired_difference) else np.nan,
                "p95_abs_explicit_minus_derived_interval_days": paired_difference.abs().quantile(0.95) if len(paired_difference) else np.nan,
                "n_interval_disagreement_over_7_days": int(paired_difference.abs().gt(7).sum()),
                "n_interval_disagreement_over_30_days": int(paired_difference.abs().gt(30).sum()),
                "note": (
                    "active-at-index is a date-derived diagnostic only; verify which episode the wide GLP fields represent"
                    if table.logical_name == "MBSCohort"
                    else "no surgery index exists in GLP1Cohort, so surgery-relative diagnostics are not applicable"
                ),
            }
        )
    return pd.DataFrame(rows)


def base_criterion(
    table: LoadedTable,
    derived: pd.DataFrame,
    concept: str,
    predicate: Callable[[pd.Series], pd.Series],
) -> tuple[pd.Series | None, str]:
    source = concept_series(table, concept)
    if source is None:
        return None, f"{concept} field absent"
    return predicate(source).fillna(False), table.concepts.get(concept) or concept


def candidate_masks(table: LoadedTable, derived: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    all_rows = pd.Series(True, index=derived.index)
    stages: list[tuple[str, str, pd.Series | None, str]] = []
    if table.logical_name == "MBSCohort":
        stages.append(("structural", "recognized sleeve or RYGB", derived["_procedure"].isin(["sleeve", "rnygb"]), "CPT mapping"))
    else:
        stages.append(("structural", "recognized incretin agent", ~derived["_agent"].isin(["<missing>", "unmapped"]), "GLP1Name mapping"))

    age_mask, age_source = base_criterion(table, derived, "age", lambda value: numeric(value).ge(18))
    bmi_mask, bmi_source = base_criterion(table, derived, "baseline_bmi", lambda value: numeric(value).between(35, 75))
    prior_mbs_mask, prior_mbs_source = base_criterion(table, derived, "prior_mbs", lambda value: numeric(value).eq(0))
    diabetes_mask, diabetes_source = base_criterion(table, derived, "diabetes", lambda value: numeric(value).eq(1))
    renal_history_mask, renal_history_source = base_criterion(
        table, derived, "dialysis_transplant", lambda value: numeric(value).eq(0)
    )
    egfr_mask, egfr_source = base_criterion(
        table, derived, "baseline_egfr", lambda value: numeric(value).isna() | numeric(value).ge(20)
    )
    retino_mask, retino_source = base_criterion(
        table, derived, "baseline_retinopathy", lambda value: numeric(value).eq(0)
    )
    stages.extend(
        [
            ("structural", "adult age at index", age_mask, age_source),
            ("structural", "baseline BMI 35-75", bmi_mask, bmi_source),
            ("structural", "no prior bariatric surgery", prior_mbs_mask, prior_mbs_source),
            ("t2d", "type 2 diabetes", diabetes_mask, diabetes_source),
            ("strict_incident", "no dialysis or transplant history", renal_history_mask, renal_history_source),
            ("strict_incident", "baseline eGFR missing or at least 20", egfr_mask, egfr_source),
            ("strict_incident", "no baseline retinopathy", retino_mask, retino_source),
        ]
    )

    running = all_rows.copy()
    rows: list[dict[str, Any]] = []
    masks: dict[str, pd.Series] = {"raw": running.copy()}
    for section, criterion, mask, source in stages:
        before = running.copy()
        applied = mask is not None
        if applied:
            running &= mask
        rows.append(
            {
                "table": table.logical_name,
                "section": section,
                "criterion": criterion,
                "source": source,
                "applied": applied,
                "n_rows_before": int(before.sum()),
                "n_rows_dropped": int((before & ~running).sum()),
                "n_rows_remaining": int(running.sum()),
                "n_patients_remaining": int(derived.loc[running, "_patient"].nunique()),
                "note": "" if applied else "not applied because source was absent",
            }
        )
        if criterion == "no prior bariatric surgery":
            masks["obesity_structural"] = running.copy()
        if criterion == "type 2 diabetes":
            masks["t2d_structural"] = running.copy()
        if criterion == "no baseline retinopathy":
            masks["strict_incident"] = running.copy()

    if "obesity_structural" not in masks:
        masks["obesity_structural"] = running.copy()
    if "t2d_structural" not in masks:
        masks["t2d_structural"] = masks["obesity_structural"].copy()
    if "strict_incident" not in masks:
        masks["strict_incident"] = masks["t2d_structural"].copy()

    legacy = masks["strict_incident"].copy()
    legacy_stages: list[tuple[str, pd.Series | None, str]] = []
    prior = concept_series(table, "prior_glp1")
    legacy_stages.append(("PriorGLP1 = 0", None if prior is None else numeric(prior).eq(0), table.concepts.get("prior_glp1") or "absent"))
    followup = concept_series(table, "active_end_days")
    legacy_stages.append(
        ("ActiveEndInterval >= 700", None if followup is None else numeric(followup).ge(700), table.concepts.get("active_end_days") or "absent")
    )
    if table.logical_name == "MBSCohort":
        cutoff = derived["_index_date"].le(pd.Timestamp("2023-05-01"))
        legacy_stages.append(("index date <= 2023-05-01", cutoff, table.concepts.get("procedure_date") or "absent"))
    else:
        cutoff = derived["_glp1_start"].le(pd.Timestamp("2023-05-01"))
        legacy_stages.append(("index date <= 2023-05-01", cutoff, table.concepts.get("glp1_start") or "absent"))
        mbs_during = concept_series(table, "mbs_during_glp1")
        legacy_stages.append(
            ("MBSduringGLP1 = 0", None if mbs_during is None else numeric(mbs_during).eq(0), table.concepts.get("mbs_during_glp1") or "absent")
        )
    for criterion, mask, source in legacy_stages:
        before = legacy.copy()
        applied = mask is not None
        if applied:
            legacy &= mask.fillna(False)
        rows.append(
            {
                "table": table.logical_name,
                "section": "legacy_future_or_exposure_restrictions",
                "criterion": criterion,
                "source": source,
                "applied": applied,
                "n_rows_before": int(before.sum()),
                "n_rows_dropped": int((before & ~legacy).sum()),
                "n_rows_remaining": int(legacy.sum()),
                "n_patients_remaining": int(derived.loc[legacy, "_patient"].nunique()),
                "note": "diagnostic only; follow-up and future treatment restrictions should not automatically define baseline eligibility",
            }
        )
    masks["legacy_strict"] = legacy
    return masks, pd.DataFrame(rows)


def count_group(design: str, population: str, arm: str, table: str, frame: pd.DataFrame) -> dict[str, Any]:
    index_date = frame["_index_date"].dropna() if "_index_date" in frame else pd.Series(dtype="datetime64[ns]")
    return {
        "design": design,
        "population": population,
        "arm": arm,
        "table": table,
        "n_rows": len(frame),
        "n_patients": frame["_patient"].nunique() if "_patient" in frame else len(frame),
        "index_date_min": "" if index_date.empty else index_date.min().date().isoformat(),
        "index_date_max": "" if index_date.empty else index_date.max().date().isoformat(),
    }


def append_group(
    groups: list[AnalysisGroup],
    rows: list[dict[str, Any]],
    design: str,
    population: str,
    arm: str,
    table: str,
    frame: pd.DataFrame,
) -> None:
    groups.append(AnalysisGroup(design=design, population=population, arm=arm, table=table, frame=frame))
    rows.append(count_group(design, population, arm, table, frame))


def classify_postop_timing(frame: pd.DataFrame) -> pd.Series:
    months = frame["_glp1_interval_days"] / DAYS_PER_MONTH
    postop = frame["_postop_glp1"]
    observed_use = postop.eq(1) | months.notna()
    result = pd.Series("timing_unavailable", index=frame.index, dtype="string")
    result.loc[postop.eq(0) & months.isna()] = "no_recorded_postop_initiation"
    result.loc[observed_use & months.lt(0)] = "recorded_before_surgery"
    result.loc[observed_use & months.ge(0) & months.lt(12)] = "initiate_before_12m_landmark"
    result.loc[observed_use & months.between(12, 18, inclusive="both")] = "initiate_12_to_18m"
    result.loc[observed_use & months.gt(18) & months.le(24)] = "initiate_after_18_through_24m"
    result.loc[observed_use & months.gt(24)] = "no_initiation_through_24m_then_later"
    return result


def classify_perioperative_strategy(frame: pd.DataFrame) -> pd.Series:
    months = frame["_glp1_interval_days"] / DAYS_PER_MONTH
    result = pd.Series("timing_unavailable", index=frame.index, dtype="string")
    result.loc[frame["_postop_glp1"].eq(0) & months.isna()] = "no_recorded_postop_restart"
    result.loc[months.lt(0)] = "recorded_preop_episode"
    result.loc[months.between(0, 3, inclusive="both")] = "restart_within_3m"
    result.loc[months.gt(3) & months.le(6)] = "restart_after_3_through_6m"
    result.loc[months.gt(6)] = "restart_after_6m"
    return result


def rescue_trigger_masks(table: LoadedTable, derived: pd.DataFrame) -> dict[str, pd.Series]:
    bmi12_col = first_column(table.frame, ("BMI12mPostEvent",))
    weight12_col = first_column(table.frame, ("Weight12mPostEvent",))
    hba1c12_col = first_column(table.frame, ("HbA1c12mPostEvent",))
    baseline_weight_col = table.concepts.get("baseline_weight")
    unavailable = pd.Series(False, index=derived.index)

    bmi12 = pd.Series(np.nan, index=derived.index) if bmi12_col is None else numeric(table.frame[bmi12_col])
    weight12 = pd.Series(np.nan, index=derived.index) if weight12_col is None else numeric(table.frame[weight12_col])
    baseline_weight = pd.Series(np.nan, index=derived.index) if baseline_weight_col is None else numeric(table.frame[baseline_weight_col])
    hba1c12 = pd.Series(np.nan, index=derived.index) if hba1c12_col is None else numeric(table.frame[hba1c12_col])
    twl = 100.0 * (baseline_weight - weight12) / baseline_weight.where(baseline_weight.gt(0))
    return {
        "BMI12 >= 35": bmi12.ge(35).fillna(False),
        "12m total weight loss < 15%": twl.lt(15).fillna(False),
        "HbA1c12 >= 6.5": hba1c12.ge(6.5).fillna(False),
        "any metabolic rescue trigger": (bmi12.ge(35) | twl.lt(15) | hba1c12.ge(6.5)).fillna(False),
        "12m trigger data available": (bmi12.notna() | twl.notna() | hba1c12.notna()).fillna(False),
        "unavailable": unavailable,
    }


def build_design_groups(
    tables: Mapping[str, LoadedTable],
    derived: Mapping[str, pd.DataFrame],
    masks: Mapping[str, Mapping[str, pd.Series]],
) -> tuple[pd.DataFrame, list[AnalysisGroup], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    groups: list[AnalysisGroup] = []
    timing_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []

    if "MBSCohort" in tables:
        mbs = derived["MBSCohort"]
        base = masks["MBSCohort"]["t2d_structural"]
        expanded = mbs.loc[base]
        for keys, group in expanded.groupby(["_procedure", "_prior_glp1"], dropna=False):
            prior_label = "missing" if pd.isna(keys[1]) else str(int(keys[1]))
            append_group(
                groups,
                rows,
                "expanded_surgery_prognosis",
                "T2D structural eligibility; prior GLP retained",
                f"{keys[0]}|prior_glp1={prior_label}",
                "MBSCohort",
                group,
            )

        direct_mbs = expanded[expanded["_prior_glp1"].eq(0)]
        for arm, group in direct_mbs.groupby("_procedure", dropna=False):
            append_group(
                groups,
                rows,
                "baseline_treatment_comparison",
                "T2D structural eligibility; baseline treatment candidates",
                str(arm),
                "MBSCohort",
                group,
            )

        postop_base = expanded[expanded["_prior_glp1"].eq(0)].copy()
        postop_base["_postop_timing"] = classify_postop_timing(postop_base)
        triggers = rescue_trigger_masks(tables["MBSCohort"], mbs)
        for trigger_name, trigger_mask in triggers.items():
            if trigger_name in {"unavailable", "12m trigger data available"}:
                continue
            eligible = postop_base.index.intersection(trigger_mask[trigger_mask].index)
            trigger_frame = postop_base.loc[eligible]
            trigger_rows.append(
                {
                    "trigger": trigger_name,
                    "n_rows": len(trigger_frame),
                    "n_patients": trigger_frame["_patient"].nunique(),
                    "n_with_12m_followup_opportunity": int(trigger_frame["_active_end_days"].ge(12 * DAYS_PER_MONTH).sum()),
                    "n_with_24m_followup_opportunity": int(trigger_frame["_active_end_days"].ge(24 * DAYS_PER_MONTH).sum()),
                    "n_with_36m_followup_opportunity": int(trigger_frame["_active_end_days"].ge(36 * DAYS_PER_MONTH).sum()),
                }
            )
            if trigger_name == "BMI12 >= 35":
                for timing, group in trigger_frame.groupby("_postop_timing", dropna=False):
                    append_group(
                        groups,
                        rows,
                        "postoperative_incretin_rescue",
                        "GLP-naive at surgery; BMI at 12m >= 35",
                        str(timing),
                        "MBSCohort",
                        group,
                    )

        for timing, group in postop_base.groupby("_postop_timing", dropna=False):
            timing_rows.append(
                {
                    "analysis": "postoperative_timing_all_GLP1_naive_surgery",
                    "timing": timing,
                    "n_rows": len(group),
                    "n_patients": group["_patient"].nunique(),
                    "median_start_month": (group["_glp1_interval_days"] / DAYS_PER_MONTH).median(),
                    "p25_start_month": (group["_glp1_interval_days"] / DAYS_PER_MONTH).quantile(0.25),
                    "p75_start_month": (group["_glp1_interval_days"] / DAYS_PER_MONTH).quantile(0.75),
                }
            )

        prior_users = expanded[expanded["_prior_glp1"].eq(1)].copy()
        append_group(
            groups,
            rows,
            "perioperative_prior_user_strategy",
            "T2D structural eligibility; PriorGLP1 = 1",
            "all_prior_users_before_strategy_split",
            "MBSCohort",
            prior_users,
        )
        prior_users["_periop_strategy"] = classify_perioperative_strategy(prior_users)
        for strategy, group in prior_users.groupby("_periop_strategy", dropna=False):
            append_group(
                groups,
                rows,
                "perioperative_prior_user_strategy",
                "T2D structural eligibility; PriorGLP1 = 1",
                str(strategy),
                "MBSCohort",
                group,
            )

        sequence = expanded.copy()
        sequence["_sequence"] = (
            "prior="
            + sequence["_prior_glp1"].map(lambda v: "missing" if pd.isna(v) else str(int(v)))
            + "|postop="
            + sequence["_postop_glp1"].map(lambda v: "missing" if pd.isna(v) else str(int(v)))
        )
        for arm, group in sequence.groupby("_sequence", dropna=False):
            append_group(
                groups,
                rows,
                "observed_treatment_sequences",
                "T2D structural eligibility",
                str(arm),
                "MBSCohort",
                group,
            )

    if "GLP1Cohort" in tables:
        glp = derived["GLP1Cohort"]
        base = masks["GLP1Cohort"]["t2d_structural"]
        direct_glp = glp.loc[base & glp["_prior_glp1"].eq(0)]
        for arm, group in direct_glp.groupby("_agent", dropna=False):
            append_group(
                groups,
                rows,
                "baseline_treatment_comparison",
                "T2D structural eligibility; baseline treatment candidates",
                str(arm),
                "GLP1Cohort",
                group,
            )

        mbs_during_col = tables["GLP1Cohort"].concepts.get("mbs_during_glp1")
        if mbs_during_col:
            glp_sequence = glp.loc[base].copy()
            glp_sequence["_sequence"] = numeric(tables["GLP1Cohort"].frame.loc[glp_sequence.index, mbs_during_col]).map(
                lambda v: "MBS_during_incretin=missing" if pd.isna(v) else f"MBS_during_incretin={int(v)}"
            )
            for arm, group in glp_sequence.groupby("_sequence", dropna=False):
                append_group(
                    groups,
                    rows,
                    "observed_treatment_sequences",
                    "T2D structural eligibility",
                    str(arm),
                    "GLP1Cohort",
                    group,
                )

    return pd.DataFrame(rows), groups, pd.DataFrame(timing_rows), pd.DataFrame(trigger_rows)


def outcome_support(groups: Sequence[AnalysisGroup], tables: Mapping[str, LoadedTable]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in groups:
        table = tables[group.table]
        for outcome, horizon_month, aliases in WIDE_OUTCOMES:
            column = first_column(table.frame, aliases)
            if column is None:
                observed = pd.Series(False, index=group.frame.index)
            else:
                observed = numeric(table.frame.loc[group.frame.index, column]).notna()
            opportunity = group.frame["_active_end_days"].ge(horizon_month * DAYS_PER_MONTH)
            eligible_opportunity = opportunity | observed
            rows.append(
                {
                    "design": group.design,
                    "population": group.population,
                    "arm": group.arm,
                    "table": group.table,
                    "outcome": outcome,
                    "horizon_month": horizon_month,
                    "source_column": column or "",
                    "n_patients": group.frame["_patient"].nunique(),
                    "n_rows": len(group.frame),
                    "n_with_followup_opportunity_or_measurement": int(eligible_opportunity.sum()),
                    "n_observed": int(observed.sum()),
                    "observed_pct_all_rows": 100.0 * observed.mean() if len(observed) else np.nan,
                    "observed_pct_with_opportunity": (
                        100.0 * observed[eligible_opportunity].mean() if eligible_opportunity.any() else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def event_support(groups: Sequence[AnalysisGroup], tables: Mapping[str, LoadedTable]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in groups:
        table = tables[group.table]
        event_pairs = []
        for event_concept, interval_concept in (
            ("mace", "mace_interval_days"),
            ("nephropathy", "nephropathy_interval_days"),
            ("retinopathy", "retinopathy_interval_days"),
        ):
            event_col = table.concepts.get(event_concept)
            interval_col = table.concepts.get(interval_concept)
            event = pd.Series(0.0, index=group.frame.index) if event_col is None else numeric(table.frame.loc[group.frame.index, event_col]).fillna(0)
            interval = pd.Series(np.nan, index=group.frame.index) if interval_col is None else numeric(table.frame.loc[group.frame.index, interval_col])
            event_pairs.append((event, interval))
        for horizon_month in (12, 24, 36, 60):
            horizon_days = horizon_month * DAYS_PER_MONTH
            event_by_horizon = pd.Series(False, index=group.frame.index)
            for event, interval in event_pairs:
                event_by_horizon |= event.eq(1) & interval.ge(0) & interval.le(horizon_days)
            known = event_by_horizon | group.frame["_active_end_days"].ge(horizon_days)
            rows.append(
                {
                    "design": group.design,
                    "population": group.population,
                    "arm": group.arm,
                    "table": group.table,
                    "horizon_month": horizon_month,
                    "n_patients": group.frame["_patient"].nunique(),
                    "n_rows": len(group.frame),
                    "n_composite_events_by_horizon": int(event_by_horizon.sum()),
                    "n_known_event_status": int(known.sum()),
                    "known_status_pct": 100.0 * known.mean() if len(known) else np.nan,
                    "note": "death and recurrent/progression definitions require protocol review; this is a raw support count",
                }
            )
    return pd.DataFrame(rows)


def calendar_counts(groups: Sequence[AnalysisGroup]) -> pd.DataFrame:
    rows = []
    for group in groups:
        years = group.frame["_index_date"].dt.year
        for year, year_group in group.frame.groupby(years, dropna=False):
            rows.append(
                {
                    "design": group.design,
                    "population": group.population,
                    "arm": group.arm,
                    "table": group.table,
                    "index_year": "missing" if pd.isna(year) else int(year),
                    "n_rows": len(year_group),
                    "n_patients": year_group["_patient"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def patient_arm_overlap(groups: Sequence[AnalysisGroup]) -> pd.DataFrame:
    """Aggregate duplicate-patient and cross-arm overlap within each design."""
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[AnalysisGroup]] = {}
    for group in groups:
        grouped.setdefault((group.design, group.population), []).append(group)

    for (design, population), design_groups in grouped.items():
        arm_patients: dict[str, set[str]] = {}
        for group in design_groups:
            label = f"{group.table}:{group.arm}"
            arm_patients.setdefault(label, set()).update(group.frame["_patient"].astype(str).unique())
        multiplicity: dict[str, int] = {}
        for patients in arm_patients.values():
            for patient in patients:
                multiplicity[patient] = multiplicity.get(patient, 0) + 1
        rows.append(
            {
                "design": design,
                "population": population,
                "record_type": "summary",
                "arm_a": "all",
                "arm_b": "all",
                "n_arms": len(arm_patients),
                "n_unique_patients": len(multiplicity),
                "n_patients_in_multiple_arms": sum(value > 1 for value in multiplicity.values()),
                "max_arms_per_patient": max(multiplicity.values(), default=0),
                "n_patients_arm_a": np.nan,
                "n_patients_arm_b": np.nan,
                "n_patients_in_both": np.nan,
                "jaccard_overlap": np.nan,
                "note": "cross-arm overlap must be resolved when one patient contributes multiple index episodes or tables",
            }
        )
        labels = sorted(arm_patients)
        for index, left in enumerate(labels):
            for right in labels[index + 1 :]:
                intersection = arm_patients[left].intersection(arm_patients[right])
                union = arm_patients[left].union(arm_patients[right])
                rows.append(
                    {
                        "design": design,
                        "population": population,
                        "record_type": "pairwise",
                        "arm_a": left,
                        "arm_b": right,
                        "n_arms": len(arm_patients),
                        "n_unique_patients": len(union),
                        "n_patients_in_multiple_arms": len(intersection),
                        "max_arms_per_patient": np.nan,
                        "n_patients_arm_a": len(arm_patients[left]),
                        "n_patients_arm_b": len(arm_patients[right]),
                        "n_patients_in_both": len(intersection),
                        "jaccard_overlap": len(intersection) / len(union) if union else np.nan,
                        "note": "pairwise patient overlap",
                    }
                )
    return pd.DataFrame(rows)


def baseline_summary(groups: Sequence[AnalysisGroup], tables: Mapping[str, LoadedTable]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    direct_groups = [group for group in groups if group.design == "baseline_treatment_comparison"]
    concepts = ("age", "baseline_bmi", "baseline_weight", "baseline_hba1c", "baseline_egfr", "baseline_creatinine")
    for group in direct_groups:
        table = tables[group.table]
        for concept in concepts:
            column = table.concepts.get(concept)
            values = pd.Series(np.nan, index=group.frame.index) if column is None else numeric(table.frame.loc[group.frame.index, column])
            valid = values.dropna()
            rows.append(
                {
                    "arm": group.arm,
                    "table": group.table,
                    "concept": concept,
                    "source_column": column or "",
                    "n_rows": len(group.frame),
                    "n_nonmissing": len(valid),
                    "nonmissing_pct": 100.0 * len(valid) / len(group.frame) if len(group.frame) else np.nan,
                    "mean": valid.mean() if len(valid) else np.nan,
                    "sd": valid.std() if len(valid) > 1 else np.nan,
                    "p05": valid.quantile(0.05) if len(valid) else np.nan,
                    "p25": valid.quantile(0.25) if len(valid) else np.nan,
                    "median": valid.median() if len(valid) else np.nan,
                    "p75": valid.quantile(0.75) if len(valid) else np.nan,
                    "p95": valid.quantile(0.95) if len(valid) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def requirement_rows() -> list[dict[str, Any]]:
    return [
        {"design": "all", "requirement": "center or contributing-organization identifier", "tables": "MBSCohort|GLP1Cohort", "patterns": r"center|site(?:id|key)|organization(?:id|key)|healthsystem|facility(?:id|key)", "importance": "recommended", "why": "center-aware validation, transportability, and clustered inference"},
        {"design": "all", "requirement": "administrative study end and exact last-observed or disenrollment date", "tables": "MBSCohort|GLP1Cohort", "patterns": r"administrativeend|studyend|lastactive|lastobserved|disenroll|censor(?:date|time)", "importance": "recommended", "why": "separate follow-up opportunity from outcome observation"},
        {"design": "expanded_surgery_prognosis", "requirement": "prior incretin exposure flag", "tables": "MBSCohort", "concept": "prior_glp1", "importance": "blocker", "why": "separate never, prior, and active users"},
        {"design": "expanded_surgery_prognosis", "requirement": "postoperative incretin exposure flag", "tables": "MBSCohort", "concept": "postop_glp1", "importance": "blocker", "why": "usual-care landmark prediction"},
        {"design": "expanded_surgery_prognosis", "requirement": "exact incretin start date or surgery-relative interval among recorded postoperative users", "tables": "MBSCohort", "special": "postop_timing_coverage", "importance": "blocker", "why": "avoid using future treatment at an earlier prediction origin"},
        {"design": "expanded_surgery_prognosis", "requirement": "exact selected BMI and HbA1c measurement dates or day offsets", "tables": "MBSCohort", "patterns": r"(?:bmi|hba1c).*(?:measurement)?(?:date|day|interval)", "importance": "recommended", "why": "audit horizon windows and informative measurement timing"},
        {"design": "baseline_treatment_comparison", "requirement": "both treatment tables", "tables": "MBSCohort|GLP1Cohort", "special": "both_tables", "importance": "blocker", "why": "construct aligned surgery and medication arms"},
        {"design": "baseline_treatment_comparison", "requirement": "ingredient or product tag", "tables": "GLP1Cohort", "concept": "glp1_name", "importance": "blocker", "why": "separate treatment versions"},
        {"design": "baseline_treatment_comparison", "requirement": "index dates for surgery and medication", "tables": "MBSCohort|GLP1Cohort", "special": "index_dates", "importance": "blocker", "why": "align time zero and calendar era"},
        {"design": "baseline_treatment_comparison", "requirement": "medication new-user washout history", "tables": "GLP1Cohort", "special": "new_user_washout", "importance": "blocker", "why": "distinguish true initiators from prevalent users"},
        {"design": "baseline_treatment_comparison", "requirement": "switching to bariatric surgery after medication initiation, including exact date", "tables": "GLP1Cohort", "special": "surgery_switch_timing", "importance": "blocker", "why": "treatment-policy and per-protocol estimands"},
        {"design": "baseline_treatment_comparison", "requirement": "dose and unit", "tables": "GLP1Cohort", "concept_all": "recent_glp1_dose|glp1_dose_unit", "importance": "recommended", "why": "differentiate treatment intensity and detect incompatible units"},
        {"design": "baseline_treatment_comparison", "requirement": "prescription, fill, or administration episodes", "tables": "GLP1Cohort", "patterns": r"order|fill|admin|dispens|dayssupply|refill", "importance": "blocker", "why": "continuous-use and per-protocol definitions"},
        {"design": "baseline_treatment_comparison", "requirement": "exact censoring and outcome measurement dates", "tables": "MBSCohort|GLP1Cohort", "patterns": r"lastactive|lastobserved|disenroll|censor|(?:bmi|hba1c).*(?:date|day|interval)", "importance": "blocker", "why": "aligned follow-up opportunity and observation-process adjustment"},
        {"design": "postoperative_incretin_rescue", "requirement": "monthly incretin exposure months 1-36", "tables": "MBSCohort", "patterns": r"(?:glp1|incretin).*month(?:0?[1-9]|[12][0-9]|3[0-6])", "expected_matches": 36, "importance": "blocker", "why": "strategy adherence and artificial censoring"},
        {"design": "postoperative_incretin_rescue", "requirement": "monthly or exact-dated BMI history", "tables": "MBSCohort", "patterns": r"bmi.*(?:month|date|day|interval)", "importance": "blocker", "why": "time-varying treatment-confounder feedback"},
        {"design": "postoperative_incretin_rescue", "requirement": "monthly or exact-dated HbA1c history", "tables": "MBSCohort", "patterns": r"hba1c.*(?:month|date|day|interval)", "importance": "blocker", "why": "time-varying treatment-confounder feedback"},
        {"design": "postoperative_incretin_rescue", "requirement": "time-varying medication history", "tables": "MBSCohort", "patterns": r"medication.*(?:month|date|day|interval)|insulin.*(?:month|date|day|interval)", "importance": "blocker", "why": "confounding and diabetes treatment changes"},
        {"design": "postoperative_incretin_rescue", "requirement": "time-varying utilization history", "tables": "MBSCohort", "patterns": r"utilization|encountercount|visitcount", "importance": "blocker", "why": "confounding and informative observation"},
        {"design": "postoperative_incretin_rescue", "requirement": "monthly observability or exact censoring", "tables": "MBSCohort", "patterns": r"observablemonth|activemonth|disenroll|lastactive|censor", "importance": "blocker", "why": "natural censoring weights"},
        {"design": "perioperative_prior_user_strategy", "requirement": "preoperative medication episode start and end", "tables": "MBSCohort", "concept_all": "glp1_start|glp1_end", "importance": "blocker", "why": "identify active treatment at surgery"},
        {"design": "perioperative_prior_user_strategy", "requirement": "postoperative restart date", "tables": "MBSCohort", "concept_any": "glp1_start|glp1_interval_days", "importance": "blocker", "why": "define continue or early-restart versus delay"},
        {"design": "perioperative_prior_user_strategy", "requirement": "multiple medication episodes or longitudinal orders", "tables": "MBSCohort", "patterns": r"episode|order|fill|admin|dispens|restart", "importance": "blocker", "why": "distinguish preoperative episode from postoperative episode"},
        {"design": "conditional_flow_challenger", "requirement": "joint BMI and HbA1c outcomes at multiple horizons", "tables": "MBSCohort|GLP1Cohort", "special": "wide_joint_outcomes", "importance": "blocker", "why": "fit and evaluate joint per-patient trajectory distributions"},
        {"design": "conditional_flow_challenger", "requirement": "valid causal design and treatment adjustment", "tables": "MBSCohort|GLP1Cohort", "special": "manual_causal_design", "importance": "blocker", "why": "treatment clamping in a predictive flow is not causal identification"},
    ]


def coverage_for_column(table: LoadedTable, column: str) -> float:
    return 100.0 * (~missing_mask(table.frame[column])).mean() if len(table.frame) else np.nan


def requirement_audit(tables: Mapping[str, LoadedTable]) -> pd.DataFrame:
    rows = []
    for spec in requirement_rows():
        requested_tables = spec["tables"].split("|")
        available_tables = [tables[name] for name in requested_tables if name in tables]
        matched: list[str] = []
        coverages: list[float] = []
        status = "absent"
        detail = ""

        if spec.get("special") == "postop_timing_coverage":
            table = tables.get("MBSCohort")
            if table is not None:
                postop_col = table.concepts.get("postop_glp1")
                timing_cols = [table.concepts.get("glp1_start"), table.concepts.get("glp1_interval_days")]
                timing_cols = [column for column in timing_cols if column]
                if postop_col and timing_cols:
                    treated = numeric(table.frame[postop_col]).eq(1)
                    timing_present = pd.Series(False, index=table.frame.index)
                    for column in timing_cols:
                        assert column is not None
                        timing_present |= ~missing_mask(table.frame[column])
                        matched.append(f"MBSCohort.{column}")
                    if treated.any():
                        conditional_coverage = 100.0 * timing_present[treated].mean()
                        coverages.append(conditional_coverage)
                        status = "present_good" if conditional_coverage >= 90 else (
                            "present_partial" if conditional_coverage >= 50 else "present_low_coverage"
                        )
                        detail = f"timing present for {conditional_coverage:.2f}% of PostOpGLP1 = 1 rows"
                    else:
                        status = "manual_review"
                        detail = "timing columns exist, but no PostOpGLP1 = 1 rows were found"
        elif spec.get("special") == "new_user_washout":
            table = tables.get("GLP1Cohort")
            if table is not None:
                prior_col = table.concepts.get("prior_glp1")
                lookback_columns = [
                    column
                    for column in table.frame.columns
                    if re.search(r"washout|lookback|prior.*(?:order|fill|admin|episode|date)", str(column), re.IGNORECASE)
                ]
                if prior_col:
                    matched.append(f"GLP1Cohort.{prior_col}")
                    coverages.append(coverage_for_column(table, prior_col))
                matched.extend(f"GLP1Cohort.{column}" for column in lookback_columns)
                if prior_col and lookback_columns:
                    status = "present_good"
                elif prior_col:
                    status = "present_partial"
                    detail = "PriorGLP1 is a proxy; no dated washout or lookback field was detected"
        elif spec.get("special") == "surgery_switch_timing":
            table = tables.get("GLP1Cohort")
            if table is not None:
                switch_col = table.concepts.get("mbs_during_glp1")
                date_columns = [
                    column
                    for column in table.frame.columns
                    if re.search(r"(?:mbs|bariatric|surgery).*(?:date|day|interval)", str(column), re.IGNORECASE)
                ]
                if switch_col:
                    matched.append(f"GLP1Cohort.{switch_col}")
                    coverages.append(coverage_for_column(table, switch_col))
                matched.extend(f"GLP1Cohort.{column}" for column in date_columns)
                if switch_col and date_columns:
                    status = "present_good"
                elif switch_col:
                    status = "present_partial"
                    detail = "MBSduringGLP1 is present, but no exact switch date or interval was detected"
        elif spec.get("special") == "both_tables":
            status = "present_good" if len(available_tables) == len(requested_tables) else "absent"
            detail = f"found {len(available_tables)} of {len(requested_tables)} tables"
        elif spec.get("special") == "index_dates":
            required = {"MBSCohort": "procedure_date", "GLP1Cohort": "glp1_start"}
            checks = []
            for name, concept in required.items():
                if name in tables and tables[name].concepts.get(concept):
                    column = tables[name].concepts[concept]
                    assert column is not None
                    matched.append(f"{name}.{column}")
                    checks.append(coverage_for_column(tables[name], column))
            coverages = checks
            status = "present_good" if len(checks) == 2 and min(checks) >= 90 else ("present_partial" if checks else "absent")
        elif spec.get("special") == "wide_joint_outcomes":
            found_families = set()
            for table in available_tables:
                for outcome, _, aliases in WIDE_OUTCOMES:
                    column = first_column(table.frame, aliases)
                    if column:
                        found_families.add((table.logical_name, outcome))
                        matched.append(f"{table.logical_name}.{column}")
            status = "present_good" if any(item[1] == "bmi" for item in found_families) and any(item[1] == "hba1c" for item in found_families) else "absent"
            detail = f"{len(matched)} wide outcome columns found"
        elif spec.get("special") == "manual_causal_design":
            status = "manual_review"
            detail = "cannot be established from schema alone"
        elif "concept" in spec or "concept_any" in spec or "concept_all" in spec:
            concepts = (spec.get("concept") or spec.get("concept_any") or spec.get("concept_all")).split("|")
            per_concept = []
            for concept in concepts:
                concept_matches = []
                for table in available_tables:
                    column = table.concepts.get(concept)
                    if column:
                        concept_matches.append((table, column))
                per_concept.append(bool(concept_matches))
                for table, column in concept_matches:
                    matched.append(f"{table.logical_name}.{column}")
                    coverages.append(coverage_for_column(table, column))
            if "concept_all" in spec:
                structurally_present = all(per_concept)
            else:
                structurally_present = any(per_concept)
            if structurally_present:
                best = max(coverages) if coverages else np.nan
                status = "present_good" if np.isfinite(best) and best >= 90 else ("present_partial" if np.isfinite(best) and best >= 50 else "present_low_coverage")
        elif "patterns" in spec:
            pattern = re.compile(spec["patterns"], re.IGNORECASE)
            for table in available_tables:
                for column in table.frame.columns:
                    if pattern.search(str(column)):
                        matched.append(f"{table.logical_name}.{column}")
                        coverages.append(coverage_for_column(table, column))
            expected = int(spec.get("expected_matches", 1))
            if len(matched) >= expected:
                status = "present_good"
            elif matched:
                status = "present_partial"
                detail = f"found {len(matched)} of at least {expected} expected columns"

        rows.append(
            {
                "design": spec["design"],
                "requirement": spec["requirement"],
                "importance": spec["importance"],
                "status": status,
                "matched_columns": "|".join(matched),
                "n_matching_columns": len(matched),
                "best_nonmissing_pct": max(coverages) if coverages else np.nan,
                "detail": detail,
                "why": spec["why"],
            }
        )
    return pd.DataFrame(rows)


def design_readiness(requirements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for design, group in requirements[requirements["design"].ne("all")].groupby("design"):
        blockers = group[group["importance"].eq("blocker")]
        blocked = blockers[blockers["status"].isin(["absent", "present_low_coverage"])]
        partial = blockers[blockers["status"].isin(["present_partial", "manual_review"])]
        if len(blocked):
            status = "blocked_by_current_extract"
        elif len(partial):
            status = "candidate_after_manual_review"
        else:
            status = "minimum_fields_present"
        rows.append(
            {
                "design": design,
                "schema_readiness": status,
                "n_blocking_requirements": len(blockers),
                "n_blocked": len(blocked),
                "n_partial_or_manual": len(partial),
                "blocked_requirements": "|".join(blocked["requirement"].astype(str)),
                "manual_review_requirements": "|".join(partial["requirement"].astype(str)),
                "interpretation": "schema gate only; arm counts, overlap, follow-up, and unmeasured confounding still determine estimability",
            }
        )
    return pd.DataFrame(rows)


def table_summary(tables: Mapping[str, LoadedTable], derived: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, table in tables.items():
        frame = derived[name]
        index_dates = frame["_index_date"].dropna()
        rows.append(
            {
                "table": name,
                "source_name": table.source_name,
                "n_rows": len(frame),
                "n_patients": frame["_patient"].nunique(),
                "n_duplicate_patient_rows": int(frame["_patient"].duplicated().sum()),
                "n_columns": len(table.frame.columns),
                "index_date_min": "" if index_dates.empty else index_dates.min().date().isoformat(),
                "index_date_max": "" if index_dates.empty else index_dates.max().date().isoformat(),
                "n_agent_tags": frame.loc[frame["_agent_raw"].ne("<missing>"), "_agent_raw"].nunique(),
                "n_unmapped_agent_rows": int(frame["_agent"].eq("unmapped").sum()),
                "patient_id_present": bool(table.concepts.get("patient_id")),
                "center_id_present": bool(table.concepts.get("center_id")),
            }
        )
    return pd.DataFrame(rows)


def privacy_safe(frame: pd.DataFrame, min_cell_size: int) -> pd.DataFrame:
    """Suppress grouped small-cell counts while retaining overall structure."""
    if frame.empty or min_cell_size <= 0:
        return frame
    out = frame.copy()
    nonsensitive_structural_counts = {
        "n_arms",
        "n_agent_tags",
        "n_blocked",
        "n_blocking_requirements",
        "n_columns",
        "n_matching_columns",
        "n_partial_or_manual",
    }
    count_columns = [
        column
        for column in out.columns
        if column.startswith("n_")
        and pd.api.types.is_numeric_dtype(out[column])
        and column not in nonsensitive_structural_counts
        and not column.startswith("n_distinct")
    ]
    if not count_columns:
        return out
    denominator = next(
        (
            column
            for column in (
                "n_patients",
                "n_unique_patients",
                "n_patients_remaining",
                "n_rows",
                "n_rows_remaining",
                "n_nonmissing",
                "n_dose_nonmissing",
                "n_known_event_status",
                "n_observed",
            )
            if column in out
        ),
        None,
    )
    if denominator is None:
        return out
    original_counts = {column: pd.to_numeric(out[column], errors="coerce") for column in count_columns}
    denominator_values = original_counts[denominator]
    row_small = denominator_values.lt(min_cell_size)
    any_positive_small_count = pd.Series(False, index=out.index)
    small_by_column: dict[str, pd.Series] = {}
    for values in original_counts.values():
        any_positive_small_count |= values.gt(0) & values.lt(min_cell_size)
    out["small_cell_suppressed"] = row_small | any_positive_small_count
    out["cell_size_display"] = denominator_values.map(
        lambda value: "" if pd.isna(value) else (f"<{min_cell_size}" if value < min_cell_size else str(int(value)))
    )
    metric_columns = [
        column
        for column in out.columns
        if pd.api.types.is_numeric_dtype(out[column])
        and not pd.api.types.is_bool_dtype(out[column])
        and column not in count_columns
        and column not in {"rank", "horizon_month", "index_year", "ordinal_position"}
    ]
    for column in count_columns + metric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    for column, values in original_counts.items():
        positive_small = values.gt(0) & values.lt(min_cell_size)
        small_by_column[column] = positive_small
        out[f"{column}_display"] = values.map(
            lambda value: "" if pd.isna(value) else (f"<{min_cell_size}" if 0 < value < min_cell_size else str(int(value)))
        )
        out.loc[row_small, f"{column}_display"] = f"<{min_cell_size}"
        out.loc[positive_small, column] = np.nan
    out.loc[row_small, count_columns] = np.nan
    out.loc[row_small, metric_columns] = np.nan

    related_metric_patterns = {
        "n_dose_nonmissing": r"dose_nonmissing_pct|^min$|^p\d+$|^median$|^max$",
        "n_nonmissing": r"nonmissing_pct|^mean$|^sd$|^p\d+$|^median$",
        "n_observed": r"observed_pct",
        "n_known_event_status": r"known_status_pct",
        "n_duration_date_pairs": r"duration.*(?:median|p95)|median_duration|p95_abs_duration",
        "n_explicit_and_date_derived_interval_pairs": r"explicit_minus_derived_interval",
        "n_patients_in_both": r"jaccard_overlap",
        "n_events": r"event_(?:pct|rate)|(?:pct|rate)_event",
        "n_missing_or_blank": r"missing_or_blank_pct",
        "n_placeholder_values": r"placeholder_pct",
    }
    for count_column, pattern in related_metric_patterns.items():
        if count_column not in small_by_column:
            continue
        related = [column for column in metric_columns if re.search(pattern, column, re.IGNORECASE)]
        if related:
            out.loc[small_by_column[count_column], related] = np.nan
    date_summary_columns = [
        column
        for column in out.columns
        if re.search(r"(?:^|_)date_(?:min|max)$|^index_date_(?:min|max)$", str(column), re.IGNORECASE)
    ]
    for column in date_summary_columns:
        out.loc[row_small, column] = ""
    return out


def ensure_aggregate_only(name: str, frame: pd.DataFrame) -> None:
    forbidden = {"patkey", "patientkey", "patientid", "patient"}
    present = forbidden.intersection({normalize_identifier(column) for column in frame.columns})
    if present:
        raise RuntimeError(f"Privacy guard rejected {name}: patient-level identifier column(s) {sorted(present)}")


def safe_write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_aggregate_only(path.name, frame)
    frame.to_csv(path, index=False)


def markdownish_rows(frame: pd.DataFrame, columns: Sequence[str], limit: int = 20) -> list[str]:
    if frame.empty:
        return ["  (none)"]
    lines = []
    for _, row in frame.head(limit).iterrows():
        bits = []
        for column in columns:
            value = row.get(column, "")
            rendered = "NA/suppressed" if not isinstance(value, (list, dict, tuple)) and pd.isna(value) else value
            bits.append(f"{column}={rendered}")
        lines.append("  - " + "; ".join(bits))
    if len(frame) > limit:
        lines.append(f"  - ... {len(frame) - limit} additional rows are in the CSV artifact")
    return lines


def with_cell_display(frame: pd.DataFrame, count_column: str) -> pd.DataFrame:
    if frame.empty or "cell_size_display" not in frame or count_column not in frame:
        return frame
    out = frame.copy()
    count_display_column = f"{count_column}_display"
    suppressed = out[count_column].isna() & out.get(
        "small_cell_suppressed", pd.Series(False, index=out.index)
    ).fillna(False)
    out[count_column] = out[count_column].map(
        lambda value: "" if pd.isna(value) else (str(int(value)) if float(value).is_integer() else str(value))
    ).astype("object")
    if count_display_column in out:
        out.loc[suppressed, count_column] = out.loc[suppressed, count_display_column]
    else:
        out.loc[suppressed, count_column] = out.loc[suppressed, "cell_size_display"]
    return out


def build_report(
    source_mode: str,
    sampled: bool,
    summaries: Mapping[str, pd.DataFrame],
    warnings: Sequence[str],
    min_cell_size: int,
) -> str:
    table_frame = summaries["table_summary"]
    tags = summaries["medication_tag_values"]
    dose = summaries["dose_profiles"]
    readiness = summaries["design_readiness"]
    counts = summaries["design_arm_counts"]
    requirements = summaries["design_requirements"]
    database_candidates = summaries["database_medication_schema_candidates"]

    lines = [
        "GLP-1 / incretin Cosmos feasibility diagnostic  (READ-ONLY, AGGREGATE-ONLY)",
        f"script version : {SCRIPT_VERSION}",
        f"generated      : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"source mode    : {source_mode}",
        f"sampled        : {sampled}",
        f"small-cell rule: counts below {min_cell_size} suppressed" if min_cell_size > 0 else "small-cell rule: disabled",
        "=" * 88,
        "",
        "[TABLES]",
    ]
    lines.extend(markdownish_rows(table_frame, ["table", "n_rows", "n_patients", "n_columns", "index_date_min", "index_date_max"]))
    if warnings:
        lines.extend(["", "[WARNINGS]"] + [f"  - {warning}" for warning in warnings])

    lines.extend(["", "[DATABASE MEDICATION SCHEMA DISCOVERY]"])
    if database_candidates.empty:
        lines.append("  (no additional medication-related schema candidates were found or schema discovery was unavailable)")
    else:
        candidate_tables = database_candidates[["TABLE_SCHEMA", "TABLE_NAME", "candidate_columns_in_table"]].drop_duplicates()
        lines.extend(markdownish_rows(candidate_tables, ["TABLE_SCHEMA", "TABLE_NAME", "candidate_columns_in_table"], limit=30))
        lines.append("  Review database_medication_schema_candidates.csv before requesting a second extract from source medication tables.")

    lines.extend(["", "[MEDICATION TAGS] proposed normalization, top raw values"])
    agent_tags = tags[tags["proposed_ingredient"].ne("")] if not tags.empty else tags
    agent_tags = with_cell_display(agent_tags, "n_patients")
    lines.extend(markdownish_rows(agent_tags, ["table", "column", "raw_value", "proposed_ingredient", "n_patients"], limit=30))
    unmapped = agent_tags[agent_tags["needs_manual_review"].eq(True)] if not agent_tags.empty else agent_tags
    lines.append(f"  unmapped raw agent tags requiring manual review: {len(unmapped)}")

    lines.extend(["", "[DOSE COVERAGE] each unit is kept separate"])
    dose_head = dose.sort_values(["table", "agent", "dose_field", "n_dose_nonmissing"], ascending=[True, True, True, False]) if not dose.empty else dose
    dose_head = with_cell_display(dose_head, "n_dose_nonmissing")
    lines.extend(markdownish_rows(dose_head, ["table", "agent", "route", "dose_unit", "dose_field", "n_dose_nonmissing", "dose_nonmissing_pct", "median", "p95"], limit=35))

    lines.extend(["", "[DESIGN SCHEMA READINESS]"])
    lines.extend(markdownish_rows(readiness, ["design", "schema_readiness", "n_blocked", "blocked_requirements", "manual_review_requirements"], limit=20))

    lines.extend(["", "[CANDIDATE ARM COUNTS]"])
    report_counts = with_cell_display(counts, "n_patients")
    lines.extend(markdownish_rows(report_counts, ["design", "population", "arm", "table", "n_patients", "index_date_min", "index_date_max"], limit=50))

    blocked = requirements[
        requirements["status"].isin(["absent", "present_low_coverage", "present_partial", "manual_review"])
    ]
    lines.extend(["", "[FIELDS TO REQUEST OR REVIEW]"])
    lines.extend(
        markdownish_rows(blocked, ["design", "requirement", "importance", "status", "matched_columns", "why"], limit=50)
    )

    lines.extend(
        [
            "",
            "[INTERPRETATION BOUNDARIES]",
            "  - More rows improve precision only if time zero, eligibility, treatment versions, and follow-up are valid.",
            "  - PriorGLP1 and PostOpGLP1 flags do not establish adherence or complete medication capture.",
            "  - No-treatment arms mean no recorded treatment in the contributing data, not proven non-use.",
            "  - A per-patient conditional flow can be evaluated as a prediction model. Causal treatment contrasts still require a target-trial estimator.",
            "  - Review all unmapped tags, dose units, date inconsistencies, arm overlap, and outcome-support tables before changing the production protocol.",
            "  - The small-cell guard is a basic diagnostic safeguard, not a substitute for formal Cosmos disclosure review before files leave the VM.",
            "",
            "CSV artifacts in this directory contain the complete aggregate diagnostics.",
            "=" * 88,
        ]
    )
    return "\n".join(lines) + "\n"


def script_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def run(args: argparse.Namespace) -> Path:
    tables, warnings, source_mode, database_schema = load_tables(args)
    sampled = args.max_rows is not None
    if sampled:
        warnings.append("--max-rows was used; all cohort counts and percentages describe a non-random truncated sample")

    derived = {name: derive_table(table) for name, table in tables.items()}
    masks: dict[str, dict[str, pd.Series]] = {}
    funnel_frames = []
    for name, table in tables.items():
        table_masks, funnel = candidate_masks(table, derived[name])
        masks[name] = table_masks
        funnel_frames.append(funnel)

    profile_frames = [profile_columns(table) for table in tables.values()]
    tag_frames = [medication_tag_values(table, derived[name], args.top_values) for name, table in tables.items()]
    coverage_frames = [coverage_by_agent(table, derived[name]) for name, table in tables.items()]
    dose_profile_frames = []
    dose_frequency_frames = []
    episode_frames = []
    reconciliation_frames = []
    for name, table in tables.items():
        profiles, frequencies = dose_profiles(table, derived[name], args.top_values)
        dose_profile_frames.append(profiles)
        dose_frequency_frames.append(frequencies)
        episode_frames.append(episode_quality(table, derived[name]))
        reconciliation_frames.append(exposure_reconciliation(table, derived[name]))

    design_counts, groups, timing, rescue_triggers = build_design_groups(tables, derived, masks)
    requirements = requirement_audit(tables)
    summaries: dict[str, pd.DataFrame] = {
        "table_summary": table_summary(tables, derived),
        "schema_inventory": schema_inventory(tables),
        "database_medication_schema_candidates": database_medication_schema_candidates(database_schema),
        "concept_inventory": concept_inventory(tables),
        "column_profiles": pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame(),
        "medication_tag_values": pd.concat(tag_frames, ignore_index=True) if tag_frames else pd.DataFrame(),
        "medication_field_coverage_by_agent": pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame(),
        "dose_profiles": pd.concat(dose_profile_frames, ignore_index=True) if dose_profile_frames else pd.DataFrame(),
        "dose_value_frequencies": pd.concat(dose_frequency_frames, ignore_index=True) if dose_frequency_frames else pd.DataFrame(),
        "episode_quality": pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame(),
        "exposure_reconciliation": (
            pd.concat(reconciliation_frames, ignore_index=True) if reconciliation_frames else pd.DataFrame()
        ),
        "cohort_funnels": pd.concat(funnel_frames, ignore_index=True) if funnel_frames else pd.DataFrame(),
        "design_arm_counts": design_counts,
        "postoperative_timing": timing,
        "rescue_trigger_counts": rescue_triggers,
        "outcome_support": outcome_support(groups, tables),
        "event_support": event_support(groups, tables),
        "calendar_counts": calendar_counts(groups),
        "patient_arm_overlap": patient_arm_overlap(groups),
        "baseline_overlap_summary": baseline_summary(groups, tables),
        "design_requirements": requirements,
        "design_readiness": design_readiness(requirements),
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    privacy_artifacts = {
        "table_summary",
        "column_profiles",
        "medication_tag_values",
        "medication_field_coverage_by_agent",
        "dose_profiles",
        "dose_value_frequencies",
        "episode_quality",
        "exposure_reconciliation",
        "cohort_funnels",
        "design_arm_counts",
        "postoperative_timing",
        "rescue_trigger_counts",
        "outcome_support",
        "event_support",
        "calendar_counts",
        "patient_arm_overlap",
        "baseline_overlap_summary",
    }
    output_summaries = {
        name: (privacy_safe(frame, args.min_cell_size) if name in privacy_artifacts else frame)
        for name, frame in summaries.items()
    }
    for name, output in output_summaries.items():
        safe_write_csv(run_dir / f"{name}.csv", output)

    report = build_report(source_mode, sampled, output_summaries, warnings, args.min_cell_size)
    (run_dir / "glp1_feasibility_report.txt").write_text(report, encoding="utf-8")
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha256(),
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_mode": source_mode,
        "sampled": sampled,
        "max_rows_per_table": args.max_rows,
        "min_cell_size": args.min_cell_size,
        "top_values_per_tag_or_dose": args.top_values,
        "tables": {name: table.source_name for name, table in tables.items()},
        "warnings": warnings,
        "artifacts": sorted(path.name for path in run_dir.iterdir() if path.is_file()),
        "privacy": "aggregate outputs only; no patient identifier columns are written",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"[debug_glp1_feasibility] wrote {run_dir} | "
        f"tables={','.join(tables)} | designs={len(summaries['design_readiness'])} | "
        f"unmapped_agent_tags={int(summaries['medication_tag_values'].get('needs_manual_review', pd.Series(dtype=bool)).sum())}",
        flush=True,
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mbs-csv", type=Path, default=None, help="Local MBSCohort CSV. Supplying either CSV enables CSV mode.")
    parser.add_argument("--glp1-csv", type=Path, default=None, help="Local GLP1Cohort CSV. Supplying either CSV enables CSV mode.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help=f"Cosmos SQL schema (default: {DEFAULT_SCHEMA}).")
    parser.add_argument("--mbs-table", default=DEFAULT_MBS_TABLE, help=f"Surgery table name (default: {DEFAULT_MBS_TABLE}).")
    parser.add_argument("--glp1-table", default=DEFAULT_GLP1_TABLE, help=f"Medication table name (default: {DEFAULT_GLP1_TABLE}).")
    parser.add_argument("--connection-timeout", type=int, default=1000, help="ODBC connection timeout in seconds.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional TOP(N) per DB table for plumbing only. Counts become sampled and non-final.")
    parser.add_argument(
        "--top-values",
        type=int,
        default=100,
        help="Maximum values retained for high-cardinality discovered fields and dose frequencies; core name/route/unit tags are complete.",
    )
    parser.add_argument("--min-cell-size", type=int, default=11, help="Suppress grouped output cells below this size. Use 0 only if local policy permits.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if args.top_values <= 0:
        parser.error("--top-values must be positive")
    if args.min_cell_size < 0:
        parser.error("--min-cell-size cannot be negative")
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 - CLI must provide an actionable failure
        print(
            f"ERROR: {exc}\n\n"
            "On the Cosmos VM, run with no CSV arguments. For a local smoke test, pass both "
            "--mbs-csv and --glp1-csv synthetic exports.",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
