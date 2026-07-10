"""Publication-ready "Table 1" (baseline characteristics) for the MBSCohort
modeling cohort, broken out by the train / validation / test split.

This is a *descriptive* companion to the modeling scripts. It answers the
question a reviewer asks first: "who is in the cohort, and are the three data
splits balanced?" It deliberately reuses ``train_flow_matching``'s data layer by
import (loaders, CPT -> surgery mapping, composite-event logic, and the EXACT
``make_stratified_splits`` used by every trainer/evaluator), so the table
describes the same post-filter cohort the models see, split patient-for-patient
the same way. The pristine core (``train_flow_matching.py``) is never modified.

Every characteristic is annotated so it is obvious which rows are model inputs:
    dagger  (U+2020) flow-matching conditioning feature (also used by the GBM risk model)
    ddagger (U+2021) additional feature used only by the gradient-boosted MACE risk model
    section (U+00A7) candidate feature available in the cohort (not currently a model input)
    a/b/c            lettered notes (postoperative incretin exposure; %TWL; composite outcome)

Continuous variables are summarized as median [Q1, Q3] (default) or mean (SD);
categorical/binary variables as n (%). A p-value column tests balance across the
three splits (Kruskal-Wallis / one-way ANOVA for continuous, chi-square for
categorical) as a sanity check on the random split -- it is not adjusted for
multiplicity.

Two run modes mirror the other satellite scripts (DB is the default so a bare
run works on the Cosmos VM):

    python make_table_one.py
        Query Cosmos MBSCohort through the fm pyodbc path.

    python make_table_one.py --csv fake_data/fake_mbs_cohort.csv
        Local mode from a saved CSV export (post-SQL).

Outputs (into a timestamped folder under runs/table_one/): the table as CSV,
a styled standalone HTML page, a rendered PNG, GitHub-flavored Markdown, and a
booktabs LaTeX table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html as html_lib
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

import train_flow_matching as fm

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - p-values degrade to blank without scipy.
    scipy_stats = None


DEFAULT_OUTPUT_DIR = fm.REPO_ROOT / "runs" / "table_one"

# Column order of the table (the modeling splits + the pooled cohort).
SPLIT_COLUMNS = ["Overall", "Train", "Validation", "Test"]

# Feature-role markers (kept ASCII-safe unicode so matplotlib/LaTeX render them).
MARK_FLOW = "†"   # dagger    -> flow-matching conditioning feature
MARK_GBM = "‡"    # ddagger   -> GBM-only risk-model feature
MARK_CAND = "§"   # section   -> candidate feature (not a current model input)

# Categorical string values that mean "not reported" and collapse to Unknown.
MASKED_TOKENS = {
    "",
    "nan",
    "none",
    "null",
    "unknown",
    "#masked",
    "*unspecified",
    "*unknown",
    "*not applicable",
    "not applicable",
}

# Display labels for the two modeled procedures.
SURGERY_DISPLAY = {"sleeve": "Sleeve gastrectomy", "rnygb": "Roux-en-Y gastric bypass"}


@dataclass
class TableConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    split_seed: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    continuous: str = "median"  # "median" -> median [Q1, Q3]; "mean" -> mean (SD)
    # Which model split the table's Train/Validation/Test columns describe. "surgery"
    # (default, unchanged behavior) -> fm.make_stratified_splits; "temporal" ->
    # fm.make_temporal_splits, so the columns match the models' out-of-time folds
    # patient-for-patient. Threaded from freeze_run so Table 1 never misaligns with the
    # temporal model.
    split_strategy: str = "surgery"


def report_saved(path: Path, description: str = "") -> Path:
    tag = f" {description}" if description else ""
    print(f"  [saved]{tag} -> {path}", flush=True)
    return path


# --------------------------------------------------------------------------- #
# Row-aligned accessors into the post-filter cohort frame
# --------------------------------------------------------------------------- #
def frame_numeric(dataset: fm.FlowDataset, canonical: str) -> np.ndarray | None:
    """Numeric column from ``dataset.frame`` by canonical name (tolerant of the
    Cosmos join suffixes ``fm.canonicalize_columns`` handles). Row-aligned with
    ``dataset.x`` / ``surgery_type``. NaNs preserved. ``None`` if absent."""
    matched = fm.find_compatible_column(list(dataset.frame.columns), canonical)
    if matched is None:
        return None
    return fm.numeric(dataset.frame[matched]).to_numpy(dtype=np.float64)


def frame_raw(dataset: fm.FlowDataset, canonical: str) -> pd.Series | None:
    matched = fm.find_compatible_column(list(dataset.frame.columns), canonical)
    if matched is None:
        return None
    return dataset.frame[matched]


def normalize_category(series: pd.Series) -> np.ndarray:
    """Strip strings; collapse masked/blank tokens to None so they land in a
    single Unknown level rather than fragmenting the category counts."""
    out = np.empty(len(series), dtype=object)
    values = series.astype("string")
    for i, value in enumerate(values.tolist()):
        if value is None:
            out[i] = None
            continue
        text = str(value).strip()
        out[i] = None if text.lower() in MASKED_TOKENS else text
    return out


def percent_total_weight_loss(dataset: fm.FlowDataset, followup_column: str) -> np.ndarray | None:
    """%TWL = (baseline weight - follow-up weight) / baseline weight * 100.

    Unit-independent (a ratio of weights), so it is reported regardless of whether
    the recorded weights are lb or kg. Observed values only (NOT GLP-1-censored):
    this is the descriptive clinical quantity, distinct from the GLP-1-masked BMI
    targets the flow model fits. NaN where either weight is missing/non-positive."""
    w0 = frame_numeric(dataset, "WeightAtEvent")
    wt = frame_numeric(dataset, followup_column)
    if w0 is None or wt is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        twl = (w0 - wt) / w0 * 100.0
    twl = np.where((w0 > 0) & np.isfinite(w0) & np.isfinite(wt), twl, np.nan)
    return twl


# --------------------------------------------------------------------------- #
# Statistics (all guarded; return NaN p-value on any degeneracy)
# --------------------------------------------------------------------------- #
def _clean_groups(groups: list[np.ndarray]) -> list[np.ndarray]:
    cleaned = [np.asarray(g, dtype=float) for g in groups]
    cleaned = [g[np.isfinite(g)] for g in cleaned]
    return [g for g in cleaned if g.size > 0]


def continuous_pvalue(groups: list[np.ndarray], mode: str) -> float:
    groups = _clean_groups(groups)
    if scipy_stats is None or len(groups) < 2:
        return float("nan")
    if np.unique(np.concatenate(groups)).size < 2:
        return float("nan")
    try:
        if mode == "mean":
            return float(scipy_stats.f_oneway(*groups).pvalue)
        return float(scipy_stats.kruskal(*groups).pvalue)
    except Exception:
        return float("nan")


def chi2_pvalue(contingency: np.ndarray) -> float:
    """chi-square on a (levels x splits) count table. Drops empty rows/cols and
    returns NaN when the table is too degenerate for a valid test."""
    table = np.asarray(contingency, dtype=float)
    if table.ndim != 2:
        return float("nan")
    table = table[table.sum(axis=1) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    if (table.sum(axis=0) == 0).any():
        return float("nan")
    if scipy_stats is None:
        return float("nan")
    try:
        return float(scipy_stats.chi2_contingency(table).pvalue)
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Cell formatting
# --------------------------------------------------------------------------- #
def fmt_continuous(values: np.ndarray, digits: int, mode: str) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "-"
    if mode == "mean":
        return f"{np.mean(values):.{digits}f} ({np.std(values, ddof=1) if values.size > 1 else 0.0:.{digits}f})"
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return f"{med:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def fmt_count_pct(n: int, denom: int) -> str:
    if denom <= 0:
        return "-"
    return f"{int(n):,} ({100.0 * n / denom:.1f})"


def fmt_pvalue(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# --------------------------------------------------------------------------- #
# Table assembly
# --------------------------------------------------------------------------- #
class TableBuilder:
    """Accumulates tidy rows for the Table 1. Each row carries its section, a
    display label, an indent level, feature-role marker(s), the four split cells,
    and a numeric p-value (NaN when not applicable)."""

    def __init__(self, dataset: fm.FlowDataset, splits: dict[str, np.ndarray], cfg: TableConfig):
        self.dataset = dataset
        self.cfg = cfg
        n = len(dataset.subject_ids)
        self.idx = {
            "Overall": np.arange(n),
            "Train": splits["train"],
            "Validation": splits["val"],
            "Test": splits["test"],
        }
        self.group_idx = [splits["train"], splits["val"], splits["test"]]
        self.rows: list[dict] = []
        self._section = ""

    # -- row emitters ------------------------------------------------------- #
    def section(self, name: str) -> None:
        self._section = name

    def _row(self, label: str, indent: int, row_type: str, cells: dict, pval: float, marker: str) -> None:
        row = {
            "section": self._section,
            "label": label,
            "indent": indent,
            "row_type": row_type,
            "marker": marker,
            "pval": pval,
        }
        row.update(cells)
        self.rows.append(row)

    def count_row(self) -> None:
        total = self.idx["Overall"].size
        cells = {}
        for col, ix in self.idx.items():
            if col == "Overall":
                cells[col] = f"{ix.size:,}"
            else:
                cells[col] = f"{ix.size:,} ({100.0 * ix.size / total:.1f})"
        self._row("No. of patients", 0, "count", cells, float("nan"), "")

    def continuous(self, label: str, values: np.ndarray | None, marker: str = "", digits: int = 1) -> None:
        if values is None:
            warnings.warn(f"Skipping continuous variable {label!r}: source column absent.", stacklevel=2)
            return
        values = np.asarray(values, dtype=float)
        cells = {col: fmt_continuous(values[ix], digits, self.cfg.continuous) for col, ix in self.idx.items()}
        pval = continuous_pvalue([values[g] for g in self.group_idx], self.cfg.continuous)
        self._row(label, 1, "var", cells, pval, marker)
        self._missing_row(values)

    def binary(self, label: str, values: np.ndarray | None, marker: str = "") -> None:
        if values is None:
            warnings.warn(f"Skipping binary variable {label!r}: source column absent.", stacklevel=2)
            return
        values = np.asarray(values, dtype=float)
        cells = {}
        for col, ix in self.idx.items():
            sub = values[ix]
            observed = np.isfinite(sub)
            cells[col] = fmt_count_pct(int(np.nansum(sub == 1)), int(observed.sum()))
        pos = [int(np.nansum(values[g] == 1)) for g in self.group_idx]
        neg = [int(np.nansum(values[g] == 0)) for g in self.group_idx]
        pval = chi2_pvalue(np.array([pos, neg]))
        self._row(label, 1, "var", cells, pval, marker)
        self._missing_row(values)

    def categorical(
        self,
        label: str,
        categories: np.ndarray | None,
        marker: str = "",
        order: list[str] | None = None,
    ) -> None:
        if categories is None:
            warnings.warn(f"Skipping categorical variable {label!r}: source column absent.", stacklevel=2)
            return
        cats = np.asarray(categories, dtype=object)
        present = np.array([value is not None for value in cats])
        if order is not None:
            levels = [lv for lv in order if (cats == lv).any()]
        else:
            unique = [lv for lv in set(cats[present].tolist())]
            levels = sorted(unique, key=lambda lv: (-int((cats == lv).sum()), str(lv)))

        contingency = np.array([[int((cats[g] == lv).sum()) for g in self.group_idx] for lv in levels])
        pval = chi2_pvalue(contingency) if len(levels) >= 2 else float("nan")
        self._row(label, 1, "var_header", {col: "" for col in self.idx}, pval, marker)
        for lv in levels:
            cells = {col: fmt_count_pct(int((cats[ix] == lv).sum()), ix.size) for col, ix in self.idx.items()}
            self._row(str(lv), 2, "level", cells, float("nan"), "")
        n_missing = int((~present).sum())
        if n_missing > 0:
            cells = {col: fmt_count_pct(int((~present[ix]).sum()), ix.size) for col, ix in self.idx.items()}
            self._row("Unknown / not reported", 2, "level", cells, float("nan"), "")

    def _missing_row(self, values: np.ndarray) -> None:
        missing = ~np.isfinite(np.asarray(values, dtype=float))
        if int(missing[self.idx["Overall"]].sum()) == 0:
            return
        cells = {col: fmt_count_pct(int(missing[ix].sum()), ix.size) for col, ix in self.idx.items()}
        self._row("Missing", 2, "missing", cells, float("nan"), "")

    # -- final frame -------------------------------------------------------- #
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["section", "label", "indent", "row_type", "marker", "pval", *SPLIT_COLUMNS])


def build_table(dataset: fm.FlowDataset, splits: dict[str, np.ndarray], cfg: TableConfig) -> pd.DataFrame:
    b = TableBuilder(dataset, splits, cfg)

    b.count_row()

    b.section("Demographics")
    b.continuous("Age at surgery, years", frame_numeric(dataset, "AgeAtEvent"), MARK_FLOW, digits=0)
    male = fm.encode_sex_male(dataset.frame[fm.find_compatible_column(list(dataset.frame.columns), "Sex")])
    sex_cat = np.array([("Male" if v == 1 else "Female" if v == 0 else None) for v in male.to_numpy(dtype=float)], dtype=object)
    b.categorical("Sex", sex_cat, MARK_FLOW, order=["Male", "Female"])
    race = frame_raw(dataset, "FirstRace")
    if race is not None:
        b.categorical("Race", normalize_category(race))
    coverage = frame_raw(dataset, "CoverageClass")
    if coverage is not None:
        b.categorical("Insurance / coverage class", normalize_category(coverage))

    b.section("Bariatric procedure")
    surgery_cat = np.array([SURGERY_DISPLAY.get(s, s) for s in dataset.surgery_type], dtype=object)
    b.categorical("Procedure", surgery_cat, MARK_FLOW, order=list(SURGERY_DISPLAY.values()))

    b.section("Anthropometrics and laboratory values at surgery")
    b.continuous("Body mass index, kg/m²", frame_numeric(dataset, "BMIatEvent"), MARK_FLOW, digits=1)
    b.continuous("Hemoglobin A1c, %", frame_numeric(dataset, "HbA1cAtEvent"), MARK_FLOW, digits=1)
    b.continuous("Serum creatinine, mg/dL", frame_numeric(dataset, "CreatinineAtEvent"), MARK_FLOW, digits=2)
    b.continuous("eGFR, mL/min/1.73 m²", frame_numeric(dataset, "eGFRatEvent"), MARK_CAND, digits=1)

    b.section("Comorbidities (past medical history)")
    b.binary("Type 2 diabetes mellitus", frame_numeric(dataset, "PMH_DM2"), MARK_GBM)
    b.binary("Hypertension", frame_numeric(dataset, "PMH_hypertension"), MARK_GBM)
    b.binary("Dyslipidemia", frame_numeric(dataset, "PMH_dyslipidemia"), MARK_FLOW)
    b.binary("Obstructive sleep apnea", frame_numeric(dataset, "PMH_OSA"), MARK_FLOW)
    b.binary("History of myocardial infarction", frame_numeric(dataset, "PMH_MI"), MARK_GBM)
    b.binary("History of stroke", frame_numeric(dataset, "PMH_stroke"), MARK_GBM)
    b.binary("Atrial fibrillation", frame_numeric(dataset, "PMH_AFib"), MARK_GBM)
    b.binary("Venous thromboembolism", frame_numeric(dataset, "PMH_VTE"), MARK_GBM)

    b.section("Glucose-lowering therapy")
    b.binary("Insulin", frame_numeric(dataset, "InsulinStatus"), MARK_FLOW)
    b.binary("Biguanide (metformin)", frame_numeric(dataset, "BiguanideStatus"), MARK_CAND)
    b.binary("SGLT2 inhibitor", frame_numeric(dataset, "SGLT2Status"), MARK_CAND)
    b.binary("Incretin-based therapyᵃ", frame_numeric(dataset, "PostOpGLP1"))

    b.section("Outcomes and follow-up")
    b.continuous("Total weight loss at 12 mo, %ᵇ", percent_total_weight_loss(dataset, "Weight12mPostEvent"), digits=1)
    b.continuous("Total weight loss at 24 mo, %ᵇ", percent_total_weight_loss(dataset, "Weight2yPostEvent"), digits=1)
    mace_dim = fm.TARGET_NAMES.index("mace_ever")
    b.binary("Composite MACE / nephropathy / retinopathyᶜ", dataset.x[:, mace_dim].astype(np.float64))
    b.binary("MACE", fm.binary_event(dataset.frame[fm.find_compatible_column(list(dataset.frame.columns), "MACE")]).to_numpy(dtype=np.float64))
    neph = frame_raw(dataset, "Nephropathy")
    if neph is not None:
        b.binary("Nephropathy", fm.binary_event(neph).to_numpy(dtype=np.float64))
    retino = frame_raw(dataset, "Retinopathy")
    if retino is not None:
        b.binary("Retinopathy", fm.binary_event(retino).to_numpy(dtype=np.float64))

    return b.to_frame()


# --------------------------------------------------------------------------- #
# Footnotes / legend
# --------------------------------------------------------------------------- #
def footnotes(cfg: TableConfig) -> list[str]:
    summary = "median [Q1, Q3]" if cfg.continuous == "median" else "mean (SD)"
    test = "Kruskal-Wallis" if cfg.continuous == "median" else "one-way ANOVA"
    if cfg.split_strategy == "temporal":
        pvalue_note = (
            f"P-values test the null of no difference across train / validation / test ({test} for "
            "continuous variables, chi-square for categorical) and are not adjusted for multiplicity. "
            "The columns are a TEMPORAL (out-of-time) split by surgery date -- earliest surgeries in "
            "Train, latest in Test -- so differences across columns are EXPECTED (calendar drift in case "
            "mix, follow-up maturity, and postoperative GLP-1 exposure) and are reported as such, not as a "
            "randomization balance check. Later-era (Test) patients also have shorter follow-up before the "
            "cohort's surgery-date cutoff, so long-horizon (5-6 yr) outcome cells are sparser in Test by "
            "construction."
        )
    else:
        pvalue_note = (
            f"P-values test the null of no difference across train / validation / test ({test} for "
            "continuous variables, chi-square for categorical). They are a balance check on the random "
            "split and are not adjusted for multiplicity. The split is stratified by bariatric procedure, "
            "so procedure balance is by design."
        )
    return [
        f"Continuous variables are summarized as {summary}; categorical variables as n (%). "
        "Percentages are column percentages (of each split's patients); for continuous variables a "
        "Missing row is shown when any value is absent.",
        pvalue_note,
        f"{MARK_FLOW} Conditioning feature for the flow-matching model (also used by the gradient-boosted risk model).",
        f"{MARK_GBM} Additional feature used by the gradient-boosted MACE risk model only.",
        f"{MARK_CAND} Candidate feature available in the cohort but not currently a model input.",
        "a. Incretin-based therapy: postoperative GLP-1 receptor agonist initiation (preoperative use was "
        "excluded by cohort definition). Post-initiation BMI/HbA1c observations are censored in modeling.",
        "b. Percent total weight loss = (weight at surgery - follow-up weight) / weight at surgery x 100, "
        "from recorded weights; observed values (not GLP-1-censored). Denominator varies with follow-up availability (see Missing).",
        "c. Composite MACE / nephropathy / retinopathy is the binary risk-prediction target modeled by the "
        "GBM and event-conditioned flow (digital twin).",
    ]


def caption(dataset: fm.FlowDataset, cfg: TableConfig) -> str:
    fracs = f"{cfg.train_frac:g}/{cfg.val_frac:g}/{cfg.test_frac:g}"
    if cfg.split_strategy == "temporal":
        split_desc = (
            f"{fracs} temporal / out-of-time split by surgery date "
            "(earliest surgeries in Train, latest in Test)"
        )
    else:
        split_desc = f"split seed {cfg.split_seed}, {fracs} stratified by procedure"
    return (
        "Table 1. Baseline characteristics of the bariatric surgery modeling cohort, "
        f"overall and by data split (n = {len(dataset.subject_ids):,}; {split_desc})."
    )


def split_ns(splits: dict[str, np.ndarray], dataset: fm.FlowDataset) -> dict[str, int]:
    return {
        "Overall": len(dataset.subject_ids),
        "Train": splits["train"].size,
        "Validation": splits["val"].size,
        "Test": splits["test"].size,
    }


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def display_label(row: pd.Series) -> str:
    """Label with role marker appended (markers are drawn inline; the HTML/PNG
    renderers additionally indent by ``row['indent']``)."""
    label = str(row["label"])
    if row["marker"]:
        label = f"{label} {row['marker']}"
    return label


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["p_value"] = out["pval"].map(fmt_pvalue)
    out = out.rename(columns={"label": "characteristic"})
    out = out[["section", "characteristic", "indent", "marker", *SPLIT_COLUMNS, "p_value"]]
    out.to_csv(path, index=False)
    report_saved(path, "Table 1 (CSV)")


def write_markdown(df: pd.DataFrame, ns: dict[str, int], cap: str, notes: list[str], path: Path) -> None:
    headers = ["Characteristic", *[f"{c} (n={ns[c]:,})" for c in SPLIT_COLUMNS], "P value"]
    lines = [f"**{cap}**", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    last_section = None
    for _, row in df.iterrows():
        if row["section"] and row["section"] != last_section:
            last_section = row["section"]
            lines.append(f"| **{last_section}** | " + " | ".join([""] * (len(headers) - 1)) + " |")
        indent = "  " * int(row["indent"])
        label = indent + display_label(row)
        if row["row_type"] in {"count", "var_header"}:
            label = f"**{label}**" if row["row_type"] == "count" else label
        cells = [str(row[c]) for c in SPLIT_COLUMNS]
        lines.append("| " + " | ".join([label, *cells, fmt_pvalue(row["pval"])]) + " |")
    lines.append("")
    lines.append("_Notes_")
    for note in notes:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_saved(path, "Table 1 (Markdown)")


def write_html(df: pd.DataFrame, ns: dict[str, int], cap: str, notes: list[str], path: Path) -> None:
    def esc(text: str) -> str:
        return html_lib.escape(str(text))

    head_cells = "".join(
        f'<th class="num">{esc(col)}<br><span class="sub">n = {ns[col]:,}</span></th>' for col in SPLIT_COLUMNS
    )
    body_rows: list[str] = []
    last_section = None
    for _, row in df.iterrows():
        if row["section"] and row["section"] != last_section:
            last_section = row["section"]
            body_rows.append(f'<tr class="section"><td colspan="6">{esc(last_section)}</td></tr>')
        classes = {
            "count": "count",
            "var": "var",
            "var_header": "varhead",
            "level": "level",
            "missing": "missing",
        }.get(row["row_type"], "var")
        pad = 8 + 20 * int(row["indent"])
        label = esc(row["label"])
        if row["marker"]:
            label += f' <sup class="mark">{esc(row["marker"])}</sup>'
        cells = "".join(f'<td class="num">{esc(row[c])}</td>' for c in SPLIT_COLUMNS)
        pval = fmt_pvalue(row["pval"])
        body_rows.append(
            f'<tr class="{classes}">'
            f'<td class="char" style="padding-left:{pad}px">{label}</td>'
            f"{cells}"
            f'<td class="num pval">{esc(pval)}</td></tr>'
        )
    notes_html = "".join(f"<li>{esc(note)}</li>" for note in notes)
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Table 1</title>
<style>
  body {{ font-family: "Helvetica Neue", Arial, sans-serif; color: #1a1a1a; margin: 32px; }}
  .wrap {{ max-width: 960px; margin: 0 auto; }}
  caption {{ caption-side: top; text-align: left; font-weight: 600; font-size: 15px;
             margin-bottom: 10px; line-height: 1.4; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  thead th {{ border-top: 2px solid #222; border-bottom: 1.5px solid #222; padding: 8px 10px;
              text-align: right; font-weight: 600; vertical-align: bottom; }}
  thead th:first-child {{ text-align: left; }}
  th .sub {{ font-weight: 400; color: #555; font-size: 11px; }}
  tbody td {{ padding: 5px 10px; border: none; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  td.char {{ text-align: left; }}
  tr.section td {{ font-weight: 700; background: #eef1f6; border-top: 1px solid #cbd2df;
                   border-bottom: 1px solid #cbd2df; padding: 6px 10px; letter-spacing: .02em; }}
  tr.count td {{ font-weight: 600; }}
  tr.varhead td.char {{ font-weight: 500; }}
  tr.level td.char {{ color: #333; }}
  tr.missing td {{ color: #888; font-style: italic; }}
  tr.missing td.char {{ font-style: italic; }}
  tbody tr.var:nth-of-type(even), tbody tr.level:nth-of-type(even) {{ }}
  sup.mark {{ color: #4056a1; font-weight: 700; }}
  td.pval {{ color: #333; }}
  tbody tr:last-child td {{ border-bottom: 2px solid #222; }}
  .notes {{ font-size: 11.5px; color: #444; margin-top: 12px; line-height: 1.5; padding-left: 18px; }}
  .notes li {{ margin-bottom: 3px; }}
</style></head>
<body><div class="wrap">
<table>
<caption>{esc(cap)}</caption>
<thead><tr><th>Characteristic</th>{head_cells}<th class="num">P value</th></tr></thead>
<tbody>
{chr(10).join(body_rows)}
</tbody>
</table>
<ul class="notes">
{notes_html}
</ul>
</div></body></html>
"""
    path.write_text(document, encoding="utf-8")
    report_saved(path, "Table 1 (HTML)")


def write_png(df: pd.DataFrame, ns: dict[str, int], cap: str, notes: list[str], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is present in the venv.
        warnings.warn(f"Skipped PNG render: {exc}", stacklevel=2)
        return

    headers = ["Characteristic", *[f"{c}\n(n={ns[c]:,})" for c in SPLIT_COLUMNS], "P value"]
    display_rows: list[list[str]] = []
    styles: list[str] = []  # per-row style tag for coloring
    last_section = None
    for _, row in df.iterrows():
        if row["section"] and row["section"] != last_section:
            last_section = row["section"]
            display_rows.append([last_section, "", "", "", "", ""])
            styles.append("section")
        indent = "   " * int(row["indent"])
        label = indent + display_label(row)
        display_rows.append([label, *[str(row[c]) for c in SPLIT_COLUMNS], fmt_pvalue(row["pval"])])
        styles.append(row["row_type"])

    import textwrap

    n_body = len(display_rows)
    wrapped_notes = [textwrap.fill(note, width=160, subsequent_indent="    ") for note in notes]
    note_lines = sum(note.count("\n") + 1 for note in wrapped_notes)

    # Size the figure to the content and split it into three stacked bands (title,
    # table, footnotes) via explicit margins, so the footnotes sit flush beneath
    # the table with no dead space regardless of row count.
    title_in, row_in, note_in, pad_in = 0.75, 0.30, 0.135, 0.20
    table_in = row_in * (n_body + 1)  # + 1 for the header row
    notes_in = note_in * note_lines + 0.15
    fig_w = 13.0
    fig_h = title_in + table_in + notes_in + pad_in
    top_frac = 1.0 - title_in / fig_h
    bottom_frac = notes_in / fig_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(cap, fontsize=12, fontweight="bold", loc="left", wrap=True, pad=10)
    fig.subplots_adjust(left=0.015, right=0.985, top=top_frac, bottom=bottom_frac)

    # bbox=[0,0,1,1] forces the table to fill the axes exactly (equal row heights),
    # which -- unlike per-cell set_height -- keeps its rendered extent in sync with
    # the axes so the footnotes below never overlap the last rows.
    table = ax.table(cellText=display_rows, colLabels=headers, cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    col_widths = [0.40, 0.1325, 0.1325, 0.1325, 0.1325, 0.07]
    for (r, c), cell in table.get_celld().items():
        cell.set_width(col_widths[c])
        cell.set_edgecolor("#d9d9d9")
        text = cell.get_text()
        if c == 0:
            text.set_ha("left")
            cell.set_text_props(ha="left")
            cell.PAD = 0.01
        if r == 0:  # header
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#40466e")
            cell.set_edgecolor("#40466e")
            continue
        style = styles[r - 1]
        if style == "section":
            cell.set_facecolor("#eef1f6")
            if c == 0:
                cell.set_text_props(weight="bold")
        elif style == "count":
            cell.set_facecolor("#ffffff")
            cell.set_text_props(weight="bold")
        elif style == "missing":
            cell.set_facecolor("#ffffff")
            cell.set_text_props(color="#8a8a8a", style="italic")
        elif style == "level":
            cell.set_facecolor("#fafafa")
        else:
            cell.set_facecolor("#ffffff")

    fig.text(0.015, bottom_frac - 0.008, "\n".join(wrapped_notes), fontsize=7.4, va="top", ha="left", color="#333333")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    report_saved(path, "Table 1 (PNG)")


def write_latex(df: pd.DataFrame, ns: dict[str, int], cap: str, notes: list[str], path: Path) -> None:
    def esc(text: str) -> str:
        # Escape LaTeX specials first, then map the unicode markers/superscripts to
        # LaTeX commands (added after escaping so their backslashes are not re-escaped).
        # The result compiles under plain pdflatex without inputenc/fontspec tricks.
        specials = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
        unicode_map = {"†": r"\dag{}", "‡": r"\ddag{}", "§": r"\S{}", "²": r"\textsuperscript{2}",
                       "ᵃ": r"\textsuperscript{a}", "ᵇ": r"\textsuperscript{b}", "ᶜ": r"\textsuperscript{c}",
                       "×": r"$\times$", "≥": r"$\geq$"}
        out = str(text)
        for key, value in specials.items():
            out = out.replace(key, value)
        for key, value in unicode_map.items():
            out = out.replace(key, value)
        return out

    header = " & ".join(["\\textbf{Characteristic}", *[f"\\textbf{{{c}}} (n={ns[c]:,})" for c in SPLIT_COLUMNS], "\\textbf{P value}"])
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{esc(cap)}}}",
        "\\begin{tabular}{l r r r r r}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    last_section = None
    for _, row in df.iterrows():
        if row["section"] and row["section"] != last_section:
            last_section = row["section"]
            lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{esc(last_section)}}}}} \\\\")
        indent = "\\quad " * int(row["indent"])
        label = indent + esc(row["label"]) + (f"\\textsuperscript{{{esc(row['marker'])}}}" if row["marker"] else "")
        if row["row_type"] == "count":
            label = f"\\textbf{{{label}}}"
        cells = [esc(row[c]) for c in SPLIT_COLUMNS]
        lines.append(" & ".join([label, *cells, esc(fmt_pvalue(row["pval"]))]) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    for note in notes:
        lines.append(f"\\par\\footnotesize {esc(note)}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_saved(path, "Table 1 (LaTeX)")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def make_output_dir(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    run_dir = root / f"table_one_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_table_splits(dataset: fm.FlowDataset, cfg: TableConfig) -> dict[str, np.ndarray]:
    """The train/val/test partition the table describes, matched to the model split.

    Routes on ``cfg.split_strategy`` so the Table 1 columns line up patient-for-patient
    with whichever fold the models use: "surgery" -> fm.make_stratified_splits (default,
    unchanged), "temporal" -> fm.make_temporal_splits (earliest surgeries -> Train, latest
    -> Test). Both consume the same split_seed/fractions as the trainers, so identical
    inputs yield the identical partition.
    """
    split_cfg = fm.TrainConfig(
        split_seed=cfg.split_seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
    )
    if cfg.split_strategy == "temporal":
        return fm.make_temporal_splits(dataset, split_cfg)
    if cfg.split_strategy != "surgery":
        raise ValueError(
            f"Unknown split_strategy: {cfg.split_strategy!r} (expected 'surgery' or 'temporal')"
        )
    return fm.make_stratified_splits(dataset, split_cfg)


def generate(dataset: fm.FlowDataset, cfg: TableConfig) -> Path:
    splits = make_table_splits(dataset, cfg)
    df = build_table(dataset, splits, cfg)
    ns = split_ns(splits, dataset)
    cap = caption(dataset, cfg)
    notes = footnotes(cfg)

    run_dir = make_output_dir(cfg.output_dir)
    write_csv(df, run_dir / "table_one.csv")
    write_html(df, ns, cap, notes, run_dir / "table_one.html")
    write_png(df, ns, cap, notes, run_dir / "table_one.png")
    write_markdown(df, ns, cap, notes, run_dir / "table_one.md")
    write_latex(df, ns, cap, notes, run_dir / "table_one.tex")

    print(
        f"\nTable 1 written for n={ns['Overall']:,} "
        f"(train={ns['Train']:,}, val={ns['Validation']:,}, test={ns['Test']:,}).",
        flush=True,
    )
    print(f"Artifacts in {run_dir}", flush=True)
    return run_dir


def generate_from_csv(csv_path: str | Path, cfg: TableConfig | None = None) -> Path:
    cfg = cfg or TableConfig()
    return generate(fm.load_dataset_from_csv(csv_path), cfg)


def generate_from_database(cfg: TableConfig | None = None) -> Path:
    cfg = cfg or TableConfig()
    return generate(fm.load_dataset_from_database(), cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=str, default=None,
                        help="Local CSV export (post-SQL). Omit to query Cosmos MBSCohort.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--continuous", choices=["median", "mean"], default="median",
                        help="Summary for continuous variables: median [Q1, Q3] (default) or mean (SD).")
    parser.add_argument("--split-strategy", type=str, default="surgery", choices=["surgery", "temporal"],
                        help="Which model fold the Train/Validation/Test columns describe: 'surgery' "
                             "(stratified, default) or 'temporal' (out-of-time, by surgery date).")
    args = parser.parse_args()

    cfg = TableConfig(
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        continuous=args.continuous,
        split_strategy=args.split_strategy,
    )
    try:
        if args.csv_path:
            generate_from_csv(args.csv_path, cfg)
        else:
            generate_from_database(cfg)
    except RuntimeError as exc:
        print(f"ERROR: {exc}\n\nPass --csv <path> to build the table from a saved CSV export.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
