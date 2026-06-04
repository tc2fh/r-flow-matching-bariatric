"""Print train/validation/test patient counts for a saved flow-matching run.

Run:
    python split_counts_flow_matching.py --run <run_dir>

For a CSV export instead of re-querying Cosmos:
    python split_counts_flow_matching.py --run <run_dir> --csv data/cosmos_mbs_flow_input.csv
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import numpy as np

import train_flow_matching as fm


def is_run_dir(path: Path) -> bool:
    return (path / "config.json").exists()


def find_latest_run(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    if is_run_dir(log_dir):
        return log_dir
    candidates = [
        path
        for path in log_dir.rglob("run_*")
        if path.is_dir() and is_run_dir(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_run_dir(path: Path | None, log_dir: Path) -> Path:
    search_root = log_dir if path is None else Path(path)
    if is_run_dir(search_root):
        return search_root

    final_model = search_root / "final_model.json"
    if final_model.exists():
        payload = json.loads(final_model.read_text(encoding="utf-8"))
        saved = Path(payload["run_dir"])
        candidates = [saved]
        if not saved.is_absolute():
            candidates.append(search_root / saved)
            candidates.append(search_root.parent / saved)
        for candidate in candidates:
            if is_run_dir(candidate):
                return candidate

    latest = find_latest_run(search_root)
    if latest is not None:
        return latest

    raise SystemExit(f"No run directory with config.json found under {search_root}")


def load_config(run_dir: Path) -> tuple[fm.TrainConfig, dict[str, Any]]:
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    valid = {field.name for field in fields(fm.TrainConfig)}
    cfg = fm.TrainConfig(**{key: value for key, value in raw.items() if key in valid})
    return cfg, raw


def load_dataset(csv_path: Path | None) -> fm.FlowDataset:
    if csv_path is not None:
        return fm.load_dataset_from_csv(csv_path)
    try:
        return fm.load_dataset_from_database()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\n\nPass --csv <path> to count splits from a saved CSV export.") from exc


def split_count_rows(dataset: fm.FlowDataset, splits: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    procedures = tuple(sorted(fm.SURGERY_TO_INDEX))
    for split_name in ("train", "val", "test"):
        idx = splits[split_name]
        row: dict[str, Any] = {"split": split_name, "patients": int(len(idx))}
        for procedure in procedures:
            row[procedure] = int((dataset.surgery_type[idx] == procedure).sum())
        rows.append(row)
    rows.append(
        {
            "split": "total",
            "patients": int(len(dataset.subject_ids)),
            **{
                procedure: int((dataset.surgery_type == procedure).sum())
                for procedure in procedures
            },
        }
    )
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0].keys())
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None, help="Run dir, Optuna study dir, or best_model dir.")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/python_flow_matching_optuna"))
    parser.add_argument("--csv", "--csv-path", dest="csv_path", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run, args.log_dir)
    cfg, raw_config = load_config(run_dir)
    dataset = load_dataset(args.csv_path)
    splits = fm.make_stratified_splits(dataset, cfg)
    rows = split_count_rows(dataset, splits)

    if args.json:
        print(json.dumps({"run_dir": str(run_dir), "counts": rows}, indent=2))
        return

    print(f"Run: {run_dir}")
    print(
        "Split config: "
        f"seed={cfg.split_seed}, train={cfg.train_frac:g}, val={cfg.val_frac:g}, test={cfg.test_frac:g}"
    )
    if "target_names" in raw_config:
        print(f"Target dimensions: {len(raw_config['target_names'])}")
    print_table(rows)


if __name__ == "__main__":
    main()
