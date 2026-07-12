"""Reproducibility helpers that work in Git and file-transfer deployments.

The Cosmos VM receives source files by manual transfer and intentionally has no
``.git`` directory. A deterministic source-tree fingerprint therefore provides
the code identity used to pair internal and temporal study runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


FINGERPRINT_ALGORITHM = "sha256-path-and-content-v1"
SOURCE_SUFFIXES = frozenset({".py"})
EXCLUDED_PARTS = frozenset({
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "runs",
    "temp",
})


def _source_files(root: Path) -> list[Path]:
    root = root.resolve()
    return sorted(
        (
            path for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SOURCE_SUFFIXES
            and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(root: str | Path) -> dict:
    """Hash first-party Python paths and bytes in a platform-independent order.

    Generated outputs, caches, virtual environments, and Git metadata are
    excluded. Per-file hashes are retained so a mismatched manual transfer can
    be diagnosed without guessing which file changed.
    """
    root = Path(root).resolve()
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in _source_files(root)
    ]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": FINGERPRINT_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(rows),
        "files": rows,
    }
