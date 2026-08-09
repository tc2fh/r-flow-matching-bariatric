#!/usr/bin/env python3
"""One-command orchestrator for the consolidated metabolic trajectory study.

The study ships as three files but one collaborator command, ``--from-scratch``:

  1. ``run_metabolic_trajectory_study.py`` - modeling layer; writes the frozen prediction store
     and its own figure book to ``<run>/FIGURES_TO_EXPORT``.
  2. ``run_secondary_analyses.py`` - analysis layer; consumes the frozen store and writes its
     figure book to ``<run>/secondary/FIGURES_TO_EXPORT``.
  3. this file - runs the two layers as SEPARATE SUBPROCESSES and merges the two figure books
     into one ``final_study_figure_book.pdf`` at the run directory root.

The subprocess boundary is the memory-bounding mechanism, not a convenience: each layer starts
with a fresh resident-set size and returns all of it to the operating system when it exits, so
peak RSS is bounded by the worst single layer rather than their sum. Neither script is ever
imported into this process.

    python qreg_improvement/run_final_study.py --from-scratch      # production, Cosmos VM
    python qreg_improvement/run_final_study.py --from-run RUN_DIR  # reuse a modeling run
    python qreg_improvement/run_final_study.py --self-test         # torch-free, ~1 s
    python qreg_improvement/run_final_study.py --smoke             # torch-free whole chain

If a long production run dies part way, resume the modeling layer directly with its own
``--resume --output-dir <run>``, then finish here with ``--from-run <run>``.

Self-contained by design: no project-local imports (a self-test asserts it), and nothing beyond
the standard library plus matplotlib, which both study scripts already require. Never torch.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
MODELING_SCRIPT = THIS_FILE.parent / "run_metabolic_trajectory_study.py"
SECONDARY_SCRIPT = THIS_FILE.parent / "run_secondary_analyses.py"

MERGED_BOOK_NAME = "final_study_figure_book.pdf"
FINAL_BUNDLE_NAME = "final_study_bundle.zip"
EXPORT_DIR_NAME = "FIGURES_TO_EXPORT"
SECONDARY_DIR_NAME = "secondary"
MODEL_BOOK = "metabolic_trajectory_figure_book.pdf"
MODEL_BUNDLE = "metabolic_trajectory_results_bundle.zip"
SECONDARY_BUNDLE = "secondary_analyses_results_bundle.zip"
# A zip records each member's mtime, so pinning it keeps two identical bundles byte-identical.
BUNDLE_TIMESTAMP = (2001, 1, 1, 0, 0, 0)

# Both books render letter-landscape pages. Fixing the merged page width and deriving the
# height from the source pixels keeps every page at its own book's aspect ratio, and deriving
# the dpi from the same pixels maps source pixels to page pixels 1:1 (no resampling).
PAGE_WIDTH_INCHES = 11.0

# A fixed CreationDate (matplotlib otherwise stamps "now") is what makes two merges of the same
# pages byte-identical. Creator/Producer carry the matplotlib version, which is fixed per
# environment.
MERGED_PDF_METADATA = {
    "Title": "Consolidated metabolic trajectory study - merged figure book",
    "Author": "Brannigan Lab",
    "CreationDate": datetime(2001, 1, 1, tzinfo=timezone.utc),
}

MERGE_RECORD_NAME = "final_run_record.json"

# ctypes/resource are stdlib and are imported only inside peak_rss_gb: this file may not import
# the study modules, so the memory probe cannot be borrowed from them and is restated here.
ALLOWED_IMPORT_ROOTS = frozenset({
    "__future__", "argparse", "ast", "ctypes", "dataclasses", "datetime", "json", "os",
    "pathlib", "re", "resource", "subprocess", "sys", "tempfile", "time", "zipfile",
    "matplotlib",
})
PROJECT_LOCAL_IMPORT_PREFIXES = (
    "qreg_improvement", "run_metabolic_trajectory_study", "run_secondary_analyses",
    "run_qreg_improvement", "causal_tte", "calibration_twin", "train_", "figures",
    "distributional_metrics")


class StageFailure(RuntimeError):
    """A stage subprocess exited non-zero. The chain aborts and never reaches the merge."""

    def __init__(self, label: str, code: int) -> None:
        super().__init__(f"stage {label!r} exited with code {code}")
        self.label, self.code = label, code


@dataclass(frozen=True)
class Plan:
    """The resolved run: where each layer writes, and the exact subprocess argv per stage."""

    mode: str
    out_dir: Path
    model_run: Path
    stages: tuple[tuple[str, list[str]], ...]

    @property
    def books(self) -> tuple[Path, ...]:
        """The two export directories, in merged-book order: modeling first, secondary second."""
        return (self.model_run / EXPORT_DIR_NAME,
                self.out_dir / SECONDARY_DIR_NAME / EXPORT_DIR_NAME)

    @property
    def merged_book(self) -> Path:
        # Deliberately at the run root: each script validates its own FIGURES_TO_EXPORT against a
        # fixed file contract and rejects any foreign file placed inside it.
        return self.out_dir / MERGED_BOOK_NAME

    @property
    def bundle(self) -> Path:
        return self.out_dir / FINAL_BUNDLE_NAME

    @property
    def bundle_members(self) -> tuple[tuple[str, Path], ...]:
        """(archive name, source) for the returnable deliverable, in reading order. The modeling
        layer writes no zip of its own today, so its book and manifest stand in; its bundle entry
        stays listed so it appears if that layer grows one, and is skipped with a note for now."""
        model, sec = self.model_run, self.out_dir / SECONDARY_DIR_NAME
        return ((MERGED_BOOK_NAME, self.merged_book),
                (f"modeling/{MODEL_BOOK}", model / EXPORT_DIR_NAME / MODEL_BOOK),
                ("modeling/run_manifest.json", model / "run_manifest.json"),
                (f"modeling/{MODEL_BUNDLE}", model / MODEL_BUNDLE),
                (f"secondary/{SECONDARY_BUNDLE}", sec / SECONDARY_BUNDLE),
                ("secondary/manifest.json", sec / "manifest.json"))


def default_run_dir() -> Path:
    """A collision-safe timestamped run directory, matching the study scripts' convention."""
    root = Path.cwd().resolve() / "results"
    stem = f"final_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}_{suffix:02d}"
        suffix += 1
    return candidate


def build_plan(mode: str, output_dir: str | None = None, from_run: str | None = None,
               enable_torch_candidates: bool = False) -> Plan:
    """Resolve directories and build the exact argv list this orchestrator will execute.

    ``enable_torch_candidates`` is forwarded to the modeling layer only, and only where this
    orchestrator actually fits models (--from-scratch). It is off by default: the PyTorch
    candidates are the memory-critical path on the production VM, so the collaborator opts in
    explicitly with ``--from-scratch --enable-torch-candidates``.
    """
    if mode == "from-run":
        if not from_run:
            raise SystemExit("--from-run requires RUN_DIR (a completed modeling run directory)")
        model_run = Path(from_run).expanduser().resolve()
        out_dir = Path(output_dir).expanduser().resolve() if output_dir else model_run
    else:
        out_dir = Path(output_dir).expanduser().resolve() if output_dir else default_run_dir()
        model_run = out_dir

    modeling = [sys.executable, str(MODELING_SCRIPT)]
    secondary = [sys.executable, str(SECONDARY_SCRIPT)]
    if mode == "from-scratch":
        # Production is the modeling script's default mode, and production already runs
        # process-per-stage orchestration internally.
        fit = modeling + ["--output-dir", str(out_dir)]
        if enable_torch_candidates:
            fit += ["--enable-torch-candidates"]
        stages = (
            ("modeling", fit),
            ("secondary", secondary + ["--from-run", str(out_dir)]),
        )
    elif mode == "from-run":
        analyse = secondary + ["--from-run", str(model_run)]
        if out_dir != model_run:
            analyse += ["--output-dir", str(out_dir)]
        stages = (("secondary", analyse),)
    elif mode == "smoke":
        # Two independent synthetic smokes: the secondary --from-run path needs Cosmos for its
        # covariate frame, so it cannot consume the smoke store locally. What this exercises is
        # the subprocess wiring and the merge, which is exactly what this file owns.
        stages = (
            ("modeling", modeling + ["--smoke", "--orchestrate", "--output-dir", str(out_dir)]),
            ("secondary", secondary + ["--smoke", "--output-dir", str(out_dir)]),
        )
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return Plan(mode=mode, out_dir=out_dir, model_run=model_run, stages=stages)


def run_stage(label: str, command: list[str], index: int, total: int) -> None:
    """Run one stage in its own process, inheriting stdio so its progress streams live."""
    print(f"[final] stage {index}/{total} START {label}\n[final]   {' '.join(command)}", flush=True)
    started = time.monotonic()
    # Unbuffered children keep the collaborator's redirected log current during long stages.
    code = subprocess.call(command, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    elapsed = time.monotonic() - started
    if code != 0:
        print(f"[final] stage {index}/{total} FAILED {label} after {elapsed:.1f}s (exit {code})",
              flush=True)
        raise StageFailure(label, code)
    print(f"[final] stage {index}/{total} DONE {label} in {elapsed:.1f}s", flush=True)


def collect_pages(export_dir: Path) -> list[Path]:
    """The numbered PNG pages of one book, in page order (the names are zero-padded)."""
    return sorted((p for p in export_dir.glob("*.png") if p.is_file()), key=lambda p: p.name)


def merge_books(books: tuple[Path, ...] | list[Path], destination: Path) -> list[Path]:
    """Bind every page of every book into one PDF and return the source pages, in order.

    matplotlib is the dependency-clean way to do this: pypdf/PyPDF2 are not installed and adding
    them is not permitted, while matplotlib is already required by both study scripts. Each
    exported PNG becomes one full-page image, so no page content is re-derived or re-styled.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pages: list[Path] = []
    for book in books:
        found = collect_pages(Path(book))
        if not found:
            raise RuntimeError(f"no figure pages found to merge in {book}")
        pages.extend(found)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (destination.name + ".tmp")
    with PdfPages(temporary, metadata=MERGED_PDF_METADATA) as pdf:
        for page in pages:
            image = mpimg.imread(page)
            height, width = int(image.shape[0]), int(image.shape[1])
            dpi = width / PAGE_WIDTH_INCHES
            figure = plt.figure(figsize=(PAGE_WIDTH_INCHES, height / dpi), dpi=dpi)
            figure.figimage(image, 0, 0, origin="upper", resize=False)
            pdf.savefig(figure, dpi=dpi)
            plt.close(figure)
            del image
    temporary.replace(destination)
    return pages


def write_bundle(members: tuple[tuple[str, Path], ...], destination: Path) -> list[str]:
    """Zip the final deliverables into one returnable archive; return the member names written.

    The member list is fixed and fully resolved before the archive is opened, and the temporary
    archive is written outside every directory read, so the bundle can never include itself. A
    member a child layer did not produce is logged and skipped, never fatal."""
    present = [(name, path) for name, path in members if path.is_file()]
    for name, path in members:
        if not path.is_file():
            print(f"[final] bundle: skipping absent member {name} ({path})", flush=True)
    if not present:
        raise RuntimeError(f"no bundle members found for {destination}")
    temporary = destination.parent / (destination.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in present:
            entry = zipfile.ZipInfo(name, date_time=BUNDLE_TIMESTAMP)
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    temporary.replace(destination)
    return [name for name, _ in present]


def peak_rss_gb() -> float | None:
    """This process's peak resident-set size in GB, or None where the platform cannot report it.

    Restated here rather than imported: this file's contract is that it imports nothing
    project-local, so it cannot borrow the study module's probe. Stdlib only - resource.getrusage
    on POSIX, the Win32 GetProcessMemoryInfo PeakWorkingSetSize on Windows. Never raises; a
    memory report must not be able to fail a run.
    """
    try:
        import resource

        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes; macOS/BSD report bytes.
        peak = int(maximum) * 1024 if sys.platform.startswith("linux") else int(maximum)
        return round(peak / (1024 ** 3), 3)
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # restype/argtypes are load-bearing, not cosmetic: left untyped, ctypes assumes a C int
        # return, so GetCurrentProcess's (HANDLE)-1 pseudo-handle is truncated to 32 bits and
        # GetProcessMemoryInfo fails on every 64-bit Windows host - the production VM included.
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD
        ]
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return round(int(counters.PeakWorkingSetSize) / (1024 ** 3), 3)
    except Exception:
        pass
    return None


def write_run_record(plan: Plan, pages: int, members: list[str], merge_peak_gb: float | None) -> Path:
    """The orchestrator's own machine-readable record, beside the merged book.

    The two child layers each write a manifest carrying their per-stage peak RSS; the merge runs
    in THIS process and so appears in neither. Section 11 asks for the probe to be visible in the
    artifacts rather than only in a console log, so the merge's peak is written here. Deliberately
    NOT a bundle member: the bundle's member list is a fixed contract, and this record is written
    after the bundle it would have to describe.
    """
    record = {
        "mode": plan.mode,
        "run_dir": str(plan.out_dir),
        "completed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "stages": [label for label, _ in plan.stages],
        "merged_book": plan.merged_book.name,
        "merged_pages": pages,
        "bundle": plan.bundle.name,
        "bundle_members": members,
        "peak_rss_gb_by_stage": {"merge": merge_peak_gb},
    }
    destination = plan.out_dir / MERGE_RECORD_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (destination.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def run_chain(plan: Plan) -> int:
    """Run every stage, then merge. A failing stage aborts before the merge and propagates."""
    print(f"[final] mode: {plan.mode}\n[final] run directory: {plan.out_dir}", flush=True)
    total = len(plan.stages)
    for index, (label, command) in enumerate(plan.stages, start=1):
        try:
            run_stage(label, command, index, total)
        except StageFailure as failure:
            print(f"[final] ABORTED at stage '{failure.label}' (exit {failure.code}). "
                  "The figure books were not merged.", file=sys.stderr, flush=True)
            return failure.code
    print(f"[final] merging {len(plan.books)} figure books", flush=True)
    pages = merge_books(plan.books, plan.merged_book)
    print(f"[final] merged figure book: {plan.merged_book} ({len(pages)} pages)")
    members = write_bundle(plan.bundle_members, plan.bundle)
    print(f"[final] final bundle: {plan.bundle} ({len(members)} members)")
    # The merge holds one full-page image at a time plus the growing PDF; this is the number that
    # would move first if a future page count or page size outgrew the VM.
    merge_peak = peak_rss_gb()
    reported = "not reported by this platform" if merge_peak is None else f"{merge_peak:.2f} GB"
    print(f"[final] peak memory after merge: {reported}", flush=True)
    record = write_run_record(plan, len(pages), members, merge_peak)
    print(f"[final] run record: {record}")
    return 0


# --- Self-test: wiring, abort-before-merge, merge correctness, single-file discipline -------
def pdf_page_count(path: Path) -> int:
    """Count PDF pages without a PDF library: count '/Type /Page' occurrences (not '/Pages')."""
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


def _write_book(directory: Path, names: list[str]) -> Path:
    """A throwaway book of tiny numbered pages, for the merge tests."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        figure = plt.figure(figsize=(2.0, 1.5), dpi=50)
        figure.text(0.1, 0.5, name)
        figure.savefig(directory / name, format="png", dpi=50)
        plt.close(figure)
    return directory


def run_self_tests() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    def raises(call) -> bool:
        try:
            call()
        except (SystemExit, RuntimeError):
            return True
        return False

    def wiring(plan: Plan, expected: list) -> tuple[bool, str]:
        actual = [[label, command] for label, command in plan.stages]
        return actual == expected, f"{[command for _, command in plan.stages]}"

    run = Path("/tmp/run_dir").resolve()
    modeling, secondary = str(MODELING_SCRIPT), str(SECONDARY_SCRIPT)

    # 1. --from-scratch: full production modeling into the run dir, then secondary --from-run it.
    #    The torch memory gate is OFF unless asked for, and when asked for it reaches the modeling
    #    layer only - the secondary fits no candidates, so forwarding it there would be noise.
    plan = build_plan("from-scratch", output_dir=str(run))
    default_ok, default_detail = wiring(plan, [
        ["modeling", [sys.executable, modeling, "--output-dir", str(run)]],
        ["secondary", [sys.executable, secondary, "--from-run", str(run)]],
    ])
    opted_in = build_plan("from-scratch", output_dir=str(run), enable_torch_candidates=True)
    opted_ok, opted_detail = wiring(opted_in, [
        ["modeling", [sys.executable, modeling, "--output-dir", str(run),
                      "--enable-torch-candidates"]],
        ["secondary", [sys.executable, secondary, "--from-run", str(run)]],
    ])
    check("01_from_scratch_command_wiring", default_ok and opted_ok,
          f"default={default_detail}; opted-in={opted_detail}")

    # 2. --from-run: modeling is skipped entirely; the secondary layer consumes the given run.
    plan_reuse = build_plan("from-run", from_run=str(run))
    check("02_from_run_command_wiring", *wiring(plan_reuse, [
        ["secondary", [sys.executable, secondary, "--from-run", str(run)]],
    ]))

    # 3. --smoke: both layers run their own synthetic smoke into the same run dir.
    plan_smoke = build_plan("smoke", output_dir=str(run))
    check("03_smoke_command_wiring", *wiring(plan_smoke, [
        ["modeling", [sys.executable, modeling, "--smoke", "--orchestrate",
                      "--output-dir", str(run)]],
        ["secondary", [sys.executable, secondary, "--smoke", "--output-dir", str(run)]],
    ]))

    # 4. Every stage is a subprocess of this interpreter, never an in-process import.
    check("04_stages_are_subprocesses_of_this_interpreter",
          all(c[0] == sys.executable and c[1].endswith(".py")
              for _, c in plan.stages + plan_smoke.stages + plan_reuse.stages), sys.executable)

    # 5. --output-dir threading, and the merged book stays out of both export directories
    #    (each script rejects foreign files inside its own FIGURES_TO_EXPORT).
    check("05_output_dir_threading_and_book_locations",
          plan.books == (run / EXPORT_DIR_NAME, run / SECONDARY_DIR_NAME / EXPORT_DIR_NAME)
          and plan.merged_book == run / MERGED_BOOK_NAME
          and all(book not in plan.merged_book.parents for book in plan.books),
          f"books={[str(b) for b in plan.books]}")

    # 6. --from-run without a run directory is refused rather than silently defaulted.
    check("06_from_run_requires_run_dir", raises(lambda: build_plan("from-run")))

    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)

        # 7. A failing stage aborts the chain: later stages never start and the merge is skipped.
        marker = workspace / "later_stage_ran.txt"
        aborting = Plan(
            mode="test",
            out_dir=workspace / "abort",
            model_run=workspace / "abort",
            stages=(
                ("ok", [sys.executable, "-c", "pass"]),
                ("boom", [sys.executable, "-c", "raise SystemExit(3)"]),
                ("later", [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]),
            ),
        )
        code = run_chain(aborting)
        check("07_failing_stage_aborts_before_merge",
              code == 3 and not marker.exists() and not aborting.merged_book.exists(),
              f"exit={code}; later_stage_ran={marker.exists()}")

        # 8. The merge binds every page of both books, in book order then page order.
        names_a, names_b = ["00_a.png", "01_a.png", "02_a.png"], ["00_b.png", "01_b.png"]
        first = _write_book(workspace / "book_a", names_a)
        second = _write_book(workspace / "book_b", names_b)
        merged = workspace / "merged" / MERGED_BOOK_NAME
        pages = merge_books((first, second), merged)
        expected_order = [first / name for name in names_a] + [second / name for name in names_b]
        counted = pdf_page_count(merged)
        check("08_merge_page_count_and_order",
              pages == expected_order and counted == len(expected_order) == 5,
              f"pages={counted}; order={[p.name for p in pages]}")

        # 9. The merge is byte-reproducible (fixed PDF CreationDate, sorted page collection).
        repeat = workspace / "merged_again" / MERGED_BOOK_NAME
        merge_books((first, second), repeat)
        check("09_merge_is_byte_deterministic",
              merged.read_bytes() == repeat.read_bytes(),
              f"{merged.stat().st_size} vs {repeat.stat().st_size} bytes")

        # 10. An empty or missing book is a loud failure, never a silently short merged book.
        check("10_missing_book_fails_loudly", raises(
            lambda: merge_books((first, workspace / "book_missing"), workspace / "never.pdf")))

        # 11. The final bundle carries the fixed member list, skips what a child did not produce
        #     (here the modeling zip, which that layer does not write today), and reproduces.
        root = workspace / "bundle"
        fake = Plan(mode="test", out_dir=root, model_run=root, stages=())
        absent = f"modeling/{MODEL_BUNDLE}"
        for name, path in fake.bundle_members:
            path.parent.mkdir(parents=True, exist_ok=True)
            if name != absent:
                path.write_bytes(name.encode())
        written = write_bundle(fake.bundle_members, fake.bundle)
        twin = workspace / "bundle_again.zip"
        write_bundle(fake.bundle_members, twin)
        with zipfile.ZipFile(fake.bundle) as archive:
            inside = archive.namelist()
        check("11_final_bundle_members_and_determinism",
              inside == written == [n for n, _ in fake.bundle_members if n != absent]
              and fake.bundle.read_bytes() == twin.read_bytes(), f"members={inside}")

    # 12. Single-file discipline: no project-local imports, stdlib plus matplotlib only.
    imported: list[str] = []
    for node in ast.walk(ast.parse(THIS_FILE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    project_local = [n for n in imported if n.startswith(PROJECT_LOCAL_IMPORT_PREFIXES)]
    outside = sorted({n for n in imported if n.split(".")[0] not in ALLOWED_IMPORT_ROOTS})
    check("12_no_project_local_or_forbidden_imports",
          not project_local and not outside,
          f"project_local={project_local}; outside_allowlist={outside}")

    # 13. The merge runs in THIS process, so neither child manifest can carry its peak RSS. The
    #     probe must report a real number on the platforms this study runs on, and the run record
    #     must carry it under the same key the two child manifests use.
    with tempfile.TemporaryDirectory() as temporary:
        peak = peak_rss_gb()
        record_plan = Plan(mode="test", out_dir=Path(temporary), model_run=Path(temporary), stages=())
        written = write_run_record(record_plan, 39, ["a", "b"], peak)
        record = json.loads(written.read_text(encoding="utf-8"))
        reportable = sys.platform.startswith(("linux", "win32", "darwin"))
        check("13_merge_peak_rss_probe_and_run_record",
              (peak is None or (isinstance(peak, float) and peak > 0.0))
              and (peak is not None or not reportable)
              and written.name == MERGE_RECORD_NAME
              and record["peak_rss_gb_by_stage"] == {"merge": peak}
              and record["merged_pages"] == 39 and record["bundle_members"] == ["a", "b"],
              f"peak={peak} GB on {sys.platform}; record={written.name}")

    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"SELF-TEST FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"SELF-TEST PASSED: {len(results)}/{len(results)} deterministic tests")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--from-scratch", action="store_true",
                       help="Acquire, model, and analyse: the full production chain (the VM path)")
    modes.add_argument("--from-run", metavar="RUN_DIR", default=None,
                       help="Reuse a completed modeling run: analyse and merge only")
    modes.add_argument("--smoke", action="store_true",
                       help="Torch-free whole-chain check on the synthetic bundle")
    modes.add_argument("--self-test", action="store_true",
                       help="Deterministic wiring and merge tests, without running either layer")
    parser.add_argument("--output-dir", default=None,
                        help="Override the run directory (defaults to a timestamped ./results dir, "
                             "or to RUN_DIR under --from-run)")
    parser.add_argument("--enable-torch-candidates", action="store_true",
                        help="Opt the PyTorch candidates into the modeling roster under "
                             "--from-scratch. OFF by default: they are the memory-critical path "
                             "on the production VM, so an installed PyTorch alone must not enrol "
                             "them. Ignored by the modes that fit no models.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_tests()
    mode = "from-scratch" if args.from_scratch else "smoke" if args.smoke else "from-run"
    return run_chain(build_plan(mode, args.output_dir, args.from_run,
                                args.enable_torch_candidates))


if __name__ == "__main__":
    raise SystemExit(main())
