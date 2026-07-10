"""Shared figure style for the W6 journal (Lancet-family) figures.

ONE place that owns: (a) vector-first matplotlib rcParams (embedded fonts, text kept
as text in SVG, TrueType in PDF), (b) the colorblind-safe palette (one blue accent +
one orange contrast + neutral grays, validated with the dataviz palette validator:
blue #2a78d6 / orange #eb6834 adjacent CVD deltaE ~97, well past the >=12 target),
(c) ``save_figure`` which writes every figure as BOTH .pdf and .svg (the deliverable)
plus a .png preview, and (d) the ONE visual language for the calibration caveat, so a
non-calibrated horizon looks the same (open marker + hatched band) in every figure.

Print/journal figures render on white/paper, so this module commits to the light
chart surface from the dataviz reference palette (a print figure has no theme toggle;
dark-mode stepping is a web-dashboard concern and is intentionally not applied here).

Every other figure module imports this and nothing style-related is defined elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Palette (dataviz reference instance, light surface). One accent + one contrast
# + status + neutral inks. Assigned by role, never cycled.
# --------------------------------------------------------------------------- #
SURFACE = "#ffffff"          # journal figures print on white paper
ACCENT = "#2a78d6"           # blue  - primary accent / "series 1" / RYGB / factual
CONTRAST = "#eb6834"         # orange - "series 2" / sleeve / counterfactual (CVD-safe vs blue)
AQUA = "#1baf7a"             # third categorical slot, only if a 3rd series is unavoidable
GOOD = "#0ca30c"             # status: good direction
CRITICAL = "#d03b3b"         # status: caution / not-trustworthy accents (with a label, never alone)

INK = "#0b0b0b"              # primary text / observed points
INK_SECONDARY = "#52514e"    # secondary text
MUTED = "#898781"            # axis labels / ticks / reference lines
GRID = "#e1e0d9"             # hairline gridlines
BASELINE = "#c3c2b7"         # axis / zero baselines

# Semantic role aliases the builders read by name (color follows the entity).
FACTUAL = ACCENT
COUNTERFACTUAL = CONTRAST
RYGB = ACCENT
SLEEVE = CONTRAST

CALIBRATION_CAUTION = (
    "Hatched band / open markers = calibration-dependent horizon (flow PIT not calibrated); "
    "intervals and threshold probabilities here are not yet trustworthy."
)


def apply_rcparams() -> None:
    """Install the vector-first, embedded-font rcParams. Idempotent."""
    mpl.rcParams.update({
        # -- vector output: editable text, no rasterization of vectors --
        "pdf.fonttype": 42,        # embed TrueType (editable text, not Type-3 outlines)
        "ps.fonttype": 42,
        "svg.fonttype": "none",    # keep SVG text as <text>, not paths (searchable, small)
        "figure.dpi": 150,
        "savefig.dpi": 200,        # only affects the .png preview; vectors are resolution-free
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        # -- type: one sans family, consistent scale --
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.titlesize": 11,
        "figure.titleweight": "semibold",
        # -- recessive chrome: no top/right spines, hairline grid, muted ticks --
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "hatch.linewidth": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.2,
    })


def style_axis(ax, *, xgrid: bool = False, ygrid: bool = True) -> None:
    """Apply the recessive-chrome look to a single axis (hairline y-grid only)."""
    if ygrid:
        ax.grid(visible=True, axis="y", color=GRID, linewidth=0.7, zorder=0)
    else:
        ax.grid(visible=False, axis="y")
    if xgrid:
        ax.grid(visible=True, axis="x", color=GRID, linewidth=0.7, zorder=0)
    else:
        ax.grid(visible=False, axis="x")
    ax.tick_params(length=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_figure(fig, stem: Path | str, *, formats: Sequence[str] = ("pdf", "svg", "png"),
                close: bool = True) -> list[Path]:
    """Write ``fig`` to ``<stem>.pdf``, ``<stem>.svg`` and a ``<stem>.png`` preview.

    ``stem`` is a path WITHOUT extension. Returns the written paths. PDF+SVG are the
    journal deliverable (true vector); PNG is a convenience raster preview.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path)
        written.append(path)
    if close:
        plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# The ONE calibration-caveat visual language, shared by every figure that plots a
# per-horizon quantity that inherits the flow's tail calibration (Fig 3/B, Fig 4,
# Fig A cols 4-5). Trustworthy == the horizon's flow PIT regime is 'calibrated'.
# --------------------------------------------------------------------------- #
def draw_calibrated_band(ax, x, lo, hi, trust, color, *, base_alpha: float = 0.16,
                         label: str | None = None) -> None:
    """Fill a predictive band: solid wash where calibrated, hatched where not."""
    x = np.asarray(x, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    trust = np.asarray(trust, bool)
    ax.fill_between(x, lo, hi, where=trust, interpolate=True, color=color,
                    alpha=base_alpha, linewidth=0, zorder=1, label=label)
    ax.fill_between(x, lo, hi, where=~trust, interpolate=True, facecolor=color,
                    alpha=0.06, hatch="////", edgecolor=color, linewidth=0, zorder=1)


def draw_calibrated_markers(ax, x, y, trust, color, *, size: float = 5.0,
                            zorder: int = 5, label: str | None = None) -> None:
    """Filled markers on calibrated horizons; open (white-filled) markers on
    calibration-dependent horizons. A 2px surface ring keeps them legible on lines."""
    x = np.asarray(x, float); y = np.asarray(y, float); trust = np.asarray(trust, bool)
    if trust.any():
        ax.plot(x[trust], y[trust], linestyle="none", marker="o", markersize=size,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=zorder, label=label)
    if (~trust).any():
        ax.plot(x[~trust], y[~trust], linestyle="none", marker="o", markersize=size,
                markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.4,
                zorder=zorder)


def calibration_legend_handles(color=MUTED):
    """Proxy handles explaining the trustworthy vs calibration-dependent marks."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color,
               markeredgecolor=SURFACE, markersize=6, label="calibrated horizon (trustworthy)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=SURFACE,
               markeredgecolor=color, markeredgewidth=1.4, markersize=6,
               label="calibration-dependent (not trustworthy)"),
        Patch(facecolor=color, alpha=0.06, hatch="////", edgecolor=color,
              label="calibration-dependent band"),
    ]


def caption(fig, text: str, *, y: float = -0.02, color=INK_SECONDARY, size: float = 7.0) -> None:
    """A small left-aligned footnote under the figure (used for the calibration caveat)."""
    fig.text(0.01, y, text, ha="left", va="top", fontsize=size, color=color, wrap=True)


def direct_label(ax, x, y, text, color, *, dx: float = 0.0, dy: float = 0.0,
                 ha: str = "left", va: str = "center", size: float = 7.5, weight="normal") -> None:
    """Selective direct label riding a mark (text in an ink token, not the series hue,
    unless the mark color is needed to disambiguate identity at the line end)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=size, color=color, weight=weight,
                annotation_clip=False, zorder=6)


# Month -> tick label used on every trajectory x-axis (years read cleaner past 12m).
def month_label(m: float) -> str:
    m = float(m)
    if m < 12:
        return f"{m:g}m"
    if abs(m / 12.0 - round(m / 12.0)) < 1e-6:
        return f"{round(m / 12.0):g}y"
    return f"{m:g}m"


def month_ticklabels(months: Iterable[float]) -> list[str]:
    return [month_label(m) for m in months]
