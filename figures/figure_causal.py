"""figure_causal.py - render the target-trial-emulation (``tte_*``) CSVs as ONE figure.

run_tte.py writes the causal deliverables as CSV/JSON only
(``tte_marginal_effects``, ``tte_rct_benchmark``, ``tte_evalue``, ``tte_c_for_benefit``,
``tte_weights_summary``, ``tte_covariate_balance_love``). Those are hard to read off the
VM, so this module turns them into a single main figure that travels back as a PNG/PDF/SVG
like every other figure.

Figure C1 (3 rows x 2 cols):
  A  BMI continuous ATE (kg/m^2) forest, 0 = null, "RYGB lower" to the left.
  B  HbA1c continuous ATE (%-points) forest.
  C  Threshold / composite risk differences - P(BMI<35), P(HbA1c<5.7), composite complication.
  D  RCT benchmark - emulated effect vs randomized anchor (overlap = design validated).
  E  E-value per contrast (how strong unmeasured confounding would need to be).
  F  Design-validity readout - PS overlap AUC, IPTW/IPCW ESS, c-for-benefit, worst |SMD|.

Real run:   OMP_NUM_THREADS=1 python -m figures.figure_causal --eval-dir runs/frozen/<ts>/evaluation
Demo:       python -m figures.figure_causal --demo --out /tmp/figc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe on a display-less VM (incl. Windows); before pyplot import

from . import style as S


# --------------------------------------------------------------------------- #
# Loading (tolerant: a missing CSV becomes an empty frame, panel self-skips)
# --------------------------------------------------------------------------- #
_CAUSAL_FILES = {
    "marginal": "tte_marginal_effects.csv",
    "rct": "tte_rct_benchmark.csv",
    "evalue": "tte_evalue.csv",
    "cfb": "tte_c_for_benefit.csv",
    "love": "tte_covariate_balance_love.csv",
    "weights": "tte_weights_summary.json",
}


def load_causal(eval_dir: str | Path) -> dict:
    eval_dir = Path(eval_dir)
    out: dict = {}
    for key, fname in _CAUSAL_FILES.items():
        p = eval_dir / fname
        if not p.exists():
            out[key] = {} if fname.endswith(".json") else pd.DataFrame()
            continue
        out[key] = json.loads(p.read_text()) if fname.endswith(".json") else pd.read_csv(p)
    return out


def demo_causal() -> dict:
    """Representative (ILLUSTRATIVE) numbers grounded in the trajectory/GBM figures already
    seen: BMI ATE ~ -4 kg/m^2, HbA1c ATE ~ -0.2 %-pts, RYGB more likely to reach threshold."""
    marginal = pd.DataFrame([
        dict(outcome="bmi_12m", group="bmi", horizon="bmi_12m", estimand="continuous_ate",
             threshold=np.nan, ate=-3.5, se=0.26, ci_lo=-4.01, ci_hi=-2.99, n_observed=3128, exploratory=False),
        dict(outcome="bmi_2y", group="bmi", horizon="bmi_2y", estimand="continuous_ate",
             threshold=np.nan, ate=-4.3, se=0.30, ci_lo=-4.89, ci_hi=-3.71, n_observed=2443, exploratory=False),
        dict(outcome="hba1c_12m", group="hba1c", horizon="hba1c_12m", estimand="continuous_ate",
             threshold=np.nan, ate=-0.21, se=0.03, ci_lo=-0.27, ci_hi=-0.15, n_observed=1357, exploratory=False),
        dict(outcome="hba1c_2y", group="hba1c", horizon="hba1c_2y", estimand="continuous_ate",
             threshold=np.nan, ate=-0.19, se=0.04, ci_lo=-0.27, ci_hi=-0.11, n_observed=1015, exploratory=False),
        dict(outcome="bmi_12m", group="bmi", horizon="bmi_12m", estimand="threshold_rd",
             threshold=35.0, ate=0.17, se=0.025, ci_lo=0.121, ci_hi=0.219, n_observed=3128, exploratory=False),
        dict(outcome="hba1c_12m", group="hba1c", horizon="hba1c_12m", estimand="threshold_rd",
             threshold=5.7, ate=0.06, se=0.02, ci_lo=0.021, ci_hi=0.099, n_observed=1357, exploratory=False),
        dict(outcome="composite_complication", group="composite", horizon="composite_complication",
             estimand="composite_rd", threshold=np.nan, ate=-0.014, se=0.008, ci_lo=-0.030, ci_hi=0.002,
             n_observed=4305, exploratory=True),
    ])
    rct = pd.DataFrame([
        dict(anchor="twl_pct_1_2y", source_horizon="bmi_2y", emulated_estimate=9.6,
             emulated_ci_lo=8.3, emulated_ci_hi=10.9, rct_point=5.0, rct_ci_lo=3.0, rct_ci_hi=7.0,
             overlaps_rct_ci=False, comparison_valid=True),
        dict(anchor="t2d_remission", source_horizon="hba1c_12m", emulated_estimate=1.38,
             emulated_ci_lo=1.16, emulated_ci_hi=1.63, rct_point=1.4, rct_ci_lo=1.15, rct_ci_hi=1.70,
             overlaps_rct_ci=True, comparison_valid=False),
    ])
    evalue = pd.DataFrame([
        dict(contrast="bmi_12m", estimand="continuous_ate", rr=1.61, e_point=2.59, e_bound=2.10),
        dict(contrast="bmi_2y", estimand="continuous_ate", rr=1.74, e_point=2.87, e_bound=2.42),
        dict(contrast="hba1c_12m", estimand="continuous_ate", rr=1.33, e_point=1.99, e_bound=1.52),
        dict(contrast="bmi_12m", estimand="threshold_rd", rr=1.45, e_point=2.27, e_bound=1.71),
        dict(contrast="composite_complication", estimand="composite_rd", rr=0.84, e_point=1.66, e_bound=1.00),
    ])
    cfb = pd.DataFrame([
        dict(outcome="bmi_12m", group="bmi", c_for_benefit=0.58, n_pairs=1540),
        dict(outcome="hba1c_12m", group="hba1c", c_for_benefit=0.55, n_pairs=1120),
    ])
    rng = np.random.default_rng(0)
    covs = ["age", "sex_male", "bmi_at_surgery", "hba1c_at_surgery", "creatinine", "eGFR",
            "insulin", "biguanide", "SGLT2", "DM2", "hypertension", "OSA", "dyslipidemia",
            "MI", "stroke", "AFib", "VTE", "svi_overall", "svi_ses", "svi_minority",
            "RUCA", "coverage_class", "surgery_year", "svi_household"]
    before = np.abs(rng.normal(0.0, 0.14, len(covs))) + 0.02
    after = before * rng.uniform(0.15, 0.5, len(covs))
    love = pd.DataFrame(dict(covariate=covs, smd_before=before, smd_after=after,
                             abs_smd_before=before, abs_smd_after=after,
                             balanced_after=after < 0.1))
    weights = {"iptw": {"ess": 3906.0, "n_kept": 4287, "n_trimmed": 18, "trim": [0.02, 0.98]},
               "ipcw_ess_by_horizon": {"bmi_12m": 3402.0, "bmi_2y": 1980.0, "hba1c_12m": 1240.0},
               "ps_model": {"backend": "xgboost", "auc": 0.68, "n_features": 24}}
    return dict(marginal=marginal, rct=rct, evalue=evalue, cfb=cfb, love=love, weights=weights, demo=True)


# --------------------------------------------------------------------------- #
# Small plotting helpers
# --------------------------------------------------------------------------- #
def _empty(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color=S.MUTED, style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)


def _forest(ax, labels, est, lo, hi, *, unit, title, null=0.0, better="lower",
            exploratory=None, color=S.ACCENT):
    """Horizontal forest: one row per contrast, point estimate + 95% CI, null reference."""
    est = np.asarray(est, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    n = len(labels)
    if n == 0 or not np.isfinite(est).any():
        _empty(ax, "no rows in CSV"); ax.set_title(title, loc="left"); return
    y = np.arange(n)[::-1]
    exploratory = [False] * n if exploratory is None else list(exploratory)
    ax.axvline(null, color=S.BASELINE, lw=1.1, zorder=1)
    for yi, e, l, h, expl in zip(y, est, lo, hi, exploratory):
        if not np.isfinite(e):
            continue
        c = S.MUTED if expl else color
        ax.plot([l, h], [yi, yi], color=c, lw=2.0, solid_capstyle="round", zorder=3,
                alpha=0.5 if expl else 1.0)
        ax.plot([e], [yi], marker="o", ms=6.5, color=c, markeredgecolor=S.SURFACE,
                markeredgewidth=1.2, zorder=4)
        ax.annotate(f"{e:+.2f} ({l:+.2f}, {h:+.2f})", (e, yi),
                    xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.6, color=S.INK_SECONDARY)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.9, (n - 1) + 0.9)
    ax.set_xlabel(unit, fontsize=7.8)
    ax.set_title(title, loc="left")
    S.style_axis(ax, xgrid=True, ygrid=False)
    # direction hint on the "good" side of the null
    lo_x, hi_x = ax.get_xlim()
    pad = 0.02 * (hi_x - lo_x)
    if better == "lower":
        ax.annotate("RYGB lower", (null - pad, n - 0.35), ha="right", va="center",
                    fontsize=6.8, color=S.GOOD, style="italic")
    else:
        ax.annotate("RYGB more likely", (null + pad, n - 0.35), ha="left", va="center",
                    fontsize=6.8, color=S.GOOD, style="italic")


def _build_fig(data: dict, stem: Path) -> list[Path]:
    S.apply_rcparams()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    demo = data.get("demo", False)
    fig = plt.figure(figsize=(11.0, 12.2))
    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.32,
                  left=0.10, right=0.97, top=0.92, bottom=0.06)
    ax_bmi = fig.add_subplot(gs[0, 0])
    ax_hba = fig.add_subplot(gs[0, 1])
    ax_rd = fig.add_subplot(gs[1, 0])
    ax_rct = fig.add_subplot(gs[1, 1])
    ax_ev = fig.add_subplot(gs[2, 0])
    ax_dv = fig.add_subplot(gs[2, 1])

    m = data["marginal"]

    # ---- A: BMI continuous ATE ----
    b = m[(m.get("estimand") == "continuous_ate") & (m.get("group") == "bmi")] if len(m) else m
    _forest(ax_bmi, list(b["horizon"]) if len(b) else [], b.get("ate", []),
            b.get("ci_lo", []), b.get("ci_hi", []),
            unit="AIPW ATE, BMI (kg/m$^2$)  -  RYGB minus sleeve",
            title="A  Weight effect (doubly-robust ATE)", better="lower", color=S.ACCENT)

    # ---- B: HbA1c continuous ATE ----
    h = m[(m.get("estimand") == "continuous_ate") & (m.get("group") == "hba1c")] if len(m) else m
    _forest(ax_hba, list(h["horizon"]) if len(h) else [], h.get("ate", []),
            h.get("ci_lo", []), h.get("ci_hi", []),
            unit="AIPW ATE, HbA1c (%-points)  -  RYGB minus sleeve",
            title="B  Glycemia effect (doubly-robust ATE)", better="lower", color=S.ACCENT)

    # ---- C: threshold + composite risk differences ----
    rd = m[m.get("estimand").isin(["threshold_rd", "composite_rd"])] if len(m) else m
    lab = []
    for _, r in rd.iterrows():
        if r["estimand"] == "composite_rd":
            lab.append("composite complication")
        else:
            thr = r.get("threshold")
            lab.append(f"P({r['group'].upper()} < {thr:g}) @ {r['horizon'].split('_')[-1]}")
    _forest(ax_rd, lab, rd.get("ate", []), rd.get("ci_lo", []), rd.get("ci_hi", []),
            unit="AIPW risk difference (probability)  -  RYGB minus sleeve",
            title="C  Threshold attainment & complication risk difference",
            better="higher", exploratory=list(rd.get("exploratory", [])) if len(rd) else None,
            color=S.CONTRAST)

    # ---- D: RCT benchmark ----
    _plot_rct(ax_rct, data["rct"])

    # ---- E: E-value ----
    _plot_evalue(ax_ev, data["evalue"])

    # ---- F: design validity readout ----
    _plot_design(ax_dv, data)

    title = "Target-trial emulation: RYGB vs sleeve causal effects (held-out test set)"
    if demo:
        title += "   -   ILLUSTRATIVE DEMO (synthetic numbers)"
    fig.suptitle(title, x=0.02, ha="left", fontsize=12.5, weight="semibold")
    cap = ("Doubly-robust AIPW (twin = outcome model; XGBoost propensity + IPCW). Negative ATE = RYGB lower; "
           "positive RD = RYGB more likely to reach the target. Composite complication (grey) is EXPLORATORY - "
           "no randomized anchor. Unmeasured confounders absent by construction: GERD, surgeon/center, smoking.")
    if demo:
        cap = "DEMO - numbers are illustrative placeholders, NOT a real run.  " + cap
    S.caption(fig, cap, y=0.035, color=S.CRITICAL if demo else S.INK_SECONDARY, size=7.0)
    return S.save_figure(fig, stem)


def _plot_rct(ax, rct: pd.DataFrame) -> None:
    """Each anchor on ONE shared, unit-free axis: the emulated effect expressed as
    distance from the trial point in half-CI-widths, so the trial 95% CI is always the
    shaded [-1, 1] band regardless of whether the anchor is %TWL or a remission RR.
    Emulated inside the band (green) = the observational design reproduced the trial."""
    if not len(rct):
        _empty(ax, "no tte_rct_benchmark.csv"); ax.set_title("D  RCT benchmark", loc="left"); return
    rows = list(rct.iterrows())
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-1, 1, color=S.MUTED, alpha=0.16, zorder=1, label="trial 95% CI")
    ax.axvline(0, color=S.BASELINE, lw=1.1, zorder=2, label="trial point estimate")
    for yi, (_, r) in zip(y, rows):
        half = (r["rct_ci_hi"] - r["rct_ci_lo"]) / 2.0
        if not np.isfinite(half) or half <= 0:
            continue
        z = lambda v: (v - r["rct_point"]) / half
        e, lo, hi = z(r["emulated_estimate"]), z(r["emulated_ci_lo"]), z(r["emulated_ci_hi"])
        ok = bool(r.get("overlaps_rct_ci", False))
        c = S.GOOD if ok else S.CRITICAL
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot([e], [yi], marker="o", ms=6.5, color=c, markeredgecolor=S.SURFACE,
                markeredgewidth=1.2, zorder=4)
        ax.annotate("overlaps RCT" if ok else "outside RCT", (e, yi), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.6, color=c)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['anchor']}\n({r['source_horizon']})" for _, r in rows], fontsize=7.0)
    ax.set_ylim(-0.9, (len(rows) - 1) + 0.9)
    ax.set_xlabel("emulated effect vs trial  (0 = trial point; shaded = trial 95% CI, in half-CI widths)",
                  fontsize=7.2)
    ax.set_title("D  Randomized-trial benchmark (in the band = design validated)", loc="left")
    ax.legend(fontsize=6.6, loc="upper right")
    S.style_axis(ax, xgrid=True, ygrid=False)


def _plot_evalue(ax, ev: pd.DataFrame) -> None:
    if not len(ev):
        _empty(ax, "no tte_evalue.csv"); ax.set_title("E  E-value", loc="left"); return
    lab = [f"{r['contrast']} ({r['estimand'].replace('_', ' ')})" for _, r in ev.iterrows()]
    y = np.arange(len(ev))[::-1]
    expl = ev["estimand"].eq("composite_rd").tolist() if "estimand" in ev else [False] * len(ev)
    for yi, (_, r), e in zip(y, ev.iterrows(), expl):
        c = S.MUTED if e else S.ACCENT
        ax.barh(yi, r["e_point"], height=0.6, color=c, alpha=0.85, zorder=3)
        if np.isfinite(r.get("e_bound", np.nan)):
            ax.plot([r["e_bound"], r["e_bound"]], [yi - 0.3, yi + 0.3], color=S.INK, lw=1.3, zorder=4)
        ax.annotate(f"{r['e_point']:.2f}", (r["e_point"], yi), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=6.8, color=S.INK_SECONDARY)
    ax.axvline(1.0, color=S.BASELINE, lw=1.1)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7.0)
    ax.set_ylim(-0.7, len(ev) - 0.3)
    ax.set_xlabel("E-value (point = bar; | = CI-limit E-value)", fontsize=7.4)
    ax.set_title("E  Sensitivity to unmeasured confounding (higher = more robust)", loc="left")
    S.style_axis(ax, xgrid=True, ygrid=False)


def _plot_design(ax, data: dict) -> None:
    w = data.get("weights", {}) or {}
    love = data.get("love", pd.DataFrame())
    cfb = data.get("cfb", pd.DataFrame())
    ps = (w.get("ps_model") or {}).get("auc", np.nan)
    iptw = (w.get("iptw") or {})
    ess, nkept, ntrim = iptw.get("ess", np.nan), iptw.get("n_kept", np.nan), iptw.get("n_trimmed", np.nan)
    worst_after = float(np.nanmax(love["abs_smd_after"])) if len(love) else np.nan
    n_imbal = int((love["abs_smd_after"] >= 0.1).sum()) if len(love) else 0
    ipcw = w.get("ipcw_ess_by_horizon", {}) or {}

    lines = [
        ("Propensity model AUC", f"{ps:.2f}" if np.isfinite(ps) else "NA",
         "overlap/positivity - near 0.5 = arms hard to tell apart (good for exchangeability)"),
        ("IPTW effective N", f"{ess:,.0f}" if np.isfinite(ess) else "NA",
         f"kept {nkept}, trimmed {ntrim} outside common support" if np.isfinite(nkept) else ""),
        ("Worst abs. SMD after weighting", f"{worst_after:.3f}" if np.isfinite(worst_after) else "NA",
         f"{n_imbal} covariate(s) still abs SMD >= 0.1 (target: 0)"),
    ]
    for _, r in cfb.iterrows():
        lines.append((f"c-for-benefit ({r['group']})", f"{r['c_for_benefit']:.2f}",
                      f"ranks who benefits; 0.5 = none  (n_pairs={int(r['n_pairs'])})"))
    if ipcw:
        rng = f"{min(ipcw.values()):,.0f} - {max(ipcw.values()):,.0f}"
        lines.append(("IPCW effective N (by horizon)", rng, "shrinks with follow-up attrition"))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("F  Design-validity readout", loc="left")
    y = 0.90
    for name, val, note in lines:
        ax.text(0.02, y, name, fontsize=8.0, color=S.INK, va="top", weight="semibold")
        ax.text(0.62, y, val, fontsize=9.0, color=S.ACCENT, va="top", weight="semibold")
        ax.text(0.02, y - 0.052, note, fontsize=6.6, color=S.MUTED, va="top")
        y -= 0.145


# --------------------------------------------------------------------------- #
# Public entrypoints
# --------------------------------------------------------------------------- #
def build(art, stem: Path, *, demo: bool = False) -> list[Path]:
    """build_all hook: render Figure C1 from ``art.eval_dir`` (or demo data)."""
    data = demo_causal() if demo else load_causal(art.eval_dir)
    if not demo and not len(data.get("marginal", [])):
        return []                       # no causal artifacts in this run -> skip cleanly
    return _build_fig(data, Path(stem))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=None, help="a frozen run's evaluation/ dir with tte_*.csv")
    ap.add_argument("--out", default=None, help="output stem dir (default alongside eval-dir)")
    ap.add_argument("--demo", action="store_true", help="render with illustrative synthetic numbers")
    args = ap.parse_args()
    if not args.demo and not args.eval_dir:
        raise SystemExit("Pass --eval-dir <run>/evaluation, or --demo.")
    data = demo_causal() if args.demo else load_causal(args.eval_dir)
    out = Path(args.out) if args.out else Path(args.eval_dir).parent / "figures" / "main"
    out.mkdir(parents=True, exist_ok=True)
    written = _build_fig(data, out / "figC1_causal_effects")
    print("wrote:", ", ".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
