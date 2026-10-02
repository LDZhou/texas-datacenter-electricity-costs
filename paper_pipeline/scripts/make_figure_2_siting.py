#!/usr/bin/env python
"""Figure 2 for the NC submission: the representative-siting experiment.

The figure carries only the six-location experiment, whose two panels are
read as a pair:

  a  can the load be served at all, graded by unserved energy
     (load_shedding_cost / VOLL from each run's metrics.csv, five-year mean)
  b  once it is served, what it does to the system price, on the capped basis
     (analysis/siting_by_year.csv 'delta_ercot_capped_lmp_$/MWh', a fixed-
     weight ERCOT price with every bus price truncated at the $5,000/MWh offer
     cap, averaged over the weather years; cells in which any year leaves more
     than 1 MWh unserved are greyed)

Both panels share the same 6 locations x 4 load scales x 3 system responses
grid, so the columns line up and the reader compares them row by row.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from . import nc_style
from .nc_style import DOUBLE_COL

DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[2] / "results" / "paper"
DEFAULT_LEGACY_ROOT = DEFAULT_RUN_ROOT / "analysis_view"
DEFAULT_ANALYSIS_DIR = DEFAULT_RUN_ROOT / "results" / "analysis"

VOLL = 5.0e6
OFFER_CAP = 5000.0
SERVED_MWH = 1.0
SCALES = [500, 1000, 3000, 5000]
LOCATIONS = ["HOUSTON", "AUSTIN", "TOLAR", "DFW_SOUTH", "ABILENE", "CHILDRESS"]
LOC_LABELS = {
    "HOUSTON": "Houston", "AUSTIN": "Austin", "TOLAR": "Tolar",
    "DFW_SOUTH": "DFW South", "ABILENE": "Abilene", "CHILDRESS": "Childress",
}
MODES = ["dispatch", "generation_storage", "generation_tx"]
MODE_LABELS = {
    "dispatch": "No new infrastructure",
    "generation_storage": "Generation + storage",
    "generation_tx": "Generation + transmission",
}
KEYS = ["year", "mode", "location", "scale_mw"]

nc_style.apply()


def load_multiloc(legacy_root: Path, run_root: Path | None = None) -> pd.DataFrame:
    """Representative-siting metrics.csv rows from the legacy view, completed
    from results/siting/<year>/multiloc_* for any mode the view lacks. The
    view is a symlink layer over the same files, so duplicates are dropped."""
    patterns = [str(legacy_root / "results" / "dc_experiments*" / "*" / "multiloc_*" / "metrics.csv")]
    if run_root is not None:
        patterns.append(str(run_root / "results" / "siting" / "*" / "multiloc_*" / "metrics.csv"))
    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    if not files:
        raise FileNotFoundError(f"no multiloc metrics.csv under {patterns}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["scale_mw"] = df["scale_mw"].astype(int)
    df = df.drop_duplicates(subset=KEYS, keep="first")
    df["unserved_MWh"] = df["load_shedding_cost"] / VOLL
    return df


def fmt_energy(mwh: float) -> str:
    if mwh <= SERVED_MWH:
        return "served"
    if mwh < 1e3:
        return f"{mwh:.0f} MWh"
    if mwh < 1e6:
        return f"{mwh / 1e3:.3g} GWh"
    return f"{mwh / 1e6:.3g} TWh"


def unserved_grids(ml: pd.DataFrame) -> dict[str, np.ndarray]:
    grids = {}
    for mode in MODES:
        g = np.full((len(LOCATIONS), len(SCALES)), np.nan)
        for i, loc in enumerate(LOCATIONS):
            for j, sc in enumerate(SCALES):
                sel = ml[(ml.location == loc) & (ml.scale_mw == sc) & (ml["mode"] == mode)]
                if len(sel):
                    g[i, j] = sel["unserved_MWh"].mean()
        grids[mode] = g
    return grids


def capped_delta_grids(siting: pd.DataFrame, served_mwh: float = SERVED_MWH,
                       value_col: str = "delta_ercot_capped_lmp_$/MWh",
                       eue_col: str = "scenario_eue_mwh") -> dict[str, np.ndarray]:
    """Mean over years of the capped system-price increase per (mode, location,
    scale); NaN where any year leaves more than `served_mwh` unserved or where
    no row exists."""
    grids = {}
    siting = siting.assign(scale_mw=siting["scale_mw"].astype(int))
    for mode in MODES:
        g = np.full((len(LOCATIONS), len(SCALES)), np.nan)
        for i, loc in enumerate(LOCATIONS):
            for j, sc in enumerate(SCALES):
                sel = siting[(siting.location == loc) & (siting.scale_mw == sc)
                             & (siting["mode"] == mode)]
                if len(sel) and (sel[eue_col] <= served_mwh).all():
                    g[i, j] = float(sel[value_col].mean())
        grids[mode] = g
    return grids


def _style_cells(ax, xlabel: bool, ylabel: bool, title: str | None) -> None:
    ax.set_xticks(range(len(SCALES)))
    ax.set_xticklabels([f"{s / 1000:g}" for s in SCALES] if xlabel else [])
    ax.set_yticks(range(len(LOCATIONS)))
    ax.set_yticklabels([LOC_LABELS[l] for l in LOCATIONS] if ylabel else [])
    if xlabel:
        ax.set_xlabel("Data-center load (GW)", labelpad=2)
    if title:
        ax.set_title(title, fontsize=6.8, pad=4)
    ax.set_xticks(np.arange(-0.5, len(SCALES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LOCATIONS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_unserved(axes, cax, grids: dict[str, np.ndarray]) -> None:
    vmax = max(float(_finite(grids).max()), SERVED_MWH * 10)
    norm = LogNorm(vmin=SERVED_MWH, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#F2F2F2")

    for k, (ax, mode) in enumerate(zip(axes, MODES)):
        g = grids[mode]
        ax.imshow(np.ma.masked_less_equal(np.ma.masked_invalid(g), SERVED_MWH),
                  cmap=cmap, norm=norm, aspect="auto")
        _style_cells(ax, xlabel=False, ylabel=(k == 0), title=MODE_LABELS[mode])
        for i in range(len(LOCATIONS)):
            for j in range(len(SCALES)):
                v = g[i, j]
                if np.isnan(v):
                    ax.text(j, i, "no run", ha="center", va="center",
                            fontsize=4.9, color="#8A8A8A", style="italic")
                elif v > SERVED_MWH:
                    col = "white" if norm(v) > 0.62 else "black"
                    ax.text(j, i, fmt_energy(v), ha="center", va="center",
                            fontsize=4.9, color=col, fontweight="bold")
                else:
                    ax.text(j, i, "served", ha="center", va="center",
                            fontsize=4.9, color="#8A8A8A")

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Mean unserved energy\n(MWh, log scale)", fontsize=6.0, labelpad=3)
    cb.ax.tick_params(labelsize=5.6)


def _finite(grids: dict[str, np.ndarray]) -> np.ndarray:
    allv = np.concatenate([g.ravel() for g in grids.values()])
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        raise ValueError("every cell is masked; nothing to colour")
    return allv


def panel_dlmp(axes, cax, grids: dict[str, np.ndarray]) -> None:
    allv = _finite(grids)
    vmax = float(allv.max())
    vmin = min(0.0, float(allv.min()))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#DBDBDB")

    for k, (ax, mode) in enumerate(zip(axes, MODES)):
        g = grids[mode]
        ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        _style_cells(ax, xlabel=True, ylabel=(k == 0), title=None)
        for i in range(len(LOCATIONS)):
            for j in range(len(SCALES)):
                v = g[i, j]
                if np.isnan(v):
                    ax.text(j, i, "not served", ha="center", va="center",
                            fontsize=4.9, color="#6E6E6E", style="italic")
                else:
                    col = "white" if (v - vmin) > (vmax - vmin) * 0.62 else "black"
                    ax.text(j, i, f"{v:+.1f}" if v < 0 else f"{v:.1f}", ha="center",
                            va="center", fontsize=5.4, color=col)

    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), cax=cax)
    cb.set_label("Mean system price increase\n(\\$/MWh, prices truncated at \\$" + f"{OFFER_CAP:,.0f}/MWh)",
                 fontsize=6.0, labelpad=3)
    cb.ax.tick_params(labelsize=5.6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT,
                        help="Experiment run root holding results/siting/<year>/<case>/.")
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT,
                        help="Legacy-layout view (results/dc_experiments*/<year>/multiloc_*).")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR,
                        help="Directory holding siting_by_year.csv from summarize_revision.")
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Directory for the generated figure files.")
    parser.add_argument("--manuscript-output-root", type=Path, default=None,
                        help="Optional second directory receiving a copy of the figure.")
    args = parser.parse_args()
    for name in ("run_root", "legacy_root", "analysis_dir", "results_root",
                 "manuscript_output_root"):
        val = getattr(args, name)
        if val is not None:
            setattr(args, name, val.expanduser().resolve())
    return args


def main() -> None:
    args = parse_args()
    ml = load_multiloc(args.legacy_root, args.run_root)
    siting = pd.read_csv(args.analysis_dir / "siting_by_year.csv")
    grids_a = unserved_grids(ml)
    grids_b = capped_delta_grids(siting)

    fig = plt.figure(figsize=(DOUBLE_COL, 4.35))
    gs = fig.add_gridspec(
        2, 4, width_ratios=[1, 1, 1, 0.045], height_ratios=[1, 1],
        left=0.088, right=0.915, top=0.925, bottom=0.085,
        wspace=0.09, hspace=0.14,
    )
    row1 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    row2 = [fig.add_subplot(gs[1, i]) for i in range(3)]
    cax1 = fig.add_subplot(gs[0, 3])
    cax2 = fig.add_subplot(gs[1, 3])

    panel_unserved(row1, cax1, grids_a)
    panel_dlmp(row2, cax2, grids_b)

    row1[0].text(-0.33, 1.14, "a", transform=row1[0].transAxes, fontsize=8.5,
                 fontweight="bold", va="top", ha="left")
    row2[0].text(-0.33, 1.05, "b", transform=row2[0].transAxes, fontsize=8.5,
                 fontweight="bold", va="top", ha="left")

    outputs = [args.results_root] + ([args.manuscript_output_root]
                                     if args.manuscript_output_root else [])
    for d in outputs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig2_siting.pdf")
        fig.savefig(d / "fig2_siting.png", dpi=450)
    plt.close(fig)
    print("wrote fig2_siting.pdf/.png")
    print("--- figure 2 cross-check ---")
    print(f"panel a years per mode: "
          + ", ".join(f"{m}: {sorted(ml[ml['mode'] == m].year.unique().tolist())}" for m in MODES))
    print(f"panel b years: {sorted(siting.year.unique().tolist())}")
    for mode in MODES:
        g = grids_b[mode]
        shown = g[np.isfinite(g)]
        rng = (f"min {shown.min():+.2f}, max {shown.max():+.2f}" if shown.size
               else "no served cell")
        print(f"  {mode:20s} capped delta LMP $/MWh: {rng}, masked cells {int(np.isnan(g).sum())}")


if __name__ == "__main__":
    main()
