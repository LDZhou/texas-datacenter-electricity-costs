#!/usr/bin/env python
"""Figure 2 for the NC submission: the representative-siting experiment.

The old Figure 2 glued three unrelated images together - the all-announced
scenario map plus these two heatmaps. The map belongs with Figure 1 (it
describes the same load), so this figure now carries only the six-location
experiment, whose two panels are read as a pair:

  a  can the load be served at all, graded by unserved energy
  b  once it is served, what it does to the system price

Both panels share the same 6 locations x 4 load scales x 3 system responses
grid, so the columns line up and the reader compares them row by row.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "nc_paper_figures"
PAPER = ROOT / "Paper" / "nc_submission" / "figures"

from nc_style import DOUBLE_COL

VOLL = 5.0e6
YEARS = [2019, 2020, 2021, 2022, 2023]
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

import nc_style

nc_style.apply()


def load_multiloc() -> pd.DataFrame:
    from paper_inventory import metric_files
    files = metric_files('multiloc')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["unserved_MWh"] = df["load_shedding_cost"] / VOLL
    return df


def load_baseline_lmp() -> dict[int, float]:
    out = {}
    for y in YEARS:
        p = ROOT / "results" / "dc_experiments" / str(y) / "none_dispatch" / "metrics.csv"
        out[y] = float(pd.read_csv(p).iloc[0]["sys_mean_lmp"])
    return out


def fmt_energy(mwh: float) -> str:
    if mwh <= 1.0:
        return "served"
    if mwh < 1e3:
        return f"{mwh:.0f} MWh"
    if mwh < 1e6:
        return f"{mwh / 1e3:.3g} GWh"
    return f"{mwh / 1e6:.3g} TWh"


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


def panel_unserved(axes, cax, ml: pd.DataFrame) -> None:
    grids = {}
    for mode in MODES:
        g = np.full((len(LOCATIONS), len(SCALES)), np.nan)
        for i, loc in enumerate(LOCATIONS):
            for j, sc in enumerate(SCALES):
                sel = ml[(ml.location == loc) & (ml.scale_mw == sc) & (ml["mode"] == mode)]
                if len(sel):
                    g[i, j] = sel["unserved_MWh"].mean()
        grids[mode] = g

    vmax = max(np.nanmax(g) for g in grids.values())
    norm = LogNorm(vmin=1.0, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#F2F2F2")

    for k, (ax, mode) in enumerate(zip(axes, MODES)):
        g = grids[mode]
        ax.imshow(np.ma.masked_less_equal(g, 1.0), cmap=cmap, norm=norm, aspect="auto")
        _style_cells(ax, xlabel=False, ylabel=(k == 0), title=MODE_LABELS[mode])
        for i in range(len(LOCATIONS)):
            for j in range(len(SCALES)):
                v = g[i, j]
                if v > 1.0:
                    col = "white" if norm(v) > 0.62 else "black"
                    ax.text(j, i, fmt_energy(v), ha="center", va="center",
                            fontsize=4.9, color=col, fontweight="bold")
                else:
                    ax.text(j, i, "served", ha="center", va="center",
                            fontsize=4.9, color="#8A8A8A")

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Mean unserved energy\n(MWh, log scale)", fontsize=6.0, labelpad=3)
    cb.ax.tick_params(labelsize=5.6)


def panel_dlmp(axes, cax, ml: pd.DataFrame, base: dict[int, float]) -> None:
    grids = {}
    for mode in MODES:
        g = np.full((len(LOCATIONS), len(SCALES)), np.nan)
        for i, loc in enumerate(LOCATIONS):
            for j, sc in enumerate(SCALES):
                sel = ml[(ml.location == loc) & (ml.scale_mw == sc) & (ml["mode"] == mode)]
                if len(sel) and (sel["unserved_MWh"] <= 1.0).all():
                    g[i, j] = np.mean([
                        float(r["sys_mean_lmp"]) - base[int(r["year"])]
                        for _, r in sel.iterrows()
                    ])
        grids[mode] = g

    vmax = max(np.nanmax(g) for g in grids.values())
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#DBDBDB")

    for k, (ax, mode) in enumerate(zip(axes, MODES)):
        g = grids[mode]
        ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        _style_cells(ax, xlabel=True, ylabel=(k == 0), title=None)
        for i in range(len(LOCATIONS)):
            for j in range(len(SCALES)):
                v = g[i, j]
                if np.isnan(v):
                    ax.text(j, i, "not served", ha="center", va="center",
                            fontsize=4.9, color="#6E6E6E", style="italic")
                else:
                    col = "white" if v > vmax * 0.62 else "black"
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=5.4, color=col)

    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=cmap), cax=cax)
    cb.set_label("Mean system price increase\n($/MWh)", fontsize=6.0, labelpad=3)
    cb.ax.tick_params(labelsize=5.6)


def main() -> None:
    ml = load_multiloc()
    base = load_baseline_lmp()

    fig = plt.figure(figsize=(DOUBLE_COL, 4.35))
    gs = fig.add_gridspec(
        2, 4, width_ratios=[1, 1, 1, 0.045], height_ratios=[1, 1],
        left=0.088, right=0.925, top=0.925, bottom=0.085,
        wspace=0.09, hspace=0.14,
    )
    row1 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    row2 = [fig.add_subplot(gs[1, i]) for i in range(3)]
    cax1 = fig.add_subplot(gs[0, 3])
    cax2 = fig.add_subplot(gs[1, 3])

    panel_unserved(row1, cax1, ml)
    panel_dlmp(row2, cax2, ml, base)

    row1[0].text(-0.33, 1.14, "a", transform=row1[0].transAxes, fontsize=8.5,
                 fontweight="bold", va="top", ha="left")
    row2[0].text(-0.33, 1.05, "b", transform=row2[0].transAxes, fontsize=8.5,
                 fontweight="bold", va="top", ha="left")

    for d in (OUT, PAPER):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig2_siting.pdf")
        fig.savefig(d / "fig2_siting.png", dpi=450)
    plt.close(fig)
    print("wrote fig2_siting.pdf/.png")


if __name__ == "__main__":
    main()
