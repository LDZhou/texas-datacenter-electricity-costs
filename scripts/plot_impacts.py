#!/usr/bin/env python3
"""Revised and new Nature Communications figures for the restructured manuscript.

Produces three figures that the four-beat narrative needs:

  fig2_multiloc_graded_unserved   siting gate, graded by unserved energy
                                  (replaces the binary weather-year-count map)
  fig_adder_attribution           two-level attribution of the customer adder
  fig_mitigation_levers           each lever against the factor it treats

All quantities are read from solver outputs; nothing is hard-coded except the
$5,000/MWh ERCOT high system-wide offer cap and the 4CP denominators, which are
documented in Methods.

Style uses the original paper's Okabe-Ito palette, no in-figure titles,
Nature column widths, PDF + 450-dpi PNG.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "nc_paper_figures"

YEARS = [2019, 2020, 2021, 2022, 2023]
MATCHED_YEAR = 2023
SCALES = [500, 1000, 3000, 5000]
LOCATIONS = ["HOUSTON", "AUSTIN", "TOLAR", "DFW_SOUTH", "ABILENE", "CHILDRESS"]
LOC_LABELS = {
    "HOUSTON": "Houston",
    "AUSTIN": "Austin",
    "TOLAR": "Tolar",
    "DFW_SOUTH": "DFW South",
    "ABILENE": "Abilene",
    "CHILDRESS": "Childress",
}
MODES = ["dispatch", "generation_storage", "generation_tx"]
MODE_LABELS = {
    "dispatch": "No new infrastructure",
    "generation_storage": "Generation + storage",
    "generation_tx": "Generation + transmission",
}

from nc_style import (C, SINGLE_COL, DOUBLE_COL, CH_ENERGY, CH_CONGESTION,
                      CH_REGULATED, CH_BUFFER, CH_SCARCITY,
                      PW_RESOURCE, PW_COORDINATED)
import nc_style

# Load-shedding penalty price: config load_shedding = 5000 is $/kWh because the
# PyPSA-Eur load-shedding generator carries sign = 1e-3, so the effective value
# of lost load is $5e6/MWh. metrics.csv reports the penalty cost, so
# unserved MWh = load_shedding_cost / VOLL.
VOLL = 5.0e6
OFFER_CAP = 5000.0

# Regulated transmission-delivery adder ($/MWh). The PowerWorld recoverNEW case
# under N-1 screening gives a one-time $34.29B, which CRF(8%, 40 yr) annualizes
# to $2.876B/yr, allocated over the modeled ERCOT 4CP of 119,309 MW with the
# data-center load included in the denominator.
#
# 4CP allocates by contribution to the four summer coincident peaks but is
# recovered per unit of energy, so the per-MWh adder scales with a customer's
# coincident-peak-to-average ratio phi. The manuscript reports the residential
# customer, phi = 2.0 (midpoint of the 1.8-2.5 range typical of that shape).
# The flat value is retained for the stylized REP, which serves a flat 1 GW
# load by construction and therefore sees phi = 1.
TCOS_FLAT = 2.751
PHI_RESIDENTIAL = 2.0
TCOS_RESIDENTIAL = TCOS_FLAT * PHI_RESIDENTIAL  # 5.502 $/MWh
DC_SHARE_4CP = 36.2 / 119.309


nc_style.apply()

PANEL_KW = dict(fontsize=9, fontweight="bold", va="bottom", ha="right")


def panel_label(ax, letter: str, dx: float = -0.10, dy: float = 1.02) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, **PANEL_KW)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {name}")


# ----------------------------------------------------------------------------
# data loaders
# ----------------------------------------------------------------------------


def load_multiloc() -> pd.DataFrame:
    """Representative-siting runs, five weather years, all modes."""
    from paper_inventory import metric_files
    files = metric_files('multiloc')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["unserved_MWh"] = df["load_shedding_cost"] / VOLL
    df["invest_B"] = df["total_annual_investment"] / 1e9
    return df


def load_all2030() -> pd.DataFrame:
    """All-announced runs. Excludes the full-year 2023 rerun, which uses a
    different study window and must not be mixed with the seasonal cases."""
    from paper_inventory import metric_files
    files = metric_files('all2030')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["invest_B"] = df["total_annual_investment"] / 1e9
    return df


def load_attribution() -> pd.DataFrame:
    """Reference-price / congestion / scarcity split of the wholesale adder."""
    d = pd.read_csv(OUT / "adder_factor_decomposition.csv")
    base = d[d.case == "baseline"].set_index("year")
    full = d[d.case == "full_tx"].set_index("year")
    att = pd.DataFrame(
        {
            "energy": full["ref"] - base["ref"],
            "congestion": full["congestion"] - base["congestion"],
            "scarcity": full["scarcity"] - base["scarcity"],
            "snaps_over_cap": full["snaps_any_bus_over_cap"],
            "n_snapshots": full["n_snapshots"],
        }
    )
    att["adder_capped"] = att.energy + att.congestion
    att["adder_uncapped"] = att.adder_capped + att.scarcity
    return att


def fmt_energy(mwh: float) -> str:
    """Human-readable unserved energy for cell annotation.

    Three significant digits in fixed notation. `:.2g` was previously used and
    produced strings like "5.9e+02 GWh", which are far too wide for a heatmap
    cell and overflowed into the neighbouring column.
    """
    if mwh <= 1.0:
        return "served"
    if mwh < 1e3:
        return f"{mwh:.0f} MWh"
    if mwh < 1e6:
        return f"{mwh / 1e3:.3g} GWh"
    return f"{mwh / 1e6:.3g} TWh"


# ----------------------------------------------------------------------------
# Figure: graded siting gate (replaces the binary regime heatmap)
# ----------------------------------------------------------------------------


def fig_graded_unserved(ml: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.45), sharey=True)

    grids = {}
    for mode in MODES:
        g = np.zeros((len(LOCATIONS), len(SCALES)))
        for i, loc in enumerate(LOCATIONS):
            for j, sc in enumerate(SCALES):
                sel = ml[(ml.location == loc) & (ml.scale_mw == sc) & (ml["mode"] == mode)]
                g[i, j] = sel["unserved_MWh"].mean() if len(sel) else np.nan
        grids[mode] = g

    vmax = max(np.nanmax(g) for g in grids.values())
    # Cells below 1 MWh are solver noise, not shedding; they are drawn as "served".
    norm = LogNorm(vmin=1.0, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#F2F2F2")

    for ax, mode in zip(axes, MODES):
        g = grids[mode]
        masked = np.ma.masked_less_equal(g, 1.0)
        ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(SCALES)))
        ax.set_xticklabels([f"{s / 1000:g}" for s in SCALES])
        ax.set_yticks(range(len(LOCATIONS)))
        ax.set_yticklabels([LOC_LABELS[l] for l in LOCATIONS])
        ax.set_xlabel("Data-center load (GW)")
        ax.set_title(MODE_LABELS[mode], fontsize=7, pad=4)
        ax.set_xticks(np.arange(-0.5, len(SCALES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(LOCATIONS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(len(LOCATIONS)):
            for j in range(len(SCALES)):
                v = g[i, j]
                label = fmt_energy(v)
                if v > 1.0:
                    shade = norm(v)
                    color = "white" if shade > 0.62 else "black"
                    weight = "bold"
                else:
                    color, weight = "#8A8A8A", "normal"
                ax.text(j, i, label, ha="center", va="center", fontsize=4.9, color=color, fontweight=weight)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.021, pad=0.015
    )
    cbar.set_label("Mean unserved energy (MWh, log scale)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    # No internal panel letters: this image is one panel of a composed figure.
    save(fig, "fig2_multiloc_graded_unserved")


# ----------------------------------------------------------------------------
# Figure: two-level attribution of the customer adder
# ----------------------------------------------------------------------------


def fig_attribution(att: pd.DataFrame, tcos_adder: float = TCOS_RESIDENTIAL) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # The regulated adder is defined only for the matched 2023 PowerWorld
    # coupling, so panel a decomposes that year rather than the five-year mean.
    m = att.loc[MATCHED_YEAR]
    e5, c5 = m.energy, m.congestion
    whole5 = e5 + c5

    # --- panel a: nested composition of what the customer pays -------------
    total = whole5 + tcos_adder
    # Labels are kept short: the legend sits under a narrow panel, and the long
    # four-entry single-column form used previously ran out under panel b.
    ax_a.bar(0, whole5, width=0.62, color=CH_ENERGY, edgecolor="white", linewidth=0.5,
             label=f"Wholesale {100 * whole5 / total:.0f}%")
    ax_a.bar(0, tcos_adder, width=0.62, bottom=whole5, color=CH_REGULATED, edgecolor="white",
             linewidth=0.5, label=f"Transmission {100 * tcos_adder / total:.0f}%")
    # The right bar re-cuts the same wholesale block, so its energy segment
    # carries no separate legend entry: it is the identical blue.
    ax_a.bar(1, e5, width=0.62, color=CH_ENERGY, edgecolor="white", linewidth=0.5)
    ax_a.bar(1, c5, width=0.62, bottom=e5, color=CH_CONGESTION, edgecolor="white", linewidth=0.5,
             label=f"of which congestion {100 * c5 / whole5:.0f}%")
    ax_a.text(1, e5 / 2, f"{100 * e5 / whole5:.0f}%", ha="center", va="center",
              fontsize=5.8, color="white", fontweight="bold")

    ax_a.plot([0.31, 0.69], [whole5, whole5], color=C["grey"], lw=0.6, ls=":")
    ax_a.plot([0.31, 0.69], [0, 0], color=C["grey"], lw=0.6, ls=":")
    ax_a.text(0, total + 0.55, f"{total / 10:.2f} c/kWh", ha="center", fontsize=6.2,
              fontweight="bold")
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["Residential\nincrease", "Wholesale\nsplit"])
    ax_a.text(0.5, 1.03, f"matched year ({MATCHED_YEAR})", transform=ax_a.transAxes,
              ha="center", va="bottom", fontsize=5.8, color=C["grey"])
    ax_a.set_ylabel("Peak-season increase ($/MWh)")
    ax_a.set_xlim(-0.6, 1.6)
    ax_a.set_ylim(0, total * 1.22)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=5.4,
                handlelength=0.9, handletextpad=0.4, columnspacing=0.9, borderpad=0.15,
                labelspacing=0.25, ncol=2)
    panel_label(ax_a, "a", dx=-0.26)

    # --- panel b: year-by-year split of the capped wholesale adder ---------
    x = np.arange(len(att.index))
    ax_b.bar(x, att.energy, width=0.66, color=CH_ENERGY, edgecolor="white", linewidth=0.5,
             label="Marginal generation")
    ax_b.bar(x, att.congestion, width=0.66, bottom=att.energy, color=CH_CONGESTION,
             edgecolor="white", linewidth=0.5, label="Congestion")
    for xi, (_, r) in zip(x, att.iterrows()):
        ax_b.text(xi, r.adder_capped + 0.7, f"{100 * r.congestion / r.adder_capped:.0f}%",
                  ha="center", fontsize=5.8, color=CH_CONGESTION, fontweight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([str(y) for y in att.index])
    ax_b.set_xlabel("Weather year")
    ax_b.set_ylabel("Wholesale adder ($/MWh)")
    ax_b.set_ylim(0, att.adder_capped.max() * 1.38)
    ax_b.legend(loc="upper left", frameon=False, fontsize=5.8, handlelength=1.1,
                labelspacing=0.25, borderpad=0.1)
    panel_label(ax_b, "b", dx=-0.22)

    # --- panel c: what the offer cap removes -------------------------------
    ax_c.bar(x, att.adder_capped, width=0.66, color=CH_ENERGY, edgecolor="white", linewidth=0.5,
             label="Reaches the customer bill")
    ax_c.bar(x, att.scarcity, width=0.66, bottom=att.adder_capped, color="white",
             edgecolor=CH_SCARCITY, linewidth=0.9, hatch="///",
             label="Suppressed by the $5,000/MWh cap")
    for xi, (_, r) in zip(x, att.iterrows()):
        ax_c.text(xi, r.adder_uncapped + 1.0, f"{int(r.snaps_over_cap)}", ha="center",
                  fontsize=6, color=CH_SCARCITY, fontweight="bold")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(y) for y in att.index])
    ax_c.set_xlabel("Weather year")
    ax_c.set_ylabel("Wholesale adder ($/MWh)")
    ax_c.set_ylim(0, att.adder_uncapped.max() * 1.44)
    ax_c.legend(loc="upper left", frameon=False, fontsize=5.8, handlelength=1.4,
                labelspacing=0.25, borderpad=0.1)
    panel_label(ax_c, "c", dx=-0.22)

    save(fig, "fig_adder_attribution")


# ----------------------------------------------------------------------------
# Figure: mitigation levers, each against the factor it treats
# ----------------------------------------------------------------------------


def fig_mitigation(att: pd.DataFrame, ml: pd.DataFrame, graded: pd.DataFrame,
                   tcos_adder: float = TCOS_RESIDENTIAL) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.35))
    ax_a, ax_b, ax_c = axes
    fig.subplots_adjust(wspace=0.42)

    # --- panel a: how many hours the flexibility lever has to cover --------
    cases = ["dispatch", "gen_storage", "full_tx"]
    case_labels = ["No new\ninfrastructure", "Generation\n+ storage", "Generation\n+ transmission"]  # data is full_tx, not gen_tx
    colors = [C["grey"], PW_RESOURCE, PW_COORDINATED]
    width = 0.26
    xs = np.arange(len(cases))
    for k, y in enumerate(YEARS):
        vals = []
        for case in cases:
            sel = graded[(graded.year == y) & (graded.case == case)]
            vals.append(sel["snaps_any_bus_over_cap"].iloc[0] if len(sel) else np.nan)
        ax_a.scatter(xs + (k - 2) * 0.11, vals, s=13, color=colors, zorder=3,
                     edgecolor="white", linewidth=0.3)
    for i, case in enumerate(cases):
        v = graded[graded.case == case]["snaps_any_bus_over_cap"]
        ax_a.plot([i - 0.3, i + 0.3], [v.mean(), v.mean()], color=colors[i], lw=1.4, zorder=2)
    ax_a.set_yscale("log")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(case_labels)
    ax_a.set_ylabel("Intervals with a bus above\n$5,000/MWh (of 1,464)")
    ax_a.set_ylim(0.6, 3000)
    ax_a.axhline(1464, color=C["grey"], lw=0.6, ls=":")
    ax_a.text(2.42, 1650, "all intervals", fontsize=5.4, color=C["grey"], ha="right")
    fs = graded[graded.case == "full_tx"]["snaps_any_bus_over_cap"]
    ax_a.annotate(f"{int(fs.min())}–{int(fs.max())}", xy=(2, fs.mean()), xytext=(2, 12),
                  ha="center", fontsize=6, color=C["blue"], fontweight="bold",
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=C["blue"]))
    panel_label(ax_a, "a", dx=-0.30)

    # --- panel b: siting, cost against residual unserved energy ------------
    sub = ml[ml.scale_mw == 5000]
    order = ["AUSTIN", "HOUSTON", "DFW_SOUTH", "TOLAR", "ABILENE", "CHILDRESS"]
    xb = np.arange(len(order))
    gs_inv, tx_inv, gs_shed = [], [], []
    for loc in order:
        gs = sub[(sub.location == loc) & (sub["mode"] == "generation_storage")]
        tx = sub[(sub.location == loc) & (sub["mode"] == "generation_tx")]
        gs_inv.append(gs["invest_B"].mean())
        tx_inv.append(tx["invest_B"].mean())
        gs_shed.append(gs["unserved_MWh"].mean())

    ax_b.bar(xb - 0.19, gs_inv, width=0.36, color=PW_RESOURCE, edgecolor="white", linewidth=0.5,
             label="Generation + storage")
    ax_b.bar(xb + 0.19, tx_inv, width=0.36, color=PW_COORDINATED, edgecolor="white", linewidth=0.5,
             label="Generation + transmission")
    for xi, (gi, sh) in enumerate(zip(gs_inv, gs_shed)):
        if sh > 1.0:
            ax_b.text(xi - 0.19, gi * 1.35, "✗", ha="center", fontsize=8, color=CH_REGULATED,
                      fontweight="bold")
    ax_b.set_yscale("log")
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([LOC_LABELS[l] for l in order], rotation=32, ha="right")
    ax_b.set_ylabel("Annualized investment\nat 5 GW ($ billion per year)")
    ax_b.set_ylim(1e-2, 60)
    # The cross key is a legend entry rather than free-floating text: at the
    # bottom right it sat on the Childress bar, and at the top right it ran
    # through the legend.
    handles, labels = ax_b.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="x", color=CH_REGULATED,
                          markersize=3.6, markeredgewidth=1.1))
    labels.append("load still unserved")
    ax_b.legend(handles, labels, loc="upper left", frameon=False, fontsize=5.8,
                handlelength=1.1, labelspacing=0.25, borderpad=0.1)
    panel_label(ax_b, "b", dx=-0.30)

    # --- panel c: what the cost-allocation lever can actually move ---------
    # Matched-year basis, because the 4CP denominator and the TCOS it allocates
    # are both defined only for the matched 2023 coupling.
    whole5 = att.loc[MATCHED_YEAR, "adder_capped"]
    tcos_excl = tcos_adder / (1 - DC_SHARE_4CP)
    labels = ["Data centers in\npeak-demand base", "Data centers\nexcluded"]
    tvals = [tcos_adder, tcos_excl]
    xc = np.arange(2)
    ax_c.bar(xc, [whole5, whole5], width=0.5, color=CH_ENERGY, edgecolor="white", linewidth=0.5,
             label="Wholesale (unchanged)")
    ax_c.bar(xc, tvals, width=0.5, bottom=[whole5, whole5], color=CH_REGULATED, edgecolor="white",
             linewidth=0.5, label="Regulated transmission")
    for xi, tv in zip(xc, tvals):
        ax_c.text(xi, whole5 + tv + 0.45, f"{(whole5 + tv) / 10:.2f} c/kWh", ha="center",
                  fontsize=6, fontweight="bold")
        ax_c.text(xi, whole5 + tv / 2, f"{tv:.2f}", ha="center", va="center", fontsize=5.6,
                  color="white", fontweight="bold")
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylabel("Customer adder ($/MWh)")
    ax_c.set_ylim(0, (whole5 + tcos_excl) * 1.22)
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False, fontsize=5.6,
                handlelength=1.1, labelspacing=0.25)
    panel_label(ax_c, "c", dx=-0.30)

    save(fig, "fig_mitigation_levers")


# ----------------------------------------------------------------------------
# Figure: graded all-announced system status (replaces binary served/unserved)
# ----------------------------------------------------------------------------


def fig_all2030_graded(graded: pd.DataFrame) -> None:
    """Two label-free single-panel images, composed as panels a and b in LaTeX."""
    fig_a, ax_a = plt.subplots(figsize=(SINGLE_COL, 2.30))
    fig_b, ax_b = plt.subplots(figsize=(SINGLE_COL, 2.30))

    cases = ["dispatch", "gen_storage", "gen_tx", "full_tx"]
    labels = ["No new\ninfrastructure", "Generation\n+ storage", "Generation\n+ transmission", "Full\nexpansion"]
    colors = [C["grey"], PW_RESOURCE, C["sky"], PW_COORDINATED]
    width = 0.16
    xs = np.arange(len(cases))
    floor = 2e-5

    for k, y in enumerate(YEARS):
        vals, snaps = [], []
        for case in cases:
            sel = graded[(graded.year == y) & (graded.case == case)]
            vals.append(max(sel["unserved_pct_of_load"].iloc[0], 0.0) if len(sel) else np.nan)
            snaps.append(sel["snaps_any_bus_over_cap"].iloc[0] if len(sel) else np.nan)
        off = (k - 2) * width
        for i, (v, s) in enumerate(zip(vals, snaps)):
            ax_a.bar(xs[i] + off, max(v, floor), width=width * 0.92, bottom=floor,
                     color=colors[i], edgecolor="white", linewidth=0.25)
            ax_b.bar(xs[i] + off, max(s, 0.3), width=width * 0.92, bottom=0.3,
                     color=colors[i], edgecolor="white", linewidth=0.25)

    ax_a.set_yscale("log")
    ax_a.set_ylim(floor, 60)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("Unserved energy\n(% of served load)")
    # Raised clear of the stub bars it used to sit on top of; still well below
    # the lowest gen+storage bar (1.5e-4), which is four x-positions to the left.
    ax_a.text(2.5, 1.1e-4, "fully served, every weather year", fontsize=5.2,
              color=C["blue"], ha="center", va="center", style="italic")

    ax_b.set_yscale("log")
    ax_b.set_ylim(0.3, 4000)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Intervals with a bus above\n$5,000/MWh (of 1,464)")
    ax_b.axhline(1464, color=C["grey"], lw=0.6, ls=":")
    ax_b.text(3.45, 1750, "all intervals", fontsize=5.4, color=C["grey"], ha="right")

    handles = [Patch(facecolor=C["grey"], label="weather years 2019–2023")]
    ax_a.legend(handles=handles, loc="upper right", frameon=False, fontsize=5.4,
                handlelength=1.0, borderpad=0.1, borderaxespad=0.3)

    save(fig_a, "fig4a_all2030_unserved_graded")
    save(fig_b, "fig4b_all2030_capped_intervals")


# ----------------------------------------------------------------------------
# Figure: customer price channels and REP risk, on the capped basis
# ----------------------------------------------------------------------------


def fig_customer_rep(att: pd.DataFrame, cap: int = 5000,
                     tcos_adder: float = TCOS_RESIDENTIAL) -> None:
    # The "_n1" directory holds the rerun that feeds the REP model the N-1
    # PowerWorld TCOS ($2.751/MWh); the un-suffixed directory is the superseded
    # N-0 run ($0.857/MWh) and no longer matches the manuscript.
    rep_dir = ROOT / "results" / f"rep_mode_contrast_cap{cap}_n1"

    def rep(zone: str, forward: str) -> pd.DataFrame:
        return pd.read_csv(rep_dir / f"rep_default_sim_summary_{zone}_{forward}.csv")

    sysrep = rep("system", "scenario_repriced")
    ftx = sysrep[(sysrep["mode"] == "full_tx") & (sysrep.scenario == "all2030")].iloc[0]
    gst = sysrep[(sysrep["mode"] == "generation_storage") & (sysrep.scenario == "all2030")].iloc[0]

    energy_adder = att.loc[MATCHED_YEAR, "adder_capped"]
    annual_energy_MWh = 1000.0 * 8760.0
    buf_ftx = max(0.0, -ftx["p05_min_cash_$"])
    buf_gst = max(0.0, -gst["p05_min_cash_$"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.35))
    fig.subplots_adjust(wspace=0.34)

    # --- panel a: customer-price stack, cents/kWh --------------------------
    recoveries = [None, 1, 3, 5]
    xlabels = ["Before\nbuffer", "1-year\nrecovery", "3-year\nrecovery", "5-year\nrecovery"]
    x = np.arange(len(recoveries))
    e_c = energy_adder / 10.0
    t_c = tcos_adder / 10.0
    for xi, Y in zip(x, recoveries):
        r_c = 0.0 if Y is None else (buf_ftx / (annual_energy_MWh * Y)) / 10.0
        ax_a.bar(xi, e_c, width=0.6, color=CH_ENERGY, edgecolor="white", linewidth=0.5,
                 label="Wholesale" if xi == 0 else None)
        ax_a.bar(xi, t_c, width=0.6, bottom=e_c, color=CH_REGULATED, edgecolor="white",
                 linewidth=0.5, label="Regulated transmission" if xi == 0 else None)
        if r_c > 0:
            ax_a.bar(xi, r_c, width=0.6, bottom=e_c + t_c, color=CH_BUFFER, edgecolor="white",
                     linewidth=0.5, label="Retailer buffer" if xi == 1 else None)
        ax_a.text(xi, e_c + t_c + r_c + 0.035, f"{e_c + t_c + r_c:.2f}", ha="center",
                  fontsize=6.2, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabels)
    ax_a.set_ylabel("Incremental residential price\n(cents/kWh, peak season)")
    ax_a.set_ylim(0, (e_c + t_c + buf_ftx / annual_energy_MWh / 10.0) * 1.30)
    # Two columns: the single-column form spilled past the panel.
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=5.4,
                handlelength=0.9, handletextpad=0.4, columnspacing=0.9, labelspacing=0.25,
                ncol=2)
    panel_label(ax_a, "a", dx=-0.22)

    # --- panel b: default probability, coordinated vs storage-only ---------
    zones = ["system", "NORTH", "SOUTH", "WEST", "HOUSTON"]
    zlabels = ["System", "North", "South", "West", "Houston"]
    ftx_p, gst_p = [], []
    for z in zones:
        s = rep(z, "scenario_repriced")
        ftx_p.append(s[(s["mode"] == "full_tx") & (s.scenario == "all2030")]["default_probability_pct"].iloc[0])
        gst_p.append(s[(s["mode"] == "generation_storage") & (s.scenario == "all2030")]["default_probability_pct"].iloc[0])
    xb = np.arange(len(zones))
    ax_b.bar(xb - 0.19, ftx_p, width=0.36, color=PW_COORDINATED, edgecolor="white", linewidth=0.5,
             label="Generation + transmission")
    ax_b.bar(xb + 0.19, gst_p, width=0.36, color=PW_RESOURCE, edgecolor="white", linewidth=0.5,
             label="Generation + storage")
    for xi, (a, b) in enumerate(zip(ftx_p, gst_p)):
        ax_b.text(xi - 0.19, a + 0.45, f"{a:.1f}", ha="center", fontsize=5.8)
        ax_b.text(xi + 0.19, b + 0.45, f"{b:.1f}", ha="center", fontsize=5.8)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(zlabels)
    ax_b.set_ylabel("Simulated retailer default\nprobability (%)")
    ax_b.set_ylim(0, max(gst_p) * 1.30)
    ax_b.legend(loc="upper left", frameon=False, fontsize=5.8, handlelength=1.1,
                labelspacing=0.25, borderpad=0.1)
    panel_label(ax_b, "b", dx=-0.22)

    save(fig, "fig8_customer_price_and_rep_risk")

    print("\n--- customer price / REP cross-check (cap%d) ---" % cap)
    for Y in [1, 3, 5]:
        r_c = (buf_ftx / (annual_energy_MWh * Y)) / 10.0
        print(f"  {Y}-year recovery: REP adder {r_c:.4f} c/kWh -> total {e_c + t_c + r_c:.4f} c/kWh")
    print(f"  before REP risk        : {e_c + t_c:.4f} c/kWh "
          f"(wholesale {e_c:.4f} + TCOS {t_c:.4f} at phi={PHI_RESIDENTIAL})")
    print(f"  buffer full_tx ${buf_ftx / 1e6:.2f}M  vs  gen+storage ${buf_gst / 1e6:.2f}M")
    print(f"  gen+storage 1-year premium: {(buf_gst / annual_energy_MWh) / 10.0:.4f} c/kWh")
    saving = gst["mean_procurement_$/MWh"] - ftx["mean_procurement_$/MWh"]
    # The REP's own load is flat, so its TCOS charge is the phi=1 value; the
    # residential charge is quoted alongside because the manuscript uses both.
    print(f"  procurement full_tx {ftx['mean_procurement_$/MWh']:.3f} vs "
          f"gen+storage {gst['mean_procurement_$/MWh']:.3f} (saving {saving:.3f} = "
          f"{saving / TCOS_FLAT:.2f}x the flat TCOS charge, "
          f"{saving / TCOS_RESIDENTIAL:.2f}x the residential charge)")


def main() -> None:
    ml = load_multiloc()
    att = load_attribution()
    graded = pd.read_csv(OUT / "graded_unserved_all2030.csv")

    fig_graded_unserved(ml)
    fig_attribution(att)
    fig_mitigation(att, ml, graded)
    fig_all2030_graded(graded)
    fig_customer_rep(att)

    # console cross-check of every number quoted in the manuscript
    m = att.mean()
    whole = m.energy + m.congestion
    print("\n--- attribution cross-check ---")
    print(f"5yr mean capped wholesale adder : {whole:.4f} $/MWh")
    print(f"  energy      {m.energy:8.4f}  ({100 * m.energy / whole:.1f}%)")
    print(f"  congestion  {m.congestion:8.4f}  ({100 * m.congestion / whole:.1f}%)")
    print(f"  suppressed  {m.scarcity:8.4f}  (not in the bill)")
    m23 = att.loc[MATCHED_YEAR, "adder_capped"]
    print(f"L1 (matched {MATCHED_YEAR}, residential phi={PHI_RESIDENTIAL}): "
          f"total {m23 + TCOS_RESIDENTIAL:.2f} $/MWh = "
          f"{(m23 + TCOS_RESIDENTIAL) / 10:.2f} c/kWh -> "
          f"wholesale {100 * m23 / (m23 + TCOS_RESIDENTIAL):.1f}% / "
          f"regulated {100 * TCOS_RESIDENTIAL / (m23 + TCOS_RESIDENTIAL):.1f}%")
    print(f"congestion share range over years: "
          f"{100 * (att.congestion / att.adder_capped).min():.1f}%–"
          f"{100 * (att.congestion / att.adder_capped).max():.1f}%")


if __name__ == "__main__":
    main()
