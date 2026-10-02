#!/usr/bin/env python3
"""Customer-cost and infrastructure figures for the Nature Communications manuscript.

Produces the manuscript figures:

  fig2_multiloc_graded_unserved   siting gate, graded by unserved energy
  fig_adder_attribution           two-level attribution of the customer adder
  fig_mitigation_levers           each lever against the factor it treats
  fig4a/fig4b_all2030_*           graded all-announced system status
  fig8_customer_price_and_rep_risk customer price channels and REP default risk

All quantities are read from solver outputs and the derived tables in the
legacy-layout view (results/nc_paper_figures/*.csv). The only external
numbers are the $5,000/MWh ERCOT high system-wide offer cap and the Texas2k
regulated transmission (TCOS) charge with its 4CP denominators, which are CLI
options documented in Methods:

  --tcos-flat            flat per-MWh TCOS charge (phi = 1), $/MWh
  --tcos-residential     residential charge (phi = 2), $/MWh
  --dc-share-4cp         data-centre share of the modelled ERCOT 4CP

Style follows nc_style: Okabe-Ito palette, no in-figure titles, Nature column
widths, PDF + 450-dpi PNG.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import nc_style
from .nc_style import (C, SINGLE_COL, DOUBLE_COL, CH_ENERGY, CH_CONGESTION,
                       CH_REGULATED, CH_BUFFER, CH_SCARCITY,
                       PW_RESOURCE, PW_COORDINATED)

DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[2] / "results" / "paper"
DEFAULT_LEGACY_ROOT = DEFAULT_RUN_ROOT / "analysis_view"
DEFAULT_REP_ROOT = DEFAULT_RUN_ROOT / "results" / "retailer"

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
MULTILOC_KEYS = ["year", "mode", "location", "scale_mw"]

# Load-shedding penalty price: config load_shedding = 5000 is $/kWh because the
# PyPSA-Eur load-shedding generator carries sign = 1e-3, so the effective value
# of lost load is $5e6/MWh. metrics.csv reports the penalty cost, so
# unserved MWh = load_shedding_cost / VOLL.
VOLL = 5.0e6
OFFER_CAP = 5000.0
N_SNAPSHOTS = 1464

# Regulated transmission-delivery adder ($/MWh) for the matched 2023 coupling.
# Texas2k engineering case (N-1 screening, recoverNEW): one-time $32.419B,
# CRF(8%, 40 yr) = 0.08386 -> $2.719B/yr, spread over the modelled ERCOT 4CP of
# 119,309 MW with the data-centre load (36.2 GW) in the denominator.
#
# 4CP allocates by contribution to the four summer coincident peaks but is
# recovered per unit of energy, so the per-MWh adder scales with a customer's
# coincident-peak-to-average ratio phi. The manuscript reports the residential
# customer, phi = 2.0; the flat value applies to the stylized REP, which
# serves a flat 1 GW load by construction and therefore sees phi = 1.
DEFAULT_TCOS_FLAT = 2.601
DEFAULT_TCOS_RESIDENTIAL = 5.203
PHI_RESIDENTIAL = 2.0
DEFAULT_DC_SHARE_4CP = 36.2 / 119.309


nc_style.apply()

PANEL_KW = dict(fontsize=9, fontweight="bold", va="bottom", ha="right")


def panel_label(ax, letter: str, dx: float = -0.10, dy: float = 1.02) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, **PANEL_KW)


def save(fig, name: str, results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(results_root / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {name}")


def pct(v: float) -> str:
    """Signed percentage label; negative shares carry their sign."""
    return f"{100 * v:+.0f}%" if v < 0 else f"{100 * v:.0f}%"


# ----------------------------------------------------------------------------
# data loaders
# ----------------------------------------------------------------------------


def load_multiloc(legacy_root: Path, run_root: Path | None = None) -> pd.DataFrame:
    """Representative-siting runs, all weather years and modes.

    Reads the legacy-layout view and completes it from
    results/siting/<year>/multiloc_* for any mode the view does not link
    (the view is a symlink layer over the same files, so rows are
    de-duplicated on year/mode/location/scale)."""
    patterns = [str(legacy_root / "results" / "dc_experiments*" / "*" / "multiloc_*" / "metrics.csv")]
    if run_root is not None:
        patterns.append(str(run_root / "results" / "siting" / "*" / "multiloc_*" / "metrics.csv"))
    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    if not files:
        raise FileNotFoundError(f"no multiloc metrics.csv under {patterns}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["scale_mw"] = df["scale_mw"].astype(int)
    df = df.drop_duplicates(subset=MULTILOC_KEYS, keep="first")
    df["unserved_MWh"] = df["load_shedding_cost"] / VOLL
    if "total_annual_investment" not in df.columns:
        df["total_annual_investment"] = np.nan
    df["invest_B"] = df["total_annual_investment"] / 1e9
    return df


def load_all2030(legacy_root: Path) -> pd.DataFrame:
    """All-announced runs. Excludes the full-year 2023 rerun, which uses a
    different study window and must not be mixed with the seasonal cases."""
    files = [
        f
        for f in glob.glob(str(legacy_root / "results" / "dc_experiments*" / "*" / "all2030_*" / "metrics.csv"))
        if "dc_experiments_fullyear" not in f
    ]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["invest_B"] = df["total_annual_investment"] / 1e9
    return df


def load_attribution(legacy_root: Path) -> pd.DataFrame:
    """Reference-price / congestion / scarcity split of the wholesale adder."""
    d = pd.read_csv(legacy_root / "results" / "nc_paper_figures" / "adder_factor_decomposition.csv")
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
    ).dropna(subset=["energy", "congestion"])
    att["adder_capped"] = att.energy + att.congestion
    att["adder_uncapped"] = att.adder_capped + att.scarcity
    att["congestion_share"] = att.congestion / att.adder_capped
    return att.sort_index()


def fmt_energy(mwh: float) -> str:
    """Human-readable unserved energy for cell annotation (three significant
    digits in fixed notation, so the label fits a heatmap cell)."""
    if mwh <= 1.0:
        return "served"
    if mwh < 1e3:
        return f"{mwh:.0f} MWh"
    if mwh < 1e6:
        return f"{mwh / 1e3:.3g} GWh"
    return f"{mwh / 1e6:.3g} TWh"


def _signed_stack(ax, x, pos, signed, width, pos_color, signed_color, net_marker=True,
                  pos_label=None, signed_label=None, signed_hatch="////"):
    """A positive bar from zero plus a signed second term: stacked on top when
    positive, drawn below zero as a hatched outline when negative. A short
    black tick marks the net (pos + signed) so the billed total stays legible
    either way. Returns the top of the drawn bars."""
    x = np.asarray(x, dtype=float)
    pos = np.asarray(pos, dtype=float)
    signed = np.asarray(signed, dtype=float)
    ax.bar(x, pos, width=width, color=pos_color, edgecolor="white", linewidth=0.5,
           label=pos_label, zorder=2)
    up = signed >= 0
    if up.any():
        ax.bar(x[up], signed[up], width=width, bottom=pos[up], color=signed_color,
               edgecolor="white", linewidth=0.5, label=signed_label, zorder=2)
    if (~up).any():
        ax.bar(x[~up], signed[~up], width=width, bottom=0.0, facecolor="white",
               edgecolor=signed_color, hatch=signed_hatch, linewidth=0.8,
               label=None if up.any() else signed_label, zorder=2)
    if net_marker:
        for xi, n in zip(x, pos + signed):
            ax.plot([xi - width * 0.55, xi + width * 0.55], [n, n], color="black",
                    lw=0.9, zorder=4, solid_capstyle="butt")
    ax.axhline(0, color=C["grey"], lw=0.5, zorder=1)
    return np.maximum(pos, pos + signed)


# ----------------------------------------------------------------------------
# Figure: graded siting gate (replaces the binary regime heatmap)
# ----------------------------------------------------------------------------


def fig_graded_unserved(ml: pd.DataFrame, results_root: Path) -> None:
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
        masked = np.ma.masked_less_equal(np.ma.masked_invalid(g), 1.0)
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
                if np.isnan(v):
                    ax.text(j, i, "no run", ha="center", va="center", fontsize=4.9,
                            color="#8A8A8A", style="italic")
                    continue
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
    save(fig, "fig2_multiloc_graded_unserved", results_root)


# ----------------------------------------------------------------------------
# Figure: two-level attribution of the customer adder
# ----------------------------------------------------------------------------


def fig_attribution(att: pd.DataFrame, results_root: Path, tcos_adder: float) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # The regulated adder is defined only for the matched 2023 Texas2k
    # coupling, so panel a decomposes that year rather than the five-year mean.
    m = att.loc[MATCHED_YEAR]
    e5, c5 = m.energy, m.congestion
    whole5 = e5 + c5

    # --- panel a: nested composition of what the customer pays -------------
    total = whole5 + tcos_adder
    ax_a.bar(0, whole5, width=0.62, color=CH_ENERGY, edgecolor="white", linewidth=0.5,
             label=f"Wholesale {pct(whole5 / total)}")
    ax_a.bar(0, tcos_adder, width=0.62, bottom=whole5, color=CH_REGULATED, edgecolor="white",
             linewidth=0.5, label=f"Transmission {pct(tcos_adder / total)}")
    # The right bar re-cuts the same wholesale block into energy and a signed
    # congestion term; a negative congestion term hangs below zero and the
    # black tick marks the net wholesale adder that reaches the bill.
    _signed_stack(ax_a, [1], [e5], [c5], 0.62, CH_ENERGY, CH_CONGESTION,
                  signed_label=f"of which congestion {pct(c5 / whole5)}")
    ax_a.text(1, e5 / 2, pct(e5 / whole5), ha="center", va="center",
              fontsize=5.8, color="white", fontweight="bold")

    ax_a.plot([0.31, 0.69], [whole5, whole5], color=C["grey"], lw=0.6, ls=":")
    ax_a.plot([0.31, 0.69], [0, 0], color=C["grey"], lw=0.6, ls=":")
    ax_a.text(0, total + total * 0.03, f"{total / 10:.2f} c/kWh", ha="center", fontsize=6.2,
              fontweight="bold")
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["Residential\nincrease", "Wholesale\nsplit"])
    ax_a.text(0.5, 1.03, f"matched year ({MATCHED_YEAR})", transform=ax_a.transAxes,
              ha="center", va="bottom", fontsize=5.8, color=C["grey"])
    ax_a.set_ylabel("Peak-season increase ($/MWh)")
    ax_a.set_xlim(-0.6, 1.6)
    ax_a.set_ylim(min(0.0, c5 * 1.6), max(total, e5) * 1.22)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=5.4,
                handlelength=0.9, handletextpad=0.4, columnspacing=0.9, borderpad=0.15,
                labelspacing=0.25, ncol=2)
    panel_label(ax_a, "a", dx=-0.26)

    # --- panel b: year-by-year split of the capped wholesale adder ---------
    x = np.arange(len(att.index))
    tops = _signed_stack(ax_b, x, att.energy, att.congestion, 0.66, CH_ENERGY, CH_CONGESTION,
                         pos_label="Marginal generation", signed_label="Congestion")
    span = tops.max() - min(0.0, att.congestion.min())
    for xi, top, share in zip(x, tops, att.congestion_share):
        ax_b.text(xi, top + span * 0.03, pct(share), ha="center", fontsize=5.8,
                  color=CH_CONGESTION, fontweight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([str(y) for y in att.index])
    ax_b.set_xlabel("Weather year")
    ax_b.set_ylabel("Wholesale adder ($/MWh)")
    ax_b.set_ylim(min(0.0, att.congestion.min() * 1.6), tops.max() * 1.38)
    handles, labels = ax_b.get_legend_handles_labels()
    handles.append(Line2D([], [], color="black", lw=0.9))
    labels.append("Net adder billed")
    ax_b.legend(handles, labels, loc="upper left", frameon=False, fontsize=5.8,
                handlelength=1.1, labelspacing=0.25, borderpad=0.1)
    panel_label(ax_b, "b", dx=-0.22)

    # --- panel c: what the offer cap removes -------------------------------
    ax_c.bar(x, att.adder_capped, width=0.66, color=CH_ENERGY, edgecolor="white", linewidth=0.5,
             label="Reaches the customer bill")
    ax_c.bar(x, att.scarcity, width=0.66, bottom=att.adder_capped, color="white",
             edgecolor=CH_SCARCITY, linewidth=0.9, hatch="///",
             label="Suppressed by the $5,000/MWh cap")
    for xi, (_, r) in zip(x, att.iterrows()):
        ax_c.text(xi, r.adder_uncapped + att.adder_uncapped.max() * 0.03,
                  f"{int(r.snaps_over_cap)}", ha="center",
                  fontsize=6, color=CH_SCARCITY, fontweight="bold")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(y) for y in att.index])
    ax_c.set_xlabel("Weather year")
    ax_c.set_ylabel("Wholesale adder ($/MWh)")
    ax_c.set_ylim(0, att.adder_uncapped.max() * 1.44)
    ax_c.legend(loc="upper left", frameon=False, fontsize=5.8, handlelength=1.4,
                labelspacing=0.25, borderpad=0.1)
    panel_label(ax_c, "c", dx=-0.22)

    save(fig, "fig_adder_attribution", results_root)


# ----------------------------------------------------------------------------
# Figure: mitigation levers, each against the factor it treats
# ----------------------------------------------------------------------------


def fig_mitigation(att: pd.DataFrame, ml: pd.DataFrame, graded: pd.DataFrame,
                   results_root: Path, tcos_adder: float, dc_share_4cp: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.35))
    ax_a, ax_b, ax_c = axes
    fig.subplots_adjust(wspace=0.42)

    # --- panel a: how many hours the flexibility lever has to cover --------
    cases = ["dispatch", "gen_storage", "full_tx"]
    case_labels = ["No new\ninfrastructure", "Generation\n+ storage", "Full\nexpansion"]
    colors = [C["grey"], PW_RESOURCE, PW_COORDINATED]
    years = sorted(graded.year.unique())
    xs = np.arange(len(cases))
    for k, y in enumerate(years):
        vals = []
        for case in cases:
            sel = graded[(graded.year == y) & (graded.case == case)]
            vals.append(sel["snaps_any_bus_over_cap"].iloc[0] if len(sel) else np.nan)
        ax_a.scatter(xs + (k - (len(years) - 1) / 2) * 0.11, vals, s=13, color=colors, zorder=3,
                     edgecolor="white", linewidth=0.3)
    for i, case in enumerate(cases):
        v = graded[graded.case == case]["snaps_any_bus_over_cap"]
        ax_a.plot([i - 0.3, i + 0.3], [v.mean(), v.mean()], color=colors[i], lw=1.4, zorder=2)
    ax_a.set_yscale("log")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(case_labels)
    ax_a.set_ylabel(f"Intervals with a bus above\n$5,000/MWh (of {N_SNAPSHOTS:,})")
    ax_a.set_ylim(0.6, 3000)
    ax_a.axhline(N_SNAPSHOTS, color=C["grey"], lw=0.6, ls=":")
    ax_a.text(2.42, N_SNAPSHOTS * 1.13, "all intervals", fontsize=5.4, color=C["grey"], ha="right")
    fs = graded[graded.case == "full_tx"]["snaps_any_bus_over_cap"]
    ax_a.annotate(f"{int(fs.min())}–{int(fs.max())}", xy=(2, fs.mean()), xytext=(2, 12),
                  ha="center", fontsize=6, color=C["blue"], fontweight="bold",
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=C["blue"]))
    panel_label(ax_a, "a", dx=-0.30)

    # --- panel b: siting, cost against residual unserved energy ------------
    sub = ml[ml.scale_mw == 5000]
    order = ["AUSTIN", "HOUSTON", "DFW_SOUTH", "TOLAR", "ABILENE", "CHILDRESS"]
    xb = np.arange(len(order))
    gs_inv, tx_inv, gs_shed, tx_shed = [], [], [], []
    for loc in order:
        gs = sub[(sub.location == loc) & (sub["mode"] == "generation_storage")]
        tx = sub[(sub.location == loc) & (sub["mode"] == "generation_tx")]
        gs_inv.append(gs["invest_B"].mean())
        tx_inv.append(tx["invest_B"].mean())
        gs_shed.append(gs["unserved_MWh"].mean())
        tx_shed.append(tx["unserved_MWh"].mean())

    ax_b.bar(xb - 0.19, gs_inv, width=0.36, color=PW_RESOURCE, edgecolor="white", linewidth=0.5,
             label="Generation + storage")
    ax_b.bar(xb + 0.19, tx_inv, width=0.36, color=PW_COORDINATED, edgecolor="white", linewidth=0.5,
             label="Generation + transmission")
    for xi, (gi, sh) in enumerate(zip(gs_inv, gs_shed)):
        if sh > 1.0:
            ax_b.text(xi - 0.19, gi * 1.35, "✗", ha="center", fontsize=8, color=CH_REGULATED,
                      fontweight="bold")
    for xi, (ti, sh) in enumerate(zip(tx_inv, tx_shed)):
        if sh > 1.0:
            ax_b.text(xi + 0.19, ti * 1.35, "✗", ha="center", fontsize=8, color=CH_REGULATED,
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
    tcos_excl = tcos_adder / (1 - dc_share_4cp)
    labels = ["Data centers in\npeak-demand base", "Data centers\nexcluded"]
    tvals = [tcos_adder, tcos_excl]
    xc = np.arange(2)
    ax_c.bar(xc, [whole5, whole5], width=0.5, color=CH_ENERGY, edgecolor="white", linewidth=0.5,
             label="Wholesale (unchanged)")
    ax_c.bar(xc, tvals, width=0.5, bottom=[whole5, whole5], color=CH_REGULATED, edgecolor="white",
             linewidth=0.5, label="Regulated transmission")
    for xi, tv in zip(xc, tvals):
        ax_c.text(xi, whole5 + tv + (whole5 + tcos_excl) * 0.03, f"{(whole5 + tv) / 10:.2f} c/kWh",
                  ha="center", fontsize=6, fontweight="bold")
        ax_c.text(xi, whole5 + tv / 2, f"{tv:.2f}", ha="center", va="center", fontsize=5.6,
                  color="white", fontweight="bold")
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylabel("Customer adder ($/MWh)")
    ax_c.set_ylim(0, (whole5 + tcos_excl) * 1.22)
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False, fontsize=5.6,
                handlelength=1.1, labelspacing=0.25)
    panel_label(ax_c, "c", dx=-0.30)

    save(fig, "fig_mitigation_levers", results_root)


# ----------------------------------------------------------------------------
# Figure: graded all-announced system status (replaces binary served/unserved)
# ----------------------------------------------------------------------------


def fig_all2030_graded(graded: pd.DataFrame, results_root: Path) -> None:
    """Two label-free single-panel images, composed as panels a and b in LaTeX.
    Cases absent from the graded table (for example while a system case is
    being re-solved) are left out rather than drawn as empty slots."""
    fig_a, ax_a = plt.subplots(figsize=(SINGLE_COL, 2.30))
    fig_b, ax_b = plt.subplots(figsize=(SINGLE_COL, 2.30))

    all_cases = ["dispatch", "gen_storage", "gen_tx", "full_tx"]
    all_labels = {"dispatch": "No new\ninfrastructure", "gen_storage": "Generation\n+ storage",
                  "gen_tx": "Generation\n+ transmission", "full_tx": "Full\nexpansion"}
    all_colors = {"dispatch": C["grey"], "gen_storage": PW_RESOURCE, "gen_tx": C["sky"],
                  "full_tx": PW_COORDINATED}
    present = set(graded.case.unique())
    cases = [c for c in all_cases if c in present]
    missing = [c for c in all_cases if c not in present]
    if missing:
        print(f"  fig4: cases missing from the graded table, not drawn: {missing}")
    labels = [all_labels[c] for c in cases]
    colors = [all_colors[c] for c in cases]
    years = sorted(graded.year.unique())
    width = 0.80 / len(years)
    xs = np.arange(len(cases))
    floor = 2e-5

    for k, y in enumerate(years):
        vals, snaps = [], []
        for case in cases:
            sel = graded[(graded.year == y) & (graded.case == case)]
            vals.append(max(sel["unserved_pct_of_load"].iloc[0], 0.0) if len(sel) else np.nan)
            snaps.append(sel["snaps_any_bus_over_cap"].iloc[0] if len(sel) else np.nan)
        off = (k - (len(years) - 1) / 2) * width
        for i, (v, s) in enumerate(zip(vals, snaps)):
            if np.isnan(v) or np.isnan(s):
                continue
            ax_a.bar(xs[i] + off, max(v, floor), width=width * 0.92, bottom=floor,
                     color=colors[i], edgecolor="white", linewidth=0.25)
            ax_b.bar(xs[i] + off, max(s, 0.3), width=width * 0.92, bottom=0.3,
                     color=colors[i], edgecolor="white", linewidth=0.25)

    ax_a.set_yscale("log")
    ax_a.set_ylim(floor, 60)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("Unserved energy\n(% of served load)")
    tx_cases = [i for i, c in enumerate(cases) if c in ("gen_tx", "full_tx")]
    if tx_cases:
        # Sits just above the stub bars of the transmission cases, which serve
        # the load in every weather year.
        ax_a.text(tx_cases[-1] + 0.42, 1.1e-4, "fully served,\nevery weather year", fontsize=5.2,
                  color=C["blue"], ha="right", va="bottom", style="italic", linespacing=1.2)

    ax_b.set_yscale("log")
    ax_b.set_ylim(0.3, 4000)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel(f"Intervals with a bus above\n$5,000/MWh (of {N_SNAPSHOTS:,})")
    ax_b.axhline(N_SNAPSHOTS, color=C["grey"], lw=0.6, ls=":")
    ax_b.text(len(cases) - 0.55, N_SNAPSHOTS * 1.2, "all intervals", fontsize=5.4,
              color=C["grey"], ha="right")

    yspan = f"{years[0]}–{years[-1]}" if years == list(range(years[0], years[-1] + 1)) \
        else ", ".join(str(y) for y in years)
    handles = [Patch(facecolor=C["grey"], label=f"weather years {yspan}")]
    ax_a.legend(handles=handles, loc="upper right", frameon=False, fontsize=5.4,
                handlelength=1.0, borderpad=0.1, borderaxespad=0.3)

    save(fig_a, "fig4a_all2030_unserved_graded", results_root)
    save(fig_b, "fig4b_all2030_capped_intervals", results_root)


# ----------------------------------------------------------------------------
# Figure: customer price channels and REP risk, on the capped basis
# ----------------------------------------------------------------------------


def fig_customer_rep(att: pd.DataFrame, rep_dir: Path, results_root: Path,
                     tcos_adder: float, tcos_flat: float, cap: float = OFFER_CAP) -> None:
    def rep(zone: str, forward: str) -> pd.DataFrame:
        return pd.read_csv(rep_dir / f"rep_default_sim_summary_{zone}_{forward}.csv")

    sysrep = rep("system", "scenario_repriced")
    ftx = sysrep[(sysrep["mode"] == "full_tx") & (sysrep.scenario == "all2030")].iloc[0]
    gst = sysrep[(sysrep["mode"] == "generation_storage") & (sysrep.scenario == "all2030")].iloc[0]

    energy_adder = att.loc[MATCHED_YEAR, "adder_capped"]
    annual_energy_MWh = 1000.0 * 8760.0
    # Solvency buffer: the cash a retailer must hold so that the 5th-percentile
    # minimum-cash path stays non-negative. Zero when that path is positive.
    buf_ftx = max(0.0, -ftx["p05_min_cash_$"])
    buf_gst = max(0.0, -gst["p05_min_cash_$"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.35))
    fig.subplots_adjust(wspace=0.34)

    # --- panel a: customer-price stack, cents/kWh --------------------------
    e_c = energy_adder / 10.0
    t_c = tcos_adder / 10.0
    if buf_ftx > 0:
        recoveries = [None, 1, 3, 5]
        xlabels = ["Before\nbuffer", "1-year\nrecovery", "3-year\nrecovery", "5-year\nrecovery"]
    else:
        recoveries = [None]
        xlabels = ["Residential\nincrease"]
    x = np.arange(len(recoveries))
    ymax = e_c + t_c
    for xi, Y in zip(x, recoveries):
        r_c = 0.0 if Y is None else (buf_ftx / (annual_energy_MWh * Y)) / 10.0
        ax_a.bar(xi, e_c, width=0.6, color=CH_ENERGY, edgecolor="white", linewidth=0.5,
                 label="Wholesale" if xi == 0 else None)
        ax_a.bar(xi, t_c, width=0.6, bottom=e_c, color=CH_REGULATED, edgecolor="white",
                 linewidth=0.5, label="Regulated transmission" if xi == 0 else None)
        if r_c > 0:
            ax_a.bar(xi, r_c, width=0.6, bottom=e_c + t_c, color=CH_BUFFER, edgecolor="white",
                     linewidth=0.5, label="Retailer buffer" if xi == 1 else None)
        ymax = max(ymax, e_c + t_c + r_c)
        ax_a.text(xi, e_c + t_c + r_c + 0.035, f"{e_c + t_c + r_c:.2f}", ha="center",
                  fontsize=6.2, fontweight="bold")
    if buf_ftx <= 0:
        ax_a.set_xlim(-0.8, 1.9)
        ax_a.text(0.55, (e_c + t_c) * 0.5,
                  "no retailer buffer premium:\nthe 5th-percentile minimum\ncash path stays positive\n"
                  f"(${ftx['p05_min_cash_$'] / 1e6:.1f}M)",
                  fontsize=5.6, color=C["dgrey"], ha="left", va="center", linespacing=1.3)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabels)
    ax_a.set_ylabel("Incremental residential price\n(cents/kWh, peak season)")
    ax_a.set_ylim(0, ymax * 1.30)
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
    pmax = max(max(ftx_p), max(gst_p), 1e-6)
    ax_b.bar(xb - 0.19, ftx_p, width=0.36, color=PW_COORDINATED, edgecolor="white", linewidth=0.5,
             label="Full expansion")
    ax_b.bar(xb + 0.19, gst_p, width=0.36, color=PW_RESOURCE, edgecolor="white", linewidth=0.5,
             label="Generation + storage")
    for xi, (a, b) in enumerate(zip(ftx_p, gst_p)):
        ax_b.text(xi - 0.19, a + pmax * 0.03, f"{a:.1f}", ha="center", fontsize=5.8)
        ax_b.text(xi + 0.19, b + pmax * 0.03, f"{b:.1f}", ha="center", fontsize=5.8)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(zlabels)
    ax_b.set_ylabel("Simulated retailer default\nprobability (%)")
    ax_b.set_ylim(0, pmax * 1.30)
    ax_b.legend(loc="upper left", frameon=False, fontsize=5.8, handlelength=1.1,
                labelspacing=0.25, borderpad=0.1)
    panel_label(ax_b, "b", dx=-0.22)

    save(fig, "fig8_customer_price_and_rep_risk", results_root)

    print(f"\n--- customer price / REP cross-check (cap {cap:,.0f}) ---")
    print(f"  REP summaries: {rep_dir}")
    if buf_ftx > 0:
        for Y in [1, 3, 5]:
            r_c = (buf_ftx / (annual_energy_MWh * Y)) / 10.0
            print(f"  {Y}-year recovery: REP adder {r_c:.4f} c/kWh -> total {e_c + t_c + r_c:.4f} c/kWh")
    else:
        print("  system-level 95th-percentile buffer is zero: no buffer premium bars drawn")
    print(f"  before REP risk        : {e_c + t_c:.4f} c/kWh "
          f"(wholesale {e_c:.4f} + TCOS {t_c:.4f} at phi={PHI_RESIDENTIAL})")
    print(f"  buffer full_tx ${buf_ftx / 1e6:.2f}M  vs  gen+storage ${buf_gst / 1e6:.2f}M")
    print(f"  gen+storage 1-year premium: {(buf_gst / annual_energy_MWh) / 10.0:.4f} c/kWh")
    print(f"  default probability system: full_tx {ftx_p[0]:.2f}% vs gen+storage {gst_p[0]:.2f}%")
    saving = gst["mean_procurement_$/MWh"] - ftx["mean_procurement_$/MWh"]
    # The REP's own load is flat, so its TCOS charge is the phi=1 value; the
    # residential charge is quoted alongside because the manuscript uses both.
    print(f"  procurement full_tx {ftx['mean_procurement_$/MWh']:.3f} vs "
          f"gen+storage {gst['mean_procurement_$/MWh']:.3f} (saving {saving:.3f} = "
          f"{saving / tcos_flat:.2f}x the flat TCOS charge, "
          f"{saving / tcos_adder:.2f}x the residential charge)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-root", "--input-root", dest="legacy_root", type=Path,
                        default=DEFAULT_LEGACY_ROOT,
                        help="Legacy-layout view holding results/dc_experiments*/ and "
                             "results/nc_paper_figures/ tables.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT,
                        help="Experiment run root; results/siting/ completes the multiloc runs.")
    parser.add_argument("--rep-root", type=Path, default=DEFAULT_REP_ROOT,
                        help="Directory with rep_default_sim_summary_<zone>_<forward>.csv.")
    parser.add_argument("--tcos-flat", type=float, default=DEFAULT_TCOS_FLAT,
                        help="Flat TCOS charge, phi = 1 ($/MWh).")
    parser.add_argument("--tcos-residential", type=float, default=DEFAULT_TCOS_RESIDENTIAL,
                        help="Residential TCOS charge, phi = 2 ($/MWh).")
    parser.add_argument("--dc-share-4cp", type=float, default=DEFAULT_DC_SHARE_4CP,
                        help="Data-centre share of the modelled ERCOT 4CP (fraction).")
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Directory for the generated figure files.")
    args = parser.parse_args()
    for name in ("legacy_root", "run_root", "rep_root", "results_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main() -> None:
    args = parse_args()
    ml = load_multiloc(args.legacy_root, args.run_root)
    att = load_attribution(args.legacy_root)
    graded = pd.read_csv(args.legacy_root / "results" / "nc_paper_figures" /
                         "graded_unserved_all2030.csv")

    fig_graded_unserved(ml, args.results_root)
    fig_attribution(att, args.results_root, tcos_adder=args.tcos_residential)
    fig_mitigation(att, ml, graded, args.results_root, tcos_adder=args.tcos_residential,
                   dc_share_4cp=args.dc_share_4cp)
    fig_all2030_graded(graded, args.results_root)
    fig_customer_rep(att, args.rep_root, args.results_root,
                     tcos_adder=args.tcos_residential, tcos_flat=args.tcos_flat)

    # console cross-check of every number quoted in the manuscript
    m = att.mean()
    whole = m.energy + m.congestion
    print("\n--- attribution cross-check ---")
    print(f"years in decomposition: {list(att.index)}; multiloc years: "
          f"{sorted(ml.year.unique().tolist())}; graded years: {sorted(graded.year.unique().tolist())}")
    print(f"TCOS flat {args.tcos_flat} $/MWh, residential {args.tcos_residential} $/MWh, "
          f"DC 4CP share {args.dc_share_4cp:.4f}")
    print(f"{len(att)}yr mean capped wholesale adder : {whole:.4f} $/MWh")
    print(f"  energy      {m.energy:8.4f}  ({100 * m.energy / whole:.1f}%)")
    print(f"  congestion  {m.congestion:8.4f}  ({100 * m.congestion / whole:.1f}%)")
    print(f"  suppressed  {m.scarcity:8.4f}  (not in the bill)")
    m23 = att.loc[MATCHED_YEAR, "adder_capped"]
    tot = m23 + args.tcos_residential
    print(f"L1 (matched {MATCHED_YEAR}, residential phi={PHI_RESIDENTIAL}): "
          f"total {tot:.2f} $/MWh = {tot / 10:.2f} c/kWh -> "
          f"wholesale {100 * m23 / tot:.1f}% / regulated {100 * args.tcos_residential / tot:.1f}%")
    print("congestion share by year: "
          + ", ".join(f"{y}: {100 * s:+.0f}%" for y, s in att.congestion_share.items()))


if __name__ == "__main__":
    main()
