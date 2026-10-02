#!/usr/bin/env python
"""Figure 1 for the NC submission.

A data-carrying figure that follows the paper's argument:

  a  where it lands - the announced capacity across the ERCOT buses that
                      receive it, coloured by zone
  b  what arrives   - a near-constant slab on top of the existing demand curve
  c  what it costs  - annualized investment under the two expansion pathways
  d  what is left   - hours per season priced above the offer cap
  e  who pays       - the residential increase, split by channel

Every number shown is computed from the result files at run time:

  a, b  results/seasonal/2023/all2030_full_tx/{dc_bus_mapping.csv, network.nc}
        (panel b caches the season's load profiles as fig1_load_profiles.csv
        next to the figure when the legacy view has none)
  c     metrics.json 'total_annual_investment' of
        results/seasonal/<year>/all2030_{generation_storage,full_tx}
  d     graded_unserved_all2030.csv 'snaps_any_bus_over_cap' (x3 h), or,
        when that table is absent, ERCOT-bus marginal prices in network.nc
  e     adder_factor_decomposition.csv (energy = ref delta, congestion =
        congestion delta, matched year full_tx vs baseline) plus the
        regulated residential TCOS charge (--tcos-residential-usd-per-mwh)

Style follows nc_style: Okabe-Ito palette, 7 pt sans, no in-figure titles,
bold lowercase panel letters, 183 mm double-column width.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import nc_style
from .nc_style import (DOUBLE_COL, C, CH_ENERGY, CH_CONGESTION,
                       CH_REGULATED, PW_RESOURCE, PW_COORDINATED)

nc_style.apply()

DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[2] / "results" / "paper"
DEFAULT_LEGACY_ROOT = DEFAULT_RUN_ROOT / "analysis_view"
DEFAULT_STATE_BOUNDARIES = Path(__file__).resolve().parents[2] / "texas2k" / "inputs" / "geography" / "state_boundaries.geojson"

MATCHED_YEAR = 2023
OFFER_CAP = 5000.0
HOURS_PER_SNAPSHOT = 3.0
# Residential customer, phi = 2 (demand at the four system peaks twice its
# average), applied to the flat Texas2k N-1 TCOS charge; see Methods.
DEFAULT_TCOS_RESIDENTIAL = 5.203

ZONE_COLORS = {"NORTH": "#0072B2", "SOUTH": "#D55E00",
               "WEST": "#009E73", "HOUSTON": "#CC79A7"}
ZONE_LABELS = {"NORTH": "North", "SOUTH": "South",
               "WEST": "West", "HOUSTON": "Houston"}


# ----------------------------------------------------------------- data access
def matched_case_dir(run_root: Path, legacy_root: Path, case: str = "all2030_full_tx",
                     year: int = MATCHED_YEAR) -> Path:
    """Directory of one seasonal system case, preferring the new layout."""
    new = run_root / "results" / "seasonal" / str(year) / case
    if (new / "network.nc").exists():
        return new
    mode = case.split("_", 1)[1]
    legacy = legacy_root / "results" / f"dc_experiments_{mode}" / str(year) / case
    if (legacy / "network.nc").exists():
        return legacy
    raise FileNotFoundError(f"no network.nc for {year} {case} under {new} or {legacy}")


def load_investment(run_root: Path, cases=("all2030_generation_storage", "all2030_full_tx")
                    ) -> pd.DataFrame:
    """Annualized investment ($bn/yr) by year for each case; years where every
    case has a metrics.json. Columns are the case names."""
    seasonal = run_root / "results" / "seasonal"
    rows = {}
    for ydir in sorted(p for p in seasonal.glob("[0-9]" * 4) if p.is_dir()):
        vals = {}
        for case in cases:
            p = ydir / case / "metrics.json"
            if p.exists():
                with open(p) as fh:
                    vals[case] = float(json.load(fh)["total_annual_investment"]) / 1e9
        if len(vals) == len(cases):
            rows[int(ydir.name)] = vals
    if not rows:
        raise FileNotFoundError(f"no complete year with {cases} under {seasonal}")
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def capped_hours_from_graded(graded: pd.DataFrame, cases=("gen_storage", "full_tx"),
                             hours_per_snapshot: float = HOURS_PER_SNAPSHOT) -> pd.DataFrame:
    """Hours per season with any ERCOT bus at or above the cap, by year and
    case, from the graded_unserved table; years where every case is present."""
    piv = graded.pivot_table(index="year", columns="case",
                             values="snaps_any_bus_over_cap", aggfunc="first")
    missing = [c for c in cases if c not in piv.columns]
    if missing:
        raise KeyError(f"graded table lacks cases {missing}")
    piv = piv[list(cases)].dropna()
    if piv.empty:
        raise ValueError("no year has every case in the graded table")
    return piv.astype(float) * hours_per_snapshot


def capped_snapshots_from_network(path: Path, cap: float = OFFER_CAP) -> int:
    """Snapshots in which any ERCOT bus prices at or above the cap, read
    straight from network.nc (same rule as compute_graded_unserved)."""
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        lmp = ds["buses_t_marginal_price"].to_pandas()
        lmp.columns = ds["buses_t_marginal_price_i"].values
        if "buses_nerc_reg" in ds:
            reg = pd.Series(ds["buses_nerc_reg"].values.astype(str), index=ds["buses_i"].values)
            ercot = reg.index[reg.str.upper().str.contains("ERCOT")]
            cols = lmp.columns.intersection(ercot)
            if len(cols):
                lmp = lmp[cols]
    finally:
        ds.close()
    return int((lmp.max(axis=1) >= cap).sum())


def load_capped_hours(run_root: Path, legacy_root: Path,
                      hours_per_snapshot: float = HOURS_PER_SNAPSHOT) -> pd.DataFrame:
    graded_path = legacy_root / "results" / "nc_paper_figures" / "graded_unserved_all2030.csv"
    if graded_path.exists():
        return capped_hours_from_graded(pd.read_csv(graded_path),
                                        hours_per_snapshot=hours_per_snapshot)
    print(f"graded table missing ({graded_path}); counting capped snapshots from network.nc")
    seasonal = run_root / "results" / "seasonal"
    rows = {}
    for ydir in sorted(p for p in seasonal.glob("[0-9]" * 4) if p.is_dir()):
        gs = ydir / "all2030_generation_storage" / "network.nc"
        ft = ydir / "all2030_full_tx" / "network.nc"
        if gs.exists() and ft.exists():
            rows[int(ydir.name)] = {
                "gen_storage": capped_snapshots_from_network(gs) * hours_per_snapshot,
                "full_tx": capped_snapshots_from_network(ft) * hours_per_snapshot,
            }
    if not rows:
        raise FileNotFoundError(f"no year with both system networks under {seasonal}")
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def bill_components(decomp: pd.DataFrame, tcos_residential: float,
                    year: int = MATCHED_YEAR) -> dict[str, float]:
    """Residential increase in cents/kWh by channel for the matched year:
    energy = reference-price delta, congestion = congestion delta (full_tx
    minus baseline; may be negative), regulated = residential TCOS charge."""
    base = decomp[(decomp.case == "baseline") & (decomp.year == year)]
    full = decomp[(decomp.case == "full_tx") & (decomp.year == year)]
    if base.empty or full.empty:
        raise ValueError(f"adder decomposition lacks baseline/full_tx rows for {year}")
    b, f = base.iloc[0], full.iloc[0]
    return {
        "energy": float(f["ref"] - b["ref"]) / 10.0,
        "congestion": float(f["congestion"] - b["congestion"]) / 10.0,
        "regulated": float(tcos_residential) / 10.0,
    }


def load_profiles(legacy_root: Path, case_dir: Path, cache_dir: Path) -> pd.DataFrame:
    """Season load profiles (dc_mw, base_mw) by snapshot. Read the legacy CSV
    when present, otherwise rebuild from network.nc and cache next to the
    figure output."""
    for p in (legacy_root / "results" / "nc_paper_figures" / "fig1_load_profiles.csv",
              cache_dir / "fig1_load_profiles.csv"):
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)
    df = profiles_from_network(case_dir / "network.nc")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_dir / "fig1_load_profiles.csv")
    print(f"rebuilt fig1_load_profiles.csv from {case_dir / 'network.nc'}")
    return df


def profiles_from_network(path: Path) -> pd.DataFrame:
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        pset = ds["loads_t_p_set"].to_pandas()
        pset.columns = ds["loads_t_p_set_i"].values.astype(str)
        idx = pd.DatetimeIndex(ds["snapshots_timestep"].values
                               if "snapshots_timestep" in ds else ds["snapshots"].values)
    finally:
        ds.close()
    pset.index = idx
    is_dc = pset.columns.str.startswith("DC_")
    return pd.DataFrame({"dc_mw": pset.loc[:, is_dc].sum(axis=1),
                         "base_mw": pset.loc[:, ~is_dc].sum(axis=1)}).rename_axis("snapshot")


# --------------------------------------------------------------------- panels
def panel_map(ax: plt.Axes, case_dir: Path, boundaries: Path) -> None:
    """Announced capacity mapped onto the ERCOT buses that receive it."""
    import xarray as xr

    mapping = pd.read_csv(case_dir / "dc_bus_mapping.csv")
    ds = xr.open_dataset(case_dir / "network.nc")
    bus_xy = pd.DataFrame({"x": ds["buses_x"].to_pandas(),
                           "y": ds["buses_y"].to_pandas()})
    ds.close()
    mapping = mapping.merge(bus_xy, left_on="bus", right_index=True, how="left")

    with open(boundaries) as fh:
        texas = json.load(fh)["features"][0]["geometry"]["coordinates"]
    for poly in texas:
        lon, lat = zip(*poly[0])
        ax.fill(lon, lat, facecolor="#F7F7F4", edgecolor="#A8A8A8",
                linewidth=0.6, zorder=0)

    ax.scatter(bus_xy["x"], bus_xy["y"], s=1.6, color="#C9C9C9",
               linewidths=0, zorder=1)
    zone_sums = mapping.groupby("zone")["capacity_mw"].sum().sort_values(ascending=False)
    for zone in zone_sums.index:
        dz = mapping[mapping["zone"] == zone]
        ax.scatter(dz["x"], dz["y"], s=dz["capacity_mw"] * 0.040,
                   color=ZONE_COLORS.get(zone, C["grey"]), alpha=0.78, linewidths=0.35,
                   edgecolors="white", zorder=2,
                   label=f"{ZONE_LABELS.get(zone, zone.title())}  {zone_sums[zone] / 1000:.1f} GW")

    leg = ax.legend(loc="lower left", frameon=False, handletextpad=0.35,
                    bbox_to_anchor=(-0.02, 0.0), fontsize=6.0,
                    labelspacing=0.32, borderpad=0.0)
    for hd in leg.legend_handles:
        hd.set_sizes([16])
    total_gw = mapping["capacity_mw"].sum() / 1000
    n_bus = mapping["bus"].nunique()
    projects = (f"{int(mapping['n_sites'].sum())} projects at "
                if "n_sites" in mapping.columns else "")
    ax.text(0.02, 0.97, f"{total_gw:.1f} GW announced\n{projects}{n_bus} buses",
            transform=ax.transAxes, fontsize=6.2, va="top", linespacing=1.35)
    ax.set_aspect(1.0 / np.cos(np.radians(31.0)))
    ax.axis("off")


def panel_load(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Near-constant data-centre load stacked on the existing demand curve."""
    # two weeks centred on the peak-demand day, so the daily cycle is visible
    peak = df["base_mw"].idxmax()
    w = df.loc[peak - pd.Timedelta(days=7): peak + pd.Timedelta(days=7)]
    x = np.arange(len(w))

    base = w["base_mw"].to_numpy() / 1e3
    dc = w["dc_mw"].to_numpy() / 1e3
    season_base = df["base_mw"] / 1e3
    season_dc = df["dc_mw"] / 1e3

    ax.fill_between(x, 0, base, facecolor=C["lgrey"], edgecolor="none", zorder=1)
    ax.plot(x, base, color=C["grey"], lw=0.7, zorder=3)
    ax.fill_between(x, base, base + dc, facecolor=C["blue"], alpha=0.88,
                    edgecolor="none", zorder=2)

    ax.set_xlim(0, len(w) - 1)
    ax.set_ylim(0, (base + dc).max() * 1.08)
    ax.set_ylabel("Demand (GW)", labelpad=2)

    step = max(1, int(round(pd.Timedelta(days=4) / (w.index[1] - w.index[0]))))
    ticks = list(range(0, len(w), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([w.index[i].strftime("%d %b") for i in ticks])
    ax.tick_params(axis="x", pad=1.5)

    ax.text(len(w) * 0.5, base.mean() + dc.mean() * 0.52,
            f"announced data centres\n{season_dc.min():.0f}-{season_dc.max():.0f} GW in every hour",
            ha="center", va="center", fontsize=6.2, color="white",
            fontweight="bold", linespacing=1.3, zorder=4)
    ax.text(len(w) * 0.5, base.mean() * 0.40,
            f"existing demand\n{season_base.min():.0f}-{season_base.max():.0f} GW",
            ha="center", va="center", fontsize=6.2, color="#4A4A4A",
            linespacing=1.3, zorder=4)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _pathway_bars(ax, means, los, his, ylabel, log=False):
    x = np.array([0, 1])
    ax.bar(x, means, width=0.54, color=[PW_RESOURCE, PW_COORDINATED],
           edgecolor="none", zorder=2)
    ax.errorbar(x, means, yerr=[means - los, his - means], fmt="none",
                ecolor="#3A3A3A", elinewidth=0.7, capsize=2.2, capthick=0.7,
                zorder=3)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(x)
    ax.set_xticklabels(["resource\nonly", "coordinated"], linespacing=1.25,
                       fontsize=6.0)
    ax.tick_params(axis="x", pad=1.5, length=0)
    ax.set_ylabel(ylabel, labelpad=2)
    if log:
        ax.set_yscale("log")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return x


def _ratio_label(ratio: float, word: str) -> str:
    return f"{ratio:.0f}x {word}" if ratio >= 10 else f"{ratio:.1f}x {word}"


def panel_invest(ax: plt.Axes, invest: pd.DataFrame) -> None:
    """Annualized investment under the two pathways (mean, min/max range)."""
    gs = invest["all2030_generation_storage"].to_numpy()
    ft = invest["all2030_full_tx"].to_numpy()
    means = np.array([gs.mean(), ft.mean()])
    x = _pathway_bars(ax, means, np.array([gs.min(), ft.min()]),
                      np.array([gs.max(), ft.max()]),
                      "Annualized investment\n($bn per year)")
    top = gs.max()
    ax.set_ylim(0, top * 1.22)
    for xi, m in zip(x, means):
        ax.text(xi, m + top * 0.04, f"{m:.1f}", ha="center", va="bottom",
                fontsize=6.6, fontweight="bold")
    yl = top * 1.07
    ax.annotate("", xy=(1, yl), xytext=(0, yl),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="#3A3A3A"))
    ax.text(0.5, yl + top * 0.02, _ratio_label(means[0] / means[1], "cheaper"),
            ha="center", va="bottom", fontsize=6.2, color="#3A3A3A")


def panel_capped(ax: plt.Axes, capped: pd.DataFrame, season_hours: float) -> None:
    """Hours per season priced above the offer cap (log scale)."""
    gs = capped["gen_storage"].to_numpy()
    ft = capped["full_tx"].to_numpy()
    means = np.array([gs.mean(), ft.mean()])
    x = _pathway_bars(ax, means, np.array([gs.min(), ft.min()]),
                      np.array([gs.max(), ft.max()]),
                      f"Hours above the price cap\n(of {season_hours:,.0f} in the season)",
                      log=True)
    ax.set_ylim(1, 9000)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1,000"])
    for xi, m in zip(x, means):
        ax.text(xi, m * 1.75, f"{m:.0f}", ha="center", va="bottom",
                fontsize=6.6, fontweight="bold")
    ax.annotate("", xy=(1, 3000), xytext=(0, 3000),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="#3A3A3A"))
    ax.text(0.5, 3500, _ratio_label(means[0] / means[1], "fewer"),
            ha="center", va="bottom", fontsize=6.2, color="#3A3A3A")


def panel_bill(ax: plt.Axes, bill: dict[str, float]) -> None:
    """Residential increase, split by channel, drawn as a waterfall.

    The three channels are signed terms, so each is a floating bar that steps
    from the running level: energy from zero, congestion from the energy level
    (downwards when negative, hatched), regulated transmission from the net
    wholesale level. The bar at the right is the true sum, stacked as net
    wholesale over regulated with the share of each channel."""
    e, g, r = bill["energy"], bill["congestion"], bill["regulated"]
    wholesale = e + g
    total = wholesale + r
    width = 0.62
    x_e, x_g, x_r, x_tot = 0.0, 1.15, 2.30, 3.65
    ymax = max(total, e, wholesale + r, e + max(g, 0.0))

    def _label(x, y, v, signed=False, **kw):
        ax.text(x, y, f"{v:+.2f}" if signed else f"{v:.2f}", ha="center",
                va="bottom", fontsize=6.0, color="#3A3A3A", zorder=4, **kw)

    ax.bar(x_e, e, width=width, color=CH_ENERGY, edgecolor="white", linewidth=0.7, zorder=2)
    _label(x_e, e + ymax * 0.02, e)

    if g >= 0:
        ax.bar(x_g, g, bottom=e, width=width, color=CH_CONGESTION, edgecolor="white",
               linewidth=0.7, zorder=2)
        _label(x_g, e + g + ymax * 0.02, g, signed=True)
    else:
        ax.bar(x_g, -g, bottom=wholesale, width=width, facecolor="white",
               edgecolor=CH_CONGESTION, hatch="////", linewidth=0.8, zorder=2)
        _label(x_g, e + ymax * 0.02, g, signed=True)

    ax.bar(x_r, r, bottom=wholesale, width=width, color=CH_REGULATED, edgecolor="white",
           linewidth=0.7, zorder=2)
    _label(x_r, wholesale + r + ymax * 0.02, r)

    # net bar: wholesale over regulated, with channel shares
    ax.bar(x_tot, wholesale, width=width, color=CH_ENERGY, edgecolor="white",
           linewidth=0.7, zorder=2)
    ax.bar(x_tot, r, bottom=wholesale, width=width, color=CH_REGULATED,
           edgecolor="white", linewidth=0.7, zorder=2)
    ax.text(x_tot, total + ymax * 0.025, f"{total:.2f}", ha="center", va="bottom",
            fontsize=7.4, fontweight="bold", zorder=4)
    ax.text(x_tot + width / 2 + 0.12, wholesale / 2,
            f"wholesale\n{100 * wholesale / total:.0f}%", ha="left", va="center",
            fontsize=5.8, linespacing=1.2, color="#3A3A3A")
    ax.text(x_tot + width / 2 + 0.12, wholesale + r / 2,
            f"regulated\n{100 * r / total:.0f}%", ha="left", va="center",
            fontsize=5.8, linespacing=1.2, color="#3A3A3A")

    # dotted connectors between the running levels
    for (x0, x1, y) in [(x_e, x_g, e), (x_g, x_r, wholesale), (x_r, x_tot, total)]:
        ax.plot([x0 + width / 2, x1 - width / 2], [y, y], lw=0.5, ls=":",
                color="#7A7A7A", zorder=1)
    ax.axhline(0, color="#3A3A3A", lw=0.6, zorder=1)

    ax.set_xlim(-0.55, x_tot + 1.55)
    ax.set_ylim(min(0.0, wholesale) - ymax * 0.02, ymax * 1.18)
    ax.set_xticks([x_e, x_g, x_r, x_tot])
    ax.set_xticklabels(["marginal\ngeneration", "congestion", "regulated\ntransmission",
                        "residential\nincrease"], fontsize=5.6, linespacing=1.2)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_ylabel("Residential increase\n(cents per kWh)", labelpad=2)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT,
                        help="Experiment run root holding results/seasonal/<year>/<case>/.")
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT,
                        help="Legacy-layout view holding results/nc_paper_figures/ tables.")
    parser.add_argument("--state-boundaries", type=Path, default=DEFAULT_STATE_BOUNDARIES,
                        help="GeoJSON with the Texas outline for panel a.")
    parser.add_argument("--tcos-residential-usd-per-mwh", type=float,
                        default=DEFAULT_TCOS_RESIDENTIAL,
                        help="Regulated transmission charge for the residential customer ($/MWh).")
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Directory for the generated figure files.")
    parser.add_argument("--manuscript-output-root", type=Path, default=None,
                        help="Optional second directory receiving a copy of the figure.")
    args = parser.parse_args()
    for name in ("run_root", "legacy_root", "state_boundaries", "results_root",
                 "manuscript_output_root"):
        val = getattr(args, name)
        if val is not None:
            setattr(args, name, val.expanduser().resolve())
    return args


def main() -> None:
    args = parse_args()
    case_dir = matched_case_dir(args.run_root, args.legacy_root)
    profiles = load_profiles(args.legacy_root, case_dir, args.results_root)
    invest = load_investment(args.run_root)
    capped = load_capped_hours(args.run_root, args.legacy_root)
    decomp = pd.read_csv(args.legacy_root / "results" / "nc_paper_figures" /
                         "adder_factor_decomposition.csv")
    bill = bill_components(decomp, args.tcos_residential_usd_per_mwh)
    season_hours = len(profiles) * HOURS_PER_SNAPSHOT

    fig = plt.figure(figsize=(DOUBLE_COL, 4.55))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.30, 1.0],
                             left=0.082, right=0.995, top=0.955, bottom=0.115,
                             hspace=0.30)
    top = outer[0].subgridspec(1, 2, width_ratios=[0.92, 1.72], wspace=0.14)
    bot = outer[1].subgridspec(1, 3, width_ratios=[0.80, 0.80, 1.35], wspace=0.52)

    ax_map = fig.add_subplot(top[0, 0])
    ax_ts = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bot[0, 0])
    ax_d = fig.add_subplot(bot[0, 1])
    ax_e = fig.add_subplot(bot[0, 2])

    panel_map(ax_map, case_dir, args.state_boundaries)
    panel_load(ax_ts, profiles)
    panel_invest(ax_c, invest)
    panel_capped(ax_d, capped, season_hours)
    panel_bill(ax_e, bill)

    for ax, lab, dx, dy in [(ax_map, "a", 0.0, 1.045), (ax_ts, "b", -0.085, 1.045),
                            (ax_c, "c", -0.30, 1.13), (ax_d, "d", -0.30, 1.13),
                            (ax_e, "e", -0.30, 1.13)]:
        ax.text(dx, dy, lab, transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", va="top", ha="left")

    outputs = [args.results_root] + ([args.manuscript_output_root]
                                     if args.manuscript_output_root else [])
    for d in outputs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig1_framework.pdf")
        fig.savefig(d / "fig1_framework.png", dpi=450)
    plt.close(fig)

    print("wrote fig1_framework.pdf/.png")
    print("--- figure 1 cross-check ---")
    print(f"years used (c): {list(invest.index)}; (d): {list(capped.index)}")
    if list(invest.index) != list(capped.index):
        print("WARNING: panels c and d use different weather-year sets "
              "(a case is missing or the graded table is stale)")
    print("investment $bn/yr:\n" + invest.round(3).to_string())
    print(f"  mean gen+storage {invest['all2030_generation_storage'].mean():.2f}, "
          f"full_tx {invest['all2030_full_tx'].mean():.2f}, ratio "
          f"{invest['all2030_generation_storage'].mean() / invest['all2030_full_tx'].mean():.2f}x")
    print("capped hours/season:\n" + capped.round(1).to_string())
    print(f"  ratio {capped['gen_storage'].mean() / capped['full_tx'].mean():.1f}x")
    tot = sum(bill.values())
    print(f"bill c/kWh ({MATCHED_YEAR}): energy {bill['energy']:.3f}, congestion "
          f"{bill['congestion']:+.3f}, regulated {bill['regulated']:.3f}, total {tot:.3f} "
          f"(wholesale {100 * (bill['energy'] + bill['congestion']) / tot:.1f}%)")


if __name__ == "__main__":
    main()
