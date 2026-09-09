#!/usr/bin/env python3
"""
validate_2021_full_year_network.py
==================================

Validate a full-year 2021 PyPSA-USA ERCOT network against historical ERCOT RTM
hub prices, and diagnose whether load shedding is material enough to affect
the LMP validation.

Fixes included:
1. ERCOT RTMLZHBSPP xlsx files may store each month in a separate sheet.
   Use --hist-xlsx to parse all sheets in one xlsx file, or --hist-dir to
   parse all matching xlsx files. The parser reads every valid sheet and
   combines them into a full-year hourly cache.
2. Tiny numerical load-shedding values should not be treated as meaningful
   shedding. The default shedding threshold is 1 MW.

Typical use:
    python scripts/validate_2021_full_year_network.py \
      --hist-xlsx data/ercot_historical/RTMLZHBSPP_2021.xlsx \
      --force-hist-reparse

Or, if the full-year parquet cache is already correct:
    python scripts/validate_2021_full_year_network.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

warnings.filterwarnings("ignore")


DEFAULT_NETWORK = Path(
    "private/validation_2021.nc"
)

DEFAULT_HIST_CACHE = Path("data/ercot_historical/rtm_hourly_2021.parquet")
DEFAULT_HIST_DIR = Path("data/ercot_historical")
DEFAULT_OUTDIR = Path("results/validation_2021_full_year")

YEAR = 2021
ZONES = ["WEST", "NORTH", "SOUTH", "HOUSTON"]

HUB_TO_ZONE = {
    "HB_WEST": "WEST",
    "HB_NORTH": "NORTH",
    "HB_SOUTH": "SOUTH",
    "HB_HOUSTON": "HOUSTON",
}

DEFAULT_SHED_MW_THRESHOLD = 1.0


def assign_ercot_zone(x: float, y: float) -> str:
    """Approximate ERCOT hub-zone assignment used in the DC scripts."""
    if x < -100.0:
        return "WEST"
    if y >= 32.0:
        return "NORTH"
    if x >= -96.5 and y < 30.5:
        return "HOUSTON"
    return "SOUTH"


def to_utc_naive_index(
    df_or_s: pd.DataFrame | pd.Series,
    source_tz: str,
) -> pd.DataFrame | pd.Series:
    """Convert index to tz-naive UTC."""
    obj = df_or_s.copy()
    idx = pd.to_datetime(obj.index)

    if idx.tz is None:
        idx = idx.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
    else:
        idx = idx.tz_convert("UTC")

    idx = idx.tz_convert("UTC").tz_localize(None)
    obj.index = idx
    obj = obj[obj.index.notna()]
    return obj.sort_index()


def safe_months(index: pd.Index) -> list[int]:
    idx = pd.to_datetime(index)
    if len(idx) == 0:
        return []
    return sorted(set(idx.month))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _parse_timestamp_from_ercot_rows(df: pd.DataFrame) -> pd.Series:
    """
    ERCOT RTMLZHBSPP convention:
      Delivery Hour: 1-24
      Delivery Interval: 1-4, each 15 min
    """
    date = pd.to_datetime(df["Delivery Date"], errors="coerce")
    hour = pd.to_numeric(df["Delivery Hour"], errors="coerce")
    interval = pd.to_numeric(df["Delivery Interval"], errors="coerce")

    return (
        date
        + pd.to_timedelta(hour - 1, unit="h")
        + pd.to_timedelta((interval - 1) * 15, unit="min")
    )


def parse_ercot_rtm_xlsx_all_sheets(xlsx_paths: list[Path]) -> pd.DataFrame:
    """
    Parse ERCOT RTMLZHBSPP xlsx files.

    Handles:
      - one xlsx containing 12 monthly sheets
      - multiple xlsx files, each with one or more sheets

    Returns hourly DataFrame with columns WEST, NORTH, SOUTH, HOUSTON.
    Timestamp is tz-naive Central local time before later timezone conversion.
    """
    if not xlsx_paths:
        raise FileNotFoundError("No xlsx files were provided for historical RTM parsing.")

    needed = {
        "Settlement Point Name",
        "Delivery Date",
        "Delivery Hour",
        "Delivery Interval",
        "Settlement Point Price",
    }

    frames: list[pd.DataFrame] = []

    print(f"  parsing {len(xlsx_paths)} historical xlsx file(s)")
    for xlsx_path in xlsx_paths:
        print(f"    reading: {xlsx_path}")
        try:
            sheets = pd.read_excel(
                xlsx_path,
                sheet_name=None,
                dtype={"Delivery Date": str},
            )
        except Exception as exc:
            print(f"      [WARN] failed to read {xlsx_path.name}: {exc}")
            continue

        print(f"      sheets: {list(sheets.keys())}")

        for sheet_name, raw in sheets.items():
            if raw is None or raw.empty:
                print(f"      [skip] {sheet_name}: empty")
                continue

            df = _normalize_columns(raw)

            if not needed.issubset(df.columns):
                print(f"      [skip] {sheet_name}: missing required columns")
                continue

            tmp = df.copy()
            tmp = tmp[tmp["Settlement Point Name"].isin(HUB_TO_ZONE.keys())].copy()
            if tmp.empty:
                print(f"      [skip] {sheet_name}: no ERCOT hub rows")
                continue

            for col in ["Delivery Hour", "Delivery Interval", "Settlement Point Price"]:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

            tmp["timestamp"] = _parse_timestamp_from_ercot_rows(tmp)
            tmp["zone"] = tmp["Settlement Point Name"].map(HUB_TO_ZONE)

            tmp = tmp.dropna(
                subset=[
                    "timestamp",
                    "Delivery Hour",
                    "Delivery Interval",
                    "Settlement Point Price",
                    "zone",
                ]
            )

            if "Repeated Hour Flag" in tmp.columns:
                tmp = tmp[tmp["Repeated Hour Flag"].astype(str).str.upper() != "Y"]

            if tmp.empty:
                print(f"      [skip] {sheet_name}: no valid rows after cleaning")
                continue

            print(
                f"      parsed {sheet_name}: rows={len(tmp)}, "
                f"range={tmp['timestamp'].min()} -> {tmp['timestamp'].max()}"
            )

            frames.append(tmp[["timestamp", "zone", "Settlement Point Price"]])

    if not frames:
        raise RuntimeError("No valid historical RTM rows were parsed from xlsx file(s).")

    full = pd.concat(frames, ignore_index=True)

    pivot_15min = full.pivot_table(
        index="timestamp",
        columns="zone",
        values="Settlement Point Price",
        aggfunc="mean",
    ).sort_index()

    hourly = pivot_15min.resample("h").mean()

    months = safe_months(hourly.index)
    print("  parsed historical xlsx summary:")
    print(f"    hourly rows: {len(hourly)}")
    print(f"    range:       {hourly.index.min()} -> {hourly.index.max()}")
    print(f"    months:      {months}")
    print(f"    columns:     {hourly.columns.tolist()}")
    print(f"    price range: {np.nanmin(hourly.values):.2f} -> {np.nanmax(hourly.values):.2f}")

    if len(months) < 12:
        print(
            f"  [WARN] historical xlsx covers only {len(months)}/12 months. "
            "Validation will only cover overlapping timestamps."
        )

    return hourly


def discover_xlsx_files(hist_dir: Path, year: int = YEAR) -> list[Path]:
    """
    Discover RTMLZHBSPP xlsx files for one specific year only.

    Important:
    Do not fall back to all years if year-specific files exist.
    Otherwise rtm_hourly_2021.parquet may accidentally contain 2019–2023.
    """
    year_specific_patterns = [
        f"*RTMLZHBSPP*{year}*.xlsx",
        f"*{year}*RTMLZHBSPP*.xlsx",
    ]

    files: list[Path] = []
    for pat in year_specific_patterns:
        files.extend(hist_dir.glob(pat))

    files = sorted(set(files))

    if files:
        return files

    # Only use generic fallback if no year-specific file exists.
    generic = sorted(set(hist_dir.glob("*RTMLZHBSPP*.xlsx")))

    # Still filter by year in filename if possible.
    filtered = [p for p in generic if str(year) in p.name]

    if filtered:
        return filtered

    raise FileNotFoundError(
        f"No RTMLZHBSPP xlsx file found for year {year} in {hist_dir}. "
        f"Use --hist-xlsx to specify the file directly."
    )


def load_historical_lmp_2021(
    hist_cache: Path,
    hist_tz: str = "US/Central",
    hist_xlsx: Path | None = None,
    hist_dir: Path = DEFAULT_HIST_DIR,
    force_reparse: bool = False,
) -> pd.DataFrame:
    """
    Load historical ERCOT hub prices and return 3h UTC-naive DataFrame.

    Priority:
      1. If --hist-xlsx is provided, parse all sheets from that xlsx.
      2. If --force-hist-reparse is set, parse xlsx files from --hist-dir.
      3. Else use parquet cache if it exists and appears complete enough.
      4. Else parse xlsx files from --hist-dir.
    """
    hist: pd.DataFrame | None = None

    if hist_xlsx is not None:
        hist = parse_ercot_rtm_xlsx_all_sheets([hist_xlsx])
        hist_cache.parent.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(hist_cache)
        print(f"  saved rebuilt historical cache: {hist_cache}")

    elif force_reparse:
        files = discover_xlsx_files(hist_dir, YEAR)
        hist = parse_ercot_rtm_xlsx_all_sheets(files)
        hist_cache.parent.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(hist_cache)
        print(f"  saved rebuilt historical cache: {hist_cache}")

    elif hist_cache.exists():
        hist = pd.read_parquet(hist_cache)
        hist.index = pd.to_datetime(hist.index)

        months = safe_months(hist.index)
        print("  loaded historical parquet:")
        print(f"    rows hourly: {len(hist)}")
        print(f"    range:       {hist.index.min()} -> {hist.index.max()}")
        print(f"    months:      {months}")
        print(f"    columns:     {hist.columns.tolist()}")

        if len(hist) < 4000 or len(months) < 6:
            print("  [WARN] historical cache looks incomplete; reparsing xlsx files")
            files = discover_xlsx_files(hist_dir, YEAR)
            hist = parse_ercot_rtm_xlsx_all_sheets(files)
            hist_cache.parent.mkdir(parents=True, exist_ok=True)
            hist.to_parquet(hist_cache)
            print(f"  saved rebuilt historical cache: {hist_cache}")

    else:
        files = discover_xlsx_files(hist_dir, YEAR)
        hist = parse_ercot_rtm_xlsx_all_sheets(files)
        hist_cache.parent.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(hist_cache)
        print(f"  saved rebuilt historical cache: {hist_cache}")

    if hist is None or hist.empty:
        raise RuntimeError("Historical LMP data could not be loaded.")

    hist = _normalize_columns(hist)

    rename = {c: HUB_TO_ZONE[c] for c in hist.columns if c in HUB_TO_ZONE}
    hist = hist.rename(columns=rename)

    keep = [z for z in ZONES if z in hist.columns]
    if len(keep) == 0:
        raise ValueError(
            f"No expected ERCOT hub/zone columns found. Columns={hist.columns.tolist()}"
        )

    hist = hist[keep].sort_index()
    hist = to_utc_naive_index(hist, hist_tz)

    hist_3h = hist.resample("3h").mean()
    hist_3h["system"] = hist_3h[keep].mean(axis=1)

    return hist_3h[["system"] + keep]


def demand_weights_by_bus(
    n: pypsa.Network,
    buses: pd.Index,
    exclude_dc: bool = True,
) -> pd.Series:
    """Use average non-DC load at each bus as aggregation weight."""
    if len(buses) == 0:
        return pd.Series(dtype=float)

    if not hasattr(n, "loads_t") or "p_set" not in n.loads_t or n.loads_t.p_set.empty:
        return pd.Series(1.0 / len(buses), index=buses)

    if n.loads.empty:
        return pd.Series(1.0 / len(buses), index=buses)

    load_ids = n.loads.index.intersection(n.loads_t.p_set.columns)

    if exclude_dc:
        load_ids = [lid for lid in load_ids if not str(lid).startswith("DC_")]

    if len(load_ids) == 0:
        return pd.Series(1.0 / len(buses), index=buses)

    mean_load = n.loads_t.p_set[load_ids].mean()
    load_buses = n.loads.loc[mean_load.index, "bus"]
    bus_load = mean_load.groupby(load_buses).sum()

    common = buses.intersection(bus_load.index)
    w = pd.Series(0.0, index=buses)
    w.loc[common] = bus_load.loc[common].astype(float).clip(lower=0)

    if w.sum() <= 0:
        return pd.Series(1.0 / len(buses), index=buses)

    return w / w.sum()


def extract_pypsa_lmp(
    n: pypsa.Network,
    pypsa_tz: str = "UTC",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract 3h system and zonal ERCOT LMP from network."""
    lmp = n.buses_t.marginal_price.copy()

    if isinstance(lmp.index, pd.MultiIndex):
        lmp.index = pd.to_datetime(lmp.index.get_level_values("timestep"))
    else:
        lmp.index = pd.to_datetime(lmp.index)

    lmp = to_utc_naive_index(lmp, pypsa_tz)

    buses = n.buses.copy()
    bus_zones = buses.apply(lambda r: assign_ercot_zone(r["x"], r["y"]), axis=1)

    out: dict[str, pd.Series] = {}

    system_buses = lmp.columns.intersection(buses.index)
    w_system = demand_weights_by_bus(n, system_buses)
    out["system"] = (lmp[system_buses] * w_system.loc[system_buses]).sum(axis=1)

    for zone in ZONES:
        zone_buses = bus_zones[bus_zones == zone].index.intersection(lmp.columns)
        if len(zone_buses) == 0:
            continue
        w = demand_weights_by_bus(n, zone_buses)
        out[zone] = (lmp[zone_buses] * w.loc[zone_buses]).sum(axis=1)

    lmp_agg = pd.DataFrame(out).sort_index()
    return lmp_agg, lmp


def detect_load_shedding(
    n: pypsa.Network,
    hours_per_step: float = 3.0,
    pypsa_tz: str = "UTC",
    shed_mw_threshold: float = DEFAULT_SHED_MW_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect meaningful load shedding from PyPSA generators."""
    if n.generators.empty or not hasattr(n, "generators_t") or "p" not in n.generators_t:
        idx = pd.to_datetime(n.snapshots)
        empty = pd.DataFrame(index=idx, data={"shed_mw": 0.0, "shed_mwh": 0.0})
        empty["has_shedding"] = False
        return to_utc_naive_index(empty, pypsa_tz), pd.DataFrame()

    gen = n.generators.copy()
    carrier = gen.get("carrier", pd.Series("", index=gen.index)).astype(str).str.lower()
    names = gen.index.astype(str).str.lower()

    mask = (
        carrier.str.contains("load", na=False)
        | carrier.str.contains("shed", na=False)
        | carrier.str.contains("unserved", na=False)
        | names.str.contains("load shedding", na=False)
        | names.str.contains("unserved", na=False)
        | names.str.contains("shed", na=False)
    )

    shed_gens = gen.index[mask].intersection(n.generators_t.p.columns)

    if len(shed_gens) == 0:
        idx = pd.to_datetime(n.snapshots)
        if isinstance(idx, pd.MultiIndex):
            idx = pd.to_datetime(idx.get_level_values("timestep"))
        empty = pd.DataFrame(index=idx, data={"shed_mw": 0.0, "shed_mwh": 0.0})
        empty["has_shedding"] = False
        return to_utc_naive_index(empty, pypsa_tz), pd.DataFrame()

    p = n.generators_t.p[shed_gens].clip(lower=0).copy()

    if isinstance(p.index, pd.MultiIndex):
        p.index = pd.to_datetime(p.index.get_level_values("timestep"))
    else:
        p.index = pd.to_datetime(p.index)

    p = to_utc_naive_index(p, pypsa_tz)

    shed_ts = pd.DataFrame(index=p.index)
    shed_ts["shed_mw"] = p.sum(axis=1)
    shed_ts["shed_mwh"] = shed_ts["shed_mw"] * hours_per_step
    shed_ts["has_shedding"] = shed_ts["shed_mw"] > shed_mw_threshold

    by_asset = pd.DataFrame({
        "generator": shed_gens,
        "carrier": gen.loc[shed_gens, "carrier"].astype(str).values,
        "bus": gen.loc[shed_gens, "bus"].astype(str).values,
        "total_shed_mwh": (p.sum(axis=0) * hours_per_step).values,
        "max_shed_mw": p.max(axis=0).values,
    })
    by_asset = by_asset.sort_values("total_shed_mwh", ascending=False)

    return shed_ts, by_asset


def summarize_validation(
    pypsa_3h: pd.DataFrame,
    hist_3h: pd.DataFrame,
    shed_ts: pd.DataFrame,
) -> pd.DataFrame:
    """Compare PyPSA and historical LMP with and without shedding timestamps."""
    common = pypsa_3h.index.intersection(hist_3h.index)
    p = pypsa_3h.loc[common]
    h = hist_3h.loc[common]

    shed_aligned = shed_ts.reindex(common)
    shed_aligned["has_shedding"] = shed_aligned.get("has_shedding", False)
    no_shed_mask = ~shed_aligned["has_shedding"].fillna(False).astype(bool)

    rows = []

    for zone in [c for c in p.columns if c in h.columns]:
        samples = {
            "all_hours": pd.Series(True, index=common),
            "excluding_shedding_hours": no_shed_mask,
        }

        for sample_name, mask in samples.items():
            pp = p.loc[mask, zone].dropna()
            hh = h.loc[mask, zone].dropna()
            idx = pp.index.intersection(hh.index)
            pp = pp.loc[idx]
            hh = hh.loc[idx]

            if len(idx) < 10:
                continue

            rows.append({
                "zone": zone,
                "sample": sample_name,
                "n_3h_steps": len(idx),
                "pypsa_mean": pp.mean(),
                "hist_mean": hh.mean(),
                "bias": (pp - hh).mean(),
                "mae": (pp - hh).abs().mean(),
                "rmse": np.sqrt(((pp - hh) ** 2).mean()),
                "corr": np.corrcoef(pp, hh)[0, 1],
                "pypsa_p95": np.percentile(pp, 95),
                "hist_p95": np.percentile(hh, 95),
                "pypsa_p99": np.percentile(pp, 99),
                "hist_p99": np.percentile(hh, 99),
            })

    return pd.DataFrame(rows)


def summarize_load_shedding(
    shed_ts: pd.DataFrame,
    lmp_3h: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize meaningful shedding and LMP levels during those periods."""
    total_steps = len(shed_ts)
    has_shed = shed_ts["has_shedding"].astype(bool) if "has_shedding" in shed_ts else shed_ts["shed_mw"] > DEFAULT_SHED_MW_THRESHOLD

    shed_steps = int(has_shed.sum())
    total_mwh_raw = float(shed_ts["shed_mwh"].sum())
    total_mwh_meaningful = float(shed_ts.loc[has_shed, "shed_mwh"].sum())
    max_mw = float(shed_ts["shed_mw"].max())
    share_steps = shed_steps / total_steps if total_steps else 0.0

    rows = [
        {"metric": "total_3h_steps", "value": total_steps},
        {"metric": "meaningful_shedding_3h_steps", "value": shed_steps},
        {"metric": "share_steps_with_meaningful_shedding", "value": share_steps},
        {"metric": "total_shed_mwh_raw", "value": total_mwh_raw},
        {"metric": "total_shed_mwh_meaningful", "value": total_mwh_meaningful},
        {"metric": "max_shed_mw_raw", "value": max_mw},
    ]

    common = shed_ts.index.intersection(lmp_3h.index)
    mask = has_shed.reindex(common).fillna(False)

    for col in lmp_3h.columns:
        if mask.any():
            rows.append({
                "metric": f"{col}_mean_lmp_during_meaningful_shedding",
                "value": float(lmp_3h.loc[common[mask], col].mean()),
            })
        rows.append({
            "metric": f"{col}_mean_lmp_no_meaningful_shedding",
            "value": float(lmp_3h.loc[common[~mask], col].mean()) if (~mask).any() else np.nan,
        })

    return pd.DataFrame(rows)


def _cap_ylim(
    *series: pd.Series,
    floor: float = -20.0,
    q: float = 99.0,
    multiplier: float = 1.25,
    hard_cap: float | None = None,
) -> tuple[float, float]:
    vals = []
    for s in series:
        vals.extend(pd.Series(s).dropna().values.tolist())
    if not vals:
        return floor, 100.0
    upper = np.nanpercentile(vals, q) * multiplier
    if hard_cap is not None:
        upper = min(upper, hard_cap)
    upper = max(upper, 50.0)
    return floor, upper


def plot_outputs(
    pypsa_3h: pd.DataFrame,
    hist_3h: pd.DataFrame,
    shed_ts: pd.DataFrame,
    outdir: Path,
):
    """Generate validation plots."""
    common = pypsa_3h.index.intersection(hist_3h.index)
    p = pypsa_3h.loc[common]
    h = hist_3h.loc[common]
    shed = shed_ts.reindex(common).copy()
    if "shed_mw" not in shed:
        shed["shed_mw"] = 0.0
    if "shed_mwh" not in shed:
        shed["shed_mwh"] = 0.0
    if "has_shedding" not in shed:
        shed["has_shedding"] = False
    shed = shed.fillna({"shed_mw": 0.0, "shed_mwh": 0.0, "has_shedding": False})

    # 1. system daily trend
    fig, ax = plt.subplots(figsize=(14, 4.8))
    p_daily = p["system"].resample("D").mean()
    h_daily = h["system"].resample("D").mean()
    ax.plot(h_daily.index, h_daily.values, label="ERCOT historical RTM hub avg", lw=1.4)
    ax.plot(p_daily.index, p_daily.values, label="PyPSA system LMP", lw=1.4)
    ax.axvspan(pd.Timestamp("2021-02-13"), pd.Timestamp("2021-02-20"), alpha=0.18, label="Winter Storm Uri")
    ax.set_ylabel("Daily mean LMP ($/MWh)")
    ax.set_title("2021 ERCOT system LMP validation")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_ylim(*_cap_ylim(p_daily, h_daily, hard_cap=1000))
    fig.savefig(outdir / "fig_2021_system_lmp_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. zonal daily trend
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, z in zip(axes, ZONES):
        if z not in p.columns or z not in h.columns:
            continue
        pp = p[z].resample("D").mean()
        hh = h[z].resample("D").mean()
        ax.plot(hh.index, hh.values, label="Historical", lw=1.1)
        ax.plot(pp.index, pp.values, label="PyPSA", lw=1.1)
        ax.axvspan(pd.Timestamp("2021-02-13"), pd.Timestamp("2021-02-20"), alpha=0.15)
        ax.set_title(z)
        ax.set_ylabel("$/MWh")
        ax.grid(alpha=0.25)
        ax.set_ylim(*_cap_ylim(pp, hh, hard_cap=1000))
    axes[0].legend()
    fig.suptitle("2021 zonal LMP validation")
    fig.savefig(outdir / "fig_2021_zonal_lmp_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3. Uri zoom
    uri_start = pd.Timestamp("2021-02-10")
    uri_end = pd.Timestamp("2021-02-25")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    pp = p.loc[uri_start:uri_end, "system"]
    hh = h.loc[uri_start:uri_end, "system"]
    ax.plot(hh.index, hh.values, label="Historical", marker="o", ms=3)
    ax.plot(pp.index, pp.values, label="PyPSA", marker="o", ms=3)
    ax.set_ylabel("3h LMP ($/MWh)")
    ax.set_title("Winter Storm Uri zoom: system LMP")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_ylim(*_cap_ylim(pp, hh, q=98, hard_cap=5000))
    fig.savefig(outdir / "fig_2021_uri_zoom.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4. duration curve
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, series in {"Historical": h["system"], "PyPSA": p["system"]}.items():
        vals = np.sort(series.dropna().values)[::-1]
        ax.plot(np.linspace(0, 100, len(vals)), vals, label=name)
    ax.set_ylim(*_cap_ylim(p["system"], h["system"], q=99.5, hard_cap=5000))
    ax.set_xlabel("Share of 3h periods (%)")
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title("2021 system LMP duration curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "fig_2021_duration_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 5. scatter
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(h["system"], p["system"], s=8, alpha=0.25)
    cap = np.nanpercentile(pd.concat([h["system"], p["system"]]), 99)
    cap = max(cap, 50)
    ax.plot([0, cap], [0, cap], "k--", lw=1)
    ax.set_xlim(-10, cap)
    ax.set_ylim(-10, cap)
    ax.set_xlabel("Historical system hub avg ($/MWh)")
    ax.set_ylabel("PyPSA system LMP ($/MWh)")
    ax.set_title("2021 3h LMP comparison")
    ax.grid(alpha=0.25)
    fig.savefig(outdir / "fig_2021_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 6. load shedding time series
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.plot(shed.index, shed["shed_mw"], lw=1.2)
    ax.axhline(DEFAULT_SHED_MW_THRESHOLD, ls="--", lw=0.9, label=f"{DEFAULT_SHED_MW_THRESHOLD:g} MW threshold")
    ax.axvspan(pd.Timestamp("2021-02-13"), pd.Timestamp("2021-02-20"), alpha=0.18, label="Winter Storm Uri")
    ax.set_ylabel("Load shedding (MW)")
    ax.set_title("PyPSA load shedding diagnostic")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "fig_2021_load_shedding.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    ap.add_argument("--hist-cache", type=Path, default=DEFAULT_HIST_CACHE)
    ap.add_argument("--hist-xlsx", type=Path, default=None, help="Specific RTMLZHBSPP xlsx file with all monthly sheets.")
    ap.add_argument("--hist-dir", type=Path, default=DEFAULT_HIST_DIR, help="Directory containing RTMLZHBSPP xlsx files.")
    ap.add_argument("--force-hist-reparse", action="store_true", help="Ignore parquet cache and rebuild from xlsx.")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--pypsa-tz", type=str, default="UTC")
    ap.add_argument("--hist-tz", type=str, default="US/Central")
    ap.add_argument("--hours-per-step", type=float, default=3.0)
    ap.add_argument("--shed-mw-threshold", type=float, default=DEFAULT_SHED_MW_THRESHOLD)

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("2021 full-year ERCOT LMP validation")
    print("=" * 80)
    print(f"network:              {args.network}")
    print(f"hist cache:           {args.hist_cache}")
    print(f"hist xlsx:            {args.hist_xlsx}")
    print(f"hist dir:             {args.hist_dir}")
    print(f"outdir:               {args.outdir}")
    print(f"pypsa tz:             {args.pypsa_tz}")
    print(f"hist tz:              {args.hist_tz}")
    print(f"shed MW threshold:    {args.shed_mw_threshold}")

    print("\n[1] Loading PyPSA network...")
    n = pypsa.Network(str(args.network))
    print(f"  snapshots:  {len(n.snapshots)}")
    print(f"  buses:      {len(n.buses)}")
    print(f"  generators: {len(n.generators)}")
    print(f"  loads:      {len(n.loads)}")

    print("\n[2] Extracting PyPSA 3h LMP...")
    pypsa_3h, nodal_lmp = extract_pypsa_lmp(n, pypsa_tz=args.pypsa_tz)
    pypsa_3h.to_csv(args.outdir / "pypsa_lmp_3h_2021.csv")
    print(pypsa_3h.describe().round(2))

    print("\n[3] Loading historical ERCOT RTM LMP...")
    hist_3h = load_historical_lmp_2021(
        hist_cache=args.hist_cache,
        hist_tz=args.hist_tz,
        hist_xlsx=args.hist_xlsx,
        hist_dir=args.hist_dir,
        force_reparse=args.force_hist_reparse,
    )
    hist_3h.to_csv(args.outdir / "historical_lmp_3h_2021.csv")
    print(hist_3h.describe().round(2))

    print("\n[4] Detecting load shedding...")
    shed_ts, shed_by_asset = detect_load_shedding(
        n,
        hours_per_step=args.hours_per_step,
        pypsa_tz=args.pypsa_tz,
        shed_mw_threshold=args.shed_mw_threshold,
    )
    shed_ts.to_csv(args.outdir / "load_shedding_2021.csv")
    shed_by_asset.to_csv(args.outdir / "load_shedding_by_asset_2021.csv", index=False)

    shed_summary = summarize_load_shedding(shed_ts, pypsa_3h)
    shed_summary.to_csv(args.outdir / "load_shedding_summary_2021.csv", index=False)
    print("\nLoad shedding summary:")
    print(shed_summary.to_string(index=False))

    print("\n[5] Aligning and summarizing validation...")
    common = pypsa_3h.index.intersection(hist_3h.index)
    shed_aligned = shed_ts.reindex(common).copy()
    if "shed_mw" not in shed_aligned:
        shed_aligned["shed_mw"] = 0.0
    if "shed_mwh" not in shed_aligned:
        shed_aligned["shed_mwh"] = 0.0
    if "has_shedding" not in shed_aligned:
        shed_aligned["has_shedding"] = False
    shed_aligned = shed_aligned.fillna({"shed_mw": 0.0, "shed_mwh": 0.0, "has_shedding": False})

    aligned = pd.concat(
        {
            "pypsa": pypsa_3h.loc[common],
            "hist": hist_3h.loc[common],
            "shed": shed_aligned,
        },
        axis=1,
    )
    aligned.to_csv(args.outdir / "aligned_lmp_3h_2021.csv")

    validation = summarize_validation(pypsa_3h, hist_3h, shed_ts)
    validation.to_csv(args.outdir / "validation_summary_2021.csv", index=False)
    print(validation.round(3).to_string(index=False))

    print("\n[6] Plotting...")
    plot_outputs(pypsa_3h, hist_3h, shed_ts, args.outdir)

    print("\n[7] Interpretation helper:")
    meaningful_shed_mwh = float(shed_ts.loc[shed_ts["has_shedding"], "shed_mwh"].sum())
    raw_shed_mwh = float(shed_ts["shed_mwh"].sum())
    shed_steps = int(shed_ts["has_shedding"].sum())
    share_steps = shed_steps / len(shed_ts) if len(shed_ts) else 0.0
    max_shed_mw = float(shed_ts["shed_mw"].max()) if len(shed_ts) else 0.0

    print(f"  Raw total shedding:        {raw_shed_mwh:.6f} MWh")
    print(f"  Meaningful shedding:       {meaningful_shed_mwh:.6f} MWh")
    print(f"  Max raw shedding:          {max_shed_mw:.6f} MW")
    print(f"  Meaningful shedding steps: {shed_steps}/{len(shed_ts)} ({share_steps:.4%})")

    if meaningful_shed_mwh == 0 and max_shed_mw < args.shed_mw_threshold:
        print("  No meaningful load shedding detected. LMP validation is not affected by shedding.")
    elif share_steps < 0.01:
        print(
            "  Meaningful load shedding is rare (<1% of 3h periods). "
            "Report validation both with and without shedding periods; the main trend is likely usable."
        )
    else:
        print(
            "  Meaningful load shedding occurs in a non-trivial share of periods. "
            "LMP during those periods may be affected by the shedding penalty."
        )

    print(f"\nDone. Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
