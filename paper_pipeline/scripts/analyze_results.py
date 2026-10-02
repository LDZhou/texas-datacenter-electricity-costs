#!/usr/bin/env python3
"""
analyze_results.py
==============================

Monte Carlo default simulation for a stylized ERCOT Load-Serving Entity / REP
under PyPSA-USA data-center scenarios.

This replaces the old deterministic "bankruptcy_prob = capital breach across
five weather years" analysis.

Cost-channel logic
------------------
1. New generation and storage are IPP / merchant / contracted assets.
   Their costs enter the REP through wholesale LMPs and forward prices.

2. New transmission is TSP-built regulated infrastructure.
   Its annualized cost is allocated to the REP/LSE using an ERCOT-style 4CP
   allocator.

3. Bankruptcy/default is simulated economically:
   each Monte Carlo path samples weather year, hedge ratio, forward premium,
   retail-price shock, pass-through, operating-cost shock, and capital buffer.
   The REP cash balance evolves through the year. A default occurs if cash
   balance falls below zero.

Core equations
--------------
Energy procurement:
    C_energy,t = h F + (1-h) lambda_t

4CP allocation:
    share_4CP = LSE_4CP_MW / ERCOT_4CP_MW
    assigned_TCOS = annual_TX_investment * share_4CP
    A_TCOS = assigned_TCOS / LSE_annual_MWh

Effective retail price:
    G_eff = G * (1 + retail_shock) + rho * A_TCOS

Non-energy cost:
    C_nonenergy = OpEx * (1 + opex_shock) + (1-rho) * A_TCOS

Margin:
    m_t = G_eff - C_energy,t - C_nonenergy

Cash:
    cash_t = capital + cumulative(m_t * L * dt)

Default:
    default = 1[min_t cash_t < 0]

Outputs
-------
results/dc_experiments/_figures/
    rep_default_sim_cache.pkl
    rep_default_sim_summary_<zone>_<forward>.csv
    rep_default_sim_raw_<zone>_<forward>.csv
    fig_rep_default_all2030_<zone>_<forward>.png
    fig_rep_default_tcos_4cp_<zone>_<forward>.png
    fig_rep_default_margin_scale_<zone>_<forward>.png

Typical run
-----------
    python -m paper_pipeline.scripts.analyze_results \
      --zones system WEST NORTH SOUTH HOUSTON \
      --forward-case both \
      --n-sims 2000 \
      --refresh-cache

This script only reads existing PyPSA results. It does not rerun PyPSA.
"""

from __future__ import annotations

import argparse
import pickle
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dc_common import DEFAULT_SCALES_MW as SCALES

ALL_YEARS = [2019, 2020, 2021, 2022, 2023]
LOCATIONS = ["HOUSTON", "AUSTIN", "TOLAR", "DFW_SOUTH", "ABILENE", "CHILDRESS"]
MODES = ["dispatch", "generation", "generation_tx"]


# =============================================================================
# Constants
# =============================================================================

MODE_COLORS = {
    "dispatch": "#777777",
    "generation": "#D55E00",
    "generation_storage": "#E69F00",
    "generation_tx": "#0072B2",
    "transmission": "#009E73",
    "full_tx": "#CC79A7",
    "expansion": "#D55E00",
    "expansion_tx": "#0072B2",
}

MODE_LABELS = {
    "dispatch": "Dispatch only",
    "generation": "Generation",
    "generation_storage": "Generation + storage",
    "generation_tx": "Generation + TX",
    "full_tx": "Generation + storage + transmission",
}

LOC_LABELS = {
    "HOUSTON": "Houston",
    "AUSTIN": "Austin",
    "TOLAR": "Tolar",
    "DFW_SOUTH": "DFW South",
    "ABILENE": "Abilene",
    "CHILDRESS": "Childress",
}

ZONES = ["WEST", "NORTH", "SOUTH", "HOUSTON"]

# Texas residential retail price fallback, cents/kWh.
# We strip TDU portion to get the energy component G.
TEXAS_RETAIL_PRICE_MONTHLY = {
    (2019, 1): 11.20, (2019, 2): 11.31, (2019, 3): 11.22, (2019, 4): 11.17,
    (2019, 5): 11.24, (2019, 6): 11.41, (2019, 7): 11.52, (2019, 8): 11.83,
    (2019, 9): 11.94, (2019, 10): 11.46, (2019, 11): 11.02, (2019, 12): 10.98,
    (2020, 1): 11.54, (2020, 2): 11.42, (2020, 3): 11.31, (2020, 4): 11.02,
    (2020, 5): 11.48, (2020, 6): 11.95, (2020, 7): 12.01, (2020, 8): 12.10,
    (2020, 9): 12.32, (2020, 10): 12.19, (2020, 11): 11.74, (2020, 12): 11.52,
    (2021, 1): 11.35, (2021, 2): 12.69, (2021, 3): 11.70, (2021, 4): 12.08,
    (2021, 5): 12.09, (2021, 6): 12.04, (2021, 7): 11.73, (2021, 8): 11.94,
    (2021, 9): 12.45, (2021, 10): 12.56, (2021, 11): 12.23, (2021, 12): 12.10,
    (2022, 1): 12.45, (2022, 2): 12.48, (2022, 3): 13.01, (2022, 4): 13.29,
    (2022, 5): 13.34, (2022, 6): 13.49, (2022, 7): 13.71, (2022, 8): 14.12,
    (2022, 9): 14.75, (2022, 10): 14.94, (2022, 11): 15.01, (2022, 12): 14.49,
    (2023, 1): 14.36, (2023, 2): 14.72, (2023, 3): 14.61, (2023, 4): 14.36,
    (2023, 5): 14.65, (2023, 6): 14.35, (2023, 7): 13.95, (2023, 8): 14.17,
    (2023, 9): 14.68, (2023, 10): 14.89, (2023, 11): 14.79, (2023, 12): 14.64,
}

TDU_FRACTION_BY_YEAR = {
    2019: 0.38,
    2020: 0.37,
    2021: 0.36,
    2022: 0.34,
    2023: 0.35,
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
})


# =============================================================================
# Experiment enumeration
# =============================================================================

def enumerate_experiments(base_dir: Path, years: Iterable[int], scenarios: Iterable[str]) -> List[dict]:
    scenarios = set(scenarios)

    # Longest alternatives first: the multiloc pattern is followed by a greedy
    # (.+) location group, so "generation" would otherwise swallow the mode of
    # e.g. multiloc_generation_storage_CHILDRESS_5000MW and report the location
    # as "storage_CHILDRESS".
    mode_re = (
        r"(dispatch|generation_storage|generation_tx|generation"
        r"|transmission|full_tx|expansion_tx|expansion)"
    )

    # Revision-v2 output names carry controlled suffixes such as
    # ``_crossover0_gas0.5``.  Keep them as identity fields rather than
    # silently discarding them and mixing sensitivities with base cases.
    suffix = r"(?:_crossover(\d+))?(?:_(.+))?"
    ml = re.compile(rf"^multiloc_{mode_re}_(.+)_(\d+)MW{suffix}$")
    a30 = re.compile(rf"^all2030_{mode_re}{suffix}$")
    none = re.compile(rf"^none_{mode_re}{suffix}$")

    out = []
    for year in years:
        ydir = base_dir / str(year)
        if not ydir.exists():
            continue

        for exp in sorted(ydir.iterdir()):
            if not exp.is_dir():
                continue

            m = ml.match(exp.name)
            if m and "multiloc" in scenarios:
                out.append({
                    "year": int(year),
                    "scenario": "multiloc",
                    "mode": m.group(1),
                    "location": m.group(2),
                    "scale_mw": int(m.group(3)),
                    "crossover": int(m.group(4) or 0),
                    "variant": m.group(5) or "base",
                    "path": exp,
                    "base_dir": str(base_dir),
                })
                continue

            m = a30.match(exp.name)
            if m and "all2030" in scenarios:
                out.append({
                    "year": int(year),
                    "scenario": "all2030",
                    "mode": m.group(1),
                    "location": None,
                    "scale_mw": None,
                    "crossover": int(m.group(2) or 0),
                    "variant": m.group(3) or "base",
                    "path": exp,
                    "base_dir": str(base_dir),
                })
                continue

            m = none.match(exp.name)
            if m and "none" in scenarios:
                out.append({
                    "year": int(year),
                    "scenario": "none",
                    "mode": m.group(1),
                    "location": None,
                    "scale_mw": None,
                    "crossover": int(m.group(2) or 0),
                    "variant": m.group(3) or "base",
                    "path": exp,
                    "base_dir": str(base_dir),
                })

    return out

def enumerate_experiments_multi(
    base_dirs: List[Path],
    years: Iterable[int],
    scenarios: Iterable[str],
    modes: Optional[Iterable[str]] = None,
) -> List[dict]:
    out = []
    seen = set()
    selected_modes = set(modes) if modes else None

    for bd in base_dirs:
        exps = enumerate_experiments(bd, years, scenarios)
        for e in exps:
            if selected_modes is not None and e["mode"] not in selected_modes:
                continue
            key = (
                e["year"],
                e["scenario"],
                e["mode"],
                e.get("location"),
                e.get("scale_mw"),
                str(e["path"]),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(e)

    return out

# =============================================================================
# Retail price
# =============================================================================

def get_texas_retail_energy_price(year: int) -> dict:
    monthly = {
        m: p for (y, m), p in TEXAS_RETAIL_PRICE_MONTHLY.items()
        if int(y) == int(year)
    }
    if not monthly:
        nearest_year = min(
            sorted({y for y, _ in TEXAS_RETAIL_PRICE_MONTHLY}),
            key=lambda y: abs(y - year),
        )
        monthly = {
            m: p for (y, m), p in TEXAS_RETAIL_PRICE_MONTHLY.items()
            if y == nearest_year
        }

    tdu_frac = TDU_FRACTION_BY_YEAR.get(year, 0.36)
    total_cents = float(np.mean(list(monthly.values())))
    energy_cents = total_cents * (1.0 - tdu_frac)

    return {
        "year": year,
        "total_retail_cents_kwh": total_cents,
        "tdu_fraction": tdu_frac,
        "energy_cents_kwh": energy_cents,
        "energy_dollar_mwh": energy_cents * 10.0,
    }


# =============================================================================
# PyPSA network helpers
# =============================================================================

def assign_ercot_zone(x: float, y: float) -> str:
    if x < -100.0:
        return "WEST"
    if y >= 32.0:
        return "NORTH"
    if x >= -96.5 and y < 30.5:
        return "HOUSTON"
    return "SOUTH"


def _snapshot_index(net) -> pd.DatetimeIndex:
    idx = net.snapshots
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values("timestep")
    return pd.DatetimeIndex(pd.to_datetime(idx))


def load_weights_by_bus(net, candidate_buses, include_dc: bool = False) -> Optional[pd.Series]:
    if not hasattr(net, "loads_t") or "p_set" not in net.loads_t:
        return None
    if net.loads_t.p_set.empty or net.loads.empty:
        return None

    load_ids = net.loads_t.p_set.columns.intersection(net.loads.index)
    if len(load_ids) == 0:
        return None

    if not include_dc:
        load_ids = [lid for lid in load_ids if not str(lid).startswith("DC_")]
        if len(load_ids) == 0:
            return None

    mean_load = net.loads_t.p_set[load_ids].mean()
    load_buses = net.loads.loc[mean_load.index, "bus"]
    bus_load = mean_load.groupby(load_buses).sum()

    common = pd.Index(candidate_buses).intersection(bus_load.index)
    if len(common) == 0:
        return None

    w = bus_load.loc[common].astype(float).clip(lower=0)
    if w.sum() <= 0:
        return None
    return w / w.sum()


def total_load_timeseries(net, include_dc: bool = True) -> pd.Series:
    if not hasattr(net, "loads_t") or "p_set" not in net.loads_t or net.loads_t.p_set.empty:
        idx = _snapshot_index(net)
        return pd.Series(np.nan, index=idx)

    p = net.loads_t.p_set.copy()
    if isinstance(p.index, pd.MultiIndex):
        p.index = pd.DatetimeIndex(p.index.get_level_values("timestep"))
    else:
        p.index = pd.DatetimeIndex(pd.to_datetime(p.index))

    if not include_dc:
        cols = [c for c in p.columns if not str(c).startswith("DC_")]
        p = p[cols]

    return p.sum(axis=1).sort_index()


def ercot_4cp_mw(net, include_dc: bool = True, fallback_to_peak: bool = True) -> float:
    """
    ERCOT 4CP approximation:
    average of system peak loads in June, July, August, and September.
    """
    load = total_load_timeseries(net, include_dc=include_dc).dropna()
    if load.empty:
        return np.nan

    cps = []
    for m in [6, 7, 8, 9]:
        s = load[load.index.month == m]
        if not s.empty:
            cps.append(float(s.max()))

    if cps:
        return float(np.mean(cps))

    return float(load.max()) if fallback_to_peak else np.nan


def annualized_load_mwh(net, hours_per_step: float = 3.0, include_dc: bool = True) -> float:
    load = total_load_timeseries(net, include_dc=include_dc).dropna()
    if load.empty:
        return np.nan

    horizon_mwh = float(load.sum() * hours_per_step)
    horizon_hours = float(len(load) * hours_per_step)
    if horizon_hours <= 0:
        return np.nan

    return horizon_mwh * 8760.0 / horizon_hours


def extract_all_zone_lmps_from_net(
    net,
    zones: List[str],
    clip_lower: float = -251.0,
    clip_upper: float = 9000.0,
    include_dc_in_weights: bool = False,
) -> Dict[str, pd.Series]:
    lmp = net.buses_t.marginal_price.copy()

    if isinstance(lmp.index, pd.MultiIndex):
        lmp.index = pd.DatetimeIndex(lmp.index.get_level_values("timestep"))
    else:
        lmp.index = pd.DatetimeIndex(pd.to_datetime(lmp.index))

    lmp = lmp.clip(lower=clip_lower, upper=clip_upper)

    buses = net.buses.copy()
    bus_zones = buses.apply(lambda r: assign_ercot_zone(r["x"], r["y"]), axis=1)

    out: Dict[str, pd.Series] = {}

    for zone in zones:
        z = zone.upper()
        if z == "SYSTEM":
            candidate = lmp.columns.intersection(buses.index)
        else:
            candidate = bus_zones[bus_zones == z].index.intersection(lmp.columns)

        if len(candidate) == 0:
            out[zone] = lmp.mean(axis=1)
            continue

        weights = load_weights_by_bus(net, candidate, include_dc=include_dc_in_weights)
        if weights is not None and len(weights) > 0:
            common = weights.index.intersection(lmp.columns)
            out[zone] = (lmp[common] * weights.loc[common]).sum(axis=1)
        else:
            out[zone] = lmp[candidate].mean(axis=1)

    return out


def read_new_capacity(exp_path: Path) -> pd.DataFrame:
    p = exp_path / "new_capacity.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if "carrier" not in df.columns and df.index.name == "carrier":
            df = df.reset_index()
        return df
    except Exception:
        return pd.DataFrame()


def compute_investment_from_new_capacity(exp_path: Path) -> dict:
    """
    Transmission is allocated through 4CP.
    Generation/storage investment is retained as diagnostic only because IPP
    costs enter the REP through LMP and forward prices.
    """
    df = read_new_capacity(exp_path)

    total_capex = 0.0
    tx_capex = 0.0
    gen_storage_capex = 0.0

    if not df.empty and "annual_investment" in df.columns:
        inv = pd.to_numeric(df["annual_investment"], errors="coerce").fillna(0.0)
        total_capex = float(inv.sum())

        if "carrier" in df.columns:
            carrier = df["carrier"].astype(str).str.lower()
            tx_mask = carrier.str.contains("transmission|line|link", regex=True, na=False)
            storage_mask = carrier.str.contains("battery|storage", regex=True, na=False)
            tx_mask = tx_mask & ~storage_mask
            tx_capex = float(inv[tx_mask].sum())
            gen_storage_capex = float(inv[~tx_mask].sum())
        else:
            gen_storage_capex = total_capex

    metrics_path = exp_path / "metrics.csv"
    if metrics_path.exists() and total_capex == 0.0:
        try:
            m = pd.read_csv(metrics_path).iloc[0]
            total_capex = float(m.get("total_annual_investment", 0.0) or 0.0)
            gen_storage_capex = total_capex
        except Exception:
            pass

    return {
        "total_capex_annual_$": total_capex,
        "tx_capex_annual_$": tx_capex,
        "gen_storage_capex_annual_$": gen_storage_capex,
    }


def compute_4cp_tcos_adder(
    tx_capex_annual: float,
    ercot_4cp_mw_value: float,
    lse_load_mw: float,
    lse_4cp_mw: Optional[float] = None,
) -> dict:
    if lse_4cp_mw is None:
        lse_4cp_mw = lse_load_mw

    lse_annual_mwh = lse_load_mw * 8760.0

    if not np.isfinite(ercot_4cp_mw_value) or ercot_4cp_mw_value <= 0 or lse_annual_mwh <= 0:
        return {
            "ercot_4cp_mw": ercot_4cp_mw_value,
            "lse_4cp_mw": lse_4cp_mw,
            "lse_4cp_share": np.nan,
            "lse_tcos_assigned_annual_$": np.nan,
            "tcos_4cp_adder_$/MWh": np.nan,
        }

    share = float(lse_4cp_mw / ercot_4cp_mw_value)
    assigned = float(tx_capex_annual * share)
    adder = float(assigned / lse_annual_mwh)

    return {
        "ercot_4cp_mw": float(ercot_4cp_mw_value),
        "lse_4cp_mw": float(lse_4cp_mw),
        "lse_4cp_share": share,
        "lse_tcos_assigned_annual_$": assigned,
        "tcos_4cp_adder_$/MWh": adder,
    }


def read_failure_flag(exp_path: Path, threshold: float = 1e6) -> bool:
    p = exp_path / "metrics.csv"
    if not p.exists():
        return False
    try:
        row = pd.read_csv(p).iloc[0]
        return float(row.get("load_shedding_cost", 0.0) or 0.0) > threshold
    except Exception:
        return False


# =============================================================================
# Cache
# =============================================================================

def build_cache(
    exps: List[dict],
    zones: List[str],
    cache_path: Path,
    lse_load_mw: float,
    lse_4cp_mw: Optional[float],
    include_dc_in_4cp_denominator: bool,
    include_dc_in_lmp_weights: bool,
    hours_per_step: float = 3.0,
    lmp_clip_lower: float = -251.0,
    lmp_clip_upper: float = 9000.0,
    metadata: Optional[dict] = None,
) -> List[dict]:
    import pypsa

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()

    for i, e in enumerate(exps, 1):
        nc = e["path"] / "network.nc"
        if not nc.exists():
            print(f"  [SKIP] missing network.nc: {e['path']}")
            continue

        try:
            net = pypsa.Network(str(nc))

            lmp_by_zone = extract_all_zone_lmps_from_net(
                net,
                zones,
                clip_lower=lmp_clip_lower,
                clip_upper=lmp_clip_upper,
                include_dc_in_weights=include_dc_in_lmp_weights,
            )
            inv = compute_investment_from_new_capacity(e["path"])

            ercot4cp = ercot_4cp_mw(
                net,
                include_dc=include_dc_in_4cp_denominator,
                fallback_to_peak=True,
            )

            tcos = compute_4cp_tcos_adder(
                tx_capex_annual=inv["tx_capex_annual_$"],
                ercot_4cp_mw_value=ercot4cp,
                lse_load_mw=lse_load_mw,
                lse_4cp_mw=lse_4cp_mw,
            )

            annual_load_mwh_system = annualized_load_mwh(
                net,
                hours_per_step=hours_per_step,
                include_dc=include_dc_in_4cp_denominator,
            )

            row = {
                "year": e["year"],
                "scenario": e["scenario"],
                "mode": e["mode"],
                "location": e.get("location"),
                "scale_mw": e.get("scale_mw"),
                "path": str(e["path"]),
                "system_failure": read_failure_flag(e["path"]),
                "lmp_by_zone": lmp_by_zone,
                "mean_lmp_by_zone": {z: float(s.mean()) for z, s in lmp_by_zone.items()},
                "p95_lmp_by_zone": {z: float(np.percentile(s, 95)) for z, s in lmp_by_zone.items()},
                "p99_lmp_by_zone": {z: float(np.percentile(s, 99)) for z, s in lmp_by_zone.items()},
                "investment": inv,
                "tcos_4cp": tcos,
                "annual_load_mwh_system": annual_load_mwh_system,
            }
            rows.append(row)

        except Exception as exc:
            print(f"  [WARN] failed {e['path'].name}: {exc}")

        if i % 20 == 0 or i == len(exps):
            elapsed = (time.time() - t0) / 60
            print(f"    [{i}/{len(exps)}] cached, elapsed={elapsed:.1f} min")

    with open(cache_path, "wb") as f:
        pickle.dump({
            "metadata": metadata,
            "zones": zones,
            "lse_load_mw": lse_load_mw,
            "lse_4cp_mw": lse_4cp_mw,
            "include_dc_in_4cp_denominator": include_dc_in_4cp_denominator,
            "include_dc_in_lmp_weights": include_dc_in_lmp_weights,
            "hours_per_step": hours_per_step,
            "rows": rows,
        }, f)

    print(f"Saved cache: {cache_path} ({len(rows)} experiments)")
    return rows


def load_or_build_cache(
    exps: List[dict],
    zones: List[str],
    cache_path: Path,
    refresh_cache: bool,
    lse_load_mw: float,
    lse_4cp_mw: Optional[float],
    include_dc_in_4cp_denominator: bool,
    include_dc_in_lmp_weights: bool,
    hours_per_step: float,
    lmp_clip_lower: float = -251.0,
    lmp_clip_upper: float = 9000.0,
    metadata: Optional[dict] = None,
) -> List[dict]:
    if cache_path.exists() and not refresh_cache:
        print(f"Using cache: {cache_path}")
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        rows = obj.get("rows", [])
        if obj.get("metadata") == metadata:
            print(f"Loaded cache with {len(rows)} experiments")
            return rows
        print("  cache metadata mismatch; rebuilding")

    print("Building REP default simulation cache from existing network.nc files ...")
    return build_cache(
        exps=exps,
        zones=zones,
        cache_path=cache_path,
        lse_load_mw=lse_load_mw,
        lse_4cp_mw=lse_4cp_mw,
        include_dc_in_4cp_denominator=include_dc_in_4cp_denominator,
        include_dc_in_lmp_weights=include_dc_in_lmp_weights,
        hours_per_step=hours_per_step,
        lmp_clip_lower=lmp_clip_lower,
        lmp_clip_upper=lmp_clip_upper,
        metadata=metadata,
    )


def cache_metadata(
    exps: List[dict],
    base_dirs: List[Path],
    years: Iterable[int],
    scenarios: Iterable[str],
    modes: Optional[Iterable[str]],
    zones: Iterable[str],
    lse_load_mw: float,
    lse_4cp_mw: Optional[float],
    include_dc_in_4cp_denominator: bool,
    include_dc_in_lmp_weights: bool,
    hours_per_step: float,
    lmp_clip_lower: float,
    lmp_clip_upper: float,
) -> dict:
    """Return the exact inputs that determine pre-simulation cache rows."""
    return {
        "version": 2,
        "base_dirs": [str(p) for p in base_dirs],
        "years": list(years),
        "scenarios": list(scenarios),
        "modes": list(modes) if modes is not None else None,
        "zones": list(zones),
        "scales": sorted({e["scale_mw"] for e in exps if e.get("scale_mw") is not None}),
        "locations": sorted({e["location"] for e in exps if e.get("location") is not None}),
        "experiments": [
            {
                "year": e["year"], "scenario": e["scenario"], "mode": e["mode"],
                "location": e.get("location"), "scale_mw": e.get("scale_mw"),
                "path": str(e["path"].resolve()),
            }
            for e in exps
        ],
        "lse_load_mw": lse_load_mw,
        "lse_4cp_mw": lse_4cp_mw,
        "include_dc_in_4cp_denominator": include_dc_in_4cp_denominator,
        "include_dc_in_lmp_weights": include_dc_in_lmp_weights,
        "hours_per_step": hours_per_step,
        "lmp_clip_lower": lmp_clip_lower,
        "lmp_clip_upper": lmp_clip_upper,
    }


# =============================================================================
# Simulation model
# =============================================================================

def build_base_forward_refs(rows: List[dict], zone: str) -> Dict[Tuple[int, str], float]:
    refs = {}

    none_rows = [r for r in rows if r["scenario"] == "none"]
    for r in none_rows:
        refs[(int(r["year"]), str(r["mode"]))] = float(r["mean_lmp_by_zone"][zone])

    years = sorted({int(r["year"]) for r in rows})
    # Cover every mode present in the data, not just the static MODES list;
    # otherwise base_forward silently falls back to the scenario's own mean
    # LMP for missing (year, mode) keys, duplicating scenario_repriced.
    modes_in_rows = {str(r["mode"]) for r in rows} | set(MODES)
    for y in years:
        available = [r for r in none_rows if int(r["year"]) == y]
        if not available:
            continue

        dispatch = [r for r in available if r["mode"] == "dispatch"]
        fallback = dispatch[0] if dispatch else available[0]

        for mode in modes_in_rows:
            refs.setdefault((y, mode), float(fallback["mean_lmp_by_zone"][zone]))

    return refs


def truncated_normal(rng, mean, sd, low, high):
    val = rng.normal(mean, sd)
    return float(np.clip(val, low, high))


def bootstrap_lmp_blocks(rng, lmp: np.ndarray, block_steps: int) -> np.ndarray:
    """
    Circular block bootstrap preserving short-run autocorrelation.
    """
    n = len(lmp)
    if n == 0 or block_steps <= 1:
        return lmp.copy()

    out = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        idx = (start + np.arange(block_steps)) % n
        out.extend(lmp[idx].tolist())
    return np.asarray(out[:n], dtype=float)


def simulate_one_path(
    rng,
    lmp_series: pd.Series,
    retail_energy_price: float,
    forward_ref_lmp: float,
    tcos_adder_mwh: float,
    lse_load_mw: float,
    hours_per_step: float,
    base_capital: float,
    base_hedge: float,
    base_forward_premium: float,
    base_opex_mwh: float,
    base_pass_through: float,
    capital_sigma_frac: float,
    hedge_sigma: float,
    forward_premium_sigma: float,
    retail_sigma_frac: float,
    opex_sigma_frac: float,
    passthrough_sigma: float,
    lmp_scale_sigma_frac: float,
    block_bootstrap_steps: int,
    include_collateral: bool,
    collateral_window_hours: float,
) -> dict:
    lmp = lmp_series.dropna().values.astype(float)
    if block_bootstrap_steps > 1:
        lmp = bootstrap_lmp_blocks(rng, lmp, block_bootstrap_steps)

    lmp_scale = max(0.05, rng.lognormal(mean=-0.5 * lmp_scale_sigma_frac**2, sigma=lmp_scale_sigma_frac))
    lmp = lmp * lmp_scale

    capital = max(1e6, rng.lognormal(mean=np.log(base_capital) - 0.5 * capital_sigma_frac**2,
                                      sigma=capital_sigma_frac))

    h = truncated_normal(rng, base_hedge, hedge_sigma, 0.0, 1.0)
    fp = truncated_normal(rng, base_forward_premium, forward_premium_sigma, -20.0, 50.0)
    rho = truncated_normal(rng, base_pass_through, passthrough_sigma, 0.0, 1.0)

    retail_shock = rng.normal(0.0, retail_sigma_frac)
    opex_shock = rng.normal(0.0, opex_sigma_frac)

    G_eff = retail_energy_price * (1.0 + retail_shock) + rho * tcos_adder_mwh
    opex_eff = base_opex_mwh * (1.0 + opex_shock) + (1.0 - rho) * tcos_adder_mwh

    F = forward_ref_lmp + fp
    procurement = h * F + (1.0 - h) * lmp
    margin = G_eff - procurement - opex_eff

    pnl = margin * lse_load_mw * hours_per_step
    cash = capital + np.cumsum(pnl)

    # Optional liquidity call: additional short-run cash requirement based on
    # rolling procurement outflow. This is not double-counted as energy cost;
    # it is a liquidity stress buffer.
    max_liquidity_call = 0.0
    if include_collateral:
        window = max(1, int(collateral_window_hours / hours_per_step))
        outflow = procurement * lse_load_mw * hours_per_step
        roll = pd.Series(outflow).rolling(window, min_periods=1).sum().values
        max_liquidity_call = float(np.max(roll))
        cash_after_liquidity = cash - max_liquidity_call
        default = bool(np.min(cash_after_liquidity) < 0)
        min_cash = float(np.min(cash_after_liquidity))
    else:
        default = bool(np.min(cash) < 0)
        min_cash = float(np.min(cash))

    return {
        "default": float(default),
        "min_cash_$": min_cash,
        "ending_cash_$": float(cash[-1]) if len(cash) else capital,
        "annual_profit_$": float(np.sum(pnl) * 8760.0 / (len(lmp) * hours_per_step)) if len(lmp) else np.nan,
        "mean_margin_$/MWh": float(np.mean(margin)) if len(margin) else np.nan,
        "p05_margin_$/MWh": float(np.percentile(margin, 5)) if len(margin) else np.nan,
        "share_negative_margin_hours": float(np.mean(margin < 0)) if len(margin) else np.nan,
        "mean_procurement_$/MWh": float(np.mean(procurement)) if len(procurement) else np.nan,
        "p99_lmp_$/MWh": float(np.percentile(lmp, 99)) if len(lmp) else np.nan,
        "capital_$": float(capital),
        "hedge_ratio": float(h),
        "forward_premium_$/MWh": float(fp),
        "forward_price_$/MWh": float(F),
        "retail_pass_through": float(rho),
        "retail_shock_frac": float(retail_shock),
        "opex_shock_frac": float(opex_shock),
        "lmp_scale": float(lmp_scale),
        "max_liquidity_call_$": float(max_liquidity_call),
    }


def scenario_key(row: dict) -> Tuple:
    return (
        row["scenario"],
        row["mode"],
        row.get("location"),
        row.get("scale_mw"),
    )


def run_default_simulation_grouped(
    rows: List[dict],
    zone: str,
    forward_case: str,
    n_sims: int,
    seed: int,
    lse_load_mw: float,
    hours_per_step: float,
    base_capital: float,
    base_hedge: float,
    base_forward_premium: float,
    base_opex_mwh: float,
    base_pass_through: float,
    capital_sigma_frac: float,
    hedge_sigma: float,
    forward_premium_sigma: float,
    retail_sigma_frac: float,
    opex_sigma_frac: float,
    passthrough_sigma: float,
    lmp_scale_sigma_frac: float,
    block_bootstrap_steps: int,
    include_collateral: bool,
    collateral_window_hours: float,
    skip_system_failure: bool,
    full_tx_tcos_adder_mwh: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    refs = build_base_forward_refs(rows, zone)

    years = sorted({int(r["year"]) for r in rows})
    retail_by_year = {
        y: get_texas_retail_energy_price(y)["energy_dollar_mwh"]
        for y in years
    }

    groups: Dict[Tuple, List[dict]] = {}
    for r in rows:
        if zone not in r["lmp_by_zone"]:
            continue
        if skip_system_failure and r.get("system_failure", False):
            continue
        groups.setdefault(scenario_key(r), []).append(r)

    raw_rows = []
    summary_rows = []

    for gidx, (key, candidates) in enumerate(sorted(groups.items(), key=lambda x: str(x[0])), 1):
        scen, mode, loc, scale = key
        if len(candidates) == 0:
            continue

        sim_results = []
        for sim in range(n_sims):
            r = candidates[int(rng.integers(0, len(candidates)))]
            year = int(r["year"])
            lmp = r["lmp_by_zone"][zone]

            if forward_case == "base_forward":
                ref_lmp = refs.get((year, str(r["mode"])), float(lmp.mean()))
            elif forward_case == "scenario_repriced":
                ref_lmp = float(lmp.mean())
            else:
                raise ValueError(forward_case)

            tcos_adder = float(r["tcos_4cp"].get("tcos_4cp_adder_$/MWh", 0.0) or 0.0)
            if (
                full_tx_tcos_adder_mwh is not None
                and scen == "all2030"
                and mode == "full_tx"
            ):
                tcos_adder = float(full_tx_tcos_adder_mwh)

            res = simulate_one_path(
                rng=rng,
                lmp_series=lmp,
                retail_energy_price=retail_by_year[year],
                forward_ref_lmp=ref_lmp,
                tcos_adder_mwh=tcos_adder,
                lse_load_mw=lse_load_mw,
                hours_per_step=hours_per_step,
                base_capital=base_capital,
                base_hedge=base_hedge,
                base_forward_premium=base_forward_premium,
                base_opex_mwh=base_opex_mwh,
                base_pass_through=base_pass_through,
                capital_sigma_frac=capital_sigma_frac,
                hedge_sigma=hedge_sigma,
                forward_premium_sigma=forward_premium_sigma,
                retail_sigma_frac=retail_sigma_frac,
                opex_sigma_frac=opex_sigma_frac,
                passthrough_sigma=passthrough_sigma,
                lmp_scale_sigma_frac=lmp_scale_sigma_frac,
                block_bootstrap_steps=block_bootstrap_steps,
                include_collateral=include_collateral,
                collateral_window_hours=collateral_window_hours,
            )
            res.update({
                "sim": sim,
                "sampled_year": year,
                "scenario": scen,
                "mode": mode,
                "location": loc,
                "scale_mw": scale,
                "zone": zone,
                "forward_case": forward_case,
                "sampled_system_failure": bool(r.get("system_failure", False)),
                "sampled_tcos_4cp_adder_$/MWh": tcos_adder,
                "sampled_tx_capex_annual_$": float(r["investment"].get("tx_capex_annual_$", 0.0)),
                "sampled_ercot_4cp_mw": float(r["tcos_4cp"].get("ercot_4cp_mw", np.nan)),
                "sampled_lse_4cp_share": float(r["tcos_4cp"].get("lse_4cp_share", np.nan)),
            })
            sim_results.append(res)

        raw = pd.DataFrame(sim_results)
        raw_rows.append(raw)

        summary_rows.append({
            "scenario": scen,
            "mode": mode,
            "location": loc,
            "scale_mw": scale,
            "zone": zone,
            "forward_case": forward_case,
            "n_weather_years_available": len(candidates),
            "n_sims": n_sims,
            "default_probability": raw["default"].mean(),
            "default_probability_pct": 100.0 * raw["default"].mean(),
            "mean_margin_$/MWh": raw["mean_margin_$/MWh"].mean(),
            "p05_margin_$/MWh": raw["p05_margin_$/MWh"].mean(),
            "mean_procurement_$/MWh": raw["mean_procurement_$/MWh"].mean(),
            "mean_tcos_4cp_adder_$/MWh": raw["sampled_tcos_4cp_adder_$/MWh"].mean(),
            "mean_annual_profit_$": raw["annual_profit_$"].mean(),
            "p05_min_cash_$": np.percentile(raw["min_cash_$"], 5),
            "mean_min_cash_$": raw["min_cash_$"].mean(),
            "mean_ending_cash_$": raw["ending_cash_$"].mean(),
            "share_negative_margin_hours": raw["share_negative_margin_hours"].mean(),
            "mean_forward_price_$/MWh": raw["forward_price_$/MWh"].mean(),
            "mean_hedge_ratio": raw["hedge_ratio"].mean(),
            "mean_pass_through": raw["retail_pass_through"].mean(),
            "mean_capital_$": raw["capital_$"].mean(),
            "mean_lse_4cp_share": raw["sampled_lse_4cp_share"].mean(),
            "mean_tx_capex_annual_$": raw["sampled_tx_capex_annual_$"].mean(),
        })

        print(
            f"    [{gidx}/{len(groups)}] {scen}/{mode}/{loc}/{scale}: "
            f"default={summary_rows[-1]['default_probability_pct']:.1f}%"
        )

    raw_df = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, raw_df


# =============================================================================
# Figures
# =============================================================================

def savefig(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def plot_default_all2030(summary: pd.DataFrame, outdir: Path, zone: str, forward_case: str):
    """
    Main REP default figure.

    Important:
    - Dispatch-only all2030 is an UNSERVED power-system regime.
      It is annotated but not plotted on the same numeric scale.
    - Only expansion and expansion_tx are plotted as normal retail-market outcomes.
    """
    sub = summary[summary["scenario"] == "all2030"].copy()
    if sub.empty:
        return

    preferred = ["generation", "generation_tx", "transmission", "full_tx", "expansion", "expansion_tx"]
    plot_modes = [m for m in preferred if m in set(sub["mode"])]

    if not plot_modes:
        return

    x = np.arange(len(plot_modes))
    colors = [MODE_COLORS[m] for m in plot_modes]
    labels = [MODE_LABELS[m] for m in plot_modes]

    default_vals = [
        float(sub[sub["mode"] == m]["default_probability_pct"].mean())
        for m in plot_modes
    ]
    margin_vals = [
        float(sub[sub["mode"] == m]["mean_margin_$/MWh"].mean())
        for m in plot_modes
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(8.4, 3.8),
        gridspec_kw={"wspace": 0.35}
    )

    # Panel a: default probability
    bars = axes[0].bar(x, default_vals, color=colors, width=0.62)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_ylabel("Simulated default probability (%)")
    axes[0].set_title("Default risk")

    ymax = max(default_vals) * 1.35 if max(default_vals) > 0 else 1.0
    ymax = max(ymax, 1.0)
    axes[0].set_ylim(0, ymax)

    for bar, val in zip(bars, default_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + ymax * 0.035,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=8
        )

    axes[0].text(
        -0.55, ymax * 0.72,
        "Dispatch only:\nUNSERVED\nnot interpreted",
        ha="center", va="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="#eeeeee", ec="#888888", lw=0.8)
    )

    # Panel b: mean margin
    bars = axes[1].bar(x, margin_vals, color=colors, width=0.62)
    axes[1].axhline(0, color="black", ls="--", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].set_ylabel("Mean margin ($/MWh)")
    axes[1].set_title("Mean margin")

    ymin = min(0, min(margin_vals) * 1.2)
    ymax2 = max(margin_vals) * 1.25 if max(margin_vals) > 0 else 1.0
    axes[1].set_ylim(ymin, ymax2)

    for bar, val in zip(bars, margin_vals):
        y = val + (ymax2 - ymin) * 0.035
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8
        )

    axes[1].text(
        -0.55, ymin + (ymax2 - ymin) * 0.72,
        "Dispatch only:\nUNSERVED\nnot interpreted",
        ha="center", va="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="#eeeeee", ec="#888888", lw=0.8)
    )

    for ax, lab in zip(axes, "ab"):
        ax.text(
            -0.12, 1.08, lab,
            transform=ax.transAxes,
            fontsize=12, fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.25)

    title_map = {
        "base_forward": "Base-forward hedging",
        "scenario_repriced": "Scenario-repriced forward contracts",
    }
    fig.suptitle(
        f"{title_map.get(forward_case, forward_case)}, {zone}",
        y=1.04, fontsize=11
    )

    savefig(fig, outdir, f"fig_rep_default_all2030_{zone}_{forward_case}")

def plot_tcos_4cp(summary: pd.DataFrame, outdir: Path, zone: str, forward_case: str):
    sub = summary[summary["scenario"] == "all2030"].copy()
    if sub.empty:
        return

    modes = [m for m in MODES if m in set(sub["mode"])]
    x = np.arange(len(modes))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), gridspec_kw={"wspace": 0.35})

    vals = [sub[sub["mode"] == m]["mean_tx_capex_annual_$"].mean() / 1e9 for m in modes]
    axes[0].bar(x, vals, color=[MODE_COLORS[m] for m in modes])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODE_LABELS[m] for m in modes], rotation=20, ha="right")
    axes[0].set_ylabel("New transmission annual investment ($B/yr)")
    axes[0].set_title("TSP-built transmission")

    vals2 = [sub[sub["mode"] == m]["mean_tcos_4cp_adder_$/MWh"].mean() for m in modes]
    axes[1].bar(x, vals2, color=[MODE_COLORS[m] for m in modes])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODE_LABELS[m] for m in modes], rotation=20, ha="right")
    axes[1].set_ylabel("4CP TCOS adder ($/MWh)")
    axes[1].set_title("4CP-allocated REP transmission cost")

    for ax, lab in zip(axes, "ab"):
        ax.text(-0.12, 1.08, lab, transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Transmission cost allocation, {zone}, {forward_case}", y=1.04, fontsize=11)
    savefig(fig, outdir, f"fig_rep_default_tcos_4cp_{zone}_{forward_case}")


def plot_margin_scale(summary: pd.DataFrame, outdir: Path, zone: str, forward_case: str):
    sub = summary[summary["scenario"] == "multiloc"].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), gridspec_kw={"wspace": 0.35})

    for mode in MODES:
        vals = []
        errs = []
        defs = []
        for scale in SCALES:
            cell = sub[(sub["mode"] == mode) & (sub["scale_mw"] == scale)]
            vals.append(cell["mean_margin_$/MWh"].mean())
            errs.append(cell["mean_margin_$/MWh"].std())
            defs.append(cell["default_probability_pct"].mean())
        axes[0].errorbar([s/1000 for s in SCALES], vals, yerr=errs,
                         fmt="o-", color=MODE_COLORS[mode], label=MODE_LABELS[mode], capsize=3)
        axes[1].plot([s/1000 for s in SCALES], defs,
                     "o-", color=MODE_COLORS[mode], label=MODE_LABELS[mode])

    axes[0].axhline(0, color="black", ls="--", lw=0.8)
    axes[0].set_xlabel("Data-center load (GW)")
    axes[0].set_ylabel("Mean margin ($/MWh)")
    axes[0].set_title("REP margin across siting experiments")

    axes[1].set_xlabel("Data-center load (GW)")
    axes[1].set_ylabel("Default probability (%)")
    axes[1].set_title("Monte Carlo default probability")
    axes[1].legend()

    for ax, lab in zip(axes, "ab"):
        ax.text(-0.12, 1.08, lab, transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)

    fig.suptitle(f"REP default simulation by data-center scale, {zone}, {forward_case}", y=1.04, fontsize=11)
    savefig(fig, outdir, f"fig_rep_default_margin_scale_{zone}_{forward_case}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--years", nargs="*", type=int, default=ALL_YEARS)
    ap.add_argument("--scenarios", nargs="*", default=["none", "multiloc", "all2030"])
    ap.add_argument(
        "--modes",
        nargs="*",
        default=None,
        help="Optional mode filter, for example: dispatch full_tx.",
    )
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--base-dirs",
        nargs="*",
        type=Path,
        default=None,
        help="Multiple result roots. If set, overrides --base-dir.",
    )
    input_group.add_argument(
        "--base-dir",
        type=Path,
        help="Experiment result root.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--zones", nargs="*", default=["system"], help="system WEST NORTH SOUTH HOUSTON")
    ap.add_argument("--forward-case", choices=["base_forward", "scenario_repriced", "both"], default="both")

    ap.add_argument("--cache-path", type=Path,
                    help="Optional cache file. Defaults below --out-dir.")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--build-cache-only", action="store_true")
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Write compact summary CSV files without large raw simulation CSV files.",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG/PDF figure generation.",
    )

    # REP base assumptions
    ap.add_argument("--rep-load", type=float, default=1000.0, help="Flat REP served load in MW")
    ap.add_argument("--rep-4cp-mw", type=float, default=None, help="REP 4CP MW. Default equals --rep-load.")
    ap.add_argument("--rep-capital", type=float, default=50e6)
    ap.add_argument("--hedge-ratio", type=float, default=0.70)
    ap.add_argument("--forward-premium", type=float, default=5.0)
    ap.add_argument("--rep-opex", type=float, default=3.0)
    ap.add_argument("--retail-pass-through", type=float, default=0.0)
    ap.add_argument("--hours-per-step", type=float, default=3.0)

    # Monte Carlo controls
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--capital-sigma-frac", type=float, default=0.30)
    ap.add_argument("--hedge-sigma", type=float, default=0.10)
    ap.add_argument("--forward-premium-sigma", type=float, default=3.0)
    ap.add_argument("--retail-sigma-frac", type=float, default=0.05)
    ap.add_argument("--opex-sigma-frac", type=float, default=0.20)
    ap.add_argument("--passthrough-sigma", type=float, default=0.10)
    ap.add_argument("--lmp-scale-sigma-frac", type=float, default=0.10)
    ap.add_argument("--block-bootstrap-steps", type=int, default=16, help="16 three-hour steps = 48h blocks")
    ap.add_argument("--include-collateral", action="store_true")
    ap.add_argument("--collateral-window-hours", type=float, default=48.0)
    ap.add_argument("--skip-system-failure", action="store_true",
                    help="Do not simulate scenarios already marked UNSERVED by the power-system model.")

    # Offer-cap clipping applied to the LMP series before simulation.
    # 9000 reproduces the previously hard-coded behaviour; 5000 matches the
    # ERCOT system-wide offer cap used by the capped-adder analysis.
    ap.add_argument("--lmp-clip-upper", type=float, default=9000.0)
    ap.add_argument("--lmp-clip-lower", type=float, default=-251.0)

    # Cost allocation controls
    ap.add_argument("--exclude-dc-from-4cp-denominator", action="store_true")
    ap.add_argument("--include-dc-in-lmp-weights", action="store_true")
    ap.add_argument(
        "--full-tx-tcos-adder-mwh",
        type=float,
        default=None,
        help=(
            "Override the all2030/full_tx TCOS adder with an external engineering "
            "estimate from the engineering study. Baseline cases "
            "remain unchanged."
        ),
    )

    args = ap.parse_args()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_path = (
        args.cache_path.expanduser().resolve()
        if args.cache_path is not None
        else args.out_dir / "rep_default_sim_cache.pkl"
    )
    if args.base_dirs is not None:
        args.base_dirs = [p.expanduser().resolve() for p in args.base_dirs]
    else:
        args.base_dir = args.base_dir.expanduser().resolve()

    zones = [z if z == "system" else z.upper() for z in args.zones]
    forward_cases = ["base_forward", "scenario_repriced"] if args.forward_case == "both" else [args.forward_case]

    print("=" * 78)
    print("REP Monte Carlo default simulation with 4CP TCOS allocation")
    print("=" * 78)
    base_dirs = args.base_dirs if args.base_dirs else [args.base_dir]
    print(f"base dirs:      {[str(p) for p in base_dirs]}")
    print(f"zones:          {zones}")
    print(f"forward case:   {args.forward_case}")
    print(f"n sims/group:   {args.n_sims}")
    print(f"REP load:       {args.rep_load:.1f} MW")
    print(f"REP 4CP:        {args.rep_4cp_mw if args.rep_4cp_mw is not None else args.rep_load:.1f} MW")
    print(f"capital mean:   ${args.rep_capital/1e6:.1f} M")
    print(f"hedge mean:     {args.hedge_ratio}")
    print(f"premium mean:   ${args.forward_premium:.2f}/MWh")
    print(f"pass-through:   {args.retail_pass_through:.0%}")
    print(f"cache:          {args.cache_path}")
    print("=" * 78)

    exps = enumerate_experiments_multi(base_dirs, args.years, args.scenarios, args.modes)
    print(f"Found {len(exps)} experiments")
    if not exps:
        return

    rows = load_or_build_cache(
        exps=exps,
        zones=zones,
        cache_path=args.cache_path,
        refresh_cache=args.refresh_cache,
        lse_load_mw=args.rep_load,
        lse_4cp_mw=args.rep_4cp_mw,
        include_dc_in_4cp_denominator=not args.exclude_dc_from_4cp_denominator,
        include_dc_in_lmp_weights=args.include_dc_in_lmp_weights,
        hours_per_step=args.hours_per_step,
        lmp_clip_lower=args.lmp_clip_lower,
        lmp_clip_upper=args.lmp_clip_upper,
        metadata=cache_metadata(
            exps, base_dirs, args.years, args.scenarios, args.modes, zones,
            args.rep_load, args.rep_4cp_mw,
            not args.exclude_dc_from_4cp_denominator,
            args.include_dc_in_lmp_weights, args.hours_per_step,
            args.lmp_clip_lower, args.lmp_clip_upper,
        ),
    )

    if args.build_cache_only:
        print("Cache built. Exiting because --build-cache-only was set.")
        return

    for zone in zones:
        combo_summaries = []
        combo_raw = []

        for fcase in forward_cases:
            print("\n" + "=" * 78)
            print(f"zone={zone}, forward={fcase}")
            print("=" * 78)

            summary, raw = run_default_simulation_grouped(
                rows=rows,
                zone=zone,
                forward_case=fcase,
                n_sims=args.n_sims,
                seed=(
                    args.seed
                    + zones.index(zone) * 1000
                    + forward_cases.index(fcase) * 100
                ),
                lse_load_mw=args.rep_load,
                hours_per_step=args.hours_per_step,
                base_capital=args.rep_capital,
                base_hedge=args.hedge_ratio,
                base_forward_premium=args.forward_premium,
                base_opex_mwh=args.rep_opex,
                base_pass_through=args.retail_pass_through,
                capital_sigma_frac=args.capital_sigma_frac,
                hedge_sigma=args.hedge_sigma,
                forward_premium_sigma=args.forward_premium_sigma,
                retail_sigma_frac=args.retail_sigma_frac,
                opex_sigma_frac=args.opex_sigma_frac,
                passthrough_sigma=args.passthrough_sigma,
                lmp_scale_sigma_frac=args.lmp_scale_sigma_frac,
                block_bootstrap_steps=args.block_bootstrap_steps,
                include_collateral=args.include_collateral,
                collateral_window_hours=args.collateral_window_hours,
                skip_system_failure=args.skip_system_failure,
                full_tx_tcos_adder_mwh=args.full_tx_tcos_adder_mwh,
            )

            suffix = f"{zone}_{fcase}"
            summary_path = args.out_dir / f"rep_default_sim_summary_{suffix}.csv"
            raw_path = args.out_dir / f"rep_default_sim_raw_{suffix}.csv"

            summary.to_csv(summary_path, index=False)
            if not args.summary_only:
                raw.to_csv(raw_path, index=False)

            print(f"Saved {summary_path}")
            if not args.summary_only:
                print(f"Saved {raw_path}")

            # Compact console summary for all2030
            a30 = summary[summary["scenario"] == "all2030"]
            if not a30.empty:
                print("\nAll2030 summary:")
                for mode in MODES:
                    cell = a30[a30["mode"] == mode]
                    if cell.empty:
                        continue
                    print(
                        f"  {mode:12s} default={cell['default_probability_pct'].mean():6.1f}%  "
                        f"margin={cell['mean_margin_$/MWh'].mean():+8.2f}  "
                        f"TCOS4CP={cell['mean_tcos_4cp_adder_$/MWh'].mean():6.2f}"
                    )

            if not args.no_plots:
                plot_default_all2030(summary, args.out_dir, zone, fcase)

            combo_summaries.append(summary)
            combo_raw.append(raw)

        if len(combo_summaries) > 1:
            both = pd.concat(combo_summaries, ignore_index=True)
            out = args.out_dir / f"rep_default_sim_summary_{zone}_both.csv"
            both.to_csv(out, index=False)
            print(f"Saved combined summary: {out}")

    print("\nDone.")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
