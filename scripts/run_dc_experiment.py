"""
run_dc_experiment.py
====================
Run a single DC impact experiment on a starting network.

Three DC scenarios:
  --dc-scenario none       — no DC, just re-solve the starting network
  --dc-scenario multiloc   — single site at one location, one capacity
  --dc-scenario all2030    — load every ERCOT-mapped TX 2030 planned DC from xlsx

Build modes:
  --mode dispatch       — lock everything, pure re-dispatch
  --mode generation     — generation extendable, storage locked
  --mode generation_storage — generation + storage extendable, transmission locked
  --mode generation_tx  — generation + lines extendable, storage locked
  --mode transmission   — transmission lines extendable only
  --mode full_tx        — generation + storage + transmission extendable
  --mode expansion_tx   — alias for full generation + storage + transmission expansion

For multiloc: pass --location LABEL --scale MW
For all2030:  pass --dc-data path/to/TX_datacenters.xlsx

Usage examples:
    # Baseline check (no DC, dispatch)
    python scripts/run_dc_experiment.py \
        --year 2019 --dc-scenario none --mode dispatch

    # One multi-loc site, generation-only expansion
    python scripts/run_dc_experiment.py \
        --year 2019 --dc-scenario multiloc \
        --location TOLAR --scale 5000 --mode generation \
        --results-root results/dc_experiments_no_storage

    # All-2030 with generation + transmission expansion
    python scripts/run_dc_experiment.py \
        --year 2019 --dc-scenario all2030 \
        --dc-data TX_datacenters.xlsx --mode generation_tx \
        --results-root results/dc_experiments_no_storage
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

# Allow `python scripts/run_dc_experiment.py` from repo root
sys.path.insert(0, str(Path(__file__).parent))
from dc_common import (
    DC_LOCATIONS,
    add_dc_load,
    add_all2030_dcs,
    assign_ercot_zone,
    compute_congestion,
    compute_lmp_summary,
    decompose_system_cost,
    extract_new_capacity,
    flatten_index,
    get_ercot_buses,
    prepare_for_mode,
    resolve_bus,
)


# =====================================================================
# Solver
# =====================================================================

GUROBI_OPTS = {
    "threads": 16,
    "method": 2,
    "crossover": 0,
    "BarHomogeneous": 1,
    "BarConvTol": 1e-4,
    "OptimalityTol": 1e-3,
    "FeasibilityTol": 1e-3,
    "ScaleFlag": 1,
    "Seed": 123,
    "AggFill": 0,
    "PreDual": 0,
    "GURO_PAR_BARDENSETHRESH": 200,
}


def solve(network: pypsa.Network, threads: int = 16) -> str:
    opts = dict(GUROBI_OPTS)
    opts["threads"] = threads
    status, condition = network.optimize(
        solver_name="gurobi",
        solver_options=opts,
        linearized_unit_commitment=True,
        transmission_losses=0,
        assign_all_duals=True,
    )
    print(f"  solver status: {status} ({condition})")
    if status != 'ok' or str(condition).lower() != 'optimal':
        raise RuntimeError(f'Solve failed: {status}, {condition}')
    if not np.isfinite(network.objective):
        raise RuntimeError('Non-finite objective')
    return status


# =====================================================================
# Output paths
# =====================================================================

def make_output_dir(year: int, dc_scenario: str, mode: str,
                    location: str = None, scale: int = None,
                    results_root: str | Path = "results/dc_experiments_no_storage") -> Path:
    """
    Build the canonical output directory for one experiment.

    {results_root}/{year}/{dc_scenario}_{mode}[_loc_scale]/
    """
    base = Path(results_root) / str(year)
    if dc_scenario == "none":
        sub = f"none_{mode}"
    elif dc_scenario == "multiloc":
        if location is None or scale is None:
            raise ValueError("multiloc needs --location and --scale")
        sub = f"multiloc_{mode}_{location}_{scale}MW"
    elif dc_scenario == "all2030":
        sub = f"all2030_{mode}"
    elif dc_scenario == "distributed":
        if location is None:
            raise ValueError("distributed needs --location")
        if scale is not None:
            sub = f"distributed_{mode}_{location}_{scale}MW"
        else:
            sub = f"distributed_{mode}_{location}"
    else:
        raise ValueError(f"unknown scenario: {dc_scenario}")
    out = base / sub
    out.mkdir(parents=True, exist_ok=True)
    return out


# =====================================================================
# Main experiment
# =====================================================================

def run_experiment(args):
    t0 = time.time()
    year = args.year
    mode = args.mode
    scenario = args.dc_scenario

    # ── 1. Load starting network ──
    if args.starting_network:
        start_path = Path(args.starting_network)
    else:
        start_path = Path(f"results/starting_networks/start_{year}.nc")

    if not start_path.exists():
        raise FileNotFoundError(
            f"Starting network not found: {start_path}\n"
            f"Run scripts/prepare_starting_network.py --year {year} first.")

    print(f"\n{'='*70}")
    print(f"DC experiment — year={year}  scenario={scenario}  mode={mode}")
    if scenario == "multiloc":
        print(f"               location={args.location}  scale={args.scale} MW")
    print(f"{'='*70}\n")
    print(f"[1] Loading {start_path}")
    network = pypsa.Network(str(start_path))
    print(f"    {len(network.buses)} buses, {len(network.lines)} lines, "
          f"{len(network.generators)} gens, {len(network.storage_units)} storage")
    print(f"    snapshots: {len(network.snapshots)}")

    ercot_buses = get_ercot_buses(network)
    print(f"    {len(ercot_buses)} ERCOT buses")

    # ── 2. Configure mode (what's extendable) ──
    print(f"\n[2] Configuring mode={mode}")
    base_caps = prepare_for_mode(network, mode)

    # ── 3. Add DC load(s) ──
    dc_info = {}
    print(f"\n[3] Adding DC load (scenario={scenario})")
    if scenario == "none":
        print("    no DC added — pure re-solve of starting network")
    elif scenario == "multiloc":
        if args.location not in DC_LOCATIONS:
            raise ValueError(
                f"location {args.location} not in {list(DC_LOCATIONS.keys())}")
        info = DC_LOCATIONS[args.location]
        bus = resolve_bus(network, info["bus"], info["x"], info["y"],
                          label=args.location)
        add_dc_load(network, bus, args.scale, args.dc_cf,
                    name=f"DC_{args.location}")
        zone_actual = assign_ercot_zone(
            network.buses.loc[bus, "x"], network.buses.loc[bus, "y"])
        print(f"    {args.location}: bus={bus} zone={zone_actual} "
              f"capacity={args.scale} MW")
        dc_info["multiloc"] = {
            "location":     args.location,
            "bus":          bus,
            "scale_mw":     args.scale,
            "zone":         zone_actual,
            "planned_2030": info.get("planned_2030_mw"),
        }
    elif scenario == "all2030":
        if not args.dc_data:
            raise ValueError("all2030 needs --dc-data")
        dc_data_path = Path(args.dc_data)
        if not dc_data_path.exists():
            raise FileNotFoundError(f"DC data file not found: {dc_data_path}")
        dc_info["all2030"] = add_all2030_dcs(network, dc_data_path, args.dc_cf)

    elif args.dc_scenario == "distributed":
        from dc_common import add_distributed_dc
        dc_data_path = Path(args.dc_data) if args.dc_data else Path("TX_datacenters.xlsx")
        # If --scale given, use it; otherwise use actual planned capacity
        kw = {"scale_mw": args.scale} if args.scale else {}
        dc_info["distributed"] = add_distributed_dc(
            network, args.location, dc_data_path, args.dc_cf, **kw)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    # ── 4. Solve ──
    print(f"\n[4] Solving ...")
    t_solve_start = time.time()
    status = solve(network, threads=args.threads)
    solve_time = time.time() - t_solve_start
    print(f"    solve time: {solve_time:.1f}s")
    if status != "ok":
        print(f"    [WARN] non-optimal status: {status}")

    # ── 5. Output dir ──
    out_dir = make_output_dir(
        year, scenario, mode,
        args.location, args.scale,
        args.results_root,
    )
    print(f"\n[5] Output dir: {out_dir}")

    # ── 6. Save solved network ──
    nc_out = out_dir / "network.nc"
    network.export_to_netcdf(str(nc_out))
    print(f"    saved network ({nc_out.stat().st_size/1e6:.1f} MB)")

    # ── 7. Compute metrics ──
    print(f"\n[6] Computing metrics")
    cost = decompose_system_cost(network)
    lmp_summary, _lmp = compute_lmp_summary(network, ercot_buses)
    congestion = compute_congestion(network, ercot_buses)

    metrics = {
        "year":         year,
        "scenario":     scenario,
        "mode":         mode,
        "location":     args.location if scenario == "multiloc" else None,
        "scale_mw":     args.scale if scenario == "multiloc" else None,
        "solve_status": status,
        "solve_time_s": round(solve_time, 1),
        "congestion_lh85": congestion,
    }
    metrics.update(cost)
    metrics.update(lmp_summary)

    # ── 8. Extract new capacity if any expansion happened ──
    EXPANSION_MODES = {
        "generation",
        "generation_storage",
        "generation_tx",
        "transmission",
        "full_tx",
        "expansion_tx",
    }

    if mode in EXPANSION_MODES:
        new_caps = extract_new_capacity(network, base_caps)

        if not new_caps.empty:
            new_caps.to_csv(out_dir / "new_capacity.csv", index=False)

            metrics["total_new_mw"] = float(new_caps["added_mw"].sum())
            metrics["total_annual_investment"] = float(
                new_caps["annual_investment"].sum()
            )

            # Standard carrier-level outputs.
            for carrier in [
                "OCGT",
                "solar",
                "onwind",
                "4hr_battery_storage",
                "transmission",
            ]:
                sub = new_caps[new_caps["carrier"] == carrier]
                key = "battery" if carrier == "4hr_battery_storage" else carrier
                metrics[f"new_{key}_mw"] = (
                    float(sub["added_mw"].sum()) if not sub.empty else 0.0
                )

            # Component-level diagnostics.
            gen_sub = new_caps[new_caps["component"] == "generators"]
            storage_sub = new_caps[new_caps["component"] == "storage_units"]
            line_sub = new_caps[new_caps["component"] == "lines"]

            metrics["new_generation_mw"] = (
                float(gen_sub["added_mw"].sum()) if not gen_sub.empty else 0.0
            )
            metrics["new_storage_mw"] = (
                float(storage_sub["added_mw"].sum()) if not storage_sub.empty else 0.0
            )
            metrics["new_transmission_mw"] = (
                float(line_sub["added_mw"].sum()) if not line_sub.empty else 0.0
            )

            # Sanity checks by mode.
            if mode == "transmission":
                if metrics["new_generation_mw"] > 1e-3 or metrics["new_storage_mw"] > 1e-3:
                    print(
                        "    [WARN] transmission-only mode built non-transmission capacity: "
                        f"gen={metrics['new_generation_mw']:.2f} MW, "
                        f"storage={metrics['new_storage_mw']:.2f} MW"
                    )

            if mode in ("generation", "generation_tx"):
                if metrics["new_storage_mw"] > 1e-3:
                    print(
                        f"    [WARN] unexpected storage expansion in no-storage mode: "
                        f"{metrics['new_storage_mw']:.2f} MW"
                    )

            print(
                f"    new capacity: {metrics['total_new_mw']:.0f} MW total, "
                f"${metrics['total_annual_investment']/1e6:.1f}M/yr investment"
            )
            print(
                f"      gen={metrics['new_generation_mw']:.0f} MW, "
                f"storage={metrics['new_storage_mw']:.0f} MW, "
                f"tx={metrics['new_transmission_mw']:.0f} MW"
            )

        else:
            print("    no new capacity built")
            metrics["total_new_mw"] = 0.0
            metrics["total_annual_investment"] = 0.0
            metrics["new_generation_mw"] = 0.0
            metrics["new_storage_mw"] = 0.0
            metrics["new_transmission_mw"] = 0.0
            metrics["new_OCGT_mw"] = 0.0
            metrics["new_solar_mw"] = 0.0
            metrics["new_onwind_mw"] = 0.0
            metrics["new_battery_mw"] = 0.0
            metrics["new_transmission_mw"] = 0.0

    # ── 9. Save metrics + DC mapping ──
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Flatten metrics into a one-row csv for easy concat later
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    if scenario == "all2030" and "all2030" in dc_info:
        mapping_df = pd.DataFrame.from_dict(dc_info["all2030"], orient="index")
        mapping_df.index.name = "load_name"
        mapping_df.to_csv(out_dir / "dc_bus_mapping.csv")

    elapsed = time.time() - t0
    print(f"\n[OK] experiment finished in {elapsed:.1f}s")
    print(f"     objective: ${metrics['objective']/1e6:.2f}M")
    print(f"     dispatch:  ${metrics['dispatch_cost']/1e6:.2f}M")
    print(f"     capex:     ${metrics['capex_total']/1e6:.2f}M")
    print(f"     sys LMP:   {metrics['sys_mean_lmp']:.2f} $/MWh")
    print(f"     congestion: {metrics['congestion_lh85']} line-hours > 85%")

    return metrics


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run one DC impact experiment")
    parser.add_argument("--year", type=int, required=True,
                        help="Year (2019-2023)")
    parser.add_argument("--dc-scenario", required=True,
                        choices=["none", "multiloc", "all2030", "distributed"])
    parser.add_argument("--mode", required=True,
                    choices=[
                        "dispatch",
                        "generation",
                        "generation_tx",
                        "generation_storage",
                        "transmission",
                        "full_tx",
                        "expansion_tx",
                    ])
    parser.add_argument("--location", default=None,
                        help="Multi-loc site label (TOLAR, ABILENE, ...)")
    parser.add_argument("--scale", type=int, default=None,
                        help="Multi-loc DC capacity in MW")
    parser.add_argument("--dc-data", default=None,
                        help="Path to TX_datacenters.xlsx (all2030 only)")
    parser.add_argument("--dc-cf", type=float, default=0.90,
                        help="DC capacity factor (default 0.90)")
    parser.add_argument("--starting-network", default=None,
                        help="Override starting network path")
    parser.add_argument(
                        "--results-root",
                        default="results/dc_experiments_no_storage",
                        help="Root directory for experiment outputs",
                    )
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    if args.dc_scenario == "multiloc":
        if not args.location or args.scale is None:
            parser.error("multiloc requires --location and --scale")
    if args.dc_scenario == "all2030":
        if not args.dc_data:
            parser.error("all2030 requires --dc-data")

    run_experiment(args)


if __name__ == "__main__":
    main()
