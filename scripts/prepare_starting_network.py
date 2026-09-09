"""
prepare_starting_network.py
============================
Build the unified starting network for one year from the vopt solve.

The vopt solve already co-optimized:
  - generator capacity (OCGT, solar, onwind, battery)
  - storage capacity
  - transmission line capacity (s_nom_extendable from ll=vopt)

We take that solved state, lock everything in place, and produce a
"starting network" that all DC experiments use as their identical
starting point.

The DC experiment script (run_dc_experiment.py) then selectively
unlocks components based on the chosen mode:
  - dispatch:     keep everything locked
  - expansion:    unlock OCGT/solar/wind/battery for additional build
  - expansion_tx: also unlock all lines for additional build

Usage:
    python scripts/prepare_starting_network.py --year 2019
"""
import argparse
from pathlib import Path

import pypsa
import pandas as pd


def default_paths(year: int):
    vopt = Path(f"results/vopt_{year}/texas/networks/"
                f"elec_s500_c185_ec_lvopt_3h_E_operations.nc")
    out = Path(f"results/starting_networks/start_{year}.nc")
    return vopt, out


def lock_to_optimal(network: pypsa.Network) -> dict:
    """
    For every component class, set nominal capacity = optimized capacity,
    then disable extendability.

    Returns count of components locked per class.
    """
    counts = {}
    for comp_name, nom, opt, ext in [
        ("generators",    "p_nom", "p_nom_opt", "p_nom_extendable"),
        ("storage_units", "p_nom", "p_nom_opt", "p_nom_extendable"),
        ("stores",        "e_nom", "e_nom_opt", "e_nom_extendable"),
        ("links",         "p_nom", "p_nom_opt", "p_nom_extendable"),
    ]:
        comp = getattr(network, comp_name, None)
        if comp is None or comp.empty:
            counts[comp_name] = 0
            continue
        if opt not in comp.columns or ext not in comp.columns:
            counts[comp_name] = 0
            continue

        was_ext = comp[ext].copy()
        n_was_ext = int(was_ext.sum())

        if n_was_ext > 0:
            comp.loc[was_ext, nom] = comp.loc[was_ext, opt]

        comp[ext] = False
        counts[comp_name] = n_was_ext

    # Lines: same logic
    lines = network.lines
    n_line_ext = 0
    if "s_nom_extendable" in lines.columns:
        was_ext = lines["s_nom_extendable"].copy()
        n_line_ext = int(was_ext.sum())
        if n_line_ext > 0 and "s_nom_opt" in lines.columns:
            lines.loc[was_ext, "s_nom"] = lines.loc[was_ext, "s_nom_opt"]
        lines["s_nom_extendable"] = False
    counts["lines"] = n_line_ext

    return counts


def prepare_starting_network(vopt_path: Path, output_path: Path):
    print(f"\n[prepare_starting_network]")
    print(f"  vopt: {vopt_path}")
    print(f"  out:  {output_path}")

    if not vopt_path.exists():
        raise FileNotFoundError(f"vopt network not found: {vopt_path}")

    print("\n[1] Loading vopt network...")
    net = pypsa.Network(str(vopt_path))
    print(f"    {len(net.buses)} buses, {len(net.lines)} lines, "
          f"{len(net.generators)} generators, "
          f"{len(net.storage_units)} storage units")
    print(f"    snapshots: {len(net.snapshots)}")
    print(f"    vopt objective: ${net.objective/1e6:.2f}M")

    # Capture some pre-lock stats for reporting
    lines = net.lines
    if "s_nom_extendable" in lines.columns:
        ext_lines = lines[lines["s_nom_extendable"]]
        if not ext_lines.empty and "s_nom_opt" in lines.columns:
            grew = (ext_lines["s_nom_opt"] > ext_lines["s_nom"] * 1.001).sum()
            avg_pre = ext_lines["s_nom"].mean()
            avg_post = ext_lines["s_nom_opt"].mean()
            print(f"    line stats (vopt solve): "
                  f"{int(grew)}/{len(ext_lines)} lines expanded, "
                  f"avg s_nom: {avg_pre:.0f} -> {avg_post:.0f} MW")

    # Save line capex/max metadata BEFORE locking — we need it for
    # later expansion_tx mode
    print("\n[2] Preserving line metadata for later expansion_tx mode...")
    line_capex = lines["capital_cost"].copy() if "capital_cost" in lines.columns else None
    line_smax  = lines["s_nom_max"].copy() if "s_nom_max" in lines.columns else None
    print(f"    saved capital_cost (mean=${(line_capex.mean() if line_capex is not None else 0):,.0f})")
    print(f"    saved s_nom_max (mean={(line_smax.mean() if line_smax is not None else 0):.0f} MW)")

    print("\n[3] Locking all capacities to *_opt values...")
    counts = lock_to_optimal(net)
    for k, v in counts.items():
        if v > 0:
            print(f"    {k}: locked {v} components")

    # Restore line metadata (lock_to_optimal didn't touch these but
    # if PyPSA version has any normalization, this guarantees they survive)
    if line_capex is not None:
        net.lines["capital_cost"] = line_capex
    if line_smax is not None:
        net.lines["s_nom_max"] = line_smax

    print("\n[4] Verification:")
    n_gen_ext = int(net.generators["p_nom_extendable"].sum()) if "p_nom_extendable" in net.generators.columns else 0
    n_sto_ext = int(net.storage_units["p_nom_extendable"].sum()) if "p_nom_extendable" in net.storage_units.columns else 0
    n_line_ext = int(net.lines["s_nom_extendable"].sum())
    print(f"    extendable: gens={n_gen_ext}, storage={n_sto_ext}, lines={n_line_ext}")
    assert n_gen_ext == 0 and n_sto_ext == 0 and n_line_ext == 0, \
        "starting network should have nothing extendable after locking"

    avg_snom = net.lines["s_nom"].mean()
    avg_smax = net.lines["s_nom_max"].mean() if "s_nom_max" in net.lines.columns else 0
    avg_capex = net.lines["capital_cost"].mean() if "capital_cost" in net.lines.columns else 0
    print(f"    avg line s_nom:     {avg_snom:.0f} MW")
    print(f"    avg line s_nom_max: {avg_smax:.0f} MW (headroom for tx expansion)")
    print(f"    avg line capex:     ${avg_capex:,.0f} per MW/yr")

    print(f"\n[5] Writing {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.export_to_netcdf(str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    print(f"    OK ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Lock vopt network to produce a starting network for DC experiments")
    parser.add_argument("--year", type=int, default=None,
                        help="Year (2019-2023). Auto-fills paths.")
    parser.add_argument("--vopt-network", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.year is not None:
        v, o = default_paths(args.year)
        if args.vopt_network is None: args.vopt_network = v
        if args.output is None: args.output = o

    if args.vopt_network is None or args.output is None:
        parser.error("need --year, or both --vopt-network and --output")

    prepare_starting_network(args.vopt_network, args.output)


if __name__ == "__main__":
    main()