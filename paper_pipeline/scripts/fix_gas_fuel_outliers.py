"""Reset gas units whose implied fuel price is an outlier for their weather year.

PyPSA-USA carries plant-level fuel costs. In the 2021 base a single OCGT unit
(bus p480210) inherits an implied gas price under half the fleet median, so
brownfield expansion at that site builds 14 GW of baseload OCGT: an artefact of
one input row, not a planning result. The rule here is deliberately narrow: a
gas unit is an outlier when its implied fuel price (fuel component of marginal
cost x efficiency) is below ``threshold`` x the median over all gas units of
that network, evaluated snapshot by snapshot on the time-varying marginal cost.
Its fuel component is replaced by the fleet-median implied fuel price divided by
its own efficiency; VOM and other components are untouched, as are all other
units. Everything changed is recorded in ``network.meta`` and in the returned
record so the correction stays auditable.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from .scale_gas_costs import GAS_CARRIERS

DEFAULT_THRESHOLD = 0.5


def _components(network):
    g = network.generators
    gas = g.index[g.carrier.isin(GAS_CARRIERS)]
    if "vom_cost" not in g.columns or g.loc[gas, "vom_cost"].isna().any():
        raise ValueError("fuel-outlier correction requires finite static vom_cost for every gas generator")
    other = g.get("other_marginal_cost", pd.Series(0.0, index=g.index))
    other = other.reindex(g.index).fillna(0.0) if hasattr(other, "reindex") else pd.Series(float(other), index=g.index)
    mc = getattr(network.generators_t, "marginal_cost", None)
    if mc is None or mc.empty or not len(mc.columns.intersection(gas)):
        mc = pd.DataFrame(np.broadcast_to(g.loc[gas, "marginal_cost"].to_numpy(), (len(network.snapshots), len(gas))),
                          index=network.snapshots, columns=gas)
    else:
        mc = mc.reindex(columns=gas)
        missing = mc.columns[mc.isna().all()]
        for column in missing:
            mc[column] = g.loc[column, "marginal_cost"]
    vom = pd.DataFrame(np.broadcast_to(g.loc[gas, "vom_cost"].to_numpy(), mc.shape), index=mc.index, columns=gas)
    dyn_vom = getattr(network.generators_t, "vom_cost", None)
    if dyn_vom is not None and not dyn_vom.empty:
        supplied = dyn_vom.columns.intersection(gas)
        if len(supplied):
            vom.loc[:, supplied] = dyn_vom.reindex(index=mc.index, columns=supplied).astype(float)
    oth = pd.DataFrame(np.broadcast_to(other.loc[gas].to_numpy(), mc.shape), index=mc.index, columns=gas)
    fuel = (mc.astype(float) - vom - oth)
    if (fuel < -1e-8).any().any():
        raise ValueError("gas marginal cost is below known VOM/other cost; unknown decomposition")
    fuel = fuel.clip(lower=0.0)
    eff = g.loc[gas, "efficiency"].astype(float)
    if (eff <= 0).any() or eff.isna().any():
        raise ValueError("gas generators need positive efficiency to imply a fuel price")
    return gas, mc, vom, oth, fuel, eff


def find_outliers(network, threshold=DEFAULT_THRESHOLD):
    gas, mc, vom, oth, fuel, eff = _components(network)
    implied = fuel.div(1.0 / eff, axis=1)  # $/MWh_fuel-equivalent = fuel component x efficiency
    median = implied.median(axis=1)
    ratio = implied.mean(axis=0) / median.mean()
    table = pd.DataFrame({"bus": network.generators.loc[gas, "bus"], "carrier": network.generators.loc[gas, "carrier"],
                          "p_nom": network.generators.loc[gas, "p_nom"], "efficiency": eff,
                          "mean_marginal_cost": mc.mean(axis=0), "mean_implied_fuel": implied.mean(axis=0),
                          "ratio_to_fleet_median": ratio})
    return table[table.ratio_to_fleet_median < threshold].sort_values("ratio_to_fleet_median"), median


def fix_gas_fuel_outliers(network, threshold=DEFAULT_THRESHOLD):
    gas, mc, vom, oth, fuel, eff = _components(network)
    outliers, median = find_outliers(network, threshold)
    record = {"threshold": float(threshold), "carriers": list(GAS_CARRIERS), "gas_units_scanned": int(len(gas)),
              "fleet_median_implied_fuel_mean": float(median.mean()),
              "corrected": [{**{k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in row.items()}, "name": name}
                            for name, row in outliers.iterrows()]}
    if outliers.empty:
        return record
    new_mc = mc.astype(float).copy()
    for name in outliers.index:
        new_fuel = median / float(eff[name])
        new_mc[name] = vom[name] + oth[name] + new_fuel
    dyn = getattr(network.generators_t, "marginal_cost", None)
    if dyn is not None and not dyn.empty:
        updated = dyn.astype({c: float for c in dyn.columns.intersection(outliers.index)}).copy()
        for name in outliers.index:
            updated[name] = new_mc[name].reindex(updated.index).to_numpy()
        network.generators_t.marginal_cost = updated
    for name in outliers.index:
        network.generators.loc[name, "marginal_cost"] = float(new_mc[name].mean())
        if "fuel_cost" in network.generators.columns and pd.notna(network.generators.loc[name, "fuel_cost"]):
            old_fuel = float(fuel[name].mean())
            if old_fuel > 0:
                network.generators.loc[name, "fuel_cost"] *= float((median.mean() / float(eff[name])) / old_fuel)
        for entry in record["corrected"]:
            if entry["name"] == name:
                entry["new_mean_marginal_cost"] = float(new_mc[name].mean())
    meta = dict(network.meta or {})
    meta["gas_fuel_outlier_fix"] = record
    network.meta = meta
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--record", type=Path, default=None, help="JSON file receiving the correction record")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--scan-only", action="store_true", help="report outliers without writing a network")
    args = parser.parse_args()
    network = pypsa.Network(args.input)
    if args.scan_only:
        outliers, median = find_outliers(network, args.threshold)
        print(f"fleet median implied fuel {median.mean():.2f}; {len(outliers)} outlier(s)")
        print(outliers.round(3).to_string())
        return
    record = fix_gas_fuel_outliers(network, args.threshold)
    record["source"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    network.export_to_netcdf(args.output)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n")
    print(f"corrected {len(record['corrected'])} gas unit(s); saved {args.output}")


if __name__ == "__main__":
    main()
