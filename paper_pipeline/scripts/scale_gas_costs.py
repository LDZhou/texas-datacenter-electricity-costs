"""Scale gas-fired generator marginal costs in a PyPSA network."""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


GAS_CARRIERS = ("CCGT", "OCGT")


def scale_gas_costs(network: pypsa.Network, factor: float) -> dict:
    """Scale the gas fuel component while leaving variable O&M unchanged.

    The returned record is also stored in ``network.meta`` to make exported
    sensitivity networks self-describing without recording environment data.
    """
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be a positive finite number")

    gas = network.generators.carrier.isin(GAS_CARRIERS)
    if "vom_cost" not in network.generators.columns or network.generators.loc[gas, "vom_cost"].isna().any():
        raise ValueError(
            "fuel-only gas scaling requires finite static vom_cost for every "
            "gas generator; unknown decomposition"
        )
    dynamic_marginal_cost = getattr(network.generators_t, "marginal_cost", None)
    dynamic_columns = (
        dynamic_marginal_cost.columns.intersection(network.generators.index[gas])
        if dynamic_marginal_cost is not None and not dynamic_marginal_cost.empty
        else []
    )
    record = {
        "factor": float(factor),
        "carriers": list(GAS_CARRIERS),
        "static_generators_scaled": int(gas.sum()),
        "timeseries_columns_scaled": int(len(dynamic_columns)),
        "decomposition": "marginal_cost_minus_static_vom_and_known_other",
    }

    # A factor-one build is an identity control. Keep the imported arrays
    # byte-for-byte untouched so the control can detect preprocessing drift.
    if factor == 1.0:
        meta = dict(network.meta or {})
        meta["gas_cost_scaling"] = record
        network.meta = meta
        return record

    # `fuel_cost` is a fuel-price input ($/MMBtu), rather than a generator's
    # fuel contribution ($/MWh). Infer the latter from actual marginal cost.
    other_static = network.generators.get("other_marginal_cost", 0.0)
    if hasattr(other_static, "reindex"):
        other_static = other_static.reindex(network.generators.index).fillna(0.0)
    fuel_static = (network.generators.loc[gas, "marginal_cost"]
                   - network.generators.loc[gas, "vom_cost"]
                   - (other_static.loc[gas] if hasattr(other_static, "loc") else other_static))
    if (fuel_static < -1e-8).any():
        raise ValueError("gas marginal cost is below known VOM/other cost; unknown decomposition")
    fuel_static = fuel_static.clip(lower=0.0)
    network.generators.loc[gas, "marginal_cost"] = (
        network.generators.loc[gas, "vom_cost"]
        + (other_static.loc[gas] if hasattr(other_static, "loc") else other_static)
        + factor * fuel_static
    )
    if "fuel_cost" in network.generators.columns:
        network.generators.loc[gas, "fuel_cost"] *= factor
    marginal_cost = getattr(network.generators_t, "marginal_cost", None)
    if marginal_cost is not None and not marginal_cost.empty:
        columns = marginal_cost.columns.intersection(network.generators.index[gas])
        if len(columns):
            static_vom = network.generators.loc[columns, "vom_cost"]
            vom = pd.DataFrame(
                np.broadcast_to(static_vom.to_numpy(), (len(marginal_cost.index), len(columns))),
                index=marginal_cost.index, columns=columns,
            )
            dynamic_vom = getattr(network.generators_t, "vom_cost", None)
            if dynamic_vom is not None and not dynamic_vom.empty:
                supplied = dynamic_vom.columns.intersection(columns)
                if len(supplied):
                    vom.loc[:, supplied] = dynamic_vom.reindex(
                        index=marginal_cost.index, columns=supplied
                    ).astype(float)
            other = getattr(network.generators_t, "other_marginal_cost", None)
            if other is None or other.empty:
                other = pd.DataFrame(0.0, index=marginal_cost.index, columns=columns)
            else:
                other = other.reindex(index=marginal_cost.index, columns=columns).fillna(0.0)
            fuel = marginal_cost.loc[:, columns].astype(float) - vom - other
            if (fuel < -1e-8).any().any():
                raise ValueError("time-series gas marginal cost is below known VOM/other cost")
            fuel = fuel.clip(lower=0.0)
            # Network imports can retain integer time-series columns. Build a
            # float copy before scaling so fractional factors never truncate.
            scaled = marginal_cost.astype({column: float for column in columns})
            scaled.columns = marginal_cost.columns.copy()
            scaled.loc[:, columns] = vom + other + factor * fuel
            network.generators_t.marginal_cost = scaled

    dynamic_fuel_price = getattr(network.generators_t, "fuel_cost", None)
    if dynamic_fuel_price is not None and not dynamic_fuel_price.empty:
        columns = dynamic_fuel_price.columns.intersection(network.generators.index[gas])
        if len(columns):
            scaled_fuel_price = dynamic_fuel_price.astype(
                {column: float for column in columns}
            )
            scaled_fuel_price.columns = dynamic_fuel_price.columns.copy()
            scaled_fuel_price.loc[:, columns] *= factor
            network.generators_t.fuel_cost = scaled_fuel_price

    meta = dict(network.meta or {})
    meta["gas_cost_scaling"] = record
    network.meta = meta
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--factor", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    network = pypsa.Network(args.input)
    record = scale_gas_costs(network, args.factor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    network.export_to_netcdf(args.output)
    print(f"scaled {record['static_generators_scaled']} gas generators by x{args.factor}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
