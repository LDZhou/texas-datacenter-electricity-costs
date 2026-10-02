"""Export a PyPSA expansion result into the canonical Texas2k asset schema.

`new_capacity.csv` records each expansion decision with its PyPSA bus but no
coordinates, and Texas2k needs longitude/latitude to place an asset on the
synthetic network. This script joins the two and keeps the PyPSA carrier label unchanged, so
that mapping a carrier to a Texas2k fuel stays a separate, auditable step.

Transmission rows are never emitted: a PyPSA line reinforcement is not a
Texas2k generation asset, and Texas2k derives its own branch upgrades.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ASSET_COLUMNS = (
    "year",
    "case",
    "location",
    "asset_type",
    "carrier",
    "zone",
    "bus",
    "bus_x",
    "bus_y",
    "added_mw",
)
COMPONENT_TO_ASSET_TYPE = {"generators": "generation", "storage_units": "storage"}


def _bus_coordinates(network_path: Path) -> pd.DataFrame:
    import pypsa

    network = pypsa.Network(str(network_path))
    return network.buses.loc[:, ["x", "y"]].rename(columns={"x": "bus_x", "y": "bus_y"})


def export_assets(
    results_dir: Path,
    destination: Path,
    *,
    year: int,
    case: str,
    location: str,
    include_storage: bool,
) -> pd.DataFrame:
    """Write one canonical asset table and return it."""
    results_dir = Path(results_dir)
    capacity = pd.read_csv(results_dir / "new_capacity.csv")
    capacity["added_mw"] = pd.to_numeric(capacity["added_mw"], errors="coerce")
    allowed = {"generators", "storage_units"} if include_storage else {"generators"}
    selected = capacity[
        capacity["component"].isin(allowed) & capacity["added_mw"].fillna(0.0).gt(0.0)
    ].copy()
    coordinates = _bus_coordinates(results_dir / "network.nc")
    unknown = sorted(set(selected["bus"]) - set(coordinates.index))
    if unknown:
        raise ValueError(f"buses missing from network.nc: {unknown[:5]}")
    selected["asset_type"] = selected["component"].map(COMPONENT_TO_ASSET_TYPE)
    selected["bus_x"] = selected["bus"].map(coordinates["bus_x"])
    selected["bus_y"] = selected["bus"].map(coordinates["bus_y"])
    selected["year"] = int(year)
    selected["case"] = case
    selected["location"] = location
    assets = selected.loc[:, list(ASSET_COLUMNS)].sort_values(
        ["carrier", "zone", "added_mw"], ascending=[True, True, False]
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assets.to_csv(destination, index=False, lineterminator="\n")
    return assets.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Export one PyPSA result directory and print its carrier totals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="PyPSA case directory holding new_capacity.csv and network.nc")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--case", required=True)
    parser.add_argument("--location", default="ALL_2030")
    parser.add_argument("--include-storage", action="store_true")
    arguments = parser.parse_args(argv)
    assets = export_assets(
        arguments.results_dir,
        arguments.destination,
        year=arguments.year,
        case=arguments.case,
        location=arguments.location,
        include_storage=arguments.include_storage,
    )
    totals = assets.groupby("carrier")["added_mw"].agg(["count", "sum"])
    print(f"Wrote {len(assets)} assets ({assets['added_mw'].sum():,.1f} MW) to {arguments.destination}")
    for carrier, row in totals.iterrows():
        print(f"  {carrier:<22} {int(row['count']):>4} rows {row['sum'] / 1000:>10.4f} GW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
