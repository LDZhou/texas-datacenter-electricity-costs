"""Regression tests for metadata retained by substation aggregation."""

import sys
from pathlib import Path

import pandas as pd
import pypsa

sys.path.insert(0, str(Path(__file__).parent.parent))

from aggregate_to_substations import aggregate_to_substations


def test_substation_aggregation_preserves_balancing_area_for_eia_demand():
    """An aggregated Texas bus still carries the BA needed by EIA demand."""
    network = pypsa.Network()
    regional_attrs = {
        "v_nom": 230.0,
        "interconnect": "texas",
        "state": "Texas",
        "country": "p48001",
        "county": "p48001",
        "balancing_area": "ERCO",
        "reeds_zone": "p60",
        "reeds_ba": "ERCO",
        "reeds_state": "TX",
    }

    for name, longitude in (("b1", -97.0), ("b2", -96.9)):
        network.add("Bus", name, x=longitude, y=31.0, **regional_attrs)
        network.buses.loc[name, "sub_id"] = 10
        network.buses.loc[name, "Pd"] = 1.0
        network.buses.loc[name, "load_weight"] = 1.0
        network.buses.loc[name, "LAF_state"] = 1.0

    network.add(
        "Line",
        "l1",
        bus0="b1",
        bus1="b2",
        r=0.01,
        x=0.1,
        s_nom=100.0,
    )
    busmap = pd.Series({"b1": "10", "b2": "10"})

    aggregated, _ = aggregate_to_substations(network, busmap, "county")

    assert aggregated.buses.loc["10", "balancing_area"] == "ERCO"
