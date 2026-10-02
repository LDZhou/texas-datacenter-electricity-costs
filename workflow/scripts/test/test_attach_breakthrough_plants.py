"""Regression tests for breakthrough hydro attachment."""

import sys
from pathlib import Path

import pandas as pd
import pypsa

sys.path.insert(0, str(Path(__file__).parent.parent))

from _helpers import get_multiindex_snapshots
from add_electricity import attach_breakthrough_renewable_plants

# Listed so that a sorted union reorders them, which is what the alignment bug needed.
PLANT_IDS = ["30", "10", "20"]


def _write_inputs(directory: Path, snapshots: int) -> dict[str, Path]:
    plants = pd.DataFrame(
        {
            "bus_id": ["1001", "1002", "1003"],
            "type": ["hydro"] * 3,
            # A zero Pmax routes p_nom through the concat branch.
            "Pmax": [0.0, 40.0, 60.0],
        },
        index=PLANT_IDS,
    )
    plants.index.name = "plant_id"
    paths = {
        "plants": directory / "plant.csv",
        "bus2sub": directory / "bus2sub.csv",
        "busmap_s": directory / "busmap_s.csv",
        "hydro": directory / "hydro.csv",
    }
    plants.to_csv(paths["plants"])
    pd.DataFrame({"Bus": ["1001", "1002", "1003"], "sub_id": ["1", "2", "3"]}).to_csv(paths["bus2sub"], index=False)
    pd.DataFrame({"sub_id": ["1", "2", "3"], "cluster_bus": ["b1", "b1", "b1"]}).to_csv(paths["busmap_s"], index=False)
    index = pd.date_range("2016-01-01", periods=snapshots, freq="h")
    pd.DataFrame(
        {"30": [5.0] * snapshots, "10": [20.0] * snapshots, "20": [30.0] * snapshots},
        index=index,
    ).to_csv(paths["hydro"])
    return paths


def test_breakthrough_hydro_keeps_plant_order_when_pmax_is_zero(tmp_path):
    """A zero-Pmax hydro plant must not reorder p_nom away from the plant index."""
    network = pypsa.Network()
    network.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 02:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    network.set_investment_periods(periods=[2030])
    network.add("Bus", "b1", x=-97.0, y=31.0, carrier="AC")
    paths = _write_inputs(tmp_path, len(network.snapshots))

    attach_breakthrough_renewable_plants(
        network,
        paths["plants"],
        paths["bus2sub"],
        paths["busmap_s"],
        {"hydro": paths["hydro"]},
        ["hydro"],
    )

    assert list(network.generators.index) == PLANT_IDS
    # p_nom is max(Pmax, peak dispatch), so the zero-Pmax plant takes its profile peak.
    assert network.generators.loc["30", "p_nom"] == 5.0
    assert network.generators.loc["10", "p_nom"] == 40.0
    assert network.generators.loc["20", "p_nom"] == 60.0
    assert (network.generators["bus"] == "b1").all()
