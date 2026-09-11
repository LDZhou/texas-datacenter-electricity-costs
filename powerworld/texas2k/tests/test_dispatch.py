"""Tests for Texas2k scenario assembly and dispatch."""

import copy

import numpy as np
import pandas as pd
import pytest
from pypower.idx_bus import BUS_I, PD
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN

from powerworld.texas2k.dispatch import assemble_scenario, map_datacenter_loads, redispatch_generation


def _base_case():
    bus = np.zeros((3, 13), dtype=float)
    bus[:, BUS_I] = [1, 2, 3]
    bus[:, 1] = [3, 1, 1]
    bus[:, PD] = [0.0, 20.0, 10.0]
    gen = np.zeros((2, 21), dtype=float)
    gen[:, GEN_BUS] = [1, 2]
    gen[:, PG] = [20.0, 10.0]
    gen[:, PMAX] = [100.0, 50.0]
    gen[:, PMIN] = 0.0
    gen[:, GEN_STATUS] = 1
    return {
        "version": "2",
        "baseMVA": 100.0,
        "bus": bus,
        "gen": gen,
        "branch": np.zeros((0, 13)),
        "gencost": np.ones((2, 6)),
        "genfuel": ["ng", "solar"],
    }


def _bus_geography():
    return pd.DataFrame(
        {
            "bus_num": [1, 2, 3],
            "longitude": [-97.0, -96.0, -95.0],
            "latitude": [31.0, 32.0, 33.0],
            "base_kv": [100.0, 200.0, 100.0],
            "station_clean": ["ALPHA", "ALPHA", "BETA"],
            "area_name": ["North", "Coast", "Coast"],
        }
    )


def _assets(carrier: str = "OCGT"):
    return pd.DataFrame(
        [
            {
                "year": 2023,
                "case": "case",
                "location": "ALL_2030",
                "asset_type": "generation",
                "carrier": carrier,
                "zone": "NORTH",
                "bus": "p1",
                "bus_x": -97.0,
                "bus_y": 31.0,
                "added_mw": 30.0,
            }
        ]
    )


def _dc_rows():
    return pd.DataFrame(
        {
            "full_address": ["site a", "site b"],
            "nearest_sub_name": ["ALPHA", "BETA"],
            "total_mw": [10.0, 5.0],
        }
    )


def test_map_datacenter_loads_weights_station_buses_by_voltage():
    """A multi-bus substation must receive one voltage-weighted aggregate load."""
    mapped, info = map_datacenter_loads(_dc_rows(), _bus_geography())

    assert mapped.set_index("bus_num")["mw"].to_dict() == pytest.approx({1: 10 / 3, 2: 20 / 3, 3: 5.0})
    assert info == {"matched_substations": 2, "mapped_buses": 3, "mapped_mw": 15.0, "unmapped_mw": 0.0}


def test_assemble_scenario_adds_each_dc_load_once():
    """Data-center rows must increase total demand exactly once."""
    base = _base_case()
    scenario, info = assemble_scenario(
        base,
        assets=_assets(),
        bus_geography=_bus_geography(),
        dc_loads=_dc_rows(),
        portfolio="generation",
    )

    assert scenario["bus"][:, PD].sum() == pytest.approx(base["bus"][:, PD].sum() + 15.0)
    assert info["dc_applied_mw"] == pytest.approx(15.0)
    assert base["bus"][:, PD].sum() == pytest.approx(30.0)


def test_generation_portfolio_expands_nearby_existing_capacity():
    """OCGT additions should expand nearby rows instead of creating an arbitrary slack unit."""
    scenario, info = assemble_scenario(
        _base_case(),
        assets=_assets(),
        bus_geography=_bus_geography(),
        dc_loads=None,
        portfolio="generation",
    )

    assert scenario["gen"][:, PMAX].sum() == pytest.approx(180.0)
    assert info["asset_mw"] == pytest.approx(30.0)
    assert info["expanded_generator_rows"] == 2


def test_generation_storage_portfolio_matches_gas_by_zone():
    """Zone matching must keep a northern gas addition off a Houston gas row."""
    base = _base_case()
    base["genfuel"] = ["ng", "ng"]
    scenario, _ = assemble_scenario(
        base,
        assets=_assets(),
        bus_geography=_bus_geography(),
        dc_loads=None,
        portfolio="generation-storage",
    )

    assert scenario["gen"][0, PMAX] == pytest.approx(130.0)
    assert scenario["gen"][1, PMAX] == pytest.approx(50.0)


def test_generation_portfolio_treats_ccgt_as_gas():
    """A CCGT addition expands existing gas rows exactly as OCGT does."""
    scenario, info = assemble_scenario(
        _base_case(),
        assets=_assets("CCGT"),
        bus_geography=_bus_geography(),
        dc_loads=None,
        portfolio="generation",
    )

    assert info["expanded_generator_rows"] == 2
    assert info["new_generator_rows"] == 0
    assert scenario["gen"][:, PMAX].sum() == pytest.approx(180.0)
    assert "other" not in scenario["genfuel"]


def test_generation_storage_portfolio_injects_ccgt_into_zone_gas():
    """A CCGT addition is injected into zone-matched gas rows."""
    base = _base_case()
    base["genfuel"] = ["ng", "ng"]
    scenario, info = assemble_scenario(
        base,
        assets=_assets("CCGT"),
        bus_geography=_bus_geography(),
        dc_loads=None,
        portfolio="generation-storage",
    )

    assert info["matched_gas_assets"] == 1
    assert info["new_generator_rows"] == 0
    assert scenario["gen"][0, PMAX] == pytest.approx(130.0)
    assert scenario["gen"][1, PMAX] == pytest.approx(50.0)


def test_gas_assets_keep_their_carrier_label():
    """Fuel mapping must not rewrite the PyPSA carrier on the asset table."""
    assets = _assets("CCGT")
    assemble_scenario(
        _base_case(),
        assets=assets,
        bus_geography=_bus_geography(),
        dc_loads=None,
        portfolio="generation",
    )

    assert assets["carrier"].tolist() == ["CCGT"]


def test_redispatch_keeps_online_generation_within_capacity():
    """Redispatch must not exceed capacity while closing a feasible load gap."""
    case = _base_case()
    case["bus"][:, PD] = [0.0, 60.0, 30.0]

    dispatched, info = redispatch_generation(case, max_utilization=1.0)

    assert info["residual_mw"] == pytest.approx(0.0)
    assert np.all(dispatched["gen"][:, PG] <= dispatched["gen"][:, PMAX] + 1e-9)
    assert np.all(dispatched["gen"][:, PG] >= dispatched["gen"][:, PMIN] - 1e-9)
    assert case["gen"][:, PG].sum() == pytest.approx(30.0)


def test_unknown_portfolio_is_rejected():
    """A misspelled portfolio must not silently choose a dispatch rule."""
    with pytest.raises(ValueError, match="portfolio"):
        assemble_scenario(
            copy.deepcopy(_base_case()),
            assets=_assets(),
            bus_geography=_bus_geography(),
            dc_loads=None,
            portfolio="generaton",
        )
