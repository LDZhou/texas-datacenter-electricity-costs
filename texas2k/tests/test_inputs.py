"""Tests for the public Texas2k input boundary."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from texas2k.data import (
    ASSET_COLUMNS,
    load_assets,
    load_datacenter_injections,
    normalize_pypsa_assets,
)

INPUTS = Path(__file__).parents[1] / "inputs"


def _asset_row(**overrides):
    row = {
        "year": 2023,
        "case": "all2030_full_tx",
        "location": "ALL_2030",
        "asset_type": "generation",
        "carrier": "OCGT",
        "zone": "NORTH",
        "bus": "p480010 0",
        "bus_x": -97.0,
        "bus_y": 31.0,
        "added_mw": 1.0,
    }
    row.update(overrides)
    return row


def test_load_assets_rejects_missing_coordinates(tmp_path):
    """An asset without coordinates cannot be mapped into Texas2k."""
    path = tmp_path / "assets.csv"
    pd.DataFrame([_asset_row(bus_x=np.nan)]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="finite bus_x and bus_y"):
        load_assets(path)


def test_load_assets_rejects_negative_capacity(tmp_path):
    """Negative additions must not silently remove Texas2k capacity."""
    path = tmp_path / "assets.csv"
    pd.DataFrame([_asset_row(added_mw=-1.0)]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-negative added_mw"):
        load_assets(path)


def test_normalize_pypsa_assets_filters_selected_scenario(tmp_path):
    """Rows from other years and storage-disabled runs must stay out."""
    source = tmp_path / "source.csv"
    destination = tmp_path / "normalized.csv"
    pd.DataFrame(
        [
            _asset_row(added_mw=5.0),
            _asset_row(asset_type="storage", carrier="4hr_battery", added_mw=2.0),
            _asset_row(year=2022, added_mw=99.0),
        ]
    ).to_csv(source, index=False)

    result = normalize_pypsa_assets(
        source,
        destination,
        year=2023,
        case="all2030_full_tx",
        location="ALL_2030",
        include_storage=False,
    )

    assert list(result.columns) == list(ASSET_COLUMNS)
    assert result["added_mw"].tolist() == [5.0]
    assert pd.read_csv(destination)["added_mw"].tolist() == [5.0]


def test_datacenter_injections_keep_positive_matched_rows(tmp_path):
    """Excluded and zero-capacity projects must not enter injected load."""
    projects_path = tmp_path / "projects.csv"
    matches_path = tmp_path / "matches.csv"
    pd.DataFrame(
        {
            "full_address": ["a", "b", "c", "d"],
            "State": ["TX"] * 4,
            "current_mw": [10.0, 0.0, 2.0, 5.0],
            "construction_mw": [0.0, 0.0, 0.0, 0.0],
            "planned_mw": [0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(projects_path, index=False)
    pd.DataFrame(
        {
            "full_address": ["a", "b", "c", "d"],
            "nearest_sub_name": ["A", "B", "C", "D"],
            "rescue_status": ["RESCUABLE", "RESCUABLE", "OUTSIDE", "RESCUABLE"],
        }
    ).to_csv(matches_path, index=False)

    rows, stats = load_datacenter_injections(projects_path, matches_path)

    assert rows["total_mw"].sum() == pytest.approx(15.0)
    assert stats == {
        "source_rows": 4,
        "matched_rows": 3,
        "injected_rows": 2,
        "injected_mw": 15.0,
    }


def test_datacenter_tables_must_align_by_address(tmp_path):
    """Equal row counts cannot mask a reordered bus-match table."""
    projects_path = tmp_path / "projects.csv"
    matches_path = tmp_path / "matches.csv"
    pd.DataFrame(
        {
            "full_address": ["a", "b"],
            "State": ["TX", "TX"],
            "current_mw": [1.0, 2.0],
            "construction_mw": [0.0, 0.0],
            "planned_mw": [0.0, 0.0],
        }
    ).to_csv(projects_path, index=False)
    pd.DataFrame(
        {
            "full_address": ["b", "a"],
            "nearest_sub_name": ["B", "A"],
            "rescue_status": ["RESCUABLE", "RESCUABLE"],
        }
    ).to_csv(matches_path, index=False)

    with pytest.raises(ValueError, match="same row order"):
        load_datacenter_injections(projects_path, matches_path)


def test_bundled_inputs_match_documented_totals():
    """Public input replacement must not silently change the paper scenario."""
    projects = pd.read_csv(INPUTS / "datacenters" / "projects.csv")
    dc_rows, dc_stats = load_datacenter_injections(
        INPUTS / "datacenters" / "projects.csv",
        INPUTS / "datacenters" / "bus_matches.csv",
    )
    generation = load_assets(INPUTS / "pypsa" / "generation_only_2023.csv")
    generation_storage = load_assets(INPUTS / "pypsa" / "generation_storage_2023.csv")

    source_mw = projects[["current_mw", "construction_mw", "planned_mw"]].sum().sum()
    assert source_mw == pytest.approx(40293.907)
    assert dc_stats["matched_rows"] == 409
    assert dc_stats["injected_rows"] == 384
    assert dc_rows["total_mw"].sum() == pytest.approx(39864.607)
    assert generation["added_mw"].sum() == pytest.approx(37323.0)
    assert generation_storage["added_mw"].sum() == pytest.approx(108982.3)
