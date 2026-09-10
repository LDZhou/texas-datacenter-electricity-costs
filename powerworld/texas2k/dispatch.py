"""Scenario assembly and dispatch rules for the Texas2k screening model."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
from pypower.idx_bus import BUS_I, PD, QD
from pypower.idx_gen import GEN_BUS, GEN_STATUS, MBASE, PG, PMAX, PMIN, QMAX, QMIN, VG

AREA_TO_ZONE = {
    "North": "NORTH",
    "North Central": "NORTH",
    "East": "NORTH",
    "South": "SOUTH",
    "South Central": "SOUTH",
    "West": "WEST",
    "Far West": "WEST",
    "Coast": "HOUSTON",
}
CARRIER_TO_FUEL = {
    "OCGT": "ng",
    "solar": "solar",
    "onwind": "wind",
    "4hr_battery_storage": "battery",
}
CAPACITY_FACTOR = {
    "nuclear": 0.95,
    "coal": 0.85,
    "hydro": 0.40,
    "solar": 0.55,
    "wind": 0.15,
    "other": 0.30,
    "dfo": 0.30,
    "wood": 0.30,
    "battery": 1.0,
}
LOAD_POWER_FACTOR = 0.98
GENERATOR_REACTIVE_FRACTION = 0.40


def _clean_station_name(name: object) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    value = re.sub(r"\s+", " ", str(name).strip().upper())
    while True:
        shortened = re.sub(r"\s+\d+(\.\d+)?\s*$", "", value)
        if shortened == value:
            return value.strip()
        value = shortened


def _strip_bus_suffix(name: object) -> str:
    value = re.sub(r"\s+", " ", str(name).strip().upper())
    without_voltage = re.sub(r"_\d+(\.\d+)?\s*$", "", value).strip()
    if without_voltage != value:
        return without_voltage
    return re.sub(r"-\d+\s*$", "", value).strip()


def _reactive_power(active_power: float, power_factor: float = LOAD_POWER_FACTOR) -> float:
    bounded = min(max(power_factor, 0.01), 0.999)
    return active_power * np.tan(np.arccos(bounded))


def map_datacenter_loads(
    datacenters: pd.DataFrame,
    bus_geography: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Map matched projects to station buses, weighted by nominal voltage."""
    buses = bus_geography.copy()
    buses["station_key"] = buses["station_clean"].map(_clean_station_name)
    buses["station_key_fallback"] = buses["station_clean"].map(_strip_bus_suffix)
    station_totals = datacenters.groupby("nearest_sub_name", dropna=True)["total_mw"].sum()
    records: list[tuple[int, float]] = []
    matched_mw = 0.0
    unmapped_mw = 0.0
    matched_substations = 0
    for station, capacity in station_totals.items():
        key = _clean_station_name(station)
        selected = buses[buses["station_key"] == key]
        if selected.empty:
            selected = buses[buses["station_key_fallback"] == key]
        if selected.empty:
            unmapped_mw += float(capacity)
            continue
        weights = selected["base_kv"].clip(lower=1.0).astype(float)
        weights /= weights.sum()
        records.extend(
            (int(bus_num), float(capacity * weight))
            for bus_num, weight in zip(selected["bus_num"], weights, strict=True)
        )
        matched_mw += float(capacity)
        matched_substations += 1
    mapped = pd.DataFrame(records, columns=["bus_num", "mw"])
    if not mapped.empty:
        mapped = mapped.groupby("bus_num", as_index=False)["mw"].sum()
    info: dict[str, float | int] = {
        "matched_substations": matched_substations,
        "mapped_buses": int(len(mapped)),
        "mapped_mw": matched_mw,
        "unmapped_mw": unmapped_mw,
    }
    return mapped, info


def _new_generator_row(template: np.ndarray, bus_num: int, capacity: float) -> np.ndarray:
    row = np.zeros(template.shape[1], dtype=float)
    row[GEN_BUS] = bus_num
    row[PMAX] = capacity
    row[PMIN] = 0.0
    row[QMAX] = GENERATOR_REACTIVE_FRACTION * capacity
    row[QMIN] = -GENERATOR_REACTIVE_FRACTION * capacity
    row[VG] = 1.0
    row[MBASE] = max(capacity / 0.9, 1.0)
    row[GEN_STATUS] = 1
    return row


def _nearest_bus(asset: pd.Series, bus_geography: pd.DataFrame) -> int:
    usable = bus_geography.dropna(subset=["longitude", "latitude"])
    distance = (usable["latitude"] - asset["bus_y"]) ** 2 + (usable["longitude"] - asset["bus_x"]) ** 2
    return int(usable.loc[distance.idxmin(), "bus_num"])


def _append_generators(
    case: dict[str, object],
    rows: list[np.ndarray],
    fuels: list[str],
) -> None:
    if not rows:
        return
    case["gen"] = np.vstack([case["gen"], np.asarray(rows)])
    gencost = np.asarray(case.get("gencost", np.empty((0, 0))))
    if gencost.size:
        case["gencost"] = np.vstack([gencost, np.tile(gencost[0], (len(rows), 1))])
    case.setdefault("genfuel", ["other"] * (case["gen"].shape[0] - len(rows)))
    case["genfuel"] = list(case["genfuel"]) + fuels


def _apply_generation_assets(
    case: dict[str, object],
    assets: pd.DataFrame,
    bus_geography: pd.DataFrame,
) -> dict[str, int]:
    generators = case["gen"]
    generator_buses = generators[:, GEN_BUS].astype(int)
    candidates = bus_geography[
        bus_geography["bus_num"].isin(np.unique(generator_buses))
    ].dropna(subset=["longitude", "latitude"])
    new_rows: list[np.ndarray] = []
    new_fuels: list[str] = []
    expanded_rows = 0
    for _, asset in assets.iterrows():
        capacity = float(asset["added_mw"])
        if capacity <= 0:
            continue
        if asset["carrier"] == "OCGT":
            distance = (candidates["latitude"] - asset["bus_y"]) ** 2 + (
                candidates["longitude"] - asset["bus_x"]
            ) ** 2
            nearby = candidates.loc[distance.nsmallest(min(8, len(distance))).index]
            indices = np.where(np.isin(generator_buses, nearby["bus_num"].to_numpy()))[0]
            if not len(indices):
                continue
            weights = np.clip(generators[indices, PMAX], 1.0, None)
            weights /= weights.sum()
            for index, weight in zip(indices, weights, strict=True):
                generators[index, PMAX] += capacity * weight
                generators[index, QMAX] = max(
                    generators[index, QMAX],
                    GENERATOR_REACTIVE_FRACTION * generators[index, PMAX],
                )
                generators[index, QMIN] = min(
                    generators[index, QMIN],
                    -GENERATOR_REACTIVE_FRACTION * generators[index, PMAX],
                )
                generators[index, MBASE] = max(
                    generators[index, MBASE],
                    generators[index, PMAX] / 0.9,
                )
                generators[index, GEN_STATUS] = 1
                expanded_rows += 1
        else:
            fuel = CARRIER_TO_FUEL.get(str(asset["carrier"]), "other")
            new_rows.append(_new_generator_row(generators, _nearest_bus(asset, bus_geography), capacity))
            new_fuels.append(fuel)
    _append_generators(case, new_rows, new_fuels)
    return {"expanded_generator_rows": expanded_rows, "new_generator_rows": len(new_rows)}


def _apply_generation_storage_assets(
    case: dict[str, object],
    assets: pd.DataFrame,
    bus_geography: pd.DataFrame,
) -> dict[str, int]:
    generators = case["gen"]
    fuels = np.asarray(case.get("genfuel", ["other"] * len(generators)))
    if len(fuels) != len(generators):
        raise ValueError("generator fuel labels must align with generator rows")
    area_by_bus = bus_geography.set_index("bus_num")["area_name"].to_dict()
    generator_zones = np.asarray(
        [AREA_TO_ZONE.get(area_by_bus.get(int(bus_num))) for bus_num in generators[:, GEN_BUS]]
    )
    new_rows: list[np.ndarray] = []
    new_fuels: list[str] = []
    matched_gas_assets = 0
    for _, asset in assets.iterrows():
        capacity = float(asset["added_mw"])
        fuel = CARRIER_TO_FUEL.get(str(asset["carrier"]))
        if capacity <= 0 or fuel is None:
            continue
        if fuel == "ng":
            indices = np.where(
                (fuels == "ng")
                & (generator_zones == asset["zone"])
                & (generators[:, GEN_STATUS] == 1)
            )[0]
            if len(indices):
                weights = np.clip(generators[indices, PMAX], 1.0, None)
                weights /= weights.sum()
                generators[indices, PMAX] += capacity * weights
                matched_gas_assets += 1
                continue
        new_rows.append(_new_generator_row(generators, _nearest_bus(asset, bus_geography), capacity))
        new_fuels.append(fuel)
    _append_generators(case, new_rows, new_fuels)
    return {"matched_gas_assets": matched_gas_assets, "new_generator_rows": len(new_rows)}


def redispatch_generation(
    case: dict[str, object],
    *,
    max_utilization: float = 0.98,
) -> tuple[dict[str, object], dict[str, float]]:
    """Balance active generation toward load using proportional available room."""
    if not 0 < max_utilization <= 1:
        raise ValueError("max_utilization must be within (0, 1]")
    dispatched = copy.deepcopy(case)
    generators = dispatched["gen"]
    online = generators[:, GEN_STATUS] == 1
    target = float(dispatched["bus"][:, PD].sum())
    initial = float(generators[online, PG].sum())
    gap = target - initial
    online_indices = np.where(online)[0]
    if gap > 0:
        limits = max_utilization * generators[online, PMAX]
        room = np.maximum(limits - generators[online, PG], 0.0)
    else:
        room = np.maximum(generators[online, PG] - generators[online, PMIN], 0.0)
    available = float(room.sum())
    moved = min(abs(gap), available)
    if moved > 0 and available > 0:
        direction = 1.0 if gap > 0 else -1.0
        generators[online_indices, PG] += direction * room * (moved / available)
    residual = target - float(generators[online, PG].sum())
    return dispatched, {
        "gap_mw": gap,
        "room_mw": available,
        "moved_mw": moved,
        "residual_mw": residual,
    }


def _dispatch_generation_storage(
    case: dict[str, object],
    original_generator_count: int,
) -> dict[str, float]:
    generators = case["gen"]
    fuels = np.asarray(case.get("genfuel", ["other"] * len(generators)))
    online = generators[:, GEN_STATUS] == 1
    non_gas_output = 0.0
    for index in np.where(online)[0]:
        fuel = fuels[index]
        if fuel == "ng":
            continue
        if index >= original_generator_count:
            generators[index, PG] = generators[index, PMAX] * CAPACITY_FACTOR.get(fuel, 0.30)
        non_gas_output += generators[index, PG]
    gas = np.where(online & (fuels == "ng"))[0]
    required = float(case["bus"][:, PD].sum()) - non_gas_output
    capacity = float(generators[gas, PMAX].sum())
    if not len(gas) or required <= 0:
        generators[gas, PG] = 0.0
    else:
        utilization = min(required / capacity, 0.98) if capacity > 0 else 0.0
        generators[gas, PG] = utilization * generators[gas, PMAX]
    return {
        "non_gas_pg_mw": non_gas_output,
        "gas_need_mw": required,
        "gas_capacity_mw": capacity,
        "gas_utilization": required / capacity if capacity > 0 else np.nan,
    }


def assemble_scenario(
    base_case: dict[str, object],
    *,
    assets: pd.DataFrame,
    bus_geography: pd.DataFrame,
    dc_loads: pd.DataFrame | None,
    portfolio: str,
) -> tuple[dict[str, object], dict[str, float | int]]:
    """Apply PyPSA assets, optional data-center load, and portfolio dispatch."""
    if portfolio not in {"generation", "generation-storage"}:
        raise ValueError("portfolio must be 'generation' or 'generation-storage'")
    scenario = copy.deepcopy(base_case)
    original_generators = scenario["gen"].shape[0]
    if portfolio == "generation":
        asset_info = _apply_generation_assets(scenario, assets, bus_geography)
    else:
        asset_info = _apply_generation_storage_assets(scenario, assets, bus_geography)
    info: dict[str, float | int] = {
        "asset_rows": int(len(assets)),
        "asset_mw": float(assets["added_mw"].sum()),
        "dc_applied_mw": 0.0,
        "mapped_dc_buses": 0,
        **asset_info,
    }
    if dc_loads is not None:
        mapped, mapping_info = map_datacenter_loads(dc_loads, bus_geography)
        bus_index = {int(bus_num): index for index, bus_num in enumerate(scenario["bus"][:, BUS_I])}
        for row in mapped.itertuples(index=False):
            index = bus_index.get(int(row.bus_num))
            if index is None:
                continue
            scenario["bus"][index, PD] += row.mw
            scenario["bus"][index, QD] += _reactive_power(row.mw)
            info["dc_applied_mw"] = float(info["dc_applied_mw"]) + float(row.mw)
        info["mapped_dc_buses"] = int(len(mapped))
        info["unmapped_dc_mw"] = float(mapping_info["unmapped_mw"])
    if portfolio == "generation-storage":
        info.update(_dispatch_generation_storage(scenario, original_generators))
    else:
        scenario, redispatch_info = redispatch_generation(scenario, max_utilization=1.0)
        info.update({f"redispatch_{key}": value for key, value in redispatch_info.items()})
    info["new_generator_rows"] = scenario["gen"].shape[0] - original_generators
    return scenario, info
