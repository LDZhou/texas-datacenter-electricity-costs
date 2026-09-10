"""Fixed engineering upgrade rules used by the Texas2k screening workflow."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pypower.idx_brch import (
    BR_B,
    BR_R,
    BR_X,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    RATE_B,
    RATE_C,
    T_BUS,
)

from powerworld.texas2k.dispatch import AREA_TO_ZONE

TIER_EDGES = (
    (0, 80, "T1_normal"),
    (80, 90, "T2_watch"),
    (90, 100, "T3_prevent"),
    (100, 110, "T4_light"),
    (110, 125, "T5_moderate"),
    (125, 150, "T6_heavy"),
    (150, np.inf, "T7_structural"),
)
MAX_UPGRADE_FACTOR = 3.0
DETOUR_FACTOR = 1.30
UPGRADE_FACTORS = {
    ("T4_light", "line"): (1.40, 0.90, 0.99, 1.00),
    ("T5_moderate", "line"): (1.75, 0.78, 1.02, 1.00),
    ("T6_heavy", "line"): (2.20, 0.50, 0.50, 2.00),
    ("T7_structural", "line"): (2.50, 0.40, 0.45, 2.20),
    ("T4_light", "transformer"): (1.50, 0.67, 0.67, 1.00),
    ("T5_moderate", "transformer"): (1.75, 0.57, 0.57, 1.00),
    ("T6_heavy", "transformer"): (2.00, 0.50, 0.50, 1.00),
    ("T7_structural", "transformer"): (3.00, 0.33, 0.33, 1.00),
}
LINE_COST = {
    115: {"T4_light": 0.85, "T5_moderate": 1.50, "T6_heavy": 2.30, "T7_structural": 4.75},
    161: {"T4_light": 1.05, "T5_moderate": 1.90, "T6_heavy": 2.90, "T7_structural": 5.75},
    230: {"T4_light": 1.35, "T5_moderate": 2.15, "T6_heavy": 3.65, "T7_structural": 7.00},
    345: {"T4_light": 1.80, "T5_moderate": 3.00, "T6_heavy": 5.00, "T7_structural": 8.75},
    500: {"T4_light": 2.50, "T5_moderate": 4.25, "T6_heavy": 6.75, "T7_structural": 10.50},
}
TRANSFORMER_COST = {
    115: {"T4_light": 3.0, "T5_moderate": 4.5, "T6_heavy": 7.5, "T7_structural": 11.0},
    161: {"T4_light": 4.5, "T5_moderate": 7.0, "T6_heavy": 11.0, "T7_structural": 16.0},
    230: {"T4_light": 8.0, "T5_moderate": 11.5, "T6_heavy": 17.0, "T7_structural": 25.0},
    345: {"T4_light": 15.0, "T5_moderate": 20.0, "T6_heavy": 28.0, "T7_structural": 40.0},
    500: {"T4_light": 20.0, "T5_moderate": 27.5, "T6_heavy": 40.0, "T7_structural": 55.0},
}
UPGRADE_LABELS = {
    ("T4_light", "line"): "Reconductor(ACSR->ACSS)",
    ("T5_moderate", "line"): "HTLS Reconductor",
    ("T6_heavy", "line"): "New Parallel Circuit",
    ("T7_structural", "line"): "New Corridor/Aggressive Parallel",
    ("T4_light", "transformer"): "Transformer Replace 1.5x",
    ("T5_moderate", "transformer"): "Transformer Replace 1.75x",
    ("T6_heavy", "transformer"): "Parallel Transformer 2x",
    ("T7_structural", "transformer"): "Major Transformer 3x",
}


def classify_loading(percent: float) -> str:
    """Map a loading percentage to the supplied seven-tier convention."""
    if pd.isna(percent):
        return "T0_na"
    for lower, upper, tier in TIER_EDGES:
        if lower <= percent < upper:
            return tier
    return "T7_structural"


def _voltage_bucket(kv: float) -> int:
    if pd.isna(kv) or kv < 140:
        return 115
    if kv < 200:
        return 161
    if kv < 290:
        return 230
    if kv < 425:
        return 345
    return 500


def estimate_upgrade_cost(row: Mapping[str, object]) -> tuple[float, str]:
    """Return capital cost in million 2024 USD and component kind."""
    tier = str(row["tier"])
    component = str(row["component"])
    if (tier, component) not in UPGRADE_FACTORS:
        return 0.0, "none"
    high_kv = row.get("high_kv", row.get("high_kV", np.nan))
    voltage = _voltage_bucket(float(high_kv))
    if component == "transformer":
        return float(TRANSFORMER_COST[voltage][tier]), "transformer"
    distance = row.get("distance_mile", np.nan)
    distance = float(distance) if pd.notna(distance) and float(distance) > 0 else 30.0
    return float(LINE_COST[voltage][tier] * distance * DETOUR_FACTOR), "line"


def haversine_miles(longitude_1, latitude_1, longitude_2, latitude_2):
    """Calculate great-circle distance in miles for scalars or arrays."""
    radius = 3958.8
    phi_1, phi_2 = np.radians(latitude_1), np.radians(latitude_2)
    delta_phi = np.radians(np.asarray(latitude_2) - np.asarray(latitude_1))
    delta_lambda = np.radians(np.asarray(longitude_2) - np.asarray(longitude_1))
    value = np.sin(delta_phi / 2) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def branch_table(result: dict[str, object], bus_geography: pd.DataFrame) -> pd.DataFrame:
    """Convert a solved PYPOWER branch matrix to a stable-ID result table."""
    branches = np.asarray(result["branch"], dtype=float)
    apparent_from = np.sqrt(branches[:, PF] ** 2 + branches[:, QF] ** 2)
    apparent_to = np.sqrt(branches[:, PT] ** 2 + branches[:, QT] ** 2)
    apparent = np.maximum(apparent_from, apparent_to)
    table = pd.DataFrame(
        {
            "branch_id": np.arange(len(branches), dtype=int),
            "from_bus": branches[:, F_BUS].astype(int),
            "to_bus": branches[:, T_BUS].astype(int),
            "rate_a": branches[:, RATE_A],
            "mva": apparent,
        }
    )
    table["loading_pct"] = np.where(table["rate_a"] > 0, 100 * table["mva"] / table["rate_a"], np.nan)
    table["tier"] = table["loading_pct"].map(classify_loading)
    geography = bus_geography.set_index("bus_num")[["longitude", "latitude", "base_kv", "area_name"]]
    for side in ("from", "to"):
        matched = geography.reindex(table[f"{side}_bus"]).reset_index(drop=True)
        table[f"{side}_lon"] = matched["longitude"].to_numpy()
        table[f"{side}_lat"] = matched["latitude"].to_numpy()
        table[f"{side}_kv"] = matched["base_kv"].to_numpy()
    table["component"] = np.where(table["from_kv"] != table["to_kv"], "transformer", "line")
    table["high_kv"] = table[["from_kv", "to_kv"]].max(axis=1)
    table["distance_mile"] = haversine_miles(
        table["from_lon"],
        table["from_lat"],
        table["to_lon"],
        table["to_lat"],
    )
    area_by_bus = bus_geography.set_index("bus_num")["area_name"].to_dict()
    table["zone"] = [AREA_TO_ZONE.get(area_by_bus.get(int(bus))) for bus in table["from_bus"]]
    return table


def apply_upgrades(
    case: dict[str, object],
    targets: pd.DataFrame,
    *,
    tier_column: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Apply fixed rating and impedance factors to target branch IDs."""
    upgraded = copy.deepcopy(case)
    branches = upgraded["branch"]
    records: list[dict[str, object]] = []
    for row in targets.to_dict(orient="records"):
        component = str(row["component"])
        tier = str(row[tier_column])
        key = (tier, component)
        if key not in UPGRADE_FACTORS:
            continue
        rate_factor, resistance_factor, reactance_factor, charging_factor = UPGRADE_FACTORS[key]
        rate_factor = min(rate_factor, MAX_UPGRADE_FACTOR)
        branch_id = int(row["branch_id"])
        base_rate = float(branches[branch_id, RATE_A])
        if base_rate <= 0:
            continue
        branches[branch_id, RATE_A] = base_rate * rate_factor
        if branches[branch_id, RATE_B] > 0:
            branches[branch_id, RATE_B] *= rate_factor
        if branches[branch_id, RATE_C] > 0:
            branches[branch_id, RATE_C] *= rate_factor
        branches[branch_id, BR_R] *= resistance_factor
        branches[branch_id, BR_X] *= reactance_factor
        branches[branch_id, BR_B] *= charging_factor
        cost_row = dict(row)
        cost_row["tier"] = tier
        cost, _ = estimate_upgrade_cost(cost_row)
        records.append(
            {
                "branch_id": branch_id,
                "from_bus": int(row["from_bus"]),
                "to_bus": int(row["to_bus"]),
                "component": component,
                "tier": tier,
                "high_kv": row.get("high_kv", row.get("high_kV", np.nan)),
                "from_kv": row.get("from_kv", np.nan),
                "to_kv": row.get("to_kv", np.nan),
                "zone": row.get("zone"),
                "base_mw": round(base_rate, 1),
                "new_mw": round(base_rate * rate_factor, 1),
                "added_mw": round(base_rate * (rate_factor - 1), 1),
                "rate_factor": rate_factor,
                "upgrade_type": UPGRADE_LABELS[key],
                "capex_musd": round(cost, 3),
                "distance_mile": round(float(row["distance_mile"]), 1)
                if pd.notna(row.get("distance_mile"))
                else np.nan,
                "from_lon": row.get("from_lon", np.nan),
                "from_lat": row.get("from_lat", np.nan),
                "to_lon": row.get("to_lon", np.nan),
                "to_lat": row.get("to_lat", np.nan),
            }
        )
    return upgraded, pd.DataFrame(records)
