"""Tests for fixed-rule Texas2k upgrade calculations."""

import numpy as np
import pandas as pd
import pytest
from pypower.idx_brch import BR_R, BR_X, RATE_A, RATE_B, RATE_C

from texas2k.upgrades import apply_upgrades, classify_loading, estimate_upgrade_cost


@pytest.mark.parametrize(
    ("loading", "tier"),
    [
        (79.9, "T1_normal"),
        (80.0, "T2_watch"),
        (90.0, "T3_prevent"),
        (100.0, "T4_light"),
        (110.0, "T5_moderate"),
        (125.0, "T6_heavy"),
        (150.0, "T7_structural"),
        (np.nan, "T0_na"),
    ],
)
def test_classify_loading_preserves_tier_boundaries(loading, tier):
    """A boundary shift would change the selected engineering action."""
    assert classify_loading(loading) == tier


def test_line_cost_uses_voltage_distance_and_detour():
    """Line capital cost must retain the supplied 1.30 route detour."""
    cost, kind = estimate_upgrade_cost(
        {
            "tier": "T4_light",
            "component": "line",
            "high_kv": 230.0,
            "distance_mile": 10.0,
        }
    )

    assert kind == "line"
    assert cost == pytest.approx(17.55)


def test_transformer_cost_does_not_scale_with_line_distance():
    """Transformer table values are lump sums in million 2024 USD."""
    cost, kind = estimate_upgrade_cost(
        {
            "tier": "T5_moderate",
            "component": "transformer",
            "high_kv": 345.0,
            "distance_mile": 100.0,
        }
    )

    assert kind == "transformer"
    assert cost == pytest.approx(20.0)


def test_apply_upgrades_changes_only_selected_branch_id():
    """Parallel circuits sharing endpoint buses must remain independently addressable."""
    branch = np.zeros((2, 13), dtype=float)
    branch[:, RATE_A] = [100.0, 100.0]
    branch[:, RATE_B] = [100.0, 100.0]
    branch[:, RATE_C] = [100.0, 100.0]
    branch[:, BR_R] = [0.1, 0.1]
    branch[:, BR_X] = [0.2, 0.2]
    case = {"branch": branch}
    targets = pd.DataFrame(
        [
            {
                "branch_id": 1,
                "from_bus": 1,
                "to_bus": 2,
                "component": "line",
                "tier": "T4_light",
                "high_kv": 230.0,
                "from_kv": 230.0,
                "to_kv": 230.0,
                "zone": "NORTH",
                "distance_mile": 10.0,
                "from_lon": -97.0,
                "from_lat": 31.0,
                "to_lon": -96.0,
                "to_lat": 31.0,
            }
        ]
    )

    upgraded_case, detail = apply_upgrades(case, targets, tier_column="tier")

    assert upgraded_case["branch"][0, RATE_A] == pytest.approx(100.0)
    assert upgraded_case["branch"][1, RATE_A] == pytest.approx(140.0)
    assert upgraded_case["branch"][1, BR_R] == pytest.approx(0.09)
    assert detail["branch_id"].tolist() == [1]
    assert detail["capex_musd"].tolist() == pytest.approx([17.55])
