import numpy as np
import pandas as pd
import pypsa
import pytest

from paper_pipeline.scripts.scale_gas_costs import scale_gas_costs


def _network(with_timeseries=True):
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=2, freq="3h"))
    network.add("Bus", "bus")
    for carrier, cost in [("OCGT", 10), ("CCGT", 20), ("solar", 30)]:
        network.add("Generator", carrier, bus="bus", carrier=carrier,
                    p_nom=1, marginal_cost=cost)
    network.generators["fuel_cost"] = [8, 15, np.nan]
    network.generators["vom_cost"] = [2, 5, np.nan]
    if with_timeseries:
        network.generators_t.marginal_cost = pd.DataFrame(
            {"OCGT": [11, 12], "CCGT": [21, 22], "solar": [31, 32]},
            index=network.snapshots,
        )
        network.generators_t.fuel_cost = pd.DataFrame(
            {"OCGT": [9, 10], "CCGT": [16, 17]}, index=network.snapshots)
        network.generators_t.vom_cost = pd.DataFrame(
            {"OCGT": [2, 2], "CCGT": [5, 5]}, index=network.snapshots)
    return network


def test_scales_only_fuel_component_and_keeps_vom_fixed():
    network = _network()
    scale_gas_costs(network, 1.5)
    assert network.generators.marginal_cost.to_dict() == {
        "OCGT": 14.0, "CCGT": 27.5, "solar": 30.0
    }
    assert network.generators_t.marginal_cost.to_dict("list") == {
        "OCGT": [15.5, 17.0], "CCGT": [29.0, 30.5], "solar": [31, 32]
    }
    assert network.generators_t.fuel_cost.to_dict("list") == {
        "OCGT": [13.5, 15.0], "CCGT": [24.0, 25.5]
    }
    assert network.meta["gas_cost_scaling"]["factor"] == 1.5
    assert network.meta["gas_cost_scaling"]["carriers"] == ["CCGT", "OCGT"]


def test_rejects_unknown_gas_cost_decomposition_and_invalid_factor():
    network = _network(with_timeseries=False)
    scale_gas_costs(network, 2)
    assert network.generators.loc["CCGT", "marginal_cost"] == 35
    unknown = _network()
    unknown.generators = unknown.generators.drop(columns=["fuel_cost", "vom_cost"])
    with pytest.raises(ValueError, match="decomposition"):
        scale_gas_costs(unknown, 2)
    for factor in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            scale_gas_costs(_network(), factor)


def test_fuel_price_units_are_inferred_from_marginal_cost_and_factor_one_is_exact():
    network = _network(with_timeseries=False)
    network.generators.loc["OCGT", ["marginal_cost", "vom_cost", "fuel_cost"]] = [30.0, 5.0, 3.0]
    scale_gas_costs(network, 1.0)
    assert network.generators.loc["OCGT", "marginal_cost"] == 30.0
    assert network.generators.loc["OCGT", "fuel_cost"] == 3.0

    network = _network(with_timeseries=False)
    network.generators.loc["OCGT", ["marginal_cost", "vom_cost", "fuel_cost"]] = [30.0, 5.0, 3.0]
    scale_gas_costs(network, 0.5)
    assert network.generators.loc["OCGT", "marginal_cost"] == 17.5


def test_factor_one_preserves_static_and_time_series_marginal_costs_exactly():
    network = _network()
    static_before = network.generators.marginal_cost.copy()
    dynamic_before = network.generators_t.marginal_cost.copy()

    scale_gas_costs(network, 1.0)

    pd.testing.assert_series_equal(network.generators.marginal_cost, static_before)
    pd.testing.assert_frame_equal(
        network.generators_t.marginal_cost, dynamic_before,
        check_column_type=False, check_dtype=False,
    )
