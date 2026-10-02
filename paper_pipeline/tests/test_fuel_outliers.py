import numpy as np
import pandas as pd
import pypsa

from paper_pipeline.scripts.fix_gas_fuel_outliers import find_outliers, fix_gas_fuel_outliers


def _network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2021-05-01", periods=3, freq="3h"))
    n.add("Bus", "bus")
    # implied fuel = (mc - vom) * eff: 20*0.5=10, 21*0.5=10.5, 30*0.33=9.9, cheap: 8*0.3=2.4
    for name, carrier, mc, eff in [("a", "CCGT", 25, 0.5), ("b", "CCGT", 26, 0.5), ("c", "OCGT", 35, 0.33), ("cheap", "OCGT", 13, 0.3), ("pv", "solar", 0, 1.0)]:
        n.add("Generator", name, bus="bus", carrier=carrier, p_nom=100, marginal_cost=mc, efficiency=eff)
    n.generators["vom_cost"] = [5, 5, 5, 5, np.nan]
    n.generators_t.marginal_cost = pd.DataFrame({"a": [25, 27, 25], "b": [26, 28, 26], "c": [35, 37, 35], "cheap": [13, 13, 13]}, index=n.snapshots)
    return n


def test_only_the_sub_threshold_unit_is_corrected_and_recorded():
    n = _network()
    outliers, _ = find_outliers(n, 0.5)
    assert list(outliers.index) == ["cheap"]
    record = fix_gas_fuel_outliers(n, 0.5)
    assert [e["name"] for e in record["corrected"]] == ["cheap"]
    # median implied fuel per snapshot ~ 10 -> new fuel component 10/0.3 = 33.3 + vom 5
    fixed = n.generators_t.marginal_cost["cheap"]
    assert abs(fixed.iloc[0] - (5 + 10.0 / 0.3)) < 0.2
    assert n.generators_t.marginal_cost["a"].tolist() == [25, 27, 25]
    assert n.generators.loc["c", "marginal_cost"] == 35
    assert n.meta["gas_fuel_outlier_fix"]["threshold"] == 0.5


def test_clean_network_is_left_untouched():
    n = _network()
    n.generators_t.marginal_cost["cheap"] = [30, 30, 30]
    before = n.generators_t.marginal_cost.copy()
    record = fix_gas_fuel_outliers(n, 0.5)
    assert record["corrected"] == []
    pd.testing.assert_frame_equal(n.generators_t.marginal_cost, before)
