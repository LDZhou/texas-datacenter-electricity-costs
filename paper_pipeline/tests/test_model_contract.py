from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from paper_pipeline.scripts.dc_common import create_dc_load_profile
from paper_pipeline.scripts.model_contract import (
    load_dc_profile_table,
    profile_fingerprint,
    validate_frozen_starting_network,
    validate_solution,
)


def test_profiles_use_a_process_independent_stable_seed():
    snapshots = pd.date_range("2023-01-01", periods=8, freq="3h")
    first = create_dc_load_profile(snapshots, 100, seed_key="DC_bus-A")
    second = create_dc_load_profile(snapshots, 100, seed_key="DC_bus-A")
    other = create_dc_load_profile(snapshots, 100, seed_key="DC_bus-B")

    pd.testing.assert_series_equal(first, second)
    assert not first.equals(other)


def test_persisted_profile_requires_exact_snapshots_and_has_stable_fingerprint(tmp_path):
    snapshots = pd.date_range("2023-01-01", periods=2, freq="3h")
    source = tmp_path / "profiles.csv"
    pd.DataFrame({"DC_A": [90.0, 91.0]}, index=snapshots).to_csv(source)

    loaded = load_dc_profile_table(source, snapshots)
    assert loaded.columns.tolist() == ["DC_A"]
    assert profile_fingerprint(loaded) == profile_fingerprint(loaded.copy())

    with pytest.raises(ValueError, match="snapshots"):
        load_dc_profile_table(source, snapshots[:1])


def test_frozen_starting_network_rejects_extendability_and_nonfinite_capacity():
    good = SimpleNamespace(
        generators=pd.DataFrame({"p_nom": [1.0], "p_nom_extendable": [False]}),
        storage_units=pd.DataFrame(), stores=pd.DataFrame(), links=pd.DataFrame(),
        lines=pd.DataFrame({"s_nom": [2.0], "s_nom_extendable": [False]}),
    )
    validate_frozen_starting_network(good)

    bad = SimpleNamespace(
        generators=pd.DataFrame({"p_nom": [np.nan], "p_nom_extendable": [True]}),
        storage_units=pd.DataFrame(), stores=pd.DataFrame(), links=pd.DataFrame(),
        lines=pd.DataFrame({"s_nom": [2.0], "s_nom_extendable": [False]}),
    )
    with pytest.raises(ValueError, match="extendable|non-finite"):
        validate_frozen_starting_network(bad)


def test_solution_balance_accounts_for_branch_port_flows_not_raw_bus_injection(caplog):
    import pypsa

    snapshots = pd.date_range("2023-01-01", periods=1, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "source")
    network.add("Bus", "sink")
    network.add("Line", "line", bus0="source", bus1="sink", s_nom=100.0, s_max_pu=0.85)
    network._objective = 0.0
    network.buses_t.p = pd.DataFrame(
        {"source": [85.0], "sink": [-85.0]}, index=snapshots)
    network.lines_t.p0 = pd.DataFrame({"line": [85.0]}, index=snapshots)
    network.lines_t.p1 = pd.DataFrame({"line": [-85.0]}, index=snapshots)

    assert validate_solution(network, "ok", "optimal")["line_upper_mw"] == 0.0

    network.lines_t.p0.loc[:, "line"] = 90.0
    network.buses_t.p.loc[:, "source"] = 90.0
    with caplog.at_level("WARNING"):
        validate_solution(network, "ok", "optimal")
    assert "line_upper_mw" in caplog.text


def test_solution_records_and_warns_above_ten_kw_without_rejecting(caplog):
    import pypsa

    snapshots = pd.date_range("2023-01-01", periods=1, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "source")
    network.add("Bus", "sink")
    network.add("Line", "line", bus0="source", bus1="sink", s_nom=100.0)
    network._objective = 0.0
    network.lines_t.p0 = pd.DataFrame({"line": [85.0]}, index=snapshots)
    network.lines_t.p1 = pd.DataFrame({"line": [-85.0]}, index=snapshots)
    network.buses_t.p = pd.DataFrame(
        {"source": [85.0063], "sink": [-85.0]}, index=snapshots
    )

    residuals = validate_solution(network, "ok", "optimal")
    assert residuals["bus_balance_mw"] == pytest.approx(0.0063)

    network.buses_t.p.loc[:, "source"] = 85.02
    with caplog.at_level("WARNING"):
        residuals = validate_solution(network, "ok", "optimal")
    assert residuals["bus_balance_mw"] == pytest.approx(0.02)
    assert "bus_balance_mw" in caplog.text
    assert "continuing" in caplog.text
