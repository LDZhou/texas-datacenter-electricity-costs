import pandas as pd
import pytest

from paper_pipeline.scripts.summarize_revision import (
    Identity,
    MetadataMismatch,
    check_inventory,
    capped_load_weighted_mean,
    capped_temporal_load_weighted_mean,
    parse_experiment_dir,
    make_figures,
    pair_cases,
    siting_tables,
    validate_pair,
    weighted_unserved_mwh,
)
from paper_pipeline.scripts.revision_matrix import build_matrix


def fingerprints(identity, start="same", dc=None):
    return {"starting_network_sha256": start, "dc_profile_sha256": dc,
            "model_contract_version": 1, "candidate_carriers": None,
            "capital_cost_factor": 1.0, "year": identity.year,
            "horizon": identity.horizon, "mode": identity.mode,
            "config_variant": identity.variant, "crossover": identity.crossover}


def test_parser_keeps_variant_suffix_and_identity():
    identity = parse_experiment_dir(
        "all2030_full_tx_crossover0_gas1.5", group="sensitivity", year=2023
    )
    assert identity.scenario == "all2030"
    assert identity.mode == "full_tx"
    assert identity.variant == "gas1.5"
    assert identity.crossover == 0
    assert identity.horizon == "seasonal"


def test_capped_lmp_uses_load_and_snapshot_weights():
    prices = pd.DataFrame({"a": [100.0, 10000.0], "b": [10.0, 10.0]})
    load = pd.DataFrame({"a": [1.0, 3.0], "b": [1.0, 1.0]})
    assert capped_load_weighted_mean(prices, load, pd.Series([1.0, 2.0]), 5000.0) == pytest.approx(3013.0)
    # The headline retains the manuscript's flat-customer convention: first
    # use mean bus-load shares, then average intervals with time weights.
    assert capped_temporal_load_weighted_mean(prices, load, pd.Series([1.0, 2.0]), 5000.0) == pytest.approx(2247.7777778)


def test_unserved_energy_applies_component_sign_and_weights():
    dispatch = pd.DataFrame({"shed": [2.0, 4.0]})
    signs = pd.Series({"shed": 0.001})
    assert weighted_unserved_mwh(dispatch, signs, pd.Series([1.0, 3.0])) == pytest.approx(0.014)


def test_pair_rejects_mismatched_horizon_or_base_fingerprint():
    baseline = Identity("seasonal", "seasonal", 2023, "none", "full_tx", "base", 0)
    scenario = Identity("full_year", "full_year", 2023, "all2030", "full_tx", "base", 0)
    with pytest.raises(MetadataMismatch, match="horizon"):
        validate_pair(
            baseline, fingerprints(baseline), scenario, fingerprints(scenario, dc="dc"),
        )

    same_horizon = Identity("seasonal", "seasonal", 2023, "all2030", "full_tx", "base", 0)
    with pytest.raises(MetadataMismatch, match="starting_network_sha256"):
        validate_pair(
            baseline, fingerprints(baseline, start="base-a"), same_horizon,
            fingerprints(same_horizon, start="base-b", dc="dc"),
        )


def test_inventory_requires_every_matrix_case_and_numerics_are_seasonal():
    observed = [Identity("numerics", "seasonal", 2023, "none", "full_tx")]
    matrix = {"numerics": [{"year": 2023, "scenario": "none", "mode": "full_tx"},
                            {"year": 2023, "scenario": "all2030", "mode": "full_tx"}]}
    missing = check_inventory(observed, matrix, {"numerics"})
    assert len(missing) == 1
    assert missing[0].horizon == "seasonal"


def test_inventory_reads_real_stage_matrix_and_scale_key():
    matrix = build_matrix()
    siting_none = next(item for item in matrix["siting"] if item["scenario"] == "none")
    observed = [Identity(siting_none["group"], siting_none["horizon"], siting_none["year"],
                         siting_none["scenario"], siting_none["mode"], siting_none["variant"],
                         siting_none["crossover"], siting_none["location"], siting_none["scale"])]
    missing = check_inventory(observed, {"siting": [siting_none]}, {"siting"})
    assert missing == []


def test_siting_uses_its_own_matching_none_baseline():
    base = Identity("siting", "seasonal", 2023, "none", "generation_storage")
    site = Identity("siting", "seasonal", 2023, "multiloc", "generation_storage", location="AUSTIN", scale_mw=5000)
    rows = [{"identity": base, "path": "base", "fingerprints": fingerprints(base), "metrics": {}},
            {"identity": site, "path": "site", "fingerprints": fingerprints(site, dc="profile"), "metrics": {}}]
    assert len(pair_cases(rows)) == 1


def test_siting_tables_average_every_weather_year_per_cell():
    rows = []
    for year, eue in ((2019, 10.0), (2023, 30.0)):
        for scale in (500, 5000):
            rows.append({"year": year, "mode": "dispatch", "location": "AUSTIN", "scale_mw": scale,
                         "scenario_eue_mwh": eue * scale / 500, "delta_ercot_capped_lmp_$/MWh": float(year - 2000)})
    by_year, mean = siting_tables(pd.DataFrame(rows))
    assert len(by_year) == 4 and len(mean) == 2
    cell = mean[mean.scale_mw == 5000].iloc[0]
    assert cell.scenario_eue_mwh == pytest.approx(200.0)
    assert cell["delta_ercot_capped_lmp_$/MWh"] == pytest.approx(21.0)
    assert cell.n_weather_years == 2 and cell.years == "2019,2023"


def test_make_figures_renders_multi_year_siting_grid(tmp_path):
    rows = []
    for year in (2019, 2023):
        for mode in ("dispatch", "generation_tx"):
            for location in ("AUSTIN", "HOUSTON"):
                for scale in (500, 5000):
                    rows.append({"group": "siting", "year": year, "mode": mode, "location": location, "scale_mw": scale,
                                 "scenario_eue_mwh": float(scale), "delta_ercot_capped_lmp_$/MWh": 1.0,
                                 "delta_annual_investment_$": 0.0, "variant": "base"})
    make_figures(pd.DataFrame(rows), tmp_path)
    assert (tmp_path / "figures" / "siting.png").exists()
    mean = pd.read_csv(tmp_path / "siting_weather_year_mean.csv")
    assert len(mean) == 8 and set(mean.n_weather_years) == {2}
