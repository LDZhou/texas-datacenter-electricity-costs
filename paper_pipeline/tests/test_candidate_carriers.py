from pathlib import Path

import pypsa
import yaml

from paper_pipeline.scripts.dc_common import prepare_for_mode


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["OCGT", "CCGT", "solar", "onwind"]


def _network():
    network = pypsa.Network()
    network.add("Bus", "bus")
    for carrier in ["OCGT", "CCGT", "solar", "onwind", "coal"]:
        network.add("Generator", carrier, bus="bus", carrier=carrier,
                    p_nom=1, capital_cost=10)
    return network


def test_configs_list_all_main_candidate_carriers_and_keep_penalty_separate():
    for name in ("seasonal.yaml", "full_year.yaml"):
        config = yaml.safe_load((ROOT / "config" / name).read_text())
        assert config["electricity"]["extendable_carriers"]["Generator"] == EXPECTED
        # 5,000 is the PyPSA-USA input in $/kWh; its sign convention yields
        # a much larger feasibility penalty than the $5,000/MWh report cap.
        assert config["solving"]["options"]["load_shedding"] == 5000


def test_generation_mode_unlocks_ccgt_by_default_and_allows_controlled_override():
    network = _network()
    prepare_for_mode(network, "generation")
    assert set(network.generators.index[network.generators.p_nom_extendable]) == set(EXPECTED)

    sensitivity = _network()
    prepare_for_mode(sensitivity, "generation",
                     candidate_carriers=["OCGT", "solar", "onwind"])
    assert set(sensitivity.generators.index[sensitivity.generators.p_nom_extendable]) == {
        "OCGT", "solar", "onwind"
    }
