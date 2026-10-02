from pathlib import Path

import pandas as pd
import yaml
from jsonschema import Draft7Validator

from paper_pipeline.scripts.generate_runtime_config import make_config


ROOT = Path(__file__).resolve().parents[1]


def _render(template, year, tmp_path):
    output = tmp_path / f"{template.stem}_{year}.yaml"
    make_config(template, year, output)
    return yaml.safe_load(output.read_text())


def test_full_year_window_is_half_open_through_january_first_for_common_and_leap_years(tmp_path):
    template = ROOT / "config" / "full_year.yaml"
    for year, hours in [(2019, 8760), (2020, 8784)]:
        config = _render(template, year, tmp_path)
        snapshots = config["snapshots"]
        assert snapshots == {
            "start": f"{year}-01-01", "end": f"{year + 1}-01-01", "inclusive": "left"
        }
        index = pd.date_range(snapshots["start"], snapshots["end"],
                              inclusive=snapshots["inclusive"], freq="3h")
        assert len(index) * 3 == hours


def test_seasonal_window_is_unchanged(tmp_path):
    config = _render(ROOT / "config" / "seasonal.yaml", 2020, tmp_path)
    assert config["snapshots"] == {
        "start": "2020-05-01", "end": "2020-10-31", "inclusive": "left"
    }


def test_runtime_config_merges_tracked_layers_and_validates(tmp_path):
    config = _render(ROOT / "config" / "seasonal.yaml", 2019, tmp_path)
    schema = yaml.safe_load(
        (ROOT.parent / "workflow" / "schemas" / "config.schema.yaml").read_text()
    )

    errors = list(Draft7Validator(schema).iter_errors(config))

    assert errors == []
    assert config["renewable"]["dataset"] == "atlite"
    assert config["renewable_scenarios"] == ["historical"]
    assert config["lines"]["types"]
    assert config["electricity"]["extendable_carriers"]["Generator"] == [
        "OCGT", "CCGT", "solar", "onwind"
    ]
