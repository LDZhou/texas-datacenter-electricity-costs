from types import SimpleNamespace

from paper_pipeline.scripts.run_experiment import (
    CAPACITY_REPORT_CARRIERS,
    make_output_dir,
    make_output_dir_from_args,
)


def test_output_paths_are_unique_for_year_scenario_mode_and_config_variant(tmp_path):
    variants = [
        (2019, "none", "generation", None, None, "base"),
        (2020, "none", "generation", None, None, "base"),
        (2019, "none", "dispatch", None, None, "base"),
        (2019, "multiloc", "generation", "TOLAR", 500, "base"),
        (2019, "multiloc", "generation", "TOLAR", 500, "gas1.5"),
    ]
    paths = [make_output_dir(*item, results_root=tmp_path) for item in variants]
    assert len(set(paths)) == len(variants)
    # Keep the historical base-layout stable while adding named sensitivities.
    assert paths[0] == tmp_path / "2019" / "none_generation"
    assert paths[-1].name.endswith("_gas1.5")


def test_crossover_is_part_of_experiment_identity(tmp_path):
    common = (2023, "all2030", "full_tx", None, None, "gas1.5")
    crossover_zero = make_output_dir(*common, results_root=tmp_path, crossover=0)
    crossover_one = make_output_dir(*common, results_root=tmp_path, crossover=1)

    assert crossover_zero != crossover_one
    assert "_crossover0" in crossover_zero.name
    assert "_crossover1" in crossover_one.name


def test_cli_args_forward_crossover_to_output_identity(tmp_path):
    args = SimpleNamespace(
        year=2023,
        dc_scenario="all2030",
        mode="full_tx",
        location=None,
        scale=None,
        config_variant="crossover1",
        results_root=tmp_path,
        crossover=1,
    )

    assert "_crossover1_crossover1" in make_output_dir_from_args(args).name


def test_capacity_metrics_include_ccgt():
    assert "CCGT" in CAPACITY_REPORT_CARRIERS
