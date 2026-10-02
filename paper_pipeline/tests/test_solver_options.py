import pytest

from paper_pipeline.scripts.run_experiment import GUROBI_OPTS, parse_solver_options


def test_overrides_are_typed_and_recorded():
    assert parse_solver_options(["ScaleFlag=2", "BarCorrectors=-1", "Note=abc"]) == {
        "ScaleFlag": 2, "BarCorrectors": -1, "Note": "abc"}
    assert parse_solver_options(None) == {}


@pytest.mark.parametrize("item", ["BarConvTol=1e-3", "crossover=1", "Method=1", "TimeLimit=10", "ScaleFlag", "=2"])
def test_contract_tolerances_and_malformed_items_are_rejected(item):
    with pytest.raises(ValueError):
        parse_solver_options([item])


def test_default_options_are_untouched_by_overrides():
    before = dict(GUROBI_OPTS)
    parse_solver_options(["ScaleFlag=2"])
    assert GUROBI_OPTS == before
