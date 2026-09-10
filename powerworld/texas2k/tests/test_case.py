"""Tests for MATPOWER parsing and DC branch loading."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from pypower.idx_brch import PF
from scipy.sparse.linalg import MatrixRankWarning

import powerworld.texas2k.case as case_module
from powerworld.texas2k.case import branch_loading, load_matpower_case, run_dc_power_flow

FIXTURES = Path(__file__).parent / "fixtures"
INPUTS = Path(__file__).parents[1] / "inputs"


def test_load_case_preserves_matrix_rows_and_base_mva():
    """Dropping a complete matrix row would corrupt every downstream index."""
    case = load_matpower_case(FIXTURES / "tiny_case.m")

    assert case["baseMVA"] == 100.0
    assert case["bus"].shape == (3, 13)
    assert case["gen"].shape == (1, 21)
    assert case["branch"].shape == (3, 13)
    assert case["gencost"].shape == (1, 6)


def test_load_case_reads_optional_generator_fuels():
    """Portfolio dispatch needs generator fuels in case row order."""
    case = load_matpower_case(INPUTS / "case" / "texas2k_series25_summer_peak.m")

    assert len(case["genfuel"]) == case["gen"].shape[0]
    assert set(case["genfuel"]) >= {"ng", "solar", "wind"}


def test_load_case_rejects_missing_required_matrix(tmp_path):
    """A malformed case must fail before a solver receives partial data."""
    malformed = tmp_path / "broken.m"
    malformed.write_text("mpc.baseMVA = 100;\nmpc.bus = [1 3 0;];\n")

    with pytest.raises(ValueError, match="gen"):
        load_matpower_case(malformed)


def test_branch_loading_uses_absolute_pf_and_zero_for_unrated_branch():
    """Reverse flow and zero ratings must not hide or create overloads."""
    branch = np.zeros((2, 17), dtype=float)
    branch[:, 5] = [100.0, 0.0]
    branch[:, PF] = [-30.0, 20.0]

    np.testing.assert_allclose(branch_loading({"branch": branch}), [30.0, 0.0])


def test_run_dc_power_flow_does_not_mutate_input():
    """Contingency copies rely on the base case remaining unchanged."""
    case = load_matpower_case(FIXTURES / "tiny_case.m")
    original = case["branch"].copy()

    result, converged = run_dc_power_flow(case)

    assert converged
    assert result["branch"].shape[1] >= 17
    np.testing.assert_array_equal(case["branch"], original)


def test_run_dc_power_flow_suppresses_solver_rank_warning(monkeypatch):
    """Per-outage rank warnings should become one aggregate skipped-case warning."""
    case = load_matpower_case(FIXTURES / "tiny_case.m")

    def rank_warning_solver(candidate, _options):
        warnings.warn("singular", MatrixRankWarning)
        return candidate, True

    monkeypatch.setattr(case_module, "rundcpf", rank_warning_solver)
    with warnings.catch_warnings(record=True) as captured:
        run_dc_power_flow(case)

    assert captured == []


def test_bundled_case_dimensions_match_texas2k_series25():
    """An accidental data replacement must be visible immediately."""
    case = load_matpower_case(INPUTS / "case" / "texas2k_series25_summer_peak.m")

    assert case["bus"].shape[0] == 2751
    assert case["gen"].shape[0] == 1099
    assert case["branch"].shape[0] == 5344
