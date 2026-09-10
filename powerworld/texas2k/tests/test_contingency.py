"""Tests for warning-tolerant N-0 and N-1 screening."""

import numpy as np
import pytest
from pypower.idx_brch import BR_STATUS, PF, RATE_A

from powerworld.texas2k.contingency import scan_n1, screen_n0


def _case():
    branch = np.zeros((3, 13), dtype=float)
    branch[:, RATE_A] = 100.0
    branch[:, BR_STATUS] = 1
    return {
        "version": "2",
        "baseMVA": 100.0,
        "bus": np.zeros((1, 13)),
        "gen": np.zeros((1, 21)),
        "branch": branch,
        "gencost": np.zeros((1, 6)),
    }


def _result(flows):
    branch = np.zeros((3, 17), dtype=float)
    branch[:, RATE_A] = 100.0
    branch[:, BR_STATUS] = 1
    branch[:, PF] = flows
    return {"branch": branch}


def test_screen_n0_requires_a_finite_converged_solution():
    """An unusable intact network cannot define an overload baseline."""
    with pytest.raises(RuntimeError, match="did not converge"):
        screen_n0(_case(), solver=lambda _: (_result([1, 2, 3]), False))
    with pytest.raises(RuntimeError, match="non-finite"):
        screen_n0(_case(), solver=lambda _: (_result([1, np.nan, 3]), True))


def test_scan_n1_records_nonfinite_solver_output_and_continues():
    """One islanded or failed outage must not discard finite contingencies."""
    calls = iter(
        [
            (_result([10.0, 20.0, 0.0]), True),
            (_result([0.0, np.nan, 0.0]), True),
            (_result([40.0, 0.0, 0.0]), True),
            (_result([5.0, 6.0, 0.0]), False),
        ]
    )

    with pytest.warns(RuntimeWarning, match="2 N-1 contingency diagnostics recorded"):
        result = scan_n1(_case(), solver=lambda _: next(calls), progress_every=0)

    assert [row.reason for row in result.skipped] == ["non_finite_flow", "not_converged"]
    assert [row.branch_id for row in result.skipped] == [0, 2]
    np.testing.assert_allclose(result.worst_loading, [40.0, 20.0, 0.0])
    assert result.worst_contingency.tolist() == [1, -1, -1]


def test_scan_n1_keeps_finite_flows_from_partially_nonfinite_result():
    """Finite islands must retain the supplied workflow's elementwise NaN behavior."""
    calls = iter(
        [
            (_result([10.0, 20.0, 0.0]), True),
            (_result([0.0, np.nan, 50.0]), True),
            (_result([10.0, 0.0, 0.0]), True),
            (_result([10.0, 20.0, 0.0]), True),
        ]
    )

    with pytest.warns(RuntimeWarning, match="1 N-1 contingency diagnostics recorded"):
        result = scan_n1(_case(), solver=lambda _: next(calls), progress_every=0)

    np.testing.assert_allclose(result.worst_loading, [10.0, 20.0, 50.0])
    assert result.worst_contingency.tolist() == [-1, -1, 0]


def test_scan_n1_records_solver_exceptions():
    """A solver exception should appear in diagnostics without ending the sweep."""
    calls = 0

    def solver(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("island")
        return _result([10.0, 20.0, 0.0]), True

    with pytest.warns(RuntimeWarning, match="1 N-1 contingency diagnostics recorded"):
        result = scan_n1(_case(), solver=solver, progress_every=0)

    assert result.skipped[0].reason == "exception"
    assert "ValueError" in result.skipped[0].detail
    assert calls == 4


def test_scan_n1_skips_inactive_base_branches():
    """Opening an already disabled row must not create a duplicate contingency."""
    case = _case()
    case["branch"][2, BR_STATUS] = 0
    calls = 0

    def solver(_):
        nonlocal calls
        calls += 1
        return _result([10.0, 20.0, 0.0]), True

    result = scan_n1(case, solver=solver, progress_every=0)

    assert calls == 3
    assert result.skipped == []
