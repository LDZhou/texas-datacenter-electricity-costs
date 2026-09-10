"""DC N-0 and exhaustive branch-outage N-1 screening."""

from __future__ import annotations

import copy
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from pypower.idx_brch import BR_STATUS, PF, RATE_A

from powerworld.texas2k.case import branch_loading, run_dc_power_flow

Solver = Callable[[dict[str, object]], tuple[dict[str, object], bool]]


@dataclass(frozen=True)
class SkippedContingency:
    """One outage whose solver output cannot enter worst-case loading."""

    branch_id: int
    reason: str
    detail: str = ""


@dataclass
class ScreeningResult:
    """Intact and worst-case loadings plus explicit skipped-outage records."""

    n0_loading: np.ndarray
    worst_loading: np.ndarray
    worst_contingency: np.ndarray
    skipped: list[SkippedContingency] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _validate_result(
    result: dict[str, object],
    converged: bool,
    expected_branches: int,
) -> np.ndarray:
    if not converged:
        raise RuntimeError("DC power flow did not converge")
    if "branch" not in result:
        raise RuntimeError("DC power flow result is missing branch data")
    branches = np.asarray(result["branch"], dtype=float)
    if branches.ndim != 2 or branches.shape[0] != expected_branches or branches.shape[1] <= PF:
        raise RuntimeError("DC power flow result has an invalid branch array")
    if not np.isfinite(branches[:, PF]).all():
        raise RuntimeError("DC power flow returned non-finite branch flow")
    return branches


def screen_n0(
    case: dict[str, object],
    *,
    solver: Solver = run_dc_power_flow,
) -> ScreeningResult:
    """Solve and validate the intact network."""
    result, converged = solver(copy.deepcopy(case))
    _validate_result(result, converged, np.asarray(case["branch"]).shape[0])
    loading = branch_loading(result, np.asarray(case["branch"])[:, RATE_A])
    return ScreeningResult(
        n0_loading=loading,
        worst_loading=loading.copy(),
        worst_contingency=np.full(len(loading), -1, dtype=int),
    )


def scan_n1(
    case: dict[str, object],
    *,
    solver: Solver = run_dc_power_flow,
    progress_every: int = 1000,
) -> ScreeningResult:
    """Scan every active branch outage, retaining only finite converged flows."""
    intact = screen_n0(case, solver=solver)
    base_branches = np.asarray(case["branch"], dtype=float)
    ratings = base_branches[:, RATE_A].copy()
    active = base_branches[:, BR_STATUS] == 1
    worst = intact.n0_loading.copy()
    worst_contingency = intact.worst_contingency.copy()
    skipped: list[SkippedContingency] = []
    started = time.monotonic()
    active_ids = np.flatnonzero(active)

    for sequence, branch_id in enumerate(active_ids, start=1):
        contingency = copy.deepcopy(case)
        contingency["branch"][branch_id, BR_STATUS] = 0
        try:
            result, converged = solver(contingency)
        except Exception as exc:  # solver libraries expose several exception types
            skipped.append(
                SkippedContingency(
                    branch_id=int(branch_id),
                    reason="exception",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if not converged:
            skipped.append(SkippedContingency(int(branch_id), "not_converged"))
            continue
        if "branch" not in result:
            skipped.append(SkippedContingency(int(branch_id), "missing_branch_data"))
            continue
        branches = np.asarray(result["branch"], dtype=float)
        if branches.ndim != 2 or branches.shape[0] != len(base_branches) or branches.shape[1] <= PF:
            skipped.append(SkippedContingency(int(branch_id), "invalid_branch_array"))
            continue
        finite_required = active.copy()
        finite_required[branch_id] = False
        if not np.isfinite(branches[finite_required, PF]).all():
            skipped.append(SkippedContingency(int(branch_id), "non_finite_flow"))
            continue
        loading = branch_loading(result, ratings)
        loading[branch_id] = 0.0
        loading[~active] = 0.0
        updated = np.isfinite(loading) & (loading > worst)
        worst[updated] = loading[updated]
        worst_contingency[updated] = int(branch_id)
        if progress_every > 0 and sequence % progress_every == 0:
            elapsed = time.monotonic() - started
            print(f"N-1 progress: {sequence}/{len(active_ids)} active outages ({elapsed:.1f} s)")

    elapsed = time.monotonic() - started
    if skipped:
        warnings.warn(
            f"{len(skipped)} N-1 contingencies skipped; see skipped_contingencies.csv",
            RuntimeWarning,
            stacklevel=2,
        )
    return ScreeningResult(
        n0_loading=intact.n0_loading,
        worst_loading=worst,
        worst_contingency=worst_contingency,
        skipped=skipped,
        elapsed_seconds=elapsed,
    )
