"""MATPOWER parsing and PYPOWER DC power-flow helpers."""

from __future__ import annotations

import copy
import re
import warnings
from pathlib import Path

import numpy as np
from pypower.api import ppoption, rundcpf
from pypower.idx_brch import PF, RATE_A

_COMMENT = re.compile(r"%.*$")


def _without_comments(text: str) -> str:
    return "\n".join(_COMMENT.sub("", line) for line in text.splitlines())


def _parse_scalar(text: str, name: str) -> float:
    match = re.search(rf"mpc\.{re.escape(name)}\s*=\s*([-+\d.eE]+)\s*;", text)
    if match is None:
        raise ValueError(f"MATPOWER case is missing scalar {name}")
    return float(match.group(1))


def _parse_matrix(text: str, name: str, *, required: bool = True) -> np.ndarray:
    match = re.search(rf"mpc\.{re.escape(name)}\s*=\s*\[(.*?)\]\s*;", text, re.DOTALL)
    if match is None:
        if required:
            raise ValueError(f"MATPOWER case is missing matrix {name}")
        return np.empty((0, 0), dtype=float)

    rows: list[list[float]] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip().rstrip(";").replace(",", " ")
        if not line:
            continue
        try:
            rows.append([float(token.strip("'")) for token in line.split()])
        except ValueError as exc:
            raise ValueError(f"MATPOWER matrix {name} contains a non-numeric value") from exc

    if not rows:
        if required:
            raise ValueError(f"MATPOWER matrix {name} is empty")
        return np.empty((0, 0), dtype=float)
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"MATPOWER matrix {name} has inconsistent row widths: {sorted(widths)}")
    return np.asarray(rows, dtype=float)


def _parse_cell_strings(text: str, name: str) -> list[str]:
    match = re.search(rf"mpc\.{re.escape(name)}\s*=\s*\{{(.*?)\}}\s*;", text, re.DOTALL)
    if match is None:
        return []
    return re.findall(r"'([^']*)'", match.group(1))


def load_matpower_case(path: Path) -> dict[str, object]:
    """Load the numeric arrays needed by PYPOWER from a MATPOWER ``.m`` case."""
    text = _without_comments(Path(path).read_text(encoding="utf-8-sig"))
    case = {
        "version": "2",
        "baseMVA": _parse_scalar(text, "baseMVA"),
        "bus": _parse_matrix(text, "bus"),
        "gen": _parse_matrix(text, "gen"),
        "branch": _parse_matrix(text, "branch"),
        "gencost": _parse_matrix(text, "gencost", required=False),
    }
    generator_fuels = _parse_cell_strings(text, "genfuel")
    if generator_fuels:
        if len(generator_fuels) != case["gen"].shape[0]:
            raise ValueError("MATPOWER genfuel row count does not match gen")
        case["genfuel"] = generator_fuels
    return case


def run_dc_power_flow(case: dict[str, object]) -> tuple[dict[str, object], bool]:
    """Run a quiet DC power flow without mutating the supplied case."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=PendingDeprecationWarning,
            module=r"pypower\..*",
        )
        result, converged = rundcpf(copy.deepcopy(case), ppoption(VERBOSE=0, OUT_ALL=0))
    return result, bool(converged)


def branch_loading(result: dict[str, object], ratings: np.ndarray | None = None) -> np.ndarray:
    """Return absolute branch loading as percent of the original MVA rating."""
    branches = np.asarray(result["branch"], dtype=float)
    branch_ratings = branches[:, RATE_A] if ratings is None else np.asarray(ratings, dtype=float)
    loading = np.zeros(branches.shape[0], dtype=float)
    rated = branch_ratings > 0
    loading[rated] = 100.0 * np.abs(branches[rated, PF]) / branch_ratings[rated]
    return loading
