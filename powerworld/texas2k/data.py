"""Validation and normalization at the PyPSA-to-Texas2k data boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ASSET_COLUMNS = (
    "year",
    "case",
    "location",
    "asset_type",
    "carrier",
    "zone",
    "bus",
    "bus_x",
    "bus_y",
    "added_mw",
)
PROJECT_COLUMNS = (
    "full_address",
    "State",
    "current_mw",
    "construction_mw",
    "planned_mw",
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def load_assets(path: Path) -> pd.DataFrame:
    """Read and validate a canonical PyPSA expansion-asset table."""
    return load_assets_from_frame(pd.read_csv(path))


def normalize_pypsa_assets(
    source: Path,
    destination: Path,
    *,
    year: int,
    case: str,
    location: str,
    include_storage: bool,
) -> pd.DataFrame:
    """Filter a coordinate-enriched PyPSA export into the canonical schema."""
    frame = pd.read_csv(source)
    _require_columns(frame, ASSET_COLUMNS, "PyPSA export")
    allowed = {"generation", "storage"} if include_storage else {"generation"}
    selected = frame.loc[
        (pd.to_numeric(frame["year"], errors="coerce") == year)
        & (frame["case"] == case)
        & (frame["location"] == location)
        & frame["asset_type"].isin(allowed),
        ASSET_COLUMNS,
    ].copy()
    selected = load_assets_from_frame(selected)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination, index=False, lineterminator="\n")
    return selected


def load_assets_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an in-memory canonical asset table without a temporary file."""
    _require_columns(frame, ASSET_COLUMNS, "asset table")
    candidate = frame.loc[:, ASSET_COLUMNS].copy()
    candidate["year"] = pd.to_numeric(candidate["year"], errors="raise").astype(int)
    for column in ("bus_x", "bus_y", "added_mw"):
        candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
    if not np.isfinite(candidate[["bus_x", "bus_y"]].to_numpy(dtype=float)).all():
        raise ValueError("asset table requires finite bus_x and bus_y coordinates")
    if not np.isfinite(candidate["added_mw"].to_numpy(dtype=float)).all():
        raise ValueError("asset table requires finite added_mw")
    if (candidate["added_mw"] < 0).any():
        raise ValueError("asset table requires non-negative added_mw")
    unsupported = sorted(set(candidate["asset_type"].dropna()) - {"generation", "storage"})
    if unsupported:
        raise ValueError(f"asset table has unsupported asset_type values: {unsupported}")
    if candidate[["case", "location", "asset_type", "carrier", "zone", "bus"]].isna().any().any():
        raise ValueError("asset table has missing identifier values")
    return candidate.reset_index(drop=True)


def load_datacenter_injections(
    projects_path: Path,
    matches_path: Path,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return positive matched data-center projects and transparent row totals."""
    projects = pd.read_csv(projects_path)
    matches = pd.read_csv(matches_path)
    _require_columns(projects, PROJECT_COLUMNS, "data-center project table")
    _require_columns(matches, ("full_address", "nearest_sub_name"), "bus-match table")
    if "rescue_status" in matches.columns:
        status_column = "rescue_status"
        matched_value = "RESCUABLE"
    elif "match_status" in matches.columns:
        status_column = "match_status"
        matched_value = "MATCHED"
    else:
        raise ValueError("bus-match table requires rescue_status or match_status")

    if len(projects) != len(matches):
        raise ValueError("project and bus-match tables must have the same row count")
    project_addresses = projects["full_address"].fillna("").astype(str).str.strip()
    match_addresses = matches["full_address"].fillna("").astype(str).str.strip()
    if not project_addresses.equals(match_addresses):
        raise ValueError("project and bus-match tables must use the same row order")

    capacities = projects[["current_mw", "construction_mw", "planned_mw"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    combined = projects.loc[:, PROJECT_COLUMNS].copy()
    combined["total_mw"] = capacities.sum(axis=1, min_count=1).fillna(0.0)
    for column in matches.columns:
        if column not in combined.columns and column != "total_mw":
            combined[column] = matches[column].to_numpy()

    matched = combined[status_column].eq(matched_value)
    injected = combined.loc[matched & combined["total_mw"].gt(0)].copy()
    injected["match_status"] = "MATCHED"
    stats: dict[str, float | int] = {
        "source_rows": int(len(combined)),
        "matched_rows": int(matched.sum()),
        "injected_rows": int(len(injected)),
        "injected_mw": float(injected["total_mw"].sum()),
    }
    return injected.reset_index(drop=True), stats
