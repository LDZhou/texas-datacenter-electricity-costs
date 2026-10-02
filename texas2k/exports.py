"""Normalized, stable-identity exports for Texas2k engineering results."""

from __future__ import annotations

import pandas as pd

UPGRADE_VALUE_COLUMNS = (
    "new_mw",
    "added_mw",
    "rate_factor",
    "upgrade_type",
    "loading_after_pct",
    "cleared",
    "cost_musd",
)


def _require_unique_branch_ids(frame: pd.DataFrame, label: str) -> None:
    if "branch_id" not in frame.columns:
        raise ValueError(f"{label} requires branch_id")
    if frame["branch_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate branch_id values")


def build_normalized_exports(
    upgrades: pd.DataFrame,
    branches: pd.DataFrame,
    *,
    portfolio: str,
    criterion: str,
    recovery: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build upgrade and all-branch tables joined only by stable branch ID."""
    _require_unique_branch_ids(upgrades, "upgrade table")
    _require_unique_branch_ids(branches, "branch table")
    metadata = {
        "portfolio": portfolio,
        "criterion": criterion.upper(),
        "recovery": recovery.upper(),
    }
    upgraded = upgrades.rename(columns={"capex_musd": "cost_musd"}).copy()
    for column, value in reversed(tuple(metadata.items())):
        if column in upgraded.columns:
            upgraded[column] = value
        else:
            upgraded.insert(0, column, value)

    all_branches = branches.copy()
    for column, value in reversed(tuple(metadata.items())):
        if column in all_branches.columns:
            all_branches[column] = value
        else:
            all_branches.insert(0, column, value)
    indexed = upgraded.set_index("branch_id")
    all_branches["is_upgraded"] = all_branches["branch_id"].isin(indexed.index)
    for column in UPGRADE_VALUE_COLUMNS:
        if column == "cost_musd":
            all_branches[column] = all_branches["branch_id"].map(indexed.get(column, pd.Series(dtype=float))).fillna(0.0)
        elif column in indexed.columns:
            all_branches[column] = all_branches["branch_id"].map(indexed[column])
    return upgraded, all_branches


def summarize_exports(
    upgrades: pd.DataFrame,
    *,
    portfolio: str,
    criterion: str,
    recovery: str,
) -> dict[str, object]:
    """Summarize costs from upgrade detail, never from the all-branch view."""
    cleared = upgrades.get("cleared", pd.Series(False, index=upgrades.index)).fillna(False).astype(bool)
    cost_column = "cost_musd" if "cost_musd" in upgrades.columns else "capex_musd"
    cost = pd.to_numeric(upgrades.get(cost_column, pd.Series(0.0, index=upgrades.index)), errors="coerce").fillna(0.0)
    return {
        "portfolio": portfolio,
        "criterion": criterion.upper(),
        "recovery": recovery.upper(),
        "n_upgraded": int(len(upgrades)),
        "cost_musd": float(cost.sum()),
        "cost_busd": float(cost.sum() / 1000),
        "cleared": int(cleared.sum()),
        "residual": int((~cleared).sum()),
    }
