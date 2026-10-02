"""Tests for stable-identity Texas2k result exports."""

import pandas as pd
import pytest

from texas2k.exports import build_normalized_exports, summarize_exports


def _parallel_branches():
    return pd.DataFrame(
        [
            {
                "branch_id": 7,
                "from_bus": 1,
                "to_bus": 2,
                "component": "line",
                "zone": "NORTH",
                "rate_a": 100.0,
                "tier": "T4_light",
                "loading_pct": 101.0,
            },
            {
                "branch_id": 8,
                "from_bus": 1,
                "to_bus": 2,
                "component": "line",
                "zone": "NORTH",
                "rate_a": 100.0,
                "tier": "T5_moderate",
                "loading_pct": 120.0,
            },
        ]
    )


def _one_upgrade():
    return pd.DataFrame(
        [
            {
                "branch_id": 8,
                "from_bus": 1,
                "to_bus": 2,
                "component": "line",
                "zone": "NORTH",
                "base_mw": 100.0,
                "new_mw": 175.0,
                "added_mw": 75.0,
                "rate_factor": 1.75,
                "upgrade_type": "HTLS Reconductor",
                "tier": "T5_moderate",
                "loading_before_pct": 120.0,
                "loading_after_pct": 90.0,
                "cleared": True,
                "capex_musd": 4.0,
            }
        ]
    )


def test_allbranch_export_marks_only_selected_parallel_circuit():
    """Endpoint equality must never broadcast costs across parallel circuits."""
    upgraded, all_branches = build_normalized_exports(
        _one_upgrade(),
        _parallel_branches(),
        portfolio="generation-storage",
        criterion="n1",
        recovery="NEW",
    )

    assert upgraded["branch_id"].tolist() == [8]
    assert all_branches.set_index("branch_id")["is_upgraded"].to_dict() == {7: False, 8: True}
    assert all_branches.set_index("branch_id")["cost_musd"].to_dict() == pytest.approx({7: 0.0, 8: 4.0})


def test_duplicate_upgrade_branch_ids_are_rejected():
    """Duplicate branch identities would double count engineering capital cost."""
    duplicates = pd.concat([_one_upgrade(), _one_upgrade()], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate branch_id"):
        build_normalized_exports(
            duplicates,
            _parallel_branches(),
            portfolio="generation-storage",
            criterion="n1",
            recovery="NEW",
        )


def test_summary_cost_comes_from_upgrade_detail_only():
    """The all-branch convenience table must not affect headline cost."""
    upgraded, all_branches = build_normalized_exports(
        _one_upgrade(),
        _parallel_branches(),
        portfolio="generation-storage",
        criterion="n1",
        recovery="NEW",
    )
    all_branches["cost_musd"] = 999.0

    summary = summarize_exports(upgraded, portfolio="generation-storage", criterion="n1", recovery="NEW")

    assert summary["n_upgraded"] == 1
    assert summary["cost_musd"] == pytest.approx(4.0)
    assert summary["cleared"] == 1
    assert summary["residual"] == 0
