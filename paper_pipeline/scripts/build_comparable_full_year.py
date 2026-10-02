"""Freeze a freshly built full-year vopt network and document its comparability.

The historical full-year vopt artifact has 2,912 snapshots and must never be
used as a starting network.  This helper accepts only a fresh 2,920-snapshot
vopt output, locks every optimized capacity through ``prepare_network``, and
compares the resulting static assets with the corrected seasonal 2023 base.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


# Every weather year reaches the optimizer with the same 2,920 three-hour
# snapshots: the workflow drops February 29, so a leap year is not longer.
# Verified on the solved 2019, 2020 and 2023 networks, each of which holds
# 224 February snapshots and ends at <year>-12-31 21:00.
FULL_YEAR_HOURS = 365 * 24


def expected_snapshots(year: int, resolution_hours: int = 3) -> tuple[int, float]:
    """Snapshot count and weighted hours for one whole year at this resolution."""
    return FULL_YEAR_HOURS // resolution_hours, float(FULL_YEAR_HOURS)


COMPONENTS = (
    ("lines", "s_nom"),
    ("generators", "p_nom"),
    ("storage_units", "p_nom"),
    ("stores", "e_nom"),
    ("links", "p_nom"),
)


def persist_full_year_profiles(network, run_root, year, dc_data, writer=None,
                               validator=None) -> Path:
    """Persist and exact-snapshot validate the all2030 full-year profile."""
    if writer is None or validator is None:
        from paper_pipeline.scripts.model_contract import load_dc_profile_table
        from paper_pipeline.scripts.prepare_revision_inputs import persist_profiles
        writer = writer or persist_profiles
        validator = validator or load_dc_profile_table
    writer(network, run_root, year, dc_data, horizon="full_year", siting=True)
    profile = Path(run_root) / "profiles" / "full_year" / str(year) / "all2030.csv"
    table = validator(profile, network.snapshots)
    count, _ = expected_snapshots(year)
    if len(table) != count:
        raise ValueError(
            f"full-year DC profile must contain {count} snapshots; got {len(table)}")
    return profile


def _component_report(baseline: pd.DataFrame, full_year: pd.DataFrame,
                      nominal_column: str) -> dict:
    baseline_ids = set(baseline.index.astype(str))
    full_year_ids = set(full_year.index.astype(str))
    common = sorted(baseline_ids & full_year_ids)
    changed = 0
    if common and nominal_column in baseline and nominal_column in full_year:
        left = pd.to_numeric(baseline.loc[common, nominal_column], errors="coerce")
        right = pd.to_numeric(full_year.loc[common, nominal_column], errors="coerce")
        changed = int((left.sub(right).abs() > 1e-6).sum())
    static_columns = [
        column for column in ("bus", "bus0", "bus1", "carrier")
        if column in baseline and column in full_year
    ]
    changed_static_attributes = 0
    for column in static_columns:
        left = baseline.loc[common, column].fillna("<missing>").astype(str)
        right = full_year.loc[common, column].fillna("<missing>").astype(str)
        changed_static_attributes += int((left != right).sum())
    return {
        "baseline_count": len(baseline_ids),
        "full_year_count": len(full_year_ids),
        "added_count": len(full_year_ids - baseline_ids),
        "removed_count": len(baseline_ids - full_year_ids),
        "changed_capacity_count": changed,
        "changed_static_attribute_count": changed_static_attributes,
        "baseline_total_nominal": float(
            pd.to_numeric(baseline.get(nominal_column, pd.Series(dtype=float)),
                          errors="coerce").fillna(0.0).sum()),
        "full_year_total_nominal": float(
            pd.to_numeric(full_year.get(nominal_column, pd.Series(dtype=float)),
                          errors="coerce").fillna(0.0).sum()),
    }


def static_comparability_report(seasonal_base, full_year_base) -> dict:
    """Return a JSON-safe static-asset and horizon comparison."""
    components = {
        name: _component_report(
            getattr(seasonal_base, name), getattr(full_year_base, name), nominal
        )
        for name, nominal in COMPONENTS
    }
    static_grid_comparable = all(
        values["added_count"] == 0
        and values["removed_count"] == 0
        and values["changed_capacity_count"] == 0
        and values["changed_static_attribute_count"] == 0
        for values in components.values()
    )
    return {
        "seasonal_snapshots": len(seasonal_base.snapshots),
        "full_year_snapshots": len(full_year_base.snapshots),
        "seasonal_weighted_hours": float(
            seasonal_base.snapshot_weightings.objective.sum()),
        "full_year_weighted_hours": float(
            full_year_base.snapshot_weightings.objective.sum()),
        "static_grid_comparable": static_grid_comparable,
        "components": components,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vopt-network", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seasonal-baseline", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dc-data", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--require-static-match", action="store_true",
        help="fail after writing the report when a pure-horizon comparison is impossible",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if not args.vopt_network.is_file() or not args.seasonal_baseline.is_file():
        raise SystemExit("vopt network and seasonal baseline must be readable files")

    import pypsa
    from paper_pipeline.scripts.prepare_network import (
        lock_to_optimal,
        prepare_starting_network,
    )

    vopt = pypsa.Network(args.vopt_network)
    snapshots = len(vopt.snapshots)
    hours = float(vopt.snapshot_weightings.objective.sum())
    expected_count, expected_hours = expected_snapshots(args.year)
    if snapshots != expected_count or abs(hours - expected_hours) > 1e-6:
        raise SystemExit(
            f"fresh full-year vopt must be {expected_count} snapshots/"
            f"{expected_hours:.0f} hours for {args.year}; got {snapshots}/{hours}"
        )

    # Compare the post-freeze state before creating the staged artifact.  A
    # strict validation failure must not leave a network that a later job could
    # mistake for an approved pure-horizon base.
    lock_to_optimal(vopt)
    report = static_comparability_report(
        pypsa.Network(args.seasonal_baseline), vopt
    )
    report.update({
        "vopt_network": str(args.vopt_network),
        "seasonal_baseline": str(args.seasonal_baseline),
        "full_year_starting_network": str(args.output),
        "interpretation": "regenerated full-year robustness",
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["static_grid_comparable"] and args.require_static_match:
        raise SystemExit(
            "full-year static assets differ from seasonal base; report written and "
            "strict pure-horizon build intentionally stopped"
        )
    prepare_starting_network(args.vopt_network, args.output)
    frozen_network = pypsa.Network(args.output)
    profile = persist_full_year_profiles(
        frozen_network, args.run_root, args.year, args.dc_data)
    report["full_year_dc_profile"] = str(profile)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
