"""End-to-end orchestration for Texas2k engineering screening runs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from powerworld.texas2k.case import load_matpower_case, run_dc_power_flow
from powerworld.texas2k.contingency import ScreeningResult, scan_n1, screen_n0
from powerworld.texas2k.data import load_assets, load_datacenter_injections
from powerworld.texas2k.dispatch import assemble_scenario
from powerworld.texas2k.upgrades import apply_upgrades, branch_table, classify_loading

PORTFOLIO_FILES = {
    "generation": "generation_only_2023.csv",
    "generation-storage": "generation_storage_2023.csv",
}


@dataclass(frozen=True)
class RunConfig:
    """Paths and scenario selectors for one engineering workflow run."""

    input_dir: Path
    output_dir: Path
    portfolio: str
    criterion: str
    progress_every: int = 1000


def _input_paths(input_dir: Path, portfolio: str) -> dict[str, Path]:
    if portfolio not in PORTFOLIO_FILES:
        raise ValueError("portfolio must be 'generation' or 'generation-storage'")
    return {
        "case": input_dir / "case" / "texas2k_series25_summer_peak.m",
        "buses": input_dir / "baseline" / "buses.csv",
        "projects": input_dir / "datacenters" / "projects.csv",
        "matches": input_dir / "datacenters" / "bus_matches.csv",
        "assets": input_dir / "pypsa" / PORTFOLIO_FILES[portfolio],
    }


def validate_input_bundle(input_dir: Path) -> dict[str, object]:
    """Validate all public inputs and return their reproducibility invariants."""
    input_dir = Path(input_dir)
    paths = _input_paths(input_dir, "generation")
    storage_paths = _input_paths(input_dir, "generation-storage")
    required_paths = {*paths.values(), storage_paths["assets"]}
    missing = sorted(str(path) for path in required_paths if not path.is_file())
    if missing:
        raise FileNotFoundError("missing Texas2k inputs: " + ", ".join(missing))
    case = load_matpower_case(paths["case"])
    buses = pd.read_csv(paths["buses"])
    if buses[["longitude", "latitude"]].isna().any().any():
        raise ValueError("baseline bus table contains missing coordinates")
    _, datacenter_stats = load_datacenter_injections(paths["projects"], paths["matches"])
    portfolios: dict[str, dict[str, float | int]] = {}
    for portfolio, filename in PORTFOLIO_FILES.items():
        assets = load_assets(input_dir / "pypsa" / filename)
        portfolios[portfolio] = {
            "rows": int(len(assets)),
            "added_mw": float(assets["added_mw"].sum()),
        }
    return {
        "case": {
            "buses": int(case["bus"].shape[0]),
            "generators": int(case["gen"].shape[0]),
            "branches": int(case["branch"].shape[0]),
        },
        "baseline_buses": int(len(buses)),
        "datacenters": datacenter_stats,
        "portfolios": portfolios,
    }


def _screen(case: dict[str, object], criterion: str, progress_every: int) -> ScreeningResult:
    if criterion == "n0":
        return screen_n0(case)
    if criterion == "n1":
        return scan_n1(case, progress_every=progress_every)
    raise ValueError("criterion must be 'n0' or 'n1'")


def _criterion_loading(screening: ScreeningResult, criterion: str) -> np.ndarray:
    return screening.n0_loading if criterion == "n0" else screening.worst_loading


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def warn_on_residuals(summary: pd.DataFrame) -> None:
    """Warn for each recovery result that retains grid violations."""
    for row in summary.itertuples(index=False):
        residual = int(row.n_residual_grid)
        if residual:
            warnings.warn(
                f"{row.recovery}: {residual} residual violations remain after fixed-rule upgrades",
                RuntimeWarning,
                stacklevel=2,
            )


def _skipped_frame(screening: ScreeningResult, phase: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "phase": phase,
                "branch_id": row.branch_id,
                "reason": row.reason,
                "detail": row.detail,
            }
            for row in screening.skipped
        ],
        columns=["phase", "branch_id", "reason", "detail"],
    )


def run_workflow(config: RunConfig) -> dict[str, Path]:
    """Run baseline, injected, and fixed-upgrade screening for one scenario."""
    input_dir = Path(config.input_dir).resolve()
    output_root = Path(config.output_dir).resolve()
    if input_dir == output_root or input_dir in output_root.parents or output_root in input_dir.parents:
        raise ValueError("output directory must differ from and remain outside the input directory")
    paths = _input_paths(input_dir, config.portfolio)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Texas2k inputs: " + ", ".join(missing))

    base_case = load_matpower_case(paths["case"])
    buses = pd.read_csv(paths["buses"])
    assets = load_assets(paths["assets"])
    datacenters, _ = load_datacenter_injections(paths["projects"], paths["matches"])
    baseline_case, baseline_info = assemble_scenario(
        base_case,
        assets=assets,
        bus_geography=buses,
        dc_loads=None,
        portfolio=config.portfolio,
    )
    injected_case, injected_info = assemble_scenario(
        base_case,
        assets=assets,
        bus_geography=buses,
        dc_loads=datacenters,
        portfolio=config.portfolio,
    )
    baseline_screening = _screen(baseline_case, config.criterion, config.progress_every)
    injected_screening = _screen(injected_case, config.criterion, config.progress_every)
    injected_result, converged = run_dc_power_flow(injected_case)
    if not converged:
        raise RuntimeError("injected intact DC power flow did not converge")
    branches = branch_table(injected_result, buses)
    baseline_loading = _criterion_loading(baseline_screening, config.criterion)
    injected_loading = _criterion_loading(injected_screening, config.criterion)
    branches["loading_n0_pct"] = injected_screening.n0_loading
    branches["loading_before_pct"] = injected_loading
    branches["base_loading_pct"] = baseline_loading
    branches["worst_contingency"] = injected_screening.worst_contingency
    branches["screening_tier"] = [classify_loading(value) for value in injected_loading]
    branches["is_over"] = branches["loading_before_pct"] >= 100
    branches["is_new"] = branches["is_over"] & (branches["base_loading_pct"] < 100)

    all_details: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    skipped_frames = [
        _skipped_frame(baseline_screening, "baseline"),
        _skipped_frame(injected_screening, "injected"),
    ]
    for recovery in ("NEW", "ALL"):
        targets = branches[branches["is_new"]] if recovery == "NEW" else branches[branches["is_over"]]
        upgraded_case, detail = apply_upgrades(injected_case, targets, tier_column="screening_tier")
        upgraded_screening = _screen(upgraded_case, config.criterion, config.progress_every)
        after_loading = _criterion_loading(upgraded_screening, config.criterion)
        skipped_frames.append(_skipped_frame(upgraded_screening, f"after_{recovery.lower()}"))
        if detail.empty:
            detail = pd.DataFrame(columns=["branch_id", "capex_musd"])
        detail["loading_before_pct"] = detail["branch_id"].map(
            branches.set_index("branch_id")["loading_before_pct"]
        )
        detail["loading_after_pct"] = detail["branch_id"].map(
            pd.Series(after_loading, index=np.arange(len(after_loading)))
        )
        detail["cleared"] = detail["loading_after_pct"] < 100
        detail["recovery"] = recovery
        detail["portfolio"] = config.portfolio
        detail["criterion"] = config.criterion.upper()
        all_details.append(detail)
        summaries.append(
            {
                "portfolio": config.portfolio,
                "criterion": config.criterion.upper(),
                "recovery": recovery,
                "n_target": int(len(targets)),
                "n_upgraded": int(len(detail)),
                "cost_musd": float(pd.to_numeric(detail["capex_musd"], errors="coerce").fillna(0).sum()),
                "cost_busd": float(pd.to_numeric(detail["capex_musd"], errors="coerce").fillna(0).sum() / 1000),
                "n_cleared": int(detail["cleared"].sum()),
                "n_residual_line": int((~detail["cleared"]).sum()),
                "n_residual_grid": int((after_loading >= 100).sum()),
                "max_loading_after_pct": float(np.nanmax(after_loading)),
                "violations_baseline": int((baseline_loading >= 100).sum()),
                "violations_injected": int((injected_loading >= 100).sum()),
                "new_violations": int(branches["is_new"].sum()),
                "dc_injected_mw": float(injected_info["dc_applied_mw"]),
                "asset_mw": float(injected_info["asset_mw"]),
                "skipped_injected": int(len(injected_screening.skipped)),
                "skipped_after": int(len(upgraded_screening.skipped)),
                "baseline_asset_mw": float(baseline_info["asset_mw"]),
            }
        )

    summary = pd.DataFrame(summaries)
    details = pd.concat(all_details, ignore_index=True)
    skipped = pd.concat(skipped_frames, ignore_index=True)
    scenario_dir = output_root / config.portfolio / config.criterion
    output_paths = {
        "summary": scenario_dir / f"summary_{config.criterion}.csv",
        "upgrades": scenario_dir / f"upgrade_detail_{config.criterion}.csv",
        "branches": scenario_dir / f"branch_all_{config.criterion}.csv",
    }
    _write_csv(summary, output_paths["summary"])
    _write_csv(details, output_paths["upgrades"])
    _write_csv(branches, output_paths["branches"])
    if config.criterion == "n1":
        output_paths["skipped"] = scenario_dir / "skipped_contingencies.csv"
        _write_csv(skipped, output_paths["skipped"])
    warn_on_residuals(summary)
    return output_paths
