"""Portable commands for the current paper experiment inventory."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "results" / "paper"
DEFAULT_DC_DATA = ROOT / "texas2k" / "inputs" / "datacenters" / "projects.csv"
DEFAULT_YEARS = (2019, 2020, 2021, 2022, 2023)


def module(name: str, *args: object) -> list[str]:
    return [sys.executable, "-m", f"paper_pipeline.scripts.{name}", *map(str, args)]


def build_commands(stage: str, *, run_root: Path, years: list[int],
                   historical_root: Path = ROOT, dc_data: Path = DEFAULT_DC_DATA,
                   group: str = "central", index: int = 0, n_sims: int = 5000) -> list[list[str]]:
    """Build reproducible commands without opening a network or scheduling work."""
    run_root = Path(run_root)
    analysis_root = run_root / "analysis_view"
    common = ["--run-root", run_root, "--historical-root", historical_root, "--dc-data", dc_data]
    if stage == "matrix":
        return [module("revision_matrix", "--output", run_root / "manifests" / "experiment_matrix.json")]
    if stage == "prepare":
        commands = [module("prepare_revision_inputs", "--year", year, *common) for year in years]
        if 2021 in years:
            commands.append(module("fix_gas_fuel_outliers", "--input", run_root / "starting_networks" / "start_2021.nc",
                                   "--output", run_root / "starting_networks" / "start_2021_gasfix.nc",
                                   "--record", run_root / "manifests" / "gas_fuel_outlier_fix_2021.json"))
        return commands
    if stage == "run-case":
        return [module("run_revision_case", *common, "--group", group, "--index", index)]
    if stage == "fuel-correction":
        return [module("fix_gas_fuel_outliers", "--input", run_root / "starting_networks" / "start_2021.nc",
                       "--output", run_root / "starting_networks" / "start_2021_gasfix.nc",
                       "--record", run_root / "manifests" / "gas_fuel_outlier_fix_2021.json")]
    if stage == "analysis":
        years_arg = ["--years", *years]
        result_root = run_root / "results"
        legacy_tables = analysis_root / "results" / "nc_paper_figures"
        return [
            module("summarize_revision", "--run-root", run_root, "--out-dir", analysis_root,
                   "--groups", "seasonal", "siting"),
            module("compute_capped_lmp_adders", *years_arg,
                   "--base-template", result_root / "seasonal" / "{y}" / "none_full_tx" / "network.nc",
                   "--full-template", result_root / "seasonal" / "{y}" / "all2030_full_tx" / "network.nc",
                   "--out", legacy_tables / "capped_lmp_adders.csv"),
            module("compute_adder_factor_decomposition", *years_arg,
                   "--results-root", result_root,
                   "--out", legacy_tables / "adder_factor_decomposition.csv"),
            module("compute_graded_unserved", *years_arg, "--multiloc",
                   "--results-root", result_root, "--outdir", legacy_tables),
        ]
    if stage == "figures":
        results_root = run_root / "results" / "figures_final"
        return [
            module("make_figure_1", "--run-root", run_root, "--legacy-root", analysis_root,
                   "--results-root", results_root),
            module("make_figure_2_siting", "--run-root", run_root, "--legacy-root", analysis_root,
                   "--analysis-dir", analysis_root, "--results-root", results_root),
            module("make_figure_3_panels", "--run-root", run_root, "--results-root", results_root),
            module("make_revision_figures", "--run-root", run_root, "--legacy-root", analysis_root,
                   "--rep-root", run_root / "results" / "retailer", "--results-root", results_root),
        ]
    if stage == "rep":
        output = run_root / "results" / "retailer"
        return [
            module("prepare_retail_inputs", "--run-root", run_root, "--years", *years),
            module("analyze_results", "--base-dir", run_root / "retailer_inputs",
                   "--out-dir", output, "--cache-path", output / "rep_contrast_cache.pkl",
                   "--years", *years, "--scenarios", "none", "all2030",
                   "--modes", "dispatch", "full_tx", "generation_storage",
                   "--zones", "system", "WEST", "NORTH", "SOUTH", "HOUSTON",
                   "--forward-case", "both", "--rep-load", "1000", "--rep-capital", "50000000",
                   "--hedge-ratio", "0.70", "--forward-premium", "5", "--rep-opex", "3",
                   "--retail-pass-through", "0", "--passthrough-sigma", "0", "--lmp-clip-upper", "5000",
                   "--full-tx-tcos-adder-mwh", "2.601", "--n-sims", n_sims, "--seed", "123",
                   "--refresh-cache", "--summary-only", "--no-plots"),
        ]
    raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("matrix", "prepare", "run-case", "fuel-correction", "analysis", "rep", "figures"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--historical-root", type=Path, default=ROOT,
                        help="PyPSA-USA worktree holding frozen historical vopt results")
    parser.add_argument("--dc-data", type=Path, default=DEFAULT_DC_DATA,
                        help="Public CSV containing mapped data-center projects")
    parser.add_argument("--years", nargs="*", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument("--group", default="central", help="Experiment-matrix group for run-case")
    parser.add_argument("--index", type=int, default=0, help="Zero-based case index within --group")
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()
    commands = build_commands(args.stage, run_root=args.run_root, years=args.years,
                              historical_root=args.historical_root, dc_data=args.dc_data,
                              group=args.group, index=args.index, n_sims=args.n_sims)
    env = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg")
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
