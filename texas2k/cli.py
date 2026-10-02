"""Command-line interface for the public Texas2k workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from texas2k.data import normalize_pypsa_assets
from texas2k.exports import build_normalized_exports, summarize_exports
from texas2k.workflow import RunConfig, run_workflow, validate_input_bundle

MODULE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_ROOT.parent
DEFAULT_INPUTS = MODULE_ROOT / "inputs"
DEFAULT_OUTPUTS = REPOSITORY_ROOT / "results" / "texas2k"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Texas2k DC engineering screening")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-inputs", help="validate case and public input totals")
    validate.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    validate.add_argument("--json", action="store_true", help="emit JSON only")

    normalize = commands.add_parser("normalize-assets", help="filter a coordinate-enriched PyPSA export")
    normalize.add_argument("source", type=Path)
    normalize.add_argument("destination", type=Path)
    normalize.add_argument("--year", type=int, default=2023)
    normalize.add_argument("--case", default="all2030_full_tx")
    normalize.add_argument("--location", default="ALL_2030")
    normalize.add_argument("--include-storage", action="store_true")

    run = commands.add_parser("run", help="run one portfolio and reliability criterion")
    run.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS)
    run.add_argument("--assets", type=Path, help="override the portfolio's normalized PyPSA asset CSV")
    run.add_argument("--portfolio", choices=("generation", "generation-storage"), required=True)
    run.add_argument("--criterion", choices=("n0", "n1"), required=True)
    run.add_argument("--progress-every", type=int, default=1000)

    export = commands.add_parser("export", help="normalize an existing workflow run")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--portfolio", choices=("generation", "generation-storage"), required=True)
    export.add_argument("--criterion", choices=("n0", "n1"), required=True)
    return parser


def _export_run(run_dir: Path, portfolio: str, criterion: str) -> list[Path]:
    details = pd.read_csv(run_dir / f"upgrade_detail_{criterion}.csv")
    branches = pd.read_csv(run_dir / f"branch_all_{criterion}.csv")
    export_dir = run_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary_rows = []
    for recovery in ("NEW", "ALL"):
        selected = details[details["recovery"] == recovery].copy()
        upgraded, all_branches = build_normalized_exports(
            selected,
            branches,
            portfolio=portfolio,
            criterion=criterion,
            recovery=recovery,
        )
        upgraded_path = export_dir / f"{criterion}_recover{recovery}_upgraded.csv"
        all_path = export_dir / f"{criterion}_recover{recovery}_allbranches.csv"
        upgraded.to_csv(upgraded_path, index=False, lineterminator="\n")
        all_branches.to_csv(all_path, index=False, lineterminator="\n")
        written.extend([upgraded_path, all_path])
        summary_rows.append(summarize_exports(upgraded, portfolio=portfolio, criterion=criterion, recovery=recovery))
    summary_path = export_dir / "engineering_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, lineterminator="\n")
    written.append(summary_path)
    return written


def main(argv: list[str] | None = None) -> int:
    """Execute one CLI subcommand and return its process exit code."""
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "validate-inputs":
            report = validate_input_bundle(arguments.input_dir)
            if arguments.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print("Texas2k inputs validated")
                print(json.dumps(report, indent=2, sort_keys=True))
        elif arguments.command == "normalize-assets":
            frame = normalize_pypsa_assets(
                arguments.source,
                arguments.destination,
                year=arguments.year,
                case=arguments.case,
                location=arguments.location,
                include_storage=arguments.include_storage,
            )
            print(f"Wrote {len(frame)} assets ({frame['added_mw'].sum():,.1f} MW) to {arguments.destination}")
        elif arguments.command == "run":
            outputs = run_workflow(
                RunConfig(
                    input_dir=arguments.input_dir,
                    output_dir=arguments.output_dir,
                    portfolio=arguments.portfolio,
                    criterion=arguments.criterion,
                    progress_every=arguments.progress_every,
                    assets_path=arguments.assets,
                )
            )
            for label, path in outputs.items():
                print(f"{label}: {path}")
        elif arguments.command == "export":
            for path in _export_run(arguments.run_dir, arguments.portfolio, arguments.criterion):
                print(path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
