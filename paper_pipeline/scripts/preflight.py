"""Validate reproducibility inputs before submitting a paper rerun."""
import argparse
import subprocess
from pathlib import Path

EXPECTED_LICENSE_ID = "2773119"


def validate_paths(paths):
    return [f"missing required input: {path}" for path in paths if not path.exists()]


def required_paths(repo: Path, historical_root: Path, config: Path, dc_data: Path | None = None):
    dc_data = dc_data or historical_root / "texas2k" / "inputs" / "datacenters" / "projects.csv"
    return [repo / "workflow" / "Snakefile", historical_root / "data",
            historical_root / "cutouts", historical_root / "resources",
            dc_data, config]


def git_value(repo: Path, argument: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *argument.split()], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dc-data", type=Path, required=True)
    parser.add_argument("--network", type=Path)
    parser.add_argument("--expected-snapshots", type=int)
    parser.add_argument("--expected-weighted-hours", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--license-id", default=EXPECTED_LICENSE_ID)
    args = parser.parse_args()
    errors = validate_paths(required_paths(args.repo, args.historical_root, args.config, args.dc_data) +
                            ([args.network] if args.network else []))
    if args.run_root.resolve() == args.historical_root.resolve():
        errors.append("run root must differ from read-only historical root")
    if git_value(args.repo, "rev-parse HEAD") != args.commit:
        errors.append("commit mismatch")
    if git_value(args.repo, "branch --show-current") != args.branch:
        errors.append("branch mismatch")
    if args.license_id != EXPECTED_LICENSE_ID:
        errors.append("unexpected WLS license ID")
    if args.output and args.output.exists() and any(args.output.iterdir()):
        errors.append(f"refusing nonempty output directory: {args.output}")
    if args.network and (args.expected_snapshots or args.expected_weighted_hours):
        import pypsa
        network = pypsa.Network(args.network)
        if args.expected_snapshots and len(network.snapshots) != args.expected_snapshots:
            errors.append("unexpected snapshot count")
        if args.expected_weighted_hours is not None:
            hours = float(network.snapshot_weightings.objective.sum())
            if abs(hours - args.expected_weighted_hours) > 1e-6:
                errors.append("unexpected weighted hours")
    if errors:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(errors))
    print("preflight ok")


if __name__ == "__main__":
    main()
