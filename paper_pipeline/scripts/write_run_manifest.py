"""Write secret-free provenance manifests for pipeline jobs."""
import argparse
import hashlib
import json
import platform
import subprocess
import shlex
from datetime import datetime, timezone
from pathlib import Path

SECRET_MARKERS = ("license", "secret", "token", "password", "key")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()
                if not any(marker in key.lower() for marker in SECRET_MARKERS)
                or key.lower() == "license_id"}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def package_versions():
    versions = {"python": platform.python_version()}
    for package in ("pypsa", "linopy", "pandas", "xarray", "scipy", "gurobipy"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package] = "unavailable"
    return versions


def git_value(repo: Path, argument: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), *shlex.split(argument)], text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_manifest(output: Path, metadata: dict | None = None, *, repo: Path | None = None,
                   config: Path | None = None, network: Path | None = None) -> dict:
    repo = repo or Path.cwd()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_value(repo, "rev-parse HEAD"),
        "branch": git_value(repo, "branch --show-current"),
        "package_versions": package_versions(),
    }
    if config and config.exists():
        payload["config"] = str(config)
        payload["config_sha256"] = sha256(config)
    if network and network.exists():
        payload["network"] = str(network)
        payload["network_sha256"] = sha256(network)
    payload.update(_safe(metadata or {}))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--network", type=Path)
    metadata_group = parser.add_mutually_exclusive_group()
    metadata_group.add_argument("--metadata-json", default=None)
    metadata_group.add_argument("--metadata-file", type=Path)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    metadata = (json.loads(args.metadata_file.read_text()) if args.metadata_file
                else json.loads(args.metadata_json or "{}"))
    if args.metrics and args.metrics.exists():
        metrics = json.loads(args.metrics.read_text())
        metadata.update({key: metrics[key] for key in ("solve_status", "objective") if key in metrics})
    write_manifest(args.output, metadata, repo=args.repo,
                   config=args.config, network=args.network)


if __name__ == "__main__":
    main()
