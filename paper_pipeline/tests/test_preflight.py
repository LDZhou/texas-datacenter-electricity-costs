from pathlib import Path

from paper_pipeline.scripts.preflight import required_paths, validate_paths


def test_validate_paths_reports_missing_required_input(tmp_path):
    missing = tmp_path / "missing.nc"
    errors = validate_paths([missing])
    assert errors == [f"missing required input: {missing}"]


def test_validate_paths_accepts_existing_path(tmp_path):
    existing = tmp_path / "input.nc"
    existing.touch()
    assert validate_paths([existing]) == []


def test_required_paths_uses_upstream_workflow_snakefile(tmp_path):
    repo = tmp_path / "repo"
    historical = tmp_path / "historical"
    config = tmp_path / "config.yaml"

    paths = required_paths(repo, historical, config)

    assert paths[0] == repo / "workflow" / "Snakefile"
    assert repo / "Snakefile" not in paths
