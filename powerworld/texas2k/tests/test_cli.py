"""Tests for the unified Texas2k workflow and command-line interface."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from powerworld.texas2k.workflow import RunConfig, run_workflow, validate_input_bundle, warn_on_residuals

MODULE_ROOT = Path(__file__).parents[1]
SMOKE_INPUTS = Path(__file__).parent / "fixtures" / "smoke_inputs"
PUBLIC_INPUTS = MODULE_ROOT / "inputs"
REPO_ROOT = Path(__file__).parents[3]


def test_run_workflow_writes_only_to_requested_scenario_directory(tmp_path):
    """A run must remain inside its explicit output directory."""
    outputs = run_workflow(
        RunConfig(
            input_dir=SMOKE_INPUTS,
            output_dir=tmp_path,
            portfolio="generation",
            criterion="n0",
        )
    )

    scenario_dir = tmp_path / "generation" / "n0"
    assert outputs["summary"] == scenario_dir / "summary_n0.csv"
    assert outputs["summary"].exists()
    assert outputs["upgrades"].exists()
    assert outputs["branches"].exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["generation"]
    assert set(pd.read_csv(outputs["summary"])["recovery"]) == {"NEW", "ALL"}


def test_warn_on_residuals_records_warning_without_raising():
    """Residual violations remain visible while the workflow can finish."""
    summary = pd.DataFrame([{"recovery": "NEW", "n_residual_grid": 3}])

    with pytest.warns(RuntimeWarning, match="3 residual violations"):
        warn_on_residuals(summary)


def test_validate_public_input_bundle_reports_documented_totals():
    """The CLI validation contract must reflect the committed public inputs."""
    report = validate_input_bundle(PUBLIC_INPUTS)

    assert report["case"] == {"buses": 2751, "generators": 1099, "branches": 5344}
    assert report["datacenters"]["source_rows"] == 417
    assert report["datacenters"]["injected_rows"] == 384
    assert report["datacenters"]["injected_mw"] == pytest.approx(39864.607)
    assert report["portfolios"]["generation"]["added_mw"] == pytest.approx(32942.9)
    assert report["portfolios"]["generation-storage"]["added_mw"] == pytest.approx(39992.7)


def test_cli_validate_inputs_emits_machine_readable_json():
    """Automation should consume validation without parsing prose."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "powerworld.texas2k.cli",
            "validate-inputs",
            "--input-dir",
            str(SMOKE_INPUTS),
            "--json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["case"] == {"buses": 3, "generators": 1, "branches": 3}
    assert report["datacenters"]["injected_mw"] == pytest.approx(10.0)


def test_cli_rejects_output_equal_to_input_directory():
    """Generated results must never overwrite immutable input data."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "powerworld.texas2k.cli",
            "run",
            "--input-dir",
            str(SMOKE_INPUTS),
            "--output-dir",
            str(SMOKE_INPUTS),
            "--portfolio",
            "generation",
            "--criterion",
            "n0",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "output directory must differ" in completed.stderr


def test_cli_run_accepts_a_normalized_pypsa_asset_override(tmp_path):
    """A newly normalized PyPSA export should run without rebuilding the input bundle."""
    output_dir = tmp_path / "output"
    assets = SMOKE_INPUTS / "pypsa" / "generation_only_2023.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "powerworld.texas2k.cli",
            "run",
            "--input-dir",
            str(SMOKE_INPUTS),
            "--assets",
            str(assets),
            "--output-dir",
            str(output_dir),
            "--portfolio",
            "generation",
            "--criterion",
            "n0",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "generation" / "n0" / "summary_n0.csv").exists()


def test_cli_export_normalizes_a_completed_run(tmp_path):
    """A completed workflow must feed the stable-ID collaborator export directly."""
    outputs = run_workflow(
        RunConfig(
            input_dir=SMOKE_INPUTS,
            output_dir=tmp_path,
            portfolio="generation",
            criterion="n0",
        )
    )
    run_dir = outputs["summary"].parent

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "powerworld.texas2k.cli",
            "export",
            "--run-dir",
            str(run_dir),
            "--portfolio",
            "generation",
            "--criterion",
            "n0",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (run_dir / "exports" / "n0_recoverNEW_upgraded.csv").exists()
    assert (run_dir / "exports" / "n0_recoverNEW_allbranches.csv").exists()
    assert (run_dir / "exports" / "engineering_summary.csv").exists()


def test_reference_summary_preserves_supplied_paper_result():
    """The committed reference index must retain the collaborator's paper row."""
    summary = pd.read_csv(MODULE_ROOT / "reference_outputs" / "engineering_summary.csv")
    row = summary[
        (summary["portfolio"] == "generation-storage")
        & (summary["criterion"] == "N1")
        & (summary["recovery"] == "NEW")
    ].squeeze()

    assert row["n_upgraded"] == 553
    assert row["cost_musd"] == pytest.approx(34289.5)
    assert row["residual_line"] == 55
    assert row["residual_grid"] == 173
