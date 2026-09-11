"""Behavior tests for the public-tree credential and path scanner."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
SCANNER = REPO_ROOT / "scripts" / "check_public_tree.py"


def test_scanner_checks_csv_without_printing_secret_value(tmp_path):
    """CSV credentials must be detected while their values stay out of logs."""
    secret = "A" * 32
    (tmp_path / "sample.csv").write_text(f"api_key,{secret}\n")
    environment = os.environ.copy()
    environment["PUBLIC_TREE_ROOT"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(SCANNER)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "sample.csv" in completed.stdout
    assert secret not in completed.stdout


def test_scanner_refuses_model_network_files(tmp_path):
    """A .nc carries the run config in its meta attribute and must never ship."""
    (tmp_path / "network.nc").write_bytes(b"CDF\x01 binary placeholder")
    environment = os.environ.copy()
    environment["PUBLIC_TREE_ROOT"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(SCANNER)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "network.nc" in completed.stdout
    assert "embeds run config" in completed.stdout
