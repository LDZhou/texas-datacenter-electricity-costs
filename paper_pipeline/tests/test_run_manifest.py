import json
import subprocess
import sys

from paper_pipeline.scripts.write_run_manifest import write_manifest


def test_manifest_excludes_license_secrets(tmp_path):
    output = tmp_path / "manifest.json"
    write_manifest(
        output=output,
        metadata={"GRB_LICENSE_FILE": "/private/license", "license_id": "2773119"},
    )
    payload = json.loads(output.read_text())
    assert payload["license_id"] == "2773119"
    assert "GRB_LICENSE_FILE" not in payload
    assert "/private/license" not in output.read_text()


def test_manifest_accepts_metadata_file_with_spaces(tmp_path):
    output = tmp_path / "manifest.json"
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"job_label": "smoke job with spaces"}')
    subprocess.run(
        [sys.executable, "-m", "paper_pipeline.scripts.write_run_manifest",
         "--output", str(output), "--metadata-file", str(metadata)],
        check=True,
    )
    assert json.loads(output.read_text())["job_label"] == "smoke job with spaces"
