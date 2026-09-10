"""Conservative public-tree checks that never print suspected credential values."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".git",
    ".venv",
    "private",
    "results",
    "logs",
    "data",
    "cutouts",
    "resources",
    "__pycache__",
    ".snakemake",
    ".superpowers",
}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".m",
    ".md",
    ".py",
    ".sbatch",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_NAMES = {".env", "config.api.yaml"}
PRIVATE_SUFFIXES = {".key", ".lic", ".pem"}
CREDENTIAL_PATTERN = re.compile(
    r"(?i)(api[_-]?key|wlssecret|wlsaccessid|access[_-]?token|password|eia)"
    r"\s*[=,:]\s*[\"']?[A-Za-z0-9_-]{24,}"
)
TOKEN_PATTERN = re.compile(
    r"gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|^-----BEGIN .*PRIVATE KEY"
)
PERSONAL_PATH_PATTERN = re.compile(
    r"/nfs/(stak|hpc)/|/Us" r"ers/[^/]+/|~/hpc" r"-share"
)


def scan_tree(root: Path) -> tuple[list[tuple[str, str]], int]:
    """Return location-only findings and the number of inspected files."""
    root = Path(root).resolve()
    findings: list[tuple[str, str]] = []
    count = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            try:
                path.resolve().relative_to(root)
            except ValueError:
                findings.append((str(relative), "external symlink"))
            continue
        if not path.is_file():
            continue
        count += 1
        if path.suffix in PRIVATE_SUFFIXES or path.name in PRIVATE_NAMES:
            findings.append((str(relative), "private file"))
        if path.stat().st_size > 95 * 1024 * 1024:
            findings.append((str(relative), "oversized Git file"))
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="replace")
        if path.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as exc:
                findings.append((str(relative), f"Python syntax line {exc.lineno}"))
        for line_number, line in enumerate(text.splitlines(), 1):
            if CREDENTIAL_PATTERN.search(line) or TOKEN_PATTERN.search(line):
                findings.append((str(relative), f"possible credential line {line_number}"))
            if PERSONAL_PATH_PATTERN.search(line):
                findings.append((str(relative), f"personal absolute path line {line_number}"))
    return findings, count


def main() -> int:
    """Scan the repository or a test-only root selected through the environment."""
    root = Path(os.environ.get("PUBLIC_TREE_ROOT", DEFAULT_ROOT))
    findings, count = scan_tree(root)
    for path, reason in findings:
        print(path, reason)
    print(f"Checked {count} files; {len(findings)} findings. This is a heuristic scan, not a guarantee.")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
