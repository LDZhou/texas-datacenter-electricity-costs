"""
generate_runtime_config.py
======================
Generate a year-specific, schema-complete config from the tracked PyPSA-USA
configuration layers and the paper overlay. YEAR_PLACEHOLDER_PLUS_ONE is
expanded first so full-year half-open windows end on January 1 of the next
year.

Used by SLURM scripts before invoking snakemake. Snakemake 7 treats the CLI
config file as the effective config during parsing, so the generator performs
the same recursive layered merge documented in workflow/Snakefile before it
writes the runtime file.

Usage:
    python -m paper_pipeline.scripts.generate_runtime_config \
        --template paper_pipeline/config/seasonal.yaml \
        --year 2019 \
        --output configs/.runtime/config_base_2019.yaml

Or get the path printed (auto-output to configs/.runtime/):
    OUT=$(python -m paper_pipeline.scripts.generate_runtime_config \
        --template paper_pipeline/config/seasonal.yaml --year 2019)
    snakemake --configfile "$OUT" ...
"""
import argparse
import sys
from pathlib import Path

import yaml

PLACEHOLDER = "YEAR_PLACEHOLDER"
NEXT_YEAR_PLACEHOLDER = "YEAR_PLACEHOLDER_PLUS_ONE"
CONFIG_LAYERS = (
    "config.slurm.yaml",
    "config.common.yaml",
    "config.plotting.yaml",
    "config.api.yaml",
    "config.sector.yaml",
    "config.default.yaml",
)
PUBLIC_CONFIG_LAYERS = {
    "config.slurm.yaml": "config.cluster.yaml",
    "config.api.yaml": "config.api.example.yaml",
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings; lists and scalars are replaced."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def layered_config(template_path: Path, overlay: dict) -> dict:
    project_root = template_path.resolve().parents[2]
    config_root = project_root / "workflow" / "repo_data" / "config"
    if not config_root.exists():
        config_root = project_root / "config"
    merged = {}
    for name in CONFIG_LAYERS:
        path = config_root / name
        if not path.exists():
            public_root = project_root / "config"
            path = public_root / PUBLIC_CONFIG_LAYERS.get(name, name)
        if not path.exists():
            raise FileNotFoundError(f"Tracked config layer not found: {path}")
        deep_merge(merged, yaml.safe_load(path.read_text()) or {})
    local_api = project_root / "config" / "config.api.yaml"
    if local_api.exists():
        deep_merge(merged, yaml.safe_load(local_api.read_text()) or {})
    return deep_merge(merged, overlay)


def make_config(template_path: Path, year: int, output_path: Path) -> Path:
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    text = template_path.read_text()

    if PLACEHOLDER not in text:
        raise ValueError(
            f"Template {template_path} contains no '{PLACEHOLDER}' — "
            f"is this the right file?"
        )

    n_replacements = text.count(PLACEHOLDER)
    new_text = text.replace(NEXT_YEAR_PLACEHOLDER, str(year + 1))
    new_text = new_text.replace(PLACEHOLDER, str(year))
    overlay = yaml.safe_load(new_text)
    config = layered_config(template_path, overlay)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False))

    print(
        f"[make_runtime_config] {template_path.name} -> {output_path} "
        f"({n_replacements} replacements, year={year})",
        file=sys.stderr,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Inject year into a config template")
    parser.add_argument("--template", required=True, type=Path,
                        help="Path to template config (with YEAR_PLACEHOLDER)")
    parser.add_argument("--year", required=True, type=int,
                        help="Year to inject (e.g. 2019)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: configs/.runtime/<template_stem>_<year>.yaml)")
    args = parser.parse_args()

    if args.output is None:
        runtime_dir = args.template.parent / ".runtime"
        args.output = runtime_dir / f"{args.template.stem}_{args.year}.yaml"

    out = make_config(args.template, args.year, args.output)
    # Print path to stdout so callers can capture it with $(...)
    print(out)


if __name__ == "__main__":
    main()
