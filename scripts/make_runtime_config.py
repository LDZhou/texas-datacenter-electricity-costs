"""
make_runtime_config.py
======================
Generate a year-specific config file from a template by replacing
YEAR_PLACEHOLDER with the actual year.

Used by SLURM scripts before invoking snakemake. Pure string replacement
(no YAML parse/serialize) so the output is byte-identical to the template
except for the year substitution — this prevents snakemake from spuriously
re-running rules due to config formatting changes.

Usage:
    python scripts/make_runtime_config.py \
        --template configs/config_base.yaml \
        --year 2019 \
        --output configs/.runtime/config_base_2019.yaml

Or get the path printed (auto-output to configs/.runtime/):
    OUT=$(python scripts/make_runtime_config.py \
        --template configs/config_base.yaml --year 2019)
    snakemake --configfile "$OUT" ...
"""
import argparse
import sys
from pathlib import Path

PLACEHOLDER = "YEAR_PLACEHOLDER"


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
    new_text = text.replace(PLACEHOLDER, str(year))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(new_text)

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