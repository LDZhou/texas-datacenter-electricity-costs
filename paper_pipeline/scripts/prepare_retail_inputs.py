"""Select the three experiment types used in the retailer analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

CASE_NAMES = ("none_dispatch", "all2030_full_tx", "all2030_generation_storage")


def prepare_view(run_root: Path, years: list[int]) -> Path:
    """Link result directories without opening any network files."""
    run_root = run_root.resolve()
    view = run_root / "retailer_inputs"
    sources = [(year, name, run_root / "results" / "seasonal" / str(year) / name)
               for year in years for name in CASE_NAMES]
    missing = [str(source) for _, _, source in sources if not source.is_dir()]
    if missing:
        raise FileNotFoundError("Missing retailer input directories: " + ", ".join(missing))
    for year in years:
        directory = view / str(year)
        if directory.exists():
            unexpected = set(p.name for p in directory.iterdir()) - set(CASE_NAMES)
            if unexpected:
                raise ValueError(f"Unexpected cases in {directory}: {sorted(unexpected)}")
    for year, name, source in sources:
        destination = view / str(year) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError(f"Conflicting retailer input link: {destination}")
        elif destination.exists():
            raise ValueError(f"Retailer input path already exists: {destination}")
        else:
            destination.symlink_to(source, target_is_directory=True)
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023])
    args = parser.parse_args()
    print(prepare_view(args.run_root, args.years))


if __name__ == "__main__":
    main()
