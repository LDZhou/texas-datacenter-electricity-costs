#!/usr/bin/env python
"""Panels c and d of the all-announced figure (Fig. 3 of the NC submission).

  fig5_all2030_annual_investment_by_year   annualized new investment by weather
                                           year for each expansion case
  fig6_all2030_mean_capacity_mix           mean new capacity across the weather
                                           years, stacked by technology

Both read metrics.json from results/seasonal/<year>/all2030_<case> for the
cases generation_storage and full_tx, plus generation_tx when its results are
present. CCGT is an expansion candidate in these results, so the capacity
mix carries the carriers OCGT, CCGT, solar, onwind, 4-hour battery storage
and transmission. Output names match the manuscript's \\includegraphics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import nc_style
from .nc_style import C, CARRIER_COLORS, PW_RESOURCE, PW_COORDINATED

nc_style.apply()

DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[2] / "results" / "paper"

# (case, label, colour) in drawing order; generation_tx is drawn only when its
# metrics.json exists for the same years as the two required cases.
REQUIRED_CASES = ["generation_storage", "full_tx"]
OPTIONAL_CASES = ["generation_tx"]
CASE_ORDER = ["generation_storage", "generation_tx", "full_tx"]
CASE_LABELS = {
    "generation_storage": "Generation + storage",
    "generation_tx": "Generation + transmission",
    "full_tx": "Full expansion",
}
CASE_COLORS = {
    "generation_storage": PW_RESOURCE,
    "generation_tx": C["sky"],
    "full_tx": PW_COORDINATED,
}
# (legend label, metrics.json key) in stacking order, bottom to top
CARRIER_KEYS = [
    ("Gas OCGT", "new_OCGT_mw"),
    ("Gas CCGT", "new_CCGT_mw"),
    ("Solar", "new_solar_mw"),
    ("Wind", "new_onwind_mw"),
    ("Battery", "new_battery_mw"),
    ("Transmission", "new_transmission_mw"),
]


def flatten(raw: dict) -> dict:
    flat: dict = {}

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            else:
                flat[k] = v

    walk(raw)
    return flat


def load_all2030_metrics(run_root: Path) -> dict[tuple[str, int], dict]:
    """{(case, year): flattened metrics.json} for every all2030 case found."""
    seasonal = run_root / "results" / "seasonal"
    out = {}
    for ydir in sorted(p for p in seasonal.glob("[0-9]" * 4) if p.is_dir()):
        for case in REQUIRED_CASES + OPTIONAL_CASES:
            p = ydir / f"all2030_{case}" / "metrics.json"
            if p.exists():
                with open(p) as fh:
                    out[(case, int(ydir.name))] = flatten(json.load(fh))
    if not out:
        raise FileNotFoundError(f"no all2030 metrics.json under {seasonal}")
    return out


def select_cases_and_years(metrics: dict[tuple[str, int], dict]) -> tuple[list[str], list[int]]:
    """Years with every required case; optional cases only if they cover the
    same years."""
    years = sorted({y for (c, y) in metrics if c == REQUIRED_CASES[0]})
    years = [y for y in years if all((c, y) in metrics for c in REQUIRED_CASES)]
    if not years:
        raise ValueError(f"no year has all of {REQUIRED_CASES}")
    cases = [c for c in CASE_ORDER if c in REQUIRED_CASES
             or all((c, y) in metrics for y in years)]
    return cases, years


def build_investment_table(metrics: dict[tuple[str, int], dict],
                           cases: list[str], years: list[int]) -> pd.DataFrame:
    """Annualized investment ($bn/yr); index year, columns case."""
    return pd.DataFrame(
        {c: [metrics[(c, y)]["total_annual_investment"] / 1e9 for y in years] for c in cases},
        index=pd.Index(years, name="year"))


def build_capacity_table(metrics: dict[tuple[str, int], dict],
                         cases: list[str], years: list[int]) -> pd.DataFrame:
    """New capacity (GW) with a MultiIndex (case, year) and one column per
    carrier label in CARRIER_KEYS; a missing key counts as zero."""
    rows = {}
    for c in cases:
        for y in years:
            m = metrics[(c, y)]
            rows[(c, y)] = {label: float(m.get(key, 0.0) or 0.0) / 1000.0
                            for label, key in CARRIER_KEYS}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["case", "year"])
    return df


def year_span(years) -> str:
    """'2019-2023' (en dash) for a contiguous run, otherwise the explicit list,
    so the label never implies a year that is not in the data."""
    ys = sorted(set(int(y) for y in years))
    if len(ys) == 1:
        return f"{ys[0]} weather year"
    if ys == list(range(ys[0], ys[-1] + 1)):
        return f"{ys[0]}\u2013{ys[-1]}"
    return ", ".join(str(y) for y in ys)


def save(fig, name: str, results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(results_root / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {name}")


def make_fig5(invest: pd.DataFrame, results_root: Path) -> None:
    cases = list(invest.columns)
    years = list(invest.index)
    fig, ax = plt.subplots(
        figsize=(5.2, 2.9),
        gridspec_kw={"left": 0.11, "right": 0.985, "top": 0.97, "bottom": 0.16},
    )
    group = 0.78
    width = group / len(cases)
    xpos = np.arange(len(years))
    for k, case in enumerate(cases):
        off = (k - (len(cases) - 1) / 2) * width
        vals = invest[case].to_numpy()
        ax.bar(xpos + off, vals, width * 0.94, color=CASE_COLORS[case],
               edgecolor="white", linewidth=0.4, label=CASE_LABELS[case])
        for x, v in zip(xpos + off, vals):
            ax.text(x, v + invest.to_numpy().max() * 0.015, f"{v:.1f}", ha="center",
                    fontsize=5.8)
    ax.set_xticks(xpos, [str(y) for y in years])
    ax.set_xlabel("Weather year")
    ax.set_ylabel("Annualized new investment ($B per year)")
    ax.set_ylim(0, invest.to_numpy().max() * 1.30)
    ax.legend(frameon=False, loc="upper right", borderaxespad=0.0, fontsize=6)
    save(fig, "fig5_all2030_annual_investment_by_year", results_root)


def make_fig6(capacity: pd.DataFrame, results_root: Path) -> None:
    mean = capacity.groupby(level="case").mean()
    cases = [c for c in CASE_ORDER if c in mean.index]
    mean = mean.loc[cases]
    labels = [CASE_LABELS[c].replace(" + ", " +\n") for c in cases]
    fig, ax = plt.subplots(
        figsize=(4.6, 3.1),
        gridspec_kw={"left": 0.12, "right": 0.98, "top": 0.97, "bottom": 0.13},
    )
    bottoms = np.zeros(len(cases))
    for carrier, _ in CARRIER_KEYS:
        vals = mean[carrier].to_numpy()
        if not np.any(vals > 0):
            continue
        ax.bar(labels, vals, 0.55, bottom=bottoms, color=CARRIER_COLORS[carrier],
               label=carrier)
        bottoms += vals
    for i, total in enumerate(bottoms):
        ax.text(i, total + bottoms.max() * 0.02, f"{total:.1f} GW", ha="center", fontsize=6.5)
    ax.set_ylim(0, bottoms.max() * 1.22)
    ax.set_ylabel(f"Mean new capacity, {year_span(capacity.index.get_level_values('year'))} (GW)")
    ax.legend(frameon=False, ncol=2, loc="upper right", borderaxespad=0.0,
              columnspacing=0.9, handletextpad=0.45, fontsize=6)
    save(fig, "fig6_all2030_mean_capacity_mix", results_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT,
                        help="Experiment run root holding results/seasonal/<year>/all2030_<case>/.")
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Directory for the generated figure files.")
    args = parser.parse_args()
    args.run_root = args.run_root.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    metrics = load_all2030_metrics(args.run_root)
    cases, years = select_cases_and_years(metrics)
    invest = build_investment_table(metrics, cases, years)
    capacity = build_capacity_table(metrics, cases, years)
    make_fig5(invest, args.results_root)
    make_fig6(capacity, args.results_root)

    print("--- figure 3 c/d cross-check ---")
    print(f"cases: {cases}; years: {years}")
    skipped = sorted({c for (c, _) in metrics} - set(cases))
    if skipped:
        print(f"  cases present for some years only, not drawn: {skipped}")
    print("investment $bn/yr:\n" + invest.round(2).to_string())
    print("mean new capacity GW:\n" + capacity.groupby(level="case").mean().round(2).to_string())


if __name__ == "__main__":
    main()
