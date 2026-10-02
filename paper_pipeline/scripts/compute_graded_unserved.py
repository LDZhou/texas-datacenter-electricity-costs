"""
compute_graded_unserved.py
==========================
Replace the binary UNSERVED flag with a graded severity measure.

Why this exists
---------------
The manuscript classifies a case as an "unserved-load regime" when the
load-shedding penalty cost exceeds $1 million. The load-shedding pseudo-
generators carry PyPSA-Eur's sign=1e-3 (kW) convention, so the effective
value of lost load in the solved networks is $5,000,000/MWh, not the
$5,000/MWh stated in Methods. At that penalty price the $1M threshold
corresponds to 0.2 MWh over a 4,392-hour season -- i.e. the flag fires on
*any* shedding at all, and a case shedding 0.2 MWh gets the same label as
one shedding 44 TWh.

This script reports the underlying quantities so the binary heatmaps
(Fig. 2b, Fig. 4a) can be redrawn on a graded scale:
  - unserved energy (MWh and % of served load)
  - number of snapshots with shedding, and peak shed power
  - LMP distribution and the fraction of bus-hours above the offer cap

Usage
-----
    python scripts/compute_graded_unserved.py
    python scripts/compute_graded_unserved.py --multiloc

Output
------
    results/nc_paper_figures/graded_unserved_all2030.csv
    results/nc_paper_figures/graded_unserved_multiloc.csv   (with --multiloc)
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")
from . import dc_common as dcc

YEARS = [2019, 2020, 2021, 2022, 2023]

# PyPSA-Eur adds load-shedding generators with sign=1e-3, so generators_t.p for
# those units is in kW while the rest of the network is in MW.
SHED_SIGN = 1e-3
SHED_MC_THRESHOLD = 1000.0   # marginal_cost above this identifies a shed unit

ALL2030_CASES = {
    "baseline": "none_full_tx",
    "dispatch": "all2030_dispatch",
    "gen_storage": "all2030_generation_storage",
    "full_tx": "all2030_full_tx",
}


def experiment_group(year: int) -> str:
    return "seasonal"


MULTILOC_GLOB = "siting/{y}/multiloc_*/network.nc"
MULTILOC_RE = re.compile(
    r"multiloc_(dispatch|generation_storage|generation_tx|generation"
    r"|transmission|full_tx|expansion_tx|expansion)_(.+)_(\d+)MW$"
)


def summarise(path: str, cap: float) -> dict:
    n = pypsa.Network(path)
    w = n.snapshot_weightings.objective

    load = float((n.loads_t.p_set.sum(axis=1) * w).sum())

    g = n.generators
    shed_i = [c for c in g.index[g.marginal_cost >= SHED_MC_THRESHOLD]
              if c in n.generators_t.p.columns]
    if shed_i:
        shed_t = n.generators_t.p[shed_i].sum(axis=1) * SHED_SIGN
        shed = float((shed_t * w).sum())
        shed_peak = float(shed_t.max())
        shed_snaps = int((shed_t > 1e-6).sum())
    else:
        shed, shed_peak, shed_snaps = 0.0, 0.0, 0

    ercot = dcc.get_ercot_buses(n)
    cols = ercot.index.intersection(n.buses_t.marginal_price.columns)
    lmp = n.buses_t.marginal_price[cols]
    arr = lmp.values.flatten()
    arr = arr[~np.isnan(arr)]

    return {
        "load_TWh": load / 1e6,
        "unserved_MWh": shed,
        "unserved_pct_of_load": 100 * shed / load if load else np.nan,
        "unserved_peak_MW": shed_peak,
        "snaps_with_shedding": shed_snaps,
        "n_snapshots": int(len(lmp)),
        "lmp_mean": float(arr.mean()),
        "lmp_max": float(arr.max()),
        "pct_bushours_over_cap": 100 * float((arr >= cap).mean()),
        "snaps_any_bus_over_cap": int((lmp.max(axis=1) >= cap).sum()),
    }


def run_all2030(cap: float, out: Path, results_root: Path, years) -> None:
    rows = []
    for y in years:
        group = experiment_group(y)
        for case, name in ALL2030_CASES.items():
            p = results_root / group / str(y) / name / "network.nc"
            if not p.exists():
                print(f"[skip] {y} {case}: missing {p}")
                continue
            rows.append({"year": y, "case": case, **summarise(str(p), cap)})
            print(f"  done {y} {case}", flush=True)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(out, index=False)

    pd.set_option("display.width", 240, "display.max_columns", 30)
    print("\n================ GRADED UNSERVED, all2030 ================")
    print(df.round(4).to_string(index=False))

    print("\n---- severity by case (five-year range) ----")
    for case in ["dispatch", "gen_storage", "gen_tx", "full_tx"]:
        d = df[df.case == case]
        if not len(d):
            continue
        print(f"  {case:12s} unserved {d.unserved_MWh.min():14,.1f} - "
              f"{d.unserved_MWh.max():14,.1f} MWh   "
              f"({d.unserved_pct_of_load.min():7.4f} - "
              f"{d.unserved_pct_of_load.max():7.4f} % of load)")
    print(f"\nsaved {out}")


def run_multiloc(cap: float, out: Path, results_root: Path, years) -> None:
    rows = []
    for y in years:
        for p in sorted(glob.glob(str(results_root / MULTILOC_GLOB.format(y=y)))):
            m = MULTILOC_RE.search(str(Path(p).parent.name))
            if not m:
                continue
            rows.append({
                "year": y, "mode": m.group(1),
                "location": m.group(2), "scale_mw": int(m.group(3)),
                **summarise(p, cap),
            })
            print(f"  done {y} {m.group(1)} {m.group(2)} {m.group(3)}MW", flush=True)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(out, index=False)

    pd.set_option("display.width", 240, "display.max_columns", 30)
    for mode in sorted(df["mode"].unique()):
        d = df[df["mode"] == mode]
        print(f"\n---- {mode}: mean unserved MWh over weather years ----")
        print(d.pivot_table(index="location", columns="scale_mw",
                            values="unserved_MWh", aggfunc="mean").round(1).to_string())
    print(f"\nsaved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=5000.0)
    ap.add_argument("--multiloc", action="store_true",
                    help="Also process the representative-siting runs (slow; ~360 networks)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Root containing the solved experiment result directories.")
    ap.add_argument("--years", nargs="*", type=int, default=YEARS)
    args = ap.parse_args()
    args.outdir = args.outdir.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()

    run_all2030(args.cap, args.outdir / "graded_unserved_all2030.csv", args.results_root, args.years)
    if args.multiloc:
        run_multiloc(args.cap, args.outdir / "graded_unserved_multiloc.csv", args.results_root, args.years)


if __name__ == "__main__":
    main()
