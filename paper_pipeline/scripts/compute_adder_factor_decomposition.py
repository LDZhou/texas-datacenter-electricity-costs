"""
compute_adder_factor_decomposition.py
=====================================
Decompose the wholesale-energy customer adder into three additive factors,
using only existing solved networks (no new optimisation).

In a lossless DC network the nodal price separates as

    lambda_i,t = ref_t + congestion_i,t ,      ref_t := min_i lambda_i,t

so, writing lambda_bar_t for the demand-weighted system price and applying the
ERCOT system-wide offer cap before the split,

    lambda_bar_t = ref_t  +  congestion_t  +  scarcity_t
      ref_t        = min_i clip(lambda_i,t)          energy / marginal unit
      congestion_t = wmean(clip(lambda)) - ref_t     deliverability rent  (>= 0)
      scarcity_t   = wmean(lambda) - wmean(clip)     price above the cap

The adder decomposition is then the case-minus-baseline difference of the
time-mean of each term. Each scenario-level component is non-negative; its
case-minus-baseline change can have either sign. The changes sum exactly to the
reported adder.

Weighting matches dc_common._bus_demand_weights_lmp, i.e. the same fixed
demand weights used for the headline sys_mean_lmp in the manuscript.

Usage
-----
    python scripts/compute_adder_factor_decomposition.py
    python scripts/compute_adder_factor_decomposition.py --cap 9000

Output
------
    results/nc_paper_figures/adder_factor_decomposition.csv
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")
from . import dc_common as dcc

YEARS = [2019, 2020, 2021, 2022, 2023]

SCENARIO_CASES = ["full_tx"]


def case_path(results_root: Path, year: int, case: str) -> Path:
    """Return an explicitly matched baseline/scenario path from the revision matrix."""
    names = {
        "baseline": "none_full_tx",
        "full_tx": "all2030_full_tx",
    }
    return results_root / "seasonal" / str(year) / names[case] / "network.nc"


def decompose(path: str, cap: float) -> dict:
    n = pypsa.Network(path)
    ercot = dcc.get_ercot_buses(n)
    cols = ercot.index.intersection(n.buses_t.marginal_price.columns)
    lmp = n.buses_t.marginal_price[cols]

    w = dcc._bus_demand_weights_lmp(n, cols).reindex(cols).fillna(0)
    if w.sum() <= 0:
        w = pd.Series(1.0, index=cols)
    w = w / w.sum()

    lmp_c = lmp.clip(upper=cap)
    wmean = (lmp * w).sum(axis=1)
    wmean_c = (lmp_c * w).sum(axis=1)
    ref = lmp_c.min(axis=1)

    arr = lmp.values.flatten()
    arr = arr[~np.isnan(arr)]

    return {
        "lmp": float(wmean.mean()),
        "lmp_capped": float(wmean_c.mean()),
        "ref": float(ref.mean()),
        "congestion": float((wmean_c - ref).mean()),
        "scarcity": float((wmean - wmean_c).mean()),
        "lmp_max": float(arr.max()),
        "snaps_over_cap": int((wmean > cap).sum()),
        "snaps_any_bus_over_cap": int((lmp.max(axis=1) >= cap).sum()),
        "n_snapshots": int(len(wmean)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=5000.0,
                    help="ERCOT system-wide offer cap ($/MWh) applied before the split")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Root containing the solved experiment result directories.")
    ap.add_argument("--years", nargs="*", type=int, default=YEARS)
    args = ap.parse_args()
    args.out = args.out.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()

    rows = []
    for y in args.years:
        for case in ("baseline", *SCENARIO_CASES):
            p = case_path(args.results_root, y, case)
            if not p.exists():
                print(f"[skip] {y} {case}: missing {p}")
                continue
            rows.append({"year": y, "case": case, **decompose(str(p), args.cap)})
            print(f"  done {y} {case}", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(args.out, index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n================ LEVELS ================")
    print(df.round(3).to_string(index=False))

    print(f"\n========= ADDER DECOMPOSITION (case - baseline), cap ${args.cap:,.0f}/MWh =========")
    print(f"{'yr':>5} {'case':13} {'adder':>8} | {'energy/ref':>18} {'congestion':>18} {'scarcity>cap':>18}")
    summary = []
    for y in args.years:
        d = df[df.year == y].set_index("case")
        if "baseline" not in d.index:
            continue
        b = d.loc["baseline"]
        for case in SCENARIO_CASES:
            if case not in d.index:
                continue
            s = d.loc[case]
            total = s.lmp - b.lmp
            parts = {
                "energy": s.ref - b.ref,
                "congestion": s.congestion - b.congestion,
                "scarcity": s.scarcity - b.scarcity,
            }
            txt = "  ".join(
                f"{v:7.2f} ({100 * v / total:5.1f}%)" for v in parts.values()
            )
            print(f"{y:>5} {case:13} {total:8.2f} | {txt}")
            summary.append({"year": y, "case": case, "adder": total, **parts})

    s = pd.DataFrame(summary)
    ft = s[s.case == "full_tx"]
    if len(ft):
        print("\n---- full_tx, five-year mean ----")
        tot = ft.adder.mean()
        for k in ["energy", "congestion", "scarcity"]:
            v = ft[k].mean()
            print(f"  {k:12s} {v:7.2f} $/MWh  ({100 * v / tot:5.1f}%)")
        print(f"  {'TOTAL':12s} {tot:7.2f} $/MWh   ({tot / 10:.2f} cents/kWh)")

    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
