"""
compute_capped_lmp_adders.py
============================
Recompute the wholesale-energy customer adder with the LMP capped at
ERCOT's system-wide offer cap ($5,000/MWh) BEFORE the demand-weighted
aggregation, so that load-shedding / congestion-stacking shadow prices
(which the paper's own methodology disclaims as non-market) no longer
dominate the system-mean LMP.

The capping is applied to the per-bus marginal_price time series and then
run through the *exact* same demand-weighting used in the manuscript
(dc_common.compute_lmp_summary), so the only change vs. the published
number is the removal of the > $5,000 tail.

Outputs (per weather year, seasonal May-Oct networks):
  - baseline / all2030_full_tx sys_mean_lmp, uncapped and capped
  - wholesale adder (uncapped vs capped), system and per zone
  - scarcity frequency (fraction of bus-hours >= $500 and >= $5000)
  - the 98/2 wholesale-vs-transmission split under each cap, using the
    fixed engineering delivery adder.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")
from . import dc_common as dcc

CAP = 5000.0            # ERCOT system-wide offer cap ($/MWh)
TCOS_ADDER = 0.857      # Engineering delivery adder ($/MWh), fixed
YEARS = [2019, 2020, 2021, 2022, 2023]

def summarise(path, cap=None):
    n = pypsa.Network(str(path))
    ercot = dcc.get_ercot_buses(n)
    bus_cols = ercot.index.intersection(n.buses_t.marginal_price.columns)
    raw = n.buses_t.marginal_price[bus_cols].values.flatten()
    raw = raw[~np.isnan(raw)]
    scarcity = {
        "frac_ge_500": float((raw >= 500).mean()),
        "frac_ge_5000": float((raw >= 5000).mean()),
        "max": float(raw.max()),
    }
    if cap is not None:
        n.buses_t.marginal_price = n.buses_t.marginal_price.clip(upper=cap)
    s, _ = dcc.compute_lmp_summary(n, ercot)
    return s, scarcity


def main():
    ap = argparse.ArgumentParser(
        description="Compute wholesale customer adders with per-bus LMP capping."
    )
    ap.add_argument("--cap", type=float, default=CAP,
                    help="ERCOT system-wide offer cap ($/MWh).")
    ap.add_argument("--tcos-adder", type=float, default=TCOS_ADDER,
                    help="Fixed engineering delivery adder ($/MWh).")
    ap.add_argument("--years", nargs="*", type=int, default=YEARS)
    ap.add_argument("--full-template", required=True,
                    help="Full-TX network path template; use {y} and optionally {group} for weather year.")
    ap.add_argument("--base-template", required=True,
                    help="Baseline network path template; use {y} and optionally {group} for weather year.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out = args.out.expanduser().resolve()

    rows = []
    for y in args.years:
        fp = Path(args.full_template.format(y=y, group="seasonal"))
        bp = Path(args.base_template.format(y=y, group="seasonal"))
        if not (fp.exists() and bp.exists()):
            print(f"[skip] {y}: missing network"); continue
        b_unc, b_sc = summarise(bp, cap=None)
        b_cap, _ = summarise(bp, cap=args.cap)
        f_unc, f_sc = summarise(fp, cap=None)
        f_cap, _ = summarise(fp, cap=args.cap)

        add_unc = f_unc["sys_mean_lmp"] - b_unc["sys_mean_lmp"]
        add_cap = f_cap["sys_mean_lmp"] - b_cap["sys_mean_lmp"]
        rows.append({
            "year": y,
            "base_lmp_unc": b_unc["sys_mean_lmp"],
            "full_lmp_unc": f_unc["sys_mean_lmp"],
            "adder_unc": add_unc,
            "base_lmp_cap": b_cap["sys_mean_lmp"],
            "full_lmp_cap": f_cap["sys_mean_lmp"],
            "adder_cap": add_cap,
            "full_frac_ge500_%": 100 * f_sc["frac_ge_500"],
            "full_frac_ge5000_%": 100 * f_sc["frac_ge_5000"],
            "full_max_lmp": f_sc["max"],
            **{f"zadd_cap_{z}": f_cap[f"lmp_{z}"] - b_cap[f"lmp_{z}"]
               for z in dcc.ERCOT_ZONES},
        })

    df = pd.DataFrame(rows).set_index("year")
    pd.set_option("display.width", 200, "display.max_columns", 40)

    print(f"\n================ WHOLESALE ADDER: uncapped vs capped@${args.cap:g} ================")
    view = df[["base_lmp_unc", "full_lmp_unc", "adder_unc",
               "base_lmp_cap", "full_lmp_cap", "adder_cap",
               "full_frac_ge500_%", "full_frac_ge5000_%", "full_max_lmp"]]
    print(view.round(2).to_string())

    print("\n---- adder in cents/kWh ----")
    ck = df[["adder_unc", "adder_cap"]] / 10.0
    ck.columns = ["adder_unc_c/kWh", "adder_cap_c/kWh"]
    print(ck.round(3).to_string())

    print("\n---- per-zone capped adder ($/MWh) ----")
    zc = df[[c for c in df.columns if c.startswith("zadd_cap_")]]
    zc.columns = [c.replace("zadd_cap_", "") for c in zc.columns]
    print(zc.round(2).to_string())

    reference_year = 2023 if 2023 in df.index else int(df.index.max())
    m = df.loc[reference_year]
    print(f"\n================ MATCHED {reference_year}: 98/2 split ================")
    for tag, a in [("uncapped (current paper)", m["adder_unc"]),
                   (f"capped @ ${args.cap:g}", m["adder_cap"])]:
        tot = a + args.tcos_adder
        print(f"{tag:26s}: wholesale {a:6.2f}  + delivery {args.tcos_adder:.3f} "
              f"= {tot:6.2f} $/MWh | wholesale share {100*a/tot:5.1f}% "
              f"delivery {100*args.tcos_adder/tot:4.1f}% | total {tot/10:.2f} c/kWh")

    print("\n================ 5-year capped adder range ================")
    print(f"  capped adder: {df['adder_cap'].min():.2f}-{df['adder_cap'].max():.2f} "
          f"$/MWh  (mean {df['adder_cap'].mean():.2f}); "
          f"{df['adder_cap'].min()/10:.2f}-{df['adder_cap'].max()/10:.2f} c/kWh")
    print(f"  {reference_year} matched is the {'MIN' if df['adder_cap'].idxmin()==reference_year else 'not-min'} of the range")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(args.out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
