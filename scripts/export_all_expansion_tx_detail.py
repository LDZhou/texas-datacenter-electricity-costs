from pathlib import Path
import re
import pandas as pd


ROOT = Path("results/dc_experiments_full_tx")
OUT_CSV = ROOT / "all_full_tx_expansion_detail.csv"
OUT_XLSX = ROOT / "all_full_tx_expansion_detail.xlsx"


def parse_case(year: str, case: str) -> dict | None:
    """
    Keep only full_tx cases:

      none_full_tx
      all2030_full_tx
      multiloc_full_tx_{LOCATION}_{SCALE}MW

    Drop:
      distributed_*
      *_dispatch
      *_generation
      *_generation_tx
      old *_expansion_tx
    """
    if case.startswith("distributed_"):
        return None

    if case == "none_full_tx":
        return {
            "year": int(year),
            "dc_scenario": "none",
            "mode": "full_tx",
            "location": None,
            "scale_mw": None,
            "case": case,
        }

    if case == "all2030_full_tx":
        return {
            "year": int(year),
            "dc_scenario": "all2030",
            "mode": "full_tx",
            "location": "ALL_2030",
            "scale_mw": None,
            "case": case,
        }

    m = re.match(r"^multiloc_full_tx_(?P<location>.+)_(?P<scale>\d+)MW$", case)
    if m:
        return {
            "year": int(year),
            "dc_scenario": "multiloc",
            "mode": "full_tx",
            "location": m.group("location"),
            "scale_mw": int(m.group("scale")),
            "case": case,
        }

    return None


def classify_component(row) -> str:
    component = str(row.get("component", ""))
    carrier = str(row.get("carrier", ""))

    if component == "generators":
        return "generation"
    if component == "storage_units":
        return "storage"
    if component == "lines" or carrier == "transmission":
        return "transmission"

    return component


def classify_powerworld_action(row) -> str:
    asset_type = str(row.get("asset_type", ""))

    if asset_type == "generation":
        return "add_or_expand_generator"
    if asset_type == "storage":
        return "add_or_expand_storage"
    if asset_type == "transmission":
        return "increase_branch_capacity"

    return "review"


def split_bus_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    For transmission rows, new_capacity.csv stores bus as:

        bus0 -> bus1

    Split it into from_bus and to_bus for PowerWorld mapping.
    For generation/storage rows, from_bus is just the local bus and to_bus is None.
    """
    df = df.copy()

    if "bus" not in df.columns:
        df["bus"] = None
        df["from_bus"] = None
        df["to_bus"] = None
        return df

    bus_str = df["bus"].astype(str)

    split_bus = bus_str.str.split(" -> ", n=1, expand=True)
    df["from_bus"] = split_bus[0]

    if split_bus.shape[1] > 1:
        df["to_bus"] = split_bus[1]
    else:
        df["to_bus"] = None

    # Clean fake "nan" strings.
    df.loc[df["from_bus"].isin(["nan", "None", ""]), "from_bus"] = None
    df.loc[df["to_bus"].isin(["nan", "None", ""]), "to_bus"] = None

    return df


def load_metrics_if_exists(exp_dir: Path) -> dict:
    """
    Optional metadata from metrics.csv. This is useful for checking solve_status
    and total_new_mw, but the script does not require it.
    """
    metrics_csv = exp_dir / "metrics.csv"
    if not metrics_csv.exists():
        return {}

    try:
        mdf = pd.read_csv(metrics_csv)
        if mdf.empty:
            return {}
        row = mdf.iloc[0].to_dict()
        keep = [
            "solve_status",
            "objective",
            "total_new_mw",
            "total_annual_investment",
            "new_generation_mw",
            "new_storage_mw",
            "new_transmission_mw",
            "new_CCGT_mw",
            "new_OCGT_mw",
            "new_solar_mw",
            "new_onwind_mw",
            "new_battery_mw",
        ]
        return {f"metrics_{k}": row.get(k) for k in keep if k in row}
    except Exception:
        return {}


def main():
    rows = []
    missing_new_capacity = []
    parsed_cases = 0

    for year_dir in sorted(ROOT.glob("*")):
        if not year_dir.is_dir():
            continue

        year = year_dir.name
        if not year.isdigit():
            continue

        for exp_dir in sorted(year_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            case = exp_dir.name
            meta = parse_case(year, case)
            if meta is None:
                continue

            parsed_cases += 1

            cap_file = exp_dir / "new_capacity.csv"
            if not cap_file.exists():
                missing_new_capacity.append(str(exp_dir))
                continue

            df = pd.read_csv(cap_file)

            if df.empty:
                continue

            # Keep only true additions.
            if "added_mw" in df.columns:
                df["added_mw"] = pd.to_numeric(df["added_mw"], errors="coerce")
                df = df[df["added_mw"].fillna(0) > 0.1].copy()

            if df.empty:
                continue

            # Add case metadata.
            for k, v in meta.items():
                df.insert(0, k, v)

            df["experiment_dir"] = str(exp_dir)

            # Add optional metrics metadata.
            metrics_meta = load_metrics_if_exists(exp_dir)
            for k, v in metrics_meta.items():
                df[k] = v

            # Classify asset type.
            df["asset_type"] = df.apply(classify_component, axis=1)

            # Split bus field for PowerWorld.
            df = split_bus_field(df)

            # Add PowerWorld-friendly helper columns.
            df["powerworld_action"] = df.apply(classify_powerworld_action, axis=1)

            df["powerworld_note"] = df.apply(
                lambda r: (
                    "Use from_bus and to_bus to map branch upgrade"
                    if r["asset_type"] == "transmission"
                    else "Use from_bus as generator/storage bus"
                ),
                axis=1,
            )

            # Stable case ID.
            df["case_id"] = df.apply(
                lambda r: (
                    f"{int(r['year'])}_{r['dc_scenario']}_{r['mode']}_"
                    f"{r['location']}_{'' if pd.isna(r['scale_mw']) else int(r['scale_mw'])}"
                ),
                axis=1,
            )

            rows.append(df)

    if not rows:
        print("No full_tx new_capacity.csv files found.")
        print(f"Checked root: {ROOT}")
        return

    out = pd.concat(rows, ignore_index=True)

    # Normalize numeric columns.
    for col in [
        "scale_mw",
        "base_mw",
        "new_mw",
        "added_mw",
        "capex_per_mw_yr",
        "annual_investment",
        "metrics_total_new_mw",
        "metrics_total_annual_investment",
        "metrics_new_generation_mw",
        "metrics_new_storage_mw",
        "metrics_new_transmission_mw",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Column order.
    preferred_cols = [
        "case_id",
        "year",
        "dc_scenario",
        "mode",
        "location",
        "scale_mw",
        "case",
        "asset_type",
        "powerworld_action",
        "component",
        "carrier",
        "zone",
        "name",
        "bus",
        "from_bus",
        "to_bus",
        "base_mw",
        "new_mw",
        "added_mw",
        "capex_per_mw_yr",
        "annual_investment",
        "metrics_solve_status",
        "metrics_total_new_mw",
        "metrics_total_annual_investment",
        "metrics_new_generation_mw",
        "metrics_new_storage_mw",
        "metrics_new_transmission_mw",
        "experiment_dir",
        "powerworld_note",
    ]

    cols = [c for c in preferred_cols if c in out.columns]
    cols += [c for c in out.columns if c not in cols]
    out = out[cols]

    # Sort.
    sort_cols = [
        c
        for c in [
            "year",
            "dc_scenario",
            "location",
            "scale_mw",
            "asset_type",
            "zone",
            "added_mw",
        ]
        if c in out.columns
    ]
    ascending = [True, True, True, True, True, True, False][: len(sort_cols)]
    out = out.sort_values(sort_cols, ascending=ascending)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # Summary 1: by case and asset type.
    summary_case = (
        out.groupby(
            [
                "year",
                "case",
                "dc_scenario",
                "location",
                "scale_mw",
                "asset_type",
                "carrier",
                "zone",
            ],
            dropna=False,
        )
        .agg(
            n_assets=("name", "count"),
            added_mw=("added_mw", "sum"),
            annual_investment=("annual_investment", "sum"),
        )
        .reset_index()
        .sort_values(["year", "case", "asset_type", "carrier", "zone"])
    )

    # Summary 2: PowerWorld-oriented branch upgrades only.
    tx_detail = out[out["asset_type"] == "transmission"].copy()

    if not tx_detail.empty:
        tx_summary = (
            tx_detail.groupby(
                [
                    "year",
                    "case",
                    "dc_scenario",
                    "location",
                    "scale_mw",
                    "name",
                    "from_bus",
                    "to_bus",
                    "zone",
                ],
                dropna=False,
            )
            .agg(
                added_mw=("added_mw", "sum"),
                new_mw=("new_mw", "max"),
                base_mw=("base_mw", "max"),
                annual_investment=("annual_investment", "sum"),
            )
            .reset_index()
            .sort_values(["year", "case", "zone", "added_mw"], ascending=[True, True, True, False])
        )
    else:
        tx_summary = pd.DataFrame()

    # Summary 3: generation/storage only.
    gen_storage_detail = out[out["asset_type"].isin(["generation", "storage"])].copy()

    if not gen_storage_detail.empty:
        gen_storage_summary = (
            gen_storage_detail.groupby(
                [
                    "year",
                    "case",
                    "dc_scenario",
                    "location",
                    "scale_mw",
                    "asset_type",
                    "carrier",
                    "from_bus",
                    "zone",
                ],
                dropna=False,
            )
            .agg(
                n_assets=("name", "count"),
                added_mw=("added_mw", "sum"),
                new_mw=("new_mw", "sum"),
                annual_investment=("annual_investment", "sum"),
            )
            .reset_index()
            .sort_values(
                ["year", "case", "asset_type", "carrier", "zone", "added_mw"],
                ascending=[True, True, True, True, True, False],
            )
        )
    else:
        gen_storage_summary = pd.DataFrame()

    # Missing report.
    missing_df = pd.DataFrame({"experiment_dir": missing_new_capacity})

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="detail", index=False)
        summary_case.to_excel(writer, sheet_name="summary_by_case", index=False)
        tx_summary.to_excel(writer, sheet_name="tx_for_powerworld", index=False)
        gen_storage_summary.to_excel(writer, sheet_name="gen_storage", index=False)
        missing_df.to_excel(writer, sheet_name="missing_new_capacity", index=False)

    missing_txt = ROOT / "full_tx_missing_new_capacity_dirs.txt"
    missing_txt.write_text("\n".join(missing_new_capacity) + ("\n" if missing_new_capacity else ""))

    print("=" * 80)
    print("Saved full_tx expansion export")
    print("=" * 80)
    print(f"Root checked:       {ROOT}")
    print(f"Parsed cases:       {parsed_cases}")
    print(f"Detail rows:        {len(out)}")
    print(f"Missing cap files:  {len(missing_new_capacity)}")
    print()
    print(f"Saved detail CSV:   {OUT_CSV}")
    print(f"Saved Excel file:   {OUT_XLSX}")
    print(f"Missing list:       {missing_txt}")
    print()
    print("Summary preview:")
    print(summary_case.head(20).to_string(index=False))


if __name__ == "__main__":
    main()