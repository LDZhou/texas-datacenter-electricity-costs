#!/usr/bin/env python3
"""Create revision-v2 summaries exclusively from one fresh run root.

The command deliberately treats output directories as data contracts: cases are
paired only inside an identical study horizon and group, and fingerprints from
the producing model must agree before deltas are emitted. It never invokes
retailer or engineering analyses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OFFER_CAP = 5000.0
GROUP_HORIZON = {"seasonal": "seasonal", "central": "seasonal", "weather": "seasonal", "sensitivity": "seasonal", "siting": "seasonal",
                 "full_year": "full_year", "siting_full_year": "full_year",
                 "numerics": "seasonal"}
_MODES = "dispatch|generation_storage|generation_tx|generation|transmission|full_tx|expansion_tx|expansion"


class MetadataMismatch(ValueError):
    """Raised when a proposed comparison is not an experiment match."""


@dataclass(frozen=True)
class Identity:
    group: str
    horizon: str
    year: int
    scenario: str
    mode: str
    variant: str = "base"
    crossover: int = 0
    location: str | None = None
    scale_mw: int | None = None


def parse_experiment_dir(name: str, *, group: str, year: int) -> Identity:
    """Parse canonical case directories, retaining all approved suffixes."""
    horizon = GROUP_HORIZON.get(group, group)
    siting = re.match(rf"^multiloc_({_MODES})_(.+)_(\d+)MW(?:_crossover(\d+))?(?:_(.+))?$", name)
    simple = re.match(rf"^(none|all2030)_({_MODES})(?:_crossover(\d+))?(?:_(.+))?$", name)
    if siting:
        return Identity(group, horizon, int(year), "multiloc", siting.group(1),
                        siting.group(5) or "base", int(siting.group(4) or 0), siting.group(2), int(siting.group(3)))
    if simple:
        return Identity(group, horizon, int(year), simple.group(1), simple.group(2),
                        simple.group(4) or "base", int(simple.group(3) or 0))
    raise ValueError(f"Unrecognised revision-v2 result directory: {name}")


def capped_load_weighted_mean(prices: pd.DataFrame, loads: pd.DataFrame,
                              snapshot_weights: pd.Series, cap: float = OFFER_CAP) -> float:
    common = prices.columns.intersection(loads.columns)
    if common.empty:
        return np.nan
    w = pd.Series(snapshot_weights, index=prices.index).reindex(prices.index).fillna(0.0)
    demand = loads.reindex(index=prices.index, columns=common).fillna(0.0).clip(lower=0.0)
    price = prices.reindex(index=prices.index, columns=common).clip(upper=cap)
    denominator = demand.mul(w, axis=0).to_numpy().sum()
    return float(price.mul(demand).mul(w, axis=0).to_numpy().sum() / denominator) if denominator > 0 else np.nan


def capped_temporal_load_weighted_mean(prices: pd.DataFrame, loads: pd.DataFrame,
                                       snapshot_weights: pd.Series, cap: float = OFFER_CAP) -> float:
    """Manuscript-compatible system LMP: fixed bus demand shares, then time mean."""
    common = prices.columns.intersection(loads.columns)
    if common.empty:
        return np.nan
    shares = loads.reindex(columns=common).mean(axis=0).clip(lower=0.0)
    if shares.sum() <= 0:
        return np.nan
    per_snapshot = prices.reindex(columns=common).clip(upper=cap).mul(shares / shares.sum(), axis=1).sum(axis=1)
    w = pd.Series(snapshot_weights, index=per_snapshot.index).fillna(0.0)
    return float((per_snapshot * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def weighted_unserved_mwh(dispatch: pd.DataFrame, signs: pd.Series, snapshot_weights: pd.Series) -> float:
    """Physical shedding: PyPSA dispatch multiplied by each component sign."""
    if dispatch.empty:
        return 0.0
    signed = dispatch.mul(signs.reindex(dispatch.columns).fillna(1.0), axis=1).clip(lower=0.0)
    return float(signed.mul(pd.Series(snapshot_weights, index=dispatch.index), axis=0).to_numpy().sum())


def validate_pair(base: Identity, base_fp: dict[str, Any], scenario: Identity,
                  scenario_fp: dict[str, Any]) -> None:
    for field in ("horizon", "year", "mode", "variant", "crossover"):
        if getattr(base, field) != getattr(scenario, field):
            raise MetadataMismatch(f"{field} differs: {getattr(base, field)!r} != {getattr(scenario, field)!r}")
    for identity, fingerprints in ((base, base_fp), (scenario, scenario_fp)):
        expected = {"year": identity.year, "horizon": identity.horizon, "mode": identity.mode,
                    "config_variant": identity.variant, "crossover": identity.crossover}
        for key, value in expected.items():
            if fingerprints.get(key) != value:
                raise MetadataMismatch(f"{key} does not match parsed output identity")
    for key in ("starting_network_sha256", "model_contract_version", "candidate_carriers", "capital_cost_factor"):
        a, b = base_fp.get(key), scenario_fp.get(key)
        if key in {"starting_network_sha256", "model_contract_version"} and (not a or not b):
            raise MetadataMismatch(f"{key} is missing or differs")
        if a != b:
            raise MetadataMismatch(f"{key} is missing or differs")
    if scenario.scenario != "none" and not scenario_fp.get("dc_profile_sha256"):
        raise MetadataMismatch("scenario dc_profile_sha256 is missing")


def _snapshot_weights(net) -> pd.Series:
    sw = net.snapshot_weightings
    col = "objective" if "objective" in sw else ("generators" if "generators" in sw else None)
    return sw[col] if col else pd.Series(1.0, index=net.snapshots)


def _ercot_buses(net) -> pd.Index:
    buses = net.buses
    for column in ("nerc_reg", "nerc_region"):
        if column in buses:
            selected = buses.index[buses[column].astype(str).str.upper().eq("ERCOT")]
            if len(selected):
                return selected
    raise ValueError("network has no ERCOT buses identified by nerc_reg/nerc_region")


def _bus_loads(net, buses: pd.Index) -> pd.DataFrame:
    """Fallback historical loads; scenario DC demand is deliberately excluded."""
    p = net.loads_t.p_set
    ids = net.loads.index.intersection(p.columns)
    load_buses = net.loads.loc[ids, "bus"]
    ids = load_buses.index[load_buses.isin(buses) & ~load_buses.index.astype(str).str.startswith("DC_")]
    data = p[ids].T.groupby(load_buses.loc[ids]).sum().T
    return data.reindex(columns=buses, fill_value=0.0)


def _headline_bus_weights(net, buses: pd.Index) -> tuple[pd.Series, str]:
    weights = (pd.to_numeric(net.buses.Pd.reindex(buses), errors="coerce").fillna(0.0).clip(lower=0.0)
               if "Pd" in net.buses else _bus_loads(net, buses).mean(axis=0).reindex(buses).fillna(0.0))
    digest = hashlib.sha256("|".join(f"{bus}:{value:.12g}" for bus, value in weights.items()).encode()).hexdigest()
    return weights, digest


def capped_temporal_bus_weighted_mean(prices: pd.DataFrame, bus_weights: pd.Series,
                                      snapshot_weights: pd.Series, cap: float = OFFER_CAP) -> float:
    shares = bus_weights.reindex(prices.columns).fillna(0.0).clip(lower=0.0)
    if shares.sum() <= 0:
        return np.nan
    per_snapshot = prices.clip(upper=cap).mul(shares / shares.sum(), axis=1).sum(axis=1)
    w = pd.Series(snapshot_weights, index=per_snapshot.index).fillna(0.0)
    return float((per_snapshot * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def _dynamic_dispatch_cost(net, weights: pd.Series) -> float:
    p = net.generators_t.p
    if p.empty:
        return 0.0
    dynamic = getattr(net.generators_t, "marginal_cost", pd.DataFrame())
    if dynamic is None or dynamic.empty:
        dynamic = pd.DataFrame(np.tile(net.generators.marginal_cost.reindex(p.columns).fillna(0.0).to_numpy(),
                                       (len(p.index), 1)), index=p.index, columns=p.columns)
    else:
        dynamic = dynamic.reindex(index=p.index, columns=p.columns)
        dynamic = dynamic.fillna(net.generators.marginal_cost.reindex(p.columns).fillna(0.0))
    return float(p.mul(dynamic).mul(weights, axis=0).to_numpy().sum())


def _co2_tonnes(net, weights: pd.Series) -> float:
    p = net.generators_t.p
    if p.empty or "carrier" not in net.generators or "co2_emissions" not in net.carriers:
        return np.nan
    carrier = net.generators.carrier.reindex(p.columns)
    factor = carrier.map(net.carriers.co2_emissions).fillna(0.0)
    efficiency = net.generators.efficiency.reindex(p.columns).replace(0.0, np.nan).fillna(1.0)
    return float(p.clip(lower=0.0).mul(factor / efficiency).mul(weights, axis=0).to_numpy().sum())


def extract_network_metrics(path: Path) -> dict[str, float]:
    """Open exactly one solved NetCDF and release it before the next case."""
    import pypsa
    net = pypsa.Network(str(path))
    weights = _snapshot_weights(net)
    buses = _ercot_buses(net)
    prices = net.buses_t.marginal_price.reindex(columns=buses)
    loads = _bus_loads(net, buses)
    headline_weights, headline_weights_hash = _headline_bus_weights(net, buses)
    lmp = capped_temporal_bus_weighted_mean(prices, headline_weights, weights)
    lmp_energy = capped_load_weighted_mean(prices, loads, weights)
    any_cap = prices.gt(OFFER_CAP).any(axis=1)
    shedding = net.generators.index[net.generators.carrier.astype(str).str.contains("load|shed", case=False, regex=True)]
    eue = weighted_unserved_mwh(net.generators_t.p.reindex(columns=shedding, fill_value=0.0),
                                net.generators.sign.reindex(shedding), weights)
    cap = pd.read_csv(path.parent / "new_capacity.csv") if (path.parent / "new_capacity.csv").exists() else pd.DataFrame()
    return {"ercot_capped_lmp_$/MWh": lmp,
            "ercot_capped_lmp_energy_weighted_$/MWh": lmp_energy, "eue_mwh": eue,
            "wholesale_weights_sha256": headline_weights_hash,
            "above_cap_hours": float(weights.reindex(any_cap.index).where(any_cap, 0.0).sum()),
            "dispatch_cost_dynamic_$": _dynamic_dispatch_cost(net, weights), "co2_tonnes": _co2_tonnes(net, weights),
            "new_capacity_mw": float(pd.to_numeric(cap.get("added_mw", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "annual_investment_$": float(pd.to_numeric(cap.get("annual_investment", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())}


def _metadata(case: Path) -> dict[str, Any]:
    metrics = json.loads((case / "metrics.json").read_text())
    fp = metrics.get("input_fingerprints", {})
    if not fp:
        raise MetadataMismatch(f"{case}: metrics.json has no input_fingerprints")
    return fp


def discover(run_root: Path, groups: Iterable[str]) -> list[tuple[Identity, Path, dict[str, Any]]]:
    cases = []
    for group in groups:
        root = run_root / "results" / group
        if not root.exists():
            continue
        for year_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
            for case in sorted(p for p in year_dir.iterdir() if p.is_dir() and (p / "network.nc").exists()):
                identity = parse_experiment_dir(case.name, group=group, year=int(year_dir.name))
                cases.append((identity, case, _metadata(case)))
    return cases


def check_inventory(observed: Iterable[Identity], matrix: dict[str, Any], requested_groups: set[str]) -> list[Identity]:
    """Return planned matrix identities absent from fresh results."""
    expected: list[Identity] = []
    stages = matrix.get("groups", matrix)
    for _stage, entries in stages.items():
        if isinstance(entries, dict):
            entries = entries.get("cases", [])
        for item in entries:
            group = item.get("group", _stage)
            if group not in requested_groups:
                continue
            expected.append(Identity(group, item.get("horizon", GROUP_HORIZON.get(group, group)), int(item["year"]),
                                     item["scenario"], item["mode"], item.get("variant", "base"),
                                     int(item.get("crossover", 0)), item.get("location"), item.get("scale_mw", item.get("scale"))))
    seen = set(observed)
    return [identity for identity in expected if identity not in seen]


def pair_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(r["identity"].group, r["identity"].horizon, r["identity"].year, r["identity"].mode,
              r["identity"].variant, r["identity"].crossover): r for r in rows if r["identity"].scenario == "none"}
    pairs = []
    rejections: list[dict[str, str]] = []
    for row in rows:
        identity = row["identity"]
        if identity.scenario == "none":
            continue
        key = (identity.group, identity.horizon, identity.year, identity.mode, identity.variant, identity.crossover)
        base = index.get(key)
        if not base:
            rejections.append({"scenario_path": str(row["path"]), "reason": "missing_baseline"})
            continue
        try:
            validate_pair(base["identity"], base["fingerprints"], identity, row["fingerprints"])
            if base["metrics"].get("wholesale_weights_sha256") != row["metrics"].get("wholesale_weights_sha256"):
                raise MetadataMismatch("wholesale_weights_sha256 differs")
        except MetadataMismatch as exc:
            rejections.append({"scenario_path": str(row["path"]), "baseline_path": str(base["path"]), "reason": str(exc)})
            continue
        pair = {**asdict(identity), "baseline_path": str(base["path"]), "scenario_path": str(row["path"])}
        for metric in ("ercot_capped_lmp_$/MWh", "ercot_capped_lmp_energy_weighted_$/MWh", "eue_mwh", "above_cap_hours", "dispatch_cost_dynamic_$", "co2_tonnes", "new_capacity_mw", "annual_investment_$"):
            pair[f"baseline_{metric}"] = base["metrics"].get(metric, np.nan)
            pair[f"scenario_{metric}"] = row["metrics"].get(metric, np.nan)
            pair[f"delta_{metric}"] = pair[f"scenario_{metric}"] - pair[f"baseline_{metric}"]
        pairs.append(pair)
    pair_cases.rejections = rejections
    return pairs


def validate_exogenous_profiles(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """DC profile must be identical across modes and controlled variants."""
    buckets: dict[tuple, set[str]] = {}
    for row in rows:
        i, fp = row["identity"], row["fingerprints"]
        if i.scenario == "none":
            continue
        key = (i.horizon, i.year, i.scenario, i.location, i.scale_mw)
        buckets.setdefault(key, set()).add(str(fp.get("dc_profile_sha256") or "MISSING"))
    return [{"identity": repr(key), "reason": "dc_profile_sha256 differs across modes/variants"}
            for key, hashes in buckets.items() if len(hashes) != 1 or "MISSING" in hashes]


def export_texas2k(case_rows: list[dict[str, Any]], out_dir: Path) -> None:
    target = out_dir / "texas2k_handoff"
    target.mkdir(parents=True, exist_ok=True)
    allowed = {"none", "all2030"}
    for row in case_rows:
        i, path = row["identity"], row["path"]
        if (i.group != "seasonal" or i.year != 2023 or i.variant != "base" or i.crossover != 0
                or i.scenario not in allowed or i.mode not in {"generation_storage", "full_tx"}):
            continue
        import pypsa
        net = pypsa.Network(str(path / "network.nc"))
        cap_path = path / "new_capacity.csv"
        cap = pd.read_csv(cap_path) if cap_path.exists() else pd.DataFrame()
        components = cap.get("component", pd.Series(index=cap.index, dtype=str)).astype(str).str.lower()
        supply = cap[components.isin(["generator", "generators", "storageunit", "storage_units"])].copy()
        if not supply.empty:
            supply["x"] = supply.bus.map(net.buses.x)
            supply["y"] = supply.bus.map(net.buses.y)
            supply[["component", "name", "bus", "x", "y", "carrier", "added_mw"]].rename(columns={"added_mw": "new_mw"}).to_csv(target / f"{i.scenario}_{i.mode}_supply.csv", index=False)
        loads = net.loads[["bus", "carrier"]].copy()
        loads["x"], loads["y"] = loads.bus.map(net.buses.x), loads.bus.map(net.buses.y)
        loads["mean_mw"] = net.loads_t.p_set.reindex(columns=loads.index).mean()
        loads.reset_index(names="load_name")[["load_name", "bus", "x", "y", "carrier", "mean_mw"]].to_csv(target / f"{i.scenario}_{i.mode}_loads.csv", index=False)
    (target / "README.md").write_text("""# Texas2k handoff\n\nThese are fresh seasonal, base-variant PyPSA 2023 supply-addition and load tables. Coordinates are PyPSA bus coordinates; `carrier` and `new_mw` preserve the planning result. They support the same expanded-supply no-DC engineering baseline and corresponding DC cases. Texas2k bus mapping and contingency analysis are downstream steps.\n""")


SITING_METRICS = ("scenario_ercot_capped_lmp_$/MWh", "delta_ercot_capped_lmp_$/MWh", "scenario_eue_mwh",
                  "delta_eue_mwh", "scenario_new_capacity_mw", "delta_new_capacity_mw",
                  "scenario_annual_investment_$", "delta_annual_investment_$",
                  "scenario_dispatch_cost_dynamic_$", "delta_dispatch_cost_dynamic_$",
                  "scenario_co2_tonnes", "delta_co2_tonnes")


def siting_tables(siting: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-year siting rows and their equal-weight mean over the available weather years."""
    keys = ["mode", "location", "scale_mw"]
    metrics = [m for m in SITING_METRICS if m in siting.columns]
    by_year = siting.sort_values(["mode", "location", "scale_mw", "year"])[["year", *keys, *metrics]].reset_index(drop=True)
    mean = by_year.groupby(keys, as_index=False, sort=True)[metrics].mean()
    counts = by_year.groupby(keys, sort=True).year.agg(n_weather_years="nunique", years=lambda s: ",".join(str(y) for y in sorted(s)))
    return by_year, mean.merge(counts.reset_index(), on=keys)


def make_figures(pairs: pd.DataFrame, out_dir: Path) -> None:
    figs = out_dir / "figures"; figs.mkdir(parents=True, exist_ok=True)
    if pairs.empty:
        return
    seasonal = pairs[pairs.group == "seasonal"]
    for metric, name, ylabel in [("delta_ercot_capped_lmp_$/MWh", "weather_price", "Capped LMP change ($/MWh)"),
                                 ("delta_annual_investment_$", "weather_investment", "Annual investment change ($)")]:
        if seasonal.empty: continue
        fig, ax = plt.subplots(figsize=(5.2, 2.7))
        for mode, sub in seasonal.groupby("mode"):
            ax.plot(sub.year, sub[metric], marker="o", label=mode)
        ax.set_xlabel("Weather year"); ax.set_ylabel(ylabel); ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(figs / f"{name}.png", dpi=300); plt.close(fig)
    sensitivity = pairs[pairs.group == "sensitivity"]
    if not sensitivity.empty:
        fig, ax = plt.subplots(figsize=(4.5, 2.7)); ax.bar(sensitivity.variant, sensitivity["delta_ercot_capped_lmp_$/MWh"], color="#0072B2")
        ax.set_ylabel("Capped LMP change ($/MWh)"); fig.tight_layout(); fig.savefig(figs / "sensitivity.png", dpi=300); plt.close(fig)
    siting = pairs[pairs.group == "siting"]
    if not siting.empty:
        by_year, mean = siting_tables(siting)
        by_year.to_csv(out_dir / "siting_by_year.csv", index=False)
        mean.to_csv(out_dir / "siting_weather_year_mean.csv", index=False)
        # DataFrame.mode is a method, so the column must be indexed by name.
        modes = list(mean["mode"].drop_duplicates()); fig, axes = plt.subplots(1, len(modes), figsize=(3.1 * len(modes), 3), squeeze=False)
        for ax, mode in zip(axes[0], modes):
            sub = mean[mean["mode"] == mode].pivot(index="location", columns="scale_mw", values="scenario_eue_mwh")
            im = ax.imshow(sub.to_numpy(), aspect="auto", cmap="YlOrRd"); ax.set_title(mode, fontsize=8)
            ax.set_xticks(range(len(sub.columns)), [str(x) for x in sub.columns]); ax.set_yticks(range(len(sub.index)), sub.index)
        n_years = siting.year.nunique()
        fig.colorbar(im, ax=axes.ravel().tolist(), label=f"EUE (MWh), mean of {n_years} weather year(s)")
        fig.tight_layout(); fig.savefig(figs / "siting.png", dpi=300); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["seasonal", "sensitivity", "full_year", "siting"])
    parser.add_argument("--matrix", type=Path, default=None,
                        help="Expected-case JSON; defaults to manifests/experiment_matrix.json")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Explicitly permit an incomplete matrix; intended only for diagnostics")
    parser.add_argument("--figures-from-pairs", action="store_true",
                        help="Rebuild figures and siting tables from an existing matched_pairs.csv in --out-dir "
                             "without reopening any network")
    args = parser.parse_args()
    if args.figures_from_pairs:
        pairs = pd.read_csv(args.out_dir / "matched_pairs.csv")
        make_figures(pairs, args.out_dir)
        print(f"Rebuilt figures from {len(pairs)} matched pairs in {args.out_dir}")
        return
    discovered = discover(args.run_root, args.groups)
    matrix_path = args.matrix or args.run_root / "manifests" / "experiment_matrix.json"
    if not args.allow_partial:
        if not matrix_path.exists():
            raise SystemExit(f"Missing expected-case matrix: {matrix_path}; use --allow-partial only for diagnostics")
        missing = check_inventory([item[0] for item in discovered], json.loads(matrix_path.read_text()), set(args.groups))
        if missing:
            rendered = ", ".join(f"{i.group}/{i.year}/{i.scenario}_{i.mode}_{i.variant}" for i in missing)
            raise SystemExit(f"Incomplete fresh-output matrix ({len(missing)} missing): {rendered}")
    rows = []
    for identity, path, fingerprints in discovered:
        rows.append({"identity": identity, "path": path, "fingerprints": fingerprints,
                     "metrics": extract_network_metrics(path / "network.nc")})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{**asdict(r["identity"]), "path": str(r["path"]), **r["metrics"]} for r in rows])
    summary.to_csv(args.out_dir / "canonical_summary.csv", index=False)
    rejections = validate_exogenous_profiles(rows)
    pairs = pd.DataFrame(pair_cases(rows)); pairs.to_csv(args.out_dir / "matched_pairs.csv", index=False)
    rejections.extend(getattr(pair_cases, "rejections", []))
    pd.DataFrame(rejections).to_csv(args.out_dir / "pair_rejections.csv", index=False)
    if rejections:
        raise SystemExit(f"Rejected {len(rejections)} invalid comparisons; see {args.out_dir / 'pair_rejections.csv'}")
    export_texas2k(rows, args.out_dir); make_figures(pairs, args.out_dir)
    print(f"Wrote {len(summary)} case summaries and {len(pairs)} validated matched pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
