"""Fail-closed model-input and post-solve contracts for DC experiments."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


CONTRACT_VERSION = 1
PROFILE_FLOAT_FORMAT = "%.17g"
PRIMAL_RESIDUAL_TOLERANCE_MW = 1e-2


def stable_seed(*parts: object) -> int:
    """Return a process-independent seed; Python's hash() is randomized."""
    payload = json.dumps([str(part) for part in parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**32)


def snapshot_labels(snapshots) -> pd.Index:
    """Canonical labels used in profile files, including MultiIndex snapshots."""
    if isinstance(snapshots, pd.MultiIndex):
        return pd.Index(["|".join(map(str, item)) for item in snapshots], name="snapshot")
    return pd.Index([str(item) for item in snapshots], name="snapshot")


def canonical_profile_bytes(table: pd.DataFrame) -> bytes:
    table = table.copy()
    table.index = snapshot_labels(table.index)
    table = table.sort_index(axis=1)
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError("DC profile contains non-finite values")
    return table.to_csv(float_format=PROFILE_FLOAT_FORMAT).encode("utf-8")


def profile_fingerprint(table: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_profile_bytes(table)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dc_profile_table(path: str | Path, snapshots) -> pd.DataFrame:
    """Read a persisted CSV profile and require the exact requested snapshots."""
    path = Path(path)
    if path.suffix.lower() in {".nc", ".netcdf"}:
        try:
            import xarray as xr
        except ImportError as exc:  # pragma: no cover - dependency-specific
            raise RuntimeError("NetCDF DC profiles require xarray") from exc
        dataset = xr.open_dataset(path)
        if "dc_profile" not in dataset:
            raise ValueError("NetCDF DC profile must contain variable 'dc_profile'")
        table = dataset["dc_profile"].to_pandas()
    else:
        table = pd.read_csv(path, index_col=0, float_precision="round_trip")

    table.index = pd.Index([str(item) for item in table.index], name="snapshot")
    expected = snapshot_labels(snapshots)
    if not table.index.equals(expected):
        raise ValueError("DC profile snapshots do not exactly match the starting network")
    if table.empty or table.columns.duplicated().any():
        raise ValueError("DC profile must contain unique load columns")
    table = table.astype(float)
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("DC profile contains non-finite values")
    return table


def profile_series(table: pd.DataFrame | None, snapshots, load_name: str,
                   generated: pd.Series) -> pd.Series:
    """Use the persisted profile verbatim, or the caller's deterministic profile."""
    if table is None:
        return generated
    if load_name not in table.columns:
        raise ValueError(f"DC profile is missing required column '{load_name}'")
    return pd.Series(table[load_name].to_numpy(dtype=float), index=snapshots, name=load_name)


def write_dc_profiles(network, path: str | Path) -> pd.DataFrame:
    """Persist every datacenter p_set curve in the canonical exchange format."""
    names = network.loads.index[network.loads.carrier.eq("datacenter")]
    values = network.loads_t.p_set.reindex(columns=names)
    if values.empty and len(names):
        raise ValueError("datacenter loads are missing p_set profiles")
    values = values.copy()
    values.index = snapshot_labels(network.snapshots)
    canonical_profile_bytes(values)  # validates before publishing the file
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(destination, float_format=PROFILE_FLOAT_FORMAT)
    return values


def validate_frozen_starting_network(network, tolerance: float = 1e-6) -> None:
    """Require a finite, fully locked historical optimum before scenario unlocks."""
    components = (
        ("generators", "p_nom", "p_nom_opt", "p_nom_extendable"),
        ("storage_units", "p_nom", "p_nom_opt", "p_nom_extendable"),
        ("stores", "e_nom", "e_nom_opt", "e_nom_extendable"),
        ("links", "p_nom", "p_nom_opt", "p_nom_extendable"),
        ("lines", "s_nom", "s_nom_opt", "s_nom_extendable"),
    )
    failures = []
    for name, nominal, optimum, extendable in components:
        table = getattr(network, name, None)
        if table is None or table.empty:
            continue
        if nominal not in table or not np.isfinite(table[nominal].astype(float)).all():
            failures.append(f"{name} has non-finite {nominal}")
        if extendable in table and table[extendable].fillna(False).astype(bool).any():
            failures.append(f"{name} remains extendable")
        if optimum in table:
            known = table[optimum].notna()
            if known.any() and not np.allclose(
                table.loc[known, nominal].astype(float), table.loc[known, optimum].astype(float),
                rtol=tolerance, atol=tolerance,
            ):
                failures.append(f"{name} nominal capacity does not equal frozen optimum")
    if failures:
        raise ValueError("invalid frozen starting network: " + "; ".join(failures))


def apply_capital_cost_factor(network, factor: float) -> None:
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("capital cost factor must be a positive finite number")
    for name in ("generators", "storage_units", "stores", "links", "lines"):
        table = getattr(network, name, None)
        if table is not None and not table.empty and "capital_cost" in table:
            table["capital_cost"] = table["capital_cost"].astype(float) * factor


def _max_abs(frame) -> float:
    if frame is None or getattr(frame, "empty", True):
        return 0.0
    values = np.asarray(frame, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite primal result")
    return float(np.max(np.abs(values))) if values.size else 0.0


def validate_solution(
    network,
    status: str,
    condition: str,
    tolerance: float = PRIMAL_RESIDUAL_TOLERANCE_MW,
) -> dict:
    """Reject invalid solves; report finite residual exceedances as warnings."""
    if status != "ok" or str(condition).lower() != "optimal":
        raise RuntimeError(f"solver did not prove optimality: status={status}, condition={condition}")
    if not np.isfinite(float(network.objective)):
        raise RuntimeError("solver returned a non-finite objective")

    residuals: dict[str, float] = {}
    # ``buses_t.p`` is the net component injection.  It intentionally is
    # non-zero at buses connected by branches, so subtract every branch-port
    # withdrawal before testing the nodal balance.  PyPSA uses positive pN
    # for a withdrawal at that port (thus p1 is normally negative at the
    # receiving end of a lossless branch).
    buses_p = getattr(network.buses_t, "p", None)
    if buses_p is None or getattr(buses_p, "empty", True):
        raise RuntimeError("cannot verify bus energy balance: buses_t.p is unavailable")
    balance = buses_p.copy().reindex(columns=network.buses.index, fill_value=0.0)
    for component in ("lines", "transformers", "links"):
        table = getattr(network, component, None)
        dynamic = getattr(network, f"{component}_t", None)
        if table is None or table.empty or dynamic is None:
            continue
        for bus_column in (column for column in table.columns if column.startswith("bus")):
            port = "p" + bus_column[3:]
            flow = getattr(dynamic, port, None)
            if flow is None or flow.empty:
                continue
            for bus, units in table.groupby(bus_column, dropna=True):
                if not isinstance(bus, str) or bus not in balance.columns:
                    continue
                columns = flow.columns.intersection(units.index)
                if len(columns):
                    balance.loc[:, bus] -= flow.loc[:, columns].sum(axis=1)
    residuals["bus_balance_mw"] = _max_abs(balance)

    for component, dispatch, nominal, upper_pu, lower_pu in (
        ("generators", "p", "p_nom", "p_max_pu", "p_min_pu"),
        ("storage_units", "p_dispatch", "p_nom", "p_max_pu", None),
    ):
        table = getattr(network, component)
        values = getattr(getattr(network, f"{component}_t", None), dispatch, None)
        if table.empty or values is None or values.empty:
            continue
        cap = table.get(f"{nominal}_opt", table[nominal]).reindex(values.columns).astype(float)
        from pypsa.descriptors import get_switchable_as_dense
        component_name = "Generator" if component == "generators" else "StorageUnit"
        upper = get_switchable_as_dense(network, component_name, upper_pu)
        upper = upper.reindex(index=values.index, columns=values.columns).fillna(1.0)
        residuals[f"{component}_upper_mw"] = _max_abs((values - upper.mul(cap, axis=1)).clip(lower=0))
        if lower_pu:
            lower = get_switchable_as_dense(network, component_name, lower_pu)
            lower = lower.reindex(index=values.index, columns=values.columns).fillna(0.0)
            residuals[f"{component}_lower_mw"] = _max_abs((lower.mul(cap, axis=1) - values).clip(lower=0))

    # Storage charging and state of charge use distinct result tables.  They
    # are bounded independently from positive dispatch in PyPSA's formulation.
    storage = getattr(network, "storage_units", None)
    storage_t = getattr(network, "storage_units_t", None)
    if storage is not None and not storage.empty and storage_t is not None:
        cap = storage.get("p_nom_opt", storage.p_nom).astype(float)
        p_store = getattr(storage_t, "p_store", None)
        if p_store is not None and not p_store.empty:
            residuals["storage_charge_upper_mw"] = _max_abs(
                (p_store - cap.reindex(p_store.columns)).clip(lower=0)
            )
        state = getattr(storage_t, "state_of_charge", None)
        if state is not None and not state.empty:
            max_hours = storage.get("max_hours", pd.Series(0.0, index=storage.index))
            energy_cap = cap * max_hours.astype(float)
            residuals["storage_energy_upper_mwh"] = _max_abs(
                (state - energy_cap.reindex(state.columns)).clip(lower=0)
            )
            residuals["storage_energy_lower_mwh"] = _max_abs((-state).clip(lower=0))

    line_p0 = getattr(network.lines_t, "p0", None)
    if line_p0 is not None and not line_p0.empty:
        cap = network.lines.get("s_nom_opt", network.lines.s_nom)
        # Unsolved/imported fixtures can carry the PyPSA default zero-valued
        # s_nom_opt. A solved expandable line has a positive optimum; all
        # other lines are bounded by their nominal capacity.
        cap = cap.where(cap.notna() & (cap > 0), network.lines.s_nom).reindex(line_p0.columns)
        from pypsa.descriptors import get_switchable_as_dense
        s_max_pu = get_switchable_as_dense(network, "Line", "s_max_pu")
        s_max_pu = s_max_pu.reindex(index=line_p0.index, columns=line_p0.columns).fillna(1.0)
        limit = s_max_pu.mul(cap, axis=1)
        residuals["line_upper_mw"] = _max_abs((line_p0.abs() - limit).clip(lower=0))
        line_p1 = getattr(network.lines_t, "p1", None)
        if line_p1 is not None and not line_p1.empty:
            residuals["line_p1_upper_mw"] = _max_abs((line_p1.abs() - limit.reindex(columns=line_p1.columns)).clip(lower=0))
    violations = {key: value for key, value in residuals.items() if value > tolerance}
    if violations:
        logging.getLogger(__name__).warning(
            "Primal residual exceeds reporting threshold %s (MW; storage energy in MWh): %s; continuing with residuals recorded",
            tolerance, violations,
        )
    return residuals
