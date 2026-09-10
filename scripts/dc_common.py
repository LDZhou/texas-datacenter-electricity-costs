"""
dc_common.py
============
Shared definitions and utilities for the DC impact experiments.

Loaded by:
  - run_dc_experiment.py  (the main solver script)
  - run_rep_risk.py       (downstream financial analysis)
  - analyze_results.py    (cross-experiment plots)

Single source of truth for:
  - DC site definitions (multi-loc + all-2030 loader)
  - ERCOT zone assignment
  - Network mode preparation (dispatch / generation / generation_tx)
  - DC load profile generation
  - Cost decomposition + new-capacity extraction

No-storage experiment family:
  - dispatch:       no expansion
  - generation:     generators extendable only
  - generation_tx:  generators and transmission extendable
  - storage remains dispatchable if already present, but is not extendable
"""
import warnings
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")


# =====================================================================
# Constants
# =====================================================================

ERCOT_ZONES = ["HOUSTON", "NORTH", "SOUTH", "WEST"]

# Multi-location DC sites — 6 representative TX 2030 hotspots
# (sourced from TX_datacenters.xlsx clustering analysis)
DC_LOCATIONS = {
    "TOLAR": {
        "bus": "p482210 0", "zone": "WEST",
        "x": -97.71, "y": 32.39,
        "desc": "Tolar / Hood County — mega-scale planned (5 GW)",
        "planned_2030_mw": 5300,
    },
    "ABILENE": {
        "bus": "p484410 0", "zone": "WEST",
        "x": -99.73, "y": 32.45,
        "desc": "Abilene cluster (Taylor/Jones/Nolan) — wind corridor",
        "planned_2030_mw": 5400,
    },
    "CHILDRESS": {
        "bus": "p480750 0", "zone": "WEST",
        "x": -100.20, "y": 34.43,
        "desc": "Childress / Panhandle — remote, weakest grid",
        "planned_2030_mw": 4250,
    },
    "DFW_SOUTH": {
        "bus": "p481390 0", "zone": "NORTH",
        "x": -96.74, "y": 32.38,
        "desc": "DFW South (Red Oak / Lancaster / Corsicana)",
        "planned_2030_mw": 2000,
    },
    "AUSTIN": {
        "bus": "p484530 0", "zone": "SOUTH",
        "x": -97.74, "y": 30.27,
        "desc": "Austin metro (Travis / San Marcos / Rockdale)",
        "planned_2030_mw": 3650,
    },
    "HOUSTON": {
        "bus": "p482010 0", "zone": "HOUSTON",
        "x": -95.37, "y": 29.76,
        "desc": "Houston metro (Harris County) — strongest grid",
        "planned_2030_mw": 1150,
    },
}

DEFAULT_SCALES_MW = [500, 1000, 3000, 5000]
MAX_ALL2030_BUS_DIST_DEG = 1.5

LIKELY_NON_ERCOT_CITIES = {
    "el paso",
    "socorro",
    "culberson county",
}

# =====================================================================
# City coordinates for geocoding TX_datacenters.xlsx addresses
# =====================================================================
# Maps lowercase city substring -> (longitude, latitude).
# load_all2030_dcs scans each address for these city names.

CITY_COORDS = {
    "amarillo": (-101.83, 35.22), "austin": (-97.74, 30.27),
    "houston": (-95.37, 29.76), "dallas": (-96.80, 32.78),
    "san antonio": (-98.49, 29.42), "fort worth": (-97.33, 32.75),
    "el paso": (-106.45, 31.76), "lubbock": (-101.85, 33.58),
    "midland": (-102.08, 31.97), "odessa": (-102.37, 31.85),
    "abilene": (-99.73, 32.45), "childress": (-100.20, 34.43),
    "sweetwater": (-100.41, 32.47), "red oak": (-96.80, 32.52),
    "lancaster": (-96.76, 32.59), "plano": (-96.70, 33.02),
    "richardson": (-96.73, 32.95), "garland": (-96.64, 32.91),
    "irving": (-96.95, 32.81), "mckinney": (-96.62, 33.20),
    "allen": (-96.67, 33.10), "hutto": (-97.55, 30.54),
    "temple": (-97.34, 31.10), "rockdale": (-97.00, 30.66),
    "san marcos": (-97.94, 29.88), "tolar": (-97.71, 32.39),
    "granbury": (-97.79, 32.44), "corsicana": (-96.47, 32.09),
    "lueders": (-99.62, 32.80), "hamlin": (-100.13, 32.88),
    "colorado city": (-100.87, 32.39), "fort stockton": (-102.88, 30.89),
    "abernathy": (-101.84, 33.83), "vernon": (-99.29, 34.15),
    "wichita falls": (-98.49, 33.91), "wink": (-103.16, 31.75),
    "pampa": (-100.96, 35.54), "hockley": (-95.82, 30.03),
    "katy": (-95.82, 29.79), "spring": (-95.42, 30.08),
    "montgomery": (-95.70, 30.39), "taylor": (-97.41, 30.57),
    "lewisville": (-97.00, 33.05), "scurry": (-96.36, 32.52),
    "grand prairie": (-97.02, 32.75), "waco": (-97.15, 31.55),
    "huntsville": (-95.55, 30.72), "brady": (-99.34, 31.13),
    "panhandle": (-101.38, 35.35), "castroville": (-98.88, 29.36),
    "mcallen": (-98.23, 26.20),
    "carrollton": (-96.89,32.98),
    "stratford": (-102.07,36.34),
    "corpus christi": (-97.4,27.8),
    "laredo": (-99.51,27.51),
    "harlingen": (-97.7,26.19),
    "arlington": (-97.11,32.74),
    "the woodlands": (-95.46,30.17),
    "farmers branch": (-96.89,32.93),
    "sugar land": (-95.63,29.62),
    "texarkana": (-94.04,33.43),
    "bryan": (-96.37,30.67),
    "desoto": (-96.86,32.59),
    "roanoke": (-97.23,33.0),
    "georgetown": (-97.68,30.63),
    "silverton": (-101.31,34.47),
    "longview": (-94.74,32.5),
    "tyler": (-95.3,32.35),
    "beaumont": (-94.1,30.08),
    "port arthur": (-93.94,29.88),
    "galveston": (-94.79,29.3),
    "victoria": (-96.99,28.81),
    "new braunfels": (-98.12,29.7),
    "round rock": (-97.68,30.51),
    "pflugerville": (-97.62,30.44),
    "cedar park": (-97.82,30.51),
    "denton": (-97.13,33.22),
    "frisco": (-96.82,33.15),
    "grapevine": (-97.08,32.93),
    "southlake": (-97.14,32.94),
    "flower mound": (-97.1,33.01),
    "coppell": (-96.99,32.95),
    "pearland": (-95.29,29.56),
    "missouri city": (-95.54,29.62),
    "friendswood": (-95.2,29.53),
    "league city": (-95.1,29.5),
    "kyle": (-97.88,29.99),
    "buda": (-97.84,30.08),
    "wolfforth": (-102.01,33.5),
    "seminole": (-102.65,32.72),
    "andrews": (-102.55,32.32),
    "big spring": (-101.48,32.25),
    "snyder": (-100.92,32.72),
    "mcgregor": (-97.4,31.44),
    "killeen": (-97.73,31.12),
    "college station": (-96.33,30.63),
    "liberty hill": (-97.92,30.67),
    "leander": (-97.85,30.58),
    "mabank": (-96.1,32.37),
    "athens": (-95.85,32.2),
    "terrell": (-96.28,32.74),
    "ennis": (-96.63,32.33),
    "bastrop": (-97.31,30.11),
    "giddings": (-96.94,30.18),
    "elgin": (-97.37,30.35),
    "lockhart": (-97.67,29.88),
    "seguin": (-97.96,29.57),
    "luling": (-97.65,29.68),
    "mont belvieu": (-94.89,29.84),
    "baytown": (-94.98,29.74),
    "deer park": (-95.12,29.7),
    "la porte": (-95.02,29.67),
    "rosenberg": (-95.81,29.56),
    "richmond": (-95.76,29.58),
    "cypress": (-95.7,29.97),
    "tomball": (-95.62,30.1),
    "conroe": (-95.46,30.31),
    "palestine": (-95.63,31.76),
    "nacogdoches": (-94.66,31.6),
    "lufkin": (-94.73,31.34),
    "paris": (-95.55,33.66),
    "sherman": (-96.61,33.64),
    "denison": (-96.54,33.76),
    "gainesville": (-97.13,33.63),
    "decatur": (-97.59,33.23),
    "azle": (-97.54,32.9),
    "burleson": (-97.32,32.54),
    "mansfield": (-97.14,32.56),
    "cleburne": (-97.39,32.34),
    "weatherford": (-97.8,32.76),
    "mineral wells": (-98.11,32.81),
    "stephenville": (-98.2,32.22),
    "brownwood": (-98.99,31.71),
    "culberson county": (-104.52,31.43),
    "ector county": (-102.54,31.87),
    "moore county": (-101.89,35.84),
    "cameron county": (-97.48,26.13),
    "pecos": (-103.49,31.42),
    "afton": (-100.79,33.77),
    "garden city": (-101.48,31.86),
    "cotulla": (-99.23,28.44),
    "seymour": (-99.26,33.59),
    "medina": (-99.24,29.8),
    "laguna park": (-97.37,31.86),
    "midlothian": (-96.99,32.48),
    "socorro": (-106.31,31.66),
}


# No-storage expansion family:
#   generation:    generators only
#   generation_tx: generators + transmission
EXPANDABLE_CARRIERS = {
    "generators": ["OCGT", "solar", "onwind"],
}

# Full expansion family:
#   full_tx / expansion_tx: generators + storage + transmission
FULL_EXPANDABLE_CARRIERS = {
    "generators": ["OCGT", "solar", "onwind"],
    "storage_units": ["4hr_battery_storage"],
}

# =====================================================================
# ERCOT zone assignment
# =====================================================================
def _bus_demand_weights_lmp(network, bus_subset) -> "pd.Series":
    """
    Demand weights for LMP averaging within a bus subset. Tries:
      1. buses.Pd column (pypsa-usa convention -- peak/case-file demand)
      2. avg loads_t.p_set per bus (excludes DC_* loads)
      3. static loads.p_set per bus (excludes DC_* loads)
      4. equal weights (with WARN)
 
    Retains the original historical-validation demand weighting so the validation
    script and the experiment metrics use IDENTICAL aggregation logic.
    """
    import pandas as pd
 
    buses = network.buses
 
    # 1) Pd
    if "Pd" in buses.columns:
        w = pd.to_numeric(buses.loc[bus_subset, "Pd"], errors="coerce")
        w = w.clip(lower=0).fillna(0)
        if w.sum() > 0:
            return w / w.sum()
 
    # 2) loads_t.p_set mean (exclude DC loads)
    loads = network.loads
    if not loads.empty:
        non_dc = ~loads.index.astype(str).str.startswith("DC_")
        loads = loads[non_dc]
 
        if (hasattr(network, "loads_t") and "p_set" in network.loads_t
                and not network.loads_t.p_set.empty):
            p_set = network.loads_t.p_set
            common = loads.index.intersection(p_set.columns)
            if len(common):
                load_mean = p_set[common].mean()
                bus_load = pd.Series(0.0, index=bus_subset)
                for lid in common:
                    b = loads.loc[lid, "bus"]
                    if b in bus_load.index:
                        bus_load[b] += float(load_mean[lid])
                if bus_load.sum() > 0:
                    return bus_load / bus_load.sum()
 
        # 3) static p_set
        if "p_set" in loads.columns:
            bus_load = pd.Series(0.0, index=bus_subset)
            for lid, row in loads.iterrows():
                b = row["bus"]
                if b in bus_load.index:
                    bus_load[b] += float(row.get("p_set", 0) or 0)
            if bus_load.sum() > 0:
                return bus_load / bus_load.sum()
 
    # 4) equal weights
    print(f"    [WARN] no demand weights for {len(bus_subset)} buses; "
          f"falling back to equal weights")
    return pd.Series(1.0 / max(len(bus_subset), 1), index=bus_subset)
 

def assign_ercot_zone(x: float, y: float) -> str:
    """
    Assign a bus to an ERCOT zone based on its (x, y) longitude/latitude.
    Simple geographic heuristic, matching the previous experiments.
    """
    if x < -100.0:
        return "WEST"
    elif y >= 32.0:
        return "NORTH"
    elif x >= -96.5 and y < 30.5:
        return "HOUSTON"
    else:
        return "SOUTH"


def get_ercot_buses(network: pypsa.Network) -> pd.DataFrame:
    """Subset buses to ERCOT only (using nerc_reg if available)."""
    buses = network.buses
    if "nerc_reg" in buses.columns:
        mask = buses["nerc_reg"].astype(str).str.upper().str.contains("ERCOT", na=False)
        ercot = buses[mask]
        if len(ercot) > 0:
            return ercot
    return buses


# =====================================================================
# Bus resolution
# =====================================================================

def resolve_bus(network: pypsa.Network, target_bus: str,
                target_x: Optional[float] = None,
                target_y: Optional[float] = None,
                label: str = "") -> str:
    """
    Return target_bus if it exists in the network, otherwise find the
    nearest bus by (x, y) coordinates.
    """
    if target_bus in network.buses.index:
        return target_bus
    if target_x is None or target_y is None:
        raise ValueError(
            f"Bus '{target_bus}' not in network and no fallback coords given")
    print(f"  [WARN] bus '{target_bus}' for {label} not found, "
          f"searching nearest to ({target_x}, {target_y})")
    buses = network.buses
    dist = ((buses["x"] - target_x) ** 2 + (buses["y"] - target_y) ** 2) ** 0.5
    nearest = dist.idxmin()
    print(f"    → using '{nearest}' (dist={dist[nearest]:.3f}°)")
    return nearest


# =====================================================================
# DC load profile
# =====================================================================

def create_dc_load_profile(snapshots, capacity_mw: float,
                           capacity_factor: float = 0.90,
                           variability: float = 0.03,
                           seed: int = 42) -> pd.Series:
    """
    Generate a stable DC load time series.

    Hyperscale DCs run flat — small seasonal swing, small noise.
    Bounded between 80% and 100% of nameplate.
    """
    rng = np.random.RandomState(seed)
    if isinstance(snapshots, pd.MultiIndex):
        idx = pd.DatetimeIndex(snapshots.get_level_values("timestep"))
    else:
        idx = pd.DatetimeIndex(snapshots)

    base = capacity_mw * capacity_factor
    seasonal = 0.05 * capacity_mw * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365)
    noise = variability * capacity_mw * rng.randn(len(idx))
    load = np.clip(base + seasonal + noise,
                   capacity_mw * 0.80, capacity_mw)
    return pd.Series(load, index=snapshots, name="datacenter")


def add_dc_load(network: pypsa.Network, bus: str, capacity_mw: float,
                capacity_factor: float = 0.90,
                name: str = "DataCenter") -> pypsa.Network:
    """Attach one DC load to the network at a given bus."""
    if bus not in network.buses.index:
        raise ValueError(f"bus '{bus}' not in network")
    profile = create_dc_load_profile(network.snapshots, capacity_mw, capacity_factor)
    network.add("Load", name, bus=bus, p_set=profile, carrier="datacenter")
    return network


def read_datacenter_table(path: Path) -> pd.DataFrame:
    """Read the public CSV or the legacy Excel data-center project table."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"unsupported data-center table format: {path.suffix}")


def add_all2030_dcs(network: pypsa.Network,
                    dc_data_path: Path,
                    capacity_factor: float = 0.90) -> dict:
    """
    Read TX_datacenters.xlsx and attach every DC load to its nearest bus.

    Columns expected: full_address, State, current_mw, construction_mw, planned_mw
    Each row's total = current + construction + planned (2030 total).

    Address -> (lon, lat) via city-substring matching against CITY_COORDS.
    Rows whose city cannot be matched are dropped with a warning.

    Multiple DCs that map to the same bus are aggregated into a single
    Load component to keep the network compact.

    Returns {load_name: {bus, capacity_mw, ...}} for later CSV export.
    """
    df = read_datacenter_table(dc_data_path)

    for col in ["current_mw", "construction_mw", "planned_mw"]:
        if col not in df.columns:
            df[col] = 0.0
    df["total_mw"] = df[["current_mw", "construction_mw", "planned_mw"]].fillna(0).sum(axis=1)
    df = df[df["total_mw"] > 0].copy()
    n_raw = len(df)

    # Geocode via city substring matching
    def _geocode(address):
        addr = str(address).lower()
        for city, (lon, lat) in CITY_COORDS.items():
            if city in addr:
                return city, lon, lat
        return None, None, None

    cities, lons, lats = [], [], []
    for addr in df["full_address"]:
        city, lon, lat = _geocode(addr)
        cities.append(city)
        lons.append(lon)
        lats.append(lat)

    df["matched_city"] = cities
    df["lon"] = lons
    df["lat"] = lats

    # Drop clearly non-ERCOT projects before nearest-bus mapping.
    non_ercot = df["matched_city"].isin(LIKELY_NON_ERCOT_CITIES)
    n_non_ercot_city = int(non_ercot.sum())
    if n_non_ercot_city:
        dropped_mw = float(df.loc[non_ercot, "total_mw"].sum())
        print(
            f"  [WARN] dropped {n_non_ercot_city} all2030 rows "
            f"matched to likely non-ERCOT cities ({dropped_mw:.0f} MW): "
            f"{sorted(df.loc[non_ercot, 'matched_city'].dropna().unique())}"
        )
    df = df[~non_ercot].copy()

    n_before = len(df)
    df = df.dropna(subset=["lon", "lat"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [WARN] dropped {n_dropped} rows (city not in CITY_COORDS)")

    # Map to nearest bus, then drop rows mapped to non-ERCOT buses.
    # This avoids forcing non-ERCOT Texas data-center sites into ERCOT.
    buses = network.buses
    ercot_bus_index = set(get_ercot_buses(network).index)

    bus_ids, dists, mapped_to_ercot = [], [], []
    for _, row in df.iterrows():
        dist = ((buses["x"] - row["lon"]) ** 2 + (buses["y"] - row["lat"]) ** 2) ** 0.5
        nearest_bus = dist.idxmin()
        nearest_dist = float(dist.min())

        bus_ids.append(nearest_bus)
        dists.append(nearest_dist)
        mapped_to_ercot.append(nearest_bus in ercot_bus_index)

    df["bus"] = bus_ids
    df["bus_dist_deg"] = dists
    df["mapped_to_ercot"] = mapped_to_ercot

    n_non_ercot = int((~df["mapped_to_ercot"]).sum())
    if n_non_ercot:
        dropped_mw = float(df.loc[~df["mapped_to_ercot"], "total_mw"].sum())
        print(
            f"  [WARN] dropped {n_non_ercot} all2030 rows "
            f"mapped to non-ERCOT buses ({dropped_mw:.0f} MW)"
        )

    df = df[df["mapped_to_ercot"]].copy()

    too_far = df["bus_dist_deg"] > MAX_ALL2030_BUS_DIST_DEG
    n_too_far = int(too_far.sum())
    if n_too_far:
        dropped_mw = float(df.loc[too_far, "total_mw"].sum())
        print(
            f"  [WARN] dropped {n_too_far} all2030 rows "
            f"farther than {MAX_ALL2030_BUS_DIST_DEG:.1f} degrees "
            f"from nearest bus ({dropped_mw:.0f} MW)"
        )

    df = df[~too_far].copy()

    # Aggregate per bus
    per_bus = df.groupby("bus").agg(
        total_mw=("total_mw", "sum"),
        n_sites=("total_mw", "count"),
    ).reset_index()

    # Attach one Load per bus
    mapping = {}
    total_mw = 0.0
    for _, row in per_bus.iterrows():
        bus = row["bus"]
        mw = float(row["total_mw"])
        load_name = f"DC_{bus}"
        # Deterministic seed per bus so same bus gives same profile across runs
        seed = int.from_bytes(hashlib.sha256(bus.encode()).digest()[:8], 'big') % 10000
        profile = create_dc_load_profile(network.snapshots, mw,
                                         capacity_factor, seed=seed)
        network.add("Load", load_name, bus=bus, p_set=profile, carrier="datacenter")
        zone = assign_ercot_zone(
            network.buses.loc[bus, "x"], network.buses.loc[bus, "y"])
        mapping[load_name] = {
            "bus":         bus,
            "capacity_mw": round(mw, 1),
            "n_sites":     int(row["n_sites"]),
            "zone":        zone,
        }
        total_mw += mw

    print(f"  {n_raw} rows with MW > 0  →  {len(df)} geocoded  "
          f"→  {len(per_bus)} unique buses  ({total_mw:.0f} MW total)")
    by_zone = {}
    for info in mapping.values():
        by_zone[info["zone"]] = by_zone.get(info["zone"], 0) + info["capacity_mw"]
    for z in ERCOT_ZONES:
        if z in by_zone:
            print(f"    {z:8s}: {by_zone[z]:7.0f} MW")

    return mapping


# =====================================================================
# Distributed DC load — spread across real xlsx sites near a location
# =====================================================================

DISTRIBUTED_RADIUS_DEG = 1.0  # ~100 km

def add_distributed_dc(network, location: str,
                        dc_data_path, capacity_factor: float = 0.90,
                        scale_mw: float = None) -> dict:
    """
    Add DC load distributed across real xlsx sites near `location`.

    By default uses actual planned capacity from xlsx (no scaling).
    If scale_mw is provided, scales proportionally to that total.

    1. Read TX_datacenters.xlsx, geocode via city matching
    2. Filter to sites within ~100km of location
    3. Map each to nearest bus, aggregate per bus
    4. Add one Load per bus (at actual or scaled MW)

    Returns {load_name: {bus, capacity_mw, n_sites, zone}} for CSV export.
    """
    import pandas as pd

    # Location center coords
    loc_info = DC_LOCATIONS.get(location)
    if loc_info is None:
        raise ValueError(f"Unknown location: {location}")
    bus_id = loc_info["bus"]
    if bus_id in network.buses.index:
        cx, cy = network.buses.loc[bus_id, ["x", "y"]]
    else:
        target = CITY_COORDS.get(location.lower())
        if target is None:
            raise ValueError(f"No coords for {location}")
        cx, cy = target
        dist = ((network.buses.x - cx)**2 + (network.buses.y - cy)**2)**0.5
        bus_id = dist.idxmin()
        cx, cy = network.buses.loc[bus_id, ["x", "y"]]

    # Read xlsx
    df = read_datacenter_table(dc_data_path)
    for col in ["current_mw", "construction_mw", "planned_mw"]:
        if col not in df.columns:
            df[col] = 0.0
    df["total_mw"] = df[["current_mw", "construction_mw", "planned_mw"]].fillna(0).sum(axis=1)
    df = df[df["total_mw"] > 0].copy()

    # Geocode
    def _geocode(address):
        addr = str(address).lower()
        for city, (lon, lat) in CITY_COORDS.items():
            if city in addr:
                return lon, lat
        return None, None

    lons, lats = [], []
    for addr in df["full_address"]:
        lon, lat = _geocode(addr)
        lons.append(lon)
        lats.append(lat)
    df["lon"] = lons
    df["lat"] = lats
    df = df.dropna(subset=["lon", "lat"])

    # Filter within radius
    df["dist_deg"] = ((df["lon"] - float(cx))**2 + (df["lat"] - float(cy))**2)**0.5
    nearby = df[df["dist_deg"] < DISTRIBUTED_RADIUS_DEG].copy()

    if nearby.empty:
        print(f"  [WARN] no xlsx sites within {DISTRIBUTED_RADIUS_DEG} deg of {location}")
        fallback_mw = scale_mw if scale_mw else 500
        return add_dc_load(network, bus_id, fallback_mw, capacity_factor)

    # Map to nearest bus
    bus_ids = []
    for _, row in nearby.iterrows():
        d = ((network.buses.x - row["lon"])**2 + (network.buses.y - row["lat"])**2)**0.5
        bus_ids.append(d.idxmin())
    nearby["bus"] = bus_ids

    # Aggregate per bus
    per_bus = nearby.groupby("bus").agg(
        raw_mw=("total_mw", "sum"),
        n_sites=("total_mw", "count"),
    ).reset_index()

    # Scale if requested, otherwise use raw
    total_raw = per_bus["raw_mw"].sum()
    if scale_mw is not None:
        ratio = scale_mw / total_raw
        per_bus["final_mw"] = per_bus["raw_mw"] * ratio
        scale_label = f"ratio={ratio:.2f}x"
    else:
        per_bus["final_mw"] = per_bus["raw_mw"]
        ratio = 1.0
        scale_label = "actual planned"

    # Add loads
    mapping = {}
    total_added = 0.0
    for _, row in per_bus.iterrows():
        bus = row["bus"]
        mw = float(row["final_mw"])
        if mw < 0.1:
            continue
        load_name = f"DC_{location}_{bus}"
        seed = int.from_bytes(hashlib.sha256((bus + location).encode()).digest()[:8], 'big') % 10000
        profile = create_dc_load_profile(network.snapshots, mw,
                                          capacity_factor, seed=seed)
        network.add("Load", load_name, bus=bus, p_set=profile, carrier="datacenter")
        zone = assign_ercot_zone(
            network.buses.loc[bus, "x"], network.buses.loc[bus, "y"])
        mapping[load_name] = {
            "bus":         bus,
            "capacity_mw": round(mw, 1),
            "n_sites":     int(row["n_sites"]),
            "zone":        zone,
        }
        total_added += mw

    print(f"  distributed {location}: {len(nearby)} sites -> {len(per_bus)} buses "
          f"({scale_label}, total={total_added:.0f} MW)")
    return mapping



# =====================================================================
# Mode preparation — controls what's extendable
# =====================================================================

def prepare_for_mode(network: pypsa.Network, mode: str) -> dict:
    """
    Configure a starting network for one experiment mode.

    mode = "dispatch":
        Lock everything. No expansion.

    mode = "generation":
        Allow generator expansion only. Storage and transmission remain locked.

    mode = "generation_storage":
        Allow generator + storage expansion. Transmission remains locked.

    mode = "generation_tx":
        Allow generator + transmission expansion. Storage remains locked.

    mode = "transmission":
        Allow transmission expansion only. Generation and storage remain locked.

    mode = "full_tx" or "expansion_tx":
        Allow generator + storage + transmission expansion.

    Returns dict with bookkeeping for later capacity-delta extraction:
        {component: pd.Series of base capacities indexed by name}
    """
    valid_modes = (
        "dispatch",
        "generation",
        "generation_storage",
        "generation_tx",
        "transmission",
        "full_tx",
        "expansion_tx",
    )
    if mode not in valid_modes:
        raise ValueError(f"unknown mode: {mode}")

    base_caps = {}

    if mode in ("generation", "generation_tx"):
        expandable_carriers = EXPANDABLE_CARRIERS
    elif mode in ("generation_storage", "full_tx", "expansion_tx"):
        expandable_carriers = FULL_EXPANDABLE_CARRIERS
    else:
        expandable_carriers = {}

    # ── Generators / storage_units / stores / links ──
    for comp_name, nom, ext in [
        ("generators",    "p_nom", "p_nom_extendable"),
        ("storage_units", "p_nom", "p_nom_extendable"),
        ("stores",        "e_nom", "e_nom_extendable"),
        ("links",         "p_nom", "p_nom_extendable"),
    ]:
        comp = getattr(network, comp_name, None)
        if comp is None or comp.empty:
            continue
        if ext not in comp.columns:
            continue

        base_caps[comp_name] = comp[nom].copy()

        # Default: lock everything
        comp[ext] = False

        min_col = f"{nom}_min"
        if min_col not in comp.columns:
            comp[min_col] = 0.0

        if comp_name in expandable_carriers:
            carriers = expandable_carriers[comp_name]
            unlock_mask = comp["carrier"].isin(carriers)

            if unlock_mask.any():
                comp.loc[unlock_mask, ext] = True
                comp.loc[unlock_mask, min_col] = comp.loc[unlock_mask, nom]

                # Keep capital cost for unlocked units, zero out locked ones.
                if "capital_cost" in comp.columns:
                    comp.loc[~unlock_mask, "capital_cost"] = 0.0
            else:
                if "capital_cost" in comp.columns:
                    comp["capital_cost"] = 0.0
        else:
            # Not expandable in this mode
            if "capital_cost" in comp.columns:
                comp["capital_cost"] = 0.0

    # ── Lines ──
    lines = network.lines
    base_caps["lines"] = lines["s_nom"].copy()

    if "s_nom_extendable" not in lines.columns:
        lines["s_nom_extendable"] = False
    if "s_nom_min" not in lines.columns:
        lines["s_nom_min"] = 0.0

    if mode in ("generation_tx", "transmission", "full_tx", "expansion_tx"):
        lines["s_nom_extendable"] = True
        lines["s_nom_min"] = lines["s_nom"].values

        n_ext = int(lines["s_nom_extendable"].sum())
        avg_max = float(lines["s_nom_max"].mean()) if "s_nom_max" in lines.columns else 0
        avg_capex = float(lines["capital_cost"].mean()) if "capital_cost" in lines.columns else 0

        if mode == "transmission":
            label = "TX-only"
        elif mode in ("full_tx", "expansion_tx"):
            label = "Gen+storage+TX"
        else:
            label = "Gen+TX"

        print(
            f"    [{label}] {n_ext} lines extendable, "
            f"avg s_nom_max={avg_max:.0f} MW, "
            f"avg capex=${avg_capex:,.0f}/MW/yr"
        )
    else:
        lines["s_nom_extendable"] = False
        if "capital_cost" in lines.columns:
            lines["capital_cost"] = 0.0

    return base_caps


# =====================================================================
# Cost decomposition
# =====================================================================

def decompose_system_cost(network: pypsa.Network) -> dict:
    """
    Extract system cost components from a solved network.

    NOTE: PyPSA's `network.objective` includes terms (status variables,
    ramp costs, storage standing losses, etc.) that we cannot fully
    reconstruct from `generators_t.p * marginal_cost` alone. Therefore:

      - `objective` is the authoritative total system cost
      - `dispatch_cost` is our best-effort reconstruction; useful for
        splitting by component but should NOT be expected to equal
        objective exactly
      - `capex_*` is exact (we control extendable units explicitly)

    Cross-experiment comparisons should use `delta_objective` for the
    headline number and `delta_capex_*` for investment breakdown.
    """
    result = {
        "objective":          float(getattr(network, "objective", float("nan"))),
        "dispatch_cost":      0.0,
        "dispatch_gen":       0.0,
        "dispatch_storage":   0.0,
        "load_shedding_cost": 0.0,
        "capex_total":        0.0,
        "capex_gen":          0.0,
        "capex_storage":      0.0,
        "capex_lines":        0.0,
    }

    # snapshot weights
    sw = network.snapshot_weightings
    if not sw.empty and "objective" in sw.columns:
        weights = sw["objective"]
    elif not sw.empty and "generators" in sw.columns:
        weights = sw["generators"]
    else:
        weights = pd.Series(1.0, index=network.snapshots)

    # ── Generator opex (split out load shedding) ──
    gens = network.generators
    if not gens.empty and "p" in network.generators_t:
        p = network.generators_t.p
        mc = gens["marginal_cost"].reindex(p.columns).fillna(0)
        is_shed = mc >= 1000  # load_shedding generators
        all_cost = (p * mc).multiply(weights, axis=0).sum().sum()
        if is_shed.any():
            shed_cols = is_shed[is_shed].index
            shed_cost = (p[shed_cols] * mc[shed_cols]).multiply(
                weights, axis=0).sum().sum()
        else:
            shed_cost = 0.0
        result["dispatch_gen"] = float(all_cost - shed_cost)
        result["load_shedding_cost"] = float(shed_cost)

    # ── Storage opex ──
    su = network.storage_units
    if not su.empty and "p_dispatch" in network.storage_units_t:
        pd_ = network.storage_units_t.p_dispatch
        mc_su = su["marginal_cost"].reindex(pd_.columns).fillna(0)
        result["dispatch_storage"] = float(
            (pd_ * mc_su).multiply(weights, axis=0).sum().sum())

    result["dispatch_cost"] = (result["dispatch_gen"]
                                + result["dispatch_storage"]
                                + result["load_shedding_cost"])

    # ── Capex on newly built gen/storage capacity ──
    for comp_name, nom_opt in [
        ("generators",    "p_nom_opt"),
        ("storage_units", "p_nom_opt"),
    ]:
        comp = getattr(network, comp_name)
        if comp.empty or "p_nom_extendable" not in comp.columns:
            continue
        ext = comp[comp["p_nom_extendable"]]
        if ext.empty:
            continue
        cc = ext["capital_cost"].fillna(0)
        opt = ext[nom_opt].fillna(0) if nom_opt in ext.columns else ext["p_nom"]
        p_min = ext["p_nom_min"].fillna(0) if "p_nom_min" in ext.columns else 0
        new_cap = (opt - p_min).clip(lower=0)
        capex = float((cc * new_cap).sum())
        if comp_name == "generators":
            result["capex_gen"] = capex
        else:
            result["capex_storage"] = capex

    # ── Capex on newly built transmission ──
    lines = network.lines
    if "s_nom_extendable" in lines.columns:
        ext_lines = lines[lines["s_nom_extendable"]]
        if not ext_lines.empty:
            cc = ext_lines["capital_cost"].fillna(0)
            s_opt = (ext_lines["s_nom_opt"].fillna(0)
                     if "s_nom_opt" in ext_lines.columns
                     else ext_lines["s_nom"])
            s_min = (ext_lines["s_nom_min"].fillna(0)
                     if "s_nom_min" in ext_lines.columns else 0)
            new_cap = (s_opt - s_min).clip(lower=0)
            result["capex_lines"] = float((cc * new_cap).sum())

    result["capex_total"] = (result["capex_gen"]
                             + result["capex_storage"]
                             + result["capex_lines"])

    return result


# =====================================================================
# New capacity extraction (for expansion / expansion_tx modes)
# =====================================================================

def extract_new_capacity(network: pypsa.Network, base_caps: dict) -> pd.DataFrame:
    """
    Per-unit listing of every component whose capacity grew vs base.

    Returns DataFrame with columns:
        component, name, bus, carrier, zone, bus_x, bus_y,
        base_mw, new_mw, added_mw,
        capex_per_mw_yr, annual_investment
    """
    rows = []

    for comp_name in ["generators", "storage_units"]:
        comp = getattr(network, comp_name)
        if comp.empty or comp_name not in base_caps:
            continue
        if "p_nom_extendable" not in comp.columns:
            continue
        ext_mask = comp["p_nom_extendable"]
        if not ext_mask.any():
            continue
        opt_col = "p_nom_opt" if "p_nom_opt" in comp.columns else "p_nom"

        for idx in comp[ext_mask].index:
            base_val = float(base_caps[comp_name].get(idx, 0.0) or 0.0)
            new_val_raw = comp.loc[idx, opt_col]
            if pd.isna(new_val_raw):
                continue
            new_val = float(new_val_raw)
            delta = new_val - base_val
            if abs(delta) < 0.1:
                continue
            capex = float(comp.loc[idx, "capital_cost"]) if "capital_cost" in comp.columns else 0.0
            bus = comp.loc[idx, "bus"]
            zone = "UNKNOWN"
            bus_x = np.nan
            bus_y = np.nan
            if bus in network.buses.index:
                bus_x = float(network.buses.loc[bus, "x"])
                bus_y = float(network.buses.loc[bus, "y"])
                zone = assign_ercot_zone(
                    bus_x, bus_y)
            rows.append({
                "component":         comp_name,
                "name":              idx,
                "bus":               bus,
                "carrier":           comp.loc[idx, "carrier"],
                "zone":              zone,
                "bus_x":             bus_x,
                "bus_y":             bus_y,
                "base_mw":           round(base_val, 1),
                "new_mw":            round(new_val, 1),
                "added_mw":          round(delta, 1),
                "capex_per_mw_yr":   round(capex, 0),
                "annual_investment": round(delta * capex, 0),
            })

    # Lines
    lines = network.lines
    if "s_nom_extendable" in lines.columns and "lines" in base_caps:
        ext_lines = lines[lines["s_nom_extendable"]]
        for idx in ext_lines.index:
            base_val = float(base_caps["lines"].get(idx, 0.0) or 0.0)
            new_val_raw = (lines.loc[idx, "s_nom_opt"]
                           if "s_nom_opt" in lines.columns
                           else lines.loc[idx, "s_nom"])
            if pd.isna(new_val_raw):
                continue
            new_val = float(new_val_raw)
            delta = new_val - base_val
            if abs(delta) < 0.1:
                continue
            capex = float(lines.loc[idx, "capital_cost"]) if "capital_cost" in lines.columns else 0.0
            bus0 = lines.loc[idx, "bus0"]
            bus1 = lines.loc[idx, "bus1"]
            zone = "UNKNOWN"
            if bus0 in network.buses.index:
                zone = assign_ercot_zone(
                    network.buses.loc[bus0, "x"], network.buses.loc[bus0, "y"])
            rows.append({
                "component":         "lines",
                "name":              idx,
                "bus":               f"{bus0} -> {bus1}",
                "carrier":           "transmission",
                "zone":              zone,
                "base_mw":           round(base_val, 1),
                "new_mw":            round(new_val, 1),
                "added_mw":          round(delta, 1),
                "capex_per_mw_yr":   round(capex, 0),
                "annual_investment": round(delta * capex, 0),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "component", "name", "bus", "carrier", "zone",
            "bus_x", "bus_y",
            "base_mw", "new_mw", "added_mw",
            "capex_per_mw_yr", "annual_investment"])
    return pd.DataFrame(rows)


# =====================================================================
# LMP / congestion utilities
# =====================================================================

def flatten_index(df):
    """If df has MultiIndex snapshots, flatten to DatetimeIndex."""
    if isinstance(df.index, pd.MultiIndex):
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index.get_level_values("timestep"))
    return df


def compute_lmp_summary(network, ercot_buses):
    """
    Demand-weighted zone LMP and demand-weighted system LMP percentiles.
 
    Returns
    -------
    summary : dict
        sys_mean_lmp, sys_p95_lmp, sys_p99_lmp -- mean / 95 / 99 percentile
            of the demand-weighted SYSTEM LMP time series across all ERCOT
            buses (matches ERCOT system-wide hub price semantics).
        max_bus_mean_lmp, max_bus_id -- annual mean per bus (unweighted),
            then take max -- identifies the worst-LMP bus.
        lmp_<ZONE> -- demand-weighted mean of the zone LMP time series.
    lmp : DataFrame
        Per-bus LMP (unchanged from before, for downstream callers).
    """
    import numpy as np
    import pandas as pd
 
    bus_cols = ercot_buses.index.intersection(
        network.buses_t.marginal_price.columns)
    lmp = flatten_index(network.buses_t.marginal_price[bus_cols])
 
    # --- system-wide demand-weighted LMP time series ---
    sys_w = _bus_demand_weights_lmp(network, bus_cols).reindex(bus_cols).fillna(0)
    if sys_w.sum() > 0:
        sys_w = sys_w / sys_w.sum()
        sys_lmp_t = (lmp * sys_w).sum(axis=1)
    else:
        sys_lmp_t = lmp.mean(axis=1)
    sys_vals = sys_lmp_t.values
 
    # bus-level mean (unweighted) for max-bus diagnostic
    sys_bus_mean = lmp.mean(axis=0)
 
    summary = {
        "sys_mean_lmp":     float(np.nanmean(sys_vals)),
        "sys_p95_lmp":      float(np.nanpercentile(sys_vals, 95)),
        "sys_p99_lmp":      float(np.nanpercentile(sys_vals, 99)),
        "max_bus_mean_lmp": float(sys_bus_mean.max()),
        "max_bus_id":       str(sys_bus_mean.idxmax()),
    }
 
    # --- per-zone demand-weighted LMP ---
    zones_series = ercot_buses.loc[bus_cols].apply(
        lambda r: assign_ercot_zone(r["x"], r["y"]), axis=1)
    for z in ERCOT_ZONES:
        zb = zones_series[zones_series == z].index
        if len(zb) == 0:
            summary[f"lmp_{z}"] = float("nan")
            continue
        zw = _bus_demand_weights_lmp(network, zb).reindex(zb).fillna(0)
        if zw.sum() > 0:
            zw = zw / zw.sum()
            summary[f"lmp_{z}"] = float((lmp[zb] * zw).sum(axis=1).mean())
        else:
            summary[f"lmp_{z}"] = float(lmp[zb].mean(axis=1).mean())
 
    return summary, lmp


def compute_congestion(network: pypsa.Network,
                       ercot_buses: pd.DataFrame,
                       threshold: float = 0.85) -> int:
    """Count line-hours where loading exceeds threshold (default 85%)."""
    lines = network.lines
    ercot_idx = set(ercot_buses.index)
    mask = lines["bus0"].isin(ercot_idx) & lines["bus1"].isin(ercot_idx)
    sub = lines[mask]
    if sub.empty or "p0" not in network.lines_t:
        return 0
    cols = sub.index.intersection(network.lines_t.p0.columns)
    if not len(cols):
        return 0
    s_nom = sub.loc[cols, "s_nom_opt"] if "s_nom_opt" in sub.columns else sub.loc[cols, "s_nom"]
    s_nom = s_nom.clip(lower=1)
    loading = network.lines_t.p0[cols].abs().div(s_nom, axis=1)
    return int((loading > threshold).sum().sum())
