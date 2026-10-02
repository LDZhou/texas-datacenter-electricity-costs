"""Freeze historical optimum and persist the common exogenous DC curves once."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pypsa

from .dc_common import DC_LOCATIONS, add_all2030_dcs, add_dc_load, resolve_bus
from .model_contract import file_sha256, validate_frozen_starting_network, write_dc_profiles
from .prepare_network import prepare_starting_network
from .revision_matrix import LOCATIONS, SCALES


def persist_profiles(network, root, year, dc_data, horizon='seasonal', siting=False):
    dest = Path(root) / 'profiles' / horizon / str(year)
    dest.mkdir(parents=True, exist_ok=True)
    net = network.copy()
    add_all2030_dcs(net, dc_data)
    write_dc_profiles(net, dest / 'all2030.csv')
    if siting:
        for location in LOCATIONS:
            info = DC_LOCATIONS[location]
            bus = resolve_bus(network, info['bus'], info['x'], info['y'], label=location)
            for scale in SCALES:
                net = network.copy()
                add_dc_load(net, bus, scale, name=f'DC_{location}')
                write_dc_profiles(net, dest / f'{location}_{scale}.csv')
    inventory = {p.name: file_sha256(p) for p in sorted(dest.glob('*.csv'))}
    (dest / 'manifest.json').write_text(json.dumps(inventory, indent=2) + '\n')


def add_siting_profiles(run_root, year, dc_data):
    """Add the per-location seasonal curves next to an already frozen base.

    The frozen starting network is read, never rewritten. The curves are built
    in a scratch directory first; the regenerated all2030 curve must be
    byte-identical to the published one before anything is moved into place.
    """
    start = Path(run_root) / f'starting_networks/start_{year}.nc'
    dest = Path(run_root) / 'profiles' / 'seasonal' / str(year)
    published = json.loads((dest / 'manifest.json').read_text())
    existing = [name for name in published if name != 'all2030.csv']
    if existing:
        raise FileExistsError(f'siting profiles already published for {year}: {existing}')
    network = pypsa.Network(start)
    validate_frozen_starting_network(network)
    with tempfile.TemporaryDirectory(dir=dest.parent) as scratch:
        persist_profiles(network, scratch, year, dc_data, siting=True)
        fresh = Path(scratch) / 'profiles' / 'seasonal' / str(year)
        regenerated = json.loads((fresh / 'manifest.json').read_text())
        if regenerated['all2030.csv'] != published['all2030.csv']:
            raise ValueError(f'regenerated all2030 curve differs from the published one for {year}')
        for name in sorted(regenerated):
            if name != 'all2030.csv':
                shutil.move(str(fresh / name), str(dest / name))
    inventory = {p.name: file_sha256(p) for p in sorted(dest.glob('*.csv'))}
    if inventory['all2030.csv'] != published['all2030.csv']:
        raise ValueError('published all2030 curve changed underneath the siting profiles')
    (dest / 'manifest.json').write_text(json.dumps(inventory, indent=2) + '\n')
    record = {'network': str(start), 'network_sha256': file_sha256(start),
              'profiles': {k: v for k, v in inventory.items() if k != 'all2030.csv'}}
    (Path(run_root) / f'manifests/siting-profiles-seasonal-{year}.json').write_text(json.dumps(record, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--historical-root', type=Path, required=True)
    p.add_argument('--dc-data', type=Path, required=True,
                   help='Public CSV or workbook containing the mapped data-center projects')
    p.add_argument('--siting-profiles-only', action='store_true',
                   help='Only add per-location seasonal DC curves for a year whose base is already frozen')
    args = p.parse_args()
    if args.siting_profiles_only:
        add_siting_profiles(args.run_root, args.year, args.dc_data)
        return
    source = args.historical_root / f'results/vopt_{args.year}/texas/networks/elec_s500_c185_ec_lvopt_3h_E_operations.nc'
    output = args.run_root / f'starting_networks/start_{args.year}.nc'
    if output.exists():
        raise FileExistsError(f'input preparation requires a fresh destination: {output}')
    prepare_starting_network(source, output)
    network = pypsa.Network(output)
    validate_frozen_starting_network(network)
    if len(network.snapshots) != 1464 or abs(network.snapshot_weightings.objective.sum() - 4392) > 1e-6:
        raise ValueError('seasonal horizon must have 1464 snapshots and 4392 weighted hours')
    persist_profiles(network, args.run_root, args.year, args.dc_data, siting=True)
    output.chmod(0o440)
    record = {'source': str(source), 'source_sha256': file_sha256(source),
              'network': str(output), 'network_sha256': file_sha256(output),
              'basis': 'historical_optimum_frozen', 'snapshots': 1464, 'weighted_hours': 4392}
    dest = args.run_root / f'manifests/base-seasonal-{args.year}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
