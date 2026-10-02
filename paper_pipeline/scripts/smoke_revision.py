"""Compute-node integration smoke on a real eight-snapshot ERCOT network."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import pypsa

from .model_contract import load_dc_profile_table, snapshot_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--historical-root', type=Path, required=True)
    p.add_argument('--dc-data', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    args = p.parse_args()
    dest = args.run_root / 'smoke_inputs' / f'{os.environ.get("SLURM_JOB_ID", "local")}_{args.index}'
    dest.mkdir(parents=True, exist_ok=False)
    net = pypsa.Network(args.run_root / 'starting_networks/start_2023.nc')
    profile = load_dc_profile_table(args.run_root / 'profiles/seasonal/2023/all2030.csv', net.snapshots)
    net.set_snapshots(net.snapshots[:8])
    start = dest / 'start.nc'
    net.export_to_netcdf(start)
    profile = profile.loc[snapshot_labels(net.snapshots)]
    profile.to_csv(dest / 'dc.csv', float_format='%.17g')
    scenario = 'none' if args.index == 0 else 'all2030'
    command = [sys.executable, '-m', 'paper_pipeline.scripts.run_experiment', '--year', '2023',
               '--dc-scenario', scenario, '--mode', 'full_tx', '--starting-network', str(start),
               '--results-root', str(dest / 'results'), '--threads', '16', '--solver-time-limit', '1800',
               '--config-variant', 'smoke', '--horizon', 'smoke']
    if args.index:
        command += ['--dc-data', str(args.dc_data),
                    '--dc-profile-file', str(dest / 'dc.csv')]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
