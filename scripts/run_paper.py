"""Portable entry point for the original paper's experiments."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ['dispatch', 'generation', 'generation_storage', 'generation_tx', 'transmission', 'full_tx']
LOCATIONS = ['HOUSTON', 'AUSTIN', 'TOLAR', 'DFW_SOUTH', 'ABILENE', 'CHILDRESS']

def run(script, *args):
    env = dict(os.environ, PYTHONHASHSEED='0', MPLBACKEND='Agg')
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, args)],
                   cwd=ROOT, env=env, check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['baseline', 'all2030', 'multiloc', 'analysis', 'rep'])
    p.add_argument('--mode', choices=MODES, default='full_tx')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--dc-data', type=Path)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--n-sims', type=int, default=5000)
    a = p.parse_args()
    if a.stage in ['baseline', 'all2030', 'multiloc']:
        count = 120 if a.stage == 'multiloc' else 5
        if not 0 <= a.index < count:
            p.error(f'index must be in 0..{count-1}')
        year = 2019 + (a.index // 24 if a.stage == 'multiloc' else a.index)
        mode = 'dispatch' if a.stage == 'baseline' else a.mode
        scenario = 'none' if a.stage == 'baseline' else a.stage
        result = 'results/dc_experiments' if a.stage == 'baseline' else f'results/dc_experiments_{mode}'
        args = ['--year', year, '--dc-scenario', scenario, '--mode', mode,
                '--results-root', result, '--threads', a.threads]
        if scenario == 'all2030':
            if not a.dc_data or not a.dc_data.is_file():
                p.error('--dc-data must name an existing private project spreadsheet')
            args += ['--dc-data', a.dc_data.resolve()]
        if scenario == 'multiloc':
            args += ['--location', LOCATIONS[(a.index % 24)//4],
                     '--scale', [500,1000,3000,5000][a.index % 4]]
        run('run_dc_experiment.py', *args)
    elif a.stage == 'analysis':
        run('compute_adder_factor_decomposition.py')
        run('compute_graded_unserved.py', '--multiloc')
    else:
        run('analyze_rep_risk.py', '--base-dirs', 'results/dc_experiments',
            'results/dc_experiments_dispatch', 'results/dc_experiments_full_tx',
            'results/dc_experiments_generation_storage', '--years', 2019,2020,2021,2022,2023,
            '--scenarios', 'none', 'all2030', '--modes', 'dispatch','full_tx','generation_storage',
            '--output-dir', 'results/rep_mode_contrast_cap5000_n1',
            '--cache-path', 'results/rep_mode_contrast_cap5000_n1/cache.pkl',
            '--zones', 'system','WEST','NORTH','SOUTH','HOUSTON', '--forward-case','both',
            '--rep-load',1000,'--rep-capital',50000000,'--hedge-ratio',0.70,
            '--forward-premium',5,'--rep-opex',3,'--retail-pass-through',0,
            '--passthrough-sigma',0,'--lmp-clip-upper',5000,
            '--full-tx-tcos-adder-mwh',2.751319804231755,
            '--n-sims',a.n_sims,'--seed',123,'--refresh-cache','--summary-only','--no-plots')

if __name__ == '__main__':
    main()
