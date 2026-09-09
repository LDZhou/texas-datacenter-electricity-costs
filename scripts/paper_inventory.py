"""Exact seasonal result selection; ignore unrelated directories."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ('dispatch','generation','generation_storage','generation_tx','transmission','full_tx')
LOCATIONS = ('HOUSTON','AUSTIN','TOLAR','DFW_SOUTH','ABILENE','CHILDRESS')

def metric_files(scenario):
    files = []
    for mode in MODES:
        for year in range(2019,2024):
            base = ROOT / 'results' / f'dc_experiments_{mode}' / str(year)
            names = [f'all2030_{mode}'] if scenario == 'all2030' else [
                f'multiloc_{mode}_{loc}_{scale}MW' for loc in LOCATIONS for scale in (500,1000,3000,5000)]
            for name in names:
                p = base / name / 'metrics.csv'
                if p.exists():
                    files.append(p)
    if not files:
        raise FileNotFoundError(f'No canonical {scenario} metrics; run experiments first')
    return files
