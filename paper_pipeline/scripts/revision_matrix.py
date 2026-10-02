"""Explicit, matched experiment inventory for the paper."""
import argparse
import json
from pathlib import Path

LOCATIONS = ('HOUSTON', 'AUSTIN', 'TOLAR', 'DFW_SOUTH', 'ABILENE', 'CHILDRESS')
SCALES = (500, 1000, 3000, 5000)
SITING_YEARS = (2023, 2019, 2020, 2021, 2022)


def case(year, scenario, mode='full_tx', group='seasonal', variant='base',
         location=None, scale=None, crossover=0, capital_factor=1.0):
    horizon = 'full_year' if group == 'full_year' else 'seasonal'
    name = f'{scenario}_{mode}'
    if scenario == 'multiloc':
        name += f'_{location}_{scale}MW'
    if variant != 'base' or crossover:
        name += f'_crossover{crossover}'
    if variant != 'base':
        name += f'_{variant}'
    profile = None
    if scenario == 'all2030':
        profile = f'profiles/{horizon}/{year}/all2030.csv'
    elif scenario == 'multiloc':
        profile = f'profiles/{horizon}/{year}/{location}_{scale}.csv'
    return dict(year=year, scenario=scenario, mode=mode, group=group,
                horizon=horizon, variant=variant, location=location, scale=scale,
                crossover=crossover, capital_factor=capital_factor, profile=profile,
                output=f'results/{group}/{year}/{name}')


def build_matrix(include_diagnostics=False):
    """Build the public inventory, with optional diagnostics kept separate."""
    m = {k: [] for k in ('central', 'weather', 'siting')}
    for year in (2023, 2019, 2020, 2021, 2022):
        dest = m['central' if year == 2023 else 'weather']
        dest.extend((case(year, 'none'), case(year, 'none', 'generation_storage'), case(year, 'all2030'),
                     case(year, 'all2030', 'generation_storage'), case(year, 'none', 'dispatch'),
                     case(year, 'all2030', 'dispatch')))
    # Seasonal siting for every weather year.
    for year in SITING_YEARS:
        for mode in ('dispatch', 'generation_storage', 'generation_tx'):
            m['siting'].append(case(year, 'none', mode, group='siting'))
            for location in LOCATIONS:
                for scale in SCALES:
                    m['siting'].append(case(year, 'multiloc', mode, group='siting', location=location, scale=scale))
    if include_diagnostics:
        m.update({k: [] for k in ('sensitivity', 'full_year', 'numerics')})
        for variant in ('gas0.5', 'gas1.5', 'ccgt_off', 'capitalhalf'):
            for scenario in ('none', 'all2030'):
                m['sensitivity'].append(case(2023, scenario, group='sensitivity', variant=variant,
                                             capital_factor=0.5 if variant == 'capitalhalf' else 1.0))
        for scenario in ('none', 'all2030'):
            m['full_year'].append(case(2023, scenario, group='full_year', variant='regenerated'))
            m['numerics'].append(case(2023, scenario, group='numerics', crossover=1))
    return m


def validate_matrix(matrix):
    outputs = [c['output'] for cases in matrix.values() for c in cases]
    if len(set(outputs)) != len(outputs):
        raise ValueError('duplicate output in experiment matrix')
    for cases in matrix.values():
        for c in cases:
            if c['scenario'] != 'none' and not c['profile']:
                raise ValueError('missing persisted DC profile')
            if Path(c['output']).is_absolute() or '..' in Path(c['output']).parts:
                raise ValueError('unsafe result path')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-diagnostics', action='store_true',
                   help='Include parameter, full-year, and numerical diagnostic cases.')
    args = p.parse_args()
    m = build_matrix(include_diagnostics=args.include_diagnostics)
    validate_matrix(m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(m, indent=2) + '\n')


if __name__ == '__main__':
    main()
