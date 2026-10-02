"""Run one immutable inventory entry and publish an execution manifest."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .model_contract import file_sha256


LICENSE_OVERAGE_MARKERS = ('Overage for too long', 'No Gurobi license', 'License server', 'too many active sessions')


def is_license_failure(output: str) -> bool:
    """True when the solver failed for want of a licence session, not a modelling reason."""
    return any(marker.lower() in output.lower() for marker in LICENSE_OVERAGE_MARKERS)


def run_with_license_retry(argv, retries, wait_seconds, runner=subprocess.run, sleep=time.sleep):
    """Run the case; on a Gurobi licence-session failure wait and try again.

    The WLS licence admits two concurrent sessions. A task that starts while
    the pool is saturated fails within a minute with 'Overage for too long';
    re-queuing it as a fresh array element cost an index each time, so the
    wait now happens inside the task. Returns the number of attempts on
    success, or minus the exit code (at least -1) on a final failure.
    """
    for attempt in range(1, retries + 2):
        proc = runner(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        sys.stdout.write(proc.stdout or '')
        sys.stdout.flush()
        if proc.returncode == 0:
            return attempt
        if attempt <= retries and is_license_failure(proc.stdout or ''):
            print(f'[license] no Gurobi session available (attempt {attempt}/{retries + 1}); '
                  f'waiting {wait_seconds}s before retrying', flush=True)
            sleep(wait_seconds)
            continue
        return -max(proc.returncode, 1)
    return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--historical-root', type=Path, required=True)
    p.add_argument('--dc-data', type=Path, required=True,
                   help='Public CSV or workbook used to construct the persisted DC profiles')
    p.add_argument('--group', required=True)
    p.add_argument('--index', required=True, type=int)
    p.add_argument('--license-retries', type=int, default=18,
                   help='how many times to wait for a Gurobi licence session before giving up (default 18)')
    p.add_argument('--license-retry-wait', type=int, default=600, help='seconds between licence retries (default 600)')
    p.add_argument('--solver-option', action='append', default=None, metavar='KEY=VALUE',
                   help='Recorded Gurobi stability override for a documented rerun of a case whose barrier stalled')
    args = p.parse_args()
    matrix = json.loads((args.run_root / 'manifests/experiment_matrix.json').read_text())
    c = matrix[args.group][args.index]
    out = args.run_root / c['output']
    if out.exists():
        raise FileExistsError(f'refusing to overwrite existing result: {out}')
    start = args.run_root / 'starting_networks' / (f'start_fy_{c["year"]}.nc' if c['horizon'] == 'full_year' else f'start_{c["year"]}.nc')
    if c['year'] == 2021 or c['variant'].startswith('gas'):
        start = args.run_root / 'starting_networks' / f'start_2023_{c["variant"]}.nc'
    if c['year'] == 2021 and not c['variant'].startswith('gas'):
        start = args.run_root / 'starting_networks' / 'start_2021_gasfix.nc'
    argv = [sys.executable, '-m', 'paper_pipeline.scripts.run_experiment',
            '--year', str(c['year']), '--dc-scenario', c['scenario'], '--mode', c['mode'],
            '--starting-network', str(start), '--results-root', str(args.run_root / 'results' / c['group']),
            '--config-variant', c['variant'], '--crossover', str(c['crossover']),
            '--capital-cost-factor', str(c['capital_factor']), '--horizon', c['horizon'],
            '--threads', '16',
            # The numerics cross-check runs Gurobi with crossover on. Barrier alone
            # takes ~900s on this model and crossover then rebuilds a 3.2M-variable
            # basis, so a 3600s budget cut it off mid-crossover and left a model
            # whose solution could not be read back.
            '--solver-time-limit', '14400' if args.group == 'numerics' else '144000']
    if c['profile']:
        argv += ['--dc-profile-file', str(args.run_root / c['profile'])]
    if c['scenario'] == 'all2030':
        argv += ['--dc-data', str(args.dc_data)]
    if c['scenario'] == 'multiloc':
        argv += ['--location', c['location'], '--scale', str(c['scale'])]
    if c['variant'] == 'ccgt_off':
        argv += ['--candidate-carriers', 'OCGT,solar,onwind']
    for item in args.solver_option or []:
        argv += ['--solver-option', item]
    record = {'case': c, 'command': argv, 'starting_network_sha256': file_sha256(start)}
    if args.solver_option:
        record['solver_option_overrides'] = list(args.solver_option)
    attempts = run_with_license_retry(argv, args.license_retries, args.license_retry_wait)
    if attempts > 1:
        record['license_retries'] = attempts - 1
    if attempts < 0:
        failure = args.run_root / f'manifests/failed-{args.group}-{args.index}.json'
        failure.write_text(json.dumps({**record, 'exit_code': -attempts if attempts < -1 else 1}, indent=2) + '\n')
        raise SystemExit(f'case {args.group}[{args.index}] failed')
    for name in ('network.nc', 'metrics.json'):
        if not (out / name).is_file():
            raise RuntimeError(f'accepted result missing {out / name}')
    record['network_sha256'] = file_sha256(out / 'network.nc')
    record['metrics_sha256'] = file_sha256(out / 'metrics.json')
    (out / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
