# Texas2k DC engineering screening

This module connects the paper's PyPSA capacity-expansion results to a synthetic
Texas transmission case. It injects the selected generation/storage portfolio
and matched data-center load, runs DC N-0 or exhaustive branch-outage N-1
screening, applies fixed engineering upgrade rules, and reports capital cost and
remaining violations.

The MATPOWER case was exported by PowerWorld Simulator. Runtime calculations use
PYPOWER, so reproducing these CSV outputs does not require a PowerWorld license
or desktop installation.

## Method boundary

This workflow provides a reproducible engineering screening estimate. Its key
boundaries are:

- DC power flow (`rundcpf`) throughout;
- one outage per active branch for N-1;
- exogenous PyPSA generation/storage additions;
- fixed overload tiers and rating/impedance multipliers;
- line and transformer unit costs in million 2024 USD;
- a maximum rating multiplier of 3.0 and line-route detour factor of 1.30;
- warning-and-record handling for non-convergent, islanded, non-finite, and
  post-upgrade residual cases.

The outputs are not an AC solution, a security-constrained expansion optimum, or
a guarantee that every N-1 violation is cleared.

## Install

From the repository root, use Python 3.11 and the root lock file:

```bash
python -m uv sync --frozen --extra dev
source .venv/bin/activate
python -m powerworld.texas2k.cli validate-inputs
```

PYPOWER is pinned to 5.1.19. Gurobi is used by the upstream PyPSA workflow and
is not used by the Texas2k DC screening commands.

## Directory layout

```text
powerworld/texas2k/
├── case.py             # MATPOWER parser and DC solver wrapper
├── data.py             # input schema, validation, and normalization
├── dispatch.py         # load mapping, portfolio injection, dispatch
├── contingency.py      # N-0 and exhaustive N-1 screening
├── upgrades.py         # branch table, tiers, engineering cost rules
├── exports.py          # stable branch-ID exports
├── workflow.py         # complete baseline/scenario/recovery run
├── cli.py              # public command-line interface
├── inputs/             # immutable public paper inputs
├── reference_outputs/  # supplied paper-result CSVs
└── tests/              # unit, integration, and synthetic smoke tests
```

Runtime outputs default to `results/powerworld/texas2k/`, which is ignored by
Git. `--output-dir` selects another location. The workflow refuses to place
outputs inside its input directory and never deletes unrelated files.

## Inputs

### Texas2k case and bus geography

`inputs/case/texas2k_series25_summer_peak.m` is the summer-peak MATPOWER export:
2,751 bus rows, 1,099 generator rows, and 5,344 branch rows. The source file
identifies PowerWorld Simulator version 24, build date March 14, 2025.

`inputs/baseline/buses.csv` provides the bus coordinates, nominal voltage,
station name, and area used for geographic mapping and line-length estimates.
All 2,751 rows have coordinates in this corrected table.

### Data centers

`inputs/datacenters/projects.csv` is a plain-CSV conversion of the 417-row paper
project table. Capacities are `current_mw + construction_mw + planned_mw`.
`inputs/datacenters/bus_matches.csv` contains the offline Texas2k station match.

The validation counts are:

| Stage | Rows | Capacity (MW) |
| --- | ---: | ---: |
| source table | 417 | 40,293.907 |
| geographically matched | 409 | 39,864.607 including zero rows |
| positive matched rows injected | 384 | 39,864.607 |

Seven El Paso rows are marked `OUTSIDE`; one Texarkana row is marked `FAR`.
Matched project capacity is aggregated by station and distributed among that
station's bus rows in proportion to nominal voltage.

The upstream PyPSA experiment uses its own city/ERCOT mapping stage and reports
371 projects totaling about 38.9 GW. Keep the stage label beside each count when
comparing results.

### PyPSA portfolios

The canonical asset schema is:

| Column | Meaning |
| --- | --- |
| `year` | planning year |
| `case` | PyPSA case identifier |
| `location` | siting scenario |
| `asset_type` | `generation` or `storage` |
| `carrier` | PyPSA carrier |
| `zone` | ERCOT zone used by dispatch mapping |
| `bus` | PyPSA bus identifier |
| `bus_x`, `bus_y` | longitude and latitude in EPSG:4326 |
| `added_mw` | added nameplate capacity in MW |

Bundled portfolios:

| File | Rows | Added capacity (MW) |
| --- | ---: | ---: |
| `generation_only_2023.csv` | 15 | 32,942.9 |
| `generation_storage_2023.csv` | 19 | 39,992.7 |

The generation-only portfolio contains 32,935.8 MW OCGT and 7.1 MW solar. The
generation-plus-storage portfolio contains 33,941.4 MW OCGT, 2,922.4 MW onshore
wind, 1,958.9 MW solar, and 1,170.0 MW four-hour battery capacity.

## Run the bundled paper portfolios

Validate inputs first:

```bash
python -m powerworld.texas2k.cli validate-inputs
python -m powerworld.texas2k.cli validate-inputs --json
```

Run any portfolio/criterion pair:

```bash
python -m powerworld.texas2k.cli run --portfolio generation --criterion n0
python -m powerworld.texas2k.cli run --portfolio generation --criterion n1
python -m powerworld.texas2k.cli run --portfolio generation-storage --criterion n0
python -m powerworld.texas2k.cli run --portfolio generation-storage --criterion n1
```

N-0 uses a few DC solves. N-1 performs complete 5,344-branch sweeps for the
baseline, injected case, `NEW` recovery, and `ALL` recovery. Runtime therefore
depends strongly on CPU and linear-algebra performance. `--progress-every 1000`
is the default; use `0` to suppress progress lines.

## Feed a new PyPSA result directly

`scripts/run_dc_experiment.py` includes `bus_x` and `bus_y` in every new
generation/storage row. Consolidate the experiment tree, then normalize one
case:

```bash
python scripts/export_all_expansion_tx_detail.py
python -m powerworld.texas2k.cli normalize-assets \
  results/dc_experiments_full_tx/all_full_tx_expansion_detail.csv \
  results/texas2k_generation_2023.csv \
  --year 2023 --case all2030_full_tx --location ALL_2030
```

Pass the result without changing the module inputs:

```bash
python -m powerworld.texas2k.cli run \
  --portfolio generation --criterion n1 \
  --assets results/texas2k_generation_2023.csv
```

For a generation-plus-storage export, include `--include-storage` during
normalization and select `--portfolio generation-storage` during the run. The
case string should match the case field produced by that PyPSA experiment.

## Outputs

Each run writes to `<output>/<portfolio>/<criterion>/`:

- `summary_n0.csv` or `summary_n1.csv`: one row each for `NEW` and `ALL`;
- `upgrade_detail_*.csv`: exact upgraded branches and costs;
- `branch_all_*.csv`: all case branches with intact/scenario diagnostics;
- `skipped_contingencies.csv`: N-1 phases and solver diagnostics. For a
  partially non-finite result, every finite branch flow still contributes to
  the worst-case envelope; exceptions, missing arrays, and non-converged cases
  contribute no flows.

`NEW` targets overloads introduced relative to the matching no-data-center
portfolio baseline. `ALL` targets every overloaded branch in the injected case.
`n_residual_line` counts targeted upgraded branches still overloaded after the
fixed action. `n_residual_grid` counts all overloaded grid branches after that
action. Residual values trigger warnings and remain valid output rows.

Create collaborator-facing normalized tables after a run:

```bash
python -m powerworld.texas2k.cli export \
  --run-dir results/powerworld/texas2k/generation-storage/n1 \
  --portfolio generation-storage --criterion n1
```

Every branch uses `branch_id`, the zero-based MATPOWER row index. Endpoint bus
pairs are retained for readability but never serve as an export join key, which
keeps parallel circuits distinct.

## Supplied reference results

`reference_outputs/engineering_summary.csv` indexes the supplied collaborator
runs. The `NEW` rows are:

| Portfolio | Criterion | Upgrades | Cost (million 2024 USD) | Residual grid violations |
| --- | --- | ---: | ---: | ---: |
| generation | N-0 | 157 | 11,005.4 | 55 |
| generation | N-1 | 522 | 36,336.3 | 152 |
| generation-storage | N-0 | 226 | 12,621.3 | 46 |
| generation-storage | N-1 | 553 | 34,289.5 | 173 |

The last row is the paper-facing engineering reference. Its 55 residual target
lines and 173 residual grid violations must accompany interpretation of the
34.2895 billion USD estimate.

On the locked public environment, the N-0 results reproduce the supplied values
at unrounded precision. The exhaustive N-1 runs each differ by one paid target
branch: generation produces 521 upgrades and 36,281.255 million USD, while
generation-plus-storage produces 552 upgrades and 34,234.473 million USD. The
case and scenario arrays are numerically identical to the supplied script's
inputs. The difference occurs in singular/islanded contingency solves and is
sensitive to the sparse linear-algebra backend. Both the supplied historical
outputs and current-run diagnostics are therefore retained; see the root
`VERIFICATION.md` for the exact comparison.

## Tests

```bash
pytest powerworld/texas2k/tests -q
python -m powerworld.texas2k.cli run \
  --input-dir powerworld/texas2k/tests/fixtures/smoke_inputs \
  --output-dir /tmp/texas2k-smoke \
  --portfolio generation-storage --criterion n1 --progress-every 0
python scripts/check_public_tree.py
```

The fast suite validates parsing, totals, dispatch, warning-only contingency
handling, upgrade costs, parallel-line identity, CLI isolation, exports, and a
small end-to-end network. It does not repeat the full 5,344-outage production
calculation on every test run.

## Data source and citation

The Texas2k Series25 system is entirely synthetic and, according to the Texas
A&M Electric Grid Test Case Repository, is provided free for commercial and
non-commercial use. Users are encouraged to register the dataset use and cite:

- A. B. Birchfield, T. Xu, K. M. Gegner, K. S. Shetye, and T. J. Overbye,
  “Grid Structural Characteristics as Validation Criteria for Synthetic
  Networks,” *IEEE Transactions on Power Systems*, 32(4), 3258–3265, 2017.
- J. Baek and A. B. Birchfield, “A tuning method for exciters and governors in
  realistic synthetic grids with dynamics,” *2023 North American Power
  Symposium*, 1–6, 2023.

Dataset page: <https://electricgrids.engr.tamu.edu/texas2k-series25/>.
Also cite PYPOWER/MATPOWER as requested by those projects. See `NOTICE.md` for
the collaborator source mapping and packaging record.
