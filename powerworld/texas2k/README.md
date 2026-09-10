# Texas2k DC engineering screening

This package maps PyPSA capacity-expansion results and data-center loads onto
the Texas2k Series25 transmission case. It runs DC N-0 or exhaustive N-1
screening, applies the paper's fixed upgrade rules, and exports transmission
upgrade costs.

## Installation

Run from the repository root with Python 3.11:

```bash
python -m uv sync --frozen --extra dev
source .venv/bin/activate
python -m powerworld.texas2k.cli validate-inputs
```

The runtime solver is PYPOWER 5.1.19.

## Package layout

```text
powerworld/texas2k/
├── case.py             # MATPOWER parser and DC power flow
├── data.py             # input validation and asset normalization
├── dispatch.py         # portfolio and data-center mapping
├── contingency.py      # N-0 and N-1 screening
├── upgrades.py         # upgrade sizing and cost rules
├── exports.py          # normalized result exports
├── workflow.py         # end-to-end orchestration
├── cli.py              # command-line interface
├── inputs/             # bundled paper inputs
├── reference_outputs/  # paper reference results
└── tests/              # unit and smoke tests
```

## Bundled inputs

### Transmission case

`inputs/case/texas2k_series25_summer_peak.m` contains:

- 2,751 buses;
- 1,099 generators;
- 5,344 branches.

`inputs/baseline/buses.csv` contains bus coordinates, nominal voltage,
station names, and Texas areas.

### Data centers

- `inputs/datacenters/projects.csv`: 417 data-center project rows.
- `inputs/datacenters/bus_matches.csv`: project-to-station matches.
- Positive matched injection: 384 rows and 39,864.607 MW.

### PyPSA portfolios

| File | Rows | Added capacity (MW) |
| --- | ---: | ---: |
| `generation_only_2023.csv` | 15 | 32,942.9 |
| `generation_storage_2023.csv` | 19 | 39,992.7 |

Asset tables use these columns:

| Column | Description |
| --- | --- |
| `year` | planning year |
| `case` | PyPSA case name |
| `location` | siting scenario |
| `asset_type` | `generation` or `storage` |
| `carrier` | PyPSA carrier |
| `zone` | ERCOT zone |
| `bus` | PyPSA bus identifier |
| `bus_x`, `bus_y` | longitude and latitude |
| `added_mw` | added nameplate capacity |

## Validate inputs

```bash
python -m powerworld.texas2k.cli validate-inputs
python -m powerworld.texas2k.cli validate-inputs --json
```

## Run bundled scenarios

```bash
python -m powerworld.texas2k.cli run --portfolio generation --criterion n0
python -m powerworld.texas2k.cli run --portfolio generation --criterion n1
python -m powerworld.texas2k.cli run --portfolio generation-storage --criterion n0
python -m powerworld.texas2k.cli run --portfolio generation-storage --criterion n1
```

The default output root is `results/powerworld/texas2k/`. Select a different
location with `--output-dir`:

```bash
python -m powerworld.texas2k.cli run \
  --portfolio generation-storage \
  --criterion n1 \
  --output-dir results/my_texas2k_run
```

N-1 runs screen every active branch for the baseline, data-center case, `NEW`
upgrade case, and `ALL` upgrade case. Progress is printed every 1,000 outages.
Set `--progress-every 0` to disable progress output.

## Run with a new PyPSA result

Aggregate the PyPSA experiment outputs:

```bash
python scripts/export_all_expansion_tx_detail.py
```

Normalize one scenario:

```bash
python -m powerworld.texas2k.cli normalize-assets \
  results/dc_experiments_full_tx/all_full_tx_expansion_detail.csv \
  results/texas2k_generation_2023.csv \
  --year 2023 \
  --case all2030_full_tx \
  --location ALL_2030
```

Run the normalized table:

```bash
python -m powerworld.texas2k.cli run \
  --portfolio generation \
  --criterion n1 \
  --assets results/texas2k_generation_2023.csv
```

For generation and storage together:

```bash
python -m powerworld.texas2k.cli normalize-assets \
  results/dc_experiments_full_tx/all_full_tx_expansion_detail.csv \
  results/texas2k_generation_storage_2023.csv \
  --year 2023 \
  --case all2030_full_tx \
  --location ALL_2030 \
  --include-storage

python -m powerworld.texas2k.cli run \
  --portfolio generation-storage \
  --criterion n1 \
  --assets results/texas2k_generation_storage_2023.csv
```

## Output files

Each scenario is written to `<output>/<portfolio>/<criterion>/`:

| File | Contents |
| --- | --- |
| `summary_n0.csv`, `summary_n1.csv` | upgrade counts, costs, and loading summary |
| `upgrade_detail_*.csv` | one row per upgraded branch |
| `branch_all_*.csv` | loading and classification for every branch |
| `skipped_contingencies.csv` | N-1 solver diagnostics by phase and branch |

`NEW` contains overloads introduced by the data-center case relative to its
matching baseline. `ALL` contains all overloaded branches in the data-center
case. `branch_id` is the zero-based MATPOWER branch-row identifier.

Create the normalized paper-analysis exports:

```bash
python -m powerworld.texas2k.cli export \
  --run-dir results/powerworld/texas2k/generation-storage/n1 \
  --portfolio generation-storage \
  --criterion n1
```

Exports are written under `<run-dir>/exports/`.

## Reference results

`reference_outputs/engineering_summary.csv` indexes the supplied paper runs.
The `NEW` rows are:

| Portfolio | Criterion | Upgrades | Cost (million 2024 USD) |
| --- | --- | ---: | ---: |
| generation | N-0 | 157 | 11,005.4 |
| generation | N-1 | 522 | 36,336.3 |
| generation-storage | N-0 | 226 | 12,621.3 |
| generation-storage | N-1 | 553 | 34,289.5 |

## Tests

```bash
pytest powerworld/texas2k/tests -q

python -m powerworld.texas2k.cli run \
  --input-dir powerworld/texas2k/tests/fixtures/smoke_inputs \
  --output-dir /tmp/texas2k-smoke \
  --portfolio generation-storage \
  --criterion n1 \
  --progress-every 0
```

## Data source

Texas2k Series25 is published by the Texas A&M Electric Grid Test Case
Repository: <https://electricgrids.engr.tamu.edu/texas2k-series25/>.

The requested Texas2k citations are listed in [`NOTICE.md`](NOTICE.md).
