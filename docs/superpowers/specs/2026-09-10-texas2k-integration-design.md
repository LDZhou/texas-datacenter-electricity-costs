# Texas2k Integration Design

## Objective

Integrate the collaborator-supplied Texas2k engineering workflow into this
repository as a self-contained public module that consumes a documented PyPSA
asset table, reproduces the paper's engineering-cost reference outputs, and
keeps private credentials, machine-specific paths, original office files, and
the source ZIP outside Git.

## Scope

The module covers the supplied PowerWorld-exported MATPOWER case, data-center
load injection, PyPSA generation and storage injection, DC N-0 and N-1
screening with PYPOWER, fixed-rule transmission upgrades, normalized output
exports, reference results, and documentation. It does not automate the
PowerWorld desktop application, claim AC convergence, optimize upgrades, or
remove residual post-upgrade violations.

The existing seasonal PyPSA experiments retain their original technology set
and results. The Texas2k module consumes their exported expansion assets; it
does not alter the PyPSA optimization model.

## Public Repository Layout

```text
powerworld/
├── README.md
└── texas2k/
    ├── README.md
    ├── NOTICE.md
    ├── __init__.py
    ├── cli.py
    ├── case.py
    ├── inputs.py
    ├── dispatch.py
    ├── contingency.py
    ├── upgrades.py
    ├── exports.py
    ├── inputs/
    │   ├── case/texas2k_series25_summer_peak.m
    │   ├── baseline/buses.csv
    │   ├── datacenters/projects.csv
    │   ├── datacenters/bus_matches.csv
    │   └── pypsa/
    │       ├── generation_only_2023.csv
    │       └── generation_storage_2023.csv
    ├── reference_outputs/
    │   ├── engineering_summary.csv
    │   ├── generation/
    │   └── generation_storage/
    └── tests/
        ├── fixtures/
        ├── test_case.py
        ├── test_inputs.py
        ├── test_contingency.py
        ├── test_upgrades.py
        └── test_exports.py
```

Files use semantic names. The supplied date-prefixed script names are retained
only in the provenance mapping in `NOTICE.md`.

## Data Boundaries and Provenance

The source archive remains at its private OneDrive location and is not copied
to the repository. The public module contains only the inputs needed to run the
engineering workflow:

- the PowerWorld Simulator 24 export of the Texas2k Series25 MATPOWER case;
- normalized CSV versions of the data-center project and bus-match tables;
- the corrected Texas2k bus-coordinate table;
- two filtered 2023 PyPSA expansion tables for generation-only and
  generation-plus-storage scenarios; and
- reference CSV outputs needed to verify the paper calculation.

Excel workbooks are converted to CSV so author metadata and opaque workbook
state are excluded. Personal absolute paths are replaced by command-line paths
and repository-relative defaults. API keys, Gurobi licenses, shell environment
files, ZIP archives, and private data remain covered by ignore rules and the
public-tree scanner.

The Texas2k case is synthetic. `NOTICE.md` and the module README cite the Texas
A&M Texas2k Series25 data page and the dataset's requested academic references.
The collaborator workflow is attributed at project-team level until the authors
provide preferred public names and identifiers.

## Canonical PyPSA Interface

The module accepts a normalized asset table with these required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `year` | integer | planning year |
| `case` | string | PyPSA experiment case |
| `location` | string | siting scenario |
| `asset_type` | string | `generation` or `storage` |
| `carrier` | string | PyPSA carrier label |
| `bus` | string | PyPSA bus identifier |
| `bus_x` | float | longitude in EPSG:4326 |
| `bus_y` | float | latitude in EPSG:4326 |
| `added_mw` | float | added nameplate capacity in MW |

The repository ships filtered examples for the published 2023
`all2030_full_tx`/`ALL_2030` case. A normalization command also accepts the
existing `scripts/export_all_expansion_tx_detail.py` CSV schema and writes the
canonical table. It validates required columns, finite coordinates, supported
asset types, and non-negative capacity before any power-flow run.

## Execution Interface

All commands run from the repository root with the root Python 3.11
environment:

```bash
python -m powerworld.texas2k.cli validate-inputs
python -m powerworld.texas2k.cli run --portfolio generation --criterion n0
python -m powerworld.texas2k.cli run --portfolio generation --criterion n1
python -m powerworld.texas2k.cli run --portfolio generation-storage --criterion n0
python -m powerworld.texas2k.cli run --portfolio generation-storage --criterion n1
python -m powerworld.texas2k.cli export --run-dir /absolute/path/to/run
```

`--input-dir` and `--output-dir` override repository defaults. Runtime results
go to ignored `results/powerworld/texas2k/`; the module never deletes files
outside the selected run directory. Reference outputs are read-only fixtures
for regression checks.

The root environment adds an exact PYPOWER dependency compatible with Python
3.11 and the existing NumPy/SciPy pins. No PowerWorld installation is required
for the supplied DC workflow because the MATPOWER case is already exported.

## Model Semantics

The workflow preserves the collaborator's scientific choices:

- DC load is the positive-capacity, geographically matched Texas data-center
  subset. Unmatched El Paso and Texarkana records remain excluded and are
  reported in validation output.
- PyPSA OCGT capacity is distributed across nearby existing generators in the
  generation-only portfolio. The generation-plus-storage portfolio uses the
  supplied zone-aware dispatch and storage treatment.
- N-0 evaluates the intact network. N-1 opens each branch independently and
  records each branch's worst finite loading.
- Non-convergent, islanded, or non-finite N-1 contingencies are recorded and
  warned about; the run continues. Summary outputs include skipped counts and
  reasons.
- Upgrade tiers, maximum three-times-rating cap, detour factor, and 2024-dollar
  MISO MTEP24 cost tables preserve the supplied calculations.
- `NEW` upgrades target overloads attributable to the data-center case;
  `ALL` upgrades target all overloads in the scenario.
- Post-upgrade residual violations are reported as warnings and retained in
  summaries. They do not fail the run.

Outputs describe a deterministic engineering screening heuristic. They do not
represent an AC solution, a security-constrained optimal transmission plan, or
complete N-1 remediation.

## Output Identity and Export Safety

Every branch receives a stable `branch_id` equal to its zero-based MATPOWER row
index. Upgrade detail and all-branch exports join exclusively on `branch_id`.
This prevents parallel circuits sharing the same endpoint bus pair from
receiving duplicated upgrade flags or costs.

Each run writes:

- `summary_<criterion>.csv` for scenario-level metrics;
- `upgrade_detail_<criterion>.csv` for upgraded branches;
- `branch_all_<criterion>.csv` for all 5,344 case branches; and
- `skipped_contingencies.csv` for N-1 runs.

The normalized exporter writes upgraded and all-branch tables plus an index
summary. Costs are summed only from upgrade-detail rows. Exports include method,
portfolio, criterion, recovery mode, input hashes, and software version fields.

## Reproduction Targets

Regression checks use the supplied outputs as historical reference values:

| Portfolio | Criterion | Recovery | Upgrades | Cost (million 2024 USD) |
| --- | --- | --- | ---: | ---: |
| generation | N-0 | NEW | 157 | 11,005.4 |
| generation | N-1 | NEW | 522 | 36,336.3 |
| generation-storage | N-0 | NEW | 226 | 12,621.3 |
| generation-storage | N-1 | NEW | 553 | 34,289.5 |

The paper-facing value is the final row. Its supplied post-upgrade result still
contains 55 residual target-line violations and 173 residual grid violations.
Documentation must present the value with this limitation.

## Verification

Fast tests cover MATPOWER parsing, asset validation, data-center aggregation,
finite-contingency handling, upgrade-cost rules, stable parallel-line exports,
and CLI path behavior using a small synthetic case. Integration checks validate
the bundled case dimensions, input totals, and reference-summary reconciliation
without running the full 5,344-contingency sweep.

The public-tree scanner covers CSV content as well as source and documentation,
reports locations without printing suspected values, rejects credentials and
personal absolute paths, and permits the explicitly documented synthetic
MATPOWER input. The final acceptance run includes the full relevant pytest
suite, the public-tree scan, CLI input validation, and a synthetic end-to-end
N-0/N-1 smoke run.
