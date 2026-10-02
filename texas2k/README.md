# Texas2k engineering screening

The `texas2k` package maps PyPSA supply additions and data-center loads onto
Texas2k Series25 case 1, screens thermal loading with PYPOWER DC power flow,
and estimates branch reinforcement costs. It includes the MATPOWER case,
coordinate tables, project-to-substation mapping, and accepted engineering
reference outputs.

## Inputs

- `inputs/case/texas2k_series25_summer_peak.m`: 2,751 buses, 1,099 generators,
  and 5,344 branches.
- `inputs/baseline/buses.csv`: bus coordinates, nominal voltage, station names,
  and areas. Coordinates are derived from the Texas2k geographic data and
  matched by station name.
- `inputs/datacenters/projects.csv` and `bus_matches.csv`: project records and
  station matches. The engineering injection contains 384 positive matched
  rows totaling 39,864.607 MW; the planning model applies its ERCOT geographic
  filter separately.
- `inputs/pypsa/generation_only_2023.csv`: the matched coordinated-expansion
  generation portfolio, totaling 37,323.0 MW.
- `inputs/pypsa/generation_storage_2023.csv`: the generation-and-storage
  portfolio for comparison experiments, totaling 108,982.3 MW.

PyPSA asset tables preserve `year`, `case`, `location`, `asset_type`, `carrier`,
`zone`, `bus`, `bus_x`, `bus_y`, and `added_mw`. CCGT and OCGT retain their
carrier labels and map to gas-fired capacity in the engineering network.

## Commands

Run from the repository root after installation:

```bash
python -m texas2k.cli validate-inputs --json
python -m texas2k.cli run --portfolio generation --criterion n0
python -m texas2k.cli run --portfolio generation --criterion n1
```

Outputs are written to `results/texas2k/generation/n0/` or `n1/`. To use a
new planning result:

```bash
python scripts/export_pypsa_assets.py \
  results/paper/results/seasonal/2023/all2030_full_tx \
  results/texas2k_assets_2023.csv --year 2023 --case all2030_full_tx
python -m texas2k.cli run --portfolio generation --criterion n1 \
  --assets results/texas2k_assets_2023.csv
```

Add `--include-storage` to the asset export and select
`--portfolio generation-storage` to evaluate the storage portfolio.

## Outputs

Each run writes:

- `summary_n0.csv` or `summary_n1.csv`: target counts, investment, cleared
  targets, residual target and grid violations.
- `upgrade_detail_*.csv`: reinforcement tier and cost for each target branch.
- `branch_all_*.csv`: intact and worst-contingency loading by branch.
- `skipped_contingencies.csv`: excluded or unsuccessful N-1 contingencies.

`NEW` reinforces branches that cross their thermal rating relative to the
matched baseline; `ALL` also reinforces baseline violations. Reinforcement
is followed by one screening pass. Residual violations remain in the outputs.
`branch_id` is the zero-based MATPOWER branch-row identifier.

Normalize a completed run for downstream analysis:

```bash
python -m texas2k.cli export --run-dir results/texas2k/generation/n1 \
  --portfolio generation --criterion n1
```

## Reference results

`reference_outputs/generation/` contains the accepted matched 2023 engineering
results. `reference_outputs/manifest.json` records their input and output
hashes. The `NEW` results are:

| Criterion | Target branches | Investment (million 2024 USD) | Cleared targets | Residual targets | Total grid violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| N-0 | 165 | 10,426.383 | 146 | 19 | 31 |
| N-1 | 519 | 32,419.304 | 451 | 68 | 143 |

The archive is verified by file hashes. A recorded execution of the available
code with the same asset table classified 517 new targets ($32.338354 billion).
The two branch-classification differences relative to the accepted archive
remain unresolved and are recorded in the manifest; exact numerical
reproduction of this engineering archive has not been verified.

## Source

Texas2k Series25 is available from the
[Texas A&M Electric Grid Test Case Repository](https://electricgrids.engr.tamu.edu/texas2k-series25/).
Source acknowledgements and requested citations are in [NOTICE.md](NOTICE.md).
