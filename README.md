# Texas data center electricity costs

Code and data for the paper's coupled PyPSA and Texas2k experiments on data
center demand, generation and transmission expansion, wholesale prices, and
residential electricity costs in ERCOT.

The workflow has two connected stages:

```text
PyPSA capacity expansion
        ↓ coordinate-enriched asset CSV
Texas2k N-0/N-1 engineering screening
        ↓ transmission upgrade cost
Residential electricity-cost analysis
```

## Repository layout

- `workflow/`: PyPSA-USA workflow and model code.
- `config/`, `configs/`: upstream and paper experiment configurations.
- `scripts/`: experiment, aggregation, analysis, and plotting commands.
- `slurm/`: Slurm launchers for the paper experiment matrix.
- `powerworld/texas2k/`: Texas2k DC screening module, inputs, tests, and
  reference outputs.
- `results/`: generated outputs.

## Installation

Use Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/LDZhou/texas-datacenter-electricity-costs.git
cd texas-datacenter-electricity-costs
python -m uv sync --frozen --extra dev
source .venv/bin/activate
ln -s workflow/repo_data repo_data
mkdir -p logs
```

PyPSA optimization uses Gurobi. Point `GRB_LICENSE_FILE` to the license file
available on the machine or cluster:

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

For EIA downloads:

```bash
cp config/config.api.example.yaml config/config.api.yaml
```

Add the EIA API key to `config/config.api.yaml`.

## PyPSA experiments

Prepared starting networks are stored as:

```text
results/starting_networks/start_2019.nc
results/starting_networks/start_2020.nc
results/starting_networks/start_2021.nc
results/starting_networks/start_2022.nc
results/starting_networks/start_2023.nc
```

Submit the five baseline years first:

```bash
sbatch --account=ACCOUNT --partition=PARTITION --array=0-4%1 slurm/base.sbatch
```

Submit experiment arrays after the starting networks are ready:

```bash
sbatch --account=ACCOUNT --partition=PARTITION --array=0-4%2 \
  slurm/experiment.sbatch baseline

sbatch --account=ACCOUNT --partition=PARTITION --array=0-4%2 \
  slurm/experiment.sbatch all2030 \
  --mode full_tx \
  --dc-data powerworld/texas2k/inputs/datacenters/projects.csv

sbatch --account=ACCOUNT --partition=PARTITION --array=0-119%2 \
  slurm/experiment.sbatch multiloc --mode full_tx
```

Available experiment modes are `dispatch`, `generation`,
`generation_storage`, `generation_tx`, `transmission`, and `full_tx`.
The all2030 array indices 0–4 select 2019–2023. The multiloc array combines
year, location, and load scale.

A single scenario can also run directly:

```bash
python scripts/run_paper.py all2030 \
  --mode full_tx \
  --index 4 \
  --dc-data powerworld/texas2k/inputs/datacenters/projects.csv
```

## Analysis and figures

After the experiment arrays finish:

```bash
python scripts/run_paper.py analysis
python scripts/run_paper.py rep
python scripts/plot_paper.py --skip-validation
```

Figures are written to `results/nc_paper_figures/`.

## Texas2k engineering screening

The included Texas2k module runs PYPOWER DC power flow on the Texas2k Series25
MATPOWER case. It supports intact-network N-0 screening, exhaustive
single-branch-outage N-1 screening, and fixed transmission-upgrade cost rules.

Validate the bundled inputs:

```bash
python -m powerworld.texas2k.cli validate-inputs
```

Run the four bundled portfolio/criterion combinations:

```bash
python -m powerworld.texas2k.cli run --portfolio generation --criterion n0
python -m powerworld.texas2k.cli run --portfolio generation --criterion n1
python -m powerworld.texas2k.cli run --portfolio generation-storage --criterion n0
python -m powerworld.texas2k.cli run --portfolio generation-storage --criterion n1
```

Outputs are written under `results/powerworld/texas2k/`. Detailed input and
output schemas are documented in
[`powerworld/texas2k/README.md`](powerworld/texas2k/README.md).

## Pass a new PyPSA result to Texas2k

Aggregate the PyPSA capacity-expansion results:

```bash
python scripts/export_all_expansion_tx_detail.py
```

Normalize the selected year, case, and location:

```bash
python -m powerworld.texas2k.cli normalize-assets \
  results/dc_experiments_full_tx/all_full_tx_expansion_detail.csv \
  results/texas2k_generation_2023.csv \
  --year 2023 \
  --case all2030_full_tx \
  --location ALL_2030
```

Run Texas2k with that asset table:

```bash
python -m powerworld.texas2k.cli run \
  --portfolio generation \
  --criterion n1 \
  --assets results/texas2k_generation_2023.csv
```

Add `--include-storage` to `normalize-assets` and select
`--portfolio generation-storage` for a generation-plus-storage result.

## Tests

```bash
pytest tests/test_public_tree.py tests/test_entrypoints.py powerworld/texas2k/tests -q
python scripts/smoke_test.py
python scripts/check_public_tree.py
```

## Sources

- PyPSA-USA: <https://github.com/PyPSA/pypsa-usa>
- Texas2k Series25: <https://electricgrids.engr.tamu.edu/texas2k-series25/>
- PYPOWER: <https://github.com/rwl/PYPOWER>

See `LICENSE.md` for the repository license.
