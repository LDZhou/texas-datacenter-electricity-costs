# Texas data center electricity costs

Code and inputs for the paper *Spatial grid constraints shape household
electricity costs from data-center growth*. The experiments connect PyPSA-USA
capacity expansion and wholesale prices to Texas2k transmission reinforcement
and residential electricity costs.

```text
Data-center locations and demand profiles
                 ↓
PyPSA generation, storage and transmission planning
                 ↓ supply additions and coordinates
Texas2k DC contingency screening and reinforcement
                 ↓ annualized transmission charge
Wholesale-price attribution and retail-provider exposure
```

## Layout

- `workflow/`, `config/`: PyPSA-USA input-building workflow.
- `configs/`, `paper_pipeline/`: paper scenarios, planning, matched analysis,
  demand profiles and final figure scripts.
- `texas2k/`: PYPOWER screening, branch reinforcement, inputs and reference runs.
- `scripts/`: public commands and planning-to-engineering asset export.
- `slurm/`: CPU job launchers.
- `reference_data/`: accepted wholesale, siting and retailer result tables.
- `results/`: locally generated networks, analyses and figures, excluded from Git.

## Installation

Use Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/LDZhou/texas-datacenter-electricity-costs.git
cd texas-datacenter-electricity-costs
uv sync --frozen --extra dev
source .venv/bin/activate
mkdir -p logs
```

The locked environment includes PyPSA 1.3.0, Linopy 0.9.1, Gurobi 11.0.3 and
PYPOWER 5.1.19. Set `GRB_LICENSE_FILE` to your local Gurobi licence file. EIA
input downloads require your own API key:

```bash
cp config/config.api.example.yaml config/config.api.yaml
```

Enter the key in that local file. Network files can embed their input
configuration and must remain outside Git.

## Planning inputs

Build the five weather-year base networks through the PyPSA-USA workflow:

```bash
sbatch --account=ACCOUNT --partition=ampere --array=0-4%2 slurm/base.sbatch
```

Input downloads and renewable resource construction follow the source URLs
and settings in `workflow/` and `configs/config_vopt.yaml`. If these base
networks already exist in another checkout, supply its path with
`--historical-root` during preparation.

After the base jobs finish:

```bash
python scripts/run_paper.py matrix
python scripts/run_paper.py prepare --historical-root .
```

Preparation freezes the historical capacities, persists common demand
profiles, and applies the gas-fuel outlier rule for 2021. Each comparison
uses the same starting network and profile with the stated expansion
permissions. All commands accept `--run-root` (default `results/paper`);
`--dry-run` prints the commands without executing them.

## Planning experiments

Run the arrays below sequentially. Keep at most two optimization jobs active
across all arrays when using a two-session Gurobi licence.

```bash
sbatch --account=ACCOUNT --partition=ampere --array=0-5%2 slurm/paper.sbatch central
sbatch --account=ACCOUNT --partition=ampere --array=0-23%2 slurm/paper.sbatch weather
sbatch --account=ACCOUNT --partition=ampere --array=0-374%2 slurm/paper.sbatch siting
```

`central` contains the 2023 announced-development cases and matched baselines;
`weather` repeats them for 2019–2022. `siting` contains the six locations,
four load scales, three infrastructure pathways and their matched baselines
across all five weather years. The main inventory contains 405 cases.
Optional diagnostic cases require their own prepared full-year bases and
controlled sensitivity inputs. Generate that separate inventory with
`python -m paper_pipeline.scripts.revision_matrix --include-diagnostics --output PATH`.

Run an individual inventory entry locally with:

```bash
python scripts/run_paper.py run-case --group central --index 0
```

The launchers use CPU resources. Supported OSU partitions are `ampere`, `eecs`,
`share`, and CPU nodes in `preempt`.

## Engineering reinforcement

The included asset table contains the matched 2023 coordinated-expansion
portfolio. Validate it and screen the engineering network:

```bash
python -m texas2k.cli validate-inputs --json
python -m texas2k.cli run --portfolio generation --criterion n1
python -m texas2k.cli export --run-dir results/texas2k/generation/n1 \
  --portfolio generation --criterion n1
```

To pass your generated planning portfolio into this stage, use
`scripts/export_pypsa_assets.py` and `--assets`, as described in
[texas2k/README.md](texas2k/README.md). Engineering investment is annualized at
8% over 40 years and allocated using four coincident peaks. Planning capital
costs are not charged again.

## Customer analysis and figures

After the planning experiments finish:

```bash
python scripts/run_paper.py analysis
python scripts/run_paper.py rep
python scripts/run_paper.py figures
```

Matched price and unserved-energy tables are written under
`results/paper/analysis_view/`; retailer results under
`results/paper/results/retailer/`; figures under
`results/paper/results/figures_final/`. Retail simulations use 5,000 paths per
zone, seed 123, the no-data-center dispatch baseline for forward procurement,
a $5,000/MWh price cap and the matched flat-load transmission
charge of $2.601/MWh. The residential example uses twice that flat-load
allocation. Edit the explicit transmission-charge input when evaluating a
different engineering case.

Accepted result tables and hashes are documented in
[reference_data/README.md](reference_data/README.md) and
[texas2k/README.md](texas2k/README.md). Solver and numerical environment details
are retained in generated run manifests.

## Checks

Input and command checks can run without optimization:

```bash
python scripts/check_public_tree.py
python -m texas2k.cli validate-inputs --json
python -m pytest tests/test_entrypoints.py tests/test_public_tree.py \
  texas2k/tests/test_inputs.py texas2k/tests/test_package.py \
  texas2k/tests/test_exports.py -q
```

The full unit and integration suites include solver calls; select them when
performing a numerical reproduction.

## Sources and licence

- [PyPSA-USA](https://github.com/PyPSA/pypsa-usa)
- [Texas2k Series25](https://electricgrids.engr.tamu.edu/texas2k-series25/)
- [PYPOWER](https://github.com/rwl/PYPOWER)

Repository code is distributed under [LICENSE.md](LICENSE.md). Texas2k source
acknowledgements are in [texas2k/NOTICE.md](texas2k/NOTICE.md).
