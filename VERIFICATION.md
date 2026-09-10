# Verification record

## Public Texas2k integration (2026-09-10)

Verified locally with Python 3.11.16, NumPy 1.26.0, SciPy 1.11.3, pandas
2.2.2, and PYPOWER 5.1.19.

- The committed input bundle validates as 2,751 buses, 1,099 generators,
  5,344 branches, 384 positive matched data-center rows totaling 39,864.607 MW,
  32,942.9 MW in the generation portfolio, and 39,992.7 MW in the
  generation-plus-storage portfolio.
- Full N-0 reruns reproduce the supplied collaborator outputs at unrounded
  precision: generation `NEW` = 157 upgrades and 11,005.415 million USD;
  generation-plus-storage `NEW` = 226 upgrades and 12,621.347 million USD.
- The locked-environment N-1 reruns each differ from the supplied output by one
  paid target branch. Generation produces 521 upgrades and 36,281.255 million
  USD, versus 522 and 36,336.3 million USD. Generation-plus-storage produces
  552 upgrades and 34,234.473 million USD, versus 553 and 34,289.5 million USD.
  The scenario bus, generator, and branch arrays were compared against the
  supplied scripts and are numerically identical. The single paid-target
  difference comes from singular/islanded sparse solves: branch 4878 is 78.4%
  in the current runs and 277.2% under contingency 4462 in the supplied output.
  Current runs record about 830 contingency diagnostics, retain all finite
  branch elements, warn once per sweep, and continue.
- A second version-sensitive branch appears only in the no-data-center N-1
  baseline (branch 5262); it does not become a new data-center target. This is
  why exact N-1 values must be reported with the locked environment and the
  supplied reference CSVs remain archived separately.
- The public-tree scanner checks tracked-like text, Python, shell, YAML, TOML,
  JSON, Markdown, CSV, and MATPOWER files without printing suspected secret
  values. Collaborator source workbooks and ZIP archives, license files, private
  API files, runtime results, and original collaborator folders remain outside
  Git. Existing upstream PyPSA input workbooks remain tracked under
  `workflow/repo_data/`.
- All 53 tests covering the public-tree guard, PyPSA-to-Texas2k entry points,
  and the Texas2k package pass. The synthetic N-1 run and normalized collaborator
  export also complete end to end. The broader upstream `pytest` selection has
  six environment/fixture failures: four require a locally installed GLPK
  executable, and two policy tests use a fixture without `rec_trading_zone`.
  These tests do not import the Texas2k package or the modified paper entry
  points.

Repeat the fast acceptance checks from the repository root:

```bash
pytest tests/test_public_tree.py tests/test_entrypoints.py powerworld/texas2k/tests -q
ruff check powerworld/texas2k scripts/check_public_tree.py tests/test_public_tree.py
python scripts/check_public_tree.py
python -m powerworld.texas2k.cli validate-inputs --json
```

## PyPSA paper workflow (2026-09-08)

Verified 2026-09-08 with Python 3.11.9, PyPSA 0.30.2 and Gurobi on OSU HPC.

- Paper-version retest `21219768`: COMPLETED, exit 0, 1 minute 55 seconds,
  all fourteen solves and downstream checks with Gurobi 13.0.1. The initial
  test below used Gurobi 12.0.2. An isolated overlay on the existing environment
  supplied Gurobi 13.0.1; a completely fresh installation was not used.
- Three entry-point tests passed, including all 755 unique matrix entries,
  original technology candidates and the paper's REP parameters.
- Updated dependency lock check and frozen installation dry-run succeeded.

- Synthetic integration job `21217138`: COMPLETED, exit 0, 1 minute 50 seconds,
  eecs partition. Fourteen optimizations including the input-network solve,
  baseline, all six modes for announced and single-location loads. The stressed
  fixture triggers new capacity. CCGT expansion remains disabled. Metrics,
  decomposition identity, REP summaries and a PNG were checked.
- Upstream Snakemake dry-run: exit 0, 21 jobs in the 2023 baseline DAG. The target
  must follow `--` so it is not interpreted as another configuration filename.
  This verifies dependency resolution, not live data acquisition or execution
  of the 21 full-scale jobs.
- Original-result plotting check: canonical historical result directories were
  read through temporary references. Plot generation completed with exit 0.
  It reproduced a 2.3960 cents/kWh pre-risk increment and 77.0:23.0 split;
  five-year capped wholesale increment 21.3446 USD/MWh. The original time index
  is a MultiIndex; the plotting entry handles it explicitly.
- The smoke test also verified local profile generation leaves the caller's
  NumPy random state unchanged. Published process-dependent seeds remain unknown.
- Python syntax, Shell syntax and public-tree heuristic checks are run separately.
  The public-tree check reports filenames/line numbers without printing values.

Not claimed: a fresh download/rebuild of all full-sized weather years, exact
reproduction of unarchived historical random profiles, execution of the separate
historical-price validation, or a PowerWorld Simulator `.pwb` automation run. The
historical plot check consumes existing scientific results; it does not replace
those tests. The Texas2k module reproduces the collaborator's MATPOWER/PYPOWER
screening path and does not require PowerWorld at runtime.

Raw solver logs, license messages, private inputs and large networks are excluded
from Git. Public checks can be repeated with `python scripts/check_public_tree.py`
using the required Python 3.11 environment.
