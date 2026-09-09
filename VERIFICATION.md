# Verification record

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
historical-price validation, or the collaborator's PowerWorld model. The historical
plot check consumes existing scientific results; it does not replace those tests.

Raw solver logs, license messages, private inputs and large networks are excluded
from Git. Public checks can be repeated with `python scripts/check_public_tree.py`
using the required Python 3.11 environment.
