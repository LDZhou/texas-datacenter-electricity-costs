# Texas2k engineering contribution

The collaborator contribution is implemented in [`texas2k/`](texas2k/). It
uses a PowerWorld Simulator-exported MATPOWER case with PYPOWER DC power flow to
screen intact (N-0) and single-branch-outage (N-1) loading, apply the supplied
fixed upgrade rules, and estimate engineering capital costs.

The workflow is an engineering screening heuristic. It does not automate the
PowerWorld application, solve an AC case, optimize a transmission plan, or
guarantee removal of every post-upgrade violation.

The PyPSA interface is a coordinate-enriched capacity table. New public PyPSA
runs write `bus_x` and `bus_y`; the Texas2k normalization command selects a year,
case, and location. The resulting CSV can be passed to `run --assets` without
moving or modifying the bundled Texas2k inputs.

The paper's engineering reference is the generation-plus-storage, N-1, `NEW`
row: 553 upgrades and 34,289.5 million 2024 USD. Its supplied verification still
contains 55 residual target-line violations and 173 residual grid violations.
These residuals are recorded and warned about; they remain part of the method's
reported limitation.

See [`texas2k/README.md`](texas2k/README.md) for installation, data definitions,
commands, output schemas, reference results, and citations.
