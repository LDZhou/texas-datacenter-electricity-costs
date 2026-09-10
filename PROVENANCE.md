# Provenance and scope

Source: the author's original PyPSA-USA project, whose upstream checkout HEAD
was `dd89d47b55100d3141e64d4d57310fc9a482ddc5` when copied. The source also
contained untracked paper scripts and local edits; that commit alone cannot
reconstruct the paper extension. This independent repository records the selected
files. Upstream project: https://github.com/PyPSA/pypsa-usa . Retained upstream
code and metadata carry their original MIT license and attribution.

The later `pypsa-usa-paper-v1` CCGT revision is not the source of this package.
No original repository history, credential file, logs, network outputs or private
data-center workbook is copied into Git. Upstream PHS geospatial layers are omitted
because the original experiment has only four-hour battery candidates; PHS is
outside scope. The rest of the upstream rule/script tree is retained because
Snakemake imports shared electricity, sector, retrieval and validation helpers.

Cleanup changes:

- Replace twelve numbered Slurm launchers with `slurm/experiment.sbatch` and a
  deterministic mode/year/location/scale entry point.
- Explicit `workflow/Snakefile` baseline entry, no automatic deletion/unlock of
  shared intermediates, sequential baseline array.
- Replace process-randomized load seeds with SHA-256; preserve per-seed legacy
  Gaussian noise using a local RandomState. Published realizations may differ.
- Canonical result selection excludes legacy, full-year and fuel-sensitivity roots.
- REP parameters match the paper-used `cap5000_n1` results, including the external
  flat-load transmission adder 2.751319804231755 USD/MWh.
- Abort non-optimal/non-finite solves before writing apparently successful outputs.
- Move personal credentials and paths out of runnable defaults.
- Package the collaborator's PowerWorld-exported Texas2k Series25 workflow as
  `powerworld.texas2k`, with PYPOWER DC N-0/N-1 screening, semantic module
  names, CSV inputs, stable branch identities, warning-only residual records,
  and supplied reference outputs.
- Add `bus_x` and `bus_y` to new PyPSA generation/storage capacity exports so a
  normalized asset table can enter Texas2k without a private enrichment workbook.
- Align the Gurobi dependency lock with the paper's 13.0.1 version (the old lock
  named 11.0.3; the original available runtime actually imported 12.0.2).
  Update the yanked transitive ConfigArgParse 1.7 release to 1.7.7.

Historical checks before cleanup: 2023 capped mean LMP 25.4038 versus 43.8619
USD/MWh; REP failure probabilities 8.96% (full_tx) and 11.14% (generation_storage);
engineering N-1 new-violation cost 34,289.5 million USD. These are reference values,
not outputs of the synthetic integration test.

The price decomposition retains the paper's minimum clipped nodal price reference.
That proxy is not an independently identified marginal generator. Packaging does
not validate this interpretation, the price-cap convention, or other scientific
assumptions. No new scientific sensitivity is implied by the cleanup.

Reproduction boundaries: historical PyPSA starting-network release is pending;
live EIA and weather downloads are not tested by the synthetic smoke test. The
2021 full-year validation requires separately supplied data. The Texas2k module
includes its synthetic case and engineering inputs; its full N-1 calculation is
kept out of the fast test suite. See `VERIFICATION.md` for tests actually completed.
