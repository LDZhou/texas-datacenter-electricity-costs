# PowerWorld contribution and interface

This directory is reserved for the collaborator's Texas2k AC power-flow and
contingency analysis. PowerWorld models and automation are not included yet.

PyPSA output for the matched case is
`results/dc_experiments_full_tx/2023/all2030_full_tx/`:
`network.nc`, `new_capacity.csv`, and `dc_bus_mapping.csv`.
`scripts/export_all_expansion_tx_detail.py` exports additional expansion details.
The collaborator should document bus mapping, units, existing-unit treatment,
the no-data-center attribution baseline, and N-0/N-1 screening separately.

Return an engineering summary with columns:
`file,criterion,recover_mode,n_upgraded,cost_musd,cost_busd,cleared,residual`.
`cost_musd` is million 2024 USD; `cost_busd` is billion 2024 USD.
The original paper used `n1_recoverNEW`, 34,289.5 million USD. This is an
external reference value, not a cost recalculated by the PyPSA smoke test.
The associated flat-load transmission adder was 2.751319804231755 USD/MWh;
the paper's residential shape multiplier is 2.

Keep credentials, proprietary cases, and data without redistribution permission
outside Git. Add a data-access README and small synthetic examples where possible.
