# Texas2k engineering workflow

The [`texas2k/`](texas2k/) package connects PyPSA capacity-expansion results to
the Texas2k Series25 transmission system.

It performs:

1. PyPSA generation and storage capacity mapping;
2. data-center load mapping to Texas2k buses;
3. DC N-0 or exhaustive branch-outage N-1 screening;
4. fixed transmission upgrade sizing and cost calculation;
5. normalized CSV export for the paper analysis.

Quick start:

```bash
python -m powerworld.texas2k.cli validate-inputs
python -m powerworld.texas2k.cli run \
  --portfolio generation-storage \
  --criterion n1
```

See [`texas2k/README.md`](texas2k/README.md) for inputs, commands, output files,
and the PyPSA interface.
