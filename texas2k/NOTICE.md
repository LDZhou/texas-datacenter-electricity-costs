# Texas2k sources and citations

## Source-code mapping

The public module uses descriptive filenames in place of the collaboration's
date-prefixed scripts:

| Original script | Public module or command |
| --- | --- |
| `0716_fix_unmapped_bus_coords.py` | `inputs/baseline/buses.csv` |
| `0717_01_dc_node_csv.py` | `data.py` |
| `0731_00_inject_base.py` | `case.py`, `dispatch.py`, `upgrades.py` |
| `0731_01_full_n0.py` | `run --portfolio generation --criterion n0` |
| `0731_02_full_n1.py` | `run --portfolio generation --criterion n1` |
| `0731_10_genstorage_base.py` | `dispatch.py` |
| `0731_11_genstorage_n0.py` | `run --portfolio generation-storage --criterion n0` |
| `0731_12_genstorage_n1.py` | `run --portfolio generation-storage --criterion n1` |
| `0731_20_export_upgrade_csvs.py` | `cli.py export` |

## Texas2k

The Texas2k Series25 case is published by the Texas A&M Electric Grid Test Case
Repository:

<https://electricgrids.engr.tamu.edu/texas2k-series25/>

Requested citations:

- A. B. Birchfield, T. Xu, K. M. Gegner, K. S. Shetye, and T. J. Overbye,
  “Grid Structural Characteristics as Validation Criteria for Synthetic
  Networks,” *IEEE Transactions on Power Systems*, 32(4), 3258–3265, 2017.
- J. Baek and A. B. Birchfield, “A tuning method for exciters and governors in
  realistic synthetic grids with dynamics,” *2023 North American Power
  Symposium*, 1–6, 2023.

## Software

- PYPOWER: <https://github.com/rwl/PYPOWER>
- MATPOWER: <https://matpower.org/>
- PyPSA-USA: <https://github.com/PyPSA/pypsa-usa>

PYPOWER uses the BSD 3-Clause license. Repository code uses the root MIT
license.
