# Texas2k contribution notice

This directory packages code and data supplied by the paper collaboration for
public reproduction. The collaborator archive was reviewed locally and remains
outside Git. Its SHA-256 digest is
`993cd53a248e9a39a234163e2a2386198f97cd183c91b9000a8ab4ec44347957`.

The public package removes personal paths, Windows-only separators, workbook
metadata, generated ZIPs, redundant backups, in-place input mutation, global
warning suppression, and date-prefixed execution names. CSV reference outputs
retain the supplied numerical results. `branch_id` replaces `row_idx` as the
stable public branch identifier.

## Source-to-module mapping

| Supplied file | Public location or replacement |
| --- | --- |
| `0716_fix_unmapped_bus_coords.py` | correction embodied in `inputs/baseline/buses.csv`; no in-place patch script retained |
| `0717_01_dc_node_csv.py` | normalized project and bus-match CSV inputs plus `data.py` validation |
| `0731_00_inject_base.py` | `case.py`, `dispatch.py`, `upgrades.py` |
| `0731_01_full_n0.py` | `workflow.py --portfolio generation --criterion n0` |
| `0731_02_full_n1.py` | `workflow.py --portfolio generation --criterion n1` |
| `0731_10_genstorage_base.py` | `dispatch.py` generation-storage portfolio |
| `0731_11_genstorage_n0.py` | `workflow.py --portfolio generation-storage --criterion n0` |
| `0731_12_genstorage_n1.py` | `workflow.py --portfolio generation-storage --criterion n1` |
| `0731_20_export_upgrade_csvs.py` | `exports.py` and `cli.py export` |

The source archive called these calculations PowerWorld analyses. The runnable
code uses a MATPOWER case exported from PowerWorld Simulator and calls PYPOWER
`rundcpf`; it contains no SimAuto/COM automation and does not open a `.pwb` case.

## Third-party data

The included Texas2k Series25 case comes from the Texas A&M Electric Grid Test
Case Repository: <https://electricgrids.engr.tamu.edu/texas2k-series25/>. The
repository describes the case as entirely synthetic, offers it free for
commercial and non-commercial use, and requests dataset registration and
academic citations. The module README lists the requested papers.

PYPOWER is distributed under the BSD 3-Clause license. The repository's source
code remains under the root MIT license. Dataset-specific attribution and
citation requests continue to apply to the Texas2k case.

The contribution is attributed to the paper collaboration as a team. Preferred
individual names, affiliations, and identifiers can be added to `CITATION.cff`
when the authors provide the public citation form.
