# Results and Provenance

Local evaluation performed on 2026-10-06, using the retained GT-volume CSV
and exactly 34 matching prediction IDs. MAPE is mean(abs(predicted-GT)/GT).
The scorer fails on incomplete IDs, duplicates, invalid values, and zero GT.

| File | All 34 | Public 10 | Private 24 |
| --- | ---: | ---: | ---: |
| `e5_w0.25.csv` | 0.19800894 | 0.20081697 | 0.19683894 |
| `sub_e5_rim.csv` | 0.17415237 | 0.16647616 | 0.17735079 |
| `sub_final_seeded.csv` | 0.17275345 | 0.16597508 | 0.17557776 |

## Hashes

- GT CSV SHA256: `0bc68472e05cadfa55fb107e740d053b7844dc581f81e30b1349289f0a6cf115`
- Final prediction SHA256: `b003f34a9d8936845a20421c4dc98dbfd9ce5c35c39fccccfcfedcb9bfaff49c`
- Rim prediction SHA256: `6352141ebc8c171baadf662814ecf125e5aebaadfc7865a952f20acf6b117a6d`
- Earlier prediction SHA256: `f8df453ee788fc56c2356738bbfc2778592561f0a8ee778b39ffa327015a7dd7`

The files themselves are excluded from this clean export. Hashes identify the
locally retained evaluation artifacts without distributing GT or dataset-derived
files. `sub_final.csv` is byte-identical to `sub_final_seeded.csv` locally.

## Historical Corrections

The old headline 0.198 refers to an earlier setting. The old best 0.1685 involved
unseeded sampling and is not used as the reproducible reference. Historical notes
recorded late-submission feedback, but the official submission metadata has not
been independently retrieved in this review.

## Benchmark Context

[Implicit-Scale 3D Reconstruction for Multi-Food Volume Estimation,
Table 2](https://arxiv.org/html/2602.13041v1#S4) reports MAPE 0.21 for MDMS,
0.31 for SGPS, 0.46 for PSHS, and 0.34 for GPT-5.2 on 24 objects.
Private-24 is the closest local subset for numerical context, not proof of an
identical controlled protocol. No competition win, new SOTA, independent held-
out validation, or superior geometric reconstruction is claimed.

The project used post-competition feedback and GT-informed crop selection.
Do not present these measurements as a leakage-free benchmark result.
