# FLIP2 Data

Protein fitness landscape splits from [FLIP2](https://flip.protein.properties) (Didi et al., ICML
2026), the successor to the original FLIP benchmark. Two of its seven datasets are downloaded here:
**TrpB** (enzyme fitness, continuous) and **NucB** (nuclease activity, ordinal bins).

Unlike [../combingym/](../combingym/), FLIP2 has no API: the splits are static files on GitHub
Pages. The script still discovers them by parsing the landing page rather than hardcoding paths, so
`--list` reports whatever FLIP2 currently publishes.
## Downloading

```bash
cd datasets/raw/flip2
uv run python download_flip2.py                         # trpb + nucb, all their splits
uv run python download_flip2.py --list                  # all 7 datasets / 16 splits
uv run python download_flip2.py --datasets ired pdz3    # fetch others
uv run python download_flip2.py --datasets trpb --splits by_position
```

## Contents

| Dataset | Splits | Rows | Seq length | Target | License |
| --- | --- | --- | --- | --- | --- |
| `trpb` | `one_to_many`, `two_to_many`, `by_position` | 228 298 | 389 | continuous fitness, −0.801 – 2.451 | CC0 |
| `nucb` | `two_to_many` | 55 759 | 142 | ordinal activity bin, 0 – 3 | CC-BY 4.0 |

* **TrpB** — β-subunit of tryptophan synthase; growth-based fitness over combinatorially complete
  landscapes. Exactly **20 variable positions** (104–108, 117–119, 162, 166, 182–186, 227, 228, 230,
  231, 301), all 20 amino acids observed at each, forming ten sub-landscapes.
  Johnston et al. 2024, *PNAS* 121(32):e2400439121.
* **NucB** — *Bacillus licheniformis* nuclease B activity at pH 7, from a four-round ML-guided design
  campaign. 118 of 142 positions vary; variants carry 1–23 substitutions.
  Thomas et al. 2024, bioRxiv 2024.03.21.585615.

## CSV format

Four columns, gzipped (`pandas.read_csv` handles `.gz` directly):

| Column | Meaning |
| --- | --- |
| `sequence` | full variant sequence (fixed length per dataset; substitutions only) |
| `target` | measured fitness — continuous for TrpB, ordinal bin for NucB |
| `set` | `train` or `test` |
| `validation` | bool; a flagged subset *of* `train`, not a third disjoint set |

**Column order differs between datasets** — NucB is `sequence,target,set,validation` while TrpB is
`sequence,set,target,validation`. Read by name, never by position.

Split sizes:

| File | train | test | validation |
| --- | --- | --- | --- |
| `trpb/one_to_many.csv.gz` | 380 | 227 918 | 76 |
| `trpb/two_to_many.csv.gz` | 10 791 | 217 507 | 2 158 |
| `trpb/by_position.csv.gz` | 39 446 | 188 852 | 7 889 |
| `nucb/two_to_many.csv.gz` | 7 455 | 48 304 | 5 367 |

> **Warning:** `nucb/two_to_many` is not the mutation-count split its name and FLIP2's description
> ("train on 0, 1 or 2 mutations") imply — 4 326 of its 7 455 training variants carry more than two
> substitutions and 9 936 test variants carry two or fewer.

### NucB target encoding

`target` is an **ordinal scale**, not a raw measurement. Verified by joining on `sequence` against
the upstream `landscape.csv`:

| `target` | Upstream `activity_level` | Rows | Share |
| --- | --- | --- | --- |
| 0 | `non-functional` | 33 890 | 60.8 % |
| 1 | `activity_greater_than_0` | 11 098 | 19.9 % |
| 2 | `activity_greater_than_WT` | 10 572 | 19.0 % |
| 3 | `activity_greater_than_A73R` | 199 | 0.36 % |

A73R is a known improved single variant, so bin 3 is "better than the best previously known
variant" — 199 sequences out of 55 759.

## Compiling

```bash
uv run datasets/engineering/raw/flip2/compile_flip2.py
uv run datasets/dataset_preprocessor.py
```

[compile_flip2.py](compile_flip2.py) writes `engineering/raw/flip2_<name>.fasta`; the shared
preprocessor then writes `engineering/flip2_<name>_max2000.fasta`, which is what
`ALSimulatorDataset.FLIP2_*` points at.

**Splits are dropped.** The `set` and `validation` columns are discarded and every row kept. These
simulations mask all labels and replay a campaign over the whole pool, so FLIP2's supervised boundary
is not meaningful here — the equivalent notion lives in [al_splits.py](../../../../al_splits.py) and is
applied at simulation time. For TrpB that also means only one of the three split files is read, since
all three carry identical `sequence` and `target` columns row for row.

### Resulting FASTA Files

| FASTA | Sequences | Length | Label |
| --- | --- | --- | --- |
| `flip2_TrpB_max2000.fasta` | 228 298 | 389 | continuous, −0.801 – 2.451 |
| `flip2_NucB_max2000.fasta` | 55 759 | 142 | ordinal bins `0`–`3` |

The `_max2000` filter drops nothing. Verified after compiling: no non-standard residues, uniform
length, unique ids, labels parse (NucB written as integers, not `2.0`).

### Sequence Ids And The Parent Sequence

FLIP2 ships **no parent sequence** for either dataset, so both are reconstructed and written out, which
is what makes the substitution-based split axes usable.

| Dataset | Parent | Ids | Why |
| --- | --- | --- | --- |
| `TrpB` | **Tm9D8∗**, written to `trpb/TrpB_wt.fasta` | substitutions, `TrpB_E105A:T106A:G107A` | ≤ 4 substitutions per variant, so ids reach only 28 characters |
| `NucB` | wild-type NucB, written to `nucb/NucB_wt.fasta` | source row index, `NucB_00000` | up to 23 substitutions per variant, which would run to 120 characters |

Having a parent is no longer what decides the id scheme — both datasets have one now — the id length is.

**TrpB's parent is Tm9D8∗**, an engineered *Thermotoga maritima* TrpB variant, **not** wild-type TmTrpB.
Johnston et al. 2024: *"Tm9D8∗ differs from wildtype TmTrpB by ten amino acid substitutions (P19G, E30G,
I69V, K96L, P140L, N167D, I184F, L213P, G228S, and T292S)."* It is reconstructed from UniProt
[`P50909`](https://www.uniprot.org/uniprotkb/P50909) (`TRPB1_THEMA`, 389 aa) with those ten applied, and
is byte-identical to the sequence the landscape itself implies. Two of the ten — `I184F` and `G228S` —
fall inside the varied positions, so the library re-randomizes sites that had already been engineered.

It could **not** have been recovered by per-position consensus: positions 183, 184, 227 and 228 form the
4-site saturation library, so every residue is heavily represented there (~30 % modal share) while the
sparse 3-site landscapes sit above 90 %. `consensus_sequence` refuses such a pool rather than returning an
arbitrary sequence.

**NucB's parent** comes from the [upstream record](https://storage.googleapis.com/nuclease_design/processed_data/landscape.csv)
(`num_mutations == 0`), since it is genuinely absent from the distributed FLIP2 CSV.