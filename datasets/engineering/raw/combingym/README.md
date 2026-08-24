# CombinGym Data
Combinatorial mutagenesis landscapes from [CombinGym](https://combingym.org), a benchmark of
*combinatorial* (multi-site) protein variants
Not on HuggingFace, but served by the CombinGym website itself, so [download_combingym.py](download_combingym.py) fetches
them from its JSON API.

## Downloading

```bash
cd datasets/raw/combingym
uv run python download_combingym.py                        # WT FASTA + DMS xlsx for the 5 proteins below
uv run python download_combingym.py --list                 # all 11 datasets CombinGym currently offers
uv run python download_combingym.py --proteins CH65 Spike  # fetch others
uv run python download_combingym.py --include msa structure  # also grab the .a2m MSA and AlphaFold .pdb
```

Every download is recorded in [metadata.json](metadata.json) with the
server-side file name, size and sha256, plus the full overview table row (DMS sites, variant space,
literature reference, MSA depth) because CombinGym's file names carry an upload timestamp and are not stable.

## Compiling
```bash
uv run datasets/engineering/raw/combingym/compile_combingym.py
uv run datasets/dataset_preprocessor.py
```
### Label Columns

| Dataset | Column | Reason |
| --- | --- | --- |
| `GB1` | `fitness` | the only label column in the file |
| `CreiLOV` | `mean` | mean over the three fluorescence replicates. The `*_log` columns are the same measurement log-scaled; the linear mean keeps the label in assay units |
| `CR9114` | `h1_mean` | H1 only. Per the CombinGym paper: *"For CR9114, we utilized only the H1 landscape because H3 and influenza B landscapes were mainly characterized by low fitness values, limiting the information content for model training."* Matches the single `binding_H1` phenotype in CombinGym's leaderboard |
| `mTagBFP2` | `combined` | the joint blue/red phenotype rather than either channel alone |
| `SaCas9` | `Mean` | mean over the three sgRNAs; CombinGym likewise reports one `activity` phenotype |

All five are "higher is better" (binding affinity, fluorescence, nuclease/nickase activity), so the
registry gives them all `MAXIMIZE`.

### Sequence Processing

| Step | Effect |
| --- | --- |
| strips CreiLOV's trailing `*` | 165 428 sequences, now 119 residues and matching the parent |
| drops rows with no label | CR9114 only: 442 of 65 536 (`h1_mean` NaNs) |
| derives sequence ids from substitutions | `GB1_V39A:D40C`, or `<name>_WT` for the parent |
| validates | all lengths equal the parent's, ids unique, else it raises |

### Resulting FASTA Files
| FASTA | Sequences | Length | Label range |
| --- | --- | --- | --- |
| `combingym_GB1_max2000.fasta` | 149 361 | 56 | 0.000 – 8.762 |
| `combingym_CreiLOV_max2000.fasta` | 165 428 | 119 | 620.276 – 15 686.305 |
| `combingym_CR9114_max2000.fasta` | 65 094 | 121 | 7.000 – 9.835 |
| `combingym_mTagBFP2_max2000.fasta` | 8 192 | 233 | 0.155 – 1.692 |
| `combingym_SaCas9_max2000.fasta` | 1 296 | 1053 | −0.380 – 1.381 |

The `_max2000` length filter drops nothing since every variant is far below 2000 residues. Verified after
compiling: no non-standard residues, no unparseable labels, no duplicate ids, one sequence length per
file.

Both the compiled FASTAs and their intermediates are **gitignored**.

## Raw Contents

| Protein | Property | WT length | DMS sites | Variant space | Measured | Reference |
| --- | --- | --- | --- | --- | --- | --- |
| `GB1` | Binding | 56 | 4 | 160 000 | 149 361 (93.4 %) | Wu et al. 2016, *eLife* 5:e16965 |
| `CreiLOV` | Fluorescence | 119 | 15 | 184 320 | 165 428 (89.8 %) | Chen et al. 2023, *ACS Synth. Biol.* 12(5):1461 |
| `CR9114` | Binding | 121 | 16 | 65 536 | 65 535 (100 %) | Phillips et al. 2021, *eLife* 10:e71393 |
| `mTagBFP2` | Fluorescence | 233 | 13 | 8 192 | 8 192 (100 %) | Poelwijk et al. 2019, *Nat. Commun.* 10:4213 |
| `SaCas9` | Enzymatic activity | 1053 | 8 | 1 296 | 1 296 (100 %) | Thean et al. 2022, *Nat. Commun.* 13:2219 |

The exact reference string CombinGym ships is in `metadata.json` under `browse_row.dmsRef`.

Layout per protein (timestamps dropped from the server file names):

```
<protein_id>/
..<protein_id>_wt.fasta # wild-type sequence used for library construction
..<protein_id>_dms.xlsx # the measured landscape
```

### Excel Schema

One sheet each (the sheet name varies: `Sheet1`, `GB1_clean`, `CR9114`, `eqFP611`). Common columns:

* `genotype` — the **full variant sequence**, not a mutation string. All rows have the WT length, so
  the landscapes are substitution-only.
* `n_mut` — number of substitutions vs. WT; `n_mut == 0` is the WT row (present in all five).
* `mutant` — only in `CreiLOV` (HGVS-ish, `p.Gly3Glu`) and `mTagBFP2` (the concatenated variable
  residues, `DVLTFNSALYNNK`).

### Labels
The label columns differ per dataset, because several proteins were measured against multiple
targets. CombinGym calls each one a **phenotype** (see [Phenotypes and splits](#phenotypes-and-splits) below):

| Protein | Phenotypes | Label columns | Range | Missing |
| --- | --- | --- | --- | --- |
| `GB1` | `binding` | `fitness` | 0 – 8.76 | none |
| `CreiLOV` | `fluorescence` | `Rep1..Rep3`, `mean`, `Rep1_log..Rep3_log`, `mean_log` | 543 – 17 348 (log: 2.74 – 4.24) | none |
| `CR9114` | `binding` to H1, H3 and influenza B antigen subtypes | `{h1,h3,fluB}_{repa,repb,repc,mean,sem}` | 6.0 – 10.1 (log10 K<sub>D</sub><sup>-1</sup>) | yes, see below |
| `mTagBFP2` (benchmarked as `eqFP611`) | `fluorescence` | `blue`, `red`, `combined` | 0.03 – 1.69 | none |
| `SaCas9` | `activity` | `sg1 (PAM: ACAAGT)`, `sg2 (PAM: GGTGGT)`, `sg3 (PAM: TGGAGT)`, `Mean` | −0.64 – 1.54 | none |

### Issues in `CR9114`
**Censoring & Missing Data**
* **The floor is how a non-binder is encoded.** Affinities are −log₁₀ K_D, and inferred K_D outside
  the titration boundaries are *pinned* to them — 10⁻⁷ M → **7.0** for H1, 10⁻⁶ M → **6.0** for
  H3/fluB. The authors state this explicitly and call the pinned values biased: "we believe it is
  not meaningful to assign −logK_D values lower than the boundary […] Rather than discard these
  variants altogether, we choose to preserve some measure of their low binding affinity (albeit
  biased) by assigning them the boundary value." So floor values are **left-censored, not missing**
  — 1 675 rows (2.6%) sit at 7.0 for H1, against 58 361 (89.1%) at 6.0 for H3 and 65 336 (99.7%)
  for fluB. That distribution is exactly why CombinGym uses H1 only.
* **A NaN is a QC failure, not a non-binder.** Replicates whose fit was untrustworthy
  (sd(log₁₀ K_D) > 1.0 or r² < 0.8) were removed before averaging. `{target}_mean` is NaN **iff all
  three replicates were removed** — verified here: 442/442 for h1, 1/1 for h3, 2/2 for fluB, no
  exceptions either way. Such a variant could be high *or* low affinity; nothing was measured

**Wrong Row Count**
- CR9114's row count as a problem: CR9114 has 65 536 rows against vs. the website's `dmsMeasured = 65535`.

**Undocumented Filtering**

The `dmsDataset` xlsx downloaded here is the **full** 65 536-variant H1 landscape (65 094 after dropping NaNs).
CombinGym's own benchmark runs on `Data/DMS/Clean/bnAbs_CR9114_H1_clean.xlsx` in
[its repository](https://github.com/sitonglab/CombinGym), which has **48 841** rows. 
From reverse-engineering, this is the raw landscape, minus all **16 384** variants carrying *both* `F29S` *and* `Y106S`,
minus the **311** remaining NaN-`h1_mean` rows.

This is **documented nowhere** and both sites are ordinary library positions in Phillips et al.
As a consequence, our split sizes are different from theirs, and their split sizes are inconsistent:
The website's browse API reports `dmsMeasured = 65535`, `dmsMeasPerc = 1.0` for CR9114, while `Data_summary.xlsx` in the same project
says `48841` and `0.745255`.


## Mutation-Order Splits

| Split | GB1 | CreiLOV | CR9114 | mTagBFP2 | SaCas9 |
| --- | --- | --- | --- | --- | --- |
| `0-vs-rest` | 1 / 149 360 | 1 / 165 427 | 1 / 65 535 | 1 / 8 191 | 1 / 1 295 |
| `1-vs-rest` | 77 / 149 284 | 21 / 165 407 | 17 / 65 519 | 14 / 8 178 | 13 / 1 283 |
| `2-vs-rest` | 2 168 / 147 193 | 197 / 165 231 | 137 / 65 399 | 92 / 8 100 | 75 / 1 221 |
| `3-vs-rest` | 28 187 / 121 174 | 1 175 / 164 253 | 697 / 64 839 | 378 / 7 814 | 255 / 1 041 |

(train / test. `RhlA` has only three splits, having no variants beyond triples.)

All splits can be derived from `n_mut`.