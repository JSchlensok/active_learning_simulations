# ColH homology search: Logan50 (MMseqs2) and PDB (Foldseek)

Sequence search of collagenase ColH against Logan50 and structure search of a ColH structure against the PDB,
run on 2026-10-29 on the HMGU cluster with the Heinzinger lab's shared tools and databases
(`/lustre/groups/shared/hei/{tools,databases}`).

## Query

ColH is the class II collagenase of *Hathewaya* (*Clostridium*) *histolytica* (AUX-II in the drug Xiaflex);
there is no human ColH.

| Input | Source | Details |
|---|---|---|
| `inputs/Q46085.fasta` | UniProt [Q46085](https://rest.uniprot.org/uniprotkb/Q46085.fasta) (COLH_HATHI) | full precursor, 1021 aa (signal 1–30, propeptide 31–40); entry version 143, sequence version 1, CRC64 `826F45EFD96F917C` |
| `inputs/9LRK.cif.gz` | RCSB [9LRK](https://files.rcsb.org/download/9LRK.cif.gz), model revision 1.1 (2026-07-01) | apo wild-type ColH, cryo-EM, 2.2 Å, [10.1038/s41467-026-71099-3](https://doi.org/10.1038/s41467-026-71099-3); chain A models UniProt 45–808 (activator and peptidase domains, PKD1) with Zn and Ca; PKD2 and the collagen-binding domain (809–1021) are unresolved |

9LRK is the highest-resolution full-length ColH entry without a bound substrate. Checksums (sha256):

```
5b1213f6a139ed79d18ed208dde6d6966871574b1ffdb7d067afc8794001510e  inputs/Q46085.fasta
d18ec19383d1ca665ec4b76e98af7c9edf10b93f9a2d3d19555f662cbc2e7b54  inputs/9LRK.cif.gz
```

## Layout

```
inputs/    query sequence and structure (scripts/fetch_inputs.sh)
scripts/   download, Slurm job, metadata and report scripts (resources and reasons in their headers)
outputs/   search and taxonomy results, tab-separated with a header line, and ENA metadata of the Logan50 source
           runs (git-ignored)
logs/      Slurm stdout/stderr, named <job name>_<job id>
report/    report template and the rendered report colh_homolog_report.html (git-ignored)
```

`outputs/` and the rendered report are not tracked; copies of the search outputs also remain in the cluster run
directory below.

## Commands

The jobs ran in `/lustre/groups/shared/hei/julius.schlensok/colh_homology_search/` (same layout), submitted
from `hpc-submit02`:

```bash
bash scripts/fetch_inputs.sh                                  # locally
R=/lustre/groups/shared/hei/julius.schlensok/colh_homology_search
rsync -a inputs scripts hpc-submit02:$R/
ssh hpc-submit02 "cd $R && mkdir -p logs && sbatch scripts/foldseek_pdb.sbatch"     # job 42125259
ssh hpc-submit02 "cd $R && sbatch scripts/mmseqs_logan50.sbatch"                     # job 42125387, crashed
ssh hpc-submit02 "cd $R && sbatch scripts/test_gpu_max_seqs.sbatch"                  # job 42126363
ssh hpc-submit02 "cd $R && sbatch scripts/mmseqs_logan50.sbatch"                     # job 42126564
ssh hpc-submit02 "cd $R && sbatch scripts/taxonomy_uniref50.sbatch"                 # job 42130720, --max-rejected 5
ssh hpc-submit02 "cd $R && sbatch scripts/test_taxonomy_unclassified.sbatch"         # job 42130807
ssh hpc-submit02 "cd $R && sbatch scripts/taxonomy_uniref50.sbatch"                 # job 42130846
ssh hpc-submit02 "cd $R && sbatch scripts/closest_uniref50.sbatch"                  # job 42130957
rsync -a hpc-submit02:$R/outputs hpc-submit02:$R/logs ./
python scripts/fetch_run_metadata.py                          # locally, needs polars
python scripts/build_report.py
```

Job 42130720 ran `taxonomy_uniref50.sbatch` without `--max-rejected`; its outputs carry the prefix `maxrej5_`.

Job 42125387 ran an earlier `mmseqs_logan50.sbatch` that called `mmseqs-logan50 easy-search` with
`--max-seqs 100000` directly (its log has the full command); the current script and `logan50_search.sh` ran
job 42126564.

## Searches

| | Foldseek | MMseqs2 |
|---|---|---|
| Query | `inputs/9LRK.cif.gz`, chain A (764 residues) | `inputs/Q46085.fasta` (1021 aa) |
| Database | PDB100 2025-01-01 (324,204 cluster representatives; predates 9LRK) | Logan50 non-human v1.0 (3.0 B sequences) |
| Tool | Foldseek 10-941cd33, CPU (only CPU searches return taxonomy) | MMseqs2 18-8cc5c, GPU via `mmseqs-logan50` (database copied to `/dev/shm`) |
| Parameters | defaults (3Di+AA, `-s 9.5`, `--max-seqs 1000`, E ≤ 10) | defaults (E ≤ 1e-3) with `--max-seqs 10000` (100000 crashed, see below) |
| Job | 42125259: `cpu_normal`, 8 cores, 16 GB; 27 s | 42126564: `gpu_normal`, 1 H100, 16 cores, 1000 GB; 1 h 08 min (52 min of it `convertalis` reading headers from lustre) |
| Output | `outputs/foldseek_pdb/9LRK_vs_pdb.tsv` | `outputs/mmseqs_logan50/Q46085_vs_logan50.tsv.gz`, `run_metadata.tsv` (ENA, 2026-10-29) |

### The `--max-seqs 100000` crash

Both Logan50 runs with `--max-seqs 100000` died in the GPU prefilter right after the database copy: job 42125387
on supergpu25 with `CUDA error: an illegal memory access was encountered : libmarv/src/dbbatching.cuh, line 69`,
job 42126564 on supergpu26 with `an illegal instruction was encountered` at the same line. The same search ran
fine against PDB seqres (77 hits) and UniRef50 (3,030 hits, against 300 with the default cap) in test job
42126363 (its `FAILED` state comes from the script's final `grep` matching nothing under `pipefail`, not from a
search), and the lab's Logan50 test passed with the default 300. So the crash needs Logan50 and a large cap; it
is not the node. Job 42126564 then fell back to `--max-seqs 10000`, which worked. libmarv (MMseqs2 commit
8cc5ce3) caps results at 512 × 1024 and stores database indices as `int32`; Logan50 has 3.0 B sequences, so
indices above 2^31 wrap, but they convert back correctly on the host and no device-side use of them turned up.
Root cause unknown; worth reporting upstream.

## Results

### Foldseek: 9LRK against the PDB

572 hits: 16 with E ≤ 1e-3, 50 with prob ≥ 0.9, 92 with alignment TM-score ≥ 0.5. Query positions below are
UniProt numbering (9LRK chain A residue 1 = UniProt 45).

- ColH's own crystal structures come back: the peptidase domain 4ARF (342–721, 99% identity, TM 0.99) and PKD1
  4U6T (725–808, 100%). Other ColH peptidase-domain entries (4AR1, 5O7E, 7ZOC) are cluster members of 4ARF in
  PDB100.
- The ColG collagenase units 2Y6I, 2Y50 and 4ARE align over 45–721 (47% identity, TM 0.67, E ~1e-60), the
  peptidase domains of ColG (7ZBV, 7Z5U) and ColT from *C. tetani* (4AR8) over 333–720 (52–56%, TM 0.97).
- The M9A collagenase units of *Grimontia* (*Vibrio*) *hollisae* (7XEB) and *Vibrio harveyi* (7VLZ) align
  over 45–599 (17% identity, TM ~0.70, E ~1e-16).
- The remaining hits are PKD-like domains on 725–808: ColH's own PKD2b (4JGU, 43% identity), ColG's PKD domain
  (2Y72, 4TN9, 4AQO), PKD1 of the *Vibrio* metalloproteases Epp and PrtV (6AEM, 4L9D), and weaker ones in other
  bacterial and human proteins (AAVR, KIAA1837).

### MMseqs2: ColH against Logan50

The prefilter returned its full 10,000 candidates and 7,879 passed E ≤ 1e-3, from 4,695 SRA runs. A hit is
assigned to a ColH domain when it spans at least half of it (see `scripts/build_report.py`).

- 6,995 hits (89%) cover only the PKD domains (PKD1 + PKD2: 4,579), 544 add the collagen-binding domain, 300
  cover part of the peptidase unit and 65 all six mature domains; 40 cover under half of any domain.
- The closest full-length relative is `SRR25704535_3472_2` (1038 aa, 58.1% identity over 99.9% of ColH); the
  next strongest are at 44–46%. No hit is ColH itself; the 91% hit is a 47-aa fragment.
- By ENA run organism, the hits come mostly from environmental metagenomes (sediment, soil, marine, hydrothermal
  vent). The full-length relatives come largely from animal genome runs (porpoise, seals, muskox, fish), i.e.
  bacteria sequenced with the host, and from *Clostridium*, *Bacillus* and *Brevibacillus* isolates.

The report (`report/colh_homolog_report.html`) has the coverage along ColH, domain combinations, identity
against coverage, source runs, taxonomy and the peptidase-unit hits in full.

### Taxonomy of the Logan50 hits

`easy-taxonomy` (approximate 2bLCA) of the 7,879 hit sequences against UniRef50 2026_03, job 42130846 (4.7 min,
`cpu_normal`, 16 cores, 160 GB), outputs `outputs/taxonomy_uniref50/logan50_hits_*`. 7,874 hits are assigned:
4,506 to Bacteria, 579 to Archaea, 25 to Eukaryota and 2,764 only to cellular organisms or root (PKD domains
occur in both Bacteria and Archaea).

- Full-length relatives: 42 of 65 in the Bacillota, mostly *Clostridium*; their closest UniRef50 entries are
  *Clostridium*, *Tissierella* and *Paenibacillus* proteins at a median 50% identity (31–100%, 10 of 65 at ≥ 90%).
- Peptidase-unit hits: Bacillota 62, Bacteroidota 31, Pseudomonadota 27, Spirochaetota 25 (all *Leptospira*,
  activator and catalytic subdomains only), Chloroflexota 21; 103 above phylum level.
- PKD-only hits: Bacteroidota, Actinomycetota, Pseudomonadota, Methanobacteriota (archaea) and others; 3,898 of
  6,995 above phylum level.

**`--max-rejected`.** With the taxonomy default of 5 (job 42130720), 1,630 hits stayed unclassified without any
UniRef50 hit, although a plain search finds strong matches for them (job 42130807: 50 such hits, 300 matches
each, best E-values down to 1e-298, none in a blacklisted taxon). The approximate 2bLCA (`lcaalign` in
`src/alignment/Alignment.cpp`) re-aligns the prefilter candidates against the top hit's region in prefilter order
and stops after `--max-rejected` consecutive failures, often before it reaches the top hit itself. With
`--max-rejected 2147483647` (job 42130846) 5 hits stay unclassified; 1,805 of the 6,249 earlier assignments
change, mostly to broader ranks, because the truncated candidate lists had made them too specific.

**Identities.** The taxonomy workflow aligns in score-only mode, so the identities in `_tophit_aln` are
estimates (e.g. 97.3% for a hit that is 58.1% identical to its match). The closest UniRef50 entry of each hit and
its identity come from a plain `easy-search` (job 42130957, `logan50_hits_closest_uniref50.tsv`); the identity is
to the cluster representative, so another cluster member can be closer.

## Caveats

- The Logan50 list holds the 10,000 best prefilter candidates only; weaker homologs beyond them are missing.
- Logan50 clusters proteins at 50% identity: hits are cluster representatives, so counts measure diversity, not
  abundance, and the run named in a hit is the representative's.
- A run's ENA organism describes the sample, not necessarily the gene (bacterial genes in animal genome runs).
- Lowest-common-ancestor assignments are conservative: a hit whose UniRef50 matches span phyla is assigned above
  them, even when its single closest match is a named genus.
- Foldseek's prefilter filled its default cap of 1000 candidates, so weak remote hits beyond the top 1000 may be
  missing; the significant hits rank well inside it.
- The GPU prefilter of MMseqs2 differs from the CPU one (it finds somewhat more hits).
- 9LRK lacks PKD2 and the collagen-binding domain, so the structure search finds homologs of those only through
  their similarity to resolved domains (4JGU via PKD1); the sequence search covers the full length.
- 2,849 Logan50 records have an empty sequence (an upstream FASTA defect, see the database README) and can never
  be hit.
