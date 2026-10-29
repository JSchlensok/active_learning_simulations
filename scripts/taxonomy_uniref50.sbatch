#!/bin/bash
# Assigns a taxon to each Logan50 hit: MMseqs2 easy-taxonomy (approximate 2bLCA) of the hit sequences against
# UniRef50 2026_03 with taxonomy, since a hit's SRA run names the sample, not the organism the gene comes from.
# CPU with the precomputed index: the lab's README needs ~120 GB to read the 108 GB index with --db-load-mode 1
# (~3 min); its 10-query test took 204 s. 7,879 queries add prefilter and alignment work, so 16 cores, 160 GB and
# 3 h leave headroom. GPU would scan all of UniRef50 once per query, which takes hours for 7,879 queries.
# Submit from the experiment root: sbatch scripts/taxonomy_uniref50.sbatch
#SBATCH --job-name=colh-taxonomy-uniref50
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --comment='agent:claude-code; taxonomy of ColH Logan50 hits against UniRef50'
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source /lustre/groups/shared/hei/tools/env.sh

DB=/lustre/groups/shared/hei/databases/mmseqs2/uniref50/2026_03/uniref50
OUT=outputs/taxonomy_uniref50
TMP=/localscratch/$USER/$SLURM_JOB_ID
mkdir -p "$OUT" "$TMP"
trap 'rm -rf -- "$TMP"' EXIT

# hit sequences from the Logan50 table (columns 2 = target, 18 = tseq)
zcat outputs/mmseqs_logan50/Q46085_vs_logan50.tsv.gz | awk -F'\t' 'NR > 1 { print ">" $2; print $18 }' > "$TMP/hits.fasta"
grep -c '^>' "$TMP/hits.fasta"

mmseqs version
# --tax-lineage 1: the _lca.tsv gets the full lineage with rank prefixes (p_, c_, ...)
# --max-rejected: the approximate 2bLCA re-aligns the prefilter candidates against the top hit's region in
# prefilter order and stops after max-rejected consecutive failures (taxonomy default 5), often before reaching
# the top hit itself, which leaves the query without hits. Job 42130720 with the default lost 1,630 of 7,879
# queries that way although a plain search finds strong hits for them (job 42130807).
mmseqs easy-taxonomy "$TMP/hits.fasta" "$DB" "$OUT/logan50_hits" "$TMP/tmp" \
    --db-load-mode 1 --threads "$SLURM_CPUS_PER_TASK" --tax-lineage 1 --max-rejected 2147483647
