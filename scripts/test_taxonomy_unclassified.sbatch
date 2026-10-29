#!/bin/bash
# Diagnoses the 1,630 Logan50 hits that easy-taxonomy (job 42130720) left unclassified without any top hit: plain
# easy-search of 50 of them and 10 classified controls against UniRef50, reporting each match's taxon, to see
# whether they have no matches or only matches whose taxon the LCA blacklists (default 12908 "unclassified
# sequences", the parent of the metagenome taxa). Same resources as taxonomy_uniref50.sbatch (index in memory).
# Submit from the experiment root: sbatch scripts/test_taxonomy_unclassified.sbatch
#SBATCH --job-name=colh-test-taxonomy-unclassified
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --comment='agent:claude-code; diagnose unclassified ColH Logan50 hits in UniRef50 taxonomy'
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source /lustre/groups/shared/hei/tools/env.sh

DB=/lustre/groups/shared/hei/databases/mmseqs2/uniref50/2026_03/uniref50
OUT=outputs/taxonomy_uniref50
TMP=/localscratch/$USER/$SLURM_JOB_ID
mkdir -p "$TMP"
trap 'rm -rf -- "$TMP"' EXIT

awk -F'\t' '$2 == 0 && n++ < 50 { print $1 } $2 != 0 && m++ < 10 { print $1 }' "$OUT/logan50_hits_lca.tsv" > "$TMP/ids"
zcat outputs/mmseqs_logan50/Q46085_vs_logan50.tsv.gz \
    | awk -F'\t' 'NR == FNR { want[$1] = 1; next } FNR > 1 && ($2 in want) { print ">" $2; print $18 }' "$TMP/ids" - \
    > "$TMP/sample.fasta"
grep -c '^>' "$TMP/sample.fasta"

mmseqs easy-search "$TMP/sample.fasta" "$DB" "$OUT/unclassified_sample_vs_uniref50.tsv" "$TMP/tmp" \
    --db-load-mode 1 --threads "$SLURM_CPUS_PER_TASK" --format-mode 4 \
    --format-output query,target,fident,qcov,evalue,bits,taxid,taxname,taxlineage
