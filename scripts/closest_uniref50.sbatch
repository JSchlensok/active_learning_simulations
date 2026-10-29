#!/bin/bash
# Closest UniRef50 entry of each Logan50 hit, with real alignment statistics: easy-taxonomy aligns in score-only
# mode (--alignment-mode 1), so the identities in its _tophit_aln are estimates from the score. A plain
# easy-search against UniRef50 2026_03 (default alignment, results per query sorted best first) keeps the first
# match per query. Same resources as taxonomy_uniref50.sbatch (index in memory).
# Submit from the experiment root: sbatch scripts/closest_uniref50.sbatch
#SBATCH --job-name=colh-closest-uniref50
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --comment='agent:claude-code; closest UniRef50 entries of ColH Logan50 hits'
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source /lustre/groups/shared/hei/tools/env.sh

DB=/lustre/groups/shared/hei/databases/mmseqs2/uniref50/2026_03/uniref50
OUT=outputs/taxonomy_uniref50
TMP=/localscratch/$USER/$SLURM_JOB_ID
mkdir -p "$OUT" "$TMP"
trap 'rm -rf -- "$TMP"' EXIT

zcat outputs/mmseqs_logan50/Q46085_vs_logan50.tsv.gz | awk -F'\t' 'NR > 1 { print ">" $2; print $18 }' > "$TMP/hits.fasta"

mmseqs easy-search "$TMP/hits.fasta" "$DB" "$TMP/all.tsv" "$TMP/tmp" \
    --db-load-mode 1 --threads "$SLURM_CPUS_PER_TASK" --format-mode 4 \
    --format-output query,target,fident,alnlen,qcov,tcov,evalue,bits,taxid,taxname
awk -F'\t' 'NR == 1 || !seen[$1]++' "$TMP/all.tsv" > "$OUT/logan50_hits_closest_uniref50.tsv"
wc -l "$OUT/logan50_hits_closest_uniref50.tsv"
