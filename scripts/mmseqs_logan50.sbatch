#!/bin/bash
# MMseqs2 GPU search of ColH (UniProt Q46085, full precursor) against Logan50 non-human v1.0 (3.0 B sequences).
# Lab recipe (databases/mmseqs2/logan50_nonhuman/v1.0/README.md): mmseqs-logan50 copies the DB (~400 GB) to
# /dev/shm, which counts toward --mem, and refuses to run below ~600 GB. 1000G and --split-memory-limit 700G as in
# the README's passing test (job 41913336: 10 queries in 26 min, mostly the 7.5 min copy, the prefilter and
# convertalis), with the recipe's 16 cores. This exceeds HMGU's GPU etiquette (<=6 cores, <=90 GB per GPU job)
# because of the RAM copy; approved by Julius on 2026-10-29. 2 h covers the test's 26 min with headroom.
# scripts/logan50_search.sh runs the search: --max-seqs 100000 lifts the default cap of 300 prefilter hits, which
# would truncate a 3 B-sequence search (the E-value cutoff, default 1e-3, does the filtering), with fallbacks if
# the GPU prefilter crashes. Job 42125387 ran easy-search with --max-seqs 100000 directly and crashed.
# Submit from the experiment root: sbatch scripts/mmseqs_logan50.sbatch
#SBATCH --job-name=colh-mmseqs-logan50
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G
#SBATCH --time=02:00:00
#SBATCH --comment='agent:claude-code; ColH MMseqs2 GPU search against Logan50'
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source /lustre/groups/shared/hei/tools/env.sh

OUT=outputs/mmseqs_logan50
export SEARCH_TMP=/localscratch/$USER/$SLURM_JOB_ID
mkdir -p "$OUT" "$SEARCH_TMP"
trap 'rm -rf -- "$SEARCH_TMP"' EXIT

mmseqs version
nvidia-smi -L
mmseqs-logan50 --run bash scripts/logan50_search.sh
gzip -f "$OUT/Q46085_vs_logan50.tsv"
