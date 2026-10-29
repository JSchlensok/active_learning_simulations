#!/bin/bash
# Diagnoses the Logan50 failure (job 42125387: "CUDA error: an illegal memory access" in the GPU ungapped
# prefilter with --max-seqs 100000; the lab's test with the default 300 passed). Runs ColH through the same GPU
# prefilter against PDB seqres (1.2 M sequences) and UniRef50 (38.8 M, many GPU batches) with --max-seqs 300 and
# 100000, so a crash pins it on max-seqs without the 1 TB Logan50 copy. H100 like the failed job (the interactive
# GPU nodes are V100s); 4 cores and 32 GB stay within GPU etiquette. UniRef50 is copied to local scratch first
# (13 GB) because the GPU prefilter reads lustre page by page.
# Submit from the experiment root: sbatch scripts/test_gpu_max_seqs.sbatch
#SBATCH --job-name=colh-test-gpu-max-seqs
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --comment='agent:claude-code; test GPU prefilter max-seqs for ColH Logan50 search'
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
source /lustre/groups/shared/hei/tools/env.sh

DBS=/lustre/groups/shared/hei/databases/mmseqs2
TMP=/localscratch/$USER/$SLURM_JOB_ID
mkdir -p "$TMP"
trap 'rm -rf -- "$TMP"' EXIT

nvidia-smi -L
cp "$DBS"/uniref50/2026_03/uniref50{,.index,.dbtype} "$TMP"/
mmseqs createdb inputs/Q46085.fasta "$TMP/query" > /dev/null

for db in "$DBS/pdb/2026-10-21/pdb" "$TMP/uniref50"; do
    for n in 300 100000; do
        res=$TMP/res_$(basename "$db")_$n
        mmseqs search "$TMP/query" "$db" "$res" "$TMP/tmp_$(basename "$db")_$n" --gpu 1 --max-seqs "$n" \
            --threads "$SLURM_CPUS_PER_TASK" > "$res.log" 2>&1
        rc=$?
        hits=$(cat "$res" "$res".[0-9]* 2>/dev/null | grep -c . || true)
        echo "RESULT db=$(basename "$db") max_seqs=$n exit=$rc hits=$hits"
        grep -E "CUDA error|died|passed prefiltering" "$res.log" | sed 's/^/  /'
    done
done
