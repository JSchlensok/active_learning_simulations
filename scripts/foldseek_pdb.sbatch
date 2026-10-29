#!/bin/bash
# Foldseek search of the ColH structure 9LRK against the lab's PDB100 snapshot (2025-01-01).
# CPU rather than GPU: only CPU searches return taxonomy (foldseek#497). The DB README's test searched 10
# structures in 9 s, so 8 cores, 16 GB (DB is 4.4 GB, read into memory by --db-load-mode 1) and 1 h are ample.
# Submit from the experiment root: sbatch scripts/foldseek_pdb.sbatch
#SBATCH --job-name=colh-foldseek-pdb
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --comment='agent:claude-code; ColH Foldseek search against PDB'
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source /lustre/groups/shared/hei/tools/env.sh

DB=/lustre/groups/shared/hei/databases/foldseek/pdb/2025-01-01/pdb
OUT=outputs/foldseek_pdb
TMP=/localscratch/$USER/$SLURM_JOB_ID
mkdir -p "$OUT" "$TMP"
trap 'rm -rf -- "$TMP"' EXIT

foldseek version
# --format-mode 4: tab-separated with a header line
foldseek easy-search inputs/9LRK.cif.gz "$DB" "$OUT/9LRK_vs_pdb.tsv" "$TMP" \
    --threads "$SLURM_CPUS_PER_TASK" --db-load-mode 1 --format-mode 4 \
    --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,qlen,tstart,tend,tlen,qcov,tcov,evalue,bits,prob,alntmscore,qtmscore,ttmscore,lddt,taxid,taxname,theader
