#!/bin/bash
# Runs inside `mmseqs-logan50 --run`, with $LOGAN50 pointing at the RAM copy of Logan50 and $SEARCH_TMP at a
# local scratch directory. Tries --max-seqs 100000 first: job 42125387 died there with "CUDA error: an illegal
# memory access" in the GPU prefilter, which the same search against PDB and UniRef50 (job 42126363) did not
# reproduce. Falls back to 10000, then to the default 300 that the lab's Logan50 test passed with, so one copy of
# the database yields the largest cap that works.
set -uo pipefail

OUT=outputs/mmseqs_logan50
for n in 100000 10000 300; do
    # --format-mode 4: tab-separated with a header line; theader holds Prodigal's SRA run/contig/coordinates
    if mmseqs easy-search inputs/Q46085.fasta "$LOGAN50" "$OUT/Q46085_vs_logan50.tsv" "$SEARCH_TMP/max_seqs_$n" \
        --gpu 1 --split-memory-limit 700G --threads "$SLURM_CPUS_PER_TASK" --max-seqs "$n" --format-mode 4 \
        --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,qlen,tstart,tend,tlen,qcov,tcov,evalue,bits,theader,tseq
    then
        echo "logan50_search: succeeded with --max-seqs $n" >&2
        exit 0
    fi
    echo "logan50_search: --max-seqs $n failed" >&2
done
exit 1
