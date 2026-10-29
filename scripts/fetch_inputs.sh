#!/bin/bash
# Downloads the query sequence and structure into inputs/. Run from the experiment root.
set -euo pipefail

mkdir -p inputs
# ColH, Hathewaya histolytica, full precursor (1021 aa)
curl -sSf -o inputs/Q46085.fasta https://rest.uniprot.org/uniprotkb/Q46085.fasta
# Apo wild-type ColH, cryo-EM, 2.2 Å; chain A models UniProt residues 45-808
curl -sSf -o inputs/9LRK.cif.gz https://files.rcsb.org/download/9LRK.cif.gz
sha256sum inputs/Q46085.fasta inputs/9LRK.cif.gz
