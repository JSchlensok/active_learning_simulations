"""Turn the CombinGym xlsx landscapes into labelled FASTAs for `dataset_preprocessor.main()`.

Usage (from this directory):

    uv run python compile_combingym.py                 # every dataset in LANDSCAPES
    uv run python compile_combingym.py --datasets GB1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from biotrainer_core.data_classes import SequenceData
from biotrainer_core.input_files import write_FASTA

HERE = Path(__file__).parent
OUTPUT_DIR = (
    HERE.parent
)  # engineering/raw/, so the shared preprocessor writes into engineering/


class Landscape(NamedTuple):
    """Which phenotype column to take, and why that one."""

    label_column: str
    rationale: str


# One phenotype per dataset. CombinGym benchmarks a subset of the available columns too (its results
# endpoint lists only `binding_H1` for CR9114, a single `activity` for SaCas9), so these follow it.
LANDSCAPES: dict[str, Landscape] = {
    "GB1": Landscape(
        label_column="fitness", rationale="The only label column in the file."
    ),
    "CreiLOV": Landscape(
        label_column="mean",
        rationale="Mean over the three fluorescence replicates. The `*_log` columns are the same "
        "measurement on a log scale; the linear mean keeps the label in assay units.",
    ),
    "CR9114": Landscape(
        label_column="h1_mean",
        rationale='H1 only, per the CombinGym paper: "For CR9114, we utilized only the H1 landscape '
        "because H3 and influenza B landscapes were mainly characterized by low fitness "
        'values, limiting the information content for model training." Matches the single '
        "`binding_H1` phenotype in CombinGym's own leaderboard.",
    ),
    "mTagBFP2": Landscape(
        label_column="combined",
        rationale="Joint blue/red phenotype rather than either channel alone.",
    ),
    "SaCas9": Landscape(
        label_column="Mean",
        rationale="Mean over the three sgRNAs. CombinGym's leaderboard likewise reports one "
        "`activity` phenotype rather than per-sgRNA numbers.",
    ),
}


def read_wildtype(dataset: str) -> str:
    """Parent sequence from the single-record FASTA shipped alongside the landscape."""
    lines = (HERE / dataset / f"{dataset}_wt.fasta").read_text().splitlines()
    return "".join(line.strip() for line in lines if line and not line.startswith(">"))


def substitution_string(sequence: str, wildtype: str) -> str:
    """1-based substitutions as `V39A:D40C`, or `WT` for the parent itself."""
    changes = [
        f"{expected}{index + 1}{actual}"
        for index, (expected, actual) in enumerate(zip(wildtype, sequence))
        if expected != actual
    ]
    return ":".join(changes) if changes else "WT"


def compile_dataset(dataset: str) -> Path:
    landscape = LANDSCAPES[dataset]
    frame = pd.read_excel(HERE / dataset / f"{dataset}_dms.xlsx")
    wildtype = read_wildtype(dataset)

    if landscape.label_column not in frame.columns:
        raise ValueError(
            f"{dataset}: no column {landscape.label_column!r}; available: {list(frame.columns)}"
        )

    n_rows = len(frame)

    # (1) CreiLOV carries a trailing stop codon on every genotype.
    sequences = frame["genotype"].astype(str).str.rstrip("*")
    n_stripped = int(
        (sequences.str.len() != frame["genotype"].astype(str).str.len()).sum()
    )

    # (3) Rows without a measurement for the chosen phenotype cannot be labelled. Note this is NOT
    # the same as a non-binder, which carries the pinned assay floor as a real value — see README.
    labels = pd.to_numeric(frame[landscape.label_column], errors="coerce")
    keep = labels.notna()
    n_dropped = int((~keep).sum())

    lengths = set(sequences[keep].str.len())
    if lengths != {len(wildtype)}:
        raise ValueError(
            f"{dataset}: variant lengths {sorted(lengths)} do not all match the "
            f"{len(wildtype)}-residue wild type."
        )

    sequence_data = [
        SequenceData(
            seq_id=f"{dataset}_{substitution_string(sequence, wildtype)}",
            seq=sequence,
            label=str(label),
        )
        for sequence, label in zip(sequences[keep], labels[keep])
    ]

    seq_ids = {data_point.seq_id for data_point in sequence_data}
    if len(seq_ids) != len(sequence_data):
        raise ValueError(
            f"{dataset}: sequence ids are not unique ({len(seq_ids)} of {len(sequence_data)})."
        )

    output_path = OUTPUT_DIR / f"combingym_{dataset}.fasta"
    n_written = write_FASTA(output_path, sequence_data)

    print(f"{dataset:<9} {n_written:>7,} sequences -> {output_path.name}")
    print(
        f"{'':<9} label={landscape.label_column!r}  rows={n_rows:,}  "
        f"stripped '*'={n_stripped:,}  dropped (no label)={n_dropped:,}"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(LANDSCAPES),
        help=f"Which landscapes to compile (default: all of {', '.join(sorted(LANDSCAPES))})",
    )
    args = parser.parse_args()

    unknown = [dataset for dataset in args.datasets if dataset not in LANDSCAPES]
    if unknown:
        print(
            f"Unknown dataset(s): {', '.join(unknown)}. Known: {', '.join(sorted(LANDSCAPES))}",
            file=sys.stderr,
        )
        return 1

    missing = [
        dataset
        for dataset in args.datasets
        if not (HERE / dataset / f"{dataset}_dms.xlsx").exists()
    ]
    if missing:
        print(
            f"Missing landscape file(s) for {', '.join(missing)}. "
            f"Run download_combingym.py first.",
            file=sys.stderr,
        )
        return 1

    for dataset in args.datasets:
        compile_dataset(dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
