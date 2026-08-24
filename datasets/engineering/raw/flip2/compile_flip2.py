"""Turn the FLIP2 split CSVs into labelled FASTAs for `dataset_preprocessor.main()`.

Counterpart to `../combingym/compile_combingym.py`: a per-source compile step that lands a labelled
FASTA one level up, in `engineering/raw/`, where the shared preprocessor writes
`engineering/flip2_<name>_max<limit>.fasta`.

**Splits are ignored.** Each source CSV carries `set` and `validation` columns holding FLIP2's
train/test assignment, and this script drops both, keeping every row. These simulations mask all
labels and replay a campaign over the whole pool, so FLIP2's supervised boundary is not meaningful
here; the equivalent notion lives in `al_splits.py` and is applied at simulation time. For TrpB that
also means only one of its three split files needs reading: all three carry identical `sequence` and
`target` columns for the same 228 298 variants, row for row, and differ only in `set`.

**FLIP2 ships no parent sequence** for either dataset, so both are reconstructed from external
sources, validated against the pool, and written out as a FASTA alongside. Every dataset here
therefore has a parent, and the mutation-aware split axes are usable for both:

* **TrpB**'s parent is **Tm9D8∗**, the engineered *Thermotoga maritima* TrpB variant the library was
  built on -- not wild-type TmTrpB. Rebuilt from UniProt P50909 plus the ten substitutions Johnston
  et al. document; see ``TRPB_PARENT`` below.
* **NucB**'s parent is taken from the upstream source of record (the ``num_mutations == 0`` row of
  ``nuclease_design/processed_data/landscape.csv``), since the distributed FLIP2 CSV omits it.

**Sequence ids differ per dataset**: TrpB carries at most 4 substitutions per variant, so substitution ids stay short (28 characters
at worst) and make a variant legible in simulation results. NucB reaches 23 substitutions, which
would run to 120 characters, so it keeps the source CSV's row index.

Usage (from this directory):

    uv run python compile_flip2.py                  # both datasets
    uv run python compile_flip2.py --datasets TrpB
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

# The NucB parent, from the upstream landscape's num_mutations == 0 row
# (storage.googleapis.com/nuclease_design/processed_data/landscape.csv). Carried here because the
# distributed FLIP2 CSV omits the wild type, and the substitution-based split axes need a parent.
NUCB_PARENT = (
    "MIKKWAVHLLFSALVLLGLSGGAAYSPQHAEGAARYDDVLYFPASRYPETGAHISDAIKAGHADVCTIERSGADKRRQ"
    "ESLKGIPTKPGFDRDEWPMAMCEEGGKGASVRYVSSSDNRGAGSWVGNRLNGYADGTRILFIVQ"
)


# The TrpB parent: Tm9D8*, an engineered TmTrpB variant, NOT wild-type TmTrpB. Johnston et al. 2024
# (PNAS 121(32):e2400439121): "Tm9D8* differs from wildtype TmTrpB by ten amino acid substitutions
# (P19G, E30G, I69V, K96L, P140L, N167D, I184F, L213P, G228S, and T292S)." Reconstructed from UniProt
# P50909 (TRPB1_THEMA, 389 aa) with those ten substitutions applied; verified identical to the
# sequence implied by the landscape (present exactly once, and no variant differs outside the 20
# varied positions). Two of the ten, I184F and G228S, fall inside the varied positions.
TRPB_PARENT = (
    "MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLH"
    "TGAHKINNAIGQVLLAKLMGKTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVK"
    "SGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPIIVRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAA"
    "GIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHSVSAGLDYSGVGPEHAYWR"
    "ETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIR"
)


class Landscape(NamedTuple):
    source: str  # split CSV to read; any one of a dataset's splits carries the full landscape
    integer_labels: bool  # ordinal bins, so write "2" rather than "2.0"
    parent: str | None  # parent sequence, if one is available
    parent_in_pool: bool | None  # whether the parent is expected among the variants
    substitution_ids: (
        bool  # ids as substitutions vs the parent, else the source row index
    )
    note: str


LANDSCAPES: dict[str, Landscape] = {
    "TrpB": Landscape(
        source="trpb/one_to_many.csv.gz",
        integer_labels=False,
        parent=TRPB_PARENT,
        parent_in_pool=True,
        substitution_ids=True,  # at most 4 substitutions per variant, so ids stay short
        note="Continuous growth-based fitness over ten combinatorially complete sub-landscapes "
        "across 20 positions. Any of the three split files gives the same landscape.",
    ),
    "NucB": Landscape(
        source="nucb/two_to_many.csv.gz",
        integer_labels=True,
        parent=NUCB_PARENT,
        parent_in_pool=False,  # documented as omitted from the distributed CSV
        substitution_ids=False,  # up to 23 substitutions per variant would give unwieldy ids
        note="Ordinal activity ladder, not a measurement: 0 non-functional, 1 active, 2 better than "
        "wild type, 3 better than the known A73R variant (199 of 55 759 sequences).",
    ),
}


def _format_label(value, integer_labels: bool) -> str:
    return str(int(value)) if integer_labels else repr(float(value))


def compile_dataset(dataset: str) -> Path:
    landscape = LANDSCAPES[dataset]
    frame = pd.read_csv(HERE / landscape.source)

    for column in ("sequence", "target"):
        if column not in frame.columns:
            raise ValueError(
                f"{dataset}: {landscape.source} has no {column!r} column; "
                f"found {list(frame.columns)}"
            )

    # Splits are deliberately dropped -- see the module docstring.
    dropped_columns = [c for c in ("set", "validation") if c in frame.columns]

    if frame["target"].isna().any():
        raise ValueError(
            f"{dataset}: {int(frame['target'].isna().sum())} rows have no target; "
            f"FLIP2 is not expected to ship missing labels."
        )

    lengths = set(frame["sequence"].astype(str).str.len())
    if len(lengths) != 1:
        raise ValueError(
            f"{dataset}: sequence lengths {sorted(lengths)[:5]} are not uniform."
        )

    width = len(str(len(frame) - 1))
    parent = landscape.parent

    def make_id(index: int, sequence: str) -> str:
        if not landscape.substitution_ids:
            return f"{dataset}_{index:0{width}d}"
        changes = ":".join(
            f"{expected}{offset + 1}{actual}"
            for offset, (expected, actual) in enumerate(zip(parent, sequence))
            if expected != actual
        )
        return f"{dataset}_{changes or 'WT'}"

    sequence_data = [
        SequenceData(
            seq_id=make_id(index, str(sequence)),
            seq=str(sequence),
            label=_format_label(target, landscape.integer_labels),
        )
        for index, (sequence, target) in enumerate(
            zip(frame["sequence"], frame["target"])
        )
    ]

    if len({data_point.seq_id for data_point in sequence_data}) != len(sequence_data):
        raise ValueError(f"{dataset}: sequence ids are not unique.")

    output_path = OUTPUT_DIR / f"flip2_{dataset}.fasta"
    n_written = write_FASTA(output_path, sequence_data)

    print(f"{dataset:<6} {n_written:>7,} sequences -> {output_path.name}")
    print(
        f"{'':<6} length={lengths.pop()}  labels={'int' if landscape.integer_labels else 'float'}"
        f"  dropped split columns={dropped_columns or 'none'}"
    )

    if landscape.parent is not None:
        _write_parent(
            dataset, sequence_data, landscape.parent, landscape.parent_in_pool
        )
    else:
        print(f"{'':<6} no parent sequence available (see module docstring)")
    return output_path


def _write_parent(
    dataset: str, pool: list[SequenceData], parent: str, expect_in_pool: bool
) -> None:
    """Write the parent as a single-record FASTA, after checking it against the pool."""
    if len(parent) != len(pool[0].seq):
        raise ValueError(
            f"{dataset}: parent length {len(parent)} != variant length {len(pool[0].seq)}."
        )

    distances = [
        sum(a != b for a, b in zip(parent, data_point.seq)) for data_point in pool
    ]
    n_exact = distances.count(0)
    if expect_in_pool and n_exact != 1:
        raise ValueError(
            f"{dataset}: expected the parent to appear exactly once among the variants, "
            f"found it {n_exact} time(s). Either the parent or the source is wrong."
        )
    if not expect_in_pool and n_exact:
        raise ValueError(
            f"{dataset}: the parent is present in the pool, but the distributed CSV is "
            f"documented as omitting it. Check whether the source changed."
        )
    # A mutagenesis library sits close to its parent; a wrong parent would put every variant far away.
    median = sorted(distances)[len(distances) // 2]
    if median > len(parent) / 4:
        raise ValueError(
            f"{dataset}: median distance to the parent is {median} of {len(parent)} "
            f"residues, far too high for a mutagenesis library. The parent looks wrong."
        )

    parent_path = HERE / dataset.lower() / f"{dataset}_wt.fasta"
    parent_path.write_text(f">{dataset}_wt\n{parent}\n")
    presence = "present once" if n_exact else "absent from the pool"
    print(
        f"{'':<6} parent -> {parent_path.relative_to(HERE)}  "
        f"({presence}, substitutions {min(distances)}-{max(distances)}, median {median})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(LANDSCAPES),
        help=f"Which landscapes to compile (default: {', '.join(sorted(LANDSCAPES))})",
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
        if not (HERE / LANDSCAPES[dataset].source).exists()
    ]
    if missing:
        print(
            f"Missing source CSV for {', '.join(missing)}. Run download_flip2.py first.",
            file=sys.stderr,
        )
        return 1

    for dataset in args.datasets:
        compile_dataset(dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
