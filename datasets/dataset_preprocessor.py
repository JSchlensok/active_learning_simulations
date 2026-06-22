from typing import List
from pathlib import Path

from biotrainer_core.data_classes import SequenceData
from biotrainer_core.input_files import read_FASTA, pgym_csv_to_fasta, write_FASTA


def remove_outliers(seq_data: List[SequenceData]):
    """ Remove outliers from the dataset based on the mean and standard deviation. (Currently not used) """
    id2label = {data_point.seq_id: float(data_point.label) for data_point in seq_data}

    # Calculate mean and standard deviation
    values = list(id2label.values())
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

    # Filter sequences
    filtered_sequences = []
    for data_point in seq_data:
        if abs(id2label[data_point.seq_id] - mean) <= 2 * std:
            filtered_sequences.append(data_point)

    return filtered_sequences


def main():
    raw_paths = [Path("raw/amylase_pet.fasta"),
                 Path("raw/biotrainer_meltome_mixed.fasta"),
                 Path("raw/scl.fasta"),
                 Path("raw/PHOT_CHLRE_Chen_2023.csv"),
                 Path("raw/exotox_merged.fasta"),
                 ]
    seq_length_limit = 2000
    for raw_path in raw_paths:
        output_path = raw_path.name.split(".")[0] + f"_max{seq_length_limit}.fasta"
        if raw_path.suffix == ".csv":
            n_seqs_in_pgym = pgym_csv_to_fasta(raw_path, output_path, single_mutations_only=True)
            assert n_seqs_in_pgym > 0, "No sequences found in the CSV file and written to FASTA file."
            seq_data = read_FASTA(output_path)
        else:
            seq_data = read_FASTA(raw_path)

        n_original_data = len(seq_data)
        seq_data_filtered = [data_point for data_point in seq_data if len(data_point.seq) < seq_length_limit]
        n_written = write_FASTA(output_path, seq_data_filtered)
        print(f"Wrote {n_written} sequences to {output_path} (out of {n_original_data} original sequences)")


if __name__ == "__main__":
    main()
