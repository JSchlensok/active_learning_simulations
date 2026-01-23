from pathlib import Path


def read_raw_fasta(fasta_path, filter_length: int = 2000):
    result = {}
    with open(fasta_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if line.startswith('>'):
                seq = lines[idx + 1].strip()
                if len(seq) < filter_length:
                    result[line.strip()] = seq

    return result


def remove_outliers(sequences):
    id2label = {}
    for header in sequences.keys():
        id = header.split(" ")[0].replace(">", "")
        label = header.split("TARGET=")[1].split(" ")[0].strip()
        id2label[id] = float(label)

    # Calculate mean and standard deviation
    values = list(id2label.values())
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

    # Filter sequences
    filtered_sequences = {}
    for header, seq in sequences.items():
        id = header.split(" ")[0].replace(">", "")
        if abs(id2label[id] - mean) <= 2 * std:
            filtered_sequences[header] = seq

    return filtered_sequences

def write_raw_fasta(fasta_path, sequences):
    with open(fasta_path, 'w') as file:
        for header, seq in sequences.items():
            file.write(f"{header}\n{seq}\n")


def main():
    raw_paths = [Path("raw/amylase_pet.fasta"), Path("raw/biotrainer_meltome_mixed.fasta"), Path("raw/scl.fasta")]
    seq_length_limit = 2000
    for raw_path in raw_paths:
        sequences = read_raw_fasta(raw_path, seq_length_limit)
        write_raw_fasta(raw_path.name.split(".")[0] + f"_max{seq_length_limit}.fasta", sequences)


if __name__ == "__main__":
    main()
