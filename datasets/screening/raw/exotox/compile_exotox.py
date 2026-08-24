import pandas as pd

from biotrainer_core.data_classes import SequenceData
from biotrainer_core.input_files import read_FASTA, write_FASTA

def read_labels(input_path: str):
    df = pd.read_csv(input_path)
    labels = {}
    for _, row in df.iterrows():
        labels[row["id"]] = str(row["label"])
    return labels


train_fasta = "exotox_X_train_SST30.fasta"
test_fasta = "exotox_X_test_SST30.fasta"
train_labels = "exotox_y_train_SST30.csv"
test_labels = "exotox_y_test_SST30.csv"

train_seqs = read_FASTA(train_fasta)
test_seqs = read_FASTA(test_fasta)

all_seqs = train_seqs + test_seqs
assert len(all_seqs) == len(train_seqs) + len(test_seqs)
assert len(all_seqs) == len(set(seq.seq_id for seq in all_seqs))

train_labels = read_labels(train_labels)
test_labels = read_labels(test_labels)

all_seq_data_with_labels = []
for seq_data in all_seqs:
    seq_id = seq_data.seq_id
    label = train_labels.get(seq_id, test_labels.get(seq_id, None))
    assert label is not None, f"Missing label for sequence {seq_id}"
    assert int(label) in [0, 1], f"Invalid label value for sequence {seq_id}: {label}"
    label_str = "EXOTOXIN" if int(label) == 1 else "NON-EXOTOXIN"
    seq_data_updated = SequenceData(seq_id=seq_data.seq_id, seq=seq_data.seq, label=label_str)
    all_seq_data_with_labels.append(seq_data_updated)

assert len(all_seq_data_with_labels) == len(all_seqs)
all_sequences = [seq_data.seq for seq_data in all_seq_data_with_labels]
assert len(set(all_sequences)) == len(all_sequences), "Duplicate sequences found in the merged FASTA file"

write_FASTA("../exotox_merged.fasta", all_seq_data_with_labels)