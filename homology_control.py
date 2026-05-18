from utils import load_dataframe

train_df, test_df = load_dataframe()

with open("peptides.fasta", "w") as f:
    for i, seq in enumerate(sequences):
        f.write(f">pep_{i}\n{seq}\n")