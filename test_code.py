import torch

import numpy as np
import pandas as pd

from utils import load_dataframe, load_esm_model, extract_representations, DataIterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# ==========================================================
# load data
# ==========================================================

train_df, test_df = load_dataframe()

from Bio.Align import substitution_matrices
import random

blosum62 = substitution_matrices.load('BLOSUM62')

def get_similar_amino_acids(aa, threshold=0):
    similar = []
    for (a1, a2), score in blosum62.items():
        if score >= threshold:
            if a1 == aa:
                similar.append(a2)
            elif a2 == aa:
                similar.append(a1)
    return similar


def blosum_mutation(sequence, mutation_rate=0.4):
    seq = list(sequence)

    for i, aa in enumerate(seq):
        if random.random() < mutation_rate:
            candidates = get_similar_amino_acids(aa)
            if candidates:
                seq[i] = random.choice(candidates)

    return "".join(seq)

print(len(train_df))

augmented_train_df = []

for seq, label in train_df.values:
    augmented = blosum_mutation(seq)
    if seq != augmented:
        augmented_train_df.append([augmented, label])

augmented_train_df = pd.DataFrame(augmented_train_df, columns=train_df.columns)
train_df = pd.concat([train_df, augmented_train_df], ignore_index=True)

print(len(train_df))

classes = np.unique(train_df["label"])
n_classes = classes.size

print(f"Classes: {classes}, n_classes: {n_classes}")