from Bio.Align import substitution_matrices
import random
import pandas as pd

from utils import load_dataframe

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


def blosum_mutation(sequence, mutation_rate=0.1):
    seq = list(sequence)

    for i, aa in enumerate(seq):
        if random.random() < mutation_rate:
            candidates = get_similar_amino_acids(aa)
            if candidates:
                seq[i] = random.choice(candidates)

    return "".join(seq)

def augmented_data(dataframe):
    augmented_train_df = []

    for seq, label in dataframe.values:
        augmented = blosum_mutation(seq)
        if seq != augmented:
            augmented_train_df.append([augmented, label])

    augmented_train_df = pd.DataFrame(augmented_train_df, columns=dataframe.columns)
    print("Добавлено: ", len(augmented_train_df))
    return pd.concat([dataframe, augmented_train_df], ignore_index=True)
