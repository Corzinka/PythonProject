from Bio.Align import substitution_matrices
import random
import pandas as pd

blosum62 = substitution_matrices.load('BLOSUM62')

from skfp.preprocessing import MolFromAminoseqTransformer
from skfp.fingerprints import ECFPFingerprint

SIMILAR_AA = {}

threshold = 1
for (a1, a2), score in blosum62.items():
    if score >= threshold and a1 != a2:
        SIMILAR_AA.setdefault(a1, []).append(a2)
        SIMILAR_AA.setdefault(a2, []).append(a1)

def blosum_mutation(sequence, mutation_rate=0.1):
    seq = list(sequence)

    for i, aa in enumerate(seq):
        if random.random() < mutation_rate:
            candidates = SIMILAR_AA.get(aa)
            if candidates:
                seq[i] = random.choice(candidates)

    return "".join(seq)

def augmented_data(dataframe):
    augmented_train_df = []
    count_pass = 0
    mol_transformer = MolFromAminoseqTransformer()

    for seq, label in dataframe.values:
        augmented = blosum_mutation(seq)
        if seq != augmented:
            mols = mol_transformer.transform(augmented)
            if None in mols:
                count_pass+=1
            else:
                augmented_train_df.append([augmented, label])

    augmented_train_df = pd.DataFrame(augmented_train_df, columns=dataframe.columns)
    print("Добавлено: ", len(augmented_train_df))
    print("Пропущено: ", count_pass)
    return pd.concat([dataframe, augmented_train_df], ignore_index=True)

#GNNRPVYIPKPRPPHPRI