import random

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def generate_peptide_from_protein(protein_seq, min_len=5, max_len=30):
    length = random.randint(min_len, max_len)
    if len(protein_seq) < length:
        return None
    start = random.randint(0, len(protein_seq) - length)
    return protein_seq[start:start + length]


def generate_negatives_from_proteins(proteins, n_samples):
    negatives = []

    while len(negatives) < n_samples:
        protein = random.choice(proteins)
        peptide = generate_peptide_from_protein(protein)
        if peptide:
            negatives.append(peptide)

    return negatives