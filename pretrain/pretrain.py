import torch

batch_size = 32

hydrophobicity_scale = {
    'A': (71.07, 6.3),
    'C': (103.14, 7.0),
    'D': (115.08, 1.0),
    'E': (129.11, 1.0),
    'F': (147.17, 7.2),
    'G': (57.05, 4.1),
    'H': (137.14, 1.3),
    'I': (113.15, 9.0),
    'K': (128.17, 0.6),
    'L': (113.15, 8.2),
    'M': (131.20, 6.4),
    'N': (114.10, 1.0),
    'P': (97.11, 2.9),
    'Q': (128.13, 1.0),
    'R': (156.18, 0.0),
    'S': (87.07, 3.6),
    'T': (101.10, 3.8),
    'V': (99.13, 8.7),
    'W': (186.20, 3.6),
    'Y': (163.17, 3.2),
    'B': (114.61, 1.0),
    'Z': (128.64, 1.0),
}

# ==========================================================
# ESM

import esm
from pretrain import DataIterator as DI

def load_esm_model():
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    return model, alphabet.get_batch_converter()

def extract_representations(model, batch_converter, data_iterator):
    model.eval()
    representations = []
    for batch_data in data_iterator:
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens

        batch_lens = (batch_tokens != 1).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])
        token_representations = results["representations"][6]

        for i, tokens_len in enumerate(batch_lens):
            seq_representation = token_representations[i, 1 : tokens_len - 1].mean(0)
            representations.append(seq_representation)

    return representations

def get_esm(data):
    model, batch_converter = load_esm_model()
    data_iterator = DI.DataIterator(data, batch_size)
    result = extract_representations(model, batch_converter, data_iterator)
    return torch.stack(result).float()

# ==========================================================
# Fingerprints

from skfp.preprocessing import MolFromAminoseqTransformer
from skfp.fingerprints import ECFPFingerprint

def get_mol_from_seq(data):
   mol_transformer = MolFromAminoseqTransformer()
   mols = mol_transformer.transform(data)
   return mols

def get_fingerprints(data):
    fp_transformer = ECFPFingerprint()
    mols = get_mol_from_seq(data)
    result = fp_transformer.transform(mols)
    return torch.tensor(result, dtype=torch.float32)

# ==========================================================
# Physchem

import numpy as np

def get_physchem(data):
    weight = []
    hydro = []

    for seq in data:
        seq_weight = []
        seq_hydro = []
        for aa in seq:
            molecular_weight, hydrophobicity = hydrophobicity_scale.get(aa, 0.0)
            seq_weight.append(molecular_weight)
            seq_hydro.append(hydrophobicity)

        weight.append(np.sum(seq_weight) + 18.08)
        hydro.append(np.mean(seq_hydro))

    return torch.tensor(np.stack([weight, hydro], axis=1), dtype=torch.float32)

# ==========================================================

def get_feature(dataframe, feature):
   if feature == 'esm':
      data = [(label, seq) for seq, label in dataframe[['sequence','label']].values]
      return get_esm(data)
   elif feature == 'fingerprint':
      data = dataframe['sequence']
      return get_fingerprints(data) 
   elif feature == 'physchem':
      data = dataframe['sequence']
      return get_physchem(data)
   else:
      return None