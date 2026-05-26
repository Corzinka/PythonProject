import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

def load_dataframe(train_path, test_path):
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    return train_df, test_df

def get_batches(esm, fp, phys, y, batch_size):
    n = esm.size(0)
    indices = torch.randperm(n)

    for i in range(0, n, batch_size):
        idx = indices[i:i+batch_size]
        yield esm[idx], fp[idx], phys[idx], y[idx]

def normalize_feature(tensor, scaler=None, fit=False):
    arr = tensor.cpu().numpy()

    if fit:
        scaler = StandardScaler()
        arr = scaler.fit_transform(arr)
    else:
        arr = scaler.transform(arr)

    return torch.tensor(arr, dtype=torch.float32, device=tensor.device), scaler