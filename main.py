import numpy as np

from sklearn.utils import compute_class_weight
import torch

from sklearn.model_selection import train_test_split

from pretrain.pretrain import get_feature

from test_model.test import evaluate_model

from train.model import AttentionFusionModel
from train.train import train_model
from utils import load_dataframe, normalize_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# load data

train_df, test_df = load_dataframe()

classes = np.unique(train_df["label"])
print(f'Размер данных: { len(train_df) }')
print(f"Классы: { classes }, кол-во классов: { classes.size }")

# ==========================================================
# pretrain

esm = get_feature(train_df, 'esm', device).to(device)
print("ESM shape:", esm.shape)  # ESM shape: torch.Size([3000, 320])

fingerprints = get_feature(train_df, 'fingerprint', device).to(device)
print("FP shape:", fingerprints.shape)

physchem = get_feature(train_df, 'physchem', device).to(device)
print("physchem shape:", physchem.shape)

esm, esm_scaler = normalize_feature(esm, fit=True)
fingerprints, fingerprints_scaler = normalize_feature(fingerprints, fit=True)
physchem, physchem_scaler = normalize_feature(physchem, fit=True)

# ==========================================================
# Model init

model = AttentionFusionModel(
    esm_dim=esm.shape[1],
    fp_dim=fingerprints.shape[1],
    phys_dim=physchem.shape[1],
    num_classes=classes.size
).to(device)

# ==========================================================
# train/validation split

indices = np.arange(len(train_df))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=train_df['label'].values
)

# split всех фичей
esm_train, esm_val = esm[train_idx], esm[val_idx]
fp_train, fp_val = fingerprints[train_idx], fingerprints[val_idx]
phys_train, phys_val = physchem[train_idx], physchem[val_idx]

y_train = torch.tensor(train_df['label'].values[train_idx], dtype=torch.long, device=device)
y_val   = torch.tensor(train_df['label'].values[val_idx], dtype=torch.long, device=device)

# ==========================================================
# train

# вычисляем веса
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_df['label'].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

model = train_model(
    model,
    esm_train, fp_train, phys_train, y_train,
    esm_val, fp_val, phys_val, y_val,
    criterion,
    device,
    classes.size,
    epochs=50,
    batch_size=64,
    patience=7
)

# ==========================================================
# test

esm_test = get_feature(test_df, 'esm', device).to(device)
fp_test = get_feature(test_df, 'fingerprint', device).to(device)
phys_test = get_feature(test_df, 'physchem', device).to(device)

esm_test, _ = normalize_feature(esm_test, scaler=esm_scaler)
fp_test, _ = normalize_feature(fp_test, scaler=fingerprints_scaler)
phys_test, _ = normalize_feature(phys_test, scaler=physchem_scaler)

y_test = torch.tensor(test_df['label'].values, dtype=torch.long, device=device)

metrics = evaluate_model(
    model,
    esm_test,
    fp_test,
    phys_test,
    y_test,
    classes.size,
    batch_size=64,
)

print("\n===== TEST RESULTS =====")
for k, v in metrics.items():
    if (k == "cm"):
        print(f"{k}:\n{v}")
    else:
        print(f"{k}: {v:.4f}")
