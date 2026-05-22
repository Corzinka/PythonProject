import random
import numpy as np
import torch

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from homology_control import create_cluster
from pretrain.pretrain import get_feature
from train.model import AttentionFusionModel
from train.train import train_model
from test_model.test import evaluate_model
from utils import load_dataframe, normalize_feature


# ==========================================================
# reproducibility
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==========================================================
# load data
# ==========================================================

train_df, test_df = load_dataframe()

train_df = create_cluster(train_df)

classes = np.unique(train_df["label"])
num_classes = len(classes)

print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")
print(f"Classes: {classes}")


# ==========================================================
# extract raw features
# ==========================================================

print("Extracting train features...")

esm_raw = get_feature(train_df, "esm")
fp_raw = get_feature(train_df, "fingerprint")
phys_raw = get_feature(train_df, "physchem")

print("Extracting test features...")

esm_test_raw = get_feature(test_df, "esm")
fp_test_raw = get_feature(test_df, "fingerprint")
phys_test_raw = get_feature(test_df, "physchem")

y_all = train_df["label"].values
y_test = torch.tensor(test_df["label"].values, dtype=torch.long)


# ==========================================================
# cross-validation
# ==========================================================

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

cv_results = []

print("\n===== CROSS VALIDATION =====\n")

for fold, (train_idx, val_idx) in enumerate(
    sgkf.split(
        X=np.zeros(len(train_df)),
        y=y_all,
        groups=train_df["cluster"]
    )
):
    print(f"\n===== FOLD {fold+1} =====")

    # -----------------------------------
    # split raw features
    # -----------------------------------

    esm_train_raw = esm_raw[train_idx]
    esm_val_raw = esm_raw[val_idx]

    fp_train_raw = fp_raw[train_idx]
    fp_val_raw = fp_raw[val_idx]

    phys_train_raw = phys_raw[train_idx]
    phys_val_raw = phys_raw[val_idx]

    y_train_np = y_all[train_idx]
    y_val_np = y_all[val_idx]

    # -----------------------------------
    # fit scaler ONLY on train fold
    # -----------------------------------

    esm_train, esm_scaler = normalize_feature(esm_train_raw, fit=True)
    esm_val, _ = normalize_feature(esm_val_raw, scaler=esm_scaler)

    fp_train, fp_scaler = normalize_feature(fp_train_raw, fit=True)
    fp_val, _ = normalize_feature(fp_val_raw, scaler=fp_scaler)

    phys_train, phys_scaler = normalize_feature(phys_train_raw, fit=True)
    phys_val, _ = normalize_feature(phys_val_raw, scaler=phys_scaler)

    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_val = torch.tensor(y_val_np, dtype=torch.long)

    # -----------------------------------
    # class weights (per fold)
    # -----------------------------------

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_np
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float
        )
    )

    # -----------------------------------
    # fold statistics
    # -----------------------------------

    print(f"Train size: {len(train_idx)}")
    print(f"Val size: {len(val_idx)}")
    print(f"Train positive ratio: {np.mean(y_train_np):.3f}")
    print(f"Val positive ratio: {np.mean(y_val_np):.3f}")
    print(f"Train clusters: {len(set(train_df.iloc[train_idx]['cluster']))}")
    print(f"Val clusters: {len(set(train_df.iloc[val_idx]['cluster']))}")

    # -----------------------------------
    # model
    # -----------------------------------

    model = AttentionFusionModel(
        esm_dim=esm_train.shape[1],
        fp_dim=fp_train.shape[1],
        phys_dim=phys_train.shape[1],
        num_classes=num_classes
    )

    model = train_model(
        model,
        esm_train,
        fp_train,
        phys_train,
        y_train,
        esm_val,
        fp_val,
        phys_val,
        y_val,
        criterion,
        num_classes=num_classes,
        epochs=50,
        batch_size=64,
        patience=7
    )

    # -----------------------------------
    # evaluate on VAL ONLY
    # -----------------------------------

    metrics = evaluate_model(
        model,
        esm_val,
        fp_val,
        phys_val,
        y_val,
        num_classes=num_classes,
        batch_size=64
    )

    cv_results.append(metrics)


# ==========================================================
# summarize CV
# ==========================================================

print("\n===== CV RESULTS =====\n")

for metric in ["pr_auc", "f1_macro"]:
    arr = np.array([m[metric] for m in cv_results])

    print(
        f"{metric}: "
        f"{arr.mean():.4f} ± {arr.std():.4f}"
    )


# ==========================================================
# FINAL TRAIN ON FULL TRAIN SET
# ==========================================================

print("\n===== FINAL TRAIN =====\n")

esm_all, esm_scaler = normalize_feature(esm_raw, fit=True)
fp_all, fp_scaler = normalize_feature(fp_raw, fit=True)
phys_all, phys_scaler = normalize_feature(phys_raw, fit=True)

esm_test, _ = normalize_feature(esm_test_raw, scaler=esm_scaler)
fp_test, _ = normalize_feature(fp_test_raw, scaler=fp_scaler)
phys_test, _ = normalize_feature(phys_test_raw, scaler=phys_scaler)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_all
)

criterion = torch.nn.CrossEntropyLoss(
    weight=torch.tensor(
        class_weights,
        dtype=torch.float
    )
)

final_model = AttentionFusionModel(
    esm_dim=esm_all.shape[1],
    fp_dim=fp_all.shape[1],
    phys_dim=phys_all.shape[1],
    num_classes=num_classes
)

final_model = train_model(
    final_model,
    esm_all,
    fp_all,
    phys_all,
    torch.tensor(y_all, dtype=torch.long),
    esm_test,
    fp_test,
    phys_test,
    y_test,
    criterion,
    num_classes=num_classes,
    epochs=50,
    batch_size=64,
    patience=7
)

# ==========================================================
# FINAL TEST (ONE TIME)
# ==========================================================

print("\n===== FINAL TEST =====\n")

test_metrics = evaluate_model(
    final_model,
    esm_test,
    fp_test,
    phys_test,
    y_test,
    num_classes=num_classes,
    batch_size=64
)

for k, v in test_metrics.items():
    if (k!='cm'):
        print(f"{k}: {v:.4f}")
    print(v)


# ==========================================================
# save final model
# ==========================================================

torch.save(
    final_model.state_dict(),
    "final_peptide_model.pt"
)

print("\nSaved: final_peptide_model.pt")