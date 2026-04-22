import torch

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from utils import load_dataframe, get_mol_from_seq, get_feature

from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

X = np.array(get_mol_from_seq(train_df['sequence'])) # train_df['sequence'].values
y = train_df['label'].values

kf = KFold(n_splits=5, shuffle=True)
all_val_f1 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n========== FOLD {fold+1} ==========")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train = torch.tensor(
        get_feature(X_train, 'fingerprint', device), # TypeError: Passed values must be RDKit Mol objects or SMILES strings, got types: {<class 'NoneType'>, <class 'rdkit.Chem.rdchem.Mol'>}
        dtype=torch.float32,
        device=device
    )

    X_val = torch.tensor(
        get_feature(X_val, 'fingerprint', device),
        dtype=torch.float32,
        device=device
    )

    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    y_val = torch.tensor(y_val, dtype=torch.long, device=device)

    input_size = 2048
    output_size = 32

    # Найханова Л.В
    
    classifier = torch.nn.Sequential(
        torch.nn.Linear(input_size, 512),
        torch.nn.LayerNorm(512),
        torch.nn.GELU(),
        torch.nn.Dropout(0.3),

        torch.nn.Linear(512, 128),
        torch.nn.GELU(),
        torch.nn.Dropout(0.2),

        torch.nn.Linear(128, 32),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),

        torch.nn.Linear(output_size, n_classes)
    ).to(device)
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0
    counter = 0

    for epoch in range(100):
        classifier.train()

        logits = classifier(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(X_val)
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
            val_preds = val_probs.argmax(axis=1)

            val_f1 = f1_score(y_val.cpu(), val_preds, average='weighted')

        # early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            counter = 0
        else:
            counter += 1

        if counter >= 5:
            break

    print(f"Fold {fold+1} best F1: {best_val_f1:.4f}")
    all_val_f1.append(best_val_f1)

print("\n=================================")
print("CV RESULTS")
print("=================================")

print(f"Mean F1: {np.mean(all_val_f1):.4f}")
print(f"Std F1: {np.std(all_val_f1):.4f}")
'''


# ==========================================================
# train/validation split
# ==========================================================

mols = get_mol_from_seq(train_df['sequence'])

# from skfp.model_selection import scaffold_train_test_split
# X_train, X_val, y_train, y_val = scaffold_train_test_split(mols, y) # scaffold дает небольшую прибавку, но тратит довольно много времени
X_train, X_val, y_train, y_val = train_test_split(mols, y)

# ==========================================================
# get feature
# ==========================================================

X_train = get_feature(X_train, 'fingerprint', device)
X_train = torch.tensor(X_train, dtype=torch.float32, device=device) # shape: [B, D_fp]

X_val = get_feature(X_val, 'fingerprint', device)
X_val = torch.tensor(X_val, dtype=torch.float32, device=device) # shape: [B, D_fp]

print("FP shape:", X_train.shape)

'''
'''

# ==========================================================
# classifier init
# ==========================================================

classifier = torch.nn.Sequential(
    torch.nn.Linear(2048, 512),
    torch.nn.LayerNorm(512),
    torch.nn.GELU(),
    torch.nn.Dropout(0.3),

    torch.nn.Linear(512, 128),
    torch.nn.GELU(),
    torch.nn.Dropout(0.2),

    torch.nn.Linear(128, n_classes)
).to(device)

# ==========================================================
# train
# ==========================================================

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    confusion_matrix,
)

import torch.optim as optim
import torch.nn.functional as F

num_classes = n_classes  # число классов

patience = 5
best_val_f1 = 0.0
counter = 0

best_metrics = None
best_state = None

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

n_epochs = 100
for epoch in range(n_epochs):
  classifier.train()
  optimizer.zero_grad()
  logits = classifier(X_train)  # logits размер [batch_size, num_classes]

  # CrossEntropyLoss принимает logits и класс-метки целиком
  loss = criterion(logits, y_train)
  loss.backward()
  optimizer.step()

  classifier.eval()
  with torch.no_grad():
    # предсказания на тренировочном наборе
    probs = F.softmax(logits, dim=1).cpu().numpy()  # вероятности по классам
    preds = probs.argmax(axis=1)
    y_true = y_train.cpu().numpy()

    # предсказания на валидации
    val_logits = classifier(X_val)
    val_probs = F.softmax(val_logits, dim=1).cpu().numpy()
    val_preds = val_probs.argmax(axis=1)
    y_val_true = y_val.cpu().numpy()

    # вычисление метрик
    # ROC-AUC и PR-AUC для мультиклассовой задачи нужно считать для каждого класса отдельно
    if n_classes == 2:
        roc_auc = roc_auc_score(y_true, probs[:, 1])
    else:
       roc_auc = roc_auc_score(y_true, probs, multi_class="ovr")
    pr_auc = average_precision_score(F.one_hot(torch.tensor(y_true, dtype=torch.long), num_classes).numpy(), probs, average='weighted')

    if n_classes == 2:
        val_roc_auc = roc_auc_score(y_val_true, val_probs[:, 1])
    else:
        val_roc_auc = roc_auc_score(y_val_true, val_probs, multi_class='ovr')

    val_f1 = f1_score(y_val_true, val_preds, average='weighted')

    f1 = f1_score(y_true, preds, average='weighted')
    mcc = matthews_corrcoef(y_true, preds)
    acc = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)

  # ==========================================================
  # early stopping
  # ==========================================================
  if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    counter = 0

    best_state = classifier.state_dict()
    best_metrics = {
        "epoch": epoch,
        "train_loss": loss.item(),
        "train_f1": f1,
        "train_roc_auc": roc_auc,
        "train_pr_auc": pr_auc,
        "train_mcc": mcc,
        "train_acc": acc,
        "val_f1": val_f1,
        "val_roc_auc": val_roc_auc,
        "cm": cm,
    }

  else:
    counter += 1

  if counter >= patience:
    print("\nEarly stopping triggered\n")
    break

# ==========================================================
# load best model
# ==========================================================

classifier.load_state_dict(best_state)

# ==========================================================
# print best model metrics
# ==========================================================

print("\n===============================================")
print("BEST MODEL")
print("===============================================")

print(f"Epoch: {best_metrics['epoch']}")
print(f"Train loss: {best_metrics['train_loss']:.4f}")
print(f"Train ROC-AUC: {best_metrics['train_roc_auc']:.4f}")
print(f"Train PR-AUC: {best_metrics['train_pr_auc']:.4f}")
print(f"Train F1: {best_metrics['train_f1']:.4f}")
print(f"Train MCC: {best_metrics['train_mcc']:.4f}")
print(f"Train ACC: {best_metrics['train_acc']:.4f}")
print(f"Validation ROC-AUC: {best_metrics['val_roc_auc']:.4f}")
print(f"Validation F1: {best_metrics['val_f1']:.4f}")
print(cm)

from model.trainer import evaluate as evaluate_model

mols = get_mol_from_seq(test_df['sequence'])
fp = get_feature(mols, 'fingerprint', device)

X_test = torch.tensor(fp, dtype=torch.float32, device=device) # shape: [B, D_fp]
y_test = torch.tensor(test_df['label'].values, dtype=torch.long, device=device)

test_metrics = evaluate_model(classifier, X_test, y_test, n_classes, device)

print("\n===============================================")
print("TEST")
print("===============================================")

print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
print(f"Test F1: {test_metrics['f1']:.4f}")
print(f"Test MCC: {test_metrics['mcc']:.4f}")
print(f"Test ACC: {test_metrics['acc']:.4f}")
print(f"Validation F1: {test_metrics['f1']:.4f}")
print(test_metrics['cm'])

'''





'''
Antibacteria

scafold
===============================================
BEST MODEL
===============================================
Epoch: 58
Train loss: 0.2210
Train ROC-AUC: 0.9721
Train PR-AUC: 0.9713
Train F1: 0.9113
Train MCC: 0.8225
Train ACC: 0.9114
Validation ROC-AUC: 0.9398
Validation F1: 0.8814
[[5155  377]
 [ 515 4485]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.9474
Test PR-AUC: 0.9419
Test F1: 0.8842
Test MCC: 0.7705
Test ACC: 0.8844
Validation F1: 0.8842
[[1441  254]
 [ 138 1557]]

simple split
===============================================
BEST MODEL
===============================================
Epoch: 47
Train loss: 0.2894
Train ROC-AUC: 0.9467
Train PR-AUC: 0.9449
Train F1: 0.8814
Train MCC: 0.7642
Train ACC: 0.8815
Validation ROC-AUC: 0.9486
Validation F1: 0.8890
[[4193  781]
 [ 342 4558]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.9410
Test PR-AUC: 0.9364
Test F1: 0.8760
Test MCC: 0.7532
Test ACC: 0.8761
Validation F1: 0.8760
[[1441  254]
 [ 166 1529]]

simple split
===============================================
BEST MODEL
===============================================
Epoch: 26
Train loss: 0.3140
Train ROC-AUC: 0.9421
Train PR-AUC: 0.9408
Train F1: 0.8716
Train MCC: 0.7499
Train ACC: 0.8720
Validation ROC-AUC: 0.9453
Validation F1: 0.8808
[[4306  688]
 [ 464 4416]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.9387
Test PR-AUC: 0.9346
Test F1: 0.8610
Test MCC: 0.7225
Test ACC: 0.8611
Validation F1: 0.8610
[[1486  209]
 [ 262 1433]]
'''

'''
Anticancer_main

scafold
===============================================
BEST MODEL
===============================================
Epoch: 10
Train loss: 0.5466
Train ROC-AUC: 0.7916
Train PR-AUC: 0.7994
Train F1: 0.6990
Train MCC: 0.4312
Train ACC: 0.7051
Validation ROC-AUC: 0.6675
Validation F1: 0.6283
[[340 217]
 [ 82 463]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.7820
Test PR-AUC: 0.7814
Test F1: 0.7109
Test MCC: 0.4283
Test ACC: 0.7122
Validation F1: 0.7109
[[111  61]
 [ 38 134]]

scafold
===============================================
BEST MODEL
===============================================
Epoch: 1
Train loss: 0.7677
Train ROC-AUC: 0.7143
Train PR-AUC: 0.7040
Train F1: 0.4346
Train MCC: 0.2331
Train ACC: 0.5472
Validation ROC-AUC: 0.6590
Validation F1: 0.6063
[[264 293]
 [ 67 478]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.7489
Test PR-AUC: 0.7476
Test F1: 0.6143
Test MCC: 0.3248
Test ACC: 0.6395
Validation F1: 0.6143
[[ 66 106]
 [ 18 154]]

simple split
===============================================
BEST MODEL
===============================================
Epoch: 2
Train loss: 0.7346
Train ROC-AUC: 0.7308
Train PR-AUC: 0.7319
Train F1: 0.3803
Train MCC: 0.1362
Train ACC: 0.5169
Validation ROC-AUC: 0.7191
Validation F1: 0.6645
[[377 133]
 [185 338]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.7801
Test PR-AUC: 0.7822
Test F1: 0.6740
Test MCC: 0.4029
Test ACC: 0.6860
Validation F1: 0.6740
[[151  21]
 [ 87  85]]

simple split
===============================================
BEST MODEL
===============================================
Epoch: 10
Train loss: 0.5858
Train ROC-AUC: 0.7361
Train PR-AUC: 0.7407
Train F1: 0.6621
Train MCC: 0.3532
Train ACC: 0.6689
Validation ROC-AUC: 0.7636
Validation F1: 0.7008
[[269 249]
 [ 71 444]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.7763
Test PR-AUC: 0.7774
Test F1: 0.6923
Test MCC: 0.3960
Test ACC: 0.6948
Validation F1: 0.6923
[[104  68]
 [ 37 135]]
'''