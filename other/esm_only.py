import torch

import numpy as np
import pandas as pd
from skfp.model_selection import scaffold_train_test_split
from sklearn.model_selection import train_test_split

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

# ==========================================================
# ESM
# ==========================================================

model, batch_converter = load_esm_model(device)
train_data = [(label, seq) for seq, label in train_df.values]

train_iterator = DataIterator(train_data, batch_size)

representations = extract_representations(model, batch_converter, train_iterator, device)
representations = torch.stack(representations) # shape: [B, D_esm]

representations = representations.to(device).float()

print("ESM shape:", representations.shape)  # ESM shape: torch.Size([3000, 320])

# ==========================================================
# train/validation split
# ==========================================================

X = representations
y = torch.tensor(train_df['label'].values, dtype=torch.long, device=device)

# X_train, X_val, y_train, y_val = scaffold_train_test_split(mols, y) # scaffold дает небольшую прибавку, но тратит довольно много времени
X_train, X_val, y_train, y_val = train_test_split(X, y)

# ==========================================================
# classifier init
# ==========================================================

classifier = torch.nn.Sequential(
    torch.nn.Linear(320, 64),
    torch.nn.LayerNorm(64),
    torch.nn.GELU(),
    torch.nn.Dropout(0.3),

    torch.nn.Linear(64, 16),
    torch.nn.GELU(),
    torch.nn.Dropout(0.2),

    torch.nn.Linear(16, n_classes)
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

patience = 10
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

test_data = [(label, seq) for seq, label in test_df.values]

test_iterator = DataIterator(test_data, batch_size)

representations = extract_representations(model, batch_converter, test_iterator, device)
representations = torch.stack(representations) # shape: [B, D_esm]

representations = representations.to(device).float()


X_test = representations
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
Antibacteria
===============================================
BEST MODEL
===============================================
Epoch: 10
Train loss: 0.4633
Train ROC-AUC: 0.9251
Train PR-AUC: 0.9194
Train F1: 0.8590
Train MCC: 0.7200
Train ACC: 0.8591
Validation ROC-AUC: 0.9564
Validation F1: 0.8980
[[4226  730]
 [ 390 4528]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.9529
Test PR-AUC: 0.9474
Test F1: 0.8977
Test MCC: 0.7994
Test ACC: 0.8979
Validation F1: 0.8977
[[1442  253]
 [  93 1602]]

===============================================
BEST MODEL
===============================================
Epoch: 19
Train loss: 0.3638
Train ROC-AUC: 0.9438
Train PR-AUC: 0.9390
Train F1: 0.8959
Train MCC: 0.7948
Train ACC: 0.8961
Validation ROC-AUC: 0.9579
Validation F1: 0.9042
[[4259  648]
 [ 342 4625]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.9574
Test PR-AUC: 0.9544
Test F1: 0.9052
Test MCC: 0.8131
Test ACC: 0.9053
Validation F1: 0.9052
[[1469  226]
 [  95 1600]]
'''

'''
Anticancer_main_train
===============================================
BEST MODEL
===============================================
Epoch: 13
Train loss: 0.6089
Train ROC-AUC: 0.7225
Train PR-AUC: 0.7156
Train F1: 0.6682
Train MCC: 0.3567
Train ACC: 0.6728
Validation ROC-AUC: 0.7346
Validation F1: 0.6736
[[291 228]
 [100 414]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.7055
Test PR-AUC: 0.7193
Test F1: 0.6305
Test MCC: 0.2722
Test ACC: 0.6337
Validation F1: 0.6305
[[ 93  79]
 [ 47 125]]

===============================================
BEST MODEL
===============================================
Epoch: 7
Train loss: 0.6512
Train ROC-AUC: 0.6963
Train PR-AUC: 0.6904
Train F1: 0.6237
Train MCC: 0.2534
Train ACC: 0.6254
Validation ROC-AUC: 0.7557
Validation F1: 0.6985
[[290 225]
 [150 368]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.6865
Test PR-AUC: 0.6941
Test F1: 0.6030
Test MCC: 0.2110
Test ACC: 0.6047
Validation F1: 0.6030
[[ 93  79]
 [ 57 115]]







===============================================
BEST MODEL
===============================================
Epoch: 30
Train loss: 0.5479
Train ROC-AUC: 0.7969
Train PR-AUC: 0.7831
Train F1: 0.7251
Train MCC: 0.4494
Train ACC: 0.7259
Validation ROC-AUC: 0.7394
Validation F1: 0.6861
[[501 227]
 [157 651]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.8418
Test PR-AUC: 0.8370
Test F1: 0.7509
Test MCC: 0.5010
Test ACC: 0.7509
Validation F1: 0.7509
[[100  35]
 [ 35 111]]

===============================================
BEST MODEL
===============================================
Epoch: 23
Train loss: 0.5805
Train ROC-AUC: 0.7668
Train PR-AUC: 0.7535
Train F1: 0.7049
Train MCC: 0.4094
Train ACC: 0.7057
Validation ROC-AUC: 0.7912
Validation F1: 0.7362
[[498 235]
 [189 614]]

===============================================
TEST
===============================================
Test ROC-AUC: 0.8538
Test PR-AUC: 0.8485
Test F1: 0.7899
Test MCC: 0.5792
Test ACC: 0.7900
Validation F1: 0.7899
[[104  31]
 [ 28 118]]
'''