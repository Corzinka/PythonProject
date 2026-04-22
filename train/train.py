import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import ( 
    roc_auc_score,
    average_precision_score,
    f1_score, 
    accuracy_score, 
    confusion_matrix 
)
from copy import deepcopy

from utils import get_batches

def predict_with_uncertainty(model, esm, fp, phys, n_samples=20):
    model.train()  # ВАЖНО: включает dropout

    preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(esm, fp, phys)
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.unsqueeze(0))

    preds = torch.cat(preds, dim=0)  # (n_samples, batch, num_classes)

    mean_probs = preds.mean(dim=0)
    std_probs = preds.std(dim=0)

    # uncertainty = среднее std по классам
    uncertainty = std_probs.mean(dim=1)

    return mean_probs, uncertainty

# =========================================
# training loop

def train_model(
    model,
    esm, fp, phys, y,
    esm_val, fp_val, phys_val, y_val,
    criterion,
    device,
    num_classes,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    patience=5
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1 = 0
    best_model = deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        train_losses = []
        all_preds, all_targets, all_probs = [], [], []

        for b_esm, b_fp, b_phys, b_y in get_batches(esm, fp, phys, y, batch_size):
            optimizer.zero_grad()

            logits = model(b_esm, b_fp, b_phys)
            loss = criterion(logits, b_y)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.detach().cpu().numpy())

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            
            all_targets.extend(b_y.detach().cpu().numpy())

        train_roc_auc = roc_auc_score(all_targets, np.array(all_probs), multi_class='ovr', average='macro')
        train_pr_auc = average_precision_score(F.one_hot(torch.tensor(all_targets, dtype=torch.long), num_classes).numpy(), all_probs, average='weighted')

        train_f1 = f1_score(all_targets, all_preds, average='macro')
        train_acc = accuracy_score(all_targets, all_preds)
        train_cm = confusion_matrix(all_targets, all_preds)

        # ===== VALIDATION =====
        model.eval()
        val_losses = []
        val_preds, val_targets, val_probs = [], [], []

        with torch.no_grad():
            for b_esm, b_fp, b_phys, b_y in get_batches(esm_val, fp_val, phys_val, y_val, batch_size):
                logits = model(b_esm, b_fp, b_phys)
                loss = criterion(logits, b_y)

                val_losses.append(loss.item())

                probs = torch.softmax(logits, dim=1)
                val_probs.extend(probs.cpu().numpy())

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())

                val_targets.extend(b_y.cpu().numpy())

        val_roc_auc = roc_auc_score(val_targets, np.array(val_probs), multi_class='ovr', average='macro')
        val_pr_auc = average_precision_score(F.one_hot(torch.tensor(val_targets, dtype=torch.long), num_classes).numpy(), val_probs, average='weighted')

        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_acc = accuracy_score(val_targets, val_preds)
        val_cm = confusion_matrix(val_targets, val_preds)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train ROC_AUC: {train_roc_auc:.4f} | "
            f"Val ROC_AUC: {val_roc_auc:.4f} | "
            f"Train PR_AUC: {train_pr_auc:.4f} | "
            f"Val PR_AUC: {val_pr_auc:.4f} | "
            f"Train ACC: {train_acc:.4f} | "
            f"Val ACC: {val_acc:.4f} | "
            f"Train Loss: {np.mean(train_losses):.4f} | "
            f"Val Loss: {np.mean(val_losses):.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val F1: {val_f1:.4f}"
            f"\nConfusion matrix (train):\n{train_cm}"
            f"\nConfusion matrix (val):\n{val_cm}"
        )

        # ===== EARLY STOPPING =====
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break

    # восстановление лучшей модели
    model.load_state_dict(best_model)

    return model