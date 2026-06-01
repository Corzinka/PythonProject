from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import ( 
    auc,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    f1_score, 
    accuracy_score, 
    confusion_matrix,
    roc_curve 
)
from utils import get_batches



def evaluate_model(model, esm, fp, phys, y, num_classes, batch_size):
    model.eval()

    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for b_esm, b_fp, b_phys, b_y in get_batches(esm, fp, phys, y, batch_size):
            logits = model(b_esm, b_fp, b_phys)

            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

            all_targets.extend(b_y.cpu().numpy())

    # ===== метрики =====
    roc_auc = roc_auc_score(all_targets, np.array(all_probs)[:, 1])
    pr_auc = average_precision_score(F.one_hot(torch.tensor(all_targets, dtype=torch.long), num_classes).numpy(), all_probs, average='weighted')
    lg_loss = log_loss(all_targets, all_probs)

    f1 = f1_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    mcc = matthews_corrcoef(all_targets, all_preds)

    pr = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)

    cm = confusion_matrix(all_targets, all_preds)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "lg_loss": lg_loss,
        "f1": f1,
        "acc": acc,
        "mcc": mcc,
        "pr": pr,
        "recall": recall,
        "cm": cm,
    }
