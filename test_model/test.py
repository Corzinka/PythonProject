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

    all_probs = np.array(all_probs)

    # ===== метрики =====
    # roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    pr_auc = average_precision_score(F.one_hot(torch.tensor(all_targets, dtype=torch.long), num_classes).numpy(), all_probs, average='weighted')

    f1 = f1_score(all_targets, all_preds, average='macro')
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    return {
        # "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_macro": f1,
        "accuracy": acc,
        "cm": cm,
    }
