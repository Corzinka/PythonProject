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

        "y_true": all_targets,
        "y_prob": all_probs,
        "y_pred": all_preds
    }


import seaborn as sns
from sklearn.calibration import calibration_curve


def plot_all_metrics(y_true, y_prob, y_pred):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, np.array(y_prob)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, np.array(y_prob)[:, 1])
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label=f"AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("pr_curve.png", dpi=300)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
