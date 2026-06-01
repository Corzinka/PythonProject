import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from baseline_1 import model_1
from baseline_2 import model_2
from baseline_3 import model_3
from baseline_RF import baseline_RF_func
from homology_control import create_cluster, create_fasta_files
from other.ConservativeSubstitutions import augmented_data
from utils import load_dataframe

def plot_all_metrics(results, variants=None):
    """
    Рисует ROC, PR и confusion matrix для каждого варианта признаков.
    results: словарь, возвращаемый model_1
    variants: список вариантов, если None — берутся все VARIANTS
    """
    if variants is None:
        variants = list(results["test"].keys())

    # ========= ROC Curves =========
    plt.figure(figsize=(8,6))
    for variant in variants:
        y_prob = results["test"][variant]["y_prob"]
        y_true = results["test"][variant]["y_true"] if "y_true" in results["test"][variant] else None
        if y_true is None:
            raise ValueError("y_true not found in results['test'][variant], add it in evaluate_metrics")

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{variant} (AUC={roc_auc:.3f})")

    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/{activity}_{model_name}_roc_curve_all_variants.png", dpi=300)
    plt.close()

    # ========= PR Curves =========
    plt.figure(figsize=(8,6))
    for variant in variants:
        y_prob = results["test"][variant]["y_prob"]
        y_true = results["test"][variant]["y_true"]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{variant} (AUC={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/{activity}_{model_name}_pr_curve_all_variants.png", dpi=300)
    plt.close()

train_df, test_df = load_dataframe(train_path="data/train_ABE.xlsx", test_path="data/test_ABE.xlsx")
activity = "ABE"
print(activity)
print(f'Размер train: {len(train_df)}')
print(f'Размер test: {len(test_df)}')
print(f'Классы: {np.unique(train_df["label"])}')

create_fasta_files(train_df, test_df)
train_df = create_cluster(train_df)

results, _ = model_1(train_df, test_df)
model_name = "RF"
plot_all_metrics(results)

model_2(train_df, test_df)
model_name = "LGBM"
plot_all_metrics(results)

#model_3(train_df, test_df)


train_df, test_df = load_dataframe(train_path="data/train_ACE.xlsx", test_path="data/test_ACE.xlsx")
activity = "ACE"
print(activity)
print(f'Размер train: {len(train_df)}')
print(f'Размер test: {len(test_df)}')
print(f'Классы: {np.unique(train_df["label"])}')

create_fasta_files(train_df, test_df)
train_df = create_cluster(train_df)

model_1(train_df, test_df)
model_name = "RF"
plot_all_metrics(results)

model_2(train_df, test_df)
model_name = "LGBM"
plot_all_metrics(results)

#model_3(train_df, test_df)


train_df, test_df = load_dataframe(train_path="data/train_AOE.xlsx", test_path="data/test_AOE.xlsx")
activity = "AOE"
print(activity)
print(f'Размер train: {len(train_df)}')
print(f'Размер test: {len(test_df)}')
print(f'Классы: {np.unique(train_df["label"])}')

create_fasta_files(train_df, test_df)
train_df = create_cluster(train_df)

model_1(train_df, test_df)
model_name = "RF"
plot_all_metrics(results)

model_2(train_df, test_df)
model_name = "LGBM"
plot_all_metrics(results)

#model_3(train_df, test_df)
