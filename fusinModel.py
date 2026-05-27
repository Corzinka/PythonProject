import numpy as np

from sklearn.utils import compute_class_weight
import torch

from pretrain.pretrain import get_feature

from test_model.test import evaluate_model, plot_all_metrics

from train.model import AttentionFusionModel
from train.train import data_slpit, train_model
from utils import load_dataframe, normalize_feature

# ==========================================================
# pretrain
def fusionModel(train_df, test_df):
    classes = np.unique(train_df["label"])
    print('Начата подготовка ...')

    esm = get_feature(train_df, 'esm')
    fingerprints = get_feature(train_df, 'fingerprint')
    physchem = get_feature(train_df, 'physchem')

    esm, esm_scaler = normalize_feature(esm, fit=True)
    fingerprints, fp_scaler = normalize_feature(fingerprints, fit=True)
    physchem, physchem_scaler = normalize_feature(physchem, fit=True)

    # ==========================================================

    esm_test = get_feature(test_df, 'esm')
    fp_test = get_feature(test_df, 'fingerprint')
    phys_test = get_feature(test_df, 'physchem')

    esm_test, _ = normalize_feature(esm_test, scaler=esm_scaler)
    fp_test, _ = normalize_feature(fp_test, scaler=fp_scaler)
    phys_test, _ = normalize_feature(phys_test, scaler=physchem_scaler)

    y_test = torch.tensor(test_df['label'].values, dtype=torch.long)

    # ==========================================================
    # вычисляем веса

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_df['label'].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ==========================================================
    # train/validation split

    esm_train, esm_val, fp_train, fp_val, phys_train, phys_val, y_train, y_val = data_slpit(esm, fingerprints, physchem, train_df)

    # ==========================================================
    # train

    print('start train')

    results_test = []
    iterations = 10

    esm_pool, fp_pool, phys_pool = [], [], []

    for i in range(iterations):
        print(f"\n=== ITERATION {i} ===")

        model = AttentionFusionModel(
            esm_dim=esm.shape[1],
            fp_dim=fingerprints.shape[1],
            phys_dim=physchem.shape[1],
            num_classes=classes.size
        )

        model = train_model(
            model,
            esm_train, fp_train, phys_train, y_train,
            esm_val, fp_val, phys_val, y_val,
            criterion,
            classes.size,
            epochs=50,
            batch_size=64,
            patience=7
        )

        metrics = evaluate_model(
            model,
            esm_test, fp_test, phys_test, y_test,
            classes.size,
            batch_size=64,
        )

        results_test.append(metrics)

    print("\n===== Результаты тестирования (Средние значения) =====\n")
    print(f"| roc_auc | pr_auc | lg_loss | f1 | acc | mcc | pr | recall |")

    roc_auc_arr = np.array([res['roc_auc'] for res in results_test])
    pr_auc_arr = np.array([res['pr_auc'] for res in results_test])
    lg_loss_arr = np.array([res['lg_loss'] for res in results_test])
    f1_macro_arr = np.array([res['f1'] for res in results_test])
    acc_arr = np.array([res['acc'] for res in results_test])
    mcc_arr = np.array([res['mcc'] for res in results_test])
    pr_arr = np.array([res['pr'] for res in results_test])
    recall_arr = np.array([res['recall'] for res in results_test])

    all_y_true = np.concatenate([r["y_true"] for r in results_test])
    all_y_prob = np.concatenate([r["y_prob"] for r in results_test])
    all_y_pred = np.concatenate([r["y_pred"] for r in results_test])

    print(f"| {roc_auc_arr.mean():.4f} ± {roc_auc_arr.std():.4f}",
        f"| {pr_auc_arr.mean():.4f} ± {pr_auc_arr.std():.4f}",
        f"| {lg_loss_arr.mean():.4f} ± {lg_loss_arr.std():.4f}",
        f"| {f1_macro_arr.mean():.4f} ± {f1_macro_arr.std():.4f}",
        f"| {acc_arr.mean():.4f} ± {acc_arr.std():.4f}",
        f"| {mcc_arr.mean():.4f} ± {mcc_arr.std():.4f}",
        f"| {pr_arr.mean():.4f} ± {pr_arr.std():.4f}",
        f"| {recall_arr.mean():.4f} ± {recall_arr.std():.4f}"
    )