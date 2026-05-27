import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score
)

from pretrain.pretrain import get_feature
from utils import normalize_feature

def baseline_LightGBM_func(train_df, test_df):
    # ==========================================================
    # features

    print("Подготовка признаков...")

    esm = get_feature(train_df, "esm")
    fingerprints = get_feature(train_df, "fingerprint")
    physchem = get_feature(train_df, "physchem")

    esm, esm_scaler = normalize_feature(esm, fit=True)
    fingerprints, fp_scaler = normalize_feature(fingerprints, fit=True)
    physchem, phys_scaler = normalize_feature(physchem, fit=True)

    esm_test = get_feature(test_df, "esm")
    fp_test = get_feature(test_df, "fingerprint")
    phys_test = get_feature(test_df, "physchem")

    esm_test, _ = normalize_feature(esm_test, scaler=esm_scaler)
    fp_test, _ = normalize_feature(fp_test, scaler=fp_scaler)
    phys_test, _ = normalize_feature(phys_test, scaler=phys_scaler)


    # ==========================================================
    # labels

    y = train_df["label"].values
    y_test = test_df["label"].values

    # ==========================================================
    # variants (ablation)

    variants = [
        ["esm", esm, esm_test],
        ["fp", fingerprints, fp_test],
        ["ph", physchem, phys_test],
        ["esm-fp", np.concatenate([esm, fingerprints], axis=1), np.concatenate([esm_test, fp_test], axis=1)],
        ["fp-ph", np.concatenate([fingerprints, physchem], axis=1), np.concatenate([fp_test, phys_test], axis=1)],
        ["esm-ph", np.concatenate([esm, physchem], axis=1), np.concatenate([esm_test, phys_test], axis=1)],
        ["all", np.concatenate([esm, fingerprints, physchem], axis=1), np.concatenate([esm_test, fp_test, phys_test], axis=1)],
    ]


    # ==========================================================
    # train/eval

    iterations = 10
    final_results = []

    for variant_name, X, X_test in variants:

        print(f"\n===== {variant_name} =====")
        results_test = []

        for seed in range(iterations):
            print(f"ITERATION {seed}")

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=seed
            )

            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(30, verbose=False)
                ]
            )

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)

            metrics = {
                "roc_auc": roc_auc_score(y_test, y_prob),
                "pr_auc": average_precision_score(y_test, y_prob),
                "lg_loss": log_loss(y_test, y_prob),
                "f1": f1_score(y_test, y_pred),
                "acc": accuracy_score(y_test, y_pred),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "pr": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred)
            }

            results_test.append(metrics)

        roc_auc_arr = np.array([r["roc_auc"] for r in results_test])
        pr_auc_arr = np.array([r["pr_auc"] for r in results_test])
        lg_loss_arr = np.array([r["lg_loss"] for r in results_test])
        f1_arr = np.array([r["f1"] for r in results_test])
        acc_arr = np.array([r["acc"] for r in results_test])
        mcc_arr = np.array([r["mcc"] for r in results_test])
        pr_arr = np.array([r["pr"] for r in results_test])
        recall_arr = np.array([r["recall"] for r in results_test])

        final_results.append([
            variant_name,
            roc_auc_arr,
            pr_auc_arr,
            lg_loss_arr,
            f1_arr,
            acc_arr,
            mcc_arr,
            pr_arr,
            recall_arr
        ])


    # ==========================================================
    # aggregate

    print("\n===== Средние результаты LightGBM =====\n")
    print("\t | roc_auc | pr_auc | lg_loss | f1 | acc | mcc | pr | recall |")
    for a in final_results:
        print(
            f"|{a[0]:8}",
            f"| {a[1].mean():.4f} ± {a[1].std():.4f}",
            f"| {a[2].mean():.4f} ± {a[2].std():.4f}",
            f"| {a[3].mean():.4f} ± {a[3].std():.4f}",
            f"| {a[4].mean():.4f} ± {a[4].std():.4f}",
            f"| {a[5].mean():.4f} ± {a[5].std():.4f}",
            f"| {a[6].mean():.4f} ± {a[6].std():.4f}",
            f"| {a[7].mean():.4f} ± {a[7].std():.4f}",
            f"| {a[8].mean():.4f} ± {a[8].std():.4f} |"
        )

'''
===== Средние результаты LightGBM =====
         | roc_auc         | pr_auc          | lg_loss         | f1              | acc             | mcc             | pr              | recall
esm      | 0.7793 ± 0.0080 | 0.7532 ± 0.0146 | 0.5642 ± 0.0110 | 0.6927 ± 0.0153 | 0.6980 ± 0.0096 | 0.3967 ± 0.0191 | 0.7050 ± 0.0122 | 0.6820 ± 0.0326
fp       | 0.8138 ± 0.0079 | 0.7858 ± 0.0104 | 0.5402 ± 0.0199 | 0.7381 ± 0.0140 | 0.7392 ± 0.0147 | 0.4786 ± 0.0293 | 0.7416 ± 0.0171 | 0.7349 ± 0.0156
ph       | 0.7158 ± 0.0153 | 0.6874 ± 0.0224 | 0.6307 ± 0.0262 | 0.6701 ± 0.0226 | 0.6596 ± 0.0185 | 0.3203 ± 0.0378 | 0.6496 ± 0.0150 | 0.6924 ± 0.0359
esm-fp   | 0.8040 ± 0.0142 | 0.7841 ± 0.0195 | 0.5480 ± 0.0259 | 0.7208 ± 0.0114 | 0.7262 ± 0.0097 | 0.4530 ± 0.0194 | 0.7354 ± 0.0144 | 0.7076 ± 0.0228
fp-ph    | 0.8089 ± 0.0080 | 0.7797 ± 0.0099 | 0.5411 ± 0.0202 | 0.7320 ± 0.0145 | 0.7326 ± 0.0126 | 0.4653 ± 0.0252 | 0.7334 ± 0.0111 | 0.7308 ± 0.0218
esm-ph   | 0.7845 ± 0.0085 | 0.7552 ± 0.0149 | 0.5586 ± 0.0120 | 0.7005 ± 0.0152 | 0.7038 ± 0.0118 | 0.4081 ± 0.0235 | 0.7085 ± 0.0152 | 0.6936 ± 0.0291
all      | 0.8078 ± 0.0114 | 0.7861 ± 0.0167 | 0.5441 ± 0.0283 | 0.7257 ± 0.0118 | 0.7297 ± 0.0108 | 0.4598 ± 0.0216 | 0.7366 ± 0.0146 | 0.7157 ± 0.0205
'''