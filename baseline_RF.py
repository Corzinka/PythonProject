import numpy as np

from sklearn.ensemble import RandomForestClassifier
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

def baseline_RF_func(train_df, test_df):
    # ==========================================================
    # features

    print("Подготовка признаков...")

    esm = get_feature(train_df, 'esm')
    fingerprints = get_feature(train_df, 'fingerprint')
    physchem = get_feature(train_df, 'physchem')

    esm, esm_scaler = normalize_feature(esm, fit=True)
    fingerprints, fp_scaler = normalize_feature(fingerprints, fit=True)
    physchem, phys_scaler = normalize_feature(physchem, fit=True)

    esm_test = get_feature(test_df, 'esm')
    fp_test = get_feature(test_df, 'fingerprint')
    phys_test = get_feature(test_df, 'physchem')

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
        { 
            "name": "esm",
            "X": esm,
            "y": y,
            "X_test": esm_test,
            "y_test": y_test,
        },
        { 
            "name": "fp",
            "X": fingerprints,
            "y": y,
            "X_test": fp_test,
            "y_test": y_test,

        },
        { 
            "name": "ph",
            "X": physchem,
            "y": y,
            "X_test": phys_test,
            "y_test": y_test,
        },
        { 
            "name": "esm-fp",
            "X": np.concatenate([esm, fingerprints], axis=1),
            "y": y,
            "X_test": np.concatenate([esm_test, fp_test], axis=1),
            "y_test": y_test,
        },
        { 
            "name": "fp-ph",
            "X": np.concatenate([fingerprints, physchem], axis=1),
            "y": y,
            "X_test": np.concatenate([fp_test, phys_test], axis=1),
            "y_test": y_test,
        },
        { 
            "name": "esm-ph",
            "X": np.concatenate([esm, physchem], axis=1),
            "y": y,
            "X_test": np.concatenate([esm_test, phys_test], axis=1),
            "y_test": y_test,
        },
        { 
            "name": "all",
            "X": np.concatenate([esm, fingerprints, physchem], axis=1),
            "y": y,
            "X_test": np.concatenate([esm_test, fp_test, phys_test], axis=1),
            "y_test": y_test,
        },
    ]

    # ==========================================================
    # train/eval

    iterations = 10
    final_results = []

    for variant in variants:
        print(f"\n===== {variant['name']} =====")
        results_test = []

        X = variant["X"]
        y = variant["y"]
        X_test = variant["X_test"]
        y_test = variant["y_test"]

        for seed in range(iterations):
            print(f"\n=== ITERATION {seed} ===")

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=seed
            )

            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

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
            variant["name"],
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

    print("\n===== Средние результаты RF =====\n")
    print("\t | roc_auc         | pr_auc          | lg_loss         | f1              | acc             | mcc             | pr              | recall|")
    for a in final_results:
        print(f"|{a[0]}\t",
            f"| {a[1].mean():.4f} ± {a[1].std():.4f}",
            f"| {a[2].mean():.4f} ± {a[2].std():.4f}",
            f"| {a[3].mean():.4f} ± {a[3].std():.4f}",
            f"| {a[4].mean():.4f} ± {a[4].std():.4f}",
            f"| {a[5].mean():.4f} ± {a[5].std():.4f}",
            f"| {a[6].mean():.4f} ± {a[6].std():.4f}",
            f"| {a[7].mean():.4f} ± {a[7].std():.4f}",
            f"| {a[8].mean():.4f} ± {a[8].std():.4f} |"
        )
    return final_results

'''
===== Средние результаты RF =====
         | roc_auc         | pr_auc          | lg_loss         | f1              | acc             | mcc             | pr              | recall         
esm      | 0.7827 ± 0.0078 | 0.7550 ± 0.0090 | 0.5491 ± 0.0067 | 0.6970 ± 0.0184 | 0.6971 ± 0.0172 | 0.3944 ± 0.0345 | 0.6972 ± 0.0174 | 0.6971 ± 0.0242
fp       | 0.8283 ± 0.0083 | 0.8028 ± 0.0137 | 0.5029 ± 0.0104 | 0.7545 ± 0.0108 | 0.7433 ± 0.0111 | 0.4887 ± 0.0224 | 0.7230 ± 0.0102 | 0.7890 ± 0.0130
ph       | 0.7383 ± 0.0097 | 0.7184 ± 0.0126 | 0.6050 ± 0.0082 | 0.6755 ± 0.0181 | 0.6698 ± 0.0172 | 0.3398 ± 0.0344 | 0.6639 ± 0.0164 | 0.6878 ± 0.0224 
esm-fp   | 0.7950 ± 0.0104 | 0.7694 ± 0.0141 | 0.5386 ± 0.0084 | 0.7069 ± 0.0132 | 0.7073 ± 0.0118 | 0.4146 ± 0.0235 | 0.7076 ± 0.0110 | 0.7064 ± 0.0186 
fp-ph    | 0.8314 ± 0.0081 | 0.8064 ± 0.0116 | 0.5015 ± 0.0091 | 0.7544 ± 0.0106 | 0.7448 ± 0.0105 | 0.4912 ± 0.0213 | 0.7269 ± 0.0100 | 0.7843 ± 0.0148 
esm-ph   | 0.7837 ± 0.0095 | 0.7574 ± 0.0109 | 0.5486 ± 0.0074 | 0.6999 ± 0.0131 | 0.7023 ± 0.0101 | 0.4049 ± 0.0201 | 0.7055 ± 0.0090 | 0.6948 ± 0.0221 
all      | 0.7947 ± 0.0068 | 0.7682 ± 0.0083 | 0.5388 ± 0.0059 | 0.7026 ± 0.0143 | 0.7052 ± 0.0107 | 0.4107 ± 0.0213 | 0.7086 ± 0.0088 | 0.6971 ± 0.0239 
'''
