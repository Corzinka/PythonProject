import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    auc, confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score,
    log_loss, f1_score, accuracy_score, matthews_corrcoef,
    precision_score, recall_score
)
from pretrain.pretrain import get_feature
from utils import normalize_feature

# ==========================================================
# reproducibility
# ==========================================================
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

VARIANTS = ["esm", "fp", "ph", "esm-fp", "fp-ph", "esm-ph", "all"]

# ==========================================================
# Helper functions
# ==========================================================
def build_features(esm, fp, ph, variant):
    """Собирает матрицу признаков в зависимости от варианта."""
    mapping = {
        "esm":    [esm],
        "fp":     [fp],
        "ph":     [ph],
        "esm-fp": [esm, fp],
        "fp-ph":  [fp, ph],
        "esm-ph": [esm, ph],
        "all":    [esm, fp, ph],
    }
    parts = mapping.get(variant)
    if parts is None:
        raise ValueError(f"Unknown variant: {variant!r}")
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

def find_best_threshold(y_true, y_prob):
    """Ищет порог с максимальным F1 по PR-кривой."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) == 0,
        0.0,
        2 * precision * recall / (precision + recall),
    )
    return float(thresholds[np.argmax(f1_scores[:-1])])

def extract_features(df):
    """Возвращает все три типа признаков: esm, fingerprint, physchem"""
    esm = get_feature(df, "esm")
    fp = get_feature(df, "fingerprint")
    phys = get_feature(df, "physchem")
    return esm, fp, phys

def normalize_fold(train_raw, val_raw):
    """Нормализует train и val признаки и возвращает нормализованные данные и scaler"""
    train_norm, scaler = normalize_feature(train_raw, fit=True)
    val_norm, _ = normalize_feature(val_raw, scaler=scaler)
    return train_norm, val_norm, scaler

def evaluate_metrics(y_true, y_prob):
    """Вычисляет все метрики классификации"""
    y_pred = (y_prob > find_best_threshold(y_true, y_prob)).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_prob": y_prob,
    }
    return metrics

# ==========================================================
# Core function
# ==========================================================
def model_1(train_df, test_df, n_splits=5, save_model_path=None):
    """
    Обучение RandomForest с CV и финальной тренировкой на полном train set.
    Возвращает словарь с CV и тестовыми метриками, а также модель.
    """
    results = {
        "cv": {variant: [] for variant in VARIANTS},
        "test": {variant: [] for variant in VARIANTS},
    }

    # ================== Extract features ==================
    print("Extracting train features...")
    esm_raw, fp_raw, phys_raw = extract_features(train_df)
    print("Extracting test features...")
    esm_test_raw, fp_test_raw, phys_test_raw = extract_features(test_df)

    y_all = train_df["label"].values
    y_test = test_df["label"].values

    for variant in VARIANTS:
        # ================== Cross-validation ==================
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        print(f"\n===== CROSS VALIDATION {variant} =====\n")
        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(X=np.zeros(len(train_df)), y=y_all, groups=train_df["cluster"])
        ):
            print(f"\n===== FOLD {fold+1} =====")

            # Split features
            esm_train_raw, esm_val_raw = esm_raw[train_idx], esm_raw[val_idx]
            fp_train_raw, fp_val_raw = fp_raw[train_idx], fp_raw[val_idx]
            phys_train_raw, phys_val_raw = phys_raw[train_idx], phys_raw[val_idx]

            y_train_np, y_val_np = y_all[train_idx], y_all[val_idx]

            # Normalize
            esm_train, esm_val, _ = normalize_fold(esm_train_raw, esm_val_raw)
            fp_train, fp_val, _ = normalize_fold(fp_train_raw, fp_val_raw)
            phys_train, phys_val, _ = normalize_fold(phys_train_raw, phys_val_raw)

            X_train = build_features(esm_train, fp_train, phys_train, variant)
            X_val = build_features(esm_val, fp_val, phys_val, variant)

            # Model
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight='balanced',
                random_state=SEED,
                n_jobs=-1
            )
            model.fit(X_train, y_train_np)

            # Evaluate
            y_prob = model.predict_proba(X_val)[:, 1]
            fold_metrics = evaluate_metrics(y_val_np, y_prob)
            results["cv"][variant].append(fold_metrics)

        # ================== Final training on full train set ==================
        esm_all, esm_scaler = normalize_feature(esm_raw, fit=True)
        fp_all, fp_scaler = normalize_feature(fp_raw, fit=True)
        phys_all, phys_scaler = normalize_feature(phys_raw, fit=True)

        esm_test, _ = normalize_feature(esm_test_raw, scaler=esm_scaler)
        fp_test, _ = normalize_feature(fp_test_raw, scaler=fp_scaler)
        phys_test, _ = normalize_feature(phys_test_raw, scaler=phys_scaler)

        X_all = build_features(esm_all, fp_all, phys_all, variant)
        X_test = build_features(esm_test, fp_test, phys_test, variant)

        final_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1
        )
        final_model.fit(X_all, y_all)

        # ================== Final evaluation on test ==================
        y_prob_test = final_model.predict_proba(X_test)[:, 1]

        results["test"][variant] = evaluate_metrics(y_test, y_prob_test)

    # ================== Summarize CV ==================
    print("\n===== CV RESULTS =====\n")
    metrics_names = ["roc_auc", "pr_auc", "log_loss", "f1", "accuracy", "mcc", "precision", "recall"]
    for variant in VARIANTS:
        print(f"\n--- {variant} ---")
        for metric in metrics_names:
            arr = np.array([fold_result[metric] for fold_result in results["cv"][variant]])
            print(f"{metric:10s}: {arr.mean():.4f} ± {arr.std():.4f}")

    print("\n===== FINAL TEST METRICS =====\n")
    for variant in VARIANTS:
        print(f"\n--- {variant} ---")
        for k, v in results["test"][variant].items():
            if k in ["y_prob", "y_true"]:
                continue
            if k != "confusion_matrix":
                print(f"{k}: {v:.4f}")
            else:
                print(v)

    return results, final_model

