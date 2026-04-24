import numpy as np

from sklearn.utils import compute_class_weight
import torch

from sklearn.model_selection import train_test_split

from pretrain.pretrain import get_feature

from test_model.test import evaluate_model

from train.model import AttentionFusionModel
from train.train import data_slpit, predict_with_uncertainty, select_candidates, train_model
from utils import load_dataframe, normalize_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_test = []

# ==========================================================
# load data

train_df, test_df = load_dataframe()

classes = np.unique(train_df["label"])
print(f'Размер данных: { len(train_df) }')
print(f"Классы: { classes }, кол-во классов: { classes.size }")

# ==========================================================
# pretrain

print('start pre-train')

esm = get_feature(train_df, 'esm', device).to(device)
fingerprints = get_feature(train_df, 'fingerprint', device).to(device)
physchem = get_feature(train_df, 'physchem', device).to(device)

esm, esm_scaler = normalize_feature(esm, fit=True)
fingerprints, fingerprints_scaler = normalize_feature(fingerprints, fit=True)
physchem, physchem_scaler = normalize_feature(physchem, fit=True)

# ==========================================================

esm_test = get_feature(test_df, 'esm', device).to(device)
fp_test = get_feature(test_df, 'fingerprint', device).to(device)
phys_test = get_feature(test_df, 'physchem', device).to(device)

esm_test, _ = normalize_feature(esm_test, scaler=esm_scaler)
fp_test, _ = normalize_feature(fp_test, scaler=fingerprints_scaler)
phys_test, _ = normalize_feature(phys_test, scaler=physchem_scaler)

y_test = torch.tensor(test_df['label'].values, dtype=torch.long, device=device)

# вычисляем веса
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_df['label'].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# ==========================================================
# Model init

print('start train')

# for i in range(1, 6):
model = AttentionFusionModel(
    esm_dim=esm.shape[1],
    fp_dim=fingerprints.shape[1],
    phys_dim=physchem.shape[1],
    num_classes=classes.size
).to(device)

# ==========================================================
# train/validation split

esm_train, esm_val, fp_train, fp_val, phys_train, phys_val, y_train, y_val = data_slpit(esm, fingerprints, physchem, train_df, device)

# ==========================================================
# train

model = train_model(
    model,
    esm_train, fp_train, phys_train, y_train,
    esm_val, fp_val, phys_val, y_val,
    criterion,
    device,
    classes.size,
    epochs=50,
    batch_size=64,
    patience=7
)

metrics = evaluate_model(
    model,
    esm_test,
    fp_test,
    phys_test,
    y_test,
    classes.size,
    batch_size=64,
)

results_test.append(metrics)

print("\n===== TEST RESULTS =====\n")
print(f"| roc_auc | pr_auc | f1_macro | acc |")
for res in results_test:
    #print(f"|---------|--------|----------|-----|")
    print(f"| {res['roc_auc']:.4f} | {res['pr_auc']:.4f} | {res['f1_macro']:.4f} | {res['accuracy']:.4f} |")

'''
model = train_model(
    model,
    esm_train, fp_train, phys_train, y_train,
    esm_val, fp_val, phys_val, y_val,
    criterion,
    device,
    classes.size,
    epochs=50,
    batch_size=64,
    patience=7
)
mean_probs, uncertainty = predict_with_uncertainty(
    model, esm_train, fp_train, phys_train
)
selected_idx = select_candidates(mean_probs, uncertainty)

print(selected_idx)

print('esm', esm_train[selected_idx])
print('fp', fp_train[selected_idx])
print('phys', phys_train[selected_idx])
'''

'''
iterations = 5
for i in range(iterations):
    print(f"\n=== ITERATION {i} ===")

    # train
    model = train_model(
        model,
        esm_train, fp_train, phys_train, y_train,
        esm_val, fp_val, phys_val, y_val,
        criterion,
        device,
        classes.size,
        epochs=50,
        batch_size=64,
        patience=7
    )

    # select
    mean_probs, uncertainty = predict_with_uncertainty(
        model, esm_pool, fp_pool, phys_pool
    )
    selected_idx = select_candidates(mean_probs, uncertainty)

    # physics (заглушка пока)
    new_esm = esm_pool[selected_idx]
    new_fp = fp_pool[selected_idx]
    new_phys = phys_pool[selected_idx]

    # TODO: здесь будет docking / MD

    # fake labels (пока)
    new_y = torch.randint(0, y_train.max()+1, (len(selected_idx),))

    # add to train
    esm_train = torch.cat([esm_train, new_esm])
    fp_train = torch.cat([fp_train, new_fp])
    phys_train = torch.cat([phys_train, new_phys])
    y_train = torch.cat([y_train, new_y])

    # remove from pool
    mask = torch.ones(len(esm_pool), dtype=bool)
    mask[selected_idx] = False

    esm_pool = esm_pool[mask]
    fp_pool = fp_pool[mask]
    phys_pool = phys_pool[mask]
'''
