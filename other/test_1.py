from skfp.metrics import multioutput_auroc_score, extract_pos_proba
from skfp.model_selection import scaffold_train_test_split
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint
from skfp.preprocessing import MolFromAminoseqTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, make_union

from utils import load_dataframe, get_feature

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    confusion_matrix,
)

train_df, test_df = load_dataframe()

mol_transformer = MolFromAminoseqTransformer()

X = mol_transformer.transform(train_df['sequence'])
y = train_df['label'].values

X_train, X_test, y_train, y_test = scaffold_train_test_split(data=X, additional_data=y, test_size=0.2)

from lightgbm import LGBMClassifier

pipeline = make_pipeline(
    make_union(ECFPFingerprint(count=True), MACCSFingerprint()),
    LGBMClassifier(),
)
pipeline.fit(X_train, y_train)

y_pred_proba = pipeline.predict_proba(X_test)
y_pred_proba_pos = extract_pos_proba(y_pred_proba)

y_pred = pipeline.predict(X_test)

# Метрики
auroc = multioutput_auroc_score(y_test, y_pred_proba_pos)  # если у вас кастомный multioutput_auroc_score
roc_auc = roc_auc_score(y_test, y_pred_proba_pos)
pr_auc = average_precision_score(y_test, y_pred_proba_pos)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Вывод
print(f"Multioutput AUROC: {auroc:.2%}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"F1: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"ACC: {acc:.4f}")
print("Confusion Matrix:")
print(cm)

'''
RandomForestClassifier - Anticancer
Multioutput AUROC: 77.33%
ROC-AUC: 0.7733
PR-AUC: 0.7754
F1: 0.7425
MCC: 0.3760
ACC: 0.6884
Confusion Matrix:
[[ 66  62]
 [ 24 124]]

RandomForestClassifier - Anticancer
Multioutput AUROC: 77.61%
ROC-AUC: 0.7761
PR-AUC: 0.7660
F1: 0.7590
MCC: 0.4213
ACC: 0.7101
Confusion Matrix:
[[ 70  58]
 [ 22 126]]

LGBMClassifier - Anticancer
Multioutput AUROC: 78.45%
ROC-AUC: 0.7845
PR-AUC: 0.8110
F1: 0.7312
MCC: 0.3714
ACC: 0.6884
Confusion Matrix:
[[ 73  55]
 [ 31 117]]

LGBMClassifier - Anticancer
Multioutput AUROC: 78.45%
ROC-AUC: 0.7845
PR-AUC: 0.8110
F1: 0.7312
MCC: 0.3714
ACC: 0.6884
Confusion Matrix:
[[ 73  55]
 [ 31 117]]

LGBMClassifier - Antibacterial
Multioutput AUROC: 97.83%
ROC-AUC: 0.9783
PR-AUC: 0.9829
F1: 0.9399
MCC: 0.8525
ACC: 0.9286
Confusion Matrix:
[[ 976   75]
 [ 113 1470]]
'''