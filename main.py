import numpy as np
import baseline_LightGBM
import baseline_RF
from utils import load_dataframe

train_df, test_df = load_dataframe("data/train_ACE.xlsx", "data/test_ACE.xlsx")

classes = np.unique(train_df["label"])

print("ACE")
print(f'Размер train: {len(train_df)}')
print(f'Размер test: {len(test_df)}')
print(f'Классы: {classes}')

baseline_RF(train_df, test_df)
baseline_LightGBM(train_df, test_df)