import numpy as np
from baseline_LightGBM import baseline_LightGBM_func
from baseline_RF import baseline_RF_func
from fusinModel import fusionModel
from other.ConservativeSubstitutions import augmented_data
from pretrain.pretrain import get_feature
from utils import load_dataframe

train_df, test_df = load_dataframe(train_path="data/train_ABE.xlsx", test_path="data/test_ABE.xlsx")
print("ABE")
train_df = augmented_data(train_df)
print(f'Размер train: {len(train_df)}')
print(f'Размер test: {len(test_df)}')
print(f'Классы: {np.unique(train_df["label"])}')
baseline_RF_func(train_df, test_df)
baseline_LightGBM_func(train_df, test_df)
fusionModel(train_df, test_df)

train_df, test_df = load_dataframe(train_path="data/train_ACE.xlsx", test_path="data/test_ACE.xlsx")
print("ACE")
train_df = augmented_data(train_df)
print(f'Размер train: {len(train_df)}')
print(f'Размер test: {len(test_df)}')
print(f'Классы: {np.unique(train_df["label"])}')
baseline_RF_func(train_df, test_df)
baseline_LightGBM_func(train_df, test_df)
fusionModel(train_df, test_df)

train_df, test_df = load_dataframe(train_path="data/train_AOE.xlsx", test_path="data/test_AOE.xlsx")
print("AOE")
train_df = augmented_data(train_df)
print(f'Размер train: {len(train_df)}')
print(f'Размер test: {len(test_df)}')
print(f'Классы: {np.unique(train_df["label"])}')
baseline_RF_func(train_df, test_df)
baseline_LightGBM_func(train_df, test_df)
fusionModel(train_df, test_df)
