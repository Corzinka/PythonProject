import numpy as np
import pandas as pd

def create_datasets(paths, name_file):
    dataframes = []
    n_classes = 1
    for path in paths:
        dataframe = pd.read_excel(path)
        dataframe.loc[dataframe['label'] == 1, 'label'] = n_classes

        dataframes.append(dataframe)
        n_classes += 1

    # Объединяем все классы в один мультиклассовый датасет
    multi_class_train = pd.concat(dataframes, ignore_index=True)

    duplicates = multi_class_train.duplicated()
    print("Есть ли дубликаты строк: ", duplicates.any())

    # Удаление дубликатов по всем столбцам
    multi_class_train = multi_class_train.drop_duplicates(keep='last')

    # Перемешиваем итоговый датасет (по желанию)
    combined_df = multi_class_train.sample(frac=1).reset_index(drop=True)

    for n_class in range(n_classes):
        print(f'{n_class}: ', combined_df[combined_df['label'] == n_class].shape[0])
    print('all:', combined_df.shape[0])

    combined_df.to_excel(name_file, index=False)
    print(f'save in {name_file}')

# ==========================================================

train_paths = [
    'data/Anticancer_main/Anticancer_main_train.xlsx',
    'data/Antioxidant/train.xlsx',
]

# ==========================================================

test_paths = [
    'data/Anticancer_main/Anticancer_main_test.xlsx',
    'data/Antioxidant/test.xlsx',
]

# ==========================================================

create_datasets(train_paths, 'train.xlsx')
create_datasets(test_paths, 'test.xlsx')
