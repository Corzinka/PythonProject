import pandas as pd
import numpy as np

def create_datasets(path, name_file, max_size=None):
    dataframe = pd.read_excel(path)

    # Если max_size не задан, сохраняем весь датасет перемешанный
    if max_size is not None and dataframe.shape[0] > max_size:
        # Определяем количество классов и сколько строк брать с каждого класса
        classes = dataframe['label'].unique()
        n_classes = len(classes)
        per_class_limit = max_size // n_classes  # Целочисленное деление

        # Сохраняем по per_class_limit строк для каждого класса
        balanced_list = []
        for cls in classes:
            cls_data = dataframe[dataframe['label'] == cls]
            if cls_data.shape[0] > per_class_limit:
                cls_sample = cls_data.sample(n=per_class_limit, random_state=42)
            else:
                cls_sample = cls_data
            balanced_list.append(cls_sample)
        
        dataframe = pd.concat(balanced_list, ignore_index=True)
        dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)  # Перемешиваем

    # Вывод статистики
    print(path)
    for n_class in dataframe['label'].unique():
        print(f'Класс {n_class}: {dataframe[dataframe["label"] == n_class].shape[0]}')
    print('Всего строк:', dataframe.shape[0])

    dataframe.to_excel(f"data/{name_file}", index=False)
    print(f'Сохранено в {name_file}')

# ==========================================================

create_datasets('data/Antibacterial/Antibacteria_train.xlsx', 'train_ABE.xlsx', max_size=1200)
create_datasets('data/Antibacterial/Antibacteria_test.xlsx', 'test_ABE.xlsx')

# ==========================================================

create_datasets('data/Anticancer_main/Anticancer_main_train.xlsx', 'train_ACE.xlsx', max_size=2000)
create_datasets('data/Anticancer_main/Anticancer_main_test.xlsx', 'test_ACE.xlsx')

# ==========================================================

create_datasets('data/Antioxidant/train.xlsx', 'train_AOE.xlsx', max_size=2000)
create_datasets('data/Antioxidant/test.xlsx', 'test_AOE.xlsx')

'''
data/Antibacterial/Antibacteria_train.xlsx
Класс 0: 6583
Класс 1: 6583
Всего строк: 13166
Сохранено в train_ABE.xlsx
data/Antibacterial/Antibacteria_test.xlsx
Класс 0: 1695
Класс 1: 1695
Всего строк: 3390
Сохранено в test_ABE.xlsx
data/Anticancer_main/Anticancer_main_train.xlsx
Класс 1: 689
Класс 0: 689
Всего строк: 1378
Сохранено в train_ACE.xlsx
data/Anticancer_main/Anticancer_main_test.xlsx
Класс 0: 172
Класс 1: 172
Всего строк: 344
Сохранено в test_ACE.xlsx
data/Antioxidant/train.xlsx
Класс 0: 541
Класс 1: 582
Всего строк: 1123
Сохранено в train_AOE.xlsx
data/Antioxidant/test.xlsx
Класс 1: 146
Класс 0: 135
Всего строк: 281
Сохранено в test_AOE.xlsx
'''