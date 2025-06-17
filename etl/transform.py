import pandas as pd
from sklearn.model_selection import train_test_split
from airflow.utils.log.logging_mixin import LoggingMixin
import os

# этап предобработки данных
def transform_data(**kwargs):
    log = LoggingMixin().log
    data_dir = os.path.join(os.path.dirname(__file__), "../data") # путь для сохранения промежуточных данных - папка data на том же уровне вложенности
    input_file = 'verified_df.csv' # имя файла с датасетом, сохраненным с прошлого этапа
    output_X_train = 'features_train.csv' # имя файла для сохранения X_train
    output_X_test = 'features_test.csv' # имя файла для сохранения X_test
    output_y_train = 'target_train.csv' # имя файла для сохранения y_train
    output_y_test = 'target_test.csv' # имя файла для сохранения y_test
    if (not os.path.exists(data_dir)) or (not os.path.isdir(data_dir)):
        log.error(f'Ошибка: Папка {data_dir} не найдена')
        raise
    df_path = os.path.join(data_dir, input_file) # задается адрес промежуточного датасета
    try:
        df = pd.read_csv(df_path, header=None)
        log.info(f"Успешно прочитан файл с датасетом {df_path}")
    except Exception as e:
        log.error(f"Ошибка при чтении файла с датасетом {df_path}: {str(e)}")
        raise
    # Блок предобработки данных:
    # удаление пропусков
    df = df.dropna()
    # удаление дубликатов
    df = df.drop_duplicates()
    # трансформация таргета
    # наш положительный класс - это 'M', т.е. злокачественная опухоль, кодируем как 1
    # 'B' - это доброкачественная опухоль, кодируем как 0
    df[1] = df[1].apply(lambda x: 0 if x=='B' else (1 if x=='M' else x))
    y = df[1] # сохраняем таргет в отдельном массиве
    # разделяем датасет на обучающую и тестовую выборки
    # из датасета признаков исключаем столбец с ID и столбец с таргетом
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[0, 1]), y,
                                                        test_size=0.2, stratify=y, random_state=42)

    # Сохраняем результат для следующих тасок Airflow
    try:
        # сохраняем 2 датафрейма и 2 series в csv
        output_path = os.path.join(data_dir, output_X_train)
        X_train.to_csv(output_path, header=None, index=False)
        output_path = os.path.join(data_dir, output_X_test)
        X_test.to_csv(output_path, header=None, index=False)
        output_path = os.path.join(data_dir, output_y_train)
        y_train.to_csv(output_path, header=False, index=False)
        output_path = os.path.join(data_dir, output_y_test)
        y_test.to_csv(output_path, header=False, index=False)
        log.info(f"Успешно сохранены данные после предобработки в папку {data_dir} - " +
                 f"файлы {output_X_train}, {output_X_test}, {output_y_train} и {output_y_test}")
    except Exception as e:
        log.error(f"Ошибка при сохранении данных после предобработки: {str(e)}")
        raise
