import pandas as pd
import numpy as np
from airflow.utils.log.logging_mixin import LoggingMixin
import os

# этап скачивания и проверки данных
def load_data(**kwargs):
    # задается адрес датасета
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    log = LoggingMixin().log
    try:
        df = pd.read_csv(url, header=None)
        log.info(f"Успешно прочитан файл с датасетом {url}")
    except Exception as e:
        log.error(f"Ошибка при скачивании и чтении файла с датасетом: {str(e)}")
        raise
    # проверка размерности
    if df.shape[1] != 32:
        log.error('Несоотвествие размера исходного файла, должно быть 32 столбца, обнаружено ' + str(df.shape[1]))
        raise
    # проверка типа данных
    if (not pd.api.types.is_integer_dtype(df[0])) or (not pd.api.types.is_object_dtype(df[1])) \
            or (not pd.api.types.is_float_dtype(df.iloc[:, 2:].values)):
        log.error('Типы данных в столбцах исходного файла не соответствуют образцу!')
        raise
    # проверка меток таргета
    if not np.array_equal(np.sort(df[1].unique()), np.array(['B', 'M'])):
        log.error('Метки таргета в исходном файле не соответствуют образцу!')
        raise
    log.info('Датасет успешно прошел проверки на количество столбцов, тип данных и метки таргета')

    output_file = 'verified_df.csv' # имя файла для сохранения скачанного датасета
    data_dir = os.path.join(os.path.dirname(__file__), "../data") # путь для сохранения промежуточных данных - папка data на том же уровне вложенности
    try:
        os.makedirs(data_dir, exist_ok=True)  # Создать папку, если её нет
        output_path = os.path.join(data_dir, output_file)
        df.to_csv(output_path, header=None, index=False) # сохраняем локально проверенный датафрейм
        log.info(f"Датасет сохранён в файле: {os.path.abspath(output_path)}")
    except Exception as e:
        log.error(f"Ошибка при сохранении данных в файле {os.path.abspath(output_path)}: {str(e)}")
        raise
