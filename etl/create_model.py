import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from airflow.utils.log.logging_mixin import LoggingMixin
from joblib import dump
import os

# этап обучения модели логистической регрессии
def fit_model(**kwargs):
    log = LoggingMixin().log
    data_dir = os.path.join(os.path.dirname(__file__), "../data") # путь для сохранения промежуточных данных - папка data на том же уровне вложенности
    results_dir = os.path.join(os.path.dirname(__file__), "../results") # путь для сохранения обученной модели
    file_X_train = 'features_train.csv' # имя файла для X_train
    file_y_train = 'target_train.csv' # имя файла для y_train
    model_file = 'lr_pipeline.joblib' # имя файла для обученной модели
    if (not os.path.exists(data_dir)) or (not os.path.isdir(data_dir)):
        log.error(f'Ошибка: Папка {data_dir} не найдена')
        raise
    path_x = os.path.join(data_dir, file_X_train) # задается адрес промежуточного датасета
    path_y = os.path.join(data_dir, file_y_train) 
    try:
        X_train = pd.read_csv(path_x, header=None)
        y_train = pd.read_csv(path_y, header=None)[0]
        log.info(f"Успешно прочитаны файлы {path_x} и {path_y}")
    except Exception as e:
        log.error(f"Ошибка при чтении файлов {path_x} и {path_y}: {str(e)}")
        raise
    
    # pipeline из 2 этапов: нормализация данных и применение к ним логистической регрессии
    # pipeline как вариант решения выбран из-за того, что по требованиям задания требуется сохранение модели
    # при этом сохранение логистической регрессии без сохранения параметров нормализации не имеет смысла
    # а сохранять их вместе проще всего в качестве pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42)) # используем гиперпараметры по умолчанию
    ])
    pipeline.fit(X_train, y_train)
    # сохранение обученной модели (сохраняем пайплайн: скейлер + логистическую регрессию)
    try:
        output_path = os.path.join(results_dir, model_file)
        dump(pipeline, output_path)
        log.info(f"Обученная модель успешно сохранена по адресу {output_path}")
    except Exception as e:
        log.error(f"Ошибка при сохранении модели: {str(e)}")
        raise
