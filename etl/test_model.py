import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from airflow.utils.log.logging_mixin import LoggingMixin
from joblib import load
import os
import json

# этап валидации модели логистической регрессии
def test_model(**kwargs):
    log = LoggingMixin().log
    data_dir = os.path.join(os.path.dirname(__file__), "../data") # путь для сохранения промежуточных данных - папка data на том же уровне вложенности
    results_dir = os.path.join(os.path.dirname(__file__), "../results") # путь для сохранения обученной модели и метрик
    file_X_test = 'features_test.csv' # имя файла для X_test
    file_y_test = 'target_test.csv' # имя файла для y_test
    file_model = 'lr_pipeline.joblib' # имя файла с обученной моделью
    file_params = 'model_params.json' # имя файла с параметрами обученной модели
    file_metrics = 'model_metrics.json' # имя файла с метриками обученной модели на тестовой выборке
    if (not os.path.exists(data_dir)) or (not os.path.isdir(data_dir)):
        log.error(f'Ошибка: Папка {data_dir} не найдена')
        raise
    if (not os.path.exists(results_dir)) or (not os.path.isdir(results_dir)):
        log.error(f'Ошибка: Папка {results_dir} не найдена')
        raise
    path_x = os.path.join(data_dir, file_X_test) # задается адрес тестового датасета
    path_y = os.path.join(data_dir, file_y_test) 
    try:
        X_test = pd.read_csv(path_x, header=None)
        y_test = pd.read_csv(path_y, header=None)[0]
        log.info(f"Успешно прочитаны файлы {path_x} и {path_y}")
    except Exception as e:
        log.error(f"Ошибка при чтении файлов {path_x} и {path_y}: {str(e)}")
        raise
    path_model = os.path.join(results_dir, file_model) 
    try:
        pipeline = load(path_model)
        log.info(f"Успешно загружена модель из файла {path_model}")
    except Exception as e:
        log.error(f"Ошибка при загрузке модели из файла {path_model}: {str(e)}")
        raise

    # прогоняем тестовые данные через pipeline
    y_pred = pipeline.predict(X_test)
    # метрики модели на тестовой выборке
    lr_accuracy = accuracy_score(y_test, y_pred)
    lr_precision = precision_score(y_test, y_pred)
    lr_recall = recall_score(y_test, y_pred)
    lr_f1 = f1_score(y_test, y_pred)
    # объединяем метрики в словарь для последующего сохранения в файл
    metrics = {
        "accuracy": lr_accuracy,
        "precision": lr_precision,
        "recall": lr_recall,
        "f1": lr_f1,
    }
    # собираем параметры обученного пайплайна для последующего сохранения в файл
    # Получаем компоненты пайплайна
    s_scaler = pipeline.named_steps['scaler']
    logreg = pipeline.named_steps['model']
    # Собираем параметры в словарь
    params = {
        "StandardScaler": {
            "mean": s_scaler.mean_.tolist(),  # преобразуем numpy array в list
            "scale": s_scaler.scale_.tolist(),
        },
        "LogisticRegression": {
            "coef": logreg.coef_.tolist(),
            "intercept": logreg.intercept_.tolist(),
            "classes": logreg.classes_.tolist(),
        }
    }

    # сохраняем параметры и метрики модели на локальном диске
    try:
        output_path = os.path.join(results_dir, file_params)
        with open(output_path, "w") as f:
            json.dump(params, f, indent=4)
        log.info(f"Параметры обученной модели успешно сохранены по адресу {output_path}")
    except Exception as e:
        log.error(f"Ошибка при сохранении параметров модели: {str(e)}")
        raise
    try:
        output_path = os.path.join(results_dir, file_metrics)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        log.info(f"Метрики обученной модели успешно сохранены по адресу {output_path}")
    except Exception as e:
        log.error(f"Ошибка при сохранении метрик модели: {str(e)}")
        raise
