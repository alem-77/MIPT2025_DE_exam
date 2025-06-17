from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os
from datetime import timedelta

# добавление пути для поиска скриптов функций в папке etl
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from etl.load_and_check import load_data # кастомная функция для загрузки и проверки датасета
from etl.transform import transform_data # кастомная функция для предобработки данных
from etl.create_model import fit_model # кастомная функция для обучения модели
from etl.test_model import test_model # кастомная функция для подсчета метрик модели на тестовой выборке и сохранения результатов на диске

# параметры ретраев по умолчанию для всех этапов
default_args = {
    'retries': 3,  # Количество попыток
    'retry_delay': timedelta(minutes=1)  # Задержка между попытками (1 минута)
}

with DAG(
    "homework_Martynov_AA",
    default_args=default_args,
    schedule_interval=None # поскольку обучение осуществляется не по графику, предусматриваю только ручной запуск
) as dag:
    load_task = PythonOperator(
        task_id='load_and_check_dataset',
        python_callable=load_data,
        retry_delay=timedelta(minutes=2)  # Здесь увеличенный параметр задержки между попытками, т.к. обращаемся по внешнему адресу
    )
    transform_task = PythonOperator(
        task_id='transform_dataset',
        python_callable=transform_data
    )
    create_model_task = PythonOperator(
        task_id='create_model',
        python_callable=fit_model
    )
    test_model_task = PythonOperator(
        task_id='calculate_metrics',
        python_callable=test_model
    )

    load_task >> transform_task >> create_model_task >> test_model_task
