from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}
