import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))


from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.pipeline.ingest import load_from_supabase
from src.pipeline.preprocess import clean_data
from src.model.train import train_from_csv


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 1),
    'retries': 1,
}

with DAG('churn_prediction_pipeline',
         default_args=default_args,
         default_view="tree",
         schedule_interval='@daily',
         catchup=False) as dag:

    def ingest_task():
        df = load_from_supabase()
        df.to_csv("data/raw.csv", index=False)

    def preprocess_task():
        import pandas as pd
        from src.pipeline.feature_engineering import transform_features
        df = pd.read_csv("data/raw.csv")
        df_cleaned = clean_data(df)
        X_train, y_train, X_test, y_test = transform_features(df_cleaned)
        pd.concat([X_train, y_train], axis=1).to_csv("data/processed.csv", index=False)

    ingest = PythonOperator(task_id='ingest_data', python_callable=ingest_task)
    preprocess = PythonOperator(task_id='preprocess_data', python_callable=preprocess_task)
    train = PythonOperator(task_id='train_model', python_callable=train_from_csv)

    ingest >> preprocess >> train