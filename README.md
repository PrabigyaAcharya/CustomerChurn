
# Customer Churn Prediction

A complete machine learning pipelline for predicting customer churn, built using

* __MLflow__ for experiment tracking and model registry
* __Airflow__ for automatino of data ingestion and traning pipeline
* __FastAPI__ for real-time model inference via /predict endpoint
* __PyTorch__ as the core deep learning framework
* __Supabase__ for data storage and ingestino




## Setup

### 1. Clone the repository

```bash
git clone https://github.com/PrabigyaAcharya/CustomerChurn.git

cd CustomerChurn
```

### 2. Create a virtual environment (Python 3.10.12)

```bash
python -m venv churn_env
source churn_env/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Start MLFlow UI
```bash
mlflow ui
```

### 4. Setup Apache Airflow 
```bash
mkdir airflow
export  AIRFLOW_HOME=$(pwd)/airflow
airflow db init
```

For creating users and setting up the server, follow [this tutorial](https://www.linkedin.com/pulse/install-apache-airflow-mac-os-ranga-reddy/)

### 6. Start Airflow Server
```bash
airflow standalone
```

### 7. Deploy
```bash
uvicorn src.serving.app:app --reload
```



## Dependencies
- Python 3.10+
- Supabase
- Apache Airflow 2
- MLFlow
- FastAPI
- PyTorch