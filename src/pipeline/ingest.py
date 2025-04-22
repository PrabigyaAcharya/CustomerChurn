import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import json

load_dotenv()

with open('src/config.json', 'r') as f:
    config = json.load(f)

def load_from_supabase(table: str = "customer_churn"):
    try:
        db_url = f'postgresql://postgres:{config["supabase"]["password"]}@{config["supabase"]["host"]}:{config["supabase"]["port"]}/postgres'
        engine = create_engine(db_url)
        query = f"SELECT * from {table}"
        # df = pd.read_sql(query, con=engine)
        with engine.connect() as conn:
            df = pd.read_sql(sql=query, con=conn.connection)
            print(f"[INFO] Loaded {df.shape[0]} rows from Supabase")
            return df
    except Exception as e:
        print(f"[ERROR] Supabase ingestion failed: {e}")
        raise
