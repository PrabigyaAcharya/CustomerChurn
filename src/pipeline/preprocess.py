import pandas as pd
from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()
non_categorical_variables = ['TotalCharges','MonthlyCharges','SeniorCitizen','tenure']

def clean_data(df: pd.DataFrame):
    df = df.copy()

    df.drop(columns=["customerID"], errors='ignore', inplace = True)

    df["TotalCharges"] = pd.to_numeric(df.TotalCharges, errors="coerce") 

    df.loc[df.tenure == 0, "TotalCharges"] = 0

    df = df.dropna(subset=["TotalCharges"])

    print(f"[INFO] Preprocessed data with {df.isnull().sum().sum()} missing values remaining.")

    #Label encoding the categorical variables
    
    categorical_df = df.drop(non_categorical_variables,axis=1)
   
    df_categorical = categorical_df.apply(label_encoder.fit_transform)

    total_features = pd.merge(df[non_categorical_variables], df_categorical, left_index=True, right_index=True)

    return total_features



    