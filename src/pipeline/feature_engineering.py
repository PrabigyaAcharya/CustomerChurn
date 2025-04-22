import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def transform_features(df: pd.DataFrame):
    df = df.copy()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_numpy = X.to_numpy()
    y_numpy = y.to_numpy()

    X_np_array = np.array(X_numpy, dtype=np.float32)
    y_np_array = np.array(y_numpy, dtype=np.float32)

    X = torch.from_numpy(X_np_array).to(torch.float32)
    y = torch.from_numpy(y_np_array).to(torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    oversample = SMOTE(k_neighbors=5)
    X_smote, y_smote = oversample.fit_resample(X_train, y_train)
    X_train, y_train = X_smote, y_smote
 
    X_train = torch.from_numpy(X_train).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.float32)


    return X_train, y_train, X_test, y_test




