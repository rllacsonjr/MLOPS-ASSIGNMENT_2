from dagster import asset
import pandas as pd
import numpy as np

@asset
def load_raw_data():
    df = pd.read_csv('../data/cirrhosis.csv')
    df = df.drop("ID", axis=1)
    df = df[df['Status'] != 'CL']
    return df

@asset
def preprocess_data(load_raw_data):
    df = load_raw_data.copy()
    df = df.dropna()
    df["Age"] = df["Age"].apply(lambda x: x / 365)
    return df

@asset
def feature_engineering(preprocess_data):
    df = preprocess_data.copy()
    df_encoded = pd.get_dummies(
        df, 
        columns=['Drug', 'Edema', 'Hepatomegaly', 'Ascites', 'Spiders', 'Sex']
    )
    return df_encoded