# cirrhosis_pipeline/assets/data_asset.py
from dagster import asset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

@asset
def load_raw_data():
    """Load the raw cirrhosis dataset."""
    df = pd.read_csv('../data/cirrhosis.csv')
    return df

@asset
def clean_data(load_raw_data):
    """Clean and preprocess the raw data."""
    df = load_raw_data.copy()
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

@asset
def transform_features(clean_data):
    """Transform and engineer features."""
    df = clean_data.copy()
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'Status':  # Don't encode the target variable yet
            df = pd.get_dummies(df, columns=[col], prefix=[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df