# cirrhosis_pipeline/assets/model_asset.py
from dagster import asset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import pandas as pd

@asset
def train_model(transform_features):
    """Train the model using the engineered features."""
    df = transform_features.copy()
    
    # Prepare features and target
    X = df.drop('Status', axis=1)
    y = df['Status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        
        # Log model performance
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        
        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
    
    return model

@asset
def make_predictions(train_model, transform_features):
    """Make predictions using the trained model."""
    model = train_model
    df = transform_features.copy()
    
    X = df.drop('Status', axis=1)
    predictions = model.predict(X)
    
    # Create prediction DataFrame
    prediction_df = pd.DataFrame({
        'Predicted_Status': predictions
    })
    
    return prediction_df