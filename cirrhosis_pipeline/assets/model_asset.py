from dagster import asset
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import mlflow

def measure_model_performance(X, y, k_neighbors_values, number_of_trials):
    lahat_training = pd.DataFrame()
    lahat_test = pd.DataFrame()
    
    for seedN in range(1, 20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seedN)
        
        training_accuracy = []
        test_accuracy = []
        
        for n_neighbors in k_neighbors_values:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', metric='cosine')
            clf.fit(X_train, y_train)
            training_accuracy.append(clf.score(X_train, y_train))
            test_accuracy.append(clf.score(X_test, y_test))
            
        lahat_training[seedN] = training_accuracy
        lahat_test[seedN] = test_accuracy
        
    return lahat_training, lahat_test

@asset
def find_optimal_k(feature_engineering):
    X = feature_engineering.drop("Status", axis=1)
    y = feature_engineering["Status"]
    
    lahat_training, lahat_test = measure_model_performance(
        X, y, k_neighbors_values=range(1, 50), number_of_trials=20
    )
    
    top_10_k = list(lahat_test.mean(axis=1).sort_values(ascending=False).head(10).index)
    _, preferred_K_nearest_test = measure_model_performance(
        X, y, k_neighbors_values=top_10_k, number_of_trials=20
    )
    
    top_index = preferred_K_nearest_test.mean(axis=1).sort_values(ascending=False).head(1).index.values[0]
    best_k = top_10_k[top_index]
    
    return {"best_k": best_k, "X": X, "y": y}

@asset
def train_model(find_optimal_k):
    best_k = find_optimal_k["best_k"]
    X = find_optimal_k["X"]
    y = find_optimal_k["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    with mlflow.start_run():
        model = KNeighborsClassifier(n_neighbors=best_k, weights='uniform', metric='cosine')
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        mlflow.log_param("best_k", best_k)
        mlflow.sklearn.log_model(model, "knn_model")
    
    return model