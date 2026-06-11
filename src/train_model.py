import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model():
    # Load dataset
    df = pd.read_csv("data/sample_customer_data.csv")

    # Features and Target
    X = df[["Age", "SubscriptionMonths", "MonthlyCharges"]]
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Start MLflow Run
    with mlflow.start_run():

        # Model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, predictions)

        print(f"Accuracy: {accuracy}")

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Save model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        print("Model logged successfully to MLflow")


if __name__ == "__main__":
    train_model()