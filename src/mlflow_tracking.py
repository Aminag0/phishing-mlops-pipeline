import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


DATA_PATH = Path("data/processed/uci_phishing_processed.csv")
MODEL_DIR = Path("models")
PLOTS_DIR = Path("artifacts/plots")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Result", axis=1)
    y = df["Result"].replace({-1: 0, 1: 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    mlflow.set_experiment("Phishing Website Detection - MLOps")

    with mlflow.start_run(run_name="Tuned_XGBoost_Final_Model"):

        model = XGBClassifier(
            subsample=1.0,
            n_estimators=200,
            max_depth=9,
            learning_rate=0.1,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 9)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("subsample", 1.0)
        mlflow.log_param("colsample_bytree", 0.8)
        mlflow.log_param("dataset", "UCI Phishing Websites Dataset")
        mlflow.log_param("features", X.shape[1])
        mlflow.log_param("test_size", 0.2)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Confusion matrix plot
        cm_plot_path = PLOTS_DIR / "confusion_matrix.png"
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title("Confusion Matrix - Tuned XGBoost")
        plt.savefig(cm_plot_path)
        plt.close()

        # ROC curve plot
        roc_plot_path = PLOTS_DIR / "roc_curve.png"
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("ROC Curve - Tuned XGBoost")
        plt.savefig(roc_plot_path)
        plt.close()

        mlflow.log_artifact(str(cm_plot_path))
        mlflow.log_artifact(str(roc_plot_path))

        # Save model locally
        local_model_path = MODEL_DIR / "final_xgboost_model.pkl"
        joblib.dump(model, local_model_path)
        mlflow.log_artifact(str(local_model_path))

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="Phishing_XGBoost_Model"
        )

        print("MLflow tracking completed.")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")
        print(f"ROC-AUC: {roc_auc * 100:.2f}%")
        print(f"Model saved to: {local_model_path}")


if __name__ == "__main__":
    main()