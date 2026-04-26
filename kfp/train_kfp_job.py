import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


DATA_PATH = Path("data/processed/uci_phishing_processed.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Starting Kubernetes ML training job...")

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

    joblib.dump(model, OUTPUT_DIR / "kfp_xgboost_model.pkl")

    metrics_text = f"""
Kubernetes Training Job Completed
Model: Tuned XGBoost
Dataset: UCI Phishing Websites Dataset
Accuracy: {accuracy * 100:.2f}%
Precision: {precision * 100:.2f}%
Recall: {recall * 100:.2f}%
F1 Score: {f1 * 100:.2f}%
ROC-AUC: {roc_auc * 100:.2f}%
"""

    with open(OUTPUT_DIR / "kfp_metrics.txt", "w") as f:
        f.write(metrics_text)

    print(metrics_text)
    print("Saved model artifact: outputs/kfp_xgboost_model.pkl")
    print("Saved metrics artifact: outputs/kfp_metrics.txt")


if __name__ == "__main__":
    main()