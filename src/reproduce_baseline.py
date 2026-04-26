import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


DATA_PATH = Path("data/processed/uci_phishing_processed.csv")
METRICS_PATH = Path("reports/metrics/baseline_metrics.csv")
MODEL_DIR = Path("models")


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Result", axis=1)
    y = df["Result"]

    # Convert labels from -1/1 to 0/1 for XGBoost compatibility
    y = y.replace({-1: 0, 1: 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining: {name}")

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        cv_scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
        kfold_acc = cv_scores.mean()

        precision = precision_score(y_test, test_pred)
        recall = recall_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred)

        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")
        print(f"10-Fold CV Accuracy: {kfold_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, test_pred))

        results.append({
            "Model": name,
            "Training Accuracy": round(train_acc * 100, 2),
            "Testing Accuracy": round(test_acc * 100, 2),
            "K-Fold Accuracy": round(kfold_acc * 100, 2),
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1 Score": round(f1 * 100, 2)
        })

        model_path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_PATH, index=False)

    print("\nBaseline reproduction completed.")
    print(results_df)
    print(f"\nMetrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()