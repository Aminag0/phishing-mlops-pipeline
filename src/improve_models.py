import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


DATA_PATH = Path("data/processed/uci_phishing_processed.csv")
MODEL_DIR = Path("models")
METRICS_DIR = Path("reports/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(name, model, X_train, X_test, y_train, y_test, X, y):
    print(f"\nEvaluating: {name}")

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    results = {
        "Model": name,
        "Training Accuracy": round(accuracy_score(y_train, train_pred) * 100, 2),
        "Testing Accuracy": round(accuracy_score(y_test, test_pred) * 100, 2),
        "K-Fold Accuracy": round(cv_scores.mean() * 100, 2),
        "Precision": round(precision_score(y_test, test_pred) * 100, 2),
        "Recall": round(recall_score(y_test, test_pred) * 100, 2),
        "F1 Score": round(f1_score(y_test, test_pred) * 100, 2),
        "ROC-AUC": round(roc_auc * 100, 2) if roc_auc is not None else None
    }

    print(results)
    return results, model


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

    results = []

    # 1. Tuned Random Forest
    print("\nTuning Random Forest...")

    rf = RandomForestClassifier(random_state=42)

    rf_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }

    rf_search = RandomizedSearchCV(
        rf,
        rf_params,
        n_iter=20,
        scoring="accuracy",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rf_search.fit(X_train, y_train)

    print("Best RF Params:", rf_search.best_params_)

    rf_result, rf_best = evaluate_model(
        "Tuned Random Forest",
        rf_search.best_estimator_,
        X_train,
        X_test,
        y_train,
        y_test,
        X,
        y
    )
    rf_result["Best Parameters"] = str(rf_search.best_params_)
    results.append(rf_result)

    # 2. Tuned XGBoost
    print("\nTuning XGBoost...")

    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

    xgb_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    xgb_search = RandomizedSearchCV(
        xgb,
        xgb_params,
        n_iter=20,
        scoring="accuracy",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    xgb_search.fit(X_train, y_train)

    print("Best XGBoost Params:", xgb_search.best_params_)

    xgb_result, xgb_best = evaluate_model(
        "Tuned XGBoost",
        xgb_search.best_estimator_,
        X_train,
        X_test,
        y_train,
        y_test,
        X,
        y
    )
    xgb_result["Best Parameters"] = str(xgb_search.best_params_)
    results.append(xgb_result)

    # 3. LightGBM
    lgbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42
    )

    lgbm_result, lgbm_best = evaluate_model(
        "LightGBM",
        lgbm_model,
        X_train,
        X_test,
        y_train,
        y_test,
        X,
        y
    )
    lgbm_result["Best Parameters"] = "Manual default strong configuration"
    results.append(lgbm_result)

    # 4. CatBoost
    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )

    cat_result, cat_best = evaluate_model(
        "CatBoost",
        cat_model,
        X_train,
        X_test,
        y_train,
        y_test,
        X,
        y
    )
    cat_result["Best Parameters"] = "Manual default strong configuration"
    results.append(cat_result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_DIR / "improved_model_metrics.csv", index=False)

    # Select best model by testing accuracy, then F1
    best_row = results_df.sort_values(
        by=["Testing Accuracy", "F1 Score"],
        ascending=False
    ).iloc[0]

    best_model_name = best_row["Model"]

    model_map = {
        "Tuned Random Forest": rf_best,
        "Tuned XGBoost": xgb_best,
        "LightGBM": lgbm_best,
        "CatBoost": cat_best
    }

    best_model = model_map[best_model_name]

    joblib.dump(best_model, MODEL_DIR / "best_improved_model.pkl")

    print("\nImprovement phase completed.")
    print(results_df)
    print(f"\nBest Model: {best_model_name}")
    print("Saved best model to: models/best_improved_model.pkl")
    print("Saved metrics to: reports/metrics/improved_model_metrics.csv")


if __name__ == "__main__":
    main()