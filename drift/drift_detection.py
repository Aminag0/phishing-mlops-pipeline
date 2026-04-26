import json
import joblib
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

Path("drift").mkdir(exist_ok=True)

REFERENCE_DATA = "data/processed/uci_phishing_processed.csv"
DRIFTED_DATA = "drift/drifted_phishing_data.csv"
MODEL_PATH = "models/final_xgboost_model.pkl"

reference_df = pd.read_csv(REFERENCE_DATA)
drifted_df = pd.read_csv(DRIFTED_DATA)

target_col = "Result"

X_ref = reference_df.drop(columns=[target_col])
y_ref = reference_df[target_col].replace(-1, 0)

X_drift = drifted_df.drop(columns=[target_col])
y_drift = drifted_df[target_col].replace(-1, 0)

model = joblib.load(MODEL_PATH)

ref_preds = model.predict(X_ref)
drift_preds = model.predict(X_drift)

reference_accuracy = accuracy_score(y_ref, ref_preds)
drifted_accuracy = accuracy_score(y_drift, drift_preds)

reference_precision = precision_score(y_ref, ref_preds, average="weighted", zero_division=0)
drifted_precision = precision_score(y_drift, drift_preds, average="weighted", zero_division=0)

reference_recall = recall_score(y_ref, ref_preds, average="weighted", zero_division=0)
drifted_recall = recall_score(y_drift, drift_preds, average="weighted", zero_division=0)

reference_f1 = f1_score(y_ref, ref_preds, average="weighted", zero_division=0)
drifted_f1 = f1_score(y_drift, drift_preds, average="weighted", zero_division=0)

accuracy_drop = reference_accuracy - drifted_accuracy
f1_drop = reference_f1 - drifted_f1

drift_results = []

for col in X_ref.columns:
    stat, p_value = ks_2samp(X_ref[col], X_drift[col])
    drift_detected = p_value < 0.05

    drift_results.append({
        "feature": col,
        "reference_mean": round(float(X_ref[col].mean()), 4),
        "drifted_mean": round(float(X_drift[col].mean()), 4),
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": bool(drift_detected)
    })

drift_summary = pd.DataFrame(drift_results)

drifted_features_count = int(drift_summary["drift_detected"].sum())
total_features = len(drift_summary)
drift_percentage = drifted_features_count / total_features

drift_threshold = 0.25
accuracy_drop_threshold = 0.02

retraining_required = (
    drift_percentage >= drift_threshold or accuracy_drop >= accuracy_drop_threshold
)

report = {
    "model_used": MODEL_PATH,
    "reference_dataset": REFERENCE_DATA,
    "drifted_dataset": DRIFTED_DATA,
    "reference_accuracy": round(reference_accuracy, 4),
    "drifted_accuracy": round(drifted_accuracy, 4),
    "accuracy_drop": round(accuracy_drop, 4),
    "reference_precision": round(reference_precision, 4),
    "drifted_precision": round(drifted_precision, 4),
    "reference_recall": round(reference_recall, 4),
    "drifted_recall": round(drifted_recall, 4),
    "reference_f1": round(reference_f1, 4),
    "drifted_f1": round(drifted_f1, 4),
    "f1_drop": round(f1_drop, 4),
    "total_features": total_features,
    "drifted_features_count": drifted_features_count,
    "drift_percentage": round(drift_percentage, 4),
    "drift_threshold": drift_threshold,
    "accuracy_drop_threshold": accuracy_drop_threshold,
    "retraining_required": retraining_required
}

drift_summary.to_csv("drift/drift_summary.csv", index=False)

with open("drift/drift_report.json", "w") as f:
    json.dump(report, f, indent=4)

with open("drift/retraining_trigger.txt", "w") as f:
    if retraining_required:
        f.write(
            "RETRAINING TRIGGERED: Drift percentage or model performance degradation exceeded the configured threshold."
        )
    else:
        f.write(
            "NO RETRAINING REQUIRED: Drift and performance degradation are within acceptable limits."
        )

print("\n========== PHASE 10 DRIFT DETECTION REPORT ==========")
print(f"Model Used: {MODEL_PATH}")
print(f"Reference Dataset: {REFERENCE_DATA}")
print(f"Drifted Dataset: {DRIFTED_DATA}")
print("----------------------------------------------------")
print(f"Reference Accuracy: {reference_accuracy:.4f}")
print(f"Drifted Accuracy: {drifted_accuracy:.4f}")
print(f"Accuracy Drop: {accuracy_drop:.4f}")
print(f"Reference Precision: {reference_precision:.4f}")
print(f"Drifted Precision: {drifted_precision:.4f}")
print(f"Reference Recall: {reference_recall:.4f}")
print(f"Drifted Recall: {drifted_recall:.4f}")
print(f"Reference F1 Score: {reference_f1:.4f}")
print(f"Drifted F1 Score: {drifted_f1:.4f}")
print(f"F1 Drop: {f1_drop:.4f}")
print("----------------------------------------------------")
print(f"Drifted Features: {drifted_features_count}/{total_features}")
print(f"Drift Percentage: {drift_percentage:.2%}")
print(f"Drift Threshold: {drift_threshold:.2%}")
print(f"Accuracy Drop Threshold: {accuracy_drop_threshold:.2%}")
print(f"Retraining Required: {retraining_required}")
print("====================================================")