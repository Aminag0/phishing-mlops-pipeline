from kfp import dsl
from kfp import compiler


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "xgboost", "joblib"]
)
def train_phishing_model():
    import pandas as pd
    import joblib
    from pathlib import Path

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from xgboost import XGBClassifier

    print("Starting Kubeflow pipeline component: train_phishing_model")

    data_path = Path("data/processed/uci_phishing_processed.csv")

    # In real Kubeflow deployment, dataset would come from PVC/object storage.
    # This component represents the pipeline training stage.
    print("Dataset expected path:", data_path)

    print("Pipeline stages:")
    print("1. Load UCI phishing dataset")
    print("2. Split train/test")
    print("3. Train tuned XGBoost model")
    print("4. Evaluate accuracy, precision, recall, F1, ROC-AUC")
    print("5. Save model artifact")


@dsl.component(
    base_image="python:3.11-slim"
)
def evaluate_model():
    print("Starting Kubeflow pipeline component: evaluate_model")
    print("Evaluation metrics: accuracy, precision, recall, F1-score, ROC-AUC")


@dsl.pipeline(
    name="phishing-mlops-kubeflow-pipeline",
    description="Kubeflow-compatible pipeline for phishing website detection MLOps workflow"
)
def phishing_mlops_pipeline():
    train_task = train_phishing_model()
    eval_task = evaluate_model()
    eval_task.after(train_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=phishing_mlops_pipeline,
        package_path="kfp/phishing_mlops_pipeline.yaml"
    )

    print("Kubeflow pipeline YAML compiled successfully.")
    print("Saved to: kfp/phishing_mlops_pipeline.yaml")