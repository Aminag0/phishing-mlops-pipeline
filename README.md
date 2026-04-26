\# End-to-End MLOps Pipeline for Phishing Website Detection with Automated CI/CD and Real-Time Monitoring



\## Overview



This project implements a complete production-style MLOps pipeline for phishing website detection using machine learning. The system covers model training, experiment tracking, API deployment, containerization, CI/CD automation, monitoring dashboards, and data drift detection.



The project was developed under the Technical Research (Track-II) category focusing on implementation and measurable improvement.



---



\## Problem Statement



Phishing websites are a major cybersecurity threat used to steal sensitive user information. Traditional ML models may perform well offline but often lack deployment readiness, monitoring, and automated maintenance.



This project solves that by combining phishing detection with a full MLOps lifecycle.



---



\## Dataset



\- UCI Phishing Websites Dataset

\- 11,055 records

\- 30 engineered URL / website features

\- Binary classification target



---



\## Models Evaluated



\- Logistic Regression

\- Decision Tree

\- Random Forest

\- Support Vector Machine

\- XGBoost

\- Tuned XGBoost (Final Model)



---



\## Final Selected Model



\*\*Tuned XGBoost\*\*



\- Accuracy: 97.69%

\- F1 Score: 97.95%

\- ROC-AUC: 99.73%



---



\## Implemented Phases



\### Phase 1 – Baseline Reproduction

Reproduced published phishing detection results using classical ML models.



\### Phase 2 – Model Improvement

Hyperparameter tuning and advanced ensemble models improved final performance.



\### Phase 3 – MLflow

Experiment tracking, metrics logging, artifact management, model registry.



\### Phase 4 – FastAPI Deployment

REST API with endpoints:



\- `/`

\- `/health`

\- `/predict`



\### Phase 5 – Dockerization

Containerized inference service for portable deployment.



\### Phase 6 – Prometheus

Real-time metrics collection:



\- requests count

\- errors

\- latency

\- prediction traffic



\### Phase 7 – Grafana

Live dashboard for API and ML serving observability.



\### Phase 8 – Kubeflow Compatibility

Kubeflow Pipeline YAML generated with KFP SDK.



\### Phase 9 – CI/CD

GitHub Actions pipeline for:



\- dependency install

\- tests

\- build checks



\### Phase 10 – Data Drift Detection

Synthetic drift introduced in 8 features.



Results:



\- Drifted Features: 8 / 30

\- Drift Percentage: 26.67%

\- Accuracy Drop: 98.64% → 79.14%

\- Retraining Trigger: Activated



---



\## Tech Stack



\- Python

\- Scikit-learn

\- XGBoost

\- FastAPI

\- MLflow

\- Docker

\- Prometheus

\- Grafana

\- GitHub Actions

\- Kubeflow Pipelines



---



\## Project Structure



```text

app/            API service

src/            training scripts

models/         saved models

data/           processed dataset

mlruns/         MLflow runs

monitoring/     Prometheus + Grafana config

kfp/            Kubeflow pipeline files

drift/          drift simulation + reports

tests/          automated tests

screenshots/    project evidence


\## Run Locally
Create Environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Start API
uvicorn app.main:app --reload
Swagger UI
http://127.0.0.1:8000/docs
Docker
docker build -t phishing-api .
docker run -p 8000:8000 phishing-api
MLflow
mlflow ui

Then open:

http://127.0.0.1:5000
CI/CD

GitHub Actions workflow automatically runs on push to main branch.

Key Outcomes
End-to-end reproducible ML system
Production deployment readiness
Automated monitoring
Drift awareness
Retraining trigger logic

