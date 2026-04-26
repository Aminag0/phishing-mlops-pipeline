from fastapi import FastAPI, Response
from pydantic import BaseModel
import joblib
import pandas as pd
import time

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(
    title="Phishing Website Detection API",
    version="1.0.0"
)

model = joblib.load("models/final_xgboost_model.pkl")

FEATURE_COLUMNS = [
    'having_IP_Address',
    'URL_Length',
    'Shortining_Service',
    'having_At_Symbol',
    'double_slash_redirecting',
    'Prefix_Suffix',
    'having_Sub_Domain',
    'SSLfinal_State',
    'Domain_registeration_length',
    'Favicon',
    'port',
    'HTTPS_token',
    'Request_URL',
    'URL_of_Anchor',
    'Links_in_tags',
    'SFH',
    'Submitting_to_email',
    'Abnormal_URL',
    'Redirect',
    'on_mouseover',
    'RightClick',
    'popUpWidnow',
    'Iframe',
    'age_of_domain',
    'DNSRecord',
    'web_traffic',
    'Page_Rank',
    'Google_Index',
    'Links_pointing_to_page',
    'Statistical_report'
]

REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests",
    ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "prediction_count",
    "Total number of predictions",
    ["prediction"]
)

ERROR_COUNT = Counter(
    "api_error_count",
    "Total number of API errors",
    ["endpoint"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds"
)


class PredictionInput(BaseModel):
    features: list


@app.get("/")
def root():
    REQUEST_COUNT.labels(endpoint="/").inc()
    return {"message": "Phishing Detection API Running"}


@app.get("/health")
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: PredictionInput):
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    start_time = time.time()

    try:
        if len(data.features) != 30:
            ERROR_COUNT.labels(endpoint="/predict").inc()
            return {"error": "Exactly 30 features required"}

        df = pd.DataFrame([data.features], columns=FEATURE_COLUMNS)

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        label = "Phishing" if pred == 1 else "Legitimate"

        PREDICTION_COUNT.labels(prediction=label).inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        return {
            "prediction": label,
            "confidence": round(float(prob), 4)
        }

    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict").inc()
        return {"error": str(e)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)