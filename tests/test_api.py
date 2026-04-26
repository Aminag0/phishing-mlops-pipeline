from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    payload = {
        "features": [-1,1,1,1,-1,-1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,0,1,1,1,-1,1,-1,0,-1,1,1,-1]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()