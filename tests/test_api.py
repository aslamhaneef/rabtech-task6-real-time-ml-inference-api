from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


VALID_PAYLOAD = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability_negative" in data
    assert "probability_positive" in data

    assert data["prediction"] in [0, 1]

    assert 0.0 <= data["probability_negative"] <= 1.0
    assert 0.0 <= data["probability_positive"] <= 1.0

    assert abs(
        data["probability_negative"]
        + data["probability_positive"]
        - 1.0
    ) < 1e-6


def test_predict_validation_error():
    invalid_payload = VALID_PAYLOAD.copy()
    del invalid_payload["age"]

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422