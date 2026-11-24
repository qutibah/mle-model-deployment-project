import pandas as pd
import pytest

from src.features import prepare_features
from src.model import load_model


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_trip(client, valid_trip):
    response = client.post("/predict", json=valid_trip)
    assert response.status_code == 200
    body = response.json()
    assert body["unit"] == "minutes"
    assert body["predicted_duration"] > 0


def test_api_matches_direct_model_call(client, valid_trip, tiny_model_path):
    # no training/serving skew: API == calling the saved model in Python
    expected = load_model(tiny_model_path).predict(prepare_features(pd.DataFrame([valid_trip])))[0]
    predicted = client.post("/predict", json=valid_trip).json()["predicted_duration"]
    assert predicted == pytest.approx(expected, abs=0.01)


def test_missing_field_returns_422(client, valid_trip):
    del valid_trip["trip_distance"]
    response = client.post("/predict", json=valid_trip)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "trip_distance"]


@pytest.mark.parametrize("field, value", [
    ("PULocationID", "abc"),
    ("PULocationID", 0),
    ("DOLocationID", 266),
    ("trip_distance", -1),
    ("trip_distance", 0),
    ("trip_distance", 150),
    ("tpep_pickup_datetime", "not a date"),
])
def test_invalid_values_return_422(client, valid_trip, field, value):
    valid_trip[field] = value
    assert client.post("/predict", json=valid_trip).status_code == 422


def test_get_on_predict_is_not_allowed(client):
    assert client.get("/predict").status_code == 405
