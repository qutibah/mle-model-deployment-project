"""Shared fixtures: a tiny model trained on fake trips, so tests run in seconds
and don't need the real 54 MB model or the parquet file."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features import prepare_features
from src.model import save_model
from src.train import build_pipeline


def make_trips(n: int = 200) -> pd.DataFrame:
    """Fake raw trips where duration grows with distance."""
    distance = [0.5 + (i % 20) for i in range(n)]
    return pd.DataFrame({
        "PULocationID": [100 + i % 5 for i in range(n)],
        "DOLocationID": [200 + i % 7 for i in range(n)],
        "trip_distance": distance,
        "tpep_pickup_datetime": pd.date_range("2025-01-01", periods=n, freq="37min"),
        "duration": [3 + 2.5 * d for d in distance],
    })


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    trips = make_trips()
    pipeline = build_pipeline(n_estimators=5, max_depth=5).fit(prepare_features(trips), trips.duration)
    return save_model(pipeline, tmp_path_factory.mktemp("model") / "model.skops")


@pytest.fixture
def client(tiny_model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_model_path))
    from app.main import app

    with TestClient(app) as c:  # "with" runs the lifespan -> model gets loaded
        yield c


@pytest.fixture
def valid_trip():
    return {
        "PULocationID": 101,
        "DOLocationID": 203,
        "trip_distance": 4.5,
        "tpep_pickup_datetime": "2025-01-15T08:30:00",
    }
