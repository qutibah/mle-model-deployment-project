import numpy as np
import pandas as pd
import pytest

from src.features import prepare_features
from src.model import DEFAULT_MODEL_PATH, load_model, save_model
from src.train import build_pipeline
from tests.conftest import make_trips


def test_save_and_load_gives_same_predictions(tmp_path):
    trips = make_trips()
    X = prepare_features(trips)
    pipeline = build_pipeline(n_estimators=5, max_depth=5).fit(X, trips.duration)

    loaded = load_model(save_model(pipeline, tmp_path / "m.skops"))
    np.testing.assert_allclose(loaded.predict(X), pipeline.predict(X))


def test_model_learns_that_longer_trips_take_longer(tiny_model_path):
    model = load_model(tiny_model_path)
    trips = make_trips(2).assign(trip_distance=[1.0, 15.0])
    short, long = model.predict(prepare_features(trips))
    assert long > short


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_model(tmp_path / "nope.skops")


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="real model not trained yet")
def test_real_model_gives_plausible_prediction():
    trip = pd.DataFrame([{"PULocationID": 161, "DOLocationID": 236, "trip_distance": 2.5,
                          "tpep_pickup_datetime": "2025-01-15T08:30:00"}])
    (minutes,) = load_model().predict(prepare_features(trip))
    assert 5 < minutes < 40
