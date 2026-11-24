import pandas as pd

from src.features import FEATURES, prepare_features


def one_trip(**overrides):
    trip = {
        "PULocationID": 161,
        "DOLocationID": 236,
        "trip_distance": 2.5,
        "tpep_pickup_datetime": pd.Timestamp("2025-01-15 08:30"),  # a Wednesday
    }
    return pd.DataFrame([{**trip, **overrides}])


def test_returns_expected_features():
    (features,) = prepare_features(one_trip())
    assert list(features) == FEATURES
    assert features == {
        "PULocationID": "161",
        "DOLocationID": "236",
        "trip_distance": 2.5,
        "pickup_hour": 8,
        "pickup_dow": 2,
    }


def test_location_ids_are_categorical_strings():
    # strings -> DictVectorizer one-hot encodes them instead of treating them as numbers
    (features,) = prepare_features(one_trip(PULocationID=1.0))
    assert features["PULocationID"] == "1"


def test_accepts_datetime_strings_like_the_api_sends():
    (from_string,) = prepare_features(one_trip(tpep_pickup_datetime="2025-01-15T08:30:00"))
    (from_timestamp,) = prepare_features(one_trip())
    assert from_string == from_timestamp


def test_no_leaky_columns_are_used():
    df = one_trip(fare_amount=50.0, tip_amount=10.0, tpep_dropoff_datetime=pd.Timestamp("2025-01-15 09:00"))
    (features,) = prepare_features(df)
    assert set(features) == set(FEATURES)
