"""The ONE place where model features are built - used by training and by the API."""

import pandas as pd

CATEGORICAL = ["PULocationID", "DOLocationID"]
NUMERICAL = ["trip_distance", "pickup_hour", "pickup_dow"]
FEATURES = CATEGORICAL + NUMERICAL

# raw columns a caller must provide (all known when the trip starts)
RAW_COLUMNS = ["PULocationID", "DOLocationID", "trip_distance", "tpep_pickup_datetime"]


def prepare_features(df: pd.DataFrame) -> list[dict]:
    """Turn raw trip rows into DictVectorizer input.

    Location IDs become strings so DictVectorizer one-hot encodes them as categories.
    """
    pickup = pd.to_datetime(df.tpep_pickup_datetime)
    features = pd.DataFrame({
        "PULocationID": df.PULocationID.astype(int).astype(str),
        "DOLocationID": df.DOLocationID.astype(int).astype(str),
        "trip_distance": df.trip_distance.astype(float),
        "pickup_hour": pickup.dt.hour,
        "pickup_dow": pickup.dt.day_of_week,
    })
    return features[FEATURES].to_dict(orient="records")
