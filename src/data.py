"""Load the raw taxi data, compute the target, and drop unrealistic trips."""

from pathlib import Path

import pandas as pd

# outlier rule decided in notebooks/01_eda.ipynb
MIN_DURATION, MAX_DURATION = 1, 60  # minutes
MAX_DISTANCE = 100  # miles


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration"] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = add_duration(df)
    keep = (
        df.duration.between(MIN_DURATION, MAX_DURATION)
        & (df.trip_distance > 0)
        & (df.trip_distance <= MAX_DISTANCE)
    )
    return df[keep]


def load_and_clean(path: str | Path) -> pd.DataFrame:
    return clean(load_raw(path))
