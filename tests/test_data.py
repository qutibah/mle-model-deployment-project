import pandas as pd

from src.data import add_duration, clean


def raw_trips(minutes, distances):
    pickup = pd.Timestamp("2025-01-10 12:00")
    return pd.DataFrame({
        "tpep_pickup_datetime": [pickup] * len(minutes),
        "tpep_dropoff_datetime": [pickup + pd.Timedelta(minutes=m) for m in minutes],
        "trip_distance": distances,
    })


def test_duration_is_in_minutes():
    df = add_duration(raw_trips([90 / 60, 15], [1, 1]))
    assert df.duration.tolist() == [1.5, 15.0]


def test_clean_removes_unrealistic_durations():
    # negative, zero, too short, ok, ok (boundary), too long
    df = clean(raw_trips([-5, 0, 0.5, 10, 60, 61], [1] * 6))
    assert df.duration.tolist() == [10, 60]


def test_clean_removes_bad_distances():
    df = clean(raw_trips([10] * 5, [-1, 0, 3, 100, 150]))
    assert df.trip_distance.tolist() == [3, 100]


def test_clean_does_not_modify_input():
    raw = raw_trips([10], [1])
    clean(raw)
    assert "duration" not in raw.columns


def test_clean_is_deterministic():
    raw = raw_trips([-5, 10, 30, 90], [1, 2, 0, 5])
    pd.testing.assert_frame_equal(clean(raw), clean(raw))
