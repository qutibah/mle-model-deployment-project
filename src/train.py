"""Train the taxi duration model, log it to MLflow, and save it for the API.

    uv run python -m src.train --help
"""

import time
from pathlib import Path

import click
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from src import data
from src.features import FEATURES, prepare_features
from src.model import DEFAULT_MODEL_PATH, TRUSTED_TYPES, save_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "yellow_tripdata_2025-01.parquet"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "nyc-taxi-duration"
RANDOM_STATE = 42


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def split(df, sample_size: int | None):
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=RANDOM_STATE)
    return train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)


def build_pipeline(**rf_params) -> Pipeline:
    return make_pipeline(
        DictVectorizer(),
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **rf_params),
    )


def train(data_path, sample_size, n_estimators, max_depth, min_samples_leaf, model_path, run_name):
    df = data.load_and_clean(data_path)
    train_df, val_df = split(df, sample_size)
    X_train, y_train = prepare_features(train_df), train_df.duration.to_numpy()
    X_val, y_val = prepare_features(val_df), val_df.duration.to_numpy()

    rf_params = {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}

    setup_mlflow()
    with mlflow.start_run(run_name=run_name):
        start = time.perf_counter()
        pipeline = build_pipeline(**rf_params).fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        metrics = {
            "train_rmse": root_mean_squared_error(y_train, pipeline.predict(X_train)),
            "val_rmse": root_mean_squared_error(y_val, pipeline.predict(X_val)),
            # the number the model has to beat: always predict the training mean
            "mean_baseline_rmse": root_mean_squared_error(y_val, np.full_like(y_val, y_train.mean())),
            "train_seconds": train_seconds,
        }
        mlflow.log_params({
            **rf_params,
            "model": "RandomForestRegressor",
            "sample_size": sample_size or len(df),
            "features": ",".join(FEATURES),
            "outlier_rule": f"{data.MIN_DURATION}<=duration<={data.MAX_DURATION}, 0<distance<={data.MAX_DISTANCE}",
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="model", skops_trusted_types=TRUSTED_TYPES)

        saved = save_model(pipeline, model_path)
        mlflow.log_param("model_path", str(saved))

    click.echo(
        f"train RMSE {metrics['train_rmse']:.2f} | val RMSE {metrics['val_rmse']:.2f} "
        f"| mean baseline {metrics['mean_baseline_rmse']:.2f} min "
        f"({train_seconds:.0f}s) -> saved to {saved}"
    )
    return pipeline, metrics


@click.command()
@click.option("--data-path", type=click.Path(exists=True, path_type=Path), default=DEFAULT_DATA_PATH, show_default=True)
@click.option("--sample-size", type=int, default=200_000, show_default=True, help="Rows to sample; 0 = use all.")
@click.option("--n-estimators", type=int, default=50, show_default=True)
@click.option("--max-depth", type=int, default=20, show_default=True)
@click.option("--min-samples-leaf", type=int, default=5, show_default=True)
@click.option("--model-path", type=click.Path(path_type=Path), default=DEFAULT_MODEL_PATH, show_default=True)
@click.option("--run-name", default="rf-train-script", show_default=True)
def main(**kwargs):
    """Train the RandomForest duration model and save it for the API."""
    train(**kwargs)


if __name__ == "__main__":
    main()
