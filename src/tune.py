"""Tune the RandomForest with Optuna and compare it against the baseline model.

    uv run python -m src.tune --help

How it works:
1. Same 200k sample and 80/20 train/validation split as src.train.
2. Optuna tries different hyperparameters. Each trial trains on a small part of the
   TRAINING set and is scored on another part of the TRAINING set ("tuning set").
   The validation set is never seen during tuning, so its RMSE stays an honest number.
3. The best hyperparameters are retrained on the full training set and evaluated on
   the validation set - the same data that gave the baseline its 4.30 RMSE.
4. Baseline vs. tuned: RMSE, training time and model size, all logged to MLflow.
"""

import time
from pathlib import Path

import click
import mlflow
import optuna
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from src import data
from src.features import prepare_features
from src.model import ROOT, TRUSTED_TYPES, save_model
from src.train import (
    DEFAULT_DATA_PATH,
    EXPERIMENT,
    RANDOM_STATE,
    TRACKING_URI,
    build_pipeline,
    split,
)

# the settings src.train uses by default
BASELINE_PARAMS = {"n_estimators": 50, "max_depth": 20, "min_samples_leaf": 5}
TUNED_MODEL_PATH = ROOT / "artifacts" / "model-tuned.skops"


def suggest_params(trial: optuna.Trial) -> dict:
    """The search space: which values Optuna is allowed to try."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 150, step=10),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        # fraction of features each split may look at (1.0 = all, like the baseline)
        "max_features": trial.suggest_categorical("max_features", [1.0, 0.5, 0.3, "sqrt"]),
    }


def fit_and_score(params, X_train, y_train, X_eval, y_eval):
    start = time.perf_counter()
    pipeline = build_pipeline(**params).fit(X_train, y_train)
    seconds = time.perf_counter() - start
    rmse = root_mean_squared_error(y_eval, pipeline.predict(X_eval))
    return pipeline, rmse, seconds


def model_size_mb(pipeline, path: Path) -> float:
    return save_model(pipeline, path).stat().st_size / 1e6


@click.command()
@click.option("--data-path", type=click.Path(exists=True, path_type=Path), default=DEFAULT_DATA_PATH, show_default=True)
@click.option("--sample-size", type=int, default=200_000, show_default=True, help="Same sample as src.train.")
@click.option("--tuning-size", type=int, default=50_000, show_default=True,
              help="Rows of the training set used inside each trial (smaller = faster trials).")
@click.option("--n-trials", type=int, default=20, show_default=True)
@click.option("--tracking-uri", default=TRACKING_URI, show_default=True)
def main(data_path, sample_size, tuning_size, n_trials, tracking_uri):
    """Search RandomForest hyperparameters with Optuna and compare with the baseline."""
    # --- data: identical split to src.train, so the validation RMSE is comparable
    df = data.load_and_clean(data_path)
    train_df, val_df = split(df, sample_size)
    X_train, y_train = prepare_features(train_df), train_df.duration.to_numpy()
    X_val, y_val = prepare_features(val_df), val_df.duration.to_numpy()

    # --- tuning data: a smaller piece of the TRAINING set, split again (80/20)
    tune_df = train_df.sample(min(tuning_size, len(train_df)), random_state=RANDOM_STATE)
    fit_df, score_df = train_test_split(tune_df, test_size=0.2, random_state=RANDOM_STATE)
    X_fit, y_fit = prepare_features(fit_df), fit_df.duration.to_numpy()
    X_score, y_score = prepare_features(score_df), score_df.duration.to_numpy()

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with mlflow.start_run(run_name="optuna-tuning"):
        mlflow.log_params({"n_trials": n_trials, "sample_size": sample_size, "tuning_size": tuning_size})

        # --- 1. the objective: Optuna calls this once per trial and tries to MINIMIZE the result
        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
                _, rmse, seconds = fit_and_score(params, X_fit, y_fit, X_score, y_score)
                mlflow.log_params(params)
                mlflow.log_metrics({"tuning_rmse": rmse, "train_seconds": seconds})
            click.echo(f"trial {trial.number:2d}: tuning RMSE {rmse:.3f}  ({seconds:5.1f}s)  {params}")
            return rmse

        # --- 2. the study: runs the trials. TPE learns from earlier trials which values look promising
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params
        click.echo(f"\nbest trial #{study.best_trial.number}: tuning RMSE {study.best_value:.3f} with {best}")

        # --- 3. fair comparison on the full training set and the untouched validation set
        click.echo("\nretraining baseline and tuned model on the full training set...")
        results = {}
        for name, params in [("baseline", BASELINE_PARAMS), ("tuned", best)]:
            pipeline, rmse, seconds = fit_and_score(params, X_train, y_train, X_val, y_val)
            path = TUNED_MODEL_PATH if name == "tuned" else TUNED_MODEL_PATH.with_name("model-baseline-compare.skops")
            results[name] = {"val_rmse": rmse, "train_seconds": seconds, "model_mb": model_size_mb(pipeline, path)}
            mlflow.log_metrics({f"{name}_{k}": v for k, v in results[name].items()})
            if name == "tuned":
                mlflow.log_params({f"best_{k}": v for k, v in best.items()})
                mlflow.sklearn.log_model(pipeline, name="tuned_model", skops_trusted_types=TRUSTED_TYPES)
        TUNED_MODEL_PATH.with_name("model-baseline-compare.skops").unlink(missing_ok=True)

    b, t = results["baseline"], results["tuned"]
    click.echo(
        f"\n{'':10}{'val RMSE':>10}{'train time':>12}{'model size':>12}\n"
        f"{'baseline':10}{b['val_rmse']:>10.3f}{b['train_seconds']:>11.0f}s{b['model_mb']:>10.0f}MB\n"
        f"{'tuned':10}{t['val_rmse']:>10.3f}{t['train_seconds']:>11.0f}s{t['model_mb']:>10.0f}MB\n"
        f"\nRMSE change: {t['val_rmse'] - b['val_rmse']:+.3f} min "
        f"({(t['val_rmse'] - b['val_rmse']) / b['val_rmse']:+.1%})\n"
        f"tuned model saved to {TUNED_MODEL_PATH} (the API still uses artifacts/model.skops)"
    )


if __name__ == "__main__":
    main()
