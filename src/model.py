"""Save and load the trained pipeline with skops (no arbitrary code execution, unlike pickle)."""

from pathlib import Path

import skops.io as sio
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "model.skops"

# Types skops won't load by default. Reviewed: these come from our own
# DictVectorizer + RandomForest pipeline, which we train and save ourselves.
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def save_model(pipeline: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(pipeline, path)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Train one first: uv run python -m src.train")
    return sio.load(path, trusted=TRUSTED_TYPES)
