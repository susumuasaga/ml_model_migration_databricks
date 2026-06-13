from pathlib import Path
from types import SimpleNamespace

import pytest

from model import SentimentModel


@pytest.fixture(scope="module")
def sentiment_model():
    model = SentimentModel()
    artifact_dir = Path(__file__).parents[1] / "model_artifact"
    weights_path = artifact_dir / "model.npz"
    dictionary_path = artifact_dir / "dictionary.json"
    context = SimpleNamespace(
        artifacts={"weights": str(weights_path), "dictionary": str(dictionary_path)}
    )
    model.load_context(context)
    return model
