from types import SimpleNamespace

import numpy as np
import pandas as pd

from model import SentimentModel


def test_save_artifact_and_load_context_round_trip(tmp_path):
    theta = np.array([1.25, -0.5, 2.0])
    theta_0 = -0.75
    dictionary = {"excellent": 0, "bad": 1, "!": 2}

    trained_model = SentimentModel(theta=theta, theta_0=theta_0, dictionary=dictionary)
    trained_model.save_artifact(tmp_path)

    loaded_model = SentimentModel()
    weights_path = tmp_path / "model.npz"
    dictionary_path = tmp_path / "dictionary.json"
    context = SimpleNamespace(
        artifacts={"weights": str(weights_path), "dictionary": str(dictionary_path)}
    )
    loaded_model.load_context(context)

    assert weights_path.is_file()
    assert dictionary_path.is_file()
    np.testing.assert_array_equal(loaded_model.theta, theta)
    assert loaded_model.theta_0 == theta_0
    assert loaded_model.dictionary == dictionary


def test_predicts_negative_sentiment(sentiment_model):
    text = (
        "Nasty No flavor. The candy is just red, No flavor. "
        "Just plain and chewy. I would never buy them again"
    )

    prediction = sentiment_model.predict(None, pd.DataFrame({"text": [text]}))

    assert prediction[0] < -0.42


def test_predicts_positive_sentiment(sentiment_model):
    text = (
        "YUMMY! You would never guess that they're sugar-free "
        "and it's so great that you can eat them pretty much guilt free! "
        "i was so impressed that i've ordered some for myself "
        "(w dark chocolate) to take to the office. "
        "These are just EXCELLENT!"
    )

    prediction = sentiment_model.predict(None, pd.DataFrame({"text": [text]}))

    assert prediction[0] > 1.1
