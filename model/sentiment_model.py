import os
import json
import string
from typing import Any, override

import numpy as np
import mlflow
import pandas as pd


class SentimentModel(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc model for sentiment classification.

    This model reconstructs a legacy linear classifier using:
        - Bag-of-Words feature extraction
        - Pre-trained parameters (theta, theta_0)

    The model is self-contained and does not depend on the
    original training environment.
    """

    def __init__(self, theta=None, theta_0=None, dictionary=None):
        super().__init__()
        self.theta = theta
        self.theta_0 = theta_0
        self.dictionary = dictionary

    def save_artifact(self, path):
        """
        Save a trained model as a portable artifact.

        Args:
            path (str | Path): Directory where the artifact will be stored.

        The artifact structure:
            - model.npz: numerical parameters
            - dictionary.json: word-to-index mapping
        """
        os.makedirs(path, exist_ok=True)

        # Save model parameters
        np.savez(
            os.path.join(path, "model.npz"),
            theta=self.theta,
            theta_0=self.theta_0,
        )

        # Save dictionary
        with open(os.path.join(path, "dictionary.json"), "w") as f:
            json.dump(self.dictionary, f)

    @override
    def load_context(self, context):
        """
        Load model artifact into memory.

        Args:
            context: MLflow context containing the model artifact paths.
        """
        path = context.artifacts["weights"]

        data = np.load(path)
        self.theta = data["theta"]
        self.theta_0 = float(data["theta_0"])

        path = context.artifacts["dictionary"]
        with open(path) as f:
            self.dictionary = json.load(f)

    def vectorize(self, texts: pd.Series):
        """
        Convert raw text into Bag-of-Words feature vectors.

        Args:
            texts (pd.Series): Input text samples.

        Returns:
            np.ndarray: Feature matrix of shape (n_samples, vocab_size).
        """
        vocab_size = len(self.dictionary)
        X = np.zeros((len(texts), vocab_size))

        punctuation_map = str.maketrans(
            {ch: f" {ch} " for ch in string.punctuation + string.digits}
        )

        for i, text in enumerate(texts):
            tokens = text.lower().translate(punctuation_map).split()
            for word in tokens:
                if word in self.dictionary:
                    idx = self.dictionary[word]
                    X[i, idx] = 1

        return X

    @override
    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ):
        """
        Generate predictions from raw text input.

        Args:
            context: MLflow context (unused during inference).
            model_input: DataFrame with a single text column named "text".
            params: Optional dictionary of parameters (unused).

        Returns:
            A numpy array of predicted confidence scores.

            The confidence score for a sample is proportional
            to the signed distance from the decision boundary.
            A positive score indicates positive sentiment,
            while a negative score indicates negative sentiment.
        """
        if not isinstance(model_input, pd.DataFrame) or list(model_input.columns) != [
            "text"
        ]:
            raise TypeError(
                'model_input must be a pandas DataFrame with one column named "text".'
            )

        X = self.vectorize(model_input["text"])

        scores = X @ self.theta + self.theta_0
        return scores


mlflow.models.set_model(SentimentModel())
