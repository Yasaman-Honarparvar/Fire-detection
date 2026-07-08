"""Train an SVM classifier on labeled pixels and use it to predict fire pixels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from fire_detection.dataset import COLUMNS, load_pixels


def train_classifier(dataset: pd.DataFrame, kernel: str = "linear", test_size: float = 0.2) -> tuple[SVC, float]:
    """Fit an SVM on the dataset and return the classifier and its held-out test accuracy."""
    x = dataset[COLUMNS]
    y = dataset["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=0, shuffle=True
    )
    classifier = SVC(kernel=kernel)
    classifier.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, classifier.predict(x_test))
    return classifier, accuracy


def predict_image(classifier: SVC, image_path: str | Path) -> pd.DataFrame:
    """Predict a fire/no-fire label for every pixel of an image."""
    pixels = load_pixels(image_path)
    df = pd.DataFrame(pixels, columns=COLUMNS)
    df["label"] = classifier.predict(df[COLUMNS])
    return df
