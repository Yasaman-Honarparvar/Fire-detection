from pathlib import Path

from fire_detection.dataset import build_dataset
from fire_detection.model import predict_image, train_classifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
FIRE_IMAGES = [DATA_DIR / "fire" / "fire.jfif", DATA_DIR / "fire" / "fire_2.jpg"]
NO_FIRE_IMAGES = [DATA_DIR / "no_fire" / "demo_2.jpg", DATA_DIR / "no_fire" / "river.jpg"]
PREDICT_IMAGE = DATA_DIR / "predict" / "firewater2.jpg"


def test_train_classifier_returns_accuracy_in_range():
    dataset = build_dataset(FIRE_IMAGES, NO_FIRE_IMAGES)
    _, accuracy = train_classifier(dataset)
    assert 0.0 <= accuracy <= 1.0


def test_predict_image_labels_every_pixel():
    dataset = build_dataset(FIRE_IMAGES, NO_FIRE_IMAGES)
    classifier, _ = train_classifier(dataset)
    predicted = predict_image(classifier, PREDICT_IMAGE)
    assert set(predicted["label"].unique()) <= {1, 2}
    assert len(predicted) > 0
