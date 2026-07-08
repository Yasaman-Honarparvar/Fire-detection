from pathlib import Path

from fire_detection.dataset import build_dataset
from fire_detection.model import predict_image, train_classifier
from fire_detection.visualize import highlight_fire

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
FIRE_IMAGES = [DATA_DIR / "fire" / "fire.jfif", DATA_DIR / "fire" / "fire_2.jpg"]
NO_FIRE_IMAGES = [DATA_DIR / "no_fire" / "demo_2.jpg", DATA_DIR / "no_fire" / "river.jpg"]
PREDICT_IMAGE = DATA_DIR / "predict" / "firewater2.jpg"


def test_highlight_fire_writes_image_matching_input_shape(tmp_path):
    import imageio.v3 as iio

    dataset = build_dataset(FIRE_IMAGES, NO_FIRE_IMAGES)
    classifier, _ = train_classifier(dataset)
    predicted = predict_image(classifier, PREDICT_IMAGE)

    output_path = tmp_path / "highlighted.jpeg"
    rgb_array = highlight_fire(PREDICT_IMAGE, predicted, output_path)

    original_shape = iio.imread(PREDICT_IMAGE)[..., :3].shape
    assert rgb_array.shape == original_shape
    assert output_path.exists()
