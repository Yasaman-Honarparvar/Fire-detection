"""Command-line entry point for the fire detection pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fire_detection.dataset import build_dataset
from fire_detection.model import predict_image, train_classifier
from fire_detection.visualize import highlight_fire

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fire_detection")

DEFAULT_FIRE_DIR = Path("data/samples/fire")
DEFAULT_NO_FIRE_DIR = Path("data/samples/no_fire")
DEFAULT_PREDICT_IMAGE = Path("data/samples/predict/firewater2.jpg")
DEFAULT_OUTPUT_DIR = Path("outputs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and highlight fire in an image using a pixel-wise SVM.")
    parser.add_argument("--fire-dir", type=Path, default=DEFAULT_FIRE_DIR, help="Directory of training images that contain fire.")
    parser.add_argument("--no-fire-dir", type=Path, default=DEFAULT_NO_FIRE_DIR, help="Directory of training images that do not contain fire.")
    parser.add_argument("--image", type=Path, default=DEFAULT_PREDICT_IMAGE, help="Image to run fire detection on.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write the dataset and highlighted image to.")
    parser.add_argument("--kernel", default="linear", choices=["linear", "rbf", "sigmoid"], help="SVM kernel to train with.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fire_images = sorted(args.fire_dir.iterdir())
    no_fire_images = sorted(args.no_fire_dir.iterdir())
    logger.info("Building dataset from %d fire and %d no-fire images", len(fire_images), len(no_fire_images))
    dataset = build_dataset(fire_images, no_fire_images)
    dataset.to_csv(args.output_dir / "image_dataset.csv", index=False)

    classifier, accuracy = train_classifier(dataset, kernel=args.kernel)
    logger.info("Trained %s-kernel SVM, held-out accuracy: %.2f%%", args.kernel, accuracy * 100)

    predicted = predict_image(classifier, args.image)
    predicted.to_csv(args.output_dir / "pred_dataset.csv", index=False)

    output_image = args.output_dir / f"highlighted_{args.image.name}"
    highlight_fire(args.image, predicted, output_image)
    logger.info("Saved highlighted image to %s", output_image)


if __name__ == "__main__":
    main()
