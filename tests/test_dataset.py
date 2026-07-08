from pathlib import Path

from fire_detection.dataset import FIRE_LABEL, NO_FIRE_LABEL, build_dataset, labeled_pixels, load_pixels

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
FIRE_IMAGE = DATA_DIR / "fire" / "fire.jfif"
NO_FIRE_IMAGE = DATA_DIR / "no_fire" / "demo_2.jpg"


def test_load_pixels_shape():
    pixels = load_pixels(FIRE_IMAGE)
    assert pixels.shape[1] == 3
    assert pixels.shape[0] > 0


def test_labeled_pixels_fire():
    df = labeled_pixels(FIRE_IMAGE, FIRE_LABEL)
    assert (df["label"] == FIRE_LABEL).all()
    assert list(df.columns) == ["red", "green", "blue", "label"]


def test_labeled_pixels_no_fire():
    df = labeled_pixels(NO_FIRE_IMAGE, NO_FIRE_LABEL)
    assert (df["label"] == NO_FIRE_LABEL).all()


def test_build_dataset_combines_both_classes():
    dataset = build_dataset([FIRE_IMAGE], [NO_FIRE_IMAGE])
    assert set(dataset["label"].unique()) == {FIRE_LABEL, NO_FIRE_LABEL}
    assert len(dataset) == len(load_pixels(FIRE_IMAGE)) + len(load_pixels(NO_FIRE_IMAGE))
