import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def load_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    images = []
    labels = []
    label_map = {}

    for label_index, person_dir in enumerate(sorted(dataset_path.iterdir())):
        if not person_dir.is_dir():
            continue
        label_map[label_index] = person_dir.name
        for image_path in person_dir.glob("*.jpg"):
            image = keras.utils.load_img(image_path, target_size=(64, 64))
            image_array = keras.utils.img_to_array(image)
            images.append(image_array)
            labels.append(label_index)

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels, dtype="int32")
    return images, labels, label_map


def build_model(num_classes):
    model = keras.Sequential(
        [
            layers.Input(shape=(64, 64, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(dataset_dir, epochs, output_dir):
    images, labels, label_map = load_dataset(dataset_dir)
    if len(images) == 0:
        raise RuntimeError("No images found in dataset. Collect images first.")

    model = build_model(num_classes=len(label_map))
    model.fit(images, labels, epochs=epochs, batch_size=16, validation_split=0.2)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(output_path / "face_cnn.h5")

    with open(output_path / "labels.json", "w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN face classifier.")
    parser.add_argument("--dataset-dir", default="data/dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.dataset_dir, args.epochs, args.output_dir)
