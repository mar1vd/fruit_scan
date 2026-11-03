"""Train a compact CNN on the Fruits-360 Kaggle directory structure."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers

DEFAULT_CLASSES = [
    "Apple Braeburn 1",
    "Apple Golden 1",
    "Apple Granny Smith 1",
    "Banana 1",
    "Cherry 1",
    "Grape Blue 1",
    "Kiwi 1",
    "Lemon 1",
    "Mango 1",
    "Orange 1",
    "Peach 1",
    "Pear 1",
    "Pineapple 1",
    "Strawberry 1",
    "Watermelon 1"
]

IMAGE_SIZE = (100, 100)
BATCH_SIZE = 32
SEED = 42


def make_datasets(data_dir: Path, selected_classes: list[str], batch_size: int):
    train_dir = data_dir / "Training"
    test_dir = data_dir / "Test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{test_dir}'. "
            "Extract the Kaggle dataset so data/fruits-360/Training and Test exist."
        )

    available_train = {p.name for p in train_dir.iterdir() if p.is_dir()}
    available_test = {p.name for p in test_dir.iterdir() if p.is_dir()}
    classes = [c for c in selected_classes if c in available_train and c in available_test]
    if len(classes) < 2:
        raise ValueError(
            "Fewer than two configured classes were found. Check folder names in "
            f"{train_dir}. Available examples: {sorted(available_train)[:15]}"
        )

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="int",
        class_names=classes, image_size=IMAGE_SIZE, batch_size=batch_size,
        shuffle=True, seed=SEED
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="int",
        class_names=classes, image_size=IMAGE_SIZE, batch_size=batch_size,
        shuffle=False
    )

    # Create validation from training data. Test remains untouched until final evaluation.
    total = tf.data.experimental.cardinality(train_ds).numpy()
    val_batches = max(1, int(total * 0.15))
    val_ds = train_ds.take(val_batches)
    train_ds = train_ds.skip(val_batches)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, classes


def build_model(num_classes: int):
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="fruitscan_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/fruits-360")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = make_datasets(
        data_dir, args.classes, args.batch_size
    )
    model = build_model(len(class_names))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        )
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": report
    }

    model.save(output_dir / "fruit_model.keras")
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "training_history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2),
        encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.65), max(8, len(class_names) * 0.55)))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, xticks_rotation=90, colorbar=False
    )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    print(f"Classes: {len(class_names)}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
