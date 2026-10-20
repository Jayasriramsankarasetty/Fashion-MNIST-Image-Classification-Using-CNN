"""
train.py: Preprocess Fashion-MNIST, train the CNN model, and evaluate performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from model import CLASS_NAMES, create_fashion_cnn


def load_and_preprocess_data():
    """
    Loads the Fashion-MNIST dataset and prepares it for CNN input.
    - Normalizes pixel values from [0, 255] to [0, 1].
    - Reshapes images to include the channel dimension (28, 28, 1).
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize to [0, 1] float32
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Expand dims to add channel axis: (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
    print(f"Input shape: {x_train.shape[1:]}")
    return (x_train, y_train), (x_test, y_test)


def plot_history(history, save_path="training_history.png"):
    """
    Saves a plot of training and validation accuracy and loss over epochs.
    """
    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history["accuracy"], label="Training Accuracy", marker="o")
    plt.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", marker="o")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history["loss"], label="Training Loss", marker="o")
    plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss", marker="o")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training history saved to {save_path}")
    plt.close()


def train(epochs=12, batch_size=64, model_path="fashion_cnn_model.keras"):
    """
    Trains the CNN model on Fashion-MNIST dataset.
    """
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = create_fashion_cnn(input_shape=(28, 28, 1), num_classes=10)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nStarting CNN training...")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[early_stop],
    )

    # Evaluate on unseen test images
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Final Test Loss:     {test_loss:.4f}")

    # Save model and plot
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")
    plot_history(history)

    return model


if __name__ == "__main__":
    train()
