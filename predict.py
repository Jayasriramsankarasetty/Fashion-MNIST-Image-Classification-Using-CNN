"""
predict.py: Evaluate trained CNN model and visualize sample predictions on unseen test images.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import CLASS_NAMES


def load_test_data():
    """Loads and preprocesses unseen test images from Fashion-MNIST."""
    _, (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test_norm = (x_test.astype(np.float32) / 255.0)
    x_test_norm = np.expand_dims(x_test_norm, axis=-1)
    return x_test, x_test_norm, y_test


def visualize_predictions(model_path="fashion_cnn_model.keras", num_samples=15, save_path="predictions_sample.png"):
    """
    Loads the trained model and visualizes sample predictions on unseen images.
    """
    print(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    raw_images, preprocessed_images, true_labels = load_test_data()

    # Overall test evaluation
    test_loss, test_acc = model.evaluate(preprocessed_images, true_labels, verbose=0)
    print(f"Overall Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Overall Test Loss:     {test_loss:.4f}")

    # Select sample indices
    indices = np.random.choice(len(true_labels), size=num_samples, replace=False)
    sample_images = preprocessed_images[indices]
    sample_raw = raw_images[indices]
    sample_true = true_labels[indices]

    # Predict
    probabilities = model.predict(sample_images, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)

    # Plot sample predictions
    rows = 3
    cols = 5
    plt.figure(figsize=(15, 9))

    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(sample_raw[i], cmap="gray")
        plt.axis("off")

        pred_idx = predicted_labels[i]
        true_idx = sample_true[i]
        confidence = probabilities[i][pred_idx] * 100

        pred_name = CLASS_NAMES[pred_idx]
        true_name = CLASS_NAMES[true_idx]

        is_correct = pred_idx == true_idx
        color = "darkgreen" if is_correct else "red"

        title = f"Pred: {pred_name} ({confidence:.1f}%)\nTrue: {true_name}"
        plt.title(title, fontsize=10, color=color, fontweight="bold")

    plt.suptitle("Fashion-MNIST CNN Sample Predictions on Unseen Test Images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Sample predictions visualization saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    visualize_predictions()
