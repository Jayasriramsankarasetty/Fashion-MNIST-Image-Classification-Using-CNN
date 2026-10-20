"""
model.py: CNN Architecture for Fashion-MNIST Classification.
"""

from tensorflow.keras import layers, models

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def create_fashion_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds a Convolutional Neural Network (CNN) for 28x28 grayscale image classification.
    """
    model = models.Sequential(
        [
            # Input Layer
            layers.Input(shape=input_shape),

            # Block 1: Feature Extraction
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Block 2: Higher-level Feature Extraction
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Block 3: Classification Head
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


if __name__ == "__main__":
    cnn_model = create_fashion_cnn()
    cnn_model.summary()
