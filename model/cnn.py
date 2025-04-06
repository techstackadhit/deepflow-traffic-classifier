import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape, num_classes=2):
    model = models.Sequential()

    # Reshape input to fit Conv1D:
    # (batch_size, features) → (batch_size, features, 1)
    model.add(layers.Reshape((input_shape, 1), input_shape=(input_shape,)))

    # First convolutional layer with 32 filters and kernel size 3
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))

    # Second convolutional layer with 64 filters
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))

    # Downsample the feature maps using max pooling
    model.add(layers.MaxPooling1D(pool_size=2))

    # Dropout for regularization (to prevent overfitting)
    model.add(layers.Dropout(0.3))

    # Flatten the feature maps to feed into the dense layers
    model.add(layers.Flatten())

    # Fully connected layer with 64 units
    model.add(layers.Dense(64, activation='relu'))

    # Output layer with softmax activation for binary classification
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model