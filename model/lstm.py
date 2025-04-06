import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(input_shape, num_classes=2):
    model = models.Sequential()

    # Reshape input: (batch_size, features) → (batch_size, features, 1)
    model.add(layers.Reshape((input_shape, 1), input_shape=(input_shape,)))

    # LSTM Layer
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.3))

    # Fully connected layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model