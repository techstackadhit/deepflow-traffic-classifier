import sys
import os

# === Add root directory to import custom modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from model.cnn import build_cnn_model
from model.utils import save_model, log_metrics_to_csv

# === Load dataset ===
train_df = pd.read_csv('data/split/train.csv')
val_df = pd.read_csv('data/split/val.csv')

X_train = train_df.drop(columns=['label']).values
y_train = train_df['label'].values

X_val = val_df.drop(columns=['label']).values
y_val = val_df['label'].values

# === Build model ===
input_dim = X_train.shape[1]
model = build_cnn_model(input_shape=input_dim, num_classes=2)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Setup training ===
epochs = 20
batch_size = 64
log_path = 'experiments/logs/cnn_log.csv'

# === Reset log file so header is written again ===
if os.path.exists(log_path):
    os.remove(log_path)
    
for epoch in range(1, epochs + 1):
    print(f"\n📦 Epoch {epoch}/{epochs}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=batch_size,
        verbose=1
    )

    train_acc = history.history['accuracy'][0]
    val_acc = history.history['val_accuracy'][0]
    train_loss = history.history['loss'][0]
    val_loss = history.history['val_loss'][0]

    # Log metrics
    log_metrics_to_csv(log_path, epoch, train_acc, val_acc, train_loss, val_loss)

# === Save model ===
save_model(model, path='model/saved/cnn_model.h5')

print("✅ Training complete. Model and log saved.")