import tensorflow as tf
import os
import csv

# === Model Save/Load Utilities ===

def save_model(model, path='model/saved/cnn_model.h5'):
    """
    Save the trained TensorFlow/Keras model to the specified path.
    Creates the directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"✅ Model saved to: {path}")

def load_model(path='model/saved/cnn_model.h5'):
    """
    Load a saved TensorFlow/Keras model from the given path.
    Raises an error if the file does not exist.
    """
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        print(f"✅ Model loaded from: {path}")
        return model
    else:
        raise FileNotFoundError(f"❌ Model file not found at {path}")

# === CSV Logging Utility ===

def log_metrics_to_csv(logfile, epoch, train_acc, val_acc, train_loss, val_loss):
    """
    Append training and validation metrics to a CSV log file.
    If the file does not exist, a header is written first.
    """
    file_exists = os.path.isfile(logfile)
    
    with open(logfile, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if file is new
        if not file_exists:
            writer.writerow(['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss'])

        # Write current metrics
        writer.writerow([epoch, train_acc, val_acc, train_loss, val_loss])
