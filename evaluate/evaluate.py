import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model.utils import load_model

def evaluate_model(model, X_test, y_test, name="Model", output_file=None):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    report = classification_report(y_test, y_pred_classes, target_names=['Normal', 'Attack'])
    matrix = confusion_matrix(y_test, y_pred_classes)

    result_text = f"\nEvaluation Result for {name}:\n\n"
    result_text += report + "\n"
    result_text += "Confusion Matrix:\n" + str(matrix) + "\n"

    print(result_text)

    # Save to file if specified
    if output_file:
        with open(output_file, 'a') as f:
            f.write(result_text)

# === Load test set ===
df_test = pd.read_csv('data/split/test.csv')
X_test = df_test.drop(columns=['label']).values
y_test = df_test['label'].values

# === Reshape for LSTM ===
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# === Load models ===
cnn_model = load_model('model/saved/cnn_model.h5')
lstm_model = load_model('model/saved/lstm_model.h5')

# === Evaluate both ===
output_path = 'evaluate/eval_report.txt'
open(output_path, 'w').close()  # optional: clear previous content

evaluate_model(cnn_model, X_test, y_test, name="CNN", output_file=output_path)
evaluate_model(lstm_model, X_test_lstm, y_test, name="LSTM", output_file=output_path)