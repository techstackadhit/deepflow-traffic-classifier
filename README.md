# DeepFlow: Network Traffic Classification Simulation using CNN and LSTM

**DeepFlow** is a deep learning-based simulation project designed to classify network traffic types using CNN and LSTM architectures. It utilizes publicly available datasets and does **not** require physical network recording or real-time traffic capture.DeepFlow is a deep learning-based simulation project for classifying network traffic patterns using CNN and LSTM architectures. It is trained on the UNSW-NB15 dataset, which captures a diverse mix of benign and malicious traffic scenarios. This allows DeepFlow to operate entirely offline, without requiring real-time packet capture or physical network monitoring.

This project builds a full pipeline—from preprocessing to evaluation—to compare deep learning models on flow-based network traffic data.

---

## Project Structure

```
deepflow-traffic-classifier/
├── data/
│   ├── raw/              # Original datasets (UNSW-NB15)
│   └── split/            # Train/val/test sets after preprocessing
│
├── evaluate/
│   ├── evaluate.py       # Evaluate CNN & LSTM on test set
│   ├── plot_metrics.py   # Plot accuracy & loss curves
│   └── eval_report.txt   # Saved evaluation results
│
├── model/
│   ├── cnn.py            # CNN architecture
│   ├── lstm.py           # LSTM architecture
│   ├── utils.py          # Save/load model & log metrics
│   └── saved/            # Trained model files (.h5)
│
├── train/
│   ├── train_cnn.py      # Training script for CNN
│   └── train_lstm.py     # Training script for LSTM
│
├── notebooks/
│   └── eda.ipynb         # EDA & feature analysis
│
├── utils/
│   └── preprocessing.py  # Data cleaning, scaling, and splitting
│
├── experiments/
│   └── logs/             # Training logs (per epoch)
│
├── requirements.txt      # Python dependencies
├── LICENSE               # (Optional) License file
└── README.md             # Project overview & instructions
```

---

## Getting Started

### 1. Clone the Repository
```
git clone https://github.com/techstackadhit/deepflow-traffic-classifier.git
cd deepflow-traffic-classifier
```

### 2. Create a Virtual Enviroment
```
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

---

## Project Goals
- Build deep learning models (CNN and LSTM) for traffic classification
- Simulate a full ML pipeline using public datasets
- Compare model accuracy, generalization, and efficiency

--- 

## Datasets Used
|Dataset	     |   Description                                |
|----------------|----------------------------------------------|
|UNSW-NB15	     | Normal vs malicious traffic flows (rich labels)|

All datasets are stored under data/raw/.

---

## Experiments
- CNN 1D (3–4 layers) vs LSTM (1–2 layers)
- Feature selection using EDA insights
-  Stratified data split from training-dataset: 80% train, 20% val

---

## Evaluation Metrics
- Accuracy
- F1-Score
- Confusion Matrix
- Training Time

---

## Tools
- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- TensorFlow or PyTorch
- Jupyter Notebook, VS Code

---

## Author / Credits

- **Author**: Aditya Arya Putranda
- **Year**: 2025  
- **Affiliation**: Independent Research 

---

## License
This project is licensed under the MIT License. Feel free to use, modify, and share.

---
