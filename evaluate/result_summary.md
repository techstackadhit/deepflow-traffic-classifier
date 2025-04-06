# 📊 Evaluation Summary – CNN vs LSTM (UNSW-NB15)

This summary compares the performance of CNN and LSTM models trained for network traffic classification using flow-based features.

---

## 🧪 Accuracy Curve

- **CNN** achieves higher validation accuracy than LSTM across most epochs.
- **LSTM** improves steadily but never surpasses CNN.
- Both models reach ~93% validation accuracy, but CNN is **more stable** overall.
- CNN converges earlier and generalizes better in early-to-mid training.

---

## 📉 Loss Curve

- **CNN** maintains lower validation loss throughout training.
- **LSTM** starts with high loss and improves, but is more **sensitive** (slightly noisier curve).
- Wider gap between LSTM’s train/val loss suggests **possible overfitting**.
- CNN’s training and validation loss are both low and tight — indicating strong generalization.

---

## ✅ Model Comparison Table

| Aspect             | CNN                         | LSTM                        |
|--------------------|------------------------------|-----------------------------|
| Train Accuracy     | High, stable                 | Competitive, improves over time |
| Val Accuracy       | Higher across most epochs    | Slightly lower overall      |
| Train/Val Loss     | Lower, consistent            | Higher & more variable      |
| Convergence Speed  | Faster                       | Slower                      |
| Stability          | ✅ Smooth curve              | ⚠️ Slight fluctuation       |

---

## 💡 Conclusion

> **CNN outperforms LSTM** in this setup on UNSW-NB15, offering better accuracy, faster convergence, and more stable learning behavior.  
> For flow-based tabular data like this, **CNN is a strong and lightweight baseline** that performs reliably without requiring sequential modeling.

---