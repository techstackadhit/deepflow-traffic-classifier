import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(logfile, label, color=None):
    df = pd.read_csv(logfile)

    # Plot Accuracy
    plt.plot(df['epoch'], df['train_acc'], linestyle='--', label=f'{label} Train Accuracy', color=color)
    plt.plot(df['epoch'], df['val_acc'], linestyle='-', label=f'{label} Val Accuracy', color=color)

def plot_losses(logfile, label, color=None):
    df = pd.read_csv(logfile)

    # Plot Loss
    plt.plot(df['epoch'], df['train_loss'], linestyle='--', label=f'{label} Train Loss', color=color)
    plt.plot(df['epoch'], df['val_loss'], linestyle='-', label=f'{label} Val Loss', color=color)

def main():
    plt.figure(figsize=(10,6))
    plot_metrics('experiments/logs/cnn_log.csv', label='CNN', color='blue')
    plot_metrics('experiments/logs/lstm_log.csv', label='LSTM', color='green')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('evaluate/accuracy_curve.png')
    plt.show()

    plt.figure(figsize=(10,6))
    plot_losses('experiments/logs/cnn_log.csv', label='CNN', color='blue')
    plot_losses('experiments/logs/lstm_log.csv', label='LSTM', color='green')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('evaluate/loss_curve.png')
    plt.show()

if __name__ == '__main__':
    main()