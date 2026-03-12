# Implement functions to calculate key performance metrics for a binary classifier, especially those suited for imbalanced data:
# Precision: TP / (TP + FP)
# Recall (True Positive Rate): TP / (TP + FN)
# F1-Score: Harmonic mean of precision and recall.
# False Alarm Rate (FAR): FP / (FP + TN)
# Also implement a function to plot a confusion matrix.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def calculate_metrics(y_true, y_pred):
    """
    Calculate key performance metrics for a binary classifier, especially those suited for imbalanced data.
    """
    # Convert to numpy arrays so element-wise comparisons work for any input type
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp)             if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)             if (tp + fn) > 0 else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) \
                                           if (precision + recall) > 0 else 0.0
    far       = fp / (fp + tn)             if (fp + tn) > 0 else 0.0

    return precision, recall, f1_score, far

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix for a binary classifier.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print(f"Confusion Matrix: {cm}")

if __name__ == "__main__":
    # Example usage
    # Notice 1s are anomalies and 0s are normal
    y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    precision, recall, f1_score, far = calculate_metrics(y_true, y_pred)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
    print(f"False Alarm Rate: {far:.2f}")
    plot_confusion_matrix(y_true, y_pred)