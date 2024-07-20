import numpy as np
import pandas as pd
from collections import Counter
import pickle
import matplotlib.pyplot as plt

class KNeighborsClassifier:
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, xi, X):
        return np.sqrt(np.sum((X - xi) ** 2, axis=1))

    def predict(self, X_test):
        predictions = []
        for xi in X_test:
            distances = self.euclidean_distance(xi, self.X_train)
            k_indices = distances.argsort()[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions

def evaluate(y_pred, y_gt):
    accuracy = np.sum(y_pred == y_gt) / len(y_gt)
    return accuracy

# Function to load data
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

# Load datasets
datasets = {
    'blobs': pd.read_pickle('data_blobs.pkl'),
    'moons': pd.read_pickle('data_moons.pkl'),
    'full': pd.read_pickle('data_full.pkl')
}

k_values = [1, 15, 30]

# Evaluate classifiers
for name, (X_train, y_train, X_test, y_test) in datasets.items():
    fig, axes = plt.subplots(2, len(k_values), figsize=(15, 8))
    fig.suptitle(f'k-NN Performance on different dataset (k = 15)', size=16)
    
    for idx, k in enumerate(k_values):
        classifier = KNeighborsClassifier(k=k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = evaluate(y_pred, y_test)
        print(f"{name.capitalize()} dataset - Accuracy with k={k}: {accuracy:.2f}")

        # Ground truth plot
        axes[0, idx].scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='x', cmap='viridis', label=f"{name.capitalize()} GT")
        axes[0, idx].set_title(f"{name.capitalize()} GT")

        # Prediction plot
        axes[1, idx].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='>', cmap='viridis', label=f"{name.capitalize()} pred")
        axes[1, idx].set_title(f"{name.capitalize()} pred")
        axes[1, idx].legend(loc='upper right')

    for ax in axes.flat:
        ax.label_outer()

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
