import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def euclidean_distance(xi, X):
    return np.sqrt(np.sum((xi - X)**2, axis=1))

class KNeighborsClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = np.empty(len(X))
        for i, xi in enumerate(X):
            distances = euclidean_distance(xi, self.X_train)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            votes = np.bincount(nearest_labels.astype(int))
            y_pred[i] = np.argmax(votes)
        return y_pred

def evaluate(y_pred, y_gt):
    return np.mean(y_pred == y_gt)

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pandas.read_pickle(f)
    return data

# Load the datasets
datasets = {
    'blobs': load_data('data_blobs.pkl'),
    'moons': load_data('data_moons.pkl'),
    'full': load_data('data_full.pkl')
}

# Color maps for each dataset
colors = {
    'blobs': ['red', 'blue', 'green', 'yellow'],
    'moons': ['red', 'yellow'],
    'full': ['red', 'yellow']
}

# Initialize, train, and evaluate k-NN classifiers for k=15 and plot results
k = 15
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

for idx, (name, data) in enumerate(datasets.items()):
    print(data)
    X_train = data[['x1_train', 'x2_train']].values
    y_train = data['y_train'].values
    X_test = data[['x1_test', 'x2_test']].values
    y_test = data['y_test'].values

    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    unique_labels = unique_labels[~np.isnan(unique_labels)]

    clf = KNeighborsClassifier(k=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = evaluate(y_pred, y_test)
    print(f'Accuracy for {name} dataset with k={k}: {accuracy:.2f}')

    # Plot ground truth data
    for label in unique_labels:
        idx_train = y_train.tolist() == label
        idx_test = y_test.tolist() == label
        label = int(label)
        color = colors[name][label]

        axes[0, idx].scatter(X_train[idx_train, 0], X_train[idx_train, 1], color=color, marker='x', label=label)
        # axes[0, idx].scatter(X_test[idx_test, 0], X_test[idx_test, 1], color=color)

    axes[0, idx].set_title(f'{name.capitalize()} GT')
    axes[0, idx].legend()

    # Plot predicted data
    for label in unique_labels:
        idx_train = y_train.tolist() == label
        idx_pred = y_pred.tolist() == label
        label = int(label)
        color = colors[name][label]

        axes[1, idx].scatter(X_train[idx_train, 0], X_train[idx_train, 1], color=color, marker='x', label=label)
        axes[1, idx].scatter(X_test[idx_pred, 0], X_test[idx_pred, 1], color=color, marker='>')

    axes[1, idx].set_title(f'{name.capitalize()} Pred')
    axes[1, idx].legend()

plt.tight_layout()
plt.show()
