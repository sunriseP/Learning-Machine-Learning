import numpy as np
import pandas as pd
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 

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

# Function to load data safely handling potential errors
def load_data(file_path):
    with open(file_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            # Handling exceptions that could arise during loading
            print(f"Error loading data: {e}")
            raise
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

# Paths to the datasets
data_paths = ['data_blobs.pkl', 'data_moons.pkl', 'data_full.pkl']
k_values = [1, 15, 30]
cmap_blobs = ListedColormap(['red', 'blue', 'green', 'yellow'])
cmap_moons_full = ListedColormap(['red', 'yellow'])
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

data_blobs = pd.read_pickle('data_blobs.pkl')
data_moons = pd.read_pickle('data_moons.pkl')
data_full = pd.read_pickle('data_full.pkl')

X_train_blobs = data_blobs[['x1_train', 'x2_train']].values
y_train_blobs = data_blobs[['y_train']].values
X_test_blobs = data_blobs[['x1_test', 'x2_test']].values
y_test_blobs = data_blobs[['y_test']].values

X_train_moons = data_moons[['x1_train', 'x2_train']].values
y_train_moons = data_moons[['y_train']].values
X_test_moons = data_moons[['x1_test', 'x2_test']].values
y_test_moons = data_moons[['y_test']].values

X_train_full = data_full[['x1_train', 'x2_train']].values
y_train_full = data_full[['y_train']].values
X_test_full = data_full[['x1_test', 'x2_test']].values
y_test_full = data_full[['y_test']].values

datasets = {
    'Blobs': (X_train_blobs, y_train_blobs, X_test_blobs, y_test_blobs),
    'Moons': (X_train_moons, y_train_moons, X_test_moons, y_test_moons),
    'Full': (X_train_full, y_train_full, X_test_full, y_test_full)
}

# Train and predict for each dataset
for i, (dataset_name, (X_train, y_train, X_test, y_test)) in enumerate(datasets.items()):
    for k in k_values:
    # Initialize k-NN classifier with k=15
        classifier = KNeighborsClassifier(k=k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Select the appropriate color map
        cmap = cmap_blobs if dataset_name == 'Blobs' else cmap_moons_full
        
        # Plot GT
        axes[0, i].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='x', label=f"{dataset_name} GT")
        axes[0, i].set_title(f"{dataset_name} GT")
        
        # Plot predictions
        axes[1, i].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=cmap, marker='>', label=f"{dataset_name} pred")
        axes[1, i].set_title(f"{dataset_name} pred")

# Add a legend with class labels
for ax, dataset_name in zip(axes[0], datasets.keys()):
    cmap = cmap_blobs if dataset_name == 'Blobs' else cmap_moons_full
    classes = np.unique(datasets[dataset_name][1])
    handles = [plt.Line2D([0], [0], color=cmap.colors[int(label)], marker='o', linestyle='', label=str(label)) for label in classes]
    ax.legend(handles=handles, title='Classes')

plt.tight_layout()
plt.show()

# for path in data_paths:
#     print(f"Evaluating dataset {path}:")
#     data = pd.read_pickle(path)
#     X_train = data[['x1_train', 'x2_train']].values
#     y_train = data['y_train'].values
#     X_test = data[['x1_test', 'x2_test']].values
#     y_test = data['y_test'].values

#     for k in k_values:
#         classifier = KNeighborsClassifier(k=k)
#         classifier.fit(X_train, y_train)
#         y_pred = classifier.predict(X_test)
#         accuracy = evaluate(y_pred, y_test)
#         print(f"Accuracy with k={k}: {accuracy:.2f}")
#         cmap = cmap_blobs if path == 'data_blobs.pkl' else cmap_others

#         # # Plotting results if needed
#         # if k == 15:
#         # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='>', label='Predicted')
#         # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='x', label='Train Data')
#         # plt.title(f'k-NN Classification Results for {path} with k={k}')
#         # plt.legend()
#         # plt.show()
#         unique_labels = np.unique(y_train)
#         legend_handles = [plt.Line2D([0], [0], color=cmap.colors[int(label)], marker='o', linestyle='', label=str(label))
#                           for label in unique_labels]
#         plt.legend(handles=legend_handles, title='Classes')
# plt.show()
