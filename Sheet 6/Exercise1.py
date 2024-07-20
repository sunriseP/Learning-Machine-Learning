import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

def euclidean_distance(xi, X):
    return np.sqrt(np.sum((xi - X)**2, axis=1))

class KNeighborsClassifier:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = np.empty(len(X))
        for i, xi in enumerate(X):
            distances = euclidean_distance(xi, self.X_train)
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            votes = np.bincount(nearest_labels.astype(int))
            y_pred[i] = np.argmax(votes)
        return y_pred

digits = load_digits()
X = digits.data
y = digits.target
# print(np.size(X)) 115008
# print(np.size(y))

train_data, test_data, train_target, test_target = train_test_split(X, y, train_size=0.7,random_state=1)
test_data, val_data, test_target, val_target = train_test_split(test_data, test_target, test_size=1/3,random_state=1)

print(np.size(train_data)) #80448
print(np.size(test_data)) #23040
print(np.size(val_data)) #11520
print(np.size(train_target)) #1257
print(np.size(test_target)) #360
print(np.size(val_target)) #180
# Completed of data segmentation

def evaluate(y_gt, y_pred):
    correct_predictions = 0
    for true, pred in zip(y_gt, y_pred):
        if true == pred:
            correct_predictions += 1
    accuracy = correct_predictions / len(y_gt)
    return accuracy

# t-SNE reduce the data dimensionality
tsne = TSNE(n_components = 4, method = 'exact',random_state=1)
train_TSNE_data = tsne.fit_transform(train_data)
test_TSNE_data = tsne.fit_transform(test_data)
# print(np.size(test_TSNE_data))
val_TSNE_data = tsne.fit_transform(val_data)
# print(np.size(val_TSNE_data))

k_values = [1, 4, 8, 16]
results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_TSNE_data, train_target)
    y_val_pred = knn.predict(val_TSNE_data)
    accuracy = evaluate(val_target, y_val_pred)
    results.append((k, accuracy))

for k, acc in zip(k_values, results):
    print(f'Accuracy for k={k}: {acc}')

# Choose the best k in [1,4,8,16]
# best_k = sorted(results, key=lambda x: x[1], reverse=True)[0][0]
# print(f'The best k is {best_k}')
knn_optimal = KNeighborsClassifier(n_neighbors=1)
knn_optimal.fit(train_TSNE_data, train_target)
y_test_pred = knn.predict(test_TSNE_data)
# print(np.size(test_target))
# print(np.size(y_test_pred))
test_accuracy = evaluate(test_target, y_test_pred)
print(f'Accuracy for test dataset with k_best: {test_accuracy:.2f}') 

# Apply t-SNE with 2 components for visualization
tsne_2 = TSNE(n_components = 2, method = 'exact')
X_test_tsne_2d = tsne_2.fit_transform(test_data)

# Visualize predicted and actual labels
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_test_tsne_2d[:, 0], X_test_tsne_2d[:, 1], c=y_test_pred, cmap='viridis', alpha=0.5)
plt.colorbar()
plt.title('Predicted Labels')
plt.subplot(1, 2, 2)
plt.scatter(X_test_tsne_2d[:, 0], X_test_tsne_2d[:, 1], c=test_target, cmap='viridis', alpha=0.5)
plt.colorbar()
plt.title('Actual Labels')
plt.show()
