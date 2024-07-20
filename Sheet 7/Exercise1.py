from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target

train_data, test_data, train_target, test_target = train_test_split(X, y, train_size=0.7,random_state=42)

tsne = TSNE(n_components = 4, method = 'exact',random_state=1)
train_TSNE_data = tsne.fit_transform(train_data)
test_TSNE_data = tsne.fit_transform(test_data)

class OneVsRest:
    def __init__(self, K):
        self.classifiers = [LogisticRegression(random_state=k) for k in range(K)]

    def fit(self, X, y):
        for i, clf in enumerate(self.classifiers):
            binary_target = (y == i).astype(int)
            clf.fit(X, binary_target)

    def predict(self, X):
        probabilities = [clf.predict_proba(X)[:, 1] for clf in self.classifiers]
        return np.argmax(probabilities, axis=0)
    
ovr = OneVsRest(K=10)
ovr.fit(train_TSNE_data, train_target)
y_pred = ovr.predict(test_TSNE_data)

def evaluate(y_gt, y_pred):
    correct_predictions = 0
    for true, pred in zip(y_gt, y_pred):
        if true == pred:
            correct_predictions += 1
    accuracy = correct_predictions / len(y_gt)
    return accuracy

accuracy = evaluate(y_pred, test_target)
print(f"Accuracy: {accuracy}")

tsne2 = TSNE(n_components = 2, method = 'exact',random_state=1)
test_TSNE_data = tsne2.fit_transform(test_data)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(test_TSNE_data[:, 0], test_TSNE_data[:, 1], c=y_pred, cmap='viridis', alpha=0.5)
plt.title('Predicted Labels')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.scatter(test_TSNE_data[:, 0], test_TSNE_data[:, 1], c=test_target, cmap='viridis', alpha=0.5)
plt.title('Actual Ground Truth Labels')
plt.colorbar()
plt.show()
