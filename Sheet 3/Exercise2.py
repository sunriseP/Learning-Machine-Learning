import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
data = digits.data
target = digits.target

fig, axes = plt.subplots(2, 5, figsize=(10, 5)) # initialize the figure
for i, ax in enumerate(axes.flatten()):
    ax.imshow(data[target == i][0].reshape(8, 8), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Digit: {i}')
plt.show()

M, N = data.shape
print(f"Number of samples (M): {M}")
print(f"Number of features (N): {N}")

# PCA
mean_vec = np.mean(data, axis=0)

centered_data = data - mean_vec

cov_matrix = np.cov(centered_data, rowvar=False)# covariance

# Perform eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)# linear algebra, eigenvalues and eigenvectors

# Sort the eigenvectors by decreasing eigenvalues
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_index]
sorted_eigenvalues = eigenvalues[sorted_index]

# Select the top K eigenvectors
K = 2
selected_vectors = sorted_eigenvectors[:, :K]

# Project the data
projected_data = np.dot(centered_data, selected_vectors)# Multiplication of matrices

# Plot the projected data
plt.figure(figsize=(8, 6))
scatter = plt.scatter(projected_data[:, 0], projected_data[:, 1], c=target, cmap='viridis', edgecolor='k', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Principal Components of Digits(K=2)')
plt.show()
