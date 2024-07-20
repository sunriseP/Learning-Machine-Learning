import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to compute the conditional probability p_j|m
def compute_p_j_given_m(data, sigma, m):
    squared_diff = np.sum((data[m] - data)**2, axis=1)
    p_j_m = np.exp(-squared_diff / (2 * sigma[m]**2))
    p_j_m[m] = 0
    sum_p_j_m = np.sum(p_j_m)
    if sum_p_j_m == 0:
        return p_j_m
    p_j_m /= sum_p_j_m
    return p_j_m

# Function to calculate the entropy of Pm and the perplexity
def compute_perplexity(p_j_m):
    H = -np.sum(p_j_m * np.log2(p_j_m + 1e-10))
    return 2**H # 2 to the power of H

# Function to perform binary search to find the sigma that achieves the target perplexity
def binary_search_perplexity(data, target_perplexity, m, tol=0.001, max_iter=100):
    lmin, lmax = 0.15, 50
    sigma = 1 
    for i in range(max_iter):
        sigma = (lmin + lmax) / 2
        p_j_m = compute_p_j_given_m(data, np.full(data.shape[0], sigma), m)
        perplexity = compute_perplexity(p_j_m)
        if abs(perplexity - target_perplexity) < tol:
            break
        if perplexity > target_perplexity:
            lmax = sigma
        else:
            lmin = sigma
    return sigma

# Load data
data = pd.read_pickle('tsne_data.pkl').values

# Compute sigma values for each point to reach the target perplexity
target_perplexity = 10
sigmas = np.array([binary_search_perplexity(data, target_perplexity, i) for i in range(data.shape[0])])

# Display the sigmas for each point
print("Sigmas for each data point:", sigmas)

# Visualize data
plt.scatter(data[:, 0], data[:, 1], c='blue')
plt.title('Original Data Points')
plt.show()
