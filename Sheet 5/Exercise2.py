import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gkernel(d, h):
    return (1 / (h * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((d / h) ** 2))

def weight_m(x, X, h):
    distances = np.array([x - xm for xm in X])
    kernels = np.array([gkernel(d, h) for d in distances])
    weights = kernels / np.sum(kernels)
    return weights

def nadaraya_watson(X, Y, x_points, h):
    y_estimates = []
    for x in x_points:
        weights = weight_m(x, X, h)
        y_estimate = np.sum(weights * Y)
        y_estimates.append(y_estimate)
    return y_estimates

# Data setup (replace with actual pandas data frame loading if needed)
data = {
    'DayID': [11, 22, 33, 44, 50, 56, 67, 70, 78, 89, 90, 100],
    'Price': [2337, 2750, 2301, 2500, 1700, 2100, 1100, 1750, 1000, 1642, 2000, 1932]
}
df = pd.DataFrame(data)

# Application of Nadaraya-Watson Estimator
X = df['DayID'].values
Y = df['Price'].values
x_points = np.linspace(1, 110, 500)
y_estimates = nadaraya_watson(X, Y, x_points, h=10)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(x_points, y_estimates, color='red', label='Nadaraya-Watson Estimate')
plt.xlabel('DayID')
plt.ylabel('Price')
plt.title('Nadaraya-Watson Price Estimation')
plt.legend()
plt.show()
