import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_pickle('classification_data.pkl')  
print(data)

# Add a column of ones for the bias term
data['bias'] = 1
features = data[['x1', 'x2', 'bias']]  # Adjust these feature names based on your dataset
target = data['label']  # Adjust if your target column is named differently

# Define the fit function to compute the weights using the Least Squares method
def fit(X, t):
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    XTy = X.T.dot(t)
    return XTX_inv.dot(XTy)

# Define the predict function to make predictions
def predict(X, w):
    y = X.dot(w)
    return np.where(y >= 0, 1, -1)

# Split the data into training and testing sets
X_train, X_test, t_train, t_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Compute weights using the training data
w = fit(X_train, t_train)

# Make predictions on the test data
t_pred = predict(X_test, w)

# Visualization
plt.figure(figsize=(12, 6))

# Plot training samples
plt.subplot(1, 2, 1)
plt.scatter(X_train['x1'], X_train['x2'], c=t_train, cmap='viridis', alpha=0.5)
plt.title('Training Samples')
plt.xlabel('x1')
plt.ylabel('x2')

# Plot test samples with predicted labels and decision boundary
plt.subplot(1, 2, 2)
plt.scatter(X_test['x1'], X_test['x2'], c=t_pred, cmap='viridis', alpha=0.5)
x_values = np.linspace(X_test['x1'].min(), X_test['x1'].max(), 100)
y_values = -(w[0] / w[1]) * x_values - (w[2] / w[1])
plt.plot(x_values, y_values, color='blue', label='Decision Boundary')
plt.title('Test Samples and Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')

plt.legend()
plt.tight_layout()
plt.show()
