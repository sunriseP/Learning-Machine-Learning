import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1(a)
a = 0.25
b = -5
c = 0.2
M = 20

# Random x
np.random.seed(0) 
x = np.random.uniform(0, 30, M)
X = x.reshape(-1, 1)

# Gaussian noise ε ~ N(0, 4.5)
noise = np.random.normal(0, np.sqrt(4.5), M)

y = a * x**2 + b * x + c + noise

for tx, ty in zip(x, y):
    print(f"x: {tx:.2f}, y: {ty:.2f}")

# Save the data as a csv file
df = pd.DataFrame(data = {
    'x_value': x,
    'y_value': y
})
df.to_csv('./Initial_data.csv', index=False)

# 1(b)(c)
def PolynomialRegression(degree = 2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))
# return class Pipeline

models = {
    "Linear Model": PolynomialRegression(1),
    "Quadratic Model": PolynomialRegression(2),
    "10th Degree Model": PolynomialRegression(10)
}

x_fit = np.linspace(0, 30, 5000).reshape(-1, 1)

predictions = {}
mse_scores = {}
for name, model in models.items():
    model.fit(X, y)
    predictions[name] = model.predict(x_fit)
    # 1(d)
    mse_scores[name] = mean_squared_error(y, model.predict(X))
    print(name, '\'s MSE: ', mse_scores[name], sep='')

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data')

for name, y_pred in predictions.items():
    plt.plot(x_fit.ravel(), y_pred, label=f'{name} Fit')

plt.title('Polynomial Regression Fit Comparison')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()