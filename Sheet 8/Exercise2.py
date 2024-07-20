import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import mode
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

dataset = pd.read_pickle("data_exercise_2.pkl")
X1 = dataset.X1
X2 = dataset.X2
X = np.column_stack((X1, X2))
y = dataset.y
df = pd.DataFrame({'x1': X1, 'x2': X2, 'y': y})
trees = []
for _ in range(30):
    df_resampled = df.sample(frac=0.01, replace=True, random_state=np.random.randint(0, 10000))
    # frac=0.01-with the train subset accounting for only 1% of the total data
    X_resampled = df_resampled[['x1', 'x2']].values
    y_resampled = df_resampled['y'].values
    tree = DecisionTreeClassifier()
    tree.fit(X_resampled, y_resampled)
    trees.append(tree)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)

# Majority voting-'mode' function return the mode and frequency
predictions = np.array([tree.predict(X_test) for tree in trees])
final_predictions = mode(predictions, axis=0)[0][0] # first[0]-array, second[0]-mode

# Cross-Entropy loss based on probability predictions
prob_predictions = np.mean([tree.predict_proba(X_test) for tree in trees], axis=0)
cross_entropy_loss = log_loss(y_test, prob_predictions)
print(f'Cross-Entropy Loss: {cross_entropy_loss}')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Ensemble Method Prediction')

ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
ax[0].set_title('Train Data')
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')

test_prob = np.max(prob_predictions, axis=1)
ax[1].scatter(X_test[:, 0], X_test[:, 1], c=test_prob, cmap='viridis')
ax[1].set_title('Test Data')
ax[1].set_xlabel('x1')
ax[1].set_ylabel('x2')

plt.show()