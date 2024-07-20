import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_pickle("gini_imp.pkl")
X1 = dataset.X1
X2 = dataset.X2
X = np.column_stack((X1, X2))
y = dataset.y

def gini_impurity(y):
    classes = np.unique(y)
    impurity = 1.0
    for cls in classes:
        p_cls = len(y[y == cls]) / len(y)
        impurity -= p_cls ** 2
    return impurity

def split_data(X, y, xdiv):
    # Boolean Mask
    set1_mask = X[:, 0] <= xdiv
    set2_mask = X[:, 0] > xdiv
    X_set1, y_set1 = X[set1_mask], y[set1_mask]
    X_set2, y_set2 = X[set2_mask], y[set2_mask]
    return X_set1, y_set1, X_set2, y_set2

def split_gini_impurity(X, y, xdiv):
    X_set1, y_set1, X_set2, y_set2 = split_data(X, y, xdiv)
    n = len(y)
    gini_set1 = gini_impurity(y_set1)
    gini_set2 = gini_impurity(y_set2)
    weighted_gini = (len(y_set1) / n) * gini_set1 + (len(y_set2) / n) * gini_set2
    return weighted_gini

xdivs = [-5, -2, 0, 1, 10]
impurities = [split_gini_impurity(X, y, xdiv) for xdiv in xdivs]

fig, axs = plt.subplots(1, 5, figsize=(20, 5))

for i, xdiv in enumerate(xdivs):
    X_set1, y_set1, X_set2, y_set2 = split_data(X, y, xdiv)
    axs[i].scatter(X_set1[:, 0], X_set1[:, 1], c=y_set1, cmap='coolwarm', alpha=1, edgecolor='k')
    axs[i].scatter(X_set2[:, 0], X_set2[:, 1], c=y_set2, cmap='coolwarm', alpha=1, edgecolor='k')
    axs[i].axvline(x=xdiv, color='red', linestyle='--')
    print(f'Gini Impurity for xdiv={xdiv} = {impurities[i]}')
    axs[i].set_title(f'Impurity = {impurities[i]:.3f}')
    axs[i].set_xlabel('x1')
    axs[i].set_ylabel('x2')

plt.suptitle('Gini Impurities')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
