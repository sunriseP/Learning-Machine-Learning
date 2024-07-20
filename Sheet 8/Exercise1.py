import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

dataset = pd.read_pickle("data_exercise_1.pkl")
X1 = dataset.X1
X2 = dataset.X2
X = np.column_stack((X1, X2))
y = dataset.y

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)

cls_tree = tree.DecisionTreeClassifier()
cls_tree = cls_tree.fit(X_train, y_train)

y_pred = cls_tree.predict(X_test)
F_Score = f1_score(y_test, y_pred, average='binary')
print(f'F-Score for test dataset: {F_Score}') 

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Decision Tree Boundary')
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train)
ax[0].set_title("Train Data")
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test)
ax[1].set_title("Test Data")
ax[1].set_xlabel('x1')
ax[1].set_ylabel('x2')

feature_1_tr, feature_2_tr = np.meshgrid(
        np.linspace(X_train[:,0].min(), X_train[:,0].max()),
        np.linspace(X_train[:,1].min(), X_train[:,1].max())
    )
feature_1_te, feature_2_te = np.meshgrid(
        np.linspace(X_test[:,0].min(), X_test[:,0].max()),
        np.linspace(X_test[:,1].min(), X_test[:,1].max())
    )
z1 = cls_tree.predict(np.c_[feature_1_tr.ravel(), feature_2_tr.ravel()])
z1 = z1.reshape(feature_1_tr.shape)
z2 = cls_tree.predict(np.c_[feature_1_te.ravel(), feature_2_te.ravel()])
z2 = z2.reshape(feature_1_te.shape)
display_train = DecisionBoundaryDisplay(xx0=feature_1_tr, xx1=feature_2_tr, response=z1, xlabel = None, ylabel = None)
display_train.plot(ax=ax[0], cmap='viridis', alpha=0.2)
display_test = DecisionBoundaryDisplay(xx0=feature_1_te, xx1=feature_2_te, response=z2, xlabel = None, ylabel = None)
display_test.plot(ax=ax[1], cmap='viridis', alpha=0.2)

plt.show()
