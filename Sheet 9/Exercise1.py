import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# np.random.seed(42)

data = pd.read_pickle('data_exercise_mlp.pkl')
print(data)

X1 = data.x_1
X2 = data.x_2
X = np.column_stack((X1, X2))
y = data.y.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

class NeuralNetwork():
    def __init__(self, feat_dim, hidden_dim, out, lr):
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.out = out
        self.lr = lr

        self.W1 = np.random.uniform(-1, 1, (feat_dim, hidden_dim))
        self.b1 = np.random.uniform(-1, 1, (1, hidden_dim))
        self.W2 = np.random.uniform(-1, 1, (hidden_dim, out))
        self.b2 = np.random.uniform(-1, 1, (1, out))
    
    def sigmoid(self, x): # activate function
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        # Z = Wx + b
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        y = self.sigmoid(self.Z2)
        self.X = X
        return y
    
    def backward(self, loss):
        dZ2 = loss
        m = X.shape[0]
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.A1 * (1 - self.A1)
        dW1 = np.dot(self.X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return gradients
    
    def update_weights(self, gradients):
        self.W1 -= self.lr * gradients["dW1"]
        self.b1 -= self.lr * gradients["db1"]
        self.W2 -= self.lr * gradients["dW2"]
        self.b2 -= self.lr * gradients["db2"]
        print(self.W1,self.b1,self.W2,self.b2)
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return loss

input_dim = X_train.shape[1]
hidden_dim = 5
output_dim = 1
lr = 0.0001

nn = NeuralNetwork(input_dim, hidden_dim, output_dim, lr)

losses = []
epochs = 100
for epoch in range(epochs):
    output = nn.forward(X_train)
    loss = nn.compute_loss(y_train, output)
    losses.append(loss)
    dZ2 = output - y_train
    gradients = nn.backward(dZ2)
    nn.update_weights(gradients)

plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

output_test = nn.forward(X_test)
y_pred = (output_test > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
print(f'Confusion matrix for test dataset: {cm}') 

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Ground truth
axs[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), cmap='viridis', s=50)
axs[0].set_title('Ground Truth')
axs[0].set_xlabel('$x_1$')
axs[0].set_ylabel('$x_2$')
axs[0].legend(*axs[0].collections[0].legend_elements())

# Predictions
axs[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred.ravel(), cmap='viridis', s=50)
axs[1].set_title('Predictions')
axs[1].set_xlabel('$x_1$')
axs[1].set_ylabel('$x_2$')
axs[1].legend(*axs[1].collections[0].legend_elements())

plt.tight_layout()
plt.show()

train_output = nn.forward(X_train)
train_accuracy = np.mean((train_output > 0.5) == y_train)
test_accuracy = np.mean(y_pred == y_test)

print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

if train_accuracy > test_accuracy:
    print("The model might be overfitting.")
elif train_accuracy < test_accuracy:
    print("The model might be underfitting.")
else:
    print("The model seems to generalize well.")