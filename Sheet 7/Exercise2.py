import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Exercise1 import train_data, test_data, train_target, test_target

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

class SoftmaxClassifier(nn.Module):
    def __init__(self, n_feat, n_classes):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(n_feat, n_classes)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)

def cross_entropy(y_pred, y_true):
    return -(y_true * torch.log(y_pred+1e-9)).sum(dim=1).mean()

# Assuming X_train and train_target are already defined from the previous exercise
num_classes = len(np.unique(train_target))  # Determine the number of unique classes
y_train_encoded = one_hot_encode(train_target, num_classes)
y_test_encoded = one_hot_encode(test_target, num_classes)

# Convert to PyTorch tensors
y_train_torch = torch.from_numpy(y_train_encoded).float()
y_test_torch = torch.from_numpy(y_test_encoded).float()
X_train_torch = torch.from_numpy(train_data).float()
X_test_torch = torch.from_numpy(test_data).float()

# Model initialization
n_features = train_data.shape[1]
model = SoftmaxClassifier(n_feat=n_features, n_classes=num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training
Loss = []
epochs = 100
len_traindata = len(train_data)

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(len_traindata):
        x = X_train_torch[i].unsqueeze(0)  # Add batch dimension
        y = y_train_torch[i].unsqueeze(0)  

        optimizer.zero_grad()  # Clear gradients
        y_pred = model(x)  # Forward pass
        loss = cross_entropy(y_pred, y)  
        loss.backward()  
        optimizer.step()  # Update weights
        epoch_loss += loss.item() * x.size(0)
        
    average_epoch_loss = epoch_loss / len_traindata  
    Loss.append(average_epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {average_epoch_loss}")

plt.plot(range(epochs), Loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()

# Model evaluation
model.eval()
test_loss = 0
len_testdata = len(test_data)

with torch.no_grad(): 
    for i in range(len_testdata):
        x = X_test_torch[i].unsqueeze(0)
        y = y_test_torch[i].unsqueeze(0)
        y_pred = model(x)
        test_loss += cross_entropy(y_pred, y).item()

average_loss = test_loss / len(test_data)
print(f"Average cross entropy loss.: {average_loss}")

