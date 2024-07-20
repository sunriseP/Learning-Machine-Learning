from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

digits = load_digits()
data = digits.data
target = digits.target

X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=42)

# Normalize the data. Dataset include 16 values, /16.0--value in 0 to 1, *2--value in 0 to 2, -1--value in -1 to 1
# X_train = (X_train / 16.0) * 2 - 1
# X_test = (X_test / 16.0) * 2 - 1

scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

class Autoencode(nn.Module):
    def __init__(self, feat_dim, latent_dim):
        super(Autoencode, self).__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.output_dim = feat_dim

        # Activate func
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.encode_linear = nn.Linear(self.feat_dim, self.latent_dim)
        self.decode_linear = nn.Linear(self.latent_dim, self.output_dim)

    def forward(self, x):
        encode_linear = self.encode_linear(x)
        encode_out = self.relu(encode_linear)

        decode_linear = self.decode_linear(encode_out)
        decode_out = self.sigmoid(decode_linear)
        return decode_out

feat_dim = X_train.shape[1]
latent_dim = 100000

model = Autoencode(feat_dim, latent_dim)
print(model)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

MSE_Loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = MSE_Loss(outputs, X_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    latent_space = model.encode_linear(X_train_tensor).numpy()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(latent_space[:, 0], latent_space[:, 1], c=y_train, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.title('Latent Space')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

with torch.no_grad():
    reconstructions = model(X_test_tensor).numpy()

fig, axes = plt.subplots(3, 2, figsize=(8, 8))
for i in range(3):
    axes[i, 0].imshow(X_test[i].reshape(8, 8))
    axes[i, 0].set_title('Input Image')
    axes[i, 1].imshow(reconstructions[i].reshape(8, 8))
    axes[i, 1].set_title('Rec. Image')
for ax in axes.flat:
    ax.set(xlabel='x1', ylabel='x2')
    ax.xaxis.set_ticks([0,5])
plt.tight_layout()
plt.show()