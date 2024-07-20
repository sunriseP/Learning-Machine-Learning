from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 加载数据
digits = load_digits()
data = digits.data
target = digits.target

X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=42)

# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # 这里使用transform而不是fit_transform来确保使用相同的缩放器

# 定义自动编码器
class Autoencoder(nn.Module):
    def __init__(self, feat_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.feat_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# 设置参数
feat_dim = X_train.shape[1]
latent_dim = 2  # 设置较小的latent_dim以便于可视化

# 初始化模型
model = Autoencoder(feat_dim, latent_dim)
print(model)

# 将数据转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 设置损失函数和优化器
MSE_Loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 15  # 增加训练的epochs以便更好地学习数据
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = MSE_Loss(outputs, X_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 提取潜在空间表示
with torch.no_grad():
    latent_space = model.encoder(X_train_tensor).numpy()

# # 使用t-SNE进行可视化
# tsne = TSNE(n_components=2, random_state=42)
# latent_2d = tsne.fit_transform(latent_space)

# 可视化潜在空间
plt.figure(figsize=(8, 6))
scatter = plt.scatter(latent_space[:, 0], latent_space[:, 1], c=y_train, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.title('Latent Space (t-SNE)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()

# 重建测试集图像
with torch.no_grad():
    reconstructions = model(X_test_tensor).numpy()

# 可视化原始图像和重建图像
fig, axes = plt.subplots(3, 2, figsize=(8, 8))
for i in range(3):
    axes[i, 0].imshow(X_test[i].reshape(8, 8), cmap='gray')
    axes[i, 0].set_title('Input Image')
    axes[i, 1].imshow(reconstructions[i].reshape(8, 8), cmap='gray')
    axes[i, 1].set_title('Rec. Image')
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
