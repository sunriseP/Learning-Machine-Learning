import matplotlib.pyplot as plt
import pandas as pd

# 先初始化一下输入的数据
df = pd.read_csv('patients.csv')

# 初始化图
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 4))

# 设定每个子图的纵横比
ax1.set_aspect(1)
ax2.set_aspect(1)

# 设定图的标题
fig.suptitle("Correlation Plot")

# 设定轴的名称
ax1.set_xlabel('Age')
ax1.set_ylabel('Height')
ax2.set_xlabel('Age')
ax2.set_ylabel('Weight')

# 设定轴的范围
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

size = 60

ax1.scatter(df['Age_norm'], df['Height_cm_norm'], s=size, c='blue', edgecolors='black', linewidth=2)
ax2.scatter(df['Age_norm'], df['Weight_kg_norm'], s=size, c='blue', edgecolors='black', linewidth=2)

fig.show()

