import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

#***********Exercise:Handing of Tabular Data***********

# 创建一个DataFrame来表示给定的数据
data = {
    'Patient ID': ['000345', '000124', '001758', '000994', '001233', '001145', '000222'],
    'Age': [45, 60, 22, 38, 36, 77, 65],
    'Height_cm': [167, 181, 158, 185, 164, 190, 180],
    'Weight_kg': [67, 78, 57, 90, 72, 75, 110]
}

df = pd.DataFrame(data)

# 用于添加新患者的字典
new_patient = pd.DataFrame({
    'Patient ID': ['001122'],
    'Age': [51],
    'Height_cm': [177],
    'Weight_kg': [81]
})

# 使用concat来添加新患者信息
df = pd.concat([df, new_patient], ignore_index=True)

# 展示数据
print(df)

# 规范化函数
def func_norm(v):
    return (v - v.min()) / (v.max() - v.min())

# 应用规范化函数到所有数值列
df_norm = df.iloc[:, 1:].apply(func_norm)

# 将规范化的数据添加为新的列
df = pd.concat([df, df_norm.add_suffix('_norm')], axis=1)
print(df)

# 保存DataFrame到CSV
df.to_csv('./patients.csv', index=False)

# 筛选年轻患者
youngest_patients = df[df['Age'] < df['Age'].mean()]
# 保存到CSV
youngest_patients.to_csv('./youngest_patients.csv', index=False)

#***********Exercise:Plots using matplotlib***********

# 加载数据
df = pd.read_csv('./patients.csv')

# 绘制直方图
#plt.hist(df['Height_cm'], bins=len(df['Height_cm'].unique()))
plt.hist(df['Height_cm'], bins=32)
plt.xticks(range(157,191))
plt.xlabel('Height [kg]')
plt.ylabel('N')
plt.title('Histogram')

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

ax1.scatter(df['Age_norm'], df['Height_cm_norm'], s=size, c='red', edgecolors='black', linewidth=2)
ax2.scatter(df['Age_norm'], df['Weight_kg_norm'], s=size, c='blue', edgecolors='black', linewidth=2)



plt.show()


ages = df['Age']
heights = df['Height_cm']
weights = df['Weight_kg']

d1 = ages - np.mean(ages)
d2 = heights - np.mean(heights)
d3 = weights - np.mean(weights)

# 计算Pearson correlation coefficients
fz = np.sum(d1 * d2)
fm = np.sqrt( np.sum(d1**2) * np.sum(d2**2) )
r_ah = fz / fm

fz = np.sum(d1 * d3)
fm = np.sqrt( np.sum(d1**2) * np.sum(d3**2) )
r_aw = fz / fm

print(f'Age and Height: {r_ah}')
print(f'Age and Weight: {r_aw}')

#***********Exercise: Similarity Metrics***********
def func_L1norm(v1, v2):
    return np.sum(np.abs(v1-v2))

def func_L2norm(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

def func_cosine_sim(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return np.dot(v1.T, v2) / (norm_v1 * norm_v2)

v1 = np.array([0.2, 0.1, 0.4, -0.4])
v2 = np.array([-0.1, -0.1, 0.8, 0.5])
print(f'res: {func_cosine_sim(v1, v2)}')