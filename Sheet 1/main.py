import pandas as pd
import matplotlib.pyplot as plt

#***********Exercise:Handing of Tabular Data

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

#***********Exercise:Plots using matplotlib

# 加载数据
df = pd.read_csv('./patients.csv')

# 绘制直方图
#plt.hist(df['Height_cm'], bins=len(df['Height_cm'].unique()))
plt.hist(df['Height_cm'], bins=32)
plt.xticks(range(157,191))
plt.show()



