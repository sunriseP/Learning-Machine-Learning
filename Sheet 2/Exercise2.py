
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import ax_ellipse

# 2(a)
def mahanalobis_distance(mu, cov, x):
  diff = mu - x
  ret = np.sqrt(np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))
  return ret

# 2(b)
df = pd.read_csv('./data_mdistance.csv')
mean = np.mean(df[['x_1', 'x_2']], axis=0)
cov = np.cov(df['x_1'], df['x_2'])
print("2-(b)")
print(f"The mean: {mean}")
print(f"The covariance matrix: {cov}")

# 2(c)
r = np.corrcoef(df['x_1'], df['x_2'])
print(f"Pearson Correlation Coefficient: {r[0, 1]}")

# 2(d)
df2 = pd.read_csv('./data_mdistance_points.csv')

points = np.array(df2[['px', 'py']])
count = len(points)
md_res = np.zeros(count)
ed_res = np.sqrt(np.power((points[:,0]-mean[0]), 2) + np.power((points[:,1]-mean[1]), 2))
for i in range(len(points)):
  md_res[i] = mahanalobis_distance(mean, cov, points[i])

print(f"Mahanalobis distance of each point: {md_res}")
print(f"Euclidean distance of each point: {ed_res}")

# 2(e)
fig, ax = plt.subplots(figsize=(12, 3))

ax.set_title('Data Plot')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax_ellipse.ax_ellipse(mean, cov, ax, n_std=1, linestyle='solid', edgecolor='red')
ax_ellipse.ax_ellipse(mean, cov, ax, n_std=2, linestyle='dashed', edgecolor='purple')
ax_ellipse.ax_ellipse(mean, cov, ax, n_std=3, linestyle='dotted', edgecolor='blue')


ax.scatter(df['x_1'], df['x_2'], c='black', s=2)
ax.scatter(mean[0], mean[1], c='red', s=10, label='$\mu$')
ax.scatter(df2['px'], df2['py'], c='red', marker='x', label='$x_p$')
ax.legend()

fig.savefig('2(e).png')
print("The answer of 2-(e) is saved to the file \"2(e).png\"\n")