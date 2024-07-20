import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 1-(a)

df = pd.read_csv('./data_entropy_binary.csv')

# filter data
df_xl0 = df[df.x_label == 0]
df_xl1 = df[df.x_label == 1]
df_yl0 = df[df.y_label == 0]
df_yl1 = df[df.y_label == 1]
x_labels = df['x_label']
y_labels = df['y_label']

# initialize plot
fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

# set some base info of the chart
ax1.set_title("Random Variable X")
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax2.set_title("Random Variable Y")
ax2.set_xlabel('$y_1$')
ax2.set_ylabel('$y_2$')

# plot the scatterplot
ax1.scatter(df_xl0['x_1'], df_xl0['x_2'], c = 'red', label = '0')
ax1.scatter(df_xl1['x_1'], df_xl1['x_2'], c = 'blue', label = '1')
ax2.scatter(df_yl0['y_1'], df_yl0['y_2'], c = 'red', label = '0')
ax2.scatter(df_yl1['y_1'], df_yl1['y_2'], c = 'blue', label = '1')

# add the legend
ax1.legend()
ax2.legend()

fix.savefig('1(a).png')
print("The answer of 1-(a) is saved to the file \"1(a).png\"\n")


# 1-(b)

def compute_probabilities(labels):
    probabilities = {}
    total_count = len(labels)

    # Count the number of each label
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # calculate the probabilities of each label
    for label, count in label_counts.items():
        probabilities[label] = count / total_count

    return probabilities

x_probabilities = compute_probabilities(x_labels)
y_probabilities = compute_probabilities(y_labels)
print("1-(b)")
print("X_Label Probabilities:", x_probabilities)
print("Y_Label Probabilities:", y_probabilities)


# 1-(c)

def compute_entropy(x):
    values = np.array(list(compute_probabilities(x).values()))
    # print(values)
    return -sum(values * np.log(values))


# 1-(d)

print("1-(d)")
print(f"Entropy for X: {compute_entropy(df['x_label'])}")
print(f"Maximal entropy for X: {np.log(2)}")
print(f"Entropy for Y: {compute_entropy(df['y_label'])}")
print(f"Maximal entropy for Y: {np.log(2)}")
print()


# 1-(e)

df2 = pd.read_csv('data_entropy_mutli.csv')

df_xls = []
df_yls = []
# filter data for each labels
for i in range(4):
    df_xls.append(df2[df2.x_label == i])
    df_yls.append(df2[df2.y_label == i])

label_colors = ['red', 'blue', 'green', 'purple']

fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

# set some base info of the chart
ax1.set_title("Random Variable X")
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax2.set_title("Random Variable Y")
ax2.set_xlabel('$y_1$')
ax2.set_ylabel('$y_2$')

# plot the scatterplot
for i in range(4):
    ax1.scatter(df_xls[i]['x_1'], df_xls[i]['x_2'], c = label_colors[i], label = str(i))
    ax2.scatter(df_yls[i]['y_1'], df_yls[i]['y_2'], c = label_colors[i], label = str(i))

# add the legend
ax1.legend()
ax2.legend()

fix.savefig('1(e).png')
print("The answer of 1-(e) is saved to the file \"1(e).png\"\n")

# 1-(f)

print("1-(f)")
print(f"Entropy for X: {compute_entropy(df2['x_label'])}")
print(f"Maximal entropy for X: {np.log(4)}")
print(f"Entropy for Y: {compute_entropy(df2['y_label'])}")
print(f"Maximal entropy for Y: {np.log(4)}")