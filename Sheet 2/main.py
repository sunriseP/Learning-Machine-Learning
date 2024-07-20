import pandas as pd
import matplotlib.pyplot as plt


############1. Exercise: Shannon Entropy
#(a)
data_eb = pd.read_csv('./data_entropy_binary.csv')
print(data_eb)

x11 = data_eb[data_eb.x_label == 0]
x12 = data_eb[data_eb.x_label == 1]
x_labels = data_eb['x_label']

y11 = data_eb[data_eb.y_label == 0]
y12 = data_eb[data_eb.y_label == 1]
y_labels = data_eb['y_label']

# x_colors = ['red' if label == 0 else 'blue' for label in x_labels]
# y_colors = ['red' if label == 0 else 'blue' for label in y_labels]

fig_x = plt.figure(figsize=(4, 3))
plt.scatter(x11['x_1'], x11['x_2'], c='r', marker='o', label = '0')
plt.scatter(x12['x_1'], x12['x_2'], c='b', marker='o', label = '1')
plt.legend()

plt.title('Random Variable X')
plt.xlabel('x1')
plt.ylabel('x2')

fig_y = plt.figure(figsize=(4, 3))
plt.scatter(y11['y_1'], y11['y_2'], c='r', marker='o', label = '0')
plt.scatter(y12['y_1'], y12['y_2'], c='b', marker='o', label = '1')
plt.legend()

plt.title('Random Variable Y')
plt.xlabel('y1')
plt.ylabel('y2')

plt.show()

#(b)
def compute_probabilities(labels):
    probabilities = {}
    total_count = len(labels)
    
    # 计算每个标签的出现次数
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
        print(label_counts[label])
    
    # 计算每个标签的概率
    for label, count in label_counts.items():
        probabilities[label] = count / total_count
    
    return probabilities

x_probabilities = compute_probabilities(x_labels)
y_probabilities = compute_probabilities(y_labels)
print("X_Label Probabilities:", x_probabilities)
print("Y_Label Probabilities:", y_probabilities)
