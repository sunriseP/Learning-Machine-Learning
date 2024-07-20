import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

y1 = np.array([0.1, 0.8, 0.62, 0.3, 0.45, 0.2, 0.2, 0.9, 0.6, 0.2, 0.1, 0.8])
y2 = np.array([0.2, 0.9, 0.45, 0.9, 0.1, 0.2, 0.55, 0.85, 0.15, 0.1, 0.3, 0.7])
y_true = np.array([0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1])

# False Positive Rate
fpr_y1 = [0.0, 0.0, 0.0, 0.0, 0.333, 0.333, 0.667, 1.0]
fpr_y2 = [0.0, 0.0, 0.0, 0.167, 0.167, 0.333, 0.5, 0.667, 1.0]
# True Positive Rate
tpr_y1 = [0.0, 0.167, 0.5, 0.667, 0.667, 0.833, 1.0, 1.0]
tpr_y2 = [0.0, 0.333, 0.667, 0.667, 0.833, 0.833, 1.0, 1.0, 1.0]
# Area Under the Curve
auc_y1 = roc_auc_score(y_true, y1)
auc_y2 = roc_auc_score(y_true, y2)

# Plotting the ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_y1, tpr_y1, label=f'Model f1 (AUC = {auc_y1:.3f})', color='blue')
plt.plot(fpr_y2, tpr_y2, label=f'Model f2 (AUC = {auc_y2:.3f})', color='green')
plt.plot([0, 1], [0, 1], 'r--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Models f1 and f2')
plt.legend()
plt.grid(True)
plt.show()

print(f"AUC for model f1: {auc_y1:.3f}")
print(f"AUC for model f2: {auc_y2:.3f}")

# Binarize predictions
y1_pred = (y1 >= 0.5).astype(int)
y2_pred = (y2 >= 0.5).astype(int)
print(f"Prediction of y1: {y1_pred}")
print(f"Prediction of y2: {y2_pred}")

# Generate confusion matrices
cm_y1 = confusion_matrix(y_true, y1_pred)
cm_y2 = confusion_matrix(y_true, y2_pred)

# Calculate metrics
metrics_y1 = precision_recall_fscore_support(y_true, y1_pred, average='binary')
metrics_y2 = precision_recall_fscore_support(y_true, y2_pred, average='binary')

# Extract sensitivity, specificity, and F1-score
sensitivity_y1, specificity_y1 = metrics_y1[1], cm_y1[0,0] / cm_y1[0,:].sum()
f1_score_y1 = metrics_y1[2]

sensitivity_y2, specificity_y2 = metrics_y2[1], cm_y2[0,0] / cm_y2[0,:].sum()
f1_score_y2 = metrics_y2[2]

# Output the results
print(f"Confusion matrices of y1: {cm_y1}")
print(f"Confusion matrices of y2: {cm_y2}")

print(f"Sensitivity of y1: {sensitivity_y1:.3f}")
print(f"Sensitivity of y2: {sensitivity_y2:.3f}")

print(f"Specificity of y1: {specificity_y1:.3f}")
print(f"Specificity of y2: {specificity_y2:.3f}")

print(f"F1-score of y1: {f1_score_y1:.3f}")
print(f"F1-score of y2: {f1_score_y2:.3f}")
