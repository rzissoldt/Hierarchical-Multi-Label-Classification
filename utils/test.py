import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

# Example true and predicted labels (replace with your data)
y_true = np.array([[1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 0]])
y_pred = np.array([[1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]])

# Compute multilabel confusion matrix
mlb = MultiLabelBinarizer()
mlb.fit(y_true)
y_true_binary = mlb.transform(y_true)
y_pred_binary = mlb.transform(y_pred)
conf_matrix = multilabel_confusion_matrix(y_true_binary, y_pred_binary)

# Combine confusion matrices for all classes
combined_conf_matrix = np.sum(conf_matrix, axis=0)

# Plot combined confusion matrix
labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
labels = np.array(labels).reshape(2, 2)
plt.imshow(combined_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Combined Confusion Matrix')
plt.xticks([0, 1], ['Predicted False', 'Predicted True'])
plt.yticks([0, 1], ['Actual False', 'Actual True'])

for j in range(2):
    for k in range(2):
        plt.text(k, j, str(labels[j][k]) + " = " + str(combined_conf_matrix[j][k]), ha='center', va='center', color='red')

plt.colorbar()
plt.tight_layout()
plt.show()
