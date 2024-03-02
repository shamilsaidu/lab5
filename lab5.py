# A1: Confusion matrix and performance metrics for classification problem
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Assuming you have a classification model named 'classifier' and X_train, y_train, X_test, y_test

data, meta = arff.loadarff("3sources_bbc1000.arff")
dataset = np.array(data.tolist(), dtype=float)
classifier = KNeighborsClassifier(n_neighbors=3)

# Replace 'classifier' with your actual model
X = dataset[:, :-1]
y = dataset[:, -1]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Fit the model
classifier.fit(X_train, y_train)

# Predictions on training set
y_train_pred = classifier.predict(X_train)

# Predictions on test set
y_test_pred = classifier.predict(X_test)

# Confusion matrix
confusion_mat_train = confusion_matrix(y_train, y_train_pred)
confusion_mat_test = confusion_matrix(y_test, y_test_pred)

# Performance metrics
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

print("Confusion Matrix (Training):")
print(confusion_mat_train)
print("Precision (Training):", precision_train)
print("Recall (Training):", recall_train)
print("F1-Score (Training):", f1_train)

print("\nConfusion Matrix (Test):")
print(confusion_mat_test)
print("Precision (Test):", precision_test)
print("Recall (Test):", recall_test)
print("F1-Score (Test):", f1_test)

# Inference about the model's learning outcome: Observe if precision, recall, and F1-Score are consistent between training and test sets.
# If training metrics are significantly better than test metrics, it might be overfitting. If both are low, it might be underfitting.

