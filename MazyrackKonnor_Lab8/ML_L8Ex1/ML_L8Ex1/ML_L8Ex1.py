import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Generate a 2D dataset that isn't linearly separable
#X, y = make_circles(n_samples=200, factor=0.1, noise=0.1, random_state=42)
df = pd.read_csv('speedLimits.csv')

X = df['Speed'].values.reshape(-1,1)
y = df['Ticket']
y_color = y.map({'T': 'red', 'NT': 'green'})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
classifiers = {}

# Train SVM models with different kernels
for kernel in kernels:
    clf = SVC(kernel=kernel, C=1.0)
    clf.fit(X_train_scaled, y_train)
    classifiers[kernel] = clf

# Evaluate the models
accuracies = {}
for kernel, clf in classifiers.items():
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[kernel] = accuracy
    print(f"Accuracy with {kernel} kernel: {accuracy:.4f}")

plt.scatter(X, y, c=y_color)
plt.xlabel("Speed")
plt.ylabel("Ticket?")
plt.title("Speed vs. Ticket")
plt.show()