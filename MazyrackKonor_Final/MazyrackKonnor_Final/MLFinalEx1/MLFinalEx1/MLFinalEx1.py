import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
print(f'Data: {data}')

X = data.data
y = data.target
print(X)
print(y)

# Standardize features (recommended for neural networks)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Define MLPClassifier to match Keras model
model = MLPClassifier(
    hidden_layer_sizes=(160, 160, 160),  # Three hidden layers: 160, 160, 160 units
    activation='relu',                    # ReLU activation
    solver='adam',                       # Adam optimizer
    max_iter=1000,                        # Equivalent to EPOCHS
    batch_size=16,                       # Equivalent to BATCH_SIZE
    alpha=0.0001,                        # L2 regularization to approximate dropout
    random_state=None,                   # No random seed to match Keras
    verbose=True,                        # Print training progress
    early_stopping=False                 # No validation during training to match Keras
)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
pred = model.predict(X_test) 
#accuracy Score
accs = accuracy_score(y_test, pred)
print(f'\nAccuracy score: {accs}')

#classification report
print(f'\nClassification Report:\n{classification_report(y_test, pred)}')

#confusion matrix
confusion_matrix_result = confusion_matrix(y_test, pred)
print(f'Confusion Matrix:\n{confusion_matrix_result}')