import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

data = pd.read_csv('lenses.csv', header=None)

X = data.iloc[:, 1:5]
y = data.iloc[:, 5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred = dt.predict(X_test)
print ('Accuracy Score: ', accuracy_score(y_test, pred))
print ('\nClassification Report:\n', classification_report(y_test, pred))
conf_matrix = confusion_matrix(y_test, pred)
print('\nConfusion Matrix\n', conf_matrix)

#print Confusion Matrix visualization (test_size=0.20, random_state=0)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
xticklabels=['Hard', 'Soft', 'None'], yticklabels=['Hard', 'Soft', 'None'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(f'Important Features: {dt.feature_importances_}')

#visualization of the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=['Age', 'Prescription', 'Astigmatic', 'Tear Rate'], class_names=['Hard',
'Soft','None'], filled=True, rounded=True)
plt.show()
