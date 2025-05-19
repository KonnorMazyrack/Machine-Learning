import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

data = pd.read_csv('balloons_extended.csv')

data['Color'] = data['Color'].map({'YELLOW': 0, 'PURPLE': 1})
data['size'] = data['size'].map({'SMALL': 0, 'LARGE': 1})
data['act'] = data['act'].map({'STRETCH': 0, 'DIP': 1})
data['age'] = data['age'].map({'ADULT': 0, 'CHILD': 1})
data['inflated'] = data['inflated'].map({'T': 1, 'F': 0})

X = data[['Color', 'size', 'act', 'age']]
y = data['inflated']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

RF = RandomForestClassifier(n_estimators=100) #default is 200
RF.fit(X_train, y_train)
pred = RF.predict(X_test)
print ('Accuracy Score: ', accuracy_score(y_test, pred))
print ('\nClassification Report:\n', classification_report(y_test, pred))
conf_matrix = confusion_matrix(y_test, pred)
print('\nConfusion Matrix\n', conf_matrix)
print(f'\nImportant Features:\n {RF.feature_importances_}')

#print Confusion Matrix visualization (test_size=0.20, random_state=0)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
xticklabels=['Not Inflated', 'Inflated'], yticklabels=['Not Inflated', 'Inflated'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

