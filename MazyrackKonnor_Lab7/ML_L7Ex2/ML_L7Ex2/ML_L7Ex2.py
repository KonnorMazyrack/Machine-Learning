import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn import preprocessing

df = pd.read_csv('golf.csv')

#convert to numerals
le = preprocessing.LabelEncoder()
weather_encoded=np.array(le.fit_transform(df.loc[:, 'Outlook']))
temp_encoded=np.array(le.fit_transform(df.loc[:, 'Temp']))
humidity_encoded=np.array(le.fit_transform(df.loc[:, 'Humidity']))
windy_encoded=np.array(le.fit_transform(df.loc[:, 'Windy']))
play_encoded=np.array(le.fit_transform(df.loc[:, 'PlayGolf']))

print(f'outlook: {weather_encoded}')
print(f'temp: {temp_encoded}')
print(f'humidity: {humidity_encoded}')
print(f'windy: {windy_encoded}')
df = pd.DataFrame({'Outlook' : weather_encoded, 'Temp' : temp_encoded, 'Humidity' : humidity_encoded, 'Windy' : windy_encoded, 'PlayGolf' : play_encoded})

X = np.array(df.loc[: , :'Windy'])
y =np.array(df.loc[:, 'PlayGolf'])
print(f'X: {X}')
print(f'y: {y}')

y2, levels = pd.factorize(df.iloc[:, 4])
print(f'y2: {y2}')
print(f'levels: {levels}')
#print(X)
#print(y)

#print(model.coef_)
#print(f'Coef:\n{model.coef_}')
#print(f'Intercept:\n{model.intercept_}')
datapoint = np.array([['Rainy', 'Hot', 'High', 'True'],['Sunny', 'Mild', 'Normal', 'False'],['Sunny', 'Cool', 'High', 'False']])

dpweather_encoded=np.array(le.fit_transform(datapoint[:, 0]))
dptemp_encoded=np.array(le.fit_transform(datapoint[:, 1]))
dphumidity_encoded=np.array(le.fit_transform(datapoint[:, 2]))
dpwindy_encoded=np.array(le.fit_transform(datapoint[:, 3]))

dp = pd.DataFrame({'Outlook' : dpweather_encoded, 'Temp' : dptemp_encoded, 'Humidity' : dphumidity_encoded, 'Windy' : dpwindy_encoded})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model = GaussianNB()

model.fit(X_train, y_train)

predicted= model.predict(dp.values)
print ("Predicted Value:", predicted)

pred = model.predict(X_test)

print(classification_report(y_test, pred))



