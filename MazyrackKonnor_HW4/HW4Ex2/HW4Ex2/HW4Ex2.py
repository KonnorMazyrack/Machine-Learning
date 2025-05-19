from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('avgHigh_jan_1895-2018.csv')

x = data[['Date']].values
y = data[['Value']].values

testSize = float(input("Enter text size: "))


index = int(len(data) * (1-testSize))

train_data = data.iloc[:index]
test_data = data.iloc[index:]

X_train = train_data[['Date']].values  
y_train = train_data['Value'].values

X_test = test_data[['Date']].values
y_test = test_data['Value'].values

reg = LinearRegression().fit(X_train, y_train)

slope, intercept, r, p, std_error = stats.linregress(X_train.flatten(), y_train.flatten())
print(f'\nSlope: {slope}, y-intercept: {intercept}, correlation: {r}, p-value: {p}, standard error: {std_error}\n')
y_pred = reg.predict(X_test)
for i in range(len(y_pred)):
    print(f'Actual: {y_test[i]}, Predicted: {y_pred[i]}')

x2 = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1,1)
y2= reg.predict(x2)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRoot Mean Square Error: {rmse}")

plt.scatter(x, y, c='blue', label='Train')
plt.scatter(X_test, y_test, c='green', label='Test')
plt.plot(x2, y2, color='red', label='Model')
plt.title(f'Slope: {slope:.2f}, Intercept: {intercept:.2f}, Test size: {testSize} ({len(data)-index}/{len(data)}), RSME: {rmse:.2f}')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()
plt.show()

