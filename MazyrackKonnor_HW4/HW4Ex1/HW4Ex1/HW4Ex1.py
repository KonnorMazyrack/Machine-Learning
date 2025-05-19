from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('avgHigh_jan_1895-2018.csv')
print(data)

x = data[['Date']]
y = data[['Value']]

x_pred = ([[201900], [202300], [202400]])

reg = LinearRegression().fit(x.values, y.values)

slope, intercept, r, p, std_error = stats.linregress(data['Date'], data['Value'])
y_pred = reg.predict(x_pred)
x2 = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
y2= reg.predict(x2)

plt.scatter(x, y, c='blue', label='Data Points')
plt.scatter(x_pred, y_pred, c='green', label='Predicted')
plt.plot(x2, y2, color='red', label="Model")
plt.title(f'January Average High Temperature. Slope: {slope:.2f}, Intercept: {intercept:.2f}')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()
plt.show()