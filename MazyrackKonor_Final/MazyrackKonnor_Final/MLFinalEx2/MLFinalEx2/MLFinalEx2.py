import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('usedcars.csv')
df['model'] = df['model'].map({'SEL': 1, 'SE' : 2, 'SES' : 3})
df['color'] = df['color'].map({'Yellow': 1, 'Gray' : 2, 'Silver' : 3, 'White' : 4, 'Blue' : 5, 'Black' : 6, 'Green' : 7, 'Red' : 8, 'Gold' : 9})
df['transmission'] = df['transmission'].map({'AUTO': 0, 'MANUAL' : 1})
X = df[['year', 'model', 'mileage', 'color', 'transmission']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

n_estimators_range = np.arange(100, 1001, 100)
rmse_error = []
for n in n_estimators_range:
    rf = RandomForestRegressor(
        n_estimators=n,
        random_state=0,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)
    test_mse = mean_squared_error(y_test, pred)
    test_rmse = np.sqrt(test_mse)
    rmse_error.append(test_rmse)
    print(f"Root Mean Square Error ({n} estimators): {test_rmse:.2f}")

minIndex = np.argmin(rmse_error)
print(f"\nMinimum RMSE: {rmse_error[minIndex]} at index: {minIndex}\n")

rf = RandomForestRegressor(n_estimators=n_estimators_range[minIndex])
rf.fit(X_train, y_train)
pred = np.array(rf.predict(X_test))
y_test = np.array(y_test)
for i in range(len(y_test)):
    print(f"Actual: ${y_test[i]}  Predicted: ${pred[i]}")

#[2017, 'SE', 113067, 'Blue', 'AUTO']
data_point = np.array([[2017, 2, 113067, 5, 0]])
dpPred = rf.predict(data_point)
print(f"\nPredicted Value of single data points: ${dpPred}")
print(f"Important Features: {rf.feature_importances_}")
print(f"Important Features: {X.columns}")
