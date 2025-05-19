import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets

data = pd.read_csv('materialsOutliers.csv')

X = data.drop(['Strength'], axis=1)
y = data['Strength']

overall_inlier_mask = np.ones(len(data), dtype=bool)

for feature in X.columns:
    X_temp = y.values.reshape(-1, 1)  
    y_temp = X[feature].values        

    ransac = linear_model.RANSACRegressor(residual_threshold=15, stop_probability=1.00).fit(X_temp, y_temp)

    inlier_mask = ransac.inlier_mask_

    overall_inlier_mask &= inlier_mask  

Xwooutliers = X[overall_inlier_mask]
ywooutliers = y[overall_inlier_mask]

lr = linear_model.LinearRegression().fit(Xwooutliers, ywooutliers)

print("Number of inliers:", np.sum(overall_inlier_mask))
print("Number of outliers:", len(data) - np.sum(overall_inlier_mask))
print("Intercept:", lr.intercept_)
print("Coefficients:", lr.coef_)

