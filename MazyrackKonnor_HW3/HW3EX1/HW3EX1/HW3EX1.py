import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

square_feet = np.array([100, 150, 185, 235, 310, 370, 420, 430, 440, 530, 600,
                         634, 718, 750, 850, 903, 978, 1010, 1050, 1990]).reshape(-1, 1)
price = np.array([12300, 18150, 20100, 23500, 31005, 359000, 44359, 52000, 53853, 61328,
                  68000, 72300, 77000, 89379, 93200, 97150, 102750, 115358, 119330, 323989])

lr = LinearRegression()
lr.fit(square_feet, price)

ransac = RANSACRegressor()
ransac.fit(square_feet, price)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

print(f"Before RANSAC: Slope: {lr.coef_[0]:.2f}, y-intercept: {lr.intercept_:.2f}")
print(f"After RANSAC: Slope: {ransac.estimator_.coef_[0]:.2f}, y-intercept: {ransac.estimator_.intercept_:.2f}")

plt.figure(figsize=(12, 6))
x_range = np.linspace(square_feet.min(), square_feet.max(), 100).reshape(-1, 1)
plt.plot(x_range, lr.predict(x_range), color='blue', label='Linear Regression Line')
plt.plot(x_range, ransac.predict(x_range), color='orange', label='RANSAC Line')
plt.scatter(square_feet[inlier_mask], price[inlier_mask], color='green', label='Inliers')
plt.scatter(square_feet[outlier_mask], price[outlier_mask], color='red', label='Outliers')
plt.title('Linear Regression Before and After RANSAC')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.legend()
plt.show()