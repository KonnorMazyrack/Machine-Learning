from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

df = fetch_california_housing(as_frame=True)

# and a single depended variable: 'GPA'
x = df.data.drop(columns=['Longitude', 'Latitude'], inplace=True)
x = df.data.iloc[::10]
#y = df.target.drop(columns=['Longitude', 'Latitude'], inplace=True)
y = df.target.iloc[::10]
x['MedHouseVal'] = y
print(x)
print(y)

sns.set(font_scale=1.1)
sns.set_style('white')
sns.pairplot(x, hue='MedHouseVal')
plt.show()





