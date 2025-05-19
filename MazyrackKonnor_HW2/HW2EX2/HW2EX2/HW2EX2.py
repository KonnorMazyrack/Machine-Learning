import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('hsbdemo.csv', header=0)
print(df)
for i in range(len(df['gender'])):
    if df.loc[i, 'gender'] == 'female':
        df.loc[i, 'gender'] = 0
    else:
        df.loc[i, 'gender'] = 1
for i in range(len(df['ses'])):
    if df.loc[i, 'ses'] == 'low':
        df.loc[i, 'ses'] = 0
    elif df.loc[i, 'ses'] == 'middle':
        df.loc[i, 'ses'] = 1
    else:
        df.loc[i, 'ses'] = 2
for i in range(len(df['schtyp'])):
    if df.loc[i, 'schtyp'] == 'public':
        df.loc[i, 'schtyp'] = 0
    else:
        df.loc[i, 'schtyp'] = 1
for i in range(len(df['honors'])):
    if df.loc[i, 'honors'] == 'not enrolled':
        df.loc[i, 'honors'] = 0
    else:
        df.loc[i, 'honors'] = 1

X = np.array(df.drop(columns=['prog']).loc[:, 'gender':'awards'])
y = np.array(df['prog'])
print(f'Before Standardization x: {X}')
X = StandardScaler().fit_transform(X)
print(f'After Standrdization x: {X}')

print('Standard')
print(X)
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(f'Variance: {explained_variance}')
print(f'Cumulative Variance: {np.cumsum(explained_variance)}')
plt.plot(np.cumsum(explained_variance))
plt.xticks(ticks=[1, 3, 5, 7, 9], labels=[2, 4, 6, 8, 10])
plt.title('PC = 1-10')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Variuance Ratio')
plt.show()


