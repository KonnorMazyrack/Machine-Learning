import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

df = pd.read_csv('recipes_muffins_cupcakes_scones.csv', header=0)
print(df)

X = np.array(df.loc[:, 'Flour':'Salt'])
features = df.loc[:, 'Flour':'Salt'].columns
y = np.array(df['Type'])
print(f'x: {X}')
print(f'y: {y}')

print(f'Before Standardization x: {X}')
X = StandardScaler().fit_transform(X)
print(f'After Standrdization x: {X}')

#Variance ratio and Cumulative Variance Ratio
print('Standard')
print(X)
pca = PCA(n_components=8)
principalComponents = pca.fit(X)
explained_variance = pca.explained_variance_ratio_
print(f'Variance Ratio: {explained_variance}')
print(f'Cumulative Variance Ratio: {np.cumsum(explained_variance)}')
plt.plot(np.cumsum(explained_variance))
plt.xticks(ticks=[1, 3, 5, 7], labels=[2, 4, 6, 8])
plt.title('PC = 1-8')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Variuance Ratio')
plt.show()

#Scatter Plot PC1 and PC2
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(f'\nVariance: {explained_variance}\n')
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf['Types'] = y
print(principalDf)

flowerClasses = ['Muffin', 'Cupcake', 'Scone']
colors = ['r', 'g', 'b']
for fClasses, color in zip(flowerClasses,colors):
    indicesToKeep = principalDf['Types'] == fClasses
    plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1'] ,principalDf.loc[indicesToKeep, 'principal component 2'], c = color)
plt.legend(flowerClasses)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title(f'PCA = 2, variance = {explained_variance}')
plt.show()

#Histogram
fig,axes =plt.subplots(4,2, figsize=(12, 9))
muffin=X[df.Type=='Muffin']
cupcake=X[df.Type=='Cupcake']
scone=X[df.Type=='Scone']
ax=axes.ravel()# flat axes with numpy ravel
for i in range(8):
  _,bins=np.histogram(X[:,i],bins=20)
  ax[i].hist(muffin[:,i],bins=bins,color='r',alpha=.2)# red color for malignant class
  ax[i].hist(cupcake[:,i],bins=bins,color='g',alpha=0.2)# alpha is           for transparency in the overlapped region 
  ax[i].hist(scone[:,i],bins=bins,color='b',alpha=0.2)
  ax[i].set_title(f'{df.columns[(i+1)]}',fontsize=9)
  ax[i].axes.get_xaxis().set_visible(False) # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
  ax[i].set_yticks(())
ax[0].legend(['muffin','cupcake', 'scone'],loc='best',fontsize=8)
plt.tight_layout()# let's make good plots
plt.show()

#Heatmap
plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1],['1st Comp','2nd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(features)),features,rotation=65,ha='left')
plt.tight_layout()
plt.show()# 

#highest and lowest variance
HVPC = 0
LVPC = 100
indexHV = 0
indexLV = 0
for i in range(len(pca.components_)):
    for b in range(len(pca.components_[0])):
        if pca.components_[i, b] > HVPC:
            HVPC = pca.components_[i, b]
            indexHV = b
        elif pca.components_[i, b] < LVPC:
            LVPC = pca.components_[i, b]
            indexLV = b
    HVPC = 0
    LVPC = 0
    print(f'Highest Variance PC{i+1}: {indexHV+1} {features[indexHV]}')
    print(f'Lowest Variance PC{i+1}: {indexLV+1} {features[indexLV]}')

#Correlation Heatmap
s=sns.heatmap(df[features].corr(),cmap='coolwarm') 
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
plt.show()











