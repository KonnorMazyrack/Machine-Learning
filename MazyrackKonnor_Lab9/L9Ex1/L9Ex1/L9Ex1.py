import pandas as pd
import io
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv('balloons_2features.csv')

data['Act'] = data['Act'].map({'Stretch': 0, 'Dip': 1})
data['Age'] = data['Age'].map({'Adult': 0, 'Child': 1})
data['Inflated'] = data['Inflated'].map({'T': 1, 'F': 0})

X = data[['Act', 'Age']]
y = data['Inflated']
print(X)
print(y)

true_count = sum(y)
false_count = len(y) - true_count
print(f"True count: {true_count}, False count: {false_count}")

# Calculate root entropy
total = len(y)
p_T = true_count / total # 8/20 = 0.4
p_F = false_count / total # 12/20 = 0.6
root_entropy = -p_T * np.log2(p_T + 1e-10) - p_F * np.log2(p_F + 1e-10)
print(f"Root Entropy: {root_entropy:.3f}")

# Calculate Act entropy
Xact = data['Act']
Total = len(y)
sTot = 0
dTot = 0
strue_count = 0
dtrue_count = 0
for i in range( len(y)):
    if (Xact[i] == 0):
        sTot += 1
        if y[i] == 1:
            strue_count += 1
    else:
        dTot += 1
        if y[i] == 1:
            dtrue_count += 1
sfalse_count = len(y) - strue_count
dfalse_count = len(y) - dtrue_count
print(f"(stretch)True count: {strue_count}, False count: {sfalse_count}")
print(f"(dip)True count: {dtrue_count}, False count: {dfalse_count}")
stp_T = strue_count / total 
stp_F = sfalse_count / total 
s_entropy = -stp_T * np.log2(stp_T + 1e-10) - stp_F * np.log2(stp_F + 1e-10)
sentropy = (sTot / Total) * s_entropy
dp_T = dtrue_count / total 
dp_F = dfalse_count / total 
d_entropy = -dp_T * np.log2(dp_T + 1e-10) - dp_F * np.log2(dp_F + 1e-10)
dentropy = (dTot / Total) * d_entropy
Infact_entropy = dentropy + sentropy
print(f"inflated|act Entropy: {Infact_entropy:.3f}")
InfGainAct = root_entropy - Infact_entropy
print(f"information gain (Act): {InfGainAct:.3f}")

# Calculate Age entropy
Xage = data['Age']
Total = len(y)
aTot = 0
cTot = 0
atrue_count = 0
ctrue_count = 0
for i in range( len(y)):
    if (Xage[i] == 0):
        aTot += 1
        if y[i] == 1:
            atrue_count += 1
    else:
        cTot += 1
        if y[i] == 1:
            ctrue_count += 1
afalse_count = len(y) - atrue_count
cfalse_count = len(y) - ctrue_count
print(f"(adult)True count: {atrue_count}, False count: {afalse_count}")
print(f"(child)True count: {ctrue_count}, False count: {cfalse_count}")
ap_T = atrue_count / total 
ap_F = afalse_count / total 
a_entropy = -ap_T * np.log2(ap_T + 1e-10) - ap_F * np.log2(ap_F + 1e-10)
aentropy = (aTot / Total) * a_entropy
cp_T = ctrue_count / total 
cp_F = cfalse_count / total 
c_entropy = -cp_T * np.log2(cp_T + 1e-10) - cp_F * np.log2(cp_F + 1e-10)
centropy = (cTot / Total) * c_entropy
Infage_entropy = aentropy + centropy
print(f"inflated|age Entropy: {Infage_entropy:.3f}")
InfGainAge = root_entropy - Infage_entropy
print(f"information gain (Age): {InfGainAge:.3f}")

#split into adult and child
df_adult = data[data['Age'] == 0]
df_child = data[data['Age'] == 1]
df_adultY = np.array(df_adult['Inflated'])
df_childY = np.array(df_child['Inflated'])

#datapoint = [['Stretch', 'Adult']]
datapoint = [0, 0]
total = 0
P_T = 0
P_F = 0
if datapoint[1] == 0:
    df_adultact = np.array(df_adult['Act'])
    for i in range(len(df_adultY)):
        total += 1
        if (df_adultact[i] == datapoint[0]):
            if df_adultY[i] == 1:
                P_T += 1
            else:
                P_F += 1
else:
    df_childact = df_child['Act']

    for i in range(len(df_childY)):
        total +=1
        if (df_childact[i] == datapoint[0]):
            if df_childY[i] == 1:
                P_T += 1
            else:
                P_F += 1
P_T = P_T / total
P_F = P_F / total
if P_T > P_F:
    print("The prediction of the datapoint is Inflated.")
else:
    print("The prediction of the datapoint is NOT Inflated.")




