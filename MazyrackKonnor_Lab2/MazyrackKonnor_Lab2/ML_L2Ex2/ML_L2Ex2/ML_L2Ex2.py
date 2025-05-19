import numpy as np
from sklearn.preprocessing import StandardScaler
data = np.array([[1, 5], [3, 2], [8, 4], [7, 14]])
X, Y = data[:, 0], data[:, 1]
print(f'X: {X}')
print(f'Y: {Y}')

Xm, Ym = 0, 0
for i in range(len(X)):
    Xm += X[i]
    Ym += Y[i]
Xm /= len(X)
Ym /= len(Y)
print(f'mean X: {Xm}')
print(f'mean Y: {Ym}')

Xstd, Ystd = 0, 0
for i in range(len(X)):
    Xstd += (X[i] - Xm)**2
    Ystd += (Y[i] - Ym)**2
Xstd /= len(X)
Ystd /= len(Y)
Xstd = Xstd**0.5
Ystd = Ystd**0.5
print(f'STDX: {Xstd}')
print(f'STDY: {Ystd}')

Xscaled = np.array([])
Yscaled = np.array([])
for i in range(len(X)):
    Xz = (X[i] - Xm)/Xstd
    Yz = (Y[i] - Ym)/Ystd
    Xscaled = np.append(Xscaled, Xz)
    Yscaled = np.append(Yscaled, Yz)
print(f'Xscaled: {Xscaled}')
print(f'Yscaled: {Yscaled}')

Xrevert = np.array([])
Yrevert = np.array([])
for i in range(len(X)):
    Xr = Xscaled[i] * Xstd + Xm
    Yr = Yscaled[i] * Ystd + Ym
    Xrevert = np.append(Xrevert, Xr)
    Yrevert = np.append(Yrevert, Yr)
print(f'Xrevert: {Xrevert}')
print(f'Yrevert: {Yrevert}')

#built in functions for X and Y
scaler = StandardScaler()
z_scaledData = scaler.fit_transform(X.reshape(-1, 1))
print(f'X Mean (before scaling): {scaler.mean_}')
print(f'X Standard deviation (before scaling): {scaler.var_**.5}')
print(f'X Data (after scaling):\n{z_scaledData.flatten()}')
revert2originalValues = z_scaledData * (scaler.var_**.5) + scaler.mean_
print(f'X Data after reverting to original values:\n{revert2originalValues.flatten()}')

z_scaledData = scaler.fit_transform(Y.reshape(-1, 1))
print(f'Y Mean (before scaling): {scaler.mean_}')
print(f'Y Standard deviation (before scaling): {scaler.var_**.5}')
print(f'Y Data (after scaling):\n{z_scaledData.flatten()}')
revert2originalValues = z_scaledData * (scaler.var_**.5) + scaler.mean_
print(f'Y Data after reverting to original values:\n{revert2originalValues.flatten()}')











