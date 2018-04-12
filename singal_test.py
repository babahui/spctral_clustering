import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

bcgin = pd.read_csv('hq.bcg.cal.csv')
fs = 125
Nyquist = fs/2

# bcg = [elem for elem in bcgin.iloc[start:start+fs*t,1]]
bcg = bcgin.iloc[:,1]
bcgRawMean = sum(bcg)/len(bcg)
bcg = [elem-bcgRawMean for elem in bcg]

#bcgtime = [elem for elem in bcgin.iloc[start:start+fs*t,0]]

#高通滤波 1Hz
b, a = signal.butter(2, 1/Nyquist ,'high')
bcg = signal.filtfilt(b, a, bcg)

#低通滤波 12Hz
b2, a2 = signal.butter(2, 10/Nyquist )
bcg = signal.filtfilt(b2, a2, bcg)

X = [] # X is sampling
# win_len = 1
# win_len = 20
# samp_points = 20
length = 150
times = 10000

for time in range(500, 500+times):
    start_p = time
    x_i = []
    for i in range(length):
        x_i.append(bcg[time+i])
    X.append(x_i)
    
plt.title("orginal sin singal")
plt.show()
X = np.asarray(X)
print("X", X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.show()

# SVD decomp
U, s, V = np.linalg.svd(X, full_matrices=False)
for i in range(len(s)):
    if i != 0:
        s[i] = 0
S = np.diag(s)
print(S)
print(U.shape, V.shape, s.shape)

# proj_data = np.dot(U, np.dot(S, V))
proj_data = U[:, 0]
# proj_data = V[0]
print("------------", proj_data.shape)
plt.plot(proj_data)
plt.show()

