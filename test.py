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

import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

X = [] # X is sampling
# win_len = 1
# win_len = 20
# samp_points = 20
length = 25
times = 1000

for time in range(500, 500+times, 1):
    start_p = time
    x_i = []
    xxxx = []
    for i in range(length):
        x_i.append(bcg[time+i])
    x_i = np.asarray(x_i)
    for xxx in x_i:
        # min_max_scaler = MinMaxScaler()
        # n_xxx = min_max_scaler.fit_transform(xxx)
        n_xxx = (xxx - np.min(x_i)) / (np.max(x_i) - np.min(x_i))
        xxxx.append(n_xxx)
    X.append(xxxx)
    # X.append(x_i)

    # new_x = [i - np.mean(xxxx) for i in xxxx]
    # kk = np.mean(n_xxx)
    # for i in range(len(xxxx)):
    #     k = n_xxx[i] - kk
    #     n_xxxx.append(k)
    # X.append(new_x)

# plt.title("orginal sin singal")
# plt.show()
X = np.asarray(X)
print("X", X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.plot(X[:, 0], X[:, 1], X[:, 2])

# SVD decomp
U, s, V = np.linalg.svd(X, full_matrices=False)
print("s value-------------------------", s)
# for i in range(len(s)):
#     if i != 0:
#         s[i] = 0
# S = np.diag(s)

print(U.shape, V.shape, s.shape)

# 第几个奇异值对应的还原的信号， 蓝色是还原的信号
for i in range(3):
    a = np.zeros(len(s))
    a[i] = s[i]
    S = np.diag(a)
    proj_data = np.dot(U, np.dot(S, V))
    plt.figure()
    # plt.scatter(proj_data)
    plt.plot(proj_data[:, 1])
    np_bcg = np.asarray(bcg)
    # reg_bcg = (np_bcg - np.mean(bcg)) / (np.std(np_bcg))
    reg_bcg = (np_bcg - np.mean(bcg)) / (np.max(np_bcg - np.min(np_bcg)))
    plt.plot(reg_bcg[500:1500], c='r')
plt.show()

# proj_data = np.dot(U, np.dot(S, V))
# proj_data = U[:, 1]
# proj_data = V[0]
# plt.figure()
# plt.plot(proj_data)

# plt.plot(U[:, 0], c='b')
# plt.plot(U[:, 1], c='y')

U_1 = U[:, 1]
re_u = []
for i in range(0, len(U_1), 50):
    start = i
    end = i + 50
    uu = U_1[start:end]
    reg_uu = [(uuu - np.min(uu)) / (np.max(uu) - np.min(uu)) for uuu in uu]
    print(len(reg_uu))
    re_u.extend(reg_uu)

# plt.figure()
# plt.plot(re_u, c='b')
# plt.plot(U_1, c='r')

# plt.figure()
# plt.plot(U[:, 0])
# plt.plot(U[:, 1])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(U[:, 0], U[:, 1], U[:, 2])
# ax.plot(U[:, 0], U[:, 1], U[:, 2])

# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(proj_data[:, 0], proj_data[:, 1], proj_data[:, 2])
# plt.show()



