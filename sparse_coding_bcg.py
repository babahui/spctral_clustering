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

length = 151
times = 1000

alldata = []
for time in range(500, 500+times):
    start_p = time
    x_i = []
    for i in range(length):
        x_i.append(bcg[time+i])
    x_i = np.asarray(x_i)
    reg_x = (x_i - np.mean(x_i)) / np.std(x_i)
    # reg_x = (x_i - np.min(x_i)) / (np.max(x_i) - np.min(x_i))
    alldata.append(reg_x)
alldata = np.asarray(alldata)
X = alldata[:, :150]
y = alldata[:, 150]

from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.01)
clf.fit(X, y)

w = clf.coef_
new_y = np.dot(X, w) # new_y 近似信号
print("w-----", w)

plt.figure()
plt.plot(y, c='r')
plt.plot(new_y, c='b')
plt.show()
