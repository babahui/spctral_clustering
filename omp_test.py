import matplotlib
import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_circles
from sklearn import linear_model
import sys
from scipy import linalg
from scipy.sparse.linalg import inv
import numpy
import scipy.sparse as sp
import scipy.sparse
from copy import deepcopy
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve

from collections import Counter

import pylab as pl

alpha = 25

def initDatasets():

    '''  here we give some data generate functions. X, Y means 2d coordinates and
    related labels.

    '''

    # 1. iris datasets
    # iris = datasets.load_iris()
    # X = iris.data[:,:2] # choose first two features
    # Y = iris.target
    # X = np.asarray(X)

    # 2. make concentric circles, unfortunately can only plot 2 circles
    # X, Y = make_circles(n_samples=50, factor=.3, noise=.05)

    # 3. make some quantiles by Gaussian
    # X, Y = make_gaussian_quantiles(cov=(1, 1), n_samples=100, n_features=2, n_classes=3)

    # 4. generate points in 3 circles, by normalized distribution
    GauSize = [1, 2 ,3, 4, 5]
    # GauSize = [1, 2 ,3]
    X1 = []
    X2 = []
    Y = []
    X = []
    for i in GauSize:
        mean = [i*15, 25]
        cov = [[3, 0], [0, 3]]
        # (x1, x2) is 2-d coordinates
        if i == 1:
            x1, x2 = np.random.multivariate_normal(mean, cov, 100).T
            X1 = np.append(X1, x1)
            X2 = np.append(X2, x2)
            y = np.dot(i, np.ones(100))
            Y = np.append(Y, y)
        if i == 2:
            x1, x2 = np.random.multivariate_normal(mean, cov, 100).T
            X1 = np.append(X1, x1)
            X2 = np.append(X2, x2)
            y = np.dot(i, np.ones(100))
            Y = np.append(Y, y)
        if i == 3:
            x1, x2 = np.random.multivariate_normal(mean, cov, 100).T
            X1 = np.append(X1, x1)
            X2 = np.append(X2, x2)
            y = np.dot(i, np.ones(100))
            Y = np.append(Y, y)
        mean2 = [15, 40]
        if i == 4:
            x1, x2 = np.random.multivariate_normal(mean2, cov, 100).T
            X1 = np.append(X1, x1)
            X2 = np.append(X2, x2)
            y = np.dot(i, np.ones(100))
            Y = np.append(Y, y)
        mean3 = [30, 40]
        if i == 5:
            x1, x2 = np.random.multivariate_normal(mean3, cov, 100).T
            X1 = np.append(X1, x1)
            X2 = np.append(X2, x2)
            y = np.dot(i, np.ones(100))
            Y = np.append(Y, y)

    for i in range(len(X1)):
        X.append([X1[i], X2[i]])

    #plot.scatter function 3 gaussian circles
    # plt.scatter(X1, X2, c=Y)
    # plt.axis('equal')
    # plt.show()

    print([row[1] for row in X])
    return X, Y, X1, X2

def plot():
    X, Y = initDatasets()

    # plot origin data
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('length')
    plt.ylabel('width')

    # plot 3 distance-based eigenvectors manifold
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    matrix = distanceMatrix(X)
    eigen_x1, eigen_x2,  eigen_x3 = matrixDecomp(matrix)
    ax.scatter(eigen_x1, eigen_x2, eigen_x3, c=Y, cmap=plt.cm.Paired)
    ax.set_title("eigenvalue*eigenvector manifold")
    ax.set_xlabel("first eigenvector")
    ax.set_ylabel("second eigenvector")
    ax.set_zlabel("third eigenvector")

    # plot 2-d eigenvectors space
    plt.figure(3, figsize=(8, 6))
    plt.scatter(eigen_x1, eigen_x3, c=Y, cmap=plt.cm.Paired)

    plt.show()

def plot2():
    X, Y, X1, X2 = initDatasets()

    # plot origin data
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X1, X2, c=Y, cmap=plt.cm.Paired)
    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('length')
    plt.ylabel('width')

    # plot 3 distance-based eigenvectors manifold
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    matrix = distanceMatrix(X)
    eigen_x1, eigen_x2,  eigen_x3 = matrixDecomp(matrix)
    ax.scatter(eigen_x1, eigen_x2, eigen_x3, c=Y, cmap=plt.cm.Paired)
    # ax.scatter(eigen_x1, eigen_x3, c=Y, cmap=plt.cm.Paired)

    ax.set_title("eigenvalue*eigenvector manifold")
    # ax.set_xlabel("first eigenvector")
    # ax.set_ylabel("second eigenvector")
    # ax.set_zlabel("third eigenvector")

    plt.show()

def distanceMatrix(X):
    matrix = []
    X = np.asarray(X)
    print(X.shape)
    # size = np.shape(X)
    size = X.shape
    # get data number and labels number
    dataNumber = size[0]
    labelsNumber = size[1]
    for i in range(dataNumber):
        for j in range(dataNumber):
            # get distance in n**2 loop, distance can be defined as l2, l1...
            dist = LA.norm(X[i] - X[j], 2)
            # dist = LA.norm(X[i] - X[j])
            matrix.append(dist)

    matrix = np.reshape(matrix, (dataNumber, dataNumber))
    return matrix

def matrixDecomp(matrix):
    n = 3
    w, v = LA.eig(matrix)
    min_index = np.argsort(w)[:n]
    index1, index2, index3 = min_index[0], min_index[1], min_index[2]
    return w[index1]*v[:, index1], w[index2]*v[:, index2], w[index3]*v[:, index3]

def multiMatrixDecomp(matrix, n):  # n means number of eigenvalues and eigenvectors
    eigenvectors = [] # eigenvectors means list of eigenvectors sorted by eigenvalues value
    w, v = LA.eig(matrix)
    min_index = np.argsort(w)[:n]
    for i in range(n):
        index = min_index[i]
        eigenvectors.append(w[index]*v[:, index])

    return eigenvectors

def clustering(n): # n means choose first n eigenvectors
    # get eigenvectors list
    X, Y = initDatasets()
    matrix = distanceMatrix(X)
    eigenvectors = multiMatrixDecomp(matrix, n)

def initVec(eigenNumber):
    # get matrix by pre-functions
    # X, Y, X1, X2 = initDatasets()
    matrix = distanceMatrix(X)

    w, v = LA.eig(matrix)
    index = np.argsort(w)[eigenNumber]
    # get 1-d vector, which is eigenvector*eigenvalue
    vector = w[index]*v[:, index]
    print("original Vector: ", vector)
    print("argsort original Vector: ", np.argsort(vector) )
    sortIndex = np.argsort(vector)
    sortedVector = [ vector[i] for i in sortIndex ]
    # print(len(sortedVector))

    label = np.zeros(len(sortIndex))
    for i in range(len(sortIndex)):
        value = sortIndex[i]
        label[i] = Y[value]
    length = len(sortedVector)

    print("sortedVector: ", sortedVector)
    print(Y)
    return sortedVector, label, X, X1, X2, Y, sortIndex

def sparse():
    # # get matrix by pre-functions
    # X, Y = initDatasets()
    # matrix = distanceMatrix(X)

    # w, v = LA.eig(matrix)
    # index = np.argsort(w)[0]
    # # get 1-d vector, which is eigenvector*eigenvalue
    # vector = w[index]*v[:, index]
    # sortIndex = np.argsort(vector)
    # sortedVector = [ vector[i] for i in sortIndex ]
    # # print(len(sortedVector))

    # label = np.zeros(len(sortIndex))
    # for i in range(len(sortIndex)):
    #     value = sortIndex[i]
    #     label[i] = Y[value]
    # length = len(sortedVector)

    sortedVector, label = initVec()
    reguVec, inv_A, diffVec = regularization(sortedVector)
    # print(inv_A.shape, len(diffVec))

    reguVecSO = secondOrderRegularization(diffVec)
    reguVecSO = np.dot(inv_A, reguVecSO)

    return sortedVector, reguVec, reguVecSO, label

def difference(vector):
    diffVec = []
    length = len(vector)
    np.append(vector, vector[-1])
    for i in range(len(vector)-1):
        diffVec.append(vector[i+1] - vector[i])
    return diffVec

def plotVector():
    sortedVector, reguVec, reguVecSO, label = sparse()
    length = len(sortedVector)
    Y = np.zeros(length)
    Z = np.ones(length)
    Q = np.dot(np.ones(length), 2)
    plt.figure(2, figsize=(8, 6))
    plt.scatter(sortedVector, Y, c=label, s=10)
    plt.scatter(reguVec, Z, s=10)
    plt.scatter(reguVecSO, Q, s=10)
    print(len(reguVec), len(reguVecSO))
    plt.xlabel('x-axis')
    plt.ylim(-2,10)
    plt.title('蓝点表示一维排序后图，橙色点表示一次差分求优化后')
    plt.show()


def regularization(sortedVector):
    length = len(sortedVector)
    y = sortedVector.copy()
    A = np.zeros((length-1, length))
    for i in range(length-1):
        A[i][i] = 1
        A[i][i+1] = -1
    inv_A = np.linalg.pinv(A)
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(inv_A, y)
    reguVec = np.dot(inv_A, clf.coef_)
    diffVec = np.dot(A, y)
    print( np.dot(A, y), sortedVector)

    return reguVec, inv_A, diffVec

def regularization2(sortedVector):
    length = len(sortedVector)
    y = sortedVector.copy()
    A = np.zeros((length-1, length))
    for i in range(length-1):
            A[i][i] = 1
            A[i][i+1] = -1

    inv_A = np.linalg.pinv(A)
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(inv_A, y)
    reguVec = np.dot(inv_A, clf.coef_)
    diffVec = np.dot(A, y)
    print( np.dot(A, y), sortedVector)

    return reguVec, inv_A, diffVec


def secondOrderRegularization(diffVec):
    length = len(diffVec)
    y = diffVec.copy()
    B = np.zeros((length-1, length))
    for i in range(length-1):
        B[i][i] = 1
        B[i][i+1] = -1
    inv_B = np.linalg.pinv(B)
    clf = linear_model.Lasso(alpha=0.01)
    clf.fit(inv_B, y)
    reguVecSO = np.dot(inv_B, clf.coef_)

    return reguVecSO

def ADMMByVectorMethod1():
    # argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a
    # used in plotVectorByADMM function

    # original input value set
    sortedVector, label, X, X1, X2, Y, sortIndex = initVec(0)
    origVec = deepcopy(sortedVector)

    length = len(sortedVector)
    A = np.zeros((length-1, length))
    for i in range(length-1):
        A[i][i] = -1
        A[i][i+1] = 1
    # print(A.shape, A)

    B = np.identity(length)
    B[length-1, length-1] = 0
    # print(B)

    x = sortedVector.copy()
    x1 = sortedVector.copy()
    v = np.dot(A, x1)
    w = np.dot(A, x1)
    e = 40
    r = 0.01
    n = 0.99 # n belong to [0.95, 0.99]
    percep = 0.1
    # a = int(percep * length)
    a = 4
    i = 1

    I = np.identity(length)

    while abs(LA.norm(np.dot(A, x), 0) - a) >= e:
        # left = numpy.linalg.inv(I + np.dot(1/r, np.dot(A.T, A)))
        left = numpy.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
        right = np.dot(1, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
        # x = np.dot(left.toarray(), right)
        x = np.dot(left, right)

        z = np.dot(A, x) + w
        if LA.norm(z, 0) <= a:
            v = deepcopy(z)
            print("z<a")
        else:
            z_abs = abs(z)
            z_sort = np.argsort(z_abs)[::-1]
            v = np.zeros(length-1)
            for i in range(a):
                index = z_sort[i]
                v[index] = z[index]

        w = w + np.dot(A, x) - v
        # r = r * n

    indexVec, unSortedLabelVec = nonzeroValue(origVec, np.dot(A, x))

    # labelVec = np.zeros(len(unSortedLabelVec))
    # for i in range(len(unSortedLabelVec)):
    #     labelVec[sortIndex[i]] = unSortedLabelVec[i]
    # print("unSortedLabelVec: ", unSortedLabelVec)
    # print("labelVec: ", labelVec)
    return origVec, indexVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec

# used in plotvectorByADMM function
def ADMMByVectorMethod2():
    # argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a
    # used in plotVectorByADMM function

    # original input value set
    sortedVector, label, X, X1, X2, Y, sortIndex = initVec(0)
    origVec = deepcopy(sortedVector)

    length = len(sortedVector)
    A = np.zeros((length-1, length))
    for i in range(length-1):
        A[i][i] = -1
        A[i][i+1] = 1
    # print(A.shape, A)

    B = np.identity(length)
    B[length-1, length-1] = 0
    # print(B)

    x = sortedVector.copy()
    x1 = sortedVector.copy()
    v = np.dot(A, x1)
    w = np.dot(A, x1)
    e = 40
    r = 0.01
    n = 0.99 # n belong to [0.95, 0.99]
    percep = 0.1
    # a = int(percep * length)
    a = 4
    i = 1

    I = np.identity(length)

    while abs(LA.norm(np.dot(A, x), 0) - a) >= e:
        # left = numpy.linalg.inv(I + np.dot(1/r, np.dot(A.T, A)))
        left = numpy.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
        right = np.dot(1, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
        # x = np.dot(left.toarray(), right)
        x = np.dot(left, right)

        z = np.dot(A, x) + w
        if LA.norm(z, 0) <= a:
            v = deepcopy(z)
            print("z<a")
        else:
            z_abs = abs(z)
            z_sort = np.argsort(z_abs)[::-1]
            v = np.zeros(length-1)
            for i in range(a):
                index = z_sort[i]
                v[index] = z[index]

        w = w + np.dot(A, x) - v
        # r = r * n

    indexVec, unSortedLabelVec = nonzeroValue(origVec, np.dot(A, x))

    # labelVec = np.zeros(len(unSortedLabelVec))
    # for i in range(len(unSortedLabelVec)):
    #     labelVec[sortIndex[i]] = unSortedLabelVec[i]
    # print("unSortedLabelVec: ", unSortedLabelVec)
    # print("labelVec: ", labelVec)
    return origVec, indexVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec

def ADMM2(eigenNumber):
    # argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a

    # original input value set
    sortedVector, label, X, X1, X2, Y, sortIndex = initVec(eigenNumber)
    origVec = deepcopy(sortedVector)

    length = len(sortedVector)
    A = np.zeros((length-1, length))
    for i in range(length-1):
        A[i][i] = -1
        A[i][i+1] = 1
    # print(A.shape, A)

    B = np.identity(length)
    B[length-1, length-1] = 0
    # print(B)

    x = sortedVector.copy()
    x1 = sortedVector.copy()
    v = np.dot(A, x1)
    w = np.dot(A, x1)
    e = 20
    r = 0.01
    n = 0.99 # n belong to [0.95, 0.99]
    percep = 0.1
    # a = int(percep * length)
    a = 3
    i = 1

    I = np.identity(length)

    while abs(LA.norm(np.dot(A, x), 0) - a) >= e:
        # left = numpy.linalg.inv(I + np.dot(1/r, np.dot(A.T, A)))
        left = numpy.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
        right = np.dot(1, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
        # x = np.dot(left.toarray(), right)
        x = np.dot(left, right)

        z = np.dot(A, x) + w
        if LA.norm(z, 0) <= a:
            v = deepcopy(z)
            print("z<a")
        else:
            z_abs = abs(z)
            z_sort = np.argsort(z_abs)[::-1]
            v = np.zeros(length-1)
            for i in range(a):
                index = z_sort[i]
                v[index] = z[index]

        w = w + np.dot(A, x) - v
        # r = r * n

    indexVec, unSortedLabelVec = nonzeroValue(origVec, np.dot(A, x))

    labelVec = np.zeros(len(unSortedLabelVec))
    for i in range(len(unSortedLabelVec)):
        labelVec[sortIndex[i]] = unSortedLabelVec[i]
    print("unSortedLabelVec: ", unSortedLabelVec)
    print("labelVec: ", labelVec)
    return origVec, indexVec, labelVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec

# wish to plot re-color point and find different label with original point
def nonzeroValue(origVec, gapVec):
    indexVec = [0]
    threshold = 1
    preIndex = 0
    labelVal = 1
    unSortedLabelVec = []

    for i in range(len(gapVec)):
        if gapVec[i] > threshold:
            indexVec.append(1)
        else:
            indexVec.append(0)

    # make last element to 1, that we can get all element in next loop
    indexVec.append(1)
    print("indexVec", indexVec[:-1])

    for i in range(len(indexVec)):
        if indexVec[i] == 1:
            indexVec[preIndex:i] = origVec[preIndex:i]
            labelVecPiece = [labelVal for i in indexVec[preIndex:i]]
            unSortedLabelVec.extend(labelVecPiece)
            labelVal = labelVal + 1
            preIndex = i

    return indexVec[:-1], unSortedLabelVec

def plotVectorByADMM():
    # x1, indexVec, labelVec, x, label, X, X1, X2, Y, sortIndex = ADMM2()
    # x means origVec, unSortedLabelVec means the final label array in vector in          x-axis
    x1, indexVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec = ADMMByVectorMethod()

    print("unSortedLabelVec: ", unSortedLabelVec)
    # print(max(set(labelVec), key=labelVec.count))
    # print(Counter(labelVec))

    # print("miss data percent : ", LA.norm(labelVec-Y, 0) / len(Y))
    origLabel = deepcopy(Y)
    i = 0
    length = len(x1)
    Y = np.zeros(length) # Y-axis all in 0
    Z = np.ones(length) # vectical coordinates all in 1

    # plot the boundry point
    coordinate1 = 0.5
    coordinate2 = -0.5
    for i in range(len(unSortedLabelVec) - 1):
        if unSortedLabelVec[i] != unSortedLabelVec[i+1]:
            plt.scatter(x1[i], coordinate1, marker='+', c='r')
            plt.scatter(x1[i+1], coordinate1, marker='+', c='r')
    for i in range(len(origLabel) - 1):
        if origLabel[i] != origLabel[i+1]:
            plt.scatter(x1[i], coordinate2, marker='+', c='r')
            plt.scatter(x1[i+1], coordinate2, marker='+', c='r')

    # matplotlib.interactive(True)
    plt.figure(1, figsize=(8, 6))
    # plt.scatter(x1, Y, c=label, s=10)
    # plt.scatter(x, Z, s=10)

    # plt.scatter(indexVec, Z, c=labelVec, s=10)
    plt.scatter(x1, Z, c=label, s=10)
    plt.ylim(-2,4)
    plt.xlim(-50, 50)
    plt.title('after ADMM optimization, a, all 90 points')

    plt.scatter(x1, Y , c=origLabel, s=10)
    # plt.scatter(x1[19], Z[19], c='r', marker='^', s=15)
    # plt.scatter(x1[20], Z[20], c='r', marker='^', s=15)
    plt.xlabel('x-axis')
    plt.ylim(-2,4)
    plt.xlim(-50, 50)
    plt.title('original plot, a, all 90 points')
    # matplotlib.interactive(False)
    plt.show()

def plotVectorByADMM():
    # x1, indexVec, labelVec, x, label, X, X1, X2, Y, sortIndex = ADMM2()
    # x means origVec, unSortedLabelVec means the final label array in vector in          x-axis
    x1, indexVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec = ADMMByVectorMethod()

    print("unSortedLabelVec: ", unSortedLabelVec)
    # print(max(set(labelVec), key=labelVec.count))
    # print(Counter(labelVec))

    # print("miss data percent : ", LA.norm(labelVec-Y, 0) / len(Y))
    origLabel = deepcopy(Y)
    i = 0
    length = len(x1)
    Y = np.zeros(length) # Y-axis all in 0
    Z = np.ones(length) # vectical coordinates all in 1

    # plot the boundry point
    coordinate1 = 0.5
    coordinate2 = -0.5
    for i in range(len(unSortedLabelVec) - 1):
        if unSortedLabelVec[i] != unSortedLabelVec[i+1]:
            plt.scatter(x1[i], coordinate1, marker='+', c='r')
            plt.scatter(x1[i+1], coordinate1, marker='+', c='r')
    for i in range(len(origLabel) - 1):
        if origLabel[i] != origLabel[i+1]:
            plt.scatter(x1[i], coordinate2, marker='+', c='r')
            plt.scatter(x1[i+1], coordinate2, marker='+', c='r')

    # matplotlib.interactive(True)
    plt.figure(1, figsize=(8, 6))
    # plt.scatter(x1, Y, c=label, s=10)
    # plt.scatter(x, Z, s=10)

    # plt.scatter(indexVec, Z, c=labelVec, s=10)
    plt.scatter(x1, Z, c=label, s=10)
    plt.ylim(-2,4)
    plt.xlim(-50, 50)
    plt.title('after ADMM optimization, a, all 90 points')

    plt.scatter(x1, Y , c=origLabel, s=10)
    # plt.scatter(x1[19], Z[19], c='r', marker='^', s=15)
    # plt.scatter(x1[20], Z[20], c='r', marker='^', s=15)
    plt.xlabel('x-axis')
    plt.ylim(-2,4)
    plt.xlim(-50, 50)
    plt.title('original plot, a, all 90 points')
    # matplotlib.interactive(False)
    plt.show()

def plotDataByADMM():
    x1, indexVec, labelVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec = ADMM2()
    # print("labelVec and Y: ", labelVec, Y)

    # print(max(set(labelVec), key=labelVec.count))
    print(Counter(labelVec))

    print("sortIndex: ", sortIndex)
    # print("miss data percent : ", LA.norm(labelVec-Y, 0) / len(Y))
    origLabel = deepcopy(Y)
    i = 0
    length = len(x1)
    Y = np.zeros(length) # Y-axis all in 0
    Z = np.ones(length) # vectical coordinates all in 1

    matplotlib.interactive(True)
    plt.figure(1, figsize=(8, 6))
    # plt.scatter(x1, Y, c=label, s=10)
    # plt.scatter(x, Z, s=10)
    # print(len(indexVec), len(labelVec))

    print("original X1 coordinates: ", X1)
    print("argsort X1: ", np.argsort(X1) )
    pre = np.argsort(X1)

    plt.scatter(X1, X2, c=labelVec, s=10)
    print("unSortedLabelVec", unSortedLabelVec)
    print("labelVec", labelVec)
    print("origLabel", origLabel)

    # # plot triangle shape in boundry
    boundry = []
    colors = []
    colorsIter = ['r', 'r', 'b', 'b', 'm', 'm', 'k', 'k', 'c' ,'c']
    for i in range(100):
        colors.extend(colorsIter)
    # markers = ['+', '+', 'x', 'x', 'h', 'h', 'H', 'H']

    testVec = unSortedLabelVec
    for i in range(1,len(testVec)-1):
        if testVec[i] != testVec[i+1] and testVec[i] == testVec[i-1]:
            boundry.append(i)
            boundry.append(i+1)
        if testVec[i] != testVec[i+1] and testVec[i] != testVec[i-1]:
            boundry.append(i)
            boundry.append(i)
    # for i in [1, 2]:
    #     leftBoundry = sortIndex[i * length/3 - 1]
    #     rightBoundry = sortIndex[i * length/3]
    #     plt.scatter(X1[leftBoundry], X2[leftBoundry], c='r', marker='+')
    #     plt.scatter(X1[rightBoundry], X2[rightBoundry], c='r', marker='+')

    print("boundry array: ", boundry)
    for boundryVal, color in zip(boundry, colors[:len(boundry)]):
        boundryIndex = sortIndex[boundryVal]
        # for boundryIndex in range(len(sortIndex)):
        #     if sortIndex[boundryIndex] == boundryVal:
        plt.scatter(X1[boundryIndex], X2[boundryIndex], c=color, marker='+')

    plt.ylim(0, 50)
    plt.xlim(0, 50)
    plt.title('figure 1 : boundry condition, a=3, all 300 points')
    plt.show()

    plt.figure(2, figsize=(8, 6))
    plt.scatter(X1, X2, c=origLabel, s=10)

    # # # plot triangle shape in boundry
    # # length = len(X1)
    # # for i in range(1,4,1):
    # #     number = i * length/3 -1
    # #     for j,num in enumerate(sortIndex):
    # #         if num == number:
    # #             pos = j
    # #             plt.scatter(x1[pos], x2[pos], c='r', marker='^', s=15)

    plt.xlabel('x-axis')
    # plt.ylim(-2,4)
    # plt.xlim(-100, 100)
    plt.ylim(0, 50)
    plt.xlim(0, 50)
    plt.title('ADMM optimization, a=10, all 90 points')
    matplotlib.interactive(False)
    plt.show()


if __name__ == "__main__":
    # plot2()

    # # plot vector by lasso method
    # plotVector()

    # plot 2d data by ADMM method
    # plotDataByADMM()

    # initdatasets()

    # plot vector by ADMM method

    # X, Y, X1, X2 = initDatasets()
    # ADMMByVectorMethod(initVec(X, Y, X1, X2, 0))

    # plotVectorByADMM()

    X, Y, X1, X2 = initDatasets()
    matrix = distanceMatrix(X)
    n = 7
    w, v = LA.eig(matrix)
    min_index = np.argsort(w)[:n]
    index1, index2, index3, index4, index5, index6, index7 = min_index[0], min_index[1], min_index[2], min_index[3], min_index[4], min_index[5], min_index[6]

    # plt.figure(0)
    # plt.scatter(v[:, index1], v[:, index2])
    # plt.show()

    # fig = plt.figure(1, figsize=(8, 6))
    # ax = Axes3D(fig, elev=-150, azim=110)
    # ax.scatter(v[:, index1], v[:, index2], v[:, index3])
    # plt.show()

    eigenvector = v[:, index3]

    sortIndex = np.argsort(eigenvector)
    sortedEigenvector = [ eigenvector[i] for i in sortIndex ]

    length = len(sortedEigenvector)
    A = np.zeros((length-1, length))
    for i in range(length-1):
        # A[i][i] = -1
        # A[i][i+1] = 1
        A[i, i] = -1
        A[i, i+1] = 1

    gradient = np.dot(A, sortedEigenvector)

    # ifft
    s = np.fft.ifft(gradient)
    t = np.arange(len(gradient))
    plt.plot(t, s)

    # OMP
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.linear_model import OrthogonalMatchingPursuitCV
    from sklearn.datasets import make_sparse_coded_signal

    # import pywt

    leng = len(gradient)
    # DWT matrix
    # dataI = np.eye(length, dtype=np.float64)
    # X = pywt.dwt2(dataI, 'haar')

    # FFT matrix
    x = np.fft.fft(np.eye(leng))

    n_nonzero_coefs = 4
    y = s

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(x, y)
    w = omp.coef_

    pl.figure(figsize=(10,8))
    # pl.plot(np.dot(X, w))
    pl.plot(y)
    pl.plot(np.dot(x, w))

    pl.figure()
    pl.plot(w, color='black')
    # pl.plot(gradient)
    # pl.plot(xfp)
    pl.show()




