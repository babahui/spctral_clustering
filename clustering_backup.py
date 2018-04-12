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
from PIL import Image
import time
import scipy.misc
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve

import matplotlib.image as mpimg

from sklearn.feature_extraction import image as img

# import initByNeighborhood function in neighbor_to_neighbor file
import sys
sys.path.append('./method_compared')
from neighbor_to_neighbor_2 import initByNeighborhood
# from floyd import initByFloyd
# from floyd import initByFloyd

# from floyd import superpixels

# MDS lib
from sklearn.manifold import MDS
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from time import time
from matplotlib.ticker import NullFormatter

alpha = 25

def initDatasets():

    '''  here we give some data generate functions. X, Y means 2d coordinates and related labels.

    '''

    # 1. iris datasets
    # iris = datasets.load_iris()
    # X = iris.data[:,:2] # choose first two features
    # Y = iris.target
    # X = np.asarray(X)
    # X1 = X[:, 0]
    # X2 = X[:, 1]

    # 2. make concentric circles, unfortunately can only plot 2 circles
    # X, Y = make_circles(n_samples=200, factor=.3, noise=.05)
    # X1 = X[:, 0]
    # X2 = X[:, 1]

    # 3. make some quantiles by Gaussian
    # X, Y = make_gaussian_quantiles(cov=(1, 1), n_samples=300, n_features=2, n_classes=3)
    # X1 = X[:, 0]
    # X2 = X[:, 1]

    # 4. generate points in 3 circles, by normalized distribution
    # # GauSize = [1, 2 ,3]
    # GauSize = [1, 2 ,3, 4, 5, 6]
    # X1 = []
    # X2 = []
    # Y = []
    # X = []
    # for i in GauSize:
    #     mean = [i*30, 30]
    #     cov = [[15, 0], [0, 15]]
    #     # (x1, x2) is 2-d coordinates
    #     if i == 1:
    #         x1, x2 = np.random.multivariate_normal(mean, cov, 100).T
    #         X1 = np.append(X1, x1)
    #         X2 = np.append(X2, x2)
    #         y = np.dot(i, np.ones(100))
    #         Y = np.append(Y, y)
    #     if i == 2:
    #         x1, x2 = np.random.multivariate_normal(mean, cov, 100).T
    #         X1 = np.append(X1, x1)
    #         X2 = np.append(X2, x2)
    #         y = np.dot(i, np.ones(100))
    #         Y = np.append(Y, y)
    #     if i == 3:
    #         x1, x2 = np.random.multivariate_normal(mean, cov, 100).T #         X1 = np.append(X1, x1)
    #         X2 = np.append(X2, x2)
    #         y = np.dot(i, np.ones(100))
    #         Y = np.append(Y, y)
    #     mean2 = [30, 60]
    #     if i == 4:
    #         x1, x2 = np.random.multivariate_normal(mean2, cov, 100).T
    #         X1 = np.append(X1, x1)
    #         X2 = np.append(X2, x2)
    #         y = np.dot(i, np.ones(100))
    #         Y = np.append(Y, y)
    #     mean3 = [60, 60]
    #     if i == 5:
    #         x1, x2 = np.random.multivariate_normal(mean3, cov, 100).T
    #         X1 = np.append(X1, x1)
    #         X2 = np.append(X2, x2)
    #         y = np.dot(i, np.ones(100))
    #         Y = np.append(Y, y)
    #     mean4 = [90, 60]
    #     if i == 6:
    #         x1, x2 = np.random.multivariate_normal(mean4, cov, 100).T
    #         X1 = np.append(X1, x1)
    #         X2 = np.append(X2, x2)
    #         y = np.dot(i, np.ones(100))
    #         Y = np.append(Y, y)


    # for i in range(len(X1)):
    #     X.append([X1[i], X2[i]])

    #plot.scatter function 3 gaussian circles
    # plt.scatter(X1, X2, c=Y)
    # plt.axis('equal')
    # plt.show()

    # 5. generate points in 3 circles, by normalized distribution, and in different direction from 4.
    GauSize = [1, 2 ,3, 4]
    X1 = []
    X2 = []
    Y = []
    X = []
    for i in GauSize:
        mean = [i*30, 35]
        cov = [[20, 0], [0, 20]]
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
        # mean2 = [45, 55]
        mean2 = [45, 65]
        if i == 3:
            x1, x2 = np.random.multivariate_normal(mean2, cov, 100).T
            X1 = np.append(X1, x1)
            X2 = np.append(X2, x2)
            y = np.dot(i, np.ones(100))
            Y = np.append(Y, y)
        # mean3 = [75, 65]
        mean3 = [75, 65]
        if i == 4:
            x1, x2 = np.random.multivariate_normal(mean3, cov, 100).T
            X1 = np.append(X1, x1)
            X2 = np.append(X2, x2)
            y = np.dot(i, np.ones(100))
            Y = np.append(Y, y)
        mean4 = [95, 30]
        if i == 5:
            x1, x2 = np.random.multivariate_normal(mean4, cov, 10).T
            X1 = np.append(X1, x1)
            X2 = np.append(X2, x2)
            y = np.dot(i, np.ones(100))
            Y = np.append(Y, y)

    for i in range(len(X1)):
        X.append([X1[i], X2[i]])

    #plot.scatter function 3 gaussian circles
    # print(Y)
    # plt.scatter(X1, X2, c=Y)
    # plt.axis('equal')
    # plt.show()

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
    # plt.figure(3, figsize=(8, 6))
    # plt.scatter(eigen_x1, eigen_x3, c=Y, cmap=plt.cm.Paired)

    # plt.show()

def plot2():
    X, Y, X1, X2 = initDatasets()

    # plot origin data
    # plt.figure(2, figsize=(8, 6))
    # plt.clf()
    # plt.scatter(X1, X2, c=Y, cmap=plt.cm.Paired)
    # # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    # plt.xlabel('length')
    # plt.ylabel('width')

    # plot 3 distance-based eigenvectors manifold
    # fig = plt.figure(1, figsize=(8, 6))
    # ax = Axes3D(fig, elev=-150, azim=110)
    matrix = distanceMatrix(X)
    eigen_x1, eigen_x2,  eigen_x3 = matrixDecomp(matrix)
    # # ax.scatter(eigen_x1, eigen_x2, eigen_x3, c=labelVec, cmap=plt.cm.Paired)
    # ax.scatter(eigen_x1, eigen_x2, eigen_x3, c=Y, cmap=plt.cm.Paired)

    # ax.set_title("eigenvalue*eigenvector manifold")
    # ax.set_xlabel("first eigenvector")
    # ax.set_ylabel("second eigenvector")
    # ax.set_zlabel("third eigenvector")

    # plt.show()

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
    n = 6
    n_dimension = 230
    w, v = LA.eig(matrix)
    min_index = np.argsort(w)[:n]
    index1, index2, index3, index4, index5, index6 = min_index[0], min_index[1], min_index[2], min_index[3], min_index[4], min_index[5]
    # return w[index1]*v[:, index1], w[index2]*v[:, index2], w[index3]*v[:, index3]

    # plot eigenvalues
    fig = plt.figure()
    # plt.scatter([1,2,3,4,5,6], [w[index1], w[index2], w[index3], w[index4], w[index5], w[index6]])
    plt.scatter([1,2,3,4,5,6], [w[index1]**2, w[index2]**2, w[index3]**2, w[index4]**2, w[index5]**2, w[index6]**2])

    w_ori = [ w[i] for i in np.argsort(w)[:5]]
    print("------------w-------------", )
    w_square = numpy.square(w_ori)
    print("-------w_square---------", w_square)
    print("-------w---------", w)
    w_length = w_square.shape[0]
    for i in range(w_length-1,-1,-1):
        w_square[i] = np.sum(w_square[:i+1])
    plt.figure()
    plt.plot(np.linspace(0, 100, w_length), w_square)
    plt.scatter(np.linspace(0, 100, w_length), w_square, c='r')
    plt.title("eigenvalues square")
    plt.show()


    # plot eigenvectors
    # fig = plt.figure()
    # plt.plot(np.linspace(0, 100, n_dimension), v[:, index1])
    # fig = plt.figure()
    # plt.plot(np.linspace(0, 100, n_dimension), v[:, index2])
    # fig = plt.figure()
    # plt.plot(np.linspace(0, 100, n_dimension), v[:, index3])
    # fig = plt.figure()
    # plt.plot(np.linspace(0, 100, n_dimension), v[:, index4])
    # fig = plt.figure()
    # plt.plot(np.linspace(0, 100, n_dimension), v[:, index5])

    # plt.show()
    return v[:, index1], v[:, index2], v[:, index3]

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

def initVec():
    # get matrix by pre-functions
    X, Y, X1, X2 = initDatasets()
    matrix = distanceMatrix(X)

    w, v = LA.eig(matrix)

    # get 1-d vector, which is eigenvector*eigenvalue
    index = np.argsort(w)[0]
    vector = w[index]*v[:, index]
    # leng = len(vector)
    # plt.scatter(vector, np.ones(leng))
    # plt.show()

    # # get 3-d vector, cal distance by x**2 + y**2 + z**2
    # index0 = np.argsort(w)[0]
    # vector0 = w[index0]*v[:, index0]
    # index1 = np.argsort(w)[1]
    # vector1 = w[index1]*v[:, index1]
    # index2 = np.argsort(w)[2]
    # vector2 = w[index2]*v[:, index2]
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=100)
    # ax.scatter(vector0, vector1, vector2)
    # vector = [ i**2 + j**2 + z**2 for i, j, z in zip(vector0, vector1, vector2)]

    # print("original Vector: ", vector)
    # print("argsort original Vector: ", np.argsort(vector) )
    sortIndex = np.argsort(vector)
    sortedVector = [ vector[i] for i in sortIndex ]
    # leng = len(sortedVector)
    # plt.scatter(sortedVector, np.ones(leng))
    # plt.show()

    label = np.zeros(len(sortIndex))
    for i in range(len(sortIndex)):
        value = sortIndex[i]
        label[i] = Y[value]
    length = len(sortedVector)

    # print("sortedVector: ", sortedVector)
    # print(Y)
    return sortedVector, label, X, X1, X2, Y, sortIndex


def initVec2(matrix=None):
    alpha = 10**16
    X1 = 0
    X2 = 0
    if matrix is None:
        # get matrix by pre-functions
        X, Y, X1, X2 = initDatasets()
        matrix = distanceMatrix(X)

    w, v = LA.eig(matrix)
    w = np.real(w)
    v = np.real(v)
    # print("----------w---------------", w)
    # print("------------w.argsort-----------", np.argsort(w))

    # get 1-d vector, which is eigenvector*eigenvalue
    # index = np.argsort(w)[0]
    # vector = w[index]*v[:, index]
    # leng = len(vector)
    # plt.scatter(vector, np.ones(leng))
    # plt.show()

    # get 3-d vector, cal distance by x**2 + y**2 + z**2
    # index0 = np.argsort(w)[0]
    # vector0 = w[index0]*v[:, index0]
    # index1 = np.argsort(w)[1]
    # vector1 = w[index1]*v[:, index1]
    # index2 = np.argsort(w)[2]
    # vector2 = w[index2]*v[:, index2]
    # index3 = np.argsort(w)[3]
    # vector3 = w[index3]*v[:, index3]
    # index4 = np.argsort(w)[4]
    # vector4 = w[index4]*v[:, index4]

    # index0 = np.argsort(w)[::-1][0]
    # vector0 = w[index0]*v[:, index0]
    # index1 = np.argsort(w)[::-1][1]
    # vector1 = w[index1]*v[:, index1]
    # index2 = np.argsort(w)[::-1][2]
    # vector2 = w[index2]*v[:, index2]
    # index3 = np.argsort(w)[::-1][3]
    # vector3 = w[index3]*v[:, index3]
    # index4 = np.argsort(w)[::-1][4]
    # vector4 = w[index4]*v[:, index4]

    index0 = np.argsort(w)[0]
    vector0 = v[:, index0]
    index1 = np.argsort(w)[1]
    vector1 = v[:, index1]
    index2 = np.argsort(w)[2]
    vector2 = v[:, index2]
    index3 = np.argsort(w)[3]
    vector3 = v[:, index3]
    index4 = np.argsort(w)[4]
    vector4 = v[:, index4]

    # index0 = np.argsort(w)[-1]
    # vector0 = v[:, index0]
    # index1 = np.argsort(w)[-2]
    # vector1 = v[:, index1]
    # index2 = np.argsort(w)[-3]
    # vector2 = v[:, index2]
    # index3 = np.argsort(w)[-4]
    # vector3 = v[:, index3]


    # index4 = np.argsort(w)[-5]
    # vector4 = v[:, index4]

    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=100)
    # ax.scatter(vector0, vector1, vector2)
    # vector = [ i**2 + j**2 + z**2 for i, j, z in zip(vector0, vector1, vector2)]

    # print("original Vector: ", vector)
    # print("argsort original Vector: ", np.argsort(vector) )
    # sortIndex = np.argsort(vector)
    # sortedVector = [ vector[i] for i in sortIndex ]
    # leng = len(sortedVector)
    # plt.scatter(sortedVector, np.ones(leng))
    # plt.show()

    sortIndex1 = np.argsort(vector0)
    sortedVector1 = [ vector0[i] for i in sortIndex1 ]
    sortIndex2 = np.argsort(vector1)
    sortedVector2 = [ vector1[i] for i in sortIndex2 ]
    sortIndex3 = np.argsort(vector2)
    sortedVector3 = [ vector2[i] for i in sortIndex3 ]

    sortIndex4 = np.argsort(vector3)
    sortedVector4 = [ vector3[i] for i in sortIndex4 ]

    sortIndex5 = np.argsort(vector4)
    sortedVector5 = [ vector4[i] for i in sortIndex5 ]


    # label1 = np.zeros(len(sortIndex1))
    # for i in range(len(sortIndex1)):
    #     value = sortIndex1[i]
    #     label1[i] = Y[value]
    # label2 = np.zeros(len(sortIndex2))
    # for i in range(len(sortIndex2)):
    #     value = sortIndex2[i]
    #     label2[i] = Y[value]
    # label3 = np.zeros(len(sortIndex3))
    # for i in range(len(sortIndex3)):
    #     value = sortIndex3[i]
    #     label3[i] = Y[value]

    # label4 = np.zeros(len(sortIndex4))
    # for i in range(len(sortIndex4)):
    #     value = sortIndex4[i]
    #     label4[i] = Y[value]

    # label5 = np.zeros(len(sortIndex5))
    # for i in range(len(sortIndex5)):
    #     value = sortIndex5[i]
    #     label5[i] = Y[value]


    # k = 5
    # index = np.argsort[:k]
    # vector = [ w[index_i]*v[:, index_i] for index_i in index]
    # sortIndex = [ np.argsort[vector_i] for vector_i in vector]
    # # sortedVector

    # print("sortedVector: ", sortedVector)
    # print(Y)
    # return sortedVector1, sortedVector2, sortedVector3, sortIndex1, sortIndex2, sortIndex3, vector0, vector1, vector2, X1, X2
    return sortedVector1, sortedVector2, sortedVector3, sortedVector4, sortedVector5, sortIndex1, sortIndex2, sortIndex3, sortIndex4, sortIndex5, vector0, vector1, vector2, vector3, vector4, X1, X2

def initVec3():
    # matrix, row, col = grayImageInit()
    # matrix, row , col = initByFloyd()
    matrix, row , col = superpixels()
    print("matrix: ", matrix)
    w, v = LA.eig(matrix)

    # # get 3-d vector, cal distance by x**2 + y**2 + z**2
    index0 = np.argsort(w)[0]
    vector0 = w[index0]*v[:, index0]
    index1 = np.argsort(w)[1]
    vector1 = w[index1]*v[:, index1]
    index2 = np.argsort(w)[2]
    vector2 = w[index2]*v[:, index2]
    index3 = np.argsort(w)[3]
    vector3 = w[index3]*v[:, index3]
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=100)
    # ax.scatter(vector0, vector1, vector2)
    # vector = [ i**2 + j**2 + z**2 for i, j, z in zip(vector0, vector1, vector2)]

    # print("original Vector: ", vector)
    # print("argsort original Vector: ", np.argsort(vector) )
    # sortIndex = np.argsort(vector)
    # sortedVector = [ vector[i] for i in sortIndex ]
    # leng = len(sortedVector)
    # plt.scatter(sortedVector, np.ones(leng))
    # plt.show()

    sortIndex1 = np.argsort(vector0)
    sortedVector1 = [ vector0[i] for i in sortIndex1 ]
    sortIndex2 = np.argsort(vector1)
    sortedVector2 = [ vector1[i] for i in sortIndex2 ]
    sortIndex3 = np.argsort(vector2)
    sortedVector3 = [ vector2[i] for i in sortIndex3 ]
    sortIndex4 = np.argsort(vector3)
    sortedVector4 = [ vector3[i] for i in sortIndex4 ]

    print("eigen decomposition done, sort eigenvectors done")

    # label1 = np.zeros(len(sortIndex1))
    # for i in range(len(sortIndex1)):
    #     value = sortIndex1[i]
    #     label1[i] = Y[value]
    # label2 = np.zeros(len(sortIndex2))
    # for i in range(len(sortIndex2)):
    #     value = sortIndex2[i]
    #     label2[i] = Y[value]
    # label3 = np.zeros(len(sortIndex3))
    # for i in range(len(sortIndex3)):
    #     value = sortIndex3[i]
    #     label3[i] = Y[value]
    return sortedVector1, sortedVector2, sortedVector3, sortedVector4, sortIndex1, sortIndex2, sortIndex3, sortIndex4, vector0, vector1, vector2, vector3, row, col

def grayImageInit(): # reduce picture size to 50*50 pixels
    image = Image.open('image/mountain2.jpg')
    image = image.resize((50, 50), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)
    # get image, and convert RGB to GRAY value by mode='L'
    grayMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='L')

    # convert RGB value to gray value, gray = 0.2989R + 0.5870G + 0.1140B,               output distance matrix by W = sqrt(a(F_i - F_j)**2 + b(X_i - X_j)**2)
    # a = 0.6
    # b = 0.4 # a,b is weight
    a = 1
    b = 1 # a,b is weight
    row = grayMatrix.shape[0]
    col = grayMatrix.shape[1]
    reshapeMatrix = np.reshape(grayMatrix, row*col)
    length = reshapeMatrix.shape[0]

    # # use numpy save distanceMatrix
    # distanceMatrix = np.zeros((length, length))
    # for i in range(length):
    #     for j in range(length):
    #         left = a * (int(reshapeMatrix[i]) - int(reshapeMatrix[j]))**2
    #         right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
    #         distanceMatrix[i, j] = np.sqrt(left + right)


    # use gradient as distanceMatrix
    # graph = img.img_to_graph(reshapeMatrix, return_as=np.ndarray)
    # beta = 5
    # eps = 1e-6
    # graph = np.exp(-beta * graph / graph.std()) + eps
    # print(graph.shape)

    # use python list to save distanceMatrix
    distanceMatrix = []
    for i in range(length):
        for j in range(length):
            left = a * (int(reshapeMatrix[i]) - int(reshapeMatrix[j]))**2
            right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
            distance = np.sqrt(left * right)
            distanceMatrix.append(distance)
    distanceMatrix = np.reshape(distanceMatrix, (length, length))
    print("distanceMatrix: ", distanceMatrix)


    # here we make element to 1000, prove sparse matrix can do the same work
    # distanceMatrix = sparseMatrix(distanceMatrix)


    return distanceMatrix, row, col

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

def ADMM():
    # argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a

    # original input value set
    sortedVector, label = initVec()
    length = len(sortedVector)
    A = np.zeros((length, length))
    for i in range(length-1):
        A[i][i] = -1
        A[i][i+1] = 1
    A[length-1][0] = 1
    A[length-1][length-1] = -1
    print(A.shape, A)

    B = np.identity(length)
    B[length-1, length-1] = 0
    # print(B)

    x = sortedVector.copy()
    x1 = sortedVector.copy()
    v = np.dot(A, x1)
    w = np.dot(A, x1)
    e = 10
    r = 1
    n = 0.99 # n belong to [0.95, 0.99]
    a = int(0.5 * length)
    # a = 2
    i = 1

    bccb = np.dot(A.T, A)
    eigenMatrix = dialog(bccb)
    # F = DFT_matrix_2d(length)
    F = np.fft.fft(np.eye(length))
    inv_F = inv(F)

    I = np.identity(length)

    while abs(LA.norm(np.dot(A, x), 0) - a) > e:
        # bccb transform to FFT
        # left = inv(I + np.dot(1/r, eigenMatrix))
        # right = np.fft.fft(x1 + np.dot(1/r, np.dot(A.T, (v - w))))
        # x = np.fft.ifft(np.dot(left, right))

        temp = inv(I + np.dot(1/r, eigenMatrix))
        left = np.dot(inv_F, np.dot(temp, F))
        right = x1 + np.dot(1/r, np.dot(A.T, (v - w)))
        x = np.dot(left, right)

        z = np.dot(A, x) + w
        if LA.norm(np.dot(B, z), 0) <= a:
            v = z.copy()
        else:
            z_abs = abs(np.dot(B, z))
            z_sort = np.argsort(z_abs)[::-1]
            v = np.zeros(length)
            for i in range(a):
                index = z_sort[i]
                v[index] = z[index]
            v = v + np.dot((I-B), z)
        # w = w + np.dot(A, x) - v r = r * n
        i = i + 1
        print(i)

    return sortedVector, x, label

def ADMM1():
    # argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a

    # original input value set
    sortedVector, label, X, X1, X2, Y, sortIndex = initVec()
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
    e = 280
    r = 0.01
    n = 0.99 # n belong to [0.95, 0.99]
    percep = 0.1
    # a = int(percep * length)
    a = 2
    i = 1

    I = np.identity(length)

    iteration = 0

    # while abs(LA.norm(np.dot(A, x), 0) - a) >= e:
    while iteration < 10:
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

        iteration += 1

    indexVec, unSortedLabelVec = nonzeroValue2(origVec, np.dot(A, x))

    # unSortedLabelVec = nonzeroValue(origVec, A.dot(x), a)

    labelVec = np.zeros(len(unSortedLabelVec))

    for i in range(len(unSortedLabelVec)):
        labelVec[sortIndex[i]] = unSortedLabelVec[i]
    # print("unSortedLabelVec: ", unSortedLabelVec)
    # print("labelVec: ", labelVec)
    print("counter unsortedLableVec and labelVec: ", Counter(unSortedLabelVec), Counter(labelVec))

    # elementIndex = np.zeros(len(sortIndex))
    # for i in range(len(sortIndex)):
    #     elementIndex[sortIndex[i]] = i
    # print("elementIndex: ", elementIndex)
    # labelVec = np.zeros(len(unSortedLabelVec))
    # for i in range(len(unSortedLabelVec)):
    #     labelVec[elementIndex[i]] = unSortedLabelVec[i]
    # print("labelVec: ", labelVec)

    # print(len(indexVec), labelVec)
    # print(origVec)
    # print([i for i in np.dot(A, x) if i > 1])
    # print(np.dot(A, x))
    # print(LA.norm(np.dot(A,x), 1))
    # print(LA.norm(np.dot(A,x), 1))
    # print(i)
    return origVec, labelVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec

def ADMM2(sortedVector, sortIndex, a):
    # argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a

    # original input value set
    # sortedVector, label, X, X1, X2, Y, sortIndex = initVec()
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
    e = 280
    r = 0.01
    n = 0.99 # n belong to [0.95, 0.99]
    percep = 0.1
    # a = int(percep * length)
    i = 1

    # I = np.identity(length)

    iteration = 0

    I = identity(length, format='lil')
    # left = numpy.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
    A_T = A.transpose()
    temp = I + A_T.dot(A) * (1/r)

    # while abs(LA.norm(np.dot(A, x), 0) - a) >= e:
    while iteration < 100:
        # left = numpy.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
        # right = np.dot(1, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
        # # x = np.dot(left.toarray(), right)
        # x = np.dot(left, right)

        right = np.dot(1, x1) + np.dot(1/r, A_T.dot(v-w))
        x = spsolve(temp, right)

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

        iteration += 1

    # indexVec, unSortedLabelVec = nonzeroValue2(origVec, np.dot(A, x))
    unSortedLabelVec = nonzeroValue(origVec, np.dot(A, x), a)

    # unSortedLabelVec = nonzeroValue(origVec, A.dot(x), a)

    labelVec = np.zeros(len(unSortedLabelVec))
    for i in range(len(unSortedLabelVec)):
        labelVec[sortIndex[i]] = unSortedLabelVec[i]
    # print("unSortedLabelVec: ", unSortedLabelVec)
    # print("labelVec: ", labelVec)
    print("counter unsortedLableVec and labelVec: ", Counter(unSortedLabelVec), Counter(labelVec))

    # elementIndex = np.zeros(len(sortIndex))
    # for i in range(len(sortIndex)):
    #     elementIndex[sortIndex[i]] = i
    # print("elementIndex: ", elementIndex)
    # labelVec = np.zeros(len(unSortedLabelVec))
    # for i in range(len(unSortedLabelVec)):
    #     labelVec[elementIndex[i]] = unSortedLabelVec[i]
    # print("labelVec: ", labelVec)

    # print(origVec)
    # print([i for i in np.dot(A, x) if i > 1])
    # print(np.dot(A, x))
    # print(LA.norm(np.dot(A,x), 1))
    # print(LA.norm(np.dot(A,x), 1))
    # print(i)
    return labelVec

def ADMM3(sortedVector, sortIndex, a, iter_time):
    # argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a

    # original input value set
    # sortedVector, label, X, X1, X2, Y, sortIndex = initVec()
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
    e = 280
    r = 0.01
    n = 0.99 # n belong to [0.95, 0.99]
    percep = 0.1
    # a = int(percep * length)
    i = 1

    I = np.identity(length)

    iteration = 0

    left = numpy.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))

    # while abs(LA.norm(np.dot(A, x), 0) - a) >= e:
    while iteration < iter_time:
        right = np.dot(1, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
        # x = np.dot(left.toarray(), right)
        x = np.dot(left, right)

        z = np.dot(A, x) + w
        if LA.norm(z, 0) <= a:
            v = deepcopy(z)
            # print("z<a")
        else:
            z_abs = abs(z)
            z_sort = np.argsort(z_abs)[::-1]
            v = np.zeros(length-1)
            for i in range(a):
                index = z_sort[i]
                v[index] = z[index]

        w = w + np.dot(A, x) - v
        # r = r * n

        iteration += 1

    # cost = LA.norm(x-sortedVector)
    indexVec, unSortedLabelVec = nonzeroValue2(origVec, np.dot(A, x))

    # unSortedLabelVec = nonzeroValue(origVec, A.dot(x), a)

    labelVec = np.zeros(len(unSortedLabelVec))
    for i in range(len(unSortedLabelVec)):
        labelVec[sortIndex[i]] = unSortedLabelVec[i]
    # print("unSortedLabelVec: ", unSortedLabelVec)
    # print("labelVec: ", labelVec)
    # print("counter unsortedLableVec and labelVec: ", Counter(unSortedLabelVec), Counter(labelVec))

    # elementIndex = np.zeros(len(sortIndex))
    # for i in range(len(sortIndex)):
    #     elementIndex[sortIndex[i]] = i
    # print("elementIndex: ", elementIndex)
    # labelVec = np.zeros(len(unSortedLabelVec))
    # for i in range(len(unSortedLabelVec)):
    #     labelVec[elementIndex[i]] = unSortedLabelVec[i]
    # print("labelVec: ", labelVec)

    # print(len(indexVec), labelVec)
    # print(origVec)
    # print([i for i in np.dot(A, x) if i > 1])
    # print(np.dot(A, x))
    # print(LA.norm(np.dot(A,x), 1))
    # print(LA.norm(np.dot(A,x), 1))
    # print(i)
    # return origVec, labelVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec

    #return an unsort vector by x and sortIndex
    unsorted_x_i = np.zeros(len(unSortedLabelVec))
    for i in range(len(unSortedLabelVec)):
        unsorted_x_i[sortIndex[i]] = x[i]
    return labelVec, unsorted_x_i

# wish to plot re-color point and find different label with original point
# method by find k-largest values
def nonzeroValue(origVec, gapVec, a):
    indexVec = []
    sortedGapVec = np.argsort(gapVec)
    for i in range(a):
        for k in range(len(sortedGapVec)):
            if sortedGapVec[k] == len(gapVec)-i-1:
                indexVec.append(k)

    sortedIndexVec = np.sort(indexVec)
    sortedIndexVec = np.append(sortedIndexVec, len(gapVec)+1)

    pre = 0 # last index, init = 0
    label = 1
    labelVec = []
    for i in sortedIndexVec:
        labelVecPiece = np.ones(i-pre) * label
        labelVec = np.append(labelVec, labelVecPiece)
        pre = i
        label = label + 1

    print("labelVec", labelVec)
    print("labelVec shape", labelVec.shape)
    return labelVec

def nonzeroValue2(origVec, gapVec):
    indexVec = [0]
    threshold = 0
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
    # print("indexVec", indexVec[:-1])

    for i in range(len(indexVec)):
        if indexVec[i] == 1:
            indexVec[preIndex:i] = origVec[preIndex:i]
            labelVecPiece = [labelVal for i in indexVec[preIndex:i]]
            unSortedLabelVec.extend(labelVecPiece)
            labelVal = labelVal + 1
            preIndex = i

    return indexVec[:-1], unSortedLabelVec


def plotVectorByADMM1():
    # x1, indexVec, labelVec, x, label, X, X1, X2, Y, sortIndex = ADMM2()
    x1, labelVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec = ADMM1()
    plt.figure()
    leng = len(x1)
    plt.scatter(x1, np.ones(leng))
    plt.scatter(x, 2 * np.ones(leng))
    plt.show()

def plotVectorByADMM():
    # x1, indexVec, labelVec, x, label, X, X1, X2, Y, sortIndex = ADMM2()
    x1, labelVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec = ADMM2()
    print("labelVec and Y: ", labelVec, Y)

    # print(max(set(labelVec), key=labelVec.count))
    # print(Counter(labelVec))
    print(Counter(unSortedLabelVec))

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

    # plt.scatter(indexVec, Z, c=labelVec, s=10)
    plt.scatter(x1, Y, c=labelVec, s=10)
    plt.ylim(-2,4)
    # plt.xlim(-200, 200)
    plt.title('after ADMM optimization, a, all 90 points')

    # plt.figure(2, figsize=(8, 6))
    # plt.scatter(x1, Z, c=origLabel, s=10)
    # # plt.scatter(x1[19], Z[19], c='r', marker='^', s=15)
    # # plt.scatter(x1[20], Z[20], c='r', marker='^', s=15)
    # plt.xlabel('x-axis')
    # plt.ylim(-2,4)
    # plt.xlim(-200, 200)
    # plt.title('original plot, a, all 90 points')
    matplotlib.interactive(False)
    plt.show()

def plotDataByADMM():
    x1, labelVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec = ADMM2()
    # print("labelVec and Y: ", labelVec, Y)

    # print(max(set(labelVec), key=labelVec.count))
    # print(Counter(labelVec))

    # print("sortIndex: ", sortIndex)
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

    # print("original X1 coordinates: ", X1)
    # print("argsort X1: ", np.argsort(X1) )
    pre = np.argsort(X1)

    plt.scatter(X1, X2, c=unSortedLabelVec, s=10)
    # print("unSortedLabelVec", unSortedLabelVec)
    # print("labelVec", labelVec)
    # print("origLabel", origLabel)

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
    # for boundryVal, color in zip(boundry, colors[:len(boundry)]):
    #     boundryIndex = sortIndex[boundryVal]
    #     # for boundryIndex in range(len(sortIndex)):
    #     #     if sortIndex[boundryIndex] == boundryVal:
    #     plt.scatter(X1[boundryIndex], X2[boundryIndex], c=color, marker='+')

    plt.ylim(0, 100)
    plt.xlim(0, 100)
    plt.title('figure 1 : boundry condition, a=3, all 300 points')
    plt.show()

    # plt.figure(2, figsize=(8, 6))
    # plt.scatter(X1, X2, c=origLabel, s=10)

    # # # # plot triangle shape in boundry
    # # # length = len(X1)
    # # # for i in range(1,4,1):
    # # #     number = i * length/3 -1
    # # #     for j,num in enumerate(sortIndex):
    # # #         if num == number:
    # # #             pos = j
    # # #             plt.scatter(x1[pos], x2[pos], c='r', marker='^', s=15)

    # plt.xlabel('x-axis')
    # # plt.ylim(-2,4)
    # # plt.xlim(-100, 100)
    # # plt.ylim(0, 50)
    # # plt.xlim(0, 50)
    # plt.title('ADMM optimization, a=10, all 90 points')
    matplotlib.interactive(False)
    plt.show()

def iterative_choose_partition1():
    sortedVector1, sortedVector2, sortedVector3, sortedVector4, sortedVector5, sortIndex1, sortIndex2, sortIndex3, sortIndex4, sortIndex5, vector0, vector1, vector2, vector3, vector4, X1, X2 = initVec2()
    vector = [vector0, vector1, vector2, vector3, vector4]
    vector = np.array(vector)
    print("------------vector shape------------------", vector.shape)
    index_vector = np.zeros(np.shape(vector0))
    label_vector = np.zeros(np.shape(vector0))
    iteration = 0
    plot_color_label = []
    # two situations will stop the iteration, first situation is isolated points appereas, second situation: cost function does not improve
    # while labeled_list_number == len(label_vector):
    while iteration < 1:
        # partition by label_vector
        labels = []
        for label in label_vector:
            if label not in labels:
                labels.append(label)
        index_vector = []
        for label in labels:
            index = [ i for i in range(len(label_vector)) if label_vector[i] == label]
            index_vector.append(index)

        # for index in index_vector:
        for i in range(len(index_vector)):
            index = index_vector[i]
            # compared in each eigenvector direction
            cost_vector = []
            original_cor = deepcopy(vector[:, index])
            for i in range(len(vector)):
                unsortVector = vector[i][index]
                sortIndex = np.argsort(unsortVector)
                sortedVector = [unsortVector[i] for i in sortIndex]
                labelVector, x = ADMM3(sortedVector, sortIndex, 1, 5000)
                after_cor = deepcopy(vector[:, index])
                after_cor[i] = x

                # calculate cost in each eigenvector's direction
                i_labels = []
                for label in labelVector:
                    if label not in i_labels:
                        i_labels.append(label)
                # print("!!!!!!!!!!!!!!!!!!!!!!!i_index_vector!!!!!!!!!!!!!!!!!!!!!!!!!", i_index_vector)
                i_index_vector = []
                for label in i_labels:
                    index_2 = [ i for i in range(len(labelVector)) if labelVector[i] == label]
                    i_index_vector.append(index_2)

                # print("!!!!!!!!!!!!!!!!!!!!!!!i_index_vector!!!!!!!!!!!!!!!!!!!!!!!!!", i_index_vector)
                cost = 0
                for index_3 in i_index_vector:
                    mean_point = np.mean(after_cor[:, index_3], axis=1)
                    print("mean_point", mean_point, mean_point.shape)
                    # for i in range(len(after_cor[:, index_3])):
                    for i in index_3:
                        original_cor[:, index_3]
                        # cost += LA.norm(original_cor[:, index_3][:, i]-mean_point)
                        cost += LA.norm(original_cor[:, i]-mean_point)
                    # print("after_cor shape-----------", after_cor.shape)

                cost_vector.append(cost)

            min_index = np.argmin(cost_vector)
            print("!!!!!!!!!!!!!!!!!!!!!!!min_index!!!!!!!!!!!!!!!!!!!!!!!!!", min_index)
            print("!!!!!!!!!!!!!!!!!!!!!!!cost_vector!!!!!!!!!!!!!!!!!!!!!!!!!", cost_vector)
            unsortVector = vector[min_index][index]
            sortIndex = np.argsort(unsortVector)
            sortedVector = [unsortVector[i] for i in sortIndex]
            labelVector, x = ADMM3(sortedVector, sortIndex, 1, 10000)
            # update label
            # max_label = np.amax(labels)
            max_label = np.amax(label_vector)
            non_difference_labelVector = [max_label+labelVector[i] for i in range(len(labelVector))]
            label_vector[index] = non_difference_labelVector

        matplotlib.interactive(False)
        # plot 3-d eigen-space
        fig = plt.figure(figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        # ax.scatter(vector0, vector1, vector2, c=label_vector, cmap=plt.cm.Paired)
        ax.scatter(vector0, vector1, vector2, c=label_vector)
        ax.set_title("color eigen-space")
        iteration += 1
        # print("label_vector", label_vector)
        print("counter label_vector: ", Counter(label_vector))

    plt.show()

def iterative_choose_partition2():
    sortedVector1, sortedVector2, sortedVector3, sortedVector4, sortedVector5, sortIndex1, sortIndex2, sortIndex3, sortIndex4, sortIndex5, vector0, vector1, vector2, vector3, vector4, X1, X2 = initVec2()
    vector = [vector0, vector1, vector2, vector3, vector4]
    vector = np.array(vector)
    print("------------vector shape------------------", vector.shape)
    # index_vector = np.zeros(np.shape(vector0))
    label_vector = np.zeros(np.shape(vector0))
    iteration = 0
    # plot_color_label = []

    # cut_point_list = np.zeros(len(vector))
    cut_point_list = []
    for i in range(len(vector)):
        cut_point_list.append(0)

    # when cost function does not improve, then iteration stops.
    while iteration < 3:
        # add previous cut result, the previous label vector
        # labels = []
        # for label in label_vector:
        #     if label not in labels:
        #         labels.append(label)
        # index_vector = []
        # for label in labels:
        #     index = [ i for i in range(len(label_vector)) if label_vector[i] == label]
        #     index_vector.append(index)

        # find best cut point along eigenvector's direction
        cost_vector = []
        for i in range(len(vector)):
            original_cor = deepcopy(vector)
            # read cut points number in this eigenvector's direction
            cut_point_number = cut_point_list[i]

            unsortVector = vector[i]
            sortIndex = np.argsort(unsortVector)
            sortedVector = [unsortVector[i] for i in sortIndex]
            labelVector, x = ADMM3(sortedVector, sortIndex, cut_point_number+1, 5000)
            after_cor = deepcopy(vector)
            after_cor[i] = x

            # calculate cost in each eigenvector's direction
            i_labels = []
            for label in labelVector:
                if label not in i_labels:
                    i_labels.append(label)
            i_index_vector = []
            for label in i_labels:
                index_2 = [ i for i in range(len(labelVector)) if labelVector[i] == label]
                i_index_vector.append(index_2)
                # print("!!!!!!!!!!!!!!!!!!!!!!!i_index_vector!!!!!!!!!!!!!!!!!!!!!!!!!", i_index_vector)
            cost = 0
            for index_3 in i_index_vector:
                mean_point = np.mean(after_cor[:, index_3], axis=1)
                print("mean_point", mean_point, mean_point.shape)
                # for i in range(len(after_cor[:, index_3])):
                for i in index_3:
                    # cost += LA.norm(original_cor[:, index_3][:, i]-mean_point)
                    cost += LA.norm(original_cor[:, i]-mean_point)
                    # print("after_cor shape-----------", after_cor.shape)
            cost_vector.append(cost)

        min_index = np.argmin(cost_vector)
        cut_point_list[min_index] = cut_point_list[min_index] + 1
        iteration += 1
        print("!!!!!!!!!!!!!!!!!!!!!!!min_index!!!!!!!!!!!!!!!!!!!!!!!!!", min_index)
        print("!!!!!!!!!!!!!!!!!!!!!!!cost_vector!!!!!!!!!!!!!!!!!!!!!!!!!", cost_vector)

    # calculate intersecton of label vectors
    labelVectorSet = []
    for i in range(len(vector)):
        if cut_point_list[i] > 0:
            unsortVector = vector[i]
            sortIndex = np.argsort(unsortVector)
            sortedVector = [unsortVector[i] for i in sortIndex]
            labelVector, x = ADMM3(sortedVector, sortIndex, cut_point_list[i], 5000)
            labelVectorSet.append(labelVector)

    labelVectorSet = np.asarray(labelVectorSet)
    element_vector = []
    for i in range(len(labelVectorSet[0])):
        element = labelVectorSet[:, i]
        if element not in element_vector:
            element_vector.append(element)
    element_label = []
    for i in range(len(labelVectorSet[0])):
        for j in range(len(element_vector)):
            if element_vector[j] == labelVectorSet[:, i]:
                element_label.append(j)
    # test elemenet_label numbers is correct
    if len(element_label) != len(labelVectorSet[0]):
        print("----------------------label numbers wrong----------------------")

    matplotlib.interactive(False)
    # plot 3-d eigen-space
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    # ax.scatter(vector0, vector1, vector2, c=label_vector, cmap=plt.cm.Paired)
    # ax.scatter(vector0, vector1, vector2, c=label_vector)
    ax.scatter(vector0, vector1, vector2, c=element_label)
    ax.set_title("color eigen-space")
    # iteration += 1
    # print("label_vector", label_vector)
    # print("counter label_vector: ", Counter(label_vector))
    print("counter label_vector: ", Counter(element_label))

    plt.show()

def iterative_choose_partition3():
    sortedVector1, sortedVector2, sortedVector3, sortedVector4, sortedVector5, sortIndex1, sortIndex2, sortIndex3, sortIndex4, sortIndex5, vector0, vector1, vector2, vector3, vector4, X1, X2 = initVec2()
    vector = [vector0, vector1, vector2, vector3, vector4]
    vector = np.array(vector)
    print("------------vector shape------------------", vector.shape)
    # index_vector = np.zeros(np.shape(vector0))
    label_vector = np.zeros((1, len(vector0)))

    # label_vector = []
    # for i in range(len(vector0)):
    #     label_vector.append(0)

    iteration = 0
    # plot_color_label = []

    cut_point_list = []
    for i in range(len(vector)):
        cut_point_list.append(0)

    best_cost_vector = []

    # when cost function does not improve, then iteration stops.
    while iteration < 10:
        # find best cut point along eigenvector's direction
        cost_vector = []
        for i in range(len(vector)):
            original_cor = deepcopy(vector)
            # read cut points number in this eigenvector's direction
            cut_point_number = cut_point_list[i]

            unsortVector = vector[i]
            sortIndex = np.argsort(unsortVector)
            sortedVector = [unsortVector[i] for i in sortIndex]
            labelVector, x = ADMM3(sortedVector, sortIndex, cut_point_number+1, 5000)
            # intersection with previous labels results
            predict_label_vector = deepcopy(label_vector)
            # print("-----------------predict_label_vector----------", predict_label_vector)
            # predict_label_vector = predict_label_vector.append(labelVector)
            predict_label_vector = np.append(predict_label_vector, [labelVector], axis=0)

            # print("-----------------predict_label_vector----------", predict_label_vector)
            element_vector = []
            for i in range(len(vector[0])):
                element = predict_label_vector[:, i]
                # print("------------element-------------", element)
                if element.tolist() not in element_vector:
                    element_vector.append(element.tolist())
            element_label = []
            for i in range(len(vector[0])):
                for j in range(len(element_vector)):
                    if element_vector[j] == predict_label_vector[:, i].tolist():
                        element_label.append(j)
            # test if element_label number is correct
            # if len(element_label) != len(predict_label_vector):
            #     print("------------label numbers wrong-----------")
            element_index = []
            for element_label_piece in element_label:
                ith_element_index = [ i for i in range(len(element_label)) if element_label[i] == element_label_piece]
                element_index.append(ith_element_index)

            # after_cor = deepcopy(vector)
            # after_cor[i] = x

            # calculate the cost
            cost = 0
            for index_3 in element_index:
                # mean_point = np.mean(after_cor[:, index_3], axis=1)
                mean_point = np.mean(original_cor[:, index_3], axis=1)
                # print("mean_point", mean_point, mean_point.shape)
                # for i in range(len(after_cor[:, index_3])):
                for i in index_3:
                    # original_cor[:, index_3]
                    # cost += LA.norm(original_cor[:, index_3][:, i]-mean_point)
                    cost += LA.norm(original_cor[:, i]-mean_point)
                    # print("after_cor shape-----------", after_cor.shape)

            cost_vector.append(cost)
        min_index = np.argmin(cost_vector)
        best_cost = cost_vector[min_index]
        best_cost_vector.append(best_cost)
        print("!!!!!!!!!!!!!!!!!!!!!!!min_index!!!!!!!!!!!!!!!!!!!!!!!!!", min_index)
        print("!!!!!!!!!!!!!!!!!!!!!!!cost_vector!!!!!!!!!!!!!!!!!!!!!!!!!", cost_vector)

        unsortVector = vector[min_index]
        sortIndex = np.argsort(unsortVector)
        sortedVector = [unsortVector[i] for i in sortIndex]
        labelVector, x = ADMM3(sortedVector, sortIndex, cut_point_list[min_index]+1, 5000)
        # update label_vector and cut_point_list
        cut_point_list[min_index] = cut_point_list[min_index] + 1
        print("-------------cut_point_list---------------", cut_point_list)
        # label_vector.append(labelVector)
        label_vector = np.append(label_vector, [labelVector], axis=0)

        iteration += 1

        # every time show figures
        element_vector = []
        for i in range(len(vector[0])):
            element = label_vector[:, i]
            if element.tolist() not in element_vector:
                element_vector.append(element.tolist())
        element_label = []
        for i in range(len(vector[0])):
            for j in range(len(element_vector)):
                if element_vector[j] == label_vector[:, i].tolist():
                    element_label.append(j)
        matplotlib.interactive(False)
        # plot 3-d eigen-space
        fig = plt.figure(figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        # ax.scatter(vector0, vector1, vector2, c=label_vector, cmap=plt.cm.Paired)
        # ax.scatter(vector0, vector1, vector2, c=label_vector)
        ax.scatter(vector0, vector1, vector2, c=element_label)
        ax.set_title("color eigen-space")
        # iteration += 1
        # print("label_vector", label_vector)
        # print("counter label_vector: ", Counter(label_vector))
        print("counter label_vector: ", Counter(element_label))

    # final element_label
    # element_vector = []
    # for i in range(len(vector[0])):
    #     element = label_vector[:, i]
    #     if element.tolist() not in element_vector:
    #         element_vector.append(element.tolist())
    # element_label = []
    # for i in range(len(vector[0])):
    #     for j in range(len(element_vector)):
    #         if element_vector[j] == label_vector[:, i].tolist():
    #             element_label.append(j)

    # matplotlib.interactive(False)
    # # plot 3-d eigen-space
    # fig = plt.figure(figsize=(8, 6))
    # ax = Axes3D(fig, elev=-150, azim=110)
    # # ax.scatter(vector0, vector1, vector2, c=label_vector, cmap=plt.cm.Paired)
    # # ax.scatter(vector0, vector1, vector2, c=label_vector)
    # ax.scatter(vector0, vector1, vector2, c=element_label)
    # ax.set_title("color eigen-space")
    # # iteration += 1
    # # print("label_vector", label_vector)
    # # print("counter label_vector: ", Counter(label_vector))
    # print("counter label_vector: ", Counter(element_label))


    # plot two_difference_vector
    matplotlib.interactive(False)
    one_difference_vector = np.diff(best_cost_vector)
    two_difference_vector = np.diff(one_difference_vector)
    fig = plt.figure()
    plt.plot(np.arange(len(two_difference_vector)), two_difference_vector)

    fig = plt.figure()
    plt.title("one difference vector")
    plt.plot(np.arange(len(one_difference_vector)), -one_difference_vector)

    plt.show()

# using L1 norm to measure the value of the gap
def iterative_choose_partition_L1_norm():
    sortedVector1, sortedVector2, sortedVector3, sortedVector4, sortedVector5, sortIndex1, sortIndex2, sortIndex3, sortIndex4, sortIndex5, vector0, vector1, vector2, vector3, vector4, X1, X2 = initVec2()
    vector = [vector0, vector1, vector2, vector3, vector4]
    vector = np.array(vector)
    print("------------vector shape------------------", vector.shape)
    # index_vector = np.zeros(np.shape(vector0))
    label_vector = np.zeros((1, len(vector0)))

    # label_vector = []
    # for i in range(len(vector0)):
    #     label_vector.append(0)

    iteration = 0
    # plot_color_label = []

    cut_point_list = []
    for i in range(len(vector)):
        cut_point_list.append(0)

    best_cost_vector = []

    # when cost function does not improve, then iteration stops.
    while iteration < 3:
        # find best cut point along eigenvector's direction
        cost_vector = []
        # loop in each eigenvector's direction
        for i in range(len(vector)):
            original_cor = deepcopy(vector)
            # read cut points number in this eigenvector's direction
            cut_point_number = cut_point_list[i]

            unsortVector = vector[i]
            sortIndex = np.argsort(unsortVector)
            sortedVector = [unsortVector[i] for i in sortIndex]
            labelVector, x = ADMM3(sortedVector, sortIndex, cut_point_number+1, 5000)
            # intersection with previous labels results
            predict_label_vector = deepcopy(label_vector)
            # print("-----------------predict_label_vector----------", predict_label_vector)
            # predict_label_vector = predict_label_vector.append(labelVector)
            predict_label_vector = np.append(predict_label_vector, [labelVector], axis=0)

            # print("-----------------predict_label_vector----------", predict_label_vector)
            element_vector = []
            for i in range(len(vector[0])):
                element = predict_label_vector[:, i]
                # print("------------element-------------", element)
                if element.tolist() not in element_vector:
                    element_vector.append(element.tolist())
            element_label = []
            for i in range(len(vector[0])):
                for j in range(len(element_vector)):
                    if element_vector[j] == predict_label_vector[:, i].tolist():
                        element_label.append(j)
            # test if element_label number is correct
            # if len(element_label) != len(predict_label_vector):
            #     print("------------label numbers wrong-----------")
            element_index = []
            for element_label_piece in element_label:
                ith_element_index = [ i for i in range(len(element_label)) if element_label[i] == element_label_piece]
                element_index.append(ith_element_index)

            # after_cor = deepcopy(vector)
            # after_cor[i] = x

            # calculate the cost
            # cost = 0
            # for index_3 in element_index:
            #     # mean_point = np.mean(after_cor[:, index_3], axis=1)
            #     mean_point = np.mean(original_cor[:, index_3], axis=1)
            #     # print("mean_point", mean_point, mean_point.shape)
            #     # for i in range(len(after_cor[:, index_3])):
            #     for i in index_3:
            #         # original_cor[:, index_3]
            #         # cost += LA.norm(original_cor[:, index_3][:, i]-mean_point)
            #         cost += LA.norm(original_cor[:, i]-mean_point)
            #         # print("after_cor shape-----------", after_cor.shape)

            # cost_vector.append(cost)

            # calculate the L1-norm cost
            gap_total = 0
            for index in element_index:
                # find biggest L1-norm gap in these points along every eigenvector's direction
                cluster_cor = original_cor[:, index]
                for direction in range(len(cluster_cor)):
                    max_index = np.argmax(cluster_cor[direction])
                    min_index = np.argmin(cluster_cor[direction])
                    gap = cluster_cor[direction][max_index] - cluster_cor[direction][min_index]
                    print("----------------gap value in each cluster and every direction", gap)
                    gap_total += gap
            cost_vector.append(gap_total)

        min_index = np.argmin(cost_vector)
        best_cost = cost_vector[min_index]
        best_cost_vector.append(best_cost)
        print("!!!!!!!!!!!!!!!!!!!!!!!min_index!!!!!!!!!!!!!!!!!!!!!!!!!", min_index)
        print("!!!!!!!!!!!!!!!!!!!!!!!cost_vector!!!!!!!!!!!!!!!!!!!!!!!!!", cost_vector)

        unsortVector = vector[min_index]
        sortIndex = np.argsort(unsortVector)
        sortedVector = [unsortVector[i] for i in sortIndex]
        labelVector, x = ADMM3(sortedVector, sortIndex, cut_point_list[min_index]+1, 5000)
        # update label_vector and cut_point_list
        cut_point_list[min_index] = cut_point_list[min_index] + 1
        print("-------------cut_point_list---------------", cut_point_list)
        # label_vector.append(labelVector)
        label_vector = np.append(label_vector, [labelVector], axis=0)

        iteration += 1

        # every time show figures
        element_vector = []
        for i in range(len(vector[0])):
            element = label_vector[:, i]
            if element.tolist() not in element_vector:
                element_vector.append(element.tolist())
        element_label = []
        for i in range(len(vector[0])):
            for j in range(len(element_vector)):
                if element_vector[j] == label_vector[:, i].tolist():
                    element_label.append(j)
        matplotlib.interactive(False)
        # plot 3-d eigen-space
        fig = plt.figure(figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        # ax.scatter(vector0, vector1, vector2, c=label_vector, cmap=plt.cm.Paired)
        # ax.scatter(vector0, vector1, vector2, c=label_vector)
        ax.scatter(vector0, vector1, vector2, c=element_label)
        ax.set_title("color eigen-space")
        # iteration += 1
        # print("label_vector", label_vector)
        # print("counter label_vector: ", Counter(label_vector))
        print("counter label_vector: ", Counter(element_label))

    # final element_label
    # element_vector = []
    # for i in range(len(vector[0])):
    #     element = label_vector[:, i]
    #     if element.tolist() not in element_vector:
    #         element_vector.append(element.tolist())
    # element_label = []
    # for i in range(len(vector[0])):
    #     for j in range(len(element_vector)):
    #         if element_vector[j] == label_vector[:, i].tolist():
    #             element_label.append(j)

    # matplotlib.interactive(False)
    # # plot 3-d eigen-space
    # fig = plt.figure(figsize=(8, 6))
    # ax = Axes3D(fig, elev=-150, azim=110)
    # # ax.scatter(vector0, vector1, vector2, c=label_vector, cmap=plt.cm.Paired)
    # # ax.scatter(vector0, vector1, vector2, c=label_vector)
    # ax.scatter(vector0, vector1, vector2, c=element_label)
    # ax.set_title("color eigen-space")
    # # iteration += 1
    # # print("label_vector", label_vector)
    # # print("counter label_vector: ", Counter(label_vector))
    # print("counter label_vector: ", Counter(element_label))


    # plot two_difference_vector
    matplotlib.interactive(False)
    one_difference_vector = np.diff(best_cost_vector)
    two_difference_vector = np.diff(one_difference_vector)
    fig = plt.figure()
    plt.plot(np.arange(len(two_difference_vector)), two_difference_vector)

    fig = plt.figure()
    plt.title("one difference vector")
    plt.plot(np.arange(len(one_difference_vector)), -one_difference_vector)

    plt.show()

# transfer matrix to labels, using in floyd.py file
def matrix_to_label(matrix, vector_space_label=None):
    sortedVector1, sortedVector2, sortedVector3, sortedVector4, sortedVector5, sortIndex1, sortIndex2, sortIndex3, sortIndex4, sortIndex5, vector0, vector1, vector2, vector3, vector4, X1, X2 = initVec2(matrix)

    # man-labor to construct vector_space_label
    # man_labor = [[0, 1, 29],
    #              [2, 3, 4, 22, 30, 31],
    #              [5, 6, 19, 21, 24, 32, 35],
    #              [7, 8, 9, 10, 23, 33],
    #              [11, 12, 13, 14, 15, 25, 26, 34, 37],
    #              [16, 17, 18, 20, 27, 28, 36],
    #              [42, 53, 73, 79, 96, 97, 109, 111, 123, 125, 134],
    #              [43, 45, 49, 50, 61, 68, 74, 80, 89, 91, 92, 102, 105, 108, 114, 115, 119, 121, 127, 130, 135],
    #              [47, 46, 48, 69, 65, 57, 71, 82, 83, 87, 93, 98, 107, 103, 116, 120, 128, 133],
    #              [39, 41, 44, 56, 62, 67, 85],
    #              [58, 51, 54, 63, 64, 72, 81, 90, 88, 100, 112, 126, 124, 131, 55, 60, 52, 59, 40, 66, 70, 75, 78, 77, 76, 94, 95, 84, 86, 99, 106, 103, 101, 117, 113, 110, 122, 129, 118, 132],
    #              [148, 159, 176],
    #              [145, 146, 147, 153, 158, 149, 164, 165, 164, 165, 181, 183, 166, 169],
    #              [137, 141, 142, 152, 157, 162, 160, 161, 170, 171, 179, 180, 184],
    #              [136],
    #              [143, 138, 154, 173, 185, 174],
    # [139, 144, 140, 151, 155, 150, 156, 163, 167, 168, 172, 175, 182, 178, 177]]

    # l = 0
    # new_vector_space_label = np.zeros(len(vector_space_label))
    # man_labor_label = np.asarray(man_labor)
    # for i in range(len(man_labor_label)):
    #     n_vector = man_labor_label[i]
    #     for j in range(len(n_vector)):
    #         k = n_vector[j]
    #         new_vector_space_label[k] = i
    #         # new_vector_space_label[man_labor_label[i, j]] = i

    # # plot eigenvector space with vector_space_label
    # # matplotlib.interactive(True)
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=110)
    # ax.scatter(vector0, vector1, vector2, c=new_vector_space_label, cmap="magma")
    # c = np.zeros(len(vector0))
    # # c[:50] = 1
    # # ax.scatter(vector0, vector1, vector2, c=c)
    # # ax.scatter(vector0, vector1, vector2)
    # ax.set_title("vector space")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    # # MDS test
    # n_points = len(vector0)
    # n_neighbors = 15
    # n_components = 2
    # color = deepcopy(new_vector_space_label)
    # X = [[vector0[i], vector1[i], vector2[i]] for i in range(len(vector0))]
    # X = np.asarray(X)

    # mds = manifold.MDS(max_iter = 100, n_init = 1)
    # Y = mds.fit_transform(X)
    # plt.figure()
    # plt.scatter(Y[:, 0], Y[:, 1], c=vector_space_label, cmap=plt.cm.Spectral)
    # plt.show()


    # fig = plt.figure(figsize=(15, 8))
    # plt.suptitle("Manifold Learning with %i points, %i neighbors(arg)"
    #              % (n_points, n_neighbors), fontsize=14)


    # ax = fig.add_subplot(251, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    # ax.view_init(4, -72)

    # methods = ['standard', 'ltsa', 'hessian', 'modified']
    # labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

    # for i, method in enumerate(methods):
    #     t0 = time()
    #     Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
    #                                         eigen_solver='auto',
    #                                         method=method).fit_transform(X)
    #     t1 = time()
    #     print("%s: %.2g sec" % (methods[i], t1 - t0))

    #     ax = fig.add_subplot(252 + i)
    #     plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    #     plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
    #     ax.xaxis.set_major_formatter(NullFormatter())
    #     ax.yaxis.set_major_formatter(NullFormatter())
    #     plt.axis('tight')

    # t0 = time()
    # Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    # t1 = time()
    # print("Isomap: %.2g sec" % (t1 - t0))
    # ax = fig.add_subplot(257)
    # plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    # plt.title("Isomap (%.2g sec)" % (t1 - t0))
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')

    # t0 = time()
    # mds = manifold.MDS(n_components, max_iter=100, n_init=1)
    # Y = mds.fit_transform(X)
    # t1 = time()
    # print("MDS: %.2g sec" % (t1 - t0))
    # ax = fig.add_subplot(258)
    # plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    # plt.title("MDS (%.2g sec)" % (t1 - t0))
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')


    # t0 = time()
    # se = manifold.SpectralEmbedding(n_components=n_components,
    #                                 n_neighbors=n_neighbors)
    # Y = se.fit_transform(X)
    # t1 = time()
    # print("SpectralEmbedding: %.2g sec" % (t1 - t0))
    # ax = fig.add_subplot(259)
    # plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    # plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')

    # t0 = time()
    # tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    # Y = tsne.fit_transform(X)
    # t1 = time()
    # print("t-SNE: %.2g sec" % (t1 - t0))
    # ax = fig.add_subplot(2, 5, 10)
    # plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    # plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')

    # plt.show()


    ''' previous method, find L2 or L1 distance to choose which eigenvector

    # vector = [vector0, vector1, vector2, vector3, vector4]
    # vector = [vector1, vector2]
    vector = [vector1]

    vector = np.array(vector)
    print("------------vector shape------------------", vector.shape)
    # index_vector = np.zeros(np.shape(vector0))
    label_vector = np.zeros((1, len(vector0)))

    # label_vector = []
    # for i in range(len(vector0)):
    #     label_vector.append(0)

    iteration = 0
    # plot_color_label = []

    cut_point_list = []
    for i in range(len(vector)):
        cut_point_list.append(0)

    best_cost_vector = []
    cost_change_rate = 1
    cost_change_difference_vector = []

    # when cost function turning point does not improve, and iterations < 10, then iteration stops.

    while iteration < 4:
    # while cost_change_rate > 0.4:
        # find best cut point along eigenvector's direction
        cost_vector = []
        for i in range(len(vector)):
            original_cor = deepcopy(vector)
            # read cut points number in this eigenvector's direction
            cut_point_number = cut_point_list[i]

            unsortVector = vector[i]
            sortIndex = np.argsort(unsortVector)
            sortedVector = [unsortVector[i] for i in sortIndex]
            labelVector, x = ADMM3(sortedVector, sortIndex, cut_point_number+1, 10000)
            # intersection with previous labels results
            predict_label_vector = deepcopy(label_vector)
            # print("-----------------predict_label_vector----------", predict_label_vector)
            # predict_label_vector = predict_label_vector.append(labelVector)
            predict_label_vector = np.append(predict_label_vector, [labelVector], axis=0)

            # print("-----------------predict_label_vector----------", predict_label_vector)
            element_vector = []
            for i in range(len(vector[0])):
                element = predict_label_vector[:, i]
                # print("------------element-------------", element)
                if element.tolist() not in element_vector:
                    element_vector.append(element.tolist())
            element_label = []
            for i in range(len(vector[0])):
                for j in range(len(element_vector)):
                    if element_vector[j] == predict_label_vector[:, i].tolist():
                        element_label.append(j)
            # test if element_label number is correct
            # if len(element_label) != len(predict_label_vector):
            #     print("------------label numbers wrong-----------")
            element_index = []
            for element_label_piece in element_label:
                ith_element_index = [ i for i in range(len(element_label)) if element_label[i] == element_label_piece]
                element_index.append(ith_element_index)

            # after_cor = deepcopy(vector)
            # after_cor[i] = x

            # calculate the L2-norm cost
            cost = 0
            for index_3 in element_index:
                # mean_point = np.mean(after_cor[:, index_3], axis=1)
                mean_point = np.mean(original_cor[:, index_3], axis=1)
                # print("mean_point", mean_point, mean_point.shape)
                # for i in range(len(after_cor[:, index_3])):
                for i in index_3:
                    # original_cor[:, index_3]
                    # cost += LA.norm(original_cor[:, index_3][:, i]-mean_point)
                    cost += LA.norm(original_cor[:, i]-mean_point)
                    # print("after_cor shape-----------", after_cor.shape)

            cost_vector.append(cost)

            # calculate the L1-norm cost
            # gap_total = 0
            # for index in element_index:
            #     # find biggest L1-norm gap in these points along every eigenvector's direction
            #     cluster_cor = original_cor[:, index]
            #     for direction in range(len(cluster_cor)):

            #         max_index = np.argmax(cluster_cor[direction])
            #         min_index = np.argmin(cluster_cor[direction])
            #         gap = cluster_cor[direction][max_index] - cluster_cor[direction][min_index]
            #         # print("----------------gap value in each cluster and every direction", gap)
            #         gap_total += gap
            # cost_vector.append(gap_total)

        min_index = np.argmin(cost_vector)
        best_cost = cost_vector[min_index]
        best_cost_vector.append(best_cost)

        one_difference_vector = np.diff(best_cost_vector)
        two_difference_vector = np.diff(one_difference_vector)
        # if len(one_difference_vector) > 1 and one_difference_vector[-1] > one_difference_vector[-2]:
        if len(two_difference_vector) > 1 and two_difference_vector[-1] > two_difference_vector[-2]:
            break

        # two_difference_vector = np.diff(one_difference_vector)

        # if len(best_cost_vector) > 1:
        #     cost_change_rate = (best_cost_vector[-2] - best_cost_vector[-1]) / best_cost_vector[-2]
        #     # cost_change_difference = best_cost_vector[-2] - best_cost_vector[-1]
        #     # cost_change_difference_vector.append(cost_change_difference)
        print("!!!!!!!!!!!!!!!!!!!!!!!min_index!!!!!!!!!!!!!!!!!!!!!!!!!", min_index)
        print("!!!!!!!!!!!!!!!!!!!!!!!cost_vector!!!!!!!!!!!!!!!!!!!!!!!!!", cost_vector)

        unsortVector = vector[min_index]
        sortIndex = np.argsort(unsortVector)
        sortedVector = [unsortVector[i] for i in sortIndex]
        labelVector, x = ADMM3(sortedVector, sortIndex, cut_point_list[min_index]+1, 10000)
        print("labelVectoe------------------------------------------", labelVector)
        # update label_vector and cut_point_list
        cut_point_list[min_index] = cut_point_list[min_index] + 1
        print("-------------cut_point_list---------------", cut_point_list)
        # label_vector.append(labelVector)
        label_vector = np.append(label_vector, [labelVector], axis=0)

        iteration += 1

    element_vector = []
    for i in range(len(vector[0])):
        element = label_vector[:, i]
        if element.tolist() not in element_vector:
            element_vector.append(element.tolist())
    element_label = []
    for i in range(len(vector[0])):
        for j in range(len(element_vector)):
            if element_vector[j] == label_vector[:, i].tolist():
                element_label.append(j)

    # plot cost
    # matplotlib.interactive(False)
    # fig = plt.figure()
    # length = len(best_cost_vector)
    # plt.plot(np.arange(length), best_cost_vector)

    # plot cost_change_diffrence_vector
    # cost_vector_i = []
    # for i in range(len(cost_change_difference_vector) - 1):
    #     cost_i = cost_change_difference_vector[i+1] - cost_change_difference_vector[i]
    #     cost_vector_i.append(cost_i)

    # one_difference_vector = np.diff(best_cost_vector)
    # two_difference_vector = np.diff(one_difference_vector)
    # matplotlib.interactive(False)
    # fig = plt.figure()
    # plt.plot(np.arange(len(two_difference_vector)), two_difference_vector)

    # fig = plt.figure()
    # plt.title("one difference vector")
    # plt.plot(np.arange(len(one_difference_vector)), -one_difference_vector)

    # plt.show()

    return element_label
    # return element_label_vector

    '''

    ''' method for choose eigenvector slice, threshold.

        notice: partition only on smallest eigenvector.
    '''
    cut_point_number = 0
    iteration = 0
    l_percent = 0.10
    l_num = 0
    # unsortVector = vector1
    unsortVector = vector1
    sortIndex = np.argsort(unsortVector)
    sortedVector = [unsortVector[i] for i in sortIndex]
    # plt.figure()
    # plt.plot(sortedVector)
    # plt.show()

    #     labelVector,x = ADMM3(sortedVector, sortIndex, cut_point_number+1, 2000)

    # while iteration < 20:
    #     labelVector, x = ADMM3(sortedVector, sortIndex, cut_point_number+1, 2000)
        # print("------------------labelVector---------", len(labelVector))
        # labels = []
        # for i in range(len(labelVector)):
        #     if labelVector[i] not in labels:
        #         labels.append(labelVector[i])
        # num = []
        # for i in range(len(labels)):
        #     l_count = 0
        #     for j in range(len(labelVector)):
        #         if labelVector[j] == labels[i]:
        #             l_count += 1
        #     num.append(l_count)
        # num_min = min(num)
        # print("--------------num--------------", num)
        # print("--------------num_min---------------", num_min)

        # # if num_min < l_percent * len(sortedVector):
        # if num_min < l_num:
        #     break
        # iteration += 1
        # cut_point_number += 1

        # fig = plt.figure()
        # ax = Axes3D(fig, elev=-150, azim=110)
        # ax.scatter(vector0, vector1, vector2, c=labelVector, cmap="magma")
        # c = np.zeros(len(vector0))
        # ax.set_title("vector space")
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")

    # print("------------------cut_point_number---------", cut_point_number+1)

    labelVector, x = ADMM3(sortedVector, sortIndex, 40, 10000)
    fig = plt.figure()
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(vector0, vector1, vector2, cmap="magma")
    # ax.scatter(vector0, vector1, vector2, c=labelVector, cmap="magma")
    c = np.zeros(len(vector0))
    ax.set_title("vector space")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig = plt.figure()
    plt.plot(vector1)

    return labelVector

def gaussian_plot():

    # sortedVector1, sortedVector2, sortedVector3, sortIndex1, sortIndex2, sortIndex3, vector0, vector1, vector2, X1, X2 = initVec2()
    sortedVector1, sortedVector2, sortedVector3, sortedVector4, sortedVector5, sortIndex1, sortIndex2, sortIndex3, sortIndex4, sortIndex5, vector0, vector1, vector2, vector3, vector4, X1, X2 = initVec2()
    labelVector1, x1 = ADMM3(sortedVector1, sortIndex1, 1, 3000)
    labelVector2, x2 = ADMM3(sortedVector2, sortIndex2, 1, 3000)
    labelVector3, x3 = ADMM3(sortedVector3, sortIndex3, 1, 3000)
    labelVector4, x4 = ADMM3(sortedVector4, sortIndex4, 1, 10000)
    labelVector5, x5 = ADMM3(sortedVector5, sortIndex5, 1, 10000)
    print("--------------cost---------------,", cost1, cost2, cost3, cost4, cost5)
    # print("---------cost3 should be minimize--------------", cost3 < cost2 and cost3 < cost1 and cost4 > cost3 and cost5 > cost3)


    matplotlib.interactive(True)
    # fig = plt.figure(1)
    # ax1 = Axes3D(fig, elev=-150, azim=110)
    # ax1.scatter(vector0, vector1, vector2, c=labelVector1)
    # ax1.set_title("clustering by first eigenvector")
    # plt.show()

    # fig = plt.figure(2)
    # ax2 = Axes3D(fig, elev=-150, azim=110)
    # ax2.scatter(vector0, vector1, vector2, c=labelVector2)
    # ax2.set_title("clustering by second eigenvector")
    # plt.show()

    # fig = plt.figure(3)
    # ax3 = Axes3D(fig, elev=-150, azim=110)
    # ax3.scatter(vector0, vector1, vector2, c=labelVector3)
    # # ax3.set_title("clustering by third eigenvector")
    # plt.show()

    fig = plt.figure(2)
    ax3 = Axes3D(fig, elev=-150, azim=110)
    ax3.scatter(vector2, vector3, vector4, c=labelVector3)
    # ax3.set_title("clustering by third eigenvector")
    plt.show()

    fig = plt.figure(3)
    ax3 = Axes3D(fig, elev=-150, azim=110)
    ax3.scatter(vector2, vector3, vector4, c=labelVector4)
    # ax3.set_title("clustering by third eigenvector")
    plt.show()

    fig = plt.figure(4)
    ax3 = Axes3D(fig, elev=-150, azim=110)
    ax3.scatter(vector2, vector3, vector4, c=labelVector5)
    # ax3.set_title("clustering by third eigenvector")
    plt.show()

    print("Counter labelVector: ", Counter(labelVector1))
    print("Counter labelVector: ", Counter(labelVector2))
    print("Counter labelVector: ", Counter(labelVector3)) 
    tmp = []
    value = []
    for i in range(len(labelVector1)):
        tmp.append([labelVector1[i], labelVector2[i],labelVector3[i]])

    for i in range(len(labelVector1)):
        if [labelVector1[i], labelVector2[i], labelVector3[i]] not in value:
            value.append([labelVector1[i], labelVector2[i], labelVector3[i]])

    # print("tmp", tmp)
    # print("value", value)

#     indices = [i for i, value in enumerate(tmp) if x == "whatever"]
#     for searchVal in value:
#         ii = np.where(tmp == searchVal)[0]
#         indices = [i for i, value in enumerate(tmp) if x == "whatever"]
#         print("ii", ii)

    reLabel = []
    for i in range(len(tmp)):
        for j in range(len(value)):
            if tmp[i] == value[j]:
                reLabel.append(j)

    # print("relabel: ", reLabel)
    fig = plt.figure(4)
    ax4 = Axes3D(fig, elev=-150, azim=110)
    ax4.scatter(vector0, vector1, vector2, c=reLabel)
    ax4.set_title("final clusters")
    plt.show()

    plt.figure(5)
    plt.scatter(X1, X2, c=reLabel)
    matplotlib.interactive(False)
    plt.show()

def image_plot():
    # plot2()

    # # plot vector by lasso method
    # plotVector()

    # plot vector by ADMM method
    # plotVectorByADMM2()

    # plot 2d data by ADMM method
    # plotDataByADMM()

    # initdatasets()

    # x1, labelVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec = ADMM2()
    # plot2(labelVec)

    # sortedVector1, sortedVector2, sortedVector3, sortIndex1, sortIndex2, sortIndex3, vector0, vector1, vector2, X1, X2 = initVec2()
    start_time = time.time()
    sortedVector1, sortedVector2, sortedVector3, sortedVector4, sortIndex1, sortIndex2, sortIndex3, sortIndex4, vector0, vector1, vector2, vector3, row, col = initVec3()
    decomposition_time = time.time()
    print("eigen decomposition time: ", decomposition_time - start_time)
    labelVector1 = ADMM3(sortedVector1, sortIndex1, 3, 100)
    labelVector2 = ADMM3(sortedVector2, sortIndex2, 3, 100)
    labelVector3 = ADMM3(sortedVector3, sortIndex3, 3, 100)
    labelVector4 = ADMM3(sortedVector4, sortIndex4, 3, 100)
    # labelVector5 = ADMM3(sortedVector5, sortIndex5, 3)
    print("admm time: ", time.time() - decomposition_time)

    matplotlib.interactive(True)
    # fig = plt.figure(1)
    # ax1 = Axes3D(fig, elev=-150, azim=110)
    # ax1.scatter(vector0, vector1, vector2, c=labelVector1)
    # ax1.set_title("clustering by first eigenvector")
    # plt.show()

    # fig = plt.figure(2)
    # ax2 = Axes3D(fig, elev=-150, azim=110)
    # ax2.scatter(vector0, vector1, vector2, c=labelVector2)
    # ax2.set_title("clustering by second eigenvector")
    # plt.show()

    # fig = plt.figure(3)
    # ax3 = Axes3D(fig, elev=-150, azim=110)
    # ax3.scatter(vector0, vector1, vector2, c=labelVector3)
    # ax3.set_title("clustering by third eigenvector")
    # plt.show()

    # print("Counter labelVector: ", Counter(labelVector1))
    # print("Counter labelVector: ", Counter(labelVector2))
    # print("Counter labelVector: ", Counter(labelVector3))

    tmp = []
    value = []

    # choose 1 vector
    # for i in range(len(labelVector1)):
    #     tmp.append([labelVector1[i]])

    # for i in range(len(labelVector1)):
    #     if [labelVector1[i]] not in value:
    #         value.append([labelVector1[i]])

    # choose 2 vectors
    for i in range(len(labelVector1)):
        tmp.append([labelVector1[i], labelVector2[i]])

    for i in range(len(labelVector1)):
        if [labelVector1[i], labelVector2[i]] not in value:
            value.append([labelVector1[i], labelVector2[i]])

    # choose 3 vectors
    # for i in range(len(labelVector1)):
    #     tmp.append([labelVector1[i], labelVector2[i],labelVector3[i]])

    # for i in range(len(labelVector1)):
    #     if [labelVector1[i], labelVector2[i], labelVector3[i]] not in value:
    #         value.append([labelVector1[i], labelVector2[i], labelVector3[i]])

    # choose 4 vectors
    # for i in range(len(labelVector1)):
    #     tmp.append([labelVector1[i], labelVector2[i],labelVector3[i], labelVector4[i]])

    # for i in range(len(labelVector1)):
    #     if [labelVector1[i], labelVector2[i], labelVector3[i], labelVector4[i]] not in value:
    #         value.append([labelVector1[i], labelVector2[i], labelVector3[i], labelVector4[i]])

    # choose 5 vector
    # for i in range(len(labelVector1)):
    #     tmp.append([labelVector1[i], labelVector2[i],labelVector3[i], labelVector4[i], labelVector5[i]])

    # for i in range(len(labelVector1)):
    #     if [labelVector1[i], labelVector2[i], labelVector3[i], labelVector4[i], labelVec5[i]] not in value:
    #         value.append([labelVector1[i], labelVector2[i], labelVector3[i], labelVector4[i], labelVector5[i]])


    # print("tmp", tmp)
    # print("value", value)

    #     indices = [i for i, value in enumerate(tmp) if x == "whatever"]
    #     for searchVal in value:
    #         ii = np.where(tmp == searchVal)[0]
    #         indices = [i for i, value in enumerate(tmp) if x == "whatever"]
    #         print("ii", ii)


    reLabel = []
    for i in range(len(tmp)):
        for j in range(len(value)):
            if tmp[i] == value[j]:
                reLabel.append(j)

    # print("relabel: ", reLabel)
    fig = plt.figure(4)
    ax4 = Axes3D(fig, elev=-150, azim=110)
    ax4.scatter(vector0, vector1, vector2, c=reLabel)
    ax4.set_title("final clusters")
    plt.show()

    # plt.figure(5)
    # plt.scatter(X1, X2, c=reLabel)
    # matplotlib.interactive(False)
    # plt.show()

    # plot image label figure
    plt.figure(5)
    reLabel = np.asarray(reLabel)
    print("reLavel shape: ", reLabel.shape)
    # reshapeLabel = np.reshape(reLabel, (20, 20))
    reshapeLabel = np.reshape(reLabel, (row, col))
    plt.imshow(reshapeLabel)
    plt.title("image label plot")
    plt.show()

    v = []
    for i in range(len(reLabel)):
        if reLabel[i] not in v:
            v.append(reLabel[i])

    # for value in v:
    #     construct = []
    #     for i in range(len(reLabel)):
    #         if reLabel[i] == value:
    #             construct.append(value)
    #         else:
    #             construct.append(numpy.nan)
    #     reconstruct = np.reshape(construct, (row, col))
    #     plt.figure()
    #     plt.imshow(reconstruct)

    # plt.show()

    # fig = plt.figure(6)
    # for i in range(row-1):
    #     for j in range(col-1):
    #         if reshapeLabel[i, j] != reshapeLabel[i+1, j] or reshapeLabel[i, j] != reshapeLabel[i, j+1]:
    #             plt.scatter(j, i, color='red', s=1)
    # origMatrix = mpimg.imread('image/resize.jpg')
    # plt.imshow(origMatrix)
    # plt.show()

    # fig = plt.figure(7)
    # plt.imshow(origMatrix)
    # matplotlib.interactive(False)
    # plt.show()

    n_regions = len(value)

    image = Image.open('image/test35.jpg')
    image = image.resize((50, 50), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)

    plt.figure(figsize=(5, 5))
    origMatrix = mpimg.imread('image/resize.jpg')
    grayMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='L')
    # plt.imshow(grayMatrix, cmap=plt.cm.gray)
    plt.imshow(origMatrix, cmap=plt.cm.gray)
    for l in range(n_regions):
        plt.contour(reshapeLabel == l, contours=1,
                    colors=[plt.cm.spectral(l / float(n_regions))])
        plt.xticks(())
        plt.yticks(())

    matplotlib.interactive(False)
    plt.show()

if __name__ == "__main__":
    # gaussian_plot()

    # # plotVectorByADMM1()

    # image_plot()

    # plot2()

    # initDatasets()

    # iterative_choose_partition1()
    # iterative_choose_partition3()
    # iterative_choose_partition_L1_norm()

    matrix_to_label()
