from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from numpy import linalg as LA
import numpy as np
import numpy
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm

import scipy.misc
from scipy import misc
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
import time
from itertools import chain

import math
# def func(i, vec):
#     for j in vec:
#         if i == j:
#             return True

def eigenvectorsPlot():
    image = Image.open('image/four-squares.png')
    image = image.resize((40, 40), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)
    colorMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='RGB')
    print("colormatrix size: ", colorMatrix.shape)

   # colorMatrix = colorMatrix.tolist()
    # for i in colorMatrix:
    #     vecset = set(vec)
    #     vecset = list(vecset)
    #     vecset = set(vecset)
    #     if not vecset.issubset(set(list(i))):
    #         vec.append(i)
    a = 1
    b = 0 # a,b is weight
    row = colorMatrix.shape[0]
    col = colorMatrix.shape[1]
    reshapeMatrix = np.reshape(colorMatrix, (row*col, 3))
    length = reshapeMatrix.shape[0]

    start_time = time.time()
    vec = []
    for i in reshapeMatrix:
        jug = 0
        for k in vec:
            if k == i.tolist():
                jug = 1
        if jug == 0:
            vec.append(i.tolist())
    print("time cost: ", time.time()-start_time)

    print("vec: ")
    for i in range(len(vec)-1):
        for j in range(i+1, len(vec)-1):
            if vec[i] == vec[j]:
                print("wrong situation")
    labelVec = []


    print("colormatrix size: ", colorMatrix.shape)
    for i in range(len(reshapeMatrix)):
        for j in range(len(vec)):
            if reshapeMatrix[i].tolist() == vec[j]:
                labelVec.append(j)

    print("labelVec size: ", len(labelVec))
    print("labelVec: ", labelVec)

    # # use numpy save distanceMatrix
    # distanceMatrix = np.zeros((length, length))
    # for i in range(length):
    #     for j in range(length):
    #         left = a * (int(reshapeMatrix[i]) - int(reshapeMatrix[j]))**2
    #         right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
    #         distanceMatrix[i, j] = np.sqrt(left + right)

    # use python list to save distanceMatrix
    distanceMatrix = []
    for i in range(length):
        for j in range(length):
            # left = a * (int(reshapematrix[i]) - int(reshapematrix[j]))**2
            left = a * LA.norm((reshapeMatrix[i] - reshapeMatrix[j]), 2)
            right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
            distance = np.sqrt(left + right)
            distanceMatrix.append(distance)
    distanceMatrix = np.reshape(distanceMatrix, (length, length))
    print("distanceMatrix: ", distanceMatrix)

    # here we make element to 1000, prove sparse matrix can do the same work
    # distanceMatrix = sparseMatrix(distanceMatrix)

    n = 3
    w, v = LA.eig(distanceMatrix)
    min_index = np.argsort(w)[:n]
    index1, index2, index3 = min_index[0], min_index[1], min_index[2]

    # plot eigenvector manifold
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, index1], v[:, index2], v[:, index3], c=labelVec)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w[index1] * v[:, index1], w[index2] * v[:, index2], w[index3] * v[:, index3], c=labelVec)

    plt.show()

# graying images, see manifold in 3 eigenvectors space
def eigenvectorsPlot_graying():
    image = Image.open('image/four-squares.png')
    image = image.resize((40, 40), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)
    colorMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='L')
    print("colormatrix size: ", colorMatrix.shape)

    a = 1
    b = 0 # a,b is weight
    row = colorMatrix.shape[0]
    col = colorMatrix.shape[1]
    reshapeMatrix = np.reshape(colorMatrix, row*col)
    length = reshapeMatrix.shape[0]

    # # use numpy save distanceMatrix
    # distanceMatrix = np.zeros((length, length))
    # for i in range(length):
    #     for j in range(length):
    #         left = a * (int(reshapeMatrix[i]) - int(reshapeMatrix[j]))**2
    #         right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
    #         distanceMatrix[i, j] = np.sqrt(left + right)

    # use python list to save distanceMatrix
    distanceMatrix = []
    for i in range(length):
        for j in range(length):
            left = a * (reshapeMatrix[i] - reshapeMatrix[j])**2
            # left = a * LA.norm((reshapeMatrix[i] - reshapeMatrix[j]), 2)
            right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
            # distance = np.sqrt(left + right)
            distance = left + right
            distanceMatrix.append(distance)
    distanceMatrix = np.reshape(distanceMatrix, (length, length))
    print("distanceMatrix: ", distanceMatrix)

    # add label
    vec = []
    for i in reshapeMatrix:
        jug = 0
        for k in vec:
            if k == i.tolist():
                jug = 1
        if jug == 0:
            vec.append(i.tolist())

    print("vec: ")
    for i in range(len(vec)-1):
        for j in range(i+1, len(vec)-1):
            if vec[i] == vec[j]:
                print("wrong situation")
    labelVec = []

    print("colormatrix size: ", colorMatrix.shape)
    for i in range(len(reshapeMatrix)):
        for j in range(len(vec)):
            if reshapeMatrix[i].tolist() == vec[j]:
                labelVec.append(j)

    print("labelVec size: ", len(labelVec))
    print("labelVec: ", labelVec)

    n = 3
    w, v = LA.eig(distanceMatrix)
    min_index = np.argsort(w)[:n]
    index1, index2, index3 = min_index[0], min_index[1], min_index[2]

    # plot eigenvector manifold
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, index1], v[:, index2], v[:, index3], c=labelVec)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w[index1] * v[:, index1], w[index2] * v[:, index2], w[index3] * v[:, index3], c=labelVec)

    plt.show()

# Gaussian distance matrix
def eigenvectorsPlot_Gaussian():
    image = Image.open('image/four-squares.png')
    image = image.resize((40, 40), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)
    colorMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='RGB')
    print("colormatrix size: ", colorMatrix.shape)

    row = colorMatrix.shape[0]
    col = colorMatrix.shape[1]
    length = row * col

    reshapeMatrix = np.reshape(colorMatrix, (row*col, 3))
    sig = 1
    gauMat = []
    for i in range(length):
        for j in range(length):
            # gauDis = math.exp(-(reshapeMatrix[i]-reshapeMatrix[j])**2/sig)
            gauDis = np.exp(-(LA.norm(reshapeMatrix[i]-reshapeMatrix[j], 2))**2 / (sig**2))
            gauMat.append(gauDis)
    distanceMatrix = np.reshape(gauMat, (length, length))

    n = 3
    w, v = LA.eig(distanceMatrix)
    min_index = np.argsort(w)[::-1][:n]
    index1, index2, index3 = min_index[0], min_index[1], min_index[2]

    # plot eigenvector manifold
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, index1], v[:, index2], v[:, index3])

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w[index1] * v[:, index1], w[index2] * v[:, index2], w[index3] * v[:, index3])

    plt.show()

# Gaussian distance matrix and nystrom method
def eigenvectorsPlot_Gaussian_nystrom():
    image = Image.open('image/four-squares.png')
    image = image.resize((40, 40), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)
    colorMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='RGB')
    print("colormatrix size: ", colorMatrix.shape)

    row = colorMatrix.shape[0]
    col = colorMatrix.shape[1]
    length = row * col

    reshapeMatrix = np.reshape(colorMatrix, (row*col, 3))
    sig = 1
    gauMat = []
    for i in range(length):
        for j in range(length):
            # gauDis = math.exp(-(reshapeMatrix[i]-reshapeMatrix[j])**2/sig)
            gauDis = np.exp(-(LA.norm(reshapeMatrix[i]-reshapeMatrix[j], 2))**2 / (sig**2))
            gauMat.append(gauDis)
    distanceMatrix = np.reshape(gauMat, (length, length))

    # nystrom
    # generate A matrix
    k = int(0.1*length)
    A = distanceMatrix[:k, :k]
    B = distanceMatrix[:k, k:]
    # A_square_inv = np.sqrt(np.linalg.pinv(A))
    A_square_inv = np.linalg.pinv(np.sqrt(A))
    S = A + A_square_inv.dot(B).dot(B.T).dot(A_square_inv)
    U, l, T = np.linalg.svd(S)
    # print("U * U_T: ", np.dot(U.T, U))
    # print("U[-1] * U[-2]_T: ", np.dot(U[-2].T, U[-1]))
    L = np.diag(l)
    v = np.dot(np.row_stack((A, B.T)), np.dot(A_square_inv, np.dot(U, np.linalg.inv(np.sqrt(L)))))

    index = np.argsort(l)
    index1 = 0
    index2 = 0
    index3 = 0
    for i in range(len(index)):
        if index[i] == len(index)-1:
            index1 = i
            print("i", i)

    for i in range(len(index)):
        if index[i] == len(index)-2:
            index2 = i
            print("i", i)

    for i in range(len(index)):
        if index[i] == len(index)-3:
            index3 = i
            print("i", i)

    # v_1_max = V[:, index1]
    # v_2_max = V[:, index2]
    # v_3_max = V[:, index3]

    # plot eigenvector manifold
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, index1], v[:, index2], v[:, index3])

    # fig = plt.figure(2)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(w[index1] * v[:, index1], w[index2] * v[:, index2], w[index3] * v[:, index3])

    plt.show()


def eigenvectorsPlot_graying_Gaussian():
    image = Image.open('image/four-squares.png')
    image = image.resize((40, 40), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)
    colorMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='L')
    print("colormatrix size: ", colorMatrix.shape)

    row = colorMatrix.shape[0]
    col = colorMatrix.shape[1]
    length = row * col

    reshapeMatrix = np.reshape(colorMatrix, row*col)
    sig = 0.1
    gauMat = []
    for i in range(length):
        for j in range(length):
            gauDis = math.exp(-(reshapeMatrix[i]-reshapeMatrix[j])**2/sig)
            # gauDis = np.exp(-LA.norm(reshapeMatrix[i]-reshapeMatrix[j], 2) / sig)
            gauMat.append(gauDis)
    distanceMatrix = np.reshape(gauMat, (length, length))

    n = 3
    w, v = LA.eig(distanceMatrix)
    max_index = np.argsort(w)[::-1][:n]
    index1, index2, index3 = max_index[0], max_index[1], max_index[2]

    # plot eigenvector manifold
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, index1], v[:, index2], v[:, index3])

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w[index1] * v[:, index1], w[index2] * v[:, index2], w[index3] * v[:, index3])

    plt.show()



if __name__ == "__main__":

    # eigenvectorsPlot()

    eigenvectorsPlot_graying()

    # eigenvectorsPlot_Gaussian()
    # eigenvectorsPlot_Gaussian_nystrom()

    # eigenvectorsPlot_graying_Gaussian()
