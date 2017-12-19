from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from numpy import linalg as LA
import scipy.misc
import numpy as np
import numpy
from collections import Counter
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
from scipy.sparse.linalg import eigsh

import time

def grayImagePlot():
    # reduce picture size to 50*50 pixels
    image = Image.open('image/four-squares.png')
    image = image.resize((40, 40), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)
    # get image, and convert RGB to GRAY value by mode='L'
    grayMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='L')
    plt.imshow(grayMatrix, cmap='gray')
    plt.show()

def grayImageInitAndOutputEigenvector():
    # reduce picture size to 50*50 pixels
    image = Image.open('image/lena.png')
    image = image.resize((100, 100), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)
    # get image, and convert RGB to GRAY value by mode='L'
    grayMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='L')

    # convert RGB value to gray value, gray = 0.2989R + 0.5870G + 0.1140B,               output distance matrix by W = sqrt(a(F_i - F_j)**2 + b(X_i - X_j)**2)
    a = 0.6
    b = 0.4 # a,b is weight
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

    # use python list to save distanceMatrix
    distanceMatrix = []
    for i in range(length):
        for j in range(length):
            left = a * (int(reshapeMatrix[i]) - int(reshapeMatrix[j]))**2
            right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
            distance = np.sqrt(left + right)
            distanceMatrix.append(distance)
    distanceMatrix = np.reshape(distanceMatrix, (length, length))
    print("distanceMatrix: ", distanceMatrix)

    # here we make element to 1000, prove sparse matrix can do the same work
    # distanceMatrix = sparseMatrix(distanceMatrix)

    # lanczos method
    # V_j = randomVector / vectorNorm
    randomVector = np.random.uniform(0, 1, size=length)
    vectorNorm = LA.norm(randomVector)
    v_j = np.dot(randomVector, 1/vectorNorm)
    v_j_mins_1 = np.zeros(length)
    b_j = np.zeros(length)
    j = 1

    while j <= length:
    # v_j = np.dot(A, v), complement by each row of A mutiply v
        w_j = []
        for m in range(length):
            distanceArray = []
            for n in range(length):
                left = a * (int(reshapeMatrix[m]) - int(reshapeMatrix[n]))**2
                right = b * ((np.floor_divide(m, col) - np.floor_divide(n, col)) ** 2 + (np.mod(m, col) - np.mod(n, col)) ** 2)
                distance = np.sqrt(left + right)
                distanceArray.append(distance)
            w_j_row = np.dot(distanceArray, v_j)
            w_j.append(w_j_row)

        a_j = np.dot(w_j, v_j)
        w_j = w_j - np.dot(a_j, v_j) - np.dot(b_j, v_j_mins_1)
        # b_j_plus_1 = LA.norm(w_j, 2)
        # v_j_plus_1 = np.dot(w_j, 1/b_j_plus_1)
        v_j_mins_1 = v_j
        b_j = LA.norm(w_j)
        v_j = np.dot(w_j, 1/b_j)
        j = j + 1
        print("a_j, b_j: ", a_j, b_j)

    return distanceMatrix, row, col

def RGBImageInit():
    # reduce picture size to 50*50 pixels
    image = Image.open('image/boundary1.jpg')
    image = image.resize((40, 40), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)
    # get RGB images
    RGBMatrix = scipy.misc.imread('image/resize.jpg', mode="RGB")

    # out matrix with position infomation and RGB value infomation
    a = 0.2
    b = 0.8 # a,b is weight
    row = RGBMatrix.shape[0]
    col = RGBMatrix.shape[1]
    reshapeMatrix = np.reshape(RGBMatrix, (row*col, 3))
    length = reshapeMatrix.shape[0]
    distanceMatrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            left = a *int((LA.norm(reshapeMatrix[i] - reshapeMatrix[j], 1)))**2
            right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
            distanceMatrix[i, j] = np.sqrt(left + right)

    # distanceMatrix = sparseMatrix(distanceMatrix)

    return distanceMatrix, row, col

def sparseMatrix(matrix):
    row = matrix.shape[0]
    col = matrix.shape[1] # get 2d matrix row and col
    percent = 0.5 # 50% element set to 1000(sparse)

    distanceMatrix = np.reshape(matrix, row*col)
    indexArray = np.argsort(distanceMatrix)[::-1]
    sparseNumber = int(percent * row * col)
    for i in range(sparseNumber):
        index = indexArray[i]
        # set element to a value, maybe 1000 big enough
        distanceMatrix[index] = 1000

    distanceMatrix = np.reshape(distanceMatrix, (row, col))
    return distanceMatrix

def ADMM():
    # argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a

    matrix, row, col = grayImageInit()

    # # use numpy eigen-decomposition method
    # w, v = LA.eig(matrix)

    # use lanczos method, A is symmetric matrix
    w, v = eigsh(matrix, k=1, which='SA') # k output 6 smallest eigenvalues

    # get 1-d vector, which is eigenvector*eigenvalue
    index = np.argsort(w)[0]
    vector = w[index]*v[:, index]

    # get 3-d vector, cal distance by x**2 + y**2 + z**2
    # index0 = np.argsort(w)[0]
    # vector0 = w[index0]*v[:, index0]
    # index1 = np.argsort(w)[1]
    # vector1 = w[index1]*v[:, index1]
    # index2 = np.argsort(w)[2]
    # vector2 = w[index2]*v[:, index2]
    # index3 = np.argsort(w)[3]
    # vector3 = w[index3]*v[:, index3]
    # vector = [ i**2 + j**2 + z**2 + d**2 for i, j, z, d in zip(vector0, vector1, vector2, vector3)]

    # print("original Vector: ", vector)
    # print("argsort original Vector: ", np.argsort(vector) )
    sortIndex = np.argsort(vector)
    sortedVector = [vector[i] for i in sortIndex]

    origVec = deepcopy(sortedVector)

    length = len(sortedVector)
    print("length: ", length)
    A = np.zeros((length-1, length))
    for i in range(length-1):
        A[i][i] = -1
        A[i][i+1] = 1
    B = np.identity(length)
    B[length-1, length-1] = 0

    x = deepcopy(sortedVector)
    x1 = deepcopy(sortedVector)
    v = np.dot(A, x1)
    w = np.dot(A, x1)
    # e = 0.9 * length
    e = 1
    r = 0.01
    n = 0.95 # n belong to [0.95, 0.99]
    a = int(0.01 * length)
    # a = 5
    countNumber = 1

    I = np.identity(length)

    while abs(LA.norm(np.dot(A, x), 0) - a) >= e:
        left = numpy.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
        right = np.dot(1, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
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
            # print("z>a")
            print("count number: ", countNumber)
            countNumber = countNumber + 1

        w = w + np.dot(A, x) - v
        r = r * n

        if countNumber > 100:
            break

    # indexVec, labelVec = nonzeroValue(origVec, np.dot(A, x))

    indexVec, unSortedLabelVec = nonzeroValue(origVec, np.dot(A, x))
    # print("indexVec: ", indexVec)

    labelVec = np.zeros(len(unSortedLabelVec))
    for i in range(len(unSortedLabelVec)):
        labelVec[sortIndex[i]] = unSortedLabelVec[i]
    # print("unSortedLabelVec: ", unSortedLabelVec)
    # print("labelVec: ", labelVec)

   # return origVec, indexVec, labelVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec
    return labelVec, row, col

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

    for i in range(len(indexVec)):
        if indexVec[i] == 1:
            indexVec[preIndex:i] = origVec[preIndex:i]
            labelVecPiece = [labelVal for i in indexVec[preIndex:i]]
            unSortedLabelVec.extend(labelVecPiece)
            labelVal = labelVal + 1
            preIndex = i

    return indexVec[:-1], unSortedLabelVec

def boundaryPlot():
    labelVec, row, col = ADMM()
    print("labelVec counter: ", Counter(labelVec))
    # print("row: ", row)
    # print("col: ", col)
    reshapeMatrix = np.reshape(labelVec, (row, col))
    for i in range(row-1):
        for j in range(col-1):
            if reshapeMatrix[i][j] != reshapeMatrix[i+1][j] or reshapeMatrix[i][j] != reshapeMatrix[i][j+1]:
                plt.scatter(j, i, color='red', s=10)
    origMatrix = scipy.misc.imread('image/resize.jpg')

    # plot 2 figures, bounary plot and color plot
    # first one is bounary figure
    plt.figure(1)
    plt.imshow(origMatrix)
    # plt.title("a = 1% pixels, in sparse distanceMatrix(set 50% element to 1000)")

    # second one is color figure
    plt.figure(2)
    plt.imshow(reshapeMatrix)

    plt.show()

def colorPlot():
    labelVec, row, col = ADMM()
    reshapeMatrix = np.reshape(labelVec, (row, col))
    # plt.imshow(reshapeMatrix, cmap='gray')
    plt.imshow(reshapeMatrix)
    # plt.title("in sparse distanceMatrix(set 50% element to 1000)")
    plt.show()

def manifoldPlot():
    matrix, row, col = imageInit()
    w, v = LA.eig(matrix)
    index = np.argsort(w)[0]
    # get 1-d vector, which is eigenvector*eigenvalue
    vector = w[index]*v[:, index]
    # print("original Vector: ", vector)
    # print("argsort original Vector: ", np.argsort(vector) )
    sortIndex = np.argsort(vector)
    sortedVector = [vector[i] for i in sortIndex]

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


if __name__ == "__main__":
    # image = scipy.misc.toimage(reshapeVec)

    # image segmentation, plot the image boundary
    # boundaryPlot()

    # recolor image
    # colorPlot()

    # grayImagePlot()

    start_time = time.time()
    matrix, row, col = grayImageInitAndOutputEigenvector()
    print(" --- %s seconds ---" % (time.time() - start_time))

