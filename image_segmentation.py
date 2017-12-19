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

def grayImagePlot():
    # reduce picture size to 50*50 pixels
    image = Image.open('image/lena.png').convert('L') # convert to gray image
    image = image.resize((100, 100), Image.ANTIALIAS)
    image.save('image/gray-lena.png', optimiza=True, quality=95)
    # get image, and convert RGB to GRAY value by mode='L'
    grayMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='L')
    plt.imshow(grayMatrix, cmap='gray')
    plt.show()

def grayImageInit():
    # reduce picture size to 50*50 pixels
    image = Image.open('image/lena.png')
    image = image.resize((40, 40), Image.ANTIALIAS)
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

    return distanceMatrix, row, col

def nystrom():
    # gray image pixels matrix generate
    row = 100
    col = 100
    length = row * col
    a = 0.8 * 100
    b = 0.1 # a,b is weight
    # a = 0.6
    # b = 0.4 # a,b is weight
    image = Image.open('image/citywall7.jpg')
    image = image.resize((row, col), Image.ANTIALIAS) # row, col means resize
    image2 = Image.open('image/citywall7.jpg').convert('LA')
    image2.save("image/resize2.png")
    image.save('image/resize.jpg', optimiza=True, quality=95)
    # get image, and convert RGB to GRAY value by mode='L'
    grayMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='L')

    reshapeMatrix = np.reshape(grayMatrix, row*col)
    # generate A, B matrix, and compute eigenvectors by nystrom
    A = []
    B = []
    m = int(0.01 * row * col) # A is m*m size

    # k means random number between 0 and length-m
    # k = int(length-m)
    k = 3000

    # generate A matrix
    for i in range(k,k+m):
        for j in range(k, k+m):
            left = a * (int(reshapeMatrix[i]) - int(reshapeMatrix[j]))**2
            right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
            distance = np.sqrt(left + right)
            A.append(distance)
    A = np.reshape(A, (m, m))
    # generate B matrix
    for i in range(k, k+m):
        for j in chain(range(k), range(k+m, length)):
            left = a * (int(reshapeMatrix[i]) - int(reshapeMatrix[j]))**2
            right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
            distance = np.sqrt(left + right)
            B.append(distance)
    B = np.reshape(B, (m, length-m))
    # generate leading eigenvector
    # A_square_inv = np.sqrt(np.linalg.pinv(A))
    A_square_inv = np.linalg.inv(np.sqrt(A))
    S = A + A_square_inv.dot(B).dot(B.T).dot(A_square_inv)
    U, l, T = np.linalg.svd(S)
    # print("U * U_T: ", np.dot(U.T, U))
    # print("U[-1] * U[-2]_T: ", np.dot(U[-2].T, U[-1]))
    L = np.diag(l)
    V = np.dot(np.row_stack((A, B.T)), np.dot(A_square_inv, np.dot(U, np.linalg.inv(np.sqrt(L)))))

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

    v_1_max = V[:, index1]
    v_2_max = V[:, index2]
    v_3_max = V[:, index3]

    min_index = np.argmin(l)
    v_min = V[:, min_index]

    max_index = np.argmax(l)
    v_max = V[:, max_index]

    # print("V * V_T : ", np.dot(v_2_max.T, v_2_max))
    # print("V * V_T : ", np.dot(V.T, V))
    # print("V[-1] * V_T[-2]", np.dot(v_2_max.T, v_max))
    # a=plt.figure(1)
    # plt.plot(v_min)
    # b=plt.figure(2)
    # plt.plot(np.sort(v_min))
    # c=plt.figure(3)
    # plt.plot(v_max)
    # d=plt.figure(4)
    # plt.plot(np.sort(v_max))

    # d=plt.figure(4)
    # plt.plot(v_2_max)
    # plt.show()

    # print("lambda: ", np.sort(l))

    # print("v_min : ", v_min)
    # print("l min: ", l[index])

    # a = plt.figure(1)
    # plt.plot(l)
    # b = plt.figure(2)
    # plt.plot(np.sort(l))
    # plt.show()

    # plot eigenvectors
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(v_1_max, v_2_max, v_3_max)
    # plt.show()
    return v_1_max * l[index1], v_2_max * l[index2], v_3_max * l[index3], row, col
    # return v_1_max, v_2_max, v_3_max, row, col
    # return v_2_max * l[two_index] + v_max * l[max_index], row, col

def nystrom2():
    # gray image pixels matrix generate
    row = 40
    col = 40
    length = row * col
    a = 0.6
    b = 0.4 # a,b is weight
    image = Image.open('image/lena.png')
    image = image.resize((row, col), Image.ANTIALIAS) # row, col means resize
    image.save('image/resize.jpg', optimiza=True, quality=95)
    # get image, and convert RGB to GRAY value by mode='L'
    grayMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='L')

    reshapeMatrix = np.reshape(grayMatrix, row*col)
    # generate A, B matrix, and compute eigenvectors by nystrom
    A = []
    B = []
    m = int(0.1 * row * col) # A is m*m size
    # generate A matrix
    for i in range(m):
        for j in range(m):
            left = a * (int(reshapeMatrix[i]) - int(reshapeMatrix[j]))**2
            right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
            distance = np.sqrt(left + right)
            A.append(distance)
    A = np.reshape(A, (m, m))
    # generate B matrix
    for i in range(m):
        for j in range(m, length):
            left = a * (int(reshapeMatrix[i]) - int(reshapeMatrix[j]))**2
            right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
            distance = np.sqrt(left + right)
            B.append(distance)
    B = np.reshape(B, (m, length-m))

    # # generate leading eigenvector
    # # A_square_inv = np.sqrt(np.linalg.pinv(A))
    # A_square_inv = np.linalg.inv(np.sqrt(A))
    # S = A + A_square_inv.dot(B).dot(B.T).dot(A_square_inv)
    # U, l, T = np.linalg.svd(S)
    # L = np.diag(l)
    # V = np.dot(np.row_stack((A, B.T)), np.dot(A_square_inv, np.dot(U, np.linalg.inv(np.sqrt(L)))))

    # index = np.argmin(l)
    # v_min = V[:, index]
    # print("v_min : ", v_min)
    # print("l min: ", l[index])

    U, l, T = np.linalg.svd(A)
    L = np.diag(l)
    V = np.row_stack((U, np.dot(B.T, np.dot(U, np.linalg.inv(L)))))

    index = np.argmin(l)
    v_min = V[:, index]

    return l[index] * v_min, row, col

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

# Gaussian distance matrix
def gaussianSimilarityMatrix():
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
    max_index = np.argsort(w)[::-1][:n]
    index1, index2, index3 = max_index[0], max_index[1], max_index[2]

    # # plot eigenvector manifold
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(v[:, index1], v[:, index2], v[:, index3])

    # fig = plt.figure(2)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(w[index1] * v[:, index1], w[index2] * v[:, index2], w[index3] * v[:, index3])

    # plt.show()

    return v[:, index1], v[:, index2], v[:, index3], row, col

def gaussianSimilarityMatrixAndNystrom():
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

    v_1_max = v[:, index1]
    v_2_max = v[:, index2]
    v_3_max = v[:, index3]

    # # plot eigenvector manifold
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(v[:, index1], v[:, index2], v[:, index3])

    # # fig = plt.figure(2)
    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(w[index1] * v[:, index1], w[index2] * v[:, index2], w[index3] * v[:, index3])

    # plt.show()

    return v_1_max, v_2_max, v_3_max, row, col

def ADMM():
    # argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a

    # matrix, row, col = grayImageInit()

    # # use numpy eigen-decomposition method
    # w, v = LA.eig(matrix)

    # use lanczos method, A is symmetric matrix
    # w, v = eigsh(matrix, k=1, which='SA') # k output 6 smallest eigenvalues

    # nystrom method
    nystrom_start_time = time.time()
    vector, vector2, vector3, row, col = nystrom()
    # vector, vector2, vector3, row, col = gaussianSimilarityMatrix()

    # get 1-d vector, which is eigenvector*eigenvalue
    # index = np.argsort(w)[0]
    # vector = w[index]*v[:, index]

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
    # 10^6
    # sortedVector = np.dot(10^3, sortedVector)
    print("sortedVector first and last element: ", sortedVector[0], sortedVector[-1])
    print("vector2: first and last element", np.sort(vector2)[0], np.sort(vector2)[-1])
    print("vector3: first and last element", np.sort(vector3)[0], np.sort(vector3)[-1])

    index11 = np.argmax(vector3)
    index22 = np.argmin(vector)


    print("nystrom time: ", time.time()-nystrom_start_time)
    origVec = deepcopy(sortedVector)

    length = len(sortedVector)
    print("length: ", length)
    # A = np.zeros((length-1, length))
    # A = csr_matrix((length-1, length))
    A = lil_matrix((length-1, length), dtype='d')
    for i in range(length-1):
        # A[i][i] = -1
        # A[i][i+1] = 1
        A[i, i] = -1
        A[i, i+1] = 1
    print("generate A step")
    # B = identity(length)
    # B[length-1, length-1] = 0

    x = deepcopy(sortedVector)
    x1 = deepcopy(sortedVector)
    # v = np.dot(A, x1)
    # w = np.dot(A, x1)
    v = A.dot(x1)
    w = A.dot(x1)
    e = 1
    r = 0.01
    n = 0.95 # n belong to [0.95, 0.99]
    # a = 5
    # a = int(0.001 * length)
    a = 20
    countNumber = 1

    I = identity(length, format='lil')
    # left = numpy.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
    A_T = A.transpose()
    temp = I + A_T.dot(A) * (1/r)
    # lil_inv = inv(lil_matrix.tocsc(temp))
    # print("memory grow faster in step 1")
    # left = csc_matrix.toarray(lil_inv)
    # # left = csc_matrix.tolil(lil_inv)
    # print("memory grow faster in step 2")

    # left = inv((I + 1/r * A_T.dot(A)).tocsc())
    print("enter to while loop")

    admm_start_time = time.time()

    # while abs(LA.norm(np.dot(A, x), 0) - a) >= e:
    while True:
        # right = np.dot(1, x1) + np.dot(1/r, np.dot(A.T, (v - w)))

        # A.T * (v-w) represent as numpy
        right = np.dot(1, x1) + np.dot(1/r, A_T.dot(v-w))

        # # x = np.dot(left, right)
        # x = left.dot(right)

        x = spsolve(temp.astype(float), right.astype(float))
        # cal x step time
        # print("x step time: ", x_time-start_time)

        # z = np.dot(A, x) + w
        z = A.dot(x) + w
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
            # print("count number: ", countNumber)
            countNumber = countNumber + 1

        # print("z step time: ", z_time-x_time)
        # w = w + np.dot(A, x) - v
        w = w + A.dot(x) - v
        # r = r * n

        if countNumber > 1000:
            break

    print("ADMM time: ", time.time()-admm_start_time)
    # indexVec, labelVec = nonzeroValue(origVec, np.dot(A, x))

    # indexVec, unSortedLabelVec = nonzeroValue(origVec, np.dot(A, x))
    # print("gap : ", np.dot(A, x))
    print("cost function error: ", LA.norm(x-sortedVector, 2))

    unSortedLabelVec = nonzeroValue(origVec, A.dot(x), a)
    print("gap : ", A.dot(x))

    labelVec = np.zeros(len(unSortedLabelVec))
    for i in range(len(unSortedLabelVec)):
        labelVec[sortIndex[i]] = unSortedLabelVec[i]
    # print("unSortedLabelVec: ", unSortedLabelVec)
    # print("labelVec: ", labelVec)

   # return origVec, indexVec, labelVec, x, label, X, X1, X2, Y, sortIndex, unSortedLabelVec
    return labelVec, row, col

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
    threshold = 0.001
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
    counternumber = 0
    # for i in range(39999):
    #     if labelVec[i] == labelVec[i+1]:
    #         counternumber = counternumber + 1
    # print("labelVec counternumber: ", counternumber)

    # print("row: ", row)
    # print("col: ", col)
    reshapeMatrix = np.reshape(labelVec, (row, col))

    for i in range(row-1):
        for j in range(col-1):
            if reshapeMatrix[i][j] != reshapeMatrix[i+1][j] or reshapeMatrix[i][j] != reshapeMatrix[i][j+1]:
                plt.scatter(j, i, color='red', s=6)
    origMatrix = scipy.misc.imread('image/resize.jpg')

    # plot 2 figures, bounary plot and color plot
    # first one is bounary figure

    plt.figure(1)
    plt.imshow(origMatrix)

    # plt.title("a = 1% pixels, in sparse distanceMatrix(set 50% element to 1000)")
    # plt.title("a=0.2, b=0.8")
    # plt.title("nystrom method")

    # second one is color figure

    plt.figure(2)
    plt.imshow(reshapeMatrix)

    # plt.title("a=0.2, b=0.8")
    # plt.title("nystrom method")

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(vector1, vector2, vector3, c=labelVec)
    # plt.figure(3)
    # plt.scatter(vector1, np.zeros(len(vector1)),  c=labelVec)

    plt.show()

def colorPlot():
    labelVec, row, col = ADMM()
    reshapeMatrix = np.reshape(labelVec, (row, col))
    # plt.imshow(reshapeMatrix, cmap='gray')
    plt.imshow(reshapeMatrix)
    # plt.title("in sparse distanceMatrix(set 50% element to 1000)")
    plt.title("a=0.8, b=0.2")
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

    # nystrom method
    # start_time = time.time()
    boundaryPlot()
    # print(" --- %s seconds ---" % (time.time() - start_time))

    # s, row, col = nystrom()

    # # a=plt.figure(1)
    # # plt.plot(s)
    # # print(np.sort(s))

    # vector1, vector2, vector3, row, col = nystrom()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(vector1, vector2, vector3)

    # plt.show()

    # plt.figure(1)
    # plt.scatter(vector1, vector2, vector3)
    # plt.show()

    # vector, vector2, vector3, row, col = nystrom()

    # sortIndex = np.argsort(vector)
    # sortedVector = [vector[i] for i in sortIndex]
    # # 10^6
    # # sortedVector = np.dot(10^3, sortedVector)
    # print("sortedVector first and last element: ", sortedVector[0], sortedVector[-1])
    # print("vector2: first and last element", np.sort(vector2)[0], np.sort(vector2)[-1])
    # print("vector3: first and last element", np.sort(vector3)[0], np.sort(vector3)[-1])
    # index11 = np.argmax(vector3)
    # index22 = np.argmin(vector)

    # labelVec = np.zeros(row*col)
    # labelVec[0:index11] = 1
    # labelVec[index11:index22] = 2
    # reshapeMatrix = np.reshape(labelVec, (row, col))

    # labelVec = np.zeros(row*col)
    # labelVec[0:4877] = 1
    # plt.figure(1)
    # plt.imshow(reshapeMatrix)


    # # fig = plt.figure(2)
    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(vector, vector2, vector3, c=labelVec)
    # # print("labelVec counter: ", Counter(labelVec))

    # plt.figure(2)
    # plt.scatter(vector, np.zeros(len(vector)),  c=labelVec)

    # plt.show()
