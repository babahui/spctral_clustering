import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import SparseCoder
from sklearn.cluster import KMeans
from sklearn.decomposition import sparse_encode
import copy
from scipy.linalg import norm

import time
from PIL import Image
from scipy import misc
import scipy.misc
from sklearn.datasets import make_circles

''' 1. use omp_test.py initialize data
    2. initialize cluster labels
    3. dictionary learning for clustering

'''
def initDatasets():

    '''  here we give some data generate functions. X, Y means 2d coordinates and
    related labels.

    '''

    # 1. iris datasets
    # iris = datasets.load_iris() # X = iris.data[:,:2] # choose first two features # Y = iris.target
    # X = np.asarray(X)

    # 2. make concentric circles, unfortunately can only plot 2 circles
    # X, Y = make_circles(n_samples=200, factor=.3, noise=.05)

    # 3. make some quantiles by Gaussian
    # X, Y = make_gaussian_quantiles(cov=(1, 1), n_samples=100, n_features=2, n_classes=3)

    # 4. generate points in 3 circles, by normalized distribution
    GauSize = [1, 2 ,3, 4, 5, 6]
    # GauSize = [1, 2 ,3]
    X1 = []
    X2 = []
    Y = []
    X = []
    for i in GauSize:
        mean = [i*25, 25]
        cov = [[5, 0], [0, 5]]
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
        # if i == 3:
        #     x1, x2 = np.random.multivariate_normal(mean, cov, 100).T
        #     X1 = np.append(X1, x1)
        #     X2 = np.append(X2, x2)
        #     y = np.dot(i, np.ones(100))
        #     Y = np.append(Y, y)
        # mean2 = [40, 40]
        # if i == 4:
        #     x1, x2 = np.random.multivariate_normal(mean2, cov, 100).T
        #     X1 = np.append(X1, x1)
        #     X2 = np.append(X2, x2)
        #     y = np.dot(i, np.ones(100))
        #     Y = np.append(Y, y)
        # mean3 = [60, 55]
        # if i == 5:
        #     x1, x2 = np.random.multivariate_normal(mean3, cov, 100).T
        #     X1 = np.append(X1, x1)
        #     X2 = np.append(X2, x2)
        #     y = np.dot(i, np.ones(100))
        #     Y = np.append(Y, y)
        # mean4 = [90, 55]
        # if i == 6:
        #     x1, x2 = np.random.multivariate_normal(mean4, cov, 100).T
        #     X1 = np.append(X1, x1)
        #     X2 = np.append(X2, x2)
        #     y = np.dot(i, np.ones(100))
        #     Y = np.append(Y, y)

    for i in range(len(X1)):
        X.append([X1[i], X2[i]])

    #plot.scatter function 3 gaussian circles
    # plt.scatter(X1, X2, c=Y)
    # plt.axis('equal')
    # plt.show()

    # print([row[1] for row in X])
    return X, Y, X1, X2
    # return X, Y

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

def imageInitialization():
    # reduce picture size to 50*50 pixels
    image = Image.open('image/four-squares.png')
    image = image.resize((40, 40), Image.ANTIALIAS)
    image.save('image/resize.jpg', optimiza=True, quality=95)
    # get image, and convert RGB to GRAY value by mode='L'
    grayMatrix = scipy.misc.imread('image/resize.jpg', flatten=False, mode='L')

    a = 0.6
    b = 0.4 # a,b is weight
    row = grayMatrix.shape[0]
    col = grayMatrix.shape[1]
    reshapeMatrix = np.reshape(grayMatrix, row*col)
    length = reshapeMatrix.shape[0]
    distanceMatrix = []
    for i in range(length):
        for j in range(length):
            left = a * (int(reshapeMatrix[i]) - int(reshapeMatrix[j]))**2
            right = b * ((np.floor_divide(i, col) - np.floor_divide(j, col)) ** 2 + (np.mod(i, col) - np.mod(j, col)) ** 2)
            distance = np.sqrt(left + right)
            distanceMatrix.append(distance)
    distanceMatrix = np.reshape(distanceMatrix, (length, length))
    print("distanceMatrix: ", distanceMatrix)

    # generate eigenvectors
    n = 7
    w, v = LA.eig(distanceMatrix)
    min_index = np.argsort(w)[:n]
    index1, index2, index3, index4, index5, index6, index7 = min_index[0], min_index[1], min_index[2], min_index[3], min_index[4], min_index[5], min_index[6]

    eigenvector1 = v[:, index1]
    eigenvector2 = v[:, index2]
    eigenvector3 = v[:, index3]
    eigenvector4 = v[:, index4]
    eigenvector5 = v[:, index5]
    eigenvector6 = v[:, index6]
    eigenvector7 = v[:, index7]

    # plot 3-d eigenvectors scatter
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=110)
    # ax.scatter(eigenvector1, eigenvector2, eigenvector3)
    # plt.show()

    vectors = [[eigenvector1[i], eigenvector2[i], eigenvector3[i], eigenvector4[i], eigenvector5[i], eigenvector6[i], eigenvector7[i]] for i in range(len(eigenvector1))]
    # vectors = [[eigenvector1[i], eigenvector2[i], eigenvector3[i]] for i in range(len(eigenvector1))]
    vectors = np.asarray(vectors)
    return vectors

# initialize eigenvectors
def initEigenvector():
    X, Y, X1, X2 = initDatasets()
    # X, Y = initDatasets()
    matrix = distanceMatrix(X)
    n = 10
    w, v = LA.eig(matrix)
    min_index = np.argsort(w)[:n]
    index1, index2, index3, index4, index5, index6, index7, index8, index9, index10 = min_index[0], min_index[1], min_index[2], min_index[3], min_index[4], min_index[5], min_index[6], min_index[7], min_index[8], min_index[9]

    eigenvector1 = v[:, index1]
    eigenvector2 = v[:, index2]
    eigenvector3 = v[:, index3]
    eigenvector4 = v[:, index4]
    eigenvector5 = v[:, index5]
    eigenvector6 = v[:, index6]
    eigenvector7 = v[:, index7]
    eigenvector8 = v[:, index8]
    eigenvector9 = v[:, index9]
    eigenvector10 = v[:, index10]

    # plot 3-d eigenvectors scatter
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=110)
    # ax.scatter(eigenvector1, eigenvector2, eigenvector3)
    # plt.show()

    vectors = [[eigenvector1[i], eigenvector2[i], eigenvector3[i], eigenvector4[i], eigenvector5[i], eigenvector6[i], eigenvector7[i]] for i in range(len(eigenvector1))]
    # vectors = [[eigenvector1[i], eigenvector2[i], eigenvector3[i]] for i in range(len(eigenvector1))]
    vectors = np.asarray(vectors)
    return vectors

# wish to initialize labels, may use standard spectral clustering
def initClusters():
    # get eigenvectors from image datasets
    # vectors = imageInitialization()
    # get eigenvectors from simulation
    vectors = initEigenvector()
    print("vectors shape: ", vectors.shape)
    dic, sparseCoder = dictionaryLearningMethod(vectors, 3)
    print("dictionary, sparseCoder shape: ", dic.shape, sparseCoder.shape)
    # construct similarity matrix
    # s1 = (sparseCoder.T).dot(sparseCoder)
    s2 = sparseCoder.dot(sparseCoder.T)

    # standard spectral clustering, find K largest eigenvectors.
    w, v = LA.eig(s2)
    m = []
    print("w eigenvalue sort, test if is sorted:", w)
    for i in np.argsort(w)[::-1][:3]:
        m.append(w[i])
    # v_k = [[v[:, 0][i], v[:, 1][i], v[:,2][i], v[:, 3][i], v[:, 4][i], v[:, 5][i]] for i in range(len(v[:, 0]))]
    # v_k = [[v[:, 0][i], v[:, 1][i], v[:,2][i], v[:, 3][i], v[:, 4][i], v[:, 5][i], v[:, 6][i], v[:, 7][i], v[:, 8][i], v[:, 9][i]] for i in range(len(v[:, 0]))]
    v_k = [[v[:, m[0]][i], v[:, m[1]][i], v[:, m[2]][i]] for i in range(len(v[:, m[0]]))]
    kmeans = KMeans(n_clusters=2).fit(v_k)
    labelIndex = kmeans.labels_
    print("init labels shape: ", labelIndex.shape)

    return labelIndex


# python dictionary learning method
def dictionaryLearningMethod(inputVec, length):
    # length = len(vectors)
    length = length
    dic = DictionaryLearning(n_components=length, alpha=0.05)
    dictionary = dic.fit(inputVec).components_
    sparseCoder = dic.fit_transform(inputVec)
    # sparseCoder = dic.fit(vectors).components_
    # dictionary = dic.fit_transform(vectors)

    return dictionary, sparseCoder

def mutipleLearningMethod(inputVec, length1, length2, alpha, beta):
    threshold = 0.01

    iteration = 0
    # initialize A and X
    # print(inputVec)
    initDic = DictionaryLearning(n_components=length1, alpha=alpha)
    A = initDic.fit(inputVec).components_
    X = initDic.fit_transform(inputVec)

    # while error < threshold:
    while iteration < 20:
        # given A, X, update for B and Y
        dic = DictionaryLearning(n_components=length2, alpha=beta)
        B = dic.fit(inputVec - X.dot(A)).components_
        Y = dic.fit_transform(inputVec - X.dot(A))

        # given B, Y, update A, X
        dic2 = DictionaryLearning(n_components=length1, alpha=alpha)
        A = dic2.fit(inputVec - Y.dot(B)).components_
        X = dic2.fit_transform(inputVec - Y.dot(B))

        error = norm(inputVec-X.dot(A)-Y.dot(B)) + alpha*norm(X, 1) + beta*norm(Y, 1)
        iteration += 1

    return A, X, B, Y

# algorithm: iterations for dictionary learning and assignment
def EM():
    alpha = 0.1
    beta = 0.1
    # get vectors from image datasets
    # vectors = imageInitialization()
    # get vectors from simulation
    vectors = initEigenvector()

    # init labels in initClusters function
    # labelIndex = initClusters()
    # initialize labels in optimial way
    labelIndex = np.append(np.zeros(100), np.ones(100))
    # labelIndex = np.append(labelIndex, 2 * np.ones(100))
    # labelIndex = np.append(labelIndex, 3 * np.ones(100))
    # labelIndex = np.append(labelIndex, 4 * np.ones(100))
    # labelIndex = np.append(labelIndex, 5 * np.ones(100))
    testLabel = copy.deepcopy(labelIndex)

    k = 2 # k means cluster numbers
    iteration = 0
    while iteration < 25:
        sortIndex = []
        sortX = []
        sortY = []
        A = []
        X = []
        B = []
        Y = []
        # label fixed, update dictionary
        for i in range(k):
            labelX = []
            # if X data's label == i, calculate minimal cost function
            for j in range(len(labelIndex)):
                if labelIndex[j] == i:
                    labelX.append(vectors[j])
                    sortIndex.append(j)

            if labelX == []:
                # D.append([])
                # print("true")
                pass
            else:
                A_piece, X_piece, B_piece, Y_piece = mutipleLearningMethod(labelX, 3, 3, alpha, beta)
                A.append(A_piece)
                X.extend(X_piece)
                B.append(B_piece)
                Y.extend(Y_piece)

        for i in range(len(sortIndex)):
            for j in range(len(sortIndex)):
                if sortIndex[j] == i:
                    sortX.append(X[j])
                    sortY.append(Y[j])

        # for i in range(len(sortIndex)):
        #     for j in range(len(sortIndex)):
        #         if sortIndex[j] == i:
        #             sorta.append(a[j])
        # dictionary fixed, update labels
        labelIndexTemp = []

        for j in range(len(labelIndex)):
            minVec = []
            # 该点与那个字典匹配使得该点的代价函数最小，则属于该类
            # e1 = (norm(vectors[j] - sorta[j].dot(D[0]))) ** 2 + alpha * norm(sorta[j], 1)
            e1 = (norm(vectors[j] - sortX[j].dot(A[0]) - sortY[j].dot(B[0]))) ** 2 + alpha * norm(sortX[j], 1) + beta * norm(sortY[j], 1)
            label = 0
            for i in range(len(A)):
                # print("D, a shape: ", len(D), len(a))
                # print("X shape: ", len(X))
                # b is sparse vector
                # minVec.append((norm(vectors[j] - b.dot(D[i]))) ** 2 + alpha * norm(b, 1))
                # e = (norm(vectors[j] - sorta[j].dot(D[i]))) ** 2 + alpha * norm(sorta[j], 1)
                e = (norm(vectors[j] - sortX[j].dot(A[i]) - sortY[j].dot(B[i]))) ** 2 + alpha * norm(sortX[j], 1) + beta * norm(sortY[j], 1)
                if e < e1:
                    e1 = e
                    label = i
            labelIndexTemp.append(label)
        labelIndex = labelIndexTemp
        print("labelIndex: ", labelIndex)
        iteration = iteration + 1

        # label更新， 备份
        # for j in range(len(labelIndex)):
        #     minVec = []
        #     # 该点与那个字典匹配使得该点的代价函数最小，则属于该类
        #     for i in range(len(D)):
        #         # print("D, a shape: ", len(D), len(a))
        #         # print("X shape: ", len(X))
        #         # b is sparse vector
        #         b = sparse_encode(X=vectors[j], dictionary=D[i])
        #         minVec.append((norm(vectors[j] - b.dot(D[i])) ** 2) + alpha * norm(b, 1))
        #         # minVec.append(norm(vectors[j] - b.dot(D[i])) ** 2)
        #     print("minVec:", minVec)
        #     label = np.argmin(minVec)
        #     labelIndexTemp.append(label)
        #     # labelIndex[j] = label
        # labelIndex = labelIndexTemp
        # print("labelIndex: ", labelIndex)
        # iteration = iteration + 1

    # print("firstly, labelIndex: ", testLabel)
    # print("finally, labelIndex: ", labelIndex)
    # Dp = D[1]
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=110)
    # # for i in range(len(D)):
    # #     # plt.scatter(D[i])
    # ax.scatter(Dp[:,0], Dp[:,1], Dp[:,2])
    # plt.show() print(len(D), len(a))
    # print(set(labelIndex))
    return testLabel, labelIndex

def test():
    # clustering results in >= 3d, manifold
    vector = initEigenvector()
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=110)
    # clabel = []
    # for i in range(100):
    #     clabel.append(0)
    # for i in range(100):
    #     clabel.append(1)
    # for i in range(100):
    #     clabel.append(2)
    # print(len(clabel))
    # ax.scatter(vector[:,0], vector[:,1], vector[:,2], c=clabel)

    dic, spa = dictionaryLearningMethod(vector, 900)
    # test if D is orth
    for i in range(len(dic)-1):
        for j in range(i+1, len(dic)):
            orth = dic[i].dot(dic[j])
    # plot line
    z = [0, 0, 0]
    for i in range(len(dic)):
        ax.plot3D([z[0], dic[i, 0]], [z[1], dic[i, 1]], [z[2], dic[i, 2]])
    ax.scatter(dic[:, 0], dic[:, 1], dic[:, 2], c='r')
    # plt.xlim(-0.1, 0.1)
    # plt.ylim(-0.1, 0.1)
    ax.set_xlim3d(-0.1, 0.1)
    ax.set_ylim3d(-0.1, 0.1)
    ax.set_zlim3d(-0.1, 0.15)

    # cal error
    alpha = 1
    error = 0
    error1 = 0
    error2 = 0
    error3 = 0
    for i in range(len(vector)):
        k = norm(vector[i] - spa[i].dot(dic), 2) ** 2 + alpha * norm(spa[i], 1)
        print(norm(spa[i], 1))
        error += k
    print("error: ", error)

    vector1 = vector[:100]
    vector2 = vector[100:200]
    vector3 = vector[200:]
    dic1, spa1 = dictionaryLearningMethod(vector[:100], 300)
    dic2, spa2 = dictionaryLearningMethod(vector[100:200], 300)
    dic3, spa3 = dictionaryLearningMethod(vector[200:], 300)
    for i in range(len(vector[:100])):
        error1 += norm(vector1[i] - spa1[i].dot(dic1)) ** 2 + alpha * norm(spa1[i], 1)
        error2 += norm(vector2[i] - spa2[i].dot(dic2)) ** 2+ alpha * norm(spa2[i], 1)
        error3 += norm(vector3[i] - spa3[i].dot(dic3)) ** 2+ alpha * norm(spa3[i], 1)
    print("idea_error: ", error1+error2+error3)

    fig1 = plt.figure()
    ax1 = Axes3D(fig1, elev=-150, azim=110)
    ax1.scatter(vector[:,0], vector[:,1], vector[:,2], c=clabel)
    for i in range(len(dic1)):
        ax1.plot3D([z[0], dic1[i, 0]], [z[1], dic1[i, 1]], [z[2], dic1[i, 2]])
    for i in range(len(dic2)):
        ax1.plot3D([z[0], dic2[i, 0]], [z[1], dic2[i, 1]], [z[2], dic2[i, 2]])
    for i in range(len(dic3)):
        ax1.plot3D([z[0], dic3[i, 0]], [z[1], dic3[i, 1]], [z[2], dic3[i, 2]])
    ax1.set_xlim3d(-0.1, 0.1)
    ax1.set_ylim3d(-0.1, 0.1)
    ax1.set_zlim3d(-0.1, 0.15)
    plt.show()
    # test if D is orth
    for i in range(len(dic1)-1):
        for j in range(i+1, len(dic1)):
            orth = dic1[i].dot(dic1[j])
            print(orth)

if __name__ == "__main__":
    testLabel, labelIndex = EM()

    # 在原图2d中观察聚类结果
    X, Y, X1, X2 = initDatasets()
    # plt.figure(1)
    # plt.scatter(X1, X2, c=testLabel)
    plt.scatter(X1, X2, c=labelIndex)
    plt.axis('equal')

    # clustering results in >= 3d, manifold
    vector = initEigenvector()
    fig = plt.figure()
    ax = Axes3D(fig, elev=-150, azim=110)
    clabel = []
    for i in range(100):
        clabel.append(0)
    for i in range(100):
        clabel.append(1)
    for i in range(100):
        clabel.append(2)
    print(len(clabel))
    ax.scatter(vector[:,0], vector[:,1], vector[:,2], c=labelIndex)
    plt.show()


    # in image
    # vector = imageInitialization()
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=110)
    # clabel = []
    # for i in range(100):
    #     clabel.append(0)
    # for i in range(100):
    #     clabel.append(1)
    # for i in range(100):
    #     clabel.append(2)
    # print(len(clabel))
    # ax.scatter(vector[:,0], vector[:,1], vector[:,2], c=labelIndex)
    # plt.show()

