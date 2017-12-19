import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import SparseCoder
from sklearn.cluster import KMeans
from sklearn.decomposition import sparse_encode

''' 1. use omp_test.py initialize data
    2. initialize cluster labels
    3. dictionary learning for clustering

'''
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

    # print([row[1] for row in X])
    return X, Y, X1, X2

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

# initialize eigenvectors
def initEigenvector():
    X, Y, X1, X2 = initDatasets()
    matrix = distanceMatrix(X)
    n = 7
    w, v = LA.eig(matrix)
    min_index = np.argsort(w)[:n]
    index1, index2, index3, index4, index5, index6, index7 = min_index[0], min_index[1], min_index[2], min_index[3], min_index[4], min_index[5], min_index[6]

    eigenvector1 = v[:, index1]
    eigenvector2 = v[:, index2]
    eigenvector3 = v[:, index3]

    # plot 3-d eigenvectors scatter
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=110)
    # ax.scatter(eigenvector1, eigenvector2, eigenvector3)
    # plt.show()

    vectors = [[eigenvector1[i], eigenvector2[i], eigenvector3[i]] for i in range(len(eigenvector1))]
    vectors = np.asarray(vectors)
    return vectors.T

# wish to initialize labels, may use standard spectral clustering
def initClusters():
    # get global dictionary
    vectors = initEigenvector()
    vectors = vectors.T
    print("vectors shape: ", vectors.shape)
    dic, sparseCoder = dictionaryLearningMethod(vectors)
    print("dictionary, sparseCoder shape: ", dic.shape, sparseCoder.shape)
    # construct similarity matrix
    s1 = (sparseCoder.T).dot(sparseCoder)
    # s2 = sparseCoder.dot(sparseCoder.T)

    # standard spectral clustering, find K largest eigenvectors.
    w, v = LA.eig(s1)
    # print("w eigenvalue sort, test if is sorted:", w)
    v_k = [[v[:, 0][i], v[:, 1][i], v[:,2][i], v[:, 3][i], v[:, 4][i]] for i in range(len(v[:, 0]))]
    kmeans = KMeans(n_clusters=5).fit(v_k)
    labelIndex = kmeans.labels_
    print("init labels shape: ", labelIndex.shape)

    return labelIndex

# python dictionary learning method
def dictionaryLearningMethod(vectors):
    # length = len(vectors)
    length = 5
    dic = DictionaryLearning(n_components=length, alpha=1)
    dictionary = dic.fit(vectors).components_
    sparseCoder = dic.fit_transform(vectors)
    # sparseCoder = dic.fit(vectors).components_
    # dictionary = dic.fit_transform(vectors)

    return dictionary.T , sparseCoder.T

# algorithm: iterations for dictionary learning and assignment
def EM():
    alpha = 1
    vectors = initEigenvector()
    print("EM : vectors shape:, ", vectors.shape)
    labelIndex = initClusters()
    print("labelindex shape: ", labelIndex.shape)
    k = 5 # k means cluster numbers
    iteration = 0
    while iteration < 10:
        D = []
        a = []
        # label fixed, update dictionary
        for i in range(k):
            X = []
            # if X data's label == i, calculate minimal cost function
            for j in range(len(labelIndex)):
                if labelIndex[j] == i:
                    X.append(vectors[:, j])
            D_piece, a_piece = dictionaryLearningMethod(X)
            D.append(D_piece)
            a.append(a_piece)

        # dictionary fixed, update labels
        for j in range(len(labelIndex)):
            minVec = []
            # 该点与那个字典匹配使得该点的代价函数最小，则属于该类
            for i in range(k):
                # print("D, a shape: ", len(D), len(a))
                # print("X shape: ", len(X))
                # a is sparse vector
                b = sparse_encode(X=vectors[:, j].T, dictionary=D[i].T)
                b = b.T
                # minVec.append(LA.norm((vectors[:, j] - D[i].dot(a[i][:, j])), 2) + alpha * LA.norm(a[i][:, j], 1))
                minVec.append(LA.norm((vectors[:, j] - D[i].dot(b), 2) + alpha * LA.norm(b, 1)))
            # print("minVec:", minVec)
            label = np.argmin(minVec)
            labelIndex[j] = label
        iteration = iteration + 1

    print("finally, labelIndex: ", labelIndex)
if __name__ == "__main__":
    EM()
