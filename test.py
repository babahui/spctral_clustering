
# from numpy.linalg import inv
# import numpy as np
# length = 300
# A = np.zeros((length-1, length))
# for i in range(length-1):
#     A[i][i] = -1
#     A[i][i+1] = 1

# I = np.identity(length)
# r = 1
# n = 0.99

# for i in range(10000):
#     left = inv(I + np.dot(1/r, np.dot(A.T, A)))
#     r = n * r

# print(left)

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

if __name__ == "__main__":
    row = 30
    col = 30
    length = row * col
    # construct our matrix
    image = np.zeros((row, col))
    image[0:15, 0:15] = 1
    # image[0:15, 15:] = 1
    vector_space_label = np.reshape(image, length)

    # first way to list image pixles
    graph = np.zeros((length, length))
    for i in range(length):
        col_i = i % col
        row_i = (i - col_i) / col
        for j in range(length):
            # col_i = i % col
            # row_i = (i - col_i) / col
            col_j = j % col
            row_j = (j - col_j) / col
            graph[i, j] = abs(image[row_i, col_i] - image[row_j, col_j])

    # second way to list image pixels
    # graph2 = np.zeros((length, length))
    # for i in range(row):
    #     for j in range(col):
    #         p_1 = i * col + j
    #         for m in range(row):
    #             for n in range(col):
    #                 p_2 = m * col + n
    #                 graph2[p_1, p_2] = abs(image[i, j] - image[m, n])


    # if graph.any == graph2.any:
    #     print("-----------it's true----------")
    # else:
    #     print("------------problem-----------")

    w, v = LA.eig(graph)
    # # get 3-d vector, cal distance by x**2 + y**2 + z**2
    index0 = np.argsort(w)[0]
    vector0 = w[index0]*v[:, index0]
    index1 = np.argsort(w)[1]
    vector1 = w[index1]*v[:, index1]
    index2 = np.argsort(w)[2]
    vector2 = w[index2]*v[:, index2]
    index3 = np.argsort(w)[3]
    vector3 = w[index3]*v[:, index3]

    # plot eigenvector space with vector_space_label
    fig = plt.figure()
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(vector0, vector1, vector2, c=vector_space_label)
    ax.set_title("vector space with labels")
    plt.show()
