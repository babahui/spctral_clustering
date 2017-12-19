import numpy as np
import scipy.misc
import scipy as sp
import time
from numpy import linalg as LA

import matplotlib.pyplot as plt

def initByNeighborhood():
    row = 20
    col = 20
    length = row * col
    matrix = scipy.misc.imread('../image/question_mark.jpg', mode="L")
    # matrix = scipy.misc.imread('image/question_mark_2.jpg', mode="L")
    matrix = sp.misc.imresize(matrix, (row, col)) / 255.

    finalDistance = []

    for i in range(row):
        for j in range(col):
            if matrix[i, j] < 1:
                matrix[i, j] = 0
    print(matrix)
    # plt.figure()
    # plt.imshow(matrix, cmap=plt.cm.gray)
    # plt.show()

    for init_i in range(row):
        for init_j in range(col):

    # for bb in range(1):
    #     for aa in range(1):
    #         init_i = 3
    #         init_j = 3

            neighbor = [[init_i, init_j]]
            points_set = [[init_i, init_j]]
            new_neighbor = []
            boundary = []
            distance = 100 * np.ones(row*col)
            distance[init_i*row+init_j] = 0

            iteration = 0

            # while iteration != 16:
            while len(points_set) != row * col:
            # while len(new_neighbor) != 0:
                for nei_point in neighbor:
                    # 新邻居点是邻居点左右上下平移一个单位的集合，并且这个集合在整个矩阵里面，不再之后的邻居点集合里面，
                    [i, j] = nei_point
                    for m in [i-1, i, i+1]:
                        for n in [j-1, j, j+1]:
                            if [m, n] not in points_set and (0 <= m <= row-1 and 0 <= n <= col-1):
                                # 新邻居点没有重复就添加到数组，对于每个邻居点找到最优的新邻居点
                                if [m, n] not in new_neighbor:
                                    new_neighbor.append([m, n])
                                if distance[m*row+n] > distance[i*row+j] + abs(matrix[i, j] - matrix[m, n]):
                                    distance[m*row+n] = distance[i*row+j] + abs(matrix[i, j] - matrix[m, n])

                points_set.extend(new_neighbor)
                # 通过新邻居点集合，更新新邻居点的距离，为了新邻居点全局最小
                for new_nei1 in new_neighbor:
                    for new_nei2 in new_neighbor:
                    # 对每一个新邻居点，如果另外一个新邻居点距离加上另外一个新邻居点与该点距离和更小，则更新这个距离
                        [m1, n1] = new_nei1
                        [m2, n2] = new_nei2
                        if dis(new_nei1, new_nei2, new_neighbor, distance, row, col) + distance[m2*row+n2] < distance[m1*row+n1]:
                            distance[m1*row+n1] = distance[m2*row+n2] + dis(new_nei1, new_nei2, new_neighbor, distance, row, col)

                # print("new_neighbor length: ", len(new_neighbor))
                # print("neighbor length: ", len(neighbor))
                # print("points_set length: ", len(points_set))
                neighbor = new_neighbor
                new_neighbor = []
                iteration = iteration + 1
            finalDistance.append(distance)

    return finalDistance, row, col

# 在新邻居点的集合中，一个点到另外一个点每次都只会走一个方向，只有两条路径，直到找到另外一个点停止，计算出路径的总距离
def dis(v1, v2, vec, value, row, col):
    [x1, y1] = v1
    [x2, y2] = v2
    new_point = [x1, y1]
    distance = 0

    while new_point == v2:
        [x, y] = new_point

        if [x-1, y] in vec:
            new_point = [x-1, y]
            distance = distance + abs(value[x*row+y] - value[(x-1)*row+y])
        elif [x+1, y] in vec:
            new_point = [x+1, y]
            distance = distance + abs(value[x*row+y] - value[(x+1)*row+y])
        elif [x, y-1] in vec:
            new_point = [x, y-1]
            distance = distance + abs(value[x*row+y] - value[x*row+y-1])
        elif [x, y+1] in vec:
            new_point = [x, y+1]
            distance = distance + abs(value[x*row+y] - value[x*row+y+1])

    return distance

if __name__ == "__main__":
        start_time = time.time()
        totalVec, row, col = initByNeighborhood()
        print(totalVec)
        print("total time: ", time.time()-start_time)

        # number = 0
        # for i in range(len(totalVec)):
        #     for j in range(len(totalVec[i])):
        #         if totalVec[i, j] == 0:
        #             number = number + 1
        # print("0 numbers : ", number)
        # totalvec = np.asarray(totalVec)
        # print("totalVec: ", totalVec)

        # if totalVec[15, 20] == totalVec[20, 15]:
        #     print("true")
        # else:
        #     print("false")

        # index = []
        # for i in range(2, row-2):
        #     for j in range(2, col-2):
        #         index.append(i*row + j)
        # totalVec = totalVec[:, index]
        # print(totalVec.shape)


