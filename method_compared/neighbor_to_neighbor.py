import numpy as np
import scipy.misc
import scipy as sp
import time
from numpy import linalg as LA

def initByNeighborhood():
    row = 24
    col = 24
    length = (row-2) * (col-2)
    length = row * col
    matrix = scipy.misc.imread('../image/question_mark.jpg', mode="L")
    # matrix = scipy.misc.imread('image/question_mark.jpg', mode="L")
    matrix = sp.misc.imresize(matrix, (row, col)) / 255.

    print(matrix)
    totalVec = []

    # for init_i in range(2, row-2):
    #     for init_j in range(2, col-2):

    for bb in range(1):
        for aa in range(1):
            init_i = 2
            init_j = 2

            # vec = -100 * np.ones(length)
            vec = np.zeros(length)

            # init point = 0
            pos = row * (init_i) + (init_j)
            vec[pos] = 0
            neighbor_pos = [[init_i, init_j]]
            new_neighbor_pos = []
            # give neighbor points value
            # update neigh_to_neigh

            iteration = 0
            exist_points = [[init_i, init_j]]
            boundary = [init_i, init_j, init_i, init_j]
            # boundary = [init_i-1, init_j-1, init_i+1, init_j+1]

            while np.count_nonzero(vec) < len(vec):
                # for nei_pos in neighbor_pos:
                #     [i, j] = nei_pos
                #     exist_path.append([i, j])
                # print(len(exist_path))

                for nei_pos in neighbor_pos:
                    [i, j] = nei_pos
                    # print("nei_pos: ", nei_pos)
                    # update new neighbor distance
                    if 0 < i < row-1 and 0 < j < col-1:
                        # update new_neighbor
                        for m in [i-1, i, i+1]:
                            for n in [j-1, j, j+1]:
                                # if [m, n] not in exist_path:
                                if boundary[0] > m or boundary[1] > n or boundary[2] < m or boundary[3] < n:
                                # if not (m == i and n == j):
                                    pos = row * m + n
                                    # if find new neighbor
                                    if vec[pos] == 0:
                                        vec[pos] = vec[i * row + j] + abs(matrix[i, j]-matrix[m, n])
                                        # print(vec[pos])
                                        new_neighbor_pos.append([m, n])
                                    elif vec[pos] > vec[i * row + j] + abs(matrix[i, j]-matrix[m, n]):
                                    # elif (boundary[0] > m or boundary[1] > n or boundary[2] < m or boundary[3] < n) and vec[pos] > vec[i * row + j] + abs(matrix[i, j]-matrix[m, n]):
                                        vec[pos] = vec[i * row + j] + abs(matrix[i, j]-matrix[m, n])
                                        # print(vec[pos])
                print("new_neighbor_pos: ", len(new_neighbor_pos))
                print("neighbor_pos: ", len(neighbor_pos))
                neighbor_pos = new_neighbor_pos
                # exist_points.extend(new_neighbor_pos)
                new_neighbor_pos = []
                # print("neighbor_pos: ", len(neighbor_pos))
                iteration = iteration + 1
                if boundary[0] == 1:
                    boundary[0] = 1
                else:
                    boundary[0] = boundary[0] - 1
                if boundary[1] == 1:
                    boundary[1] = 1
                else:
                    boundary[1] = boundary[1] - 1
                if boundary[2] == row-1:
                    boundary[2] = row-1
                else:
                    boundary[2] = boundary[2] + 1
                if boundary[3] == col-1:
                    boundary[3] = col-1
                else:
                    boundary[3] = boundary[3] + 1

                # print(iteration)
                print("boundary: ", boundary)

            totalVec.append(vec)

    # return middle points
    # index = []
    # totalVec = np.asarray(totalVec)
    # print(totalVec)
    # for i in range(2, row-2):
    #     for j in range(2, col-2):
    #         index.append(i*row + j)
    # totalVec = totalVec[:, index]
    # print(totalVec.shape)

    # return totalVec, row-4, col-4
    return vec, row-4, col-4

if __name__ == "__main__":
        start_time = time.time()
        totalVec, row, col = initByNeighborhood()
        # print(totalVec)
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


