# superpixels method
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel
from skimage.color import rgb2gray

# floyd method lib
import numpy as np
import numpy as np
import scipy.misc
import scipy as sp
import time
from copy import deepcopy
from scipy.sparse.csgraph import floyd_warshall, dijkstra, bellman_ford, shortest_path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import csgraph_from_dense


# ADMM method lib
from numpy import linalg as LA
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

# others
import numpy
import matplotlib
from skimage.io import imsave

N = 400
IMAGE_NAME = "118035.jpg"
#IMG = "../image/BSDS300/images/test/" + IMAGE_NAME
# IMG = "/home/yy/matlab/BSR/BSDS500/data/images/test/" + IMAGE_NAME

IMG = "/home/yy/matlab/BSR/BSDS500/data/images/train/" + IMAGE_NAME

PATH = "../compared_image/" + "our_" +IMAGE_NAME
GRAYPATH = "../compared_image/" + "our_segmentation_" +IMAGE_NAME

def superpixels():
    image = scipy.misc.imread('../image/test37.jpg', mode="RGB")
    image_gray = scipy.misc.imread('../image/test37.jpg', mode="L")
    image_gray = np.asarray(image_gray)
    image_gray = np.reshape(image_gray, image_gray.shape[0]*image_gray.shape[1])
    image = img_as_float(image)
    segments = slic(image, n_segments=N, sigma=5)
    segments_copy = slic(image, n_segments=N, sigma=5)
    print("segments: ", segments)

    # show the output of SLIC
    matplotlib.interactive(True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.show()

    image = np.asarray(image)
    segments = np.asarray(segments)
    row = segments.shape[0]
    col = segments.shape[1]
    print("row, col: ", row, col)
    image = np.reshape(image, (image.shape[0]*image.shape[1], 3))
    segments = np.reshape(segments, segments.shape[0]*segments.shape[1])
    position = []
    max_num = np.max(segments)
    # print("maximum: ", max_num)
    for i in range(max_num+1):
        pos = [index for index in range(len(image)) if segments[index] == i]
        position.append(pos)

    # print("postion length: ", len(position))
    # print("postion: ", position)
    average = [np.average(image_gray[i]) for i in range(len(position))]

    # print("average: ", average)
    # print("average length: ", len(average))
    matrix = []
    for i in range(len(average)):
        r = []
        for j in range(len(average)):
            metric = abs(average[i]-average[j])
            r.append(metric)
        matrix.append(r)

    # distance metric = (x_i - x_j)
    # matrix = []
    # for i in position:
    #     for j in position:
    #         matrix_i = []
    #         differece = 0
    #         for m in image[i]:
    #             for n in image[j]:
    #                 differece = differece + abs((m-n)[0]) + abs((m-n)[1]) + abs((m-n)[2])
    #                 print("in interation")
    #         matrix_i.append(differece)
    #     matrix.append(matrix_i)
    # print("matrix construct compelet")

    return matrix, position, row, col, segments_copy

def superpixels_neighbor_method():
    image = scipy.misc.imread(IMG, mode="RGB")
    # image = scipy.misc.imread('../image/question_reshape.jpg', mode="L")
    # image = img_as_float(image)
    print("image matrix, ", image)
    # image = image / 255

    # segmentation method compare
    # segments = slic(image, n_segments=N, sigma=1)
    # segments_copy = slic(image, n_segments=N, sigma=1)

    segments = felzenszwalb(image, scale=100, sigma=0.8, min_size=20)
    segments_copy = felzenszwalb(image, scale=100, sigma=0.8, min_size=20)

    # segments = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
    # segments_copy = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)

    # segments = quickshift(image, kernel_size=3, max_dist=10, ratio=0.5)
    # segments_copy = quickshift(image, kernel_size=3, max_dist=10, ratio=0.5)

    # gradient = sobel(rgb2gray(image))
    # segments = watershed(gradient, markers=250, compactness=0.001)
    # segments_copy = watershed(gradient, markers=250, compactness=0.001)

    print("segments: ", segments)

    # # show the output of SLIC
    # matplotlib.interactive(True)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(image, segments))
    # plt.show()

    image = np.asarray(image)
    segments = np.asarray(segments)
    row = image.shape[0]
    col = image.shape[1]
    # row = segments.shape[0]
    # col = segments.shape[1]
    # image = np.reshape(image, (image.shape[0]*image.shape[1], 3))
    segments_copy = np.reshape(segments_copy, segments_copy.shape[0]*segments_copy.shape[1])

    # image_gray = scipy.misc.imread('../image/test37.jpg', mode="L")
    # image_gray = np.asarray(image_gray)
    # image_gray = np.reshape(image_gray, image_gray.shape[0]*image_gray.shape[1])
    # average = [np.average(image_gray[i]) for i in range(len(position))]

    # generate position method 1
    segmentsLabel = []
    for i in range(row):
        for j in range(col):
            l = segments[i, j]
            if l not in segmentsLabel:
                segmentsLabel.append(l)

    position = []
    for i in segmentsLabel:
        pixel_position = []
        for m in range(row):
            for n in range(col):
                if segments[m, n] == i:
                    pixel_position.append([m, n])
        position.append(pixel_position)

    # generate position method 2
    # max_num = np.max(segments_copy)
    # position = []
    # for i in range(max_num+1):
    #     if i in segments_copy:
    #         pixel_position = []
    #         for m in range(row):
    #             for n in range(col):
    #                 if segments[m, n] == i:
    #                     pixel_position.append([m, n])
    #         position.append(pixel_position)
    #     # else:
    #     #     position.append([])

    for i in position:
        if i == []:
            print("position exist null list")
    # print("last position element: ", position[321])

    print("position length: ", len(position))

    # graph init
    graph = 1000000 * np.ones((len(position), len(position))) - 1000000 * np.eye(len(position))
    # graph = np.zeros((len(position), len(position)))
    # graph = np.full((len(position), len(position)), np.inf)

    print("before graph: ", graph)

    # generate average
    average = []
    for i in range(len(position)):
        val = 0
        for j in position[i]:
            [m, n] = j
            val += 0.299*image[m, n, 0] + 0.587*image[m, n, 1] + 0.114*image[m, n, 2]
            # val += image[m, n]
        average.append(val/len(position[i]))

    # print("average: ", average)
    # avePos = []
    # for i in range(len(position)):
    #     cori = 0
    #     corj = 0
    #     for j in position[i]:
    #         [m, n] = j
    #         cori += m
    #         corj += n
    #     avePos.append([cori/len(position[i]), corj/len(position[i])])

    # init graph previous method, superpixels connected in a region
    # threshold = 0.1
    # for i in range(len(position)):
    #     for j in range(len(position)):
    #         # avePos differece, distance information
    #         m = avePos[i][0] - avePos[j][0]
    #         n = avePos[i][1] - avePos[j][1]
    #         # print("m,n:", m, n)
    #         # gray method
    #         # if abs(average[i] - average[j]) <= threshold:
    #         #     # print("graph value: ", abs(average[i]-average[j]))
    #         #     graph[i, j] = abs(average[i] - average[j])
    #         # RGB method
    #         x = abs(average[i][0]-average[j][0]) + abs(average[i][1] - average[j][1]) + abs(average[i][2]-average[j][2])
    #         # print(x)
    #         if x <= threshold and m**2+n**2 <= 2000:
    #         # if m**2+n**2 <= 1000:
    #         # if x <= threshold:
    #             # print("graph value: ", abs(average[i]-average[j]))
    #             graph[i, j] = x

    # init graph method, superpixels connected by neighbor
    for i in range(row):
        for j in range(col):
            # neighbor = [[i, j-1], [i, j+1], [i-1, j], [i+1, j]]
            neighbor = [[i-1, j-1], [i-1, j+1], [i+1, j-1], [i+1, j+1], [i, j-1], [i, j+1], [i-1, j], [i+1, j]]

            # neighbor = []
            # length = 2
            # for m in range(i-length, i+length):
            #     for n in range(j-length, j+length):
            #         neighbor.append([m, n])

            for nei in neighbor:
                if 0 <= nei[0] < row and 0 <= nei[1] < col and segments[i, j] != segments[nei[0], nei[1]]:
                    label1 = segments[i, j]
                    label2 = segments[nei[0], nei[1]]
                    # position 1
                    for k in range(len(segmentsLabel)):
                        if segmentsLabel[k] == label1:
                            index1 = k
                    for k in range(len(segmentsLabel)):
                        if segmentsLabel[k] == label2:
                            index2 = k
                    # x = abs(average[index1][0]-average[index2][0]) + abs(average[index1][1] - average[index2][1]) + abs(average[index1][2]-average[index2][2])
                    x = abs(average[index1]-average[index2])

                    # position 2
                    # x = abs(average[label1][0]-average[label2][0]) + abs(average[label1][1] - average[label2][1]) + abs(average[label1][2]-average[label2][2])
                    # x = abs(average[label1] - average[label2])
                    # graph[label1, label2] = x
                    graph[index1, index2] = x

    # print("graph: ", graph)

    # sparse graph
    # graph = csr_matrix(graph)
    # graph = csgraph_from_dense(graph, null_value=np.inf)
    # graph = np.ma.masked_invalid(graph)

    # number = 0
    # for i in graph:
    #     for j in i:
    #         if j != 0:
    #             number += 1
    # print("number: ", number)

    print("graph construct compelet")
    # graph construct
    # start_time = time.time()

    # graph = floyd_func(graph)

    # print("floyd time: ", time.time()-start_time)

    # for i in graph:
    #     for j in i:
    #         if np.isinf(j) == True:
    #             graph = floyd_warshall(graph, directed=False, unweighted=False)

    graph = floyd_warshall(graph, directed=False, unweighted=False)
    # graph = dijkstra(graph, directed=True)
    # graph = dijkstra(graph, directed=True)
    # graph = bellman_ford(graph, directed=True)
    print("graph full-distance generated")

    # for i in range(graph.shape[0]):
    #     if graph[0, i] > 0.2:
    #         print(graph[0, i])
    print("graph calculate compelet")

    # convert inf to 0
    # number = 0
    # for i in range(graph.shape[0]):
    #     for j in range(graph.shape[1]):
    #         if graph[i, j] == np.inf:
    #             number += 1
    #             graph[i, j] = 0
    # print("number: ", number)

    for i in range(len(position)):
        for j in range(len(position)):
            if graph[i, j] == 1000000:
                print("graph problem")

    print("after graph: ", graph)
    return graph, position, row, col, segments_copy

def boundary_detection(pixel_pos1, pixel_pos2):
    result = False
    for i in pixel_pos1:
        for j in pixel_pos2:
            if abs(i[0]-j[0])**2 + abs(i[1]-j[1])**2 <= 2:
                result = True
    return result

def floyd_func(graph):
    v = graph.shape[0]
    for k in range(0,v):
        for i in range(0,v):
            for j in range(0,v):
                if graph[i,j] > graph[i,k] + graph[k,j]:
                    graph[i,j] = graph[i,k] + graph[k,j]

    return graph

def initByFloyd():
    # length = row * col
    # matrix = scipy.misc.imread('image/four-squares.png', mode="L")
    matrix = scipy.misc.imread('../image/question_mark.jpg', mode="L")
    # matrix = scipy.misc.imread('image/citywall1.jpg', mode="L")
    row = matrix.shape[0]
    col = matrix.shape[1]
    matrix = sp.misc.imresize(matrix, (row, col)) / 255.
    for i in range(row):
        for j in range(col):
            if matrix[i, j] > 0.5:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0

    # scipy.misc.imsave('../image/question_reshape.jpg', matrix)
    # use RGB information
    # matrix = scipy.misc.imread('image/test35.jpg', mode="RGB")
    # matrix = sp.misc.imresize(matrix, (row, col)) / 255.

    print("matrix: ", matrix)

    # plt.figure()
    # plt.imshow(matrix)
    # plt.show()

    graph = []
    for i in range(row):
        for j in range(col):
            vec = 300000 * np.ones(row*col)
            # vec = np.zeros(row*col)
            pos = i * row + j
            for m in [i-1, i, i+1]:
                for n in [j-1, j, j+1]:
                    if 0 <= m <= row-1 and 0 <= n <= col-1:
                        # vec[m*row+n] = abs(matrix[i, j] - matrix[m, n])
                        vec[m*row+n] = abs(matrix[i, j, 1]- matrix[m, n, 1]) + abs(matrix[i, j, 2] - matrix[m, n, 2]) + abs(matrix[i, j, 0] - matrix[m, n, 0])

            graph.append(vec)
    graph = np.asarray(graph)
    print("graph", graph)
    print("graph shape", graph.shape)

    v = len(graph)

    # path reconstruction matrix
    # p = np.zeros(graph.shape)
    # for i in range(0,v):
    #     for j in range(0,v):
    #         p[i,j] = 0
    #         if (i != j and graph[i,j] == 30000):
    #             p[i,j] = -30000
    #             graph[i,j] = 30000
                # set zeros to any large number which is bigger then the longest way

                # print("graph:", graph)
                # print("p", p)

    # floyd method
    # for k in range(0,v):
    #     for i in range(0,v):
    #         for j in range(0,v):
    #             if graph[i,j] > graph[i,k] + graph[k,j]:
    #                 graph[i,j] = graph[i,k] + graph[k,j]

    # floyd sparse method
    graph = floyd_warshall(graph, directed=False, unweighted=False)
    # graph = dijkstra(graph, directed=False, unweighted=False)
    # print(graph.shape)

    # graph_inter = []
    # neighbor method
    # while not np.array_equal(graph, graph_inter):
    #     graph_inter = deepcopy(graph)
    #     for i in range(0, v):
    #         for j in range(0, v):
    #             # if k is neighbor to i
    #             [m, n] = [(i-i%row)/row, i%row]
    #             K = []
    #             for x in [m-1, m, m+1]:
    #                 for y in [n-1, n, n+1]:
    #                     if 0 <= x <= row-1 and 0 <= y <= col-1 and (x != m or y != n):
    #                         k = x*row + y
    #                         K.append(k)
    #             for k in K:
    #                 if graph[i, j] > graph[i, k] + graph[k, j]:
    #                     graph[i, j] = graph[i, k] + graph[k, j]

    # neighbor function method
    # graph_inter = deepcopy(graph)
    # for i in range(0, v):
    #     for j in range(0, v):
    #         graph[i, j] = func(graph, i, j)

    # teacher's method
    # graph_inter = []
    # while not np.array_equal(graph, graph_inter):
    #     graph_inter = deepcopy(graph)
    #     for i in range(0, v):
    #         val_j = []
    #         for j in range(0, v):
    #             [m, n] = [(i-i%row)/row, i%row]
    #             [m2, n2] = [(j-j%row)/row, j%row]
    #             if (m-m2)**2 + (n-n2)**2 > 2:
    #                 K = []
    #                 val_k = []
    #                 for x in [m-1, m, m+1]:
    #                     for y in [n-1, n, n+1]:
    #                         if 0 <= x <= row-1 and 0 <= y <= col-1 and (x != m or y != n):
    #                             k = x*row + y
    #                             K.append(k)
    #                 for k in K:
    #                     # val_k.append(graph[i, k] + graph[k, j])
    #                     val_k.append(float(graph[i, k]) + float(graph[k, j]))
    #                     # val_k.append(max(graph[i, k], graph[k, j]))
    #                 val_k_largest = max(val_k)
    #         val_j.append(val_k_largest)
    #         graph[i, val_j.index(min(val_j))] = min(val_j)
    #     print(graph)
    return graph, row, col

def func(graph, i, j):
    res = graph[i, j]
    [m, n] = [(i-i%row)/row, i%row]
    [m2, n2] = [(j-j%row)/row, j%row]
    if (m-m2)**2 + (n-n2)**2 <= 2:
        val = graph[i, j]
    else:
        K = []
        for x in [m-1, m, m+1]:
            for y in [n-1, n, n+1]:
                if 0 <= x <= row-1 and 0 <= y <= col-1 and (x != m or y != n):
                    k = x*row + y
                    K.append(k)
        if k in K:
            if res > graph[i, k] + func(graph, k, j):
                res = graph[i, k] + func(graph, k, j)
            val = res

    return val

def classMatrix(matrix):
    # matrix, row, col = superpixels()
    w, v = LA.eig(matrix)

    # plot eigenvalues squares
    w_ori = [ w[i] for i in np.argsort(w)[:10] ]
    w_square = numpy.square(w_ori)
    print("-------w_square---------", w_square)
    print("-------w---------", w_ori)
    w_length = w_square.shape[0]
    for i in range(w_length-1,-1,-1):
        w_square[i] = np.sum(w_square[:i+1])
    plt.figure()
    plt.plot(np.linspace(0, 100, w_length), w_square)
    plt.scatter(np.linspace(0, 100, w_length), w_square, c='r')
    plt.title("eigenvalues square")
    plt.show()

    # get 3-d vector, cal distance by x**2 + y**2 + z**2
    index0 = np.argsort(w)[0]
    vector0 = w[index0]*v[:, index0]
    index1 = np.argsort(w)[1]
    vector1 = w[index1]*v[:, index1]
    index2 = np.argsort(w)[2]
    vector2 = w[index2]*v[:, index2]
    index3 = np.argsort(w)[3]
    vector3 = w[index3]*v[:, index3]

    # plot eigenvectors
    # n_dimension = vector0.shape[0]
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

    plt.show()

    # sortIndex1 = np.argsort(vector0)
    # sortedVector1 = [vector0[i] for i in sortIndex1]
    # sortIndex2 = np.argsort(vector1)
    # sortedVector2 = [vector1[i] for i in sortIndex2]
    # sortIndex3 = np.argsort(vector2)
    # sortedVector3 = [vector2[i] for i in sortIndex3]
    # sortIndex4 = np.argsort(vector3)
    # sortedVector4 = [vector3[i] for i in sortIndex4]

    # get 10 vectors
    choose_number = 1
    labelVector = []
    for i in range(0, choose_number):
        index = np.argsort(w)[i]
        vector = w[index]*v[:, index]
        sortIndex = np.argsort(vector)
        sortedVector = [vector[i] for i in sortIndex]
        labelVectorPiece = ADMM3(sortedVector, sortIndex, 10, 10000)
        labelVector.append(labelVectorPiece)

    # for i in range(choose_number, 6):
    #     index = np.argsort(w)[i]
    #     vector = w[index]*v[:, index]
    #     sortIndex = np.argsort(vector)
    #     sortedVector = [vector[i] for i in sortIndex]
    #     labelVectorPiece = ADMM3(sortedVector, sortIndex, 1, 5000)
    #     labelVector.append(labelVectorPiece)

    # for i in range(4, 8):
    #     index = np.argsort(w)[i]
    #     vector = w[index]*v[:, index]
    #     sortIndex = np.argsort(vector)
    #     sortedVector = [vector[i] for i in sortIndex]
    #     labelVectorPiece = ADMM3(sortedVector, sortIndex, 2, 10000)
    #     labelVector.append(labelVectorPiece)

    print("eigen decomposition done, sort eigenvectors done")

    # labelVector1 = ADMM3(sortedVector1, sortIndex1, 2, 5000)
    # labelVector2 = ADMM3(sortedVector2, sortIndex2, 2, 5000)
    # labelVector3 = ADMM3(sortedVector3, sortIndex3, 2, 5000)
    # labelVector4 = ADMM3(sortedVector4, sortIndex4, 2, 5000)

    tmp = []
    value = []

    # choose 1 vector
    # for i in range(len(labelVector4)):
    #     tmp.append([labelVector1[i]])

    # for i in range(len(labelVector4)):
    #     if [labelVector1[i]] not in value:
    #         value.append([labelVector4[i]])

    # choose 2 vectors
    # for i in range(len(labelVector1)):
    #     tmp.append([labelVector1[i], labelVector2[i]])

    # for i in range(len(labelVector1)):
    #     if [labelVector1[i], labelVector2[i]] not in value:
    #         value.append([labelVector1[i], labelVector2[i]])

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

    # choose 10 vectors
    # for i in range(len(labelVector[0])):
    #     k = [labelVector[0][i], labelVector[1][i],labelVector[2][i], labelVector[3][i], labelVector[4][i], labelVector[5][i], labelVector[6][i], labelVector[7][i], labelVector[8][i], labelVector[9][i]]
    #     tmp.append(k)

    # for i in range(len(labelVector[0])):
    #     k = [labelVector[0][i], labelVector[1][i],labelVector[2][i], labelVector[3][i], labelVector[4][i], labelVector[5][i], labelVector[6][i], labelVector[7][i], labelVector[8][i], labelVector[9][i]]
    #     if k not in value:
    #         value.append(k)

    # choose n vectors
    for i in range(len(labelVector[0])):
        a = []
        for j in range(choose_number):
            a.append(labelVector[j][i])
        tmp.append(a)

    for i in range(len(labelVector[0])):
        a = []
        for j in range(choose_number):
            a.append(labelVector[j][i])
        if a not in value:
            value.append(a)

    reLabel = []
    for i in range(len(tmp)):
        for j in range(len(value)):
            if tmp[i] == value[j]:
                reLabel.append(j)

    print("reLabel length: ", len(reLabel))

    # fig = plt.figure(4)
    # ax4 = Axes3D(fig, elev=-150, azim=110)
    # ax4.scatter(vector0, vector1, vector2, c=reLabel)
    # ax4.set_title("final clusters")
    # plt.show()

    return reLabel

def ADMM3(sortedVector, sortIndex, a, iter_time):
    # argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a
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
    return labelVec

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
    # threshold = 0.000001
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
    # print("indexVec", indexVec[:-1])

    for i in range(len(indexVec)):
        if indexVec[i] == 1:
            indexVec[preIndex:i] = origVec[preIndex:i]
            labelVecPiece = [labelVal for i in indexVec[preIndex:i]]
            unSortedLabelVec.extend(labelVecPiece)
            labelVal = labelVal + 1
            preIndex = i

    return indexVec[:-1], unSortedLabelVec


if __name__ == "__main__":
    # start_time = time.time()
    # initByFloyd()
    # print(time.time()-start_time)

    # image = scipy.misc.imread('../image/question_reshape.jpg', mode="L")
    # print(image)

    # superpixels method
    # matrix, position, row, col, segments = superpixels()
    # superpixels neighbor method
    start_time = time.time()
    matrix, position, row, col, segments = superpixels_neighbor_method()
    reLabel = classMatrix(matrix)

    labelVec = []
    for i in range(len(reLabel)):
        if reLabel[i] not in labelVec:
            labelVec.append(reLabel[i])

    print("labelVec: ", labelVec)

    labelPos = []

    finalLabel = np.ones((row, col))
    for i in range(len(labelVec)):
        labelPosPiece = [j for j in range(len(reLabel)) if reLabel[j] == labelVec[i]]
        # print("labelPosPiece: ", labelPosPiece)
        labelPiece = []
        for k in labelPosPiece:
            labelPiece.extend(position[k])
        color = int(i)
        for m in labelPiece:
            [cor_i, cor_j] = m
            # cor_j = m % col
            # cor_i = (m-cor_j) / col
            finalLabel[cor_i, cor_j] = color

    finalLabel = np.asarray(finalLabel, dtype=int)
    print("finalLabel: ", finalLabel)
    print("time cost", time.time()-start_time)
    # show the output of superpixels method
    image = scipy.misc.imread(IMG, mode="RGB")
    image = img_as_float(image)
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    plt.imshow(mark_boundaries(image, finalLabel, color=(0,0,1)))
    # imsave(PATH, mark_boundaries(image, finalLabel, color=(0,0,1)))

    # our segmentation figure
    seg_boundary = find_boundaries(finalLabel).astype(np.uint8)
    for i in range(0, seg_boundary.shape[0]):
        for j in range(0, seg_boundary.shape[1]):
            if seg_boundary[i, j] == 1:
                seg_boundary[i, j] = 255

    imsave(GRAYPATH, seg_boundary, cmap='gray')

    # plt.figure(2)
    # label = find_boundaries(finalLabel)
    # plt.imshow(label)
    # plt.imshow(mark_boundaries(image, segments))
    matplotlib.interactive(False)
    plt.show()

