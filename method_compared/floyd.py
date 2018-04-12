# superpixels method
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
# from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import felzenszwalb, slic, quickshift
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

# import from ../clustering-backup.py
import sys
sys.path.insert(0, '/home/yy/hust_lab/CV/github_spectral_clustering')
sys.path.insert(0, '/home/yy/hust_lab/CV/github_spectral_clustering/reconstruct_project')
import admm_algorithms
import clustering_backup

import math

# import spectral clustering method
from sklearn.cluster import spectral_clustering
from sklearn import cluster as clusters

# method in SLIC+Ncut
from skimage import data, segmentation, color
from skimage.future import graph

import networkx as nx
from scipy import sparse
from scipy.sparse import linalg as SCI

import pylab as pl


N = 400

# in /image/test
# IMAGE_NAME = "249061.jpg"

# IMAGE_NAME = "175083.jpg"

# IMAGE_NAME = "175083.jpg"

# IMG = "/home/yy/matlab/BSR/BSDS500/data/images/test/" + IMAGE_NAME

# in /image/train
# IMAGE_NAME = "118035.jpg"

# IMAGE_NAME = "113009.jpg"

# IMAGE_NAME = "124084.jpg"

IMAGE_NAME = "65019.jpg"

# IMAGE_NAME = "242078.jpg"

# IMAGE_NAME = "35070.jpg"

# IMAGE_NAME = "24063.jpg"

# IMAGE_NAME = "60079.jpg"

# IMAGE_NAME = "302003.jpg"

# IMAGE_NAME = "113016.jpg"

# IMAGE_NAME = "113044.jpg"


IMG = "/home/yy/matlab/BSR/BSDS500/data/images/train/" + IMAGE_NAME


# IMAGE_NAME = 'pic1.jpg'
# IMAGE_NAME = 'test19.jpg'
# IMAGE_NAME = 'four-squares.png'
# IMAGE_NAME = '30*30_test1.png'
# IMAGE_NAME = '30*30_test7.png'
# IMAGE_NAME = '30*30_test6.png'

# IMAGE_NAME = '10*10_test1.png'

# IMAGE_NAME = '11.png'

# IMG = "../image/" + IMAGE_NAME

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
    # matplotlib.interactive(True)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(image, segments))
    # plt.show()

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
    image = np.asarray(image)
    # image = scipy.misc.imread('../image/question_reshape.jpg', mode="L")
    # image = img_as_float(image)
    print("image matrix, ", image)
    # image = image / 255

    # segmentation method compare
    segments = slic(image, n_segments=N, sigma=1)
    segments_copy = slic(image, n_segments=N, sigma=1)

    # segments = felzenszwalb(image, scale=100, sigma=0.8, min_size=20)
    # segments_copy = felzenszwalb(image, scale=100, sigma=0.8, min_size=20)

    # segments = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
    # segments_copy = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)

    # segments = quickshift(image, kernel_size=3, max_dist=10, ratio=0.5)
    # segments_copy = quickshift(image, kernel_size=3, max_dist=10, ratio=0.5)

    # gradient = sobel(rgb2gray(image))
    # segments = watershed(gradient, markers=250, compactness=0.001)
    # segments_copy = watershed(gradient, markers=250, compactness=0.001)

    print("segments: ", segments)

    image = np.asarray(image)
    segments = np.asarray(segments)
    row = image.shape[0]
    col = image.shape[1] # row = segments.shape[0]
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
    for i in range(len(segmentsLabel)-1):
        if segmentsLabel[i] + 1 != segmentsLabel[i+1]:
            print("False")


    position = []
    for i in segmentsLabel:
        pixel_position = []
        for m in range(row):
            for n in range(col):
                if segments[m, n] == i:
                    pixel_position.append([m, n])
        position.append(pixel_position)

    # # show the output of SLIC
    # fig = plt.figure()
    # # ax = fig.add_subplot(1, 1, 1)
    # # ax.imshow(mark_boundaries(image, segments))
    # # plt.imshow(mark_boundaries(image, segments, color=(0, 0, 1)))
    # label_segments_fi = deepcopy(segments)
    # # label_min = [100, 250]
    # # label_max = [200, 300]
    # # for i in range(label_segments.shape[0]):
    # #     for j in range(label_segments.shape[1]):
    # #         if label_min[0] <= label_segments[i, j] <= label_max[0] or label_min[1] <= label_segments[i, j] <= label_max[1]:
    # #             label_segments[i, j] = 1000
    # for i in range(len(position)-50, len(position)):
    #     for pos in position[i]:
    #         [m, n] = pos
    #         label_segments_fi[m, n] = 1000
    # plt.imshow(mark_boundaries(image, label_segments_fi, color=(0, 0, 1)))
    # plt.show()

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

    '''
    here are some methods about init graph to test if they work.

    '''
    # init graph by first method, by color distance metric between superpixels.

    # distance metric: by average value
    # average = []
    # for i in range(len(position)):
    #     val = 0
    #     for j in position[i]:
    #         [m, n] = j
    #         val += 0.299*image[m, n, 0] + 0.587*image[m, n, 1] + 0.114*image[m, n, 2]
    #         # val += image[m, n]
    #     average.append(val/len(position[i]))

    # length = len(position)
    # graph = np.zeros((length, length))
    # for i in range(length):
    #     for j in range(length):
    #         graph[i, j] = abs(average[i] - average[j]) ** 2

    # constructing distance with calculating each pixel
    # length = len(position)
    # graph = np.zeros((length, length))
    # for i in range(length):
    #     for j in range(length):
    #         val = 0
    #         for pixel_i in position[i]:
    #             [m, n] = pixel_i
    #             for pixel_j in position[j]:
    #                 [m2, n2] = pixel_j
    #                 val += 0.299*(image[m, n, 0]-image[m2, n2, 0]) + 0.587*(image[m, n, 1]-image[m2, n2, 1]) + 0.114*(image[m, n, 2]-image[m2, n2, 2])
    #                 # val += 0.299*(image[m, n, 0]-image[m2, n2, 0]) + 0.587*(image[m, n, 1]-image[m2, n2, 1]) + 0.114*(image[m, n, 2]-image[m2, n2, 2])

    #         graph[i, j] = val / (len(position[i] * len(position[j])))

    # distance metric: max(x_i-x_j)+max(x_i-x_j)
    # length = len(position)
    # maxium = []
    # minium = []
    # for i in range(length):
    #     val = []
    #     for j in position[i]:
    #         [m, n] = j
    #         # val_m_n = np.dot(np.array[0.299, 0.587, 0.114]).T, image[m, n]
    #         val_m_n = np.dot(np.array([0.299, 0.587, 0.114]).T, image[m, n])
    #         val.append(val_m_n)
    #     # val = [np.dot(np.array([0.299, 0.587, 0.114]).T, image[j]) for j in position[i]]
    #     maxium.append(np.max(val))
    #     minium.append(np.min(val))
    # graph = np.zeros((length, length))
    # for i in range(length):
    #     for j in range(length):
    #         # graph[i, j] = abs(maxium[i] - maxium[j]) + abs(minium[i] - minium[j])
    #         # graph[i, j] = abs(maxium[i] - maxium[j])
    #         graph[i, j] = abs(abs(maxium[i] - minium[i]) - abs(maxium[j] - minium[j]))

    # print("-----------graph constructing done-----------")
    # for i in range(row):
    #     for j in range(col):
    #         p_1 = i * col + j
    #         for m in range(row):
    #             for n in range(col):
    #                 p_2 = m * col + n
    #                 graph[p_1, p_2] = abs(average[i, j] - average[m, n])


    # # init graph by second method: by continuity
    # # graph init
    # graph = 1000000 * np.ones((len(position), len(position))) - 1000000 * np.eye(len(position))
    # # graph = np.zeros((len(position), len(position)))
    # # graph = np.full((len(position), len(position)), np.inf)

    # print("before graph: ", graph)

    # # generate average
    # average = []
    # for i in range(len(position)):
    #     val = 0
    #     for j in position[i]:
    #         [m, n] = j
    #         val += 0.299*image[m, n, 0] + 0.587*image[m, n, 1] + 0.114*image[m, n, 2]
    #         # val += image[m, n]
    #     average.append(val/len(position[i]))

    # # print("average: ", average)
    # # avePos = []
    # # for i in range(len(position)):
    # #     cori = 0
    # #     corj = 0
    # #     for j in position[i]:
    # #         [m, n] = j
    # #         cori += m
    # #         corj += n
    # #     avePos.append([cori/len(position[i]), corj/len(position[i])])

    # # init graph previous method, superpixels connected in a region
    # # threshold = 0.1
    # # for i in range(len(position)):
    # #     for j in range(len(position)):
    # #         # avePos differece, distance information
    # #         m = avePos[i][0] - avePos[j][0]
    # #         n = avePos[i][1] - avePos[j][1]
    # #         # print("m,n:", m, n)
    # #         # gray method
    # #         # if abs(average[i] - average[j]) <= threshold:
    # #         #     # print("graph value: ", abs(average[i]-average[j]))
    # #         #     graph[i, j] = abs(average[i] - average[j])
    # #         # RGB method
    # #         x = abs(average[i][0]-average[j][0]) + abs(average[i][1] - average[j][1]) + abs(average[i][2]-average[j][2])
    # #         # print(x)
    # #         if x <= threshold and m**2+n**2 <= 2000:
    # #         # if m**2+n**2 <= 1000:
    # #         # if x <= threshold:
    # #             # print("graph value: ", abs(average[i]-average[j]))
    # #             graph[i, j] = x

    # # init graph method, superpixels connected by neighbor
    # for i in range(row):
    #     for j in range(col):
    #         # neighbor = [[i, j-1], [i, j+1], [i-1, j], [i+1, j]]
    #         neighbor = [[i-1, j-1], [i-1, j+1], [i+1, j-1], [i+1, j+1], [i, j-1], [i, j+1], [i-1, j], [i+1, j]]

    #         # neighbor = []
    #         # length = 2
    #         # for m in range(i-length, i+length):
    #         #     for n in range(j-length, j+length):
    #         #         neighbor.append([m, n])

    #         for nei in neighbor:
    #             if 0 <= nei[0] < row and 0 <= nei[1] < col and segments[i, j] != segments[nei[0], nei[1]]:
    #                 label1 = segments[i, j]
    #                 label2 = segments[nei[0], nei[1]]
    #                 # position 1
    #                 for k in range(len(segmentsLabel)):
    #                     if segmentsLabel[k] == label1:
    #                         index1 = k
    #                 for k in range(len(segmentsLabel)):
    #                     if segmentsLabel[k] == label2:
    #                         index2 = k
    #                 # x = abs(average[index1][0]-average[index2][0]) + abs(average[index1][1] - average[index2][1]) + abs(average[index1][2]-average[index2][2])
    #                 x = abs(average[index1]-average[index2])

    #                 # position 2
    #                 # x = abs(average[label1][0]-average[label2][0]) + abs(average[label1][1] - average[label2][1]) + abs(average[label1][2]-average[label2][2])
    #                 # x = abs(average[label1] - average[label2])
    #                 # graph[label1, label2] = x
    #                 graph[index1, index2] = x

    # # print("graph: ", graph)


    # init graph by third method: also continuity but different with method 2
    graph_row = len(position)
    graph_col = len(position)
    graph = 1000000 * np.ones((len(position), len(position))) - 1000000 * np.eye(len(position))

    # generate average color value and red, green, blue color values
    average = []
    red_average = []
    green_average = []
    blue_average = []
    for i in range(len(position)):
        val = 0
        red_val = 0
        green_val = 0
        blue_val = 0
        for j in position[i]:
            [m, n] = j
            val += 0.299*image[m, n, 0] + 0.587*image[m, n, 1] + 0.114*image[m, n, 2]
            # red_val += 0.299*image[m, n, 0]
            # green_val += 0.587*image[m, n, 1]
            # blue_val += 0.114*image[m, n, 2]

            red_val += image[m, n, 0]
            green_val += image[m, n, 1]
            blue_val += image[m, n, 2]
            # val += image[m, n]
        average.append(val/len(position[i]))
        red_average.append(red_val/len(position[i]))
        green_average.append(green_val/len(position[i]))
        blue_average.append(blue_val/len(position[i]))

    # generate average postition
    avePos = []
    for i in range(len(position)):
        cori = 0
        corj = 0
        for j in position[i]:
            [m, n] = j
            cori += m
            corj += n
        avePos.append([cori/len(position[i]), corj/len(position[i])])

    # record superpixels center position if RGB satisfy requrement.
    r_thre = 100
    g_thre = 100
    b_thre = 100
    plot_position = []
    distance_label = []
    num_marker = 0
    vector_space_label = []
    spe_color_position = []
    # vector_space_label = np.zeros(len(position))
    label_segments = deepcopy(segments)
    # for i in range(len(position)):
    for i in segmentsLabel:
        if red_average[i] >= r_thre and green_average[i] >= g_thre and blue_average[i] <= b_thre:
            plot_position.append(avePos[i])
            distance_label.append(num_marker)
            num_marker += 1
            vector_space_label.append(1)
            spe_color_position.append(avePos[i])
            for pos in position[i]:
                [m, n] = pos
                label_segments[m, n] = 1000
        # elif red_average[i] <= 100 and green_average[i] <= 100 and blue_average[i] <= 100:
        #     vector_space_label.append(2)
        #     for pos in position[i]:
        #         [m, n] = pos
        #         label_segments[m, n] = 2000
        else:
            vector_space_label.append(0)


    # show the output of SLIC
    # fig = plt.figure()
    # # ax = fig.add_subplot(1, 1, 1)
    # # ax.imshow(mark_boundaries(image, segments))
    # # plt.imshow(mark_boundaries(image, segments, color=(0, 0, 1)))
    # plt.imshow(mark_boundaries(image, label_segments, color=(0, 0, 1)))
    # plot_position = np.asarray(plot_position)
    # # plt.scatter(plot_position[:, 1],  plot_position[:, 0])
    # for x, y, z in zip(plot_position[:, 1], plot_position[:, 0], distance_label):
    #     pl.text(x, y, str(z), color="red", fontsize=12)
    # # pl.plot(plot_position[:, 1],  plot_position[:, 0], distance_label, color="red", fontsize=12)
    # plt.show()

    # fully connected
    # color_array = []
    # for i in range(graph_row):
    #     for j in range(graph_col):
    #         if continuity[i, j] == 1:
    #             color_array.append([abs(red_average[i]-red_average[j]), abs(green_average[i]-green_average[j]), abs(blue_average[i]-blue_average[j])])
    #             # color_array.append([abs(red_average[i]-red_average[j]) / (len(red_average) ** 2) , abs(green_average[i]-green_average[j]) / (len(green_average) ** 2), abs(blue_average[i]-blue_average[j]) / (len(blue_average) ** 2)])
    # [red_var, green_var, blue_var] = np.var(color_array, axis=0).tolist()
    # [red_mean, green_mean, blue_mean] = np.mean(color_array, axis=0).tolist()
    # color_array_np = np.asarray(color_array)

    for i in range(graph_row):
        for j in range(graph_col):
            graph[i, j] = abs(average[i] - average[j])
                # graph[i, j] = math.exp(-(average[i]-average[j])**2 / elp)

    distance_graph = np.zeros((len(spe_color_position), len(spe_color_position)))
    for i in range(len(spe_color_position)):
        for j in range(len(spe_color_position)):
            [x1, y1] = spe_color_position[i]
            [x2, y2] = spe_color_position[j]
            # print(x1, y1, x2, y2)
            distance_graph[i, j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    ''' add constraint on neighbor and color.

    # # if superpixels i,j is neighbor, then continuity[i,j] = 1
    # continuity = np.zeros((graph_row, graph_col))
    # numbers = 3
    # distance_constraint = numbers * math.sqrt(row * col / len(position))
    # for i in range(graph_row):
    #     for j in range(graph_col):
    #         # if LA.norm(avePos[i] - avePos[j]) < distance_constraint:
    #         if math.sqrt((avePos[i][0]-avePos[j][0])**2 + (avePos[i][1]-avePos[j][1])**2) < distance_constraint:
    #             continuity[i, j] = 1
    # count = 0
    # for i in range(len(continuity)):
    #     for j in range(len(continuity)):
    #         if continuity[i, j] == 1:
    #             count += 1
    # print("continuity graph: --------------------------------", continuity)
    # print("count---------------------------------------", count)

    # # connected superpixels' color distance
    # color_array = []
    # for i in range(graph_row):
    #     for j in range(graph_col):
    #         if continuity[i, j] == 1:
    #             color_array.append([abs(red_average[i]-red_average[j]), abs(green_average[i]-green_average[j]), abs(blue_average[i]-blue_average[j])])
    #             # color_array.append([abs(red_average[i]-red_average[j]) / (len(red_average) ** 2) , abs(green_average[i]-green_average[j]) / (len(green_average) ** 2), abs(blue_average[i]-blue_average[j]) / (len(blue_average) ** 2)])
    # [red_var, green_var, blue_var] = np.var(color_array, axis=0).tolist()
    # [red_mean, green_mean, blue_mean] = np.mean(color_array, axis=0).tolist()
    # color_array_np = np.asarray(color_array)
    # # print("red_var, green_var, blue_var: ", red_var, green_var, blue_var)
    # # print("red_mean, green_mean, blue_mean: ", red_mean, green_mean, blue_mean)

    # # color_metric method 1
    # # use spectral clustering method to cluster 2 labels, and get the color_metric
    # color_metric = spectral_clustering_method(color_array_np)


    # # # color_metric method 2
    # # # only allowed 20% nodes is connected.
    # # # percent = 0.2
    # # rgb_value = []
    # # average_value = []
    # # for i in range(graph_row):
    # #     for j in range(graph_col):
    # #         if continuity[i, j] == 1:
    # #             # color value metric 1: average color
    # # #             average_value.append(abs(average[i]-average[j]))
    # # # sort_average_value = sorted(average_value)
    # # # average_metric = sort_average_value[int(percent * len(sort_average_value))]
    # #             # color value metric 2: rgb color value
    # #             rgb_value.append(min([abs(red_average[i]-red_average[j]), abs(green_average[i]-green_average[j]), abs(blue_average[i]-blue_average[j])]))
    # # sort_rgb_value = sorted(rgb_value)
    # # color_metric = sort_rgb_value[int(percent * len(rgb_value))]
    # # print("---------color_metric------", color_metric)

    # # init graph use color_metric
    # # if continuity[i, j] = 1, then graph[i, j] = Gaussian(i,j)
    # elp = 1
    # for i in range(graph_row):
    #     for j in range(graph_col):
    #         if continuity[i, j] == 1:
    #             # choose different distance metric
    #             # print(average[i]-average[j])
    #             # graph[i, j] = math.exp(-(average[i]-average[j])**2 / elp)

    #             # if abs(average[i] -average[j]) < average_metric:
    #             # color_metric is a 3*1 vector
    #             if abs(red_average[i] - red_average[j]) < color_metric[0] and abs(green_average[i]-green_average[j]) < color_metric[2] and abs(blue_average[i]- blue_average[j]) < color_metric[4]:
    #                 graph[i, j] = 0
    #             else:
    #                 # graph[i, j] = (average[i] - average[j])**2
    #                 graph[i, j] = 0.299*abs(red_average[i]-red_average[j]) + 0.587*abs(green_average[i]-green_average[j]) + 0.114*abs(blue_average[i]-blue_average[j])
    #                 # graph[i, j] = abs(average[i] - average[j])

    # # # plot hist
    # # data1 = color_array_np[:, 0]
    # # data2 = color_array_np[:, 1]
    # # data3 = color_array_np[:, 2]
    # # bins = np.arange(0, 10, 0.5)
    # # fig = plt.figure()
    # # plt.hist(data1, bins = bins, alpha=0.3, label="red")
    # # plt.hist(data2, bins = bins, alpha=0.3, label="green")
    # # plt.hist(data3, bins = bins, alpha=0.3, label="blue")
    # # plt.legend(loc='upper right')
    # # plt.show()


    # # # calculate std
    # # red_std = np.std(red_average)
    # # green_std = np.std(green_average)
    # # blue_std = np.std(blue_average)
    # # print("red_std, green_std, blue_std--------------------", red_std, green_std, blue_std)

    # # # calculate var
    # # red_var = np.var(red_average)
    # # green_var = np.var(green_average)
    # # blue_var = np.var(blue_average)
    # # print("red_var, green_var, blue_var: ", red_var, green_var, blue_var)

    # # red_mean = np.mean(red_average)
    # # green_mean = np.mean(green_average)
    # # blue_mean = np.mean(blue_average)
    # # print("red_mean, green_mean, blue_mean: ", red_mean, green_mean, blue_mean)

    # # test if init graph is connected
    # # new position list

    # final_list = []
    # for i in range(graph_row):
    #     for j in range(i+1, graph_col):
    #         if graph[i, j] == 0:
    #             if i not in final_list:
    #                 final_list.append(i)
    #             if j not in final_list:
    #                 final_list.append(j)

    # # final_list = []
    # # for i in range(graph_row):
    # #     for j in range(i+1, graph_col):
    # #         if graph[i, j] == 0:
    # #             final_list.append([i, j])

    # print("segment matrix shape: ", segments.shape)
    # segment3 = deepcopy(segments)
    # segment_label2 = np.asarray(segment3, dtype=int)
    # for i in final_list:
    #     for j in position[i]:
    #         [m, n] = j
    #         segment_label2[m, n] = 2000
    # count2 = 0 # count2: connected nodes
    # count3 = 0 # count3: not connected nodes
    # for i in range(segment_label2.shape[0]):
    #     for j in range(segment_label2.shape[1]):
    #         if segment_label2[i, j] == 2000:
    #             count2 += 1
    #         else:
    #             count3 += 1
    # print("count2: ---------------------------------------:", count2)
    # print("count3: ---------------------------------------:", count3)
    '''

    # first_plot mark
    # image = scipy.misc.imread(IMG, mode="RGB")
    # image = img_as_float(image)
    # matplotlib.interactive(True)
    # fig = plt.figure()
    # plt.imshow(mark_boundaries(image, segment_label2, color=(0, 0, 1)))
    # plt.show()

    '''
    after init graph, then constructing graph by different methods:
    1. sparse graph
    2. floyd method(use library)
    3. floyd method(our function)
    4. other methods like dijkstra and so on
    '''

    # # sparse graph
    # # graph = csr_matrix(graph)
    # # graph = csgraph_from_dense(graph, null_value=np.inf)
    # # graph = np.ma.masked_invalid(graph)

    # # number = 0
    # # for i in graph:
    # #     for j in i:
    # #         if j != 0:
    # #             number += 1
    # # print("number: ", number)

    # print("graph construct compelet")
    # # graph construct
    start_time = time.time()
    # graph = floyd_func(graph)
    print("graph construct compelet, and time cost:", time.time()-start_time)

    # continuity final graph
    continuity_label_pos = []
    for i in range(graph_row):
        for j in range(i+1, graph_col):
            if graph[i, j] == 0:
                continuity_label_pos.append(i)
                continuity_label_pos.append(j)

    # continuity_label = np.ones((row, col))
    # continuity_label = np.asarray(continuity_label, dtype=int)
    # for i in continuity_label_pos:
    #     pos = position[continuity_label_pos[i]]
    #     # for j in range(len(pos)):
    #     for j in pos:
    #         [m, n] = j
    #         continuity_label[m, n] = 2

    segments2 = deepcopy(segments)
    segments_label = np.asarray(segments2, dtype=int)
    # for i in continuity_label_pos:
    for i in range(len(continuity_label_pos)):
        pos = position[continuity_label_pos[i]]
        # for j in range(len(pos)):
        for j in pos:
            [m, n] = j
            segments_label[m, n] = 2500

    count4 = 0 # connected nodes
    count5 = 0 # not connected nodes
    for i in range(segments_label.shape[0]):
        for j in range(segments_label.shape[1]):
            if segments_label[i, j] == 2500:
                count4 += 1
            else:
                count5 += 1
    print("count4-------------------------", count4)
    print("count5-------------------------", count5)

    # second_plot mark
    # image = scipy.misc.imread(IMG, mode="RGB")
    # image = img_as_float(image)
    # fig = plt.figure()
    # plt.imshow(mark_boundaries(image, segments_label, color=(0, 0, 1)))
    # # plt.imshow(mark_boundaries(image, segments2, color=(0, 0, 1)))
    # plt.show()

    # # print("floyd time: ", time.time()-start_time)

    # # for i in graph:
    # #     for j in i:
    # #         if np.isinf(j) == True:
    # #             graph = floyd_warshall(graph, directed=False, unweighted=False)

    # graph = floyd_warshall(graph, directed=False, unweighted=False)
    # # graph = dijkstra(graph, directed=True)
    # # graph = dijkstra(graph, directed=True)
    # # graph = bellman_ford(graph, directed=True)
    # print("graph full-distance generated")

    # # for i in range(graph.shape[0]):
    # #     if graph[0, i] > 0.2:
    # #         print(graph[0, i])
    # print("graph calculate compelet")

    # for i in range(len(position)):
    #     for j in range(len(position)):
    #         if graph[i, j] == 1000000:
    #             print("graph problem")

    print("after graph: ", graph)
    return graph, position, row, col, segments_copy, vector_space_label, distance_graph, distance_label

def spectral_clustering_method(color_array_np):
    color_metric = []
    # # seperate r,g,b
    # for i in range(3):
    #     rgb_color_array = color_array_np[:, i]
    #     print("-----------------------rgb color array", rgb_color_array.shape)
    #     spectral = clusters.SpectralClustering(n_clusters=2, affinity='rbf')
    #     spectral.fit(rgb_color_array)
    #     y_pred = spectral.fit_predict(rgb_color_array)
    #     # init rgb value
    #     rgb_value_1_min = rgb_color_array[0]
    #     rgb_value_1_max = rgb_color_array[0]
    #     rgb_value_2_min = rgb_color_array[0]
    #     rgb_value_2_max = rgb_color_array[0]
    #     label1 = y_pred[0]
    #     for i in range(len(y_pred)):
    #         if y_pred[i] == label1:
    #             if rgb_color_array[i] > rgb_value_1_max:
    #                 rgb_value_1_max = rgb_color_array[i]
    #             if rgb_color_array[i] < rgb_value_1_min:
    #                 rgb_value_1_min = rgb_color_array[i]
    #         if y_pred[i] != label1:
    #             if rgb_color_array[i] > rgb_value_2_max:
    #                 rgb_value_2_max = rgb_color_array[i]
    #             if rgb_color_array[i] < rgb_value_2_min:
    #                 rgb_value_2_min = rgb_color_array[i]

    #     if rgb_value_1_max <= rgb_value_2_min:
    #         color_metric.append(rgb_value_1_max, rgb_value_2_min)
    #     if rgb_value_2_max <= rgb_value_1_min:
    #         color_metric.append(rgb_value_2_max, rgb_value_1_min)

    # # r,g,b as a vector
    # spectral = clusters.SpectralClustering(n_clusters=2, affinity='rbf')
    # spectral.fit(rgb_array_np)
    spectral = clusters.SpectralClustering(n_clusters=2, affinity='rbf')
    spectral.fit(color_array_np)
    y_pred = spectral.fit_predict(color_array_np)
    for i in range(3):
        rgb_color_array = color_array_np[:, i]
        print("-----------------------rgb color array", rgb_color_array.shape)
        # init rgb value
        rgb_value_1_min = rgb_color_array[0]
        rgb_value_1_max = rgb_color_array[0]
        rgb_value_2_min = rgb_color_array[0]
        rgb_value_2_max = rgb_color_array[0]
        label1 = y_pred[0]
        for i in range(len(y_pred)):
            if y_pred[i] == label1:
                if rgb_color_array[i] > rgb_value_1_max:
                    rgb_value_1_max = rgb_color_array[i]
                if rgb_color_array[i] < rgb_value_1_min:
                    rgb_value_1_min = rgb_color_array[i]
            if y_pred[i] != label1:
                if rgb_color_array[i] > rgb_value_2_max:
                    rgb_value_2_max = rgb_color_array[i]
                if rgb_color_array[i] < rgb_value_2_min:
                    rgb_value_2_min = rgb_color_array[i]

        if rgb_value_1_max <= rgb_value_2_min:
            color_metric.append(rgb_value_1_max, rgb_value_2_min)
        if rgb_value_2_max <= rgb_value_1_min:
            color_metric.append(rgb_value_2_max, rgb_value_1_min)



    return color_metric


def small_scale_image_floyd_method():
    # image = scipy.misc.imread(IMG, mode="RGB")
    image = scipy.misc.imread(IMG, mode="L")
    # image = scipy.misc.imread('../image/question_reshape.jpg', mode="L")
    # image = img_as_float(image)
    # image = image / 255

    image = np.asarray(image)
    print("-------------image--------------", image.shape)
    # segments = np.asarray(segments)
    row = image.shape[0]
    col = image.shape[1]

    # convert gray value to 0 or 255
    vector_space_label = []
    for i in range(row):
        for j in range(col):
            if image[i, j] < 255/2:
                # image[i, j] = 0
                vector_space_label.append(0)
            else:
                # image[i, j] = 1
                vector_space_label.append(1)
    print("image matrix, ", image)

    # # plot the image after converting
    # labels = np.reshape(vector_space_label, (row, col))
    # plt.figure()
    # plt.imshow(labels)
    # plt.title("image labels plot")
    # plt.show()

    length = row * col

    # only use color information to construct graph matrix
    # first method to try it
    # graph = np.zeros((length, length))
    # for i in range(length):
    #     for j in range(length):
    #         col_i = i % col
    #         row_i = (i - col_i) / col
    #         col_j = j % col
    #         row_j = (i - col_j) / col
    #         graph[i, j] = abs(image[row_i, col_i] - image[row_j, col_j])

    # second way to try it
    graph = np.zeros((length, length))
    for i in range(row):
        for j in range(col):
            p_1 = i * col + j
            for m in range(row):
                for n in range(col):
                    p_2 = m * col + n
                    graph[p_1, p_2] = abs(image[i, j] - image[m, n])

    # use both distance information and color information
    # graph = np.zeros((length, length))
    # for i in range(length):
    #     for j in range(length):
    #         col_i = i % col
    #         row_i = (i - col_i) / col
    #         col_j = j % col
    #         row_j = (i - col_j) / col
    #         graph[i, j] = 1 * abs(image[row_i, col_i] - image[row_j, col_j]) + 1 * math.sqrt((row_i - row_j)**2 + (col_i - col_j)**2)


    # # graph init
    # # graph = 1000000 * np.ones((len(position), len(position))) - 1000000 * np.eye(len(position))
    # graph = 1000000 * np.ones((length, length)) - 1000000 * np.eye(length)
    # # graph = 1000000 * np.zeros((length, length))
    # print("before graph: ", graph)

    # # init graph method, superpixels connected by neighbor
    # for i in range(row):
    #     for j in range(col):
    #         # neighbor = [[i, j-1], [i, j+1], [i-1, j], [i+1, j]]
    #         neighbor = [[i-1, j-1], [i-1, j+1], [i+1, j-1], [i+1, j+1], [i, j-1], [i, j+1], [i-1, j], [i+1, j]]

    #         for nei in neighbor:
    #             if 0 <= nei[0] < row and 0 <= nei[1] < col:
    #                 index = i * row + j
    #                 neighbor_index = nei[0] * row + nei[1]
    #                 # test if this is no use, for graph[i, j] == graph[j, i]
    #                 print("----------image[i,j,0]", image[i,j,0])
    #                 graph[index, neighbor_index] = abs((image[i, j, 0]-image[nei[0], nei[1], 0])*0.299 + (image[i, j, 1]-image[nei[0], nei[1], 1])*0.587 + (image[i, j, 2]-image[nei[0], nei[1], 2])*0.114)
    #                 graph[neighbor_index, index] = abs((image[i, j, 0]-image[nei[0], nei[1], 0])*0.299 + (image[i, j, 1]-image[nei[0], nei[1], 1])*0.587 + (image[i, j, 2]-image[nei[0], nei[1], 2])*0.114)
    #                 # graph[neighbor_index, index] = abs((image[i, j, 0]-image[nei[0], nei[1], 0])*0.299 + (image[i, j, 1]-image[nei[0], nei[1], 1])*0.587 + (image[i, j, 2]-image[nei[0], nei[1], 2])*0.114)
    #                 # graph[index, neighbor_index] = abs(image[i, j] - image[nei[0], nei[1]])
    #                 # graph[neighbor_index, index] = abs(image[i, j] - image[nei[0], nei[1]])

    # graph = floyd_func(graph)

    # graph = floyd_warshall(graph, directed=False, unweighted=False)
    # graph = dijkstra(graph, directed=True)
    # graph = dijkstra(graph, directed=True)
    # graph = bellman_ford(graph, directed=True)
    print("graph construct compelet")

    # for i in range(graph.shape[0]):
    #     if graph[0, i] > 0.2:
    #         print(graph[0, i])
    # print("graph calculate compelet")

    # convert inf to 0
    # number = 0
    # for i in range(graph.shape[0]):
    #     for j in range(graph.shape[1]):
    #         if graph[i, j] == np.inf:
    #             number += 1
    #             graph[i, j] = 0
    # print("number: ", number)

    for i in range(length):
        for j in range(length):
            if graph[i, j] == 1000000:
                print("graph problem")

    print("after graph: ", graph)
    # return graph, position, row, col, segments_copy
    return graph, row, col, vector_space_label

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

    # get 3-d vector, cal distance by x**2 + y**2 + z**2
    index0 = np.argsort(w)[0]
    vector0 = w[index0]*v[:, index0]
    index1 = np.argsort(w)[1]
    vector1 = w[index1]*v[:, index1]
    index2 = np.argsort(w)[2]
    vector2 = w[index2]*v[:, index2]
    index3 = np.argsort(w)[3]
    vector3 = w[index3]*v[:, index3]

    # sortIndex1 = np.argsort(vector0)
    # sortedVector1 = [vector0[i] for i in sortIndex1]
    # sortIndex2 = np.argsort(vector1)
    # sortedVector2 = [vector1[i] for i in sortIndex2]
    # sortIndex3 = np.argsort(vector2)
    # sortedVector3 = [vector2[i] for i in sortIndex3]
    # sortIndex4 = np.argsort(vector3)
    # sortedVector4 = [vector3[i] for i in sortIndex4]

    # get 10 vectors
    choose_number = 3
    labelVector = []
    for i in range(0, choose_number):
        index = np.argsort(w)[i]
        vector = w[index]*v[:, index]
        sortIndex = np.argsort(vector)
        sortedVector = [vector[i] for i in sortIndex]
        labelVectorPiece = ADMM3(sortedVector, sortIndex, 2, 5000)
        labelVector.append(labelVectorPiece)

    for i in range(choose_number, 6):
        index = np.argsort(w)[i]
        vector = w[index]*v[:, index]
        sortIndex = np.argsort(vector)
        sortedVector = [vector[i] for i in sortIndex]
        labelVectorPiece = ADMM3(sortedVector, sortIndex, 1, 5000)
        labelVector.append(labelVectorPiece)

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


def superpixels_neighbor_method_plot(reLabel, position, row, col, segments, image, vector_space_label=None, distance_graph=None, distance_label=None):
    # start_time = time.time()
    # initByFloyd()
    # print(time.time()-start_time)

    # matrix, position, row, col, segments = superpixels_neighbor_method(percent)

    # image = scipy.misc.imread('../image/question_reshape.jpg', mode="L")
    # print(image)

    # superpixels method
    # matrix, position, row, col, segments = superpixels()
    # superpixels neighbor method
    start_time = time.time()

    # tradition method, parameters given by users
    # reLabel = classMatrix(matrix)

    # new method, auto choose parameters
    # reLabel = clustering_backup.matrix_to_label(matrix, vector_space_label)


    # reLabel = clustering_backup.matrix_to_label(distance_graph, distance_label)


    # print("reLabel-----------------------:", reLabel)

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
    # image = scipy.misc.imread(IMG, mode="RGB")
    # image = img_as_float(image)

    # matplotlib.interactive(True)

    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    # plot color by color=(R, G, B)
    # plt.imshow(mark_boundaries(image, finalLabel, color=(0,0,1)))

    # another plot method, different contour color for different labels
    plt.imshow(image)
    for l in range(len(labelVec)):
        plt.contour(finalLabel==l, contour=1, colors=[plt.cm.spectral(l/float(len(labelVec)))])

    matplotlib.interactive(False)
    plt.show()
    # imsave(PATH, mark_boundaries(image, finalLabel, color=(0,0,1)))

    # our segmentation figure
    # seg_boundary = find_boundaries(finalLabel).astype(np.uint8)
    # for i in range(0, seg_boundary.shape[0]):
    #     for j in range(0, seg_boundary.shape[1]):
    #         if seg_boundary[i, j] == 1:
    #             seg_boundary[i, j] = 255
    # imsave(GRAYPATH, seg_boundary, cmap='gray')

    # plt.figure(2)
    # label = find_boundaries(finalLabel)
    # plt.imshow(label)
    # plt.imshow(mark_boundaries(image, segments))
    # matplotlib.interactive(False)
    # plt.show()

def small_scale_image_method_plot():
    start_time = time.time()
    matrix, row, col, vector_space_label = small_scale_image_floyd_method()

    # tradition method, parameters given by users
    # reLabel = classMatrix(matrix)

    # new method, auto choose parameters

    # # plot each iteration image segmentation
    # reLabel_vector = clustering_backup.matrix_to_label(matrix)
    # for reLabel in reLabel_vector:
    #     finalLabel = np.reshape(reLabel, (row,col))
    #     finalLabel = np.asarray(finalLabel, dtype=int)
    #     print("finalLabel: ", finalLabel)
    #     print("time cost", time.time()-start_time)
    #     # show the output of superpixels method
    #     image = scipy.misc.imread(IMG, mode="RGB")
    #     image = img_as_float(image)
    #     fig = plt.figure()
    #     # ax = fig.add_subplot(1, 1, 1)
    #     plt.imshow(mark_boundaries(image, finalLabel, color=(0,0,1)))

    # plot the final image segmentation
    reLabel = clustering_backup.matrix_to_label(matrix, vector_space_label)
    finalLabel = np.reshape(reLabel, (row,col))
    finalLabel = np.asarray(finalLabel, dtype=int)
    print("finalLabel: ", finalLabel)
    print("time cost", time.time()-start_time)
    # show the output of superpixels method
    # image = scipy.misc.imread(IMG, mode="RGB")
    image = scipy.misc.imread(IMG, mode="L")
    image = img_as_float(image)
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    plt.imshow(mark_boundaries(image, finalLabel, color=(0,0,1)))
    # imsave(PATH, mark_boundaries(image, finalLabel, color=(0,0,1)))

    # our segmentation figure
    # seg_boundary = find_boundaries(finalLabel).astype(np.uint8)
    # for i in range(0, seg_boundary.shape[0]):
    #     for j in range(0, seg_boundary.shape[1]):
    #         if seg_boundary[i, j] == 1:
    #             seg_boundary[i, j] = 255

    # imsave(GRAYPATH, seg_boundary, cmap='gray')

    # plt.figure(2)
    # label = find_boundaries(finalLabel)
    # plt.imshow(label)
    # plt.imshow(mark_boundaries(image, segments))
    matplotlib.interactive(False)
    plt.show()


def slic_our_method():
    '''
    SLIC to construct distance matrix, our partition method to eigenspace
    '''

    # IMAGE_NAME = "113009.jpg"
    # IMAGE_NAME = "3063.jpg"
    IMAGE_NAME = "385028.jpg"

 #   IMAGE = "/home/yy/matlab/BSR/BSDS500/data/images/train/" + IMAGE_NAME
    IMAGE = "/home/yy/berkeley_datasets/BSR/BSDS500/data/images/train/" + IMAGE_NAME
    # IMAGE = "/home/yy/matlab/BSR/BSDS500/data/images/test/" + IMAGE_NAME
    img = scipy.misc.imread(IMAGE)

    # img = data.coffee()
    segments = segmentation.slic(img, compactness=30, n_segments=N)
    segments_copy = deepcopy(segments)
    # segments = np.asarray(segments)
    # row = image.shape[0]
    # col = image.shape[1] # row = segments.shape[0]
    row = img.shape[0]
    col = img.shape[1]
    segments_copy = np.reshape(segments_copy, segments_copy.shape[0]*segments_copy.shape[1])


    # generate position method
    segmentsLabel = []
    for i in range(row):
        for j in range(col):
            l = segments[i, j]
            if l not in segmentsLabel:
                segmentsLabel.append(l)

    # print(segmentsLabel)
    print(len(segmentsLabel))

    position = []
    for i in segmentsLabel:
        pixel_position = []
        for m in range(row):
            for n in range(col):
                if segments[m, n] == i:
                    pixel_position.append([m, n])
        position.append(pixel_position)

    # generate graph matrix
    g = graph.rag_mean_color(img, segments, mode='similarity', connectivity=4)
    # g = graph.rag_mean_color(img, segments, mode='distance', connectivity=3)
    # print(g.edges(data=True))
    print(len(g.edges(data=True)))

    w = nx.to_scipy_sparse_matrix(g, format='csc')
    entries = w.sum(axis=0)
    d = sparse.dia_matrix((entries, 0), shape=w.shape).tocsc()
    m = w.shape[0]
    d2 = d.copy()
    d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)

    matrix = d2 * (d - w) * d2
    matrix = matrix.toarray()

    print("-------------------------matrix:", matrix)
    # distance of superpixels not neighbor = inf
    # inf_matrix = deepcopy(matrix)
    # for i in range(matrix.shape[0]):
    #     for j in range(matrix.shape[1]):
    #         if matrix[i, j] == 0:
    #             inf_matrix[i, j] = 10000

    # distane of superpixels with continuity
    # continue_matrix = floyd_func(matrix)
    # print("continu matrix", continue_matrix)

    matplotlib.interactive(True)

    vals, vectors = SCI.eigsh(d2 * (d - w) * d2, which='SM', k=min(100, m - 2))
    vals, vectors = np.real(vals), np.real(vectors)
    index1, index2, index3 = np.argsort(vals)[0], np.argsort(vals)[1], np.argsort(vals)[2]
    ev1, ev2, ev3 = vectors[:, index1], vectors[:, index2], vectors[:, index3]
    # sortIndex = np.argsort(ev2)
    # sortedVector = [ev2[i] for i in sortIndex]
    # reLabel, mmm = clustering_backup.ADMM3(sortedVector, sortIndex, 10, 100000)
    reLabel = admm_algorithms.admm(n_vector=ev2, num_cuts=10)
    fig = plt.figure()
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(ev1, ev2, ev3, c=reLabel)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("n_cut eigenspace")

    fig = plt.figure()
    # plt.plot(ev2)
    plt.scatter(ev2, np.ones(len(ev2)), c=reLabel)
    plt.title("n_cut vector")

    # print("vals", vals)
    # # plot eigenvector space with vector_space_label
    # fig = plt.figure()
    # ax = Axes3D(fig, elev=-150, azim=110)
    # # ax.scatter(vector0, vector1, vector2, c=vector_space_label)
    # # ax.scatter(vector0, vector1, vector2)
    # # ax.scatter(vals[0]*vectors[:, 0], vals[1]*vectors[:, 1], vals[2]*vectors[:, 2])
    # ax.scatter(vals[-1]*vectors[:, -1], vals[-2]*vectors[:, -2], vals[-3]*vectors[:, -3])
    # # ax.set_title("vector space with labels")
    # plt.show()


    # # init the matrix
    # length = len(segmentsLabel)
    # # matrix = np.zeros((length, length))
    # matrix = np.identity(length)
    # count_new = 0
    # for x, y, d in g.edges(data=True):
    #     matrix[x, y] = abs(d['weight'])
    #     matrix[y, x] = abs(d['weight'])
    #     count_new += 1
    # print("count", count_new)

    # fig = plt.figure()
    # lc = graph.show_rag(segments, g, img)
    # cbar = plt.colorbar(lc)
    # print(cbar)
    # plt.show()

    # parameters: matrix, position, row, col, segments

    superpixels_neighbor_method_plot(reLabel, position, row, col, segments, img)
    # superpixels_neighbor_method_plot(matrix, position, row, col, segments, img)
    # superpixels_neighbor_method_plot(inf_matrix, position, row, col, segments, img)
    # superpixels_neighbor_method_plot(continue_matrix, position, row, col, segments, img)

if __name__ == "__main__":
    # superpixels_neighbor_method_plot()

    # small_scale_image_method_plot()

    # percent = 0.6
    # # for percent in [0.2, 0.4]:
    # # for percent in np.arange(0.1, 0.9, 0.1):
    # if True:
    #     # print("----------percent---------:", percent)

    #     # matrix, position, row, col, segments = superpixels_neighbor_method(percent)
    #     # superpixels_neighbor_method_plot(matrix, position, row, col, segments)

    #     superpixels_neighbor_method_plot(percent)

    # plt.show()

    slic_our_method()

    # image = scipy.misc.imread(IMG, mode="RGB")
    # matrix, position, row, col, segments, vector_space_label, distance_graph, distance_label = superpixels_neighbor_method()
    # superpixels_neighbor_method_plot(matrix, position, row, col, segments, image, vector_space_label, distance_graph, distance_label)
