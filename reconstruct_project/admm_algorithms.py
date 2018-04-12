"""
===============
admm algorithms
===============

A partition method with ADMM method, code by pure python.
"""

print(__doc__)

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

def admm(n_vector, num_cuts=2, n_iter=10000):
    """  Assign labels to input vector.

    Minimize cost funtion: argmin_x' | x - x' |^2, s.t. v' = Ax', |v|_0 <= a,
    while x is the input vector, a is cut_number.

    Parameters
    ----------
    num_cuts : int
        Number of iteration will cut on this vector/eigenvector.
    n_iter : int
        itertion times of sub-problems.
    n_vector : ndarray
        Input vector, as x in the cost function.

    Returns
    -------
    out : ndarray
        The new labeled array.

    """

    # sort the input vector and get the sorted index
    sort_index = np.argsort(n_vector)
    sort_vector = [n_vector[i] for i in sort_index]
    sort_vector = np.asarray(sort_vector)
    length = len(sort_vector)

    # initilize variables of cost function
    A = np.zeros((length-1, length))
    for i in range(length-1):
        A[i][i] = -1
        A[i][i+1] = 1
    I = np.identity(length)
    iteration = 0
    x = deepcopy(sort_vector)
    x1 = deepcopy(sort_vector)
    v = np.dot(A, x1)
    w = np.dot(A, x1)
    r = 0.01
    n = 0.99 # n belong to [0.95, 0.99]

    ''' solve cost function by three sub-equations,
    x' = inv(I + 1/r * A.T * A) * (x1 + 1/r * A.T * (v - w))
    v' = A * x + w, if |z|_0 <= a,
      or sort(A*x+w)[:a] , if |z|_0 > a
    w' = w + A * x - v
    '''
    # left part of x out of while block to save computation time.
    left = np.linalg.inv(np.dot(1, I) + np.dot(1/r, np.dot(A.T, A)))
    while iteration < n_iter:
        right = np.dot(1, x1) + np.dot(1/r, np.dot(A.T, (v - w)))
        x = np.dot(left, right)
        z = np.dot(A, x) + w
        if np.linalg.norm(z, 0) <= num_cuts:
            v = deepcopy(z)
            # print("z<a")
        else:
            z_abs = abs(z)
            z_sort = np.argsort(z_abs)[::-1]
            v = np.zeros(length-1)
            for i in range(num_cuts):
                index = z_sort[i]
                v[index] = z[index]
        w = w + np.dot(A, x) - v

        # It seems reducing r in each iteration will converage fast (?)
        # r = r * n
        iteration += 1

    # assign labels by final gaps
    gap = np.dot(A, x)
    # print("___________gap_____________:", gap)
    k = num_cuts
    big_k_gap = np.sort(np.argsort(gap)[::-1][:k])
    print(big_k_gap)
    assign_label = np.zeros(length)
    label = 1
    for i in range(len(big_k_gap)-1):
        begin_indice = big_k_gap[i] + 1
        end_indice = big_k_gap[i+1] + 1
        print("begin and end", begin_indice, end_indice)
        assign_label[begin_indice:end_indice] = label
        label += 1
    assign_label[big_k_gap[-1]:] = label

    label_vector = np.zeros(length)
    for i in range(length):
        label_vector[sort_index[i]] = assign_label[i]
    # labelVec = np.zeros(len(unSortedLabelVec))
    # for i in range(len(unSortedLabelVec)):
    #     labelVec[sortIndex[i]] = unSortedLabelVec[i]
    return label_vector

if __name__ == "__main__":
    input_vector = np.random.rand(1000)
    # label_vector = admm(input_vector=input_vector)
    label_vector = admm(n_vector=input_vector)
    print(label_vector)
    plt.scatter(input_vector, np.ones(len(input_vector)), c=label_vector)
    plt.show()
