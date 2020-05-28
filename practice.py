import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import struct
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from random import seed
from random import randint

# matplotlib inline

# scroll to bottom to call each part individually

# GIVEN
train = dict()
test = dict()


def get_images(filename):
    with gzip.GzipFile(Path('mnist', filename), 'rb') as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        print(magic, size, rows, cols)
        images = np.frombuffer(f.read(), dtype=np.dtype('B'))
    return images.reshape(size, rows, cols)


train['image'] = get_images('train-images-idx3-ubyte.gz')
test['image'] = get_images('t10k-images-idx3-ubyte.gz')
print(train['image'].shape, test['image'].shape)


def get_labels(filename):
    with gzip.GzipFile(Path('mnist', filename), 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        # print(magic, num)

        labels = np.frombuffer(f.read(), dtype=np.dtype('B'))
    return labels


train['label'] = get_labels('train-labels-idx1-ubyte.gz')
test['label'] = get_labels('t10k-labels-idx1-ubyte.gz')
testLen = 5000
testLen2 = 60000


# PART 1


def percentError(valA, valE):
    return abs(((valA-valE)/valE))*100


def sign(p):
    if (p >= 0):
        return 1
    else:
        return -1


def setY():
    # num = 0
    Ys = np.ones([10, testLen])

    for j in range(10):
        T = np.ones([testLen])
        T *= -1
        for i in range(testLen):
            if(test['label'][i] == j):
                T[i] *= -1
        Ys[j] = T
    return Ys


def getY(number, v):

    return v[number]


def xTilda(x):
    xT = x.T
    dProduct = xT @ x
    inv = np.linalg.pinv(dProduct)
    A = inv @ xT
    return A


def Beta(x, y):
    return x @ y


def getBeta(x, y, xt, v):
    weights = np.zeros([10])
    for j in range(testLen):
        for i in range(10):
            B = Beta(xt, getY(i, v))
            weights[i] = x[j] @ B
        maxElement = np.amax(weights)
        index = np.where(weights == maxElement)
        y[j] = index[0][0]
    return y


def offSet(B, X, Y):
    BT = B.T
    val = BT @ X
    return Y - val


def getXA(Arr, l):
    xa = np.ones([l, 784])
    for i in range(l):  # len(test(['label]))
        image = np.asarray(Arr[i]).flatten()
        xa[i] = image

    return xa


def part1(xa):

    print("\nPart 1 starting....")
    print('getting xtilda')
    xat = xTilda(xa)
    print("setting ys")
    val = setY()
    ya = np.ones([testLen])
    print("getting weights")
    B = Beta(xat, getY(0, val))
    ya = getBeta(xa, ya, xat, val)
    v = offSet(B, xat, ya)
    c = correct(ya)
    print("The offset is: ", v)
    print("Accuracy: ", (c/testLen)*100, "%")


def correct(y):
    inc = 0
    for i in range(testLen):
        if (y[i] == test['label'][i]):
            inc += 1
    return inc


# PART 2

def dH(x):
    newArr = np.zeros((28, 28))
    n = np.reshape(x, (28, 28))

    for i in range(28):
        for j in range(27):
            newArr[i][j] = (n[i][j]-n[i][j+1])**2
    return newArr


def dV(x):
    newArr = np.zeros((28, 28))
    n = np.reshape(x, (28, 28))

    for i in range(27):
        for j in range(28):
            newArr[i][j] = (n[i][j]-n[i+i][j])**2
    return newArr


def part2(xa, index):
    newArr = np.ones([testLen, 784])
    newArr2 = np.ones([testLen, 784])
    newArr3 = np.ones([testLen, 784])
    newArr4 = np.ones([testLen, 784])

    plank = 1
    plank2 = 10 ** -3
    plank3 = 10 ** -6
    plank4 = 10 ** -9
    v2 = setY()

    print("\npart 2 starting...")
    for i in range(testLen):
        dh = dH(xa[i])
        dv = dH(xa[i])
        sums = dv+dh
        dhv = np.asarray(sums.flatten())
        newArr[i] = xa[i] + dhv*plank
        newArr2[i] = xa[i] + dhv*plank2
        newArr3[i] = xa[i] + dhv*plank3
        newArr4[i] = xa[i] + dhv*plank4
    print("newarray made")
    print("setting Ys")
    v2 = setY()
    ya2 = np.ones([testLen])

    print("\n𝜆 = 1")
    print("getting xtilda")
    nAT = xTilda(newArr)
    print("getting ya")
    ya2 = getBeta(newArr, ya2, nAT, v2)
    print("Calculating accuracy...")
    c = correct(ya2)
    print("Accuracy: ", c/testLen*100, "%")

    print("\n𝜆 = 10^-3")
    print("getting xtilda")
    nAT = xTilda(newArr2)
    print("getting ya")
    ya2 = getBeta(newArr2, ya2, nAT, v2)
    print("Calculating accuracy...")
    c = correct(ya2)
    print("Accuracy: ", c/testLen*100, "%")

    print("\n𝜆 = 10^-6")
    print("getting xtilda")
    nAT = xTilda(newArr3)
    print("getting ya")
    ya2 = getBeta(newArr3, ya2, nAT, v2)
    print("Calculating accuracy...")
    c = correct(ya2)
    print("Accuracy: ", c/testLen*100, "%")

    print("\n𝜆 = 10^-9")
    print("getting xtilda")
    nAT = xTilda(newArr4)
    print("getting ya")
    ya2 = getBeta(newArr4, ya2, nAT, v2)
    print("Calculating accuracy...")
    c = correct(ya2)
    print("Accuracy: ", c/testLen*100, "%")

    # print(dh)
    disp = np.reshape(newArr[index], (28, 28))
    disp2 = np.reshape(newArr2[index], (28, 28))
    disp3 = np.reshape(newArr3[index], (28, 28))
    disp4 = np.reshape(newArr4[index], (28, 28))

    print("example of different values of Lambda plotting...")
    fig, ax = plt.subplots()
    plt.suptitle("Part2: Lambda = 1")
    _ = ax.imshow(disp, cmap='gray')

    fig, ax = plt.subplots()
    plt.suptitle("Part2: Lambda = 10^-3")
    _ = ax.imshow(disp2, cmap='gray')

    fig, ax = plt.subplots()
    plt.suptitle("Part2: Lambda = 10^-6")
    _ = ax.imshow(disp3, cmap='gray')

    fig, ax = plt.subplots()
    plt.suptitle("part2: Lambda = 10^-9")
    _ = ax.imshow(disp4, cmap='gray')

    plt.show()


def getIndex(Arr, l, K):
    seed(1)
    value = randint(0, 5000)
    indx = 0
    for i in range(l):
        if(Arr[value+i] == int(K)):
            indx = value+i
            break
    return indx


def part3(X, indx):
    print("\nPart 3 starting...")

    inpt = X[indx].reshape(28, 28)
    inpt = StandardScaler().fit_transform(inpt)
    mx = np.mean(inpt)
    cov_mat = np.cov(inpt.T)
    # Compute the eigen values and vectors using numpy
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                 for i in range(len(eig_vals))]

    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    print(eig_pairs)

    Arr = []
    vArr = []

    print("10 highest eigenvalues and corresponding Vectors: ")
    for i in range(28):
        Arr.append(eig_pairs[i][1].T)
        vArr.append(eig_pairs[i][0].T)
        if(i < 10):
            print(eig_pairs[i])
    Arr = np.asarray(Arr)
    print("Eigen Values and Eigen Vectors Printed!!")
    repmat = np.tile(mx, (28, 1))
    pca_data = Arr@(inpt-repmat).T  # y=A𝜆

    Xr = repmat+(Arr.T@pca_data).T  # Xr= (A^T y)^T

    fig, ax = plt.subplots()
    plt.suptitle("Part3: Original")
    _ = ax.imshow(X[indx].reshape(28, 28), cmap='gray')

    fig, ax = plt.subplots()
    plt.suptitle("Part3: Eigen Vectors")
    _ = ax.imshow(Xr, cmap='gray')
    plt.show()


K = input("enter a number to get values: ")
x1 = getXA(test['image'], testLen)
x2 = getXA(train['image'], testLen2)
index1 = getIndex(test['label'], testLen, K)
index2 = getIndex(train['label'], testLen2, K)
# part1(x1)
# part2(x1, index1)
part3(x2, index2)
