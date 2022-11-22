import numpy as np
import matplotlib.pyplot as plt
import math


def init(X):
    c = 2
    # y = mx + c
    n = len(X)
    m = n
    pred = []
    for i in range(n):
        pred.append(m*X[i] + c)

    return pred


def Loss(Y, pred):
    m = len(Y)
    sq = 0
    for i in range(m):
        sq += (Y[i] - pred[i])**2

    mse = sq / m
    return mse


def der(X, Y, pred):
    m = len(X)
    da0 = 0
    for i in range(m):
        da0 += pred[i] - Y[i]
    da0 = (da0*2)/m

    da1 = 0
    for i in range(m):
        da1 += (pred[i] - Y[i]) * X[i]
    da1 = (da1 * 2) / m

    return da0, da1


def grad_descent(X, Y, d0, d1, n):
    m = len(X)
    for i in range(0, n):
        hyp = init(X)
        loss =


if __name__ == '__main__':
    X = np.random.randint(10, size=10)
    Y = np.random.randint(10, size=10)
    # print(X)
    # print(Y)

    pre = init(X)
    ans = Loss(Y, pre)
    d0, d1 = der(X, Y, pre)

    print(pre)
    print(ans)
    print(d0)
    print(d1)

    # a, b = np.polyfit(X, Y, 1)
    # plt.scatter(X, Y)
    # plt.plot(X, a*X+b)
    # plt.title('My first graph!')

    plt.show()
