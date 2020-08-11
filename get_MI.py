from math import log
import pandas as pd
import numpy as np

def mutual_information(X, Y):
    '''计算互信息'''
    X = pd.Series(X)
    Y = pd.Series(Y)

    total = len(X)
    eps = 1.4e-45

    # 四等分
    x_0 = X.quantile(0)
    x_1 = X.quantile(1 / 4)
    x_2 = X.quantile(2 / 4)
    x_3 = X.quantile(3 / 4)
    x_4 = X.quantile(1) + eps

    y_0 = Y.quantile(0)
    y_1 = Y.quantile(1 / 4)
    y_2 = Y.quantile(2 / 4)
    y_3 = Y.quantile(3 / 4)
    y_4 = Y.quantile(1) + eps
    x_range = [[x_0, x_1],
               [x_1, x_2],
               [x_2, x_3],
               [x_3, x_4]
               ]
    y_range = [[y_0, y_1],
               [y_1, y_2],
               [y_2, y_3],
               [y_3, y_4]
               ]
    # 计算互信息
    MI = 0
    for x in x_range:
        for y in y_range:
            x_occur = np.where((X >= x[0]) & (X < x[1]))
            y_occur = np.where((Y >= y[0]) & (Y < y[1]))
            xy_occur = np.intersect1d(x_occur, y_occur)
            # print(y,y_occur)
            px = 1.0 * len(x_occur[0]) / total
            py = 1.0 * len(y_occur[0]) / total
            pxy = 1.0 * len(xy_occur) / total
            # print(px,py,pxy)
            MI = MI + pxy * log(pxy / (px * py) + eps, 2)
    # 标准化互信息
    Hx = 0
    for x in x_range:
        x_occurCount = 1.0 * len(np.where((X >= x[0]) & (X < x[1]))[0])
        Hx = Hx - x_occurCount / total * log(x_occurCount / total + eps, 2)
    Hy = 0
    for y in y_range:
        y_occurCount = 1.0 * len(np.where((Y >= y[0]) & (Y < y[1]))[0])
        Hy = Hy - y_occurCount / total * log(y_occurCount / total + eps, 2)
    MI = 2.0 * MI / (Hx + Hy)
    return MI