# -*- coding: utf-8 -*-
"""
梯度上升法算法找到最佳回归系数
"""
import numpy as np
import matplotlib.pyplot as plt

# 加载训练集数据


def loadDataSet():
    dataMat = []
    labelMat = []
    filename = 'testSet.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[-1]))
    return dataMat, labelMat

# sigmoid的函数


def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))


# 梯度上升法,确定最佳回归系数


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for i in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()

# 绘制分类边界图


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()  # 加载数据集
    dataArr = np.array(dataMat)   # 转换成numpy的array数组
    print('训练数据集:\n', dataMat)
    n = len(dataMat)      # 数据个数
    print('训练样本个数:', n)
    xcord1 = []                  # 正样本为1
    ycord1 = []
    xcord2 = []                  # 负样本为0
    ycord2 = []
    for i in range(n):  # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])  # 1为正样本
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])  # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord1, ycord1, s=20, c='blue', marker='s', alpha=0.5)  # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='yellow', alpha=0.5)  # 绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]  # 分隔线方程
    ax.plot(x, y)
    plt.title('BestFit', fontsize=24)  # 绘制title
    plt.xlabel('X1', fontsize=14)
    plt.ylabel('X2', fontsize=14)  # 绘制label
    plt.show()




if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    print('最佳回归系数:\n', gradAscent(dataMat, labelMat))
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)