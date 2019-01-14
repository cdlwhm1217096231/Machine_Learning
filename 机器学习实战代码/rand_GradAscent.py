#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-20 10:50:44
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.font_manager import FontProperties


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
    return 1.0 / (1 + np.exp(-inX))


# 梯度上升法,确定最佳回归系数


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    weights_array = np.array([])
    for i in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA(), weights_array  # 将矩阵转换为数组，并返回


"""
改进的随机梯度上升算法,确定最佳回归系数
"""


def randGradAscent(dataMatrix, classLabels, numIter=150):  # 默认迭代次数150次
    m, n = np.shape(dataMatrix)         # 返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                # 参数初始化
    weights_array = np.array([])        # 存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + i + j) + 0.01                        # alpha在每次迭代时，都会调整，降低alpha的大小，每次减小1/(j+i)
            randIndex = int(random.uniform(0, len(dataIndex)))    # 随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))     # 选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                    # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            weights_array = np.append(weights_array, weights, axis=0)   # 添加回归系数到数组中
            del (dataIndex[randIndex])            # 删除已经使用的样本
    weights_array = weights_array.reshape(numIter * m, n)    # 改变维度
    return weights, weights_array
# 绘制分类边界图


# def plotBestFit(weights):
#     dataMat, labelMat = loadDataSet()  # 加载数据集
#     dataArr = np.array(dataMat)   # 转换成numpy的array数组
#     print('训练数据集:\n', dataMat)
#     n = len(dataMat)      # 数据个数
#     print('训练样本个数:', n)
#     xcord1 = []                  # 正样本为1
#     ycord1 = []
#     xcord2 = []                  # 负样本为0
#     ycord2 = []
#     for i in range(n):  # 根据数据集标签进行分类
#         if int(labelMat[i]) == 1:
#             xcord1.append(dataArr[i, 1])
#             ycord1.append(dataArr[i, 2])  # 1为正样本
#         else:
#             xcord2.append(dataArr[i, 1])
#             ycord2.append(dataArr[i, 2])  # 0为负样本
#     fig = plt.figure()
#     ax = fig.add_subplot(111)  # 添加subplot
#     ax.scatter(xcord1, ycord1, s=20, c='blue', marker='s', alpha=0.5)  # 绘制正样本
#     ax.scatter(xcord2, ycord2, s=20, c='yellow', alpha=0.5)  # 绘制负样本
#     x = np.arange(-3.0, 3.0, 0.1)
#     y = (-weights[0] - weights[1] * x) / weights[2]  # 分隔线方程
#     ax.plot(x, y)
#     plt.title('BestFit', fontsize=24)  # 绘制title
#     plt.xlabel('X1', fontsize=14)
#     plt.ylabel('X2', fontsize=14)  # 绘制label
#     plt.show()

# 绘制回归系数与迭代次数的关系
def plotWeights(weights_array1,weights_array2):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)   # 设置汉字格式
    #  将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20,10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')


    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1, weights_array1 = randGradAscent(np.array(dataMat), labelMat)
    print('改进的随机梯度上升法的最佳回归系数:\n', weights1)
    weights2, weights_array2 = gradAscent(dataMat, labelMat)
    print('梯度上升法的最佳回归系数:\n', weights2)
    plotWeights(weights_array1, weights_array2)
    # plotBestFit(weights)
