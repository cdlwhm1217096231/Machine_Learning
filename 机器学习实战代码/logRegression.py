# -*- coding: utf-8 -*-
import numpy as np
import random
"""
当数据集较小时，使用梯度上升算法
当数据集较大时，使用改进的随机梯度上升算法
"""

# sigmoid（）函数


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

# 改进的随机梯度上升算法


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 返回dataMatrix的大小。m为行数,n为列数。
    m, n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 降低alpha的大小，每次减小1/(j+i)。
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex))
                            )                # 随机选取样本
            # 选择随机选取的一个样本，计算h
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - \
                h                                 # 计算误差
            weights = weights + alpha * error * \
                dataMatrix[randIndex]       # 更新回归系数
            # 删除已经使用的样本
            del(dataIndex[randIndex])
    return weights                                                             # 返回

#  梯度上升算法


def gradAscent(dataMatIn, classLabels):
    # 转换成numpy的mat
    dataMatrix = np.mat(dataMatIn)
    # 转换成numpy的mat,并进行转置
    labelMat = np.mat(classLabels).transpose()
    # 返回dataMatrix的大小。m为行数,n为列数。
    m, n = np.shape(dataMatrix)
    # 移动步长,也就是学习速率,控制更新的幅度。
    alpha = 0.01
    maxCycles = 500                                                        # 最大迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # 梯度上升矢量化公式
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    # 将矩阵转换为数组，并返回
    return weights.getA()

# 分类函数


def classifyVector(inX, weights):
    p = sigmoid(sum(inX * weights))
    if p > 0.5:
        return 1.0
    else:
        return 0.0


# 使用Python写的Logistic分类器做预测-----使用改进的随机梯度上升算法训练
# def colicTest():
"""下面是训练部分，得到最优的回归系数"""
#     frTrain = open('horseColicTraining.txt')                                        # 打开训练集                                              # 打开测试集
#     trainingSet = []
#     trainingLabels = []
#     for line in frTrain.readlines():
#         currLine = line.strip().split('\t')
#         lineArr = []   # 存放特征的列表
#         for i in range(len(currLine)-1):
#             lineArr.append(float(currLine[i]))
#         trainingSet.append(lineArr) # 除去类别标签的训练集数据，只含有特征
#         trainingLabels.append(float(currLine[-1])) # 训练集数据，只含有类别标签
#     trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)        # 使用改进的随机梯度上升算法训练，得到回归系数
"""下面是测试部分"""
# frTest = open('horseColicTest.txt')   # 打开测试集
#     errorCount = 0  # 分类错误次数，初始化为0
#     numTestVec = 0.0
#     for line in frTest.readlines():
#         numTestVec += 1.0
#         currLine = line.strip().split('\t')
#         lineArr =[]
#         for i in range(len(currLine)-1):
#             lineArr.append(float(currLine[i]))
#         if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
#             errorCount += 1
#     errorRate = (float(errorCount)/numTestVec) * 100                                 # 错误率计算
#     print("测试集错误率为: %.2f%%" % errorRate)
#     print("测试集所属类别:", int(classifyVector(np.array(lineArr), trainWeights[:, 0])))

# 使用Python写的Logistic分类器做预测------使用梯度上升算法进行训练


def colicTest():
    """下面是训练部分，得到最优的回归系数"""
    frTrain = open(
        'horseColicTraining.txt')                                        # 打开训练集
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = gradAscent(np.array(trainingSet),
                              trainingLabels)        # 使用梯度上升算法进行训练，得到回归系数
    """下面是测试部分"""
    frTest = open('horseColicTest.txt')  # 打开测试集
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[:, 0])) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec) * 100                                 # 错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)
    print("测试集所属类别:", int(classifyVector(np.array(lineArr), trainWeights[:, 0])))


if __name__ == '__main__':
    colicTest()
