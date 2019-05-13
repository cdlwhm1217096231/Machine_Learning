#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-13 11:27:16
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties



# 加载数据集
def loadDataSet(fileName):
    num_feature = len(open(fileName).readline().split("\t"))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(num_feature - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 构建单层决策树分类函数
def stumpClassify(dataMat, dimen, threshval, threshIneq):
    """
        dataMat:数据矩阵
        dimen:第dimen列，即第几个特征
        threshval:阈值
        threshIneq:标志
        返回值retArray：分类结果
    """
    retArray = np.ones((np.shape(dataMat)[0], 1))  # 初始化retArray为1
    if threshIneq == "lt":
        retArray[dataMat[:, dimen] <= threshval] = -1.0   # 如果小于阈值,则赋值为-1
    else:
        retArray[dataMat[:, dimen] > threshval] = 1.0   # 如果大于阈值,则赋值为-1
    return retArray


# 找到数据集上最佳的单层决策树，单层决策树是指只考虑其中的一个特征，在该特征的基础上进行分类，寻找分类错误率最低的阈值即可。例如本文中的例子是，如果以第一列特征为基础，阈值选择1.3，并且设置>1.3的为-1,<1.3的为+1,这样就构造出了一个二分类器
def buildStump(dataMat, classLabels, D):
    """
        dataMat:数据矩阵
        classLabels:数据标签
        D：样本权重
        返回值是:bestStump:最佳单层决策树信息;minError:最小误差;bestClasEst：最佳的分类结果
    """
    dataMat = np.matrix(dataMat)
    labelMat = np.matrix(classLabels).T
    m, n = np.shape(dataMat)
    numSteps = 10.0
    bestStump = {}  # 存储最佳单层决策树信息的字典
    bestClasEst = np.mat(np.zeros((m, 1)))   # 最佳分类结果
    minError = float("inf")
    for i in range(n):  # 遍历所有特征
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ["lt", "gt"]:
                threshval = (rangeMin + float(j) * stepSize)  # 计算阈值
                predictVals = stumpClassify(
                    dataMat, i, threshval, inequal)  # 计算分类结果
                errArr = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                errArr[predictVals == labelMat] = 0  # 分类完全正确，赋值为0
                # 基于权重向量D而不是其他错误计算指标来评价分类器的，不同的分类器计算方法不一样
                weightedError = D.T * errArr  # 计算弱分类器的分类错误率---这里没有采用常规方法来评价这个分类器的分类准确率，而是乘上了样本权重D
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                    i, threshval, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictVals.copy()
                    bestStump["dim"] = i
                    bestStump["thresh"] = threshval
                    bestStump["ineq"] = inequal
    return bestStump, minError, bestClasEst


# 使用Adaboost算法提升弱分类器性能
def adbBoostTrainDS(dataMat, classLabels, numIt=50):
    """
        dataMat:数据矩阵
        classLabels:标签矩阵
        numIt:最大训练次数
        返回值：weakClassArr  训练好的分类器   aggClassEst:类别估计累计值
    """
    weakClassArr = []
    m = np.shape(dataMat)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化样本权重D
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(
            dataMat, classLabels, D)  # 构建单个单层决策树
        # 计算弱分类器权重alpha,使error不等于0,因为分母不能为0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump["alpha"] = alpha   # 存储每个弱分类器的权重alpha
        weakClassArr.append(bestStump)  # 存储单个单层决策树
        print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha *
                            np.mat(classLabels).T, classEst)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # 计算AdaBoost误差,当误差为0时，退出循环
        aggClassEst += alpha * classEst  # 计算类别估计累计值--注意这里包括了目前已经训练好的每一个弱分类器
        print("aggClassEst:", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(
            classLabels).T, np.ones((m, 1)))  # 目前集成分类器的分类误差
        errorRate = aggErrors.sum() / m  # 集成分类器分类错误率，如果错误率为0，则整个集成算法停止，训练完成
        print("total error:", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


# 绘制ROC(接受者操作特性曲线)

def plot_ROC(predStrengths, classLabels):
    """
        predStrengths:分类器的预测强度
        classLabels:类别
    """
    font = FontProperties(fname=r"C:\\Windows\Fonts\\simsun.ttc", size=14)
    cur = (1.0, 1.0)  # 绘制光标位置
    ysum = 0.0  # 用于计算AUC
    numPosClas = np.sum(np.array(classLabels) == 1.0)  # 统计正类的数量
    yStep = 1 / float(numPosClas)  # y轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)  # x轴步长

    sortedIndicies = predStrengths.argsort()  # 预测强度排序,从低到高
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ysum += cur[1]  # 注意，每次cur[1]都可能变化了
        ax.plot([cur[0], cur[0] - delX],
                [cur[1], cur[1] - delY], c='b')  # 绘制ROC
        cur = (cur[0] - delX, cur[1] - delY)  # 更新绘制光标的位置
    ax.plot([0, 1], [0, 1], 'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties=font)
    plt.xlabel('假阳率', FontProperties=font)
    plt.ylabel('真阳率', FontProperties=font)
    ax.axis([0, 1, 0, 1])
    print('AUC面积为:', ysum * xStep)  # 计算AUC
    plt.show()


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet(
        "E:\\Code\\local_code\\机器学习实战\\DataSet\\horseColicTraining2.txt")
    weakClassArr, aggClassEst = adbBoostTrainDS(dataMat, labelMat, 50)
    plot_ROC(aggClassEst.T, labelMat)
