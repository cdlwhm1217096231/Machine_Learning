#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-17 08:27:10
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from numpy import *
import urllib
import json


# K-means聚类算法，无监督学习算法


# 加载数据集
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = map(float, curLine)
        dataMat.append(list(fltLine))
    return dataMat  # 返回的是一个list，不是一个矩阵mat


# 欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))   # L2范数


# 构造出K个随机选择的初始化聚类中心
def randCent(dataset, K):
    n = shape(dataset)[1]
    centroids = mat(zeros((K, n)))  # 聚类中心矩阵初始化
    for j in range(n):  # 遍历每一维特征，找到每一维特征的最大和最小值来保证随机选择的聚类中心处于整个数据集中
        minj = min(dataset[:, j])
        rangej = float(max(dataset[:, j]) - minj)
        centroids[:, j] = mat(minj + rangej * random.rand(K, 1))
    return centroids


# KMeans算法
def KMeans(dataset, K, dist=distEclud, createCent=randCent):
    m = shape(dataset)[0]   # 所有的样本数
    clusterAssment = mat(zeros((m, 2)))  # 用于存储每个样本属于哪个簇和到聚类中心的最短距离
    # 创建初始的K个聚类中心
    centeroids = createCent(dataset, K)
    clusterChanged = True  # 循环停的条件：当某个样本的聚类中心不在改变时，停止迭代
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 遍历每个样本
            minDist = inf  # 最近距离初始化
            minIndex = -1  # 距离最近的是哪个聚类中心，此时是进行初始化
            for j in range(K):  # 遍历K个聚类中心
                # 对于所有的聚类中心，计算每个样本距离所有的聚类中心中的哪个最近
                # 使用欧式距离表示误差用来评价聚类效果
                distji = dist(centeroids[j, :], dataset[i, :])
                if distji < minDist:
                    minDist = distji
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 没有找到最近的距离，继续内循环
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist
        print("-------------------循环一次完成！---------------")
        print("聚类中心:\n", centeroids)
        for center in range(K):  # 更新聚类中心的位置：将聚类中心移动到每个簇的均值处，开始下一次循环
            # 得到属于第center个簇的所有样本ptsInClust
            ptsInClust = dataset[nonzero(clusterAssment[:, 0].A == center)[0]]
            centeroids[center, :] = mean(ptsInClust, axis=0)
    return centeroids, clusterAssment


"""
    KMeans聚类算法中，K是一个用户自定义的参数，如何知道K的选择是否正确？如何才能知道生成的簇比较好？
    clusterAssment矩阵中存储着每个样本属于哪个簇以及每个样本距离簇的最近距离(即误差),下面使用此误差来评价
    聚类算法的质量。
    衡量距离聚类算法效果的指标是SSE：误差平方和，即clusterAssment矩阵中的第二列的所有元素之和！
    SSE越小，表示样本越接近于聚类中心，聚类效果越好
"""


# 使用后处理来提高聚类性能
"""
    聚类的目标：在保持原有簇数量不变的条件下，提高簇的质量
    将具有最大SSE值的簇划分成两个簇，具体实现时，可以将最大簇包含的样本样过滤出来，然后在这些样本点上运行KMeans算法，令其中的K=2
"""


# 二分K-Means聚类算法
def biKMeans(dataSet, K, distMeas=distEclud):
    m = shape(dataSet)[0]  # 样本点数量
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 将所有样本看出一个簇，下面进行簇划分
    centList = [centroid0]  # 将划分得到的簇放在一个list中
    # 计算每个样本的误差
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < K):  # 划分得到的簇数量小于用户自定义的簇数量，继续划分
        lowestSSE = inf
        for i in range(len(centList)):
            # 得到属于第i个簇中的所有样本pstInCurrCluster
            ptsInCurrCluster = dataSet[nonzero(
                clusterAssment[:, 0].A == i)[0], :]
            # 对每个簇都划分成2个簇
            centroidMat, splitClustAss = KMeans(
                ptsInCurrCluster, 2, distMeas)
            # 计算划分后的总误差
            sseSplit = sum(splitClustAss[:, 1])
            # 计算没有划分前的不在当前第i个簇中的样本的总误差
            sseNotSplit = sum(clusterAssment[nonzero(
                clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit:\n", sseSplit)
            print("sseNotSplit:\n", sseNotSplit)
            # 如果对第i个簇的样本划分后的误差加上不是第i个簇样本的误差之和<没有对第i个簇样本进行划分前的误差和，则表示该此划分有效,SSE值是最大程度降低的
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 下面是实际的划分操作
        # 下面两行代码表示对划分后的新族进行重新编号 0 1 2 3 ...
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(
            centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)
                     [0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 添加到list中，保存每个簇
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        # 重新分配新的簇和对应的SSE
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[
            0], :] = bestClustAss
    return mat(centList), clusterAssment


if __name__ == "__main__":
    mydata = loadDataSet("E:\\Code\\local_code\\机器学习实战\\DataSet\\testSet.txt")
    mymat = mat(mydata)
    print("第一维特征的最小值:", min(mymat[:, 0]))
    print("第一维特征的最大值:", max(mymat[:, 0]))
    randCenter = randCent(mymat, K=2)
    print("随机选择的两个聚类中心:\n", randCenter)
    print("-----------KMeans聚类-------------")
    result = KMeans(mymat, 4)
    print(result)
    print("------------------------------------二分 KMeans聚类----------------------------------------")
    dataMat = mat(loadDataSet(
        "E:\\Code\\local_code\\机器学习实战\\DataSet\\testSet2.txt"))
    myCentroids, clustAssing = biKMeans(dataMat, 3)
