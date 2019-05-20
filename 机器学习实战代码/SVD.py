#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-20 10:34:35
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from numpy import *
from numpy import linalg as la


def loadDataSet():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


def loadDataSet2():
    return [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]


def loadExData():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# 相似度计算函数
# 欧式距离---列向量参与运算
def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))  # la.norm是计算列向量的L2范数


# 皮尔逊系数
def pearSim(inA, inB):
    if (len(inA) < 3):
        return 1.0
    else:
        return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


# 余弦相似度
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


# 基于物品(列)进行相似度的推荐引擎
def standEst(dataMat, user, simMeas, item):
    """
        dataMat:数据集
        user:用户
        simMeans:相似度衡量方式
        item:物品
    """
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overlap = nonzero(logical_and(
            dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overlap) == 0:
            similarity = 0.0
        else:
            similarity = simMeas(dataMat[overlap, item], dataMat[overlap, j])
            print("the %d and %d similarity is: %f" % (item, j, similarity))
            simTotal += similarity
            ratSimTotal += similarity * userRating
    if simTotal == 0.0:
        return 0
    else:
        return ratSimTotal / simTotal


# 基于SVD的评分估计
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[:4])  # 包含奇异值能量的90%
    xformedItems = dataMat.T * U[:, :4]
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        else:
            similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
            print("the %d and %d similarity is: %f" % (item, j, similarity))
            simTotal += similarity
            ratSimTotal += similarity * userRating
    if simTotal == 0.0:
        return 0
    else:
        return ratSimTotal / simTotal


# 推荐函数
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # 寻找未评分的物品
    if len(unratedItems) == 0.0:
        return "you rated everything"
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


if __name__ == "__main__":
    dataMat = loadDataSet()
    print("----------奇异值计算-----------")
    U, Sigma, VT = la.svd(dataMat)
    print("Sigma:\n", Sigma)
    print("---------------重构原始矩阵dataMat------------")
    Sigma3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    dataMatload = U[:, :3] * Sigma3 * VT[:3, :]
    print("重构的原始矩阵:\n", dataMatload)
    print("----------------几种相似度计算方法-----------------")
    myData = mat(dataMat)
    print("欧式距离:\n", ecludSim(myData[:, 0], myData[:, 4]))
    print("欧式距离:\n", ecludSim(myData[:, 4], myData[:, 4]))
    print("皮尔逊系数:\n", pearSim(myData[:, 0], myData[:, 4]))
    print("皮尔逊系数:\n", pearSim(myData[:, 4], myData[:, 4]))
    print("余弦相似度:\n", cosSim(myData[:, 0], myData[:, 4]))
    print("余弦相似度:\n", cosSim(myData[:, 4], myData[:, 4]))
    print("------------基于物品相似度的推荐引擎-----------")
    dataMat2 = loadDataSet2()
    myData2 = mat(dataMat2)
    print("默认推荐结果:\n", recommend(myData2, user=2))
    print("使用欧式距离,推荐结果:\n", recommend(myData2, user=2, simMeas=ecludSim))
    print("使用皮尔逊系数,推荐结果:\n", recommend(myData2, user=2, simMeas=pearSim))
    print("------------基于SVD相似度的推荐引擎-----------")
    myData3 = mat(loadExData())
    U, Sigma, VT = la.svd(myData3)
    print("奇异值矩阵Sigma:\n", Sigma)
    Sigma2 = Sigma ** 2
    sumSigma = sum(Sigma2)
    print("总能量:", sumSigma)
    print("总能量的90%:", sumSigma * 0.9)
    print("前两个元素所包含的能量:", sum(Sigma2[:2]))
    print("前三个元素所包含的能量:", sum(Sigma2[:3]))  # 将11维矩阵降维到3维矩阵
    print("使用默认相似度的SVD进行评分:", recommend(myData3, user=1, estMethod=svdEst))
    print("使用皮尔逊系数的SVD进行评分:", recommend(myData3, user=1,
                                  estMethod=svdEst, simMeas=pearSim))
