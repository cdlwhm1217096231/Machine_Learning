
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-19 15:30:22
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from numpy import *


# 加载数据集
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)


# PCA算法
def pca(dataMat, topNfeature=9999999):
    meanVals = mean(dataMat, axis=0)  # 计算原始数据集的均值
    meanRemoved = dataMat - meanVals  # 减去原始数据集的均值
    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差矩阵
    eigVals, eigVects = linalg.eig(mat(covMat))  # 求特征值和特征向量
    eigValInd = argsort(eigVals)  # 对特征值进行从小到大排序
    # 根据特征值排序结果的逆序，可以得到topNfeature个最大特征向量
    eigValInd = eigValInd[:-(topNfeature + 1):-1]
    redEigVects = eigVects[:, eigValInd]  # 上面的特征向量将构成后面对数据进行转换的矩阵
    lowDDataMat = meanRemoved * redEigVects  # 降维后的数据集
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean():
    dataMat = loadDataSet(
        "E:\\Code\\local_code\\机器学习实战\\DataSet\\secom.data", ' ')
    numfeatures = shape(dataMat)[1]
    for i in range(numfeatures):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:, i].A))[0], i])
        dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat
