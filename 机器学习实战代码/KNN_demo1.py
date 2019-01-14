#!/usr/bin/env  python
# -*- coding: utf-8 -*-
from os import listdir
import numpy as np
import operator


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # numpy函数shape[0]返回dataSet的行数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # # inX在列方向上重复共1次(横向),在行方向上重复共dataSetSize次
    sqDiffMat = diffMat**2  # 特征相减后进行平方
    sqDistances = sqDiffMat.sum(axis=1)  # 列元素压缩为一列，具体压缩方法是各列元素相加，最终得到一列
    distances = sqDistances**0.5  # 开方得欧式距离
    sortedDistIndices = distances.argsort()  # 返回distances中元素从小到大排序后的索引值
    classCount = {}  # 定义一个记录类别次数的字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]  # 取出前k个元素的类别
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计算类别次数
        # key=operator.itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(0)根据字典的键进行排序
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # reverse降序排序字典
        return sortedClassCount[0][0]  # 返回次数最多的类别,即所要分类的类别
"""
img2Vector(filename)函数说明:将一幅32x32的二进制图像转换为1x1024向量。

参数:
	filename - 文件名
返回值:
	returnVect - 返回的二进制图像的1x1024向量
"""


def img2Vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect
"""
函数说明:手写数字分类测试
参数:
	无
返回值:
	无
"""


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2Vector('trainingDigits/%s' % (fileNameStr))
#         以上部分为训练集部分,下面部分是测试集部分
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s' % (fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类器返回结果为%d\t真实结果为%d" % (classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print("分类器分类错误总次数%d" % errorCount)
    print("错误率:%f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    handwritingClassTest()









