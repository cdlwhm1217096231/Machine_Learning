#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-13 09:13:52
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$


# 使用sklearn机器学习库，训练马疝病数据集，预测病马死亡率
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# 加载数据集
def loadDataSet(fileName):
    num_feature = len((open(fileName).readline().split("\t")))  # 特征的数量
    print("总的列数量:", num_feature)
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")  # 对其中的每行进行分割处理
        print("curLine:\n", curLine)
        for i in range(num_feature - 1):  # 不包括最后一列，最后一列是label
            lineArr.append(float(curLine[i]))  # 特征
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))  # 标签
    return dataMat, labelMat


if __name__ == "__main__":
    dataMat, classLabels = loadDataSet(
        "E:\\Code\\local_code\\机器学习实战\\DataSet\\horseColicTraining2.txt")
    print("----------------------训练过程--------------")
    adaboost_fun = AdaBoostClassifier(DecisionTreeClassifier(
        max_depth=2), algorithm="SAMME", n_estimators=10)
    adaboost_fun.fit(dataMat, classLabels)
    predictions = adaboost_fun.predict(dataMat)
    errArr = np.mat(np.ones((len(dataMat), 1)))
    print("训练集的错误率:%.3f%%" %
          float(errArr[predictions != classLabels].sum() / len(dataMat) * 100))
    print("----------------------预测过程--------------")
    testMat, testLabels = loadDataSet(
        "E:\\Code\\local_code\\机器学习实战\\DataSet\\horseColicTest2.txt")
    predictions = adaboost_fun.predict(testMat)
    errArr = np.mat(np.ones((len(testMat), 1)))
    print("测试集的错误率:%.3f%%" %
          float(errArr[predictions != testLabels].sum() / len(testMat) * 100))
