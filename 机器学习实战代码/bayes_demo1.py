#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-05 10:21:49
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

"""
朴素贝叶斯算法改进:
1、利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，即计算p(w0|1)p(w1|1)p(w2|1)。
如果其中有一个概率值为0，那么最后概率的乘积也为0
解决办法：为了降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2
这种做法就叫做拉普拉斯平滑(Laplace Smoothing)又被称为加1平滑，是比较常用的平滑方法，它就是为了解决0概率问题。
2、另外一个遇到的问题就是下溢出，这是由于太多很小的数相乘造成的
解决方法：对乘积结果取自然对数。通过求对数可以避免下溢出或者浮点数舍入导致的错误。同时，采用自然对数进行处理不会有任何损失
"""
import numpy as np
from functools import reduce

# 加载每一篇文档，然后将每一篇文档(每个人的留言)经过切分，得到每个列表


def loadDateSet():
    primaryVocabularyLists = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                              ['maybe', 'not', 'take', 'him',
                                  'to', 'dog', 'park', 'stupid'],
                              ['my', 'dalmation', 'is', 'so',
                               'cute', 'I', 'love', 'him'],
                              ['stop', 'posting', 'stupid',
                                  'worthless', 'garbage'],
                              ['mr', 'licks', 'ate', 'my', 'steak',
                               'how', 'to', 'stop', 'him'],
                              ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]     # 每篇文档所属的类别，1代表是侮辱性词汇，0代表不是侮辱性词汇
    return primaryVocabularyLists, classVec
# 将每篇文档中的单词提取出来组成一个集合(即每个单词只出现一次)，此集合称为词汇表


def createVocabList(dataSet):  # dataSet --- 每篇文档或每条留言切分后得到的每个列表组成的样本数据集
    vocabuSet = set([])  # 创建一个空的集合，用于存放所有留言中出现的单词
    for document in dataSet:  # 遍历每篇文档/每条留言切分后得到的每个列表
        vocabuSet = vocabuSet | set(document)  # 取并集
    return list(vocabuSet)  # vocabuSet -----存放的是：所有留言中出现的单词(每个单词只出现一次)


# vocabulist即vocabuSet,inputSet即一篇新的文档或者一条新的留言
# 如果inputSet中的单词出现在vocabuList中，则记为1，反之记为0
# 返回的结果returnVec是将每篇文档/每条留言中的单词数字化，转化为词向量


def setOfWords2Vec(vocabuList, inputSet):
    returnVec = [0] * len(vocabuList)  # 创建一个与vocabuList相同长度的全0的向量
    for word in inputSet:   # 遍历每条留言/每篇文档中的每个单词
        if word in vocabuList:    # 如果每篇文档/每条留言中的单词出现在vocabuList中，则returnVec将对应于vocabuList索引位置上的元素置1，反之未出现时置0
            returnVec[vocabuList.index(word)] = 1
        else:
            print('单词:%s不在词汇表中！' % word)
    return returnVec  # returnVec是将每篇文档/每条留言中的单词数字化，转化为词向量

# 训练算法函数
# trainMatrix ---- 每篇文档/每条留言中的单词转化为数字后得到的词向量(returnVec)组成的训练矩阵trainMatrix
# trainCategory ----- 每篇文档所属的类别，即loadDataSet返回的类别标签向量classVec


def trainNB0(trainMatrix, trainCategory):
    # 训练矩阵的行数即含有的训练文档总数
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])                             # 计算每篇文档含有的单词个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)        # 文档属于侮辱类别的概率p(1)
    # 创建numpy.ones数组,存放每个单词出现的次数初始化为1，拉普拉斯平滑法，解决概率乘积为0的问题
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0  #  分母初始化为2,拉普拉斯平滑
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 统计属于侮辱类单词的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        # 统计属于非侮辱类单词的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    print('属于侮辱性类别中每个单词出现的总次数:', p1Num)
    print('属于侮辱性类别总的单词数目:', p1Denom)
    print('属于非侮辱性类别中每个单词出现的总次数:', p0Num)
    print('属于非侮辱性类别总的单词数目:', p0Denom)
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive
# 朴素贝叶斯分类函数


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    # 对应元素相乘, logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0
# 测试算法


def testingNB():
    primaryVocabularyLists, classVec = loadDateSet()
    print('primaryVocabularyLists:\n', primaryVocabularyLists)
    myVocabuList = createVocabList(primaryVocabularyLists)  # 创建词汇表，每个单词只出现一次
    print('myVocabuList:\n', myVocabuList)   # 打印输出一下词汇表
    trainMat = []  # trainMat是将每篇文档或每条留言转化为数字0，1后的结果即returnVec，主要利用的是SetOfWords2Vec()函数来实现
    for primaryVocabularyList in primaryVocabularyLists:
        trainMat.append(setOfWords2Vec(myVocabuList, primaryVocabularyList))
    print('单词转化为数字后的训练矩阵trainMat:\n', trainMat)
    print('需要训练的总文档数目:', len(trainMat))  # trainMat是由0，1组成的多维数组，称为训练矩阵
    print('词汇表中的总词条数:', len(trainMat[0]))
    p0V, p1V, pAb = trainNB0(trainMat, classVec)   # 开始训练分类函数
    print('属于非侮辱类单词的条件概率数组p0V:\n', p0V)
    print('属于侮辱类单词的条件概率数组p1V:\n', p1V)
    print('类别标签向量classVec:\n', classVec)
    print('文档属于侮辱类文档的概率pAb:\n', pAb)
    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabuList, testEntry))  # 测试样本向量化
    print('thisDoc测试样本1向量化后转化为0,1数字矩阵的结果：', thisDoc)
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果

    testEntry = ['stupid', 'garbage']  # 测试样本2
    thisDoc = np.array(setOfWords2Vec(myVocabuList, testEntry))  # 测试样本向量化
    print('thisDoc测试样本2向量化后转化为0,1数字矩阵的结果：', thisDoc)
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')     # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果


if __name__ == '__main__':
    testingNB()
